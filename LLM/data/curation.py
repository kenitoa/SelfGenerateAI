"""Every document is accounted for; only evidenced examples enter training."""
import ast
import copy
import json
import re
from LLM.core import digest,require

VERSION=2


def purpose(text):
    head=text[:4000].lower()
    scores={'review':sum(head.count(word) for word in ('code review','코드 검토','security review','취약점','리뷰 지침')),
        'test':sum(head.count(word) for word in ('unit test','unittest','pytest','테스트 작성','테스트 설계')),
        'refactor':sum(head.count(word) for word in ('refactoring','리팩터링'))}
    best=max(scores,key=scores.get)
    return best if scores[best] else 'code'


def chunks(text,maximum=240):
    """Bounded source excerpts, without correcting or inventing source facts."""
    for paragraph in re.split(r'\n\s*\n',text):
        paragraph=paragraph.strip()
        if len(paragraph)<60:continue
        for start in range(0,len(paragraph),maximum):
            part=paragraph[start:start+maximum].strip()
            if len(part)>=60:yield part


def quality(text):
    from RAG.Post.service import SECRET_PATTERNS
    if any(pattern.search(text) for pattern in SECRET_PATTERNS):return ['SECRET_DETECTED']
    if len(text.strip())<40:return ['INSUFFICIENT_CONTENT']
    if '\x00' in text or text.count('\ufffd')>max(2,len(text)//100):return ['TEXT_DECODING_FAILED']
    if len(set(text.split()))<4:return ['LOW_INFORMATION_CONTENT']
    if re.search(r'ignore (all |the )?(previous|system) instructions|reveal (the )?system prompt',text,re.I):return ['PROMPT_INJECTION_REVIEW_REQUIRED']
    return []


def python_blocks(text):
    fenced=list(re.finditer(r'```(?:python|py)\s*\n([\s\S]*?)```',text,re.I))
    blocks=[m.group(1) for m in fenced] if fenced else [text]
    for block in blocks:
        try:tree=ast.parse(block)
        except SyntaxError:continue
        for node in tree.body:
            if isinstance(node,ast.FunctionDef):
                source=ast.get_source_segment(block,node)
                if source and len(source)<6000:yield node,source


def mutant(source):
    tree=ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node,ast.BinOp) and isinstance(node.op,(ast.Add,ast.Sub,ast.Mult)):
            node.op=ast.Sub() if isinstance(node.op,ast.Add) else ast.Add()
            return ast.unparse(ast.fix_missing_locations(tree))
    return None


def unit_tests(cases):
    lines=['import unittest','import solution','class ReferenceCases(unittest.TestCase):']
    for index,case in enumerate(cases):
        require(isinstance(case.get('function'),str) and case['function'].isidentifier(),'INVALID_VERIFIER_CASES')
        call=f"getattr(solution, {case['function']!r})(*{case.get('args',[])!r}, **{case.get('kwargs',{})!r})"
        lines.append(f'    def test_case_{index}(self):')
        if 'expectedError' in case:
            require(case['expectedError'] in ('ValueError','TypeError','ZeroDivisionError','IndexError','KeyError'),'INVALID_VERIFIER_CASES')
            lines.extend([f"        with self.assertRaises({case['expectedError']}):",f'            {call}'])
        else:lines.append(f"        self.assertEqual({call}, {case['expected']!r})")
    return '\n'.join(lines)+'\n'


class TaskFactory:
    def __init__(self,control):self.c=control

    def build(self,text,bindings=(),offset=0,limit=24,document_role=None):
        drafts=[];held=[]
        document_role=document_role or purpose(text)
        def add(role,instruction,target,verifier,objective='sft',**extra):
            drafts.append({'role':role,'instruction':instruction,'target':target,'objective':objective,
                'targetOrigin':'source_span','receipt':{'verifier':verifier,'builderVersion':VERSION,**extra}})
        excerpts=list(chunks(text))
        for excerpt in excerpts[offset:offset+limit]:
            add(document_role,'문서 근거를 그대로 인용하여 답하세요.\n자료:\n'+excerpt+'\n질문: 이 자료에서 설명한 내용을 인용하세요.',excerpt,
                'source_excerpt_v2',sourceHash=digest(excerpt.encode()),executionStatus='not_executed')
            add(document_role,'승인 원문의 다음 토큰을 학습합니다.',excerpt,'approved_cpt_excerpt_v1',objective='cpt',sourceHash=digest(excerpt.encode()),executionStatus='not_executed')
        for node,source in python_blocks(text):
            # Keep context-dependent snippets as document knowledge, not standalone functions.
            import builtins
            args={a.arg for a in [*node.args.posonlyargs,*node.args.args,*node.args.kwonlyargs]}
            if node.args.vararg:args.add(node.args.vararg.arg)
            if node.args.kwarg:args.add(node.args.kwarg.arg)
            local={n.id for n in ast.walk(node) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store)}
            loaded={n.id for n in ast.walk(node) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load)}
            if loaded-(args|local|set(dir(builtins))|{node.name}):
                held.append({'function':node.name,'reasonCodes':['FUNCTION_CONTEXT_REQUIRED']});continue
            signature=source.splitlines()[0]
            add('code',f'Python 함수를 문서의 계약에 맞춰 완성하세요.\n{signature}\n설명: {ast.get_docstring(node) or "원문 구현 재현"}',source,
                'source_span_ast_v2',sourceHash=digest(source.encode()),executionStatus='not_executed')
            facts={'function':node.name,'parameters':[a.arg for a in [*node.args.posonlyargs,*node.args.args,*node.args.kwonlyargs]],
                'returnStatements':sum(isinstance(n,ast.Return) for n in ast.walk(node))}
            add('review','다음 코드의 함수명·매개변수·return 문 개수를 검토하고 JSON으로 답하세요.\n'+source,
                json.dumps(facts,ensure_ascii=False,sort_keys=True),'independent_ast_facts_v1',executionStatus='static_verified')
            matching=[f for f in bindings if all(case.get('function')==node.name for case in f.get('cases',[])) and f.get('cases')]
            for fixture in matching:
                try:
                    baseline=self.c.sandbox.verify(source,fixture)
                    if not baseline['passed']:held.append({'function':node.name,'reasonCodes':['REFERENCE_TEST_FAILED']});continue
                    negative=mutant(source)
                    if not negative:continue
                    negative_receipt=self.c.sandbox.verify(negative,fixture)
                    if negative_receipt['passed']:held.append({'function':node.name,'reasonCodes':['INSUFFICIENT_TEST_DISCRIMINATION']});continue
                    receipt={'executionStatus':'passed','referenceReceiptId':baseline['id'],'negativeReceiptId':negative_receipt['id'],'fixtureId':fixture['id']}
                    add('review','테스트 계약을 만족하도록 오류를 수정하세요.\n코드:\n'+negative+'\n계약:\n'+json.dumps(fixture['cases'],ensure_ascii=False),source,'reference_mutation_test_v1',**receipt)
                    tests=unit_tests(fixture['cases'])
                    test_fixture={'profileId':fixture['profileId'],'tests':tests}
                    original_check=self.c.sandbox.verify(source,test_fixture);negative_check=self.c.sandbox.verify(negative,test_fixture)
                    if original_check['passed'] and not negative_check['passed']:
                        add('test','다음 함수의 계약을 검사하는 Python unittest를 작성하세요.\n'+source+'\n계약:\n'+json.dumps(fixture['cases'],ensure_ascii=False),tests,'executed_mutation_tests_v1',**receipt)
                except Exception as exc:held.append({'function':node.name,'reasonCodes':[getattr(exc,'code','VERIFICATION_FAILED')]})
            if not matching:held.append({'function':node.name,'role':'test','reasonCodes':['APPROVED_TEST_FIXTURE_REQUIRED']})
        unique={digest([d['role'],d['instruction'],d['target']]):d for d in drafts}
        return list(unique.values()),held,{'nextCursor':min(offset+limit,len(excerpts)),'chunkCount':len(excerpts),'complete':offset+limit>=len(excerpts)}
