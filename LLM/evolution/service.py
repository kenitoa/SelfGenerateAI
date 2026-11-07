"""Generate, isolate, test, version and atomically apply model-written source."""
import ast
import json
import os
from pathlib import Path
import re
import threading
import time
from datetime import datetime,timezone,timedelta
from LLM.core import require,digest,file_hash,atomic_json,uid,now

TARGET='LLM/extensions/document_policy.py'
GOAL='문서의 의미와 코드 문자를 보존하면서 앞뒤 공백과 CRLF 줄바꿈을 정리한다. transform(document)는 text와 tags 목록을 반환한다. 입력 객체를 수정하지 않는다.'


def validate_source(source):
    require(isinstance(source,str) and 0<len(source.encode())<=16000,'SELF_CODE_SIZE_LIMIT')
    try:tree=ast.parse(source)
    except SyntaxError:require(False,'SELF_CODE_SYNTAX_ERROR')
    require(any(isinstance(n,ast.FunctionDef) and n.name=='transform' for n in tree.body),'SELF_CODE_ENTRYPOINT_REQUIRED')
    for node in ast.walk(tree):
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            modules=[a.name.split('.')[0] for a in node.names] if isinstance(node,ast.Import) else [node.module]
            require(all(m in ('re','json','unicodedata','math','string') for m in modules),'SELF_CODE_IMPORT_DENIED')
        if isinstance(node,ast.Name):require(node.id not in ('open','exec','eval','compile','__import__','globals','locals','getattr','setattr','delattr','input','breakpoint','exit','quit'),'SELF_CODE_OPERATION_DENIED')
        if isinstance(node,ast.Attribute):require(not node.attr.startswith('__'),'SELF_CODE_OPERATION_DENIED')
    return source


def validate_transform(original,result):
    require(isinstance(result,dict) and isinstance(result.get('text'),str) and isinstance(result.get('tags'),list),'INVALID_DOCUMENT_TRANSFORM')
    require(all(isinstance(t,str) and len(t)<=64 for t in result['tags']) and len(result['tags'])<=16,'INVALID_DOCUMENT_TRANSFORM')
    require(original.split()==result['text'].split(),'DOCUMENT_SEMANTICS_CHANGED')
    try:before=ast.parse(original)
    except SyntaxError:before=None
    if before is not None:
        try:after=ast.parse(result['text'])
        except SyntaxError:require(False,'DOCUMENT_SEMANTICS_CHANGED')
        require(ast.dump(before,include_attributes=False)==ast.dump(after,include_attributes=False),'DOCUMENT_SEMANTICS_CHANGED')
    else:
        from LLM.data.curation import python_blocks
        prior=[ast.dump(node,include_attributes=False) for node,_ in python_blocks(original)]
        current=[ast.dump(node,include_attributes=False) for node,_ in python_blocks(result['text'])]
        require(prior==current,'DOCUMENT_SEMANTICS_CHANGED')
    return result


class SelfEvolution:
    def __init__(self,control):
        self.c=control;self.store=control.store
        self.project=Path(__file__).resolve().parents[2];self.lock=threading.RLock();self.last_check=None

    def policy(self):
        return {**{'id':'self-code','version':0,'enabled':True,'autoApply':True,'fixtureId':None,'intervalHours':24,'goal':GOAL},**self.store.get('policy','self-code',{})}

    def configure(self,body,expected=None):
        require(not set(body)-{'enabled','autoApply','fixtureId','intervalHours','goal'},'UNKNOWN_SELF_CODE_POLICY')
        value={**self.policy(),**body}
        require(isinstance(value['enabled'],bool) and isinstance(value['autoApply'],bool) and isinstance(value['intervalHours'],int) and value['intervalHours']>=1,'INVALID_SELF_CODE_POLICY')
        require(isinstance(value['goal'],str) and 1<=len(value['goal'])<=2000,'INVALID_SELF_CODE_POLICY')
        if value['fixtureId']:
            fixture=self.store.get('fixture',value['fixtureId']);profile=self.store.get('verifier_profile',fixture['profileId'])
            require(profile['language']=='python' and fixture.get('cases'),'INVALID_SELF_CODE_FIXTURE')
            require(all(c.get('function')=='transform' for c in fixture['cases']),'INVALID_SELF_CODE_FIXTURE')
        self.last_check=None
        return self.store.put('policy',value,'self-code',expected=expected)

    def active_hash(self):
        return digest((self.project/TARGET).read_text('utf-8').encode())

    def transform(self,document):
        active=self.store.get('self_code_slot','document',{})
        if not active:
            # The checked-in initial transform is trusted; never import generated modules.
            return {'text':document['text'].strip(),'tags':[]}
        with self.lock:
            require(self.active_hash()==active['sourceHash'],'SELF_CODE_HASH_MISMATCH')
            source=(self.project/TARGET).read_text('utf-8')
            profile=self.store.get('verifier_profile',active['profileId'])
            values=self.c.sandbox.run_python(source,[{'function':'transform','args':[document]}],profile)
            require(len(values)==1 and 'value' in values[0],'DOCUMENT_TRANSFORM_FAILED')
            return validate_transform(document['text'],values[0]['value'])

    def check(self):
        policy=self.policy()
        if not policy['enabled'] or (self.last_check is not None and time.monotonic()-self.last_check<policy['intervalHours']*3600):return
        self.last_check=time.monotonic()
        if not policy['fixtureId'] or not self.c.sandbox.available():
            self.store.put('setting',{'state':'waiting_verifier','reasonCodes':['SELF_CODE_VERIFIER_REQUIRED']},'selfCodeStatus');return
        models=[a for a in self.store.list('artifact') if a.get('state')=='ready' and not a.get('fixture') and a['kind']!='adapter']
        if not models:return
        self.enqueue({'artifactId':models[-1]['id']})

    def enqueue(self,body,key=None):
        model=self.store.get('artifact',body['artifactId'])
        require(not model.get('comparisonOnly'),'COMPARISON_ONLY_MODEL',status=409)
        require(not model.get('fixture') and model['kind']!='adapter','SELF_CODE_MODEL_REQUIRED')
        if model.get('origin')=='random_init' or model.get('foundationProjectId'):
            self._foundation_eligibility(model)
        with self.store.tx():
            if model.get('origin')=='random_init' or model.get('foundationProjectId'):
                self._budget('proposals')
            result=self.store.enqueue('evolve',{'artifactId':model['id'],'sourceHash':self.active_hash(),'goal':self.policy()['goal'],'fixtureId':self.policy()['fixtureId']},key)
            if model.get('origin')=='random_init' or model.get('foundationProjectId'):
                self.store.put('foundation_self_code_attempt',{'artifactId':model['id'],'week':self._week(),'state':'reserved'},result['id'])
            return result

    def _foundation_eligibility(self,model):
        self.c.registry.verify(model)
        eligibility=self.store.get('foundation_self_code_eligibility',model['id'],{})
        require(eligibility.get('passed') is True and eligibility.get('artifactId')==model['id']
            and eligibility.get('state')=='eligible' and eligibility.get('fixtureId')==self.policy().get('fixtureId')
            and eligibility.get('sourceHash')==self.active_hash()
            and not eligibility.get('testOnly'),'SELF_CODE_ELIGIBILITY_REQUIRED',status=409)
        fixture=self.store.get('fixture',self.policy()['fixtureId'])
        cert=self.store.get('certificate',eligibility.get('certificateId') or '',{})
        require(eligibility.get('fixtureHash')==digest(fixture) and cert.get('passed') is True
            and cert.get('usable') is True and cert.get('purpose')=='usable_release' and cert.get('state')!='revoked'
            and cert.get('artifactId')==model['id'],'SELF_CODE_ELIGIBILITY_REQUIRED',status=409)
        return eligibility

    @staticmethod
    def _week():
        return '-'.join(map(str,datetime.now(timezone(timedelta(hours=9))).isocalendar()[:2]))

    def _budget(self,kind):
        week=self._week()
        if kind=='proposals':
            count=sum(r.get('week')==week for r in self.store.list('foundation_self_code_attempt'))
            require(count<3,'SELF_CODE_PROPOSAL_BUDGET_EXHAUSTED',status=409)
        else:
            count=sum(r.get('appliedWeek')==week for r in self.store.list('self_code_proposal'))
            require(count<1,'SELF_CODE_APPLY_BUDGET_EXHAUSTED',status=409)

    def run(self,job):
        policy=self.policy();source=(self.project/TARGET).read_text('utf-8');body=job['payload']
        require(digest(source.encode())==body['sourceHash'],'SELF_CODE_VERSION_CONFLICT',status=409)
        model=self.store.get('artifact',body['artifactId'])
        content='Write only a Python module with transform(document) returning text and tags. No file, network or process access.\nGoal: '+body['goal']+'\nModule:\n'+source
        require(not model.get('comparisonOnly'),'COMPARISON_ONLY_MODEL',status=409)
        if model.get('origin')=='random_init' or model.get('foundationProjectId'):
            generated,timing=self.c.foundation.runtime.text(model['id'],[{'role':'user','content':content}],min(128,int(model.get('trainedContextTokens',128))//2))
        else:
            generated,timing=self.c.inference.text(body['artifactId'],'<|im_start|>user\n'+content+'\n<|im_end|>\n<|im_start|>assistant\n',768)
        blocks=re.findall(r'```(?:python)?\s*\n([\s\S]*?)```',generated)
        proposed=(blocks[0] if blocks else generated).strip()+'\n'
        record=self.store.put('self_code_proposal',{'target':TARGET,'modelArtifactId':body['artifactId'],'baseHash':body['sourceHash'],
            'source':proposed,'sourceHash':digest(proposed.encode()),'state':'proposed','goal':body['goal'],'timings':timing,'fixtureId':body['fixtureId']},job['id'])
        try:
            validate_source(proposed)
            require(record['sourceHash']!=body['sourceHash'],'SELF_CODE_NO_CHANGE')
            require(body['fixtureId'] and self.c.sandbox.available(),'SELF_CODE_VERIFIER_REQUIRED',status=503)
            fixture=self.store.get('fixture',body['fixtureId']);profile=self.store.get('verifier_profile',fixture['profileId'])
            baseline=self.c.sandbox.verify(source,fixture);candidate=self.c.sandbox.verify(proposed,fixture)
            require(candidate['passed'],'SELF_CODE_REGRESSION_FAILED')
            inputs=[{k:v for k,v in c.items() if k not in ('expected','expectedError')} for c in fixture['cases']]
            values=self.c.sandbox.run_python(proposed,inputs,profile)
            require(len(values)==len(fixture['cases']),'SELF_CODE_REGRESSION_FAILED')
            for case,value in zip(fixture['cases'],values):validate_transform(case['args'][0]['text'],value.get('value'))
            require(not baseline['passed'],'SELF_CODE_IMPROVEMENT_UNCONFIRMED')
            record=self.store.put('self_code_proposal',{'state':'verified','baselineReceiptId':baseline['id'],'candidateReceiptId':candidate['id'],
                'profileId':fixture['profileId'],'functionalImprovement':True},record['id'])
            if self.policy()['enabled'] and self.policy()['autoApply'] and self.policy()['fixtureId']==body['fixtureId']:
                record=self.apply(record['id'],job)
                if self.c.bridge:
                    try:self.feedback(record)
                    except Exception as exc:self.store.event('self_code.feedback_held',record['id'],{'code':getattr(exc,'code','FEEDBACK_FAILED')})
        except Exception as exc:record=self.store.put('self_code_proposal',{'state':'held','reasonCodes':[getattr(exc,'code','SELF_CODE_FAILED')]},record['id'])
        self.store.put('setting',{'state':record['state'],'proposalId':record['id'],'reasonCodes':record.get('reasonCodes',[])},'selfCodeStatus')
        return record

    def apply(self,identity,job=None):
        with self.lock:
            record=self.store.get('self_code_proposal',identity)
            require(record['state']=='verified' and record.get('functionalImprovement'),'SELF_CODE_NOT_VERIFIED')
            require(record['target']==TARGET and self.active_hash()==record['baseHash'],'SELF_CODE_VERSION_CONFLICT',status=409)
            artifact=self.store.get('artifact',record['modelArtifactId'])
            self.c.registry.verify(artifact)
            is_foundation=artifact.get('origin')=='random_init' or bool(artifact.get('foundationProjectId'))
            if is_foundation:self._foundation_eligibility(artifact);self._budget('apply')
            if job:self.store.assert_lease(job)
            candidate=self.store.get('verifier_receipt',record['candidateReceiptId'])
            require(candidate['passed'] and candidate['codeHash']==record['sourceHash'],'SELF_CODE_RECEIPT_MISMATCH')
            target=self.project/TARGET;old=target.read_text('utf-8')
            backup=self.store.root/'self-code'/identity;backup.mkdir(parents=True,exist_ok=True)
            atomic_json(backup/'manifest.json',{'target':TARGET,'before':old,'beforeHash':record['baseHash'],'afterHash':record['sourceHash'],
                'previousSlot':self.store.get('self_code_slot','document',{})})
            temporary=target.with_suffix('.py.pending')
            try:
                with temporary.open('w',encoding='utf-8',newline='\n') as out:out.write(record['source']);out.flush();os.fsync(out.fileno())
                require(file_hash(temporary)==record['sourceHash'],'SELF_CODE_HASH_MISMATCH')
                os.replace(temporary,target)
                try:
                    canary=self.c.sandbox.verify(target.read_text('utf-8'),self.store.get('fixture',record['fixtureId']))
                    require(canary.get('passed') is True and canary.get('codeHash')==record['sourceHash'],'SELF_CODE_CANARY_FAILED')
                    if job:self.store.assert_lease(job)
                    with self.store.tx():
                        if is_foundation:self._budget('apply')
                        self.store.put('self_code_slot',{'proposalId':identity,'sourceHash':record['sourceHash'],'profileId':record['profileId']},'document')
                        return self.store.put('self_code_proposal',{'state':'applied','appliedAt':now(),'appliedWeek':self._week(),
                            'canaryReceiptId':canary.get('id')},identity)
                except BaseException:
                    with temporary.open('w',encoding='utf-8',newline='\n') as out:out.write(old);out.flush();os.fsync(out.fileno())
                    os.replace(temporary,target)
                    self.store.put('self_code_proposal',{'state':'held','reasonCodes':['SELF_CODE_CANARY_FAILED']},identity)
                    raise
            finally:temporary.unlink(missing_ok=True)

    def rollback(self):
        with self.lock,self.store.tx():
            slot=self.store.get('self_code_slot','document');record=self.store.get('self_code_proposal',slot['proposalId'])
            require(self.active_hash()==record['sourceHash'],'SELF_CODE_VERSION_CONFLICT')
            manifest=json.loads((self.store.root/'self-code'/record['id']/'manifest.json').read_text('utf-8'))
            require(digest(manifest['before'].encode())==manifest['beforeHash'],'SELF_CODE_HASH_MISMATCH')
            target=self.project/TARGET;temporary=target.with_suffix('.py.rollback')
            with temporary.open('w',encoding='utf-8',newline='\n') as out:out.write(manifest['before']);out.flush();os.fsync(out.fileno())
            os.replace(temporary,target)
            if manifest.get('previousSlot'):self.store.put('self_code_slot',manifest['previousSlot'],'document')
            else:self.store.db.execute("DELETE FROM records WHERE kind='self_code_slot' AND id='document'")
            return self.store.put('self_code_proposal',{'state':'rolled_back'},record['id'])

    def reconcile(self):
        """Recover only a known verified revision, never overwrite an unknown edit."""
        with self.lock,self.store.tx():
            slot=self.store.get('self_code_slot','document',{})
            if not slot:return {'state':'unconfigured'}
            record=self.store.get('self_code_proposal',slot['proposalId'])
            current=self.active_hash()
            if current==slot['sourceHash']:return {'state':'consistent','proposalId':record['id']}
            require(current==record['baseHash'],'SELF_CODE_VERSION_CONFLICT',status=409)
            receipt=self.store.get('verifier_receipt',record['candidateReceiptId'])
            require(receipt['passed'] and receipt['codeHash']==record['sourceHash']==slot['sourceHash'],'SELF_CODE_RECEIPT_MISMATCH')
            validate_source(record['source'])
            target=self.project/TARGET;temporary=target.with_suffix('.py.restore')
            with temporary.open('w',encoding='utf-8',newline='\n') as out:out.write(record['source']);out.flush();os.fsync(out.fileno())
            os.replace(temporary,target)
            return {'state':'restored','proposalId':record['id']}

    def setup(self,profile_id):
        profile=self.store.get('verifier_profile',profile_id)
        from LLM.verification.broker import pinned
        require(profile['language']=='python' and pinned(profile),'PINNED_SANDBOX_IMAGE_REQUIRED')
        if profile.get('purpose')!='self_code_regression':
            profile=self.store.put('verifier_profile',{'language':'python','image':profile['image'],'purpose':'self_code_regression'})
            profile_id=profile['id']
        # Fixed controller-authored expectations; generated code cannot edit these.
        texts=[' hello ','one\r\ntwo','\r\nalpha\r\nbeta\r\n','def add(a, b):\n    return a + b\n','한글 문서\r\n다음 줄',
            'x\ty','\n','a'*1000,' curly {json} ','a\n\n b','   x\r\ny  ','  \t끝\t ']
        cases=[{'function':'transform','args':[{'text':t,'documentId':str(i)}],
            'expected':{'text':t.replace('\r\n','\n').strip(),'tags':[]}} for i,t in enumerate(texts)]
        fixture=self.store.put('fixture',{'profileId':profile_id,'cases':cases,'purpose':'self_code_regression',
            'trustedControllerFixture':True,'independent':True,'targetOrigin':'controller'})
        return self.configure({'fixtureId':fixture['id']})

    def feedback(self,proposal):
        """Verified code revisions re-enter RAG with model-origin labels and lineage."""
        require(proposal['state']=='applied','SELF_CODE_NOT_VERIFIED')
        bridge=self.c.bridge;identity='self-code-feedback'
        source=self.store.get('setting',identity,{}).get('sourceId')
        if not source:
            row=bridge.pre.create_source({'kind':'upload','name':'검증된 자기 코드 수정','trainingPolicy':'allowed','retrievalPolicy':'approved'})
            source=row['id'];bridge.store.put('source',{'origin':'model_self_code'},source)
            self.store.put('setting',{'sourceId':source},identity)
            self.c.scheduler.configure({'allowedSourceIds':list(dict.fromkeys([*self.c.scheduler.policy()['allowedSourceIds'],source]))})
        receipt=bridge.pre.upload([('self_code_'+proposal['id']+'.py',proposal['source'].encode())],source_id=source,mode='append')
        return self.store.put('self_code_feedback',{'sourceId':source,'proposalId':proposal['id'],'upload':receipt,'state':'indexing','targetOrigin':'teacher'},proposal['id'])

    def advance_feedback(self):
        if not self.c.bridge:return
        bridge=self.c.bridge
        for record in self.store.list('self_code_feedback'):
            if record['state']!='indexing':continue
            documents=[bridge.pre.document(i) for i in record['upload'].get('documentIds',[])]
            if not documents or any(not d.get('currentVersionId') for d in documents):continue
            proposal=self.store.get('self_code_proposal',record['proposalId'])
            for document in documents:
                self.store.put('document_verification',{'fixtureIds':[proposal['fixtureId']]},document['currentVersionId'])
            self.store.put('self_code_feedback',{'state':'ready','versionIds':[d['currentVersionId'] for d in documents]},record['id'])
