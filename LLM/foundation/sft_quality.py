"""Source-bound SFT context audit/preparation. Never edits an active corpus.

Static closure is an input-completeness check, never an execution oracle.
Only exact licensed function spans are targets; annotation postponement is an
explicit trusted execution contract, not a fabricated source import.
"""
from __future__ import annotations

import argparse
import ast
import builtins
from collections import Counter
import json
from pathlib import Path
import symtable
import zipfile

from LLM.core import require,digest,file_hash,atomic_json,canonical,now
from .source_context_contracts import selection,stamp_fields,verify_quality_record,source_parent_fields,matches_fields
from .source_context_inputs import source_inputs

VERSION='seed-source-context-sft-v1'
STDLIB={'abc','ast','base64','binascii','bisect','collections','contextlib','copy','dataclasses',
        'decimal','enum','fractions','functools','hashlib','heapq','html','inspect','itertools',
        'json','math','numbers','operator','re','statistics','string','struct','textwrap',
        'types','typing','unicodedata','urllib.parse'}
DYNAMIC={'eval','exec','compile','__import__','open','input','breakpoint','globals','locals','vars','getattr','setattr','delattr'}


def free_names(text,*,shadowed_builtins=None):
    tree=ast.parse(text);defined={node.name for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}
    result={'body':set(),'definition':set(),'annotation':set()}
    def visit(table,annotation=False):
        kind=str(table.get_type()).lower();annotation=annotation or 'annotation' in kind or 'type parameters' in kind
        category='annotation' if annotation else 'definition' if 'module' in kind else 'body'
        for symbol in table.get_symbols():
            if symbol.is_referenced() and symbol.is_global() and (symbol.get_name() not in vars(builtins)
                or symbol.get_name() in (shadowed_builtins or set())) and symbol.get_name() not in defined:
                result[category].add(symbol.get_name())
        for child in table.get_children():visit(child,annotation)
    visit(symtable.symtable(text,'<immutable-source-span>','exec'))
    return result


def _allowed_import(node):
    if isinstance(node,ast.Import):modules=[alias.name for alias in node.names]
    elif isinstance(node,ast.ImportFrom):
        if node.level or any(alias.name=='*' for alias in node.names):return False
        modules=[node.module or '']
    else:return True
    return all(module in STDLIB or any(module.startswith(root+'.') for root in STDLIB if root!='urllib.parse') for module in modules)


def _safe_code(node):
    for child in ast.walk(node):
        if isinstance(child,(ast.Global,ast.Nonlocal)):return False
        if isinstance(child,(ast.Import,ast.ImportFrom)) and not _allowed_import(child):return False
        if isinstance(child,ast.Call) and isinstance(child.func,ast.Name) and child.func.id in DYNAMIC:return False
        if isinstance(child,ast.Attribute) and child.attr.startswith('__'):return False
    return True


def _bindings(tree):
    values={}
    for node in tree.body:
        if isinstance(node,ast.Import):names=[alias.asname or alias.name.split('.')[0] for alias in node.names]
        elif isinstance(node,ast.ImportFrom):names=[alias.asname or alias.name for alias in node.names]
        elif isinstance(node,(ast.FunctionDef,ast.ClassDef)):names=[node.name]
        elif isinstance(node,(ast.Assign,ast.AnnAssign)):
            targets=node.targets if isinstance(node,ast.Assign) else [node.target]
            names=[item.id for target in targets for item in ast.walk(target) if isinstance(item,ast.Name) and isinstance(item.ctx,ast.Store)]
        else:continue
        for name in names:
            if name in values:values[name]=None  # Ambiguous rebinding cannot be resolved by guessing.
            else:values[name]=node
    return values


def contextualize(source,function,tokenizer,*,maximum=512,target_limit=128,contract=VERSION):
    output=ast.get_source_segment(source,function)
    if not output or function.decorator_list or function.type_params or not _safe_code(function):return None,'UNSUPPORTED_FUNCTION'
    doc=ast.get_docstring(function)
    if not doc:return None,'MISSING_TASK_DESCRIPTION'
    implementation=[node for node in function.body if not (isinstance(node,ast.Expr) and isinstance(node.value,ast.Constant)
                    and isinstance(node.value.value,(str,type(Ellipsis)))) and not isinstance(node,ast.Pass)]
    if not implementation:return None,'STUB_OR_EMPTY_IMPLEMENTATION'
    if len(implementation)==1 and isinstance(implementation[0],ast.Raise):
        raised=implementation[0].exc
        if isinstance(raised,ast.Name) and raised.id=='NotImplementedError' or isinstance(raised,ast.Call) and isinstance(raised.func,ast.Name) and raised.func.id=='NotImplementedError':
            return None,'STUB_OR_EMPTY_IMPLEMENTATION'
    targets=len(tokenizer.encode_ordinary(output))+1
    if not 24<=targets<=target_limit:return None,'TARGET_TOKEN_BUDGET'
    tree=ast.parse(source);bindings=_bindings(tree);chosen={};visiting=set()
    module_table=symtable.symtable(source,'<verified-original-module>','exec')
    shadowed={symbol.get_name() for symbol in module_table.get_symbols()
              if symbol.get_name() in vars(builtins) and (symbol.is_assigned() or symbol.is_imported())}
    names=free_names(output,shadowed_builtins=shadowed);required=names['body']|names['definition']
    def resolve(name):
        if name==function.name or name in vars(builtins) and name not in shadowed:return True
        if name in visiting:return False
        node=bindings.get(name)
        if node is None or node is function:return False
        identity=node.lineno
        if identity in chosen:return True
        if len(chosen)>=8:return False
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            if not _allowed_import(node):return False
        elif isinstance(node,(ast.Assign,ast.AnnAssign)):
            try:value=ast.literal_eval(node.value)
            except (ValueError,TypeError):return False
            if len(repr(value))>4096:return False
        elif isinstance(node,ast.FunctionDef):
            if node.decorator_list or node.type_params or not _safe_code(node):return False
        else:return False
        fragment=ast.get_source_segment(source,node)
        if not fragment or len(fragment)>4096:return False
        visiting.add(name)
        # Ordinary imports and literal constants define their names directly.
        dependencies=free_names(fragment,shadowed_builtins=shadowed) if isinstance(node,ast.FunctionDef) else {'body':set(),'definition':set()}
        if not all(resolve(value) for value in dependencies['body']|dependencies['definition']):return False
        visiting.remove(name);chosen[identity]=(node,fragment);return True
    if not all(resolve(name) for name in sorted(required)):return None,'UNRESOLVED_OR_NONSTDLIB_CONTEXT'
    context='\n\n'.join(fragment for _,fragment in sorted(chosen.values(),key=lambda pair:pair[0].lineno))
    signature='def '+function.name+'('+ast.unparse(function.args)+')'
    if function.returns:signature+=' -> '+ast.unparse(function.returns)
    description=doc.split('\n\n',1)[0][:400]
    instruction='Implement the specified Python function. Return only the function.\nRuntime contract: annotations are postponed with from __future__ import annotations. Type-hint evaluation is outside this task.\n'
    if context:instruction+='Original module context:\n```python\n'+context+'\n```\n'
    instruction+='Signature: '+signature+'\nDescription: '+description
    ids=tokenizer.assemble([{'role':'user','content':instruction},{'role':'assistant','content':output}],assistant_prefix=False)
    if len(ids)>maximum:return None,'TOTAL_TOKEN_BUDGET'
    closure=[]
    for node,fragment in sorted(chosen.values(),key=lambda pair:pair[0].lineno):
        closure.append({'startLine':node.lineno,'endLine':node.end_lineno,'sha256':digest(fragment.encode()),
                        'kind':type(node).__name__})
    return {'instruction':instruction,'output':output,'targetTokens':targets,'totalTokens':len(ids),
            'requiredBodyGlobals':sorted(required),'annotationNames':sorted(names['annotation']),
            'contextBindings':closure,'contract':contract,'contextSha256':digest(context.encode()),
            'runtimeExecuted':False,'staticContextClosed':True},None


def _data(root):
    from .acceptance import require_local_path
    root=Path(root).resolve()
    for path in (root,root/'control',root/'rag'):require_local_path(path,root)
    ready_path=root/'reports/readiness.json';ready=json.loads(ready_path.read_text('utf-8'))
    require(ready.get('state')=='ready' and ready.get('nasModified') is False,'PREPARATION_NOT_READY')
    folder=root/'control/foundation/corpora'/ready['sftCorpusId'];manifest_path=folder/'manifest.json'
    require(file_hash(manifest_path)==ready['sftCorpusHash'],'CORPUS_HASH_MISMATCH')
    manifest=json.loads(manifest_path.read_text('utf-8'))
    for name,checksum in manifest['files'].items():require(file_hash(folder/name)==checksum,'CORPUS_HASH_MISMATCH')
    return root,ready,folder,manifest


def _rows(folder,manifest,split):
    for name in manifest['splitShards'][split]:
        with (folder/name).open(encoding='utf-8') as handle:
            for line in handle:yield json.loads(line)


def audit(root,*,report_name="sft-source-context-audit-v1.json"):
    require(report_name in ("sft-source-context-audit-v1.json","sft-source-context-audit-v3.json"),"SOURCE_CONTEXT_VERSION_UNSUPPORTED")
    import apsw
    root,ready,folder,manifest=_data(root);before=file_hash(root/'reports/readiness.json')
    counts=Counter();frequencies=Counter();details=[]
    db=apsw.Connection(str(root/'control/metadata/llm.db'),flags=apsw.SQLITE_OPEN_READONLY)
    try:
        for row in _rows(folder,manifest,'train'):
            names=free_names(row['output']);all_names=set.union(*names.values());runtime=names['body']|names['definition']
            counts.update(total=1,anyExternal=bool(all_names),runtimeExternal=bool(runtime),bodyExternal=bool(names['body']),
                          definitionExternal=bool(names['definition']),annotationExternal=bool(names['annotation']),noExternalDetected=not all_names)
            frequencies.update(runtime);verification=row.get('independentVerification',{})
            counts['runtimeExecutedTrue']+=verification.get('runtimeExecuted') is True
            counts['runtimeExecutedFalse']+=verification.get('runtimeExecuted') is False
            fixture={}
            if verification.get('fixtureId'):
                found=list(db.execute("SELECT body FROM records WHERE kind='fixture' AND id=?",(verification['fixtureId'],)))
                fixture=json.loads(found[0][0]) if found else {}
                counts['fixturePresent']+=bool(fixture);counts['fixtureRuntimeVerifiedTrue']+=fixture.get('runtimeVerified') is True
                counts['fixtureRuntimeVerifiedFalse']+=fixture.get('runtimeVerified') is False;counts['literalCases']+=len(fixture.get('cases',[]))
                counts['fixtureProfileMissing']+=fixture.get('profileId') is None
            else:counts['fixtureMissing']+=1
            require(row.get('sourceSpan',{}).get('targetSha256')==digest(row['output'].encode()),'SOURCE_TARGET_HASH_MISMATCH')
            details.append({'function':ast.parse(row['output']).body[0].name,'exampleHash':row['exampleHash'],'groupId':row['groupId'],
                'sourceSpan':row['sourceSpan'],'freeNames':{key:sorted(value) for key,value in names.items()},
                'instructionHash':digest(row['instruction'].encode()),'instructionHasNoSourceContext':row['instruction'].startswith('Implement ') and '\n' not in row['instruction'],
                'runtimeExecuted':verification.get('runtimeExecuted') is True,'fixtureId':verification.get('fixtureId'),
                'fixtureRuntimeVerified':fixture.get('runtimeVerified') is True})
    finally:db.close()
    require(file_hash(root/'reports/readiness.json')==before,'PREPARATION_CHANGED')
    _data(root)
    report={'schemaVersion':1,'kind':'sft_source_context_audit','implementationVersion':VERSION,'observedAt':now(),
            'readinessHash':before,'originalCorpusId':ready['sftCorpusId'],'originalCorpusHash':ready['sftCorpusHash'],
            'counts':dict(counts),'runtimeExternalNameFrequency':frequencies.most_common(),'examples':details,
            'staticAnalysisOnly':True,'runtimeExecuted':False,'modelWeightsRead':False,'modelWeightsWritten':False,
            'originalReadinessAndCorpusUnchanged':True,'nasModified':False,
            'limitations':['Static names do not establish execution correctness or task-specification sufficiency.',
                           'Annotation dependencies are separate from Python-version-specific evaluation timing.',
                           'Builtin shadowing and dynamic attribute/string lookup require further source/runtime review.']}
    report['reportHash']=digest(report)
    path=root/'reports'/report_name;atomic_json(path,report)
    return report


def _source_references(archive,proof,group,cache,*,legacy=False):
    """One immutable archive parse per V3 source family, even if rows interleave."""
    from .prepare_data import _test_references
    key=(group,proof['revision'],proof['archiveSha256'])
    if not legacy and key in cache:return cache[key]
    with zipfile.ZipFile(archive) as package:references=_test_references(package)
    if not legacy:cache[key]=references
    return references


def prepare(root,*,preparation_version='v2'):
    plan=selection(preparation_version)
    from .prepare_data import ACTOR,_test_references
    from .tokenizer import FoundationTokenizer
    from .pc_validation import attach
    from LLM.service import Control
    from .corpus import build,rows
    root,ready,folder,manifest=_data(root);original_hash=file_hash(root/'reports/readiness.json');original=audit(root,report_name=plan['auditFile'])
    mapping=json.loads((root/'reports/source-map.json').read_text('utf-8'));require(mapping['actor']==ACTOR,'SOURCE_ACTOR_MISMATCH')
    sources={row['groupId']:row for row in mapping['sources'] if row['split'] in ('train','dev')}
    require(sum(row['split']=='train' for row in sources.values())==20,'TRAIN_FAMILY_COUNT_REQUIRED')
    tok=FoundationTokenizer(root/'control/foundation/tokenizers'/ready['tokenizerId']/'tokenizer.json')
    candidates=[];rejected=Counter();source_receipts={};references_cache={}
    parent_fields={}
    with source_inputs(root,ready,plan,sources) as (raw_parent,batches):
        if raw_parent is not None:
            parent_fields=source_parent_fields(raw_parent,project_id=ready['projectId'],
                tokenizer_id=ready['tokenizerId'],tokenizer_hash=ready['tokenizerHash'])
        for group,mapped,archive,proof,source_rows in batches:
            source_receipts[group]=proof
            references=_source_references(archive,proof,group,references_cache,legacy=plan['version']=='v2')
            for source in source_rows:
                if source['language'] not in (('py','pyi','python') if plan['version']=='v3' else ('py','pyi')):continue
                try:tree=ast.parse(source['text'])
                except SyntaxError:continue
                for function in tree.body:
                    if not isinstance(function,ast.FunctionDef) or function.name.startswith('_') or function.name not in references:continue
                    item,reason=contextualize(source['text'],function,tok,maximum=plan['maximumContextTokens'],
                        target_limit=plan['maximumTargetTokens'],contract=plan['staticVersion'])
                    if not item:rejected[reason]+=1;continue
                    item.update(source=source,sourceId=mapped['sourceId'],groupId=group,split=mapped['split'],reference=references[function.name][0],
                                function=function.name,functionStartLine=function.lineno,functionEndLine=function.end_lineno)
                    candidates.append(item)
    # Bound each family without allowing one repository to meet a diversity gate.
    selected=[];per_group=Counter();seen=set()
    for item in sorted(candidates,key=lambda row:(row['groupId'],row['totalTokens'],digest(row['output'].encode()))):
        key=digest(ast.dump(ast.parse(item['output']),include_attributes=False))
        if key in seen:rejected['EXACT_TARGET_DUPLICATE']+=1;continue
        maximum=32 if item['split']=='train' else 4
        if per_group[item['groupId']]>=maximum:rejected['FAMILY_EXAMPLE_CAP']+=1;continue
        seen.add(key);per_group[item['groupId']]+=1;selected.append(item)
    control=Control(root/'control',worker=False,execution_scope='pc_validation');rag=None
    try:
        rag=attach(control,root/'rag');project=control.foundation.project(ready['projectId']);active_before=project.get('activeSftCorpusId')
        policy_sources=set(project['policy'].get('allowedSourceIds',[]));allowed=sorted({row['sourceId'] for row in selected}&policy_sources)
        exported=control.bridge.foundation_corpus(project['id'],sorted({row['source']['versionId'] for row in selected}),allowed)
        authoritative={row['versionId']:row for row in exported['records']};records=[];accepted=[]
        for item in selected:
            source=item['source'];authority=authoritative.get(source['versionId'])
            if not authority or authority['sourceId']!=item['sourceId'] or authority['text']!=source['text']:
                rejected['CURRENT_RAG_PERMISSION_OR_SOURCE_UNAVAILABLE']+=1;continue
            require(authority['ragSplit']==('train' if item['split']=='train' else 'validation'),'SPLIT_LEAKAGE')
            base=source['sourceSpan']['startLine']-1
            span={**source['sourceSpan'],'startLine':base+item['functionStartLine'],'endLine':base+item['functionEndLine'],
                  'targetSha256':digest(item['output'].encode())}
            receipt={'actor':ACTOR,'kind':'delegated_source_context_review',**stamp_fields(plan),**parent_fields,'implementationVersion':plan['staticVersion'],
                     'runtimeExecuted':False,'staticContextClosed':True,'trustedExecutionPrelude':'from __future__ import annotations',
                     'typeHintEvaluationSupported':False,'contextSha256':item['contextSha256'],'contextBindings':item['contextBindings'],
                     'totalTokens':item['totalTokens'],'targetTokens':item['targetTokens'],'sourceReceipt':source_receipts[item['groupId']]}
            verification={'kind':'official_test_source_reference','source':item['reference'],'runtimeExecuted':False,
                          'state':'linked_not_executed','linkageBasis':'same_function_name_not_runtime_resolved','actor':ACTOR}
            records.append({**authority,'text':item['instruction']+'\n'+item['output'],'instruction':item['instruction'],'output':item['output'],
                            'reviewStatus':'approved','permissionStatus':'allowed','sourceSpan':span,'provenance':source['provenance'],
                            'approvalReceipt':receipt,'independentVerification':verification,'targetOrigin':'source_span'})
            accepted.append(item)
        corpus=build(control.store,project,records,{'kind':'sft','policyVersion':exported['policyVersion']})
        final=list(rows(control.store,corpus,'train',purpose='sft_quality_audit'));dev=list(rows(control.store,corpus,'dev',purpose='sft_quality_audit'))
        groups={row['groupId'] for row in final};target_tokens=sum(len(tok.encode_ordinary(row['output']))+1 for row in final)
        trained_groups={row['groupId'].rstrip('/').removesuffix('.git').lower() for row in mapping['sources'] if row['split']=='train'}
        require(groups<=trained_groups and not groups&{row['groupId'] for row in dev},'SPLIT_LEAKAGE')
        good=corpus['state']=='ready' and len(final)>=32 and len(groups)>=3 and target_tokens>=4096 and bool(dev)
        require(control.store.get('foundation_project',project['id']).get('activeSftCorpusId')==active_before,'ACTIVE_CORPUS_CHANGED')
        require(file_hash(root/'reports/readiness.json')==original_hash,'PREPARATION_CHANGED');_data(root)
        report={'schemaVersion':1,**stamp_fields(plan),**parent_fields,'implementationVersion':plan['staticVersion'],'state':'ready' if good else 'held','projectId':project['id'],
            'originalReadinessHash':original_hash,'originalCorpusId':ready['sftCorpusId'],'originalCorpusHash':ready['sftCorpusHash'],
            'originalAuditHash':original['reportHash'],'corpusId':corpus['id'],'corpusHash':corpus['manifestHash'],
            'trainExamples':len(final),'trainGroups':len(groups),'trainTargetTokens':target_tokens,'devExamples':len(dev),
            'trainFamilies':sorted(groups),'sourceReceipts':source_receipts,'candidateCount':len(candidates),'rejectedCounts':dict(rejected),
            'originalAnyExternal':original['counts']['anyExternal'],'originalRuntimeExternal':original['counts']['runtimeExternal'],
            'newUnresolvedBodyGlobals':0,'trustedPostponedAnnotations':True,'runtimeExecuted':False,'runtimeVerified':0,
            'approvalActor':ACTOR,'targetOrigin':'source_span','modelGeneratedTargets':0,'holdoutFamilyIntersections':0,
            'originalActiveCorpusUnchanged':True,'originalReadinessAndCorpusUnchanged':True,'nasModified':False,
            'maximumTotalTokens':max((row['approvalReceipt']['totalTokens'] for row in final),default=0),
            'maximumTargetTokens':max((row['approvalReceipt']['targetTokens'] for row in final),default=0),
            'executionContract':plan['staticVersion'],'qualityGateRequired':True,'servingEligible':False,'observedAt':now()}
        report['reportHash']=digest(report)
        record=control.store.put('foundation_sft_quality_preparation',report,digest([project['id'],plan['staticVersion'],corpus['manifestHash']]))
        atomic_json(root/'reports'/plan['qualityFile'],record)
        return record
    finally:
        control.store.close()
        if rag is not None:rag.close()


def register(root,*,preparation_version='v2'):
    plan=selection(preparation_version)
    """Approve NEW RAG examples/snapshot and seal a separately exportable dataset."""
    from .prepare_data import ACTOR,export_inventory
    from .pc_validation import attach
    from .tokenizer import load as load_tokenizer
    from .corpus import rows,build,verify
    from LLM.service import Control
    root,ready,folder,manifest=_data(root);original_hash=file_hash(root/'reports/readiness.json')
    implementation_hash=file_hash(Path(__file__))
    quality=json.loads((root/'reports'/plan['qualityFile']).read_text('utf-8'))
    require(quality.get('state')=='ready' and quality.get('originalReadinessHash')==original_hash
        and quality.get('implementationVersion')==plan['staticVersion']
        and quality.get('executionContract')==plan['staticVersion']
        and all(quality.get(key)==value for key,value in stamp_fields(plan).items()),'QUALITY_PREPARATION_NOT_READY')
    control=Control(root/'control',worker=False,execution_scope='pc_validation');rag=None
    try:
        authoritative=control.store.get('foundation_sft_quality_preparation',quality['id'])
        require(authoritative.get('reportHash')==quality['reportHash'],'QUALITY_PREPARATION_CHANGED')
        verify_quality_record(plan,quality,authoritative)
        raw_parent=control.store.get('foundation_data_preparation',plan['rawPreparationId'],{}) if plan['version']=='v3' else None
        parent_fields=source_parent_fields(raw_parent,project_id=ready['projectId'],tokenizer_id=ready['tokenizerId'],tokenizer_hash=ready['tokenizerHash']) if plan['version']=='v3' else {}
        require(matches_fields(quality,parent_fields),'SOURCE_CONTEXT_RAW_PREPARATION_CHANGED')
        existing=control.store.get('foundation_data_preparation',plan['preparationId'],{})
        if existing:
            require(existing.get('contextQualityPreparationId')==quality['id'] and existing.get('state')=='ready'
                and all(existing.get(key)==value for key,value in stamp_fields(plan).items()),'QUALITY_PREPARATION_CHANGED')
            return existing
        rag=attach(control,root/'rag');bridge=control.bridge;post=bridge.post;pre=bridge.pre
        project=control.foundation.project(ready['projectId']);active_before=project.get('activeSftCorpusId')
        token=load_tokenizer(control.store,ready['tokenizerId']);require(token.tokenizer_hash==ready['tokenizerHash'],'TOKENIZER_HASH_MISMATCH')
        source_corpus=control.store.get('foundation_corpus',quality['corpusId']);verify(control.store,source_corpus)
        created=[];original_examples={row['id']:canonical(row) for row in rag.list('training_example')
                                    if row['id'] in rag.get('training_snapshot',ready['ragSnapshotId'])['exampleIds']}
        for split in ('train','dev'):
            for row in rows(control.store,source_corpus,split,purpose='context_sft_registration'):
                version=rag.get('version',row['versionId']);source=pre.content(version['id']);tree=ast.parse(source)
                target=next((node for node in tree.body if isinstance(node,ast.FunctionDef)
                             and ast.get_source_segment(source,node)==row['output']),None)
                require(target is not None and digest(row['output'].encode())==row['sourceSpan']['targetSha256'],'SOURCE_TARGET_HASH_MISMATCH')
                prepared,reason=contextualize(source,target,token,maximum=plan['maximumContextTokens'],
                    target_limit=plan['maximumTargetTokens'],contract=plan['staticVersion'])
                require(prepared is not None and prepared['instruction']==row['instruction'] and prepared['output']==row['output'],
                        'SOURCE_CONTEXT_MISMATCH',reason=reason)
                identity=digest([row['instruction'],row['output'],row['versionId']])
                reused=next((item for item in rag.list('training_example') if item.get('contextExampleHash')==identity
                             and item.get('contextQualityPreparationId')==quality['id']),None)
                if reused:
                    require(reused.get('input')==row['instruction'] and reused.get('target')==row['output']
                            and reused.get('reviewedBy')==ACTOR and reused.get('reviewStatus')=='approved','CONTEXT_APPROVAL_CHANGED')
                    created.append(reused['id']);continue
                candidate=post.create_example({'input':row['instruction'],'target':row['output'],'lineage':row['lineage']})
                reviewed=post.review_example(candidate['id'],{'decision':'approve','permissionStatus':'allowed',
                    'reason':f"Delegated source-context review: exact immutable target, stdlib/literal/helper closure, native {plan['maximumContextTokens']}/{plan['maximumTargetTokens']} budget; no runtime correctness claim."})
                verification=dict(row['independentVerification']);reference=verification['source']
                verification['assertionBlobHash']=pre._blob(reference['assertion'])
                if reference.get('cases'):
                    fixture=control.store.put('fixture',{'cases':reference['cases'],'purpose':'official_source_sft_reference',
                        'sourceEvidence':reference,'registeredBy':ACTOR,'profileId':None,'runtimeVerified':False,
                        'contextQualityPreparationId':quality['id']})
                    verification['fixtureId']=fixture['id']
                approval={**row['approvalReceipt'],**stamp_fields(plan),**parent_fields,'actor':ACTOR,'kind':'delegated_source_context_review',
                          'sourceVersionStartLine':version['sourceSpan']['startLine']}
                rag.put('training_example',{'targetOrigin':'source_span','reviewedBy':ACTOR,
                    'approvalKind':'delegated_source_context_review','sourceSpan':row['sourceSpan'],
                    'independentVerification':verification,'approvalReceipt':approval,'executionStatus':'not_executed',
                    'contextExampleHash':identity,'contextQualityPreparationId':quality['id']},reviewed['id'])
                created.append(reviewed['id'])
        # The RAG exporter requires all three roles. Reuse its old test-only
        # example unchanged; never relabel a train family to manufacture a test.
        old_snapshot=rag.get('training_snapshot',ready['ragSnapshotId'])
        from RAG.Post.service import _manifest_examples
        old_root=rag.root/old_snapshot['storagePath'];old_manifest=json.loads((old_root/'manifest.json').read_text('utf-8'))
        heldout_ids=[item['id'] for item in _manifest_examples(old_root,old_manifest) if item['split']=='test']
        require(heldout_ids,'PRESERVED_TEST_SPLIT_REQUIRED')
        snapshot=post.create_snapshot({'exampleIds':[ *created,*heldout_ids ],'name':'Official source context '+preparation_version+'; runtime pending'})
        snapshot=post._export(snapshot['id'])
        exported=bridge.foundation_sft_corpus(project['id'],snapshot['id'],project['policy']['allowedSourceIds'])
        for row in exported['records']:
            example=rag.get('training_example',row['exampleId'])
            row.update(sourceSpan=example['sourceSpan'],independentVerification=example['independentVerification'],
                       approvalReceipt=example.get('approvalReceipt',{'actor':ACTOR,'kind':example['approvalKind'],'runtimeExecuted':False}))
        final=build(control.store,project,exported['records'],{'kind':'sft','policyVersion':exported['policyVersion']})
        training=list(rows(control.store,final,'train',purpose='context_sft_final_audit'));development=list(rows(control.store,final,'dev',purpose='context_sft_final_audit'))
        train_groups={row['groupId'] for row in training};dev_groups={row['groupId'] for row in development}
        target_tokens=sum(len(token.encode_ordinary(row['output']))+1 for row in training)
        require(final['state']=='ready' and len(training)>=32 and len(train_groups)>=3 and target_tokens>=4096 and development
                and not train_groups&dev_groups,'CONTEXT_SFT_REQUIREMENTS_NOT_MET')
        require(all(rag.get('training_example',identity)==json.loads(body) for identity,body in original_examples.items()),'ORIGINAL_EXAMPLE_CHANGED')
        require(control.store.get('foundation_project',project['id']).get('activeSftCorpusId')==active_before
                and file_hash(root/'reports/readiness.json')==original_hash and file_hash(Path(__file__))==implementation_hash,'PREPARATION_CHANGED')
        original=control.store.get('foundation_data_preparation','official-v1');requirements={**original['requirements'],
            'sftApprovedExamples':len(training),'sftTrainGroups':len(train_groups),'sftTargetTokens':target_tokens}
        report={key:original[key] for key in ('schemaVersion','actor','projectId','rawCorpusId','rawCorpusHash','tokenizerId','tokenizerHash',
                'ragSourceIds','ragRoot','controlRoot','rightsSourceCount')}
        report.update(**stamp_fields(plan),**parent_fields)
        if raw_parent is not None:
            requirements.update(raw_parent['requirements'])
            report.update(ragRoot=str(rag.root),controlRoot=str(control.store.root))
        report.update(state='ready',preparationKind='delegated_source_context_review',contextQualityPreparationId=quality['id'],
            originalPreparationId='official-v1',originalReadinessHash=original_hash,implementationVersion=plan['staticVersion'],implementationHash=implementation_hash,
            sftCorpusId=final['id'],sftCorpusHash=final['manifestHash'],ragSnapshotId=snapshot['id'],requirements=requirements,
            sftReview={'approvalActor':ACTOR,'approvalKind':'delegated_source_context_review','approvedExamples':len(created),
                      'trainTargetTokens':target_tokens,'runtimeTestsExecuted':False,'runtimeVerified':0,
                      'preservedOriginalTestExampleIds':heldout_ids,'newExampleIds':created},
            originalExamplesChanged=0,originalActiveCorpusUnchanged=True,modelTrainingPerformed=False,nasModified=False,
            servingEligible=False,qualityGateRequired=True,reasonCodes=[],observedAt=now())
        path=control.store.root/'reports/preparations'/(plan['preparationId']+'.json');atomic_json(path,report)
        record=control.store.put('foundation_data_preparation',{**report,'reportPath':path.relative_to(control.store.root).as_posix()},plan['preparationId'])
        inventory=export_inventory(control.store,record['id'],rag)
        require(inventory['metadata']['containsWeights'] is False and inventory['metadata']['preparationId']==plan['preparationId'],'DATASET_EXPORT_INVALID')
        control.store.put('foundation_sft_quality_preparation',{'registeredCorpusId':final['id'],'registeredCorpusHash':final['manifestHash'],
            'ragSnapshotId':snapshot['id'],'dataPreparationId':record['id'],'registrationState':'ready'},quality['id'])
        atomic_json(root/'reports'/(plan['preparationId']+'.json'),record)
        return record
    finally:
        control.store.close()
        if rag is not None:rag.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['audit','prepare','register']);parser.add_argument('root')
    parser.add_argument('--preparation-version',choices=['v2','v3'],default='v2')
    args=parser.parse_args()
    result=audit(args.root) if args.command=='audit' else {'prepare':prepare,'register':register}[args.command](args.root,preparation_version=args.preparation_version)
    print(canonical({key:result.get(key) for key in ('state','reportHash','corpusId','trainExamples','trainGroups','trainTargetTokens','devExamples','runtimeVerified')}))


def verify_runtime(control,rag,preparation_id,profile_id):
    from .sft_runtime_verification import verify_runtime as verify
    return verify(control,rag,preparation_id,profile_id)


if __name__=='__main__':main()
