"""Docker-only source-context data verification; never model-quality evidence."""
from __future__ import annotations
import ast
import json
import math
from pathlib import Path
import zipfile

from LLM.core import Fault,require,digest,file_hash,atomic_json,now
from .prepare_data import ACTOR
from .sft_quality import contextualize,VERSION
from .source_context_contracts import runtime_contract,runtime_evidence_fields,verify_runtime_source_parent


def source_runtime_program(row,source,tokenizer,*,preparation=None):
    plan=runtime_contract(preparation,row)
    require(digest(row['output'].encode())==row['sourceSpan']['targetSha256'],'SOURCE_TARGET_HASH_MISMATCH')
    tree=ast.parse(source)
    function=next((node for node in tree.body if isinstance(node,ast.FunctionDef)
                   and ast.get_source_segment(source,node)==row['output']),None)
    require(function is not None,'SOURCE_TARGET_HASH_MISMATCH')
    prepared,reason=contextualize(source,function,tokenizer,maximum=plan['maximumContextTokens'],
        target_limit=plan['maximumTargetTokens'],contract=plan['staticVersion'])
    require(prepared is not None and prepared['instruction']==row['instruction']
            and prepared['contextSha256']==row['approvalReceipt']['contextSha256'],
            'SOURCE_CONTEXT_MISMATCH',reason=reason)
    fragments=[]
    for binding in prepared['contextBindings']:
        node=next((node for node in tree.body if node.lineno==binding['startLine'] and node.end_lineno==binding['endLine']),None)
        require(node is not None,'SOURCE_CONTEXT_MISMATCH')
        fragment=ast.get_source_segment(source,node)
        require(digest(fragment.encode())==binding['sha256'],'SOURCE_CONTEXT_MISMATCH');fragments.append(fragment)
    context='\n\n'.join(fragments)
    require(digest(context.encode())==prepared['contextSha256'],'SOURCE_CONTEXT_MISMATCH')
    probe_name='_seed_definition_'+digest(row['output'].encode())[:16]
    code='from __future__ import annotations\n'+context+'\n\n'+row['output']+'\n\ndef '+probe_name+'():\n    return callable('+function.name+')\n'
    ast.parse(code)  # Syntax inspection only. Target code is never executed here.
    prepared={**prepared,'sourceFunctionStartLine':function.lineno,'sourceFunctionEndLine':function.end_lineno}
    return code,probe_name,function.name,prepared


def verified_receipt(store,receipt,code,fixture,profile):
    require(isinstance(receipt,dict) and receipt.get('id'),'SOURCE_RUNTIME_RECEIPT_INVALID')
    current=store.get('verifier_receipt',receipt['id'])
    require(current==receipt and current.get('codeHash')==digest(code.encode())
            and current.get('fixtureHash')==digest(fixture) and current.get('environmentDigest')==profile['image'],
            'SOURCE_RUNTIME_RECEIPT_INVALID')
    probe=store.get('verifier_probe',current.get('probeReceiptId') or '')
    require(probe.get('passed') is True and probe.get('state')=='passed' and probe.get('profileId')==profile['id']
            and probe.get('profileHash')==digest(profile) and probe.get('imageDigest')==profile['image'],
            'SOURCE_RUNTIME_RECEIPT_INVALID')
    return current.get('passed') is True and current.get('state')=='passed',probe


def _fixture(store,body,key):
    existing=store.get('fixture',key,{})
    if existing:
        require(all(existing.get(k)==v for k,v in body.items()),'SOURCE_RUNTIME_FIXTURE_CHANGED')
        return existing
    return store.put('fixture',body,key)


def _native_json(value):
    if value is None or type(value) in (bool,int,str):return True
    if type(value) is float:return math.isfinite(value)
    if type(value) is list:return all(_native_json(item) for item in value)
    if type(value) is dict:return all(type(key) is str and _native_json(item) for key,item in value.items())
    return False


def literal_cases_from_assertion(assertion,function_name):
    """Only exact scalar equality with lossless JSON inputs is an oracle here.

    Container/tuple result coercion by JSON is not a semantic type proof.
    Those cases remain pending rather than receiving a fabricated pass.
    """
    try:
        node=ast.parse(assertion).body[0]
        test=node.test if isinstance(node,ast.Assert) else None
        if not isinstance(test,ast.Compare) or len(test.ops)!=1 or not isinstance(test.ops[0],ast.Eq):return []
        call=test.left
        if not isinstance(call,ast.Call):return []
        name=call.func.id if isinstance(call.func,ast.Name) else call.func.attr if isinstance(call.func,ast.Attribute) else None
        if name!=function_name or any(keyword.arg is None for keyword in call.keywords):return []
        args=[ast.literal_eval(value) for value in call.args]
        kwargs={value.arg:ast.literal_eval(value.value) for value in call.keywords};expected=ast.literal_eval(test.comparators[0])
        if type(expected) not in (type(None),bool,int,float,str) or not _native_json([args,kwargs,expected]):return []
        return [{'function':function_name,'args':args,'kwargs':kwargs,'expected':expected}]
    except (ValueError,TypeError,SyntaxError):return []


def _source_cases(control,preparation,example,version,function_name,*,binding_out=None):
    verification=example.get('independentVerification',{});fixture_id=verification.get('fixtureId')
    if not fixture_id:return [],'NO_REGISTERED_LITERAL_CASE'
    fixture=control.store.get('fixture',fixture_id);reference=verification['source']
    require(fixture.get('sourceEvidence')==reference,'SOURCE_TEST_REFERENCE_MISMATCH')
    cases=literal_cases_from_assertion(reference['assertion'],function_name)
    if not cases:return [],'NO_LOSSLESS_SCALAR_LITERAL_CASE'
    require(all(case in fixture.get('cases',[]) and case['function']==function_name for case in cases),'SOURCE_TEST_REFERENCE_MISMATCH')
    require(getattr(control,'bridge',None) is not None and example.get('id') and version.get('id'),
            'SOURCE_TEST_REFERENCE_MISMATCH')
    from .source_oracle_binding import binding_evidence
    evidence=binding_evidence(control,preparation,example,version,function_name)
    if binding_out is not None:
        binding_out.update(oracleBindingId=evidence['id'],oracleBindingHash=digest(evidence),
            oracleBindingState=evidence['state'],oracleBindingVersion=evidence['helperVersion'])
    if evidence['state']!='matched':return [],evidence['binding']['reasonCode']
    return cases,None


def verify_runtime(control,rag,preparation_id,profile_id):
    from .corpus import verify,rows
    from .tokenizer import load
    from LLM.verification.broker import pinned
    from .acceptance import require_local_path
    require(getattr(control,'execution_scope',None)=='pc_validation','PC_VALIDATION_SCOPE_REQUIRED')
    require_local_path(control.store.root);require_local_path(rag.root)
    store=control.store;preparation=store.get('foundation_data_preparation',preparation_id)
    require(preparation.get('state')=='ready' and preparation.get('actor')==ACTOR
            and preparation.get('preparationKind')=='delegated_source_context_review','QUALITY_PREPARATION_NOT_READY')
    plan=runtime_contract(preparation)
    verify_runtime_source_parent(store,preparation)
    project=control.foundation.project(preparation['projectId']);control.foundation.validate_policy(project)
    bridge=control.bridge;require(bridge is not None and bridge.store.root==rag.root,'RAG_POLICY_GATEWAY_REQUIRED')
    profile=store.get('verifier_profile',profile_id)
    require(profile.get('language')=='python' and profile.get('purpose')=='code' and pinned(profile),'PINNED_SANDBOX_IMAGE_REQUIRED')
    corpus=store.get('foundation_corpus',preparation['sftCorpusId']);verify(store,corpus)
    require(corpus['manifestHash']==preparation['sftCorpusHash'],'CORPUS_HASH_MISMATCH')
    token=load(store,preparation['tokenizerId']);require(token.tokenizer_hash==preparation['tokenizerHash'],'TOKENIZER_HASH_MISMATCH')
    report={'projectId':project['id'],'preparationId':preparation_id,**runtime_evidence_fields(plan,preparation),'corpusId':corpus['id'],'corpusHash':corpus['manifestHash'],
            'tokenizerHash':token.tokenizer_hash,'profileId':profile['id'],'profileHash':digest(profile),'imageDigest':profile['image'],
            'executionScope':getattr(control,'execution_scope','nas'),'implementationVersion':plan['runtimeVersion'],
            'implementationHash':file_hash(Path(__file__)),'purpose':'source_span_data_execution',
            'definitionLoadOnlyIsNotSemanticCorrectness':True,'modelQualityPassed':False,'servingEligible':False,
            'originalExamplesAndCorpusMutated':False,'nasModified':False,'observedAt':now(),'rows':[],
            'definitionVerified':0,'runtimeVerified':0,'semanticCasesExecuted':0,'semanticCaseFailures':0}
    from .source_oracle_binding import VERSION as binding_version,implementation_hash as binding_hash
    binding_implementation=binding_hash()
    namespace=[plan['runtimeVersion'],binding_version,binding_implementation,file_hash(Path(__file__))]
    report.update(oracleBindingVersion=binding_version,oracleBindingImplementationHash=binding_implementation,
        runtimeFixtureNamespaceHash=digest(namespace),semanticPendingCount=0)
    try:
        probe=control.sandbox.probe(profile)
        require(probe.get('passed') is True and probe.get('state')=='passed' and probe.get('profileId')==profile['id'] and probe.get('profileHash')==digest(profile)
                and probe.get('imageDigest')==profile['image'],'SANDBOX_OS_PROBE_REQUIRED')
        report.update(initialProbeReceiptId=probe['id'],initialProbeHash=digest(probe))
        snapshot=bridge.post.snapshot(preparation['ragSnapshotId']);require(snapshot['status']=='ready','SNAPSHOT_NOT_READY')
        exported=bridge.foundation_sft_corpus(project['id'],snapshot['id'],project['policy']['allowedSourceIds'])
        authorized={row['exampleId']:row for row in exported['records']}
        source_rows=[*rows(store,corpus,'train',purpose='source_runtime_verification'),*rows(store,corpus,'dev',purpose='source_runtime_verification')]
        for row in source_rows:
            identity=row['permissionReceipt']['exampleId'];authority=authorized.get(identity)
            require(authority and authority['instruction']==row['instruction'] and authority['output']==row['output']
                    and authority['versionId']==row['versionId'] and authority['documentId']==row['documentId']
                    and authority['sourceId']==row['sourceId']
                    and authority.get('ragSplit')==('train' if row['split']=='train' else 'validation')
                    and row['permissionReceipt']['manifestHash']==snapshot['manifestHash'],'POLICY_REVOKED')
            example=rag.get('training_example',identity);require(example.get('reviewedBy')==ACTOR
                and example.get('approvalKind')=='delegated_source_context_review','SFT_APPROVAL_REQUIRED')
            require(row['approvalReceipt'].get('actor')==ACTOR and row['approvalReceipt'].get('kind')=='delegated_source_context_review'
                    and example.get('approvalReceipt',{}).get('contextSha256')==row['approvalReceipt'].get('contextSha256'),
                    'SOURCE_CONTEXT_MISMATCH')
            runtime_contract(preparation,{'approvalReceipt':example.get('approvalReceipt',{})})
            version=rag.get('version',row['versionId']);source=bridge.pre.content(version['id'])
            require(digest(source.encode())==version['rawBlobHash']==row['sourceSpan']['contentSha256'],'SOURCE_CONTENT_CHANGED')
            code,probe_name,function_name,prepared=source_runtime_program(row,source,token,preparation=preparation)
            version_span=version['sourceSpan'];base=version_span['startLine']-1
            require(row['sourceSpan']['path']==version_span['path']
                    and row['sourceSpan']['sourceFileSha256']==version_span['sourceFileSha256']
                    and row['sourceSpan']['startLine']==base+prepared['sourceFunctionStartLine']
                    and row['sourceSpan']['endLine']==base+prepared['sourceFunctionEndLine'],'SOURCE_TARGET_HASH_MISMATCH')
            common={'profileId':profile['id'],'verifierProfileHash':digest(profile),'profileImageDigest':profile['image'],
                    'registeredBy':plan['runtimeVersion'].removeprefix('seed-'),'sourceCorpusHash':corpus['manifestHash'],
                    'sourceExampleId':identity,'sourceTargetHash':row['sourceSpan']['targetSha256'],
                    'oracleBindingVersion':binding_version,'runtimeFixtureNamespaceHash':digest(namespace)}
            fixture=_fixture(store,{**common,'purpose':'source_definition_load_only','cases':[{'function':probe_name,'args':[],'kwargs':{},'expected':True}]},
                             digest([corpus['manifestHash'],identity,digest(profile),namespace,'definition-load']))
            result=control.sandbox.verify(code,fixture);passed,actual_probe=verified_receipt(store,result,code,fixture,profile)
            item={'exampleId':identity,'exampleHash':row['exampleHash'],'sourceVersionId':version['id'],
                  'sourceContentHash':version['rawBlobHash'],'sourceFileSha256':row['sourceSpan']['sourceFileSha256'],
                  'sourceSpanHash':digest(row['sourceSpan']),'targetHash':row['sourceSpan']['targetSha256'],
                  'contextHash':prepared['contextSha256'],'programHash':digest(code.encode()),
                  'totalTokens':prepared['totalTokens'],'targetTokens':prepared['targetTokens'],
                  'definitionVerified':passed,'definitionFixtureId':fixture['id'],'definitionFixtureHash':digest(fixture),
                  'definitionReceiptId':result['id'],'definitionReceiptHash':digest(result),
                  'probeReceiptId':actual_probe['id'],'probeReceiptHash':digest(actual_probe),
                  'semanticVerified':False,'semanticCaseCount':0,'reasonCodes':[] if passed else ['DEFINITION_LOAD_FAILED']}
            report['definitionVerified']+=passed
            # Eligibility is resolved from source bindings for every registered
            # witness, independently of this execution's success/failure.
            cases,reason=_source_cases(control,preparation,example,version,function_name,binding_out=item)
            if cases and passed:
                semantic=_fixture(store,{**common,'purpose':'official_source_scalar_cases','cases':cases,
                                  'sourceFixtureId':example['independentVerification']['fixtureId'],
                                  'sourceAssertionHash':example['independentVerification']['assertionBlobHash']},
                                  digest([corpus['manifestHash'],identity,digest(profile),namespace,'semantic-scalar']))
                answer=control.sandbox.verify(code,semantic);semantic_passed,semantic_probe=verified_receipt(store,answer,code,semantic,profile)
                item.update(semanticVerified=semantic_passed,semanticCaseCount=len(cases),semanticFixtureId=semantic['id'],
                    semanticFixtureHash=digest(semantic),semanticReceiptId=answer['id'],semanticReceiptHash=digest(answer),
                    semanticProbeReceiptId=semantic_probe['id'],semanticProbeReceiptHash=digest(semantic_probe))
                report['runtimeVerified']+=semantic_passed;report['semanticCasesExecuted']+=len(cases)
                report['semanticCaseFailures']+=not semantic_passed
                if not semantic_passed:item['reasonCodes'].append('OFFICIAL_LITERAL_CASE_FAILED')
            elif reason:
                item['semanticPendingReason']=reason;report['semanticPendingCount']+=1
            report['rows'].append(item)
            control.foundation.validate_policy(control.foundation.project(project['id']))
        require(source_rows,'SFT_DATA_REQUIRED')
        passed=report['definitionVerified']==len(source_rows) and report['semanticCaseFailures']==0
        report.update(passed=passed,state='passed' if passed else 'held',reasonCodes=[] if passed else ['SOURCE_RUNTIME_CHECK_FAILED'],
                      exampleCount=len(source_rows),ragSnapshotId=snapshot['id'],ragSnapshotHash=snapshot['manifestHash'])
    except Fault as error:report.update(passed=False,state='held',reasonCodes=[error.code])
    report['reportHash']=digest(report)
    value=store.put('foundation_sft_runtime_verification',report)
    atomic_json(store.root/'reports/preparations'/(preparation_id+'-runtime-'+value['id']+'.json'),value)
    return value
