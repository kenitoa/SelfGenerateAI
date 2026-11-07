"""Trusted local model acceptance runners for requirements 7 and 10.

There is deliberately no score-upload API. Software tests cannot issue these
passes. Service fault injection operates only on a fresh local copy. Specialist
acceptance requires actual purpose-specific training, seven weight artifacts,
and current independent model/domain certificates.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import itertools
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time

import apsw

from LLM.core import Fault,Store,atomic_json,canonical,digest,file_hash,inside,now,require,uid
from RAG.core import DomainError

RUNNER='seed-local-model-acceptance-v1'
ROLES={'A':('code','code','python'),'B':('review','review','ko'),'C':('test','tests','python')}
MATRIX=(('A',),('B',),('C',),('A','B'),('A','C'),('B','C'),('A','B','C'))
ROOT=Path(__file__).resolve().parents[2]
_GATES={}


def _local(control):
    from .acceptance import require_local_path
    require(getattr(control,'execution_scope',None)=='pc_validation','PC_VALIDATION_SCOPE_REQUIRED')
    require_local_path(control.store.root)
    bridge=getattr(control,'bridge',None);require(bridge is not None,'RAG_POLICY_GATEWAY_REQUIRED')
    require_local_path(bridge.store.root)


def _idle(control):
    require(not control.store.rows("SELECT id FROM jobs WHERE state IN ('queued','running','paused','pause_requested','cancelling') LIMIT 1"),
        'HEAVY_WORK_IN_PROGRESS',status=409)
    require(not any(thread.is_alive() for thread in getattr(control,'threads',[])),
        'SOURCE_WORKERS_MUST_BE_STOPPED',status=409)
    require(not control.bridge.store.read("SELECT id FROM jobs WHERE state IN ('running','cancelling') LIMIT 1"),
        'RAG_WORK_IN_PROGRESS',status=409)


def _evidence_root(control):
    from .acceptance import require_local_path
    root=ROOT/'LLM/runtime-data/model-acceptance'/digest(str(control.store.root.resolve()))[:24]
    require_local_path(root);return root


def _record(control,number,project_id,body):
    from .acceptance import _seal,source_snapshot
    with_store=Store(_evidence_root(control))
    try:
        record=_seal(with_store,'foundation_model_requirement_'+str(number),{
            'runnerVersion':RUNNER,'requirementNumber':number,'projectId':project_id,
            'sourceStoreHash':digest(str(control.store.root.resolve())),
            'sourceHash':source_snapshot()['hash'],'executionScope':'pc_validation',
            'nasModified':False,'activationPerformed':False,**body})
        atomic_json(with_store.root/'reports'/('requirement-'+str(number)+'-'+record['id']+'.json'),record)
        return record
    finally:with_store.close()


def _blocked(control,number,project_id,error,**extra):
    return _record(control,number,project_id,{'passed':False,'state':'blocked',
        'reasonCodes':[getattr(error,'code','MODEL_ACCEPTANCE_FAILED')],
        'details':getattr(error,'details',{}),'actualModelGatePassed':False,**extra})


def _db_hashes(control):
    result={}
    for label,path in (('control',control.store.root/'metadata/llm.db'),('rag',control.bridge.store.db_path)):
        for suffix in ('','-wal'):
            target=Path(str(path)+suffix)
            result[label+suffix]=file_hash(target) if target.is_file() else None
    return result


def _rights(control,project):
    """Read authoritative RAG decisions/immutable rights; never refresh or approve."""
    bridge=control.bridge;allowed=project.get('policy',{}).get('allowedSourceIds',[])
    allowed=control.store.get('foundation_policy',project['id'],project.get('policy',{})).get('allowedSourceIds',allowed)
    require(allowed,'CORPUS_PERMISSION_REQUIRED')
    receipts=bridge.store.list('source_rights_receipt');sources=[];evidence=[]
    for identity in sorted(allowed):
        source=bridge.store.get('source',identity)
        require(source.get('trainingPolicy')=='allowed' and not source.get('upstreamDeleted'),'POLICY_REVOKED')
        current=[row for row in receipts if row.get('sourceId')==identity and row.get('state')=='rights_verified'
            and row.get('decision','allowed')=='allowed']
        require(current,'SOURCE_RIGHTS_RECEIPT_REQUIRED',sourceId=identity)
        for receipt in current:
            for item in receipt.get('licenses',[]):
                path=inside(bridge.store.root,item.get('relativePath',''))
                require(path.is_file() and not path.is_symlink() and file_hash(path)==item.get('sha256'),'SOURCE_RIGHTS_RECEIPT_CHANGED')
            evidence.append((receipt['id'],digest(receipt)))
        sources.append((identity,digest(source)))
    return {'policyVersion':bridge.store.setting('policyVersion',0),'dataRevision':bridge.store.setting('trainingDataRevision',0),
        'sources':sources,'rights':sorted(evidence),'ragRootHash':digest(str(bridge.store.root.resolve()))}


def _proof(control,artifact,certificate):
    """Read actual source-owned graded records, not generic software receipts."""
    from .evaluation import first_usable_gate,subsequent_gate
    store=control.store;result=[];problems=set();baseline_proofs=[]
    require(certificate.get('passed') is True and certificate.get('usable') is True
        and certificate.get('purpose')=='usable_release' and certificate.get('state') not in ('revoked','quarantined')
        and not certificate.get('testOnly') and not certificate.get('fixture')
        and certificate.get('artifactId')==artifact['id'],'USABLE_FOUNDATION_MODEL_REQUIRED')
    identities=certificate.get('evaluationIds',[]);require(identities,'INDEPENDENT_EXAM_REQUIRED')
    for identity in identities:
        record=store.get('foundation_evaluation',identity)
        require(record.get('state')=='completed' and record.get('kind')=='release' and record.get('productionCertified') is True
            and record.get('gate',{}).get('passed') is True and not record.get('fixture') and record.get('artifactId')==artifact['id'],
            'INDEPENDENT_EXAM_REQUIRED')
        suite=store.get('foundation_suite',record['suiteId'])
        require(suite.get('state')=='ready' and not suite.get('fixture') and suite['suiteHash']==record['suiteHash']
            and suite['suiteHash']==digest({key:suite[key] for key in ('kind','tasks','fixture')}),'EVALUATION_SUITE_CHANGED')
        rows=record['results'][artifact['id']];by_task={row['taskId']:row for row in rows}
        require(len(by_task)==len(suite['tasks']) and {task['id'] for task in suite['tasks']}==set(by_task),'INDEPENDENT_EXAM_REQUIRED')
        for task in suite['tasks']:
            fixed=store.get('fixture',task['fixtureId']);row=by_task[task['id']]
            require(fixed.get('independent') is True and not fixed.get('testOnly') and not fixed.get('fixture')
                and digest(fixed)==task['fixtureHash'],'EVALUATION_FIXTURE_CHANGED')
            if not task.get('critical'):problems.add(task['problemFingerprint'])
            if row.get('passed'):
                receipt=store.get('verifier_receipt',row.get('receiptId') or '',{})
                probe=store.get('verifier_probe',receipt.get('probeReceiptId') or '',{})
                profile=store.get('verifier_profile',fixed.get('profileId') or '',{})
                from LLM.verification.broker import pinned
                require(receipt.get('passed') is True and receipt.get('state')=='passed' and receipt.get('fixtureHash')==digest(fixed)
                    and receipt.get('codeHash') and probe.get('passed') is True and probe.get('limits',{}).get('dockerInspectPassed') is True
                    and probe.get('profileHash')==digest(profile) and pinned(profile)
                    and receipt.get('environmentDigest')==profile.get('image'),'ACTUAL_VERIFIER_EVIDENCE_REQUIRED')
            require(row.get('timings',{}).get('runtimeResourceObserved') is True
                and isinstance(row.get('timings',{}).get('peakRssBytes'),int),'ACTUAL_MODEL_RUNTIME_EVIDENCE_REQUIRED')
        baselines={key:value for key,value in record['results'].items() if key!=artifact['id']}
        require(set(baselines)==set(record.get('baselineIds',[])),'INDEPENDENT_EXAM_REQUIRED')
        for baseline_id in sorted(baselines):
            registered=store.get('artifact',baseline_id,{})
            if registered:
                require(registered.get('state') in ('ready','verified') and not registered.get('fixture') and not registered.get('testOnly'),
                    'BASELINE_NO_LONGER_AVAILABLE')
                control.registry.verify(registered)
                extra=None
                if registered.get('comparisonOnly'):
                    from .market_baseline import verify as verify_market
                    official=verify_market(control,baseline_id)
                    extra=[official['id'],official.get('version'),official.get('receiptHash')]
                baseline_proofs.append([baseline_id,registered.get('version'),registered.get('manifestHash'),extra])
            else:
                checkpoint=store.get('foundation_checkpoint',baseline_id,{}) or store.get('foundation_genesis',baseline_id,{})
                require(checkpoint and checkpoint.get('state') in ('completed','ready','verified') and not checkpoint.get('fixture') and not checkpoint.get('testOnly'),
                    'BASELINE_NO_LONGER_AVAILABLE')
                from .checkpoint import verify as verify_checkpoint
                verify_checkpoint(store,checkpoint)
                baseline_proofs.append([baseline_id,checkpoint.get('version'),checkpoint.get('manifestHash'),None])
        gate_key=digest([record['id'],record.get('version'),suite['id'],suite.get('version'),suite['suiteHash']])
        if gate_key not in _GATES:
            gate=subsequent_gate(rows,baselines) if baselines else first_usable_gate(rows)
            require(gate['passed'],'MODEL_QUALITY_REQUIRED')
            if len(_GATES)>=64:_GATES.pop(next(iter(_GATES)))
            _GATES[gate_key]=True
        require(record['gate'].get('runtimeEvidence',{}).get('passed') is True,'MODEL_QUALITY_REQUIRED')
        result.append((record['id'],record.get('version'),suite['id'],suite.get('version'),suite['suiteHash']))
    graduation=store.get('foundation_graduation',certificate.get('qualifiedGraduationId') or '',{})
    require(graduation.get('state')=='qualified','FOUNDATION_QUALIFICATION_REQUIRED')
    qualification=store.get('foundation_evaluation',graduation.get('evaluationId') or '',{})
    require(qualification.get('kind')=='qualification' and qualification.get('state')=='completed'
        and qualification.get('gate',{}).get('passed') is True and not qualification.get('fixture')
        and qualification.get('targetTokenCount',0)>=100000 and qualification.get('independentGroupCount',0)>=20,
        'FOUNDATION_QUALIFICATION_REQUIRED')
    return {'exams':result,'problemFingerprints':sorted(problems),'baselineReferences':baseline_proofs,
        'graduation':(graduation['id'],graduation.get('version')),'qualification':(qualification['id'],qualification.get('version'))}


def _model(control,project_id,release_id=None,artifact_id=None):
    from .origins import FOUNDATION_ORIGINS
    store=control.store;project=store.get('foundation_project',project_id)
    require(not project.get('testOnly') and not project.get('revocationPending') and project.get('state')!='revoked',
        'USABLE_FOUNDATION_MODEL_REQUIRED')
    require(project.get('executionScope')=='pc_validation','PC_VALIDATION_SCOPE_REQUIRED')
    releases=[row for row in store.list('foundation_release') if row.get('projectId',row.get('foundationProjectId'))==project_id
        and (not release_id or row['id']==release_id) and (not artifact_id or row.get('artifactId')==artifact_id)]
    require(releases,'USABLE_FOUNDATION_MODEL_REQUIRED')
    failures=[]
    for row in sorted(releases,key=lambda item:item.get('updatedAt',''),reverse=True):
        try:
            release={**row,**store.get('release',row['id'],{})}
            artifact=store.get('artifact',release['artifactId']);certificate=(store.get('certificate',release.get('certificateId') or '',{})
                or store.get('foundation_certificate',release.get('certificateId') or '',{}))
            require(release.get('state') in ('ready','active','retired') and artifact.get('origin') in FOUNDATION_ORIGINS
                and not artifact.get('comparisonOnly') and not artifact.get('testOnly') and not artifact.get('fixture')
                and artifact.get('state')=='ready' and artifact.get('foundationProjectId')==project_id
                and artifact.get('executionScope')=='pc_validation' and certificate.get('executionScope')=='pc_validation',
                'USABLE_FOUNDATION_MODEL_REQUIRED')
            root=control.registry.verify(artifact);proof=_proof(control,artifact,certificate)
            profile=release.get('profile',{});profile_hash=digest({key:value for key,value in profile.items() if key!='profileHash'})
            require(profile_hash==release.get('profileHash')==certificate.get('profileHash') and profile.get('profileHash')==profile_hash
                and profile.get('templateFormat')=='seed-structured-v1' and profile.get('provider')=='foundation'
                and profile.get('artifactManifestHash')==artifact['manifestHash'] and profile.get('tokenizerHash')==artifact['tokenizerHash']
                and profile.get('tokenizerSha256')==file_hash(root/'tokenizer.json'),'RELEASE_PROFILE_MISMATCH')
            binding={'projectId':project_id,'projectVersion':project.get('version'),
                'policyVersion':store.get('foundation_policy',project_id,{}).get('version'),
                'releaseId':release['id'],'releaseVersion':release.get('version'),'artifactId':artifact['id'],
                'artifactVersion':artifact.get('version'),'artifactManifestHash':artifact['manifestHash'],
                'certificateId':certificate['id'],'certificateVersion':certificate.get('version'),
                'profileHash':profile_hash,'tokenizerHash':artifact['tokenizerHash'],'proof':proof,
                'rights':_rights(control,project)}
            return json.loads(canonical(binding)),release,artifact,certificate
        except Fault as error:failures.append(error.code)
    raise Fault('CURRENT_CERTIFIED_MODEL_REQUIRED',status=409,details={'reasonCodes':sorted(set(failures))})


def _software(control):
    from .acceptance import _latest_valid,source_snapshot
    from .acceptance_checks import verify_software
    source=source_snapshot()['hash'];software=_latest_valid(control.store,'foundation_software_acceptance',source)
    require(software is not None,'CURRENT_SOFTWARE_ACCEPTANCE_REQUIRED')
    verify_software(control.store,software)
    return software['id'],software['evidenceHash']


def _copy_tree(source,destination,budget):
    source=Path(source);destination=Path(destination);total=0
    if not source.exists():return total
    for parent,folders,files in os.walk(source,followlinks=False):
        current=Path(parent);require(not current.is_symlink(),'MODEL_ACCEPTANCE_COPY_UNSAFE')
        folders[:]=[name for name in folders if name not in ('job-results','token-shards','staging')]
        require(all(not (current/name).is_symlink() for name in folders),'MODEL_ACCEPTANCE_COPY_UNSAFE')
        target=destination/current.relative_to(source);target.mkdir(parents=True,exist_ok=True)
        for name in files:
            path=current/name;require(path.is_file() and not path.is_symlink(),'MODEL_ACCEPTANCE_COPY_UNSAFE')
            total+=path.stat().st_size;require(total<=budget,'MODEL_ACCEPTANCE_COPY_LIMIT')
            shutil.copy2(path,target/name)
    return total


def _copy_database(source,target):
    """Never open the original: even a readonly SQLite open can create its WAL."""
    target.parent.mkdir(parents=True,exist_ok=True)
    staging=target.parent/('.snapshot-'+uid());staging.mkdir()
    local=staging/'source.db';original=copy=None
    try:
        before={suffix:file_hash(Path(str(source)+suffix)) if Path(str(source)+suffix).is_file() else None
            for suffix in ('','-wal')}
        for suffix,checksum in before.items():
            if checksum is not None:
                shutil.copy2(Path(str(source)+suffix),Path(str(local)+suffix))
                require(file_hash(Path(str(local)+suffix))==checksum,'MODEL_ACCEPTANCE_SOURCE_CHANGED')
        after={suffix:file_hash(Path(str(source)+suffix)) if Path(str(source)+suffix).is_file() else None
            for suffix in ('','-wal')}
        require(before==after,'MODEL_ACCEPTANCE_SOURCE_CHANGED')
        original=apsw.Connection(str(local));copy=apsw.Connection(str(target))
        with copy.backup('main',original,'main') as operation:
            while not operation.done:operation.step(256)
    finally:
        if copy:copy.close()
        if original:original.close()
        shutil.rmtree(staging)


def _clone(control,work,maximum=8*1024**3):
    require(work.resolve().is_relative_to((_evidence_root(control)/'work').resolve()) and not work.exists(),'MODEL_ACCEPTANCE_COPY_UNSAFE')
    available=shutil.disk_usage(work.parent.parent).free-1024**3
    require(available>0,'DISK_LOW');maximum=min(maximum,available)
    work.mkdir(parents=True);copied=0
    for name in ('artifacts','foundation','releases','snapshots'):
        copied+=_copy_tree(control.store.root/name,work/'control'/name,maximum-copied)
    for name in ('blobs','exports','indexes','rights'):
        copied+=_copy_tree(control.bridge.store.root/name,work/'rag'/name,maximum-copied)
    _copy_database(control.store.root/'metadata/llm.db',work/'control/metadata/llm.db')
    _copy_database(control.bridge.store.db_path,work/'rag/metadata/seed.db')
    return {'copiedBytes':copied,'controlRoot':work/'control','ragRoot':work/'rag'}


@contextmanager
def _environment(values):
    previous={key:os.environ.get(key) for key in values}
    try:
        os.environ.update(values);yield
    finally:
        for key,value in previous.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value


def _stream(response):
    require(response.status_code==200,'LOCAL_MODEL_HTTP_FAILED',httpStatus=response.status_code)
    require(len(response.content)<=256*1024,'LOCAL_MODEL_HTTP_OUTPUT_LIMIT')
    packets=[json.loads(line) for line in response.text.splitlines() if line.strip()]
    require(packets and packets[-1].get('done') is True and not packets[-1].get('error'),'INCOMPLETE_ENGINE_RESPONSE')
    text=''.join(item.get('response','') for item in packets)
    require(text and packets[-1].get('eval_count',0)>0,'EMPTY_MODEL_GENERATION')
    return text,packets[-1]


def _service_copy(control,copy,binding,release):
    import httpx
    import uvicorn
    from RAG.api import create_app
    values={'SEED_LLM_ENABLED':'1','SEED_LLM_DATA_DIR':str(copy['controlRoot']),
        'SEED_LLM_WORKER':'0','SEED_LLM_TOKEN_FILE':'','SEED_ACCESS_IP':'127.0.0.1',
        'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
    with _environment(values):app=create_app(copy['ragRoot'],start_worker=False,local_dev=False)
    clone=app.state.llm;clone.execution_scope='pc_validation'
    password=secrets.token_urlsafe(32);app.state.auth.set_password(password)
    sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM);sock.bind(('127.0.0.1',0));sock.listen(32)
    port=sock.getsockname()[1];base='http://127.0.0.1:'+str(port)
    server=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=port,log_level='error',access_log=False,timeout_graceful_shutdown=10))
    thread=threading.Thread(target=lambda:server.run(sockets=[sock]),daemon=True);thread.start()
    result={};selected=[]
    try:
        deadline=time.monotonic()+30
        while not server.started:
            require(thread.is_alive() and time.monotonic()<deadline,'LOCAL_MODEL_HTTP_START_FAILED');time.sleep(.05)
        with httpx.Client(base_url=base,timeout=300,trust_env=False) as client:
            result['authenticationRequired']=client.get('/api/v1/sources').status_code==401
            login=client.post('/api/v1/auth/login',json={'password':password},headers={'Origin':base})
            require(login.status_code==200,'LOCAL_MODEL_AUTH_FAILED')
            headers={'Origin':base,'X-CSRF-Token':login.json()['csrfToken']}
            result['csrfRequired']=client.post('/api/v1/retrieval/search',json={'query':'return'}).status_code==403
            allowed=clone.store.get('foundation_policy',binding['projectId'])['allowedSourceIds']
            search=client.post('/api/v1/retrieval/search',json={'query':'return','limit':2,'sourceIds':allowed},headers=headers)
            require(search.status_code==200 and search.json().get('items'),'RAG_EVIDENCE_REQUIRED')
            profile=release['profile']
            evidence=[{**item,'label':'S'+str(index+1)} for index,item in enumerate(search.json()['items'])]
            messages,citations,context,output=clone.bridge.foundation_context({'content':'Complete a short Python function using the references.','mode':'code'},evidence,profile)
            require(citations,'RAG_EVIDENCE_REQUIRED');selected=[(item.get('sourceId'),item.get('documentId'),item.get('versionId')) for item in citations]
            body={'releaseId':release['id'],'model':profile['model'],'profileHash':profile['profileHash'],
                'messages':messages,'mode':'code','options':{'num_predict':8},'stream':True}
            direct=clone.foundation.runtime.generate(body)
            text,finished=_stream(client.post('/api/v1/llm/foundation/generate',json=body,headers=headers))
            result['directNativeEqual']=text==direct['response'] and finished['eval_count']==direct['eval_count'] and finished['prompt_eval_count']==direct['prompt_eval_count']
            result['profileBound']=finished.get('releaseId')==release['id'] and finished.get('profileHash')==profile['profileHash']
            wrong={**body,'profileHash':'0'*64}
            denied=client.post('/api/v1/llm/foundation/generate',json=wrong,headers=headers)
            result['wrongProfileDenied']=denied.status_code>=400
            original=clone.store.get('foundation_project',binding['projectId'])
            clone.store.put('foundation_project',{'revocationPending':True},binding['projectId'])
            denied=client.post('/api/v1/llm/foundation/generate',json=body,headers=headers)
            result['privateCopyRevocationDenied']=denied.status_code in (403,409,503) and denied.json().get('error',{}).get('code')=='POLICY_REVOKED'
            clone.store.put('foundation_project',{**original,'revocationPending':original.get('revocationPending',False)},binding['projectId'])
            restored,_=_stream(client.post('/api/v1/llm/foundation/generate',json=body,headers=headers))
            result['privateCopyRestorePassed']=restored==text
            result.update(outputHash=digest(text.encode()),nativeOutputHash=digest(direct['response'].encode()),
                generatedTokens=finished['eval_count'],inputTokens=finished['prompt_eval_count'],requestHash=digest(body),
                selectedRagLineage=selected,ragTraceId=search.json().get('traceId'),contextLimit=context,outputBudget=8,
                servedArtifactId=binding['artifactId'],servedProfileHash=profile['profileHash'],
                testCopyProjectId=binding['projectId'],faultInjectionScope='fresh_local_copy_only')
            require(all(result[key] is True for key in ('authenticationRequired','csrfRequired','directNativeEqual','profileBound',
                'wrongProfileDenied','privateCopyRevocationDenied','privateCopyRestorePassed')),'LOCAL_MODEL_SERVICE_ACCEPTANCE_FAILED')
    finally:
        password=None;server.should_exit=True;thread.join(30);sock.close()
        require(not thread.is_alive(),'LOCAL_MODEL_SERVICE_NOT_STOPPED')
        clone.foundation.runtime.expire(force=True);clone.store.close()
    return result


def _role_rows(control,project,branch,snapshot_id):
    """Approved purpose-specific data; a code-only corpus cannot stand in for B/C."""
    import ast
    snapshot=control.store.get('snapshot',snapshot_id)
    require(snapshot.get('state')=='verified' and snapshot.get('familyId')==branch['familyId'],'SNAPSHOT_NOT_VERIFIED_BY_TRAINER')
    path=inside(control.store.root,snapshot['storagePath'])/'examples.json'
    require(path.is_file() and path.stat().st_size<=32*1024**2,'PURPOSE_SNAPSHOT_REQUIRES_BOUNDED_SUBSET')
    role=branch['specialization'];groups=set();tokens=0;positives=negatives=0;hashes=[]
    from .tokenizer import load
    tokenizer=load(control.store,project['tokenizerId'])
    rows=control.snapshots.examples(snapshot_id,'train')
    for row in rows:
        if row.get('branchId') not in (None,branch['id']) or row.get('objective','sft')!='sft':continue
        example=control.bridge.store.get('training_example',row['exampleId'])
        require(example.get('reviewStatus')=='approved' and example.get('permissionStatus')=='allowed'
            and example.get('targetOrigin','human') in ('human','source_span') and not example.get('legacyDemo')
            and not control.bridge.post._example_reasons(example),'PURPOSE_SPECIFIC_APPROVED_SFT_REQUIRED')
        require(example.get('contentHash')==row['hash'],'SNAPSHOT_ROW_HASH_MISMATCH')
        require(example.get('branchId')==branch['id'] or all(
            control.store.get('source_purpose',link['sourceId'],{}).get('specialization')==role for link in row['lineage']),
            'PURPOSE_SPECIFIC_APPROVED_SFT_REQUIRED')
        for link in row['lineage']:
            require(control.bridge.pre.document(link['documentId']).get('currentVersionId')==link['versionId'],'POLICY_REVOKED')
            require(control.bridge.store.get('source',link['sourceId']).get('trainingPolicy')=='allowed','POLICY_REVOKED')
        try:
            if role=='review':
                target=json.loads(row['output'])
                require(isinstance(target,dict) and set(target)=={'findings'} and isinstance(target['findings'],list),
                    'PURPOSE_TARGET_CONTRACT_MISMATCH')
                require(all(isinstance(finding,dict) and set(finding)=={'ruleId','line'} and finding['ruleId'] in
                    ('mutable_default','bare_except','dynamic_eval') and type(finding['line']) is int and finding['line']>0
                    for finding in target['findings']),'PURPOSE_TARGET_CONTRACT_MISMATCH')
                positives+=bool(target['findings']);negatives+=not target['findings']
            else:
                tree=ast.parse(row['output']);functions=[node for node in tree.body if isinstance(node,ast.FunctionDef)]
                require(functions,'PURPOSE_TARGET_CONTRACT_MISMATCH')
                if role=='test':
                    from .domains import _tests_source
                    _tests_source(row['output'])
                else:require(any(node.name!='check' for node in functions),'PURPOSE_TARGET_CONTRACT_MISMATCH')
        except (ValueError,SyntaxError,TypeError):raise Fault('PURPOSE_TARGET_CONTRACT_MISMATCH') from None
        groups.add(row['group']);hashes.append(row['hash']);tokens+=len(tokenizer.encode_ordinary(row['output']))+1
    require(len(hashes)>=32 and len(groups)>=3 and tokens>=4096,'PURPOSE_SPECIFIC_APPROVED_SFT_REQUIRED',
        role=role,approvedTrainExamples=len(hashes),trainGroups=len(groups),targetTokens=tokens)
    require(role!='review' or positives and negatives,'PURPOSE_REVIEW_POSITIVE_AND_NEGATIVE_REQUIRED')
    return {'snapshotId':snapshot_id,'snapshotHash':snapshot['manifestHash'],'role':role,
        'trainExamples':len(hashes),'trainGroups':len(groups),'targetTokens':tokens,'exampleHashes':sorted(hashes)}


def _branches(control,anchor):
    result={}
    for label,(role,_,_) in ROLES.items():
        found=[row for row in control.store.list('branch') if row.get('familyId')==anchor['familyId'] and row.get('specialization')==role]
        require(len(found)==1,'PURPOSE_SPECIFIC_BRANCH_REQUIRED',role=role)
        require(found[0].get('anchorArtifactId')==anchor['id'],'INCOMPATIBLE_BASE')
        result[label]=found[0]
    return result


def _anchor(control,project_id):
    """Select the family anchor even after newer specialist releases exist."""
    candidates=[]
    for family in control.store.list('family'):
        identity=family.get('anchorArtifactId')
        if not identity:continue
        artifact=control.store.get('artifact',identity,{})
        if artifact.get('foundationProjectId')==project_id and artifact.get('kind')=='base':candidates.append(identity)
    require(candidates,'CERTIFIED_SCRATCH_ANCHOR_REQUIRED')
    for identity in sorted(candidates):
        try:return _model(control,project_id,artifact_id=identity)
        except Fault:continue
    raise Fault('CERTIFIED_SCRATCH_ANCHOR_REQUIRED',status=409)


def _matrix(control,project_id,anchor):
    branches=_branches(control,anchor);members={};datasets={};weights={}
    from LLM.merging.service import tensor_map,read_tensor
    import torch
    base=tensor_map(control.registry.verify(anchor))
    for label,branch in branches.items():
        identity=branch.get('candidateArtifactId') or branch.get('currentArtifactId')
        require(identity,'SPECIALIST_TRAINING_REQUIRED',role=branch['specialization'])
        artifact=control.store.get('artifact',identity)
        require(artifact.get('kind')=='specialist_checkpoint' and artifact.get('branchId')==branch['id']
            and artifact.get('familyId')==anchor['familyId'] and not artifact.get('fixture') and not artifact.get('testOnly')
            and not artifact.get('comparisonOnly'),'ACTUAL_SPECIALIST_REQUIRED')
        run=control.store.get('training_run',artifact.get('trainingRunId') or '')
        require(run.get('state')=='completed' and run.get('completedSteps',0)>0 and run.get('anchorArtifactId')==anchor['id']
            and run.get('outputArtifactId')==identity and run.get('initialTrainableHash')!=run.get('trainedHash'),
            'ACTUAL_SPECIALIST_TRAINING_REQUIRED')
        data=_role_rows(control,control.store.get('foundation_project',project_id),branch,run['snapshotId'])
        mapping=tensor_map(control.registry.verify(artifact))
        require(mapping.keys()==base.keys() and any(not torch.equal(read_tensor(base,key),read_tensor(mapping,key)) for key in base),
            'SPECIALIST_WEIGHTS_UNCHANGED')
        members[label]=artifact;datasets[label]=data;weights[label]=mapping
    require(len({digest(data['exampleHashes']) for data in datasets.values()})==3,'PURPOSE_DATA_REUSED')
    outputs={''.join((label,)):artifact for label,artifact in members.items()};parity={}
    for labels in MATRIX:
        if len(labels)==1:continue
        parents=[members[label] for label in labels]
        check=control.merger.preflight({'artifactIds':[parent['id'] for parent in parents],'method':'linear_delta'})
        require(check['compatible'],'INCOMPATIBLE_COMBINATION',reasonCodes=check['reasonCodes'])
        combination=control.store.get('combination',check['recipeHash'],{})
        require(combination.get('state') in ('built','ready') and combination.get('outputArtifactId'),
            'ACTUAL_COMBINATION_REQUIRED',combination=''.join(labels))
        artifact=control.store.get('artifact',combination['outputArtifactId'])
        require(artifact.get('kind')=='merged_checkpoint' and set(artifact.get('parents',[]))=={parent['id'] for parent in parents}
            and artifact.get('recipeHash')==combination['id'],'COMBINATION_LINEAGE_MISMATCH')
        actual=tensor_map(control.registry.verify(artifact));require(actual.keys()==base.keys(),'INCOMPATIBLE_TENSOR_SCHEMA')
        recipe=combination['recipe'];require(digest(recipe)==combination['id'],'MERGE_RECIPE_HASH_MISMATCH')
        for key in actual:
            original=read_tensor(base,key).float();delta=torch.zeros_like(original)
            for member in recipe['members']:
                label=next(label for label in labels if members[label]['id']==member['artifactId'])
                delta+=(read_tensor(weights[label],key).float()-original)*member['coefficient']
            expected=original+delta
            require(torch.equal(read_tensor(actual,key).float(),expected),'COMBINATION_WEIGHT_MISMATCH')
        outputs[''.join(labels)]=artifact;parity[''.join(labels)]={'tensorCount':len(actual),'recipeHash':combination['id'],
            'outputManifestHash':artifact['manifestHash'],'singleCombinedWeightArtifact':True}
    bindings={};used=set();domains={}
    from .domains import authorize
    for labels in MATRIX:
        key=''.join(labels);artifact=outputs[key]
        binding,*_=_model(control,project_id,artifact_id=artifact['id'])
        problems=set(binding['proof']['problemFingerprints']);require(not used&problems,'EVALUATION_PROBLEM_REUSED');used.update(problems)
        required={anchor['id'],*[members[label]['id'] for label in labels if len(labels)>1]}
        for identity,_,_,_,_ in binding['proof']['exams']:
            record=control.store.get('foundation_evaluation',identity)
            require(required<=set(record.get('baselineIds',[])),'SPECIALIST_BASELINE_REQUIRED')
        domain_proofs=[]
        for label in labels:
            _,mode,language=ROLES[label]
            if mode=='code':continue
            approved=authorize(control.store,artifact['id'],mode,language)
            measured=control.store.get('foundation_domain_evaluation',approved['evaluationId'])
            require(measured.get('productionCertified') is True and measured.get('gate',{}).get('passed') is True
                and not measured.get('fixture') and required<=set(measured.get('baselineIds',[])),
                'PURPOSE_INDEPENDENT_QUALITY_REQUIRED')
            suite=control.store.get('foundation_domain_suite',measured['suiteId'])
            require(suite.get('state')=='ready' and suite.get('suiteHash')==measured.get('suiteHash'),'DOMAIN_SUITE_NOT_READY')
            domain_proofs.append((approved['id'],approved.get('version'),measured['id'],measured.get('version'),suite['suiteHash']))
        bindings[key]=binding;domains[key]=domain_proofs
    return {'anchorArtifactId':anchor['id'],'anchorManifestHash':anchor['manifestHash'],
        'branches':{label:(branch['id'],branch.get('version')) for label,branch in branches.items()},
        'purposeData':datasets,'models':bindings,'domainProofs':domains,'combinationParity':parity}


def run_specialization(control,project_id):
    try:
        _local(control);_idle(control);base,release,anchor,certificate=_anchor(control,project_id)
        software=_software(control);matrix=_matrix(control,project_id,anchor)
        return _record(control,10,project_id,{'passed':True,'state':'passed','reasonCodes':[],
            'claimScope':'actual_same_anchor_purpose_specialists_and_seven_independently_qualified_weight_models',
            'binding':base,'matrix':matrix,'softwareEvidence':software,'actualModelGatePassed':True,
            'externalMarketTrainingPerformed':False})
    except (Fault,OSError,ValueError,RuntimeError) as error:return _blocked(control,10,project_id,error)


def _legacy_step(control,project_id,kind,payload):
    """One explicit local CPU step; respect the existing cross-process heavy lease."""
    from .evaluation import _EvaluationGuard
    _idle(control);require(kind in ('train','merge'),'MODEL_ACCEPTANCE_JOB_NOT_ALLOWED')
    with control.store.tx():
        require(not control.store.rows("SELECT id FROM jobs WHERE state IN ('queued','running','paused','pause_requested','cancelling') LIMIT 1"),'HEAVY_WORK_IN_PROGRESS')
        queued=control.store.enqueue(kind,{**payload,'acceptanceProjectId':project_id})
        job=control.store.claim('local-specialization-'+uid(),{kind})
        require(job and job['id']==queued['id'],'MODEL_ACCEPTANCE_JOB_CLAIM_FAILED')
    process=None;began=time.monotonic();last_refresh=0
    try:
        import psutil
        if kind=='train':
            branch=control.store.get('branch',payload['branchId']);artifacts=[branch['anchorArtifactId']]
        else:
            recipe=control.store.get('combination',payload['combinationId'])['recipe'];artifacts=[row['artifactId'] for row in recipe['members']]
        guard=_EvaluationGuard(control,project_id,artifacts,job=job)
        destination=_evidence_root(control)/'workers'/job['id'];destination.mkdir(parents=True)
        output=destination/'result.json';command=[sys.executable,'-X','utf8','-m','LLM.foundation.acceptance_model','worker',
            '--control-root',str(control.store.root),'--rag-root',str(control.bridge.store.root),'--project-id',project_id,
            '--job-id',job['id'],'--owner',job['owner'],'--fence',str(job['fence']),'--result-path',str(output)]
        env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'1',
            'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false'}
        with (destination/'worker.log').open('wb') as log:
            process=subprocess.Popen(command,env=env,stdout=log,stderr=log,cwd=ROOT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));observed=psutil.Process(process.pid)
            while process.poll() is None:
                guard.check();control.store.heartbeat(job)
                if time.monotonic()-last_refresh>=5:
                    guard.refresh_policy();last_refresh=time.monotonic()
                try:rss=observed.memory_info().rss+sum(child.memory_info().rss for child in observed.children(recursive=True))
                except psutil.NoSuchProcess:rss=0
                require(rss<=4*1024**3,'MODEL_ACCEPTANCE_WORKER_MEMORY_LIMIT')
                require(time.monotonic()-began<1800 and log.tell()<8*1024**2,'MODEL_ACCEPTANCE_WORKER_LIMIT')
                time.sleep(.5)
            require(process.wait(timeout=5)==0 and output.is_file(),'MODEL_ACCEPTANCE_WORKER_FAILED')
        result=json.loads(output.read_text('utf-8'));guard.check()
        require(result.get('fence')==job['fence'] and not result.get('error'),'MODEL_ACCEPTANCE_WORKER_FAILED',
            workerError=result.get('error'))
        control.store.finish(job,'completed',result['result'])
        return {'state':'progressed','jobId':job['id'],'jobType':kind,'result':result['result'],'passed':False,
            'actualModelGatePassed':False,'activationPerformed':False,'nasModified':False}
    except Exception as error:
        if process and process.poll() is None:
            try:
                parent=psutil.Process(process.pid)
                for child in parent.children(recursive=True):child.kill()
            except psutil.NoSuchProcess:pass
            process.kill();process.wait(timeout=10)
        try:control.store.finish(job,'failed',error={'code':getattr(error,'code','MODEL_ACCEPTANCE_WORKER_FAILED')})
        except Fault:pass
        raise


def advance_specialization(control,project_id,snapshot_ids):
    """Opt-in single action, never inferred by assess or the reboot callback.

    Approved role datasets are input IDs, never uploaded scores/permissions.
    Quality examinations/certification use existing foundation/domain APIs and
    their unchanged independent-bank/day/week budgets.
    """
    try:
        _local(control);_idle(control);_,_,anchor,_=_anchor(control,project_id)
        _software(control)
        existing=control.store.list('branch');branches={}
        for label,(role,_,_) in ROLES.items():
            found=[row for row in existing if row.get('familyId')==anchor['familyId'] and row.get('specialization')==role]
            require(len(found)<=1,'PURPOSE_SPECIFIC_BRANCH_REQUIRED')
            if not found:
                branch=control.registry.branch({'familyId':anchor['familyId'],'label':label,'specialization':role})
                return {'state':'branch_prepared','branchId':branch['id'],'role':role,'passed':False,
                    'reasonCodes':['PURPOSE_SPECIFIC_APPROVED_SFT_REQUIRED'],'activationPerformed':False,'nasModified':False}
            branches[label]=found[0]
        project=control.store.get('foundation_project',project_id);members={}
        for index,(label,branch) in enumerate(branches.items()):
            identity=branch.get('candidateArtifactId') or branch.get('currentArtifactId')
            if not identity:
                snapshot_id=snapshot_ids.get(label);require(snapshot_id,'PURPOSE_SPECIFIC_APPROVED_SFT_REQUIRED',role=branch['specialization'])
                _role_rows(control,project,branch,snapshot_id)
                config={'rank':8,'loraAlpha':16,'dropout':0.0,'epochs':1,'batchSize':1,'gradientAccumulation':8,
                    'learningRate':.0002,'maxSequence':min(512,anchor.get('trainedContextTokens',128)),
                    'maxSteps':64,'checkpointEvery':16,'seed':42+index,'device':'cpu','objective':'sft'}
                return _legacy_step(control,project_id,'train',{'branchId':branch['id'],'snapshotId':snapshot_id,'config':config})
            members[label]=control.store.get('artifact',identity)
            run=control.store.get('training_run',members[label].get('trainingRunId') or '',{})
            require(members[label].get('kind')=='specialist_checkpoint' and run.get('state')=='completed'
                and run.get('anchorArtifactId')==anchor['id'] and run.get('completedSteps',0)>0,
                'ACTUAL_SPECIALIST_TRAINING_REQUIRED')
            _role_rows(control,project,branch,run['snapshotId'])
        # Float-weight construction can proceed without declaring it usable.
        for labels in MATRIX:
            if len(labels)==1:continue
            parents=[members[label]['id'] for label in labels]
            combo=control.merger.recipe({'artifactIds':parents,'method':'linear_delta'})
            if not combo.get('outputArtifactId'):
                return _legacy_step(control,project_id,'merge',{'combinationId':combo['id']})
        return run_specialization(control,project_id)
    except (Fault,OSError,ValueError,RuntimeError) as error:
        return _blocked(control,10,project_id,error,availableNextSteps=['provide_verified_role_snapshots',
            'run_independent_foundation_release_and_purpose_domain_evaluations','prepare_release_and_domain_certificate',
            'rerun_specialization_acceptance'])


def _worker(control,args):
    job=control.store.job(args.job_id);control.store.assert_lease(job)
    require(job.get('owner')==args.owner and job.get('fence')==args.fence and job.get('state')=='running'
        and job.get('payload',{}).get('acceptanceProjectId')==args.project_id,'LEASE_EXPIRED')
    from .evaluation import _EvaluationGuard
    if job['type']=='train':
        branch=control.store.get('branch',job['payload']['branchId'])
        anchor=control.store.get('artifact',branch['anchorArtifactId'])
        require(not anchor.get('comparisonOnly') and anchor.get('foundationProjectId')==args.project_id,'COMPARISON_ONLY_MODEL')
        guard=_EvaluationGuard(control,args.project_id,[anchor['id']],job=job)
        guard.pin('snapshot',control.store.get('snapshot',job['payload']['snapshotId']))
        _role_rows(control,control.store.get('foundation_project',args.project_id),branch,job['payload']['snapshotId'])
        control.trainer.should_yield=lambda:(guard.check() or False)
        result=control.trainer.run(job)
    else:
        require(job['type']=='merge','MODEL_ACCEPTANCE_JOB_NOT_ALLOWED')
        recipe=control.store.get('combination',job['payload']['combinationId'])['recipe']
        guard=_EvaluationGuard(control,args.project_id,[row['artifactId'] for row in recipe['members']],job=job)
        require(all(control.store.get('artifact',row['artifactId']).get('foundationProjectId')==args.project_id for row in recipe['members']),
            'INCOMPATIBLE_BASE')
        result=control.merger.build(job)
    guard.check();atomic_json(args.result_path,{'fence':job['fence'],'result':result})


def verify(control,record,number,source_hash):
    from .acceptance import verify_evidence
    require(number in (7,10) and record.get('runnerVersion')==RUNNER and record.get('requirementNumber')==number
        and not record.get('testOnly') and not record.get('fixture')
        and record.get('passed') is True and record.get('actualModelGatePassed') is True
        and record.get('executionScope')=='pc_validation' and record.get('activationPerformed') is False
        and record.get('nasModified') is False and record.get('sourceStoreHash')==digest(str(control.store.root.resolve())),
        'ACTUAL_MODEL_REQUIREMENT_EVIDENCE_REQUIRED')
    require(record.get('claimScope')==('actual_local_authenticated_rag_and_foundation_http_native_equivalence' if number==7 else
        'actual_same_anchor_purpose_specialists_and_seven_independently_qualified_weight_models'),'ACTUAL_MODEL_REQUIREMENT_EVIDENCE_REQUIRED')
    verify_evidence(record,expected_source=source_hash);_local(control)
    require(tuple(record.get('softwareEvidence',()))==_software(control),'CURRENT_SOFTWARE_ACCEPTANCE_REQUIRED')
    current,*_= _model(control,record['projectId'],record['binding']['releaseId'])
    require(current==record['binding'],'MODEL_ACCEPTANCE_SOURCE_CHANGED')
    if number==7:
        observations=record.get('observations',{})
        require(record.get('sourceDatabaseDigestsBefore')==record.get('sourceDatabaseDigestsAfter')
            and record.get('sourceArtifactFileHashesBefore')==record.get('sourceArtifactFileHashesAfter')
            and bool(record.get('sourceArtifactFileHashesBefore'))
            and record.get('originalStateModified') is False and record.get('revocationScope')=='fresh_local_copy_only'
            and all(observations.get(key) is True for key in ('authenticationRequired','csrfRequired','directNativeEqual','profileBound',
                'wrongProfileDenied','privateCopyRevocationDenied','privateCopyRestorePassed'))
            and observations.get('generatedTokens',0)>0 and observations.get('selectedRagLineage')
            and observations.get('outputHash')==observations.get('nativeOutputHash')
            and observations.get('servedArtifactId')==current['artifactId'] and observations.get('servedProfileHash')==current['profileHash'],
            'ACTUAL_LOCAL_SERVICE_EVIDENCE_REQUIRED')
    else:
        matrix=record.get('matrix',{});require(set(matrix.get('models',{}))=={''.join(labels) for labels in MATRIX},'ACTUAL_SPECIALIST_MATRIX_REQUIRED')
        # Reread current immutable artifacts/certificates/data. Tensor parity was
        # measured by this runner; manifest hashes bind those same bytes now.
        for label,binding in matrix['models'].items():
            actual,*_=_model(control,record['projectId'],binding['releaseId'])
            require(actual==binding,'MODEL_ACCEPTANCE_SOURCE_CHANGED')
        anchor=control.store.get('artifact',matrix['anchorArtifactId']);control.registry.verify(anchor)
        branches=_branches(control,anchor)
        require(matrix['branches']=={label:[branch['id'],branch.get('version')] for label,branch in branches.items()}
            or matrix['branches']=={label:(branch['id'],branch.get('version')) for label,branch in branches.items()},'MODEL_ACCEPTANCE_SOURCE_CHANGED')
        for label,branch in branches.items():
            stored=matrix['purposeData'][label]
            current_data=_role_rows(control,control.store.get('foundation_project',record['projectId']),branch,stored['snapshotId'])
            require(current_data==stored,'MODEL_ACCEPTANCE_SOURCE_CHANGED')
        for key,parity in matrix['combinationParity'].items():
            combo=control.store.get('combination',parity['recipeHash'])
            require(combo.get('state') in ('built','ready') and digest(combo['recipe'])==parity['recipeHash']
                and combo.get('outputArtifactId')==matrix['models'][key]['artifactId']
                and parity['outputManifestHash']==matrix['models'][key]['artifactManifestHash'],'MODEL_ACCEPTANCE_SOURCE_CHANGED')
        from .domains import authorize
        for key,proofs in matrix['domainProofs'].items():
            artifact_id=matrix['models'][key]['artifactId']
            for label in key:
                _,mode,language=ROLES[label]
                if mode!='code':
                    current_domain=authorize(control.store,artifact_id,mode,language)
                    require(any(row[0]==current_domain['id'] and row[1]==current_domain.get('version') for row in proofs),'MODEL_ACCEPTANCE_SOURCE_CHANGED')
            for _,_,evaluation_id,revision,suite_hash in proofs:
                measured=control.store.get('foundation_domain_evaluation',evaluation_id)
                suite=control.store.get('foundation_domain_suite',measured['suiteId'])
                require(measured.get('state')=='completed' and measured.get('version')==revision and measured.get('gate',{}).get('passed') is True
                    and suite.get('state')=='ready' and suite.get('suiteHash')==suite_hash,'MODEL_ACCEPTANCE_SOURCE_CHANGED')
    return record


def latest(control,project_id,number,source_hash):
    root=_evidence_root(control)
    if not (root/'metadata/llm.db').is_file():return None
    store=Store(root)
    try:
        rows=sorted((row for row in store.list('foundation_model_requirement_'+str(number)) if row.get('projectId')==project_id),
            key=lambda row:row.get('observedAt',''),reverse=True)
        for row in rows:
            try:return verify(control,row,number,source_hash)
            except (Fault,OSError,ValueError,KeyError):continue
    finally:store.close()
    return None


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('service','specialization','advance-specialization','worker'))
    parser.add_argument('--control-root',required=True);parser.add_argument('--rag-root',required=True)
    parser.add_argument('--project-id',required=True);parser.add_argument('--release-id')
    parser.add_argument('--snapshot-code');parser.add_argument('--snapshot-review');parser.add_argument('--snapshot-test')
    parser.add_argument('--job-id');parser.add_argument('--owner');parser.add_argument('--fence',type=int);parser.add_argument('--result-path')
    args=parser.parse_args()
    from LLM.service import Control
    from LLM.integration import RagBridge
    from RAG.core import Store as RagStore
    from RAG.Pre.service import PreService
    from RAG.Post.service import PostService
    control=Control(args.control_root,worker=False,execution_scope='pc_validation');rag=RagStore(args.rag_root)
    try:
        pre=PreService(rag);post=PostService(rag,pre);control.bridge=RagBridge(control,rag,pre,post)
        if args.action=='worker':
            require(args.job_id and args.owner and args.fence is not None and args.result_path,'MODEL_ACCEPTANCE_JOB_NOT_ALLOWED')
            target=Path(args.result_path).resolve()
            require(target.is_relative_to((_evidence_root(control)/'workers'/args.job_id).resolve()),'MODEL_ACCEPTANCE_COPY_UNSAFE')
            try:_worker(control,args)
            except Exception as error:
                atomic_json(target,{'fence':args.fence,'error':{'code':getattr(error,'code','MODEL_ACCEPTANCE_WORKER_FAILED')}})
                raise
            return
        if args.action=='service':result=run_service(control,args.project_id,args.release_id)
        elif args.action=='advance-specialization':result=advance_specialization(control,args.project_id,{
            'A':args.snapshot_code,'B':args.snapshot_review,'C':args.snapshot_test})
        else:result=run_specialization(control,args.project_id)
        print(json.dumps(result,ensure_ascii=False,indent=2))
    finally:control.foundation.runtime.expire(force=True);control.store.close();rag.close()


def run_service(control,project_id,release_id=None):
    """Actual loopback HTTP and actual native model, only after genuine quality."""
    work=None
    try:
        _local(control);_idle(control)
        binding,release,artifact,certificate=_model(control,project_id,release_id)
        software=_software(control);before=_db_hashes(control)
        original_files={name:file_hash(control.registry.directory(artifact)/name) for name in artifact['files']}
        root=_evidence_root(control);(root/'work').mkdir(parents=True,exist_ok=True);work=root/'work'/uid()
        copy=_clone(control,work);observed=_service_copy(control,copy,binding,release)
        require(work.resolve().is_relative_to((_evidence_root(control)/'work').resolve()) and not work.is_symlink(),'MODEL_ACCEPTANCE_COPY_UNSAFE')
        shutil.rmtree(work);work=None
        after=_db_hashes(control)
        require(before==after,'SOURCE_DATABASE_CHANGED_DURING_SERVICE_ACCEPTANCE')
        current,*_= _model(control,project_id,release['id'])
        require(current==binding,'MODEL_ACCEPTANCE_SOURCE_CHANGED')
        current_files={name:file_hash(control.registry.directory(artifact)/name) for name in artifact['files']}
        require(current_files==original_files,'MODEL_ACCEPTANCE_SOURCE_CHANGED')
        require(_software(control)==software,'CURRENT_SOFTWARE_ACCEPTANCE_REQUIRED')
        require(before==_db_hashes(control),'SOURCE_DATABASE_CHANGED_DURING_SERVICE_ACCEPTANCE')
        return _record(control,7,project_id,{'passed':True,'state':'passed','reasonCodes':[],
            'claimScope':'actual_local_authenticated_rag_and_foundation_http_native_equivalence',
            'binding':binding,'softwareEvidence':software,'sourceDatabaseDigestsBefore':before,'sourceDatabaseDigestsAfter':after,
            'sourceArtifactFileHashesBefore':original_files,'sourceArtifactFileHashesAfter':current_files,
            'originalStateModified':False,'revocationScope':'fresh_local_copy_only','actualModelGatePassed':True,
            'observations':observed,'cloneDataBytes':copy['copiedBytes']})
    except (Fault,DomainError,OSError,ValueError,RuntimeError) as error:return _blocked(control,7,project_id,error)
    finally:
        if work and work.exists():
            require(work.resolve().is_relative_to((_evidence_root(control)/'work').resolve()) and not work.is_symlink(),'MODEL_ACCEPTANCE_COPY_UNSAFE')
            try:shutil.rmtree(work)
            except OSError:pass  # A failed private cleanup cannot mutate the original roots.


if __name__=='__main__':main()
