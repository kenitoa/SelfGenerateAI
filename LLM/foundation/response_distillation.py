"""Train-only cross-tokenizer response distillation with actual sandbox receipts.

Teacher outputs retain their origin in a separate signed/policy-bound corpus.
They are never treated as source spans or release/evaluation answers.
"""
from __future__ import annotations

import math
import re
import time
from LLM.core import Fault, require, digest, canonical, now
from .schemas import integer, real
from .distillation import _teacher


def configure_policy(control, project_id, body):
    require(isinstance(body,dict) and body.get('approved') is True, 'TEACHER_POLICY_EXPLICIT_APPROVAL_REQUIRED')
    project = control.foundation.project(project_id)
    approvals = body.get('teacherApprovalIds',[])
    require(isinstance(approvals,list) and 0 < len(approvals) <= 3, 'PINNED_TEACHER_APPROVAL_REQUIRED')
    for identity in approvals:
        approved = control.store.get('foundation_teacher_approval',identity)
        require(approved.get('projectId') == project_id and approved.get('approved') is True
            and approved.get('allowedMode') == 'verified_response_sft','PINNED_TEACHER_APPROVAL_REQUIRED')
        teacher = _teacher(control,approved['teacherArtifactId'])
        require(teacher['manifestHash'] == approved['teacherManifestHash'], 'TEACHER_REVISION_MISMATCH')
    value = {'projectId':project_id,'enabled':body.get('enabled',False) is True,'approved':True,
        'teacherApprovalIds':sorted(set(approvals)),
        'maxTeacherFraction':real(body.get('maxTeacherFraction',.1),'maxTeacherFraction',.001,.1),
        'maxResponses':integer(body.get('maxResponses',20),'maxResponses',1,100),
        'maxOutputTokens':integer(body.get('maxOutputTokens',128),'maxOutputTokens',1,128),
        'maxActiveSeconds':integer(body.get('maxActiveSeconds',900),'maxActiveSeconds',1,900),
        'origin':'verified_response_sft','automaticDownload':False,'ordinaryCorpusTeacherFraction':0.0,
        'testOnly':project.get('testOnly') is True}
    value['policyHash'] = digest(value)
    existing=control.store.get('foundation_teacher_policy',project_id,{})
    if existing.get('policyHash')==value['policyHash']:return existing
    return control.store.put('foundation_teacher_policy',value,project_id)


def _policy(control,project_id):
    policy = control.store.get('foundation_teacher_policy',project_id)
    require(policy.get('enabled') is True and policy.get('approved') is True
        and 0 < policy.get('maxTeacherFraction',0) <= .1,'TEACHER_POLICY_DISABLED')
    return policy


def _source_record(row):
    lineage = row.get('lineage',[])
    require(lineage,'LINEAGE_REQUIRED')
    link = lineage[0]
    return {**row,'id':row.get('versionId',link.get('versionId')),
        'versionId':row.get('versionId',link.get('versionId')),'sourceId':row.get('sourceId',link.get('sourceId')),
        'documentId':row.get('documentId',link.get('documentId')),'trainingPolicy':'allowed',
        'lineageGroups':row['groupIds'],'ragSplit':row['split'],'reviewStatus':'approved','permissionStatus':'allowed'}


def _train_fixture(store,fixture_id,row):
    fixture = store.get('fixture',fixture_id)
    require(fixture.get('split') == 'train' and fixture.get('neverTrain') is not True
        and fixture.get('independent') is True and (fixture.get('cases') or fixture.get('tests')),
        'TRAIN_ONLY_FIXTURE_REQUIRED')
    require(fixture.get('sourceTaskHash') == row['hash'] and fixture.get('prompt') == row['instruction'],
        'TEACHER_SOURCE_TASK_MISMATCH')
    for group in row['groupIds']:
        split = store.get('foundation_split',digest(group),{})
        require(split.get('split','train') == 'train','EVALUATION_DATA_NOT_FOR_TRAINING')
    for suite_kind in ('foundation_suite','suite'):
        for suite in store.list(suite_kind):
            if suite.get('kind') not in ('dev','release','audit','regression'): continue
            for task in suite.get('tasks',[]):
                require(task.get('fixtureId') != fixture_id and not set(task.get('lineageGroups',[])) & set(row['groupIds']),
                    'EVALUATION_DATA_NOT_FOR_TRAINING')
    fingerprint = digest({'prompt':row['instruction'],'codePrefix':fixture.get('codePrefix',''),
        'cases':fixture.get('cases'),'tests':fixture.get('tests')})
    require(not store.get('foundation_eval_problem',fingerprint,{}), 'EVALUATION_DATA_NOT_FOR_TRAINING')
    return fixture


def _receipt_valid(store,record,policy, *, current=True):
    approval = store.get('foundation_teacher_approval',record['teacherApprovalId'])
    require(approval.get('approved') is True and approval.get('projectId') == policy['projectId']
        and approval.get('allowedMode') == 'verified_response_sft'
        and approval.get('teacherManifestHash') == record['teacherManifestHash']
        and approval.get('teacherArtifactId') == record['teacherArtifactId'], 'TEACHER_REVISION_MISMATCH')
    require(record['teacherApprovalId'] in policy['teacherApprovalIds'] and record['teacherPolicyId'] == policy['id']
        and record['teacherPolicyVersion'] == policy['version'],'TEACHER_POLICY_CHANGED')
    teacher = store.get('foundation_checkpoint',record['teacherArtifactId'],{}) or store.get('artifact',record['teacherArtifactId'])
    require(teacher.get('state') in ('ready','verified','completed') and teacher.get('manifestHash') == record['teacherManifestHash'],
        'TEACHER_REVISION_MISMATCH')
    query=store.get('foundation_teacher_query',record['teacherQueryId'])
    require(query.get('purpose')=='train_response_distillation' and query.get('split')=='train'
        and query.get('responseGenerated') is True and query.get('responseHash')==digest(record['output'].encode())
        and query.get('sourceTaskHash')==record['sourceTaskHash']
        and query.get('teacherArtifactId')==record['teacherArtifactId']
        and query.get('teacherManifestHash')==record['teacherManifestHash']
        and query.get('teacherPolicyHash')==policy['policyHash'],'TEACHER_QUERY_RECEIPT_MISMATCH')
    receipt = store.get('verifier_receipt',record['verifierReceiptId'])
    require(receipt.get('passed') is True and receipt.get('state') == 'passed'
        and receipt.get('codeHash') == digest(record['output'].encode())
        and digest(receipt) == record['verifierReceiptHash'],'TEACHER_VERIFIER_RECEIPT_MISMATCH')
    fixture = store.get('fixture',record['fixtureId'])
    require(receipt.get('fixtureHash') == digest(fixture) and fixture.get('split') == 'train', 'TRAIN_ONLY_FIXTURE_REQUIRED')
    _train_fixture(store,record['fixtureId'],{'hash':record['sourceTaskHash'],
        'instruction':record['instruction'],'groupIds':record['groupIds']})
    probe = store.get('verifier_probe',receipt.get('probeReceiptId',''))
    limits = probe.get('limits',{})
    require(probe.get('passed') is True and probe.get('state') == 'passed'
        and probe.get('profileId') == fixture.get('profileId')
        and probe.get('profileHash') == digest(store.get('verifier_profile',fixture.get('profileId','')))
        and receipt.get('environmentDigest') == probe.get('environmentDigest')
        and all(limits.get(key) is True for key in ('networkOnlyLoopback','rootReadonly','capEffZero',
            'noNewPrivileges','dockerSocketAbsent','hostDataMountsAbsent','dockerInspectPassed'))
        and isinstance(limits.get('pids'),int) and 0 < limits['pids'] <= 32
        and isinstance(limits.get('memoryBytes'),int) and 0 < limits['memoryBytes'] <= 256*1024**2,
        'SANDBOX_PROBE_REQUIRED')
    return True


def validate_corpus(store,record):
    """Current approval/registry checks usable by readers and optimizer boundaries."""
    require(record.get('kind') == 'foundation_teacher_corpus','TEACHER_CORPUS_REQUIRED')
    authoritative = store.get('foundation_teacher_corpus',record['id'])
    require(store.get('foundation_project',record['projectId']).get('state') != 'revoked','POLICY_REVOKED')
    require(authoritative.get('manifestHash') == record.get('manifestHash')
        and authoritative.get('state') in ('ready','verified'), 'TEACHER_CORPUS_REVOKED')
    policy = store.get('foundation_teacher_policy',record['projectId'])
    require(policy.get('enabled') is True and policy.get('approved') is True
        and record.get('teacherPolicyHash') == policy.get('policyHash')
        and record.get('teacherPolicyVersion') == policy.get('version')
        and 0 <= record.get('actualTeacherFraction',1) <= policy.get('maxTeacherFraction',0) <= .1,
        'TEACHER_POLICY_CHANGED')
    for approval_id in policy['teacherApprovalIds']:
        approval=store.get('foundation_teacher_approval',approval_id)
        require(approval.get('approved') is True and approval.get('projectId')==record['projectId']
            and approval.get('allowedMode')=='verified_response_sft','TEACHER_POLICY_CHANGED')
        teacher=store.get('artifact',approval['teacherArtifactId'],{}) or store.get('foundation_checkpoint',approval['teacherArtifactId'],{})
        require(teacher.get('state') in ('ready','verified','completed')
            and teacher.get('manifestHash')==approval['teacherManifestHash'],'TEACHER_REVISION_MISMATCH')
    return policy


def validate_row(store,corpus,row):
    policy = validate_corpus(store,corpus)
    if row.get('targetOrigin') == 'teacher': return _receipt_valid(store,row,policy)
    require(row.get('targetOrigin','source_span') not in ('model_generated','synthetic_model'), 'HUMAN_REPLAY_REQUIRED')
    return True


def generate(control,job):
    """Generate candidates from fixed train tasks, verify, then build separate corpus."""
    from . import corpus as corpus_module
    from .tokenizer import load
    store,body = control.store,job['payload']
    project = control.foundation.project(body['projectId']);policy = _policy(control,project['id'])
    require(project.get('testOnly') or body.get('teacherPolicyVersion') == policy['version']
        and body.get('teacherPolicyHash') == policy['policyHash'],'TEACHER_POLICY_CHANGED')
    require(project.get('testOnly') or project['policy'].get('enabled'), 'AUTOMATION_DISABLED')
    source = store.get('foundation_corpus',body['sourceCorpusId'])
    require(source.get('kind','foundation_corpus') == 'foundation_corpus' and source.get('projectId') == project['id']
        and source.get('state') == 'ready' and source.get('objective') == 'sft','APPROVED_SFT_CORPUS_REQUIRED')
    tasks = body.get('tasks',[])
    require(isinstance(tasks,list) and 0 < len(tasks) <= policy['maxResponses'], 'TRAIN_TASKS_REQUIRED')
    require(len({t.get('sourceHash') for t in tasks}) == len(tasks),'DUPLICATE_TEACHER_TASK')
    wanted = {task['sourceHash']:task for task in tasks}; selected = {}; humans = []; human_targets = 0
    tok = load(store,project['tokenizerId'])
    human_bytes = 0
    # Dev is copied only as validation data; no dev prompt reaches a teacher.
    for split in ('train','dev'):
        for row in corpus_module.rows(store,source,split,purpose='distillation_human_train' if split=='train' else 'dev'):
            require(row.get('targetOrigin','source_span') not in ('teacher','model_generated','synthetic_model'), 'HUMAN_REPLAY_REQUIRED')
            require(isinstance(row.get('instruction'),str) and isinstance(row.get('output'),str), 'APPROVED_SFT_CORPUS_REQUIRED')
            human_bytes += len(row['text'].encode());require(human_bytes <= 128*1024**2,'TEACHER_REPLAY_LIMIT')
            humans.append(_source_record(row))
            if split == 'train':
                human_targets += len(tok.encode_ordinary(row['output']))+1
                if row['hash'] in wanted: selected[row['hash']] = row
    require(selected.keys() == wanted.keys(), 'TEACHER_SOURCE_TASK_NOT_TRAIN')
    require(control.sandbox.available(),'VERIFIER_UNAVAILABLE',status=503)
    pinned = store.get('foundation_teacher_approval',body['teacherApprovalId'])
    require(pinned['id'] in policy['teacherApprovalIds'] and pinned.get('allowedMode') == 'verified_response_sft',
        'PINNED_TEACHER_APPROVAL_REQUIRED')
    teacher = _teacher(control,pinned['teacherArtifactId'])
    require(teacher['manifestHash'] == pinned['teacherManifestHash'], 'TEACHER_REVISION_MISMATCH')
    require(teacher.get('format','safetensors') == 'safetensors','LOCAL_FLOAT_TEACHER_REQUIRED')
    teacher_budget = math.floor(human_targets*policy['maxTeacherFraction']/(1-policy['maxTeacherFraction']))
    accepted, excluded, learned_teacher = [], [], 0
    started = time.monotonic()
    for index,task in enumerate(tasks):
        current=store.assert_lease(job)
        require(current['state']=='running','JOB_CANCELLED',status=409)
        require(store.get('foundation_project',project['id']).get('state') != 'revoked','POLICY_REVOKED')
        if not project.get('testOnly'):control.foundation.validate_policy(control.foundation.project(project['id']))
        require(_policy(control,project['id'])['version'] == policy['version'],'TEACHER_POLICY_CHANGED')
        if time.monotonic()-started >= policy['maxActiveSeconds']:break
        row = selected[task['sourceHash']]; fixture = _train_fixture(store,task['fixtureId'],row)
        messages = [{'role':'user','content':row['instruction']}]
        query_id=digest([job['id'],row['hash'],teacher['manifestHash']])
        store.put('foundation_teacher_query',{'projectId':project['id'],'runId':job['id'],'split':'train',
            'purpose':'train_response_distillation','sourceTaskHash':row['hash'],'sourceCorpusId':source['id'],
            'sourceCorpusHash':source['manifestHash'],'teacherArtifactId':teacher['id'],
            'teacherManifestHash':teacher['manifestHash'],'promptHash':digest(row['instruction'].encode()),
            'templateHash':teacher.get('templateHash'),'teacherPolicyHash':policy['policyHash']},query_id)
        if teacher.get('foundationProjectId') or teacher.get('origin') in ('random_init','architecture_growth'):
            text,timing = control.foundation.runtime.text(teacher['id'],messages,policy['maxOutputTokens'],purpose='distillation_train')
        else:
            prompt = control.inference.messages_prompt(teacher['id'],messages)
            text,timing = control.inference.text(teacher['id'],prompt,policy['maxOutputTokens'])
        from .corpus import secret_reasons
        if secret_reasons(text):
            excluded.append({'sourceHash':row['hash'],'reasonCode':'TEACHER_RESPONSE_SECRET'});continue
        from .evaluation import _code
        try:answer = _code(text)
        except Fault:
            excluded.append({'sourceHash':row['hash'],'reasonCode':'NON_CODE_TEACHER_RESPONSE'});continue
        store.put('foundation_teacher_query',{'responseHash':digest(answer.encode()),'responseGenerated':True},query_id)
        proof = control.sandbox.verify(answer,fixture)
        if proof.get('passed') is not True:
            excluded.append({'sourceHash':row['hash'],'reasonCode':'TEACHER_OUTPUT_FAILED_INDEPENDENT_TESTS'});continue
        target_count = len(tok.encode_ordinary(answer))+1
        if learned_teacher+target_count > teacher_budget:
            excluded.append({'sourceHash':row['hash'],'reasonCode':'TEACHER_FRACTION_BUDGET'});continue
        record = {**_source_record(row),'text':canonical({'instruction':row['instruction'],'output':answer}),
            'output':answer,'split':'train','targetOrigin':'teacher','teacherArtifactId':teacher['id'],
            'teacherManifestHash':teacher['manifestHash'],'verifierReceiptId':proof['id'],
            'verifierReceiptHash':digest(proof),'sourceTaskHash':row['hash'],'teacherApprovalId':pinned['id'],
            'probeReceiptId':proof.get('probeReceiptId'),'teacherPolicyId':policy['id'],'teacherPolicyVersion':policy['version'],
            'fixtureId':fixture['id'],'generationProfileHash':digest([teacher['manifestHash'],teacher.get('templateHash'),policy['maxOutputTokens']]),
            'teacherQueryId':query_id,
            'generationPurpose':'train_response_distillation','teacherTiming':timing}
        _receipt_valid(store,record,policy)
        accepted.append(record);learned_teacher += target_count
        store.progress(job,'response_distillation',index+1,len(tasks),'train_tasks',accepted=len(accepted))
    require(accepted,'NO_VERIFIED_DISTILLATION_EXAMPLES')
    def validate(record):
        require(record['sourceTaskHash'] in selected,'TEACHER_SOURCE_TASK_NOT_TRAIN')
        _train_fixture(store,record['fixtureId'],selected[record['sourceTaskHash']])
        return _receipt_valid(store,record,policy)
    corpus = corpus_module.build(store,project,[*humans,*accepted],{'objective':'sft',
        'teacherPolicyHash':policy['policyHash'],'teacherPolicyId':policy['id'],'teacherPolicyVersion':policy['version']},
        teacher_validator=validate,record_kind='foundation_teacher_corpus')
    audit = store.put('foundation_response_distillation',{'projectId':project['id'],'sourceCorpusId':source['id'],
        'sourceCorpusHash':source['manifestHash'],'teacherArtifactId':teacher['id'],'teacherManifestHash':teacher['manifestHash'],
        'teacherApprovalId':pinned['id'],'teacherPolicyHash':policy['policyHash'],'corpusId':corpus['id'],
        'acceptedCount':len(accepted),'excluded':excluded,'teacherTargetTokens':learned_teacher,'humanTargetTokens':human_targets,
        'actualTeacherFraction':learned_teacher/(human_targets+learned_teacher),'state':'verified_candidates',
        'servingEligible':False,'qualityGateRequired':True,'productionCertified':False},job['id'])
    return {'state':'completed','corpusId':corpus['id'],'auditId':audit['id'],'teacherOriginPreserved':True,
        'qualityGateRequired':True,'trainingStarted':False,'servingEligible':False}


def mix_blocks(factory, *, teacher_fraction=.1, max_group_blocks=32, max_group_target_tokens=8192, report=None):
    """Yield atomic optimizer envelopes; each envelope learns at most 10% teacher.

    The trainer must process an envelope in one optimizer update, splitting its
    blocks into configured CPU microbatches. Never truncate an SFT answer or
    commit a partial envelope. Two bounded passes avoid unbounded tee caches.
    """
    require(callable(factory) and 0 < teacher_fraction <= .1, 'TEACHER_MIX_CONFIG_INVALID')
    integer(max_group_blocks,'maxGroupBlocks',2,64)
    integer(max_group_target_tokens,'maxGroupTargetTokens',1,65536)
    humans = (b for b in factory() if b.get('targetOrigin','source_span') != 'teacher')
    teachers = (b for b in factory() if b.get('targetOrigin') == 'teacher')
    result = report if report is not None else {}
    total_human, total_teacher, dropped, index = 0,0,0,0
    def envelope(blocks,teacher_targets):
        nonlocal total_human,total_teacher,index
        targets = sum(b['targetCount'] for b in blocks)
        require(targets > 0 and teacher_targets/targets <= teacher_fraction+1e-12, 'TEACHER_ACTUAL_BATCH_FRACTION_EXCEEDED')
        total_teacher += teacher_targets; total_human += targets-teacher_targets
        record = {'mixGroup':True,'blocks':blocks,'targetCount':targets,'teacherTargetTokens':teacher_targets,
            'humanTargetTokens':targets-teacher_targets,'actualTeacherFraction':teacher_targets/targets,
            'groupHash':digest([[b['blockHash'],b.get('targetOrigin','source_span')] for b in blocks]),'index':index}
        index += 1
        result.update(selectedHumanTargetTokens=total_human,selectedTeacherTargetTokens=total_teacher,
            actualTeacherFraction=total_teacher/(total_human+total_teacher),droppedTeacherTargetTokens=dropped)
        return record
    for teacher in teachers:
        need = math.ceil(teacher['targetCount']*(1-teacher_fraction)/teacher_fraction)
        group, count = [],0
        while count < need and len(group) < max_group_blocks-1:
            human = next(humans,None)
            if human is None: break
            require(human.get('targetOrigin','source_span') not in ('model_generated','synthetic_model'), 'HUMAN_REPLAY_REQUIRED')
            group.append(human);count += human['targetCount']
            if count+teacher['targetCount'] > max_group_target_tokens: break
        if count >= need and count+teacher['targetCount'] <= max_group_target_tokens:
            yield envelope([*group,teacher],teacher['targetCount'])
        else:
            dropped += teacher['targetCount']
            result['deficitReasonCodes'] = ['INSUFFICIENT_HUMAN_TARGETS_FOR_ATOMIC_MIX']
            if group: yield envelope(group,0)
    pending = []
    for human in humans:
        pending.append(human)
        if len(pending) >= min(8,max_group_blocks):
            yield envelope(pending,0);pending=[]
    if pending: yield envelope(pending,0)
    result['droppedTeacherTargetTokens'] = dropped
