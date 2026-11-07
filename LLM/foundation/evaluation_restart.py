"""Explicit PC-only, same-examination resource restart.

This is an operator action, not a general cancelled-job resume API. It never
changes an examination payload, a committed answer, or consumption ownership.
File verification and process inspection take place outside write transactions.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import time

from LLM.core import HEAVY_JOBS, digest, inside, now, require, uid

ENV_KEYS = tuple('SEED_EVALUATION_' + name for name in
                 ('JOB_ID', 'OWNER', 'FENCE', 'CONTEXT_HASH', 'CPU_THREADS', 'SCOPE'))
FORBIDDEN = frozenset(('revoked', 'quarantined', 'failed', 'archived', 'deleted', 'reclaiming', 'evicted'))
RESTART_KIND = 'foundation_evaluation_restart'
RUNTIME_KIND = 'foundation_evaluation_runtime'
PREFLIGHT_KIND = 'foundation_evaluation_preflight_retry'
PREFLIGHT_POINTER = 'foundation_evaluation_preflight_runtime'
MAX_PREFLIGHT_RETRIES = 3


def _threads(value):
    require(type(value) is int and 1 <= value <= 4, 'EVALUATION_CPU_THREADS_INVALID')
    return value


def _project_semantic(record):
    # Cancellation changes only these operational fields. Backend receipts are
    # freshly observed on child entry; their immutable platform fields remain.
    value = {k:v for k,v in record.items() if k not in ('version', 'updatedAt', 'stage', 'reasonCodes')}
    if isinstance(value.get('backendReceipt'), dict):
        value['backendReceipt'] = {k:v for k,v in value['backendReceipt'].items()
                                   if k not in ('observedAt', 'receiptHash')}
    return value


def _scope(store, job, execution_scope):
    require(execution_scope == 'pc_validation', 'PC_RELEASE_EVALUATION_REQUIRED', status=403)
    require(job['type'] == 'foundation_evaluate', 'PC_RELEASE_EVALUATION_REQUIRED', status=403)
    body = job['payload']
    project = store.get('foundation_project', body.get('projectId') or '', {})
    suite = store.get('foundation_suite', body.get('suiteId') or '', {})
    require(project.get('executionScope') == 'pc_validation' and not project.get('testOnly')
            and not project.get('fixture') and not suite.get('fixture') and suite.get('kind') == 'release'
            and suite.get('state') == 'ready' and not store.get('foundation_domain_suite', suite.get('id') or '', {}),
            'PC_RELEASE_EVALUATION_REQUIRED', status=403)
    require(project.get('state') not in FORBIDDEN and project.get('stage') != 'revoked'
            and not project.get('revocationPending'), 'POLICY_REVOKED', status=409)
    require(suite.get('suiteHash') == digest({k:suite[k] for k in ('kind', 'tasks', 'fixture')}),
            'EVALUATION_SUITE_CHANGED', status=409)
    return project, suite


def _implementation():
    from .acceptance import source_snapshot
    return source_snapshot()['hash']


def _quality_contract():
    """Gate/decoding logic is immutable; thread-controller source is separate."""
    import inspect
    from . import evaluation
    from .runtime import FoundationRuntime
    from LLM.inference.runtime import Inference
    functions = (evaluation.wilson, evaluation._group_results, evaluation.first_usable_gate,
        evaluation.paired_interval, evaluation.subsequent_gate, evaluation._code,
        evaluation.FoundationEvaluator.evaluate, FoundationRuntime._tokens,
        FoundationRuntime.text, Inference.messages_prompt, Inference.text)
    return {function.__module__+'.'+function.__qualname__:digest(inspect.getsource(function).encode())
            for function in functions}


def _processes(store, job):
    """Identify the exact old child and its owning runner, without storing args."""
    import psutil
    matches = []
    for process in psutil.process_iter(['pid', 'cmdline', 'create_time', 'ppid']):
        try:
            args = process.info['cmdline'] or []
            def option(name):
                return args[args.index(name)+1] if name in args and args.index(name)+1 < len(args) else None
            if ('LLM.foundation.worker' not in args or option('--job-id') != job['id']
                    or option('--owner') != job['owner'] or option('--fence') != str(job['fence'])):
                continue
            directory = option('--data-dir')
            require(directory and Path(directory).resolve() == store.root, 'EVALUATION_WORKER_IDENTITY_MISMATCH')
            parent = process.parent()
            require(parent is not None and parent.pid != os.getpid(), 'EVALUATION_EXTERNAL_RUNNER_REQUIRED')
            matches.append([{'pid':process.pid, 'created':process.create_time(), 'role':'child'},
                            {'pid':parent.pid, 'created':parent.create_time(), 'role':'runner'}])
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
        except psutil.AccessDenied:
            # Unrelated protected processes are not proof of the job's identity.
            continue
    require(len(matches) == 1, 'EVALUATION_WORKER_IDENTITY_REQUIRED', status=409)
    return matches[0]


def _processes_gone(store, job, identities):
    import psutil
    require(isinstance(identities, list) and {r.get('role') for r in identities} == {'child', 'runner'},
            'EVALUATION_WORKER_IDENTITY_REQUIRED')
    for identity in identities:
        try:
            process = psutil.Process(identity['pid'])
            require(process.create_time() != identity['created'] or not process.is_running(),
                    'EVALUATION_OLD_WORKER_ONLINE', status=409)
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            pass
        except psutil.AccessDenied:
            require(False, 'EVALUATION_PROCESS_STATE_UNKNOWN', status=409)
    # Catch any still-running replacement child for the old owner/fence as well.
    for process in psutil.process_iter(['cmdline']):
        try:
            args = process.info['cmdline'] or []
            if 'LLM.foundation.worker' in args and '--job-id' in args:
                require(args[args.index('--job-id')+1] != job['id'], 'EVALUATION_OLD_WORKER_ONLINE', status=409)
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue


def _verify_files(control, bindings):
    from . import checkpoint, corpus, tokenizer
    for item in bindings:
        kind, record = item['kind'], item['record']
        require(record.get('state') not in FORBIDDEN and not record.get('revocationPending'), 'POLICY_REVOKED', status=409)
        if kind in ('foundation_checkpoint', 'foundation_genesis'):
            checkpoint.verify(control.store, record)
        elif kind in ('foundation_corpus', 'foundation_teacher_corpus'):
            corpus.verify(control.store, record)
        elif kind == 'foundation_tokenizer':
            tokenizer.load(control.store, record)
        elif kind == 'artifact':
            control.registry.verify(record)
            if record.get('comparisonOnly'):
                from .market_baseline import verify
                verify(control, record['id'])


def _rag_snapshot(control):
    bridge = getattr(control, 'bridge', None)
    require(bridge is not None, 'RAG_POLICY_GATEWAY_REQUIRED', status=503)
    store = bridge.store
    before = [store.setting('policyVersion', 0), store.setting('trainingDataRevision', 0)]
    # Large document/version changes are fenced by trainingDataRevision. Rights
    # receipts and source-unit metadata additionally receive exact revision pins.
    pins = []
    for row in store.read("SELECT kind,id,body,updated_at FROM records WHERE kind IN ('source','source_unit','source_rights_receipt','training_snapshot') ORDER BY kind,id"):
        body = json.loads(row['body'])
        body = {k:v for k,v in body.items() if k not in ('validatedDataRevision', 'validatedPolicyVersion', 'updatedAt')}
        pins.append({'kind':row['kind'], 'id':row['id'], 'hash':digest(body),
                     'updated':None if row['kind'] == 'training_snapshot' else row['updated_at']})
    require(before == [store.setting('policyVersion', 0), store.setting('trainingDataRevision', 0)],
            'EVALUATION_CONTEXT_CHANGED', status=409)
    return {'epochs':before, 'pins':pins}


def _snapshot(control, job):
    """Heavy verification first; returned pins are metadata-only CAS inputs."""
    store = control.store
    require(store.db.get_autocommit(), 'EVALUATION_TRANSACTION_BOUNDARY_REQUIRED')
    project, suite = _scope(store, job, getattr(control, 'execution_scope', 'nas'))
    control.foundation.validate_policy(project)
    project, suite = _scope(store, job, control.execution_scope)
    bindings = {}; projects = {}
    def add(kind, record):
        key = (kind, record.get('id'))
        require(key[1], 'EVALUATION_DEPENDENCY_MISSING')
        if key in bindings:return
        require(record.get('state') not in FORBIDDEN and not record.get('revocationPending'), 'POLICY_REVOKED', status=409)
        bindings[key] = {'kind':kind, 'record':record}
    def add_project(identity):
        if identity in projects:return
        value = store.get('foundation_project', identity)
        require(value.get('executionScope') == 'pc_validation' and value.get('state') not in FORBIDDEN
                and value.get('stage') != 'revoked' and not value.get('revocationPending'), 'POLICY_REVOKED', status=409)
        projects[identity] = value
        add('foundation_policy', store.get('foundation_policy', identity))
        for field in ('activeCorpusId', 'activeSftCorpusId'):
            if value.get(field):add('foundation_corpus', store.get('foundation_corpus', value[field]))
        if value.get('tokenizerId'):add('foundation_tokenizer', store.get('foundation_tokenizer', value['tokenizerId']))
    def artifact(identity):
        from .evaluation import _artifact_record
        kind, value = _artifact_record(store, identity)
        if (kind,identity) in bindings:return
        require(not value.get('fixture') and not value.get('testOnly'), 'FIXTURE_NOT_FOR_PRODUCTION')
        add(kind,value)
        owner = value.get('projectId', value.get('foundationProjectId'))
        if owner:add_project(owner)
        token = value.get('tokenizerId', value.get('tokenizerArtifactId'))
        if token:add('foundation_tokenizer',store.get('foundation_tokenizer',token))
        data = value.get('corpusId',value.get('corpusSnapshotId'))
        if data:add('foundation_corpus',store.get('foundation_corpus',data))
        for field in ('sourceCheckpointId','parentCheckpointId'):
            if value.get(field):artifact(value[field])
        if value.get('comparisonOnly'):
            from .market_baseline import verify
            add('foundation_market_baseline',verify(control,identity))
    add_project(project['id'])
    candidate = job['payload'].get('artifactId', job['payload'].get('checkpointId'))
    require(candidate, 'ARTIFACT_NOT_FOUND')
    artifact(candidate)
    # Include every baseline actually represented by the payload or release
    # context, including evaluator-added prior champion/growth parent baselines.
    baselines = list(dict.fromkeys(job['payload'].get('baselineIds', [])))
    if project.get('origin') == 'architecture_growth' and project.get('sourceCheckpointId'):
        baselines = list(dict.fromkeys([project['sourceCheckpointId'],*baselines]))
    slot = store.get('slot', 'champion', {})
    active = store.get('release', slot.get('activeReleaseId') or '', {})
    if project.get('origin') == 'architecture_growth' and active.get('artifactId'):
        from .evaluation import _artifact_record
        _,current = _artifact_record(store,active['artifactId'])
        if current.get('foundationProjectId') or current.get('origin') in ('random_init','architecture_growth'):
            baselines = list(dict.fromkeys([active['artifactId'],*baselines]))
    releases = []
    for record in store.list('foundation_release'):
        if record.get('projectId', record.get('foundationProjectId')) == project['id']:
            add('foundation_release', record)
            mirror = store.get('release',record['id'],{})
            if mirror:add('release',mirror)
            combined = {**record,**mirror}
            certificate = combined.get('certificateId')
            if certificate:
                for kind in ('certificate','foundation_certificate'):
                    selected = store.get(kind,certificate,{})
                    if selected:
                        add(kind,selected)
                        break
            if combined.get('activatedAt') or combined['id'] == slot.get('activeReleaseId') or combined.get('state') == 'active':
                releases.append(combined)
    releases.sort(key=lambda r:(r['id']==slot.get('activeReleaseId'),r.get('activatedAt',r.get('createdAt',''))),reverse=True)
    if releases:baselines.insert(0,releases[0]['artifactId'])
    identities = list(dict.fromkeys([candidate,*baselines]))
    for identity in identities:artifact(identity)
    add('foundation_suite', suite)
    for task in suite['tasks']:
        fixture = store.get('fixture',task['fixtureId'])
        require(digest(fixture) == task.get('fixtureHash') and fixture.get('independent') is True
                and not fixture.get('fixture') and not fixture.get('testOnly'), 'EVALUATION_FIXTURE_CHANGED')
        add('fixture', fixture)
        add('verifier_profile', store.get('verifier_profile',fixture['profileId']))
        for group in task.get('lineageGroups',[]):
            split = store.get('foundation_split',digest(group),{})
            require(split.get('split') not in ('train','contaminated'),'SPLIT_LEAKAGE')
            if split:add('foundation_split',split)
    _verify_files(control,list(bindings.values()))
    rag = _rag_snapshot(control)
    pins = [{'kind':kind,'id':identity,'version':value['record']['version'], 'hash':digest(value['record'])}
            for (kind,identity),value in sorted(bindings.items())]
    context = {'schemaVersion':1,'jobId':job['id'],'payloadHash':digest(job['payload']),
        'projectId':project['id'],'artifactIds':identities,'suiteId':suite['id'],
        'projects':{key:_project_semantic(value) for key,value in sorted(projects.items())},
        'dependencies':pins,'rag':rag,'releaseContext':{'slot':slot,'active':active},
        'qualityImplementation':_quality_contract(),
        'decoding':{'dtype':'float32','strategy':'greedy_argmax','outputCeiling':128,
                    'taskBudgets':{t['id']:t.get('maxTokens',128) for t in suite['tasks']}}}
    return context


def _cache(store, job, context):
    suite = store.get('foundation_suite',context['suiteId'])
    tasks = {row['id']:row for row in suite['tasks']}
    records = []
    for entry in store.rows("SELECT id,body FROM records WHERE kind='foundation_eval_task' AND json_extract(body,'$.evaluationId')=? ORDER BY id",(job['id'],)):
        value = json.loads(entry['body']); result = value.get('result',{}); task = tasks.get(result.get('taskId'))
        require(task and value.get('artifactId') in context['artifactIds'] and value.get('evaluationId') == job['id']
                and entry['id'] == digest([job['id'],value['artifactId'],task['id']])
                and value.get('version') == 1 and value.get('neverTrain') is True
                and result.get('neverTrain') is True and type(result.get('passed')) is bool
                and type(result.get('syntaxPassed')) is bool and result.get('group') == task['group']
                and result.get('critical') == (task.get('critical') is True)
                and isinstance(result.get('responseHash'),str) and len(result['responseHash']) == 64,
                'EVALUATION_CACHED_RESPONSE_INVALID',status=409)
        records.append({'id':entry['id'],'version':value['version'],'hash':digest(value),
                        'responseHash':result['responseHash']})
    return records


def _consumption(store, job, context):
    suite = store.get('foundation_suite',context['suiteId'])
    keys = [('foundation_eval_shard',digest([suite['suiteHash'],suite['shardId']]))]
    keys += [('foundation_eval_problem',task['problemFingerprint']) for task in suite['tasks'] if not task.get('critical')]
    pins = []
    for kind, identity in sorted(set(keys)):
        row = store.get(kind,identity,{})
        require(row.get('evaluationId') == job['id'] and row.get('projectId') == context['projectId']
                and row.get('state') == 'consumed' and row.get('neverTrain') is True,
                'EVALUATION_CONSUMPTION_CHANGED',status=409)
        if kind == 'foundation_eval_shard':
            require(row.get('suiteHash') == suite['suiteHash'] and row.get('shardId') == suite['shardId'],
                    'EVALUATION_CONSUMPTION_CHANGED',status=409)
        pins.append({'kind':kind,'id':identity,'version':row['version'],'hash':digest(row)})
    _consumption_keys(store,job['id'],pins)
    return pins


def _consumption_keys(store,job_id,pins):
    for kind in ('foundation_eval_shard','foundation_eval_problem'):
        rows = store.rows("SELECT id FROM records WHERE kind=? AND json_extract(body,'$.evaluationId')=? ORDER BY id",(kind,job_id))
        require(rows == [{'id':p['id']} for p in pins if p['kind'] == kind],
                'EVALUATION_CONSUMPTION_CHANGED',status=409)


def _preserved(original, current):
    current = {row['id']:row for row in current}
    require(all(current.get(row['id']) == row for row in original), 'EVALUATION_CACHED_RESPONSE_CHANGED',status=409)


def _check_metadata(store, context, cache, consumption):
    # No file hashing, models, process inspection, or policy refresh here.
    require(not store.get('setting','deploymentMaintenance',{}).get('enabled'),'DEPLOYMENT_MAINTENANCE')
    for pin in [*context['dependencies'],*consumption]:
        rows = store.rows("SELECT json_extract(body,'$.version') version,json_extract(body,'$.state') state,json_extract(body,'$.revocationPending') pending FROM records WHERE kind=? AND id=?",(pin['kind'],pin['id']))
        require(rows and rows[0]['version'] == pin['version'] and rows[0]['state'] not in FORBIDDEN
                and not rows[0]['pending'],'EVALUATION_CONTEXT_CHANGED',status=409)
    _consumption_keys(store,context['jobId'],consumption)
    for identity, expected in context['projects'].items():
        project = store.get('foundation_project',identity,{})
        require(project.get('stage') != 'revoked' and project.get('state') not in FORBIDDEN
                and not project.get('revocationPending') and _project_semantic(project) == expected,
                'EVALUATION_CONTEXT_CHANGED',status=409)
    require(context['releaseContext'] == {'slot':store.get('slot','champion',{}),
        'active':store.get('release',context['releaseContext']['slot'].get('activeReleaseId') or '',{})},
        'EVALUATION_CONTEXT_CHANGED',status=409)
    current = store.rows("SELECT id,json_extract(body,'$.version') version FROM records WHERE kind='foundation_eval_task' AND json_extract(body,'$.evaluationId')=? ORDER BY id",(context['jobId'],))
    require(current == [{'id':row['id'],'version':row['version']} for row in cache],
            'EVALUATION_CACHED_RESPONSE_CHANGED',status=409)


@contextmanager
def _locked(control, context):
    # Fixed RAG -> LLM lock order. Only metadata revisions are read under locks.
    rag = control.bridge.store
    with rag.tx():
        require(context['rag']['epochs'] == [rag.setting('policyVersion',0),rag.setting('trainingDataRevision',0)],
                'EVALUATION_CONTEXT_CHANGED',status=409)
        for pin in context['rag']['pins']:
            if pin['updated'] is None:continue
            row = rag.one('SELECT updated_at FROM records WHERE kind=? AND id=?',(pin['kind'],pin['id']))
            require(row and row['updated_at'] == pin['updated'],'EVALUATION_CONTEXT_CHANGED',status=409)
        with control.store.tx():yield


def _same_job(job, marker):
    require(job['id'] == marker['jobId'] and job['owner'] == marker['sourceOwner']
            and job['fence'] == marker['sourceFence'] and digest(job['payload']) == marker['payloadHash'],
            'EVALUATION_RESTART_JOB_CHANGED',status=409)


def request_restart(control, job_id, cpu_threads=4):
    """Freeze current inputs/answers, record operator intent, request cancellation."""
    _threads(cpu_threads)
    store = control.store; job = store.job(job_id)
    require(job['state'] == 'running' and job.get('owner'), 'EVALUATION_RESTART_NOT_RUNNING',status=409)
    require(not store.get(RUNTIME_KIND,job_id,{}), 'EVALUATION_RESTART_ALREADY_CONFIGURED',status=409)
    context = _snapshot(control,job)
    processes = _processes(store,job)
    cache = _cache(store,job,context); consumption = _consumption(store,job,context)
    marker_id = uid(); implementation = _implementation()
    marker = {'schemaVersion':1,'state':'requested','jobId':job_id,'projectId':context['projectId'],
        'sourceOwner':job['owner'],'sourceFence':job['fence'],'payloadHash':digest(job['payload']),
        'cpuThreads':cpu_threads,'executionScope':'pc_validation','operatorIntent':'approved_pc_resource_change',
        'approvedResourceOnlyControllerDiff':True,'context':context,'contextHash':digest(context),
        'cachedResponses':cache,'cachedResponseHash':digest(cache),'cachedCount':len(cache),'consumption':consumption,
        'consumptionHash':digest(consumption),'workerProcesses':processes,
        'implementationAtRequest':implementation,'requestedAt':now(),'weightsChanged':False,
        'decodingChanged':False,'qualityCriteriaChanged':False,'confirmedAnswersChanged':False}
    with _locked(control,context):
        current = store.job(job_id); _same_job(current,marker)
        require(current['state'] == 'running','EVALUATION_RESTART_NOT_RUNNING',status=409)
        _check_metadata(store,context,cache,consumption)
        result = store.put(RESTART_KIND,marker,marker_id,expected=0)
        control.command(job_id,'cancel')
        store.event('evaluation.resource_restart_requested',job_id,{
            'restartId':marker_id,'cpuThreads':cpu_threads,'contextHash':marker['contextHash'],
            'cachedResponseHash':marker['cachedResponseHash'],'cachedCount':len(cache)})
    return result


def _stopped_snapshot(control, job_id, marker_id):
    store = control.store; marker = store.get(RESTART_KIND,marker_id,{}); job = store.job(job_id)
    require(marker.get('state') in ('requested','stopped') and marker.get('operatorIntent') == 'approved_pc_resource_change'
            and marker.get('executionScope') == 'pc_validation','EVALUATION_RESTART_MARKER_REQUIRED',status=409)
    _threads(marker.get('cpuThreads'))
    require(marker.get('contextHash') == digest(marker.get('context'))
            and marker.get('cachedResponseHash') == digest(marker.get('cachedResponses'))
            and marker.get('consumptionHash') == digest(marker.get('consumption')),
            'EVALUATION_RESTART_MARKER_CHANGED',status=409)
    _same_job(job,marker)
    require(job['state'] == 'cancelled','EVALUATION_OLD_JOB_NOT_CANCELLED',status=409)
    _processes_gone(store,job,marker['workerProcesses'])
    context = _snapshot(control,job)
    require(digest(context) == marker['contextHash'],'EVALUATION_CONTEXT_CHANGED',status=409)
    cache = _cache(store,job,context); _preserved(marker['cachedResponses'],cache)
    if marker['state'] == 'stopped':
        require(marker.get('stoppedResponses') == cache,'EVALUATION_CACHED_RESPONSE_CHANGED',status=409)
    consumption = _consumption(store,job,context)
    require(consumption == marker['consumption'],'EVALUATION_CONSUMPTION_CHANGED',status=409)
    return marker,job,context,cache,consumption


def confirm_stopped(control, job_id, marker_id):
    """Fence lease release/offline metadata only after old OS processes are gone."""
    store = control.store
    marker,job,context,cache,consumption = _stopped_snapshot(control,job_id,marker_id)
    with _locked(control,context):
        current = store.job(job_id); _same_job(current,marker)
        require(current['state'] == 'cancelled','EVALUATION_OLD_JOB_NOT_CANCELLED',status=409)
        require(store.get(RESTART_KIND,marker_id)['version'] == marker['version'],'VERSION_CONFLICT',status=412)
        _check_metadata(store,context,cache,consumption)
        store.db.execute('UPDATE jobs SET lease=0,updated=? WHERE id=?',(now(),job_id))
        worker = store.get('worker',job['owner'],{})
        store.put('worker',{'state':'offline','observedAt':now(),'restartId':marker_id,
                           'offlineProof':'original_child_and_runner_processes_gone'},job['owner'],expected=worker.get('version',0))
        result = store.put(RESTART_KIND,{'state':'stopped','stoppedAt':now(),
            'stoppedResponses':cache,'stoppedResponseHash':digest(cache),'stoppedCount':len(cache)},marker_id,expected=marker['version'])
        store.event('evaluation.resource_restart_stopped',job_id,{'restartId':marker_id,'cachedResponseHash':digest(cache),'cachedCount':len(cache)})
    return result


def resume(control, job_id, marker_id):
    """Queue the identical cancelled job; only its next claim increases fence."""
    store = control.store
    marker,job,context,cache,consumption = _stopped_snapshot(control,job_id,marker_id)
    require(marker['state'] == 'stopped' and job['lease'] == 0,'EVALUATION_STOP_CONFIRMATION_REQUIRED',status=409)
    require(store.get('worker',job['owner'],{}).get('state') == 'offline','EVALUATION_OLD_WORKER_ONLINE',status=409)
    implementation = _implementation()
    with _locked(control,context):
        current = store.job(job_id); _same_job(current,marker)
        require(current['state'] == 'cancelled' and current['lease'] == 0,'EVALUATION_STOP_CONFIRMATION_REQUIRED',status=409)
        require(store.get('worker',job['owner'],{}).get('state') == 'offline','EVALUATION_OLD_WORKER_ONLINE',status=409)
        require(store.get(RESTART_KIND,marker_id)['version'] == marker['version'],'VERSION_CONFLICT',status=412)
        _check_metadata(store,context,cache,consumption)
        require(not store.get(RUNTIME_KIND,job_id,{}),'EVALUATION_RESTART_ALREADY_CONFIGURED',status=409)
        store.put(RUNTIME_KIND,{'schemaVersion':1,'state':'approved','jobId':job_id,'restartId':marker_id,
            'markerVersion':marker['version']+1,
            'payloadHash':marker['payloadHash'],'contextHash':marker['contextHash'],
            'sourceFence':job['fence'],'resumeFence':job['fence']+1,'cpuThreads':marker['cpuThreads'],
            'executionScope':'pc_validation','context':context,'consumption':consumption,
            'implementationAtResume':implementation},job_id,expected=0)
        store.put(RESTART_KIND,{'state':'resumed','resumedAt':now(),'resumedResponses':cache,
            'resumedResponseHash':digest(cache),'resumedCount':len(cache),
            'implementationAtResume':implementation,'nextFence':job['fence']+1,
            'previousJobResultHash':digest(job.get('result')),'previousJobErrorHash':digest(job.get('error'))},marker_id,expected=marker['version'])
        store.db.execute("UPDATE jobs SET state='queued',owner=NULL,lease=0,result=NULL,error=NULL,updated=? WHERE id=?",(now(),job_id))
        store.event('evaluation.resource_restart_resumed',job_id,{'restartId':marker_id,'cpuThreads':marker['cpuThreads'],
            'contextHash':marker['contextHash'],'cachedResponseHash':digest(cache),'cachedCount':len(cache),'fence':job['fence']})
        result = store.job(job_id)
    return {**result,'restartId':marker_id,'contextHash':marker['contextHash'],'cpuThreads':marker['cpuThreads']}


def _approval(store, job):
    record = store.get(RUNTIME_KIND,job['id'],{})
    require(record, 'EVALUATION_RESTART_MARKER_REQUIRED', status=409)
    marker = store.get(RESTART_KIND,record.get('restartId') or '',{})
    require(record.get('version') == 1 and record.get('state') == 'approved'
            and record.get('executionScope') == 'pc_validation' and marker.get('state') == 'resumed'
            and marker.get('version') == record.get('markerVersion')
            and marker.get('operatorIntent') == 'approved_pc_resource_change'
            and marker.get('jobId') == job['id'] and record.get('jobId') == job['id']
            and record.get('contextHash') == marker.get('contextHash') == digest(record.get('context'))
            and record.get('payloadHash') == marker.get('payloadHash') == digest(job['payload'])
            and record.get('sourceFence') == marker.get('sourceFence')
            and record.get('cpuThreads') == marker.get('cpuThreads'), 'EVALUATION_RUNTIME_CONTEXT_CHANGED',status=409)
    return record,marker


def _preflight_runtime(store, job, base, marker):
    pointer = store.get(PREFLIGHT_POINTER,job['id'],{})
    if not pointer:return base
    require(pointer.get('jobId') == job['id'] and pointer.get('restartId') == marker['id']
            and type(pointer.get('retryNo')) is int and 1 <= pointer['retryNo'] <= MAX_PREFLIGHT_RETRIES
            and pointer.get('version') == pointer['retryNo'], 'EVALUATION_PREFLIGHT_RETRY_CHANGED',status=409)
    identity = pointer.get('retryId'); latest = None; expected_hash = pointer.get('receiptHash')
    for number in range(pointer['retryNo'],0,-1):
        receipt = store.get(PREFLIGHT_KIND,identity or '',{})
        body = {k:v for k,v in receipt.items() if k not in ('id','version','createdAt','updatedAt','receiptHash')}
        require(receipt.get('version') == 1 and receipt.get('state') == 'approved'
                and receipt.get('receiptHash') == expected_hash == digest(body)
                and receipt.get('retryNo') == number and receipt.get('jobId') == job['id']
                and receipt.get('restartId') == marker['id'] and receipt.get('markerVersion') == marker['version']
                and receipt.get('baseRuntimeHash') == digest(base)
                and receipt.get('contextHash') == base['contextHash']
                and receipt.get('payloadHash') == base['payloadHash']
                and receipt.get('cachedResponseHash') == marker.get('resumedResponseHash')
                and receipt.get('consumptionHash') == marker.get('consumptionHash')
                and receipt.get('sourceFence') == base['resumeFence']+number-1
                and receipt.get('nextFence') == base['resumeFence']+number
                and receipt.get('cpuThreads') == base['cpuThreads']
                and receipt.get('executionScope') == 'pc_validation'
                and receipt.get('operatorIntent') == 'approved_pc_preflight_infrastructure_retry',
                'EVALUATION_PREFLIGHT_RETRY_CHANGED',status=409)
        if latest is None:latest = receipt
        identity = receipt.get('previousRetryId'); expected_hash = receipt.get('previousRetryHash')
    require(identity is None and expected_hash is None,'EVALUATION_PREFLIGHT_RETRY_CHANGED',status=409)
    return {**base,'resumeFence':latest['nextFence'],
        'implementationAtResume':latest['implementationAtRetry'],'preflightRetryId':latest['id']}


def _held_preflight(store, job, context):
    evaluation = store.get('foundation_evaluation',job['id'],{})
    gate = evaluation.get('gate',{})
    require(job['state'] == 'completed' and job.get('error') is None and job.get('owner')
            and evaluation and job.get('result') == evaluation
            and evaluation.get('state') == 'held' and evaluation.get('kind') == 'release'
            and evaluation.get('fixture') is False and evaluation.get('productionCertified') is False
            and evaluation.get('neverTrain') is True and evaluation.get('results') == {}
            and gate.get('passed') is False and bool(gate.get('reasonCodes'))
            and set(gate['reasonCodes']) == {'SANDBOX_PROBE_REQUIRED'}
            and evaluation.get('projectId') == context['projectId']
            and evaluation.get('suiteId') == context['suiteId']
            and evaluation.get('suiteHash') == store.get('foundation_suite',context['suiteId'])['suiteHash']
            and evaluation.get('artifactId') == context['artifactIds'][0]
            and list(dict.fromkeys([evaluation.get('artifactId'),*evaluation.get('baselineIds',[])])) == context['artifactIds'],
            'EVALUATION_INFRASTRUCTURE_PREFLIGHT_REQUIRED',status=409)
    return evaluation


def _attempt_proof(store, job, base):
    rows = store.rows("SELECT id FROM records WHERE kind='foundation_evaluation_attempt' AND json_extract(body,'$.jobId')=? AND json_extract(body,'$.fence')=?",(job['id'],job['fence']))
    require(len(rows) == 1,'EVALUATION_ATTEMPT_PROOF_REQUIRED',status=409)
    attempt = store.get('foundation_evaluation_attempt',rows[0]['id'])
    measured = store.get('foundation_evaluation_worker_environment',attempt['id'],{})
    for value in (attempt,measured):
        require(value.get('version') == 1 and value.get('jobId') == job['id']
                and value.get('owner') == job['owner'] and value.get('fence') == job['fence']
                and value.get('executionScope') == 'pc_validation' and value.get('cpuThreads') == base['cpuThreads'],
                'EVALUATION_ATTEMPT_PROOF_REQUIRED',status=409)
    require(attempt.get('payloadHash') == base['payloadHash'] and attempt.get('restartId') == base['restartId']
            and attempt.get('contextHash') == base['contextHash']
            and attempt.get('threadChoiceContextHash') == base['contextHash']
            and attempt.get('expectedTorchThreads') == measured.get('torchNumThreads') == base['cpuThreads']
            and attempt.get('expectedInteropThreads') == measured.get('torchNumInteropThreads') == 1
            and measured.get('attemptId') == attempt['id'] and measured.get('attemptHash') == digest(attempt)
            and all(attempt.get('environment',{}).get(key) == measured.get('environment',{}).get(key) == str(base['cpuThreads'])
                    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')),
            'EVALUATION_ATTEMPT_PROOF_REQUIRED',status=409)
    path = inside(store.root,'foundation/job-results/'+job['id']+'/result.json')
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 1024**2,
            'EVALUATION_WORKER_RESULT_REQUIRED',status=409)
    raw = path.read_bytes(); result = json.loads(raw)
    require(result.get('fence') == job['fence'] and result.get('result') == job['result']
            and not result.get('error') and result.get('evaluationAttemptId') == attempt['id']
            and result.get('finalTorchNumThreads') == base['cpuThreads']
            and result.get('finalTorchNumInteropThreads') == 1,'EVALUATION_WORKER_RESULT_REQUIRED',status=409)
    process = store.get('foundation_evaluation_process',attempt['id'],{})
    if process:
        body = {k:v for k,v in process.items() if k not in ('id','version','createdAt','updatedAt','observationHash')}
        require(process.get('version') == 1 and process.get('observationHash') == digest(body)
                and process.get('attemptId') == attempt['id'] and process.get('attemptHash') == digest(attempt)
                and all(process.get(key) == value for key,value in {
                    'jobId':job['id'],'owner':job['owner'],'fence':job['fence'],'cpuThreads':base['cpuThreads'],
                    'executionScope':'pc_validation','contextHash':base['contextHash'],
                    'restartId':base['restartId'],'payloadHash':base['payloadHash']}.items())
                and measured.get('process') == process.get('child'), 'EVALUATION_PROCESS_PROOF_REQUIRED',status=409)
    return {'attempt':attempt,'environment':measured,'process':process or None,
            'workerResult':{'sha256':digest(raw),'sizeBytes':len(raw),'value':result}}


def _preflight_absence(store, job, process_record):
    """Current exclusivity, honestly distinguished from historic PID evidence."""
    import psutil
    if process_record:
        identities = []
        for role in ('runner','child'):
            identity = process_record.get(role,{})
            require(type(identity.get('pid')) is int and type(identity.get('createTime')) in (int,float),
                    'EVALUATION_PROCESS_PROOF_REQUIRED')
            identities.append({'pid':identity['pid'],'created':identity['createTime'],'role':role})
        _processes_gone(store,job,identities)
    for process in psutil.process_iter(['pid','name','cmdline']):
        if process.pid == os.getpid():continue
        try:
            args = process.info.get('cmdline')
            if args is None:
                name = (process.info.get('name') or '').lower()
                require(not (name.startswith('python') or name in ('py','py.exe')),
                        'EVALUATION_PROCESS_STATE_UNKNOWN',status=409)
                continue
            def option(name):
                return args[args.index(name)+1] if name in args and args.index(name)+1 < len(args) else None
            require(option('--job-id') != job['id'],'EVALUATION_OLD_WORKER_ONLINE',status=409)
            directory = option('--data-dir')
            if directory and Path(directory).resolve() == store.root:
                require('--serve' not in args,'EVALUATION_OLD_WORKER_ONLINE',status=409)
        except (psutil.NoSuchProcess,psutil.ZombieProcess):
            continue
        except psutil.AccessDenied:
            require(False,'EVALUATION_PROCESS_STATE_UNKNOWN',status=409)
    return {'historicalPidProofAvailable':bool(process_record),'currentMatchingProcessCount':0,
            'unreadablePythonProcessCount':0,'scope':'all_job_id_processes_and_shared_root_serve',
            'observedAt':now(),'observedAtEpoch':time.time()}


def _fresh_preflight_probes(store, context):
    proofs = []
    for pin in context['dependencies']:
        if pin['kind'] != 'verifier_profile':continue
        profile = store.get('verifier_profile',pin['id'])
        pointer = store.get('verifier_probe_cache',profile['id'],{})
        receipt = store.get('verifier_probe',pointer.get('receiptId') or '',{})
        limits = receipt.get('limits',{})
        require(receipt.get('version') == 1 and receipt.get('passed') is True and receipt.get('state') == 'passed'
                and receipt.get('profileId') == profile['id'] and receipt.get('profileHash') == digest(profile)
                and receipt.get('environmentDigest') == profile.get('image')
                and receipt.get('imageDigest') == profile.get('image') and receipt.get('runtimeImageId')
                and receipt.get('expiresAt',0) > time.time()
                and type(limits.get('memoryBytes')) is int and 0 < limits['memoryBytes'] <= 268435456
                and type(limits.get('pids')) is int and 0 < limits['pids'] <= 32
                and limits.get('uid') == 65534 and limits.get('cpuset') == '0'
                and all(limits.get(key) is True for key in ('networkOnlyLoopback','rootReadonly','capEffZero',
                    'noNewPrivileges','dockerSocketAbsent','workDataAbsent','dockerInspectPassed','hostDataMountsAbsent')),
                'SANDBOX_OS_PROBE_REQUIRED',status=503)
        proofs.append({'profileId':profile['id'],'pointer':pointer,'receipt':receipt})
    require(proofs,'SANDBOX_OS_PROBE_REQUIRED',status=503)
    return proofs


def retry_preflight(control, job_id, marker_id):
    """Retry only a measured, empty-results sandbox preflight on the same exam.

    No historic process identity is invented for older workers. They require a
    current complete process-absence observation; newer PID receipts also fence
    their exact original runner/child identities. Call only after all source
    fixes are frozen and the same registered profiles have fresh kernel proofs.
    """
    store = control.store; job = store.job(job_id)
    base,marker = _approval(store,job)
    require(marker['id'] == marker_id and base['cpuThreads'] == 4,'EVALUATION_RESTART_MARKER_REQUIRED',status=409)
    effective = _preflight_runtime(store,job,base,marker)
    require(job['fence'] == effective['resumeFence'],'EVALUATION_RUNTIME_CONTEXT_CHANGED',status=409)
    context = _snapshot(control,job)
    require(digest(context) == base['contextHash'],'EVALUATION_CONTEXT_CHANGED',status=409)
    evaluation = _held_preflight(store,job,context)
    cache = _cache(store,job,context)
    require(cache == marker.get('resumedResponses') and digest(cache) == marker.get('resumedResponseHash'),
            'EVALUATION_CACHED_RESPONSE_CHANGED',status=409)
    consumption = _consumption(store,job,context)
    require(consumption == marker.get('consumption'),'EVALUATION_CONSUMPTION_CHANGED',status=409)
    pointer = store.get(PREFLIGHT_POINTER,job_id,{})
    previous = store.get(PREFLIGHT_KIND,pointer.get('retryId') or '',{})
    number = pointer.get('retryNo',0)+1
    require(number <= MAX_PREFLIGHT_RETRIES,'EVALUATION_PREFLIGHT_RETRY_LIMIT',status=409)
    attempt = _attempt_proof(store,job,base)
    probes = _fresh_preflight_probes(store,context)
    implementation = _implementation()
    absence = _preflight_absence(store,job,attempt['process'])
    worker = store.get('worker',job['owner'],{})
    receipt_id = uid()
    body = {'schemaVersion':1,'state':'approved','operatorIntent':'approved_pc_preflight_infrastructure_retry',
        'jobId':job_id,'restartId':marker_id,'markerVersion':marker['version'],'retryNo':number,
        'baseRuntimeHash':digest(base),'contextHash':base['contextHash'],'payloadHash':base['payloadHash'],
        'cachedResponseHash':digest(cache),'cachedCount':len(cache),'consumptionHash':digest(consumption),
        'sourceFence':job['fence'],'nextFence':job['fence']+1,'sourceOwner':job['owner'],
        'cpuThreads':4,'executionScope':'pc_validation','implementationAtRetry':implementation,
        'previousRetryId':previous.get('id'),'previousRetryHash':previous.get('receiptHash'),
        'heldEvaluation':evaluation,'attemptProof':attempt,'probeProofs':probes,'processAbsence':absence,
        'historicalPidProofAvailable':absence['historicalPidProofAvailable'],
        'oldLease':job['lease'],'leaseReleased':True,'confirmedAnswersChanged':False,
        'weightsChanged':False,'decodingChanged':False,'qualityCriteriaChanged':False,'observedAt':now()}
    body['receiptHash'] = digest(body)
    with _locked(control,context):
        current = store.job(job_id)
        require(all(current.get(k) == job.get(k) for k in ('type','state','payload','result','error','owner','fence','lease')),
                'EVALUATION_RESTART_JOB_CHANGED',status=409)
        require(_held_preflight(store,current,context) == evaluation,'EVALUATION_PREFLIGHT_RETRY_CHANGED',status=409)
        require(store.get(RUNTIME_KIND,job_id)['version'] == base['version']
                and store.get(RESTART_KIND,marker_id)['version'] == marker['version']
                and store.get(PREFLIGHT_POINTER,job_id,{}).get('version',0) == pointer.get('version',0),
                'EVALUATION_PREFLIGHT_RETRY_CHANGED',status=409)
        _check_metadata(store,context,cache,consumption)
        require(0 <= time.time()-absence['observedAtEpoch'] <= 10,'EVALUATION_PROCESS_STATE_STALE',status=409)
        busy = store.rows("SELECT id,type FROM jobs WHERE id<>? AND state IN ('queued','running','pause_requested','cancelling')",(job_id,))
        require(not any(row['type'] in HEAVY_JOBS for row in busy),'EVALUATION_CONCURRENT_WORK_REQUIRED_OFFLINE',status=409)
        for kind,value in [('foundation_evaluation_attempt',attempt['attempt']),
                           ('foundation_evaluation_worker_environment',attempt['environment']),
                           *([('foundation_evaluation_process',attempt['process'])] if attempt['process'] else [])]:
            require(store.get(kind,value['id'])['version'] == value['version'],'EVALUATION_ATTEMPT_PROOF_REQUIRED',status=409)
        for proof in probes:
            require(store.get('verifier_probe_cache',proof['profileId'])['version'] == proof['pointer']['version']
                    and store.get('verifier_probe',proof['receipt']['id'])['version'] == proof['receipt']['version']
                    and proof['receipt']['expiresAt'] > time.time(),'SANDBOX_OS_PROBE_REQUIRED',status=503)
        store.put('worker',{'state':'offline','observedAt':now(),'preflightRetryId':receipt_id,
            'offlineProof':'current_complete_process_absence','historicalPidProofAvailable':absence['historicalPidProofAvailable']},
            job['owner'],expected=worker.get('version',0))
        receipt = store.put(PREFLIGHT_KIND,body,receipt_id,expected=0)
        store.put(PREFLIGHT_POINTER,{'jobId':job_id,'restartId':marker_id,'retryId':receipt_id,
            'retryNo':number,'receiptHash':receipt['receiptHash']},job_id,expected=pointer.get('version',0))
        store.db.execute("UPDATE jobs SET state='queued',owner=NULL,lease=0,result=NULL,error=NULL,updated=? WHERE id=?",(now(),job_id))
        store.event('evaluation.infrastructure_preflight_retried',job_id,{
            'restartId':marker_id,'preflightRetryId':receipt_id,'retryNo':number,'fence':job['fence'],
            'contextHash':base['contextHash'],'cachedResponseHash':digest(cache),'cachedCount':len(cache),
            'historicalPidProofAvailable':absence['historicalPidProofAvailable']})
        result = store.job(job_id)
    return {**result,'restartId':marker_id,'preflightRetryId':receipt_id,
            'contextHash':base['contextHash'],'cpuThreads':4,
            'historicalPidProofAvailable':absence['historicalPidProofAvailable']}


def _runtime(store, job, scope):
    if not store.get(RUNTIME_KIND,job['id'],{}):return None
    _scope(store,job,scope)
    base,marker = _approval(store,job)
    record = _preflight_runtime(store,job,base,marker)
    require(job['state'] == 'running' and job.get('owner') and job['fence'] == record['resumeFence'],
            'EVALUATION_RUNTIME_CONTEXT_CHANGED',status=409)
    current = store.assert_lease(job)
    require(current['state'] == 'running','JOB_CANCELLED',status=409)
    _threads(record.get('cpuThreads'))
    # CPU-controller edits are separately recorded at resume. Once queued, the
    # approved attempt's complete implementation must remain frozen too.
    require(store.db.get_autocommit(),'EVALUATION_TRANSACTION_BOUNDARY_REQUIRED')
    require(_implementation() == record.get('implementationAtResume'),
            'EVALUATION_IMPLEMENTATION_CHANGED',status=409)
    # These cheap pins remain authoritative in the model child. Normal evaluator
    # guards independently fence every response and final publication as well.
    cache = _cache(store,job,record['context'])
    _preserved(marker.get('resumedResponses',[]),cache)
    _check_metadata(store,record['context'],cache,record['consumption'])
    return record


def runtime_threads(control, job):
    record = _runtime(control.store,job,getattr(control,'execution_scope','nas'))
    return record['cpuThreads'] if record else None


def runtime_metadata(control, job):
    record = _runtime(control.store,job,getattr(control,'execution_scope','nas'))
    require(record is not None,'EVALUATION_RESTART_MARKER_REQUIRED',status=409)
    return {'cpuThreads':record['cpuThreads'],'executionScope':'pc_validation','jobId':job['id'],
        'owner':job['owner'],'fence':job['fence'],'contextHash':record['contextHash'],
        'restartId':record['restartId'],'payloadHash':record['payloadHash'],
        **({'preflightRetryId':record['preflightRetryId']} if record.get('preflightRetryId') else {})}


def child_runtime_threads(store):
    present = [key in os.environ for key in ENV_KEYS]
    if not any(present):return 1
    require(all(present) and os.environ.get('SEED_EXECUTION_SCOPE') == 'pc_validation'
            and os.environ['SEED_EVALUATION_SCOPE'] == 'pc_validation','PC_RELEASE_EVALUATION_REQUIRED',status=403)
    raw_threads = os.environ['SEED_EVALUATION_CPU_THREADS']; raw_fence = os.environ['SEED_EVALUATION_FENCE']
    require(raw_threads in ('1','2','3','4') and raw_fence.isdigit(),'EVALUATION_CPU_THREADS_INVALID')
    job = store.job(os.environ['SEED_EVALUATION_JOB_ID'])
    require(job['owner'] == os.environ['SEED_EVALUATION_OWNER'] and job['fence'] == int(raw_fence),
            'EVALUATION_RUNTIME_CONTEXT_CHANGED',status=409)
    record = _runtime(store,job,'pc_validation')
    require(record is not None and record['contextHash'] == os.environ['SEED_EVALUATION_CONTEXT_HASH']
            and record['cpuThreads'] == int(raw_threads),'EVALUATION_RUNTIME_CONTEXT_CHANGED',status=409)
    return record['cpuThreads']
