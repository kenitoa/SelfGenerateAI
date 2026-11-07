"""Read actual transition evidence; software green never becomes model quality.

No records, scores, certificates, jobs or files are created here. Producers are
the existing trusted curriculum, domain, trainer, evaluator and release paths.
"""
from __future__ import annotations

import json
import math
import re

from LLM.core import Fault,digest,file_hash,inside,require

NUMBERS=frozenset((17,18,20,21))
BAD_STATES=frozenset(('revoked','quarantined','failed','archived','deleted','reclaiming','evicted'))
PATHS={
    17:['Curriculum.configure/advance','FoundationController.calibrate/start_training/evaluate'],
    18:['domains.register_suite/evaluate/certify/authorize','graduation.certify_usable'],
    20:['foundation_distill or verified-response SFT','foundation_format F16 and Q4_K_M',
        'foundation_evaluate and usable certificate for native and each format'],
    21:['growth.experiment/adopt','foundation training/qualification/evaluate','graduation.certify_usable'],
}


def _real(record,*,allow_random_baseline=False):
    require(record and not any(record.get(key) for key in ('fixture','testOnly','fixtureOnly'))
            and (allow_random_baseline or not record.get('randomBaselineOnly')),
            'FIXTURE_NOT_FOR_PRODUCTION',status=409)
    require(record.get('state') not in BAD_STATES and not record.get('revocationPending'),'POLICY_REVOKED',status=409)
    return record


def _integer(value,minimum=1):
    return type(value) is int and value>=minimum


class _Proof:
    def __init__(self,control):
        self.control=control;self.store=control.store;self.records={};self.jobs={};self.checkpoints={}
        self.scope=getattr(control,'execution_scope','nas');self.absent=set()

    def optional(self,kind,identity):
        if not self.store.get(kind,identity,{}):self.absent.add((kind,identity));return {}
        return self.get(kind,identity)

    def mark(self):
        return set(self.records),set(self.jobs),set(self.checkpoints),set(self.absent)

    def rollback(self,mark):
        self.records={k:v for k,v in self.records.items() if k in mark[0]}
        self.jobs={k:v for k,v in self.jobs.items() if k in mark[1]}
        self.checkpoints={k:v for k,v in self.checkpoints.items() if k in mark[2]}
        self.absent=set(mark[3])

    def get(self,kind,identity):
        require(isinstance(identity,str) and identity,'TRANSITION_EVIDENCE_REQUIRED',status=409)
        record=self.store.get(kind,identity,{})
        require(record,'TRANSITION_EVIDENCE_REQUIRED',status=409,kind=kind,id=identity)
        key=(kind,identity);old=self.records.get(key)
        require(not old or old['version']==record['version'],'TRANSITION_EVIDENCE_CHANGED',status=409)
        self.records[key]=record
        return record

    def job(self,identity):
        row=self.store.job(identity)
        require(row.get('state')=='completed','COMPLETED_EXECUTION_REQUIRED',status=409)
        value={key:row.get(key) for key in ('type','state','payload','result','error','fence')}
        require(identity not in self.jobs or self.jobs[identity]==value,'TRANSITION_EVIDENCE_CHANGED',status=409)
        self.jobs[identity]=value
        return row

    def execution(self,kind,record,*,job_type='foundation_evaluate'):
        """The completed worker result is a second durable execution binding."""
        job=self.job(record['id'])
        emitted=job.get('result') or {}
        require(job.get('type')==job_type and emitted.get('id')==record['id']
            and all(record.get(key)==value for key,value in emitted.items()
                    if key not in ('version','updatedAt','createdAt')),
            'ACTUAL_EVALUATION_REQUIRED')
        return job

    def project(self,identity):
        from .origins import assert_project_available,FOUNDATION_ORIGINS
        from .backend import verify
        project=_real(self.get('foundation_project',identity))
        require(project.get('origin') in FOUNDATION_ORIGINS and project.get('stage')!='revoked','FOUNDATION_ORIGIN_REQUIRED')
        require(project.get('executionScope','nas')==self.scope,'EXECUTION_SCOPE_REQUALIFICATION_REQUIRED',status=409)
        assert_project_available(self.store,project)
        self.control.foundation.validate_policy(self.control.foundation.project(identity))
        backend=project.get('backendReceipt')
        require(backend and backend.get('receiptHash'),'BACKEND_CERTIFICATION_REQUIRED')
        verify(backend,strict_platform=self.scope=='nas')
        if project.get('sourceProjectId'):self.project(project['sourceProjectId'])
        return project

    def corpus(self,identity):
        from .corpus import resolve,verify
        row=_real(resolve(self.store,identity));self.get('foundation_corpus',row['id'])
        if row.get('kind')=='foundation_teacher_corpus':self.get('foundation_teacher_corpus',row['id'])
        verify(self.store,row)
        return row

    def artifact(self,identity):
        row=self.store.get('artifact',identity,{})
        if row:
            row=_real(self.get('artifact',identity));self.control.registry.verify(row)
            require(row.get('executionScope','nas')==self.scope,'EXECUTION_SCOPE_REQUALIFICATION_REQUIRED')
            if row.get('foundationProjectId'):self.project(row['foundationProjectId'])
            return row
        return self.checkpoint(identity)[0]

    def checkpoint(self,identity,*,trained=False,allow_random_baseline=False):
        if identity not in self.checkpoints:
            from .checkpoint import verify,_unpack
            kind='foundation_checkpoint' if self.store.get('foundation_checkpoint',identity,{}) else 'foundation_genesis'
            row=_real(self.get(kind,identity),allow_random_baseline=allow_random_baseline);root,manifest=verify(self.store,row)
            _real(manifest)
            tree=json.loads((root/'state.json').read_text('utf-8'))
            node=next((value for key,value in tree.get('items',[]) if key=={'type':'scalar','value':'state'}),None)
            saved=_unpack(node,{})
            require(isinstance(saved,dict),'CHECKPOINT_SCHEMA_INVALID')
            _real(saved,allow_random_baseline=allow_random_baseline)
            for key in ('projectId','origin','executionScope','trustedGrowthReceipt','pretrainedWeightArtifactIds',
                        'runId','parentCheckpointId','stage','completedOptimizerSteps',
                        'committedTargetTokens','trainedContextTokens','distillationMode','teacherManifestHashes',
                        'teacherApprovalIds','teacherArtifactId','teacherManifestHash',
                        'initialTensorDigest','trainedTensorDigest','committedTeacherTargetTokens','committedHumanTargetTokens'):
                require(row.get(key)==saved.get(key),'CHECKPOINT_STATE_MISMATCH',status=409,field=key)
            require(saved.get('executionScope','nas')==self.scope,'EXECUTION_SCOPE_REQUALIFICATION_REQUIRED')
            self.project(row['projectId'])
            token=_real(self.get('foundation_tokenizer',row['tokenizerId']))
            require(token.get('projectId')==row['projectId'] and token.get('tokenizerHash')==row.get('tokenizerHash'),
                    'TOKENIZER_HASH_MISMATCH')
            from .tokenizer import load
            load(self.store,token)
            self.checkpoints[identity]=(row,saved,root,manifest)
        row,saved,root,manifest=self.checkpoints[identity]
        require(allow_random_baseline or not saved.get('randomBaselineOnly'),'TRUE_RANDOM_BASELINE_NOT_A_CANDIDATE')
        if trained:
            require(_integer(saved.get('completedOptimizerSteps')) and _integer(saved.get('committedTargetTokens')),
                    'ACTUAL_TRAINING_REQUIRED',status=409)
        return row,saved,root,manifest

    def training(self,checkpoint_id):
        checkpoint,saved,root,manifest=self.checkpoint(checkpoint_id,trained=True)
        run=_real(self.get('foundation_run',saved.get('runId')));job=self.job(run['id'])
        require(run.get('state')=='completed' and run.get('projectId')==checkpoint['projectId']
            and job['payload'].get('projectId')==checkpoint['projectId']
            and job['type'] in ('foundation_pretrain','foundation_cpt','foundation_sft','foundation_distill')
            and (job.get('result') or {}).get('checkpointId')==checkpoint_id
            and run.get('lastCheckpointId',run.get('checkpointId'))==checkpoint_id,
            'COMPLETED_TRAINING_REQUIRED',status=409)
        require(run.get('completedOptimizerSteps')==saved['completedOptimizerSteps']
            and run.get('committedTargetTokens')==saved['committedTargetTokens'],'TRAINING_COUNTER_MISMATCH')
        corpus=self.corpus(saved['corpusId'])
        require(corpus['projectId']==checkpoint['projectId'] and corpus['manifestHash']==saved.get('corpusHash'),
                'CORPUS_HASH_MISMATCH')
        parent,_,_,parent_manifest=self.checkpoint(saved['parentCheckpointId'])
        require(parent.get('tokenizerHash')==checkpoint.get('tokenizerHash')
            and parent_manifest['files']['model.safetensors']!=manifest['files']['model.safetensors'],
            'TRAINED_WEIGHTS_UNCHANGED')
        ledger=self.get('foundation_consumption',run['id'])
        require(ledger.get('checkpointId')==checkpoint_id and ledger.get('projectId')==checkpoint['projectId']
            and ledger.get('committedTargetTokens')==saved['committedTargetTokens'], 'CONSUMPTION_EVIDENCE_MISMATCH')
        return checkpoint,saved,run,job,corpus

    def finish(self):
        for kind,identity in self.absent:
            require(not self.store.get(kind,identity,{}),'TRANSITION_EVIDENCE_CHANGED',status=409)
        for (kind,identity),old in self.records.items():
            row=self.store.get(kind,identity,{})
            # A mutation killed by a generated test is affirmative evidence;
            # the verifier's expected failed outcome is not a revoked model.
            invalid=row.get('state') in BAD_STATES and not (kind=='verifier_receipt' and row.get('state')=='failed')
            require(row.get('version')==old['version'] and not invalid
                and not row.get('revocationPending'),'TRANSITION_EVIDENCE_CHANGED',status=409)
        for identity,old in self.jobs.items():
            row=self.store.job(identity)
            require(old=={key:row.get(key) for key in old},'TRANSITION_EVIDENCE_CHANGED',status=409)

    def evidence(self):
        return [{'kind':kind,'id':identity,'version':row['version']}
                for (kind,identity),row in sorted(self.records.items())]


def _verifier(proof,receipt_id,fixture,*,passed=None):
    from LLM.verification.broker import pinned
    receipt=proof.get('verifier_receipt',receipt_id)
    require(receipt.get('fixtureHash')==digest(fixture) and receipt.get('expectedAnswersExposed') is False
        and re.fullmatch('[0-9a-f]{64}',receipt.get('codeHash','')),'VERIFIER_EVIDENCE_MISMATCH')
    if passed is not None:
        require(receipt.get('passed') is passed and receipt.get('state')==('passed' if passed else 'failed'),
                'VERIFIER_EVIDENCE_MISMATCH')
    profile=proof.get('verifier_profile',fixture.get('profileId'));probe=proof.get('verifier_probe',receipt.get('probeReceiptId'))
    limits=probe.get('limits',{})
    require(pinned(profile) and probe.get('passed') is True and probe.get('state')=='passed'
        and probe.get('profileId')==profile['id'] and probe.get('profileHash')==digest(profile)
        and receipt.get('environmentDigest')==probe.get('environmentDigest')==profile['image']
        and all(limits.get(key) is True for key in ('networkOnlyLoopback','rootReadonly','capEffZero',
            'noNewPrivileges','dockerSocketAbsent','hostDataMountsAbsent','dockerInspectPassed')),
        'ACTUAL_SANDBOX_PROOF_REQUIRED')
    require(_integer(limits.get('pids')) and limits['pids']<=32
        and _integer(limits.get('memoryBytes')) and limits['memoryBytes']<=256*1024**2,'ACTUAL_SANDBOX_PROOF_REQUIRED')
    return receipt


def _rows(proof,evaluation,suite,identity,*,domain=None):
    rows=evaluation.get('results',{}).get(identity)
    tasks=suite.get('tasks',[])
    require(isinstance(rows,list) and tasks and len(rows)==len(tasks)
        and len({row.get('taskId') for row in rows})==len(rows)
        and {row.get('taskId') for row in rows}=={task['id'] for task in tasks},'EVALUATION_COVERAGE_MISMATCH')
    by_id={task['id']:task for task in tasks}
    for row in rows:
        task=by_id[row['taskId']]
        require(type(row.get('passed')) is bool and type(row.get('syntaxPassed')) is bool
            and row.get('group')==task['group'] and row.get('neverTrain') is True
            and bool(row.get('critical'))==bool(task.get('critical'))
            and row.get('safetyCategory')==task.get('safetyCategory')
            and row.get('criticalFailure') is bool(task.get('critical') and not row['passed']),
            'EVALUATION_ROW_MISMATCH')
        key=digest([evaluation['id'],identity,task['problemFingerprint'],task['id']]) if domain else digest([evaluation['id'],identity,task['id']])
        ledger=proof.get('foundation_domain_task' if domain else 'foundation_eval_task',key)
        require(ledger.get('evaluationId')==evaluation['id'] and ledger.get('artifactId')==identity
            and ledger.get('result')==row and ledger.get('neverTrain') is True,'ACTUAL_EVALUATION_TASK_REQUIRED')
        # A malformed domain answer is retained as a failed assigned task. Its
        # producer records the exception before it can record a response digest.
        require(bool(re.fullmatch('[0-9a-f]{64}',row.get('responseHash','')))
            or (domain and row['passed'] is False and isinstance(row.get('reasonCode'),str)
                and bool(row['reasonCode'])),'ACTUAL_EVALUATION_TASK_REQUIRED')
        for field,hash_field in (('fixtureId','fixtureHash'),('testFixtureId','testFixtureHash')):
            if task.get(field):
                fixed=_real(proof.get('fixture',task[field]))
                require(fixed.get('independent') is True and digest(fixed)==task.get(hash_field)
                    and fixed.get('targetOrigin') not in ('teacher','model_generated'),'EVALUATION_FIXTURE_CHANGED')
        if not domain:
            if row['passed']:_verifier(proof,row.get('receiptId'),proof.get('fixture',task['fixtureId']),passed=True)
        elif domain['purpose']=='tests' and row['passed']:
            receipts=row.get('receiptIds',[])
            require(len(receipts)==1+len(task.get('mutants',[])) and len(receipts)>=3,'MUTATION_EVIDENCE_REQUIRED')
            fixed=proof.get('fixture',task['testFixtureId'])
            for index,receipt_id in enumerate(receipts):_verifier(proof,receipt_id,fixed,passed=index==0)
    return rows


def _evaluation(proof,identity,candidate,*,paired=(),domain=None):
    from .evaluation import first_usable_gate,subsequent_gate
    kind='foundation_domain_evaluation' if domain else 'foundation_evaluation'
    result=_real(proof.get(kind,identity));job=proof.execution(kind,result)
    candidate_record=proof.artifact(candidate)
    project_id=candidate_record.get('foundationProjectId',candidate_record.get('projectId'))
    require(result.get('state')=='completed' and result.get('artifactId')==candidate
        and result.get('projectId')==project_id and result.get('neverTrain') is True,
        'ACTUAL_EVALUATION_REQUIRED')
    suite=proof.get('foundation_domain_suite' if domain else 'foundation_suite',result.get('suiteId'))
    keys=('domainId','domainHash','kind','tasks') if domain else ('kind','tasks','fixture')
    require(suite.get('state')=='ready' and suite.get('suiteHash')==result.get('suiteHash')==digest({k:suite[k] for k in keys}),
            'EVALUATION_SUITE_CHANGED')
    if domain:
        require(suite.get('domainHash')==digest({k:domain[k] for k in ('purpose','language','oracleType','scope','policyVersion')}),
                'DOMAIN_SUITE_NOT_READY')
    require(job['payload'].get('suiteId')==suite['id']
        and job['payload'].get('artifactId',job['payload'].get('checkpointId'))==candidate,
        'EVALUATION_JOB_MISMATCH')
    expected_split='dev' if suite['kind']=='dev' else 'release'
    for task in suite['tasks']:
        require(task.get('lineageGroups'),'EVALUATION_LINEAGE_REQUIRED')
        for group in task['lineageGroups']:
            split=proof.get('foundation_split',digest(group))
            require(split.get('split')==expected_split and split.get('evaluationReserved') is True,'SPLIT_LEAKAGE')
    baseline_ids=result.get('baselineIds',[])
    require(len(set(baseline_ids))==len(baseline_ids) and candidate not in baseline_ids
        and set(paired)<=set(baseline_ids) and set(result.get('results',{}))=={candidate,*baseline_ids},
        'EVALUATION_PAIR_MISMATCH')
    candidate_rows=_rows(proof,result,suite,candidate,domain=domain)
    old={}
    for baseline in baseline_ids:
        proof.artifact(baseline);old[baseline]=_rows(proof,result,suite,baseline,domain=domain)
    if result.get('kind')=='release':
        shard=proof.get('foundation_eval_shard',digest([suite['suiteHash'],suite['shardId']]))
        require(shard.get('evaluationId')==identity and shard.get('state')=='consumed','ACTUAL_EVALUATION_RESERVATION_REQUIRED')
        for task in suite['tasks']:
            if not task.get('critical'):
                used=proof.get('foundation_eval_problem',task['problemFingerprint'])
                require(used.get('evaluationId')==identity and used.get('neverTrain') is True,'EVALUATION_PROBLEM_REUSED')
        gate=subsequent_gate(candidate_rows,old,fixture=False) if old else first_usable_gate(candidate_rows,fixture=False)
        require(result.get('productionCertified') is True and result.get('gate',{}).get('passed') is True and gate['passed'],
                'INDEPENDENT_QUALITY_REQUIRED',status=409)
        if paired:require(gate['groupCount']>=200 and old,'PAIRED_200_GROUPS_REQUIRED')
        from .resources import foundation_budget
        budget=foundation_budget(proof.store,test_only=False)
        limit=min(12*1024**3,budget['hostWorkloadBudgetBytes'],budget.get('memoryLimitBytes') or 2**63)
        for row in candidate_rows:
            timing=row.get('timings',{})
            require(timing.get('runtimeResourceObserved') is True and _integer(timing.get('peakRssBytes'))
                and timing['peakRssBytes']<=limit,
                    'RUNTIME_RESOURCE_GATE_REQUIRED')
    else:
        require(not paired and result.get('kind') in ('dev','regression'),'RELEASE_EVALUATION_REQUIRED')
    return result,suite


def _usable(proof,project,candidate):
    failures=[]
    for row in proof.store.list('foundation_release'):
        if row.get('projectId',row.get('foundationProjectId'))!=project['id']:continue
        artifact=proof.store.get('artifact',row.get('artifactId',''),{})
        exact=artifact.get('id')==candidate or (artifact.get('format')=='safetensors' and artifact.get('sourceCheckpointId')==candidate)
        if not exact:continue
        mark=proof.mark()
        try:
            release,certificate,artifact,_,_=proof.control.foundation.runtime._release({'releaseId':row['id'],'mode':'code','language':'python'})
            require(release.get('state') in ('ready','active'),'CURRENT_USABLE_CERTIFICATE_REQUIRED')
            proof.get('foundation_release',row['id'])
            for kind in ('release','certificate'):
                identity=row['id'] if kind=='release' else certificate['id']
                if proof.store.get(kind,identity,{}):proof.get(kind,identity)
            _real(proof.get('foundation_certificate',certificate['id']));proof.artifact(artifact['id'])
            require(certificate.get('projectId')==project['id'] and certificate.get('checkpointId')==candidate
                and certificate.get('executionScope')==proof.scope,'CURRENT_USABLE_CERTIFICATE_REQUIRED')
            qualification=proof.get('foundation_graduation',certificate.get('qualifiedGraduationId'))
            require(qualification.get('state')=='qualified' and qualification.get('projectId')==project['id'],
                    'FOUNDATION_QUALIFICATION_REQUIRED')
            _qualification(proof,project,qualification)
            pointer=artifact.get('sourceCheckpointId') if artifact.get('format')=='safetensors' else artifact.get('sourceArtifactId')
            ancestors=set()
            while pointer and pointer not in ancestors and len(ancestors)<4096:
                ancestors.add(pointer)
                registered=proof.store.get('artifact',pointer,{})
                if registered:
                    registered=proof.artifact(pointer)
                    pointer=registered.get('sourceCheckpointId') or registered.get('sourceArtifactId')
                else:
                    _,state,_,_=proof.checkpoint(pointer)
                    pointer=state.get('parentCheckpointId') or state.get('sourceCheckpointId')
            require(qualification.get('checkpointId') in ancestors,'FOUNDATION_QUALIFICATION_REQUIRED')
            require(certificate.get('evaluationIds'),'CURRENT_USABLE_CERTIFICATE_REQUIRED')
            return release,certificate,artifact
        except Fault as error:proof.rollback(mark);failures.append(error.code)
    raise Fault('CURRENT_USABLE_CERTIFICATE_REQUIRED',status=409,details={'causes':sorted(set(failures))})


def _qualification(proof,project,qualification):
    """Recheck executed LM qualification and its two earlier checkpoint controls."""
    measured=_real(proof.get('foundation_evaluation',qualification.get('evaluationId')))
    job=proof.execution('foundation_evaluation',measured)
    require(measured.get('kind')=='qualification' and measured.get('state')=='completed'
        and measured.get('projectId')==project['id'] and measured.get('neverTrain') is True
        and measured.get('artifactId')==qualification.get('checkpointId')
        and job['payload'].get('artifactId',job['payload'].get('checkpointId'))==measured['artifactId']
        and measured.get('gate',{}).get('passed') is True
        and _integer(measured.get('targetTokenCount'),100000) and measured['targetTokenCount']<=1_000_511
        and _integer(measured.get('independentGroupCount'),20),'FOUNDATION_QUALIFICATION_REQUIRED')
    checkpoint,_,_,_=proof.checkpoint(measured['artifactId'],trained=True)
    corpus=proof.corpus(measured['corpusId'])
    require(corpus['projectId']==project['id'] and corpus['manifestHash']==measured.get('corpusHash')
        and checkpoint.get('tokenizerHash')==measured.get('tokenizerHash')
        and measured['independentGroupCount']<=corpus.get('groupCounts',{}).get('dev',0),
        'FOUNDATION_QUALIFICATION_REQUIRED')
    environment=measured.get('evaluationEnvironment',{})
    require(environment.get('executionScope')==proof.scope
        and environment.get('backendReceiptHash')==project['backendReceipt']['receiptHash'],
        'EXECUTION_SCOPE_REQUALIFICATION_REQUIRED')
    nll=measured.get('nll',{})
    require(all(type(nll.get(key)) in (int,float) and math.isfinite(nll[key]) and nll[key]>0
                for key in ('candidate','random','unigram'))
        and nll['candidate']<=.9*nll['random'] and nll['candidate']<=.9*nll['unigram'],
        'FOUNDATION_QUALITY_NOT_REACHED')
    previous=sorted((row for row in proof.store.list('foundation_evaluation')
        if row.get('kind')=='qualification' and row.get('projectId')==project['id']
        and row.get('corpusHash')==corpus['manifestHash'] and row.get('artifactId')!=measured['artifactId']
        and row.get('createdAt','')<=measured.get('createdAt','')),key=lambda row:row.get('createdAt',''),reverse=True)
    history=[];seen=set()
    for row in previous:
        if row.get('artifactId') in seen:continue
        seen.add(row.get('artifactId'));history.append(row)
        if len(history)==2:break
    require(len(history)==2,'QUALIFICATION_HISTORY_REQUIRED')
    for row in history:
        row=_real(proof.get('foundation_evaluation',row['id']));proof.execution('foundation_evaluation',row)
        old=row.get('nll',{}).get('candidate')
        require(row.get('state')=='completed' and row.get('tokenizerHash')==measured['tokenizerHash']
            and type(old) in (int,float) and math.isfinite(old) and old>0 and nll['candidate']<=old*1.02,
            'LM_DEV_REGRESSION')
        proof.checkpoint(row['artifactId'],trained=True)
    return measured


def _profile(proof,project,training_job,checkpoint):
    profile=_real(proof.get('foundation_profile',training_job['payload'].get('profileId')))
    config=training_job['payload']['config'];stage=training_job['payload']['stage']
    require(profile.get('state')=='passed' and profile.get('passed') is True
        and profile.get('measurementOnly') is True and profile.get('productionTrainedTokens')==0
        and profile.get('projectId')==project['id'] and profile.get('executionScope')==proof.scope
        and profile.get('stage')==stage and profile.get('configHash')==digest(config)
        and profile.get('config')==config and profile.get('tokenizerHash')==checkpoint['tokenizerHash']
        and profile.get('architectureHashV2')==checkpoint['architectureHashV2']
        and profile.get('warmupSteps')==3 and _integer(profile.get('measuredSteps'),5)
        and _integer(profile.get('measuredTargetTokens')),'CALIBRATION_EVIDENCE_REQUIRED')
    require((profile.get('backendReceipt') or {}).get('receiptHash')==project['backendReceipt']['receiptHash'],
            'BACKEND_RECEIPT_MISMATCH')
    completed=[job for job in proof.store.rows("SELECT id FROM jobs WHERE type='foundation_calibrate' AND state='completed'")
        if (proof.store.job(job['id']).get('result') or {}).get('id')==profile['id']]
    require(completed,'CALIBRATION_EXECUTION_REQUIRED')
    executed=proof.job(completed[0]['id'])
    measured_config=executed['payload'].get('config',{})
    require(executed['payload'].get('projectId')==project['id']
        and executed['payload'].get('stage')==stage
        and all(measured_config.get(key)==value for key,value in config.items())
        and set(measured_config)-set(config)<={'measurementSteps','warmupSteps','maxSeconds'}
        and executed['payload'].get('corpusId')==profile.get('corpusId')
        and all(profile.get(key)==value for key,value in executed['result'].items()
            if key not in ('version','updatedAt','createdAt')),'CALIBRATION_EXECUTION_REQUIRED')
    require(all(type(profile.get(key)) in (int,float) and math.isfinite(profile[key]) and profile[key]>0
        for key in ('effectiveTokensPerSecond','meanStepSeconds','p95StepSeconds','peakRssBytes','checkpointSeconds')),
        'CALIBRATION_EVIDENCE_REQUIRED')
    from .resources import machine_fingerprint
    require(profile.get('environment',{}).get('machineFingerprint')==machine_fingerprint(),
        'CALIBRATION_ENVIRONMENT_CHANGED')
    return profile


def _curriculum(proof,project):
    from .curriculum import development_gate
    plan=_real(proof.get('foundation_curriculum',project['id']));steps=plan.get('steps',[]);history=plan.get('history',[])
    require(plan.get('enabled') is True and plan.get('state')=='completed' and 2<=len(steps)<=8
        and plan.get('index')==len(steps) and plan.get('planHash')==digest(steps)
        and len(history)==len(steps),'COMPLETED_MULTI_STEP_CURRICULUM_REQUIRED')
    require([row.get('index') for row in history]==list(range(len(steps))),'CURRICULUM_HISTORY_CHANGED')
    previous=None;contexts=[];jobs=set();evaluations=set()
    for step,entry in zip(steps,history):
        require(step.get('minGroups',0)>=20 and step.get('minPassRate',0)>=.7
            and step.get('minSyntaxRate',0)>=.95 and 0<=step.get('maxRegression',1)<=.02,'CURRICULUM_GATE_POLICY_REQUIRED')
        checkpoint,saved,run,job,corpus=proof.training(entry['checkpointId'])
        require(checkpoint['projectId']==project['id'] and saved.get('stage')==step['stage']
            and corpus['id']==step['corpusId'] and corpus['manifestHash']==step['corpusHash']
            and job['type']=='foundation_'+step['stage']
            and all(job['payload']['config'].get(k)==v for k,v in step['config'].items()),'CURRICULUM_RUN_MISMATCH')
        if previous:require(saved.get('parentCheckpointId')==previous,'CURRICULUM_PARENT_CHANGED')
        require(_integer(checkpoint.get('trainedContextTokens'),step['sequenceLength'])
            and (not contexts or step['sequenceLength']>=contexts[-1]),'CURRICULUM_CONTEXT_NOT_TRAINED')
        _profile(proof,project,job,checkpoint)
        evaluation,suite=_evaluation(proof,entry['evaluationId'],checkpoint['id'])
        require(evaluation.get('kind') in ('dev','regression') and suite['id']==step['suiteId']
            and suite['suiteHash']==step['suiteHash'] and saved['parentCheckpointId'] in evaluation.get('baselineIds',[]),
            'CURRICULUM_EVALUATION_MISMATCH')
        decision=development_gate(evaluation,checkpoint['id'],saved['parentCheckpointId'],step)
        require(decision['passed'] and entry.get('decision')==decision,'CURRICULUM_QUALITY_NOT_REACHED')
        require(proof.get('foundation_automation_job',job['id']).get('curriculumId')==plan['id'],
                'AUTOMATIC_CURRICULUM_EXECUTION_REQUIRED')
        jobs.add(job['id']);evaluations.add(evaluation['id']);contexts.append(step['sequenceLength']);previous=checkpoint['id']
    require(len(jobs)==len(steps) and len(evaluations)==len(steps)
        and plan.get('parentCheckpointId')==previous,'CURRICULUM_HISTORY_CHANGED')
    return {'completedSteps':len(steps),'trainedContexts':contexts,'finalCheckpointId':previous}


def _domains(proof,project):
    from .domains import authorize,source_facts,source_findings
    latest=project.get('latestCheckpointId');require(latest,'TRAINED_CANDIDATE_REQUIRED')
    _,base,artifact=_usable(proof,project,latest)
    for identity in base['evaluationIds']:_evaluation(proof,identity,latest)
    approved=[]
    expected=(('explanation','ko','source_facts','python_source_fact_claims'),
              ('review','ko','static_findings','python_static_findings'),
              ('tests','python','mutation_tests','python_mutation_test_generation'))
    for purpose,language,oracle,scope in expected:
        cert=_real(authorize(proof.store,artifact['id'],purpose,language));proof.get('foundation_domain_certificate',cert['id'])
        require(cert.get('baseCertificateId')==base['id'] and cert.get('projectId')==project['id'],'DOMAIN_BASE_CERTIFICATE_CHANGED')
        domain=_real(proof.get('foundation_domain',cert.get('domainId')))
        require((domain.get('purpose'),domain.get('language'),domain.get('oracleType'),domain.get('scope'))
                ==(purpose,language,oracle,scope),'TYPED_PYTHON_DOMAIN_REQUIRED')
        evaluation,suite=_evaluation(proof,cert.get('evaluationId'),artifact['id'],domain=domain)
        require(evaluation.get('kind')=='release' and cert.get('suiteHash')==suite['suiteHash'],'DOMAIN_QUALITY_REQUIRED')
        for task in suite['tasks']:
            if purpose=='explanation':
                facts=source_facts(task['sourceCode'])
                require(task.get('oracleFacts')=={key:facts[key] for key in task['factIds']},'SOURCE_ORACLE_CHANGED')
            elif purpose=='review':require(task.get('oracleFindings')==source_findings(task['sourceCode']),'SOURCE_ORACLE_CHANGED')
        approved.append({'purpose':purpose,'language':language,'scope':scope,'certificateId':cert['id']})
    return {'approvedScopes':approved,'baseCertificateId':base['id'],'artifactId':artifact['id']}


def _paired_quality(proof,project,candidate,baselines):
    release,certificate,artifact=_usable(proof,project,candidate)
    required=list(dict.fromkeys([*baselines,*_current_baselines(proof,candidate)]))
    records=[]
    for identity in certificate['evaluationIds']:
        result,_=_evaluation(proof,identity,candidate,paired=required)
        require(result.get('kind')=='release','RELEASE_EVALUATION_REQUIRED');records.append(result['id'])
    return {'releaseId':release['id'],'certificateId':certificate['id'],'artifactId':artifact['id'],
            'evaluationIds':records,'baselineIds':required}


def _current_baselines(proof,candidate):
    slot=proof.optional('slot','champion')
    identity=slot.get('activeReleaseId')
    if not identity:return []
    release=proof.optional('release',identity) or proof.get('foundation_release',identity)
    active=proof.artifact(release['artifactId'])
    same=(active['id']==candidate or (active.get('format')=='safetensors' and active.get('sourceCheckpointId')==candidate))
    if same:
        identity=slot.get('previousReleaseId')
        if not identity:return []
        release=proof.optional('release',identity) or proof.get('foundation_release',identity)
        active=proof.artifact(release['artifactId'])
    _real(release)
    return [active['id']]


def _distilled_ancestor(proof,identity):
    seen=set()
    while identity and identity not in seen and len(seen)<4096:
        seen.add(identity);checkpoint,saved,_,_=proof.checkpoint(identity)
        mode=saved.get('distillationMode')
        actual=saved.get('stage')=='distillation' and mode=='same_tokenizer_logits'
        actual=actual or (mode=='verified_response_sft' and saved.get('stage')=='sft'
                         and _integer(saved.get('committedTeacherTargetTokens')))
        if actual:return proof.training(identity)
        identity=saved.get('parentCheckpointId') or saved.get('sourceCheckpointId')
    raise Fault('ACTUAL_TRAINED_DISTILLATION_REQUIRED',status=409)


def _distillation(proof,project):
    latest=project.get('latestCheckpointId');require(latest,'TRAINED_CANDIDATE_REQUIRED')
    final,_,_,_,_=proof.training(latest)
    checkpoint,saved,run,job,corpus=_distilled_ancestor(proof,latest)
    require(saved.get('teacherManifestHashes') and saved.get('teacherApprovalIds'),'TEACHER_PROVENANCE_REQUIRED')
    for approval_id in saved['teacherApprovalIds']:
        approval=proof.get('foundation_teacher_approval',approval_id)
        require(approval.get('approved') is True and approval.get('teacherManifestHash')==saved['teacherManifestHashes'].get(approval.get('teacherArtifactId')),
                'TEACHER_REVISION_MISMATCH')
        teacher=proof.artifact(approval['teacherArtifactId'])
        require(not teacher.get('comparisonOnly') and teacher['manifestHash']==approval['teacherManifestHash'],
                'TEACHER_REVISION_MISMATCH')
    if saved.get('distillationMode')=='verified_response_sft':
        from .corpus import rows
        from .response_distillation import validate_row
        require(corpus.get('kind')=='foundation_teacher_corpus' and saved.get('teacherPolicyHash')==corpus.get('teacherPolicyHash'),
                'TEACHER_CORPUS_REQUIRED')
        teacher=saved['committedTeacherTargetTokens'];human=saved.get('committedHumanTargetTokens',0)
        require(_integer(human) and teacher/(teacher+human)<=.1,'SYNTHETIC_RATIO_EXCEEDED')
        for row in rows(proof.store,corpus,'train'):validate_row(proof.store,corpus,row)
    elif saved.get('distillationMode')=='same_tokenizer_logits':
        require(job['type']=='foundation_distill' and run.get('trainingMode')=='full_parameter_logits_distillation'
            and saved.get('distillationRecipeHash'),'ACTUAL_TRAINED_DISTILLATION_REQUIRED')
    else:raise Fault('ACTUAL_TRAINED_DISTILLATION_REQUIRED')
    native=_paired_quality(proof,project,latest,[saved['parentCheckpointId']])
    return final,native


def _formats(proof,project,checkpoint):
    from .gguf_runtime import verify_conformance
    result={};source=None
    for artifact in proof.store.list('artifact'):
        if (artifact.get('format')!='safetensors' or artifact.get('foundationProjectId')!=project['id']
            or artifact.get('sourceCheckpointId')!=checkpoint['id']):continue
        if artifact.get('state') not in BAD_STATES and not artifact.get('fixture') and not artifact.get('testOnly'):
            source=proof.artifact(artifact['id']);break
    require(source,'FINAL_CHECKPOINT_FORMAT_SOURCE_REQUIRED')
    for quantization in ('F16','Q4_K_M'):
        failures=[]
        for artifact in proof.store.list('artifact'):
            if artifact.get('format')!='gguf' or artifact.get('quantization')!=quantization or artifact.get('sourceArtifactId')!=source['id']:continue
            mark=proof.mark()
            try:
                _real(artifact);proof.artifact(artifact['id'])
                validation,_=verify_conformance(proof.store,artifact,proof.control.registry)
                proof.get('foundation_format_validation',validation['id'])
                require(validation.get('fixture') is False and validation.get('executionScope')==proof.scope
                    and validation.get('sourceManifestHash')==source['manifestHash'],'GGUF_CONFORMANCE_MISMATCH')
                result[quantization]={'artifactId':artifact['id'],'validationId':validation['id'],'sourceArtifactId':source['id']};break
            except Fault as error:proof.rollback(mark);failures.append(error.code)
        require(quantization in result,'FINAL_TWO_FORMAT_NUMERIC_REQUIRED',status=409,format=quantization,causes=failures)
    return result


def _growth(proof,project):
    from .schemas import valid_growth_config
    candidates=[project] if project.get('origin')=='architecture_growth' else [row for row in proof.store.list('foundation_project') if row.get('sourceProjectId')==project['id']]
    errors=[]
    for row in candidates:
        mark=proof.mark()
        try:
            grown=proof.project(row['id']);require(grown.get('origin')=='architecture_growth','GROWTH_PROJECT_REQUIRED')
            receipt=grown.get('trustedGrowthReceipt');require(valid_growth_config(grown['config'],receipt),'GROWTH_RECEIPT_REQUIRED')
            candidate=_real(proof.get('foundation_growth_candidate',grown.get('acceptedArchitectureRecipeId')))
            require(candidate.get('state')=='adopted' and candidate.get('adoptedProjectId')==grown['id']
                and candidate.get('adoptedGenesisId')==grown.get('genesisId')
                and candidate.get('trustedGrowthReceipt')==receipt and receipt.get('candidateId')==candidate['id'],
                'ACTUAL_GROWTH_ADOPTION_REQUIRED')
            require(receipt.get('receiptHash')==digest({k:v for k,v in receipt.items() if k!='receiptHash'})
                and receipt.get('evidenceHash')==digest(candidate.get('evidence'))
                and candidate.get('evidence',{}).get('passed') is True,'GROWTH_RECEIPT_REQUIRED')
            from .transfer import _safe
            path=_safe(proof.store.root/candidate['storagePath']);inside(proof.store.root,candidate['storagePath'])
            manifest_path=path/'manifest.json'
            require(not path.is_symlink() and file_hash(manifest_path)==candidate['manifestHash'],'GROWTH_MANIFEST_MISMATCH')
            manifest=json.loads(manifest_path.read_text('utf-8'))
            for name,checksum in manifest['files'].items():
                file=inside(path,name);require(not file.is_symlink() and file_hash(file)==checksum,'GROWTH_MANIFEST_MISMATCH')
            for key in ('sourceCheckpointId','sourceManifestHash','sourceProjectId','sourceGenesisId','config','recipe','recipeHash',
                'tokenizerId','tokenizerHash','candidateTensorDigest','parentTensorDigest','trustedGrowthReceipt','evidence'):
                require(candidate.get(key)==manifest.get(key),'GROWTH_MANIFEST_MISMATCH',field=key)
            require(manifest.get('trustedGrowthReceipt')==receipt and candidate.get('config')==grown['config']
                and candidate.get('sourceProjectId')==grown['sourceProjectId']
                and candidate.get('sourceCheckpointId')==grown['sourceCheckpointId'],'GROWTH_MANIFEST_MISMATCH')
            executions={}
            for row_job in proof.store.rows("SELECT id FROM jobs WHERE type IN ('foundation_growth','foundation_growth_adopt') AND state='completed'"):
                job=proof.store.job(row_job['id']);emitted=job.get('result') or {};body=job['payload']
                if job['type']=='foundation_growth' and emitted.get('id')==candidate['id']:
                    require(body.get('projectId')==grown['sourceProjectId'] and body.get('parentCheckpointId')==grown['sourceCheckpointId']
                        and emitted.get('manifestHash')==candidate['manifestHash'],'ACTUAL_GROWTH_EXPERIMENT_REQUIRED')
                    executions['experiment']=proof.job(job['id'])['id']
                elif job['type']=='foundation_growth_adopt' and emitted.get('candidateId')==candidate['id']:
                    require(body.get('candidateId')==candidate['id'] and emitted.get('project',{}).get('id')==grown['id']
                        and emitted.get('genesis',{}).get('id')==grown['genesisId'],'ACTUAL_GROWTH_ADOPTION_REQUIRED')
                    executions['adoption']=proof.job(job['id'])['id']
            require(len(executions)==2,'ACTUAL_GROWTH_EXECUTIONS_REQUIRED')
            source,_,_,_=proof.checkpoint(grown['sourceCheckpointId'],trained=True)
            genesis,initial,_,_=proof.checkpoint(grown['genesisId'])
            require(source['manifestHash']==receipt.get('sourceManifestHash')
                and initial.get('initialTensorDigest')==candidate.get('candidateTensorDigest')
                and genesis.get('origin')=='architecture_growth','GROWTH_SOURCE_CHANGED')
            baseline,baseline_state,_,_=proof.checkpoint(grown.get('randomBaselineGenesisId'),allow_random_baseline=True)
            require(baseline_state.get('randomBaselineOnly') is True and baseline_state.get('origin')=='random_init'
                and baseline_state.get('pretrainedWeightArtifactIds')==[],'TRUE_RANDOM_BASELINE_REQUIRED')
            latest=grown.get('latestCheckpointId');require(latest,'POST_GROWTH_TRAINING_REQUIRED')
            final,_,_,_,_=proof.training(latest)
            require(final.get('origin')=='architecture_growth' and final.get('genesisId',grown['genesisId'])==grown['genesisId'],
                    'GROWTH_PROVENANCE_CHANGED')
            quality=_paired_quality(proof,grown,latest,[source['id']])
            return {'projectId':grown['id'],'candidateId':candidate['id'],'sourceCheckpointId':source['id'],
                'executionJobIds':executions,**quality}
        except Fault as error:proof.rollback(mark);errors.append(error.code)
    raise Fault('QUALIFIED_GROWTH_TRANSITION_REQUIRED',status=409,details={'causes':sorted(set(errors))})


def _format_and_distillation_components(proof,project):
    components={'nativeDistillationQualityPassed':False,'twoFormatNumericPassed':False,
                'formatFunctionalCertification':False,'formatFunctionalResults':{}}
    failures={};mark=proof.mark()
    try:
        _,native=_distillation(proof,project)
        components.update(nativeDistillationQualityPassed=True,native=native)
    except Fault as error:
        proof.rollback(mark);failures['nativeDistillationQuality']=error.code
    formats={};mark=proof.mark()
    try:
        latest=project.get('latestCheckpointId');require(latest,'TRAINED_CANDIDATE_REQUIRED')
        checkpoint,_,_,_=proof.checkpoint(latest,trained=True)
        formats=_formats(proof,project,checkpoint)
        components.update(twoFormatNumericPassed=True,formats=formats)
    except Fault as error:
        proof.rollback(mark);failures['twoFormatNumeric']=error.code
    for name,entry in formats.items():
        mark=proof.mark()
        try:
            quality=_paired_quality(proof,project,entry['artifactId'],[entry['sourceArtifactId']])
            components['formatFunctionalResults'][name]={'passed':True,**quality}
        except Fault as error:
            proof.rollback(mark);failures[name+'FunctionalQuality']=error.code
            components['formatFunctionalResults'][name]={'passed':False,'reasonCodes':[error.code]}
    components['formatFunctionalCertification']=bool(formats) and len(formats)==2 and all(
        row.get('passed') for row in components['formatFunctionalResults'].values())
    components['componentFailures']=failures
    return components,failures


def assess_requirement(control,number,project_id,*,software=None):
    """Current source-computed result, never a caller-provided score receipt."""
    require(type(number) is int and number in NUMBERS,'UNSUPPORTED_ACCEPTANCE_REQUIREMENT')
    proof=_Proof(control);components={};reasons=[];details={}
    try:
        if software:
            from .acceptance_checks import verify_software
            verify_software(control.store,software)
        project=proof.project(project_id)
        if number==17:components['curriculum']=_curriculum(proof,project)
        elif number==18:components['typedPythonDomains']=_domains(proof,project)
        elif number==21:components['growth']=_growth(proof,project)
        else:
            components,failures=_format_and_distillation_components(proof,project)
            require(not failures and components['nativeDistillationQualityPassed']
                and components['twoFormatNumericPassed'] and components['formatFunctionalCertification'],
                'FORMATS_AND_DISTILLATION_QUALITY_REQUIRED',status=409,components=failures)
        proof.project(project_id);proof.finish()
    except (Fault,KeyError,TypeError,ValueError,OSError) as error:
        reasons=[getattr(error,'code','TRANSITION_EVIDENCE_INVALID')];details=getattr(error,'details',{})
    evidence=proof.evidence()
    return {'requirementNumber':number,'projectId':project_id,'passed':not reasons,
        'state':'passed' if not reasons else 'blocked','reasonCodes':reasons,'failureDetails':details,
        'claimScope':'actual_transition_and_current_quality','executionScope':proof.scope,
        'evidenceIds':sorted({row['id'] for row in evidence}),'evidenceRecords':evidence,
        'softwareEvidenceId':software.get('id') if software else None,'components':components,
        'availableExecutionPaths':PATHS[number],
        'missingOperationalPaths':(['fresh independent format-specific 200-group evaluations and usable certificates']
            if number==20 and not components.get('formatFunctionalCertification') else []),
        'modelEvaluationPerformedByThisCheck':False,'nasModified':False}


def assess_all(control,project_id,*,software=None):
    return {number:assess_requirement(control,number,project_id,software=software) for number in sorted(NUMBERS)}
