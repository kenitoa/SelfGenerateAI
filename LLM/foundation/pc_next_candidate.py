"""Explicit local PC branch training with one persistent four-hour wall budget.

Importing this module performs no IO. The CLI prints a plan unless --execute is
present. It never deploys, activates, downloads weights or edits an old ledger.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict,dataclass
import json
import math
import os
from pathlib import Path
import re
import time

from LLM.core import Fault,atomic_json,digest,now,require

KIND='foundation_pc_next_candidate'
TERMINAL=frozenset(('completed','failed','cancelled'))
TRAIN=frozenset(('foundation_cpt','foundation_sft'))


@dataclass(frozen=True)
class Plan:
    project_id:str=''
    parent_checkpoint_id:str=''
    raw_preparation_id:str='official-python-v2'
    sft_preparation_id:str='official-context-v3'
    runtime_verification_id:str=''
    max_seconds:int=14400
    tokens_per_cpt:int=4_000_000
    reserve_seconds:int=180
    execution_scope:str='pc_validation'

    def validate(self):
        require(self.execution_scope=='pc_validation','PC_VALIDATION_SCOPE_REQUIRED')
        require(type(self.max_seconds) is int and 1<=self.max_seconds<=14400,'PC_NEXT_BUDGET_INVALID')
        require(type(self.tokens_per_cpt) is int and 1<=self.tokens_per_cpt<=4_000_000,'PC_NEXT_BUDGET_INVALID')
        require(type(self.reserve_seconds) is int and 0<=self.reserve_seconds<self.max_seconds,'PC_NEXT_BUDGET_INVALID')
        require(self.project_id and self.parent_checkpoint_id and self.runtime_verification_id,'PC_NEXT_INPUTS_REQUIRED')
        return self

    def recipe(self,index):
        sft=index==3
        return {'sequenceLength':512,'microBatchSize':4,'gradientAccumulation':2,'cpuThreads':4,
            'learningRate':5e-5 if sft else 1e-4,'maxTokenExposures':self.tokens_per_cpt,
            'maxEpochs':2 if sft else 3,'sliceSeconds':900,'maxActiveSeconds':self.max_seconds,
            'checkpointEvery':100,'checkpointSeconds':120}


class WallClock:
    def time(self):return time.time()
    def monotonic(self):return time.monotonic()
    def boot_id(self):
        import psutil
        return int(psutil.boot_time())
    def sleep(self,seconds):time.sleep(seconds)


def local_preparation_root(value):
    from LLM.foundation.acceptance import require_local_path
    from LLM import core
    require(not str(value).startswith(('\\\\','//')),'LOCAL_PC_DATA_REQUIRED')
    absolute=Path(os.path.abspath(value));workspace=Path(core.__file__).resolve().parents[1]
    require(absolute.is_relative_to(workspace/'LLM/runtime-data/preparation'),'LOCAL_PC_DATA_REQUIRED')
    for part in (absolute,*absolute.parents):
        require(not part.is_symlink() and not getattr(part,'is_junction',lambda:False)(),'LOCAL_PC_DATA_REQUIRED')
    return require_local_path(absolute,workspace/'LLM/runtime-data/preparation')


def original_plan(root,runtime_verification_id,*,parent_checkpoint_id=None):
    root=local_preparation_root(root)
    original=local_report(root,'pc-candidate.json')
    require(original.get('state')=='candidate_trained' and original.get('executionScope')=='pc_validation'
        and original.get('nasModified') is False,'PC_NEXT_ORIGINAL_REPORT_REQUIRED')
    parent=parent_checkpoint_id or original.get('phaseCheckpoints',{}).get('2')
    return Plan(project_id=original['projectId'],parent_checkpoint_id=parent or '',
        runtime_verification_id=runtime_verification_id).validate()


def local_report(root,name):
    path=Path(root)/'reports'/name
    for part in (path,path.parent):
        require(not part.is_symlink() and not getattr(part,'is_junction',lambda:False)(),'LOCAL_PC_DATA_REQUIRED')
    require(path.is_file() and path.stat().st_size<=32*1024**2,'PC_NEXT_ORIGINAL_REPORT_REQUIRED')
    return json.loads(path.read_text('utf-8'))


def runtime_implementation_fields(preparation):
    """Current execution semantics, independently of any supplied pass flag."""
    from LLM.core import file_hash
    from LLM.foundation import sft_runtime_verification,source_oracle_binding
    from LLM.foundation.source_context_contracts import runtime_contract,runtime_evidence_fields
    execution=runtime_contract(preparation)
    runtime_hash=file_hash(Path(sft_runtime_verification.__file__))
    binding_hash=source_oracle_binding.implementation_hash()
    namespace=[execution['runtimeVersion'],source_oracle_binding.VERSION,binding_hash,runtime_hash]
    return {'implementationVersion':execution['runtimeVersion'],'implementationHash':runtime_hash,
        **runtime_evidence_fields(execution,preparation),
        'oracleBindingVersion':source_oracle_binding.VERSION,'oracleBindingImplementationHash':binding_hash,
        'runtimeFixtureNamespaceHash':digest(namespace)}


def require_current_runtime(record,preparation):
    from LLM.foundation.source_context_contracts import matches_fields
    fields=runtime_implementation_fields(preparation)
    body={k:v for k,v in record.items() if k not in ('id','version','createdAt','updatedAt','reportHash')}
    require(matches_fields(record,fields) and record.get('reportHash')==digest(body)
        and record.get('purpose')=='source_span_data_execution' and record.get('modelQualityPassed') is False
        and record.get('servingEligible') is False and record.get('definitionLoadOnlyIsNotSemanticCorrectness') is True,
        'PC_NEXT_SFT_RUNTIME_IMPLEMENTATION_CHANGED',status=409)
    return fields


def require_current_oracle_binding(binding,row,current,corpus):
    keys=('oracleBindingId','oracleBindingHash','oracleBindingState','oracleBindingVersion')
    count=row.get('semanticCaseCount',0)
    require(type(count) is int and count>=0 and type(row.get('semanticVerified')) is bool,
        'PC_NEXT_SFT_ORACLE_BINDING_CHANGED',status=409)
    if binding is None:
        require(not any(key in row for key in keys) and row['semanticVerified'] is False and count==0,
            'PC_NEXT_SFT_ORACLE_BINDING_REQUIRED',status=409)
        return
    require(all(row.get(key) for key in keys) and binding.get('id')==row['oracleBindingId']
        and digest(binding)==row['oracleBindingHash']
        and binding.get('helperVersion')==row['oracleBindingVersion']==current['oracleBindingVersion']
        and binding.get('helperHash')==current['oracleBindingImplementationHash']
        and binding.get('state')==row['oracleBindingState'] and binding['state'] in ('matched','mismatch','unresolved')
        and binding.get('sourceExampleId')==row['exampleId']
        and binding.get('sourceVersionId')==row['sourceVersionId']
        and binding.get('sourceCorpusId')==corpus['id'] and binding.get('sourceCorpusHash')==corpus['manifestHash'],
        'PC_NEXT_SFT_ORACLE_BINDING_CHANGED',status=409)
    matched=binding['state']=='matched'
    require(matched is row['semanticVerified'] and (count>0) is matched,
        'PC_NEXT_SFT_ORACLE_BINDING_CHANGED',status=409)


def require_current_oracle_eligibility(row,binding_fields,case_count):
    from LLM.foundation.source_context_contracts import matches_fields
    keys=('oracleBindingId','oracleBindingHash','oracleBindingState','oracleBindingVersion')
    require((binding_fields and matches_fields(row,binding_fields) or
        not binding_fields and not any(key in row for key in keys))
        and type(row.get('semanticCaseCount')) is int and row['semanticCaseCount']==case_count
        and row.get('semanticVerified') is (case_count>0),'PC_NEXT_SFT_ORACLE_ELIGIBILITY_CHANGED',status=409)


class Coordinator:
    """Scheduling only. Adapter methods own the existing guarded controller IO."""
    def __init__(self,adapter,plan,operation_id,*,clock=None):
        self.adapter=adapter;self.plan=plan.validate();self.operation_id=operation_id
        require(isinstance(operation_id,str) and re.fullmatch('[A-Za-z0-9_-]{1,100}',operation_id),
            'PC_NEXT_OPERATION_REQUIRED')
        self.clock=clock or WallClock();self.ledger={};self.current_job=None;self.started=False

    def monotonic(self):return getattr(self.clock,'monotonic',self.clock.time)()
    def boot_id(self):return getattr(self.clock,'boot_id',lambda:'injected-test-clock')()

    def elapsed(self):
        require(self.ledger.get('bootId')==self.boot_id(),'PC_NEXT_REBOOT_REQUIRES_REVIEW',status=409)
        monotonic_elapsed=self.monotonic()-self.ledger['startedMonotonic']
        require(monotonic_elapsed>=0,'PC_NEXT_CLOCK_ROLLBACK',status=409)
        return max(self.ledger.get('elapsedSeconds',0),self.clock.time()-self.ledger['startedEpoch'],monotonic_elapsed)

    def update(self,**fields):
        stamp=self.clock.time()
        if self.ledger.get('lastObservedEpoch',stamp)>stamp+1:
            raise Fault('PC_NEXT_CLOCK_ROLLBACK',status=409)
        elapsed=self.elapsed()
        value={**self.ledger,**fields,'lastObservedEpoch':stamp,'elapsedSeconds':elapsed,
            'observedAt':now(),'nasModified':False,'activationPerformed':False,'actualModelQualityPassed':False}
        self.ledger=self.adapter.save(value,self.operation_id)
        return self.ledger

    def cleanup_update(self,**fields):
        try:return self.update(**fields)
        except Fault as error:
            if error.code not in ('PC_NEXT_CLOCK_ROLLBACK','PC_NEXT_REBOOT_REQUIRES_REVIEW'):raise
            # Clock uncertainty never authorizes more compute or resets time.
            self.ledger=self.adapter.save({**self.ledger,**fields,'state':'held','clockValidationHeld':True,
                'reasonCodes':sorted(set([*self.ledger.get('reasonCodes',[]),error.code])),
                'observedAt':now()},self.operation_id)
            return self.ledger

    def remaining(self):
        stamp=self.clock.time()
        require(stamp>=self.ledger.get('lastObservedEpoch',stamp)-1,'PC_NEXT_CLOCK_ROLLBACK',status=409)
        elapsed=self.elapsed()
        return min(self.ledger['deadlineEpoch']-stamp,self.plan.max_seconds-elapsed)

    def budget(self):
        require(self.remaining()>self.plan.reserve_seconds,'PC_NEXT_WALL_BUDGET_EXHAUSTED',status=409)

    def grant(self):
        policy=self.adapter.policy()
        require(policy.get('enabled') is False,'PC_NEXT_AUTOMATION_MUST_BE_DISABLED',status=409)
        previous=self.ledger.get('policyGrant')
        if previous and not previous.get('restored'):
            require(policy.get('dailyActiveSeconds')==previous['grantedCap'],'PC_NEXT_POLICY_OWNERSHIP_CHANGED',status=409)
            return
        cap=math.ceil(self.adapter.used_today()+self.plan.max_seconds)
        require(1<=cap<=86400,'PC_NEXT_DAILY_CAP_INVALID',status=409)
        grant={'oldCap':policy['dailyActiveSeconds'],'grantedCap':cap,'previousVersion':policy['version'],
            'restored':False,'intentRecordedAt':now()}
        if hasattr(self.adapter,'coordinator_identity'):grant['coordinator']=self.adapter.coordinator_identity()
        self.update(policyGrant=grant)
        changed=self.adapter.configure_cap(cap,policy['version'])
        self.update(policyGrant={**grant,'appliedVersion':changed['version']})

    def restore(self):
        grant=self.ledger.get('policyGrant')
        if not grant or grant.get('restored'):return
        policy=self.adapter.policy()
        if policy.get('dailyActiveSeconds')!=grant['grantedCap']:
            self.cleanup_update(state='held',reasonCodes=sorted(set([*self.ledger.get('reasonCodes',[]),'PC_NEXT_POLICY_OWNERSHIP_CHANGED'])),
                policyRestore={'state':'held','reason':'PC_NEXT_POLICY_OWNERSHIP_CHANGED'})
            return
        changed=self.adapter.configure_cap(grant['oldCap'],policy['version'])
        self.cleanup_update(policyGrant={**grant,'restored':True,'restoredVersion':changed['version']},
            policyRestore={'state':'restored','onlyDailyActiveSecondsChanged':True})

    def wait(self,job):
        self.current_job=job['id']
        while True:
            self.budget();current=self.adapter.job(job['id'])
            if current['state']=='completed':return current.get('result') or {}
            if current['state'] in ('failed','cancelled'):
                error=current.get('error') or {};attempts=self.ledger.get('sqliteRecoveryAttempts',{})
                recoverable=(current['state']=='failed' and current['type'] in TRAIN
                    and error.get('code')=='JOB_FAILED' and 'database is locked' in error.get('message','').lower())
                if recoverable and attempts.get(job['id'],0)<3:
                    self.update(sqliteRecoveryAttempts={**attempts,job['id']:attempts.get(job['id'],0)+1})
                    self.clock.sleep(2);self.budget();self.adapter.command(job['id'],'retry');continue
                raise Fault(error.get('code') or 'PC_NEXT_JOB_FAILED',status=409)
            if current['state']=='paused':
                require(current['type'] in TRAIN,'PC_NEXT_NONTRAIN_PAUSE_REQUIRES_REVIEW',status=409)
                reason=self.adapter.pause_reason(job['id'])
                require(reason in ('SLICE_LIMIT','WORKER_SHUTDOWN',None),reason or 'PC_NEXT_JOB_PAUSED',status=409)
                self.clock.sleep(2);self.budget();self.adapter.command(job['id'],'resume')
            if self.clock.time()-self.ledger.get('lastObservedEpoch',0)>=10:self.update()
            self.clock.sleep(.5)

    def phase(self,index,parent):
        self.budget();stage='sft' if index==3 else 'cpt';key=str(index)
        corpus=self.ledger['inputs']['sftCorpusId' if index==3 else 'rawCorpusId']
        config=self.plan.recipe(index)
        if index==3:
            config['maxTokenExposures']=min(self.plan.tokens_per_cpt,max(1,self.ledger['inputs']['sftTargetTokens']*2))
        body={'parentCheckpointId':parent,'corpusId':corpus,'tokenizerId':self.ledger['inputs']['tokenizerId'],
            'stage':stage,'continueCorpus':index in (1,2),'config':config}
        job_id=self.ledger.get('phaseJobs',{}).get(key)
        if not job_id:
            if not self.adapter.has_profile(body):
                calibration_id=self.ledger.get('calibrationJobs',{}).get(key)
                calibration=self.adapter.job(calibration_id) if calibration_id else None
                if calibration is None:
                    calibration=self.adapter.calibrate(body)
                    self.update(calibrationJobs={**self.ledger.get('calibrationJobs',{}),key:calibration['id']})
                result=self.wait(calibration)
                require(result.get('passed') is True,'PC_NEXT_CALIBRATION_NOT_PASSED',status=409)
            self.budget();job=self.adapter.train(body)
            job_id=job['id'];self.update(phaseJobs={**self.ledger.get('phaseJobs',{}),key:job_id})
        job=self.adapter.job(job_id);self.adapter.assert_job(job,body,'foundation_'+stage)
        self.update(state='running',stage=stage,phase=index,jobId=job_id,
            phaseRequestedMaxTokens={**self.ledger.get('phaseRequestedMaxTokens',{}),key:config['maxTokenExposures']})
        self.wait(job);cp=self.adapter.checkpoint(job_id)
        self.update(checkpointId=cp['id'],lastVerifiedCheckpointId=cp['id'],
            phaseCheckpoints={**self.ledger.get('phaseCheckpoints',{}),key:cp['id']},
            phaseActualTokens={**self.ledger.get('phaseActualTokens',{}),key:cp.get('committedTargetTokens',0)})
        if index<3:
            evaluation_id=self.ledger.get('qualificationJobs',{}).get(key)
            if not evaluation_id:
                evaluation=self.adapter.qualify(cp['id'],self.ledger['inputs']['rawCorpusId'])
                evaluation_id=evaluation['id'];self.update(qualificationJobs={**self.ledger.get('qualificationJobs',{}),key:evaluation_id})
            self.update(stage='qualification',jobId=evaluation_id)
            self.wait(self.adapter.job(evaluation_id));evaluation=self.adapter.qualification(evaluation_id,cp['id'],self.ledger['inputs'])
            gate=evaluation.get('gate') or {};reasons=set(gate.get('reasonCodes') or [])
            self.update(qualificationGates={**self.ledger.get('qualificationGates',{}),key:gate})
            require(gate.get('passed') is True or index<2 and reasons=={'QUALIFICATION_HISTORY_REQUIRED'},
                'PC_NEXT_QUALIFICATION_HELD',status=409)
        self.update(completedPhases=sorted(set([*self.ledger.get('completedPhases',[]),index])))
        return cp['id']

    def workflow(self):
        old=self.adapter.load(self.operation_id)
        if old:
            require(old.get('planHash')==digest(asdict(self.plan)),'PC_NEXT_PLAN_CHANGED',status=409)
            self.ledger=old
            if old.get('state')=='candidate_trained':return
            if self.remaining()<=self.plan.reserve_seconds:
                self.adapter.expire_jobs(self.ledger)
                raise Fault('PC_NEXT_WALL_BUDGET_EXHAUSTED',status=409)
        proof=self.adapter.preflight(self.plan,self.ledger)
        if self.ledger:
            require(proof==self.ledger['inputs'],'PC_NEXT_INPUTS_CHANGED',status=409)
        else:
            start=self.clock.time()
            self.ledger=self.adapter.create(self.operation_id,{
                'schemaVersion':1,'operationId':self.operation_id,'projectId':self.plan.project_id,
                'executionScope':'pc_validation','plan':asdict(self.plan),'planHash':digest(asdict(self.plan)),
                'inputs':proof,'startedEpoch':start,'deadlineEpoch':start+self.plan.max_seconds,
                'startedMonotonic':self.monotonic(),'bootId':self.boot_id(),
                'lastObservedEpoch':start,'elapsedSeconds':0,'phaseJobs':{},'completedPhases':[],
                'checkpointId':self.plan.parent_checkpoint_id,'lastVerifiedCheckpointId':self.plan.parent_checkpoint_id,
                'calibrationJobs':{},'qualificationJobs':{},'phaseCheckpoints':{},'phaseActualTokens':{},
                'state':'ready','qualityCriteriaLowered':False,'nasModified':False,'activationPerformed':False})
        self.budget();self.grant();self.adapter.start();self.started=True
        parent=self.plan.parent_checkpoint_id
        for index in range(4):
            parent=self.phase(index,parent)
        self.adapter.verify_originals(self.ledger['inputs'])
        self.update(state='candidate_trained',stage='independent_exams_required',checkpointId=parent,
            reasonCodes=['NEW_UNSEEN_RELEASE_SELF_CODE_DOMAIN_EXAMS_REQUIRED'])

    def run(self):
        require(self.adapter.execution_scope=='pc_validation','PC_VALIDATION_SCOPE_REQUIRED')
        self.adapter.acquire(self.operation_id)
        try:
            self.workflow()
        except Exception as error:
            if self.ledger:
                self.cleanup_update(state='held',reasonCodes=[getattr(error,'code','PC_NEXT_WORKFLOW_FAILED')])
            else:raise
        finally:
            cleanup=[]
            try:
                if self.started:
                    try:
                        self.adapter.drain(self.current_job,self.ledger['deadlineEpoch'])
                        current=self.adapter.job(self.current_job) if self.current_job else None
                        if current and current['type'] in TRAIN and current['state'] in ('paused','completed'):
                            checkpoint=self.adapter.checkpoint(current['id'])
                            key=str(self.ledger['phase']);actual=checkpoint.get('committedTargetTokens',0)
                            self.cleanup_update(checkpointId=checkpoint['id'],lastVerifiedCheckpointId=checkpoint['id'],
                                checkpointCommitted=True,
                                phaseActualTokens={**self.ledger.get('phaseActualTokens',{}),key:actual},
                                phaseUnderfillTokens={**self.ledger.get('phaseUnderfillTokens',{}),
                                    key:max(0,self.ledger.get('phaseRequestedMaxTokens',{}).get(key,0)-actual)})
                    except Exception as error:cleanup.append(getattr(error,'code','PC_NEXT_DRAIN_FAILED'))
                    try:self.adapter.stop()
                    except Exception as error:cleanup.append(getattr(error,'code','PC_NEXT_STOP_FAILED'))
                    self.started=False
                if self.ledger:
                    try:self.restore()
                    except Exception as error:cleanup.append(getattr(error,'code','PC_NEXT_POLICY_RESTORE_FAILED'))
                    try:self.adapter.verify_originals(self.ledger['inputs'])
                    except Exception as error:cleanup.append(getattr(error,'code','PC_NEXT_ORIGINAL_CHANGED'))
                    if cleanup:self.cleanup_update(state='held',reasonCodes=sorted(set([*self.ledger.get('reasonCodes',[]),*cleanup])))
            finally:self.adapter.release(self.operation_id)
        return self.ledger


class LocalAdapter:
    """Existing controller APIs only; instantiated solely by explicit --execute."""
    execution_scope='pc_validation'

    def __init__(self,root,plan):
        from LLM.foundation.acceptance import require_local_path
        from LLM.service import Control
        from LLM.foundation.pc_validation import attach
        self.root=local_preparation_root(root);self.plan=plan
        for path in (self.root,self.root/'control',self.root/'rag'):
            require_local_path(path,self.root)
            require(not path.is_symlink(),'LOCAL_PC_DATA_REQUIRED')
        self.control=Control(self.root/'control',worker=True,execution_scope='pc_validation')
        self.store=self.control.store;self.rag=attach(self.control,self.root/'rag')
        self.lock_path=self.store.root/'metadata/pc-next-candidate.lock'
        self.lock=None;self.fatal=None;self.inputs=None;self.deadline=None

    def acquire(self,operation_id):
        import psutil
        self.processes_gone()
        require(not self.lock_path.is_symlink(),'PC_NEXT_LOCK_UNSAFE')
        if self.lock_path.exists():
            require(self.lock_path.stat().st_size<=8192,'PC_NEXT_LOCK_UNSAFE')
            value=json.loads(self.lock_path.read_text('utf-8'))
            try:alive=psutil.Process(value['pid']).create_time()==value['createTime']
            except psutil.NoSuchProcess:alive=False
            require(not alive,'PC_NEXT_COORDINATOR_ALREADY_RUNNING',status=409)
            self.lock_path.unlink()
        body={'operationId':operation_id,'pid':os.getpid(),'createTime':psutil.Process(os.getpid()).create_time()}
        handle=os.open(self.lock_path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        with os.fdopen(handle,'w',encoding='utf-8') as output:
            json.dump(body,output);output.flush();os.fsync(output.fileno())
        self.lock=body

    def release(self,operation_id):
        if self.lock and self.lock_path.is_file():
            require(json.loads(self.lock_path.read_text('utf-8'))==self.lock,'PC_NEXT_LOCK_OWNER_CHANGED')
            self.lock_path.unlink();self.lock=None

    @staticmethod
    def budget_fields(body):
        return {**{key:body[key] for key in ('planHash','startedEpoch','deadlineEpoch','startedMonotonic','bootId')},
            'maxSeconds':body['plan']['max_seconds']}

    def load(self,operation_id):
        from .pc_budget_guard import recover_operation
        recover_operation(self.store,operation_id)
        value=self.store.get(KIND,operation_id,{})
        if not value:return value
        grant=self.store.get('foundation_pc_next_authorization',value.get('authorizationId') or '',{})
        require(grant.get('version')==1 and grant.get('operationId')==operation_id
            and value.get('authorizationHash')==digest(grant) and grant.get('budget')==self.budget_fields(value),
            'PC_NEXT_BUDGET_AUTHORITY_CHANGED',status=409)
        return value

    def save(self,body,operation_id):
        self.deadline=body['deadlineEpoch']
        result=self.store.put(KIND,body,operation_id,expected=body.get('version',0))
        atomic_json(self.root/'reports'/('pc-next-candidate-'+operation_id+'.json'),result)
        return result

    def create(self,operation_id,body):
        authorization=digest([self.plan.project_id,self.plan.parent_checkpoint_id,
            self.plan.raw_preparation_id,self.plan.sft_preparation_id])
        with self.store.tx():
            grant=self.store.get('foundation_pc_next_authorization',authorization,{})
            require(not grant or grant.get('operationId')==operation_id,'PC_NEXT_GRANT_ALREADY_USED',status=409)
            require(not grant,'PC_NEXT_LEDGER_MISSING_REQUIRES_REVIEW',status=409)
            require(not self.store.get(KIND,operation_id,{}),'PC_NEXT_LEDGER_ALREADY_EXISTS',status=409)
            grant=self.store.put('foundation_pc_next_authorization',{
                'operationId':operation_id,'projectId':self.plan.project_id,'planHash':body['planHash'],
                'startedEpoch':body['startedEpoch'],'deadlineEpoch':body['deadlineEpoch'],
                'budget':self.budget_fields(body),'executionScope':'pc_validation','nasModified':False},authorization,expected=0)
            return self.save({**body,'authorizationId':authorization,'authorizationHash':digest(grant)},operation_id)

    def policy(self):return self.store.get('foundation_policy',self.plan.project_id)

    def coordinator_identity(self):
        from .pc_budget_guard import identity
        return identity()

    def used_today(self):
        from LLM.foundation.resources import used_today
        return used_today(self.store)

    def configure_cap(self,cap,version):
        return self.control.foundation.configure(self.plan.project_id,{'dailyActiveSeconds':cap},expected=version)

    def originals(self):
        from LLM.core import file_hash
        from LLM.evolution.service import TARGET
        from LLM.foundation.checkpoint import verify
        old=self.store.get('foundation_pc_validation','pc-candidate-'+self.plan.project_id)
        report=local_report(self.root,'pc-candidate.json')
        exam=local_report(self.root,'post-reboot-independent-exam.json')
        require(digest(report)==digest(old) and exam.get('projectId')==self.plan.project_id
            and exam.get('checkpointId')==old['checkpointId'] and exam.get('state')=='actual_exam_completed'
            and exam.get('modelScoringPerformed') is True,'PC_NEXT_ORIGINAL_REPORT_REQUIRED')
        exam_job=self.store.job(exam['jobId'])
        require(exam_job['state']=='completed' and exam_job['payload'].get('projectId')==self.plan.project_id
            and exam_job['payload'].get('checkpointId')==old['checkpointId'],'PC_NEXT_ORIGINAL_REPORT_REQUIRED')
        answers=[{'id':row['id'],'hash':digest(json.loads(row['body']))} for row in self.store.rows(
            "SELECT id,body FROM records WHERE kind='foundation_eval_task' AND json_extract(body,'$.evaluationId')=? ORDER BY id",
            (exam['jobId'],))]
        checkpoints={}
        for key in {self.plan.parent_checkpoint_id,old['checkpointId']}:
            checkpoint=self.store.get('foundation_checkpoint',key);verify(self.store,checkpoint)
            checkpoints[key]=digest(checkpoint)
        return {'oldLedgerHash':digest(old),'oldReportHash':digest(report),'oldEvaluationId':exam['jobId'],
            'oldExamReportHash':digest(exam),'oldCheckpoints':checkpoints,'oldAnswerCount':len(answers),
            'oldAnswersHash':digest(answers),'extensionHash':file_hash(self.control.evolution.project/TARGET),
            'championSlot':digest(self.store.get('slot','champion',{})),
            'selfCodeSlot':digest(self.store.get('self_code_slot','document',{}))}

    def verify_originals(self,inputs):
        require(self.originals()==inputs['originals'],'PC_NEXT_ORIGINAL_CHANGED',status=409)

    def processes_gone(self):
        import psutil
        for record in self.store.list('foundation_evaluation_process'):
            job=self.store.job(record['jobId'])
            if job['payload'].get('projectId')!=self.plan.project_id:continue
            for role in ('runner','child'):
                proof=record[role]
                try:alive=psutil.Process(proof['pid']).create_time()==proof['createTime']
                except psutil.NoSuchProcess:alive=False
                require(not alive,'PC_NEXT_OLD_PROCESS_PRESENT',status=409)
        for process in psutil.process_iter(['pid','name']):
            if process.pid==os.getpid() or not (process.info['name'] or '').lower().startswith(('python','pythonw')):continue
            try:args=process.cmdline()
            except psutil.NoSuchProcess:continue
            except psutil.AccessDenied:raise Fault('PC_NEXT_PROCESS_INSPECTION_REQUIRED',status=409) from None
            if 'LLM.foundation.worker' in args and '--data-dir' in args:
                root=Path(args[args.index('--data-dir')+1]).resolve()
                require(root!=self.store.root.resolve(),'PC_NEXT_OLD_PROCESS_PRESENT',status=409)

    def runtime_proof(self,preparation,corpus):
        import ast
        from LLM.foundation.acceptance_transitions import _Proof,_verifier
        from LLM.foundation import source_oracle_binding
        from LLM.foundation.sft_runtime_verification import _source_cases
        proof=_Proof(self.control)
        record=proof.get('foundation_sft_runtime_verification',self.plan.runtime_verification_id)
        current=require_current_runtime(record,preparation)
        require(record.get('state')=='passed' and record.get('passed') is True
            and record.get('executionScope')=='pc_validation' and record.get('projectId')==self.plan.project_id
            and record.get('preparationId')==preparation['id'] and record.get('corpusId')==corpus['id']
            and record.get('corpusHash')==corpus['manifestHash'] and record.get('semanticCaseFailures')==0
            and record.get('exampleCount',0)>0 and record.get('definitionVerified')==record['exampleCount']
            and len(record.get('rows',[]))==record['exampleCount'],'PC_NEXT_SFT_RUNTIME_PROOF_REQUIRED',status=409)
        for row in record['rows']:
            fixture=proof.get('fixture',row['definitionFixtureId'])
            require(fixture.get('runtimeFixtureNamespaceHash')==current['runtimeFixtureNamespaceHash']
                and fixture.get('oracleBindingVersion')==current['oracleBindingVersion'],
                'PC_NEXT_SFT_RUNTIME_IMPLEMENTATION_CHANGED',status=409)
            receipt=_verifier(proof,row['definitionReceiptId'],fixture,passed=True)
            require(digest(fixture)==row['definitionFixtureHash'] and digest(receipt)==row['definitionReceiptHash']
                and receipt.get('codeHash')==row['programHash'],'PC_NEXT_SFT_RUNTIME_PROOF_CHANGED',status=409)
            bridge=getattr(self.control,'bridge',None)
            require(bridge is not None,'RAG_POLICY_GATEWAY_REQUIRED',status=503)
            example=bridge.store.get('training_example',row['exampleId'])
            version=bridge.store.get('version',row['sourceVersionId'])
            tree=ast.parse(example['target'])
            require(len(tree.body)==1 and isinstance(tree.body[0],ast.FunctionDef),'SOURCE_TARGET_HASH_MISMATCH')
            binding_fields={}
            cases,_=_source_cases(self.control,preparation,example,version,tree.body[0].name,binding_out=binding_fields)
            require_current_oracle_eligibility(row,binding_fields,len(cases))
            if binding_fields:
                binding=proof.get(source_oracle_binding.KIND,row['oracleBindingId'])
                require_current_oracle_binding(binding,row,current,corpus)
            else:
                require_current_oracle_binding(None,row,current,corpus)
            if row.get('semanticVerified'):
                fixture=proof.get('fixture',row['semanticFixtureId'])
                require(fixture.get('runtimeFixtureNamespaceHash')==current['runtimeFixtureNamespaceHash']
                    and fixture.get('oracleBindingVersion')==current['oracleBindingVersion'],
                    'PC_NEXT_SFT_RUNTIME_IMPLEMENTATION_CHANGED',status=409)
                receipt=_verifier(proof,row['semanticReceiptId'],fixture,passed=True)
                require(digest(fixture)==row['semanticFixtureHash'] and digest(receipt)==row['semanticReceiptHash']
                    and receipt.get('codeHash')==row['programHash'],'PC_NEXT_SFT_RUNTIME_PROOF_CHANGED',status=409)
        proof.finish();return digest(record)

    def preflight(self,plan,ledger):
        from LLM.foundation.acceptance import source_snapshot
        from LLM.core import file_hash
        from LLM.foundation.backend import verify as verify_backend
        from LLM.foundation.checkpoint import verify as verify_checkpoint
        from LLM.foundation.corpus import verify as verify_corpus
        from LLM.foundation.prepare_data import ACTOR
        project=self.control.foundation.project(plan.project_id)
        require(project.get('executionScope')=='pc_validation' and project.get('origin')=='random_init'
            and not project.get('testOnly') and not project.get('fixture'),'REAL_PC_RANDOM_INIT_PROJECT_REQUIRED')
        require(self.policy().get('enabled') is False,'PC_NEXT_AUTOMATION_MUST_BE_DISABLED')
        self.control.foundation.validate_policy(project);self.processes_gone()
        own=set(ledger.get('phaseJobs',{}).values())|set(ledger.get('calibrationJobs',{}).values())|set(ledger.get('qualificationJobs',{}).values())
        for row in self.store.rows("SELECT id,state FROM jobs WHERE state IN ('queued','running','pause_requested','paused','cancelling')"):
            require(row['id'] in own and row['state'] in ('paused','queued'),'PC_NEXT_PENDING_JOB_REQUIRES_REVIEW',status=409)
        backend=verify_backend(strict_platform=False)
        parent=self.store.get('foundation_checkpoint',plan.parent_checkpoint_id);verify_checkpoint(self.store,parent)
        parent_run=self.store.get('foundation_run',parent.get('runId') or '')
        parent_job=self.store.job(parent.get('runId') or '')
        require(parent.get('projectId')==plan.project_id and parent.get('state')=='ready'
            and not parent.get('fixture') and not parent.get('testOnly') and parent.get('trainedContextTokens',0)>=512
            and parent_job.get('state')=='completed' and parent_job.get('type')=='foundation_cpt'
            and parent_run.get('stage')=='cpt' and parent_run.get('config',{}).get('sequenceLength')==512
            and parent_job.get('payload',{}).get('projectId')==plan.project_id,
            'PC_NEXT_PARENT_NOT_READY')
        raw=self.store.get('foundation_data_preparation',plan.raw_preparation_id)
        sft=self.store.get('foundation_data_preparation',plan.sft_preparation_id)
        for record in (raw,sft):
            require(record.get('state')=='ready' and record.get('actor')==ACTOR and record.get('projectId')==plan.project_id
                and record.get('tokenizerId')==project['tokenizerId'] and record.get('nasModified') is False,
                'PC_NEXT_PREPARATION_NOT_READY')
        raw_corpus=self.store.get('foundation_corpus',raw['rawCorpusId'])
        sft_corpus=self.store.get('foundation_corpus',sft['sftCorpusId'])
        verify_corpus(self.store,raw_corpus);verify_corpus(self.store,sft_corpus)
        require(raw_corpus['manifestHash']==raw['rawCorpusHash'] and sft_corpus['manifestHash']==sft['sftCorpusHash']
            and sft.get('rawPreparationId')==raw['id'] and sft.get('rawCorpusId')==raw_corpus['id']
            and sft.get('rawSelectedVersionIdsHash')==raw.get('selectedVersionIdsHash')
            and raw.get('holdoutUnchanged') is True and raw.get('originalRightsPreserved') is True,
            'PC_NEXT_PREPARATION_CHANGED')
        token=self.store.get('foundation_tokenizer',project['tokenizerId'])
        require(raw['tokenizerHash']==sft['tokenizerHash']==token['tokenizerHash']==parent['tokenizerHash'],
            'TOKENIZER_HASH_MISMATCH')
        proof={'rawPreparationId':raw['id'],'rawPreparationHash':digest(raw),'rawCorpusId':raw_corpus['id'],
            'rawCorpusHash':raw_corpus['manifestHash'],'selectedVersionIdsHash':raw['selectedVersionIdsHash'],
            'sftPreparationId':sft['id'],'sftPreparationHash':digest(sft),'sftCorpusId':sft_corpus['id'],
            'sftCorpusHash':sft_corpus['manifestHash'],'sftTargetTokens':sft_corpus['sftTargetTokens'],
            'tokenizerId':token['id'],'tokenizerHash':token['tokenizerHash'],
            'runtimeVerificationHash':self.runtime_proof(sft,sft_corpus),'runtimeVerificationId':plan.runtime_verification_id,
            'sourceHash':source_snapshot()['hash'],'backendLockHash':backend['backendLockHash'],
            'coordinatorImplementationHash':file_hash(Path(__file__)),
            'parentCheckpointHash':digest(parent),'originals':self.originals()}
        self.inputs=proof
        return proof

    def start(self):
        import threading
        from LLM.foundation.service import JOB_TYPES
        from .pc_budget_guard import identity
        ledger=self.load(self.lock['operationId'])
        self.control.pc_budget_operation={'operationId':ledger['operationId'],
            'authorizationId':ledger['authorizationId'],'authorizationHash':ledger['authorizationHash'],
            'coordinator':identity()}
        self.control.stop_event.clear()
        self.store.put('worker',{'capabilities':sorted(JOB_TYPES),'state':'online','observedAt':now(),
            'executionScope':'pc_validation','foundationSharedVolume':True},self.control.owner)
        def policy_heartbeat():
            from LLM.foundation.acceptance import source_snapshot
            while not self.control.stop_event.wait(5):
                try:
                    require(source_snapshot()['hash']==self.inputs['sourceHash'],'PC_NEXT_SOURCE_CHANGED')
                    require(self.policy().get('enabled') is False,'PC_NEXT_AUTOMATION_MUST_BE_DISABLED')
                    if self.deadline and time.time()>=self.deadline-self.plan.reserve_seconds:self.control.stop_event.set();return
                    if self.control.active:self.store.heartbeat(self.control.active)
                    self.store.put('worker',{'state':'online','observedAt':now()},self.control.owner)
                    self.control.bridge.refresh_policy()
                except Exception as error:
                    self.fatal=error;self.control.stop_event.set();return
        # Deliberately omit the general scheduler/GC/self-code/release supervisor.
        for target,name in ((self.control._work,'seed-llm-worker'),(policy_heartbeat,'pc-next-policy-heartbeat')):
            thread=threading.Thread(target=target,daemon=True,name=name);thread.start();self.control.threads.append(thread)

    def stop(self):
        self.control.stop()
        require(not any(thread.is_alive() for thread in self.control.threads),'PC_NEXT_WORKER_NOT_STOPPED',status=409)
        worker=self.store.get('worker',self.control.owner,{})
        if worker:self.store.put('worker',{'state':'offline','observedAt':now()},self.control.owner,expected=worker['version'])

    def job(self,identity):
        if self.fatal:raise self.fatal
        return self.store.job(identity)

    def command(self,identity,action):return self.control.foundation.command(identity,action)
    def pause_reason(self,identity):return self.store.get('foundation_run',identity,{}).get('pauseReason')

    def expire_jobs(self,ledger):
        identities=set(ledger.get('phaseJobs',{}).values())|set(ledger.get('calibrationJobs',{}).values())|set(ledger.get('qualificationJobs',{}).values())
        for identity in identities:
            job=self.store.job(identity)
            require(job['payload'].get('projectId')==self.plan.project_id,'PC_NEXT_JOB_CHANGED')
            if job['state'] in ('queued','running'):self.control.command(identity,'cancel')

    def has_profile(self,body):
        p,corpus,parent,token,cfg=self.control.foundation.training_inputs(self.plan.project_id,body,body['stage'])
        return bool(self.control.foundation.profile_for(p,cfg,body['stage'],corpus))

    def calibrate(self,body):return self.control.foundation.calibrate(self.plan.project_id,body)
    def train(self,body):return self.control.foundation.start_training(self.plan.project_id,body,body['stage'])

    def assert_job(self,job,body,kind):
        p,corpus,parent,token,cfg=self.control.foundation.training_inputs(self.plan.project_id,body,body['stage'])
        expected={'projectId':p['id'],'corpusId':corpus['id'],'parentCheckpointId':parent['id'],
            'tokenizerId':token['id'],'stage':body['stage'],'config':cfg,'continueCorpus':body['continueCorpus']}
        require(job['type']==kind and all(job['payload'].get(k)==v for k,v in expected.items()),'PC_NEXT_JOB_CHANGED',status=409)

    def checkpoint(self,job_id):
        from LLM.foundation.checkpoint import verify
        run=self.store.get('foundation_run',job_id)
        identity=run.get('lastCheckpointId') or run.get('checkpointId')
        require(identity and run.get('completedOptimizerSteps',0)>0,'NO_EFFECTIVE_TRAINING')
        cp=self.store.get('foundation_checkpoint',identity);verify(self.store,cp)
        require(cp.get('runId')==job_id and cp.get('projectId')==self.plan.project_id and cp.get('state')=='ready',
            'PC_NEXT_CHECKPOINT_CHANGED')
        return cp

    def qualify(self,checkpoint,corpus):
        return self.control.foundation.evaluate({'projectId':self.plan.project_id,'checkpointId':checkpoint,
            'kind':'qualification','corpusId':corpus})

    def qualification(self,identity,checkpoint,inputs):
        value=self.store.get('foundation_evaluation',identity)
        require(value.get('state')=='completed' and value.get('kind')=='qualification'
            and value.get('artifactId')==checkpoint and value.get('corpusId')==inputs['rawCorpusId']
            and value.get('corpusHash')==inputs['rawCorpusHash'] and value.get('tokenizerHash')==inputs['tokenizerHash']
            and value.get('targetTokenCount',0)>=100_000 and value.get('independentGroupCount',0)>=20
            and not value.get('fixture') and set(value.get('nll',{}))=={'candidate','random','unigram'}
            and all(type(v) in (int,float) and math.isfinite(v) and v>0
                for v in value.get('nll',{}).values()),'PC_NEXT_QUALIFICATION_CHANGED')
        return value

    def drain(self,identity,deadline):
        if not identity:return
        current=self.store.job(identity)
        if current['state'] in TERMINAL or current['state']=='paused':return
        if current['type'] in TRAIN and current['state']=='running':self.command(identity,'pause')
        elif current['state'] not in ('pause_requested','cancelling'):self.control.command(identity,'cancel')
        while time.time()<deadline-5:
            current=self.store.job(identity)
            if current['state'] in TERMINAL or current['state']=='paused':
                if current['type'] in TRAIN and current['state']=='paused':self.checkpoint(identity)
                return
            time.sleep(.5)
        raise Fault('PC_NEXT_WORKER_DRAIN_INCOMPLETE',status=409)

    def close(self):
        self.rag.close();self.store.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation-root',default='LLM/runtime-data/preparation/official-v1')
    parser.add_argument('--operation-id',required=True)
    parser.add_argument('--runtime-verification-id',required=True)
    parser.add_argument('--parent-checkpoint-id',default=None)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args();plan=original_plan(args.preparation_root,args.runtime_verification_id,
        parent_checkpoint_id=args.parent_checkpoint_id)
    require(re.fullmatch('[A-Za-z0-9_-]{1,100}',args.operation_id),'PC_NEXT_OPERATION_REQUIRED')
    if not args.execute:
        print(json.dumps({'state':'plan_only','plan':asdict(plan),'operationId':args.operation_id,
            'storeModified':False,'modelExecutionPerformed':False,'nasModified':False},indent=2));return
    adapter=LocalAdapter(args.preparation_root,plan)
    try:
        result=Coordinator(adapter,plan,args.operation_id).run()
        print(json.dumps({k:result.get(k) for k in ('state','stage','operationId','checkpointId','reasonCodes',
            'elapsedSeconds','deadlineEpoch','policyRestore','nasModified','activationPerformed')},ensure_ascii=False,indent=2))
        if result.get('state')!='candidate_trained':raise SystemExit(2)
    finally:adapter.close()


if __name__=='__main__':main()
