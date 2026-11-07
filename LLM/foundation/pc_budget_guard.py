"""Opt-in PC operation deadline. Environment names locate Store authority only.

The hard timer never accesses SQLite or waits for checkpoint/cleanup. A second
thread requests a normal yield and narrowly recovers an orphan's owned cap.
OS shutdown/termination of this process also terminates both threads; the next
coordinator invocation recovers an outstanding grant without restarting work.
"""
from __future__ import annotations

import os
import threading
import time

from LLM.core import Fault,Store,digest,now,require,uid

KIND='foundation_pc_budget_binding'
LEDGER='foundation_pc_next_candidate'
AUTH='foundation_pc_next_authorization'
ENV='SEED_PC_BUDGET_BINDING_ID'
TRAIN=frozenset(('foundation_cpt','foundation_sft'))
TYPES=TRAIN|{'foundation_calibrate','foundation_evaluate'}
HARD_MARGIN_SECONDS=1.0
ORPHAN_GRACE_SECONDS=120.0


def identity():
    import psutil
    return {'pid':os.getpid(),'createTime':psutil.Process(os.getpid()).create_time()}


def alive(value):
    import psutil
    try:return psutil.Process(value['pid']).create_time()==value['createTime']
    except psutil.NoSuchProcess:return False
    # Access denied is uncertainty, never evidence that an owner has died.


def budget_fields(ledger):
    return {**{key:ledger[key] for key in ('planHash','startedEpoch','deadlineEpoch','startedMonotonic','bootId')},
        'maxSeconds':ledger['plan']['max_seconds']}


def _authority(store,binding,job=None,*,creating=False):
    if not creating:
        require(binding.get('id') and store.get(KIND,binding['id'])==binding,
            'PC_BUDGET_AUTHORITY_CHANGED',status=409)
    require(binding.get('version')==1 and binding.get('executionScope')=='pc_validation',
        'PC_BUDGET_AUTHORITY_CHANGED',status=409)
    auth=store.get(AUTH,binding['authorizationId'])
    ledger=store.get(LEDGER,binding['operationId'])
    require(auth.get('version')==1 and digest(auth)==binding['authorizationHash']==ledger.get('authorizationHash')
        and auth.get('operationId')==ledger.get('operationId')==binding['operationId']
        and auth.get('projectId')==ledger.get('projectId')==binding['projectId']
        and auth.get('executionScope')==ledger.get('executionScope')=='pc_validation'
        and auth.get('budget')==binding['budget']==budget_fields(ledger)
        and ledger.get('authorizationId')==binding['authorizationId'],
        'PC_BUDGET_AUTHORITY_CHANGED',status=409)
    budget=binding['budget'];plan=ledger['plan']
    require(type(budget['maxSeconds']) is int and 1<=budget['maxSeconds']<=14400
        and ledger['planHash']==digest(plan)
        and budget['deadlineEpoch']==budget['startedEpoch']+budget['maxSeconds']
        and plan.get('execution_scope')=='pc_validation'
        and 0<=binding['reserveSeconds']<budget['maxSeconds']
        and binding['reserveSeconds']==plan['reserve_seconds'], 'PC_BUDGET_AUTHORITY_CHANGED',status=409)
    grant=ledger.get('policyGrant') or {}
    require(all(grant.get(key)==binding['policyGrant'].get(key) for key in
        ('oldCap','grantedCap','previousVersion','appliedVersion','intentRecordedAt','coordinator')),
        'PC_BUDGET_GRANT_CHANGED',status=409)
    require(grant.get('coordinator')==binding['coordinator'] and all(type(grant.get(key)) is int
        and 1<=grant[key]<=86400 for key in ('oldCap','grantedCap')),
        'PC_BUDGET_GRANT_CHANGED',status=409)
    project=store.get('foundation_project',binding['projectId'])
    require(project.get('executionScope')=='pc_validation','PC_VALIDATION_SCOPE_REQUIRED',status=403)
    current=store.job(binding['jobId'])
    if job is not None:
        require(all(current.get(key)==job.get(key) for key in ('id','type','owner','fence')),
            'PC_BUDGET_JOB_CHANGED',status=409)
    members=set().union(*(set(ledger.get(key,{}).values()) for key in
        ('phaseJobs','calibrationJobs','qualificationJobs')))
    require(current['id'] in members and current['type'] in TYPES
        and current['type']==binding['jobType'] and current['payload'].get('projectId')==binding['projectId']
        and digest(current['payload'])==binding['payloadHash']
        and current['owner']==binding['owner'] and current['fence']==binding['fence'],
        'PC_BUDGET_JOB_CHANGED',status=409)
    return ledger,current


def bind(control,job,*,registration_timeout=5):
    """Called immediately before spawn. Binding and lease check share one TX."""
    context=getattr(control,'pc_budget_operation',None)
    if context is None:return None
    require(control.execution_scope=='pc_validation','PC_VALIDATION_SCOPE_REQUIRED',status=403)
    store=control.store;end=time.monotonic()+registration_timeout
    while True:
        ledger=store.get(LEDGER,context['operationId'])
        members=set().union(*(set(ledger.get(key,{}).values()) for key in
            ('phaseJobs','calibrationJobs','qualificationJobs')))
        if job['id'] in members:break
        require(time.monotonic()<end,'PC_BUDGET_JOB_NOT_REGISTERED',status=409)
        time.sleep(.02)
    require(alive(context['coordinator']),'PC_BUDGET_COORDINATOR_GONE',status=409)
    require(ledger.get('authorizationId')==context['authorizationId']
        and ledger.get('authorizationHash')==context['authorizationHash'],
        'PC_BUDGET_AUTHORITY_CHANGED',status=409)
    grant=ledger.get('policyGrant') or {}
    require(grant.get('appliedVersion') and grant.get('restored') is False,
        'PC_BUDGET_GRANT_REQUIRED',status=409)
    body={'schemaVersion':1,'executionScope':'pc_validation','operationId':context['operationId'],
        'authorizationId':context['authorizationId'],'authorizationHash':context['authorizationHash'],
        'projectId':ledger['projectId'],'jobId':job['id'],'jobType':job['type'],'owner':job['owner'],
        'fence':job['fence'],'payloadHash':digest(job['payload']),'budget':budget_fields(ledger),
        'reserveSeconds':ledger['plan']['reserve_seconds'],'policyGrant':grant,
        'coordinator':context['coordinator'],'observedAt':now()}
    # Synthetic version is used only for validating a not-yet-published record.
    with store.tx():
        current=store.assert_lease(job)
        require(current['state']=='running','PC_BUDGET_JOB_NOT_RUNNING',status=409)
        _authority(store,{**body,'version':1},job,creating=True)
        require(remaining(body)>body['reserveSeconds'],'PC_NEXT_WALL_BUDGET_EXHAUSTED',status=409)
        return store.put(KIND,body,uid(),expected=0)


def remaining(binding):
    import psutil
    budget=binding['budget']
    require(int(psutil.boot_time())==budget['bootId'],'PC_NEXT_REBOOT_REQUIRES_REVIEW',status=409)
    elapsed=time.monotonic()-budget['startedMonotonic']
    require(elapsed>=0,'PC_NEXT_CLOCK_ROLLBACK',status=409)
    return min(budget['deadlineEpoch']-time.time(),budget['maxSeconds']-elapsed)


def recover_cap(store,binding):
    """Metadata-only CAS; never replaces a policy body or accounting counters."""
    require(not alive(binding['coordinator']),'PC_BUDGET_COORDINATOR_PRESENT',status=409)
    with store.tx():
        ledger,_=_authority(store,binding)
        return _restore_cap(store,ledger,{'bindingId':binding['id'],'bindingHash':digest(binding)})


def _restore_cap(store,ledger,evidence):
    grant=ledger['policyGrant']
    if grant.get('restored'):return {'state':'already_restored'}
    policy=store.get('foundation_policy',ledger['projectId'])
    if policy.get('dailyActiveSeconds')!=grant['grantedCap']:
        state='held';reason='PC_NEXT_POLICY_OWNERSHIP_CHANGED';restored_grant=grant
    else:
        restored=store.put('foundation_policy',{'dailyActiveSeconds':grant['oldCap']},
            policy['id'],expected=policy['version'])
        state='restored';reason='PC_NEXT_ORPHAN_RECOVERED_REQUIRES_REVIEW'
        restored_grant={**grant,'restored':True,'restoredVersion':restored['version']}
    receipt=store.put('foundation_pc_budget_recovery',{**evidence,'operationId':ledger['operationId'],'state':state,
        'reason':reason,'onlyDailyActiveSecondsChanged':state=='restored','observedAt':now()},expected=0)
    store.put(LEDGER,{'policyGrant':restored_grant,'policyRestore':{'state':state,'reason':reason,
        'recoveryId':receipt['id'],'onlyDailyActiveSecondsChanged':state=='restored'},
        'state':'held','orphanRecoveryRequiresReview':True,
        'reasonCodes':sorted(set([*ledger.get('reasonCodes',[]),reason]))},ledger['id'],expected=ledger['version'])
    return receipt


def recover_operation(store,operation_id):
    ledger=store.get(LEDGER,operation_id,{})
    if not ledger:return
    require(not ledger.get('orphanRecoveryRequiresReview'),'PC_NEXT_ORPHAN_RECOVERY_REQUIRES_REVIEW',status=409)
    grant=ledger.get('policyGrant') or {}
    if not grant or grant.get('restored'):return
    if grant.get('coordinator') and not alive(grant['coordinator']):
        # Covers grant->first-spawn gaps without manufacturing a job binding.
        with store.tx():
            ledger=store.get(LEDGER,operation_id)
            auth=store.get(AUTH,ledger.get('authorizationId',''))
            require(auth.get('version')==1 and digest(auth)==ledger.get('authorizationHash')
                and auth.get('operationId')==operation_id and auth.get('projectId')==ledger['projectId']
                and auth.get('budget')==budget_fields(ledger)
                and auth.get('executionScope')==ledger.get('executionScope')=='pc_validation'
                and store.get('foundation_project',ledger['projectId']).get('executionScope')=='pc_validation',
                'PC_BUDGET_AUTHORITY_CHANGED',status=409)
            require(not alive(ledger['policyGrant']['coordinator']),'PC_BUDGET_COORDINATOR_PRESENT',status=409)
            _restore_cap(store,ledger,{'authorizationId':auth['id'],'authorizationHash':digest(auth),
                'grantHash':digest(ledger['policyGrant']),'gapRecovery':True})
        raise Fault('PC_NEXT_ORPHAN_RECOVERY_REQUIRES_REVIEW',status=409)
    # Bound executions are the authoritative record of which coordinator owned
    # the temporary grant. No new grant or work is authorized by recovery.
    rows=store.rows("SELECT body FROM records WHERE kind=? AND json_extract(body,'$.operationId')=? ORDER BY rowid DESC",
        (KIND,operation_id))
    if rows:
        import json
        binding=json.loads(rows[0]['body'])
        if not alive(binding['coordinator']):
            recover_cap(store,binding)
            raise Fault('PC_NEXT_ORPHAN_RECOVERY_REQUIRES_REVIEW',status=409)


class Guard:
    def __init__(self,store,binding,job):
        self.root=store.root;self.binding=binding;self.job=job
        _authority(store,binding,job)
        require(store.assert_lease(job)['state']=='running','PC_BUDGET_JOB_NOT_RUNNING',status=409)
        self.stopped=threading.Event();self.lock=threading.Lock();self.orphan_stop=float('inf')
        # Compute timers only after Store authority validation. No Store call is
        # permitted in _hard; even a permanently locked database cannot extend it.
        self.hard_stop=time.monotonic()+max(0,remaining(binding)-HARD_MARGIN_SECONDS)
        self.hard=threading.Thread(target=self._hard,daemon=True,name='pc-budget-hard-stop')
        self.monitor=threading.Thread(target=self._monitor,daemon=True,name='pc-budget-yield')

    def start(self):self.hard.start();self.monitor.start();return self

    def _hard(self):
        while not self.stopped.wait(.05):
            try:parent_gone=not alive(self.binding['coordinator'])
            except Exception:parent_gone=True
            if parent_gone:
                with self.lock:self.orphan_stop=min(self.orphan_stop,time.monotonic()+ORPHAN_GRACE_SECONDS)
            with self.lock:deadline=min(self.hard_stop,self.orphan_stop)
            if time.monotonic()>=deadline or time.time()>=self.binding['budget']['deadlineEpoch']-HARD_MARGIN_SECONDS:
                os._exit(124)

    def _monitor(self):
        store=None;yielded=False;recovered=False
        try:
            while not self.stopped.wait(.1):
                try:parent_gone=not alive(self.binding['coordinator'])
                except Exception:parent_gone=True  # uncertainty stops compute, never grants recovery
                if parent_gone:
                    with self.lock:self.orphan_stop=min(self.orphan_stop,time.monotonic()+ORPHAN_GRACE_SECONDS)
                due=parent_gone or time.monotonic()>=self.hard_stop+HARD_MARGIN_SECONDS-self.binding['reserveSeconds']
                if not due:continue
                try:
                    if store is None:
                        store=Store(self.root);store.db.set_busy_timeout(100)
                    if not yielded:
                        with store.tx():
                            _,current=_authority(store,self.binding,self.job)
                            store.assert_lease(self.job)
                            if current['state']=='running':
                                state='pause_requested' if current['type'] in TRAIN else 'cancelling'
                                store.db.execute('UPDATE jobs SET state=?,updated=? WHERE id=? AND owner=? AND fence=? AND state=\'running\'',
                                    (state,now(),current['id'],current['owner'],current['fence']))
                                if current['type'] in TRAIN:store.put('foundation_run',{
                                    'pauseReason':'PC_OPERATION_PARENT_GONE' if parent_gone else 'PC_OPERATION_DEADLINE'},current['id'])
                            yielded=True
                    if parent_gone and not recovered:
                        recover_cap(store,self.binding);recovered=True
                except Exception:
                    # Never let SQLite contention delay the independent timer.
                    pass
        finally:
            if store:store.close()

    def close(self):
        self.stopped.set();self.hard.join(timeout=.5);self.monitor.join(timeout=.5)


def start(control,job):
    binding_id=os.environ.get(ENV)
    if not binding_id:return None
    require(control.execution_scope=='pc_validation','PC_VALIDATION_SCOPE_REQUIRED',status=403)
    binding=control.store.get(KIND,binding_id)
    return Guard(control.store,binding,job).start()
