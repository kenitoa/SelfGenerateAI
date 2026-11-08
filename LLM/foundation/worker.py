"""Network-isolated, shared-volume NAS worker and bounded model subprocess."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
from LLM.core import Store, Fault, atomic_json, require, digest, now


def evaluation_environment(control,job):
    """Measure the actual child setting; environment alone is not approval."""
    import os
    attempt_id=os.environ.get('SEED_EVALUATION_ATTEMPT_ID')
    if not attempt_id:return None
    from .evaluation_restart import child_runtime_threads
    expected=child_runtime_threads(control.store)
    attempt=control.store.get('foundation_evaluation_attempt',attempt_id)
    require(job['type']=='foundation_evaluate' and all(attempt.get(key)==value for key,value in
        {'jobId':job['id'],'owner':job['owner'],'fence':job['fence'],'cpuThreads':expected,
         'executionScope':control.execution_scope,'payloadHash':digest(job['payload'])}.items()),
        'EVALUATION_RUNTIME_CONTEXT_CHANGED',status=409)
    environment={name:os.environ.get(name) for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}
    require(all(value==str(expected) for value in environment.values()),'EVALUATION_CPU_ENV_MISMATCH',status=409)
    import torch
    if attempt.get('expectedInteropThreads')==1:
        require(expected==4 and control.execution_scope=='pc_validation'
            and os.environ.get('SEED_EVALUATION_SCOPE')=='pc_validation','PC_RELEASE_EVALUATION_REQUIRED',status=403)
        try:torch.set_num_interop_threads(1)
        except RuntimeError as error:
            print('EVALUATION_INTEROP_THREADS_FAILED: '+str(error),file=sys.stderr,flush=True)
            raise Fault('EVALUATION_INTEROP_THREADS_FAILED',status=503) from error
        require(torch.get_num_interop_threads()==1,'EVALUATION_INTEROP_THREADS_FAILED',status=503)
    observed=torch.get_num_threads()
    require(observed==expected,'EVALUATION_CPU_THREADS_NOT_OBSERVED',status=409)
    process_identity={}
    if os.environ.get('SEED_EVALUATION_JOB_ID') and control.execution_scope=='pc_validation':
        import psutil
        process_identity={'process':{'pid':os.getpid(),'createTime':psutil.Process(os.getpid()).create_time()}}
    return control.store.put('foundation_evaluation_worker_environment',{
        'schemaVersion':1,'attemptId':attempt_id,'attemptHash':digest(attempt),'jobId':job['id'],
        'owner':job['owner'],'fence':job['fence'],'executionScope':control.execution_scope,'cpuThreads':expected,
        'torchNumThreads':observed,'torchNumInteropThreads':torch.get_num_interop_threads(),
        'torchVersion':torch.__version__,'environment':environment,**process_identity,
        'threadChoiceContextHash':attempt['threadChoiceContextHash'],'observedAt':now()},attempt_id,expected=0)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',required=True)
    parser.add_argument('--job-id')
    parser.add_argument('--owner')
    parser.add_argument('--fence',type=int)
    parser.add_argument('--serve',action='store_true')
    parser.add_argument('--health',action='store_true')
    args=parser.parse_args()
    if args.health:
        from .resources import isolated_network
        from .worker_status import has_shared_worker
        store=Store.open_existing(Path(args.data_dir),read_only=True)
        try:
            if not has_shared_worker(store,max_age=45) or not isolated_network():raise SystemExit(1)
        finally:store.close()
        return
    from LLM.service import Control
    if args.serve:
        import os,time,signal,threading
        os.environ['SEED_FOUNDATION_WORKER']='1'
        from .resources import isolated_network
        if not isolated_network():raise SystemExit('OS network isolation is required')
        from .backend import verify
        verify()
        control=Control(Path(args.data_dir),worker=True)
        stopped=threading.Event()
        signal.signal(signal.SIGTERM,lambda *_:stopped.set())
        signal.signal(signal.SIGINT,lambda *_:stopped.set())
        control.start()
        try:
            while not stopped.wait(1):pass
        except KeyboardInterrupt:pass
        finally:control.stop()
        return
    import os
    control=Control(Path(args.data_dir),worker=False,execution_scope=os.environ.get('SEED_EXECUTION_SCOPE','nas'))
    result_path=control.store.root/'foundation/job-results'/str(args.job_id)/'result.json'
    result={'fence':args.fence}
    budget_guard=None
    try:
        job=control.store.job(args.job_id)
        if job['owner']!=args.owner or job['fence']!=args.fence:raise Fault('LEASE_EXPIRED',status=409)
        control.store.assert_lease(job)
        from .pc_budget_guard import start as start_pc_budget_guard
        budget_guard=start_pc_budget_guard(control,job)
        measured=evaluation_environment(control,job)
        result['result']=control.foundation.execute_local(job)
        if measured:
            import torch
            require(torch.get_num_threads()==measured['cpuThreads'],'EVALUATION_CPU_THREADS_NOT_OBSERVED',status=409)
            result.update(evaluationAttemptId=measured['attemptId'],finalTorchNumThreads=torch.get_num_threads(),
                finalTorchNumInteropThreads=torch.get_num_interop_threads())
    except Exception as error:
        result['error']={'code':getattr(error,'code','FOUNDATION_WORKER_FAILED'),'status':getattr(error,'status',503)}
    finally:
        if budget_guard:budget_guard.close()
        atomic_json(result_path,result);control.store.close()
    if result.get('error'):raise SystemExit(1)


if __name__=='__main__':main()
