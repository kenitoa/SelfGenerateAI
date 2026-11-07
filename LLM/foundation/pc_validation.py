"""Real, bounded PC candidate training. Never deploys or claims NAS measurements."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from LLM.core import Fault, atomic_json, digest, now, require


def attach(control, rag_root):
    from RAG.core import Store
    from RAG.Pre.service import PreService
    from RAG.Post.service import PostService
    from LLM.integration import RagBridge
    rag = Store(rag_root)
    pre = PreService(rag)
    post = PostService(rag, pre)
    bridge = RagBridge(control, rag, pre, post)
    control.bridge = bridge
    control.releases.rag = bridge
    bridge.refresh_policy()
    return rag


def run(preparation_root, *, max_seconds=14400, tokens_per_phase=1_000_000,cpu_threads=None,micro_batch_size=None):
    require(type(max_seconds) is int and 60 <= max_seconds <= 86400, 'CONFIG_INVALID')
    require(type(tokens_per_phase) is int and 65536 <= tokens_per_phase <= 100_000_000, 'CONFIG_INVALID')
    require(cpu_threads is None or type(cpu_threads) is int and 1<=cpu_threads<=4,'CONFIG_INVALID')
    require(micro_batch_size is None or type(micro_batch_size) is int and micro_batch_size in (1,2,4),'CONFIG_INVALID')
    root = Path(preparation_root).resolve()
    from .acceptance import require_local_path
    require_local_path(root)
    ready = root/'reports/readiness.json'
    require(ready.is_file(), 'PREPARATION_NOT_READY')
    preparation = json.loads(ready.read_text('utf-8'))
    require(preparation.get('state') == 'ready' and preparation.get('nasModified') is False,
            'PREPARATION_NOT_READY')
    control_root = Path(preparation['controlRoot']).resolve()
    rag_root = Path(preparation['ragRoot']).resolve()
    # This local CLI cannot redirect training or write a NAS share.
    for path in (root, control_root, rag_root):
        require_local_path(path,root)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                      TOKENIZERS_PARALLELISM='false', RAYON_NUM_THREADS='1')
    from LLM.service import Control
    from .backend import verify
    from .acceptance import source_snapshot
    control = Control(control_root, worker=True, execution_scope='pc_validation')
    rag = None
    identity = preparation['projectId']
    ledger_id = 'pc-candidate-'+identity
    began = time.monotonic()
    ledger = {}
    def update(**values):
        nonlocal ledger
        ledger = control.store.put('foundation_pc_validation', {
            **values, 'projectId':identity, 'executionScope':'pc_validation',
            'testOnly':False, 'nasModified':False, 'nasMeasured':False,
            'qualityCriteriaLowered':False, 'observedAt':now(),
            'elapsedSeconds':time.monotonic()-began}, ledger_id)
        atomic_json(root/'reports/pc-candidate.json', ledger)
        print(json.dumps({k:ledger.get(k) for k in
            ('state','stage','jobId','checkpointId','reasonCodes','elapsedSeconds')},
            ensure_ascii=False), flush=True)
    def budget_check():
        require(time.monotonic()-began < max_seconds, 'PC_VALIDATION_TIME_LIMIT', status=409)
    def wait(job):
        while True:
            budget_check()
            current = control.store.job(job['id'])
            if current['state'] == 'completed':return current.get('result', {})
            if current['state'] in ('failed','cancelled'):
                error=current.get('error') or {}
                recoverable=(current['state']=='failed' and current['type'] in ('foundation_pretrain','foundation_cpt','foundation_sft')
                    and error.get('code')=='JOB_FAILED' and 'database is locked' in error.get('message','').lower())
                counts=ledger.get('recoveryAttempts',{})
                if recoverable and counts.get(current['id'],0)<3:
                    update(recoveryAttempts={**counts,current['id']:counts.get(current['id'],0)+1})
                    time.sleep(2);control.foundation.command(current['id'],'retry');continue
                raise Fault((current.get('error') or {}).get('code', 'PC_CANDIDATE_JOB_FAILED'), status=409)
            budget_check()
            if current['state'] == 'paused':
                require(current['type'] in ('foundation_pretrain','foundation_cpt','foundation_sft'),
                        'PC_NONTRAIN_JOB_RETRY_REQUIRED',status=409)
                reason = control.store.get('foundation_run', current['id'], {}).get('pauseReason')
                require(reason in ('SLICE_LIMIT', 'WORKER_SHUTDOWN', None),
                        reason or 'PC_CANDIDATE_PAUSED', status=409)
                time.sleep(2)
                control.foundation.command(current['id'], 'resume')
            time.sleep(.5)
    try:
        backend = verify(strict_platform=False)
        project = control.store.get('foundation_project', identity)
        require(not project.get('testOnly') and project.get('origin') == 'random_init',
                'REAL_RANDOM_INIT_PROJECT_REQUIRED')
        if project.get('executionScope', 'nas') != 'pc_validation':
            require(not project.get('genesisId') and not project.get('latestCheckpointId')
                    and not control.foundation._pending(identity), 'EXECUTION_SCOPE_REQUALIFICATION_REQUIRED')
            control.store.put('foundation_project', {'executionScope':'pc_validation',
                'pcValidationAuthorized':True}, identity)
            control.store.event('foundation.pc_validation.authorized', identity,
                                {'source':'explicit_user_authorization','nasModified':False})
        rag = attach(control, rag_root)
        from .prepare_data import ACTOR
        prepared_record = control.store.get('foundation_data_preparation', 'official-v1')
        require(prepared_record.get('actor') == ACTOR and prepared_record.get('state') == 'ready'
                and prepared_record.get('projectId') == identity
                and prepared_record.get('rawCorpusHash') == preparation.get('rawCorpusHash')
                and prepared_record.get('tokenizerHash') == preparation.get('tokenizerHash'),
                'PREPARATION_CHANGED')
        previous = control.store.get('foundation_pc_validation', ledger_id, {})
        recipe={'phases':[('pretrain',128),('cpt',256),('cpt',512),('sft',512)],'tokensPerPhase':tokens_per_phase}
        if cpu_threads is not None:recipe['newRunCpuThreads']=cpu_threads
        if micro_batch_size is not None:recipe['newRunMicroBatchSize']=micro_batch_size
        recipe_hash=digest(recipe)
        if previous.get('jobs'):
            require(previous.get('preparationHash')==digest(preparation),'PREPARATION_CHANGED')
            require(not previous.get('recipeHash') or previous['recipeHash']==recipe_hash,'PC_VALIDATION_RECIPE_CHANGED')
        phase_jobs=dict(previous.get('phaseJobs',{}))
        # Reconcile older ledgers using actual run configuration, never a supplied score.
        for job_id in previous.get('jobs',[]):
            run_record=control.store.get('foundation_run',job_id,{})
            for phase,(stage,context) in enumerate([('pretrain',128),('cpt',256),('cpt',512),('sft',512)]):
                if run_record.get('stage')==stage and run_record.get('config',{}).get('sequenceLength')==context:phase_jobs[str(phase)]=job_id
        update(state='running', stage='initialization', backendReceipt=backend,
               sourceHash=source_snapshot()['hash'], preparationHash=digest(preparation),
               jobs=previous.get('jobs', []), completedPhases=previous.get('completedPhases', []),
               evaluations=previous.get('evaluations', []), reasonCodes=[],recipeHash=recipe_hash,
               phaseJobs=phase_jobs,phaseCheckpoints=previous.get('phaseCheckpoints',{}),
               phaseEvaluations=previous.get('phaseEvaluations',{}),calibrationJobs=previous.get('calibrationJobs',{}))
        control.start()
        project = control.store.get('foundation_project', identity)
        if not project.get('genesisId'):
            pending=control.foundation._pending(identity)
            existing=next((row for row in pending if row['type']=='foundation_initialize'),None)
            job = existing or control.foundation.initialize(identity)
            update(jobId=job['id'],initializationJobId=job['id'])
            wait(job)
        phases = [('pretrain',128), ('cpt',256), ('cpt',512), ('sft',512)]
        for index, (stage, context) in enumerate(phases):
            budget_check()
            config = {'sequenceLength':context, 'maxTokenExposures':tokens_per_phase,
                      'sliceSeconds':900, 'maxActiveSeconds':max_seconds,
                      'maxEpochs':2 if stage == 'sft' else 3}
            if cpu_threads is not None:config['cpuThreads']=cpu_threads
            if micro_batch_size is not None:
                config.update(microBatchSize=micro_batch_size,gradientAccumulation=8//micro_batch_size)
            if stage == 'sft':config['learningRate'] = 5e-5
            body = {'stage':stage, 'config':config}
            saved_job=ledger.get('phaseJobs',{}).get(str(index))
            if saved_job:
                job=control.store.job(saved_job)
                require(job['type']=='foundation_'+stage and job['payload'].get('projectId')==identity
                        and job['payload'].get('config',{}).get('sequenceLength')==context
                        and job['payload'].get('config',{}).get('maxTokenExposures')==tokens_per_phase,
                        'PC_VALIDATION_RECIPE_CHANGED')
                update(stage=stage+'_'+str(context),jobId=job['id'])
            else:
                p, corpus, parent, token, effective = control.foundation.training_inputs(identity, body, stage)
                if not control.foundation.profile_for(p, effective, stage,corpus):
                    calibration_id=ledger.get('calibrationJobs',{}).get(str(index))
                    calibration=control.store.job(calibration_id) if calibration_id else None
                    if calibration and calibration['state']=='paused':
                        with control.store.tx():control.store.db.execute("UPDATE jobs SET state='cancelled',updated=? WHERE id=? AND state='paused'",(now(),calibration_id))
                    if calibration is None or calibration['state'] in ('failed','cancelled','paused','completed'):
                        calibration=control.foundation.calibrate(identity,body)
                    update(stage='calibration_'+stage+'_'+str(context),jobId=calibration['id'],
                           calibrationJobs={**ledger.get('calibrationJobs',{}),str(index):calibration['id']})
                    result=wait(calibration)
                    require(result.get('passed') is True,'PC_CALIBRATION_NOT_PASSED',status=409)
                job=control.foundation.start_training(identity,body,stage)
                update(stage=stage+'_'+str(context),jobId=job['id'],jobs=[*ledger.get('jobs',[]),job['id']],
                       phaseJobs={**ledger.get('phaseJobs',{}),str(index):job['id']})
            wait(job)
            run_record=control.store.get('foundation_run',job['id'])
            cp=run_record.get('lastCheckpointId') or run_record.get('checkpointId')
            require(cp and run_record.get('completedOptimizerSteps',0)>0,'NO_EFFECTIVE_TRAINING')
            from .checkpoint import verify as verify_checkpoint
            verify_checkpoint(control.store,control.store.get('foundation_checkpoint',cp))
            update(checkpointId=cp,phaseCheckpoints={**ledger.get('phaseCheckpoints',{}),str(index):cp})
            if stage != 'sft':
                exam_id=ledger.get('phaseEvaluations',{}).get(str(index))
                exam=control.store.job(exam_id) if exam_id else None
                if exam is None:
                    old=next((row for row in control.store.list('foundation_evaluation') if row.get('projectId')==identity and row.get('kind')=='qualification' and row.get('artifactId')==cp and row.get('state')=='completed'),None)
                    if old:exam=control.store.job(old['id'])
                if exam is None:
                    exam=control.foundation.evaluate({'projectId':identity,'checkpointId':cp,'kind':'qualification','corpusId':preparation['rawCorpusId']})
                update(stage='qualification', jobId=exam['id'])
                wait(exam)
                evaluation = control.store.get('foundation_evaluation',exam['id'])
                require(evaluation.get('artifactId')==cp and evaluation.get('kind')=='qualification','PC_VALIDATION_EXAM_MISMATCH')
                update(evaluations=list(dict.fromkeys([*ledger.get('evaluations',[]),evaluation['id']])),
                       phaseEvaluations={**ledger.get('phaseEvaluations',{}),str(index):evaluation['id']},
                       lastQualificationGate=evaluation.get('gate'))
            update(completedPhases=sorted(set([*ledger.get('completedPhases',[]),index])))
        # Completion of training is separate from passing independent model exams.
        update(state='candidate_trained', stage='independent_exams_required',
               reasonCodes=['INDEPENDENT_RELEASE_SELF_CODE_SPECIALIZATION_EXAMS_REQUIRED'],
               actualModelQualityPassed=False)
        return ledger
    except Exception as error:
        update(state='held', stage=ledger.get('stage','preflight'),
               reasonCodes=[getattr(error,'code','PC_VALIDATION_FAILED')],
               actualModelQualityPassed=False)
        raise
    finally:
        control.stop()
        if rag is not None:rag.close()
        control.store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation-root',default='LLM/runtime-data/preparation/official-v1')
    parser.add_argument('--max-seconds',type=int,default=14400)
    parser.add_argument('--tokens-per-phase',type=int,default=1_000_000)
    parser.add_argument('--cpu-threads',type=int,default=None)
    parser.add_argument('--micro-batch-size',type=int,default=None)
    args = parser.parse_args()
    run(args.preparation_root,max_seconds=args.max_seconds,tokens_per_phase=args.tokens_per_phase,
        cpu_threads=args.cpu_threads,micro_batch_size=args.micro_batch_size)


if __name__ == '__main__':main()
