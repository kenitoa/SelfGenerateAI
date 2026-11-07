"""Sequential, budgeted curricula advanced only by recorded independent dev results."""
from __future__ import annotations

from LLM.core import Fault, require, digest, now
from .schemas import integer, real


def development_gate(evaluation, artifact_id, baseline_id, step):
    """Consume executed verifier rows, never scores supplied in curriculum config."""
    require(evaluation.get('artifactId') == artifact_id and evaluation.get('state', 'completed') in ('completed','held'),
            'CURRICULUM_EVALUATION_MISMATCH')
    rows = evaluation.get('results', {}).get(artifact_id, [])
    groups = {}
    for row in rows:
        if row.get('critical'): continue
        require(isinstance(row.get('group'),str) and isinstance(row.get('passed'),bool), 'INVALID_EVALUATION_RESULT')
        groups.setdefault(row['group'], []).append(row)
    total = len(groups)
    passed = sum(all(r['passed'] for r in values) for values in groups.values())
    syntax = sum(all(r.get('syntaxPassed',r['passed']) for r in values) for values in groups.values())
    rate = passed/total if total else 0.0
    reasons = [code for code in evaluation.get('gate',{}).get('reasonCodes',[]) if code in
        ('VERIFIER_UNAVAILABLE','SANDBOX_PROBE_REQUIRED','CRITICAL_REGRESSION','EVALUATION_FIXTURE_CHANGED')]
    if total < step.get('minGroups',20): reasons.append('INSUFFICIENT_DEV_COVERAGE')
    if rate < step.get('minPassRate',.7): reasons.append('CURRICULUM_QUALITY_NOT_REACHED')
    if not total or syntax/total < step.get('minSyntaxRate',.95): reasons.append('CURRICULUM_SYNTAX_NOT_REACHED')
    if any(r.get('criticalFailure') or r.get('verifierUnavailable') for r in rows): reasons.append('CURRICULUM_VERIFIER_OR_SAFETY_FAILED')
    previous = evaluation.get('results', {}).get(baseline_id, []) if baseline_id else []
    if baseline_id:
        old = {}
        for row in previous:
            if not row.get('critical'): old.setdefault(row['group'], []).append(row)
        if old.keys() != groups.keys(): reasons.append('CURRICULUM_BASELINE_REQUIRED')
        elif old:
            old_rate = sum(all(r['passed'] for r in values) for values in old.values())/len(old)
            if rate+step.get('maxRegression',.02) < old_rate: reasons.append('CURRICULUM_REGRESSION')
    return {'passed':not reasons,'reasonCodes':reasons,'groupCount':total,'passRate':rate,
        'syntaxRate':syntax/total if total else None, 'productionCertified':False}


class Curriculum:
    def __init__(self, control): self.control, self.store = control, control.store

    def configure(self, project_id, body):
        project = self.control.foundation.project(project_id)
        require(isinstance(body,dict) and isinstance(body.get('steps'),list) and 0 < len(body['steps']) <= 8,
                'CURRICULUM_STEPS_REQUIRED')
        clean = []
        for index, item in enumerate(body['steps']):
            require(isinstance(item,dict) and item.get('stage','sft') in ('pretrain','cpt','sft'), 'CONFIG_INVALID')
            corpus = self.store.get('foundation_corpus',item.get('corpusId',''))
            suite = self.store.get('foundation_suite',item.get('suiteId',''))
            require(corpus.get('projectId') == project_id and corpus.get('state') == 'ready', 'CORPUS_NOT_READY')
            require(suite.get('state') == 'ready' and suite.get('kind') in ('dev','regression'), 'CURRICULUM_DEV_SUITE_REQUIRED')
            length = integer(item.get('sequenceLength',128), 'sequenceLength', 8, project['config']['max_position_embeddings'])
            if clean: require(length >= clean[-1]['sequenceLength'], 'CURRICULUM_CONTEXT_MUST_NOT_SHRINK')
            config = dict(item.get('config',{}))
            config.update(sequenceLength=length,
                maxTokenExposures=integer(item.get('maxTargetTokens',65536),'maxTargetTokens',1,1_000_000),
                maxEpochs=2 if item.get('stage','sft') == 'sft' else 3,
                sliceSeconds=min(900,project['policy'].get('sliceSeconds',900)),
                maxActiveSeconds=min(14400,project['policy'].get('dailyActiveSeconds',14400)))
            clean.append({'name':str(item.get('name',f'stage-{index+1}'))[:64], 'stage':item.get('stage','sft'),
                'corpusId':corpus['id'],'corpusHash':corpus['manifestHash'],'suiteId':suite['id'],'suiteHash':suite['suiteHash'],
                'sequenceLength':length,'config':config,
                'minGroups':integer(item.get('minGroups',20),'minGroups',20,1000),
                'minPassRate':real(item.get('minPassRate',.7),'minPassRate',.5,1),
                'minSyntaxRate':real(item.get('minSyntaxRate',.95),'minSyntaxRate',.9,1),
                'maxRegression':real(item.get('maxRegression',.02),'maxRegression',0,.02)})
        previous = self.store.get('foundation_curriculum',project_id,{})
        require(not previous or previous.get('state') in ('completed','held','disabled'), 'CURRICULUM_BUSY')
        return self.store.put('foundation_curriculum', {'projectId':project_id,'enabled':body.get('enabled',True) is True,
            'steps':clean,'planHash':digest(clean),'index':0,'state':'ready','history':[],
            'parentCheckpointId':project.get('latestCheckpointId') or project.get('genesisId'),
            'activeJobId':None,'evaluationJobId':None,'calibrationJobId':None,'candidateCheckpointId':None,
            'servingEligible':False,'automaticPromotion':False}, project_id)

    def _update(self, plan, state, reasons=None, **fields):
        return self.store.put('foundation_curriculum', {'state':state,'reasonCodes':reasons or [],**fields},plan['id'])

    def advance(self, project_id):
        plan = self.store.get('foundation_curriculum',project_id,{})
        if not plan or not plan.get('enabled'): return plan or {'state':'disabled'}
        if plan.get('state') in ('completed','held'): return plan
        try:
            project = self.control.foundation.project(project_id)
            require(not self.store.get('setting','deploymentMaintenance',{}).get('enabled'), 'DEPLOYMENT_MAINTENANCE')
            require(project.get('testOnly') or project['policy'].get('enabled'), 'AUTOMATION_DISABLED')
            if not project.get('testOnly'):
                require(project['policy'].get('automation','always_on_idle') != 'manual','AUTOMATION_DISABLED')
                scheduler=getattr(self.control,'foundation_scheduler',None)
                require(scheduler is not None and scheduler._window(project['policy']),'OUTSIDE_TRAINING_WINDOW')
            from .resources import used_today
            require(used_today(self.store) < project['policy'].get('dailyActiveSeconds',14400), 'DAILY_BUDGET_EXHAUSTED')
            require(digest(plan['steps']) == plan['planHash'], 'CURRICULUM_PLAN_HASH_MISMATCH')
            index = plan['index']
            if index == len(plan['steps']): return self._update(plan,'completed')
            step = plan['steps'][index]
            corpus = self.store.get('foundation_corpus',step['corpusId'])
            suite = self.store.get('foundation_suite',step['suiteId'])
            require(corpus.get('state') == 'ready' and corpus['manifestHash'] == step['corpusHash'], 'POLICY_REVOKED')
            require(suite.get('state') == 'ready' and suite['suiteHash'] == step['suiteHash']
                    and suite['kind'] in ('dev','regression'), 'CURRICULUM_DEV_SUITE_REQUIRED')
            if plan.get('calibrationJobId'):
                measured = self.store.job(plan['calibrationJobId'])
                if measured['state'] in ('failed','cancelled'):
                    return self._update(plan,'held',['CURRICULUM_CALIBRATION_FAILED'])
                if measured['state'] != 'completed': return plan
                profile = self.store.get('foundation_profile',(measured.get('result') or {}).get('id',''))
                require(profile.get('passed') is True and profile.get('projectId') == project_id,
                    'CURRICULUM_CALIBRATION_FAILED')
                return self._update(plan,'ready',calibrationJobId=None)
            if plan.get('evaluationJobId'):
                job = self.store.job(plan['evaluationJobId'])
                if job['state'] in ('failed','cancelled'): return self._update(plan,'held',['CURRICULUM_EVALUATION_FAILED'])
                if job['state'] != 'completed': return plan
                evaluation_id = (job.get('result') or {}).get('id',job['id'])
                evaluation = self.store.get('foundation_evaluation',evaluation_id)
                require(evaluation.get('suiteHash') == step['suiteHash'], 'CURRICULUM_EVALUATION_MISMATCH')
                decision = development_gate(evaluation,plan['candidateCheckpointId'],plan['parentCheckpointId'],step)
                history = [*plan['history'],{'index':index,'checkpointId':plan['candidateCheckpointId'],
                    'evaluationId':evaluation_id,'decision':decision,'at':now()}]
                if not decision['passed']: return self._update(plan,'held',decision['reasonCodes'],history=history)
                return self._update(plan,'ready',index=index+1,parentCheckpointId=plan['candidateCheckpointId'],
                    activeJobId=None,evaluationJobId=None,candidateCheckpointId=None,history=history)
            if plan.get('activeJobId'):
                job = self.store.job(plan['activeJobId'])
                if job['state'] in ('failed','cancelled'): return self._update(plan,'held',['CURRICULUM_TRAINING_FAILED'])
                if job['state'] == 'paused':
                    run = self.store.get('foundation_run',job['id'],{})
                    automatic = run.get('pauseReason') in ('FOREGROUND_REQUEST','SLICE_LIMIT','WORKER_SHUTDOWN',
                        'RESOURCE_PRESSURE','DAILY_BUDGET_EXHAUSTED','OUTSIDE_TRAINING_WINDOW','DEPLOYMENT_MAINTENANCE')
                    if automatic and (run.get('checkpointId') or run.get('lastCheckpointId')):
                        self.control.foundation.command(job['id'],'resume')
                        return self._update(plan,'training')
                if job['state'] != 'completed': return self._update(plan,'paused' if job['state'] == 'paused' else 'training')
                checkpoint_id = (job.get('result') or {}).get('checkpointId')
                checkpoint = self.store.get('foundation_checkpoint',checkpoint_id or '')
                require(checkpoint.get('projectId') == project_id and checkpoint.get('completedOptimizerSteps',0)>0,
                        'CURRICULUM_REAL_TRAINING_REQUIRED')
                evaluation = self.control.foundation.evaluate({'projectId':project_id,'artifactId':checkpoint_id,
                    'suiteId':step['suiteId'],'baselineIds':[plan['parentCheckpointId']]},
                    key='curriculum-eval:'+plan['planHash']+':'+str(index))
                return self._update(plan,'evaluating',candidateCheckpointId=checkpoint_id,evaluationJobId=evaluation['id'])
            busy = self.store.rows("SELECT id FROM jobs WHERE state IN ('queued','running','paused','pause_requested','cancelling') AND type <> 'backup_maintenance'")
            require(not busy,'GLOBAL_MODEL_JOB_BUSY')
            job = self.control.foundation.start_training(project_id,{'corpusId':step['corpusId'],
                'parentCheckpointId':plan['parentCheckpointId'],'config':step['config']},stage=step['stage'],
                key='curriculum-train:'+plan['planHash']+':'+str(index))
            self.store.put('foundation_automation_job',{'projectId':project_id,'jobId':job['id'],'curriculumId':plan['id']},job['id'])
            return self._update(plan,'training',activeJobId=job['id'])
        except Fault as exc:
            if exc.code == 'CALIBRATION_REQUIRED':
                try:
                    measured = self.control.foundation.calibrate(project_id,{'corpusId':step['corpusId'],
                        'parentCheckpointId':plan['parentCheckpointId'],'stage':step['stage'],'config':step['config']},
                        key='curriculum-profile:'+plan['planHash']+':'+str(index))
                    return self._update(plan,'calibrating',calibrationJobId=measured['id'])
                except Fault as calibration:
                    return self._update(plan,'waiting',[calibration.code])
            waiting = {'DAILY_BUDGET_EXHAUSTED','DEPLOYMENT_MAINTENANCE','GLOBAL_MODEL_JOB_BUSY',
                'CALIBRATION_REQUIRED','RESOURCE_PRESSURE','WORKER_UNAVAILABLE','ISOLATED_TRAINER_REQUIRED',
                'FOUNDATION_BUSY','AUTOMATION_DISABLED','OUTSIDE_TRAINING_WINDOW'}
            return self._update(plan,'waiting' if exc.code in waiting else 'held',[exc.code])
