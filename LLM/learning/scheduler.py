from datetime import datetime,timezone,timedelta
from zoneinfo import ZoneInfo
from LLM.core import require,digest,now

DEFAULT_POLICY={'enabled':False,'mode':'auto','timezone':'Asia/Seoul','checkIntervalMinutes':60,
    'trainingWindow':{'start':'02:00','end':'06:00'},'minNewTrainExamples':256,'minNewTargetTokens':65536,
    'cooldownHours':24,'maxFamilyTrainRunsPerDay':1,'maxSyntheticRatio':.3,'allowedSourceIds':[],
    'permittedTaskTypes':['source_grounded_completion','source_excerpt','review','test'],'autoTrain':True,'autoEvaluate':True,'autoPromote':True,'deploymentFormat':'gguf'}

class Scheduler:
    def __init__(self,control):self.control=control;self.store=control.store;self.last_check=None
    def policy(self):return self.store.get('policy','learning',{}) or {**DEFAULT_POLICY,'version':0,'id':'learning'}

    def configure(self,body,expected=None):
        old=self.policy();value={**old,**body}
        require(value.get('timezone')=='Asia/Seoul','UNSUPPORTED_TIMEZONE')
        for key in ('minNewTrainExamples','minNewTargetTokens','cooldownHours','maxFamilyTrainRunsPerDay','checkIntervalMinutes'):
            require(isinstance(value.get(key),int) and value[key]>=1,'INVALID_LEARNING_POLICY',field=key)
        for k in ('start','end'):
            datetime.strptime(value['trainingWindow'][k],'%H:%M')
        require(isinstance(value.get('allowedSourceIds'),list) and 0<=value.get('maxSyntheticRatio',.3)<=.3,'INVALID_LEARNING_POLICY')
        require(value.get('deploymentFormat') in ('gguf','safetensors'),'INVALID_DEPLOYMENT_FORMAT')
        for field in ('enabled','autoTrain','autoEvaluate','autoPromote'):
            require(isinstance(value.get(field),bool),'INVALID_LEARNING_POLICY',field=field)
        if value.get('enabled'):
            blockers=self.control.activation_blockers(value)
            require(not blockers,'LEARNING_ACTIVATION_BLOCKED',status=409,reasons=blockers)
        result=self.store.put('policy',value,'learning',expected=expected)
        if self.control.bridge:self.control.bridge.sync_learning_policy(result)
        return result

    def check(self,manual=False):
        policy=self.policy()
        if not policy.get('enabled') and not manual:return {'state':'paused','reasonCodes':['AUTOMATION_DISABLED']}
        if not policy.get('autoTrain',True):return {'state':'paused','reasonCodes':['AUTO_TRAIN_DISABLED']}
        if not manual and self.last_check and (datetime.now(timezone.utc)-self.last_check).total_seconds()<policy['checkIntervalMinutes']*60:return {'state':'waiting_window'}
        self.last_check=datetime.now(timezone.utc)
        if self.control.bridge and (policy.get('enabled') or manual):
            pending=self.store.rows("SELECT id FROM jobs WHERE type='curate' AND state IN ('queued','running') LIMIT 1")
            if not pending:self.store.enqueue('curate',{'policyVersion':policy['version']})
        blockers=self.control.activation_blockers()
        if blockers:return self.store.put('setting',{'state':'waiting_worker','reasonCodes':blockers},'learningStatus')
        clock=datetime.now(ZoneInfo('Asia/Seoul')).strftime('%H:%M');window=policy['trainingWindow']
        in_window=window['start']<=clock<window['end'] if window['start']<window['end'] else clock>=window['start'] or clock<window['end']
        if not manual and not in_window:return self.store.put('setting',{'state':'waiting_window','reasonCodes':['OUTSIDE_TRAINING_WINDOW']},'learningStatus')
        if any(c['state'] in ('queued','training','building','evaluating','packaging','deploying','release_evaluating') for c in self.store.list('cycle')):return {'state':'queued','reasonCodes':['CYCLE_ALREADY_ACTIVE']}
        failures=sorted([c for c in self.store.list('cycle') if c['state'] in ('held','failed','completed')],key=lambda c:c['createdAt'],reverse=True)
        if len(failures)>=3 and all(c['state']=='failed' for c in failures[:3]):
            self.configure({'enabled':False});return {'state':'paused','reasonCodes':['CONSECUTIVE_FAILURE_CIRCUIT_OPEN']}
        for branch in self.store.list('branch'):
            for snapshot in sorted(self.store.list('snapshot'),key=lambda s:s['createdAt'],reverse=True):
                if snapshot['familyId']!=branch['familyId'] or snapshot['state']!='verified':continue
                objective='sft'
                fresh=self.control.snapshots.new_examples(branch['id'],snapshot['id'],objective)
                if len(fresh)<policy['minNewTrainExamples']:
                    objective='cpt';fresh=self.control.snapshots.new_examples(branch['id'],snapshot['id'],objective)
                if len(fresh)<policy['minNewTrainExamples']:continue
                anchor=self.store.get('artifact',branch['anchorArtifactId'])
                from transformers import AutoTokenizer
                tokenizer=AutoTokenizer.from_pretrained(self.control.registry.verify(anchor),local_files_only=True,trust_remote_code=False)
                tokens=sum(len(tokenizer.encode(r['output'],add_special_tokens=False)) for r in fresh)
                if tokens<policy['minNewTargetTokens']:continue
                completed=[r for r in self.store.list('training_run') if r.get('state')=='completed' and self.store.get('branch',r['branchId'])['familyId']==branch['familyId']]
                today=datetime.now(ZoneInfo('Asia/Seoul')).date()
                day_runs=[r for r in self.store.list('training_run') if r.get('completedSteps',0)>0 and self.store.get('branch',r['branchId'])['familyId']==branch['familyId'] and datetime.fromisoformat(r['createdAt'].replace('Z','+00:00')).astimezone(ZoneInfo('Asia/Seoul')).date()==today]
                if len(day_runs)>=policy['maxFamilyTrainRunsPerDay']:continue
                recent=[r for r in completed if datetime.fromisoformat(r['finishedAt'].replace('Z','+00:00'))>datetime.now(timezone.utc)-timedelta(hours=policy['cooldownHours'])]
                if recent:continue
                key=digest([branch['id'],snapshot['manifestHash'],branch.get('currentAdapterId'),objective,policy['version']])
                cycle=self.store.get('cycle',key,{})
                if cycle:continue
                job=self.control.start_training({'branchId':branch['id'],'snapshotId':snapshot['id'],'model':branch['id'],'config':{'objective':objective}},key='cycle:'+key)
                cycle=self.store.put('cycle',{'state':'training','branchId':branch['id'],'snapshotId':snapshot['id'],'trainingJobId':job['id'],
                    'policyVersion':policy['version'],'newTrainExamples':len(fresh),'newTargetTokens':tokens,'reasonCodes':[]},key)
                return cycle
        return self.store.put('setting',{'state':'waiting_data','reasonCodes':['INSUFFICIENT_NEW_TRAIN_DATA']},'learningStatus')

    def advance(self):
        for cycle in self.store.list('cycle'):
            if cycle['state']=='training':
                job=self.store.job(cycle['trainingJobId'])
                if job['state']=='failed':self.store.put('cycle',{'state':'failed','error':job['error']},cycle['id']);continue
                if job['state']!='completed':continue
                output=job['result']['artifactId']
                if not self.policy().get('autoEvaluate',True):
                    self.store.put('cycle',{'state':'held','candidateArtifactId':output,'reasonCodes':['AUTO_EVALUATE_DISABLED']},cycle['id']);continue
                branch=self.store.get('branch',cycle['branchId'])
                suites=[s for s in self.store.list('suite') if s['kind']=='dev' and not s.get('fixture') and (not s.get('specialization') or s['specialization']==branch['specialization'])]
                if not suites:self.store.put('cycle',{'state':'held','candidateArtifactId':output,'reasonCodes':['DEV_SUITE_REQUIRED']},cycle['id']);continue
                branch=self.store.get('branch',cycle['branchId']);baseline=branch.get('currentArtifactId') or branch['anchorArtifactId']
                evaluation=self.store.enqueue('evaluate',{'artifactId':output,'suiteId':suites[0]['id'],'baselineIds':list(dict.fromkeys([baseline,branch['anchorArtifactId']]))},'cycle-eval:'+cycle['id'])
                self.store.put('cycle',{'state':'evaluating','candidateArtifactId':output,'evaluationJobId':evaluation['id']},cycle['id'])
            elif cycle['state']=='evaluating':
                job=self.store.job(cycle['evaluationJobId'])
                if job['state'] not in ('completed','failed'):continue
                if job['state']=='failed':self.store.put('cycle',{'state':'held','reasonCodes':['EVALUATION_FAILED']},cycle['id']);continue
                result=job['result']
                if not result['gate']['passed']:
                    self.store.put('cycle',{'state':'held','reasonCodes':result['gate']['reasonCodes'],'evaluationId':result['id']},cycle['id']);continue
                branch=self.store.get('branch',cycle['branchId']);run=self.store.get('training_run',cycle['trainingJobId'])
                self.store.put('branch',{'currentArtifactId':cycle['candidateArtifactId'],'currentAdapterId':run['adapterArtifactId']},branch['id'])
                selection=self.store.get('setting','selectedCombination',{}).get('combinationId')
                if selection:
                    combo=self.store.get('combination',selection);ids=[]
                    for member in combo['recipe']['members']:
                        artifact=self.store.get('artifact',member['artifactId']);b=self.store.get('branch',artifact['branchId'])
                        ids.append(b.get('currentArtifactId') or artifact['id'])
                    combo=self.control.merger.recipe({'artifactIds':ids,'weights':[m['coefficient'] for m in combo['recipe']['members']],
                        **{k:combo['recipe'][k] for k in ('method','deltaScale','density','seed','outputDtype')}})
                    job=self.store.enqueue('merge',{'combinationId':combo['id']},'cycle-build:'+cycle['id'])
                    self.store.put('cycle',{'state':'building','buildJobId':job['id'],'combinationId':combo['id']},cycle['id'])
                else:self._package(cycle,cycle['candidateArtifactId'])
            elif cycle['state']=='building':
                job=self.store.job(cycle['buildJobId'])
                if job['state']=='completed':self._package(cycle,job['result']['artifactId'])
                elif job['state']=='failed':self._held(cycle,'MERGE_FAILED')
            elif cycle['state']=='packaging':
                job=self.store.job(cycle['packageJobId'])
                if job['state']=='completed':self._release_evaluate(cycle,job['result']['id'])
                elif job['state']=='failed':self._held(cycle,job['error']['code'])
            elif cycle['state']=='release_evaluating':
                jobs=[self.store.job(i) for i in cycle['releaseEvaluationJobIds']]
                if any(j['state']=='failed' for j in jobs):self._held(cycle,'RELEASE_EVALUATION_FAILED');continue
                if not all(j['state']=='completed' for j in jobs):continue
                certificate=self.control.evaluator.certify({'artifactId':cycle['deploymentArtifactId'],'evaluationIds':[j['result']['id'] for j in jobs]})
                if not certificate['passed']:self._held(cycle,*certificate['reasonCodes']);continue
                release=self.control.releases.prepare({'artifactId':cycle['deploymentArtifactId'],'certificateId':certificate['id']})
                if self.policy().get('autoPromote'):
                    try:
                        slot=self.control.releases.activate(release['id'])
                        self.store.put('cycle',{'state':'completed','releaseId':release['id'],'activeReleaseId':slot['activeReleaseId']},cycle['id'])
                    except Exception as exc:self._held(cycle,getattr(exc,'code','DEPLOYMENT_FAILED'))
                else:self.store.put('cycle',{'state':'held','releaseId':release['id'],'reasonCodes':['MANUAL_PROMOTION_POLICY']},cycle['id'])

    def _held(self,cycle,*reasons):
        self.store.put('cycle',{'state':'held','reasonCodes':list(reasons)},cycle['id'])

    def _package(self,cycle,artifact_id):
        if self.policy().get('deploymentFormat','gguf')=='gguf':
            if not self.control.capabilities()['quantization']:self._held(cycle,'QUANTIZATION_TOOLS_UNAVAILABLE');return
            job=self.store.enqueue('quantize',{'artifactId':artifact_id},'cycle-package:'+cycle['id'])
            self.store.put('cycle',{'state':'packaging','packageJobId':job['id']},cycle['id'])
        else:self._release_evaluate(cycle,artifact_id)

    def _release_evaluate(self,cycle,artifact_id):
        suites={s['kind']:s for s in self.store.list('suite') if not s.get('fixture')}
        if not all(k in suites for k in ('release','regression','evidence','unanswerable')):self._held(cycle,'RELEASE_SUITES_REQUIRED');return
        branch=self.store.get('branch',cycle['branchId']);baselines=[branch['anchorArtifactId']]
        slot=self.store.get('slot','champion',{})
        if slot:baselines.append(self.store.get('release',slot['activeReleaseId'])['artifactId'])
        if cycle.get('combinationId'):baselines.extend(m['artifactId'] for m in self.store.get('combination',cycle['combinationId'])['recipe']['members'])
        jobs=[self.store.enqueue('evaluate',{'artifactId':artifact_id,'suiteId':suites[k]['id'],'baselineIds':list(dict.fromkeys(baselines))},'cycle-release:'+cycle['id']+':'+k) for k in ('release','regression','evidence','unanswerable')]
        self.store.put('cycle',{'state':'release_evaluating','deploymentArtifactId':artifact_id,'releaseEvaluationJobIds':[j['id'] for j in jobs]},cycle['id'])
