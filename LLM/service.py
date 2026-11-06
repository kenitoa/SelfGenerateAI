from __future__ import annotations
import importlib.util
import json
import os
import shutil
import threading
import time
from pathlib import Path
import psutil
from LLM.core import Store,Fault,require,digest,now,uid
from LLM.registry import Registry
from LLM.data import Snapshots
from LLM.merging import Merger
from LLM.training import Trainer
from LLM.inference import Inference
from LLM.verification import Sandbox
from LLM.evaluation import Evaluator
from LLM.release import Releases
from LLM.learning import Scheduler

class Control:
    def __init__(self,root,allowed_roots=(),worker=False):
        self.store=Store(root);self.registry=Registry(self.store,allowed_roots)
        self.snapshots=Snapshots(self.store);self.merger=Merger(self.store,self.registry)
        self.trainer=Trainer(self.store,self.registry,self.snapshots);self.inference=Inference(self.store,self.registry)
        self.sandbox=Sandbox(self.store);self.evaluator=Evaluator(self.store,self.registry,self.inference,self.sandbox)
        self.releases=Releases(self.store,self.registry);self.scheduler=Scheduler(self)
        self.releases.inference=self.inference
        self.trainer.should_yield=lambda:self.inference.foreground_waiters>0
        from LLM.merging.advanced import RecipeSearch
        self.recipe_search=RecipeSearch(self)
        from LLM.evolution import SelfEvolution
        self.evolution=SelfEvolution(self)
        self.bridge=None;self.worker_enabled=worker;self.stop_event=threading.Event();self.threads=[];self.active=None;self.owner='local-'+uid()

    def activation_blockers(self,policy=None):
        blockers=[]
        if not self.has_worker('train'):blockers.append('WORKER_UNAVAILABLE')
        if not all(importlib.util.find_spec(m) for m in ('torch','peft','transformers','safetensors')):blockers.append('TRAINING_BACKEND_UNAVAILABLE')
        if not self.store.list('family'):blockers.append('MODEL_NOT_REGISTERED')
        if not self.store.list('branch'):blockers.append('BRANCH_REQUIRED')
        if self.bridge is None:blockers.append('RAG_POLICY_GATEWAY_REQUIRED')
        if not (policy or self.scheduler.policy()).get('allowedSourceIds'):blockers.append('TRAINING_SOURCES_NOT_APPROVED')
        elif self.bridge:
            sources={s['id']:s for s in self.bridge.store.list('source')}
            if any(identity not in sources or sources[identity].get('trainingPolicy')!='allowed' for identity in (policy or self.scheduler.policy())['allowedSourceIds']):
                blockers.append('SOURCE_TRAINING_NOT_ALLOWED')
        return blockers

    def capabilities(self):
        from LLM.workers.resources import observe,budget
        backend=all(importlib.util.find_spec(m) for m in ('torch','peft','transformers','safetensors'))
        return {'registry':True,'generation':backend,'training':self.has_worker('train'),'merging':self.has_worker('merge'),
            'distillation':backend and self.sandbox.available(),'verification':self.sandbox.available(),
            'quantization':bool(self.store.get('setting','tools',{}).get('quantizer')),'scheduling':True,
            'selfModification':{'enabled':self.evolution.policy()['enabled'],'verifierAvailable':self.sandbox.available(),'target':'LLM/extensions/document_policy.py'},
            'activationBlockers':self.activation_blockers(),'observedAt':now(),
            'resources':budget(self.store)}

    def start_training(self,body,key=None):
        require(self.has_worker('train'),'WORKER_UNAVAILABLE',status=503)
        branch_id=body.get('branchId')
        if not branch_id:
            branches=[b for b in self.store.list('branch') if body.get('model') in (b['id'],b['label'],b.get('anchorArtifactId'))]
            require(len(branches)==1,'TRAINING_TARGET_REQUIRED',status=409)
            branch_id=branches[0]['id']
        branch=self.store.get('branch',branch_id)
        snapshot=self.store.get('snapshot',body.get('snapshotId') or self.store.get('setting','selectedSnapshot',{}).get('snapshotId',''))
        require(snapshot['familyId']==branch['familyId'] and snapshot['state']=='verified','SNAPSHOT_NOT_VERIFIED_BY_TRAINER',status=409)
        if self.bridge:self.bridge.validate_snapshot(snapshot)
        cfg=self.trainer.config(body.get('config',body))
        with self.store.tx():
            active=self.store.rows("SELECT id FROM jobs WHERE type='train' AND state IN ('queued','running','paused','pause_requested')")
            if key:
                old=self.store.rows('SELECT id FROM jobs WHERE request_key=?',(key,))
                if old:return self.store.job(old[0]['id'])
            require(not active,'TRAINER_BUSY',status=409)
            return self.store.enqueue('train',{'branchId':branch_id,'snapshotId':snapshot['id'],'config':cfg},key)

    def training_status(self):
        selected=self.store.get('setting','selectedSnapshot',{}).get('snapshotId')
        snapshot=self.store.get('snapshot',selected,{}) if selected else {}
        if snapshot and self.bridge:self.bridge.validate_snapshot(snapshot)
        jobs=self.store.rows("SELECT id FROM jobs WHERE type='train' ORDER BY created DESC")
        runs=[];active=None
        for row in jobs:
            job=self.store.job(row['id']);run=self.store.get('training_run',row['id'],{})
            state=job['state']
            if state in ('completed','failed','cancelled'):
                runs.append({'id':job['id'],'state':state,'finishedAt':run.get('finishedAt',job['updatedAt'])})
            elif active is None:
                mapped={'queued':'preparing','running':run.get('state','preparing'),'pause_requested':'running','paused':'paused','cancelling':'running'}.get(state,'preparing')
                active={'id':job['id'],'state':mapped,'completedSteps':run.get('completedSteps',0),'totalSteps':run.get('totalSteps',0),
                    'epoch':run.get('epoch'),'loss':run.get('loss'),'updatedAt':run.get('updatedAt',job['updatedAt']),
                    'capabilities':{'pause':state=='running' and bool(run.get('completedSteps')) and not (job.get('owner') or '').startswith('remote:'),'resume':state=='paused','cancel':True}}
        counts=snapshot.get('splitCounts',{})
        return {'job':active,'runs':runs,'readySamples':snapshot.get('exampleCount',0) if snapshot.get('state')=='verified' else 0,
            'canStart':bool(snapshot and snapshot['state']=='verified' and not active and self.has_worker('train')),
            'snapshotId':selected,'snapshotManifestHash':snapshot.get('manifestHash'),'verifiedSnapshotIds':[s['id'] for s in self.store.list('snapshot') if s['state']=='verified'],
            'trainSamples':counts.get('train',0),'validationSamples':counts.get('validation',0),'testSamples':counts.get('test',0),
            'reasonCodes':[] if snapshot else ['SNAPSHOT_REQUIRED']}

    def has_worker(self,kind):
        if self.worker_enabled and all(importlib.util.find_spec(m) for m in ('torch','peft','transformers','safetensors')):return True
        from datetime import datetime,timezone
        return any(kind in w.get('capabilities',[]) and w.get('state')=='online' and
            (datetime.now(timezone.utc)-datetime.fromisoformat(w['observedAt'].replace('Z','+00:00'))).total_seconds()<60
            for w in self.store.list('worker'))

    def request_certification(self,body,key=None):
        artifact=self.store.get('artifact',body['artifactId']);require(not artifact.get('fixture'),'FIXTURE_NOT_FOR_PRODUCTION',status=409)
        anchor=self.store.get('family',artifact['familyId'])['anchorArtifactId']
        if artifact['id']==anchor and not self.store.get('slot','champion',{}):
            job=self.store.enqueue('bootstrap',{'artifactId':artifact['id']},key)
            return self.store.put('certification_request',{'artifactId':artifact['id'],'jobIds':[job['id']],'state':'bootstrap','bootstrap':True})
        suites={s['kind']:s for s in self.store.list('suite') if not s.get('fixture')}
        require(all(k in suites for k in ('release','regression','evidence','unanswerable')),'RELEASE_SUITES_REQUIRED',status=409)
        anchor=self.store.get('family',artifact['familyId'])['anchorArtifactId']
        bootstrap=artifact['id']==anchor
        baselines=[anchor] if not bootstrap else []
        baselines.extend(p for p in artifact.get('parents',[]) if self.store.get('artifact',p).get('kind')=='specialist_checkpoint')
        slot=self.store.get('slot','champion',{})
        if slot:baselines.append(self.store.get('release',slot['activeReleaseId'])['artifactId'])
        jobs=[self.store.enqueue('evaluate',{'artifactId':artifact['id'],'suiteId':suites[k]['id'],'baselineIds':list(dict.fromkeys(baselines)),
            'bootstrap':bootstrap},key=(key or uid())+':'+k) for k in ('release','regression','evidence','unanswerable')]
        return self.store.put('certification_request',{'artifactId':artifact['id'],'jobIds':[j['id'] for j in jobs],'state':'evaluating','bootstrap':bootstrap})

    def advance_certifications(self):
        for request in self.store.list('certification_request'):
            if request['state']=='bootstrap':
                job=self.store.job(request['jobIds'][0])
                if job['state']=='failed':self.store.put('certification_request',{'state':'held','reasonCodes':[job['error']['code']]},request['id'])
                elif job['state']=='completed':
                    certificate=job['result']
                    if certificate['passed']:
                        release=self.releases.prepare({'artifactId':request['artifactId'],'certificateId':certificate['id']})
                        self.store.put('certification_request',{'state':'ready','certificateId':certificate['id'],'releaseId':release['id']},request['id'])
                    else:self.store.put('certification_request',{'state':'held','reasonCodes':certificate['reasonCodes']},request['id'])
                continue
            if request['state']!='evaluating':continue
            jobs=[self.store.job(i) for i in request['jobIds']]
            if any(j['state']=='failed' for j in jobs):self.store.put('certification_request',{'state':'held','reasonCodes':['EVALUATION_FAILED']},request['id']);continue
            if not all(j['state']=='completed' for j in jobs):continue
            certificate=self.evaluator.certify({'artifactId':request['artifactId'],'evaluationIds':[j['result']['id'] for j in jobs],'bootstrap':request['bootstrap']})
            if certificate['passed']:
                release=self.releases.prepare({'artifactId':request['artifactId'],'certificateId':certificate['id']})
                self.store.put('certification_request',{'state':'ready','certificateId':certificate['id'],'releaseId':release['id']},request['id'])
            else:self.store.put('certification_request',{'state':'held','reasonCodes':certificate['reasonCodes']},request['id'])

    def advance_distillations(self):
        if not self.bridge:return
        for item in self.store.list('distillation'):
            if item['state']=='waiting_export':
                snapshot=self.bridge.post.snapshot(item['snapshotId'])
                if snapshot['status']=='revoked':self.store.put('distillation',{'state':'held','reasonCodes':['POLICY_REVOKED']},item['id']);continue
                if snapshot['status']!='ready':continue
                branch=self.store.get('branch',item['branchId']);self.bridge.import_snapshot(snapshot['id'],branch['familyId'])
                job=self.start_training({'branchId':branch['id'],'snapshotId':snapshot['id']},'distill-train:'+item['id'])
                payload={**job['payload'],'distillationId':item['id']}
                from LLM.core import canonical
                with self.store.tx():self.store.db.execute('UPDATE jobs SET payload=? WHERE id=?',(canonical(payload),job['id']))
                self.store.put('distillation',{'state':'training','trainingJobId':job['id']},item['id'])
            elif item['state']=='training':
                job=self.store.job(item['trainingJobId'])
                if job['state']=='completed':self.store.put('distillation',{'state':'built','outputArtifactId':job['result']['artifactId']},item['id'])
                elif job['state']=='failed':self.store.put('distillation',{'state':'held','error':job['error']},item['id'])

    def command(self,identity,action):
        with self.store.tx():
            job=self.store.job(identity)
            if action=='pause':
                require(job['type']=='train' and job['state']=='running','PAUSE_NOT_AVAILABLE',status=409)
                require(not (job.get('owner') or '').startswith('remote:'),'REMOTE_PAUSE_UNAVAILABLE',status=409)
                self.store.db.execute("UPDATE jobs SET state='pause_requested' WHERE id=?",(identity,))
            elif action=='resume':
                require(job['state']=='paused','RESUME_NOT_AVAILABLE',status=409)
                run=self.store.get('training_run',identity);require(run.get('checkpointId'),'CHECKPOINT_REQUIRED')
                payload={**job['payload'],'checkpointId':run['checkpointId']}
                from LLM.core import canonical
                self.store.db.execute("UPDATE jobs SET state='queued',payload=?,lease=0 WHERE id=?",(canonical(payload),identity))
            elif action=='cancel':
                require(job['state'] not in ('completed','failed'),'JOB_ALREADY_FINISHED',status=409)
                state='cancelling' if job['state'] in ('running','pause_requested') else 'cancelled'
                self.store.db.execute('UPDATE jobs SET state=? WHERE id=?',(state,identity))
            else:raise Fault('INVALID_JOB_ACTION')
        return self.store.job(identity)

    def start(self):
        if self.threads:return
        self.stop_event.clear()
        try:self.store.put('setting',self.evolution.reconcile(),'selfCodeHealth')
        except Fault as exc:self.store.put('setting',{'state':'held','reasonCodes':[exc.code]},'selfCodeHealth')
        try:self.store.put('setting',self.releases.reconcile(),'deploymentHealth')
        except Fault as exc:self.store.put('setting',{'state':'needs_review','reasonCodes':[exc.code]},'deploymentHealth')
        if self.worker_enabled:
            self.store.put('worker',{'capabilities':['train','merge','evaluate','quantize'],'state':'online','observedAt':now()},self.owner)
            t=threading.Thread(target=self._work,daemon=True,name='seed-llm-worker');t.start();self.threads.append(t)
        t=threading.Thread(target=self._supervise,daemon=True,name='seed-llm-supervisor');t.start();self.threads.append(t)

    def stop(self):
        self.stop_event.set()
        for t in self.threads:t.join(timeout=5)

    def step(self):
        if self.inference.foreground_waiters or not self.inference.lock.acquire(blocking=False):return False
        try:return self._step_locked()
        finally:self.inference.lock.release()

    def _step_locked(self):
        if self.store.get('setting','deploymentMaintenance',{}).get('enabled'):return False
        pending=self.store.rows("SELECT id FROM jobs WHERE state='queued' OR (state='running' AND lease<?) LIMIT 1",(time.time(),))
        if not pending:return False
        if not self.inference.expire(force=True):return False
        job=self.store.claim(self.owner,{'train','merge','evaluate','quantize','distill','bootstrap','compress','curate','evolve'})
        if not job:return False
        self.active=job
        started=time.monotonic()
        deadline=self.store.get('setting','resourcePolicy',{}).get('backgroundDeadlineSeconds',7200)
        heartbeat_stop=threading.Event()
        def keep_lease():
            while not heartbeat_stop.wait(10):
                try:
                    self.store.heartbeat(job)
                    from LLM.workers.resources import budget
                    resources=budget(self.store)
                    memory_exceeded=resources['processRssBytes']>resources['maxRssBytes']
                    if time.monotonic()-started>deadline or memory_exceeded:
                        with self.store.tx():self.store.db.execute("UPDATE jobs SET state='cancelling' WHERE id=? AND state='running'",(job['id'],))
                        self.store.event('job.budget_exceeded',job['id'],{'deadlineSeconds':deadline,'memoryExceeded':memory_exceeded})
                except Fault:return
        lease_thread=threading.Thread(target=keep_lease,daemon=True);lease_thread.start()
        try:
            from LLM.workers.resources import budget
            resources=budget(self.store);available=min(resources['availableRamBytes'],max(0,resources['maxRssBytes']-resources['processRssBytes']))
            require(available>=768*1024**2,'RESOURCE_PRESSURE',status=503)
            require(resources['freeDiskBytes']>=2*1024**3,'DISK_LOW',status=503)
            if job['type']=='train':
                branch=self.store.get('branch',job['payload']['branchId']);anchor=self.store.get('artifact',branch['anchorArtifactId'])
                estimated=anchor['parameterCount']*4+512*1024**2
                require(available-max(512*1024**2,int(available*.2))>=estimated,'RESOURCE_PRESSURE',status=503)
            if self.bridge:self.bridge.refresh_policy()
            if job['type']=='train':
                if self.bridge:self.bridge.validate_snapshot(self.store.get('snapshot',job['payload']['snapshotId']))
                result=self.trainer.run(job)
            elif job['type']=='merge':result=self.merger.build(job)
            elif job['type']=='evaluate':result=self.evaluator.evaluate(job)
            elif job['type']=='bootstrap':result=self.evaluator.bootstrap(job)
            elif job['type']=='quantize':result=self.releases.package(job)
            elif job['type']=='compress':
                from LLM.merging.advanced import compress_adapter
                result=compress_adapter(self,job)
            elif job['type']=='curate':
                require(self.bridge is not None,'RAG_POLICY_GATEWAY_REQUIRED')
                result=self.bridge.produce_candidates(self.scheduler.policy())
            elif job['type']=='evolve':result=self.evolution.run(job)
            else:
                from LLM.distillation import Distiller
                result=Distiller(self).run(job)
            state='paused' if result.get('state') in ('paused','yielded') else 'completed'
            self.store.finish(job,state,result)
            if result.get('state')=='yielded':self.command(job['id'],'resume')
        except BaseException as exc:
            if not isinstance(exc,Exception):raise
            code=getattr(exc,'code','JOB_FAILED')
            error={'code':code,'message':str(exc)[:500]}
            try:
                self.store.finish(job,'cancelled' if code=='JOB_CANCELLED' else 'failed',error=error)
                if job['type']=='train':self.store.put('training_run',{'state':'cancelled' if code=='JOB_CANCELLED' else 'failed','finishedAt':now(),'error':error},job['id'])
            except Fault:pass
        finally:
            heartbeat_stop.set();lease_thread.join(timeout=1);self.active=None
        return True

    def _work(self):
        while not self.stop_event.is_set():
            try:
                if not self.step():self.stop_event.wait(.5)
            except Exception:self.stop_event.wait(2)

    def _supervise(self):
        last_gc=None
        while not self.stop_event.wait(5):
            try:
                if self.active:self.store.heartbeat(self.active)
                self.inference.expire()
                self.scheduler.advance();self.scheduler.check();self.advance_certifications();self.advance_distillations()
                self.recipe_search.advance()
                self.evolution.check()
                self.evolution.advance_feedback()
                if self.bridge:
                    from LLM.data.curator import Curator
                    Curator(self.bridge).import_exports()
                if last_gc is None or time.monotonic()-last_gc>=86400:
                    from LLM.storage.retention import collect
                    retention=self.store.get('setting','retentionPolicy',{})
                    result=collect(self.store,apply=retention.get('autoGc',True))
                    if result['state']!='held':last_gc=time.monotonic()
                if self.bridge:self.bridge.refresh_policy()
                if self.bridge and any(d.get('state')=='preparing' for d in self.store.list('deployment')):
                    self.store.put('setting',self.releases.reconcile(),'deploymentHealth')
            except Exception as exc:
                self.store.event('supervisor.error','control',{'code':getattr(exc,'code','SUPERVISOR_FAILED')})
