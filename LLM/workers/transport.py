import hashlib
import json
import secrets
import time
from pathlib import Path
from LLM.core import require,uid,digest,file_hash,inside,now

class Transport:
    def __init__(self,control):self.c=control;self.s=control.store

    def register(self,body):
        require(isinstance(body.get('capabilities',['train','merge','evaluate']),list) and set(body.get('capabilities',['train','merge','evaluate'])).issubset({'train','merge','evaluate','quantize','distill'}),'INVALID_WORKER_CAPABILITIES')
        token=secrets.token_urlsafe(40);identity=uid()
        worker=self.s.put('worker',{'name':str(body.get('name','worker'))[:100],'capabilities':body.get('capabilities',['train','merge','evaluate']),
            'tokenHash':hashlib.sha256(token.encode()).hexdigest(),'state':'registered','observedAt':now()},identity)
        return {'worker':{k:v for k,v in worker.items() if k!='tokenHash'},'token':token}

    def authorize(self,token):
        h=hashlib.sha256(token.encode()).hexdigest()
        return next((w for w in self.s.list('worker') if secrets.compare_digest(w.get('tokenHash',''),h) and w.get('state')!='revoked'),None)

    def assignment(self,worker_id):
        worker=self.s.get('worker',worker_id)
        self.s.put('worker',{'state':'online','observedAt':now()},worker_id)
        if self.s.get('setting','deploymentMaintenance',{}).get('enabled'):return {'job':None}
        job=self.s.claim('remote:'+worker_id,set(worker['capabilities']))
        if not job:return {'job':None}
        with self.s.tx():self.s.db.execute('UPDATE jobs SET lease=? WHERE id=?',(time.time()+600,job['id']))
        payload=job['payload'];branch=self.s.get('branch',payload['branchId']) if payload.get('branchId') else None
        ids=set()
        if branch:ids.update(x for x in (branch.get('anchorArtifactId'),branch.get('currentAdapterId')) if x)
        ids.update(payload.get('baselineIds',[]))
        if payload.get('artifactId'):ids.add(payload['artifactId'])
        if payload.get('combinationId'):
            combo=self.s.get('combination',payload['combinationId']);ids.add(combo['recipe']['anchorArtifactId']);ids.update(m['artifactId'] for m in combo['recipe']['members'])
        for _ in range(len(self.s.list('family'))+1):
            before=len(ids)
            for identity in list(ids):
                artifact=self.s.get('artifact',identity);family=self.s.get('family',artifact['familyId'])
                ids.add(family['anchorArtifactId']);ids.update(family.get('marketBaselineIds',[]))
            if len(ids)==before:break
        artifacts=[self.s.get('artifact',i) for i in ids]
        family_ids={a['familyId'] for a in artifacts};snapshots=[]
        if payload.get('snapshotId'):
            snap=self.s.get('snapshot',payload['snapshotId'])
            require(self.c.bridge is not None,'RAG_POLICY_GATEWAY_REQUIRED',status=503);self.c.bridge.validate_snapshot(snap)
            snapshots=[s for s in self.s.list('snapshot') if s['familyId']==snap['familyId'] and s['state']=='verified']
            for s in snapshots:self.c.bridge.validate_snapshot(s)
        fixtures=[];profiles=[]
        if payload.get('suiteId'):
            suite=self.s.get('suite',payload['suiteId'])
            fixtures=[self.s.get('fixture',t['fixtureId']) for t in suite['tasks'] if t.get('fixtureId')]
            profiles=[self.s.get('verifier_profile',f['profileId']) for f in fixtures]
        assignment={'job':job,'artifacts':artifacts,'snapshots':snapshots,'families':[self.s.get('family',i) for i in family_ids],
            'branches':[branch] if branch else [],'combinations':[combo] if payload.get('combinationId') else [],
            'suites':[self.s.get('suite',payload['suiteId'])] if payload.get('suiteId') else [],'fixtures':fixtures,'verifierProfiles':profiles,
            'consumed':self.s.rows('SELECT * FROM consumed WHERE branch=?',(branch['id'],)) if branch else [],
            'splits':self.s.rows('SELECT * FROM splits WHERE family=?',(branch['familyId'],)) if branch else [],
            'policyEpoch':self.s.get('setting','policyEpoch',{}),'workerId':worker_id}
        self.s.put('assignment',{'workerId':worker_id,'jobId':job['id'],'fence':job['fence'],'artifactIds':sorted(ids),
            'snapshotIds':[s['id'] for s in snapshots]},job['id'])
        return assignment

    def check(self,worker_id,job_id):
        job=self.s.job(job_id);require(job['owner']=='remote:'+worker_id,'WORKER_JOB_SCOPE_DENIED',status=403)
        self.s.assert_lease(job)
        return job

    def input_file(self,worker_id,job_id,kind,identity,name):
        self.check(worker_id,job_id);assignment=self.s.get('assignment',job_id)
        require(identity in assignment['artifactIds' if kind=='artifact' else 'snapshotIds'],'WORKER_ARTIFACT_SCOPE_DENIED',status=403)
        record=self.s.get(kind,identity)
        if kind=='artifact':
            require(name in record['files'],'ARTIFACT_FILE_NOT_FOUND',status=404);root=self.c.registry.verify(record)
        else:
            root=self.s.root/record['storagePath'];require(name in ('manifest.json','train.jsonl','validation.jsonl','test.jsonl','examples.json'),'SNAPSHOT_FILE_NOT_FOUND',status=404)
        return inside(root,name)

    def output_path(self,worker_id,job_id,artifact_id,name):
        self.check(worker_id,job_id)
        require(len(artifact_id)<128 and all(c.isalnum() or c in '-_' for c in artifact_id),'INVALID_ARTIFACT_ID')
        require(len(name)<200 and Path(name).suffix in ('.json','.safetensors','.gguf','.txt','.model','.md','.jinja') and '/' not in name and '\\' not in name,'OUTPUT_FILE_NOT_ALLOWED')
        root=self.s.root/'staging'/('remote-'+job_id)/artifact_id;root.mkdir(parents=True,exist_ok=True)
        return inside(root,name)

    def complete(self,worker_id,job_id,body):
        job=self.check(worker_id,job_id)
        require(body.get('fencingToken')==job['fence'],'LEASE_EXPIRED',status=409)
        if job['payload'].get('snapshotId'):
            require(self.c.bridge is not None,'RAG_POLICY_GATEWAY_REQUIRED',status=503)
            self.c.bridge.validate_snapshot(self.s.get('snapshot',job['payload']['snapshotId']))
        for identity in self.s.get('assignment',job_id)['artifactIds']:
            self.c.registry.verify(self.s.get('artifact',identity))
        if body.get('state') in ('failed','cancelled'):
            self.s.finish(job,body['state'],error={'code':'REMOTE_JOB_FAILED' if body['state']=='failed' else 'JOB_CANCELLED','message':str((body.get('error') or {}).get('code','Worker stopped'))})
            return self.s.job(job_id)
        require(body.get('state')=='completed','INVALID_WORKER_RESULT')
        for artifact in body.get('artifacts',[]):
            root=self.s.root/'staging'/('remote-'+job_id)/artifact['id']
            info=self.c.registry.inspect(root,artifact['kind'])
            require(info['manifestHash']==artifact['manifestHash'],'ARTIFACT_HASH_MISMATCH')
            family=self.s.get('family',artifact['familyId'])
            anchor=self.s.get('artifact',family['anchorArtifactId'])
            allowed=self.s.get('assignment',job_id)['artifactIds']
            require(artifact['familyId']==anchor['familyId'] and all(p in allowed or any(a['id']==p for a in body.get('artifacts',[])) for p in artifact.get('parents',[])),'WORKER_ARTIFACT_SCOPE_DENIED',status=403)
            metadata={k:artifact[k] for k in ('snapshotId','snapshotIds','trainingRunId','adapterArtifactId','recipeHash') if k in artifact}
            published=self.c.registry.publish(root,kind=artifact['kind'],family_id=family['id'],parents=artifact.get('parents'),branch_id=artifact.get('branchId'),license_refs=anchor['licenseRefs'],metadata=metadata)
            require(published['id']==artifact['id'],'ARTIFACT_HASH_MISMATCH')
        run=body.get('trainingRun')
        if run:
            require(job['type']=='train' and run['id']==job_id and run['branchId']==job['payload']['branchId'] and run['snapshotId']==job['payload']['snapshotId'],'WORKER_JOB_SCOPE_DENIED',status=403)
            require(isinstance(run.get('completedSteps'),int) and run['completedSteps']>0 and run['completedSteps']==run.get('totalSteps'),'INVALID_TRAINER_RESPONSE')
            self.s.put('training_run',run,job_id)
            self.c.snapshots.consume(run['branchId'],run['datasetRows'],job_id)
            self.s.put('branch',{'candidateArtifactId':run['outputArtifactId'],'candidateAdapterId':run['adapterArtifactId']},run['branchId'])
        if job['type']=='merge':self.s.put('combination',{'outputArtifactId':body['result']['artifactId'],'state':'built'},job['payload']['combinationId'])
        if job['type']=='evaluate':self.s.put('evaluation',body['result'],job_id)
        self.s.finish(job,'completed',body.get('result'))
        return self.s.job(job_id)
