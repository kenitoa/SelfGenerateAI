"""Durable from-scratch controller. User data never becomes executable code."""
from __future__ import annotations
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from LLM.core import Fault, require, digest, canonical, now, uid, atomic_json, file_hash, inside
from .schemas import PRESET, architecture_config, parameter_count, integer, real, record_path
from .resources import DEFAULT_POLICY, admit, foundation_budget, isolated_network, charge_time, used_today

JOB_TYPES = frozenset({'foundation_corpus','foundation_tokenizer','foundation_initialize',
                      'foundation_calibrate','foundation_pretrain','foundation_cpt','foundation_sft','foundation_evaluate','foundation_self_code_evaluate',
                      'foundation_distill','foundation_response_distill','foundation_growth','foundation_growth_adopt','foundation_relearn','foundation_format'})
TRAIN_TYPES = frozenset({'foundation_pretrain','foundation_cpt','foundation_sft','foundation_distill'})


class FoundationController:
    def __init__(self, control):
        self.control = control; self.store = control.store
        self.store.db.execute('''CREATE TABLE IF NOT EXISTS foundation_active_time(
            run TEXT NOT NULL,day TEXT NOT NULL,seconds REAL NOT NULL,PRIMARY KEY(run,day));
            CREATE TABLE IF NOT EXISTS foundation_evaluation_usage(
            suite TEXT NOT NULL,shard TEXT NOT NULL,candidate TEXT NOT NULL,at TEXT NOT NULL,PRIMARY KEY(suite,shard));''')
        from .trainer import FoundationTrainer
        from .runtime import FoundationRuntime
        from .evaluation import FoundationEvaluator
        self.trainer = FoundationTrainer(self.store)
        self.runtime = FoundationRuntime(self.store, artifacts=control.registry)
        self.runtime.policy_validator = lambda project:self.validate_policy(self._project(project['id']))
        self.evaluator = FoundationEvaluator(control)

    def _project(self, identity):
        value = self.store.get('foundation_project', identity)
        value['policy'] = self.store.get('foundation_policy', identity, {**DEFAULT_POLICY,'version':0,'id':identity})
        require(value.get('state') != 'revoked' and not value.get('revocationPending'), 'POLICY_REVOKED', status=409)
        from .origins import assert_project_available
        assert_project_available(self.store,value)
        return value

    def project(self, identity):return self._project(identity)

    def status(self, identity):
        p = self.store.get('foundation_project', identity)
        p['policy'] = self.store.get('foundation_policy', identity, {**DEFAULT_POLICY,'version':0,'id':identity})
        for field, kind in [('runs','foundation_run'),('corpora','foundation_corpus'),('tokenizers','foundation_tokenizer'),
                            ('initializations','foundation_genesis'),('evaluations','foundation_evaluation'),('profiles','foundation_profile')]:
            p[field] = [r for r in self.store.list(kind) if r.get('projectId') == identity]
        p['revision'] = p['version']; p['resources'] = foundation_budget(self.store, test_only=p.get('testOnly') is True)
        p['trainedContextTokens']=0 if not p.get('latestCheckpointId') else None
        latest_checkpoint=self.store.get('foundation_checkpoint',p.get('latestCheckpointId') or '',{})
        if latest_checkpoint:
            # Read only authenticated scalar training state for the status view;
            # model loading/serving performs the full tensor verification itself.
            try:
                root=record_path(self.store,latest_checkpoint)
                manifest_path=root/'manifest.json';state_path=root/'state.json'
                require(latest_checkpoint.get('state') in ('ready','verified','completed')
                        and latest_checkpoint.get('projectId')==identity and manifest_path.is_file()
                        and state_path.is_file() and not manifest_path.is_symlink() and not state_path.is_symlink(),
                        'CHECKPOINT_NOT_READY')
                require(file_hash(manifest_path)==latest_checkpoint.get('manifestHash'),'CHECKPOINT_HASH_MISMATCH')
                manifest=json.loads(manifest_path.read_text('utf-8'))
                require(manifest.get('id')==latest_checkpoint['id']
                        and manifest.get('tokenizerHash')==latest_checkpoint.get('tokenizerHash')
                        and file_hash(state_path)==manifest.get('files',{}).get('state.json'),'CHECKPOINT_HASH_MISMATCH')
                from .checkpoint import _unpack
                tree=json.loads(state_path.read_text('utf-8'))
                state=next(value for key,value in tree['items'] if key=={'type':'scalar','value':'state'})
                state=_unpack(state,{})
                context=state.get('trainedContextTokens')
                require(state.get('projectId')==identity and state.get('completedOptimizerSteps',0)>0
                        and isinstance(context,int) and not isinstance(context,bool)
                        and 0<context<=p['config']['max_position_embeddings']
                        and context==latest_checkpoint.get('trainedContextTokens'),'CHECKPOINT_STATE_MISMATCH')
                p['trainedContextTokens']=context
                p['trainedContextCheckpointId']=latest_checkpoint['id']
            except (Fault,ValueError,KeyError,StopIteration,TypeError):
                p['trainedContextReasonCode']='CHECKPOINT_STATE_UNVERIFIED'
        reasons = []
        if not p['policy'].get('allowedSourceIds'):reasons.append('CORPUS_PERMISSION_REQUIRED')
        if not p.get('activeCorpusId'):reasons.append('INSUFFICIENT_CORPUS')
        if not self.has_worker():reasons.append('ISOLATED_TRAINER_REQUIRED')
        if p.get('state') == 'revoked':reasons.append('POLICY_REVOKED')
        p['reasonCodes'] = list(dict.fromkeys([*p.get('reasonCodes',[]),*reasons]))
        from .retention import plan
        cached=getattr(self,'_storage_plans',{})
        saved=cached.get(identity,{})
        if time.monotonic()-saved.get('at',-1e20)>30:
            cached[identity]={'at':time.monotonic(),'value':plan(self.store,p)}
            self._storage_plans=cached
        p['storage']=cached[identity]['value']
        p['capabilityFlags'] = {'canPrepareCorpus': bool(p['policy'].get('allowedSourceIds')),
            'canInitialize':bool(p.get('tokenizerId')), 'canTrain':bool(p.get('genesisId') or p.get('latestCheckpointId')) and self.has_worker(),
            'canServe':p.get('state')!='revoked' and not p.get('revocationPending') and any(r.get('projectId')==identity and
                self.store.get('release',r['id'],r).get('state')=='active' for r in self.store.list('foundation_release')),
            'selfCodeEligible':any(r.get('projectId')==identity and r.get('state')=='eligible' and r.get('passed') is True
                for r in self.store.list('foundation_self_code_eligibility'))}
        p['approvedScopes']=[{key:row.get(key) for key in ('artifactId','purpose','language','scope','baseCertificateId')}
            for row in self.store.list('foundation_domain_certificate') if row.get('projectId')==identity and row.get('state')=='active' and row.get('passed') is True]
        return p

    def has_worker(self):
        if self.control.execution_scope=='pc_validation' and self.control.worker_enabled:return True
        if self.control.worker_enabled and isolated_network():return True
        current = datetime.now(timezone.utc)
        for record in self.store.list('worker'):
            try:recent = (current-datetime.fromisoformat(record['observedAt'].replace('Z','+00:00'))).total_seconds()<60
            except (KeyError,ValueError):continue
            if recent and record.get('executionScope','nas')==self.control.execution_scope and record.get('state')=='online' and record.get('foundationSharedVolume') is True and 'foundation_pretrain' in record.get('capabilities',[]):return True
        return False

    def create_project(self, body, *, trusted_test=False,trusted_growth_receipt=None):
        require(isinstance(body,dict),'CONFIG_INVALID')
        allowed={'name','origin','pretrainedWeightArtifactIds','architecture','architecturePreset','automation','policyId','seed'}
        require(body.get('origin','random_init')=='random_init' and not body.get('pretrainedWeightArtifactIds'), 'EXTERNAL_WEIGHTS_FORBIDDEN')
        forbidden={'testOnly','fixture','servingEligible','capabilityFlags','stage','passed','certificateId','genesisId','tokenizerId'}
        require(not (forbidden & set(body)) or trusted_test, 'SERVER_CONTROLLED_FIELD', status=403)
        require(trusted_test or not (set(body)-allowed),'CONFIG_INVALID',field='project')
        require(body.get('policyId','foundation-local-v1')=='foundation-local-v1','CONFIG_INVALID',field='policyId')
        config=architecture_config(body.get('architecture',body.get('architecturePreset',PRESET)),test_only=trusted_test,
            growth_receipt=trusted_growth_receipt)
        name=body.get('name','SEED-S0');require(isinstance(name,str) and 1<=len(name.strip())<=80,'CONFIG_INVALID',field='name')
        mode=body.get('automation','always_on_idle');mode='night_window' if mode=='night' else mode
        require(mode in ('always_on_idle','night_window','manual'),'CONFIG_INVALID',field='automation')
        policy={**DEFAULT_POLICY,'automation':mode,'enabled':False}
        seed=integer(body.get('seed',42),'seed',0,2**32-1)
        if trusted_test:policy.update({'minCorpusBytes':1,'minTrainGroups':1,'dailyActiveSeconds':14400})
        with self.store.tx():
            p=self.store.put('foundation_project',{'name':name.strip(),'origin':'random_init','pretrainedWeightArtifactIds':[],
                'architecturePreset':PRESET,'config':config,'configHash':digest(config),'architectureHashV2':digest(config),
                'parameterCount':parameter_count(config),'seed':seed,'stage':'waiting_corpus','state':'active','testOnly':trusted_test,
                'executionScope':self.control.execution_scope,
                'trustedGrowthReceipt':trusted_growth_receipt,
                'acceptedArchitectureRecipeId':(trusted_growth_receipt or {}).get('candidateId'),
                'activeCorpusId':None,'tokenizerId':None,'genesisId':None,'latestCheckpointId':None,'reasonCodes':[]})
            self.store.put('foundation_policy',policy,p['id'])
        self.store.event('foundation.project.created',p['id'],{'origin':'random_init'})
        return self.status(p['id'])

    def configure(self, identity, body, expected=None):
        project=self._project(identity); old=project['policy']
        allowed=set(DEFAULT_POLICY)
        require(not (set(body)-allowed-{'expectedVersion'}),'CONFIG_INVALID',field='policy')
        value={**old,**{k:v for k,v in body.items() if k in allowed}}
        value['automation']='night_window' if value['automation']=='night' else value['automation']
        require(value['automation'] in ('always_on_idle','night_window','manual'),'CONFIG_INVALID')
        require(isinstance(value['enabled'],bool),'CONFIG_INVALID',field='enabled')
        require(isinstance(value['allowedSourceIds'],list) and all(isinstance(s,str) and s for s in value['allowedSourceIds']),'CONFIG_INVALID')
        for k,low,high in [('dailyActiveSeconds',1,86400),('sliceSeconds',10,900),('restSeconds',0,86400),('minNewTargetTokens',1,2**40),
                           ('minCorpusBytes',1,2**40),('minTrainGroups',1,100000),('maxConsecutiveFailures',1,10),('checkIntervalSeconds',10,86400),('quotaBytes',1024**2,2**50)]:
            integer(value[k],k,low,high)
        if not project.get('testOnly'):
            require(value['minCorpusBytes']>=16*1024**2 and value['minTrainGroups']>=20,'CORPUS_MINIMUM_CANNOT_BE_LOWERED')
        real(value['maxSyntheticRatio'],'maxSyntheticRatio',0,0)  # teacher-free P0
        window=value['trainingWindow'];require(isinstance(window,dict) and set(window)=={'start','end'},'CONFIG_INVALID')
        for v in window.values():require(isinstance(v,str) and re.fullmatch(r'(?:[01]\d|2[0-3]):[0-5]\d',v),'CONFIG_INVALID')
        if value['allowedSourceIds']:
            require(self.control.bridge is not None or project.get('testOnly'),'RAG_POLICY_GATEWAY_REQUIRED',status=503)
            if self.control.bridge:
                sources={s['id']:s for s in self.control.bridge.store.list('source')}
                require(all(s in sources and sources[s].get('trainingPolicy')=='allowed' for s in value['allowedSourceIds']),'SOURCE_NOT_ALLOWED',status=403)
        if value['enabled']:require(value['allowedSourceIds'],'CORPUS_PERMISSION_REQUIRED',status=409)
        version=expected if expected is not None else body.get('expectedVersion')
        require(isinstance(version,int) and not isinstance(version,bool),'VERSION_REQUIRED',status=428)
        result=self.store.put('foundation_policy',value,identity,expected=version)
        removed=set(old.get('allowedSourceIds',[]))-set(value['allowedSourceIds'])
        if removed:
            from .corpus import iter_members
            affected=False
            for corpus in self.store.list('foundation_corpus'):
                if corpus.get('projectId')!=identity or corpus.get('state') not in ('ready','verified'):continue
                affected |= any(link.get('sourceId') in removed for member in iter_members(self.store,corpus) for link in member.get('lineage',[]))
            if affected:
                self.revoke(identity,{'reason':'project_source_permission_withdrawn','selectors':{'sourceIds':sorted(removed)}})
                result=self.store.get('foundation_policy',identity)
        self.store.event('foundation.policy.updated',identity,{'version':result['version']})
        return result

    def _pending(self, identity):
        rows=self.store.rows("SELECT id FROM jobs WHERE type LIKE 'foundation_%' AND state IN ('queued','running','pause_requested','paused','cancelling')")
        return [self.store.job(r['id']) for r in rows if self.store.job(r['id'])['payload'].get('projectId')==identity]

    def enqueue(self, kind, project_id, payload, key=None):
        project=self._project(project_id);require(kind in JOB_TYPES,'CONFIG_INVALID')
        require(not self.store.get('setting','deploymentMaintenance',{}).get('enabled'),'DEPLOYMENT_MAINTENANCE',status=503)
        require(not self._pending(project_id),'FOUNDATION_BUSY',status=409)
        if not project.get('testOnly'):require(self.has_worker(),'ISOLATED_TRAINER_REQUIRED',status=503)
        job=self.store.enqueue(kind,{'projectId':project_id,**payload},key)
        return {**job,'jobId':job['id'],'phase':job.get('progress',{}).get('stage','queued'),
            'reasonCodes':[],'revision':job.get('fence',0)}

    def create_corpus(self, identity, body=None, key=None):
        body=body or {};project=self._project(identity)
        require(not self._pending(identity),'FOUNDATION_BUSY',status=409)
        require(project.get('testOnly') or self.has_worker(),'ISOLATED_TRAINER_REQUIRED',status=503)
        require(self.control.bridge is not None,'RAG_POLICY_GATEWAY_REQUIRED',status=503)
        allowed=project['policy'].get('allowedSourceIds',[])
        selected=body.get('allowedSourceIds',allowed)
        require(isinstance(selected,list) and selected and set(selected)<=set(allowed),'SOURCE_NOT_ALLOWED',status=403)
        purpose=body.get('kind','raw')
        require(purpose in ('raw','sft'),'CONFIG_INVALID',field='kind')
        exported=(self.control.bridge.foundation_sft_corpus_page(identity,body.get('snapshotId'),selected,maxBytes=8*1024**2) if purpose=='sft'
                  else self.control.bridge.foundation_corpus(identity,body.get('sourceVersionIds'),selected))
        from .retention import admit as quota_admit
        request_id=uid();root=inside(self.store.root,'foundation/inputs/'+request_id);root.mkdir(parents=True)
        path=root/'records.jsonl';count=0;estimated=0;excluded={};first=exported
        self.store.put('foundation_input',{'projectId':identity,'storagePath':root.relative_to(self.store.root).as_posix(),
            'state':'building','purpose':purpose},request_id)
        work=None
        try:
            with path.open('w',encoding='utf-8',newline='\n') as out:
                while True:
                    records=exported.get('records',[])
                    from .relearning import filter_records
                    filtered=filter_records(self.control,project,records)
                    if len(filtered)!=len(records):excluded['RELEARNING_EXCLUSION']=excluded.get('RELEARNING_EXCLUSION',0)+len(records)-len(filtered)
                    records=filtered
                    page_bytes=sum(len(canonical(row).encode('utf-8'))+1 for row in records)
                    estimated+=page_bytes
                    quota_admit(self.store,project,additional_bytes=estimated*2+page_bytes)
                    if page_bytes:
                        from .storage_reservations import begin,commit
                        work=begin(self.store,project,page_bytes,purpose='corpus-input')
                        for row in records:out.write(canonical(row)+'\n');count+=1
                        out.flush()
                        with self.store.tx():commit(self.store,work,page_bytes)
                        work=None
                    for reason,amount in exported.get('excludedCounts',{}).items():excluded[reason]=excluded.get(reason,0)+amount
                    cursor=exported.get('nextCursor') if purpose=='sft' else None
                    if not cursor:break
                    exported=self.control.bridge.foundation_sft_corpus_page(identity,first['snapshotId'],selected,cursor=cursor,maxBytes=8*1024**2)
                    require(exported['manifestHash']==first['manifestHash'] and exported['policyVersion']==first['policyVersion'],'POLICY_STATE_CHANGED',status=409)
                require(count,'INSUFFICIENT_CORPUS',status=409)
                out.flush();os.fsync(out.fileno())
        except BaseException:
            require(root.resolve().is_relative_to((self.store.root/'foundation'/'inputs').resolve()),'PATH_OUTSIDE_SCOPE')
            import shutil
            shutil.rmtree(root,ignore_errors=True)
            if work:
                from .storage_reservations import abort
                abort(self.store,work,files_removed=not root.exists())
            self.store.put('foundation_input',{'state':'deleted' if not root.exists() else 'failed',
                'reasonCodes':['CORPUS_INPUT_NOT_PUBLISHED']},request_id)
            raise
        input_record=self.store.put('foundation_input',{'projectId':identity,'storagePath':root.relative_to(self.store.root).as_posix(),
            'inputHash':file_hash(path),'state':'ready','purpose':purpose},request_id)
        return self.enqueue('foundation_corpus',identity,{'inputPath':path.relative_to(self.store.root).as_posix(),'inputId':input_record['id'],
            'inputHash':file_hash(path),'policyVersion':exported.get('policyVersion',0),'purpose':purpose,
            'selection':first.get('selection'),'upstreamExcludedCounts':excluded,
            'ragSnapshotId':exported.get('snapshotId',body.get('snapshotId'))},key)

    def prepare_sft_corpus(self,identity):return self.create_corpus(identity,{'kind':'sft'})

    def start_tokenizer(self, identity, body=None, key=None):
        p=self._project(identity);body=body or {};corpus_id=body.get('corpusSnapshotId',body.get('corpusId',p.get('activeCorpusId')))
        from .corpus import resolve
        corpus=resolve(self.store,corpus_id or '')
        require(corpus.get('kind','foundation_corpus')=='foundation_corpus','TOKENIZER_TEACHER_DATA_FORBIDDEN')
        require(corpus['projectId']==identity and corpus['state']=='ready','CORPUS_NOT_READY',status=409)
        require(not p.get('genesisId'),'TOKENIZER_FROZEN_NEW_PROJECT_REQUIRED',status=409)
        return self.enqueue('foundation_tokenizer',identity,{'corpusId':corpus_id,'config':body.get('config',{})},key)

    def initialize(self, identity, body=None, key=None):
        p=self._project(identity);body=body or {}
        require(not p.get('genesisId'),'GENESIS_ALREADY_EXISTS',status=409)
        tokenizer_id=body.get('tokenizerArtifactId',body.get('tokenizerId',p.get('tokenizerId')))
        tokenizer=self.store.get('foundation_tokenizer',tokenizer_id or '')
        require(tokenizer['projectId']==identity and tokenizer['state']=='ready','TOKENIZER_NOT_READY',status=409)
        seed=integer(body.get('seed',p.get('seed',42)),'seed',0,2**32-1)
        return self.enqueue('foundation_initialize',identity,{'tokenizerId':tokenizer_id,'seed':seed},key)

    def training_inputs(self, identity, body=None, stage='pretrain'):
        p=self._project(identity);body=body or {};require(stage in ('pretrain','cpt','sft'),'CONFIG_INVALID')
        parent=body.get('parentCheckpointId',p.get('latestCheckpointId') or p.get('genesisId'))
        require(parent,'GENESIS_REQUIRED',status=409)
        corpus_id=body.get('corpusSnapshotId',body.get('corpusId',p.get('activeSftCorpusId') if stage=='sft' else p.get('activeCorpusId')))
        from .corpus import resolve
        corpus=resolve(self.store,corpus_id or '')
        require(corpus['projectId']==identity and corpus['state']=='ready','CORPUS_NOT_READY',status=409)
        if stage=='sft':require(corpus.get('sftExampleCount',0)>0 or corpus.get('objective')=='sft' and corpus.get('splitCounts',{}).get('train',0)>0,'APPROVED_SFT_CORPUS_REQUIRED',status=409)
        tokenizer_id=body.get('tokenizerArtifactId',body.get('tokenizerId',p.get('tokenizerId')))
        requested=dict(body.get('config',{}))
        if stage=='sft':requested.setdefault('learningRate',5e-5);requested.setdefault('maxEpochs',2)
        requested['sliceSeconds']=min(requested.get('sliceSeconds',900),p['policy'].get('sliceSeconds',900))
        requested['maxActiveSeconds']=min(requested.get('maxActiveSeconds',14400),p['policy'].get('dailyActiveSeconds',14400))
        config=self.trainer.config(requested,maximum_context=p['config']['max_position_embeddings'])
        require(self.control.execution_scope=='pc_validation' or config.get('cpuThreads',1)==1,'NAS_CPU_THREADS_FIXED')
        parent_record=self.store.get('foundation_checkpoint',parent,{}) or self.store.get('foundation_genesis',parent,{})
        token_record=self.store.get('foundation_tokenizer',tokenizer_id or '')
        require(parent_record.get('projectId')==identity and parent_record.get('state') in ('ready','verified','completed'),'PARENT_NOT_READY',status=409)
        require(not parent_record.get('randomBaselineOnly'),'RANDOM_BASELINE_NOT_TRAINING_PARENT',status=409)
        require(token_record.get('projectId')==identity and token_record.get('state')=='ready','TOKENIZER_NOT_READY',status=409)
        require(parent_record.get('tokenizerHash')==token_record.get('tokenizerHash'),'TOKENIZER_HASH_MISMATCH',status=409)
        return p,corpus,parent_record,token_record,config

    def profile_for(self, project, config, stage='pretrain',corpus=None):
        from .backend import LOCK_PATH
        from .resources import machine_fingerprint
        fingerprint=machine_fingerprint()
        token=self.store.get('foundation_tokenizer',project.get('tokenizerId') or '',{})
        parent=self.store.get('foundation_checkpoint',project.get('latestCheckpointId') or '',{}) or self.store.get('foundation_genesis',project.get('genesisId') or '',{})
        expected_architecture=parent.get('architectureHashV2',project.get('architectureHashV2'))
        from .packing import teacher_mix_descriptor
        mix=teacher_mix_descriptor(corpus or {})
        return next((r for r in self.store.list('foundation_profile') if r.get('projectId')==project['id']
            and r.get('passed') is True and r.get('state')=='passed' and r.get('configHash')==digest(config) and r.get('stage','pretrain')==stage
            and (stage!='sft' or corpus and r.get('corpusId')==corpus.get('id')
                and (not r.get('corpusHash') or r['corpusHash']==corpus.get('manifestHash')))
            and r.get('tokenizerHash')==token.get('tokenizerHash') and r.get('architectureHashV2')==expected_architecture
            and r.get('teacherMixHash')==(digest(mix) if mix else None)
            and (project.get('testOnly') or r.get('backendReceipt',{}).get('backendLockHash')==file_hash(LOCK_PATH))
            and (project.get('testOnly') or r.get('environment',{}).get('machineFingerprint')==fingerprint)
            and bool(r.get('testOnly'))==bool(project.get('testOnly'))),None)

    def calibrate(self,identity,body=None,key=None):
        body=body or {};stage=body.get('stage','pretrain')
        p,corpus,parent,token,config=self.training_inputs(identity,body,stage)
        require(set(body.get('measurement',{}))<={'measurementSteps','warmupSteps','maxSeconds'},'CONFIG_INVALID')
        return self.enqueue('foundation_calibrate',identity,{'stage':stage,'corpusId':corpus['id'],'parentCheckpointId':parent['id'],
            'tokenizerId':token['id'],'config':{**config,**body.get('measurement',{})}},key)

    def start_training(self, identity, body=None, stage='pretrain', key=None):
        body=body or {};p,corpus,parent_record,token_record,config=self.training_inputs(identity,body,stage)
        parent=parent_record['id'];corpus_id=corpus['id'];tokenizer_id=token_record['id']
        profile=self.profile_for(p,config,stage,corpus)
        require(p.get('testOnly') or profile,'CALIBRATION_REQUIRED',status=409)
        with self.store.tx():
            job=self.enqueue('foundation_'+stage,identity,{'stage':stage,'corpusId':corpus_id,'tokenizerId':tokenizer_id,
                'parentCheckpointId':parent,'config':config,'profileId':profile['id'] if profile else None,
                'continueCorpus':body.get('continueCorpus') is True},key)
            run=self.store.get('foundation_run',job['id'],{})
            if not run:self.store.put('foundation_run',{'projectId':identity,'stage':stage,'state':'queued','corpusId':corpus_id,
                'tokenizerId':tokenizer_id,'parentCheckpointId':parent,'config':config,'trainingMode':'full_parameters',
                'completedOptimizerSteps':0,'committedTargetTokens':0,'attemptedTargetTokens':0,'observedLoss':None,
                'effectiveTokensPerSecond':None,'peakRssBytes':None,'etaStatus':'unmeasured'},job['id'])
            self.store.put('foundation_project',{'latestRunId':job['id'],'reasonCodes':[]},identity)
        return job

    def command(self, identity, action):
        job=self.store.job(identity);require(job['type'] in TRAIN_TYPES,'FOUNDATION_TRAINING_RUN_REQUIRED')
        if action=='retry':return self._retry_failed(job)
        with self.store.tx():
            if action=='pause':
                require(job['state']=='running','PAUSE_NOT_AVAILABLE',status=409)
                self.store.db.execute("UPDATE jobs SET state='pause_requested' WHERE id=?",(identity,))
                self.store.put('foundation_run',{'pauseReason':'USER_PAUSED'},identity)
            elif action=='resume':
                require(job['state']=='paused','RESUME_NOT_AVAILABLE',status=409)
                run=self.store.get('foundation_run',identity)
                checkpoint=run.get('checkpointId') or run.get('lastCheckpointId')
                require(checkpoint,'CHECKPOINT_REQUIRED')
                payload={**job['payload'],'checkpointId':checkpoint}
                self.store.db.execute("UPDATE jobs SET state='queued',payload=?,owner=NULL,lease=0 WHERE id=?",(canonical(payload),identity))
                self.store.put('foundation_run',{'state':'queued','pauseReason':None},identity)
            elif action=='cancel':
                require(job['state'] not in ('completed','cancelled','failed'),'JOB_ALREADY_FINISHED',status=409)
                state='cancelling' if job['state'] in ('running','pause_requested') else 'cancelled'
                self.store.db.execute('UPDATE jobs SET state=? WHERE id=?',(state,identity))
                if state=='cancelled':self.store.put('foundation_run',{'state':'cancelled'},identity)
            else:raise Fault('INVALID_JOB_ACTION')
        return self.store.job(identity)

    def _retry_failed(self,job):
        """Explicit recovery from an intact same-run commit, with a new fence."""
        require(job['type'] in ('foundation_pretrain','foundation_cpt','foundation_sft')
                and job['state']=='failed','RETRY_NOT_AVAILABLE',status=409)
        project=self._project(job['payload']['projectId']);self.validate_policy(project)
        run=self.store.get('foundation_run',job['id'])
        checkpoint_id=run.get('lastCheckpointId') or run.get('checkpointId')
        require(checkpoint_id,'CHECKPOINT_REQUIRED')
        record=self.store.get('foundation_checkpoint',checkpoint_id)
        from .checkpoint import verify
        root,manifest=verify(self.store,record)  # File hashing never holds the queue write lock.
        body=job['payload'];stage=job['type'].removeprefix('foundation_')
        _,corpus,parent,token,config=self.training_inputs(project['id'],body,stage)
        from .packing import teacher_mix_descriptor
        mix=teacher_mix_descriptor(corpus)
        expected_input=digest([project['id'],parent['id'],corpus['manifestHash'],token['tokenizerHash'],stage,config]
                              +([mix] if mix else []))
        from .checkpoint import _unpack
        tree=json.loads((root/'state.json').read_text('utf-8'))
        saved_tree=next((value for key,value in tree.get('items',[]) if key=={'type':'scalar','value':'state'}),None)
        saved=_unpack(saved_tree,{})
        require(record.get('state')=='ready' and record.get('runId')==job['id']
                and record.get('projectId')==project['id']
                and record.get('inputHash')==run.get('inputHash')==expected_input
                and record.get('runConfigHash')==digest(body['config'])==digest(config)
                and saved.get('inputHash')==expected_input and saved.get('runId')==job['id']
                and saved.get('projectId')==project['id'] and saved.get('runConfigHash')==digest(config)
                and manifest.get('tokenizerId')==record.get('tokenizerId')==token['id']
                and all(saved.get(key)==record.get(key) for key in
                        ('corpusId','parentCheckpointId','stage','completedOptimizerSteps','cursor','committedTargetTokens')),
                'CHECKPOINT_INCOMPATIBLE',status=409)
        with self.store.tx():
            current=self.store.job(job['id'])
            require(current['state']=='failed' and current['fence']==job['fence'] and current['payload']==job['payload']
                    and self.store.get('foundation_run',job['id'])['version']==run['version']
                    and self.store.get('foundation_checkpoint',checkpoint_id)['version']==record['version'],
                    'VERSION_CONFLICT',status=409)
            self._project(project['id'])
            require(not self._pending(project['id']),'FOUNDATION_BUSY',status=409)
            self.store.db.execute("UPDATE jobs SET state='queued',payload=?,owner=NULL,lease=0,error=NULL,updated=? WHERE id=?",
                (canonical({**current['payload'],'checkpointId':checkpoint_id}),now(),job['id']))
            self.store.put('foundation_run',{'state':'queued','pauseReason':None,'error':None,
                'recoveredFromCheckpointId':checkpoint_id,'recoveryCount':run.get('recoveryCount',0)+1},job['id'])
            self.store.put('foundation_project',{'latestCheckpointId':checkpoint_id,'reasonCodes':[]},project['id'])
            self.store.event('foundation.run.retry',job['id'],{'checkpointId':checkpoint_id,'previousFence':job['fence']})
        return self.store.job(job['id'])

    control_run=command

    def run_status(self,identity):
        value=self.store.get('foundation_run',identity)
        job=self.store.job(identity)
        progress=job.get('progress',{})
        live={k:progress[k] for k in ('completedOptimizerSteps','observedLoss','attemptedTargetTokens','effectiveTokensPerSecond')
            if k in progress and job['state'] in ('running','pause_requested')}
        return {**value,**live,'progress':progress,'phase':progress.get('stage',job['state']),
                'checkpointCommittedOptimizerSteps':value.get('completedOptimizerSteps'),
                'jobState':job['state'],'state':job['state'],
                'capabilities':{'pause':job['state']=='running','resume':job['state']=='paused',
                                'retry':job['state']=='failed' and bool(value.get('lastCheckpointId') or value.get('checkpointId')),
                                'cancel':job['state'] in ('queued','running','paused','pause_requested')}}

    def validate_policy(self, project):
        if project.get('testOnly'):return
        require(project.get('executionScope','nas')==self.control.execution_scope,'EXECUTION_SCOPE_REQUALIFICATION_REQUIRED',status=409)
        if self.control.bridge:self.control.bridge.refresh_policy()
        require(project.get('state')!='revoked' and not project.get('revocationPending'),'POLICY_REVOKED',status=409)
        epoch=self.store.get('setting','policyEpoch',{})
        require(epoch.get('observedAt'),'RAG_POLICY_GATEWAY_REQUIRED',status=503)
        age=(datetime.now(timezone.utc)-datetime.fromisoformat(epoch['observedAt'].replace('Z','+00:00'))).total_seconds()
        require(age<60,'POLICY_STATE_STALE',status=409)
        current=self.store.get('foundation_project',project['id'])
        require(current.get('state')!='revoked' and not current.get('revocationPending'),'POLICY_REVOKED',status=409)
        return epoch

    def refresh_policy(self):
        """RAG is authoritative. Withdrawals invalidate every exposed descendant."""
        if getattr(self,'_refreshing_policy',False):return
        self._refreshing_policy=True
        try:return self._audit_policy()
        finally:self._refreshing_policy=False

    def _audit_policy(self):
        bridge=self.control.bridge
        if bridge is None:return
        epoch=bridge.store.setting('policyVersion',0)
        data_revision=bridge.store.setting('trainingDataRevision',0)
        previous=self.store.get('setting','foundationPolicyAudit',{})
        if previous.get('epoch')==epoch and previous.get('trainingDataRevision')==data_revision:return
        sources={r['id']:r for r in bridge.store.list('source')}
        documents={r['id']:r for r in bridge.store.list('document')}
        versions={r['id']:r for r in bridge.store.list('version')}
        for project in self.store.list('foundation_project'):
            if project.get('state')=='revoked' or project.get('testOnly'):continue
            invalid=False;invalid_versions=set()
            from .corpus import iter_members
            for corpus in self.store.list('foundation_corpus'):
                if corpus.get('projectId')!=project['id'] or corpus.get('state') not in ('ready','verified'):continue
                for member in iter_members(self.store,corpus):
                    for link in member.get('lineage',[]):
                        source=sources.get(link.get('sourceId'),{});document=documents.get(link.get('documentId'),{})
                        version=versions.get(link.get('versionId'),{})
                        bad = (source.get('trainingPolicy')!='allowed' or source.get('upstreamDeleted') is True
                            or not document or document.get('deletedAt') is not None or not document.get('enabled',True)
                            or document.get('trainingPolicy')=='denied' or not version or version.get('qualityStatus')!='passed')
                        invalid |= bad
                        if bad and link.get('versionId'):invalid_versions.add(link['versionId'])
                # Harmless unrelated policy changes need no retraining or tokenizer reset.
                if not invalid:self.store.put('foundation_corpus',{'policyVersion':epoch},corpus['id'])
            if invalid:self.revoke(project['id'],{'reason':'upstream_permission_or_quality_revoked','selectors':{'versionIds':sorted(invalid_versions)}})
        self.store.put('setting',{'epoch':epoch,'trainingDataRevision':data_revision,'observedAt':now()},'foundationPolicyAudit')

    def execute_local(self, job):
        """Trusted entrypoint for a network-isolated shared-volume worker."""
        p=self._project(job['payload']['projectId']);self.validate_policy(p)
        if not p.get('testOnly'):
            from .backend import verify
            receipt=verify(p.get('backendReceipt'),strict_platform=self.control.execution_scope=='nas')
            self.store.put('foundation_project',{'backendReceipt':receipt},p['id'])
            p={**p,'backendReceipt':receipt}
        admit(self.store,p)
        from .retention import admit as quota_admit
        self.store.progress(job,'preparing',0,None,'stages',preparingPhase='quota_admission')
        quota_admit(self.store,p)
        kind=job['type'];body=job['payload']
        def policy_validator(corpus):
            current=self._project(p['id']);epoch=self.validate_policy(current)
            if not p.get('testOnly'):require(corpus.get('policyVersion',0)==epoch.get('epoch',0),'CORPUS_POLICY_REFRESH_REQUIRED',status=409)
        self.trainer.policy_validator=policy_validator
        if kind=='foundation_corpus':
            from .corpus import build
            if not p.get('testOnly'):
                require(body.get('policyVersion')==self.validate_policy(p).get('epoch'),'CORPUS_POLICY_REFRESH_REQUIRED',status=409)
            path=inside(self.store.root,body['inputPath']);require(file_hash(path)==body['inputHash'],'CORPUS_HASH_MISMATCH')
            with path.open(encoding='utf-8') as f:
                records=(json.loads(line) for line in f if line.strip())
                value=build(self.store,p,records,{'policyVersion':body.get('policyVersion',0),**p['policy'],
                    'objective':'sft' if body.get('purpose')=='sft' else 'causal_lm'})
            purpose=body.get('purpose','raw')
            value=self.store.put('foundation_corpus',{'purpose':purpose,'corpusKind':purpose,
                'sftExampleCount':sum(value.get('splitCounts',{}).values()) if purpose=='sft' else 0,
                'selection':body.get('selection'),'upstreamExcludedCounts':body.get('upstreamExcludedCounts',{}),
                'ragSnapshotId':body.get('ragSnapshotId')},value['id'])
            fields={'activeSftCorpusId':value['id']} if purpose=='sft' else {'activeCorpusId':value['id']}
            if not p.get('tokenizerId'):fields['stage']='waiting_tokenizer'
            self.store.put('foundation_project',{**fields,'reasonCodes':[]},p['id'])
            if body.get('inputId') and value.get('state')=='ready':
                require(path.parent==inside(self.store.root,'foundation/inputs/'+body['inputId']) and not path.is_symlink(),'PATH_OUTSIDE_SCOPE')
                path.unlink();path.parent.rmdir()
                self.store.put('foundation_input',{'state':'deleted','deletedAt':now(),'corpusId':value['id']},body['inputId'])
            return value
        if kind=='foundation_tokenizer':
            from .tokenizer import build
            value=build(self.store,p,self.store.get('foundation_corpus',body['corpusId']),body.get('config',{}))
            self.store.put('foundation_project',{'tokenizerId':value['id'],'stage':'tokenizer_ready','reasonCodes':[]},p['id'])
            return value
        if kind=='foundation_initialize':
            from .architecture import initialize
            value=initialize(self.store,{**p,'seed':body.get('seed',p.get('seed',42))},self.store.get('foundation_tokenizer',body['tokenizerId']))
            self.store.put('foundation_project',{'genesisId':value['id'],'seed':value['seed'],'stage':'genesis_ready','reasonCodes':[]},p['id'])
            return value
        if kind=='foundation_calibrate':
            from .calibration import run
            from .corpus import resolve
            value=run(self.store,p,resolve(self.store,body['corpusId']),body['tokenizerId'],
                body['parentCheckpointId'],body['config'],job)
            self.store.put('foundation_project',{'latestProfileId':value['id'],'reasonCodes':value.get('reasonCodes',[])},p['id'])
            return value
        if kind=='foundation_format':
            from .format_jobs import run
            return run(self.control,job)
        if kind=='foundation_distill':
            from .distillation import run
            result=run(self.control,job)
            if result.get('checkpointId'):
                self.store.put('foundation_project',{'latestCheckpointId':result['checkpointId'],'latestRunId':job['id'],
                    'stage':'foundation_candidate','reasonCodes':[]},p['id'])
            return result
        if kind=='foundation_response_distill':
            from .response_distillation import generate
            return generate(self.control,job)
        if kind in TRAIN_TYPES:
            from .corpus import resolve
            require(p.get('testOnly') or self.profile_for(p,body['config'],body['stage'],resolve(self.store,body['corpusId'])),'CALIBRATION_REQUIRED',status=409)
            self.store.put('foundation_project',{'stage':'instruction_training' if body['stage']=='sft' else 'pretraining',
                'latestRunId':job['id'],'reasonCodes':[]},p['id'])
            value=self.trainer.run(job)
            run=self.store.get('foundation_run',job['id'],{})
            checkpoint=value.get('checkpointId') or value.get('lastCheckpointId') or run.get('checkpointId') or run.get('lastCheckpointId')
            updates={'stage':('pretraining' if value.get('state') in ('paused','yielded') else 'foundation_candidate') if body['stage']!='sft' else 'assistant_candidate','latestRunId':job['id']}
            if checkpoint:updates['latestCheckpointId']=checkpoint
            self.store.put('foundation_project',updates,p['id']);return value
        if kind=='foundation_evaluate':
            domain_suite=self.store.get('foundation_domain_suite',body.get('suiteId',''),{})
            if domain_suite:
                if domain_suite['kind']=='audit':
                    from .audit import run
                    require(body.get('audit') is True,'AUDIT_NOT_FOR_TUNING',status=403)
                    return run(self.control,job)
                from .domains import evaluate
                return evaluate(self.control,job)
            suite=self.store.get('foundation_suite',body.get('suiteId',''),{})
            if suite.get('kind')=='audit':
                require(body.get('audit') is True,'AUDIT_NOT_FOR_TUNING',status=403)
                from .audit import reserve,aggregate
                if self.control.sandbox.available():reserve(self.store,suite,job['id'],p['id'])
                result=self.evaluator.evaluate(job)
                self.store.put('foundation_audit_feedback',aggregate(result),result['id'])
                return result
            return self.evaluator.evaluate(job)
        if kind=='foundation_relearn':
            from .relearning import execute
            return execute(self.control,job)
        if kind=='foundation_growth':
            from .growth import experiment
            return experiment(self.store,p['id'],body['parentCheckpointId'],body['recipe'],job)
        if kind=='foundation_growth_adopt':
            from .growth import adopt
            return adopt(self.control,body['candidateId'],body.get('options',{}))
        if kind=='foundation_self_code_evaluate':
            from .selfcode import evaluate
            return evaluate(self.control,job)
        raise Fault('UNSUPPORTED_FOUNDATION_JOB')

    def execute(self, job):
        """Model work lives in a child, so timeout/memory failures can be contained."""
        p=self._project(job['payload']['projectId'])
        require(p.get('testOnly') or isolated_network() or self.control.execution_scope=='pc_validation','ISOLATED_TRAINER_REQUIRED',status=503)
        require(p.get('testOnly') or p.get('executionScope','nas')==self.control.execution_scope,'EXECUTION_SCOPE_REQUALIFICATION_REQUIRED',status=409)
        directory=inside(self.store.root,'foundation/job-results/'+job['id']);directory.mkdir(parents=True,exist_ok=True)
        result_path=directory/'result.json'
        if result_path.exists():result_path.unlink()
        env=os.environ.copy();env.update({'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'1',
            'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false','RAYON_NUM_THREADS':'1'})
        for name in tuple(env):
            if name.startswith('SEED_EVALUATION_'):env.pop(name)
        from .pc_budget_guard import ENV as budget_env,bind as bind_pc_budget
        env.pop(budget_env,None)
        env['SEED_EXECUTION_SCOPE']=self.control.execution_scope
        command=[sys.executable,'-m','LLM.foundation.worker','--data-dir',str(self.store.root),'--job-id',job['id'],
                 '--owner',job['owner'],'--fence',str(job['fence'])]
        started=time.time();attempt=uid();failure=None;peak_rss=0;last_metric=0
        evaluation_attempt=None;process_observation=None
        if job['type']=='foundation_evaluate':
            from .evaluation_restart import runtime_threads,runtime_metadata
            approved_threads=runtime_threads(self.control,job)
            suite=self.store.get('foundation_suite',job['payload'].get('suiteId') or '',{})
            release=suite.get('kind')=='release' and not self.store.get('foundation_domain_suite',suite.get('id',''),{})
            threads=1 if approved_threads is None else approved_threads
            require(type(threads) is int and 1<=threads<=4,'EVALUATION_CPU_THREADS_INVALID')
            if approved_threads is not None:
                require(release and self.control.execution_scope=='pc_validation' and p.get('executionScope')=='pc_validation',
                    'PC_RELEASE_EVALUATION_REQUIRED',status=403)
                choice=runtime_metadata(self.control,job)
                require(all(choice.get(key)==value for key,value in {'cpuThreads':threads,'jobId':job['id'],
                    'executionScope':'pc_validation','owner':job['owner'],'fence':job['fence'],'payloadHash':digest(job['payload'])}.items())
                    and re.fullmatch('[0-9a-f]{64}',choice.get('contextHash','')),'EVALUATION_RUNTIME_CONTEXT_CHANGED',status=409)
                for suffix,value in {'JOB_ID':job['id'],'OWNER':job['owner'],'FENCE':job['fence'],
                    'CONTEXT_HASH':choice['contextHash'],'CPU_THREADS':threads,'SCOPE':'pc_validation'}.items():
                    env['SEED_EVALUATION_'+suffix]=str(value)
            else:
                choice={'cpuThreads':1,'jobId':job['id'],'owner':job['owner'],'fence':job['fence'],
                    'executionScope':self.control.execution_scope,'payloadHash':digest(job['payload']),'reason':'default_worker_cpu_limit'}
                choice['contextHash']=digest(choice)
            if release:
                for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):env[name]=str(threads)
                env['SEED_EVALUATION_ATTEMPT_ID']=attempt
                with self.store.tx():
                    self.store.assert_lease(job)
                    evaluation_attempt=self.store.put('foundation_evaluation_attempt',{
                        'schemaVersion':1,'jobId':job['id'],'projectId':p['id'],'jobType':job['type'],
                        **choice,'attemptId':attempt,'threadChoiceContextHash':choice['contextHash'],
                        'environment':{name:env[name] for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
                        'expectedTorchThreads':threads,'weightsChanged':False,'decodingChanged':False,
                        'expectedInteropThreads':1 if approved_threads==4 else None,
                        'threadBudgetScope':'native_and_intra_op_pools_not_total_os_threads',
                        'resourceSelectionOnly':True,'observedAt':now()},attempt,expected=0)
        remaining=max(0,p['policy'].get('dailyActiveSeconds',14400)-used_today(self.store))
        pc_binding=bind_pc_budget(self.control,job)
        if pc_binding:env[budget_env]=pc_binding['id']
        with (directory/'worker.log').open('wb') as output:
            process=subprocess.Popen(command,stdout=output,stderr=output,env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            try:
                import psutil
                observed=psutil.Process(process.pid)
                if evaluation_attempt and approved_threads is not None:
                    runner=psutil.Process(os.getpid())
                    observation={'schemaVersion':1,'attemptId':attempt,'attemptHash':digest(evaluation_attempt),
                        'jobId':job['id'],'owner':job['owner'],'fence':job['fence'],'executionScope':'pc_validation',
                        'cpuThreads':evaluation_attempt['cpuThreads'],'contextHash':evaluation_attempt['contextHash'],
                        'restartId':evaluation_attempt['restartId'],'payloadHash':digest(job['payload']),
                        'runner':{'pid':os.getpid(),'createTime':runner.create_time()},
                        'child':{'pid':process.pid,'createTime':observed.create_time()},'observedAt':now()}
                    observation['observationHash']=digest(observation)
                    with self.store.tx():
                        self.store.assert_lease(job)
                        process_observation=self.store.put('foundation_evaluation_process',observation,attempt,expected=0)
                grace=None
                while process.poll() is None:
                    self.store.heartbeat(job)
                    state=self.store.assert_lease(job)['state']
                    try:rss=observed.memory_info().rss+sum(x.memory_info().rss for x in observed.children(recursive=True) if x.is_running())
                    except psutil.NoSuchProcess:
                        if process.poll() is not None:break
                        continue
                    elapsed=time.time()-started
                    peak_rss=max(peak_rss,rss)
                    if job['type'] in TRAIN_TYPES and elapsed-last_metric>=2:
                        self.store.put('foundation_run',{'peakRssBytes':peak_rss},job['id']);last_metric=elapsed
                    policy=p['policy'];budget=foundation_budget(self.store,test_only=p.get('testOnly'))
                    pressure=not p.get('testOnly') and (rss>4*1024**3 or budget['availableRamBytes']<1024**3
                        or budget.get('hostAvailableRamBytes',budget['availableRamBytes'])<4*1024**3 or budget['freeDiskBytes']<5*1024**3)
                    timeout=elapsed>min(policy.get('sliceSeconds',900),remaining) and job['type'] in TRAIN_TYPES
                    stopping=self.control.stop_event.is_set() or self.store.get('setting','deploymentMaintenance',{}).get('enabled')
                    if pressure or timeout or stopping:
                        with self.store.tx():self.store.db.execute("UPDATE jobs SET state='pause_requested' WHERE id=? AND state='running'",(job['id'],))
                        if job['type'] in TRAIN_TYPES:
                            reason=('RESOURCE_PRESSURE' if pressure else 'DAILY_BUDGET_EXHAUSTED' if elapsed>=remaining
                                else 'SLICE_LIMIT' if timeout else 'DEPLOYMENT_MAINTENANCE' if self.store.get('setting','deploymentMaintenance',{}).get('enabled') else 'WORKER_SHUTDOWN')
                            self.store.put('foundation_run',{'pauseReason':reason},job['id'])
                        grace=grace or time.monotonic();failure='RESOURCE_PRESSURE' if pressure else None
                    if grace and time.monotonic()-grace>60:
                        process.kill();failure=failure or 'CHECKPOINT_GRACE_EXCEEDED';break
                    if elapsed>7200 and job['type'] not in TRAIN_TYPES:
                        process.kill();failure='BACKGROUND_DEADLINE_EXCEEDED';break
                    if job['type']=='foundation_calibrate' and elapsed>job['payload'].get('config',{}).get('maxSeconds',300)+60:
                        process.kill();failure='PROFILE_TIME_LIMIT';break
                    time.sleep(.5)
                process.wait(timeout=10)
            finally:
                if process.poll() is None:process.kill();process.wait(timeout=10)
                if job['type'] in TRAIN_TYPES or job['type'] in ('foundation_calibrate','foundation_response_distill','foundation_growth','foundation_growth_adopt','foundation_format'):
                    charge_time(self.store,job['id']+':'+attempt,started,time.time()-started)
        if failure:raise Fault(failure,status=503)
        require(result_path.is_file(),'FOUNDATION_WORKER_FAILED',status=503)
        result=json.loads(result_path.read_text('utf-8'))
        require(result.get('fence')==job['fence'],'LEASE_EXPIRED',status=409)
        if result.get('error'):raise Fault(result['error']['code'],status=result['error'].get('status',422))
        if evaluation_attempt:
            measured=self.store.get('foundation_evaluation_worker_environment',attempt,{})
            require(result.get('evaluationAttemptId')==attempt and measured.get('attemptHash')==digest(evaluation_attempt)
                and measured.get('jobId')==job['id'] and measured.get('fence')==job['fence']
                and measured.get('cpuThreads')==evaluation_attempt['cpuThreads']==measured.get('torchNumThreads')
                and result.get('finalTorchNumThreads')==evaluation_attempt['cpuThreads']
                and (evaluation_attempt.get('expectedInteropThreads') is None or
                    measured.get('torchNumInteropThreads')==result.get('finalTorchNumInteropThreads')==1),
                'EVALUATION_CPU_THREADS_NOT_OBSERVED',status=409)
            if process_observation:
                require(measured.get('process')==process_observation['child'],
                    'EVALUATION_PROCESS_IDENTITY_CHANGED',status=409)
        return result['result']

    def evaluate(self, body, key=None):
        identity=body.get('projectId')
        reference=body.get('checkpointId') or body.get('artifactId') or ''
        artifact=self.store.get('foundation_checkpoint',reference,{}) or self.store.get('artifact',reference,{})
        if not identity:
            identity=artifact.get('projectId',artifact.get('foundationProjectId'))
        p=self._project(identity)
        require(artifact.get('projectId',artifact.get('foundationProjectId'))==identity,'INCOMPATIBLE_BASE',status=409)
        return self.enqueue('foundation_evaluate',identity,body,key)

    def curriculum(self,identity,body=None):
        from .curriculum import Curriculum
        return Curriculum(self.control).configure(identity,body) if body is not None else self.store.get('foundation_curriculum',identity,{})

    def start_distillation(self,identity,body,key=None):
        p=self._project(identity)
        parent=body.get('parentCheckpointId') or p.get('latestCheckpointId') or p.get('genesisId')
        corpus=body.get('corpusId') or p.get('activeCorpusId')
        require(parent and corpus,'GENESIS_AND_CORPUS_REQUIRED',status=409)
        require(set(body)<={'parentCheckpointId','corpusId','teacherArtifactId','teacherManifestHash','config'},'CONFIG_INVALID')
        return self.enqueue('foundation_distill',identity,{**body,'stage':'distillation','parentCheckpointId':parent,'corpusId':corpus},key)

    def start_response_distillation(self,identity,body,key=None):
        from .response_distillation import _policy
        self._project(identity);policy=_policy(self.control,identity)
        require(set(body)<={'sourceCorpusId','teacherApprovalId','tasks'},'CONFIG_INVALID')
        tasks=body.get('tasks',[])
        require(isinstance(tasks,list) and 0<len(tasks)<=policy['maxResponses'],'TRAIN_TASKS_REQUIRED')
        for task in tasks:
            require(isinstance(task,dict) and set(task)=={'sourceHash','fixtureId'}
                and isinstance(task['sourceHash'],str) and re.fullmatch('[0-9a-f]{64}',task['sourceHash'])
                and isinstance(task['fixtureId'],str) and 0<len(task['fixtureId'])<=200,'TRAIN_TASKS_REQUIRED')
        return self.enqueue('foundation_response_distill',identity,{**body,'teacherPolicyHash':policy['policyHash'],
            'teacherPolicyVersion':policy['version']},key)

    def adopt_growth(self,identity,body):
        from .growth import adopt
        candidate=self.store.get('foundation_growth_candidate',body.get('candidateId',''))
        require(candidate.get('sourceProjectId')==identity,'GROWTH_SOURCE_CHANGED')
        require(set(body)<={'candidateId','name','autoTrain'},'CONFIG_INVALID')
        return self.enqueue('foundation_growth_adopt',identity,{'candidateId':candidate['id'],
            'options':{k:v for k,v in body.items() if k!='candidateId'}},'growth-adopt:'+candidate['id'])

    def start_growth(self,identity,body,key=None):
        p=self._project(identity);parent=body.get('parentCheckpointId') or p.get('latestCheckpointId')
        require(parent,'TRAINED_CHECKPOINT_REQUIRED',status=409)
        checkpoint=self.store.get('foundation_checkpoint',parent)
        require(checkpoint.get('projectId')==identity and checkpoint.get('completedOptimizerSteps',0)>0,'TRAINED_CHECKPOINT_REQUIRED',status=409)
        return self.enqueue('foundation_growth',identity,{'parentCheckpointId':parent,'recipe':body.get('recipe',body)},key)

    def advance_extensions(self):
        if self.store.get('setting','deploymentMaintenance',{}).get('enabled'):return
        from .growth import advance as advance_growth
        for policy in self.store.list('foundation_growth_policy'):
            if policy.get('enabled'):advance_growth(self.control,policy['projectId'])
        from .curriculum import Curriculum
        for plan in self.store.list('foundation_curriculum'):
            if plan.get('enabled') and plan.get('state') not in ('completed','held','disabled'):
                Curriculum(self.control).advance(plan['projectId'])
        from .relearning import advance
        for plan in self.store.list('foundation_relearning_plan'):
            if plan.get('state') not in ('completed','held','failed','cancelled') and (plan.get('state')!='planned' or plan.get('autoStart')):
                try:advance(self.control,plan['id'])
                except Fault as error:self.store.put('foundation_relearning_plan',{'state':'held','reasonCodes':[error.code]},plan['id'])

    def maintain_storage(self):
        if self.store.get('setting','deploymentMaintenance',{}).get('enabled'):return
        from LLM.storage.lifecycle import expire_backups
        # A durable next-check time survives controller restarts.
        old=self.store.get('setting','backupLifecycle',{})
        if old.get('nextCheckEpoch',0)>time.time():return
        result=expire_backups(self.store)
        self.store.put('setting',{'state':result['state'],'nextCheckEpoch':time.time()+3600,
                       'deletedCount':result['deletedCount'],'observedAt':now()},'backupLifecycle')

    def graduate(self, body):
        from .graduation import graduate
        p=self._project(body['projectId']);checkpoint=self.store.get('foundation_checkpoint',body.get('checkpointId') or p.get('latestCheckpointId') or '')
        evaluation=self.store.get('foundation_evaluation',body['evaluationId'])
        return graduate(self.store,p,checkpoint,evaluation,registry=self.control.registry)

    def prepare_release(self, body):
        from .graduation import certify_usable
        p=self._project(body['projectId'])
        reference=body.get('checkpointId') or body.get('artifactId') or p.get('latestCheckpointId') or ''
        checkpoint=self.store.get('foundation_checkpoint',reference,{}) or self.store.get('artifact',reference)
        require(checkpoint.get('projectId',checkpoint.get('foundationProjectId'))==p['id'],'INCOMPATIBLE_BASE')
        evaluation=self.store.get('foundation_evaluation',body['evaluationId'])
        result=certify_usable(self.store,p,checkpoint,evaluation,registry=self.control.registry)
        certificate,release=result['certificate'],result['release']
        from .tokenizer import load
        token_record=self.store.get('foundation_tokenizer',release['tokenizerId']);load(self.store,token_record)
        source=record_path(self.store,token_record)/'tokenizer.json'
        directory=inside(self.store.root,'releases/'+release['id']);directory.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,directory/'tokenizer.json')
        profile={**release['profile'],'provider':'foundation','releaseId':release['id'],
                 'tokenizerRevision':token_record['tokenizerHash'],'tokenizerSha256':file_hash(source),
                 'contextLength':release['profile'].get('trainedContextTokens',512),'addSpecialTokens':False}
        profile_hash=digest({k:v for k,v in profile.items() if k!='profileHash'})
        profile['profileHash']=profile_hash
        certificate={**certificate,'profileHash':profile_hash,'purpose':'usable_release','usable':True}
        atomic_json(directory/'generation-profile.json',profile)
        self.store.put('foundation_certificate',certificate,certificate['id'])
        self.store.put('certificate',certificate,certificate['id'])
        release={**release,'profile':profile,'profileHash':profile_hash,'tokenizerPath':str(directory/'tokenizer.json')}
        self.store.put('foundation_release',release,release['id'])
        result['release']=self.store.put('release',release,release['id'])
        result['certificate']=self.store.get('certificate',certificate['id'])
        return result

    def stream(self,body):
        with self.control.inference.foreground():
            prepared=self.runtime.prepare(body);request_id=uid()
            try:
                for line in self.runtime.stream(body):
                    item=json.loads(line)
                    item.update({'requestId':request_id,'releaseId':prepared['releaseId'],'profileHash':prepared['profileHash'],
                                 'delta':item.get('response','')})
                    if item.get('done'):item.update({'finishReason':item.get('done_reason','stop'),'inputTokens':item.get('prompt_eval_count'),'outputTokens':item.get('eval_count')})
                    yield canonical(item)+'\n'
            except Fault as error:
                yield canonical({'requestId':request_id,'releaseId':prepared['releaseId'],'done':True,'finishReason':'error','error':{'code':error.code}})+'\n'

    def activate_release(self,identity,expected=None):
        release=self.store.get('release',identity)
        from .schemas import FOUNDATION_ORIGINS
        require(release.get('origin') in FOUNDATION_ORIGINS,'FOUNDATION_ORIGIN_REQUIRED')
        slot=self.store.get('slot','champion',{})
        return self.control.releases.activate(identity,slot.get('version',0) if expected is None else expected)

    def growth(self, identity):
        self.store.get('foundation_project',identity)
        evaluations=[r for r in self.store.list('foundation_evaluation') if r.get('projectId')==identity]
        return {'items':evaluations,'runs':[r for r in self.store.list('foundation_run') if r.get('projectId')==identity]}

    def selective_revoke(self,identity,selectors,reason='source_policy_revoked'):
        return self.revoke(identity,{'selectors':selectors,'reason':reason})

    def revoke_teacher(self,artifact_ids,reason='teacher_policy_revoked'):
        """Registry withdrawal follows actual teacher weight ancestry, not grants alone."""
        affected=set(artifact_ids);projects=set()
        for approval in self.store.list('foundation_teacher_approval'):
            if approval.get('teacherArtifactId') in affected:
                self.store.put('foundation_teacher_approval',{'approved':False,'state':'revoked','revocationReason':reason},approval['id'])
        for checkpoint in self.store.list('foundation_checkpoint'):
            refs=set(checkpoint.get('teacherArtifactIds') or [])|{checkpoint.get('teacherArtifactId')}
            if refs&affected and checkpoint.get('projectId'):projects.add(checkpoint['projectId'])
        for pid in sorted(projects):
            project=self.store.get('foundation_project',pid,{})
            if project and project.get('state')!='revoked':
                self.revoke(pid,{'selectors':{'teacherArtifactIds':sorted(affected)},'reason':reason})
        return sorted(projects)

    def revoke(self, identity, body):
        existing=self.store.get('foundation_project',identity)
        if existing.get('state')=='revoked':return {**existing,'affectedProjectIds':[identity],'relearningPlanIds':[]}
        supplied=str(body.get('reason','source_policy_revoked'))
        reason=supplied if re.fullmatch(r'[A-Za-z0-9_.:-]{1,100}',supplied) else 'operator_revocation'
        selected=body.get('selectors') or {'allProjectData':True}
        projects={identity};artifact_ids=set();checkpoint_ids=set()
        known_projects={row['id'] for row in self.store.list('foundation_project')}
        # Growth and teacher-derived weights inherit their source's withdrawal.
        # A separately proven clean relearning project has no sourceProjectId link.
        changed=True
        while changed:
            before=(len(projects),len(artifact_ids),len(checkpoint_ids))
            for project in self.store.list('foundation_project'):
                if project.get('sourceProjectId') in projects:projects.add(project['id'])
            for checkpoint in self.store.list('foundation_checkpoint'):
                if checkpoint.get('projectId') in projects:checkpoint_ids.add(checkpoint['id'])
                refs=set(checkpoint.get('teacherArtifactIds') or [])|{checkpoint.get('teacherArtifactId')}
                if refs&(artifact_ids|checkpoint_ids) and checkpoint.get('projectId') in known_projects:projects.add(checkpoint['projectId'])
            for artifact in self.store.list('artifact'):
                if artifact.get('foundationProjectId') in projects or artifact.get('sourceProjectId') in projects or set(artifact.get('parents',[]))&artifact_ids:
                    artifact_ids.add(artifact['id'])
                    if artifact.get('foundationProjectId') in known_projects:projects.add(artifact['foundationProjectId'])
            changed=before!=(len(projects),len(artifact_ids),len(checkpoint_ids))
        enabled={pid:self.store.get('foundation_policy',pid,{}).get('enabled') is True for pid in projects}
        previous={pid:self.store.get('foundation_project',pid) for pid in projects};plans=[]
        with self.store.tx():
            for pid in projects:self.store.put('foundation_project',{'revocationPending':True,'reasonCodes':['POLICY_REVOKED']},pid)
        # Proof failures must never prevent the actual safety withdrawal.
        from .relearning import plan as plan_relearning
        for pid in sorted(projects):
            if previous[pid].get('state')=='revoked':continue
            try:
                planned=plan_relearning(self.control,pid,selected)
                planned=self.store.put('foundation_relearning_plan',{'autoStart':bool(enabled[pid] and body.get('retrain',True) is True)},planned['id'])
                plans.append(planned['id'])
            except Exception as error:
                self.store.event('foundation.relearning.held',pid,{'reasonCode':getattr(error,'code','RELEARNING_PROOF_UNAVAILABLE')})
        kinds=('foundation_corpus','foundation_tokenizer','foundation_genesis','foundation_checkpoint','foundation_profile',
               'foundation_release','foundation_certificate','release','certificate','foundation_growth_candidate','foundation_growth_family',
               'foundation_domain_certificate','foundation_domain_evaluation','foundation_self_code_eligibility','foundation_teacher_approval')
        kinds=(*kinds,'foundation_teacher_corpus','foundation_teacher_policy','foundation_response_distillation')
        with self.store.tx():
            for pid in projects:
                self.store.put('foundation_project',{'state':'revoked','stage':'revoked','revocationPending':False,'reasonCodes':['POLICY_REVOKED']},pid)
                self.store.put('foundation_policy',{'enabled':False},pid)
            for kind in kinds:
                for record in self.store.list(kind):
                    if (record.get('projectId',record.get('foundationProjectId')) in projects or record.get('sourceProjectId') in projects
                            or record.get('artifactId') in artifact_ids
                            or (set(record.get('teacherArtifactIds') or [])|{record.get('teacherArtifactId')})&(artifact_ids|checkpoint_ids)):
                        fields={'state':'revoked','revocationReason':reason}
                        if kind=='foundation_teacher_approval':fields['approved']=False
                        self.store.put(kind,fields,record['id'])
            for artifact_id in artifact_ids:self.control.registry.revoke(artifact_id,reason)
            for pid in projects:
                for job in self._pending(pid):
                    self.store.db.execute("UPDATE jobs SET state=? WHERE id=?",('cancelling' if job['state'] in ('running','pause_requested','cancelling') else 'cancelled',job['id']))
            for planned in self.store.list('foundation_relearning_plan'):
                if planned.get('newProjectId') in projects and planned.get('state')!='completed':
                    self.store.put('foundation_relearning_plan',{'state':'held','reasonCodes':['POLICY_REVOKED']},planned['id'])
        while self.store.get('self_code_slot','document',{}):
            slot=self.store.get('self_code_slot','document')
            proposal=self.store.get('self_code_proposal',slot['proposalId'])
            model=self.store.get('artifact',proposal.get('modelArtifactId') or '',{})
            if model.get('state')!='revoked':break
            try:self.control.evolution.rollback()
            except Fault:
                self.store.put('policy',{'enabled':False},'self-code')
                self.store.event('foundation.self_code.rollback_held',identity,{'reasonCode':'SELF_CODE_VERSION_CONFLICT'})
                break
        self.runtime.expire(force=True)
        for pid in projects:
            try:self._recover_service(pid)
            except Exception as error:self.store.event('foundation.service.recovery_held',pid,{'reasonCode':getattr(error,'code','NO_CLEAN_RELEASE')})
        try:
            from LLM.storage.lifecycle import revoke_backups
            backup_result=revoke_backups(self.store,{'projectIds':sorted(projects),'artifactIds':sorted(artifact_ids)})
            if backup_result.get('state')=='held':self.store.event('foundation.backup.revocation_held',identity,{'reasonCodes':backup_result.get('reasonCodes',[])})
        except Exception as error:self.store.event('foundation.backup.revocation_held',identity,{'reasonCode':getattr(error,'code','BACKUP_DELETION_HELD')})
        for plan_id in plans:
            planned=self.store.get('foundation_relearning_plan',plan_id)
            if planned.get('autoStart'):
                try:
                    from .relearning import start
                    start(self.control,plan_id)
                except Exception as error:self.store.put('foundation_relearning_plan',{'state':'held','reasonCodes':[getattr(error,'code','RELEARNING_NOT_READY')]},plan_id)
        self.store.event('foundation.policy.revoked',identity,{'reason':'POLICY_REVOKED','affectedProjectIds':sorted(projects)})
        return {**self.store.get('foundation_project',identity),'affectedProjectIds':sorted(projects),'relearningPlanIds':plans}

    def _recover_service(self, revoked_project):
        bridge=self.control.bridge
        if bridge is None:return
        slot=self.store.get('slot','champion',{})
        active=self.store.get('release',slot.get('activeReleaseId') or '',{})
        if active.get('projectId',active.get('foundationProjectId'))!=revoked_project:return
        candidates=sorted(self.store.list('release'),key=lambda r:r.get('activatedAt',''),reverse=True)
        for candidate in candidates:
            if candidate.get('state') not in ('retired','ready') or not candidate.get('activatedAt'):continue
            try:
                artifact=self.store.get('artifact',candidate['artifactId']);self.control.registry.verify(artifact)
                cert=self.store.get('certificate',candidate['certificateId'])
                require(cert.get('passed') and cert.get('state')!='revoked','RELEASE_NOT_CERTIFIED')
                self.control.releases.activate(candidate['id'],slot.get('version',0));return
            except Fault:continue
        with bridge.deployment_lock,bridge.store.tx():
            settings=bridge.store.setting('settings',{})
            bridge.store.set_setting('settings',{**settings,'model':'','activeReleaseId':None,'version':settings.get('version',0)+1})
            bridge.post.generation_profile=None;bridge.post.model_status_cache=None;bridge.post.tokenizer_cache=None
        self.store.put('slot',{'activeReleaseId':None,'previousReleaseId':None,'reasonCodes':['NO_CLEAN_RELEASE']},'champion')
        self.store.event('foundation.service.blocked',revoked_project,{'reasonCode':'NO_CLEAN_RELEASE'})

    def dispatch(self, path, method, body, key=None, expected=None):
        bits=[b for b in path.strip('/').split('/') if b]
        if bits==['projects']:
            require(method in ('GET','POST'),'METHOD_NOT_ALLOWED',status=405)
            return self.create_project(body) if method=='POST' else {'items':[self.status(p['id']) for p in self.store.list('foundation_project')]}
        if len(bits)==2 and bits[0]=='projects' and method=='GET':return self.status(bits[1])
        if len(bits)==3 and bits[0]=='projects':
            identity,action=bits[1:]
            if action=='policy':
                require(method in ('GET','PATCH'),'METHOD_NOT_ALLOWED',status=405)
                return self.configure(identity,body,int(expected.strip('"')) if expected else body.get('expectedVersion')) if method=='PATCH' else self._project(identity)['policy']
            if action=='growth' and method=='GET':return self.growth(identity)
            if action=='acceptance' and method=='GET':
                from .acceptance import assess
                return assess(self.control,identity)
            if action=='curriculum' and method=='GET':return self.curriculum(identity)
            if action=='teacher-policy' and method=='GET':return self.store.get('foundation_teacher_policy',identity,{})
            if action=='growth-policy' and method=='GET':return self.store.get('foundation_growth_policy',identity,{})
            require(method=='POST','METHOD_NOT_ALLOWED',status=405)
            if action=='corpora':return self.create_corpus(identity,body,key)
            if action=='tokenizers':return self.start_tokenizer(identity,body,key)
            if action=='initializations':return self.initialize(identity,body,key)
            if action=='calibrate':return self.calibrate(identity,body,key)
            if action=='curriculum':return self.curriculum(identity,body)
            if action=='distillations':return self.start_distillation(identity,body,key)
            if action=='response-distillations':return self.start_response_distillation(identity,body,key)
            if action=='teacher-policy':
                from .response_distillation import configure_policy
                return configure_policy(self.control,identity,body)
            if action=='formats':
                return self.enqueue('foundation_format',identity,{
                    'artifactId':body.get('artifactId'),'checkpointId':body.get('checkpointId') or self._project(identity).get('latestCheckpointId'),
                    'quantization':body.get('quantization','F16')},key)
            if action=='growth-experiments':return self.start_growth(identity,body,key)
            if action=='growth-adoptions':return self.adopt_growth(identity,body)
            if action=='growth-policy':
                from .growth import configure_policy
                return configure_policy(self.control,identity,body)
            if action=='teachers':
                from .distillation import approve_teacher
                return approve_teacher(self.control,identity,body)
            if action=='relearning-plans':
                from .relearning import plan
                return plan(self.control,identity,body.get('selectors'))
            if action=='retention':
                from .retention import enforce
                return enforce(self.store,self._project(identity),dry_run=body.get('apply') is not True)
            if action in ('pretraining-runs','cpt-runs','sft-runs'):return self.start_training(identity,body,{'pretraining-runs':'pretrain','cpt-runs':'cpt','sft-runs':'sft'}[action],key)
            if action=='revoke':return self.revoke(identity,body)
        if len(bits)==3 and bits[0]=='runs':
            if bits[2]=='checkpoints' and method=='GET':return {'items':[r for r in self.store.list('foundation_checkpoint') if r.get('runId')==bits[1]]}
            require(method=='POST','METHOD_NOT_ALLOWED',status=405);return self.command(bits[1],bits[2])
        mapping={'corpora':'foundation_corpus','tokenizers':'foundation_tokenizer','initializations':'foundation_genesis','runs':'foundation_run','evaluations':'foundation_evaluation','profiles':'foundation_profile',
                 'domain-evaluations':'foundation_domain_evaluation','domain-certificates':'foundation_domain_certificate','domain-suites':'foundation_domain_suite',
                 'relearning-plans':'foundation_relearning_plan','audits':'foundation_audit_feedback'}
        if len(bits)==1 and bits[0] in mapping and method=='GET':return {'items':self.store.list(mapping[bits[0]])}
        if len(bits)==2 and bits[0] in mapping and method=='GET':return self.run_status(bits[1]) if bits[0]=='runs' else self.store.get(mapping[bits[0]],bits[1])
        if bits==['generate'] and method=='POST':
            self.runtime.prepare(body)
            from fastapi.responses import StreamingResponse
            return StreamingResponse(self.stream(body),media_type='application/x-ndjson')
        if bits==['evaluations']:return self.evaluate(body,key) if method=='POST' else {'items':self.store.list('foundation_evaluation')}
        if bits==['domain-evaluations']:
            require(method=='POST','METHOD_NOT_ALLOWED',status=405)
            require(self.store.get('foundation_domain_suite',body.get('suiteId',''),{}),'DOMAIN_SUITE_REQUIRED')
            return self.evaluate(body,key)
        if bits==['audits']:
            require(method=='POST','METHOD_NOT_ALLOWED',status=405)
            suite=self.store.get('foundation_domain_suite',body.get('suiteId',''),{}) or self.store.get('foundation_suite',body.get('suiteId',''),{})
            require(suite.get('kind')=='audit','AUDIT_SUITE_REQUIRED')
            return self.evaluate({**body,'audit':True},key)
        if bits==['acceptance'] and method=='GET':
            from .acceptance import assess
            return assess(self.control)
        if bits==['domains']:
            if method=='GET':return {'items':self.store.list('foundation_domain')}
            require(method=='POST','METHOD_NOT_ALLOWED',status=405)
            from .domains import register_domain
            return register_domain(self.store,body)
        if bits==['domain-suites']:
            if method=='GET':return {'items':self.store.list('foundation_domain_suite')}
            require(method=='POST','METHOD_NOT_ALLOWED',status=405)
            from .domains import register_suite
            return register_suite(self.control,body)
        if bits==['domain-certificates']:
            if method=='GET':return {'items':self.store.list('foundation_domain_certificate')}
            require(method=='POST','METHOD_NOT_ALLOWED',status=405)
            from .domains import certify
            return certify(self.control,body['evaluationId'])
        if len(bits)==3 and bits[0]=='relearning-plans' and bits[2]=='start' and method=='POST':
            from .relearning import start
            return start(self.control,bits[1])
        if len(bits)==3 and bits[0]=='relearning-plans' and bits[2]=='cancel' and method=='POST':
            value=self.store.get('foundation_relearning_plan',bits[1])
            require(value.get('state') not in ('completed','cancelled'),'JOB_ALREADY_FINISHED',status=409)
            if value.get('newProjectId'):
                self.store.put('foundation_policy',{'enabled':False},value['newProjectId'])
                for job in self._pending(value['newProjectId']):
                    self.store.db.execute('UPDATE jobs SET state=? WHERE id=?',('cancelling' if job['state'] in ('running','pause_requested','cancelling') else 'cancelled',job['id']))
            return self.store.put('foundation_relearning_plan',{'state':'cancelled','autoStart':False,'reasonCodes':['USER_CANCELLED']},value['id'])
        if bits==['suites']:return self.evaluator.register_suite(body) if method=='POST' else {'items':self.store.list('foundation_suite')}
        if bits==['self-code','suites']:
            if method=='GET':return {'items':self.store.list('foundation_self_code_suite')}
            require(method=='POST','METHOD_NOT_ALLOWED',status=405)
            from .selfcode import register_suite
            return register_suite(self.control,body)
        if bits==['self-code','evaluations']:
            if method=='GET':return {'items':self.store.list('foundation_self_code_eligibility')}
            require(method=='POST','METHOD_NOT_ALLOWED',status=405)
            return self.enqueue('foundation_self_code_evaluate',body['projectId'],body,key)
        if bits==['graduations'] and method=='POST':return self.graduate(body)
        if bits==['release-candidates'] and method=='POST':return self.prepare_release(body)
        if len(bits)==3 and bits[0]=='releases' and bits[2]=='activate' and method=='POST':return self.activate_release(bits[1],int(expected.strip('"')) if expected else None)
        raise Fault('ENDPOINT_NOT_FOUND',status=404)
