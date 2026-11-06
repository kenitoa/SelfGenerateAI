"""RAG-owned bridge. Source permissions remain authoritative in RAG."""
import ast
import json
import os
import threading
from pathlib import Path
from starlette.concurrency import run_in_threadpool
from fastapi import Request
from fastapi.responses import StreamingResponse
from LLM.core import Fault,require,digest,now,file_hash

class RagBridge:
    def __init__(self,control,rag_store,pre,post):
        self.c,self.store,self.pre,self.post=control,rag_store,pre,post
        self.last_policy=0
        self.deployment_lock=threading.RLock()
        original_run=post.run
        def pinned_run(run_id):
            with self.deployment_lock:return original_run(run_id)
        post.run=pinned_run

    def validate_snapshot(self,snapshot):
        current=self.post.snapshot(snapshot['id'])
        require(current['status']=='ready' and current['manifestHash']==snapshot['manifestHash'],'POLICY_REVOKED',status=409)
        return current

    def import_snapshot(self,identity,family_id):
        snapshot=self.post.snapshot(identity)
        require(snapshot['status']=='ready','SNAPSHOT_NOT_READY',status=409)
        root=(self.store.root/snapshot['storagePath']).resolve()
        require(root.is_relative_to((self.store.root/'exports').resolve()),'SNAPSHOT_PATH_INVALID',status=403)
        def validate(manifest):
            require(manifest['id']==identity,'SNAPSHOT_ID_MISMATCH')
            for e in manifest['examples']:
                record=self.store.get('training_example',e['id'])
                require(record['reviewStatus']=='approved' and not self.post._example_reasons(record),'POLICY_REVOKED',status=403)
        receipt=self.c.snapshots.ingest(root,snapshot['manifestHash'],family_id,self.store.setting('policyVersion',0),validate)
        metadata={e['id']:{'targetOrigin':self.store.get('training_example',e['id']).get('targetOrigin','human'),
            'objective':self.store.get('training_example',e['id']).get('objective','sft'),
            'branchId':self.store.get('training_example',e['id']).get('branchId')} for e in json.loads((root/'manifest.json').read_text('utf-8'))['examples']}
        receipt=self.c.store.put('snapshot',{'exampleMetadata':metadata},identity)
        self.c.store.put('setting',{'snapshotId':identity},'selectedSnapshot')
        self.store.put('training_snapshot',{'trainerVerified':True,'trainerReceipt':receipt},identity)
        self.store.set_setting('selectedTrainingSnapshot',identity)
        current=self.store.setting('settings',{})
        if not current.get('training',{}).get('endpoint'):
            self.store.set_setting('settings',{**current,'training':{**current.get('training',{}),'endpoint':'http://127.0.0.1:8100'},'version':current.get('version',0)+1})
        return receipt

    def sync_learning_policy(self,policy):
        current=self.store.setting('settings',{})
        if current.get('auto')!=policy.get('enabled'):
            self.store.set_setting('settings',{**current,'auto':bool(policy.get('enabled')),'version':current.get('version',0)+1})

    def refresh_policy(self):
        epoch=self.store.setting('policyVersion',0)
        for snapshot in self.c.store.list('snapshot'):
            if snapshot['state']!='verified':continue
            try:self.validate_snapshot(snapshot)
            except Exception:
                self.c.store.put('snapshot',{'state':'revoked'},snapshot['id'])
                for a in self.c.store.list('artifact'):
                    if a.get('snapshotId')==snapshot['id'] or snapshot['id'] in a.get('snapshotIds',[]):self.c.registry.revoke(a['id'],'snapshot_revoked')
        self.c.store.put('setting',{'epoch':epoch,'observedAt':now()},'policyEpoch');self.last_policy=epoch

    def produce_candidates(self,policy):
        from LLM.data.curator import Curator
        return Curator(self).run(policy)

    def activate_release(self,release):
        from tokenizers import Tokenizer
        profile=release['profile'];tokenizer=Path(release['tokenizerPath'])
        require(file_hash(tokenizer)==profile['tokenizerSha256'],'RELEASE_PROFILE_MISMATCH')
        parsed=Tokenizer.from_file(str(tokenizer))
        require(parsed.encode('<|im_start|>assistant\n',add_special_tokens=False).ids,'INVALID_TEMPLATE')
        with self.deployment_lock,self.post.lock,self.store.tx():
            current=self.store.setting('settings',{})
            self.store.set_setting('settings',{**current,'model':release['engineAlias'],'tokenizerPath':str(tokenizer),
                'endpoint':self.c.store.get('setting','inferenceEndpoint',{}).get('url','http://127.0.0.1:8100'),
                'contextTokens':profile['contextLength'],'activeReleaseId':release['id'],'version':current.get('version',0)+1})
            self.post.tokenizer_cache=None;self.post.generation_profile=profile;self.post.model_status_cache=None
        return {'activeReleaseId':release['id']}

    def active_release(self):
        return self.store.setting('settings',{}).get('activeReleaseId')

def attach(app,rag_store,pre,post):
    """Register before RAG's static mount; no model work occurs on import."""
    from LLM.service import Control
    from LLM.api.server import dispatch
    from RAG.core import DomainError
    root=Path(os.environ.get('SEED_LLM_DATA_DIR',str(Path(__file__).parent/'runtime-data')))
    allowed=[root/'imports',root/'staging']
    allowed.extend(Path(p) for p in os.environ.get('SEED_LLM_MODEL_ROOTS','').split(os.pathsep) if p)
    c=Control(root,allowed,worker=os.environ.get('SEED_LLM_WORKER','0')=='1')
    bridge=RagBridge(c,rag_store,pre,post);c.bridge=bridge;c.releases.rag=bridge;app.state.llm=c
    token_file=os.environ.get('SEED_LLM_TOKEN_FILE')
    original_engine=post._engine
    def authenticated_engine(endpoint=None,timeout=10):
        connection=original_engine(endpoint,timeout)
        target=endpoint or post._config().get('endpoint','')
        if token_file and target.rstrip('/')=='http://127.0.0.1:8100':
            secret=Path(token_file).read_text().strip();original_open=connection.open
            def authorized_open(method,path,payload=None,headers=None):
                return original_open(method,path,payload,{**(headers or {}),'Authorization':'Bearer '+secret})
            connection.open=authorized_open
        return connection
    post._engine=authenticated_engine
    def translated(callback):
        try:return callback()
        except Fault as exc:raise DomainError(exc.code,exc.message,exc.status,exc.details) from exc

    @app.api_route('/api/v1/llm/{path:path}',methods=['GET','POST','PATCH'])
    async def llm(path:str,request:Request):
        body=await request.json() if request.method!='GET' else {}
        return await run_in_threadpool(translated,lambda:dispatch(c,path,request.method,body,request.headers.get('idempotency-key'),request.headers.get('if-match')))

    @app.get('/api/v1/llm-events')
    def events(after:int=0):return {'items':c.store.rows('SELECT * FROM events WHERE sequence>? ORDER BY sequence LIMIT 100',(after,))}

    # The existing trainer gateway remains the public contract; selecting "embedded"
    # installs the in-process adapter without any browser access to internal ports.
    original_status,original_command=post.trainer_status,post.trainer_command
    def use_embedded():return rag_store.setting('settings',{}).get('training',{}).get('endpoint')=='embedded'
    def status():
        if not use_embedded():return original_status()
        selected=rag_store.setting('settings',{}).get('trainingSnapshotId') or rag_store.setting('selectedTrainingSnapshot',None)
        if selected and not c.store.get('snapshot',selected,{}):
            families=c.store.list('family')
            if len(families)==1:translated(lambda:bridge.import_snapshot(selected,families[0]['id']))
        if selected:c.store.put('setting',{'snapshotId':selected},'selectedSnapshot')
        return translated(c.training_status)
    def command(action,body=None,job_id=None):
        if not use_embedded():return original_command(action,body,job_id)
        return translated(lambda:c.start_training(body or {}) if action=='start' else c.command(job_id,action))
    post.trainer_status=status;post.trainer_command=command
    return c
