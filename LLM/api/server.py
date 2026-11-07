import asyncio
import json
import secrets
from contextlib import asynccontextmanager
from fastapi import FastAPI,Request
from fastapi.responses import JSONResponse,StreamingResponse
from starlette.concurrency import run_in_threadpool
from LLM.core import Fault,require,uid
from LLM.service import Control
from LLM.workers.transport import Transport

def create_app(root,*,token=None,local=False,worker=False,allowed_roots=(),control=None):
    require(bool(token) or local,'SERVICE_AUTH_REQUIRED')
    control=control or Control(root,allowed_roots,worker)
    @asynccontextmanager
    async def lifespan(app):
        control.start()
        try:yield
        finally:control.stop()
    app=FastAPI(title='SEED LLM Control',version='1.0.0',lifespan=lifespan)
    app.state.control=control
    transport=Transport(control)

    @app.exception_handler(Fault)
    async def fault(request,exc):return JSONResponse({'error':{'code':exc.code,'message':exc.message,'details':exc.details,'requestId':uid(),'retryable':exc.status==503}},status_code=exc.status)

    @app.middleware('http')
    async def boundary(request,call_next):
        if request.url.path!='/health/live':
            supplied=request.headers.get('authorization','').removeprefix('Bearer ')
            trusted=token and secrets.compare_digest(supplied,token)
            loopback=local and request.client and request.client.host in ('127.0.0.1','::1','testclient')
            worker_identity=transport.authorize(supplied) if supplied else None
            if worker_identity:
                if not request.url.path.startswith('/internal/v1/workers/'+worker_identity['id']+'/'):
                    return JSONResponse({'error':{'code':'WORKER_SCOPE_DENIED'}},status_code=403)
                request.state.worker_id=worker_identity['id']
            elif not (trusted or loopback):return JSONResponse({'error':{'code':'SERVICE_AUTH_REQUIRED'}},status_code=401)
        limit=10*1024**3 if '/outputs/' in request.url.path else 1024**2
        if int(request.headers.get('content-length','0') or 0)>limit:return JSONResponse({'error':{'code':'REQUEST_TOO_LARGE'}},status_code=413)
        return await call_next(request)

    @app.get('/health/live')
    def live():return {'status':'alive'}
    @app.get('/api/tags')
    def tags():return control.inference.tags()
    @app.post('/api/generate')
    async def generate(request:Request):return StreamingResponse(control.inference.stream(await request.json()),media_type='application/x-ndjson')
    @app.get('/api/training/status')
    def status():return control.training_status()
    @app.post('/api/training/jobs',status_code=202)
    async def train(request:Request):return control.start_training(await request.json(),request.headers.get('idempotency-key'))
    @app.post('/api/training/jobs/{identity}/{action}')
    def command(identity:str,action:str):return control.command(identity,action)

    @app.post('/internal/v1/foundation/generate')
    async def foundation_generate(request:Request):
        body=await request.json()
        await run_in_threadpool(control.foundation.runtime.prepare,body)
        return StreamingResponse(control.foundation.stream(body),media_type='application/x-ndjson')

    @app.post('/internal/v1/workers/register')
    async def register(request:Request):return transport.register(await request.json())
    @app.post('/internal/v1/workers/{identity}/{action}')
    async def remote_action(identity:str,action:str,request:Request):
        body=await request.json()
        if action=='claim':return transport.assignment(identity)
        if action=='heartbeat':
            job=transport.check(identity,body['jobId']);require(job['fence']==body['fencingToken'],'LEASE_EXPIRED',status=409)
            control.store.heartbeat(job)
            telemetry=body.get('trainingTelemetry')
            if telemetry and job['type']=='train':
                done,total=telemetry.get('completedSteps'),telemetry.get('totalSteps')
                require(isinstance(done,int) and isinstance(total,int) and 0<=done<=total,'INVALID_TRAINER_RESPONSE')
                previous=control.store.get('training_run',job['id'],{})
                require(done>=previous.get('completedSteps',0),'INVALID_TRAINER_RESPONSE')
                control.store.put('training_run',{**job['payload'],'completedSteps':done,'totalSteps':total,'state':telemetry.get('state','running'),
                    'loss':telemetry.get('loss'),'epoch':telemetry.get('epoch')},job['id'])
            from LLM.core import now
            control.store.put('worker',{'state':'online','observedAt':now()},identity)
            return {'state':job['state'],'fencingToken':job['fence']}
        if action=='complete':return transport.complete(identity,body['jobId'],body)
        raise Fault('INVALID_WORKER_ACTION')
    @app.get('/internal/v1/workers/{identity}/files/{job_id}/{kind}/{artifact_id}/{name:path}')
    def input_file(identity:str,job_id:str,kind:str,artifact_id:str,name:str):
        from fastapi.responses import FileResponse
        require(kind in ('artifact','snapshot'),'INVALID_FILE_KIND')
        return FileResponse(transport.input_file(identity,job_id,kind,artifact_id,name))
    @app.put('/internal/v1/workers/{identity}/outputs/{job_id}/{artifact_id}/{name}')
    async def output_file(identity:str,job_id:str,artifact_id:str,name:str,request:Request):
        from LLM.core import file_hash
        target=transport.output_path(identity,job_id,artifact_id,name);temporary=target.with_suffix(target.suffix+'.partial');size=0
        with temporary.open('wb') as handle:
            async for chunk in request.stream():
                size+=len(chunk);require(size<=10*1024**3,'ARTIFACT_TOO_LARGE',status=413);handle.write(chunk)
        transport.check(identity,job_id);temporary.replace(target)
        return {'sha256':file_hash(target),'bytes':size}

    @app.api_route('/internal/v1/{path:path}',methods=['GET','POST','PATCH'])
    async def internal(path:str,request:Request):
        body=await request.json() if request.method!='GET' else {}
        result=await run_in_threadpool(dispatch,control,path,request.method,body,request.headers.get('idempotency-key'),request.headers.get('if-match'))
        return foundation_http_response(path,request.method,result)
    return app

def foundation_http_response(path,method,result):
    """Apply additive foundation status codes after durable JSON idempotency."""
    if method!='POST' or not path.startswith('foundation/') or not isinstance(result,dict):return result
    if str(result.get('type','')).startswith('foundation_') and result.get('state')=='queued':
        return JSONResponse(result,status_code=202)
    if path in ('foundation/projects','foundation/suites','foundation/self-code/suites',
                'foundation/graduations','foundation/release-candidates'):
        return JSONResponse(result,status_code=201)
    return result

def dispatch(c,path,method,body,key=None,expected=None):
    from LLM.core import canonical,digest
    if method!='GET':require(not c.store.get('setting','deploymentMaintenance',{}).get('enabled'),'DEPLOYMENT_MAINTENANCE',status=503)
    if path=='foundation/generate':
        require(method=='POST','METHOD_NOT_ALLOWED',status=405)
        return c.foundation.dispatch('generate',method,body,key,expected)
    commands={'learning/run','learning/pause','learning/resume','combinations/preflight','certifications','adapter-compressions','garbage-collections','deployments/champion/rollback','curation/run','curation/bind','self-code/run','self-code/setup','self-code/rollback','market-baselines'}
    action=path.split('/')[-1]
    if path in commands or action in ('activate','build','cancel','pause','resume','select'):
        require(method=='POST','METHOD_NOT_ALLOWED',status=405)
    if method=='PATCH':require(expected is not None,'VERSION_REQUIRED',status=428)
    if path.endswith('/activate') or path=='deployments/champion/rollback':require(expected is not None,'VERSION_REQUIRED',status=428)
    if expected is not None:require(expected.strip('"').isdigit(),'INVALID_VERSION')
    if method!='POST':return _dispatch(c,path,method,body,key,expected)
    require(key,'IDEMPOTENCY_KEY_REQUIRED')
    identity=path+':'+key;request_hash=digest(body)
    with c.store.tx():
        old=c.store.rows('SELECT request_hash,result FROM idempotency WHERE key=?',(identity,))
        if old:
            require(old[0]['request_hash']==request_hash,'IDEMPOTENCY_CONFLICT',status=409)
            require(old[0]['result'] is not None,'REQUEST_IN_PROGRESS',status=409)
            return json.loads(old[0]['result'])
        c.store.db.execute('INSERT INTO idempotency VALUES(?,?,NULL)',(identity,request_hash))
    try:result=_dispatch(c,path,method,body,key,expected)
    except BaseException:
        with c.store.tx():c.store.db.execute('DELETE FROM idempotency WHERE key=?',(identity,))
        raise
    with c.store.tx():c.store.db.execute('UPDATE idempotency SET result=? WHERE key=?',(canonical(result),identity))
    return result

def _dispatch(c,path,method,body,key=None,expected=None):
    if path=='foundation' or path.startswith('foundation/'):
        return c.foundation.dispatch(path.removeprefix('foundation').lstrip('/'),method,body,key,expected)
    if method in ('POST','PATCH') and method=='POST':require(key,'IDEMPOTENCY_KEY_REQUIRED')
    if path=='capabilities':return c.capabilities()
    if path=='resource-policy':
        from LLM.workers.resources import budget,configure
        if method=='PATCH':return configure(c.store,body,int(expected.strip('"')))
        return {**c.store.get('setting','resourcePolicy',{'version':0,'mode':'adaptive'}),'effective':budget(c.store)}
    if path=='curation':return {'items':c.store.list('document_curation'),'nextCursor':None}
    if path=='curation/routes':
        if method=='GET':return {'items':c.store.list('source_purpose'),'nextCursor':None}
        require(c.bridge is not None,'RAG_POLICY_GATEWAY_REQUIRED');c.bridge.store.get('source',body['sourceId'])
        require(body.get('specialization') in ('code','review','test','refactor'),'INVALID_SOURCE_SPECIALIZATION')
        return c.store.put('source_purpose',{'specialization':body['specialization']},body['sourceId'])
    if path=='curation/run':return c.store.enqueue('curate',{'policyVersion':c.scheduler.policy()['version']},key)
    if path=='curation/bind':
        require(c.bridge is not None,'RAG_POLICY_GATEWAY_REQUIRED')
        c.bridge.pre.content(body['versionId'])
        require(isinstance(body.get('fixtureIds'),list) and 1<=len(body['fixtureIds'])<=20,'INVALID_VERIFIER_BINDING')
        for identity in body['fixtureIds']:c.store.get('fixture',identity)
        return c.store.put('document_verification',{'fixtureIds':body['fixtureIds']},body['versionId'])
    if path=='self-code/policy':return c.evolution.configure(body,int(expected.strip('"'))) if method=='PATCH' else c.evolution.policy()
    if path=='self-code':return {'policy':c.evolution.policy(),'status':c.store.get('setting','selfCodeStatus',{}),'items':c.store.list('self_code_proposal'),'target':'LLM/extensions/document_policy.py'}
    if path=='self-code/run':return c.evolution.enqueue(body,key)
    if path=='self-code/setup':return c.evolution.setup(body['profileId'])
    if path=='self-code/rollback':return c.evolution.rollback()
    if path=='progress':return {'items':c.evaluator.progress(),'nextCursor':None}
    if path=='market-baselines':
        family=c.store.get('family',body['familyId']);artifact=c.store.get('artifact',body['artifactId'])
        require(not artifact.get('fixture') and artifact['kind']!='adapter','MARKET_BASELINE_REQUIRED')
        c.registry.verify(artifact)
        return c.store.put('family',{'marketBaselineIds':sorted(set([*family.get('marketBaselineIds',[]),artifact['id']]))},family['id'])
    if path=='models':return c.registry.register(body) if method=='POST' else {'items':c.store.list('artifact'),'nextCursor':None}
    if path=='families':return {'items':c.store.list('family'),'nextCursor':None}
    if path=='branches':return c.registry.branch(body) if method=='POST' else {'items':c.store.list('branch'),'nextCursor':None}
    if path=='combinations/preflight':return c.merger.preflight(body)
    if path=='combinations':return c.merger.recipe(body) if method=='POST' else {'items':c.store.list('combination'),'nextCursor':None}
    if path=='recipe-searches':return c.recipe_search.create(body) if method=='POST' else {'items':c.store.list('recipe_search'),'nextCursor':None}
    if path=='adapter-compressions':return c.store.enqueue('compress',body,key)
    if path=='garbage-collections':
        from LLM.storage.retention import collect
        return collect(c.store,apply=body.get('apply') is True)
    bits=path.split('/')
    if len(bits)==3 and bits[0]=='combinations' and bits[2]=='build':
        require(method=='POST','METHOD_NOT_ALLOWED',status=405)
        combo=c.store.get('combination',bits[1])
        c.store.put('setting',{'combinationId':combo['id']},'selectedCombination')
        return c.store.enqueue('merge',{'combinationId':bits[1]},key='build:'+bits[1])
    if path=='learning/policy':return c.scheduler.configure(body,int(expected.strip('"')) if expected else None) if method=='PATCH' else c.scheduler.policy()
    if path=='learning/status':return {'policy':c.scheduler.policy(),'status':c.store.get('setting','learningStatus',{}),'cycles':c.store.list('cycle')}
    if path=='learning/run':return c.scheduler.check(manual=True)
    if path in ('learning/pause','learning/resume'):return c.scheduler.configure({'enabled':path.endswith('resume')})
    if path=='evaluations':return c.store.enqueue('evaluate',body,key) if method=='POST' else {'items':c.store.list('evaluation'),'nextCursor':None}
    if path=='distillations':return c.store.enqueue('distill',body,key) if method=='POST' else {'items':c.store.list('distillation'),'nextCursor':None}
    if path=='certifications':return c.evaluator.certify(body)
    if path=='certification-runs':return c.request_certification(body,key) if method=='POST' else {'items':c.store.list('certification_request'),'nextCursor':None}
    if path=='backups':
        from LLM.storage import backup
        return backup(c.store) if method=='POST' else {'items':c.store.list('backup'),'nextCursor':None}
    if path=='suites':return c.evaluator.register_suite(body) if method=='POST' else {'items':c.store.list('suite'),'nextCursor':None}
    if path=='releases':return c.releases.prepare(body) if method=='POST' else {'items':c.store.list('release'),'nextCursor':None}
    if path=='deployments/champion':return c.store.get('slot','champion',{'version':0,'activeReleaseId':None})
    if len(bits)==3 and bits[0]=='releases' and bits[2]=='activate':return c.releases.activate(bits[1],int(expected.strip('"')))
    if path=='deployments/champion/rollback':return c.releases.rollback(int(expected.strip('"')))
    if path=='workers':return {'items':[{k:v for k,v in w.items() if k!='tokenHash'} for w in c.store.list('worker')],'nextCursor':None}
    if path=='cycles':return {'items':c.store.list('cycle'),'nextCursor':None}
    if path=='snapshot-imports':
        require(c.bridge is not None,'RAG_SNAPSHOT_GATEWAY_REQUIRED',status=503)
        return c.bridge.import_snapshot(body['snapshotId'],body['familyId'])
    if len(bits)==3 and bits[0]=='snapshots' and bits[2]=='select':
        c.store.get('snapshot',bits[1]);return c.store.put('setting',{'snapshotId':bits[1]},'selectedSnapshot')
    mapping={'models':'artifact','branches':'branch','families':'family','combinations':'combination','evaluations':'evaluation','cycles':'cycle','releases':'release','snapshot-imports':'snapshot'}
    if len(bits)==2 and bits[0] in mapping:return c.store.get(mapping[bits[0]],bits[1])
    if len(bits)==2 and bits[0] in ('jobs','builds'):return c.store.job(bits[1])
    if len(bits)==3 and bits[0] in ('jobs','builds'):return c.command(bits[1],bits[2])
    raise Fault('ENDPOINT_NOT_FOUND',status=404)
