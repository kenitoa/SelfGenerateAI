"""The stable Interface <-> RAG contract; Pre and Post own the domain logic."""
from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from RAG.core import Store, DomainError, digest, now, uid
from RAG.operations import Runtime, system_status, resource_guard
from RAG.security import Auth


class BodyLimitMiddleware:
    def __init__(self, app):
        self.app=app

    async def __call__(self, scope, receive, send):
        if scope['type']!='http' or scope.get('method') in ('GET','HEAD','OPTIONS'):
            return await self.app(scope,receive,send)
        maximum=26*1024**2 if scope.get('path')=='/api/v1/uploads' else 1024**2
        received=0;started=False
        async def bounded_receive():
            nonlocal received
            message=await receive()
            received+=len(message.get('body',b''))
            if received>maximum:raise DomainError('REQUEST_TOO_LARGE','요청 본문 용량 상한을 초과했습니다.',413)
            return message
        async def tracked_send(message):
            nonlocal started
            if message['type']=='http.response.start':started=True
            await send(message)
        try:await self.app(scope,bounded_receive,tracked_send)
        except DomainError as exc:
            if started:raise
            response=JSONResponse({'error':{'code':exc.code,'message':exc.message,'retryable':False,'requestId':uid(),'details':{}}},status_code=exc.status)
            await response(scope,receive,send)


def create_app(data_dir=None, *, start_worker=True, local_dev=False, pre_service=None, post_service=None):
    from RAG.Pre.service import PreService
    from RAG.Post.service import PostService
    store=Store(data_dir or os.environ.get("SEED_DATA_DIR", str(Path(__file__).parent/"data")))
    pre=pre_service or PreService(store)
    post=post_service or PostService(store,pre)
    runtime=Runtime(store,pre,post)
    auth=Auth(store,local_dev=local_dev)

    @asynccontextmanager
    async def lifespan(app):
        if start_worker:runtime.start()
        try:yield
        finally:
            runtime.stop()
            if hasattr(app.state,'llm'):app.state.llm.stop()
            store.close()

    app=FastAPI(title="SEED RAG",version="1.0.0",lifespan=lifespan)
    app.state.store,app.state.pre,app.state.post=store,pre,post
    app.state.runtime,app.state.auth=runtime,auth

    def error_response(exc, request_id=None):
        response=JSONResponse({"error":{"code":exc.code,"message":exc.message,"requestId":request_id or uid(),"retryable":exc.status in (429,503),"details":exc.details}},status_code=exc.status)
        if exc.status==429:response.headers['Retry-After']='10'
        return response

    @app.exception_handler(DomainError)
    async def domain_error(request,exc):return error_response(exc,getattr(request.state,'request_id',None))

    @app.exception_handler(RequestValidationError)
    async def validation_error(request,exc):
        return error_response(DomainError("VALIDATION_ERROR","요청 형식을 확인하세요.",422,{"fields":[{"location":list(e['loc']),"message":e['msg']} for e in exc.errors()]}),getattr(request.state,'request_id',None))

    @app.exception_handler(json.JSONDecodeError)
    async def invalid_json(request,exc):
        return error_response(DomainError('INVALID_JSON','올바른 JSON 요청이 필요합니다.',422),getattr(request.state,'request_id',None))

    @app.middleware('http')
    async def boundary(request,call_next):
        request.state.request_id=uid()
        try:
            if request.method in ('POST','PATCH','PUT') and request.headers.get('content-type','').split(';')[0].strip()=='application/json':
                try:
                    payload=await request.json()
                    if not isinstance(payload,dict):raise DomainError('INVALID_JSON_OBJECT','JSON 객체 요청이 필요합니다.',422)
                    json.dumps(payload,allow_nan=False)
                except (json.JSONDecodeError,ValueError):raise DomainError('INVALID_JSON','올바른 JSON 요청이 필요합니다.',422)
            if request.url.path.startswith('/api/') and request.url.path not in ('/api/v1/auth/me','/api/v1/auth/login','/api/v1/health/live','/api/v1/health/ready'):
                request.state.session=auth.authorize(request)
            response=await call_next(request)
        except DomainError as exc:response=error_response(exc,request.state.request_id)
        response.headers['X-Request-ID']=request.state.request_id
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Referrer-Policy']='same-origin'
        response.headers['X-Frame-Options']='DENY'
        if request.url.path.startswith('/api/'):response.headers['Cache-Control']='no-store'
        return response

    def items(values,request):
        try:limit=min(100,max(1,int(request.query_params.get('limit',100))))
        except (ValueError,TypeError):raise DomainError('INVALID_LIMIT','목록 limit은 정수여야 합니다.',422)
        values=sorted(values,key=lambda v:(v.get('createdAt',''),str(v.get('id',''))),reverse=True)
        cursor=request.query_params.get('cursor')
        if cursor:
            positions=[i for i,v in enumerate(values) if str(v.get('id'))==cursor]
            if not positions:raise DomainError("INVALID_CURSOR","잘못된 페이지 위치입니다.",422)
            values=values[positions[0]+1:]
        selected=values[:limit]
        return {"items":selected,"nextCursor":str(selected[-1]['id']) if len(values)>limit else None}

    async def mutation(request,body,callback,status=200,require_key=True):
        key=request.headers.get('idempotency-key')
        if require_key and not key:raise DomainError("IDEMPOTENCY_KEY_REQUIRED","Idempotency-Key 헤더가 필요합니다.",422)
        if not key:return await run_in_threadpool(callback)
        if len(key)>128:raise DomainError("INVALID_IDEMPOTENCY_KEY","요청 키가 너무 깁니다.",422)
        identity=request.url.path+':'+key;request_hash=digest(body)
        with store.tx() as db:
            db.execute('DELETE FROM idempotency WHERE created_at<?',(time.time()-86400,))
            existing=db.execute('SELECT * FROM idempotency WHERE key=?',(identity,)).fetchone()
            if existing:
                if existing['request_hash']!=request_hash:raise DomainError("IDEMPOTENCY_CONFLICT","같은 요청 키의 내용이 다릅니다.",409)
                if existing['status']=='pending':raise DomainError("REQUEST_IN_PROGRESS","같은 요청을 처리하고 있습니다.",409)
                return JSONResponse(json.loads(existing['resource']),status_code=status)
            db.execute('INSERT INTO idempotency VALUES(?,?,?,?,?)',(identity,request_hash,'pending',None,time.time()))
        try:
            value=await run_in_threadpool(callback)
        except BaseException:
            with store.tx() as db:db.execute('DELETE FROM idempotency WHERE key=?',(identity,))
            raise
        with store.tx() as db:db.execute("UPDATE idempotency SET status='completed',resource=? WHERE key=?",(json.dumps(value,ensure_ascii=False),identity))
        return JSONResponse(value,status_code=status)

    @app.get('/api/v1/auth/me')
    def me(request:Request,response:Response):
        session=auth.session(request)
        if not session and auth.is_local(request) and not store.setting('admin_password_hash'):
            session=auth.create_session(response,local=True)
        return {"authenticated":bool(session),"bootstrapRequired":not bool(store.setting('admin_password_hash')),
                "localAccess":bool(session and session.get('local')),"csrfToken":session['csrfToken'] if session else None}

    @app.post('/api/v1/auth/login')
    async def login(request:Request,response:Response):
        body=await request.json()
        return auth.login(request,response,body.get('password',''))

    @app.post('/api/v1/auth/logout')
    def logout(request:Request,response:Response):
        store.delete('session',request.state.session['id']);response.delete_cookie('seed_session',path='/');return {"authenticated":False}

    @app.get('/api/v1/health/live')
    def live():return {"status":"alive","observedAt":now()}

    @app.get('/api/v1/health/ready')
    def ready():
        status=pre.readiness()
        valid=status.get('indexReady',False)
        return JSONResponse({"status":"ready" if valid else "not_ready","sqliteVersion":store.sqlite_version,**status},status_code=200 if valid else 503)

    @app.get('/api/v1/capabilities')
    def capabilities():
        readiness=pre.readiness();model=post.model_status()
        return {"ingestion":True,"lexicalSearch":True,"denseSearch":readiness.get('denseSearch',False),
                "generation":model.get('generationAvailable',False),"training":bool(store.setting('settings',{}).get('training',{}).get('endpoint')),"scheduling":True,
                "sourceSync":True,"backups":True,"trainingExport":True,"evaluation":True,"reasonCodes":model.get('reasonCodes',[model['reasonCode']] if model.get('reasonCode') else []),
                "observedAt":now(),"pre":readiness,"post":model}

    @app.get('/api/v1/system/status')
    def system():return system_status(store)

    @app.get('/api/v1/dashboard')
    def dashboard(period:str='week'):
        documents=pre.documents();readiness=pre.readiness()
        days=7 if period=='week' else 30
        cutoff=(datetime.now(timezone.utc)+timedelta(hours=9)-timedelta(days=days-1)).date().isoformat()
        series=sorted([m for m in store.list('metric_daily') if m.get('date','')>=cutoff],key=lambda m:m['date'])
        ready_docs=[d for d in documents if d.get('state')=='ready']
        model=post.model_status()
        counts={state:len([d for d in documents if d.get('state')==state]) for state in ('processing','needs_review','ready','failed')}
        trainer=store.setting('lastTrainerStatus')
        return {"observedAt":now(),"stale":False,"dataOrigin":"server",
                "metrics":{"collectedDocuments":len([d for d in documents if d.get('state') not in ('deleted','excluded')]),
                           "indexedChunks":sum(d.get('chunkCount',0) for d in ready_docs),
                           "approvedTrainingExamples":len([x for x in post.examples() if x.get('reviewStatus')=='approved']),
                           "completedTrainingRuns":len([r for r in trainer.get('runs',[]) if r.get('state')=='completed']) if trainer else None,"loadedModelParameters":model.get('parameters')},
                "dependencies":{"rag":"ready","inference":model.get('connection','disconnected'),"trainer":"connected" if trainer else "disconnected"},
                "searchMode":readiness.get('searchMode','lexical_only'),"series":series,
                "pipeline":{"queued":store.one("SELECT count(*) AS n FROM jobs WHERE state='queued'")['n'],"processing":counts['processing'],"needsReview":counts['needs_review'],"readyDocuments":counts['ready'],"failed":counts['failed']}}

    @app.get('/api/v1/activities')
    def activities(request:Request):
        rows=store.read("SELECT * FROM activities ORDER BY sequence DESC LIMIT 500")
        return items([{**r,'id':str(r['sequence']),'createdAt':r['created_at'],'entityId':r['entity_id']} for r in rows],request)

    @app.get('/api/v1/settings')
    def settings():return store.setting('settings',{})

    @app.patch('/api/v1/settings')
    async def update_settings(request:Request):
        body=await request.json()
        if not isinstance(body,dict):raise DomainError('INVALID_SETTINGS','설정은 JSON 객체여야 합니다.',422)
        def endpoint(value,empty=False):
            from urllib.parse import urlparse
            if empty and value=='':return
            if not isinstance(value,str) or len(value)>2000:raise DomainError('INVALID_ENDPOINT','올바른 엔진 주소가 필요합니다.',422)
            parsed=urlparse(value)
            try:parsed.port
            except ValueError:raise DomainError('INVALID_ENDPOINT','올바른 포트가 필요합니다.',422)
            if parsed.scheme not in ('http','https') or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:raise DomainError('INVALID_ENDPOINT','올바른 엔진 주소가 필요합니다.',422)
        for key in ('embedding','training','retrieval'):
            if key in body and not isinstance(body[key],dict):raise DomainError('INVALID_SETTINGS',f'{key}는 객체여야 합니다.',422)
        if 'memory' in body and str(body['memory']) not in ('1','2','4','8','16','24','32'):raise DomainError('INVALID_MEMORY','메모리 설정 범위를 확인하세요.',422)
        if 'auto' in body and not isinstance(body['auto'],bool):raise DomainError('INVALID_SETTINGS','auto는 boolean이어야 합니다.',422)
        if 'schedule' in body and (not isinstance(body['schedule'],str) or not re.fullmatch(r'(?:[01]\d|2[0-3]):[0-5]\d',body['schedule'])):raise DomainError('INVALID_SCHEDULE','예약 시각은 HH:MM 형식입니다.',422)
        for key in ('contextTokens','maxOutputTokens'):
            if key in body and (not isinstance(body[key],int) or isinstance(body[key],bool) or not 128<=body[key]<=131072):raise DomainError('INVALID_TOKEN_BUDGET','토큰 예산 범위를 확인하세요.',422)
        for key in ('tokenizerPath','model'):
            if key in body and (not isinstance(body[key],str) or len(body[key])>2000):raise DomainError('INVALID_SETTINGS',f'{key} 값을 확인하세요.',422)
        for key in ('allowedSourceHosts','allowedOrigins'):
            if key in body and (not isinstance(body[key],list) or len(body[key])>100 or any(not isinstance(v,str) or len(v)>2000 for v in body[key])):raise DomainError('INVALID_SETTINGS',f'{key}는 문자열 목록이어야 합니다.',422)
        training=body.get('training',{})
        if 'endpoint' in training:endpoint(training['endpoint'],empty=True)
        try:
            if 'epochs' in training and not 1<=int(training['epochs'])<=100:raise ValueError()
            if 'batch' in training and int(training['batch']) not in (1,2,4,8):raise ValueError()
            if 'rate' in training and not 0<float(training['rate'])<=1:raise ValueError()
        except (ValueError,TypeError):raise DomainError('INVALID_TRAINING_CONFIG','학습 설정 범위를 확인하세요.',422)
        embedding=body.get('embedding',{})
        if 'enabled' in embedding and not isinstance(embedding['enabled'],bool):raise DomainError('INVALID_EMBEDDING_CONFIG','임베딩 활성화 값이 올바르지 않습니다.',422)
        for key in ('directory','modelPath','tokenizerPath','revision'):
            if key in embedding and not isinstance(embedding[key],str):raise DomainError('INVALID_EMBEDDING_CONFIG','임베딩 경로·리비전을 확인하세요.',422)
        with store.tx():
            old=store.setting('settings',{})
            expected=request.headers.get('if-match','').strip('"')
            if not expected or expected!=str(old.get('version',1)):raise DomainError("SETTINGS_CONFLICT","설정을 새로고침한 뒤 저장하세요.",412)
            allowed={'name','endpoint','model','memory','schedule','auto','training','contextTokens','maxOutputTokens','tokenizerPath','embedding','allowedSourceHosts','retrieval','allowedOrigins','trainingSnapshotId'}
            unknown=set(body)-allowed-{'version'}
            if unknown:raise DomainError('UNKNOWN_SETTING','지원하지 않는 설정입니다.',422,{'fields':sorted(unknown)})
            for key in ('endpoint',):
                if key in body:endpoint(body[key])
            if 'name' in body and (not isinstance(body['name'],str) or not body['name'].strip() or len(body['name'])>24):raise DomainError('INVALID_NAME','이름은 1~24자입니다.',422)
            updated={**old,**{k:v for k,v in body.items() if k in allowed},'version':old.get('version',1)+1}
            if updated.get('maxOutputTokens',700)>=updated.get('contextTokens',4096):raise DomainError('INVALID_TOKEN_BUDGET','출력 예산은 전체 컨텍스트보다 작아야 합니다.',422)
            if body.get('trainingSnapshotId'):
                snapshot=post.snapshot(body['trainingSnapshotId'])
                if snapshot.get('status')!='ready':raise DomainError('SNAPSHOT_NOT_READY','준비된 학습 스냅샷을 선택하세요.',409)
            if 'auto' in body:
                if hasattr(app.state,'llm'):
                    from LLM.core import Fault
                    try:app.state.llm.scheduler.configure({'enabled':bool(body['auto'])})
                    except Fault as exc:raise DomainError(exc.code,exc.message,exc.status,exc.details) from exc
                elif body.get('auto'):
                    raise DomainError('TRAINER_SCHEDULER_UNAVAILABLE','자동 모델 학습은 LLM 통합 서버에서 설정하세요.',409)
            if body.get('trainingSnapshotId'):store.set_setting('selectedTrainingSnapshot',body['trainingSnapshotId'])
            if 'memory' in body and hasattr(app.state,'llm'):
                app.state.llm.store.put('setting',{'mode':'fixed','maxRssBytes':int(float(body['memory'])*1024**3)},'resourcePolicy')
            store.set_setting('settings',updated)
        store.activity('settings_updated','workspace','환경 설정 저장')
        return updated

    @app.post('/api/v1/uploads',status_code=202)
    async def upload(request:Request):
        resource_guard(store)
        length=request.headers.get('content-length')
        try:
            if length and int(length)>26*1024**2:raise DomainError('UPLOAD_TOO_LARGE','요청 전체 크기를 초과했습니다.',413)
        except ValueError:raise DomainError('INVALID_CONTENT_LENGTH','요청 길이가 올바르지 않습니다.',422)
        form=await request.form(max_files=10,max_fields=30,max_part_size=26*1024**2)
        files=[]
        total=0
        for file in form.getlist('files[]') or form.getlist('files'):
            if not hasattr(file,'read'):continue
            raw=await file.read(10*1024**2+1);await file.close();total+=len(raw)
            if len(raw)>10*1024**2 or total>25*1024**2:raise DomainError('UPLOAD_TOO_LARGE','파일 또는 요청 용량 상한을 초과했습니다.',413)
            files.append((file.filename,raw))
        if not files:raise DomainError('FILES_REQUIRED','파일을 선택하세요.',422)
        if len(files)>10:raise DomainError('TOO_MANY_FILES','한 번에 최대 10개 파일입니다.',422)
        body={'files':[(name,digest(raw)) for name,raw in files], 'uploadKeys':form.getlist('uploadKeys[]') or None, 'datasetId':form.get('datasetId'), 'mode':form.get('mode','replace_snapshot'),'sourceId':form.get('sourceId'),'consent':str(form.get('retrievalConsent','false')).lower()=='true'}
        return await mutation(request,body,lambda:pre.upload(files,dataset_id=body['datasetId'],source_id=body['sourceId'],mode=body['mode'],retrieval_consent=body['consent'],upload_keys=form.getlist('uploadKeys[]') or None),202)

    @app.get('/api/v1/datasets')
    def datasets(request:Request):return items(pre.list_datasets(),request)
    @app.post('/api/v1/datasets',status_code=201)
    async def new_dataset(request:Request):
        body=await request.json();return await mutation(request,body,lambda:pre.create_dataset(body),201)
    @app.get('/api/v1/datasets/{id}')
    def dataset(id:str):return pre.dataset(id)
    @app.patch('/api/v1/datasets/{id}')
    async def patch_dataset(id:str,request:Request):return pre.update_dataset(id,await request.json())
    @app.delete('/api/v1/datasets/{id}')
    def delete_dataset(id:str):return pre.delete_dataset(id)
    @app.get('/api/v1/datasets/{id}/documents')
    def dataset_documents(id:str,request:Request):return items(pre.documents(id),request)
    @app.get('/api/v1/documents')
    def documents(request:Request):return items(pre.documents(),request)
    @app.get('/api/v1/documents/{id}')
    def document(id:str):return pre.document(id)
    @app.get('/api/v1/documents/{id}/versions')
    def versions(id:str,request:Request):return items(pre.versions(id),request)
    @app.get('/api/v1/document-versions/{id}/content')
    def content(id:str):return Response(pre.content(id),media_type='text/plain; charset=utf-8',headers={'Content-Disposition':'inline','X-Content-Type-Options':'nosniff'})
    @app.get('/api/v1/document-versions/{id}/chunks')
    def chunks(id:str,request:Request):return items(pre.chunks(id),request)
    @app.post('/api/v1/documents/{id}/review')
    async def review(id:str,request:Request):
        body=await request.json();return await mutation(request,body,lambda:pre.review(id,body))
    @app.delete('/api/v1/documents/{id}')
    def remove_document(id:str):return pre.delete_document(id)

    @app.get('/api/v1/sources')
    def sources(request:Request):return items(pre.sources(),request)
    @app.post('/api/v1/sources',status_code=201)
    async def new_source(request:Request):
        body=await request.json();return await mutation(request,body,lambda:pre.create_source(body),201)
    @app.get('/api/v1/sources/{id}')
    def source(id:str):return store.get('source',id)
    @app.patch('/api/v1/sources/{id}')
    async def patch_source(id:str,request:Request):return pre.update_source(id,await request.json())
    @app.delete('/api/v1/sources/{id}')
    def delete_source(id:str,purgeDocuments:bool=False):return pre.delete_source(id,purge=purgeDocuments)
    @app.post('/api/v1/sources/{id}/sync',status_code=202)
    async def sync_source(id:str,request:Request):return await mutation(request,{},lambda:runtime.operations.sync_source(id),202)
    @app.get('/api/v1/ingestion-jobs')
    def jobs(request:Request):return items([store._job(r) for r in store.read("SELECT * FROM jobs ORDER BY created_at DESC LIMIT 500")],request)
    @app.get('/api/v1/ingestion-jobs/{id}')
    def job(id:str):return store.job(id)
    @app.post('/api/v1/ingestion-jobs/{id}/cancel')
    def cancel_job(id:str):return store.cancel_job(id)
    @app.post('/api/v1/ingestion-jobs/{id}/retry')
    def retry_job(id:str):return store.retry_job(id)
    @app.post('/api/v1/indexes/rebuild',status_code=202)
    async def rebuild(request:Request):
        body=await request.json();return await mutation(request,body,lambda:store.enqueue('reindex',body),202)
    @app.get('/api/v1/indexes')
    def indexes(request:Request):return items(pre.indexes(),request)
    @app.post('/api/v1/indexes/{id}/activate')
    def activate(id:str):return pre.activate_index(id)
    @app.post('/api/v1/retrieval/search')
    async def search(request:Request):return await run_in_threadpool(pre.search,await request.json())
    @app.get('/api/v1/retrieval-traces/{id}')
    def trace(id:str):return store.get('trace',id)

    @app.get('/api/v1/threads')
    def threads(request:Request):return items(post.threads(),request)
    @app.post('/api/v1/threads',status_code=201)
    async def thread_create(request:Request):
        body=await request.json();return await mutation(request,body,lambda:post.create_thread(body),201)
    @app.post('/api/v1/threads/import')
    async def import_threads(request:Request):
        body=await request.json();return await mutation(request,body,lambda:post.import_threads(body))
    @app.get('/api/v1/threads/{id}')
    def thread(id:str):return post.thread(id)
    @app.delete('/api/v1/threads/{id}')
    def delete_thread(id:str):return post.delete_thread(id)
    @app.post('/api/v1/threads/{id}/messages',status_code=202)
    async def message(id:str,request:Request):
        body=await request.json();return await mutation(request,body,lambda:post.start_message(id,body),202)
    @app.get('/api/v1/generation-runs/{id}')
    def run(id:str):return post.run(id)
    @app.get('/api/v1/generation-runs/{id}/events')
    async def events(id:str,request:Request):
        post.run(id)
        try:after=int(request.headers.get('last-event-id') or request.query_params.get('after',0))
        except ValueError:raise DomainError('INVALID_EVENT_CURSOR','잘못된 이벤트 위치입니다.',422)
        post.events(id,after)  # Validate cursor/retention before streaming headers.
        async def stream():
            cursor=after;last_ping=time.monotonic()
            post.subscribe(id,1)
            try:
                while True:
                    if await request.is_disconnected():break
                    batch=await run_in_threadpool(post.events,id,cursor)
                    for event in batch:
                        sequence=event.get('sequence',event.get('id',0))
                        cursor=int(sequence)
                        yield f"id: {sequence}\nevent: {event.get('event','message')}\ndata: {json.dumps(event.get('data',{}),ensure_ascii=False)}\n\n"
                    current=await run_in_threadpool(post.run,id)
                    if current.get('state',current.get('status')) in ('completed','failed','cancelled','insufficient_evidence'):
                        # Completion may race the preceding event read. Drain the
                        # committed terminal event before closing the subscription.
                        tail=await run_in_threadpool(post.events,id,cursor)
                        for event in tail:
                            sequence=event.get('sequence',event.get('id',0));cursor=int(sequence)
                            yield f"id: {sequence}\nevent: {event.get('event','message')}\ndata: {json.dumps(event.get('data',{}),ensure_ascii=False)}\n\n"
                        break
                    if time.monotonic()-last_ping>10:
                        yield ': heartbeat\n\n';last_ping=time.monotonic()
                    await asyncio.sleep(.15)
            finally:post.subscribe(id,-1)
        return StreamingResponse(stream(),media_type='text/event-stream',headers={'X-Accel-Buffering':'no','Cache-Control':'no-cache'})
    @app.post('/api/v1/generation-runs/{id}/cancel')
    def cancel_run(id:str):return post.cancel_run(id)
    @app.post('/api/v1/messages/{id}/feedback')
    async def feedback(id:str,request:Request):
        body=await request.json();return await mutation(request,body,lambda:post.feedback(id,body))

    @app.get('/api/v1/models')
    def models():return post.models()
    @app.post('/api/v1/connections/test')
    async def connection_test(request:Request):return await run_in_threadpool(post.test_connection,await request.json())
    @app.get('/api/v1/schedules')
    def schedules(request:Request):return items(runtime.operations.schedules(),request)
    @app.post('/api/v1/schedules',status_code=201)
    async def schedule(request:Request):
        body=await request.json();return await mutation(request,body,lambda:runtime.operations.save_schedule(body),201)
    @app.patch('/api/v1/schedules/{id}')
    async def edit_schedule(id:str,request:Request):
        old=store.get('schedule',id);return runtime.operations.save_schedule({**old,**await request.json()},id)
    @app.delete('/api/v1/schedules/{id}')
    def delete_schedule(id:str):store.delete('schedule',id);return {'deleted':True}
    @app.get('/api/v1/training-examples')
    def examples(request:Request):return items(post.examples(),request)
    @app.post('/api/v1/training-examples',status_code=201)
    async def new_example(request:Request):
        body=await request.json();return await mutation(request,body,lambda:post.create_example(body),201)
    @app.post('/api/v1/training-examples/{id}/review')
    async def example_review(id:str,request:Request):
        body=await request.json();return await mutation(request,body,lambda:post.review_example(id,body))
    @app.post('/api/v1/training-snapshots',status_code=202)
    async def export(request:Request):
        body=await request.json();return await mutation(request,body,lambda:post.create_snapshot(body),202)
    @app.get('/api/v1/training-snapshots')
    def snapshots(request:Request):return items(store.list('training_snapshot'),request)
    @app.get('/api/v1/training-snapshots/{id}')
    def snapshot(id:str):return post.snapshot(id)
    @app.get('/api/training/status')
    def trainer_status():return post.trainer_status()
    @app.post('/api/training/jobs')
    async def training_start(request:Request):return await run_in_threadpool(post.trainer_command,'start',await request.json())
    @app.post('/api/training/jobs/{id}/{action}')
    def training_command(id:str,action:str):
        if action not in ('pause','resume','cancel'):raise DomainError('INVALID_ACTION','지원하지 않는 학습 제어입니다.',422)
        return post.trainer_command(action,job_id=id)
    @app.post('/api/v1/backups',status_code=202)
    async def backup(request:Request):
        body=await request.json();identity=uid()
        return await mutation(request,body,lambda:store.enqueue('backup',{'backupId':identity}),202)
    @app.get('/api/v1/backups')
    def backups(request:Request):return items(store.list('backup'),request)
    @app.get('/api/v1/backups/{id}')
    def backup_result(id:str):return store.get('backup',id)
    @app.post('/api/v1/evaluations',status_code=202)
    async def evaluate(request:Request):
        body=await request.json()
        if not isinstance(body,dict) or body.get('split','dev') not in ('dev','holdout'):raise DomainError('INVALID_EVALUATION','dev 또는 holdout 평가를 지정하세요.',422)
        if body.get('suite')=='builtin-code':
            if body.get('split','dev')!='dev':raise DomainError('INVALID_EVALUATION','기본 평가셋은 dev 전용입니다.',422)
            return await mutation(request,body,lambda:store.enqueue('evaluate',{'suite':'builtin-code','evaluationId':uid()},priority=70),202)
        if body.get('suite'):raise DomainError('INVALID_EVALUATION','지원하지 않는 평가셋입니다.',422)
        from RAG.evaluation.service import validate_questions
        questions=body.get('questions')
        validate_questions(questions)
        def start_evaluation():
            if body.get('split')=='holdout':
                with store.tx():
                    old=store.setting('evaluationHoldout',{})
                    ids={key:set(old.get(key,[])) for key in ('versionIds','documentIds','sourceIds','splitGroups')}
                    for q in questions:
                        for doc_id in q.get('expectedDocumentIds',[]):
                            doc=pre.document(doc_id);ids['documentIds'].add(doc['id']);ids['sourceIds'].add(doc['sourceId']);ids['versionIds'].add(doc['currentVersionId'])
                        for chunk_id in q.get('expectedChunkIds',[]):
                            chunk=store.get('chunk',chunk_id)
                            doc=pre.document(chunk['documentId'])
                            ids['versionIds'].add(chunk['versionId']);ids['documentIds'].add(doc['id']);ids['sourceIds'].add(doc['sourceId'])
                        ids['splitGroups'].update(q.get('splitGroups',[]))
                    store.set_setting('evaluationHoldout',{key:sorted(values) for key,values in ids.items()})
            return store.enqueue('evaluate',{**body,'evaluationId':uid()})
        return await mutation(request,body,start_evaluation,202)
    @app.get('/api/v1/evaluations')
    def evaluations(request:Request):return items(store.list('evaluation'),request)
    @app.get('/api/v1/evaluations/{id}')
    def evaluation_detail(id:str):return store.get('evaluation',id)

    @app.get('/api/v1/knowledge/status')
    def knowledge_status():
        documents=[d for d in store.list('document') if not d.get('deletedAt')]
        versions={v['id']:v for v in store.list('version')}
        current=[versions[d['currentVersionId']] for d in documents if d['currentVersionId'] in versions]
        scores=[v['qualityRating']['score'] for v in current if v.get('qualityRating')]
        encoder=pre.encoder()
        return {'retrieval':pre.readiness(),'embedding':{'available':encoder.available,'reason':encoder.reason,'profile':encoder.profile},
                'documents':len(documents),'analyzedDocuments':sum(bool(v.get('codeStructure')) for v in current),
                'symbols':sum(len(v.get('codeStructure',{}).get('symbols',[])) for v in current),
                'calls':sum(len(v.get('codeStructure',{}).get('calls',[])) for v in current),
                'quality':{'scoredDocuments':len(scores),'meanScore':round(sum(scores)/len(scores),1) if scores else None,'testExecution':'not_run'},
                'upstreamDeletedSources':sum(bool(s.get('upstreamDeleted')) for s in store.list('source')),'observedAt':now()}
    @app.post('/api/v1/knowledge/refresh',status_code=202)
    async def knowledge_refresh(request:Request):return await mutation(request,{},lambda:store.enqueue('reindex',{},priority=40),202)
    @app.post('/api/v1/embedding/install',status_code=202)
    async def embedding_install(request:Request):
        def enqueue_install():
            active=store.one("SELECT id FROM jobs WHERE kind='embedding_install' AND state IN ('queued','running','retry_wait') LIMIT 1")
            return store.job(active['id']) if active else store.enqueue('embedding_install',{},priority=40)
        return await mutation(request,{},enqueue_install,202)
    @app.get('/api/v1/documents/{id}/structure')
    def document_structure(id:str):
        doc=pre.document(id);pre.content(doc['currentVersionId'])
        return {'documentId':id,'versionId':doc['currentVersionId'],'structure':doc.get('codeStructure'),'qualityRating':doc.get('qualityRating')}

    @app.get('/api/v1/crawler/status')
    def crawler_status():return runtime.crawler.status()
    @app.get('/api/v1/crawler/config')
    def crawler_config():return runtime.crawler.config()
    @app.patch('/api/v1/crawler/config')
    async def crawler_configure(request:Request):return runtime.crawler.configure(await request.json())
    @app.post('/api/v1/crawler/run',status_code=202)
    async def crawler_run(request:Request):return await mutation(request,{},runtime.crawler.enqueue,202)
    @app.get('/api/v1/crawler/frontier')
    def crawler_frontier():return {'items':runtime.crawler.frontier(),'nextCursor':None}

    if os.environ.get('SEED_LLM_ENABLED')=='1':
        from LLM.integration import attach
        attach(app,store,pre,post)

    # Last registered middleware is outermost, including pre-auth JSON parsing.
    app.add_middleware(BodyLimitMiddleware)
    dist=Path(__file__).parent.parent/'Interface'/'dist'
    if dist.is_dir():
        app.mount('/',StaticFiles(directory=str(dist),html=True),name='interface')
    return app
