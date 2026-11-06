"""Persistent code-first crawl frontier, automatic discovery and RAG ingestion."""
from __future__ import annotations

import asyncio
import base64
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime,timedelta,timezone
from urllib.parse import urlsplit,unquote

from RAG.core import DomainError,digest,now,uid
from .web import PublicHTTP,Robots,canonical,extract_code,SOURCE_EXTENSIONS
from .discovery import Discovery
from .balance import select as balanced_select, record as record_balance, labels

DEFAULTS={
    'version':1,'enabled':True,'intervalMinutes':60,'maxPagesPerRun':6,'dailyPages':60,
    'maxDepth':2,'maxFrontier':3000,'delaySeconds':1.5,'maxPageBytes':2*1024**2,
    'githubEnabled':True,'sitemapEnabled':True,'feedEnabled':True,'commonCrawlEnabled':True,
    'browserEnabled':True,'maxRendered':1,'codeOnly':True,
    'fairScheduling':True,'languageWeights':{'python':1,'javascript':1,'typescript':1,'go':1,'rust':1,'java':1,'cpp':1},
    'seedUrls':['https://docs.python.org/3/tutorial/','https://fastapi.tiangolo.com/tutorial/','https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide'],
    'feedUrls':['https://github.blog/changelog/feed/'],
    'githubQueries':['language:Python size:<500 archived:false fork:false','language:TypeScript size:<500 archived:false fork:false','language:Go size:<500 archived:false fork:false','language:Rust size:<500 archived:false fork:false','language:Java size:<500 archived:false fork:false','language:C++ size:<500 archived:false fork:false'],
    'commonCrawlPatterns':['docs.python.org/3/tutorial/introduction.html'],
}


class CrawlerService:
    def __init__(self,store,pre):
        self.store,self.pre=store,pre
        with store.tx() as db:
            db.executescript('''CREATE TABLE IF NOT EXISTS crawl_frontier(
                id TEXT PRIMARY KEY,url TEXT NOT NULL UNIQUE,kind TEXT NOT NULL,origin TEXT NOT NULL,
                depth INTEGER NOT NULL DEFAULT 0,priority INTEGER NOT NULL DEFAULT 50,state TEXT NOT NULL DEFAULT 'queued',
                meta TEXT NOT NULL,attempt INTEGER NOT NULL DEFAULT 0,next_at REAL NOT NULL DEFAULT 0,
                last_error TEXT,created_at TEXT NOT NULL,updated_at TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS crawl_due ON crawl_frontier(state,next_at,priority,created_at);''')
        if store.setting('crawler') is None:store.set_setting('crawler',dict(DEFAULTS))
        # Earlier crawler versions stored immutable URLs without a repository watch.
        for source in store.list('source'):
            if not source.get('crawlerManaged'):continue
            parsed=urlsplit(source.get('url',''));parts=parsed.path.strip('/').split('/')
            if parsed.hostname=='raw.githubusercontent.com' and len(parts)>=4:
                repository='/'.join(parts[:2]);path=unquote('/'.join(parts[3:]))
                family=source.get('repositoryFamily') or repository
                store.put('source',{'repository':repository,'repositoryPath':path,'repositoryFamily':family if family.startswith('https://github.com/') else 'https://github.com/'+family},id=source['id'])
                self._track_repository({'repository':repository,'revision':parts[2]})

    def config(self):return {**DEFAULTS,**self.store.setting('crawler',{})}

    def configure(self,body):
        if not isinstance(body,dict):raise DomainError('INVALID_CRAWLER_CONFIG','크롤러 설정은 객체여야 합니다.',422)
        if set(body)-set(DEFAULTS):raise DomainError('INVALID_CRAWLER_CONFIG','지원하지 않는 크롤러 설정입니다.',422)
        with self.store.tx():
            old=self.config()
            if body.get('version')!=old['version']:raise DomainError('CRAWLER_CONFIG_CONFLICT','크롤러 설정을 새로고침하세요.',412)
            cfg={**old,**body,'version':old['version']+1}
            for key in ('enabled','githubEnabled','sitemapEnabled','feedEnabled','commonCrawlEnabled','browserEnabled','codeOnly','fairScheduling'):
                if not isinstance(cfg[key],bool):raise DomainError('INVALID_CRAWLER_CONFIG',f'{key}는 boolean이어야 합니다.',422)
            ranges={'intervalMinutes':(5,1440),'maxPagesPerRun':(1,20),'dailyPages':(1,1000),'maxDepth':(0,4),'maxFrontier':(10,20000),'maxRendered':(0,2),'maxPageBytes':(65536,2*1024**2)}
            for key,(low,high) in ranges.items():
                if not isinstance(cfg[key],int) or isinstance(cfg[key],bool) or not low<=cfg[key]<=high:raise DomainError('INVALID_CRAWLER_CONFIG',f'{key} 범위를 확인하세요.',422)
            if not isinstance(cfg['delaySeconds'],(int,float)) or isinstance(cfg['delaySeconds'],bool) or not 1<=cfg['delaySeconds']<=30:raise DomainError('INVALID_CRAWLER_CONFIG','요청 간격은 1~30초입니다.',422)
            if not isinstance(cfg['languageWeights'],dict) or len(cfg['languageWeights'])>40 or any(not isinstance(k,str) or len(k)>40 or isinstance(v,bool) or not isinstance(v,(int,float)) or not .1<=v<=10 for k,v in cfg['languageWeights'].items()):raise DomainError('INVALID_CRAWLER_CONFIG','언어 가중치는 0.1~10입니다.',422)
            for key in ('seedUrls','feedUrls','githubQueries','commonCrawlPatterns'):
                if not isinstance(cfg[key],list) or len(cfg[key])>30 or any(not isinstance(v,str) or not v.strip() or len(v)>2000 for v in cfg[key]):raise DomainError('INVALID_CRAWLER_CONFIG',f'{key} 목록을 확인하세요.',422)
            cfg['seedUrls']=[canonical(v) for v in cfg['seedUrls']]
            cfg['feedUrls']=[canonical(v) for v in cfg['feedUrls']]
            self.store.set_setting('crawler',cfg)
            if not cfg['enabled']:
                for row in self.store.read("SELECT id FROM jobs WHERE kind='crawl_cycle' AND state IN ('queued','running','retry_wait')"):
                    self.store.cancel_job(row['id'])
        self.store.activity('crawler_configured','workspace','코드 자동 수집 설정 변경')
        return cfg

    def add(self,item,depth=0):
        url=canonical(item['url']);kind=item.get('kind','page')
        if kind not in ('page','code','sitemap','feed','archive'):raise DomainError('INVALID_FRONTIER_KIND','지원하지 않는 수집 항목입니다.',422)
        cfg=self.config()
        # Discard document actions, downloads and unbounded query permutations.
        parsed=urlsplit(url)
        if len(parsed.query)>512 or any(word in parsed.path.lower() for word in ('/logout','/login','/signup','/cart/','/search')):return None
        if depth>cfg['maxDepth']:return None
        with self.store.tx() as db:
            if item.get('sourceKey'):
                previous=db.execute("SELECT * FROM crawl_frontier WHERE json_extract(meta,'$.sourceKey')=? LIMIT 1",(item['sourceKey'],)).fetchone()
                if previous:
                    old_meta=json.loads(previous['meta'])
                    if old_meta.get('revision')!=item.get('revision'):
                        db.execute("UPDATE crawl_frontier SET url=?,meta=?,state='queued',next_at=0,last_error=NULL,updated_at=? WHERE id=?",(url,json.dumps(item,ensure_ascii=False),now(),previous['id']))
                    return previous['id']
            old=db.execute('SELECT id FROM crawl_frontier WHERE url=?',(url,)).fetchone()
            if old:return old['id']
            if db.execute('SELECT count(*) AS n FROM crawl_frontier').fetchone()['n']>=cfg['maxFrontier']:return None
            identity=uid();stamp=now()
            priority=10 if kind=='code' or any(x in url.lower() for x in ('example','tutorial','reference','/library/')) else 30
            db.execute('INSERT INTO crawl_frontier(id,url,kind,origin,depth,priority,meta,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)',
                       (identity,url,kind,item.get('origin','link'),depth,priority,json.dumps(item,ensure_ascii=False),stamp,stamp))
        return identity

    def status(self):
        counts={r['state']:r['n'] for r in self.store.read('SELECT state,count(*) AS n FROM crawl_frontier GROUP BY state')}
        packages={}
        for package in ('scrapy','httpx','selectolax','trafilatura','feedparser','playwright','warcio'):
            try:packages[package]=importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:packages[package]=None
        day=datetime.now(timezone(timedelta(hours=9))).date().isoformat()
        source_ids={s['id'] for s in self.store.list('source') if s.get('crawlerManaged')}
        documents=[d for d in self.pre.documents() if d.get('sourceId') in source_ids]
        return {'config':self.config(),'frontier':counts,'today':self.store.setting('crawlerDay:'+day,{'pages':0,'codeDocuments':0,'bytes':0}),
                'indexedDocuments':sum(d['state']=='ready' for d in documents),'reviewDocuments':sum(d['state']=='needs_review' for d in documents),
                'lastRun':self.store.setting('crawlerLastRun'),'nextRunAt':self.store.setting('crawlerNextAt',0),'packages':packages,'balance':self.store.setting('crawlerBalance:'+day,{}),'trackedRepositories':len(self.store.list('crawl_repository')),
                'technologies':{'githubRestApi':True,'scrapy':bool(packages['scrapy']),'sitemap':True,'rssAtom':bool(packages['feedparser']),
                                'httpx':bool(packages['httpx']),'selectolax':bool(packages['selectolax']),'trafilatura':bool(packages['trafilatura']),
                                'playwright':bool(packages['playwright']),'commonCrawl':bool(packages['warcio']),'robotparser':True},
                'learningMode':'rag_index','weightTraining':False,'observedAt':now()}

    def enqueue(self):
        with self.store.tx():
            if not self.config()['enabled']:raise DomainError('CRAWLER_PAUSED','자동 크롤링이 중지되어 있습니다.',409)
            active=self.store.one("SELECT id FROM jobs WHERE kind='crawl_cycle' AND state IN ('queued','running','retry_wait','cancelling') LIMIT 1")
            return self.store.job(active['id']) if active else self.store.enqueue('crawl_cycle',{},priority=60)

    def tick(self):
        cfg=self.config()
        if cfg['enabled'] and time.time()>=self.store.setting('crawlerNextAt',0):
            self.enqueue();self.store.set_setting('crawlerNextAt',time.time()+cfg['intervalMinutes']*60)

    def frontier(self):
        return [{**row,'meta':json.loads(row['meta'])} for row in self.store.read('SELECT * FROM crawl_frontier ORDER BY updated_at DESC LIMIT 100')]

    def _done(self,url,state='visited',error=None):
        delay=7*86400 if state in ('visited','skipped') else 6*3600
        with self.store.tx() as db:
            db.execute('UPDATE crawl_frontier SET state=?,attempt=attempt+1,last_error=?,next_at=?,updated_at=? WHERE url=?',(state,error,time.time()+delay,now(),url))

    def _snapshot(self,snapshot):
        if not snapshot.get('complete'):return
        name=snapshot['repository'];paths=set(snapshot['paths']);changed=False
        with self.store.tx():
            self.store.put('crawl_repository',{**snapshot,'checkedAt':now()},id=digest(name)[:36])
            for row in self.store.read("SELECT id,meta FROM crawl_frontier WHERE json_extract(meta,'$.repository')=?",(name,)):
                if json.loads(row['meta']).get('path') not in paths:
                    # Old immutable URLs must not resurrect a deleted logical file.
                    with self.store.tx() as db:db.execute("UPDATE crawl_frontier SET state='removed',next_at=253402300799,updated_at=? WHERE id=?",(now(),row['id']))
            for source in self.store.list('source'):
                if not source.get('crawlerManaged') or source.get('repository')!=name:continue
                deleted=source.get('repositoryPath') not in paths
                if deleted!=bool(source.get('upstreamDeleted')):
                    self.store.put('source',{'upstreamDeleted':deleted,'upstreamCheckedAt':now(),'upstreamRevision':snapshot['revision']},id=source['id']);changed=True
            if changed:
                self.pre._bump_policy();self.store.enqueue('index',{},key='upstream:'+name+':'+snapshot['revision'])

    def _track_repository(self,meta):
        name=meta.get('repository')
        if not name:return
        identity=digest(name)[:36]
        try:self.store.get('crawl_repository',identity)
        except DomainError:self.store.put('crawl_repository',{'repository':name,'revision':meta.get('revision'),'complete':False,'paths':[],'fileCursor':0},id=identity)

    def _source_for(self,url,meta=None):
        key=(meta or {}).get('sourceKey') or 'web:'+canonical(url)
        try:return self.store.get('source',self.store.get('crawl_source_map',digest(key)[:36])['sourceId'])
        except DomainError:return None

    def _missing(self,url,status,meta=None,reason='UPSTREAM_MISSING'):
        source=self._source_for(url,meta)
        if not source:return
        with self.store.tx():
            count=source.get('upstreamMissingCount',0)+1
            self.store.put('source',{'upstreamMissingCount':count,'upstreamCheckedAt':now(),'upstreamRemovalReason':reason},id=source['id'])
            if status==410 or count>=2:
                self.store.put('source',{'upstreamDeleted':True},id=source['id']);self.pre._bump_policy()
                self.store.enqueue('index',{},key='upstream-missing:'+source['id']+':'+str(count))

    async def discover(self,cfg,job=None,archive_budget=2):
        def check_cancel(exc):
            if getattr(exc,'code','') in ('CANCELLED','JOB_CANCELLED','LEASE_EXPIRED'):raise exc
        allowed=self.store.setting('settings',{}).get('allowedSourceHosts') or []
        transport=PublicHTTP(allowed,guard=(lambda:self.store.assert_lease(job['id'],job['fencingToken'])) if job else None);robots=Robots(transport,self.store.setting('crawlerRobots',{}));discovery=Discovery(transport,robots)
        discovery.cursors={r['repository']:r.get('fileCursor',0) for r in self.store.list('crawl_repository')}
        errors=[];found=0;archives=archive_codes=archive_bytes=0
        for url in cfg['seedUrls']:found+=bool(self.add({'url':url,'kind':'page','origin':'seed'}))
        if cfg['feedEnabled']:
            for url in cfg['feedUrls']:self.add({'url':url,'kind':'feed','origin':'seed_feed'})
        if cfg['sitemapEnabled']:
            for url in cfg['seedUrls'][:3]:
                try:
                    await robots.allowed(url)
                    origin=f'{urlsplit(url).scheme}://{urlsplit(url).netloc}'
                    for sitemap in (robots.sitemaps.get(origin) or [origin+'/sitemap.xml'])[:2]:self.add({'url':sitemap,'kind':'sitemap','origin':'robots_sitemap'})
                except DomainError as exc:
                    check_cancel(exc);errors.append({'stage':'sitemap_discovery','code':exc.code})
        if cfg['githubEnabled'] and cfg['githubQueries'] and time.time()>=self.store.setting('crawlerGithubNextAt',0):
            cursor=self.store.setting('crawlerGithubCursor',0);query=cfg['githubQueries'][cursor%len(cfg['githubQueries'])]
            try:
                for item in await discovery.github(query,page=1+(cursor//len(cfg['githubQueries']))%10,max_files=20):found+=bool(self.add(item))
                for snapshot in discovery.snapshots:self._snapshot(snapshot)
                tracked=self.store.list('crawl_repository')
                if tracked:
                    refresh_cursor=self.store.setting('crawlerRefreshCursor',0);repository=tracked[refresh_cursor%len(tracked)]
                    paths=[s['repositoryPath'] for s in self.store.list('source') if s.get('repository')==repository['repository'] and s.get('repositoryPath')]
                    if paths:
                        for item in await discovery.refresh(repository['repository'],max_files=20,tracked_paths=paths):found+=bool(self.add(item))
                        self._snapshot(discovery.snapshots[-1])
                    self.store.set_setting('crawlerRefreshCursor',refresh_cursor+1)
                self.store.set_setting('crawlerGithubNextAt',time.time()+60)
            except (DomainError,ValueError,KeyError) as exc:
                check_cancel(exc)
                errors.append({'stage':'github','code':getattr(exc,'code','INVALID_API_RESPONSE')})
                self.store.set_setting('crawlerGithubNextAt',time.time()+(3600 if getattr(exc,'code','')=='CRAWL_RATE_LIMIT' else 300))
            self.store.set_setting('crawlerGithubCursor',cursor+1)
        # Common Crawl is queried at most once a day, never downloaded in full.
        if cfg['commonCrawlEnabled'] and cfg['commonCrawlPatterns'] and time.time()-self.store.setting('crawlerArchiveDiscoveryAt',0)>86400:
            archive_cursor=self.store.setting('crawlerArchiveCursor',0)
            try:
                pattern=cfg['commonCrawlPatterns'][archive_cursor%len(cfg['commonCrawlPatterns'])]
                for item in await discovery.commoncrawl(pattern,limit=2):found+=bool(self.add(item))
            except (DomainError,ValueError,KeyError) as exc:
                check_cancel(exc);errors.append({'stage':'commoncrawl','code':getattr(exc,'code','INVALID_API_RESPONSE')})
            self.store.set_setting('crawlerArchiveDiscoveryAt',time.time())
            self.store.set_setting('crawlerArchiveCursor',archive_cursor+1)
        rows=self.store.read("SELECT * FROM crawl_frontier WHERE kind IN ('sitemap','feed','archive') AND next_at<=? ORDER BY priority,created_at LIMIT 3",(time.time(),))
        for row in rows:
            item=json.loads(row['meta'])
            try:
                if row['kind']=='archive':
                    if archives>=archive_budget:continue
                    capture=await discovery.warc(item)
                    extracted=extract_code(capture['body'],item['url'],capture['headers']['content-type'])
                    ingested=self.ingest({**extracted,'url':item['url'],'raw':base64.b64encode(capture['body']).decode(),'archive':item['archive']},item)
                    archives+=1;archive_codes+=bool(ingested['ingested']);archive_bytes+=len(capture['body'])
                    self._done(row['url'],'visited')
                    continue
                values=await (discovery.sitemap(row['url']) if row['kind']=='sitemap' else discovery.feed(row['url']))
                for child in values:found+=bool(self.add(child,depth=row['depth']+1))
                self._done(row['url'],'visited')
            except (DomainError,ValueError,KeyError) as exc:
                check_cancel(exc)
                code=getattr(exc,'code','DISCOVERY_PARSE_ERROR');errors.append({'stage':row['kind'],'code':code});self._done(row['url'],'blocked' if code=='ROBOTS_DENIED' else 'failed',code)
        self.store.set_setting('crawlerRobots',robots.cache)
        return {'found':found,'errors':errors,'archivePages':archives,'archiveDocuments':archive_codes,'archiveBytes':archive_bytes}

    def ingest(self,page,meta=None):
        meta=meta or {};cfg=self.config()
        if meta.get('repository'):
            try:snapshot=self.store.get('crawl_repository',digest(meta['repository'])[:36])
            except DomainError:snapshot=None
            if snapshot and snapshot.get('complete') and meta.get('path') not in snapshot['paths']:return {'ingested':False,'reason':'UPSTREAM_DELETED'}
        if cfg['codeOnly'] and not page.get('codeCount'):
            raw=base64.b64decode(page['raw'],validate=True)
            dynamic=any(marker in raw.lower() for marker in (b'__next',b'id="root"',b'id="app"',b'enable javascript'))
            if not page.get('renderError') and (not dynamic or page.get('rendered')):self._missing(page['url'],404,meta,reason='CODE_REMOVED')
            return {'ingested':False,'reason':'NO_CODE'}
        raw=base64.b64decode(page['raw'],validate=True)
        if len(raw)>cfg['maxPageBytes']:raise DomainError('SOURCE_TOO_LARGE','수집 원문 상한 초과',413)
        url=canonical(page['url']);key=meta.get('sourceKey') or 'web:'+url
        identity=digest(key)[:36]
        try:
            mapping=self.store.get('crawl_source_map',identity)
            source=self.store.get('source',mapping['sourceId'])
        except DomainError:
            dataset=next((d for d in self.pre.list_datasets() if d.get('crawlerManaged')),None)
            if not dataset:
                dataset=self.pre.create_dataset({'name':'자동 수집 · 코드'});self.store.put('dataset',{'crawlerManaged':True},id=dataset['id'])
            source=self.pre.create_source({'kind':'web_document','name':page.get('title',url)[:120],'datasetId':dataset['id'],'locator':{'url':url},'retrievalPolicy':'approved','trainingPolicy':'unknown','licenseId':meta.get('licenseId')})
            # Source ID remains server generated; logical discovery key is separate.
            self.store.put('crawl_source_map',{'sourceId':source['id'],'sourceKey':key},id=identity)
        if source.get('retrievalPolicy')=='denied' or not source.get('collectionEnabled',True):return {'ingested':False,'reason':'SOURCE_DISABLED'}
        self._track_repository(meta)
        capture_hash=self.pre._blob(raw)
        rendered_hash=None
        if page.get('renderedRaw'):
            rendered=base64.b64decode(page['renderedRaw'],validate=True)
            if len(rendered)>cfg['maxPageBytes']:raise DomainError('SOURCE_TOO_LARGE','렌더링 문서 상한 초과',413)
            rendered_hash=self.pre._blob(rendered)
        self.store.put('source',{'locator':{'url':url},'url':url,'crawlerManaged':True,'rawCaptureHash':capture_hash,'captureKind':'archive' if page.get('archive') else 'live',
                                'renderedCaptureHash':rendered_hash,'archive':page.get('archive'),'codeCount':page.get('codeCount',0),'extractor':page.get('extractor'),'repositoryFamily':meta.get('repositoryFamily'),
                                'repository':meta.get('repository'),'repositoryPath':meta.get('path'),'repositorySignals':meta.get('repositorySignals',{}),'repositoryUpdatedAt':meta.get('repositoryUpdatedAt'),
                                'discoveryRevision':meta.get('revision'),'etag':page.get('etag'),'lastModified':page.get('lastModified'),'upstreamDeleted':False,'upstreamMissingCount':0,'upstreamRemovalReason':None,'discoveryOrigin':meta.get('origin','link'),'lastSyncedAt':now()},id=source['id'])
        name=meta.get('path') or page.get('name') or 'page.md'
        result=self.pre.ingest_source_files(source['id'],[(name,page['text'].encode())],revision=meta.get('revision') or digest(page['text'].encode()))
        self.store.activity('crawler_code_collected',source['id'],'공개 코드 수집 · 정제/색인 대기')
        return {'ingested':True,'sourceId':source['id'],'jobId':result.get('jobId'),'captureHash':capture_hash}

    def run_scrapy(self,config,job):
        root=Path(__file__).resolve().parents[3]
        process=subprocess.Popen([sys.executable,'-m','RAG.Pre.crawling.scrapy_runner'],cwd=str(root),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        payload=json.dumps(config).encode();started=time.monotonic()
        try:
            while True:
                try:
                    stdout,stderr=process.communicate(input=payload,timeout=1);break
                except subprocess.TimeoutExpired:
                    payload=None
                    self.store.assert_lease(job['id'],job['fencingToken'])
                    if time.monotonic()-started>180:raise DomainError('CRAWLER_TIMEOUT','크롤링 실행 상한 초과',503)
            if process.returncode!=0:
                raise DomainError('SCRAPY_FAILED','Scrapy 작업자 실행 실패',503,{'stderr':stderr.decode(errors='replace')[-1500:]})
            return [json.loads(line) for line in stdout.splitlines() if line.startswith(b'{')]
        finally:
            if process.poll() is None:
                import psutil
                try:
                    for child in psutil.Process(process.pid).children(recursive=True):
                        try:child.kill()
                        except psutil.Error:pass
                except psutil.Error:pass
                process.kill();process.communicate()

    def process_job(self,job):
        cfg=self.config()
        if not cfg['enabled']:raise DomainError('CRAWLER_PAUSED','자동 크롤링이 중지되었습니다.',409)
        day=datetime.now(timezone(timedelta(hours=9))).date().isoformat();key='crawlerDay:'+day
        totals=self.store.setting(key,{'pages':0,'codeDocuments':0,'bytes':0})
        remaining=min(cfg['maxPagesPerRun'],max(0,cfg['dailyPages']-totals['pages']))
        if remaining==0:return {'state':'daily_limit','visited':0}
        self.store.update_job(job['id'],stage='discovering',unit='pages')
        target=job.get('payload',{}).get('targetSourceId')
        if target:
            source=self.store.get('source',target)
            if not source.get('collectionEnabled',True):raise DomainError('SOURCE_DISABLED','수집이 중지된 소스입니다.',409)
            mapping=next((m for m in self.store.list('crawl_source_map') if m['sourceId']==target),None)
            row=self.store.one("SELECT * FROM crawl_frontier WHERE json_extract(meta,'$.sourceKey')=?",(mapping['sourceKey'],)) if mapping and mapping['sourceKey'].startswith('github:') else self.store.one('SELECT * FROM crawl_frontier WHERE url=?',(source['url'],))
            if not row:
                self.add({'url':source['url'],'kind':'page','origin':'source_sync'})
                row=self.store.one('SELECT * FROM crawl_frontier WHERE url=?',(source['url'],))
            target_rows=[row] if row and row['state']!='removed' else []
            discovery={'found':0,'errors':[]};remaining=min(remaining,1)
        else:discovery=asyncio.run(self.discover(cfg,job=job,archive_budget=remaining))
        remaining-=discovery.get('archivePages',0)
        self.store.assert_lease(job['id'],job['fencingToken'])
        rows=target_rows if target else self.store.read("SELECT * FROM crawl_frontier WHERE kind IN ('page','code') AND next_at<=? ORDER BY priority,created_at LIMIT ?",(time.time(),cfg['maxFrontier'] if cfg['fairScheduling'] else remaining))
        balance=self.store.setting('crawlerBalance:'+day,{})
        rows=balanced_select(rows,remaining,balance,cfg) if cfg['fairScheduling'] else rows
        visited=discovery.get('archivePages',0);ingested=discovery.get('archiveDocuments',0);totals['bytes']+=discovery.get('archiveBytes',0);errors=discovery['errors']
        if rows:
            domains=list({urlsplit(row['url']).hostname for row in rows})
            request_urls=[]
            for row in rows:
                source=self._source_for(row['url'],json.loads(row['meta']));headers={}
                if source and source.get('etag'):headers['If-None-Match']=source['etag']
                if source and source.get('lastModified'):headers['If-Modified-Since']=source['lastModified']
                request_urls.append({'url':row['url'],'depth':row['depth'],'headers':headers})
            worker_cfg={**cfg,'maxPages':remaining,'followLinks':False,'urls':request_urls,'domains':domains,
                        'allowedHosts':self.store.setting('settings',{}).get('allowedSourceHosts') or [],'robotsCache':self.store.setting('crawlerRobots',{})}
            self.store.update_job(job['id'],stage='crawling',totalItems=remaining+visited)
            for event in self.run_scrapy(worker_cfg,job):
                if event.get('type')=='finished':self.store.set_setting('crawlerRobots',event.get('robotsCache',{}));continue
                if event.get('type') in ('not_modified','upstream_missing'):
                    row=next((r for r in rows if r['url']==event['url']),None);meta=json.loads(row['meta']) if row else {}
                    if event['type']=='not_modified':
                        source=self._source_for(event['url'],meta)
                        if source:
                            self.store.put('source',{'lastSyncedAt':now(),'upstreamMissingCount':0,'upstreamDeleted':False,'lastCheckedRevision':meta.get('revision')},id=source['id'])
                            if source.get('upstreamDeleted'):
                                self.pre._bump_policy();self.store.enqueue('index',{},key='upstream-restored:'+source['id']+':'+uid())
                        self._done(event['url'])
                    else:self._missing(event['url'],event['status'],meta);self._done(event['url'],'failed','UPSTREAM_MISSING')
                    visited+=1
                    if row:record_balance(balance,row)
                    continue
                if event.get('type')=='error':
                    self._done(event['url'],'blocked' if event['code']=='ROBOTS_DENIED' else 'failed',event['code']);errors.append(event);continue
                if event.get('type')!='page':continue
                self.store.assert_lease(job['id'],job['fencingToken'])
                url=canonical(event['url']);row=next((r for r in rows if r['url']==event.get('requestUrl',url)),None);meta=json.loads(row['meta']) if row else {'origin':'link'}
                self.add({'url':url,'kind':'page','origin':meta.get('origin','link')},event.get('depth',0))
                result=self.ingest(event,meta);visited+=1;ingested+=bool(result['ingested'])
                record_balance(balance,row or {'url':url,'meta':meta})
                totals['bytes']+=len(base64.b64decode(event['raw']))
                self._done(url,'visited' if result['ingested'] else 'skipped',event.get('renderError'))
                if row and row['url']!=url:self._done(row['url'],'visited')
                for link in event.get('links',[])[:50]:
                    if urlsplit(link).hostname in domains:self.add({'url':link,'kind':'page','origin':'link'},event.get('depth',0)+1)
                for item in event.get('discovery',[]):
                    if cfg['feedEnabled']:self.add(item,event.get('depth',0)+1)
                self.store.update_job(job['id'],processedItems=visited)
        totals['pages']+=visited;totals['codeDocuments']+=ingested;self.store.set_setting(key,totals)
        self.store.set_setting('crawlerBalance:'+day,balance)
        result={'finishedAt':now(),'discovered':discovery['found'],'visited':visited,'ingested':ingested,'errors':errors[:30],'learningMode':'rag_index','modelWeightsChanged':False}
        self.store.set_setting('crawlerLastRun',result);self.store.activity('crawler_cycle_completed',job['id'],f'자동 코드 수집 · {visited}페이지 / {ingested}코드 문서')
        return result
