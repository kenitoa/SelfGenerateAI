"""One isolated Scrapy process per bounded batch (reactors are not restarted)."""
from __future__ import annotations

import asyncio
import base64
import json
import sys
import time
from urllib.parse import urlsplit

import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.exceptions import IgnoreRequest
from scrapy.http import HtmlResponse
from scrapy.linkextractors import LinkExtractor

from RAG.core import DomainError
from RAG.Pre.crawling.web import PublicHTTP, Robots, Renderer, extract_code

CONFIG = {}
HTTP = None
POLICY = None


def emit(value):
    print(json.dumps(value,ensure_ascii=True,default=str),flush=True)


class SafeDownloader:
    @classmethod
    def from_crawler(cls,crawler):
        self=cls();self.config=crawler.settings['SEED_CONFIG']
        self.http=PublicHTTP(self.config.get('allowedHosts',[]),max_bytes=self.config.get('maxPageBytes',2*1024**2))
        self.policy=Robots(self.http,self.config.get('robotsCache',{}))
        crawler.seed_robots=self.policy
        return self

    async def process_request(self,request,spider=None):
        # Return a Response here so Scrapy never opens its own DNS/network path.
        try:
            if not await self.policy.allowed(request.url):
                emit({'type':'error','url':request.url,'code':'ROBOTS_DENIED'})
                raise IgnoreRequest('robots policy')
            origin=f'{urlsplit(request.url).scheme}://{urlsplit(request.url).netloc}'
            await asyncio.sleep(max(self.config.get('delaySeconds',1),self.policy.delays.get(origin,1)))
            started=time.perf_counter()
            result=await self.http.get(request.url,headers=dict(request.meta.get('conditionalHeaders',{})),allow_status=(304,404,410))
            request.meta['download_latency']=time.perf_counter()-started
            if result['status'] in (304,404,410):
                emit({'type':'not_modified' if result['status']==304 else 'upstream_missing','url':request.url,'status':result['status']})
                raise IgnoreRequest('conditional response')
            return HtmlResponse(url=result['url'],status=result['status'],body=result['body'],headers=result['headers'],request=request,encoding='utf-8')
        except DomainError as exc:
            emit({'type':'error','url':request.url,'code':exc.code,'details':exc.details})
            raise IgnoreRequest(exc.code)


class CodeSpider(scrapy.Spider):
    name='seed_code'

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.count=0;self.rendered=0
        self.allowed_domains=CONFIG['domains']
        self.links=LinkExtractor(allow_domains=self.allowed_domains,deny=(r'/login',r'/signup',r'/search\?',r'\?replytocom=',r'/logout'))

    async def start(self):
        for item in CONFIG['urls']:
            yield scrapy.Request(item['url'],callback=self.parse,errback=self.failed,meta={'seedDepth':item.get('depth',0),'conditionalHeaders':item.get('headers',{})})

    def failed(self,failure):
        if failure.check(IgnoreRequest):return
        emit({'type':'error','url':failure.request.url,'code':'REQUEST_FAILED','message':failure.getErrorMessage()[:500]})

    async def parse(self,response):
        if self.count>=CONFIG['maxPages']:return
        self.count+=1
        body=response.body
        content_type=response.headers.get('Content-Type',b'text/html').decode('ascii',errors='ignore')
        try:
            result=extract_code(body,response.url,content_type)
            rendered=None;render_error=None
            if result['codeCount']==0 and CONFIG.get('browserEnabled') and self.rendered<CONFIG.get('maxRendered',1) and any(marker in body.lower() for marker in (b'__next',b'id="root"',b'id="app"',b'enable javascript')):
                self.rendered+=1
                try:
                    rendered=await Renderer(HTTP).render(response.url)
                    result=extract_code(rendered,response.url)
                    result['rendered']=True
                except Exception as exc:render_error=getattr(exc,'code','RENDER_FAILED')
            depth=response.meta.get('seedDepth',0)+response.meta.get('depth',0)
            emit({'type':'page','url':response.url,'requestUrl':response.request.url,'depth':depth,'etag':response.headers.get('ETag',b'').decode(),'lastModified':response.headers.get('Last-Modified',b'').decode(),'raw':base64.b64encode(body).decode(),'renderedRaw':base64.b64encode(rendered).decode() if rendered else None,'renderError':render_error,**result})
            if CONFIG.get('followLinks',True) and depth<CONFIG['maxDepth']:
                for link in self.links.extract_links(response)[:CONFIG.get('maxLinksPerPage',25)]:
                    if self.count<CONFIG['maxPages']:
                        yield scrapy.Request(link.url,callback=self.parse,errback=self.failed,meta={'seedDepth':response.meta.get('seedDepth',0)})
        except Exception as exc:
            emit({'type':'error','url':response.url,'code':getattr(exc,'code','EXTRACTION_FAILED')})


def main():
    global CONFIG,HTTP,POLICY
    CONFIG=json.loads(sys.stdin.read())
    HTTP=PublicHTTP(CONFIG.get('allowedHosts',[]),max_bytes=CONFIG.get('maxPageBytes',2*1024**2))
    POLICY=Robots(HTTP,CONFIG.get('robotsCache',{}))
    if sys.platform=='win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    process=CrawlerProcess(settings={
        'USER_AGENT':'SEED-CodeCrawler/1.0','LOG_ENABLED':True,'LOG_LEVEL':'ERROR','TELNETCONSOLE_ENABLED':False,
        'SEED_CONFIG':CONFIG,
        'TWISTED_REACTOR':'twisted.internet.asyncioreactor.AsyncioSelectorReactor',
        'CONCURRENT_REQUESTS':2,'CONCURRENT_REQUESTS_PER_DOMAIN':1,'DOWNLOAD_DELAY':CONFIG.get('delaySeconds',1),
        'AUTOTHROTTLE_ENABLED':True,'AUTOTHROTTLE_START_DELAY':1,'AUTOTHROTTLE_MAX_DELAY':30,'AUTOTHROTTLE_TARGET_CONCURRENCY':1,
        'DOWNLOAD_TIMEOUT':25,'DEPTH_LIMIT':CONFIG['maxDepth'],'RETRY_ENABLED':False,
        'ROBOTSTXT_OBEY':False, # Explicit urllib.robotparser enforcement in SafeDownloader.
        'COOKIES_ENABLED':False,'HTTPCACHE_ENABLED':False,
        'DOWNLOADER_MIDDLEWARES':{'RAG.Pre.crawling.scrapy_runner.SafeDownloader':900},
        'FEED_EXPORT_ENCODING':'utf-8','WARN_ON_GENERATOR_RETURN_VALUE':False,
    })
    crawler=process.create_crawler(CodeSpider)
    deferred=process.crawl(crawler)
    deferred.addErrback(lambda failure:emit({'type':'error','url':CONFIG['urls'][0]['url'],'code':'SCRAPY_START_FAILED','message':failure.getErrorMessage()[:1000]}))
    process.start(install_signal_handlers=False)
    emit({'type':'finished','robotsCache':getattr(crawler,'seed_robots',POLICY).cache,'stats':crawler.stats.get_stats() if crawler.stats else {}})


if __name__=='__main__':main()
