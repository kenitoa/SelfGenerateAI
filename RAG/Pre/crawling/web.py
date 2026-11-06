"""HTTPX, robots, HTML extraction, feeds, sitemaps and optional rendering.

All network paths use one DNS-pinned, public-address-only HTTP transport.
Library text parsers never make their own network requests.
"""
from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import re
import time
from pathlib import PurePosixPath
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import httpx
from defusedxml import ElementTree
from selectolax.lexbor import LexborHTMLParser
import feedparser
import trafilatura

from RAG.core import DomainError

USER_AGENT = 'SEED-CodeCrawler/1.0'
CODE_EXTENSIONS = {'.py','.js','.jsx','.ts','.tsx','.sql','.html','.css','.rs','.go','.c','.cpp','.cc','.h','.hpp','.cs','.java','.kt','.kts','.swift','.m','.mm','.rb','.php','.sh','.bash','.ps1','.r','.jl','.lua','.dart','.scala','.ex','.exs','.erl','.hrl','.hs','.clj','.cljs','.vue','.svelte','.graphql','.gql','.proto','.zig','.nim','.pl','.pm','.f90','.f95','.lisp','.el','.scm','.rkt','.ml','.mli','.fs','.fsx','.vb','.v','.sv','.vhd','.vhdl','.asm','.s','.bat','.cmd','.groovy','.gradle','.sol','.move','.nix','.tf','.hcl','.cmake','.make','.json','.jsonl','.yml','.yaml','.toml','.md','.txt'}
SOURCE_EXTENSIONS = CODE_EXTENSIONS - {'.html','.md','.txt','.json','.jsonl','.yml','.yaml','.toml'}


def canonical(url, base=None):
    if not isinstance(url,str) or len(url)>4096:
        raise DomainError('INVALID_CRAWL_URL','수집 URL 길이·형식을 확인하세요.',422)
    parsed=urlsplit(urljoin(base,url) if base else url)
    try:port=parsed.port
    except ValueError:raise DomainError('SOURCE_BLOCKED','잘못된 포트입니다.',422)
    if parsed.scheme not in ('http','https') or not parsed.hostname or parsed.username or parsed.password or port not in (None,80,443):
        raise DomainError('SOURCE_BLOCKED','공개 HTTP/HTTPS 주소만 허용합니다.',403)
    host=parsed.hostname.rstrip('.').lower().encode('idna').decode()
    authority=f'[{host}]' if ':' in host else host
    if port and port != (443 if parsed.scheme=='https' else 80):authority+=f':{port}'
    return urlunsplit((parsed.scheme,authority,parsed.path or '/',parsed.query,''))


class PublicHTTP:
    def __init__(self, allowed_hosts=(), max_bytes=2*1024**2, client=None, guard=None):
        self.allowed={str(v).lower() for v in allowed_hosts}
        self.max_bytes=max_bytes
        self.client=client
        self.guard=guard

    async def get(self,url,headers=None,max_bytes=None,allow_status=()):
        if self.guard:self.guard()
        try:
            async with asyncio.timeout(30):
                result=await self._get(url,headers,max_bytes,allow_status)
        except TimeoutError as exc:raise DomainError('SOURCE_TIMEOUT','수집 전체 요청 시간 초과',503) from exc
        if self.guard:self.guard()
        return result

    async def _get(self,url,headers=None,max_bytes=None,allow_status=()):
        import socket
        current=canonical(url);maximum=max_bytes or self.max_bytes
        credential_origin=urlsplit(current).netloc
        async with httpx.AsyncClient(trust_env=False,follow_redirects=False,timeout=httpx.Timeout(20,connect=8),limits=httpx.Limits(max_connections=2,max_keepalive_connections=0)) if self.client is None else _Borrowed(self.client) as client:
            for redirect in range(4):
                parsed=urlsplit(current);host=parsed.hostname
                if self.allowed and host not in self.allowed:
                    raise DomainError('SOURCE_BLOCKED','허용하지 않은 수집 호스트입니다.',403)
                try:
                    answers=await asyncio.to_thread(socket.getaddrinfo,host,parsed.port or (443 if parsed.scheme=='https' else 80),0,socket.SOCK_STREAM)
                    addresses=list(dict.fromkeys(row[4][0] for row in answers))
                except OSError as exc:raise DomainError('SOURCE_UNAVAILABLE','수집 호스트 확인 실패',503) from exc
                if not addresses or any(not ipaddress.ip_address(ip).is_global for ip in addresses):
                    raise DomainError('SOURCE_BLOCKED','사설·로컬·메타데이터 주소 수집을 차단했습니다.',403)
                destination=httpx.URL(current).copy_with(host=addresses[0])
                outgoing={'User-Agent':USER_AGENT,'Accept-Encoding':'identity',**(headers or {}),'Host':parsed.netloc}
                if parsed.netloc!=credential_origin:
                    outgoing={k:v for k,v in outgoing.items() if k.lower() not in ('authorization','cookie','proxy-authorization')}
                try:
                    async with client.stream('GET',destination,headers=outgoing,extensions={'sni_hostname':host}) as response:
                        if response.status_code in (301,302,303,307,308):
                            if redirect==3 or 'location' not in response.headers:raise DomainError('SOURCE_BLOCKED','리다이렉트 한도 초과',422)
                            current=canonical(response.headers['location'],current)
                            continue
                        if response.status_code in (429,503) or (response.status_code==403 and response.headers.get('x-ratelimit-remaining')=='0'):
                            raise DomainError('CRAWL_RATE_LIMIT','수집 서버가 대기를 요청했습니다.',503,{'retryAfter':response.headers.get('retry-after'),'url':current})
                        if response.status_code not in (200,206,*allow_status):
                            raise DomainError('SOURCE_HTTP_ERROR',f'수집 응답 오류 ({response.status_code})',422,{'httpStatus':response.status_code})
                        if int(response.headers.get('content-length') or 0)>maximum:raise DomainError('SOURCE_TOO_LARGE','수집 문서 상한 초과',413)
                        data=bytearray()
                        async for part in response.aiter_bytes():
                            data.extend(part)
                            if len(data)>maximum:raise DomainError('SOURCE_TOO_LARGE','압축 해제 후 수집 문서 상한 초과',413)
                        return {'url':current,'status':response.status_code,'headers':dict(response.headers),'body':bytes(data)}
                except DomainError:raise
                except (httpx.HTTPError,ValueError) as exc:raise DomainError('SOURCE_UNAVAILABLE','수집 연결 실패',503) from exc
        raise DomainError('SOURCE_BLOCKED','리다이렉트 오류',422)


class _Borrowed:
    def __init__(self,client):self.client=client
    async def __aenter__(self):return self.client
    async def __aexit__(self,*args):pass


class Robots:
    def __init__(self,transport,cache=None):
        self.transport=transport;self.cache=cache if cache is not None else {};self.delays={};self.sitemaps={}

    async def allowed(self,url):
        url=canonical(url);parsed=urlsplit(url);origin=f'{parsed.scheme}://{parsed.netloc}'
        cached=self.cache.get(origin)
        if not cached or time.time()-cached.get('fetchedAt',0)>86400:
            try:
                result=await self.transport.get(origin+'/robots.txt',max_bytes=512*1024,allow_status=(401,403,404,410))
                cached={'fetchedAt':time.time(),'status':result['status'],'text':result['body'].decode('utf-8',errors='replace')}
            except DomainError:
                # A network error must never silently grant crawl permission.
                return False
            self.cache[origin]=cached
        status=cached['status']
        if status in (401,403):return False
        if status in (404,410):return True
        parser=RobotFileParser();parser.set_url(origin+'/robots.txt');parser.parse(cached['text'].splitlines())
        self.delays[origin]=max(1.0,float(parser.crawl_delay(USER_AGENT) or parser.crawl_delay('*') or 1))
        self.sitemaps[origin]=parser.site_maps() or []
        return parser.can_fetch(USER_AGENT,url)


def sitemap_urls(raw,base,limit=100):
    try:root=ElementTree.fromstring(raw)
    except Exception as exc:raise DomainError('INVALID_SITEMAP','잘못된 sitemap XML입니다.',422) from exc
    result=[]
    for node in root.iter():
        if node.tag.rsplit('}',1)[-1]=='loc' and node.text:
            try:result.append(canonical(node.text.strip(),base))
            except DomainError:continue
            if len(result)>=limit:break
    return list(dict.fromkeys(result))


def feed_urls(raw,base,limit=100):
    parsed=feedparser.parse(raw)
    links=[]
    for entry in parsed.entries[:limit]:
        if entry.get('link'):
            try:links.append(canonical(entry.link,base))
            except DomainError:pass
    return list(dict.fromkeys(links))


def extract_code(raw,url,content_type='text/html'):
    suffix=PurePosixPath(urlsplit(url).path).suffix.lower()
    if suffix in SOURCE_EXTENSIONS and 'html' not in content_type:
        text=raw.decode('utf-8-sig')
        if '\x00' in text:raise DomainError('BINARY_NOT_ALLOWED','바이너리 코드는 수집하지 않습니다.',422)
        return {'title':PurePosixPath(urlsplit(url).path).name,'text':text,'name':PurePosixPath(urlsplit(url).path).name,'codeCount':1,'links':[],'discovery':[],'extractor':'raw-code'}
    html=raw.decode('utf-8',errors='replace')
    tree=LexborHTMLParser(html)
    title=tree.css_first('title');title=title.text(strip=True) if title else url
    snippets=[];hashes=set()
    for node in tree.css('pre')[:80]:
        for br in node.css('br'):br.replace_with('\n')
        code=node.text(separator='',strip=False)
        if len(code.strip())<12 or len(code.encode())>64*1024:continue
        hashed=hashlib.sha256(code.encode()).hexdigest()
        if hashed in hashes:continue
        hashes.add(hashed)
        child=node.css_first('code');attributes=(child.attributes if child else node.attributes)
        language=re.search(r'(?:language-|lang-)([a-zA-Z0-9_+-]+)',attributes.get('class',''))
        fence='`'*(max([len(m.group()) for m in re.finditer(r'`{3,}',code)]+[2])+1)
        snippets.append(f'{fence}{language.group(1) if language else ""}\n{code}\n{fence}')
    discovery=[];links=[]
    for node in tree.css('a[href]')[:500]:
        try:links.append(canonical(node.attributes['href'],url))
        except DomainError:pass
    for node in tree.css('link[href]')[:50]:
        kind=node.attributes.get('type','').lower()
        if any(t in kind for t in ('rss','atom','feed+json')):
            try:discovery.append({'kind':'feed','url':canonical(node.attributes['href'],url)})
            except DomainError:pass
    summary=trafilatura.extract(html,include_comments=False,include_tables=False,include_links=False,favor_precision=True) or ''
    text=f'# {title}\n\nSource: {url}\n\n'+summary[:1800]+'\n\n'+'\n\n'.join(snippets)
    return {'title':title[:300],'name':'page.md','text':text,'codeCount':len(snippets),'links':list(dict.fromkeys(links)),'discovery':discovery,'extractor':'selectolax+trafilatura'}


class Renderer:
    """No browser profile, cookies, local files or unvalidated browser networking."""
    def __init__(self,transport):self.transport=transport

    async def render(self,url):
        from playwright.async_api import async_playwright, Error
        requests=0;total=0
        async with async_playwright() as driver:
            try:browser=await driver.chromium.launch(headless=True,args=['--disable-background-networking','--disable-extensions','--disable-quic'])
            except Error as exc:raise DomainError('BROWSER_UNAVAILABLE','Playwright Chromium 설치 상태를 확인하세요.',503) from exc
            try:
                context=await browser.new_context(service_workers='block',accept_downloads=False,permissions=[])
                async def close_socket(route):await route.close()
                await context.route_web_socket('**/*',close_socket)
                await context.add_init_script('window.RTCPeerConnection = undefined; window.webkitRTCPeerConnection = undefined;')
                async def route_request(route):
                    nonlocal requests,total
                    request=route.request
                    if request.method!='GET' or request.resource_type in ('image','media','font','websocket') or requests>=25:
                        await route.abort();return
                    requests+=1
                    try:
                        result=await self.transport.get(request.url,max_bytes=2*1024**2)
                        total+=len(result['body'])
                        if total>8*1024**2:await route.abort();return
                        headers={k:v for k,v in result['headers'].items() if k.lower() in ('content-type','cache-control','content-security-policy')}
                        await route.fulfill(status=result['status'],headers=headers,body=result['body'])
                    except Exception:await route.abort()
                await context.route('**/*',route_request)
                page=await context.new_page()
                await page.goto(canonical(url),wait_until='domcontentloaded',timeout=20000)
                await page.wait_for_timeout(500)
                html=(await page.content()).encode()
                if len(html)>2*1024**2:raise DomainError('SOURCE_TOO_LARGE','렌더링 문서 상한 초과',413)
                return html
            finally:await browser.close()
