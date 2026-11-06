"""API discovery, XML/feeds and bounded Common Crawl WARC range retrieval."""
from __future__ import annotations

import asyncio
import gzip
import io
import json
import os
from pathlib import Path
from urllib.parse import urlencode, urlsplit, quote

from warcio.archiveiterator import ArchiveIterator

from RAG.core import DomainError
from .web import SOURCE_EXTENSIONS, canonical, sitemap_urls, feed_urls


class Discovery:
    def __init__(self,transport,robots):self.transport,self.robots=transport,robots;self.snapshots=[];self.cursors={}

    def github_headers(self):
        headers={'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'}
        token_path=os.environ.get('SEED_GITHUB_TOKEN_FILE')
        if token_path:
            token=Path(token_path).read_text(encoding='utf-8').strip()
            if len(token)>4096:raise DomainError('INVALID_GITHUB_TOKEN','GitHub 토큰 파일 크기 오류',422)
            headers['Authorization']='Bearer '+token
        return headers

    async def github(self,query,page=1,max_files=30):
        headers=self.github_headers()
        search='https://api.github.com/search/repositories?'+urlencode({'q':query+' is:public','sort':'updated','order':'desc','per_page':2,'page':min(10,page)})
        result=await self.transport.get(search,headers=headers,max_bytes=1024**2)
        repos=json.loads(result['body']).get('items',[])
        urls=[]
        for repo in repos[:1]:
            if repo.get('private'):continue
            name=repo.get('full_name','')
            if len(name.split('/'))!=2:continue
            if repo.get('fork'):
                repo=json.loads((await self.transport.get('https://api.github.com/repos/'+quote(name,safe='/'),headers=headers,max_bytes=1024**2))['body'])
            urls.extend(await self.repository(repo,max_files))
        return urls

    async def refresh(self,name,max_files=20,tracked_paths=()):
        if len(name.split('/'))!=2:raise DomainError('INVALID_REPOSITORY','저장소 경로 오류',422)
        repo=json.loads((await self.transport.get('https://api.github.com/repos/'+quote(name,safe='/'),headers=self.github_headers(),max_bytes=1024**2))['body'])
        return await self.repository(repo,max_files,tracked_paths)

    async def repository(self,repo,max_files=20,tracked_paths=()):
        if repo.get('private'):raise DomainError('SOURCE_BLOCKED','공개 저장소만 수집합니다.',403)
        name=repo['full_name'];headers=self.github_headers();base='https://api.github.com/repos/'+quote(name,safe='/')
        family=(repo.get('source') or repo.get('parent') or repo).get('full_name',name)
        commit=json.loads((await self.transport.get(base+'/commits/'+quote(repo.get('default_branch','HEAD'),safe=''),headers=headers,max_bytes=1024**2))['body'])
        revision=commit.get('sha')
        if not revision:return []
        tree=json.loads((await self.transport.get(base+'/git/trees/'+revision+'?recursive=1',headers=headers,max_bytes=2*1024**2))['body'])
        if tree.get('truncated'):raise DomainError('INCOMPLETE_REPOSITORY_TREE','불완전한 트리로 파일 삭제를 판단하지 않습니다.',422)
        files=[item for item in tree.get('tree',[]) if item.get('type')=='blob' and item.get('mode') in ('100644','100755')]
        all_paths=[item['path'] for item in files]
        signals={'hasTests':any(p.lower().startswith(('tests/','test/')) or '/test_' in p.lower() or p.lower().endswith(('_test.go','.test.ts','.test.js')) for p in all_paths),
                 'hasReadme':any(Path(p).name.lower().startswith('readme') for p in all_paths)}
        snapshot={'repository':name,'revision':revision,'branch':repo.get('default_branch','HEAD'),'paths':all_paths,'complete':True,'signals':signals}
        urls=[];tracked=set(tracked_paths);eligible=[]
        for item in sorted(files,key=lambda f:(f['path'] not in tracked,f['path'])):
            path=item['path']
            if Path(path).suffix.lower() not in SOURCE_EXTENSIONS:continue
            if any(part in ('node_modules','vendor','dist','.git','build') or part.startswith('.env') for part in path.split('/')):continue
            if path.endswith(('.lock','lock.json','lock.yaml')) or item.get('size',0)>512*1024:continue
            eligible.append(item)
        cursor=self.cursors.get(name,0)%max(1,len(eligible))
        selected=(eligible[cursor:]+eligible[:cursor])[:max_files]
        snapshot['fileCursor']=(cursor+len(selected))%max(1,len(eligible));self.cursors[name]=snapshot['fileCursor']
        self.snapshots.append(snapshot)
        for item in selected:
            path=item['path']
            urls.append({'url':f'https://raw.githubusercontent.com/{name}/{revision}/{quote(path,safe="/")}',
                         'kind':'code','origin':'github_api','repository':name,'path':path,'revision':revision,'blobRevision':item.get('sha'),
                         'licenseId':(repo.get('license') or {}).get('spdx_id'),'sourceUrl':repo.get('html_url'),
                         'repositorySignals':signals,'repositoryUpdatedAt':repo.get('pushed_at'),'language':(repo.get('language') or '').lower(),
                         'sourceKey':f'github:{name}:{path}','repositoryFamily':'https://github.com/'+family})
        return urls

    async def sitemap(self,url,limit=50):
        if not await self.robots.allowed(url):return []
        raw=(await self.transport.get(url,max_bytes=2*1024**2))['body']
        if raw.startswith(b'\x1f\x8b'):
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:raw=stream.read(2*1024**2+1)
            if len(raw)>2*1024**2:raise DomainError('SOURCE_TOO_LARGE','sitemap 압축 상한 초과',413)
        return [{'url':u,'kind':'sitemap' if urlsplit(u).path.endswith(('.xml','.xml.gz')) else 'page','origin':'sitemap'} for u in sitemap_urls(raw,url,limit)]

    async def feed(self,url,limit=30):
        if not await self.robots.allowed(url):return []
        raw=(await self.transport.get(url,max_bytes=1024**2))['body']
        return [{'url':u,'kind':'page','origin':'rss_atom'} for u in feed_urls(raw,url,limit)]

    async def commoncrawl(self,pattern,limit=3):
        indexes=json.loads((await self.transport.get('https://index.commoncrawl.org/collinfo.json',max_bytes=1024**2))['body'])
        if not indexes:return []
        selected=None;result=None
        for index in indexes[:3]:
            endpoint=index.get('cdx-api','')
            parsed=urlsplit(endpoint)
            if parsed.scheme!='https' or parsed.hostname!='index.commoncrawl.org':raise DomainError('INVALID_ARCHIVE_INDEX','잘못된 Common Crawl 인덱스',422)
            query=endpoint+'?'+urlencode({'url':pattern,'output':'json','filter':'status:200','collapse':'urlkey','limit':min(10,limit)})
            result=await self.transport.get(query,max_bytes=1024**2,allow_status=(404,))
            # CDX uses 404 for no matching captures; try bounded older indexes.
            if result['status']!=404:
                selected=index;break
        if selected is None:return []
        records=[]
        for line in result['body'].splitlines()[:limit]:
            try:
                row=json.loads(line);url=canonical(row['url'])
                filename=row['filename'];offset=int(row['offset']);length=int(row['length'])
                if not filename.startswith('crawl-data/') or '..' in filename.split('/') or offset<0 or not 0<length<=2*1024**2:continue
                records.append({'url':url,'kind':'archive','origin':'commoncrawl','archive':{'filename':filename,'offset':offset,'length':length,'timestamp':row.get('timestamp'),'index':selected.get('id')}})
            except (KeyError,ValueError,DomainError):continue
        return records

    async def warc(self,item):
        if not await self.robots.allowed(item['url']):raise DomainError('ROBOTS_DENIED','보관 자료의 원 사이트 수집 정책을 확인하세요.',403)
        info=item['archive'];offset=int(info['offset']);length=int(info['length'])
        filename=info['filename']
        if not filename.startswith('crawl-data/') or '..' in filename.split('/') or not 0<length<=2*1024**2 or offset<0:raise DomainError('INVALID_WARC_RANGE','잘못된 WARC 범위',422)
        result=await self.transport.get('https://data.commoncrawl.org/'+filename,headers={'Range':f'bytes={offset}-{offset+length-1}'},max_bytes=length)
        if result['status']!=206:raise DomainError('ARCHIVE_RANGE_REQUIRED','전체 WARC 다운로드는 허용하지 않습니다.',422)
        raw=result['body']
        if raw.startswith(b'\x1f\x8b'):
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:raw=stream.read(3*1024**2+1)
        if len(raw)>3*1024**2:raise DomainError('SOURCE_TOO_LARGE','WARC 압축 해제 상한 초과',413)
        for record in ArchiveIterator(io.BytesIO(raw)):
            if record.rec_type!='response':continue
            if canonical(record.rec_headers.get_header('WARC-Target-URI'))!=canonical(item['url']):continue
            body=record.content_stream().read(2*1024**2+1)
            if len(body)>2*1024**2:raise DomainError('SOURCE_TOO_LARGE','보관 문서 상한 초과',413)
            return {'body':body,'url':item['url'],'headers':{'content-type':record.http_headers.get_header('Content-Type') or 'text/html'}}
        raise DomainError('WARC_DOCUMENT_MISSING','보관 레코드와 대상 URL이 일치하지 않습니다.',422)
