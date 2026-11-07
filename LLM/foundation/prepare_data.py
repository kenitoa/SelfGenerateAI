"""Auditable local preparation of explicitly delegated, public official sources.

Never opens a NAS database for writing, deploys code, or executes collected code.
The catalogue declares family-level splits before tokenizer/model work. License
text, immutable revision, archive hash, and per-span hashes remain beside data.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import time
import urllib.error
import urllib.request
import zipfile

from LLM.core import atomic_json, canonical, digest, file_hash, now, require, uid

ACTOR = 'user-delegated-public-source-audit-v1'
POLICY = 'official-permissive-source-v1'
CATALOG = {
 'train': ['django/django','numpy/numpy','scipy/scipy','pandas-dev/pandas','sympy/sympy',
    'sqlalchemy/sqlalchemy','psf/requests','encode/httpx','pydantic/pydantic','fastapi/fastapi',
    'pytest-dev/pytest','pallets/flask','python-attrs/attrs','python/mypy','pypa/pip',
    'pypa/setuptools','python-poetry/poetry','ipython/ipython','PyCQA/isort','Textualize/rich'],
 'dev': ['pallets/werkzeug','pallets/jinja','pallets/click','pallets/itsdangerous','pallets/markupsafe',
    'encode/httpcore','agronholm/anyio','encode/starlette','urllib3/urllib3','pypa/packaging',
    'pypa/virtualenv','pypa/build','pypa/installer','pypa/twine','python-poetry/poetry-core',
    'pytest-dev/pluggy','pytest-dev/iniconfig','hynek/structlog','psf/black','tornadoweb/tornado'],
 'release': ['python-trio/trio','jmespath/jmespath.py'],
 'audit': ['pallets/quart','dateutil/dateutil'],
}
EXCLUDE_PARTS = frozenset({'vendor','vendored','_vendor','_vendored','third_party','third-party',
    'extern','externals','external','node_modules','dist','build','.git','licenses','license',
    'tests','test','testing','benchmarks','fixtures','_generated','generated'})
EXTENSIONS = frozenset({'.py','.pyi','.rs','.js','.ts','.tsx','.c','.h','.cpp','.rst','.md'})
MAX_ARCHIVE_BYTES = 256*1024**2
MAX_MEMBER_BYTES = 2*1024**2
SPAN_BYTES = 60*1024
LICENSE_NAMES = ('LICENSE','LICENSE.txt','LICENSE.rst','LICENSE.md','LICENSE-MIT','LICENSE-APACHE',
                 'LICENSE.BSD','LICENSE.APACHE','LICENSE.MIT','LICENSE.APACHE2','COPYING','COPYING.txt',
                 'NOTICE','NOTICE.txt','COPYRIGHT','AUTHORS','AUTHORS.md','AUTHORS.txt','AUTHORS.rst')


def request(url):
    require(url.startswith(('https://api.github.com/','https://codeload.github.com/','https://raw.githubusercontent.com/')),
            'OFFICIAL_SOURCE_URL_REQUIRED')
    return urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent':'SelfGenerateAI-public-source-audit/1',
        'Accept':'application/vnd.github+json'}), timeout=45)


def identify_license(text):
    normalized=' '.join(text.lower().split())
    if all(s in normalized for s in ('permission is hereby granted, free of charge','deal in the software without restriction','the software is provided "as is"')):
        return 'MIT'
    if 'apache license' in normalized and 'version 2.0' in normalized and 'grant of patent license' in normalized:
        return 'Apache-2.0'
    if 'redistribution and use in source and binary forms' in normalized and 'this software is provided' in normalized:
        return 'BSD-3-Clause' if 'neither the name' in normalized else 'BSD-2-Clause'
    return None


def _spans(text):
    lines=text.splitlines(keepends=True);pending=[];size=0;start=1
    for index,line in enumerate(lines,1):
        amount=len(line.encode())
        if amount>SPAN_BYTES:
            if pending:yield start,index-1,''.join(pending);pending=[];size=0
            start=index+1;continue
        if pending and size+amount>SPAN_BYTES:
            yield start,index-1,''.join(pending);pending=[];size=0;start=index
        if not pending:start=index
        pending.append(line);size+=amount
    if pending:yield start,len(lines),''.join(pending)


def collect_one(root, repo, split):
    require(re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',repo),'INVALID_REPOSITORY')
    destination=root/'sources'/repo.replace('/','--');destination.mkdir(parents=True,exist_ok=True)
    existing=destination/'receipt.json'
    if existing.exists():
        receipt=json.loads(existing.read_text('utf-8'))
        if receipt.get('state')=='rights_verified' and (destination/'records.jsonl').exists():return receipt
    with request('https://api.github.com/repos/'+repo) as response:metadata=json.load(response)
    canonical_repo=metadata['full_name'];family=metadata.get('source',{}).get('full_name') or canonical_repo
    require(family.lower()==canonical_repo.lower() or metadata.get('fork') is True,'FORK_LINEAGE_UNRESOLVED')
    # Git's immutable HEAD protocol avoids spending an API request per commit.
    refs=subprocess.run(['git','ls-remote','https://github.com/'+canonical_repo+'.git','HEAD'],
                        capture_output=True,text=True,timeout=45,check=True).stdout
    revision=refs.split()[0];require(re.fullmatch(r'[0-9a-f]{40}',revision),'IMMUTABLE_REVISION_REQUIRED')
    atomic_json(destination/'repository.json',metadata)
    archive=destination/(revision+'.zip');url='https://codeload.github.com/'+canonical_repo+'/zip/'+revision
    if not archive.exists():
        temporary=archive.with_suffix('.part');size=0
        with request(url) as response,temporary.open('wb') as target:
            while chunk:=response.read(1024**2):
                size+=len(chunk);require(size<=MAX_ARCHIVE_BYTES,'ARCHIVE_BYTE_LIMIT');target.write(chunk)
        temporary.replace(archive)
    archive_hash=file_hash(archive)
    licenses=[];excluded=Counter();records=0;total=0
    maximum=8*1024**2 if split=='train' else 1024**2
    with zipfile.ZipFile(archive) as package:
        infos=package.infolist();prefix=infos[0].filename.split('/')[0]+'/'
        members={i.filename.removeprefix(prefix):i for i in infos if i.filename.startswith(prefix)}
        for name in LICENSE_NAMES:
            if name not in members or members[name].file_size>128*1024:continue
            raw=package.read(members[name]);text=raw.decode('utf-8-sig')
            license_id=identify_license(text) if name.upper().startswith(('LICENSE','COPYING')) else None
            (destination/name.replace('/','--')).write_bytes(raw)
            licenses.append({'path':name,'sha256':digest(raw),'licenseId':license_id,
                'url':f'https://github.com/{canonical_repo}/blob/{revision}/{name}'})
        allowed=[row['licenseId'] for row in licenses if row['licenseId']]
        require(allowed,'PERMISSIVE_LICENSE_EVIDENCE_REQUIRED')
        # A nested license marks a different rights scope. Exclude it unless a
        # later explicit per-file audit resolves it; no inherited blanket grant.
        nested_scopes={str(PurePosixPath(name).parent) for name in members
            if PurePosixPath(name).name.upper().startswith(('LICENSE','COPYING')) and len(PurePosixPath(name).parts)>1}
        output=destination/'records.jsonl'
        with output.open('w',encoding='utf-8',newline='\n') as handle:
            for name in sorted(members):
                info=members[name];path=PurePosixPath(name)
                if info.is_dir() or path.suffix.lower() not in EXTENSIONS:continue
                if path.is_absolute() or '..' in path.parts or any(p.lower() in EXCLUDE_PARTS for p in path.parts):continue
                if any(name.startswith(scope+'/') for scope in nested_scopes):excluded['NESTED_LICENSE_SCOPE']+=1;continue
                if info.file_size>MAX_MEMBER_BYTES:excluded['SOURCE_FILE_TOO_LARGE']+=1;continue
                if (info.external_attr>>16)&0o170000==0o120000:continue
                raw=package.read(info)
                try:text=raw.decode('utf-8-sig')
                except UnicodeError:excluded['INVALID_UTF8']+=1;continue
                # Explicitly foreign SPDX declarations never inherit root rights.
                spdx=re.findall(r'SPDX-License-Identifier:\s*([^\r\n*]+)',text)
                if any(not all(part in {'MIT','Apache-2.0','BSD-2-Clause','BSD-3-Clause','OR','AND','(',')'}
                    for part in declaration.replace('(',' ( ').replace(')',' ) ').split()) for declaration in spdx):
                    excluded['FOREIGN_FILE_LICENSE']+=1;continue
                from .corpus import secret_reasons
                for start,end,span in _spans(text):
                    amount=len(span.encode())
                    if not span.strip() or amount<128:continue
                    if secret_reasons(span):excluded['SECRET_OR_PERSONAL_VALUE']+=1;continue
                    if total+amount>maximum:excluded['GROUP_BYTE_QUOTA']+=1;continue
                    checksum=digest(span.encode());document_id=digest([canonical_repo,revision,name,start,end])
                    row={'id':document_id,'documentId':document_id,'versionId':document_id,'sourceId':'official:'+canonical_repo,
                        'groupId':'repository:https://github.com/'+family,'repositoryFamily':'https://github.com/'+family,
                        'text':span,'language':path.suffix.lower().lstrip('.'),'trainingPolicy':'allowed','policyVersion':POLICY,
                        'targetOrigin':'source_span','ragSplit':split,'split':split,
                        'sourceSpan':{'path':name,'startLine':start,'endLine':end,'sourceFileSha256':digest(raw),'contentSha256':checksum},
                        'provenance':{'repository':canonical_repo,'family':family,'revision':revision,'archiveSha256':archive_hash,
                            'url':f'https://github.com/{canonical_repo}/blob/{revision}/{name}#L{start}-L{end}'},
                        'permissionReceipt':{'actor':ACTOR,'policy':POLICY,'decision':'allowed','licenseEvidence':licenses,
                            'delegation':'user requested public official licensed data preparation','reviewedAt':now()}}
                    handle.write(canonical(row)+'\n');total+=amount;records+=1
        archive_hash=file_hash(archive)
    receipt={'schemaVersion':1,'repository':canonical_repo,'repositoryFamily':family,'fork':metadata.get('fork') is True,
        'revision':revision,'archiveUrl':url,'archiveSha256':archive_hash,'licenses':licenses,'licenseIds':sorted(set(allowed)),
        'split':split,'splitPlanning':'declared before any tokenizer/model training; family-level 20 train /20 dev plus independent release/audit',
        'actor':ACTOR,'policy':POLICY,'state':'rights_verified','rawBytes':total,'recordCount':records,
        'recordsSha256':file_hash(destination/'records.jsonl'),'exclusions':dict(excluded),'observedAt':now()}
    atomic_json(existing,receipt);return receipt


def collect(root, workers=4):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    atomic_json(root/'split-plan.json',{'actor':ACTOR,'declaredAt':now(),'catalog':CATALOG,'modelTrainingPerformed':False})
    results=[]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures={executor.submit(collect_one,root,repo,split):(repo,split) for split,repos in CATALOG.items() for repo in repos}
        for future in as_completed(futures):
            repo,split=futures[future]
            try:result=future.result()
            except Exception as exc:result={'repository':repo,'split':split,'state':'held','reasonCodes':[getattr(exc,'code',type(exc).__name__)]}
            results.append(result);print(canonical({k:result.get(k) for k in ('repository','split','state','rawBytes','recordCount','reasonCodes')}),flush=True)
            atomic_json(root/'collection-report.json',{'actor':ACTOR,'sources':sorted(results,key=lambda r:r['repository']),
                'observedAt':now(),'modelTrainingPerformed':False})
    return results


def _jsonl(path):
    with Path(path).open('rb') as handle:
        while line:=handle.readline(2*1024**2+1):
            require(len(line)<=2*1024**2,'PREPARATION_ROW_TOO_LARGE')
            if line.strip():yield json.loads(line)


def ensure_attributions(root,rag):
    """Retain primary NOTICE/COPYRIGHT/AUTHORS evidence without rewriting grants."""
    count=0
    for rights in list(rag.list('source_rights_receipt')):
        if rights.get('approvalKind')!='delegated_primary_license_review':continue
        source=rag.get('source',rights['sourceId']);repo=rights['repository']
        candidates=[p for p in (Path(root)/'sources').glob('*/receipt.json') if json.loads(p.read_text('utf-8')).get('repository')==repo]
        if not candidates:continue
        directory=candidates[0].parent;archive=directory/(rights['revision']+'.zip');evidence=[]
        with zipfile.ZipFile(archive) as package:
            for member in package.infolist():
                parts=member.filename.split('/')
                if len(parts)!=2 or not parts[-1].upper().startswith(('NOTICE','COPYRIGHT','AUTHORS')):continue
                require(member.file_size<=2*1024**2,'ATTRIBUTION_EVIDENCE_TOO_LARGE')
                raw=package.read(member);relative='rights/'+source['id']+'/'+parts[-1]
                path=rag.root/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
                evidence.append({'path':parts[-1],'relativePath':relative,'sha256':digest(raw),
                    'url':f"https://github.com/{repo}/blob/{rights['revision']}/{parts[-1]}",'licenseId':None})
        if evidence and not source.get('attributionReceiptIds'):
            supplement=rag.put('source_rights_receipt',{**{k:v for k,v in rights.items() if k not in ('id','createdAt','updatedAt')},
                'approvalKind':'attribution_supplement','supplementsReceiptId':rights['id'],
                'licenseEvidence':[ *rights['licenseEvidence'],*evidence],'sourceGrantChanged':False,'observedAt':now()})
            rag.put('source',{'attributionReceiptIds':[supplement['id']]},source['id'])
        count+=len(evidence)
    return count


def audit_rag_data(root):
    import apsw
    root=Path(root);connection=apsw.Connection(str(root/'metadata'/'seed.db'),flags=apsw.SQLITE_OPEN_READONLY)
    connection.set_busy_timeout(5000)
    try:
        data={kind:[json.loads(r[0]) for r in connection.execute('SELECT body FROM records WHERE kind=?',(kind,))]
            for kind in ('source','document','version','training_example','training_snapshot')}
        sources={r['id']:r for r in data['source']};versions={r['id']:r for r in data['version']};eligible=[]
        for document in data['document']:
            source=sources.get(document.get('sourceId'),{});version=versions.get(document.get('currentVersionId'),{})
            if source.get('trainingPolicy')=='allowed' and source.get('retrievalPolicy')=='approved' and not document.get('deletedAt') and document.get('enabled',True) and document.get('trainingPolicy')!='denied' and document.get('retrievalPolicy')=='approved' and version.get('qualityStatus')=='passed':eligible.append(version)
        raw=0
        for version in eligible:
            checksum=version.get('rawBlobHash','');path=root/'blobs'/checksum[:2]/checksum
            if path.is_file():raw+=path.stat().st_size
        return {'sourceCount':len(sources),'documentCount':len(data['document']),'trainingPolicyCounts':dict(Counter(r.get('trainingPolicy','unknown') for r in sources.values())),
            'qualityCounts':dict(Counter(r.get('qualityStatus','unknown') for r in versions.values())),
            'eligibleCurrentDocuments':len(eligible),'eligibleRawBytes':raw,'sftExamples':len(data['training_example']),
            'approvedSft':sum(r.get('reviewStatus')=='approved' and r.get('permissionStatus')=='allowed' for r in data['training_example']),
            'readySnapshots':sum(r.get('status')=='ready' for r in data['training_snapshot']),'accessMode':'read_only','observedAt':now()}
    finally:connection.close()


def _ingest(root, rag, pre):
    from RAG.Pre.processing import quality, normalize, digest as rag_digest, text_language
    source_ids=[];origin_roles={};manifest=[];changed=False
    for path in sorted((root/'sources').glob('*/receipt.json')):
        receipt=json.loads(path.read_text('utf-8'))
        if receipt.get('state')!='rights_verified':continue
        directory=path.parent
        require(file_hash(directory/'records.jsonl')==receipt['recordsSha256'],'PREPARATION_HASH_MISMATCH')
        identity=digest([receipt['repository'],receipt['revision'],POLICY])
        existing=next((r for r in rag.list('source_rights_receipt') if r.get('preparationIdentity')==identity),None)
        if existing:source=rag.get('source',existing['sourceId'])
        else:
            changed=True
            source=pre.create_source({'kind':'git_repository','name':receipt['repository'],
                'url':'https://github.com/'+receipt['repository'],'trainingPolicy':'unknown','retrievalPolicy':'pending'})
            rights=root/'rag'/'rights'/source['id'];rights.mkdir(parents=True,exist_ok=True)
            evidence=[]
            for item in receipt['licenses']:
                data=(directory/item['path'].replace('/','--')).read_bytes()
                require(digest(data)==item['sha256'],'LICENSE_HASH_MISMATCH')
                name=item['path'].replace('/','--');(rights/name).write_bytes(data)
                evidence.append({**item,'relativePath':(rights/name).relative_to(rag.root).as_posix()})
            for name in ('repository.json','receipt.json'):
                data=(directory/name).read_bytes();(rights/name).write_bytes(data)
                evidence.append({'path':name,'sha256':digest(data),'relativePath':(rights/name).relative_to(rag.root).as_posix()})
            audited=rag.put('source_rights_receipt',{'preparationIdentity':identity,'sourceId':source['id'],
                'actor':ACTOR,'policy':POLICY,'decision':'allowed','licenseIds':receipt['licenseIds'],'licenseEvidence':evidence,
                'state':'rights_verified','licenses':[item for item in evidence if item.get('licenseId')],
                'repository':receipt['repository'],'repositoryFamily':receipt['repositoryFamily'],'revision':receipt['revision'],
                'archiveSha256':receipt['archiveSha256'],'primaryRepositoryMetadata':receipt['repository'],
                'delegation':'User explicitly delegated collection and permission review of official publicly licensed corpus',
                'approvalKind':'delegated_primary_license_review','observedAt':now()})
            source=rag.put('source',{'trainingPolicy':'allowed','retrievalPolicy':'approved','reviewedBy':ACTOR,
                'reviewedAt':now(),'reviewReason':'Primary immutable LICENSE + repository archive verified; delegated machine review, not human approval',
                'rightsReceiptId':audited['id'],'licenseId':' OR '.join(receipt['licenseIds']),
                'licenseUrl':receipt['licenses'][0]['url'],'repositoryFamily':'https://github.com/'+receipt['repositoryFamily'],
                'lastSuccessfulRevision':receipt['revision'],'policyVersion':2},source['id'])
        source_ids.append(source['id']);group='repository:'+source['repositoryFamily']
        origin_roles[group]={'dev':'validation','release':'test'}.get(receipt['split'],receipt['split'])
        with rag.tx():
            for row in _jsonl(directory/'records.jsonl'):
                document_id=row['documentId'];version_id=row['versionId']
                if rag.one("SELECT id FROM records WHERE kind='version' AND id=?",(version_id,)):continue
                changed=True
                text=row['text'];status,reasons=quality(text);normalized,_=normalize(text,row['language'])
                raw_hash=pre._blob(text);unit_key=row['sourceSpan']['path']+':'+str(row['sourceSpan']['startLine'])+'-'+str(row['sourceSpan']['endLine'])
                unit_id=digest([source['id'],unit_key])
                rag.put('source_unit',{'sourceId':source['id'],'unitKey':unit_key,'unitKind':'verified_source_span',
                    'rawBlobHash':raw_hash,'currentSnapshotHash':raw_hash,'revision':receipt['revision'],'sourceSpan':row['sourceSpan']},unit_id)
                rag.put('document',{'name':row['sourceSpan']['path'],'datasetId':source['datasetId'],'sourceId':source['id'],
                    'sourceUnitId':unit_id,'externalKey':row['sourceSpan']['path']+':'+str(row['sourceSpan']['startLine']),
                    'path':row['sourceSpan']['path'],'enabled':True,'deletedAt':None,'currentVersionId':version_id,
                    'retrievalPolicy':'approved','trainingPolicy':'allowed','policyVersion':1,'state':'ingested'},document_id)
                rag.put('version',{'documentId':document_id,'sourceId':source['id'],'rawBlobHash':raw_hash,
                    'normalizedBlobHash':pre._blob(normalized),'rawFileBlobHash':raw_hash,'contentHash':rag_digest(text),
                    'normalizedHash':rag_digest(normalized),'textLength':len(normalized),'revision':receipt['revision'],
                    'language':{'.py':'python','py':'python','pyi':'python','rs':'rust','js':'javascript','ts':'typescript','tsx':'typescript'}.get(row['language'],row['language']),
                    'textLanguage':text_language(normalized),'qualityStatus':status,'qualityReasons':reasons,
                    'parserVersion':'official-source-span-v1','recordPositions':[],'sourceSpan':row['sourceSpan'],
                    'provenance':row['provenance'],'targetOrigin':'source_span','rightsReceiptId':source.get('rightsReceiptId')},version_id)
        manifest.append({'sourceId':source['id'],'groupId':group,'split':receipt['split'],'rightsReceiptId':source.get('rightsReceiptId')})
        print(canonical({'stage':'rag_ingested','repository':receipt['repository'],'sourceId':source['id']}),flush=True)
    rag.set_setting('trainingStableOrigins',origin_roles)
    if changed:rag.set_setting('policyVersion',int(rag.setting('policyVersion',0))+1)
    atomic_json(root/'reports'/'source-map.json',{'actor':ACTOR,'sources':manifest,'nasModified':False})
    return source_ids,manifest


def _approved_raw_rows(rag,pre,source_ids,*,selected_version_ids=None):
    sources={identity:rag.get('source',identity) for identity in source_ids};cursor=''
    selected=set(selected_version_ids) if selected_version_ids is not None else None
    epochs=(rag.setting('policyVersion',0),rag.setting('trainingDataRevision',0))
    while True:
        page=rag.read("SELECT d.id,d.body document,v.body version FROM records d JOIN records v ON v.kind='version' AND v.id=json_extract(d.body,'$.currentVersionId') WHERE d.kind='document' AND d.id>? ORDER BY d.id LIMIT 256",(cursor,))
        if not page:break
        for item in page:
            document=json.loads(item['document']);version=json.loads(item['version']);source=sources.get(document['sourceId'])
            if selected is not None and version['id'] not in selected:continue
            if not source or version.get('qualityStatus')!='passed':continue
            require(epochs==(rag.setting('policyVersion',0),rag.setting('trainingDataRevision',0)),'POLICY_STATE_CHANGED')
            try:text=pre.content(version['id'])
            except Exception:continue
            group='repository:'+source['repositoryFamily'];role=rag.setting('trainingStableOrigins',{}).get(group)
            yield {'documentId':document['id'],'versionId':version['id'],'sourceId':source['id'],'groupId':group,
                'text':text,'trainingPolicy':'allowed','policyVersion':rag.setting('policyVersion',0),'targetOrigin':'source_span',
                'language':version.get('language'),'ragSplit':role,'sourceSpan':version['sourceSpan'],'provenance':version['provenance'],
                'permissionReceipt':{'actor':ACTOR,'policy':POLICY,'rightsReceiptId':source['rightsReceiptId'],
                    'sourcePolicyVersion':source['policyVersion'],'trainingPolicy':'allowed'}}
        cursor=page[-1]['id']

    require(epochs==(rag.setting('policyVersion',0),rag.setting('trainingDataRevision',0)),'POLICY_STATE_CHANGED')


def _test_references(archive):
    references={};scanned=0
    for member in archive.infolist():
        path=member.filename.lower()
        if not path.endswith('.py') or not any(marker in path for marker in ('/test_','/tests/','/test/')) or member.file_size>512*1024:continue
        if scanned+member.file_size>32*1024**2:break
        raw=archive.read(member);scanned+=len(raw)
        try:text=raw.decode('utf-8');tree=ast.parse(text)
        except (UnicodeError,SyntaxError):continue
        for node in ast.walk(tree):
            unittest_call=node.value if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) else None
            if unittest_call and not (isinstance(unittest_call.func,ast.Attribute) and unittest_call.func.attr in ('assertEqual','assertTrue','assertFalse','assertIsNone')):unittest_call=None
            if not isinstance(node,ast.Assert) and unittest_call is None:continue
            expression=node.test if isinstance(node,ast.Assert) else unittest_call
            calls=[n for n in ast.walk(expression) if isinstance(n,ast.Call)]
            for call in calls:
                name=call.func.id if isinstance(call.func,ast.Name) else call.func.attr if isinstance(call.func,ast.Attribute) else None
                if not name or len(references.get(name,[]))>=2:continue
                example={'path':member.filename.split('/',1)[1],'sourceSha256':digest(raw),'startLine':node.lineno,
                    'endLine':node.end_lineno,'assertion':ast.get_source_segment(text,node),'cases':[]}
                comparison=expression if isinstance(expression,ast.Compare) else None
                if comparison and comparison.left is call and len(comparison.ops)==1 and isinstance(comparison.ops[0],ast.Eq):
                    try:
                        args=[ast.literal_eval(a) for a in call.args];kwargs={a.arg:ast.literal_eval(a.value) for a in call.keywords};expected=ast.literal_eval(comparison.comparators[0])
                        case={'function':name,'args':args,'kwargs':kwargs,'expected':expected};canonical(case);example['cases']=[case]
                    except (ValueError,TypeError,SyntaxError):pass
                references.setdefault(name,[]).append(example)
    return references


def _prepare_sft(root,rag,pre,post,control,project,tokenizer,source_map):
    examples=[];selected=Counter();target_tokens=0;source_by_repo={rag.get('source',row['sourceId'])['name']:row for row in source_map}
    by_id={row['sourceId']:row for row in source_map}
    for example in rag.list('training_example'):
        source=by_id.get(example.get('lineage',[{}])[0].get('sourceId'))
        if not source or example.get('reviewedBy')!=ACTOR or example.get('reviewStatus')!='approved' or example.get('targetOrigin')!='source_span':continue
        if post._example_reasons(example) or example.get('sourceSpan',{}).get('targetSha256')!=digest(example['target'].encode()):continue
        if example['target'] not in pre.content(example['lineage'][0]['versionId']):continue
        examples.append(example);selected[source['sourceId']]+=1
        if source['split']=='train':target_tokens+=len(tokenizer.encode_ordinary(example['target']))+1
    for path in sorted((root/'sources').glob('*/receipt.json')):
        receipt=json.loads(path.read_text('utf-8'));mapped=source_by_repo.get(receipt.get('repository'))
        if not mapped or mapped['split']=='audit':continue
        if mapped['split']=='train' and target_tokens>=4096 and sum(m['split']=='train' and selected[m['sourceId']]>0 for m in source_map)>=3:continue
        if mapped['split']!='train' and any(m['split']==mapped['split'] and selected[m['sourceId']]>0 for m in source_map):continue
        archive_path=next(path.parent.glob(receipt['revision']+'.zip'))
        with zipfile.ZipFile(archive_path) as package:
            refs=_test_references(package)
        maximum=20 if mapped['split']=='train' else 8
        for row in _jsonl(path.parent/'records.jsonl'):
            if selected[mapped['sourceId']]>=maximum:break
            if row['language'] not in ('py','pyi'):continue
            try:tree=ast.parse(row['text'])
            except SyntaxError:continue
            for function in tree.body:
                if selected[mapped['sourceId']]>=maximum:break
                if not isinstance(function,ast.FunctionDef) or function.decorator_list or function.name.startswith('_') or function.name not in refs:continue
                output=ast.get_source_segment(row['text'],function)
                if not output or len(output)<(30 if mapped['split']=='release' else 100):continue
                doc=ast.get_docstring(function) or '';description=doc.splitlines()[0][:120] if doc else function.name.replace('_',' ')
                instruction='Implement '+function.name+': '+description
                count=len(tokenizer.encode_ordinary(output))+1
                if count<(10 if mapped['split']=='release' else 40) or len(tokenizer.encode_ordinary(instruction))+count+5>128:continue
                try:
                    pre.content(row['versionId'])
                    candidate=post.create_example({'input':instruction,'target':output,'lineage':[{'sourceId':mapped['sourceId'],
                        'documentId':row['documentId'],'versionId':row['versionId']}]})
                    reviewed=post.review_example(candidate['id'],{'decision':'approve','permissionStatus':'allowed',
                        'reason':'Delegated review: immutable licensed source span, exact AST function text, syntax parse and separately hashed official test assertion reference. Runtime tests are not executed.'})
                except Exception:continue
                reference=refs[function.name][0];test_blob=pre._blob(reference['assertion'])
                verification={'kind':'official_test_source_reference','state':'linked_not_executed','source':reference,
                    'assertionBlobHash':test_blob,'runtimeExecuted':False,'actor':ACTOR}
                if reference['cases']:
                    fixture=control.store.put('fixture',{'cases':reference['cases'],'purpose':'official_source_sft_reference',
                        'sourceEvidence':reference,'registeredBy':ACTOR,'profileId':None,'runtimeVerified':False})
                    verification['fixtureId']=fixture['id']
                span={**row['sourceSpan'],'startLine':row['sourceSpan']['startLine']+function.lineno-1,
                    'endLine':row['sourceSpan']['startLine']+function.end_lineno-1,'targetSha256':digest(output.encode())}
                reviewed=rag.put('training_example',{'targetOrigin':'source_span','reviewedBy':ACTOR,'approvalKind':'delegated_source_span_review',
                    'sourceSpan':span,'independentVerification':verification,'executionStatus':'not_executed'},reviewed['id'])
                examples.append(reviewed);selected[mapped['sourceId']]+=1
                if mapped['split']=='train':target_tokens+=count
    require(target_tokens>=4096 and sum(mapped['split']=='train' and selected[mapped['sourceId']]>0 for mapped in source_map)>=3,
            'INSUFFICIENT_APPROVED_SFT_DATA',targetTokens=target_tokens,counts=dict(selected))
    # Pin the planned family roles before the snapshot allocator runs, including
    # its connected-component keys. No quality metric influences this decision.
    assignments=post._split_assignments(examples);roles=rag.setting('trainingStableOrigins',{})
    ledger={assignments[e['id']]:roles[post._lineage_groups(e['lineage'])[0]] for e in examples}
    require({'train','validation','test'}<=set(ledger.values()),'INSUFFICIENT_SPLIT_GROUPS')
    rag.set_setting('trainingStableSplits',ledger)
    snapshot=post.create_snapshot({'exampleIds':[e['id'] for e in examples],'name':'Official source spans, delegated provenance review'})
    snapshot=post._export(snapshot['id'])
    exported=control.bridge.foundation_sft_corpus(project['id'],snapshot['id'],[m['sourceId'] for m in source_map])
    for record in exported['records']:
        source=rag.get('training_example',record['exampleId'])
        record.update(sourceSpan=source['sourceSpan'],independentVerification=source['independentVerification'],
                      approvalReceipt={'actor':ACTOR,'kind':'delegated_source_span_review','runtimeExecuted':False})
    from .corpus import build
    result=build(control.store,project,exported['records'],{'kind':'sft','policyVersion':rag.setting('policyVersion',0)})
    return result,snapshot,{'approvedExamples':len(examples),'trainTargetTokens':target_tokens,'sourceCounts':dict(selected),
                           'runtimeTestsExecuted':False,'approvalActor':ACTOR}


def finish_sft(root):
    """Resume after tokenizer publication without rebuilding or relearning raw data."""
    from RAG.core import Store as RagStore
    from RAG.Pre.service import PreService
    from RAG.Post.service import PostService
    from LLM.service import Control
    from LLM.integration import RagBridge
    from . import corpus,tokenizer as token_module
    root=Path(root).resolve();rag=RagStore(root/'rag');pre=PreService(rag);post=PostService(rag,pre);control=Control(root/'control',worker=False)
    control.bridge=RagBridge(control,rag,pre,post);control.releases.rag=control.bridge
    try:
        project=control.foundation._project(control.store.list('foundation_project')[0]['id'])
        raw=corpus.resolve(control.store,project['activeCorpusId']);tokens=control.store.get('foundation_tokenizer',project['tokenizerId']);tok=token_module.load(control.store,tokens)
        source_map=json.loads((root/'reports'/'source-map.json').read_text('utf-8'))['sources']
        sft,snapshot,sft_report=_prepare_sft(root,rag,pre,post,control,project,tok,source_map)
        require(sft['state']=='ready','INSUFFICIENT_APPROVED_SFT_DATA',details=sft.get('reasonCodes'))
        control.store.put('foundation_project',{'activeSftCorpusId':sft['id']},project['id'])
        dev_tokens=0;dev_groups=set()
        for row in corpus.rows(control.store,raw,'dev'):
            dev_tokens+=len(tok.encode_ordinary(row['text']))+1;dev_groups.update(row['groupIds'])
        requirements={'trainRawBytes':raw['splitBytes']['train'],'trainGroups':raw['groupCounts']['train'],
            'devTargetTokens':dev_tokens,'devGroups':len(dev_groups),'vocabSize':tok.vocab_size,
            'sftApprovedExamples':sft['splitCounts']['train'],'sftTrainGroups':sft['groupCounts']['train'],'sftTargetTokens':sft['sftTargetTokens']}
        require(requirements['trainRawBytes']>=16*1024**2 and requirements['trainGroups']>=20 and dev_tokens>=100000
            and len(dev_groups)>=20 and tok.vocab_size==16384,'DATA_READINESS_REQUIRED')
        report={'schemaVersion':1,'actor':ACTOR,'state':'ready','projectId':project['id'],'rawCorpusId':raw['id'],'rawCorpusHash':raw['manifestHash'],
            'tokenizerId':tokens['id'],'tokenizerHash':tokens['tokenizerHash'],'sftCorpusId':sft['id'],'sftCorpusHash':sft['manifestHash'],
            'ragSnapshotId':snapshot['id'],'requirements':requirements,'sftReview':sft_report,'rightsSourceCount':len(source_map),
            'ragSourceIds':[m['sourceId'] for m in source_map],'ragRoot':str(rag.root),'controlRoot':str(control.store.root),
            'reasonCodes':[],'modelTrainingPerformed':False,'nasModified':False,'observedAt':now()}
        atomic_json(root/'reports'/'readiness.json',report);path=control.store.root/'reports'/'preparations'/'official-v1.json';atomic_json(path,report)
        preparation=control.store.put('foundation_data_preparation',{**report,'reportPath':path.relative_to(control.store.root).as_posix()},'official-v1')
        print(canonical(report),flush=True);return preparation
    finally:control.stop();control.store.close()


def prepare(root):
    from RAG.core import Store as RagStore
    from RAG.Pre.service import PreService
    from RAG.Post.service import PostService
    from LLM.service import Control
    from LLM.integration import RagBridge
    from . import corpus,tokenizer as token_module
    root=Path(root).resolve();(root/'reports').mkdir(parents=True,exist_ok=True)
    rag=RagStore(root/'rag');pre=PreService(rag);post=PostService(rag,pre);control=Control(root/'control',worker=False)
    control.bridge=RagBridge(control,rag,pre,post);control.releases.rag=control.bridge
    try:
        source_ids,source_map=_ingest(root,rag,pre)
        ensure_attributions(root,rag)
        # One pre-exposure migration fixes the historical policy which poisoned
        # an entire unrelated repository for one shared boilerplate document.
        # The predeclared family roles and immutable source bytes do not change.
        migration=control.store.get('foundation_preparation_migration','independent-exact-copy-v2',{})
        if not migration and control.store.list('foundation_split_conflict'):
            exposures=control.store.list('foundation_exposure')
            require(not any(row.get('purpose') not in ('tokenizer','data_readiness') for row in exposures)
                and not control.store.list('foundation_checkpoint') and not control.store.list('foundation_genesis')
                and not control.store.list('foundation_tokenizer'),'PREPARATION_ALREADY_EXPOSED')
            backup={kind:control.store.list(kind) for kind in ('foundation_split','foundation_content_split','foundation_split_conflict')}
            atomic_json(root/'reports'/'pre-exposure-split-audit-v1.json',backup)
            with control.store.tx():
                for kind in backup:control.store.db.execute('DELETE FROM records WHERE kind=?',(kind,))
                for old_corpus in control.store.list('foundation_corpus'):
                    control.store.put('foundation_corpus',{'state':'revoked','reasonCodes':['PRE_TRAINING_DEDUP_POLICY_REBUILD']},old_corpus['id'])
                control.store.put('foundation_preparation_migration',{'actor':ACTOR,'reason':'Independent cross-split exact copies are excluded individually; family split plan unchanged',
                    'sourceBytesChanged':False,'familySplitRolesChanged':False,'uncommittedTokenizerAttemptDiscarded':bool(exposures),
                    'modelTrainingPerformed':False,'backupSha256':file_hash(root/'reports'/'pre-exposure-split-audit-v1.json')},'independent-exact-copy-v2')
        for mapped in source_map:
            source=rag.get('source',mapped['sourceId']);group=mapped['groupId'].lower()
            canonical_group='repository:'+source['repositoryFamily'].lower()
            control.store.put('foundation_repository_family',{'groupId':group,'canonicalGroupId':canonical_group,
                'sourceId':source['id'],'rightsReceiptId':source['rightsReceiptId'],'actor':ACTOR},digest(group))
        old=control.store.list('foundation_project')
        project=control.foundation.status(old[0]['id']) if old else control.foundation.create_project({'name':'Official licensed corpus preparation',
            'origin':'random_init','pretrainedWeightArtifactIds':[],'automation':'manual'})
        control.foundation.configure(project['id'],{'allowedSourceIds':source_ids,'enabled':False,'automation':'manual'},project['policy']['version'])
        project=control.foundation._project(project['id'])
        raw=corpus.build(control.store,project,_approved_raw_rows(rag,pre,source_ids),{'policyVersion':rag.setting('policyVersion',0)})
        require(raw['state']=='ready','INSUFFICIENT_CORPUS',details=raw.get('reasonCodes'))
        control.store.put('foundation_project',{'activeCorpusId':raw['id'],'stage':'corpus_ready'},project['id'])
        print(canonical({'stage':'raw_corpus_ready','bytes':raw['splitBytes'],'groups':raw['groupCounts']}),flush=True)
        tokens=token_module.build(control.store,project,raw);tok=token_module.load(control.store,tokens)
        control.store.put('foundation_project',{'tokenizerId':tokens['id'],'stage':'tokenizer_ready'},project['id']);project=control.foundation._project(project['id'])
        dev_tokens=0;dev_groups=set()
        for row in corpus.rows(control.store,raw,'dev',purpose='data_readiness'):
            dev_tokens+=len(tok.encode_ordinary(row['text']))+1;dev_groups.update(row['groupIds'])
        require(len(dev_groups)>=20 and dev_tokens>=100000,'INSUFFICIENT_DEV_DATA',devGroups=len(dev_groups),devTokens=dev_tokens)
        sft,snapshot,sft_report=_prepare_sft(root,rag,pre,post,control,project,tok,source_map)
        require(sft['state']=='ready','INSUFFICIENT_APPROVED_SFT_DATA',details=sft.get('reasonCodes'))
        control.store.put('foundation_project',{'activeSftCorpusId':sft['id']},project['id'])
        requirements={'trainRawBytes':raw['splitBytes']['train'],'trainGroups':raw['groupCounts']['train'],
            'devTargetTokens':dev_tokens,'devGroups':len(dev_groups),'vocabSize':tok.vocab_size,
            'sftApprovedExamples':sft['splitCounts']['train'],'sftTrainGroups':sft['groupCounts']['train'],
            'sftTargetTokens':sft['sftTargetTokens']}
        report={'schemaVersion':1,'actor':ACTOR,'state':'ready','projectId':project['id'],'rawCorpusId':raw['id'],
            'rawCorpusHash':raw['manifestHash'],'tokenizerId':tokens['id'],'tokenizerHash':tokens['tokenizerHash'],
            'sftCorpusId':sft['id'],'sftCorpusHash':sft['manifestHash'],'ragSnapshotId':snapshot['id'],
            'requirements':requirements,'sftReview':sft_report,'rightsSourceCount':len(source_ids),'ragSourceIds':source_ids,
            'ragRoot':str(rag.root),'controlRoot':str(control.store.root),'reasonCodes':[],
            'modelTrainingPerformed':False,'nasModified':False,'observedAt':now()}
        atomic_json(root/'reports'/'readiness.json',report)
        local_report=control.store.root/'reports'/'preparations'/'official-v1.json';atomic_json(local_report,report)
        preparation=control.store.put('foundation_data_preparation',{**report,'reportPath':local_report.relative_to(control.store.root).as_posix()},'official-v1')
        print(canonical(report),flush=True);return preparation
    finally:control.stop();control.store.close()


def export_inventory(llm_store,preparation_id,rag_store):
    """Allowlisted immutable dataset graph for offline transfer; never weights."""
    from .corpus import iter_members,verify
    preparation=llm_store.get('foundation_data_preparation',preparation_id)
    require(preparation.get('state')=='ready' and preparation.get('actor')==ACTOR,'PREPARATION_NOT_READY')
    records=[];files={};sources=set(preparation['ragSourceIds']);versions=set();documents=set();groups=set();hashes=set()
    def add_record(scope,kind,record):records.append({'scope':scope,'kind':kind,'id':record['id'],'body':record})
    def add_file(scope,relative):
        root=rag_store.root if scope=='rag' else llm_store.root
        path=(root/relative).resolve();require(path.is_relative_to(root) and path.is_file() and not path.is_symlink(),'PREPARATION_PATH_INVALID')
        files[(scope,relative)]={'scope':scope,'relativePath':relative,'sha256':file_hash(path),'sizeBytes':path.stat().st_size}
    add_record('llm','foundation_data_preparation',preparation)
    if preparation.get('implementationVersion')=='seed-source-context-sft-v3':
        from .source_context_contracts import verify_runtime_source_parent
        raw_parent=verify_runtime_source_parent(llm_store,preparation)
        add_record('llm','foundation_data_preparation',raw_parent)
        add_file('llm',raw_parent['reportPath'])
    for kind,identity in (('foundation_project',preparation['projectId']),('foundation_policy',preparation['projectId']),('foundation_tokenizer',preparation['tokenizerId'])):
        record=llm_store.get(kind,identity)
        if kind=='foundation_project':
            # Project IDs link data lineage. PC-trained weight/quality/runtime
            # references are deliberately outside this data-only projection.
            record={k:v for k,v in record.items() if not any(word in k.lower() for word in ('checkpoint','genesis','release','certificate','evaluation','runid','quality','serving','profile','capability'))}
            record.update(origin='random_init',pretrainedWeightArtifactIds=[],stage='tokenizer_ready',state='active',
                genesisId=None,latestCheckpointId=None,latestRunId=None,trainedContextTokens=0,
                activeCorpusId=preparation['rawCorpusId'],activeSftCorpusId=preparation['sftCorpusId'],tokenizerId=preparation['tokenizerId'],
                executionScope='nas',nasRequalificationRequired=True,dataOnlyImport=True)
            from .source_context_contracts import project_preparation_fields
            record.update(project_preparation_fields(preparation))
        elif kind=='foundation_policy':
            record={**record,'enabled':False,'automation':'manual','workerScope':'nas-local','workerMemoryBytes':6*1024**3}
        add_record('llm',kind,record)
        if kind=='foundation_tokenizer':
            root=llm_store.root/record['storagePath'];manifest=json.loads((root/'manifest.json').read_text('utf-8'))
            for name in (*manifest['files'],'manifest.json'):add_file('llm',(root/name).relative_to(llm_store.root).as_posix())
    corpus_ids=(preparation['rawCorpusId'],preparation['sftCorpusId'])
    if preparation.get('implementationVersion')=='seed-source-context-sft-v3':
        token=llm_store.get('foundation_tokenizer',preparation['tokenizerId'])
        origin=llm_store.get('foundation_corpus',token.get('corpusId') or '',{})
        require(origin.get('state')=='ready' and origin.get('projectId')==preparation['projectId']
            and token.get('projectId')==preparation['projectId'] and token.get('state')=='ready'
            and token.get('tokenizerHash')==preparation['tokenizerHash']
            and origin.get('manifestHash')==token.get('trainCorpusHash'),
            'TOKENIZER_SOURCE_CORPUS_REQUIRED')
        corpus_ids=tuple(dict.fromkeys((*corpus_ids,origin['id'])))
    for identity in corpus_ids:
        record=llm_store.get('foundation_corpus',identity);add_record('llm','foundation_corpus',record);root,manifest=verify(llm_store,record)
        for name in (*manifest['files'],'manifest.json'):add_file('llm',(root/name).relative_to(llm_store.root).as_posix())
        for member in iter_members(llm_store,record):
            hashes.add(member['hash']);groups.update(member['groupIds'])
            for link in member['lineage']:
                versions.add(link['versionId']);documents.add(link['documentId'])
                if preparation.get('implementationVersion')=='seed-source-context-sft-v3':sources.add(link['sourceId'])
    for identity in sources:
        source=rag_store.get('source',identity)
        if preparation.get('implementationVersion')=='seed-source-context-sft-v3':
            require(source.get('trainingPolicy')=='allowed','SOURCE_NOT_ALLOWED')
        add_record('rag','source',source)
        rights=rag_store.get('source_rights_receipt',source['rightsReceiptId'])
        require(rights.get('actor')==ACTOR and rights.get('state')=='rights_verified','RIGHTS_RECEIPT_REQUIRED')
        add_record('rag','source_rights_receipt',rights)
        for evidence in rights['licenseEvidence']:add_file('rag',evidence['relativePath'])
        for supplement_id in source.get('attributionReceiptIds',[]):
            supplement=rag_store.get('source_rights_receipt',supplement_id);add_record('rag','source_rights_receipt',supplement)
            for evidence in supplement['licenseEvidence']:add_file('rag',evidence['relativePath'])
    units=set();blobs=set();fixtures=set()
    for identity in documents:
        record=rag_store.get('document',identity);add_record('rag','document',record);units.add(record['sourceUnitId'])
    for kind,identities in (('source_unit',units),('version',versions)):
        for identity in identities:
            record=rag_store.get(kind,identity);add_record('rag',kind,record)
            for key in ('rawBlobHash','rawFileBlobHash','normalizedBlobHash'):
                if record.get(key):blobs.add(record[key])
    snapshot=rag_store.get('training_snapshot',preparation['ragSnapshotId']);add_record('rag','training_snapshot',snapshot)
    exported_examples={}
    for identity in snapshot['exampleIds']:
        record=rag_store.get('training_example',identity);add_record('rag','training_example',record)
        exported_examples[identity]=record
        evidence=record.get('independentVerification',{})
        if evidence.get('assertionBlobHash'):blobs.add(evidence['assertionBlobHash'])
        if evidence.get('fixtureId'):fixtures.add(evidence['fixtureId'])
    for identity in fixtures:add_record('llm','fixture',llm_store.get('fixture',identity))
    if preparation.get('implementationVersion')=='seed-source-context-sft-v3':
        from types import SimpleNamespace
        from . import source_oracle_binding as oracle
        control=SimpleNamespace(store=llm_store,bridge=SimpleNamespace(store=rag_store))
        for binding in llm_store.list(oracle.KIND):
            if not (binding.get('preparationId')==preparation_id
                and binding.get('sourceCorpusId')==preparation['sftCorpusId']
                and binding.get('sourceCorpusHash')==preparation['sftCorpusHash']
                and binding.get('sourceExampleId') in exported_examples
                and binding.get('helperHash')==oracle.implementation_hash()):continue
            example=exported_examples[binding['sourceExampleId']]
            version=rag_store.get('version',binding['sourceVersionId'])
            function=ast.parse(example['target']).body[0]
            require(isinstance(function,ast.FunctionDef),'SOURCE_ORACLE_TARGET_CHANGED')
            current=oracle.binding_evidence(control,preparation,example,version,function.name)
            require(current==binding and binding.get('actor')==ACTOR
                and binding.get('hostSourceExecuted') is False,'SOURCE_ORACLE_BINDING_CHANGED')
            add_record('llm',oracle.KIND,binding)
            blobs.update(binding[key] for key in ('targetModuleBlobHash','testModuleBlobHash','assertionBlobHash'))
    root=rag_store.root/snapshot['storagePath'];manifest=json.loads((root/'manifest.json').read_text('utf-8'))
    require(file_hash(root/'manifest.json')==snapshot['manifestHash'],'SNAPSHOT_HASH_MISMATCH')
    for name in (*manifest['files'],'manifest.json'):add_file('rag',(root/name).relative_to(rag_store.root).as_posix())
    for checksum in blobs:add_file('rag','blobs/'+checksum[:2]+'/'+checksum)
    for kind in ('foundation_split','foundation_content_split','foundation_repository_family','foundation_near_duplicate','foundation_split_conflict'):
        for record in llm_store.list(kind):
            if record.get('groupId') in groups or set(record.get('groupIds',[]))&groups or record.get('contentHash') in hashes:
                add_record('llm',kind,record)
    if llm_store.rows("SELECT name FROM sqlite_master WHERE type='table' AND name='foundation_near_documents'"):
        for near in llm_store.rows('SELECT * FROM foundation_near_documents'):
            if near['hash'] not in hashes:continue
            fingerprint={'id':digest([near['namespace'],near['hash']]),'version':1,'contentHash':near['hash'],'namespace':near['namespace'],
                'split':near['split'],'groupIds':json.loads(near['groups_json']),'sketch':json.loads(near['sketch_json']),
                'method':'bottom-k-token5-jaccard-v1','derivedFromPreparationId':preparation_id,
                'createdAt':preparation['createdAt'],'updatedAt':preparation['createdAt']}
            add_record('llm','foundation_near_fingerprint',fingerprint)
    add_file('llm',preparation['reportPath'])
    if preparation.get('splitAuditPath'):add_file('llm',preparation['splitAuditPath'])
    return {'metadata':{'kind':'foundation_prepared_dataset','preparationId':preparation_id,'projectId':preparation['projectId'],
        'actor':ACTOR,'requirements':preparation['requirements'],'sourceIds':sorted(sources),'ragSnapshotId':snapshot['id'],
        'modelTrainingPerformed':False,'containsWeights':False,'nasModified':False,
        'projectProjection':'data-only-v1','executionScope':'nas','nasRequalificationRequired':True},
        'records':sorted(records,key=lambda r:(r['scope'],r['kind'],r['id'])),'files':sorted(files.values(),key=lambda r:(r['scope'],r['relativePath']))}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default='LLM/runtime-data/preparation/official-v1')
    parser.add_argument('--workers',type=int,default=4);parser.add_argument('command',choices=['collect','prepare','finish-sft','attributions','audit'])
    parser.add_argument('--rag-root',default='RAG/data');parser.add_argument('--nas-config')
    args=parser.parse_args()
    if args.command=='collect':collect(args.output,args.workers)
    elif args.command=='prepare':prepare(args.output)
    elif args.command=='finish-sft':finish_sft(args.output)
    elif args.command=='attributions':
        from RAG.core import Store as RagStore
        print(canonical({'attributionFiles':ensure_attributions(Path(args.output),RagStore(Path(args.output)/'rag'))}))
    else:
        report={'local':audit_rag_data(args.rag_root),'nasModified':False,'modelTrainingPerformed':False}
        if args.nas_config:
            settings=json.loads(Path(args.nas_config).read_text('utf-8-sig'));report['nas']=audit_rag_data(Path(settings['share'])/'data'/'rag')
        atomic_json(Path(args.output)/'reports'/'operational-audit.json',report);print(canonical(report))


if __name__=='__main__':main()
