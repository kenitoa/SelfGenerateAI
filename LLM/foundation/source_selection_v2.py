"""Add-only, same-archive PC raw corpus reselection.

No CLI or import side effects. An explicit operator calls preparation only after active model work has stopped.
"""
from __future__ import annotations
from collections import defaultdict
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

from LLM.core import HEAVY_JOBS, atomic_json, digest, file_hash, now, require
from LLM.foundation.prepare_data import ACTOR, POLICY, EXCLUDE_PARTS, EXTENSIONS, MAX_MEMBER_BYTES
from LLM.foundation.corpus import secret_reasons
from RAG.Pre.processing import normalize, quality, text_language

VERSION='official-python-v2'
SAMPLING={'kind':'python-focused-targets-v2','weights':{'python':90,'other_code':5,'docs':5}}


def _rows(path):
    with Path(path).open('rb') as handle:
        while line:=handle.readline(2*1024**2+1):
            require(len(line)<=2*1024**2,'SELECTION_ROW_LIMIT')
            yield json.loads(line)


def _protected(rag):
    # Small metadata bodies, no open transaction over archives or file work.
    return {(r['kind'],r['id']):r['body'] for r in rag.read("SELECT kind,id,body FROM records WHERE kind IN ('source','source_unit','document','version','source_rights_receipt')")}


def _preserved(rag,before):
    current=_protected(rag)
    require(all(current.get(k)==value for k,value in before.items()),'ORIGINAL_RAG_RECORD_CHANGED',status=409)


def _holdout(store,corpus):
    root=store.root/corpus['storagePath'];manifest=json.loads((root/'manifest.json').read_text('utf-8'))
    require(file_hash(root/'manifest.json')==corpus['manifestHash'],'CORPUS_HASH_MISMATCH')
    versions=set();hashes=defaultdict(set)
    for split,names in manifest['splitShards'].items():
        if split=='train':continue
        for name in names:
            require(file_hash(root/name)==manifest['files'][name],'CORPUS_HASH_MISMATCH')
            for row in _rows(root/name):
                hashes[split].add((row['hash'],tuple(row['groupIds'])))
                versions.update(link['versionId'] for link in row.get('lineage',[]) if link.get('versionId'))
                versions.add(row['versionId'])
    return versions,{k:sorted(v) for k,v in hashes.items()}


def _verified_span(archive,item,rights):
    require(item.get('split')=='train' and item.get('modelGenerated') is False
            and item.get('repository')==rights['repository'] and item.get('revision')==rights['revision']
            and item.get('archiveSha256')==rights['archiveSha256'],'SELECTION_RIGHTS_MISMATCH')
    context=getattr(archive,'_selection_context',None)
    if context is None:
        infos=archive.infolist();prefix=infos[0].filename.split('/')[0]+'/'
        members={i.filename.removeprefix(prefix):i for i in infos if i.filename.startswith(prefix)}
        nested=[str(PurePosixPath(n).parent)+'/' for n in members
            if len(PurePosixPath(n).parts)>1 and PurePosixPath(n).name.upper().startswith(('LICENSE','COPYING'))]
        context={'members':members,'nested':nested,'lastFile':None};archive._selection_context=context
    name=item['path'];path=PurePosixPath(name)
    require(not path.is_absolute() and '..' not in path.parts and path.suffix.lower() in EXTENSIONS
            and not any(p.lower() in EXCLUDE_PARTS for p in path.parts),'SELECTION_PATH_EXCLUDED')
    members=context['members']
    require(name in members,'SOURCE_FILE_MISSING')
    require(not any(name.startswith(scope) for scope in context['nested']),'NESTED_LICENSE_SCOPE')
    info=members[name]
    require(not info.is_dir() and info.file_size<=MAX_MEMBER_BYTES
            and (info.external_attr>>16)&0o170000!=0o120000,'SOURCE_FILE_INVALID')
    previous=context['lastFile']
    if previous and previous[0]==name:checksum,text=previous[1:]
    else:
        raw=archive.read(info);checksum=digest(raw);text=raw.decode('utf-8-sig')
        context['lastFile']=(name,checksum,text)
    require(checksum==item['sourceFileSha256'],'SOURCE_FILE_HASH_MISMATCH')
    declarations=re.findall(r'SPDX-License-Identifier:\s*([^\r\n*]+)',text)
    require(not any(not all(part in {'MIT','Apache-2.0','BSD-2-Clause','BSD-3-Clause','OR','AND','(',')'}
        for part in declaration.replace('(',' ( ').replace(')',' ) ').split()) for declaration in declarations),'FOREIGN_FILE_LICENSE')
    require(type(item['startLine']) is int and type(item['endLine']) is int
            and 1<=item['startLine']<=item['endLine'],'SOURCE_SPAN_INVALID')
    lines=text.splitlines(keepends=True)
    require(item['endLine']<=len(lines),'SOURCE_SPAN_INVALID')
    span=''.join(lines[item['startLine']-1:item['endLine']])
    require(digest(span.encode())==item['sha256'] and len(span.encode())==item['rawBytes']
            and not secret_reasons(span) and quality(span)[0]=='passed','SOURCE_SPAN_NOT_APPROVED')
    return span,path.suffix.lower().lstrip('.')


def add_selection(rag,pre,source_root,inventory):
    """Add new immutable spans under existing grants. Never retire old units."""
    before=_protected(rag);seen=set();selected=[];imported=0
    rights_cache={};archive_cache={};source_cache={}
    try:
        for item in _rows(inventory):
            identity=digest([item['repository'],item['revision'],item['path'],item['startLine'],item['endLine']])
            require(identity not in seen,'SELECTION_DUPLICATE_ID');seen.add(identity)
            rights_id=item['rightsReceiptId']
            if rights_id not in rights_cache:
                rights=rag.get('source_rights_receipt',rights_id);source=rag.get('source',rights['sourceId'])
                require(rights.get('state')=='rights_verified' and rights.get('decision')=='allowed'
                    and rights.get('actor')==ACTOR and rights.get('policy')==POLICY
                    and digest(rights)==item['rightsReceiptHash'] and source.get('trainingPolicy')=='allowed'
                    and source.get('retrievalPolicy')=='approved' and source.get('rightsReceiptId')==rights_id,
                    'SELECTION_RIGHTS_MISMATCH')
                group='repository:'+source['repositoryFamily']
                require(rag.setting('trainingStableOrigins',{}).get(group)=='train','SPLIT_LEAKAGE')
                archive_path=Path(source_root)/'sources'/rights['repository'].replace('/','--')/(rights['revision']+'.zip')
                require(file_hash(archive_path)==rights['archiveSha256'],'ARCHIVE_HASH_MISMATCH')
                rights_cache[rights_id]=rights;source_cache[rights_id]=source;archive_cache[rights_id]=zipfile.ZipFile(archive_path)
            rights=rights_cache[rights_id];source=source_cache[rights_id]
            span,language=_verified_span(archive_cache[rights_id],item,rights)
            existing=rag.one("SELECT body FROM records WHERE kind='version' AND id=?",(identity,))
            if existing:
                old=json.loads(existing['body']);document=rag.get('document',identity)
                require(old.get('rawBlobHash')==digest(span.encode()) and old.get('sourceId')==source['id']
                    and old.get('revision')==rights['revision'] and old.get('qualityStatus')=='passed'
                    and document.get('currentVersionId')==identity and document.get('trainingPolicy')!='denied'
                    and document.get('retrievalPolicy')=='approved','SELECTION_EXISTING_VERSION_CONFLICT')
                selected.append(identity);continue
            normalized,_=normalize(span,language);raw_hash=pre._blob(span);normalized_hash=pre._blob(normalized)
            position={'path':item['path'],'startLine':item['startLine'],'endLine':item['endLine'],
                'sourceFileSha256':item['sourceFileSha256'],'contentSha256':item['sha256']}
            unit_key=item['path']+':'+str(item['startLine'])+'-'+str(item['endLine']);unit_id=digest([source['id'],unit_key])
            provenance={'repository':rights['repository'],'family':rights['repositoryFamily'],'revision':rights['revision'],
                'archiveSha256':rights['archiveSha256'],'url':'https://github.com/'+rights['repository']+'/blob/'+rights['revision']+'/'+item['path']+'#L'+str(item['startLine'])+'-L'+str(item['endLine'])}
            with rag.tx():
                require(not rag.one("SELECT id FROM records WHERE kind IN ('document','version') AND id=?",(identity,)),
                        'SELECTION_EXISTING_VERSION_CONFLICT')
                current=rag.get('source',source['id']);current_rights=rag.get('source_rights_receipt',rights_id)
                require(current==source and current_rights==rights,'POLICY_STATE_CHANGED')
                old_unit=rag.one("SELECT body FROM records WHERE kind='source_unit' AND id=?",(unit_id,))
                require(not old_unit,'SELECTION_EXISTING_UNIT_CONFLICT')
                rag.put('source_unit',{'sourceId':source['id'],'unitKey':unit_key,'unitKind':'verified_source_span',
                    'rawBlobHash':raw_hash,'currentSnapshotHash':raw_hash,'revision':rights['revision'],'sourceSpan':position},unit_id)
                rag.put('document',{'name':item['path'],'datasetId':source['datasetId'],'sourceId':source['id'],
                    'sourceUnitId':unit_id,'externalKey':item['path']+':'+str(item['startLine']),'path':item['path'],
                    'enabled':True,'deletedAt':None,'currentVersionId':identity,'retrievalPolicy':'approved',
                    'trainingPolicy':'allowed','policyVersion':1,'state':'ingested'},identity)
                rag.put('version',{'documentId':identity,'sourceId':source['id'],'rawBlobHash':raw_hash,
                    'normalizedBlobHash':normalized_hash,'rawFileBlobHash':raw_hash,'contentHash':digest(span.encode()),
                    'normalizedHash':digest(normalized.encode()),'textLength':len(normalized),'revision':rights['revision'],
                    'language':{'py':'python','pyi':'python','rs':'rust','js':'javascript','ts':'typescript','tsx':'typescript'}.get(language,language),
                    'textLanguage':text_language(normalized),'qualityStatus':'passed','qualityReasons':[],
                    'parserVersion':VERSION,'recordPositions':[],'sourceSpan':position,'provenance':provenance,
                    'targetOrigin':'source_span','rightsReceiptId':rights_id},identity)
            selected.append(identity);imported+=1
        _preserved(rag,before)
        if imported:rag.set_setting('policyVersion',int(rag.setting('policyVersion',0))+1)
        return {'selectedVersionIds':selected,'importedVersionCount':imported,'originalRightsPreserved':True,
            'originalRagRecordsUnchanged':True,'inventoryHash':file_hash(inventory),'rightsReceiptIds':sorted(rights_cache),
            'observedPolicyVersion':rag.setting('policyVersion',0),
            'observedDataRevision':rag.setting('trainingDataRevision',0)}
    finally:
        for archive in archive_cache.values():archive.close()


def prepare(control,rag,pre,source_root,inventory,*,project_id,measure_tokens=True):
    """Explicit post-exam operator call; returns IDs without changing active data."""
    require(control.execution_scope=='pc_validation','PC_VALIDATION_SCOPE_REQUIRED')
    from LLM.foundation.acceptance import require_local_path
    for path in (control.store.root,rag.root,source_root,inventory):require_local_path(path)
    busy=control.store.rows("SELECT type FROM jobs WHERE state IN ('queued','running','pause_requested','cancelling')")
    require(not any(r['type'] in HEAVY_JOBS for r in busy),'ACTIVE_MODEL_WORK_BLOCKS_DATA_CHANGE')
    from LLM.foundation import corpus,tokenizer
    from LLM.foundation.prepare_data import _approved_raw_rows
    project={**control.store.get('foundation_project',project_id),
        'policy':control.store.get('foundation_policy',project_id)}
    require(project.get('executionScope')=='pc_validation' and project.get('state')!='revoked'
        and not project.get('revocationPending'),'PC_VALIDATION_SCOPE_REQUIRED')
    original=corpus.resolve(control.store,project['activeCorpusId'])
    holdout_ids,expected_holdout=_holdout(control.store,original)
    imported=add_selection(rag,pre,source_root,inventory)
    selected=sorted(set(imported['selectedVersionIds'])|holdout_ids)
    source_ids=project['policy']['allowedSourceIds']
    # The optional filter is part of the staged prepare_data patch. RAG keeps
    # old+new; no unselected original document can enter the new corpus.
    new=corpus.build(control.store,project,_approved_raw_rows(rag,pre,source_ids,selected_version_ids=selected),
        {'policyVersion':rag.setting('policyVersion',0),'samplingPolicy':SAMPLING})
    actual_versions=set();actual_holdout={}
    for member in corpus.iter_members(control.store,new):
        actual_versions.update(link['versionId'] for link in member.get('lineage',[]) if link.get('versionId'))
        if member.get('versionId'):actual_versions.add(member['versionId'])
    require(actual_versions<=set(selected),'UNSELECTED_CORPUS_VERSION')
    _,actual_holdout=_holdout(control.store,new)
    require(actual_holdout==expected_holdout,'HOLDOUT_IDENTITY_CHANGED')
    tok=tokenizer.load(control.store,project['tokenizerId']);counts={}
    if measure_tokens:
        for split in ('train','dev'):
            counts[split]=sum(len(tok.encode_ordinary(row['text']))+1 for row in corpus.rows(control.store,new,split))
    missing=sorted(set(selected)-actual_versions)
    report={'id':VERSION,'actor':ACTOR,'state':'ready' if new['state']=='ready' and measure_tokens else 'held',
        'projectId':project_id,'rawCorpusId':new['id'],'rawCorpusHash':new['manifestHash'],
        'sourcePreparationId':'official-v1','sourceRawCorpusId':original['id'],'sourceRawCorpusHash':original['manifestHash'],
        'tokenizerId':project['tokenizerId'],'tokenizerHash':tok.tokenizer_hash,'selectionVersion':VERSION,
        'requestedVersionIdsHash':digest(selected),'selectedVersionIdsHash':digest(sorted(actual_versions)),
        'selectedVersionIds':sorted(actual_versions),'acceptedVersionIdsHash':digest(sorted(actual_versions)),
        'excludedSelectedVersionIds':missing,'holdoutUnchanged':True,'originalRightsPreserved':True,
        'originalRagRecordsUnchanged':True,'selectionImport':{k:v for k,v in imported.items() if k!='selectedVersionIds'},
        'requirements':{'trainRawBytes':new['splitBytes']['train'],'trainGroups':new['groupCounts']['train'],
            'devGroups':new['groupCounts']['dev'],'devTargetTokens':counts.get('dev'),'trainTargetTokens':counts.get('train'),
            'vocabSize':tok.vocab_size},'samplingPolicy':SAMPLING,'requiresNewCalibration':True,
        'continueCorpus':False,'projectActiveDataChanged':False,'modelTrainingPerformed':False,'nasModified':False,
        'reasonCodes':new.get('reasonCodes',[]) or ([] if measure_tokens else ['TOKEN_COUNT_REQUIRED']),'observedAt':now()}
    if (new['splitBytes']['train']<16*1024**2 or new['groupCounts']['train']<20
            or new['groupCounts']['dev']<20 or (counts.get('dev') or 0)<100000 or tok.vocab_size!=16384):
        report.update(state='held',reasonCodes=sorted(set([*report['reasonCodes'],'DATA_READINESS_REQUIRED'])))
    path=control.store.root/'reports/preparations'/('official-python-v2-'+new['id']+'.json')
    atomic_json(path,report)
    atomic_json(Path(source_root)/'reports/readiness-code-v2.json',report)
    record=control.store.put('foundation_data_preparation',{**report,'reportPath':path.relative_to(control.store.root).as_posix()},VERSION)
    control.foundation.validate_policy(control.store.get('foundation_project',project_id))
    require(control.store.get('foundation_project',project_id)['activeCorpusId']==original['id']
        and corpus.resolve(control.store,original['id'])['state'] in ('ready','verified'),'PARENT_DATA_CHANGED')
    return record
