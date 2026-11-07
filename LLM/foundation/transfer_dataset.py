"""Fixed-ID offline import of audited RAG rights/blobs and prepared LLM data.

Applying requires an operator-approved package hash and leaves LLM maintenance
enabled. Two database commits are journaled; an interruption stays review-held.
No jobs, weights, credentials, settings or release certificates are imported.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
from pathlib import Path
import re
import shutil
import stat
import zipfile

from LLM.core import require, digest, file_hash, canonical, now, uid, atomic_json
from .transfer import _safe, CHUNK, MAX_PACKAGE_BYTES

RAG_KINDS = frozenset(('source','source_unit','document','version','training_example','training_snapshot','source_rights_receipt'))
LLM_KINDS = frozenset(('foundation_project','foundation_policy','foundation_corpus','foundation_tokenizer',
    'foundation_data_preparation','foundation_split','foundation_content_split','foundation_repository_family',
    'foundation_near_duplicate','foundation_split_conflict','foundation_near_fingerprint','fixture',
    'foundation_source_oracle_binding'))
ACTOR = 'user-delegated-public-source-audit-v1'


def _file_allowed(scope, relative):
    if not isinstance(relative,str) or '\\' in relative: return False
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts or len(path.parts) < 2: return False
    if scope == 'rag': return path.parts[0] in ('blobs','exports','rights')
    return scope == 'llm' and ((len(path.parts) >= 3 and path.parts[0] == 'foundation'
        and path.parts[1] in ('corpora','tokenizers','preparation','preparations')) or path.parts[0] == 'reports')


def _inventory_valid(inventory):
    require(isinstance(inventory,dict) and isinstance(inventory.get('metadata'),dict)
        and isinstance(inventory.get('records'),list) and isinstance(inventory.get('files'),list), 'TRANSFER_INVENTORY_INVALID')
    require(len(inventory['records']) <= 100000 and len(inventory['files']) <= 100000, 'TRANSFER_SIZE_LIMIT')
    identities, filenames = set(), set()
    sources, receipts, allowed, evidence = set(), set(), set(), {}
    record_bytes = 0
    bindings=[];indexed={}
    require(len(canonical(inventory['metadata']).encode()) <= 1024**2, 'TRANSFER_SIZE_LIMIT')
    for item in inventory['records']:
        scope,kind,identity,body = (item.get(k) for k in ('scope','kind','id','body'))
        require(scope in ('rag','llm') and kind in (RAG_KINDS if scope == 'rag' else LLM_KINDS)
            and isinstance(identity,str) and isinstance(body,dict) and body.get('id') == identity
            and (scope,kind,identity) not in identities, 'TRANSFER_RECORD_INVALID')
        amount = len(canonical(body).encode()); record_bytes += amount
        require(amount <= 8*1024**2 and record_bytes <= 128*1024**2, 'TRANSFER_SIZE_LIMIT')
        identities.add((scope,kind,identity))
        indexed[(scope,kind,identity)]=body
        if kind=='foundation_source_oracle_binding':
            from . import source_oracle_binding as oracle
            core={key:value for key,value in body.items() if key not in ('id','version','createdAt','updatedAt','receiptHash')}
            require(body.get('version')==1 and body.get('receiptHash')==digest(core)
                and body.get('helperVersion')==oracle.VERSION and body.get('helperHash')==oracle.implementation_hash()
                and body.get('actor')==ACTOR and body.get('state') in ('matched','mismatch','unresolved')
                and body.get('neverTrain') is True and body.get('modelQualityEvidence') is False
                and body.get('hostSourceExecuted') is False,'TRANSFER_ORACLE_BINDING_INVALID')
            bindings.append(body)
        if kind == 'source':
            require(body.get('trainingPolicy') == 'allowed', 'SOURCE_NOT_ALLOWED'); sources.add(identity)
        if kind == 'source_rights_receipt':
            require(body.get('actor') == ACTOR and body.get('state') in ('rights_verified','approved','verified'),
                    'TRANSFER_RIGHTS_RECEIPT_REQUIRED')
            source_id = body.get('sourceId',identity)
            require(re.fullmatch('[0-9a-f]{40}',str(body.get('revision','')))
                and re.fullmatch('[0-9a-f]{64}',str(body.get('archiveSha256',''))), 'TRANSFER_RIGHTS_RECEIPT_REQUIRED')
            licenses = body.get('licenses',[])
            permitted = [r for r in licenses if r.get('licenseId') in ('MIT','Apache-2.0','BSD-2-Clause','BSD-3-Clause','PSF-2.0')]
            require(permitted, 'TRANSFER_RIGHTS_RECEIPT_REQUIRED')
            evidence[source_id] = {r.get('sha256') for r in permitted}
            receipts.add(source_id)
        if kind == 'foundation_policy': allowed.update(body.get('allowedSourceIds',[]))
        if kind == 'foundation_project':
            require(not body.get('genesisId') and not body.get('latestCheckpointId'), 'TRANSFER_WEIGHTS_FORBIDDEN')
    require(bool(sources) and sources <= receipts and allowed <= sources, 'TRANSFER_RIGHTS_RECEIPT_REQUIRED')
    total = 0
    for item in inventory['files']:
        scope,relative = item.get('scope'),item.get('relativePath')
        require(_file_allowed(scope,relative) and (scope,relative) not in filenames, 'TRANSFER_PATH_UNSAFE')
        require(isinstance(item.get('sizeBytes'),int) and 0 <= item['sizeBytes'] <= MAX_PACKAGE_BYTES
            and isinstance(item.get('sha256'),str) and len(item['sha256']) == 64, 'TRANSFER_INVENTORY_INVALID')
        if scope == 'rag' and Path(relative).parts[0] == 'blobs':
            require(Path(relative).name == item['sha256'], 'TRANSFER_BLOB_HASH_MISMATCH')
        filenames.add((scope,relative)); total += item['sizeBytes']
    require(total <= MAX_PACKAGE_BYTES, 'TRANSFER_SIZE_LIMIT')
    for binding in bindings:
        prep=indexed.get(('llm','foundation_data_preparation',binding.get('preparationId')),{} )
        example=indexed.get(('rag','training_example',binding.get('sourceExampleId')),{} )
        version=indexed.get(('rag','version',binding.get('sourceVersionId')),{} )
        rights=indexed.get(('rag','source_rights_receipt',binding.get('rightsReceiptId')),{} )
        fixture=indexed.get(('llm','fixture',binding.get('originalFixtureId')),{} )
        require(prep.get('implementationVersion')=='seed-source-context-sft-v3'
            and prep.get('sftCorpusId')==binding.get('sourceCorpusId')
            and prep.get('sftCorpusHash')==binding.get('sourceCorpusHash')
            and example and digest(example)==binding.get('sourceExampleHash')
            and version and rights and fixture and digest(rights)==binding.get('rightsReceiptHash')
            and digest(fixture)==binding.get('originalFixtureHash')
            and digest(example.get('independentVerification',{}).get('source'))==binding.get('referenceHash'),
            'TRANSFER_ORACLE_BINDING_GRAPH_INCOMPLETE')
        for field,maximum in (('targetModuleBlobHash',2*1024**2),('testModuleBlobHash',512*1024),('assertionBlobHash',512*1024)):
            checksum=binding.get(field)
            require(isinstance(checksum,str) and re.fullmatch('[0-9a-f]{64}',checksum),
                'TRANSFER_ORACLE_BINDING_INVALID')
            relative='blobs/'+checksum[:2]+'/'+checksum
            require(any(item['scope']=='rag' and item['relativePath']==relative
                and item['sha256']==checksum and item['sizeBytes']<=maximum for item in inventory['files']),
                'TRANSFER_ORACLE_BINDING_BLOB_MISSING')
    for source_id, hashes in evidence.items():
        shipped = {item['sha256'] for item in inventory['files'] if item['scope'] == 'rag'
            and len(Path(item['relativePath']).parts) >= 3
            and Path(item['relativePath']).parts[:2] == ('rights',source_id)}
        require(hashes <= shipped, 'TRANSFER_LICENSE_EVIDENCE_MISSING')
    return total


def export_inventory_package(llm_store, rag_store, inventory, destination):
    total = _inventory_valid(inventory)
    output = _safe(destination); require(not output.exists(), 'TRANSFER_DESTINATION_EXISTS')
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary = output.with_name(output.name+'.tmp-'+uid())
    records = ('\n'.join(canonical(record) for record in inventory['records'])+'\n').encode('utf-8')
    require(len(records) <= 128*1024**2, 'TRANSFER_SIZE_LIMIT')
    file_records = {item['scope']+'/'+item['relativePath']:item for item in inventory['files']}
    manifest = {'schemaVersion':1,'kind':'foundation_prepared_dataset','metadata':inventory['metadata'],
        'recordsHash':hashlib.sha256(records).hexdigest(),'recordCount':len(inventory['records']),
        'files':file_records,'containsWeights':False,'preservesFixedIds':True,'rightsActor':ACTOR,
        'automaticTraining':False,'requiresOperatorHashApproval':True}
    manifest['packageId'] = digest(manifest)
    try:
        with zipfile.ZipFile(temporary,'x',compression=zipfile.ZIP_STORED,allowZip64=True) as archive:
            archive.writestr('records.jsonl',records)
            for name,item in sorted(file_records.items()):
                root = llm_store.root if item['scope'] == 'llm' else rag_store.root
                source = _safe(root/item['relativePath'])
                require(source.is_relative_to(root) and source.is_file() and source.stat().st_size == item['sizeBytes']
                    and file_hash(source) == item['sha256'], 'TRANSFER_HASH_MISMATCH')
                archive.write(source,name)
            archive.writestr('manifest.json',canonical(manifest).encode('utf-8'))
        with temporary.open('r+b') as handle:
            import os
            os.fsync(handle.fileno())
        inspect_preparation(temporary)
        temporary.replace(output)
    finally:
        if temporary.exists(): temporary.unlink()
    return {'path':str(output),'packageId':manifest['packageId'],'sha256':file_hash(output),
        'dataBytes':total,'recordCount':manifest['recordCount'],'deployed':False,'nasModified':False}


def export_preparation(llm_store, rag_store, preparation_id, destination):
    from .prepare_data import export_inventory
    return export_inventory_package(llm_store,rag_store,export_inventory(llm_store,preparation_id,rag_store),destination)


def inspect_preparation(path):
    path = _safe(path)
    require(path.is_file() and path.stat().st_size <= MAX_PACKAGE_BYTES+256*1024**2, 'TRANSFER_SIZE_LIMIT')
    with zipfile.ZipFile(path,'r') as archive:
        entries = archive.infolist(); names = [item.filename for item in entries]
        require(len(entries) <= 100002 and len(set(names)) == len(names)
            and {'manifest.json','records.jsonl'} <= set(names), 'TRANSFER_ARCHIVE_INVALID')
        for item in entries:
            require(not item.is_dir() and not stat.S_ISLNK(item.external_attr >> 16)
                and not item.flag_bits & 1 and item.compress_type in (zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED)
                and item.file_size <= MAX_PACKAGE_BYTES, 'TRANSFER_ARCHIVE_INVALID')
        require(archive.getinfo('manifest.json').file_size <= 32*1024**2
            and archive.getinfo('records.jsonl').file_size <= 128*1024**2, 'TRANSFER_SIZE_LIMIT')
        manifest = json.loads(archive.read('manifest.json').decode('utf-8'))
        require(manifest.get('schemaVersion') == 1 and manifest.get('kind') == 'foundation_prepared_dataset'
            and manifest.get('containsWeights') is False and manifest.get('rightsActor') == ACTOR
            and manifest.get('packageId') == digest({k:v for k,v in manifest.items() if k != 'packageId'}),
            'TRANSFER_MANIFEST_INVALID')
        require(set(manifest['files']) == set(names)-{'manifest.json','records.jsonl'}, 'TRANSFER_ARCHIVE_INVALID')
        raw = archive.read('records.jsonl')
        require(hashlib.sha256(raw).hexdigest() == manifest['recordsHash'], 'TRANSFER_HASH_MISMATCH')
        records = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line]
        inventory = {'metadata':manifest['metadata'],'records':records,'files':list(manifest['files'].values())}
        total = _inventory_valid(inventory)
        require(len(records) == manifest['recordCount'], 'TRANSFER_MANIFEST_INVALID')
        for name,item in manifest['files'].items():
            require(name == item['scope']+'/'+item['relativePath'], 'TRANSFER_PATH_UNSAFE')
            checksum = hashlib.sha256(); count = 0
            with archive.open(name) as source:
                while chunk := source.read(CHUNK): checksum.update(chunk); count += len(chunk)
            require(checksum.hexdigest() == item['sha256'] and count == item['sizeBytes'], 'TRANSFER_HASH_MISMATCH')
    return {'manifest':manifest,'inventory':inventory,'packageFileHash':file_hash(path),'dataBytes':total,
        'validated':True,'nasModified':False,'deployed':False,'automaticTraining':False}


def _existing(store,kind,identity):
    try: return store.get(kind,identity)
    except Exception as exc:
        if getattr(exc,'code',None) == 'NOT_FOUND': return None
        raise


def _collisions(llm_store,rag_store,inventory):
    for item in inventory['records']:
        store = llm_store if item['scope'] == 'llm' else rag_store
        old = _existing(store,item['kind'],item['id'])
        require(old is None or old == item['body'], 'TRANSFER_RECORD_CONFLICT', kind=item['kind'],id=item['id'])
    for item in inventory['files']:
        root = llm_store.root if item['scope'] == 'llm' else rag_store.root
        target = _safe(root/item['relativePath'])
        require(target.is_relative_to(root) and (not target.exists() or target.is_file()
            and file_hash(target) == item['sha256']), 'TRANSFER_FILE_CONFLICT')


def import_preparation(llm_store, rag_store, path, *, dry_run=True, expected_package_hash=None, approve_rights=False,target_kind='nas'):
    require(target_kind in ('nas','local_validation'),'TRANSFER_TARGET_KIND_INVALID')
    if target_kind=='local_validation':
        from .acceptance import require_local_path
        require_local_path(llm_store.root);require_local_path(rag_store.root)
    inspection = inspect_preparation(path); inventory,manifest = inspection['inventory'],inspection['manifest']
    require(expected_package_hash is None or expected_package_hash == inspection['packageFileHash'], 'TRANSFER_HASH_MISMATCH')
    # SQLite read locks can alter a WAL shared-memory file. Dry-run deliberately
    # never opens target records; authoritative ID/rights checks happen at apply.
    for item in inventory['files']:
        root = llm_store.root if item['scope'] == 'llm' else rag_store.root
        target = _safe(root/item['relativePath'])
        require(target.is_relative_to(root) and (not target.exists() or target.is_file()
            and file_hash(target) == item['sha256']), 'TRANSFER_FILE_CONFLICT')
    for scope, store in (('llm',llm_store),('rag',rag_store)):
        needed = sum(item['sizeBytes'] for item in inventory['files'] if item['scope'] == scope
            and not (store.root/item['relativePath']).exists())
        require(shutil.disk_usage(store.root).free >= needed+512*1024**2, 'TRANSFER_DISK_LOW')
    report = {key:value for key,value in inspection.items() if key != 'inventory'}
    report.update(dryRun=dry_run,state='validated',trainingStarted=False,
        targetRecordValidation='deferred_until_apply' if dry_run else 'authoritative',
        targetKind=target_kind,targetModified=False,nasModified=False)
    if dry_run: return report
    require(approve_rights is True and expected_package_hash == inspection['packageFileHash'], 'TRANSFER_OPERATOR_APPROVAL_REQUIRED')
    _collisions(llm_store,rag_store,inventory)
    identity = manifest['packageId']
    with llm_store.tx():
        require(not llm_store.rows("SELECT id FROM jobs WHERE state IN ('running','pause_requested','cancelling') LIMIT 1"),
            'TRANSFER_ACTIVE_JOBS')
        llm_store.put('setting',{'enabled':True,'restoreReviewRequired':True,'reason':'OFFLINE_DATA_IMPORT_REVIEW_REQUIRED'},'deploymentMaintenance')
        llm_store.put('offline_transfer',{'state':'importing','packageHash':inspection['packageFileHash'],
            'packageId':identity,'recordCount':manifest['recordCount'],'automaticTraining':False},identity)
    # Each blob is CAS-published before any record references it. Interrupted
    # cross-DB work remains safe under persistent deployment maintenance.
    with zipfile.ZipFile(path,'r') as archive:
        for name,item in manifest['files'].items():
            root = llm_store.root if item['scope'] == 'llm' else rag_store.root
            target = _safe(root/item['relativePath'])
            if target.exists(): continue
            target.parent.mkdir(parents=True,exist_ok=True)
            temporary = target.with_name(target.name+'.import-'+uid())
            try:
                with archive.open(name) as source, temporary.open('xb') as out:
                    shutil.copyfileobj(source,out,length=CHUNK);out.flush()
                    import os
                    os.fsync(out.fileno())
                require(file_hash(temporary) == item['sha256'], 'TRANSFER_HASH_MISMATCH')
                # Link is an exclusive CAS publication, unlike overwrite rename.
                import os
                try: os.link(temporary,target)
                except FileExistsError: require(file_hash(target) == item['sha256'], 'TRANSFER_FILE_CONFLICT')
            finally:
                if temporary.exists(): temporary.unlink()
    with llm_store.tx():
        with rag_store.tx() as rag_db:
            _collisions(llm_store,rag_store,inventory)
            for item in inventory['records']:
                body = canonical(item['body'])
                if item['scope'] == 'llm':
                    llm_store.db.execute('INSERT OR IGNORE INTO records(kind,id,body) VALUES(?,?,?)',(item['kind'],item['id'],body))
                else:
                    rag_db.execute('INSERT OR IGNORE INTO records(kind,id,body,created_at,updated_at) VALUES(?,?,?,?,?)',
                        (item['kind'],item['id'],body,item['body'].get('createdAt',now()),item['body'].get('updatedAt',now())))
            llm_store.put('offline_transfer',{'state':'imported_review_required','completedAt':now(),
                'fixedIdsPreserved':True,'rightsActor':ACTOR,'productionTrainingTokens':0},identity)
    report.update(state='imported_review_required',nasModified=target_kind=='nas',targetModified=True,deploymentMaintenance=True,
        fixedIdsPreserved=True,productionTrainingTokens=0)
    return report
