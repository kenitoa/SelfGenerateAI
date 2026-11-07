"""Offline, content-addressed approved-data packages; imports never activate training."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import zipfile

from LLM.core import require, digest, file_hash, atomic_json, canonical, inside, uid

MAX_PACKAGE_BYTES = 20*1024**3
CHUNK = 1024**2


def _safe(path):
    path = Path(path).absolute()
    require(all(not part.is_symlink() for part in (*path.parents, path)), 'TRANSFER_PATH_UNSAFE')
    return path.resolve()


def export_package(store, project_id, corpus_id, destination, checkpoint_id=None, *, include_dev=False):
    """Export approved corpus rows only. No weights, release/audit data or secrets."""
    from .corpus import verify, rows
    project = store.get('foundation_project', project_id)
    corpus = store.get('foundation_corpus', corpus_id)
    require(corpus['projectId'] == project_id and corpus.get('state') == 'ready'
            and project.get('state') != 'revoked', 'TRANSFER_POLICY_REVOKED')
    verify(store, corpus)
    policy = store.get('foundation_policy', project_id, {})
    require(policy.get('allowedSourceIds') or project.get('testOnly'), 'CORPUS_PERMISSION_REQUIRED')
    token = store.get('foundation_tokenizer', project.get('tokenizerId') or '')
    output = _safe(destination)
    require(not output.exists(), 'TRANSFER_DESTINATION_EXISTS', status=409)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name+'.tmp-'+uid())
    metadata = {'schemaVersion':1, 'kind':'foundation_training_data', 'projectId':project_id,
        'corpusId':corpus_id, 'corpusHash':corpus['manifestHash'], 'tokenizerId':token['id'],
        'tokenizerHash':token['tokenizerHash'], 'configHash':digest(project['config']),
        'allowedSourceIds':sorted(policy.get('allowedSourceIds',[])), 'policyVersion':policy.get('version',0),
        'checkpointReference':checkpoint_id, 'containsWeights':False, 'releaseAuditIncluded':False,
        'testOnly':project.get('testOnly') is True, 'automaticTraining':False}
    metadata['requestHash'] = digest(metadata)
    files, total = {}, 0
    try:
        with zipfile.ZipFile(temporary, 'x', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for split in (('train','dev') if include_dev else ('train',)):
                checksum, count, size = hashlib.sha256(), 0, 0
                with archive.open(split+'.jsonl','w',force_zip64=True) as out:
                    for row in rows(store, corpus, split, purpose='transfer_export'):
                        lineage = row.get('lineage',[])
                        require(project.get('testOnly') or lineage and all(item.get('sourceId') in metadata['allowedSourceIds']
                            for item in lineage), 'SOURCE_NOT_ALLOWED')
                        encoded = (canonical(row)+'\n').encode('utf-8')
                        total += len(encoded); size += len(encoded)
                        require(total <= MAX_PACKAGE_BYTES, 'TRANSFER_SIZE_LIMIT')
                        checksum.update(encoded); out.write(encoded); count += 1
                files[split+'.jsonl'] = {'sha256':checksum.hexdigest(), 'sizeBytes':size, 'rows':count, 'split':split}
            require(files['train.jsonl']['rows'] > 0, 'EMPTY_TRAIN_CORPUS')
            manifest = {**metadata, 'files':files}
            manifest['packageId'] = digest(manifest)
            archive.writestr('manifest.json', canonical(manifest).encode('utf-8'))
        with temporary.open('r+b') as handle: os.fsync(handle.fileno())
        require(store.get('foundation_corpus',corpus_id).get('state') == 'ready'
                and store.get('foundation_project',project_id).get('state') != 'revoked', 'TRANSFER_POLICY_REVOKED')
        temporary.replace(output)
    finally:
        if temporary.exists(): temporary.unlink()
    return {'path':str(output), 'packageId':manifest['packageId'], 'requestHash':manifest['requestHash'],
        'sha256':file_hash(output), 'sizeBytes':output.stat().st_size, 'deployed':False, 'nasModified':False}


def inspect_package(path):
    path = _safe(path)
    require(path.is_file() and path.stat().st_size <= MAX_PACKAGE_BYTES+2*1024**2, 'TRANSFER_SIZE_LIMIT')
    with zipfile.ZipFile(path, 'r') as archive:
        entries = archive.infolist(); names = [item.filename for item in entries]
        require(len(entries) <= 3 and len(set(names)) == len(names) and 'manifest.json' in names
                and set(names) <= {'manifest.json','train.jsonl','dev.jsonl'}, 'TRANSFER_ARCHIVE_INVALID')
        for item in entries:
            require(not item.is_dir() and not stat.S_ISLNK(item.external_attr >> 16)
                    and not item.flag_bits & 1 and item.file_size <= MAX_PACKAGE_BYTES
                    and item.compress_type in (zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED), 'TRANSFER_ARCHIVE_INVALID')
        require(archive.getinfo('manifest.json').file_size <= 1024**2, 'TRANSFER_MANIFEST_INVALID')
        manifest = json.loads(archive.read('manifest.json').decode('utf-8'))
        require(manifest.get('schemaVersion') == 1 and manifest.get('kind') == 'foundation_training_data'
                and manifest.get('containsWeights') is False and manifest.get('releaseAuditIncluded') is False
                and manifest.get('packageId') == digest({k:v for k,v in manifest.items() if k != 'packageId'}),
                'TRANSFER_MANIFEST_INVALID')
        require(set(manifest.get('files',{})) == set(names)-{'manifest.json'} and 'train.jsonl' in manifest['files'],
                'TRANSFER_MANIFEST_INVALID')
        metadata = {k:v for k,v in manifest.items() if k not in ('files','packageId','requestHash')}
        require(manifest.get('requestHash') == digest(metadata), 'TRANSFER_REQUEST_MISMATCH')
        total = 0
        for name, record in manifest['files'].items():
            checksum, length, count = hashlib.sha256(), 0, 0
            require(record.get('split') == name.removesuffix('.jsonl'), 'SPLIT_LEAKAGE')
            with archive.open(name) as source:
                while line := source.readline(8*1024**2+1):
                    require(len(line) <= 8*1024**2, 'CORPUS_RECORD_TOO_LARGE')
                    checksum.update(line); length += len(line); total += len(line); count += 1
                    require(total <= MAX_PACKAGE_BYTES, 'TRANSFER_SIZE_LIMIT')
                    row = json.loads(line.decode('utf-8'))
                    require(row.get('split') == record['split'] and isinstance(row.get('text'),str)
                        and len(row['text'].encode('utf-8')) <= 2*1024**2
                        and row.get('hash') == digest(row['text'].encode('utf-8')), 'TRANSFER_ROW_INVALID')
            require(checksum.hexdigest() == record.get('sha256') and length == record.get('sizeBytes')
                    and count == record.get('rows'), 'TRANSFER_HASH_MISMATCH')
    return {'manifest':manifest, 'packageFileHash':file_hash(path), 'sizeBytes':path.stat().st_size,
        'validated':True, 'automaticTraining':False, 'nasModified':False}


def import_package(store, path, expected_request_hash=None, dry_run=True):
    """Validate against NAS project policy. Apply stores a quarantined input only."""
    require(isinstance(dry_run,bool), 'CONFIG_INVALID')
    inspection = inspect_package(path); manifest = inspection['manifest']
    require(expected_request_hash is None or manifest['requestHash'] == expected_request_hash, 'TRANSFER_REQUEST_MISMATCH')
    project = store.get('foundation_project',manifest['projectId'])
    policy = store.get('foundation_policy',project['id'],{})
    token = store.get('foundation_tokenizer',project.get('tokenizerId') or '')
    require(project.get('state') != 'revoked' and token.get('tokenizerHash') == manifest['tokenizerHash']
            and digest(project['config']) == manifest['configHash'], 'TRANSFER_PROJECT_MISMATCH')
    require(project.get('testOnly') or not manifest.get('testOnly'), 'FIXTURE_NOT_FOR_PRODUCTION')
    require(project.get('testOnly') or set(manifest['allowedSourceIds']) <= set(policy.get('allowedSourceIds',[])),
            'SOURCE_NOT_ALLOWED')
    report = {**inspection, 'dryRun':dry_run, 'state':'validated' if dry_run else 'quarantined',
        'trainingStarted':False, 'deployed':False, 'nasModified':not dry_run}
    if dry_run: return report
    from .retention import admit
    admit(store,project,sum(item['sizeBytes'] for item in manifest['files'].values())+1024**2)
    identity = manifest['packageId']
    destination = _safe(store.root/'foundation'/'inputs'/('offline-'+identity))
    require(destination.is_relative_to(store.root) and not destination.exists(), 'TRANSFER_DESTINATION_EXISTS')
    stage = _safe(store.root/'foundation'/'staging'/('transfer-'+uid())); stage.mkdir(parents=True)
    try:
        with zipfile.ZipFile(path,'r') as archive:
            for name in manifest['files']:
                with archive.open(name) as source, (stage/name).open('xb') as out:
                    shutil.copyfileobj(source,out,length=CHUNK); out.flush(); os.fsync(out.fileno())
                require(file_hash(stage/name) == manifest['files'][name]['sha256'], 'TRANSFER_HASH_MISMATCH')
        atomic_json(stage/'manifest.json',manifest)
        destination.parent.mkdir(parents=True,exist_ok=True)
        with store.tx():
            require(store.get('foundation_project',project['id']).get('state') != 'revoked', 'POLICY_REVOKED')
            stage.replace(destination)
            record = store.put('foundation_input', {'projectId':project['id'], 'state':'quarantined',
                'storagePath':destination.relative_to(store.root).as_posix(), 'packageHash':inspection['packageFileHash'],
                'requestHash':manifest['requestHash'], 'automaticTraining':False, 'requiresRagPolicyRevalidation':True}, identity)
            report['inputId'] = record['id']
    finally:
        if stage.exists(): shutil.rmtree(stage)
    return report


def upload_command(package, host, destination, *, expected_hash=None):
    """Prepare argv only; caller must explicitly execute sending. No deployment."""
    with zipfile.ZipFile(_safe(package),'r') as archive:
        require(archive.getinfo('manifest.json').file_size <= 32*1024**2, 'TRANSFER_MANIFEST_INVALID')
        kind = json.loads(archive.read('manifest.json').decode('utf-8')).get('kind')
    if kind == 'foundation_prepared_dataset':
        from .transfer_dataset import inspect_preparation
        report = inspect_preparation(package)
    else:
        report = inspect_package(package)
    require(expected_hash is None or report['packageFileHash'] == expected_hash, 'TRANSFER_HASH_MISMATCH')
    require(isinstance(host,str) and re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*@[A-Za-z0-9][A-Za-z0-9_.-]*',host), 'TRANSFER_HOST_INVALID')
    require(isinstance(destination,str) and re.fullmatch(r'/[A-Za-z0-9_./-]+',destination)
            and '..' not in Path(destination).parts, 'TRANSFER_PATH_UNSAFE')
    require(re.fullmatch(r'[A-Za-z0-9_.-]+',Path(package).name), 'TRANSFER_PATH_UNSAFE')
    return {'argv':['scp','--',str(_safe(package)),host+':'+destination.rstrip('/')+'/'+Path(package).name],
        'dryRun':True, 'deployed':False, 'nasModified':False, 'packageFileHash':report['packageFileHash']}


def export_preparation(llm_store, rag_store, preparation_id, destination):
    from .transfer_dataset import export_preparation as export
    return export(llm_store,rag_store,preparation_id,destination)


def inspect_preparation(path):
    from .transfer_dataset import inspect_preparation as inspect
    return inspect(path)


def import_preparation(llm_store, rag_store, path, **options):
    from .transfer_dataset import import_preparation as incoming
    return incoming(llm_store,rag_store,path,**options)
