"""Committed model backups; raw inputs/scratch are excluded; restores need review."""
import contextlib
import json
import os
from pathlib import Path
import re
import shutil
import threading
import time

import apsw
from LLM.core import require, uid, file_hash, atomic_json, now, canonical, Store
from . import encryption
from datetime import datetime, timezone, timedelta

LEGACY_DIRS = ('artifacts', 'snapshots', 'releases', 'checkpoints', 'self-code')
FOUNDATION_DIRS = ('corpora', 'tokenizers', 'genesis', 'checkpoints', 'token-shards', 'growth')
ACTIVE = ('running', 'pause_requested', 'cancelling')
TRAINING = ('train', 'foundation_pretrain', 'foundation_cpt', 'foundation_sft')


def _root(value):
    path = Path(value).absolute()
    require(all(not p.is_symlink() for p in (*path.parents, path)), 'BACKUP_PATH_UNSAFE')
    return path.resolve()


def _path(root, relative):
    require(isinstance(relative, str) and relative and '\\' not in relative, 'BACKUP_PATH_UNSAFE')
    parts = Path(relative).parts
    require(not Path(relative).is_absolute() and '..' not in parts and '.' not in parts, 'BACKUP_PATH_UNSAFE')
    target = root
    for part in parts:
        target = target/part
        require(not target.is_symlink(), 'BACKUP_PATH_UNSAFE')
    require(target.resolve().is_relative_to(root), 'BACKUP_PATH_UNSAFE')
    return target


def _allowed(name):
    parts = Path(name).parts
    return name == 'llm.db' or (len(parts) >= 2 and parts[0] in LEGACY_DIRS) or (
        len(parts) >= 3 and parts[0] == 'foundation' and parts[1] in FOUNDATION_DIRS)


def _files(root):
    """Never follow links or special files, even links to another local dir."""
    require(root.is_dir() and not root.is_symlink(), 'BACKUP_PATH_UNSAFE')
    with os.scandir(root) as entries:
        for entry in entries:
            require(not entry.is_symlink(), 'BACKUP_PATH_UNSAFE')
            path = Path(entry.path)
            if entry.is_dir(follow_symlinks=False):
                yield from _files(path)
            else:
                require(entry.is_file(follow_symlinks=False), 'BACKUP_PATH_UNSAFE')
                yield path


def _records(db, kind):
    return [json.loads(row[0]) for row in db.execute('SELECT body FROM records WHERE kind=?', (kind,))]


def _pending(db, excluded=None):
    require(not any(row.get('revocationPending') for row in _records(db,'foundation_project')),'BACKUP_POLICY_REVOCATION_PENDING',status=409)
    require(not any(r.get('state') == 'deleting' for r in _records(db, 'foundation_retention_intent')),
            'BACKUP_GC_IN_PROGRESS', status=409)
    require(not any(r.get('id') == 'storageGcInProgress' and r.get('enabled') for r in _records(db, 'setting')),
            'BACKUP_GC_IN_PROGRESS', status=409)
    for identity, kind, state, payload in db.execute("SELECT id,type,state,payload FROM jobs WHERE state IN ('running','pause_requested','cancelling','queued','paused')"):
        if identity == excluded: continue
        require(state not in ACTIVE, 'BACKUP_ACTIVE_JOBS', status=409, jobId=identity)
        body = json.loads(payload)
        run_kind = 'training_run' if kind == 'train' else 'foundation_run'
        run = next((r for r in _records(db, run_kind) if r['id'] == identity), {})
        checkpoint = body.get('checkpointId') or run.get('checkpointId') or run.get('lastCheckpointId')
        require(kind in TRAINING and checkpoint, 'BACKUP_UNFINISHED_INPUTS', status=409, jobId=identity)
        record_kind = 'checkpoint' if kind == 'train' else 'foundation_checkpoint'
        require(any(r['id'] == checkpoint and r.get('state') not in ('deleted', 'archived', 'reclaiming')
                    for r in _records(db, record_kind)), 'BACKUP_CHECKPOINT_REQUIRED', status=409, jobId=identity)
    require(not any(r.get('state') not in ('deleted', 'cancelled', 'archived') for r in _records(db, 'foundation_input')),
            'BACKUP_UNFINISHED_INPUTS', status=409)


def _integrity(database):
    db = apsw.Connection(str(database), flags=apsw.SQLITE_OPEN_READONLY)
    try:
        require(list(db.execute('PRAGMA integrity_check')) == [('ok',)], 'BACKUP_DATABASE_CORRUPT')
    finally:
        db.close()


def _foundation_integrity(root, database):
    db = apsw.Connection(str(database), flags=apsw.SQLITE_OPEN_READONLY)
    needed = set()
    try:
        for kind, body in db.execute("SELECT kind,body FROM records WHERE kind LIKE 'foundation_%'"):
            record = json.loads(body)
            if kind in ('foundation_retention_intent', 'foundation_input'):
                continue  # audit/deleted input metadata is outside model backup scope
            if not record.get('storagePath') or record.get('state') in ('deleted', 'archived', 'revoked', 'quarantined', 'evicted'):
                continue
            relative = record['storagePath']
            require(_allowed(relative) and relative.startswith('foundation/'), 'BACKUP_FOUNDATION_PATH_UNSUPPORTED')
            directory = _path(root, relative)
            require(directory.is_dir(), 'BACKUP_FOUNDATION_FILES_MISSING')
            require(record.get('manifestHash'), 'BACKUP_MANIFEST_INVALID')
            if record.get('manifestHash'):
                manifest_path = _path(root, relative+'/manifest.json')
                require(manifest_path.is_file() and file_hash(manifest_path) == record['manifestHash'], 'BACKUP_HASH_MISMATCH')
                require(manifest_path.stat().st_size <= 16*1024**2, 'BACKUP_MANIFEST_INVALID')
                manifest = json.loads(manifest_path.read_text('utf-8'))
                needed.add(manifest_path.relative_to(root).as_posix())
                for name, expected in manifest.get('files', {}).items():
                    target = _path(directory, name)
                    require(target.is_file() and file_hash(target) == expected, 'BACKUP_HASH_MISMATCH')
                    needed.add(target.relative_to(root).as_posix())
    finally:
        db.close()
    return needed


def _assert_gate(store, identity, errors):
    require(not errors, 'BACKUP_LEASE_LOST', status=409)
    maintenance = store.get('setting', 'deploymentMaintenance', {})
    lease = store.get('setting', 'foregroundLease', {})
    require(maintenance.get('enabled') and maintenance.get('backupOwner') == identity
            and lease.get('owner') == identity and lease.get('expires', 0) > time.time(),
            'BACKUP_LEASE_LOST', status=409)


@contextlib.contextmanager
def _quiesced(store, identity):
    # The sentinel pins legacy GC. Foreground ownership is honored atomically
    # inside Store.claim; deployment maintenance also gates normal workers.
    with store.tx():
        _pending(store.db)
        previous = store.get('setting', 'deploymentMaintenance', {})
        require(not previous.get('backupInProgress'), 'BACKUP_ALREADY_RUNNING', status=409)
        foreground = store.get('setting', 'foregroundLease', {})
        require(foreground.get('expires', 0) <= time.time(), 'BACKUP_FOREGROUND_ACTIVE', status=409)
        current = store.put('setting', {'enabled': True, 'backupOwner': identity, 'backupInProgress': True}, 'deploymentMaintenance')
        store.put('setting', {'owner': identity, 'expires': time.time()+90, 'purpose': 'backup'}, 'foregroundLease')
        store.db.execute("INSERT INTO jobs(id,type,state,payload,created,updated,progress) VALUES(?,'backup_maintenance','paused',?,?,?,?)",
            (identity, canonical({'maintenanceOnly': True}), now(), now(), canonical({'stage': 'backup'})))
    ended, errors = threading.Event(), []
    def heartbeat():
        while not ended.wait(10):
            try:
                with store.tx():
                    require(store.get('setting', 'foregroundLease', {}).get('owner') == identity, 'BACKUP_LEASE_LOST')
                    require(store.get('setting', 'deploymentMaintenance', {}).get('backupOwner') == identity, 'BACKUP_LEASE_LOST')
                    store.put('setting', {'expires': time.time()+90}, 'foregroundLease')
            except Exception as exc:
                errors.append(exc); return
    thread = threading.Thread(target=heartbeat, daemon=True); thread.start()
    success = False
    try:
        yield errors
        require(not errors, 'BACKUP_LEASE_LOST', status=409)
        success = True
    finally:
        ended.set(); thread.join(timeout=1)
        with store.tx():
            value = store.get('setting', 'deploymentMaintenance', {})
            if value.get('backupOwner') == identity:
                enabled = previous.get('enabled', False) if value.get('version') == current['version'] else True
                store.put('setting', {'enabled': enabled, 'backupOwner': None, 'backupInProgress': False}, 'deploymentMaintenance')
            if store.get('setting', 'foregroundLease', {}).get('owner') == identity:
                store.db.execute("DELETE FROM records WHERE kind='setting' AND id='foregroundLease'")
            store.db.execute('UPDATE jobs SET state=?,updated=? WHERE id=?', ('completed' if success else 'failed', now(), identity))


def _lineage(database):
    result = {name: set() for name in ('sourceIds', 'documentIds', 'versionIds', 'projectIds', 'artifactIds')}
    names = {'sourceId': 'sourceIds', 'documentId': 'documentIds', 'versionId': 'versionIds',
             'projectId': 'projectIds', 'foundationProjectId': 'projectIds', 'artifactId': 'artifactIds'}
    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in names and isinstance(item, str): result[names[key]].add(item)
                visit(item)
        elif isinstance(value, list):
            for item in value: visit(item)
    db = apsw.Connection(str(database), flags=apsw.SQLITE_OPEN_READONLY)
    try:
        for kind, body in db.execute('SELECT kind,body FROM records'):
            record = json.loads(body); visit(record)
            if kind == 'foundation_project': result['projectIds'].add(record['id'])
            if kind == 'artifact': result['artifactIds'].add(record['id'])
    finally: db.close()
    return {key: sorted(value) for key, value in result.items()}


def backup(store, *, expires_days=30, key_id=None):
    require(isinstance(expires_days, int) and not isinstance(expires_days, bool) and 1 <= expires_days <= 365, 'BACKUP_EXPIRY_INVALID')
    identity = uid()
    directory = _path(store.root, 'backups/'+identity)
    own_key = key_id is None
    key_id = key_id or 'backup-'+identity
    key = encryption.get_key(store, key_id, create=own_key)
    scratch = encryption.key_directory(store)/('snapshot-'+identity+'.db')
    success = False
    try:
      with _quiesced(store, identity) as errors:
        directory.mkdir(parents=True)
        database = scratch
        database.touch(mode=0o600, exist_ok=False); encryption.private(database, create=True)
        # No source write transaction is held during SQLite/model file copying.
        with store.lock:
            target = apsw.Connection(str(database))
            try:
                with target.backup('main', store.db, 'main') as operation:
                    while not operation.done: operation.step(100)
                target.execute('PRAGMA journal_mode=DELETE')
            finally:
                target.close()
        copied_db = apsw.Connection(str(database), flags=apsw.SQLITE_OPEN_READONLY)
        try:
            _pending(copied_db, identity)
        finally:
            copied_db.close()
        # Only DB-referenced, manifest-verified foundation files are copied.
        # Uncommitted/orphan/private files elsewhere in foundation stay out.
        source_files = [('llm.db', database)]
        excluded_prefixes=[]
        source_db=apsw.Connection(str(database),flags=apsw.SQLITE_OPEN_READONLY)
        try:
            for body, in source_db.execute('SELECT body FROM records'):
                row=json.loads(body)
                if row.get('storagePath') and row.get('state') in ('revoked','quarantined','deleted','archived','evicted'):
                    excluded_prefixes.append(row['storagePath'].rstrip('/')+'/')
        finally:source_db.close()
        for name in LEGACY_DIRS:
            source = _path(store.root, name)
            if not source.exists(): continue
            source_files.extend((path.relative_to(store.root).as_posix(), path) for path in _files(source)
                                if not any((path.relative_to(store.root).as_posix()+'/').startswith(prefix) for prefix in excluded_prefixes))
        source_files.extend((name, _path(store.root, name)) for name in sorted(_foundation_integrity(store.root, database)))
        plaintext_files, files = {}, {}
        for relative, path in source_files:
            _assert_gate(store, identity, errors)
            destination = _path(directory, relative)
            before = file_hash(path)
            hashes = encryption.encrypt_file(path, destination, key, [identity, relative])
            require(hashes['plaintextHash'] == before == file_hash(path), 'BACKUP_SOURCE_CHANGED', status=409)
            plaintext_files[relative] = before; files[relative] = hashes['ciphertextHash']
        _integrity(database)
        expires = (datetime.now(timezone.utc)+timedelta(days=expires_days)).isoformat()
        lineage = _lineage(database)
        lineage_db=apsw.Connection(str(database), flags=apsw.SQLITE_OPEN_READONLY)
        try:corpora=_records(lineage_db,'foundation_corpus')
        finally:lineage_db.close()
        for row in corpora:
            if row.get('storagePath') and row.get('state') not in ('deleted','archived','revoked','quarantined'):
                path = _path(store.root, row['storagePath']+'/manifest.json')
                require(path.is_file() and file_hash(path)==row.get('manifestHash'), 'BACKUP_HASH_MISMATCH')
                from LLM.foundation.corpus import iter_members
                for member in iter_members(store,row):
                    for link in member.get('lineage', []):
                        for field, target in [('sourceId','sourceIds'),('documentId','documentIds'),('versionId','versionIds')]:
                            if link.get(field):lineage[target]=sorted(set(lineage[target])|{link[field]})
        manifest = {'id': identity, 'schemaVersion': 3, 'createdAt': now(), 'files': files, 'completed': True,
            'includedFoundationDirectories': list(FOUNDATION_DIRS),
            'excluded': ['foundation/inputs', 'foundation/staging', 'foundation/job-results', 'staging', 'sandbox-rpc', 'RAG raw sources'],
            'restoreRequiresMaintenanceReview': True, 'encrypted': True, 'algorithm':'AES-256-GCM-chunk-v1',
            'keyId': key_id, 'expiresAt': expires}
        # Plain hashes and original source membership are authenticated and encrypted too.
        sealed = encryption.key_directory(store)/('manifest-'+identity+'.json')
        atomic_json(sealed, {'files':plaintext_files,'lineage':lineage,'backupId':identity,'expiresAt':expires})
        encryption.private(sealed, create=True)
        try:
            encrypted = encryption.encrypt_file(sealed, directory/'sealed-manifest', key, [identity,'sealed-manifest'])
            manifest['sealedManifestHash'] = encrypted['ciphertextHash']
        finally: sealed.unlink(missing_ok=True)
        with store.tx():
            _assert_gate(store, identity, errors)
            atomic_json(directory/'backup-manifest.json', manifest)
            result = store.put('backup', {'storagePath': directory.relative_to(store.root).as_posix(), 'manifest': manifest,
                     'state':'ready','keyId':key_id,'ownedKey':own_key,'expiresAt':expires,'lineage':lineage}, identity)
        success = True
      return result
    finally:
        scratch.unlink(missing_ok=True)
        for suffix in ('-journal','-wal','-shm'):Path(str(scratch)+suffix).unlink(missing_ok=True)
        if not success:
            if directory.exists():shutil.rmtree(directory)
            if own_key:encryption.destroy_key(store,key_id)


def restore(directory, destination, *, key_store=None, allow_legacy=False):
    root, dest = _root(directory), _root(destination)
    require(not dest.exists() or (dest.is_dir() and not any(dest.iterdir())), 'RESTORE_DESTINATION_NOT_EMPTY')
    require(not dest.is_relative_to(root) and not root.is_relative_to(dest), 'BACKUP_PATH_UNSAFE')
    manifest_path = _path(root, 'backup-manifest.json')
    require(manifest_path.is_file() and manifest_path.stat().st_size <= 64*1024**2, 'BACKUP_MANIFEST_INVALID')
    manifest = json.loads(manifest_path.read_text('utf-8'))
    require(manifest.get('completed') is True and manifest.get('schemaVersion') in (1, 2, 3), 'BACKUP_INCOMPLETE')
    if manifest.get('schemaVersion') == 3:
        return _restore_encrypted(root, dest, manifest, key_store=key_store)
    require(allow_legacy is True, 'LEGACY_UNENCRYPTED_BACKUP_REVIEW_REQUIRED', status=409)
    authority_owned=key_store is None
    if key_store is None:
        owner=root.parent.parent
        require((owner/'metadata'/'llm.db').is_file(),'BACKUP_AUTHORITY_REQUIRED',status=409)
        key_store=Store(owner)
    try:
        authority=key_store.get('backup',manifest['id'])
        require(authority.get('state','ready')=='ready' and not key_store.get('backup_revocation',manifest['id'],{}),'BACKUP_REVOKED',status=403)
        require(authority.get('manifest')==manifest,'BACKUP_MANIFEST_CHANGED',status=409)
        if authority.get('expiresAt'):
            require(datetime.fromisoformat(authority['expiresAt'].replace('Z','+00:00'))>datetime.now(timezone.utc),'BACKUP_EXPIRED',status=409)
    finally:
        if authority_owned:key_store.close()
    files = manifest.get('files')
    require(isinstance(files, dict) and 'llm.db' in files, 'BACKUP_MANIFEST_INVALID')
    for name, checksum in files.items():
        require(_allowed(name) and isinstance(checksum, str) and re.fullmatch('[0-9a-f]{64}', checksum), 'BACKUP_MANIFEST_INVALID')
        source = _path(root, name)
        require(source.is_file() and file_hash(source) == checksum, 'BACKUP_HASH_MISMATCH')
    _integrity(root/'llm.db')
    _foundation_integrity(root, root/'llm.db')
    dest.mkdir(parents=True, exist_ok=True)
    for name, checksum in files.items():
        target = _path(dest, 'metadata/llm.db' if name == 'llm.db' else name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_path(root, name), target)
        require(file_hash(target) == checksum, 'BACKUP_HASH_MISMATCH')
    restored = Store(dest)
    try:
        with restored.tx():
            restored.put('setting', {'enabled': True, 'backupOwner': None, 'backupInProgress': False,
                'restoredFromBackupId': manifest['id'], 'restoreReviewRequired': True,
                'reason': 'BACKUP_RESTORE_REVIEW_REQUIRED'}, 'deploymentMaintenance')
            restored.db.execute("DELETE FROM records WHERE kind='setting' AND id='foregroundLease'")
            restored.db.execute("UPDATE jobs SET state='completed',lease=0,updated=? WHERE id=? AND type='backup_maintenance'", (now(), manifest['id']))
            restored.event('backup.restored', manifest['id'], {'maintenanceReviewRequired': True})
    finally:
        restored.close()
    _integrity(dest/'metadata'/'llm.db')
    _foundation_integrity(dest, dest/'metadata'/'llm.db')
    return {'destination': str(dest), 'backupId': manifest['id'], 'deploymentMaintenance': True, 'reviewRequired': True}


def _restore_encrypted(root, dest, manifest, *, key_store=None):
    owned_store = key_store is None
    if key_store is None:
        owner = root.parent.parent
        require((owner/'metadata'/'llm.db').is_file(), 'BACKUP_AUTHORITY_REQUIRED', status=409)
        key_store = Store(owner)
    try:
        record = key_store.get('backup', manifest['id'])
        require(record.get('state') == 'ready' and not key_store.get('backup_revocation', manifest['id'], {}), 'BACKUP_REVOKED', status=403)
        require(record.get('manifest') == manifest, 'BACKUP_MANIFEST_CHANGED', status=409)
        require(datetime.fromisoformat(manifest['expiresAt'].replace('Z','+00:00')) > datetime.now(timezone.utc), 'BACKUP_EXPIRED', status=409)
        files=manifest.get('files');require(isinstance(files,dict) and 'llm.db' in files,'BACKUP_MANIFEST_INVALID')
        for name, checksum in files.items():
            require(_allowed(name) and re.fullmatch('[0-9a-f]{64}',checksum or ''),'BACKUP_MANIFEST_INVALID')
            source=_path(root,name);require(source.is_file() and file_hash(source)==checksum,'BACKUP_HASH_MISMATCH')
        sealed=_path(root,'sealed-manifest')
        require(sealed.is_file() and file_hash(sealed)==manifest.get('sealedManifestHash'),'BACKUP_HASH_MISMATCH')
        key=encryption.get_key(key_store,manifest['keyId'])
        scratch=encryption.key_directory(key_store)/('restore-manifest-'+uid()+'.json')
        try:
            encryption.decrypt_file(sealed,scratch,key,[manifest['id'],'sealed-manifest']);encryption.private(scratch,create=True)
            private_manifest=json.loads(scratch.read_text('utf-8'))
        finally:scratch.unlink(missing_ok=True)
        require(private_manifest.get('backupId')==manifest['id'] and set(private_manifest.get('files',{}))==set(files),'BACKUP_MANIFEST_INVALID')
        require(private_manifest.get('expiresAt')==manifest['expiresAt'],'BACKUP_MANIFEST_INVALID')
        # Publish only a complete authenticated restore; partial output is cleaned.
        staging=dest.with_name(dest.name+'.restore-'+uid());staging.mkdir(mode=0o700,parents=True)
        try:
            for name in files:
                target=_path(staging,'metadata/llm.db' if name=='llm.db' else name)
                actual=encryption.decrypt_file(_path(root,name),target,key,[manifest['id'],name])
                require(actual==private_manifest['files'][name],'BACKUP_HASH_MISMATCH')
            _integrity(staging/'metadata'/'llm.db');_foundation_integrity(staging,staging/'metadata'/'llm.db')
            require(key_store.get('backup',manifest['id']).get('state')=='ready' and not key_store.get('backup_revocation',manifest['id'],{}),'BACKUP_REVOKED',status=403)
            restored=Store(staging)
            try:
                restored.put('setting',{'enabled':True,'backupOwner':None,'backupInProgress':False,'restoredFromBackupId':manifest['id'],
                             'restoreReviewRequired':True,'reason':'BACKUP_RESTORE_REVIEW_REQUIRED'},'deploymentMaintenance')
                restored.db.execute("DELETE FROM records WHERE kind='setting' AND id='foregroundLease'")
                restored.db.execute("UPDATE jobs SET state='completed',lease=0,updated=? WHERE type='backup_maintenance'",(now(),))
                restored.event('backup.restored',manifest['id'],{'maintenanceReviewRequired':True,'encrypted':True})
            finally:restored.close()
            if dest.exists():dest.rmdir()
            os.replace(staging,dest)
        finally:
            if staging.exists():shutil.rmtree(staging)
        return {'destination':str(dest),'backupId':manifest['id'],'deploymentMaintenance':True,'reviewRequired':True,'encrypted':True}
    finally:
        if owned_store:key_store.close()
