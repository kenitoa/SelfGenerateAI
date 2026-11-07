"""Expiry and source-revocation erasure for owned encrypted backup packages."""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from LLM.core import digest, file_hash, now, require
from .encryption import destroy_key

SELECTORS = frozenset(('sourceIds','documentIds','versionIds','projectIds','artifactIds'))


def selectors(value):
    require(isinstance(value,dict) and bool(value) and not set(value)-SELECTORS,'BACKUP_SELECTOR_REQUIRED')
    result={}
    for key,items in value.items():
        require(isinstance(items,list) and all(isinstance(item,str) and item for item in items),'BACKUP_SELECTOR_INVALID')
        result[key]=set(items)
    require(any(result.values()),'BACKUP_SELECTOR_REQUIRED')
    return result


def delete_backup(store, identity, reason='expired'):
    require(reason in ('expired','permission_revoked','operator_delete'),'BACKUP_DELETE_REASON_INVALID')
    record=store.get('backup',identity)
    if record.get('state')=='deleted':return record
    path=store.root/'backups'/identity
    require(record.get('storagePath')=='backups/'+identity and path.resolve().parent==(store.root/'backups').resolve()
            and all(not item.is_symlink() for item in (path,*path.parents)),'BACKUP_DELETE_PATH_UNSAFE')
    if path.exists():
        require(path.is_dir(),'BACKUP_DELETE_PATH_UNSAFE')
        manifest_path=path/'backup-manifest.json'
        require(manifest_path.is_file() and not manifest_path.is_symlink(),'BACKUP_MANIFEST_INVALID')
        manifest=json.loads(manifest_path.read_text('utf-8'))
        require(manifest==record.get('manifest') and manifest.get('encrypted') is True,'BACKUP_MANIFEST_CHANGED')
        owned=set(manifest.get('files',{}))|{'sealed-manifest','backup-manifest.json'}
        actual=set()
        for item in path.rglob('*'):
            require(not item.is_symlink(),'BACKUP_DELETE_PATH_UNSAFE')
            if item.is_file():actual.add(item.relative_to(path).as_posix())
            else:require(item.is_dir(),'BACKUP_DELETE_PATH_UNSAFE')
        require(actual==owned,'BACKUP_UNOWNED_FILES')
        for name,checksum in manifest['files'].items():
            item=path/name
            require(item.resolve().is_relative_to(path.resolve()) and file_hash(item)==checksum,'BACKUP_HASH_MISMATCH')
        require(file_hash(path/'sealed-manifest')==manifest.get('sealedManifestHash'),'BACKUP_HASH_MISMATCH')
    with store.tx():
        current=store.get('backup',identity)
        require(current.get('version')==record.get('version'),'BACKUP_VERSION_CONFLICT',status=409)
        store.put('backup_revocation',{'reason':reason,'revokedAt':now(),'lineageHash':digest(record.get('lineage',{}))},identity)
        store.put('backup',{'state':'deleting','deleteReason':reason},identity)
    # The key is erased before ciphertext. A crash can leave inaccessible ciphertext,
    # never a backup that remains authorized for restoration.
    if record.get('ownedKey'):
        destroy_key(store,record['keyId'])
    if path.exists():shutil.rmtree(path)
    result=store.put('backup',{'state':'deleted','deletedAt':now(),'keyDestroyed':bool(record.get('ownedKey'))},identity)
    store.event('backup.deleted',identity,{'reason':reason})
    return result


def revoke_backups(store, value):
    selected=selectors(value)
    affected=[record['id'] for record in store.list('backup') if record.get('state')!='deleted' and
              any(set(record.get('lineage',{}).get(key,[]))&items for key,items in selected.items())]
    with store.tx():
        for identity in affected:
            record=store.get('backup',identity)
            store.put('backup_revocation',{'reason':'permission_revoked','revokedAt':now(),'lineageHash':digest(record.get('lineage',{}))},identity)
            store.put('backup',{'state':'revoked'},identity)
    rows=[];held=[]
    for identity in affected:
        try:rows.append(delete_backup(store,identity,'permission_revoked'))
        except Exception as error:
            record=store.get('backup',identity)
            if record.get('ownedKey'):
                try:destroy_key(store,record['keyId'])
                except Exception:pass
            held.append({'backupId':identity,'reasonCode':getattr(error,'code','BACKUP_DELETION_HELD')})
    return {'state':'held' if held else 'completed','backupIds':affected,'deletedCount':len(rows),'held':held,
            'reasonCodes':sorted({row['reasonCode'] for row in held}),'mathematicalUnlearning':False}


def expire_backups(store):
    clock=datetime.now(timezone.utc);rows=[];held=[]
    for record in store.list('backup'):
        if record.get('state')=='deleted':continue
        if record.get('state') not in ('deleting','revoked') and not (record.get('expiresAt') and datetime.fromisoformat(record['expiresAt'].replace('Z','+00:00'))<=clock):continue
        reason=record.get('deleteReason','permission_revoked' if record.get('state')=='revoked' else 'expired')
        try:rows.append(delete_backup(store,record['id'],reason))
        except Exception as error:
            with store.tx():
                store.put('backup_revocation',{'reason':reason,'revokedAt':now()},record['id'])
                store.put('backup',{'state':'revoked','deleteReason':reason},record['id'])
            if record.get('ownedKey'):
                try:destroy_key(store,record['keyId'])
                except Exception:pass
            held.append({'backupId':record['id'],'reasonCode':getattr(error,'code','BACKUP_DELETION_HELD')})
    return {'state':'held' if held else 'completed','backupIds':[row['id'] for row in rows],'deletedCount':len(rows),'held':held}
