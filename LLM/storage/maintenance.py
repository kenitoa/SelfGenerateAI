import json
import shutil
from pathlib import Path
import apsw
from LLM.core import require,uid,file_hash,atomic_json,now,inside

def backup(store):
    identity=uid();directory=store.root/'backups'/identity;directory.mkdir()
    database=directory/'llm.db'
    with store.lock:
        target=apsw.Connection(str(database))
        with target.backup('main',store.db,'main') as operation:
            while not operation.done:operation.step(100)
        target.close()
        # Immutable directories cannot be garbage-collected while this lock is held.
        for name in ('artifacts','snapshots','releases','checkpoints','self-code'):
            if (store.root/name).exists():shutil.copytree(store.root/name,directory/name)
    files={p.relative_to(directory).as_posix():file_hash(p) for p in directory.rglob('*') if p.is_file()}
    manifest={'id':identity,'schemaVersion':1,'createdAt':now(),'files':files,'completed':True}
    atomic_json(directory/'backup-manifest.json',manifest)
    return store.put('backup',{'storagePath':directory.relative_to(store.root).as_posix(),'manifest':manifest},identity)

def restore(directory,destination):
    root=Path(directory).resolve();dest=Path(destination).resolve()
    require(not dest.exists() or not any(dest.iterdir()),'RESTORE_DESTINATION_NOT_EMPTY')
    manifest=json.loads((root/'backup-manifest.json').read_text('utf-8'))
    require(manifest.get('completed') is True,'BACKUP_INCOMPLETE')
    for name,checksum in manifest['files'].items():require(file_hash(inside(root,name))==checksum,'BACKUP_HASH_MISMATCH')
    db=apsw.Connection(str(root/'llm.db'),flags=apsw.SQLITE_OPEN_READONLY)
    require(list(db.execute('PRAGMA integrity_check'))==[('ok',)],'BACKUP_DATABASE_CORRUPT');db.close()
    dest.mkdir(parents=True,exist_ok=True)
    for name in manifest['files']:
        target=dest/'metadata'/'llm.db' if name=='llm.db' else inside(dest,name)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(inside(root,name),target)
    return {'destination':str(dest),'backupId':manifest['id']}
