"""Hash-bound snapshot row offsets; targets are read one bounded line at a time."""
import base64
import hashlib
import hmac
import json
from pathlib import Path
import secrets

import apsw

from LLM.core import atomic_json,canonical,digest,file_hash,require,uid

MAX_LINE=2*1024**2
_VERIFIED={}


def checked_hash(path,expected):
    path=Path(path);require(path.is_file() and not path.is_symlink(),'SNAPSHOT_PATH_INVALID')
    stat=path.stat();signature=(str(path.resolve()),stat.st_size,stat.st_mtime_ns,expected)
    if signature not in _VERIFIED:
        require(file_hash(path)==expected,'SNAPSHOT_HASH_MISMATCH')
        if len(_VERIFIED)>=128:_VERIFIED.pop(next(iter(_VERIFIED)))
        _VERIFIED[signature]=True


def index_snapshot(bridge,snapshot,project_id):
    from RAG.Post.service import _hash,_manifest_examples
    from LLM.foundation.storage_reservations import begin,commit,abort
    store=bridge.c.store;require(store is not None,'FOUNDATION_STORE_REQUIRED')
    project=store.get('foundation_project',project_id)
    root=(bridge.store.root/snapshot['storagePath']).resolve()
    require(root.is_relative_to((bridge.store.root/'exports').resolve()),'SNAPSHOT_PATH_INVALID')
    checked_hash(root/'manifest.json',snapshot['manifestHash'])
    identity=digest([project_id,snapshot['manifestHash']]);old=store.get('foundation_sft_index',identity,{})
    if old and old.get('state')=='ready':
        require(old.get('projectId')==project_id and old.get('snapshotHash')==snapshot['manifestHash'],'SNAPSHOT_INDEX_OWNER_MISMATCH')
        for name,checksum in old['sourceFiles'].items():checked_hash(root/name,checksum)
        target=store.root/old['indexPath'];checked_hash(target,old['indexHash'])
        return root,target,old
    require(not old or old.get('state')=='deleted','SNAPSHOT_INDEX_RECONCILIATION_REQUIRED',status=409)
    manifest=json.loads((root/'manifest.json').read_text('utf-8'))
    require(manifest.get('schemaVersion') in (1,2) and manifest.get('format')=='instruction_output','SNAPSHOT_SCHEMA_UNSUPPORTED')
    destination=store.root/'foundation'/'snapshot-index'/identity
    require(not destination.is_symlink() and (not destination.exists() or not any(destination.iterdir())),'SNAPSHOT_INDEX_RECONCILIATION_REQUIRED',status=409)
    destination.mkdir(parents=True,exist_ok=True)
    store.put('foundation_sft_index',{'projectId':project_id,'snapshotId':snapshot['id'],'snapshotHash':snapshot['manifestHash'],
        'sourceFiles':manifest['files'],'state':'building','storagePath':destination.relative_to(store.root).as_posix()},identity)
    temporary=destination/('index-'+uid()+'.sqlite');db=None;count=0;work=None
    def write(estimate,action):
        nonlocal work
        work=begin(store,project,estimate,purpose='sft-snapshot-index')
        before=temporary.stat().st_size if temporary.exists() else 0
        if db is not None:
            list(db.execute('PRAGMA max_page_count='+str(max(16,(before+estimate)//4096))))
        action()
        actual=max(0,temporary.stat().st_size-before)
        with store.tx():commit(store,work,actual)
        work=None
    def batch(sql,items,amount):
        if not items:return
        def apply():
            db.execute('BEGIN')
            try:
                list(db.executemany(sql,items));db.execute('COMMIT')
            except Exception:
                db.execute('ROLLBACK');raise
        write(amount*4+65536,apply)
    try:
        db=apsw.Connection(str(temporary))
        write(65536,lambda:list(db.execute('PRAGMA journal_mode=OFF;CREATE TABLE positions(ordinal INTEGER PRIMARY KEY,split TEXT,file TEXT,offset INTEGER,length INTEGER,metadata TEXT);CREATE TABLE expected(ordinal INTEGER PRIMARY KEY,split TEXT,metadata TEXT);CREATE INDEX expected_split ON expected(split,ordinal);')))
        pending=[];pending_bytes=0
        for number,meta in enumerate(_manifest_examples(root,manifest),1):
            encoded=canonical(meta);amount=len(encoded.encode());require(amount<=MAX_LINE,'SNAPSHOT_ROW_TOO_LARGE',status=413)
            pending.append((number,meta['split'],encoded));pending_bytes+=amount
            if len(pending)>=32 or pending_bytes>=1024**2:
                batch('INSERT INTO expected VALUES(?,?,?)',pending,pending_bytes);pending=[];pending_bytes=0
        batch('INSERT INTO expected VALUES(?,?,?)',pending,pending_bytes)
        for split in ('train','validation','test'):
            expected=iter(json.loads(item[0]) for item in db.execute('SELECT metadata FROM expected WHERE split=? ORDER BY ordinal',(split,)));actual=0
            pending=[];pending_bytes=0
            for name in manifest.get('splitShards',{}).get(split,[split+'.jsonl']):
                require(Path(name).name==name and name in manifest['files'],'SNAPSHOT_PATH_INVALID')
                checked_hash(root/name,manifest['files'][name])
                with (root/name).open('rb') as handle:
                    while True:
                        offset=handle.tell();line=handle.readline(MAX_LINE+1)
                        if not line:break
                        require(len(line)<=MAX_LINE,'SNAPSHOT_ROW_TOO_LARGE',status=413)
                        if not line.strip():continue
                        meta=next(expected,None);require(meta is not None,'SNAPSHOT_SAMPLE_COUNT_MISMATCH')
                        row=json.loads(line);example=bridge.store.get('training_example',meta['id'])
                        require(_hash({'input':row['instruction'],'target':row['output']})==meta['hash']==example['contentHash'],'SNAPSHOT_ROW_HASH_MISMATCH')
                        encoded=canonical(meta);count+=1;actual+=1
                        pending.append((count,split,name,offset,len(line),encoded));pending_bytes+=len(encoded.encode())
                        if len(pending)>=32 or pending_bytes>=1024**2:
                            batch('INSERT INTO positions VALUES(?,?,?,?,?,?)',pending,pending_bytes);pending=[];pending_bytes=0
            batch('INSERT INTO positions VALUES(?,?,?,?,?,?)',pending,pending_bytes)
            require(next(expected,None) is None and actual==manifest['splitCounts'][split],'SNAPSHOT_SAMPLE_COUNT_MISMATCH')
        require(count==manifest['exampleCount'],'SNAPSHOT_SAMPLE_COUNT_MISMATCH')
        # Freed pages stay inside this charged file; VACUUM would create an
        # unreserved second copy proportional to the entire metadata catalog.
        write(65536,lambda:list(db.execute('DROP TABLE expected;')))
        db.close();db=None;target=destination/'index.sqlite';temporary.replace(target)
        record=store.put('foundation_sft_index',{'projectId':project_id,'snapshotId':snapshot['id'],'snapshotHash':snapshot['manifestHash'],'state':'ready',
            'sourceFiles':manifest['files'],'rowCount':count,'indexHash':file_hash(target),
            'indexPath':target.relative_to(store.root).as_posix(),'storagePath':destination.relative_to(store.root).as_posix()},identity)
        return root,target,record
    except Exception:
        if db is not None:db.close();db=None
        temporary.unlink(missing_ok=True)
        if work is not None:abort(store,work,files_removed=not temporary.exists())
        if not any(destination.iterdir()):
            destination.rmdir();store.put('foundation_sft_index',{'state':'deleted'},identity)
        raise
    finally:
        if db is not None:db.close()
        temporary.unlink(missing_ok=True)


def cursor_key(store):
    with store.tx():
        record=store.get('setting','foundationSftCursorKey',{})
        if not record:record=store.put('setting',{'secret':secrets.token_hex(32)},'foundationSftCursorKey')
        return bytes.fromhex(record['secret'])


def encode_cursor(store,binding,ordinal):
    raw=canonical({**binding,'ordinal':ordinal}).encode();signature=hmac.new(cursor_key(store),raw,hashlib.sha256).digest()
    return base64.urlsafe_b64encode(raw+signature).decode().rstrip('=')


def decode_cursor(store,value,binding):
    if not value:return 0
    try:
        packed=base64.urlsafe_b64decode(value+'='*(-len(value)%4));raw,signature=packed[:-32],packed[-32:]
        require(hmac.compare_digest(hmac.new(cursor_key(store),raw,hashlib.sha256).digest(),signature),'INVALID_SFT_CURSOR')
        body=json.loads(raw);ordinal=body.pop('ordinal')
        require(body==binding and type(ordinal) is int and ordinal>=0,'STALE_SFT_CURSOR',status=409)
        return ordinal
    except (ValueError,TypeError,KeyError):require(False,'INVALID_SFT_CURSOR')
