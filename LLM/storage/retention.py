"""Conservative file retention; provenance records survive file archival."""
import shutil
import json
import threading
import time
from datetime import datetime,timezone,timedelta
from pathlib import Path
from LLM.core import require,inside,now,uid,digest,file_hash


def collect(store,apply=False):
    if not apply:
        result=_collect(store,False);result['longLived']=collect_long_lived(store,apply=False);return result
    identity=uid()
    with store.tx():
        require(not store.get('setting','deploymentMaintenance',{}).get('enabled'),'DEPLOYMENT_MAINTENANCE',status=409)
        require(not store.get('setting','storageGcInProgress',{}),'STORAGE_GC_BUSY',status=409)
        store.put('setting',{'enabled':True,'owner':identity,'startedAt':now()},'storageGcInProgress')
    try:
        result=_collect(store,True);result['longLived']=collect_long_lived(store,apply=True);return result
    finally:
        with store.tx():
            if store.get('setting','storageGcInProgress',{}).get('owner')==identity:
                store.db.execute("DELETE FROM records WHERE kind='setting' AND id='storageGcInProgress'")


def _collect(store,apply=False):
    with store.lock:
        if store.rows("SELECT id FROM jobs WHERE state IN ('queued','running','paused','pause_requested','cancelling') LIMIT 1"):
            return {'state':'held','reasonCodes':['JOBS_PIN_STORAGE'],'items':[],'applied':False}
        clock=datetime.now(timezone.utc);cutoff=clock-timedelta(days=90);failed_cutoff=clock-timedelta(days=7)
        def recent(record,limit=cutoff):
            return datetime.fromisoformat(record['createdAt'].replace('Z','+00:00'))>=limit
        artifacts=store.list('artifact');by_id={a['id']:a for a in artifacts}
        pinned={f['anchorArtifactId'] for f in store.list('family')}
        for branch in store.list('branch'):
            pinned.update(branch.get(k) for k in ('currentArtifactId','currentAdapterId','candidateArtifactId','candidateAdapterId') if branch.get(k))
            history=sorted([a for a in artifacts if a.get('branchId')==branch['id'] and a['kind'] in ('specialist_checkpoint','distilled_checkpoint')],key=lambda a:a['createdAt'],reverse=True)[:2]
            for artifact in history:
                pinned.add(artifact['id'])
                if artifact.get('adapterArtifactId'):pinned.add(artifact['adapterArtifactId'])
        releases=store.list('release')
        retained=[r for r in releases if r['state'] in ('active','ready')]
        retained+=sorted([r for r in releases if r['state']=='retired'],key=lambda r:r['updatedAt'],reverse=True)[:2]
        pinned.update(r['artifactId'] for r in retained)
        for evaluation in store.list('evaluation'):
            if recent(evaluation):pinned.update(evaluation.get('results',{}))
        items=[]
        for artifact in artifacts:
            if artifact['id'] in pinned or recent(artifact) or artifact['state']=='archived':continue
            path=inside(store.root,artifact['storagePath'])
            require(path.parent==store.root/'artifacts' and path.name==artifact['id'],'GC_SCOPE_DENIED')
            if path.exists():items.append({'kind':'artifact','id':artifact['id'],'path':path.relative_to(store.root).as_posix()})
        runs={r['id']:r for r in store.list('training_run')}
        for checkpoint in store.list('checkpoint'):
            run=runs.get(checkpoint['runId'],{})
            if run.get('outputArtifactId') in pinned:continue
            if recent(checkpoint,failed_cutoff if run.get('state') in ('failed','cancelled') else cutoff):continue
            path=inside(store.root,checkpoint['storagePath'])
            require(path.parent==store.root/'checkpoints' and path.name==checkpoint['id'],'GC_SCOPE_DENIED')
            if path.exists():items.append({'kind':'checkpoint','id':checkpoint['id'],'path':path.relative_to(store.root).as_posix()})
        for path in (store.root/'staging').iterdir():
            require(not path.is_symlink() and path.resolve().parent==store.root/'staging','GC_SCOPE_DENIED')
            if datetime.fromtimestamp(path.stat().st_mtime,timezone.utc)<failed_cutoff:
                items.append({'kind':'staging','id':path.name,'path':path.relative_to(store.root).as_posix()})
        if apply:
            for item in items:
                path=inside(store.root,item['path'])
                if item['kind']=='artifact':store.put('artifact',{'state':'archived','archivedAt':now()},item['id'])
                elif item['kind']=='checkpoint':store.put('checkpoint',{'state':'archived','archivedAt':now()},item['id'])
                if path.is_dir():shutil.rmtree(path)
                else:path.unlink(missing_ok=True)
        result={'state':'completed','items':items,'applied':apply,'observedAt':now(),'provenanceRecordsPreserved':True}
        if apply:store.put('maintenance',result,'last-gc')
        return result


def collect_long_lived(store,apply=False,*,cache_days=30,log_days=30,event_days=365):
    if not apply:return _collect_long_lived(store,False,cache_days=cache_days,log_days=log_days,event_days=event_days)
    if store.rows("SELECT id FROM jobs WHERE state IN ('running','pause_requested','cancelling') LIMIT 1"):
        return {'state':'held','reasonCodes':['GC_ACTIVE_JOBS'],'items':[],'applied':False}
    if store.get('setting','foregroundLease',{}).get('expires',0)>time.time():
        return {'state':'held','reasonCodes':['GC_FOREGROUND_ACTIVE'],'items':[],'applied':False}
    owner='gc-'+uid();ended=threading.Event();errors=[]
    with store.tx():
        require(not store.get('setting','deploymentMaintenance',{}).get('enabled'),'DEPLOYMENT_MAINTENANCE',status=409)
        require(store.get('setting','foregroundLease',{}).get('expires',0)<=time.time(),'GC_FOREGROUND_ACTIVE',status=409)
        require(not store.rows("SELECT id FROM jobs WHERE state IN ('running','pause_requested','cancelling') LIMIT 1"),'GC_ACTIVE_JOBS',status=409)
        store.put('setting',{'owner':owner,'expires':time.time()+90,'purpose':'garbage_collection'},'foregroundLease')
    def refresh():
        while not ended.wait(15):
            try:
                with store.tx():
                    require(store.get('setting','foregroundLease',{}).get('owner')==owner,'GC_LEASE_LOST')
                    store.put('setting',{'expires':time.time()+90},'foregroundLease')
            except Exception as exc:errors.append(exc);return
    thread=threading.Thread(target=refresh,daemon=True);thread.start()
    try:
        result=_collect_long_lived(store,True,cache_days=cache_days,log_days=log_days,event_days=event_days)
        require(not errors and store.get('setting','foregroundLease',{}).get('owner')==owner,'GC_LEASE_LOST')
        return result
    finally:
        ended.set();thread.join(timeout=1)
        with store.tx():
            if store.get('setting','foregroundLease',{}).get('owner')==owner:store.db.execute("DELETE FROM records WHERE kind='setting' AND id='foregroundLease'")


def _collect_long_lived(store,apply=False,*,cache_days=30,log_days=30,event_days=365):
    """Reclaim reproducible token caches and terminal-worker logs, never lineage.

    Active and paused run inputs remain pinned. Unknown files are left untouched.
    Events keep their monotone sequence; clients before the watermark must resync.
    """
    require(all(isinstance(value,int) and not isinstance(value,bool) and 1<=value<=3650 for value in (cache_days,log_days,event_days)),
            'RETENTION_POLICY_INVALID')
    if apply:require(not store.get('setting','deploymentMaintenance',{}).get('enabled'),'DEPLOYMENT_MAINTENANCE',status=409)
    clock=datetime.now(timezone.utc);cutoff=clock-timedelta(days=cache_days);log_cutoff=clock-timedelta(days=log_days)
    pending=[store.job(row['id']) for row in store.rows("SELECT id FROM jobs WHERE state IN ('queued','running','paused','pause_requested','cancelling')")]
    pinned_corpora=set();pinned_tokens=set()
    for project in store.list('foundation_project'):
        if project.get('state')!='revoked':
            pinned_corpora.update(project.get(field) for field in ('activeCorpusId','activeSftCorpusId') if project.get(field))
            if project.get('tokenizerId'):pinned_tokens.add(project['tokenizerId'])
    for job in pending:
        body=job.get('payload') or {}
        if body.get('corpusId'):pinned_corpora.add(body['corpusId'])
        if body.get('tokenizerId'):pinned_tokens.add(body['tokenizerId'])
    items=[];held=[]
    active=any(job['state'] in ('running','pause_requested','cancelling') for job in pending)
    for record in store.list('foundation_token_shard'):
        if record.get('state') in ('evicted','deleted'):continue
        project=store.get('foundation_project',record.get('projectId',''),{})
        # Frozen tokenizer reuse does not pin every historical corpus cache.
        # Active/paused corpus inputs are sufficient; old caches are reproducible.
        if record.get('corpusId') in pinned_corpora:continue
        if active:continue
        stamp=datetime.fromisoformat(record.get('updatedAt',record['createdAt']).replace('Z','+00:00'))
        if stamp>=cutoff and project.get('state')!='revoked':continue
        relative='foundation/token-shards/'+record['id'];path=inside(store.root,relative)
        require(record.get('storagePath')==relative and not path.is_symlink(),'GC_SCOPE_DENIED')
        if not path.exists():continue
        manifest_path=path/'manifest.json'
        require(manifest_path.is_file() and file_hash(manifest_path)==record.get('manifestHash'),'TOKEN_SHARD_HASH_MISMATCH')
        manifest=json.loads(manifest_path.read_text('utf-8'));owned=set(manifest.get('files',{}))|{'manifest.json'}
        actual=set()
        for item in path.rglob('*'):
            require(not item.is_symlink(),'GC_SCOPE_DENIED')
            if item.is_file():actual.add(item.relative_to(path).as_posix())
        if actual!=owned:held.append({'kind':'token_cache','id':record['id'],'reasonCode':'GC_UNOWNED_FILES'});continue
        for name,checksum in manifest['files'].items():require(file_hash(inside(path,name))==checksum,'TOKEN_SHARD_HASH_MISMATCH')
        items.append({'kind':'token_cache','id':record['id'],'path':relative,'manifestHash':record['manifestHash']})
    jobs={row['id']:store.job(row['id']) for row in store.rows("SELECT id FROM jobs WHERE state IN ('completed','cancelled','failed')")}
    base=store.root/'foundation'/'job-results'
    if base.exists():
        require(not base.is_symlink(),'GC_SCOPE_DENIED')
        for path in base.iterdir():
            require(not path.is_symlink(),'GC_SCOPE_DENIED')
            job=jobs.get(path.name)
            if not job or not path.is_dir() or datetime.fromisoformat(job['updatedAt'].replace('Z','+00:00'))>=log_cutoff:continue
            files=list(path.iterdir())
            if any(item.is_symlink() or not item.is_file() or item.name not in ('worker.log','result.json','format-stage-0.log','format-stage-1.log') for item in files):
                held.append({'kind':'worker_log','id':path.name,'reasonCode':'GC_UNOWNED_FILES'});continue
            items.append({'kind':'worker_log','id':path.name,'path':path.relative_to(store.root).as_posix(),
                          'files':{item.name:file_hash(item) for item in files}})
    latest=store.rows('SELECT COALESCE(MAX(sequence),0) AS value FROM events')[0]['value'];keep_before=max(0,latest-10000)
    event_cutoff=(clock-timedelta(days=event_days)).isoformat().replace('+00:00','Z')
    progress_cutoff=(clock-timedelta(days=log_days)).isoformat().replace('+00:00','Z')
    event_rows=store.rows("SELECT sequence,type,entity,data,at FROM events WHERE sequence<? AND (at<? OR (at<? AND type LIKE '%.progress')) ORDER BY sequence LIMIT 5000",
                          (keep_before,event_cutoff,progress_cutoff))
    if apply:
        for item in items:
            require(not store.get('setting','deploymentMaintenance',{}).get('enabled'),'DEPLOYMENT_MAINTENANCE',status=409)
            require(not store.rows("SELECT id FROM jobs WHERE state IN ('running','pause_requested','cancelling') LIMIT 1"),'GC_ACTIVE_JOBS',status=409)
            path=inside(store.root,item['path']);require(not path.is_symlink(),'GC_SCOPE_DENIED')
            if item['kind']=='token_cache':
                record=store.get('foundation_token_shard',item['id'])
                require(record.get('manifestHash')==item['manifestHash'],'TOKEN_SHARD_HASH_MISMATCH')
                store.put('foundation_token_shard',{'state':'evicted','evictedAt':now(),'evictionReason':'retention'},item['id'])
            else:
                job=store.job(item['id']);require(job['state'] in ('completed','cancelled','failed'),'GC_ACTIVE_JOBS')
                require(all(file_hash(path/name)==checksum for name,checksum in item['files'].items()),'GC_SOURCE_CHANGED')
            require(path.resolve().is_relative_to(store.root.resolve()) and path.resolve()!=store.root.resolve(),'GC_SCOPE_DENIED')
            shutil.rmtree(path)
        if event_rows:
            with store.tx():
                store.db.executemany('DELETE FROM events WHERE sequence=?',[(row['sequence'],) for row in event_rows])
                old=store.get('setting','eventRetention',{})
                store.put('setting',{'prunedThroughSequence':max(old.get('prunedThroughSequence',0),event_rows[-1]['sequence']),
                          'deletedCount':old.get('deletedCount',0)+len(event_rows),'lastBatchHash':digest(event_rows),'observedAt':now()},'eventRetention')
        store.put('maintenance',{'state':'completed','itemCount':len(items),'eventsDeleted':len(event_rows),'observedAt':now()},'long-lived-gc')
    return {'state':'completed','applied':apply,'items':items,'held':held,'eventsEligible':len(event_rows),
            'lineageRecordsPreserved':True,'nextEventCursorAction':'resync_when_before_prunedThroughSequence'}
