"""Conservative file retention; provenance records survive file archival."""
import shutil
from datetime import datetime,timezone,timedelta
from pathlib import Path
from LLM.core import require,inside,now


def collect(store,apply=False):
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
