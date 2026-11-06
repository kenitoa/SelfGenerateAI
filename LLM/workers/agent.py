"""Run on a registered private worker. Transfers only scoped, hashed files."""
import argparse
import json
import time
import threading
from pathlib import Path
import httpx
from LLM.core import require,file_hash,atomic_json,canonical
from LLM.service import Control

def run(endpoint,worker_id,token,root,once=False):
    client=httpx.Client(base_url=endpoint,headers={'Authorization':'Bearer '+token},timeout=60,follow_redirects=False)
    require(endpoint.startswith('https://') or endpoint.startswith(('http://127.0.0.1:','http://localhost:')),'WORKER_TLS_REQUIRED')
    def post(path,body):
        response=client.post(path,json=body,headers={'Idempotency-Key':str(time.time_ns())});response.raise_for_status();return response.json()
    prefix='/internal/v1/workers/'+worker_id
    while True:
        assignment=post(prefix+'/claim',{})
        if not assignment.get('job'):
            if once:return
            time.sleep(5);continue
        job=assignment['job'];work=Path(root)/job['id'];c=Control(work,worker=True)
        for kind,records in [('family',assignment['families']),('branch',assignment['branches']),('combination',assignment['combinations']),('suite',assignment['suites']),('fixture',assignment.get('fixtures',[])),('verifier_profile',assignment.get('verifierProfiles',[]))]:
            for record in records:c.store.put(kind,record,record['id'])
        with c.store.tx():
            for row in assignment.get('consumed',[]):c.store.db.execute('INSERT OR REPLACE INTO consumed VALUES(?,?,?,?,?)',tuple(row[k] for k in ('branch','hash','example','run','at')))
            for row in assignment.get('splits',[]):c.store.db.execute('INSERT OR REPLACE INTO splits VALUES(?,?,?)',tuple(row[k] for k in ('family','group_id','split')))
        for kind,records in [('artifact',assignment['artifacts']),('snapshot',assignment['snapshots'])]:
            for record in records:
                dest=work/('artifacts' if kind=='artifact' else 'snapshots')/record['id'];dest.mkdir(parents=True,exist_ok=True)
                names=record['files'] if kind=='artifact' else ['manifest.json','train.jsonl','validation.jsonl','test.jsonl','examples.json']
                for name in names:
                    (dest/name).parent.mkdir(parents=True,exist_ok=True)
                    with client.stream('GET',prefix+f'/files/{job["id"]}/{kind}/{record["id"]}/{name}') as response:
                        response.raise_for_status()
                        with (dest/name).open('wb') as output:
                            for chunk in response.iter_bytes(1024**2):output.write(chunk)
                    if kind=='artifact':require(file_hash(dest/name)==record['files'][name],'ARTIFACT_HASH_MISMATCH')
                c.store.put(kind,{**record,'storagePath':dest.relative_to(work).as_posix()},record['id'])
        if c.store.rows('SELECT id FROM jobs WHERE id=?',(job['id'],)):
            with c.store.tx():c.store.db.execute("UPDATE jobs SET state='queued',payload=?,owner=NULL,lease=0,error=NULL,result=NULL WHERE id=?",(canonical(job['payload']),job['id']))
        else:
            local=c.store.enqueue(job['type'],job['payload'])
            # Preserve the control-plane run identity for checkpoints and outputs.
            with c.store.tx():c.store.db.execute('UPDATE jobs SET id=? WHERE id=?',(job['id'],local['id']))
        ended=threading.Event()
        def heartbeat():
            while not ended.wait(10):
                try:
                    run=c.store.get('training_run',job['id'],{})
                    state=post(prefix+'/heartbeat',{'jobId':job['id'],'fencingToken':job['fence'],
                        'trainingTelemetry':{k:run.get(k) for k in ('state','completedSteps','totalSteps','loss','epoch')} if run else None})
                    if state['state']=='cancelling':c.command(job['id'],'cancel')
                except Exception:
                    c.command(job['id'],'cancel');return
        thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
        try:
            c.step();result=c.store.job(job['id']);artifacts=[a for a in c.store.list('artifact') if a['id'] not in {x['id'] for x in assignment['artifacts']}]
            for a in artifacts:
                for name in a['files']:
                    with (c.registry.directory(a)/name).open('rb') as content:
                        response=client.put(prefix+f'/outputs/{job["id"]}/{a["id"]}/{name}',content=content)
                        response.raise_for_status()
            post(prefix+'/complete',{'jobId':job['id'],'fencingToken':job['fence'],'state':result['state'],'error':result['error'],
                'result':result['result'],'artifacts':artifacts,'trainingRun':c.store.get('training_run',job['id'],{}) or None})
        finally:ended.set();thread.join(timeout=2)
        if once:return

def main():
    p=argparse.ArgumentParser();p.add_argument('--endpoint',required=True);p.add_argument('--worker-id',required=True);p.add_argument('--token-file',required=True);p.add_argument('--data-dir',required=True);p.add_argument('--once',action='store_true');a=p.parse_args()
    run(a.endpoint,a.worker_id,Path(a.token_file).read_text().strip(),Path(a.data_dir).resolve(),a.once)

if __name__=='__main__':main()
