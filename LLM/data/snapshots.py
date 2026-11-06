from __future__ import annotations
import json
import shutil
from pathlib import Path
from LLM.core import require, digest, file_hash, atomic_json, uid, now

SPLITS=("train","validation","test")

class Snapshots:
    def __init__(self, store):self.store=store

    def ingest(self, directory, manifest_hash, family_id, policy_epoch, policy_validator):
        root=Path(directory).resolve()
        require(root.is_dir() and not any(p.is_symlink() for p in root.rglob("*")),"SNAPSHOT_PATH_INVALID")
        require(file_hash(root/"manifest.json")==manifest_hash,"SNAPSHOT_HASH_MISMATCH")
        manifest=json.loads((root/"manifest.json").read_text("utf-8"))
        require(manifest.get("schemaVersion")==1 and manifest.get("format")=="instruction_output","SNAPSHOT_SCHEMA_UNSUPPORTED")
        require(callable(policy_validator),"POLICY_VALIDATOR_REQUIRED",status=409)
        policy_validator(manifest)
        examples=manifest.get("examples",[])
        require(len(examples)==manifest.get("exampleCount") and len({e['id'] for e in examples})==len(examples),"SNAPSHOT_SAMPLE_COUNT_MISMATCH")
        records=[]; group_splits={}
        for split in SPLITS:
            name=split+".jsonl"
            require(file_hash(root/name)==manifest["files"].get(name),"SNAPSHOT_HASH_MISMATCH")
            raw=(root/name).read_text("utf-8").splitlines()
            rows=[json.loads(line) for line in raw if line.strip()]
            metadata=[e for e in examples if e.get("split")==split]
            require(len(rows)==len(metadata)==manifest["splitCounts"].get(split),"SNAPSHOT_SAMPLE_COUNT_MISMATCH")
            require(bool(rows),"INSUFFICIENT_SPLIT_GROUPS")
            for row, meta in zip(rows,metadata):
                require(isinstance(row.get("instruction"),str) and row["instruction"].strip() and isinstance(row.get("output"),str) and row["output"].strip(),"INVALID_TRAINING_EXAMPLE")
                require(digest({"input":row["instruction"],"target":row["output"]})==meta.get("hash"),"SNAPSHOT_ROW_HASH_MISMATCH")
                require(meta.get("permissionStatus")=="allowed" and bool(meta.get("lineage")),"POLICY_NOT_APPROVED",status=403)
                from RAG.Post.service import SECRET_PATTERNS
                require(not any(p.search(row['instruction']+'\n'+row['output']) for p in SECRET_PATTERNS),"SECRET_DETECTED",status=403)
                group=meta.get("splitGroup")
                require(isinstance(group,str) and group,"INVALID_SPLIT_GROUP")
                require(group not in group_splits or group_splits[group]==split,"SPLIT_LEAKAGE",status=409)
                group_splits[group]=split
                records.append({**row,"exampleId":meta["id"],"hash":meta["hash"],"group":group,"split":split,"lineage":meta["lineage"]})
        identity=manifest["id"]
        old=self.store.get("snapshot",identity,{})
        if old:
            require(old["manifestHash"]==manifest_hash and old["familyId"]==family_id,"SNAPSHOT_IMMUTABLE",status=409)
            require(old["state"]=="verified","POLICY_REVOKED",status=403)
            return old
        destination=self.store.root/"snapshots"/digest([identity,manifest_hash])
        staging=self.store.root/"staging"/("snapshot-"+uid())
        staging.mkdir()
        for name in ["manifest.json",*[s+".jsonl" for s in SPLITS]]:shutil.copyfile(root/name,staging/name)
        require(file_hash(staging/"manifest.json")==manifest_hash,"SNAPSHOT_HASH_MISMATCH")
        for name,h in manifest["files"].items():
            require(name in [s+".jsonl" for s in SPLITS] and file_hash(staging/name)==h,"SNAPSHOT_HASH_MISMATCH")
        atomic_json(staging/"examples.json",records)
        examples_hash=file_hash(staging/'examples.json')
        with self.store.tx():
            for group,split in group_splits.items():
                previous=self.store.rows("SELECT split FROM splits WHERE family=? AND group_id=?",(family_id,group))
                require(not previous or previous[0]["split"]==split,"SPLIT_LEAKAGE",status=409,group=group)
                self.store.db.execute("INSERT OR IGNORE INTO splits VALUES(?,?,?)",(family_id,group,split))
            if not destination.exists():staging.replace(destination)
            value=self.store.put("snapshot",{"familyId":family_id,"manifestHash":manifest_hash,"state":"verified",
                "exampleCount":len(records),"splitCounts":manifest["splitCounts"],"policyEpoch":policy_epoch,
                "storagePath":destination.relative_to(self.store.root).as_posix(),"examplesHash":examples_hash,"verifiedAt":now(),"verifierVersion":1},identity)
        return value

    def examples(self, snapshot_id, split=None):
        item=self.store.get("snapshot",snapshot_id)
        require(item["state"]=="verified","POLICY_REVOKED",status=409)
        root=self.store.root/item["storagePath"]
        require(file_hash(root/"manifest.json")==item["manifestHash"],"SNAPSHOT_HASH_MISMATCH")
        manifest=json.loads((root/"manifest.json").read_text('utf-8'))
        for name,h in manifest['files'].items():require(file_hash(root/name)==h,"SNAPSHOT_HASH_MISMATCH")
        require(file_hash(root/'examples.json')==item['examplesHash'],'SNAPSHOT_ROW_HASH_MISMATCH')
        records=json.loads((root/"examples.json").read_text("utf-8"))
        expected=[e for e in manifest['examples']]
        require({r['exampleId']:r['hash'] for r in records}=={e['id']:e['hash'] for e in expected},"SNAPSHOT_ROW_HASH_MISMATCH")
        for row in records:require(digest({'input':row['instruction'],'target':row['output']})==row['hash'],"SNAPSHOT_ROW_HASH_MISMATCH")
        metadata=item.get('exampleMetadata',{})
        return [{**r,**metadata.get(r['exampleId'],{}),'snapshotId':snapshot_id} for r in records if split is None or r["split"]==split]

    def new_examples(self, branch_id, snapshot_id, objective=None):
        consumed={r['hash'] for r in self.store.rows("SELECT hash FROM consumed WHERE branch=?",(branch_id,))}
        return [r for r in self.examples(snapshot_id,"train") if r['hash'] not in consumed and r.get('branchId') in (None,branch_id) and (objective is None or r.get('objective','sft')==objective)]

    def consume(self, branch_id, records, run_id):
        with self.store.tx():
            for r in records:self.store.db.execute("INSERT OR IGNORE INTO consumed VALUES(?,?,?,?,?)",(branch_id,r['hash'],r['exampleId'],run_id,now()))

    def batch(self,branch_id,snapshot_id,tokenizer,objective='sft'):
        branch=self.store.get('branch',branch_id)
        all_current=[r for r in self.examples(snapshot_id,'train') if r.get('branchId') in (None,branch_id) and r.get('objective','sft')==objective]
        consumed={r['hash'] for r in self.store.rows('SELECT hash FROM consumed WHERE branch=?',(branch_id,))}
        fresh=[r for r in all_current if r['hash'] not in consumed]
        if not fresh:return all_current,{'bootstrapReplay':False,'newTargetTokenRatio':0,'explicitRetraining':True}
        pool={}
        for s in self.store.list('snapshot'):
            if s['familyId']!=branch['familyId'] or s['state']!='verified':continue
            for r in self.examples(s['id'],'train'):
                if r['hash'] in consumed and r.get('branchId') in (None,branch_id) and r.get('objective','sft')==objective:pool[r['hash']]=r
        tokens=lambda r:len(tokenizer.encode(r['output'],add_special_tokens=False))
        target=sum(tokens(r) for r in fresh)*.3/.7;replay=[];count=0
        for key in sorted(pool):
            if count>=target:break
            replay.append(pool[key]);count+=tokens(pool[key])
        new_count=sum(tokens(r) for r in fresh)
        return fresh+replay,{'bootstrapReplay':not consumed,'newTargetTokenRatio':new_count/(new_count+count) if new_count+count else 0,
            'newExamples':len(fresh),'replayExamples':len(replay),'replayTargetTokens':count}
