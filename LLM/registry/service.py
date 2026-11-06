from __future__ import annotations
import json
import shutil
from pathlib import Path
from LLM.core import Fault, require, digest, file_hash, atomic_json, inside, uid

CONFIG_KEYS=("architectures","model_type","hidden_size","num_hidden_layers","num_attention_heads","num_key_value_heads","intermediate_size","vocab_size","tie_word_embeddings","rope_theta","rope_scaling")
TOKEN_FILES=("tokenizer.json","tokenizer_config.json","special_tokens_map.json","vocab.json","merges.txt","tokenizer.model")

class Registry:
    def __init__(self, store, allowed_roots=()):
        self.store=store
        self.allowed_roots=[Path(p).resolve() for p in allowed_roots]
        self.verified_files={}

    def directory(self, artifact):
        return inside(self.store.root, artifact["storagePath"])

    def verify(self, artifact):
        require(artifact['state']!='archived','ARTIFACT_ARCHIVED',status=409)
        root=self.directory(artifact)
        for name, checksum in artifact["files"].items():
            p=inside(root,name)
            require(p.is_file(),"ARTIFACT_HASH_MISMATCH",status=409,artifactId=artifact["id"],file=name)
            signature=(p.stat().st_size,p.stat().st_mtime_ns,checksum)
            if self.verified_files.get(str(p))!=signature:
                require(file_hash(p)==checksum,"ARTIFACT_HASH_MISMATCH",status=409,artifactId=artifact["id"],file=name)
                self.verified_files[str(p)]=signature
        require(artifact["state"] not in ("revoked","quarantined","failed"),"POLICY_REVOKED",status=409)
        return root

    def inspect(self, directory, kind):
        from safetensors import safe_open
        root=Path(directory)
        files={p.relative_to(root).as_posix():file_hash(p) for p in sorted(root.rglob("*")) if p.is_file() and p.name!="artifact-manifest.json"}
        require(bool(files),"EMPTY_ARTIFACT")
        schema={}; parameters=0
        for name in files:
            if name.endswith(".safetensors"):
                with safe_open(str(root/name),framework="pt",device="cpu") as f:
                    for key in f.keys():
                        require(key not in schema,"DUPLICATE_TENSOR")
                        shape=f.get_slice(key).get_shape()
                        schema[key]=shape
                        n=1
                        for d in shape:n*=d
                        parameters+=n
        require(bool(schema) or kind=="quantized","MERGE_INPUT_REQUIRES_FLOAT_WEIGHTS")
        config=json.loads((root/"config.json").read_text("utf-8")) if (root/"config.json").exists() else {}
        tokenizer={name:files[name] for name in TOKEN_FILES if name in files}
        template=json.loads((root/"tokenizer_config.json").read_text("utf-8")).get("chat_template","") if (root/"tokenizer_config.json").exists() else ""
        return {"files":files,"manifestHash":digest(files),"architectureHash":digest({k:config.get(k) for k in CONFIG_KEYS}),
                "tensorSchemaHash":digest(schema),"tensorSchema":schema,"parameterCount":parameters,
                "tokenizerHash":digest(tokenizer),"templateHash":digest(template),"sizeBytes":sum((root/n).stat().st_size for n in files)}

    def publish(self, directory, *, kind, family_id=None, parents=None, branch_id=None, license_refs=None, metadata=None):
        require(kind in {"base","adapter","specialist_checkpoint","merged_checkpoint","distilled_checkpoint","quantized","training_checkpoint"},"INVALID_ARTIFACT_KIND")
        root=Path(directory).resolve()
        require(not any(p.is_symlink() for p in root.rglob("*")),"SYMLINK_NOT_ALLOWED")
        info=self.inspect(root,kind)
        identity=kind+"-"+info["manifestHash"][:24]
        destination=self.store.root/"artifacts"/identity
        if not destination.exists():
            temp=self.store.root/"staging"/(identity+"-"+uid())
            shutil.copytree(root,temp)
            require(self.inspect(temp,kind)["manifestHash"]==info["manifestHash"],"ARTIFACT_HASH_MISMATCH")
            atomic_json(temp/"artifact-manifest.json",info)
            temp.replace(destination)
        old=self.store.get("artifact",identity,{})
        if old:
            require(old.get("familyId")==family_id or kind=="base","ARTIFACT_FAMILY_CONFLICT",status=409)
            if old.get('state')=='archived':old=self.store.put('artifact',{'state':'verified'},identity)
            self.verify(old);return old
        inherited_fixture=any(self.store.get('artifact',p,{}).get('fixture') for p in (parents or []))
        value=self.store.put("artifact",{"kind":kind,"familyId":family_id,"parents":parents or [],"branchId":branch_id,
            "storagePath":destination.relative_to(self.store.root).as_posix(),"state":"verified","licenseRefs":license_refs or [],
            "format":"gguf" if kind=="quantized" else "safetensors",**info,**(metadata or {}),"fixture":inherited_fixture or bool((metadata or {}).get('fixture'))},identity)
        self.store.event("artifact.verified",identity,{"kind":kind})
        return value

    def register(self, body):
        root=Path(body.get("directory","")).resolve()
        require(any(root.is_relative_to(p) for p in self.allowed_roots),"MODEL_PATH_NOT_ALLOWED",status=403)
        require(root.is_dir() and not root.is_symlink(),"MODEL_DIRECTORY_REQUIRED")
        require(bool(body.get("licenseRefs")),"MODEL_LICENSE_REQUIRED")
        require((root/"config.json").is_file() and (root/"tokenizer.json").is_file(),"MODEL_BUNDLE_INCOMPLETE")
        require(not list(root.glob("*.bin")),"PICKLE_WEIGHTS_NOT_ALLOWED")
        require(sum(p.stat().st_size for p in root.rglob("*") if p.is_file())<=int(body.get("maxBytes",20*1024**3)),"MODEL_TOO_LARGE")
        artifact=self.publish(root,kind="base",license_refs=body["licenseRefs"],metadata={"sourceRevision":body.get("revision"),"fixture":body.get("fixture") is True})
        family_id="family-"+digest([artifact["manifestHash"],artifact["tokenizerHash"]])[:20]
        family=self.store.get("family",family_id,{}) or self.store.put("family",{"anchorArtifactId":artifact["id"],
            "architectureHash":artifact["architectureHash"],"tensorSchemaHash":artifact["tensorSchemaHash"],
            "tokenizerHash":artifact["tokenizerHash"],"templateHash":artifact["templateHash"],"name":body.get("name",root.name)},family_id)
        artifact=self.store.put("artifact",{"familyId":family_id},artifact["id"])
        return {"artifact":artifact,"family":family}

    def branch(self, body):
        family=self.store.get("family",body.get("familyId",""))
        label=str(body.get("label","")).strip()
        require(bool(label) and len(label)<=32,"INVALID_BRANCH_LABEL")
        require(not any(b["familyId"]==family["id"] and b["label"]==label for b in self.store.list("branch")),"BRANCH_EXISTS",status=409)
        return self.store.put("branch",{"familyId":family["id"],"label":label,"specialization":body.get("specialization","code"),
            "currentArtifactId":None,"currentAdapterId":None,"anchorArtifactId":family["anchorArtifactId"]})

    def revoke(self, artifact_id, reason):
        affected={artifact_id}
        for _ in range(len(self.store.list("artifact"))+1):
            before=len(affected)
            for a in self.store.list("artifact"):
                if set(a.get("parents",[])) & affected: affected.add(a["id"])
            if before==len(affected):break
        for identity in affected:self.store.put("artifact",{"state":"revoked","revocationReason":reason},identity)
        return sorted(affected)
