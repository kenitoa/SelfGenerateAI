from __future__ import annotations
import json
import shutil
from pathlib import Path
from LLM.core import Fault, require, digest, file_hash, atomic_json, inside, uid
from LLM.foundation.origins import FOUNDATION_ORIGINS,assert_project_available

CONFIG_KEYS=("architectures","model_type","hidden_size","num_hidden_layers","num_attention_heads","num_key_value_heads","intermediate_size","vocab_size","tie_word_embeddings","rope_theta","rope_scaling",
             "hidden_act","rms_norm_eps","attention_bias","mlp_bias","max_position_embeddings","head_dim","pretraining_tp")
TOKEN_FILES=("tokenizer.json","tokenizer_config.json","special_tokens_map.json","vocab.json","merges.txt","tokenizer.model")
FOUNDATION_FIELDS=("origin","foundationProjectId","projectId","genesisId","sourceCheckpointId","tokenizerId","tokenizerArtifactId",
                   "tokenizerHash","architectureHashV2","trainedContextTokens","testOnly","foundationOriginReceipt")
RESERVED_METADATA={"id","kind","familyId","parents","branchId","storagePath","state","format","files",
                   "manifestHash","architectureHash","tensorSchemaHash","tensorSchema","sizeBytes",
                   "tokenizerAssetsHash","templateHash","foundationProvenanceHash"}

class Registry:
    def __init__(self, store, allowed_roots=()):
        self.store=store
        self.allowed_roots=[Path(p).resolve() for p in allowed_roots]
        self.verified_files={}

    def directory(self, artifact):
        return inside(self.store.root, artifact["storagePath"])

    def verify(self, artifact):
        require(artifact['state']!='archived','ARTIFACT_ARCHIVED',status=409)
        if artifact.get('foundationProjectId'):
            project=self.store.get('foundation_project',artifact['foundationProjectId'],{})
            require(project,'FOUNDATION_PROVENANCE_MISMATCH',status=409)
            require(project.get('state')!='revoked' and not project.get('revocationPending'),'POLICY_REVOKED',status=409)
            assert_project_available(self.store,project)
            source=self.store.get('foundation_checkpoint',artifact.get('sourceCheckpointId') or '',{})
            require(source.get('projectId')==project['id'] and source.get('tokenizerHash')==artifact.get('tokenizerHash'),
                'FOUNDATION_PROVENANCE_MISMATCH',status=409)
            teacher_refs=dict(source.get('teacherManifestHashes') or {})
            if source.get('teacherArtifactId'):teacher_refs[source['teacherArtifactId']]=source.get('teacherManifestHash')
            teacher_refs.update(artifact.get('teacherManifestHashes') or {})
            approval_ids=set(source.get('teacherApprovalIds') or [])|set(artifact.get('teacherApprovalIds') or [])
            for approval_id in approval_ids:
                approval=self.store.get('foundation_teacher_approval',approval_id,{})
                require(approval.get('approved') is True,'POLICY_REVOKED',status=409)
            for identity,checksum in teacher_refs.items():
                teacher=self.store.get('artifact',identity,{}) or self.store.get('foundation_checkpoint',identity,{})
                require(teacher.get('state') in ('ready','verified','completed') and teacher.get('manifestHash')==checksum,
                    'POLICY_REVOKED',status=409)
                teacher_project=teacher.get('foundationProjectId',teacher.get('projectId'))
                if teacher_project:assert_project_available(self.store,self.store.get('foundation_project',teacher_project))
        root=self.directory(artifact)
        for name, checksum in artifact["files"].items():
            p=inside(root,name)
            require(p.is_file(),"ARTIFACT_HASH_MISMATCH",status=409,artifactId=artifact["id"],file=name)
            signature=(p.stat().st_size,p.stat().st_mtime_ns,checksum)
            if self.verified_files.get(str(p))!=signature:
                require(file_hash(p)==checksum,"ARTIFACT_HASH_MISMATCH",status=409,artifactId=artifact["id"],file=name)
                self.verified_files[str(p)]=signature
        require(artifact["state"] not in ("revoked","quarantined","failed"),"POLICY_REVOKED",status=409)
        if artifact.get('origin') in FOUNDATION_ORIGINS or artifact.get('foundationProjectId'):
            require(artifact.get('foundationProvenanceHash')==digest({k:artifact.get(k) for k in FOUNDATION_FIELDS}),
                    'FOUNDATION_PROVENANCE_MISMATCH',status=409)
            if artifact.get('origin') == 'architecture_growth':
                from LLM.foundation.schemas import valid_growth_config
                project = self.store.get('foundation_project',artifact['foundationProjectId'])
                receipt = project.get('trustedGrowthReceipt')
                require(project.get('state') != 'revoked' and receipt == artifact.get('trustedGrowthReceipt')
                    and project.get('acceptedArchitectureRecipeId') == (receipt or {}).get('candidateId')
                    and (project.get('testOnly') or valid_growth_config(project['config'],receipt)), 'GROWTH_RECEIPT_REQUIRED')
        return root

    def current_architecture_hash(self, artifact):
        """Compare verified configs under today's keys without rewriting history.

        Older registry records used fewer CONFIG_KEYS. Their historical digest
        is retained, while compatibility compares the actual immutable config.
        Foundation V2/provenance checks remain independent and mandatory.
        """
        root = self.verify(artifact)
        path = root/'config.json'
        require('config.json' in artifact['files'] and path.is_file()
                and path.stat().st_size <= 1024*1024, 'INCOMPATIBLE_ARCHITECTURE')
        config = json.loads(path.read_text('utf-8'))
        return digest({key: config.get(key) for key in CONFIG_KEYS})

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
                "tokenizerHash":digest(tokenizer),"tokenizerAssetsHash":digest(tokenizer),
                "templateHash":digest(template),"sizeBytes":sum((root/n).stat().st_size for n in files)}

    def _foundation_metadata(self, root, kind, family_id, parent_records, metadata, info):
        """Bind scratch provenance to actual trusted inputs, never caller labels."""
        foundations=[p for p in parent_records if p.get('origin') in FOUNDATION_ORIGINS or p.get('foundationProjectId')]
        if family_id:
            family=self.store.get('family',family_id)
            anchor=self.store.get('artifact',family['anchorArtifactId'])
            if anchor.get('origin') in FOUNDATION_ORIGINS or anchor.get('foundationProjectId'):
                self.verify(anchor)
                foundations=[anchor,*foundations]
        if foundations:
            source=foundations[0]
            for parent in foundations:
                self.verify(parent)
                for field in ('origin','foundationProjectId','genesisId','tokenizerId','tokenizerHash','architectureHashV2'):
                    require(parent.get(field)==source.get(field),'MERGE_FAMILY_INCOMPATIBLE',field=field)
            require(source.get('origin') in FOUNDATION_ORIGINS and all(p.get('origin')==source.get('origin') and p.get('foundationProjectId')==source['foundationProjectId']
                        for p in parent_records),'MERGE_FAMILY_INCOMPATIBLE')
            expected={key:source.get(key) for key in FOUNDATION_FIELDS}
            expected.update({key:source.get(key) for key in ('trustedGrowthReceipt','executionScope',
                'pretrainedWeightArtifactIds','sourceProjectId','sourceGenesisId','teacherArtifactId','teacherArtifactIds',
                'teacherManifestHash','teacherManifestHashes','distillationMode','teacherApprovalIds')})
            contexts=[p.get('trainedContextTokens') for p in foundations]
            require(all(isinstance(n,int) and not isinstance(n,bool) and n>=0 for n in contexts),'FOUNDATION_CONTEXT_UNKNOWN')
            expected['trainedContextTokens']=min(contexts)
            expected['testOnly']=any(p.get('testOnly') or p.get('fixture') for p in foundations)
            # Adapter-only checkpoints have no tokenizer/config files of their own.
            if kind not in ('adapter','quantized'):
                require(info['architectureHash']==self.current_architecture_hash(source),'MERGE_FAMILY_INCOMPATIBLE',field='architectureHash')
                require(info['tokenizerAssetsHash']==source.get('tokenizerAssetsHash'),'INCOMPATIBLE_TOKENIZER')
                require(info['templateHash']==source['templateHash'],'INCOMPATIBLE_TEMPLATE')
        elif metadata.get('origin') in FOUNDATION_ORIGINS or metadata.get('foundationProjectId'):
            require(kind=='base' and not parent_records,'FOUNDATION_ORIGIN_REQUIRED')
            from LLM.foundation.checkpoint import verify as verify_checkpoint
            source=self.store.get('foundation_checkpoint',metadata.get('sourceCheckpointId',''))
            cp_root,manifest=verify_checkpoint(self.store,source)
            project=self.store.get('foundation_project',source['projectId'])
            require(source.get('origin','random_init') in FOUNDATION_ORIGINS
                    and source.get('origin','random_init')==project.get('origin','random_init'),'FOUNDATION_ORIGIN_REQUIRED')
            token_id=source.get('tokenizerId',source.get('tokenizerArtifactId'))
            token=self.store.get('foundation_tokenizer',token_id)
            from LLM.foundation.tokenizer import load as load_tokenizer
            load_tokenizer(self.store,token)
            token_root=inside(self.store.root,token['storagePath'])
            assets=digest({name:file_hash(token_root/name) for name in TOKEN_FILES if (token_root/name).is_file()})
            require(info['tokenizerAssetsHash']==assets,'INCOMPATIBLE_TOKENIZER')
            cfg=json.loads((root/'config.json').read_text('utf-8'))
            require(all(cfg.get(key)==value for key,value in manifest['config'].items()),'INCOMPATIBLE_ARCHITECTURE')
            pointer=source;seen=set()
            while pointer.get('parentCheckpointId') and pointer['id'] not in seen:
                seen.add(pointer['id']);parent_id=pointer['parentCheckpointId']
                pointer=self.store.get('foundation_checkpoint',parent_id,{}) or self.store.get('foundation_genesis',parent_id)
            genesis_id=pointer['id']
            require(bool(self.store.get('foundation_genesis',genesis_id,{})),'FOUNDATION_ORIGIN_REQUIRED')
            self._same_foundation_weights(root,cp_root,manifest,cfg)
            expected={'origin':source.get('origin','random_init'),'foundationProjectId':source['projectId'],'projectId':source['projectId'],'sourceCheckpointId':source['id'],
                'genesisId':genesis_id,'tokenizerId':token_id,'tokenizerArtifactId':token_id,
                'tokenizerHash':token['tokenizerHash'],'architectureHashV2':source['architectureHashV2'],
                'trainedContextTokens':source.get('trainedContextTokens',0),
                'testOnly':bool(project.get('testOnly') or source.get('testOnly') or source.get('fixture')),
                'foundationOriginReceipt':digest([source['manifestHash'],token['tokenizerHash'],source['architectureHashV2']])}
            expected.update({'trustedGrowthReceipt':project.get('trustedGrowthReceipt'),
                'executionScope':project.get('executionScope','nas'),
                'pretrainedWeightArtifactIds':project.get('pretrainedWeightArtifactIds',[]),
                'sourceProjectId':project.get('sourceProjectId'),'sourceGenesisId':project.get('sourceGenesisId')})
            expected.update({key:source.get(key) for key in ('teacherArtifactId','teacherArtifactIds',
                'teacherManifestHash','teacherManifestHashes','distillationMode','teacherApprovalIds')})
        else:
            require(not any(key in metadata for key in ('genesisId','foundationProjectId','foundationOriginReceipt')),
                    'FOUNDATION_ORIGIN_REQUIRED')
            require('tokenizerHash' not in metadata or metadata['tokenizerHash']==info['tokenizerHash'],
                    'ARTIFACT_METADATA_OVERRIDE',field='tokenizerHash')
            return metadata
        for key,value in expected.items():
            if key in metadata:
                require(metadata[key]==value,'FOUNDATION_PROVENANCE_MISMATCH',field=key)
        require(metadata.get('servingEligible',False) is False,'FOUNDATION_RELEASE_CERTIFICATE_REQUIRED')
        result={**metadata,**expected,'servingEligible':False}
        result['foundationProvenanceHash']=digest({k:result.get(k) for k in FOUNDATION_FIELDS})
        return result

    def _same_foundation_weights(self, root, cp_root, manifest, config):
        """A valid checkpoint ID cannot relabel unrelated bytes as scratch weights."""
        from safetensors import safe_open
        import torch
        incoming={}
        for path in root.glob('*.safetensors'):
            with safe_open(str(path),framework='pt',device='cpu') as handle:
                for key in handle.keys():incoming[key]=path
        with safe_open(str(cp_root/'model.safetensors'),framework='pt',device='cpu') as handle:
            aliases=manifest.get('tensorAliases',{})
            expected=set(handle.keys())|set(aliases)
            available=set(incoming)
            if config.get('tie_word_embeddings'):
                for key,other in (('lm_head.weight','model.embed_tokens.weight'),('model.embed_tokens.weight','lm_head.weight')):
                    if key not in incoming and other in incoming:incoming[key]=incoming[other]
            require(set(incoming)==expected,'INCOMPATIBLE_TENSOR_SCHEMA')
            for key in sorted(expected):
                owner=aliases.get(key,key)
                left=handle.get_tensor(owner)
                path=incoming[key]
                with safe_open(str(path),framework='pt',device='cpu') as exported:
                    actual_key=key if key in exported.keys() else ('model.embed_tokens.weight' if key=='lm_head.weight' else 'lm_head.weight')
                    right=exported.get_tensor(actual_key)
                require(left.shape==right.shape and torch.equal(left,right),'FOUNDATION_WEIGHT_PROVENANCE_MISMATCH')

    def publish(self, directory, *, kind, family_id=None, parents=None, branch_id=None, license_refs=None, metadata=None):
        require(kind in {"base","adapter","specialist_checkpoint","merged_checkpoint","distilled_checkpoint","quantized","training_checkpoint"},"INVALID_ARTIFACT_KIND")
        metadata=dict(metadata or {})
        require(not set(metadata)&RESERVED_METADATA,'ARTIFACT_METADATA_OVERRIDE',fields=sorted(set(metadata)&RESERVED_METADATA))
        root=Path(directory).resolve()
        require(not any(p.is_symlink() for p in root.rglob("*")),"SYMLINK_NOT_ALLOWED")
        info=self.inspect(root,kind)
        parent_records=[self.store.get('artifact',p) for p in (parents or [])]
        for parent in parent_records:self.verify(parent)
        protected=[p for p in self.store.list('artifact') if p.get('comparisonOnly')]
        weight_hashes=sorted(value for name,value in info['files'].items() if name.endswith('.safetensors'))
        protected_bytes=bool(weight_hashes and any(weight_hashes==sorted(value for name,value in item['files'].items()
            if name.endswith('.safetensors')) for item in protected))
        require(not (protected_bytes or any(p.get('comparisonOnly') for p in parent_records))
            or metadata.get('comparisonOnly') is True,'COMPARISON_ONLY_MODEL',status=409)
        if 'parameterCount' in metadata:
            require(kind=='quantized' and parent_records and metadata['parameterCount']==parent_records[0]['parameterCount'],
                    'ARTIFACT_METADATA_OVERRIDE',field='parameterCount')
        metadata=self._foundation_metadata(root,kind,family_id,parent_records,metadata,info)
        legacy_identity=kind+"-"+info["manifestHash"][:24]
        proof=metadata.get('foundationProvenanceHash')
        if proof:
            # Equal bytes in a newly proven Genesis/branch remain a different
            # immutable lineage; never rebind or reactivate a revoked old row.
            legacy=self.store.get('artifact',legacy_identity,{})
            same_legacy=(legacy and legacy.get('foundationProvenanceHash')==proof
                and legacy.get('parents',[])==(parents or []) and legacy.get('branchId')==branch_id
                and legacy.get('executionScope','nas')==metadata.get('executionScope','nas'))
            identity=legacy_identity if same_legacy else kind+'-'+digest([info['manifestHash'],proof,
                parents or [],branch_id,metadata.get('executionScope','nas')])[:24]
        else:
            require(not any(item.get('manifestHash')==info['manifestHash'] and item.get('foundationProvenanceHash')
                for item in self.store.list('artifact')),'FOUNDATION_PROVENANCE_MISMATCH',status=409)
            identity=legacy_identity
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
            require(old.get('parents',[])==(parents or []) and old.get('branchId')==branch_id,'ARTIFACT_PROVENANCE_CONFLICT',status=409)
            require(old.get('foundationProvenanceHash')==metadata.get('foundationProvenanceHash'),
                    'FOUNDATION_PROVENANCE_MISMATCH',status=409)
            if old.get('state')=='archived':old=self.store.put('artifact',{'state':'verified'},identity)
            self.verify(old);return old
        inherited_fixture=any(p.get('fixture') for p in parent_records)
        with self.store.tx():
            project_ids={p.get('foundationProjectId') for p in parent_records}|{metadata.get('foundationProjectId')}
            for project_id in project_ids-{None}:
                project=self.store.get('foundation_project',project_id)
                require(project.get('state')!='revoked' and not project.get('revocationPending'),'POLICY_REVOKED',status=409)
                assert_project_available(self.store,project)
            for parent in parent_records:
                current=self.store.get('artifact',parent['id'])
                require(current.get('state') not in ('revoked','quarantined','failed','archived')
                    and current.get('manifestHash')==parent['manifestHash'],'POLICY_REVOKED',status=409)
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
        require(artifact.get('origin') not in FOUNDATION_ORIGINS,'FOUNDATION_REGISTRATION_REQUIRED',status=409)
        family_id="family-"+digest([artifact["manifestHash"],artifact["tokenizerHash"]])[:20]
        family=self.store.get("family",family_id,{}) or self.store.put("family",{"anchorArtifactId":artifact["id"],
            "architectureHash":artifact["architectureHash"],"tensorSchemaHash":artifact["tensorSchemaHash"],
            "tokenizerHash":artifact["tokenizerHash"],"templateHash":artifact["templateHash"],"name":body.get("name",root.name)},family_id)
        artifact=self.store.put("artifact",{"familyId":family_id},artifact["id"])
        return {"artifact":artifact,"family":family}

    def branch(self, body):
        family=self.store.get("family",body.get("familyId",""))
        require(not self.store.get('artifact',family['anchorArtifactId']).get('comparisonOnly'),
                'COMPARISON_ONLY_MODEL',status=409)
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
        callback=getattr(self,'on_revoke',None)
        if callable(callback):callback(sorted(affected),reason)
        return sorted(affected)
