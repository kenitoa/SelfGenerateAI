"""Actual tensor merges. No inference routing or factor-wise LoRA averaging."""
import json
import math
import shutil
from pathlib import Path
from LLM.core import require, digest, atomic_json, uid
from LLM.foundation.origins import FOUNDATION_ORIGINS

def tensor_map(directory):
    from safetensors import safe_open
    result={}
    for p in sorted(Path(directory).glob("*.safetensors")):
        with safe_open(str(p),framework="pt",device="cpu") as f:
            for k in f.keys():
                require(k not in result,"DUPLICATE_TENSOR")
                result[k]=p
    return result

def read_tensor(mapping, key):
    from safetensors import safe_open
    with safe_open(str(mapping[key]),framework="pt",device="cpu") as f:return f.get_tensor(key)

def copy_metadata(source, destination):
    for p in Path(source).iterdir():
        if p.is_file() and not (p.name.endswith((".safetensors",".index.json",".bin")) or p.name=="artifact-manifest.json"):
            shutil.copyfile(p,Path(destination)/p.name)

def exact_lora_delta(a, b, scale):
    return b.float() @ a.float() * scale

class Merger:
    def __init__(self, store, registry):self.store,self.registry=store,registry

    def preflight(self, body):
        ids=body.get("artifactIds") or [m["artifactId"] for m in body.get("members",[])]
        require(isinstance(ids,list) and 1<=len(ids)<=3 and len(set(ids))==len(ids),"INVALID_COMBINATION")
        parents=[self.store.get("artifact",i) for i in ids]
        require(not any(parent.get('comparisonOnly') for parent in parents),'COMPARISON_ONLY_MODEL',status=409)
        weights=body.get("weights") or [m.get("weight",1) for m in body.get("members",[])] or [1]*len(ids)
        require(len(weights)==len(ids) and all(isinstance(w,(float,int)) and not isinstance(w,bool) and math.isfinite(w) and w>0 for w in weights),"INVALID_MERGE_WEIGHTS")
        members=sorted(zip(parents,weights),key=lambda x:x[0]["id"])
        parents=[x[0] for x in members];weights=[x[1]/sum(weights) for x in members]
        family=parents[0]["familyId"]; reasons=[]
        family_record=self.store.get('family',family)
        anchor=self.store.get('artifact',family_record['anchorArtifactId'])
        self.registry.verify(anchor)
        anchor_architecture = self.registry.current_architecture_hash(anchor)
        parent_architectures = {}
        method=body.get("method","linear_delta")
        require(method in {"linear_delta","task_arithmetic","ties","dare_linear","dare_ties","slerp"},"UNSUPPORTED_MERGE_METHOD")
        if method=="slerp" and len(parents)!=2:reasons.append("SLERP_REQUIRES_TWO_MODELS")
        branches=[p.get("branchId") for p in parents]
        if len(ids)>1 and (any(b is None for b in branches) or len(set(branches))!=len(branches) or any(p['kind'] not in ('specialist_checkpoint',) for p in parents)):
            reasons.append("DUPLICATE_BRANCH_OR_NESTED_COMBINATION")
        for p in parents:
            parent_architectures[p['id']] = self.registry.current_architecture_hash(p)
            if p["familyId"]!=family:reasons.append("INCOMPATIBLE_BASE")
            if p.get('tokenizerHash')!=family_record['tokenizerHash']:reasons.append('INCOMPATIBLE_TOKENIZER')
            if p.get('templateHash')!=family_record['templateHash']:reasons.append('INCOMPATIBLE_TEMPLATE')
            if parent_architectures[p['id']]!=anchor_architecture:reasons.append('INCOMPATIBLE_ARCHITECTURE')
            if p.get('tensorSchemaHash')!=anchor.get('tensorSchemaHash'):reasons.append('INCOMPATIBLE_TENSOR_SCHEMA')
            if anchor.get('origin') in FOUNDATION_ORIGINS or p.get('origin') in FOUNDATION_ORIGINS or anchor.get('foundationProjectId') or p.get('foundationProjectId'):
                for field in ('origin','foundationProjectId','genesisId','tokenizerId','tokenizerHash','architectureHashV2'):
                    if p.get(field)!=anchor.get(field):reasons.append('MERGE_FAMILY_INCOMPATIBLE')
                if p.get('tokenizerAssetsHash')!=anchor.get('tokenizerAssetsHash'):reasons.append('INCOMPATIBLE_TOKENIZER')
            for key,code in (("tensorSchemaHash","INCOMPATIBLE_TENSOR_SCHEMA"),("tokenizerHash","INCOMPATIBLE_TOKENIZER"),("templateHash","INCOMPATIBLE_TEMPLATE")):
                if p[key]!=parents[0][key]:reasons.append(code)
            if p.get('format')!='safetensors':reasons.append("MERGE_INPUT_REQUIRES_FLOAT_WEIGHTS")
            self.registry.verify(p)
        scale=body.get("deltaScale",1)
        require(isinstance(scale,(int,float)) and not isinstance(scale,bool) and math.isfinite(scale) and 0<scale<=2,"INVALID_DELTA_SCALE")
        density=body.get("density",.8)
        require(isinstance(density,(int,float)) and not isinstance(density,bool) and math.isfinite(density) and 0<density<=1,"INVALID_DENSITY")
        seed=body.get('seed',42)
        require(isinstance(seed,int) and not isinstance(seed,bool) and 0<=seed<2**32,'INVALID_MERGE_SEED')
        recipe={"familyId":family,"anchorArtifactId":self.store.get("family",family)["anchorArtifactId"],
            "members":[{"artifactId":p['id'],"digest":p['manifestHash'],"coefficient":w} for p,w in zip(parents,weights)],
            "method":method,"deltaScale":scale,"density":density,"seed":seed,
            "computeDtype":"float32","outputDtype":body.get("outputDtype","float32"),"implementationVersion":1,
            "tokenizerHash":parents[0]["tokenizerHash"],"templateHash":parents[0]["templateHash"]}
        require(recipe['outputDtype'] in ('float32','float16','bfloat16'),"INVALID_OUTPUT_DTYPE")
        recipe_hash=digest(recipe)
        old=self.store.get("combination",recipe_hash,{})
        labels=[self.store.get('branch',p['branchId'])['label'] if p.get('branchId') else p['id'] for p in parents]
        selected_artifact=old.get('servingArtifactId') or old.get('outputArtifactId')
        artifact=self.store.get('artifact',selected_artifact,{}) if selected_artifact else {}
        if len(parents)==1:artifact=parents[0]
        ready=artifact.get('state')=='ready'
        largest=max((math.prod(shape) for shape in parents[0].get('tensorSchema',{}).values()),default=0)
        estimated_ram=largest*4*(3+2*len(parents))+64*1024**2
        return {"displayName":"+".join(sorted(labels)),"compatible":not reasons,"reasonCodes":sorted(set(reasons)),
            "recipeHash":recipe_hash,"recipe":recipe,"cachedArtifactId":artifact.get('id'),"state":"ready" if ready else old.get('state','not_built'),
            "resolvedMembers":[{"artifactId":p['id'],"label":label,"branchId":p.get('branchId')} for p,label in zip(parents,labels)],
            "normalizedWeights":weights,"estimatedResources":{"peakRamBytes":estimated_ram,"workDiskBytes":sum(p['sizeBytes'] for p in parents)*2,"measured":False},
            "distillationEligibility":bool(reasons),"actions":{"build":not reasons and len(ids)>1,"use":ready}}

    def recipe(self, body):
        check=self.preflight(body)
        require(check['compatible'],"INCOMPATIBLE_COMBINATION",status=409,reasons=check['reasonCodes'])
        existing=self.store.get("combination",check['recipeHash'],{})
        if existing:return existing
        return self.store.put("combination",{**check,"state":"not_built"},check['recipeHash'])

    def build(self, job):
        import torch
        from safetensors.torch import save_file
        torch.set_num_threads(1)
        combo=self.store.get('combination',job['payload']['combinationId'])
        recipe=combo['recipe']
        require(digest(recipe)==combo['id']==combo['recipeHash'],'MERGE_RECIPE_HASH_MISMATCH',status=409)
        parents=[self.store.get('artifact',m['artifactId']) for m in recipe['members']]
        fresh=self.preflight({'artifactIds':[p['id'] for p in parents], 'weights':[m['coefficient'] for m in recipe['members']],**{k:recipe[k] for k in ('method','deltaScale','density','seed','outputDtype')}})
        require(fresh['compatible'],"INCOMPATIBLE_COMBINATION",status=409)
        require(fresh['recipeHash']==combo['recipeHash'],'MERGE_RECIPE_CHANGED',status=409)
        from LLM.workers.resources import observe
        available=observe(self.store.root)['availableRamBytes']
        require(available-max(512*1024**2,int(available*.2))>=fresh['estimatedResources']['peakRamBytes'],'RESOURCE_PRESSURE',status=503)
        anchor=self.store.get('artifact',recipe['anchorArtifactId'])
        base=tensor_map(self.registry.verify(anchor)); maps=[tensor_map(self.registry.verify(p)) for p in parents]
        require(all(set(m)==set(base) for m in maps),'INCOMPATIBLE_TENSOR_SCHEMA')
        destination=self.store.root/'staging'/('merge-'+uid());destination.mkdir()
        copy_metadata(self.registry.directory(anchor),destination)
        keys=sorted(base); index={}; shard={}; shard_bytes=0; shard_no=0
        def flush():
            nonlocal shard,shard_bytes,shard_no
            if not shard:return
            name=f'model-{shard_no:05d}.safetensors';save_file(shard,str(destination/name))
            index.update({k:name for k in shard});shard={};shard_bytes=0;shard_no+=1
        for position,key in enumerate(keys):
            current=self.store.assert_lease(job)
            require(current['state'] not in ('cancelling','cancelled'),'JOB_CANCELLED',status=409)
            original=read_tensor(base,key).float()
            values=[read_tensor(m,key).float() for m in maps]
            require(all(v.shape==original.shape for v in values),'INCOMPATIBLE_TENSOR_SCHEMA')
            if recipe['method']=='slerp':
                u,v=values; uf,vf=u.flatten(),v.flatten();den=uf.norm()*vf.norm()
                dot=torch.clamp(torch.dot(uf,vf)/den,-1,1) if den>0 else torch.tensor(1.)
                theta=torch.acos(dot); t=recipe['members'][1]['coefficient']
                merged=((1-t)*u+t*v) if abs(float(dot))>.9995 else (torch.sin((1-t)*theta)*u+torch.sin(t*theta)*v)/torch.sin(theta)
            else:
                deltas=[v-original for v in values]
                for i,d in enumerate(deltas):
                    if recipe['method'].startswith('dare'):
                        g=torch.Generator().manual_seed((recipe['seed']+int(digest([key,i])[:12],16))%(2**63-1))
                        d=d*(torch.rand(d.shape,generator=g)<recipe['density'])/recipe['density']
                    if 'ties' in recipe['method']:
                        cutoff=torch.quantile(d.abs().flatten(),1-recipe['density']) if d.numel()>1 else 0
                        d=d*(d.abs()>=cutoff)
                    deltas[i]=d*recipe['members'][i]['coefficient']
                if 'ties' in recipe['method']:
                    stack=torch.stack(deltas);sign=stack.sum(0).sign();mask=(stack.sign()==sign)&(stack!=0)
                    delta=(stack*mask).sum(0)/mask.sum(0).clamp(min=1)
                else:delta=sum(deltas,torch.zeros_like(original))
                merged=original+recipe['deltaScale']*delta
            require(bool(torch.isfinite(merged).all()),'NONFINITE_MERGE')
            result=merged.to(getattr(torch,recipe['outputDtype'])).contiguous()
            if shard_bytes+result.numel()*result.element_size()>64*1024**2:flush()
            shard[key]=result;shard_bytes+=result.numel()*result.element_size()
            self.store.progress(job,'merging',position+1,len(keys),'tensors')
        flush();atomic_json(destination/'model.safetensors.index.json',{'metadata':{},'weight_map':index})
        atomic_json(destination/'merge-recipe.json',recipe)
        for parent in parents:self.registry.verify(self.store.get('artifact',parent['id']))
        artifact=self.registry.publish(destination,kind='merged_checkpoint',family_id=anchor['familyId'],parents=[p['id'] for p in parents],
            license_refs=anchor['licenseRefs'],metadata={'recipeHash':combo['id']})
        with self.store.tx():
            self.store.assert_lease(job)
            self.store.put('combination',{'state':'built','outputArtifactId':artifact['id'],'buildId':job['id']},combo['id'])
        return {'artifactId':artifact['id'],'combinationId':combo['id'],'singleArtifact':True}
