"""Bounded dev-only recipe search and explicit approximate adapter compression."""
import json
import shutil
from LLM.core import require,uid,atomic_json


def truncated_delta(a,b,scale,rank):
    import torch
    delta=(b.float()@a.float())*scale
    require(1<=rank<=min(delta.shape),'INVALID_SVD_RANK')
    u,s,vh=torch.linalg.svd(delta,full_matrices=False)
    left=u[:,:rank]*s[:rank].sqrt()
    right=s[:rank].sqrt().unsqueeze(1)*vh[:rank]
    denominator=float(delta.norm())
    error=float((delta-left@right).norm())/denominator if denominator else 0.
    return right.contiguous(),left.contiguous(),error


def compress_adapter(control,job):
    from safetensors.torch import load_file,save_file
    artifact=control.store.get('artifact',job['payload']['artifactId'])
    require(artifact['kind']=='adapter','ADAPTER_REQUIRED')
    root=control.registry.verify(artifact)
    config=json.loads((root/'adapter_config.json').read_text('utf-8'))
    require(not config.get('rank_pattern') and not config.get('alpha_pattern') and not config.get('use_dora') and not config.get('use_rslora'),'UNSUPPORTED_SVD_ADAPTER')
    rank=job['payload']['rank'];require(isinstance(rank,int) and 1<=rank<=config['r'],'INVALID_SVD_RANK')
    values=load_file(str(root/'adapter_model.safetensors'));output={};errors={}
    for name,a in values.items():
        if '.lora_A.' not in name:
            if '.lora_B.' not in name:output[name]=a
            continue
        control.store.assert_lease(job)
        partner=name.replace('.lora_A.','.lora_B.')
        require(partner in values,'INVALID_ADAPTER_TENSORS')
        right,left,error=truncated_delta(a,values[partner],config['lora_alpha']/config['r'],rank)
        output[name]=right;output[partner]=left;errors[name]=error
    require(bool(errors),'INVALID_ADAPTER_TENSORS')
    dest=control.store.root/'staging'/('svd-'+uid());dest.mkdir()
    for path in root.iterdir():
        if path.is_file() and path.suffix!='.safetensors' and path.name!='artifact-manifest.json':shutil.copyfile(path,dest/path.name)
    config.update(r=rank,lora_alpha=rank)
    atomic_json(dest/'adapter_config.json',config)
    save_file(output,str(dest/'adapter_model.safetensors'))
    receipt={'rank':rank,'relativeFrobeniusErrors':errors,'approximate':True,'qualityCertified':False}
    atomic_json(dest/'svd-receipt.json',receipt)
    result=control.registry.publish(dest,kind='adapter',family_id=artifact['familyId'],parents=[artifact['id']],branch_id=artifact.get('branchId'),license_refs=artifact['licenseRefs'],metadata={'compression':receipt})
    return {'artifactId':result['id'],**receipt}


class RecipeSearch:
    def __init__(self,control):self.c=control;self.store=control.store

    def create(self,body):
        suite=self.store.get('suite',body['suiteId'])
        require(suite['kind']=='dev','SEARCH_REQUIRES_DEV_SUITE')
        require(suite.get('fixture') or len(suite['tasks'])>=60,'INSUFFICIENT_DEV_SUITE')
        candidates=body.get('candidates')
        require(isinstance(candidates,list) and 1<=len(candidates)<=3,'SEARCH_LIMIT_EXCEEDED')
        combos=[self.c.merger.recipe({**candidate,'artifactIds':body['artifactIds']}) for candidate in candidates]
        require(len({v['id'] for v in combos})==len(combos),'DUPLICATE_SEARCH_RECIPE')
        identity=uid()
        jobs=[self.store.enqueue('merge',{'combinationId':v['id']},'search:'+identity+':'+v['id']) for v in combos]
        return self.store.put('recipe_search',{'state':'building','suiteId':suite['id'],'artifactIds':body['artifactIds'],
            'combinationIds':[v['id'] for v in combos],'buildJobIds':[j['id'] for j in jobs],'baselineIds':body.get('baselineIds',body['artifactIds'])},identity)

    def advance(self):
        for search in self.store.list('recipe_search'):
            if search['state']=='building':
                jobs=[self.store.job(i) for i in search['buildJobIds']]
                if any(j['state'] in ('failed','cancelled') for j in jobs):
                    self.store.put('recipe_search',{'state':'held','reasonCodes':['BUILD_FAILED']},search['id']);continue
                if not all(j['state']=='completed' for j in jobs):continue
                evaluations=[self.store.enqueue('evaluate',{'artifactId':j['result']['artifactId'],'suiteId':search['suiteId'],'baselineIds':search['baselineIds']},'search-eval:'+search['id']+':'+j['id']) for j in jobs]
                self.store.put('recipe_search',{'state':'evaluating','evaluationJobIds':[j['id'] for j in evaluations]},search['id'])
            elif search['state']=='evaluating':
                jobs=[self.store.job(i) for i in search['evaluationJobIds']]
                if any(j['state'] in ('failed','cancelled') for j in jobs):
                    self.store.put('recipe_search',{'state':'held','reasonCodes':['EVALUATION_FAILED']},search['id']);continue
                if not all(j['state']=='completed' for j in jobs):continue
                passed=[j['result'] for j in jobs if j['result']['gate']['passed']]
                best=max(passed,key=lambda r:sum(t['passed'] for t in r['results'][r['artifactId']]),default=None)
                self.store.put('recipe_search',{'state':'completed' if best else 'held','selectedArtifactId':best['artifactId'] if best else None,
                    'evaluationIds':[j['result']['id'] for j in jobs],'reasonCodes':[] if best else ['EVALUATION_INCONCLUSIVE'],'productionCertified':False},search['id'])
