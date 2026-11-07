"""Bounded function-preserving growth experiments; never promote by size.

Only SwiGLU FFN widening with zero output columns and zero-residual decoder
insertion are supported. Width, vocabulary, attention and tokenizer stay fixed.
These are derived-weight candidates, not new random initializations.
"""
from __future__ import annotations

import json
import shutil
from LLM.core import require, digest, file_hash, atomic_json, inside, uid, now
from .schemas import ARCHITECTURE, architecture_config, parameter_count, integer


def configure_policy(control,project_id,body):
    """One-time operator scope for bounded, dev-triggered candidate experiments."""
    from .schemas import real
    require(isinstance(body,dict) and body.get('approved') is True,'GROWTH_POLICY_EXPLICIT_APPROVAL_REQUIRED')
    project=control.foundation.project(project_id)
    operation=body.get('operation','ffn_expand')
    require(operation in ('ffn_expand','insert_identity_layer'),'UNSUPPORTED_GROWTH_OPERATION')
    increment=integer(body.get('ffnIncrement',min(128,project['config']['intermediate_size'])),'ffnIncrement',8,
        min(1536,project['config']['intermediate_size']))
    require(increment%8==0,'GROWTH_DIMENSION_ALIGNMENT')
    policy={'projectId':project_id,'enabled':body.get('enabled',False) is True,'approved':True,
        'operation':operation,'ffnIncrement':increment,'autoAdopt':body.get('autoAdopt',True) is True,
        'autoTrain':body.get('autoTrain',True) is True,'maxGrowthDepth':integer(body.get('maxGrowthDepth',2),'maxGrowthDepth',1,4),
        'plateauWindow':integer(body.get('plateauWindow',3),'plateauWindow',3,10),
        'minRelativeImprovement':real(body.get('minRelativeImprovement',.01),'minRelativeImprovement',0,.1),
        'maxExperimentsPerDay':1,'maxParameters':32_000_000,'qualityGateRequired':True}
    policy['policyHash']=digest(policy)
    existing=control.store.get('foundation_growth_policy',project_id,{})
    if existing.get('policyHash')==policy['policyHash']:return existing
    running=control.store.get('foundation_growth_automation',project_id,{})
    require(running.get('state') not in ('experimenting','adopting'),'GROWTH_EXPERIMENT_BUSY')
    result=control.store.put('foundation_growth_policy',policy,project_id)
    control.store.put('foundation_growth_automation',{'projectId':project_id,'state':'ready',
        'activeJobId':None,'adoptionJobId':None,'reasonCodes':[]},project_id)
    return result


def advance(control,project_id):
    """No elapsed-time/parameter-count proxy can substitute for measured dev loss."""
    from datetime import datetime,timezone,timedelta
    from .resources import used_today
    store=control.store;policy=store.get('foundation_growth_policy',project_id,{})
    if not policy.get('enabled'):return {'state':'disabled','blocksDefault':False}
    project=control.foundation.project(project_id)
    def status(state,reasons=None,**fields):
        return store.put('foundation_growth_automation',{'projectId':project_id,'state':state,
            'reasonCodes':reasons or [],'blocksDefault':False,**fields},project_id)
    try:
        require(policy.get('approved') is True and not store.get('setting','deploymentMaintenance',{}).get('enabled'),
            'GROWTH_POLICY_DISABLED')
        require(project['policy'].get('enabled') is True,'AUTOMATION_DISABLED')
        require(used_today(store)<project['policy'].get('dailyActiveSeconds',14400),'DAILY_BUDGET_EXHAUSTED')
        state=store.get('foundation_growth_automation',project_id,{})
        if state.get('adoptionJobId'):
            job=store.job(state['adoptionJobId'])
            if job['state'] in ('failed','cancelled'):return status('held',['GROWTH_ADOPTION_FAILED'])
            if job['state']!='completed':return status('adopting',adoptionJobId=job['id'],blocksDefault=True)
            adopted=job['result']['project']
            if adopted.get('growthDepth',1)<policy['maxGrowthDepth']:
                inherited={k:v for k,v in policy.items() if k not in ('id','version','createdAt','updatedAt','policyHash')}
                inherited['projectId']=adopted['id'];inherited['policyHash']=digest(inherited)
                store.put('foundation_growth_policy',inherited,adopted['id'])
            return status('candidate_learning',adoptedProjectId=adopted['id'],adoptionJobId=None,
                activeJobId=None,lastSourceCheckpointId=state.get('lastSourceCheckpointId'))
        if state.get('activeJobId'):
            job=store.job(state['activeJobId'])
            if job['state'] in ('failed','cancelled'):return status('held',['GROWTH_EXPERIMENT_FAILED'])
            if job['state']!='completed':return status('experimenting',activeJobId=job['id'],blocksDefault=True)
            candidate=store.get('foundation_growth_candidate',job['result']['id'])
            if not policy['autoAdopt']:return status('candidate_ready',activeJobId=None,candidateId=candidate['id'])
            pending=control.foundation.adopt_growth(project_id,{'candidateId':candidate['id'],'autoTrain':policy['autoTrain']})
            return status('adopting',candidateId=candidate['id'],adoptionJobId=pending['id'],activeJobId=None,blocksDefault=True)
        if state.get('state')=='held':return state
        require(project.get('growthDepth',0)<policy['maxGrowthDepth'],'GROWTH_DEPTH_LIMIT')
        checkpoint_id=project.get('latestCheckpointId')
        require(checkpoint_id,'TRAINED_CHECKPOINT_REQUIRED')
        if state.get('lastSourceCheckpointId')==checkpoint_id:return status('waiting',['NEW_SOURCE_CHECKPOINT_REQUIRED'])
        require(any(row.get('projectId')==project_id and row.get('state')=='qualified'
            for row in store.list('foundation_graduation')),'FOUNDATION_QUALIFICATION_REQUIRED')
        evaluations=sorted((row for row in store.list('foundation_evaluation') if row.get('projectId')==project_id
            and row.get('kind') in ('qualification','lm_dev') and row.get('state')=='completed'
            and (project.get('testOnly') or not row.get('fixture'))),key=lambda row:row.get('createdAt',''),reverse=True)
        require(evaluations and evaluations[0].get('artifactId')==checkpoint_id,'GROWTH_CURRENT_DEV_MEASUREMENT_REQUIRED')
        comparable=[row for row in evaluations if row.get('corpusHash')==evaluations[0].get('corpusHash')
            and row.get('tokenizerHash')==evaluations[0].get('tokenizerHash')]
        seen=set();window=[]
        for row in comparable:
            if row['artifactId'] in seen:continue
            seen.add(row['artifactId']);window.append(row)
            if len(window)==policy['plateauWindow']:break
        require(len(window)==policy['plateauWindow'],'GROWTH_DEV_HISTORY_REQUIRED')
        losses=[row.get('nll',{}).get('candidate') for row in window]
        import math
        require(all(isinstance(value,(int,float)) and math.isfinite(value) and value>0 for value in losses),'GROWTH_DEV_HISTORY_REQUIRED')
        improvement=(losses[-1]-losses[0])/losses[-1]
        require(improvement<policy['minRelativeImprovement'],'GROWTH_DEV_STILL_IMPROVING')
        today=datetime.now(timezone(timedelta(hours=9))).date()
        attempted=[row for row in store.list('foundation_growth_candidate') if row.get('sourceProjectId')==project_id
            and datetime.fromisoformat(row['createdAt'].replace('Z','+00:00')).astimezone(timezone(timedelta(hours=9))).date()==today]
        require(len(attempted)<1,'GROWTH_DAILY_EXPERIMENT_LIMIT')
        recipe={'operation':policy['operation']}
        if recipe['operation']=='ffn_expand':recipe['intermediateSize']=project['config']['intermediate_size']+policy['ffnIncrement']
        else:recipe['position']=project['config']['num_hidden_layers']
        job=control.foundation.start_growth(project_id,{'parentCheckpointId':checkpoint_id,'recipe':recipe},
            key='growth-auto:'+policy['policyHash']+':'+checkpoint_id)
        return status('experimenting',activeJobId=job['id'],lastSourceCheckpointId=checkpoint_id,
            plateauEvidenceIds=[row['id'] for row in window],measuredRelativeImprovement=improvement,blocksDefault=True)
    except Exception as exc:
        if not isinstance(exc,(ValueError,KeyError,TypeError)) and not hasattr(exc,'code'):raise
        return status('waiting',[getattr(exc,'code','GROWTH_METADATA_INVALID')])


def transform(model, config, body):
    """Return a real FP32 HF model and exact mapping recipe, without I/O/network."""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = architecture_config(config, test_only=True)
    operation = body.get('operation')
    require(operation in ('ffn_expand', 'insert_identity_layer'), 'UNSUPPORTED_GROWTH_OPERATION')
    target = dict(cfg)
    if operation == 'ffn_expand':
        width = integer(body.get('intermediateSize'), 'intermediateSize', cfg['intermediate_size']+1,
                        cfg['intermediate_size']*2)
        require(width % 8 == 0, 'GROWTH_DIMENSION_ALIGNMENT')
        target['intermediate_size'] = width
        position = None
    else:
        position = integer(body.get('position', cfg['num_hidden_layers']), 'position', 0, cfg['num_hidden_layers'])
        target['num_hidden_layers'] += 1
    require(parameter_count(target) <= min(32_000_000,2*parameter_count(cfg)), 'GROWTH_RESOURCE_BOUND_EXCEEDED')
    seed = integer(body.get('seed', 42), 'seed', 0, 2**32-1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        native = LlamaConfig(**target); native._attn_implementation = 'eager'
        grown = LlamaForCausalLM(native).to(device='cpu', dtype=torch.float32)
    source, destination = model.state_dict(), grown.state_dict()
    with torch.no_grad():
        for name, value in destination.items():
            source_name = name
            if operation == 'insert_identity_layer' and name.startswith('model.layers.'):
                layer = int(name.split('.')[2])
                if layer == position:
                    if name.endswith(('self_attn.o_proj.weight', 'mlp.down_proj.weight')): value.zero_()
                    continue
                old_layer = layer if layer < position else layer-1
                source_name = name.replace(f'model.layers.{layer}.', f'model.layers.{old_layer}.', 1)
            if operation == 'ffn_expand' and '.mlp.' in name:
                if name.endswith(('gate_proj.weight','up_proj.weight')):
                    value[:cfg['intermediate_size']].copy_(source[name])
                elif name.endswith('down_proj.weight'):
                    value.zero_(); value[:, :cfg['intermediate_size']].copy_(source[name])
                else: require(False, 'UNSUPPORTED_GROWTH_TENSOR')
            else:
                require(source_name in source and value.shape == source[source_name].shape, 'GROWTH_MAPPING_MISMATCH')
                value.copy_(source[source_name])
    grown.tie_weights(); grown.config.use_cache = False; grown.eval()
    require(sum(p.numel() for p in grown.parameters()) == parameter_count(target), 'MODEL_PARAMETER_COUNT_MISMATCH')
    require(all(bool(torch.isfinite(p).all()) for p in grown.parameters()), 'MODEL_NONFINITE')
    return grown, target, {'operation':operation, 'position':position, 'seed':seed,
        'sourceConfigHash':digest(cfg), 'targetConfigHash':digest(target),
        'sourceParameterCount':parameter_count(cfg), 'targetParameterCount':parameter_count(target),
        'mappingVersion':'seed-function-preserving-growth-v1'}


def compare_logits(original, grown, config, *, maximum_context=None):
    """Measured finite-probe evidence; does not claim model-quality improvement."""
    import torch
    context = min(config['max_position_embeddings'], maximum_context or config['max_position_embeddings'])
    lengths = sorted({min(context, value) for value in (8, 16, 32, 64)})
    maximum_error = 0.0
    original.eval(); grown.eval()
    with torch.inference_mode():
        for length in lengths:
            ids = torch.tensor([[1]+[12+(i*17)%(config['vocab_size']-12) for i in range(length-1)]])
            left, right = original(input_ids=ids, use_cache=False).logits, grown(input_ids=ids, use_cache=False).logits
            require(bool(torch.isfinite(left).all()) and bool(torch.isfinite(right).all()), 'MODEL_NONFINITE')
            maximum_error = max(maximum_error, float((left-right).abs().max()))
            require(torch.allclose(left, right, atol=1e-5, rtol=1e-4), 'GROWTH_FUNCTION_PRESERVATION_FAILED')
    return {'passed':True, 'probeLengths':lengths, 'maxAbsoluteLogitError':maximum_error,
        'absoluteTolerance':1e-5, 'relativeTolerance':1e-4, 'qualityImprovementMeasured':False,
        'scope':'finite deterministic FP32 probes; algebraic zero-residual/zero-column mapping'}


def experiment(store, project_id, parent_checkpoint_id, body, job=None):
    from .architecture import load_model, model_digest
    from .checkpoint import verify
    from .tokenizer import load
    from .resources import admit
    from .retention import admit as quota_admit
    project = store.get('foundation_project', project_id)
    parent = store.get('foundation_checkpoint', parent_checkpoint_id, {}) or store.get('foundation_genesis', parent_checkpoint_id)
    require(parent['projectId'] == project_id and parent.get('state') == 'ready', 'GROWTH_PARENT_MISMATCH')
    verify(store, parent)
    if not project.get('testOnly'):
        project['policy'] = store.get('foundation_policy', project_id)
        admit(store, project)
    quota_admit(store, project, parent['parameterCount']*12+1024**2)
    if job is not None: store.assert_lease(job)
    original = load_model(store, parent)
    before = model_digest(original)
    grown, config, recipe = transform(original, parent['config'], body)
    evidence = compare_logits(original, grown, parent['config'], maximum_context=parent.get('trainedContextTokens') or None)
    require(model_digest(original) == before, 'GROWTH_PARENT_MUTATED')
    token = store.get('foundation_tokenizer', parent['tokenizerId']); load(store, token)
    identity = digest([parent['manifestHash'], recipe, model_digest(grown)])
    family_id, genesis_id = 'growth-family-'+identity[:24], 'growth-genesis-'+identity[:24]
    stage = inside(store.root, 'foundation/staging/growth-'+uid()); stage.mkdir(parents=True)
    grown.save_pretrained(stage, safe_serialization=True)
    token_root = inside(store.root, token['storagePath'])
    token_manifest = json.loads((token_root/'manifest.json').read_text('utf-8'))
    for name in token_manifest['files']:
        path = token_root/name
        require(not path.is_symlink(), 'ARTIFACT_PATH_INVALID')
        if path.is_file() and path.name != 'manifest.json': shutil.copyfile(path, stage/path.name)
    import transformers
    provenance = {'origin':'architecture_growth', 'sourceProjectId':project_id, 'sourceCheckpointId':parent['id'],
        'sourceManifestHash':parent['manifestHash'], 'sourceGenesisId':project.get('genesisId'),
        'familyId':family_id, 'genesisId':genesis_id, 'tokenizerId':token['id'], 'tokenizerHash':token['tokenizerHash'],
        'config':config, 'architectureHashV2':digest([config,transformers.__version__]), 'recipe':recipe, 'recipeHash':digest(recipe),
        'evidence':evidence, 'parentTensorDigest':before, 'candidateTensorDigest':model_digest(grown),
        'trainedContextTokens':parent.get('trainedContextTokens',0), 'pretrainedWeightArtifactIds':[parent['id']],
        'servingEligible':False, 'qualityGateRequired':True, 'testOnly':project.get('testOnly') is True}
    receipt = {'schemaVersion':1, 'kind':'verified_growth_recipe', 'candidateId':identity,
        'sourceConfig':parent['config'], 'targetConfig':config, 'recipe':recipe, 'recipeHash':digest(recipe),
        'evidenceHash':digest(evidence), 'parentTensorDigest':before, 'candidateTensorDigest':model_digest(grown),
        'tokenizerHash':token['tokenizerHash'], 'sourceManifestHash':parent['manifestHash']}
    receipt['receiptHash'] = digest(receipt)
    provenance['trustedGrowthReceipt'] = receipt
    atomic_json(stage/'growth-provenance.json', provenance)
    manifest = {'schemaVersion':1, 'id':identity, 'kind':'foundation_growth_candidate', **provenance,
        'files':{p.name:file_hash(p) for p in stage.iterdir() if p.is_file()}}
    atomic_json(stage/'manifest.json', manifest)
    destination = inside(store.root, 'foundation/growth/'+identity); destination.parent.mkdir(parents=True, exist_ok=True)
    with store.tx():
        if job is not None: store.assert_lease(job)
        current_project=store.get('foundation_project', project_id)
        require(current_project.get('state') != 'revoked' and not current_project.get('revocationPending'), 'POLICY_REVOKED')
        if destination.exists():
            require(file_hash(destination/'manifest.json') == file_hash(stage/'manifest.json'), 'GROWTH_IMMUTABLE_CONFLICT')
            shutil.rmtree(stage)
        else: stage.replace(destination)
        record = store.put('foundation_growth_candidate', {**provenance, 'projectId':project_id, 'state':'candidate',
            'storagePath':destination.relative_to(store.root).as_posix(), 'manifestHash':file_hash(destination/'manifest.json'),
            'parameterCount':parameter_count(config), 'createdAt':now()}, identity)
        store.put('foundation_growth_family', {'projectId':project_id, 'origin':'architecture_growth',
            'candidateArtifactId':identity, 'genesisId':genesis_id, 'sourceGenesisId':project.get('genesisId'),
            'state':'candidate', 'servingEligible':False, 'qualityGateRequired':True}, family_id)
    return record


def adopt(control,candidate_id,body=None):
    """Reproduce a stored transformation, then create a separate unqualified line."""
    import torch
    from transformers import LlamaForCausalLM
    from . import architecture, checkpoint
    from .tokenizer import load
    from .schemas import valid_growth_config
    from .resources import DEFAULT_POLICY
    from .transfer import _safe
    body = body or {};store = control.store
    require(not set(body)-{'name','autoTrain'},'CONFIG_INVALID')
    candidate = store.get('foundation_growth_candidate',candidate_id)
    require(candidate.get('state') in ('candidate','adopted') and candidate.get('origin') == 'architecture_growth'
        and candidate.get('evidence',{}).get('passed') is True,'GROWTH_CANDIDATE_NOT_VERIFIED')
    source_project = store.get('foundation_project',candidate['sourceProjectId'])
    require(source_project.get('state') != 'revoked' and source_project.get('stage') != 'revoked'
        and not source_project.get('revocationPending'),'POLICY_REVOKED')
    require(source_project.get('testOnly') or source_project.get('executionScope','nas') == control.execution_scope,
        'EXECUTION_SCOPE_REQUALIFICATION_REQUIRED')
    source = store.get('foundation_checkpoint',candidate['sourceCheckpointId'],{}) or store.get('foundation_genesis',candidate['sourceCheckpointId'])
    checkpoint.verify(store,source)
    require(source['manifestHash'] == candidate['sourceManifestHash'] and source.get('state') == 'ready','GROWTH_SOURCE_CHANGED')
    root = _safe(store.root/candidate['storagePath'])
    require(root == store.root/'foundation'/'growth'/candidate_id and file_hash(root/'manifest.json') == candidate['manifestHash'],
        'GROWTH_MANIFEST_MISMATCH')
    manifest = json.loads((root/'manifest.json').read_text('utf-8'))
    for name,checksum in manifest['files'].items():
        path = _safe(root/name)
        require(path.is_relative_to(root) and path.is_file() and file_hash(path) == checksum,'GROWTH_MANIFEST_MISMATCH')
    for key in ('sourceCheckpointId','sourceManifestHash','sourceProjectId','sourceGenesisId','config','recipe','recipeHash',
        'tokenizerId','tokenizerHash','candidateTensorDigest','parentTensorDigest','trustedGrowthReceipt','evidence'):
        require(candidate.get(key) == manifest.get(key),'GROWTH_MANIFEST_MISMATCH',field=key)
    receipt = candidate['trustedGrowthReceipt']
    require(receipt.get('receiptHash') == digest({k:v for k,v in receipt.items() if k != 'receiptHash'})
        and receipt.get('targetConfig') == candidate['config'] and receipt.get('sourceConfig') == source['config']
        and receipt.get('evidenceHash') == digest(candidate['evidence']), 'GROWTH_RECEIPT_REQUIRED')
    require(source_project.get('testOnly') or valid_growth_config(candidate['config'],receipt),'GROWTH_RECEIPT_REQUIRED')
    project_id = 'growth-project-'+candidate_id[:24]
    old = store.get('foundation_project',project_id,{})
    if old:
        require(old.get('acceptedArchitectureRecipeId') == candidate_id and old.get('origin') == 'architecture_growth',
            'GROWTH_ADOPTION_CONFLICT')
        if old.get('genesisId') and old.get('randomBaselineGenesisId'):
            return {'project':old,'genesis':store.get('foundation_genesis',old['genesisId']),
                'randomBaselineGenesis':store.get('foundation_genesis',old['randomBaselineGenesisId']),
                'candidateId':candidate_id,'servingEligible':False,'qualityGateRequired':True,'priorCertificatesInherited':False}
    original = architecture.load_model(store,source)
    require(architecture.model_digest(original) == candidate['parentTensorDigest'],'GROWTH_SOURCE_CHANGED')
    recipe_body = {'operation':candidate['recipe']['operation'],'seed':candidate['recipe']['seed'],
        'position':candidate['recipe'].get('position'),'intermediateSize':candidate['config']['intermediate_size']}
    grown,cfg,reproduced = transform(original,source['config'],recipe_body)
    require(reproduced == candidate['recipe'] and architecture.model_digest(grown) == candidate['candidateTensorDigest'],
        'GROWTH_REPRODUCTION_FAILED')
    compare_logits(original,grown,source['config'],maximum_context=source.get('trainedContextTokens') or None)
    exported = LlamaForCausalLM.from_pretrained(root,local_files_only=True,trust_remote_code=False,
        dtype=torch.float32,attn_implementation='eager')
    require(architecture.model_digest(exported) == candidate['candidateTensorDigest'],'GROWTH_REPRODUCTION_FAILED')
    del exported, original
    token = store.get('foundation_tokenizer',candidate['tokenizerId']);load(store,token)
    token_id = 'growth-tokenizer-'+digest([project_id,token['id'],token['tokenizerHash']])[:24]
    token_destination = _safe(store.root/'foundation'/'tokenizers'/token_id)
    token_stage = _safe(store.root/'foundation'/'staging'/('adopt-tokenizer-'+uid()));token_stage.mkdir(parents=True)
    token_manifest = json.loads((store.root/token['storagePath']/'manifest.json').read_text('utf-8'))
    for name in {*token_manifest['files'],'manifest.json'}:
        source_file = _safe(store.root/token['storagePath']/name)
        require(source_file.is_relative_to(store.root/token['storagePath']), 'ARTIFACT_PATH_INVALID')
        shutil.copyfile(source_file,token_stage/name)
    token_destination.parent.mkdir(parents=True,exist_ok=True)
    if token_destination.exists():
        require(file_hash(token_destination/'manifest.json') == file_hash(token_stage/'manifest.json'),'GROWTH_ADOPTION_CONFLICT')
        shutil.rmtree(token_stage)
    else:token_stage.replace(token_destination)
    source_policy = store.get('foundation_policy',source_project['id'],{})
    name = body.get('name',source_project.get('name','SEED')+' grown')
    require(isinstance(name,str) and 0 < len(name.strip()) <= 80,'CONFIG_INVALID')
    with store.tx():
        latest_source=store.get('foundation_project',source_project['id'])
        require(latest_source.get('state')!='revoked' and not latest_source.get('revocationPending'),'POLICY_REVOKED')
        project = store.put('foundation_project',{'name':name.strip(),'origin':'architecture_growth','state':'active',
            'stage':'adopting_growth','architecturePreset':'verified-growth-v1','config':cfg,'configHash':digest(cfg),
            'architectureHashV2':candidate['architectureHashV2'],'parameterCount':parameter_count(cfg),
            'trustedGrowthReceipt':receipt,'acceptedArchitectureRecipeId':candidate_id,
            'sourceProjectId':source_project['id'],'sourceCheckpointId':source['id'],'sourceGenesisId':candidate['sourceGenesisId'],
            'growthDepth':source_project.get('growthDepth',0)+1,
            'pretrainedWeightArtifactIds':[source['id']],'executionScope':source_project.get('executionScope',control.execution_scope),
            'plannedFamilyId':candidate['familyId'],'testOnly':source_project.get('testOnly') is True,
            'seed':candidate['recipe']['seed'],'tokenizerId':token_id,'genesisId':None,'latestCheckpointId':None,
            'activeCorpusId':None,'activeSftCorpusId':None,'servingEligible':False,'capabilityFlags':{},'reasonCodes':[]},project_id)
        policy = {key:source_policy.get(key,value) for key,value in DEFAULT_POLICY.items()}
        if project['testOnly']:policy.update(minCorpusBytes=1,minTrainGroups=1)
        policy['enabled'] = body.get('autoTrain') is True
        store.put('foundation_policy',policy,project_id)
        new_token = store.put('foundation_tokenizer',{**{k:v for k,v in token.items() if k not in ('id','version','createdAt','updatedAt')},
            'projectId':project_id,'sourceTokenizerId':token['id'],'storagePath':token_destination.relative_to(store.root).as_posix()},token_id)
    state = {'projectId':project_id,'origin':'architecture_growth','trustedGrowthReceipt':receipt,
        'sourceProjectId':source_project['id'],'sourceCheckpointId':source['id'],'sourceGenesisId':candidate['sourceGenesisId'],
        'executionScope':project['executionScope'],'pretrainedWeightArtifactIds':[source['id']],
        'seed':project['seed'],'completedOptimizerSteps':0,'committedTargetTokens':0,'attemptedTargetTokens':0,
        'trainedContextTokens':source.get('trainedContextTokens',0),'cursor':{'epoch':0,'block':0},
        'testOnly':project['testOnly'],'servingEligible':False,'initialTensorDigest':architecture.model_digest(grown)}
    for key in ('teacherArtifactId','teacherArtifactIds','teacherManifestHash','teacherManifestHashes','teacherApprovalIds','distillationMode'):
        if key in source:state[key]=source[key]
    genesis = next((record for record in store.list('foundation_genesis') if record.get('projectId') == project_id
        and not record.get('randomBaselineOnly') and record.get('initialTensorDigest') == state['initialTensorDigest']
        and record.get('trustedGrowthReceipt') == receipt),None)
    if genesis: checkpoint.verify(store,genesis)
    else: genesis = checkpoint.save(store,grown,cfg,new_token,state,record_kind='foundation_genesis')
    del grown
    # A derived initialization is not a random baseline. Keep a separate fresh
    # control model for qualification under exactly the enlarged topology.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(project['seed'])
        random_model = architecture.create_model(cfg,test_only=project['testOnly'],growth_receipt=receipt)
    random_state = {**state,'origin':'random_init','pretrainedWeightArtifactIds':[],
        'sourceProjectId':None,'sourceCheckpointId':None,'sourceGenesisId':None,'trainedContextTokens':0,
        'teacherArtifactId':None,'teacherArtifactIds':[],'teacherManifestHash':None,'teacherManifestHashes':{},
        'teacherApprovalIds':[],'distillationMode':None,
        'randomBaselineOnly':True,'initialTensorDigest':architecture.model_digest(random_model)}
    random_baseline = next((record for record in store.list('foundation_genesis') if record.get('projectId') == project_id
        and record.get('randomBaselineOnly') and record.get('initialTensorDigest') == random_state['initialTensorDigest']
        and record.get('trustedGrowthReceipt') == receipt),None)
    if random_baseline: checkpoint.verify(store,random_baseline)
    else: random_baseline = checkpoint.save(store,random_model,cfg,new_token,random_state,record_kind='foundation_genesis')
    with store.tx():
        latest_source=store.get('foundation_project',source_project['id'])
        target_project=store.get('foundation_project',project_id)
        require(latest_source.get('state') != 'revoked' and not latest_source.get('revocationPending')
            and target_project.get('state')!='revoked' and not target_project.get('revocationPending'),'POLICY_REVOKED')
        project = store.put('foundation_project',{'genesisId':genesis['id'],'randomBaselineGenesisId':random_baseline['id'],
            'stage':'waiting_corpus','servingEligible':False},project_id)
        store.put('foundation_growth_candidate',{'state':'adopted','adoptedProjectId':project_id,
            'adoptedGenesisId':genesis['id'],'servingEligible':False},candidate_id)
        store.put('foundation_growth_family',{'state':'awaiting_qualification','projectId':project_id,
            'genesisId':genesis['id'],'randomBaselineGenesisId':random_baseline['id'],'servingEligible':False},candidate['familyId'])
    return {'project':project,'genesis':genesis,'randomBaselineGenesis':random_baseline,'candidateId':candidate_id,
        'servingEligible':False,'qualityGateRequired':True,'priorCertificatesInherited':False}
