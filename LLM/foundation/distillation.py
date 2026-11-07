"""Actual offline logit distillation from explicitly pinned local teachers."""
from __future__ import annotations

import time
from LLM.core import require, digest, now
from .schemas import integer, real


def objective(student_logits, teacher_logits, labels, *, temperature=2., alpha=.5):
    import torch
    import torch.nn.functional as F
    require(student_logits.shape == teacher_logits.shape and labels.shape == student_logits.shape[:2],
            'DISTILLATION_LOGIT_SHAPE_MISMATCH')
    mask = labels[:,1:] != -100
    require(bool(mask.any()), 'NO_TRAINABLE_EXAMPLES')
    left, right = student_logits[:,:-1][mask].float(), teacher_logits[:,:-1][mask].float()
    target = labels[:,1:][mask]
    ce = F.cross_entropy(left,target,reduction='mean')
    soft = F.kl_div(F.log_softmax(left/temperature,dim=-1),F.softmax(right/temperature,dim=-1),reduction='batchmean')
    loss = alpha*ce+(1-alpha)*temperature**2*soft
    require(bool(torch.isfinite(loss)), 'DISTILLATION_NONFINITE')
    return loss, {'crossEntropy':float(ce.detach()),'klDivergence':float(soft.detach()),
        'temperature':temperature,'alpha':alpha,'targetCount':int(mask.sum())}


def _teacher(control, identity):
    store = control.store
    record = store.get('foundation_checkpoint',identity,{}) or store.get('artifact',identity)
    require(not record.get('comparisonOnly'), 'COMPARISON_ONLY_MODEL', status=409)
    if store.get('foundation_checkpoint',identity,{}):
        from .checkpoint import verify
        verify(store,record)
    else: control.registry.verify(record)
    require(record.get('state') in ('ready','verified','completed') and record.get('format','safetensors') == 'safetensors',
            'DISTILLATION_TEACHER_NOT_READY')
    return record


def approve_teacher(control, project_id, body):
    """Server-called approval binds a registered artifact to exact immutable bytes."""
    teacher = _teacher(control,body['teacherArtifactId'])
    require(body.get('teacherManifestHash') == teacher['manifestHash'] and body.get('approved') is True,
            'PINNED_TEACHER_APPROVAL_REQUIRED')
    project = control.store.get('foundation_project',project_id)
    require(project.get('testOnly') or not (teacher.get('fixture') or teacher.get('testOnly')), 'FIXTURE_NOT_FOR_PRODUCTION')
    mode = body.get('mode','same_tokenizer_logits')
    require(mode in ('same_tokenizer_logits','verified_response_sft'), 'DISTILLATION_MODE_INVALID')
    return control.store.put('foundation_teacher_approval', {'projectId':project_id,'teacherArtifactId':teacher['id'],
        'teacherManifestHash':teacher['manifestHash'],'sourceRevision':teacher['manifestHash'],
        'declaredSourceRevision':teacher.get('sourceRevision'),
        'approved':True,'allowedMode':mode,'automaticDownload':False},
        digest([project_id,teacher['id'],teacher['manifestHash']]
            + ([mode] if mode != 'same_tokenizer_logits' else [])))


def run(control, job):
    import torch
    from . import architecture, checkpoint, packing, tokenizer
    from .resources import admit
    from .retention import admit as quota_admit
    store, body = control.store, job['payload']
    project = store.get('foundation_project',body['projectId'])
    project['policy'] = store.get('foundation_policy',project['id'],{})
    parent = store.get('foundation_checkpoint',body['parentCheckpointId'],{}) or store.get('foundation_genesis',body['parentCheckpointId'])
    require(not parent.get('randomBaselineOnly'),'RANDOM_BASELINE_NOT_TRAINING_PARENT')
    teacher = _teacher(control,body['teacherArtifactId'])
    require(body.get('teacherManifestHash') == teacher['manifestHash'], 'TEACHER_REVISION_MISMATCH')
    approval_id = digest([project['id'],teacher['id'],teacher['manifestHash']])
    approval = store.get('foundation_teacher_approval',approval_id,{})
    require(approval.get('approved') is True and approval.get('teacherManifestHash') == teacher['manifestHash']
            and approval.get('allowedMode') == 'same_tokenizer_logits',
            'PINNED_TEACHER_APPROVAL_REQUIRED')
    token = store.get('foundation_tokenizer',project['tokenizerId'])
    require(parent.get('projectId') == project['id'] and parent['tokenizerHash'] == token['tokenizerHash'], 'CHECKPOINT_INCOMPATIBLE')
    require(teacher.get('tokenizerHash') == token['tokenizerHash'], 'RESPONSE_SFT_VERIFIED_DATA_REQUIRED')
    require(teacher.get('trainedContextTokens',0) >= 8, 'TEACHER_TRAINED_CONTEXT_REQUIRED')
    from .corpus import resolve
    corpus = resolve(store,body['corpusId'])
    require(corpus.get('projectId') == project['id'] and corpus.get('state') == 'ready', 'CORPUS_NOT_READY')
    require(corpus.get('kind','foundation_corpus') == 'foundation_corpus','LOGITS_DISTILLATION_RAW_CORPUS_REQUIRED')
    cfg = body.get('config',{})
    require(not set(cfg)-{'sequenceLength','maxSteps','maxActiveSeconds','temperature','alpha','learningRate','cpuThreads'},'CONFIG_INVALID')
    threads=integer(cfg.get('cpuThreads',1),'cpuThreads',1,4)
    require(project.get('executionScope','nas')=='pc_validation' or threads==1,'NAS_CPU_THREADS_FIXED')
    torch.set_num_threads(threads)
    length = integer(cfg.get('sequenceLength',128),'sequenceLength',8,min(parent['config']['max_position_embeddings'],teacher['trainedContextTokens']))
    steps = integer(cfg.get('maxSteps',5),'maxSteps',1,20)
    seconds = integer(cfg.get('maxActiveSeconds',900),'maxActiveSeconds',1,min(900,project['policy'].get('dailyActiveSeconds',14400)))
    temperature = real(cfg.get('temperature',2.),'temperature',.1,10,positive=True)
    alpha = real(cfg.get('alpha',.5),'alpha',0,1)
    learning_rate = real(cfg.get('learningRate',5e-5),'learningRate',1e-8,1e-3,positive=True)
    recipe = {'sequenceLength':length,'maxSteps':steps,'maxActiveSeconds':seconds,'temperature':temperature,'alpha':alpha,'learningRate':learning_rate}
    if 'cpuThreads' in cfg:recipe['cpuThreads']=threads
    recipe_hash = digest([recipe,parent['manifestHash'],teacher['manifestHash'],corpus['manifestHash'],token['tokenizerHash']])
    estimated = parent['parameterCount']*16+teacher['parameterCount']*4+512*1024**2
    require(estimated <= 6*1024**3, 'DISTILLATION_MEMORY_BUDGET_EXCEEDED')
    if not project.get('testOnly'):
        status = admit(store,project)
        require(estimated < status['availableForJobBytes'], 'RESOURCE_PRESSURE')
        from .backend import verify
        verify(project.get('backendReceipt'),strict_platform=control.execution_scope=='nas')
    quota_admit(store,project,parent['parameterCount']*20+1024**2)
    prior = store.get('foundation_run',job['id'],{})
    if prior.get('state') == 'completed' and prior.get('lastCheckpointId'):
        return {'state':'completed','checkpointId':prior['lastCheckpointId'],'runId':job['id']}
    selected = store.get('foundation_checkpoint',prior['lastCheckpointId']) if prior.get('lastCheckpointId') else parent
    student = architecture.load_model(store,selected); student.train()
    teacher_model, _ = control.foundation.runtime.load(teacher['id'],purpose='evaluation')
    teacher_model.eval(); teacher_model.requires_grad_(False)
    optimizer = torch.optim.AdamW(student.parameters(),lr=learning_rate,weight_decay=.1)
    state = {'projectId':project['id'],'runId':job['id'],'origin':parent.get('origin','random_init'),
        'trustedGrowthReceipt':project.get('trustedGrowthReceipt'),'executionScope':project.get('executionScope','nas'),
        'pretrainedWeightArtifactIds':parent.get('pretrainedWeightArtifactIds',[]),
        'parentCheckpointId':parent['id'],'corpusId':corpus['id'],'corpusHash':corpus['manifestHash'],
        'teacherArtifactId':teacher['id'],'teacherManifestHash':teacher['manifestHash'],'approvalId':approval_id,
        'teacherArtifactIds':[teacher['id']],'teacherManifestHashes':{teacher['id']:teacher['manifestHash']},
        'teacherApprovalIds':[approval_id],'distillationMode':'same_tokenizer_logits',
        'configHash':digest(parent['config']),'distillationRecipeHash':recipe_hash,'stage':'distillation',
        'completedOptimizerSteps':0,'committedTargetTokens':0,'attemptedTargetTokens':0,
        'trainedContextTokens':parent.get('trainedContextTokens',0),'targetNamespace':'distillation_logits',
        'cursor':{'epoch':0,'block':0},'consumedTargetRanges':{},'testOnly':project.get('testOnly') is True,
        'servingEligible':False,'qualityGateRequired':True}
    if prior.get('lastCheckpointId'):
        state = checkpoint.restore(store,selected,student,optimizer)
        require(state.get('distillationRecipeHash') == recipe_hash, 'DISTILLATION_RESUME_MISMATCH')
    inherited_teachers=dict(parent.get('teacherManifestHashes') or {})
    if parent.get('teacherArtifactId'):inherited_teachers[parent['teacherArtifactId']]=parent.get('teacherManifestHash')
    inherited_teachers.update(state.get('teacherManifestHashes') or {})
    state['teacherManifestHashes']=inherited_teachers;state['teacherArtifactIds']=sorted(inherited_teachers)
    state['teacherApprovalIds']=sorted(set(parent.get('teacherApprovalIds') or [])|set(state.get('teacherApprovalIds') or []))
    store.put('foundation_run',{**state,'state':'running','trainingMode':'full_parameter_logits_distillation'},job['id'])
    tok = tokenizer.load(store,token)
    started = time.monotonic()
    outcome, metrics = 'completed', {}
    try:
        for block in packing.iter_blocks(store,corpus,tok,length,shuffle=False,skip_blocks=state['cursor']['block'],purpose='distillation_train'):
            current = store.assert_lease(job)
            if current['state'] in ('cancelling','cancelled'): outcome='cancelled'; break
            if current['state'] == 'pause_requested' or time.monotonic()-started >= seconds:
                outcome='paused'; break
            require(store.get('foundation_teacher_approval',approval_id).get('approved') is True
                    and resolve(store,corpus['id']).get('state') == 'ready'
                    and store.get('foundation_project',project['id']).get('state') != 'revoked', 'POLICY_REVOKED')
            _teacher(control,teacher['id'])
            batch = packing.tensor_batch(torch,[block]); optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(): teacher_logits = teacher_model(**{k:v for k,v in batch.items() if k != 'labels'}).logits
            student_logits = student(**{k:v for k,v in batch.items() if k != 'labels'}).logits
            loss, metrics = objective(student_logits,teacher_logits,batch['labels'],temperature=temperature,alpha=alpha)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(student.parameters(),1.)
            require(bool(torch.isfinite(norm)) and all(bool(torch.isfinite(p.grad).all()) for p in student.parameters() if p.grad is not None), 'DISTILLATION_NONFINITE')
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            require(all(bool(torch.isfinite(p).all()) for p in student.parameters()), 'DISTILLATION_NONFINITE')
            state['completedOptimizerSteps'] += 1; state['committedTargetTokens'] += block['targetCount']
            state['attemptedTargetTokens'] = state['committedTargetTokens']; state['cursor']['block'] += 1
            state['trainedContextTokens'] = max(state['trainedContextTokens'],len(block['inputIds']))
            metrics['loss'] = float(loss.detach()); state['lastMetrics'] = metrics
            saved = checkpoint.save(store,student,parent['config'],token,state,optimizer,job=job)
            store.put('foundation_run',{'lastCheckpointId':saved['id'],'checkpointId':saved['id'],**state},job['id'])
            store.progress(job,'distilling',state['completedOptimizerSteps'],steps,'optimizer_steps',targetTokens=state['committedTargetTokens'])
            if state['completedOptimizerSteps'] >= steps: break
        if state['completedOptimizerSteps'] == 0 and outcome in ('paused','cancelled'):
            saved = checkpoint.save(store,student,parent['config'],token,state,optimizer,job=job)
            store.put('foundation_run',{'lastCheckpointId':saved['id'],'checkpointId':saved['id'],**state},job['id'])
        else:
            require(state['completedOptimizerSteps'] > 0, 'NO_TRAINABLE_EXAMPLES')
    finally:
        control.foundation.runtime.expire(force=True)
    latest = store.get('foundation_run',job['id'])['lastCheckpointId']
    store.put('foundation_run',{'state':outcome,'lastCheckpointId':latest,'completedOptimizerSteps':state['completedOptimizerSteps'],
        'committedTargetTokens':state['committedTargetTokens'],'finishedAt':now() if outcome == 'completed' else None},job['id'])
    return {'state':outcome,'checkpointId':latest,'runId':job['id'],'metrics':metrics,'qualityGateRequired':True,
        'servingEligible':False,'teacherManifestHash':teacher['manifestHash']}
