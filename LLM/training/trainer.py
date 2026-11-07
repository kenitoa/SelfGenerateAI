"""Local-only, standard LoRA SFT with real optimizer telemetry and checkpoints."""
import json
import math
import random
from pathlib import Path
from LLM.core import require, digest, file_hash, atomic_json, now, uid

DEFAULTS={'rank':8,'loraAlpha':16,'dropout':.05,'epochs':1,'batchSize':1,'gradientAccumulation':8,
          'learningRate':.0002,'maxSequence':512,'seed':42,'maxSteps':0,'checkpointEvery':25,
          'targetModules':['q_proj','v_proj'],'device':'cpu','gradientClipping':1.0,'earlyStoppingPatience':2,'minValidationDelta':.0001,'objective':'sft','dpoBeta':.1}

def encode_example(tokenizer, row, maximum):
    if hasattr(tokenizer,'encode_ordinary') and hasattr(tokenizer,'assemble'):
        start=tokenizer.assemble([{'role':'user','content':row['instruction']}],assistant_prefix=True)
        target=[*tokenizer.encode_ordinary(row['output']),7]
        require(target and len(start)+len(target)<=maximum,'INPUT_TOO_LARGE_FOR_TRAINING')
        return {'input_ids':start+target,'labels':[-100]*len(start)+target}
    prefix='<|im_start|>user\n'+row['instruction']+'<|im_end|>\n<|im_start|>assistant\n'
    start=tokenizer.encode(prefix,add_special_tokens=False)
    target=tokenizer.encode(row['output']+'<|im_end|>',add_special_tokens=False)
    require(target and len(start)+len(target)<=maximum,'INPUT_TOO_LARGE_FOR_TRAINING')
    return {'input_ids':start+target,'labels':[-100]*len(start)+target}

class Trainer:
    def __init__(self, store, registry, snapshots):self.store,self.registry,self.snapshots=store,registry,snapshots

    def config(self, body):
        result={**DEFAULTS,**{k:v for k,v in body.items() if k in DEFAULTS}}
        for k,lo,hi in [('rank',1,64),('epochs',1,100),('batchSize',1,8),('gradientAccumulation',1,128),('maxSequence',32,8192),('checkpointEvery',1,10000)]:
            require(isinstance(result[k],int) and not isinstance(result[k],bool) and lo<=result[k]<=hi,'INVALID_TRAINING_CONFIG',field=k)
        require(isinstance(result['learningRate'],(int,float)) and math.isfinite(result['learningRate']) and 0<result['learningRate']<=.01,'INVALID_TRAINING_CONFIG')
        require(result['device'] in ('cpu','cuda'),'INVALID_TRAINING_DEVICE')
        require(result['objective'] in ('sft','dpo','cpt'),'INVALID_TRAINING_OBJECTIVE')
        require(isinstance(result['dpoBeta'],(int,float)) and not isinstance(result['dpoBeta'],bool) and math.isfinite(result['dpoBeta']) and 0<result['dpoBeta']<=1,'INVALID_DPO_BETA')
        for k,lo,hi in [('dropout',0,1),('loraAlpha',.001,4096),('gradientClipping',.001,100),('minValidationDelta',0,10)]:
            require(isinstance(result[k],(int,float)) and not isinstance(result[k],bool) and math.isfinite(result[k]) and lo<=result[k]<=hi,'INVALID_TRAINING_CONFIG',field=k)
        for k,lo,hi in [('maxSteps',0,10000000),('earlyStoppingPatience',1,100),('seed',0,2**32-1)]:
            require(isinstance(result[k],int) and not isinstance(result[k],bool) and lo<=result[k]<=hi,'INVALID_TRAINING_CONFIG',field=k)
        require(isinstance(result['targetModules'],list) and result['targetModules'] and all(isinstance(k,str) for k in result['targetModules']),'INVALID_TARGET_MODULES')
        return result

    def run(self, job):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model, PeftModel
        from LLM.training.objectives import prepare,loss as objective_loss
        torch.set_num_threads(1)
        body=job['payload'];cfg=self.config(body.get('config',body))
        branch=self.store.get('branch',body['branchId']);anchor=self.store.get('artifact',branch['anchorArtifactId'])
        require(not anchor.get('comparisonOnly'),'COMPARISON_ONLY_MODEL',status=409)
        root=self.registry.verify(anchor)
        snapshot=self.store.get('snapshot',body['snapshotId'])
        require(snapshot['familyId']==branch['familyId'],'INCOMPATIBLE_BASE')
        prior_run=self.store.get('training_run',job['id'],{})
        if prior_run.get('state')=='completed' and prior_run.get('outputArtifactId'):
            self.registry.verify(self.store.get('artifact',prior_run['outputArtifactId']))
            return {'artifactId':prior_run['outputArtifactId'],'adapterArtifactId':prior_run['adapterArtifactId'],'completedSteps':prior_run['completedSteps'],'state':'completed'}
        foundation=anchor.get('origin')=='random_init' or bool(anchor.get('foundationProjectId'))
        if foundation:
            from LLM.foundation.tokenizer import load as load_native
            tokenizer=load_native(self.store,anchor['tokenizerId'])
            require(cfg['maxSequence']<=min(512,int(anchor.get('trainedContextTokens',128))),'INPUT_TOO_LARGE_FOR_TRAINING')
            require(cfg['device']=='cpu' and cfg['objective']=='sft','FOUNDATION_SPECIALIZATION_SFT_REQUIRED')
        else:tokenizer=AutoTokenizer.from_pretrained(root,local_files_only=True,trust_remote_code=False)
        if prior_run.get('datasetRows'):
            rows=prior_run['datasetRows'];batch_receipt=prior_run['batchReceipt']
        else:rows,batch_receipt=self.snapshots.batch(branch['id'],snapshot['id'],tokenizer,cfg['objective'])
        target_tokens=sum(len(tokenizer.encode(r['output'],add_special_tokens=False)) for r in rows)
        synthetic_tokens=sum(len(tokenizer.encode(r['output'],add_special_tokens=False)) for r in rows if r.get('targetOrigin')=='teacher')
        require(not target_tokens or synthetic_tokens/target_tokens<=(0 if foundation else .3),'SYNTHETIC_RATIO_EXCEEDED',status=409)
        prepared=[]; used=[];excluded=[]
        for row in rows:
            try:prepared.append(prepare(tokenizer,row,cfg['maxSequence'],cfg['objective'],encode_example));used.append(row)
            except Exception as exc:
                if getattr(exc,'code',None)!='INPUT_TOO_LARGE_FOR_TRAINING':raise
                excluded.append(row['exampleId'])
        require(bool(prepared),'NO_TRAINABLE_EXAMPLES')
        torch.manual_seed(cfg['seed']);random.seed(cfg['seed'])
        require(cfg['device']!='cuda' or torch.cuda.is_available(),'TRAINING_DEVICE_UNAVAILABLE',status=503)
        model=AutoModelForCausalLM.from_pretrained(root,local_files_only=True,trust_remote_code=False,
            dtype=torch.float32,attn_implementation='eager').to(cfg['device'])
        names={name.split('.')[-1] for name,_ in model.named_modules()}
        require(set(cfg['targetModules']).issubset(names),'TARGET_MODULE_NOT_FOUND')
        model.config.use_cache=False
        run=self.store.get('training_run',job['id'],{})
        resume_id=body.get('checkpointId') or (run.get('checkpointId') if job.get('attempt',1)>1 else None)
        if resume_id:
            checkpoint=self.store.get('checkpoint',resume_id)
            require(checkpoint['runId']==job['id'] and checkpoint['configHash']==digest(cfg) and checkpoint['snapshotHash']==snapshot['manifestHash'],'CHECKPOINT_MISMATCH')
            cp=self.store.root/checkpoint['storagePath']
            require(file_hash(cp/('optimizer.json' if foundation else 'optimizer.pt'))==checkpoint['optimizerHash'],'CHECKPOINT_HASH_MISMATCH')
            for name,checksum in checkpoint.get('files',{}).items():require(file_hash(cp/name)==checksum,'CHECKPOINT_HASH_MISMATCH')
            model=PeftModel.from_pretrained(model,cp/'adapter',is_trainable=True,local_files_only=True)
        elif branch.get('currentAdapterId'):
            adapter=self.store.get('artifact',branch['currentAdapterId'])
            require(adapter['familyId']==anchor['familyId'],'INCOMPATIBLE_BASE')
            model=PeftModel.from_pretrained(model,self.registry.verify(adapter),is_trainable=True,local_files_only=True)
        else:
            model=get_peft_model(model,LoraConfig(r=cfg['rank'],lora_alpha=cfg['loraAlpha'],lora_dropout=cfg['dropout'],bias='none',task_type='CAUSAL_LM',target_modules=cfg['targetModules']))
        parameters=[p for p in model.parameters() if p.requires_grad]
        require(bool(parameters),'NO_TRAINABLE_PARAMETERS')
        before=digest([digest(p.detach().cpu().float().numpy().tobytes()) for p in parameters])
        optimizer=torch.optim.AdamW(parameters,lr=cfg['learningRate'])
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:1.0)
        validation=[prepare(tokenizer,r,cfg['maxSequence'],cfg['objective'],encode_example) for r in self.snapshots.examples(snapshot['id'],'validation')
            if r.get('branchId') in (None,branch['id']) and r.get('objective','sft')==cfg['objective']]
        require(validation,'VALIDATION_DATA_REQUIRED')
        batch_count=math.ceil(len(prepared)/cfg['batchSize'])
        steps_per_epoch=math.ceil(batch_count/cfg['gradientAccumulation'])
        total=steps_per_epoch*cfg['epochs']
        if cfg.get('maxSteps'):total=min(total,int(cfg['maxSteps']))
        completed=0; cursor=0; loss_value=None
        best_validation=float('inf');bad_epochs=0;validation_history=[];early_stopped=False
        if resume_id:
            if foundation:
                from LLM.foundation.checkpoint import _unpack
                from safetensors.torch import load_file
                saved=_unpack(json.loads((cp/'optimizer.json').read_text('utf-8')),load_file(str(cp/'optimizer.safetensors')))
            else:saved=torch.load(cp/'optimizer.pt',map_location=cfg['device'],weights_only=True)
            optimizer.load_state_dict(saved['optimizer']);completed=saved['completedSteps'];cursor=saved['cursor']
            torch.set_rng_state(saved['rng'].cpu());random.setstate(saved['pythonRng'])
            if saved.get('cudaRng') and torch.cuda.is_available():torch.cuda.set_rng_state_all(saved['cudaRng'])
            before=saved['initialHash']
            if saved.get('scheduler'):scheduler.load_state_dict(saved['scheduler'])
            best_validation=saved.get('bestValidation') if saved.get('bestValidation') is not None else float('inf');bad_epochs=saved.get('badEpochs',0)
            validation_history=saved.get('validationHistory',[])
        run=self.store.put('training_run',{'branchId':branch['id'],'snapshotId':snapshot['id'],'snapshotHash':snapshot['manifestHash'],
            'config':cfg,'configHash':digest(cfg),'anchorArtifactId':anchor['id'],'state':'running','completedSteps':completed,
            'totalSteps':total,'excludedExampleIds':excluded,'trainSamples':len(prepared),'plannedTotalSteps':total,'datasetRows':used,'batchReceipt':batch_receipt},job['id'])
        model.train();optimizer.zero_grad(set_to_none=True)

        def checkpoint_save(cursor_value):
            identity=uid();dest=self.store.root/'staging'/('checkpoint-'+identity);dest.mkdir()
            model.save_pretrained(dest/'adapter',safe_serialization=True)
            optimizer_state={'optimizer':optimizer.state_dict(),'completedSteps':completed,'cursor':cursor_value,'rng':torch.get_rng_state(),
                        'pythonRng':random.getstate(),'cudaRng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,'initialHash':before,'scheduler':scheduler.state_dict(),
                        'bestValidation':best_validation if math.isfinite(best_validation) else None,'badEpochs':bad_epochs,'validationHistory':validation_history}
            if foundation:
                from LLM.foundation.checkpoint import _pack
                from safetensors.torch import save_file
                tensors={};tree=_pack(optimizer_state,tensors)
                save_file(tensors,str(dest/'optimizer.safetensors'));atomic_json(dest/'optimizer.json',tree)
            else:torch.save(optimizer_state,dest/'optimizer.pt')
            import os
            for path in dest.rglob('*'):
                if path.is_file():
                    with path.open('r+b') as handle:handle.flush();os.fsync(handle.fileno())
            published=self.store.root/'checkpoints'/identity;dest.replace(published);dest=published
            record=self.store.put('checkpoint',{'runId':job['id'],'storagePath':dest.relative_to(self.store.root).as_posix(),
                'optimizerHash':file_hash(dest/('optimizer.json' if foundation else 'optimizer.pt')),'configHash':digest(cfg),'snapshotHash':snapshot['manifestHash'],
                'completedSteps':completed,'cursor':cursor_value,'files':{p.relative_to(dest).as_posix():file_hash(p) for p in dest.rglob('*') if p.is_file()}},identity)
            self.store.put('training_run',{'checkpointId':identity},job['id'])
            return record

        for epoch in range(cfg['epochs']):
            batches=[prepared[i:i+cfg['batchSize']] for i in range(0,len(prepared),cfg['batchSize'])]
            for group_start in range(0,len(batches),cfg['gradientAccumulation']):
                group=batches[group_start:group_start+cfg['gradientAccumulation']]
                group_end=epoch*len(batches)+group_start+len(group)
                if group_end<=cursor:continue
                if completed>=total:break
                current=self.store.assert_lease(job)
                require(current['state'] not in ('cancelling','cancelled'),'JOB_CANCELLED',status=409)
                yielding=bool(getattr(self,'should_yield',lambda:False)())
                if current['state']=='pause_requested' or yielding:
                    saved=checkpoint_save(cursor)
                    self.store.put('training_run',{'state':'paused'},job['id'])
                    return {'state':'yielded' if yielding and current['state']!='pause_requested' else 'paused','checkpointId':saved['id']}
                for batch in group:
                    pad=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                    require(pad is not None,'TOKENIZER_PADDING_REQUIRED')
                    measured_loss=objective_loss(torch,model,batch,pad,cfg['device'],cfg['objective'],cfg['dpoBeta'])
                    require(bool(torch.isfinite(measured_loss)),'NONFINITE_TRAINING')
                    (measured_loss/len(group)).backward();loss_value=float(measured_loss.detach())
                norm=torch.nn.utils.clip_grad_norm_(parameters,cfg['gradientClipping'])
                require(bool(torch.isfinite(norm)),'NONFINITE_TRAINING')
                optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True);completed+=1;cursor=group_end
                self.store.put('training_run',{'state':'running','completedSteps':completed,'loss':loss_value,'epoch':epoch+(group_start+len(group))/len(batches)},job['id'])
                self.store.progress(job,'training',completed,total,'optimizer_steps')
                if completed%cfg['checkpointEvery']==0:checkpoint_save(cursor)
            model.eval();validation_losses=[]
            with torch.inference_mode():
                for row in validation:
                    pad=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                    value=objective_loss(torch,model,[row],pad,cfg['device'],cfg['objective'],cfg['dpoBeta'])
                    require(bool(torch.isfinite(value)),'NONFINITE_VALIDATION')
                    validation_losses.append(float(value))
            measured=sum(validation_losses)/len(validation_losses)
            validation_history.append({'epoch':epoch+1,'loss':measured,'completedSteps':completed})
            if measured<best_validation-cfg['minValidationDelta']:
                best_validation=measured;bad_epochs=0
                model.save_pretrained(self.store.root/'checkpoints'/('best-'+job['id']),safe_serialization=True)
            else:bad_epochs+=1
            self.store.put('training_run',{'validationLoss':measured,'validationHistory':validation_history},job['id'])
            model.train()
            if bad_epochs>=cfg['earlyStoppingPatience']:
                early_stopped=True;break
            if completed>=total:break
        require(completed>0,'NO_EFFECTIVE_TRAINING')
        after=digest([digest(p.detach().cpu().float().numpy().tobytes()) for p in parameters])
        require(after!=before,'NO_EFFECTIVE_TRAINING')
        checkpoint_save(cursor)
        best_path=self.store.root/'checkpoints'/('best-'+job['id'])/'adapter_model.safetensors'
        if best_path.exists():
            from safetensors.torch import load_file
            from peft import set_peft_model_state_dict
            set_peft_model_state_dict(model,load_file(str(best_path)))
            after=digest([digest(p.detach().cpu().float().numpy().tobytes()) for p in parameters])
        self.store.put('training_run',{'state':'validating'},job['id'])
        for identity in {r.get('snapshotId',snapshot['id']) for r in used}:
            require(self.store.get('snapshot',identity)['state']=='verified','POLICY_REVOKED',status=409)
        adapter_dir=self.store.root/'staging'/('adapter-'+uid());model.save_pretrained(adapter_dir,safe_serialization=True)
        adapter_parents=[anchor['id']]+([branch['currentAdapterId']] if branch.get('currentAdapterId') else [])
        adapter=self.registry.publish(adapter_dir,kind='adapter',family_id=anchor['familyId'],parents=adapter_parents,branch_id=branch['id'],license_refs=anchor['licenseRefs'],
            metadata={'trainingRunId':job['id'],'snapshotId':snapshot['id'],'snapshotIds':sorted({r.get('snapshotId',snapshot['id']) for r in used}),'initialTrainableHash':before,'trainedHash':after})
        full=model.merge_and_unload(safe_merge=True)
        full_dir=self.store.root/'staging'/('specialist-'+uid());full.save_pretrained(full_dir,safe_serialization=True)
        if not foundation:tokenizer.save_pretrained(full_dir)
        # Preserve byte-identical family tokenizer assets; HF reserialization may
        # change JSON metadata without changing the learned tokenizer semantics.
        import shutil
        from LLM.registry.service import TOKEN_FILES
        for name in TOKEN_FILES:
            if (root/name).is_file():shutil.copyfile(root/name,full_dir/name)
        artifact_kind='distilled_checkpoint' if body.get('distillationId') else 'specialist_checkpoint'
        teachers=self.store.get('distillation',body['distillationId'])['teacherArtifactIds'] if body.get('distillationId') else []
        artifact=self.registry.publish(full_dir,kind=artifact_kind,family_id=anchor['familyId'],parents=[anchor['id'],adapter['id'],*teachers],branch_id=branch['id'],license_refs=anchor['licenseRefs'],
            metadata={'trainingRunId':job['id'],'snapshotId':snapshot['id'],'snapshotIds':sorted({r.get('snapshotId',snapshot['id']) for r in used}),'adapterArtifactId':adapter['id']})
        with self.store.tx():
            self.store.assert_lease(job)
            self.snapshots.consume(branch['id'],used,job['id'])
            self.store.put('training_run',{'state':'completed','completedSteps':completed,'totalSteps':completed,'finishedAt':now(),
                'outputArtifactId':artifact['id'],'adapterArtifactId':adapter['id'],'initialTrainableHash':before,'trainedHash':after,
                'earlyStopped':early_stopped,'validationHistory':validation_history},job['id'])
            self.store.put('branch',{'candidateArtifactId':artifact['id'],'candidateAdapterId':adapter['id']},branch['id'])
        return {'artifactId':artifact['id'],'adapterArtifactId':adapter['id'],'completedSteps':completed,'state':'completed'}
