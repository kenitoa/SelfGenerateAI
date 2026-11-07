import json
import threading
import time
import shutil
import subprocess
from contextlib import contextmanager
import httpx
from LLM.core import require, file_hash, now

class Inference:
    def __init__(self, store, registry):
        self.store,self.registry=store,registry
        self.lock=threading.RLock();self.loaded=None;self.model=None;self.tokenizer=None
        self.foreground_waiters=0;self.waiter_lock=threading.Lock()
        self.last_used=time.monotonic()

    @contextmanager
    def foreground(self):
        with self.waiter_lock:self.foreground_waiters+=1
        try:
            with self.store.foreground_slot():
                with self.lock:
                    with self.waiter_lock:self.foreground_waiters-=1
                    try:yield
                    finally:
                        with self.waiter_lock:self.foreground_waiters+=1
        finally:
            with self.waiter_lock:self.foreground_waiters-=1

    def expire(self,force=False):
        if not self.lock.acquire(blocking=False):return False
        try:
            if force:self._template_tokenizer=None
            if self.loaded and (force or time.monotonic()-self.last_used>60):
                self.model=None;self.tokenizer=None;self.loaded=None
                import gc
                gc.collect()
            if force and hasattr(self,'foundation'):return self.foundation.expire(force=True)
            return True
        finally:self.lock.release()

    def tags(self):
        return {'models':[{'name':r['engineAlias'],'model':r['engineAlias'],'digest':r['engineDigest'],
            'size':self.store.get('artifact',r['artifactId'])['sizeBytes'],
            'details':{'parameter_size':str(self.store.get('artifact',r['artifactId'])['parameterCount'])}}
            for r in self.store.list('release') if r['state'] in ('ready','active','retired') and self.store.get('artifact',r['artifactId'])['state']=='ready']}

    def load(self, artifact_id):
        import torch
        from transformers import AutoTokenizer,AutoModelForCausalLM
        artifact=self.store.get('artifact',artifact_id);directory=self.registry.verify(artifact)
        self.last_used=time.monotonic()
        require(artifact.get('format')=='safetensors','RUNTIME_BACKEND_REQUIRED',status=503)
        if self.loaded!=artifact_id:
            self.model=None;self.tokenizer=None
            import gc
            gc.collect()
            from LLM.workers.resources import budget
            resources=budget(self.store)
            available=min(resources['availableRamBytes'],max(0,resources['maxRssBytes']-resources['processRssBytes']))
            estimated=artifact['parameterCount']*4+256*1024**2
            require(available-max(512*1024**2,int(available*.2))>=estimated,'RESOURCE_PRESSURE',status=503)
            from LLM.foundation.evaluation_restart import child_runtime_threads
            torch.set_num_threads(child_runtime_threads(self.store))
            self.tokenizer=AutoTokenizer.from_pretrained(directory,local_files_only=True,trust_remote_code=False)
            self.model=AutoModelForCausalLM.from_pretrained(directory,local_files_only=True,trust_remote_code=False,dtype=torch.float32,attn_implementation='eager')
            self.model.eval();self.loaded=artifact_id
        return self.model,self.tokenizer

    def messages_prompt(self,artifact_id,messages):
        """Use a market baseline's pinned local template, without downloading code."""
        from transformers import AutoTokenizer
        artifact=self.store.get('artifact',artifact_id);root=self.registry.verify(artifact)
        cached=getattr(self,'_template_tokenizer',None)
        if not cached or cached[0]!=artifact['manifestHash']:
            cached=(artifact['manifestHash'],AutoTokenizer.from_pretrained(root,local_files_only=True,trust_remote_code=False))
            self._template_tokenizer=cached
        require(cached[1].chat_template,'GENERATION_PROFILE_INCOMPATIBLE',status=409)
        return cached[1].apply_chat_template(messages,tokenize=False,add_generation_prompt=True)

    def text(self,artifact_id,prompt,max_tokens=128):
        import torch
        with self.lock:
            artifact=self.store.get('artifact',artifact_id)
            if artifact.get('origin')=='random_init' or artifact.get('foundationProjectId'):
                require(hasattr(self,'foundation'),'FOUNDATION_RUNTIME_REQUIRED',status=503)
                return self.foundation.text(artifact_id,prompt,max_tokens,purpose='evaluation')
            if artifact['format']=='gguf':
                runtime=self.ollama_register(artifact)
                started=time.monotonic()
                response=httpx.post('http://127.0.0.1:11434/api/generate',json={'model':runtime['alias'],'prompt':prompt,'raw':True,'stream':False,'options':{'num_predict':max_tokens,'temperature':0}},timeout=300)
                response.raise_for_status();value=response.json();require(value.get('done') is True,'INCOMPLETE_ENGINE_RESPONSE')
                import psutil
                processes=[]
                for p in psutil.process_iter(['name','memory_info']):
                    if 'ollama' in (p.info.get('name') or '').lower():processes.append(p.info['memory_info'].rss)
                return value['response'],{**{k:value.get(k,0) for k in ('prompt_eval_count','eval_count','total_duration')},
                    'runtimeResourceObserved':bool(processes),'runtimeRssBytes':sum(processes)}
            model,tokenizer=self.load(artifact_id)
            inputs=tokenizer(prompt,return_tensors='pt',add_special_tokens=False,return_token_type_ids=False)
            limit=int(getattr(model.config,'max_position_embeddings',2048))
            require(inputs['input_ids'].shape[1]+max_tokens<=limit,'INPUT_TOO_LARGE_FOR_MODEL')
            started=time.monotonic()
            with torch.inference_mode():
                generated=model.generate(**inputs,max_new_tokens=max_tokens,do_sample=False,pad_token_id=tokenizer.eos_token_id)
            result=tokenizer.decode(generated[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
            return result,{'prompt_eval_count':int(inputs['input_ids'].shape[1]),'eval_count':int(generated.shape[1]-inputs['input_ids'].shape[1]),
                'total_duration':int((time.monotonic()-started)*1e9),'cpuThreads':torch.get_num_threads()}

    def ollama_register(self,artifact):
        root=self.registry.verify(artifact);runtime=self.store.get('runtime_model',artifact['id'],{})
        if runtime:return runtime
        tool=shutil.which('ollama');require(tool is not None,'OLLAMA_RUNTIME_UNAVAILABLE',status=503)
        alias='seed-build/'+artifact['id']
        modelfile=self.store.root/'staging'/(artifact['id']+'.Modelfile')
        modelfile.write_text('FROM '+json.dumps(str(root/'model.gguf'))+'\n','utf-8')
        result=subprocess.run([tool,'create',alias,'-f',str(modelfile)],capture_output=True,timeout=600)
        require(result.returncode==0,'RUNTIME_REGISTRATION_FAILED',status=503)
        models=httpx.get('http://127.0.0.1:11434/api/tags',timeout=10).json()['models']
        installed=next((m for m in models if m.get('name') in (alias,alias+':latest')),None)
        require(installed is not None,'MODEL_NOT_INSTALLED')
        return self.store.put('runtime_model',{'alias':installed['name'],'digest':installed['digest']},artifact['id'])

    def stream(self,body):
        require(body.get('raw') is True and isinstance(body.get('prompt'),str),'RAW_PROMPT_REQUIRED')
        release=next((r for r in self.store.list('release') if r['engineAlias']==body.get('model') and r['state'] in ('ready','active','retired')),None)
        require(release is not None,'MODEL_NOT_INSTALLED',status=503)
        # One immutable artifact is resolved before generation and retained for this call.
        identity=release['artifactId'];options=body.get('options',{})
        maximum=int(options.get('num_predict',700));require(1<=maximum<=8192,'INVALID_TOKEN_BUDGET')
        artifact=self.store.get('artifact',identity)
        require(not (artifact.get('origin')=='random_init' or artifact.get('foundationProjectId')),'GENERATION_PROFILE_INCOMPATIBLE',status=409)
        if artifact['format']=='gguf':
            runtime=self.ollama_register(artifact)
            with self.foreground(),httpx.stream('POST','http://127.0.0.1:11434/api/generate',json={**body,'model':runtime['alias']},timeout=300) as response:
                response.raise_for_status()
                done=False
                for line in response.iter_lines():
                    if not line:continue
                    require(self.store.get('artifact',identity)['state'] not in ('revoked','quarantined'),'POLICY_REVOKED',status=403)
                    value=json.loads(line);value['model']=release['engineAlias'];done=value.get('done',False)
                    yield json.dumps(value,ensure_ascii=False)+'\n'
                require(done,'INCOMPLETE_ENGINE_RESPONSE',status=503)
            return
        import torch
        from transformers import TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
        cancelled=threading.Event()
        class Stop(StoppingCriteria):
            def __call__(self,*args,**kwargs):return cancelled.is_set()
        with self.foreground():
            model,tokenizer=self.load(identity)
            inputs=tokenizer(body['prompt'],return_tensors='pt',add_special_tokens=False,return_token_type_ids=False)
            context=min(int(options.get('num_ctx',4096)),int(getattr(model.config,'max_position_embeddings',4096)))
            require(inputs['input_ids'].shape[1]+maximum<=context,'INPUT_TOO_LARGE_FOR_MODEL')
            streamer=TextIteratorStreamer(tokenizer,skip_prompt=True,skip_special_tokens=True,timeout=90)
            failures=[];tokens=[];started=time.monotonic()
            def generate():
                try:
                    with torch.inference_mode():
                        output=model.generate(**inputs,max_new_tokens=maximum,do_sample=False,streamer=streamer,
                            pad_token_id=tokenizer.eos_token_id,stopping_criteria=StoppingCriteriaList([Stop()]))
                    tokens.append(int(output.shape[1]-inputs['input_ids'].shape[1]))
                except Exception as exc:failures.append(exc);streamer.on_finalized_text('',stream_end=True)
            thread=threading.Thread(target=generate,daemon=True);thread.start()
            try:
                for delta in streamer:
                    require(self.store.get('artifact',identity)['state'] not in ('revoked','quarantined'),'POLICY_REVOKED',status=403)
                    yield json.dumps({'model':release['engineAlias'],'response':delta,'done':False},ensure_ascii=False)+'\n'
                thread.join(timeout=5)
                if failures:raise failures[0]
                require(not thread.is_alive(),'INFERENCE_TIMEOUT',status=503)
                yield json.dumps({'model':release['engineAlias'],'response':'','done':True,'done_reason':'stop',
                    'prompt_eval_count':int(inputs['input_ids'].shape[1]),'eval_count':tokens[0] if tokens else 0,
                    'total_duration':int((time.monotonic()-started)*1e9)})+'\n'
            finally:
                cancelled.set();thread.join(timeout=5)
