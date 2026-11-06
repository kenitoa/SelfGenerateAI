"""python -m LLM: local control, integrated RAG, model import and diagnostics."""
import argparse
import json
import os
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description='SEED LLM — actual learning and merged artifacts')
    parser.add_argument('--data-dir',default=os.environ.get('SEED_LLM_DATA_DIR',str(Path(__file__).parent/'runtime-data')))
    sub=parser.add_subparsers(dest='command',required=True)
    serve=sub.add_parser('serve');serve.add_argument('--host',default='127.0.0.1');serve.add_argument('--port',type=int,default=8100)
    serve.add_argument('--local',action='store_true');serve.add_argument('--worker',action='store_true');serve.add_argument('--integrated',action='store_true')
    serve.add_argument('--rag-data-dir',default=str(Path(__file__).parent.parent/'RAG'/'data'));serve.add_argument('--token-file')
    install=sub.add_parser('install-model');install.add_argument('--repo',default='Qwen/Qwen2.5-Coder-0.5B-Instruct');install.add_argument('--revision',default='main')
    register=sub.add_parser('register');register.add_argument('directory');register.add_argument('--license',required=True);register.add_argument('--name',default='Imported model')
    tools=sub.add_parser('configure-tools');tools.add_argument('--converter',required=True);tools.add_argument('--quantizer',required=True)
    verifier=sub.add_parser('register-verifier');verifier.add_argument('--image',required=True);verifier.add_argument('--language',choices=['python','javascript','typescript','sql'],default='python')
    fixture=sub.add_parser('register-fixture');fixture.add_argument('file')
    resources=sub.add_parser('configure-resources');resources.add_argument('--nas-ram-gb',type=int,default=32);resources.add_argument('--reserve-gb',type=int,default=8)
    evolution=sub.add_parser('setup-self-code');evolution.add_argument('--profile-id',required=True)
    sub.add_parser('backup')
    gc=sub.add_parser('gc');gc.add_argument('--apply',action='store_true')
    recovery=sub.add_parser('restore');recovery.add_argument('--backup',required=True);recovery.add_argument('--destination',required=True)
    sub.add_parser('check')
    args=parser.parse_args();root=Path(args.data_dir).resolve()
    os.environ['SEED_LLM_DATA_DIR']=str(root)
    if args.command=='serve':
        require_local=args.host in ('localhost','127.0.0.1','::1')
        if args.local and not require_local:parser.error('--local은 loopback에서만 가능합니다.')
        import uvicorn
        from LLM.api import create_app
        if args.integrated:
            os.environ['SEED_LLM_ENABLED']='1';os.environ['SEED_LLM_WORKER']='1' if args.worker else '0'
            import secrets
            token_path=root/'metadata'/'service-token';token_path.parent.mkdir(parents=True,exist_ok=True)
            if not token_path.exists():
                descriptor=os.open(token_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
                with os.fdopen(descriptor,'w') as handle:handle.write(secrets.token_urlsafe(40))
            service_token=token_path.read_text().strip();os.environ['SEED_LLM_TOKEN_FILE']=str(token_path)
            from RAG.api import create_app as rag_app
            app=rag_app(args.rag_data_dir,local_dev=args.local)
            control=app.state.llm;control.store.put('setting',{'url':'http://127.0.0.1:8100'},'inferenceEndpoint')
            facade=create_app(root,token=service_token,control=control)
            import threading
            thread=threading.Thread(target=lambda:uvicorn.run(facade,host='127.0.0.1',port=8100,log_level='warning'),daemon=True);thread.start()
            uvicorn.run(app,host=args.host,port=args.port,workers=1,proxy_headers=False)
        else:
            token=Path(args.token_file).read_text().strip() if args.token_file else None
            uvicorn.run(create_app(root,token=token,local=args.local,worker=args.worker,allowed_roots=[root/'imports']),host=args.host,port=args.port,workers=1,proxy_headers=False)
        return
    from LLM.service import Control
    c=Control(root,[root/'imports',Path(getattr(args,'directory',root/'imports')).resolve()])
    if args.command=='register':
        print(json.dumps(c.registry.register({'directory':args.directory,'name':args.name,'licenseRefs':[args.license]}),ensure_ascii=False,indent=2))
    elif args.command=='configure-resources':
        from LLM.workers.resources import configure,budget
        configure(c.store,{'mode':'adaptive','nasRamBytes':args.nas_ram_gb*1024**3,'systemReserveBytes':args.reserve_gb*1024**3})
        print(json.dumps(budget(c.store),ensure_ascii=False,indent=2))
    elif args.command=='setup-self-code':print(json.dumps(c.evolution.setup(args.profile_id),ensure_ascii=False,indent=2))
    elif args.command=='install-model':
        from huggingface_hub import HfApi,snapshot_download
        from LLM.core import require
        require(args.repo in ('Qwen/Qwen2.5-Coder-0.5B-Instruct','Qwen/Qwen2.5-Coder-1.5B-Instruct'),'MODEL_REPOSITORY_NOT_ALLOWED')
        info=HfApi().model_info(args.repo,revision=args.revision)
        dest=root/'imports'/(args.repo.split('/')[-1]+'-'+info.sha[:12])
        snapshot_download(args.repo,revision=info.sha,local_dir=dest,allow_patterns=['*.safetensors','config.json','generation_config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.json','merges.txt'],max_workers=2)
        # Hub metadata/cache and symlink files are not model payloads.
        import shutil
        shutil.rmtree(dest/'.cache',ignore_errors=True)
        value=c.registry.register({'directory':str(dest),'name':args.repo,'revision':info.sha,'licenseRefs':['Apache-2.0']})
        print(json.dumps({'artifactId':value['artifact']['id'],'familyId':value['family']['id'],'revision':info.sha},indent=2))
    elif args.command=='configure-tools':
        from LLM.core import require,file_hash
        converter=Path(args.converter).resolve();quantizer=Path(args.quantizer).resolve()
        require(converter.is_file() and converter.suffix=='.py' and quantizer.is_file(),'TOOLCHAIN_FILES_REQUIRED')
        result=c.store.put('setting',{'ggufConverter':str(converter),'converterHash':file_hash(converter),'quantizer':str(quantizer),'quantizerHash':file_hash(quantizer)},'tools')
        print('Registered pinned conversion tools:',result['id'])
    elif args.command=='register-verifier':
        from LLM.core import require
        require('@sha256:' in args.image,'PINNED_SANDBOX_IMAGE_REQUIRED')
        print(c.store.put('verifier_profile',{'language':args.language,'image':args.image})['id'])
    elif args.command=='register-fixture':
        from LLM.core import require
        body=json.loads(Path(args.file).read_text('utf-8'));require(body.get('profileId') and (body.get('tests') or body.get('cases')),'INVALID_VERIFIER_FIXTURE')
        c.store.get('verifier_profile',body['profileId']);print(c.store.put('fixture',body)['id'])
    elif args.command=='backup':
        from LLM.storage import backup
        print(backup(c.store)['id'])
    elif args.command=='gc':
        from LLM.storage.retention import collect
        print(json.dumps(collect(c.store,args.apply),ensure_ascii=False,indent=2))
    elif args.command=='restore':
        from LLM.storage import restore
        print(json.dumps(restore(args.backup,args.destination),ensure_ascii=False))
    else:print(json.dumps(c.capabilities(),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
