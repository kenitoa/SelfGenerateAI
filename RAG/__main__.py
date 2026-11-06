"""Run with python -m RAG. No project-wide or LLM directory writes."""
import argparse
import getpass
import os
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description="SEED Interface ↔ RAG (Pre ↔ Post)")
    parser.add_argument('--data-dir',default=os.environ.get('SEED_DATA_DIR',str(Path(__file__).parent/'data')))
    commands=parser.add_subparsers(dest='command',required=True)
    serve=commands.add_parser('serve');serve.add_argument('--host',default='127.0.0.1');serve.add_argument('--port',type=int,default=8000);serve.add_argument('--local',action='store_true',help='Loopback-only development session; prohibited on network binds')
    commands.add_parser('init-admin')
    restore=commands.add_parser('restore');restore.add_argument('--backup-id',required=True);restore.add_argument('--destination',required=True)
    commands.add_parser('check')
    commands.add_parser('install-embedding',help='Install pinned multilingual E5 INT8 in the RAG data directory')
    args=parser.parse_args()
    if args.command=='serve':
        if args.local and args.host not in ('127.0.0.1','::1','localhost'):
            parser.error('--local은 loopback 주소에서만 사용할 수 있습니다.')
        import uvicorn
        from RAG.api import create_app
        uvicorn.run(create_app(args.data_dir,local_dev=args.local),host=args.host,port=args.port,workers=1,proxy_headers=False,log_level='info')
        return
    from RAG.core import Store
    store=Store(args.data_dir)
    if args.command=='install-embedding':
        from RAG.Pre.model_setup import install
        result=install(store)
        print('Embedding installed:',result['directory'])
    elif args.command=='init-admin':
        from RAG.security import Auth
        password=getpass.getpass('관리자 비밀번호 (12자 이상): ')
        if password!=getpass.getpass('비밀번호 확인: '):raise SystemExit('비밀번호가 일치하지 않습니다.')
        Auth(store).set_password(password);print('관리자 설정 완료. 기존 세션은 해제되었습니다.')
    elif args.command=='restore':
        from RAG.Pre.service import PreService
        from RAG.Post.service import PostService
        from RAG.operations import Operations
        pre=PreService(store);result=Operations(store,pre,PostService(store,pre)).restore(args.backup_id,args.destination)
        print('복원 완료:',result['destination'])
    else:
        from RAG.Pre.service import PreService
        from RAG.operations import system_status
        import json
        print(json.dumps({'sqlite':store.sqlite_version,'pre':PreService(store).readiness(),'system':system_status(store)},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
