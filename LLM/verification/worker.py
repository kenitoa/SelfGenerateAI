"""Trusted Docker driver. Generated source is only passed to isolated containers."""
import argparse
import os
import subprocess
import threading
import time
from LLM.core import Store, uid
from LLM.verification.broker import VerifierBroker
from LLM.verification.sandbox import Sandbox


class VerifierDriver:
    def __init__(self, store):
        os.environ['SEED_VERIFIER_DRIVER']='1'
        self.store=store;self.broker=VerifierBroker(store);self.owner='verifier-'+uid()
        self.sandbox=Sandbox(store);self._recovered=False;self._cleanup_pending=None

    def heartbeat(self):
        caps=self.sandbox.local_capabilities()
        if self._cleanup_pending:
            request,sandbox=self._cleanup_pending
            if sandbox.cancel():self.broker.release(request);self._cleanup_pending=None
        if not self.broker.heartbeat(self.owner,{**caps,'available':caps['available'] and self._recovered}):return False
        if not self._recovered:
            # A previous driver may have died with a running Docker container.
            # Container names are deterministic private request UUIDs.
            for row in self.broker.orphans(self.owner):
                try:
                    result=subprocess.run(['docker','rm','-f','seed-rpc-'+row['id']],capture_output=True,timeout=3)
                    if result.returncode and b'No such container' not in (result.stderr or b''):return False
                except (OSError,subprocess.TimeoutExpired):return False
                self.broker.abandon(row['id'],row['owner'])
            self._recovered=True
            self.broker.heartbeat(self.owner,caps)
        return True

    def serve_once(self):
        if not self.heartbeat():return False
        if not self.sandbox.local_capabilities()['available']:return False
        request=self.broker.claim(self.owner)
        if request is None:return False
        sandbox=Sandbox(self.store,request_id=request['id']);ended=threading.Event()
        def cancel_when_expired():
            while not ended.wait(.1):
                if not self.broker.active(request):sandbox.cancel();return
        watcher=threading.Thread(target=cancel_when_expired,daemon=True);watcher.start()
        try:
            profile=self.broker.validate(request['payload']);body=request['payload']
            if body['kind']=='probe':
                receipt=sandbox.probe(profile)
                self.broker.complete(request,{'receiptId':receipt['id']})
            elif body['kind']=='verify':
                fixture=self.store.get('fixture',body['fixtureId'])
                receipt=sandbox.verify(body['code'],fixture)
                self.broker.complete(request,{'receiptId':receipt['id']})
            else:
                values=sandbox.run_python(body['code'],body['inputs'],profile)
                self.broker.complete(request,{'values':values})
        except Exception as exc:
            # Never serialize source, Docker output, exception text, or paths.
            self.broker.complete(request,error=getattr(exc,'code','VERIFIER_EXECUTION_FAILED'))
        finally:
            ended.set();watcher.join(timeout=1)
            if sandbox.cancel():self.broker.release(request)
            else:self._cleanup_pending=(request,sandbox)
        return True

    def serve(self):
        stopped=threading.Event()
        def pulse():
            while not stopped.wait(5):self.heartbeat()
        self.heartbeat();thread=threading.Thread(target=pulse,daemon=True);thread.start()
        try:
            while True:
                if not self.serve_once():time.sleep(.2)
        finally:
            stopped.set();thread.join(timeout=1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',required=True)
    parser.add_argument('--serve',action='store_true')
    parser.add_argument('--health',action='store_true')
    args=parser.parse_args()
    store=Store(args.data_dir)
    try:
        if args.health:
            # Liveness is independent of Docker limit support. A NAS may remain
            # healthy while the UI correctly holds evaluation for missing caps.
            return 0 if VerifierBroker(store).status()['alive'] else 1
        if not args.serve:parser.error('--serve or --health is required')
        VerifierDriver(store).serve()
    finally:store.close()
    return 0


if __name__=='__main__':raise SystemExit(main())
