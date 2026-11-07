"""Untrusted generated code only runs in an explicitly registered sandbox image."""
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from LLM.core import require,digest,uid,now
from LLM.verification.broker import VerifierBroker,pinned,MAX_CODE,MAX_JSON

# Docker Desktop can need several seconds to create its first container after
# an idle period. This deadline applies only to our fixed trusted kernel probe;
# generated-code execution retains its own strict wall-time and output limits.
PROBE_CREATE_SECONDS=10


def _strict_json(raw):
    def unsupported(value):raise ValueError('Non-finite JSON output')
    return json.loads(raw,parse_constant=unsupported)

PROBE_SCRIPT=r'''import json,os,pathlib
def read(paths):
 for path in paths:
  try:return pathlib.Path(path).read_text().strip()
  except OSError:pass
 return None
def limit(paths):
 try:return int(read(paths))
 except (ValueError,TypeError):return None
status={}
for line in pathlib.Path('/proc/self/status').read_text().splitlines():
 if ':' in line:
  key,value=line.split(':',1);status[key]=value.strip()
memory=limit(['/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory/memory.limit_in_bytes'])
pids=limit(['/sys/fs/cgroup/pids.max','/sys/fs/cgroup/pids/pids.max'])
limits={'memoryBytes':memory,'pids':pids,'uid':os.getuid(),
 'cpuset':read(['/sys/fs/cgroup/cpuset.cpus.effective','/sys/fs/cgroup/cpuset/cpuset.cpus']),
 'networkOnlyLoopback':set(os.listdir('/sys/class/net'))=={'lo'},
 'rootReadonly':bool(os.statvfs('/').f_flag & os.ST_RDONLY),
 'capEffZero':int(status.get('CapEff','1'),16)==0,'noNewPrivileges':status.get('NoNewPrivs')=='1',
 'dockerSocketAbsent':not any(pathlib.Path(p).exists() for p in ['/var/run/docker.sock','/run/docker.sock']),
 'workDataAbsent':not pathlib.Path('/work').exists()}
passed=(memory is not None and 0<memory<=268435456 and pids is not None and 0<pids<=32
 and limits['uid']==65534 and limits['cpuset']=='0' and all(limits[k] for k in
 ['networkOnlyLoopback','rootReadonly','capEffZero','noNewPrivileges','dockerSocketAbsent','workDataAbsent']))
print(json.dumps({'passed':passed,'limits':limits},allow_nan=False))
'''

class Sandbox:
    def __init__(self,store,request_id=None):
        self.store=store;self._capabilities=None;self._checked=0
        self._request_id=request_id
        self._containers=set();self._container_lock=threading.RLock();self._cancelled=threading.Event()

    def available(self):
        return self.capabilities()['available']

    def capabilities(self):
        local=self.local_capabilities()
        if local['available'] or os.environ.get('SEED_VERIFIER_DRIVER')=='1':return local
        remote=VerifierBroker(self.store).status()
        return remote if remote.get('alive') else local

    def local_capabilities(self):
        if self._capabilities is not None and time.monotonic()-self._checked<5:return self._capabilities
        value={'available':False,'reasonCodes':['SANDBOX_UNAVAILABLE'],'observedAt':now()}
        if shutil.which('docker'):
            try:
                result=subprocess.run(['docker','info','--format','{{json .}}'],capture_output=True,timeout=5,text=True)
                info=json.loads(result.stdout) if result.returncode==0 else {}
                # Docker 29 can omit Cpuset even when --cpuset-cpus works.
                # Unknown reporting permits only entry into our fixed kernel
                # probe; every generated-code path still requires its proof.
                missing=[name for name in ('MemoryLimit','PidsLimit') if info.get(name) is not True]
                if info.get('Cpuset') is not None and info.get('Cpuset') is not True:missing.append('Cpuset')
                unknown=['Cpuset'] if info and info.get('Cpuset') is None else []
                value.update(available=bool(info) and not missing,reasonCodes=['SANDBOX_LIMIT_UNSUPPORTED'] if missing else [],
                    missingLimits=missing,unknownLimits=unknown,
                    engineConfirmedLimits=[name for name in ('MemoryLimit','PidsLimit','Cpuset') if info.get(name) is True],
                    kernelProbeRequired=True,capabilityScope='fixed_kernel_probe_preflight',
                    engineVersion=info.get('ServerVersion'),cpuMode='cpuset')
            except (OSError,ValueError,subprocess.TimeoutExpired):pass
        self._capabilities=value;self._checked=time.monotonic();return value

    def _limits(self):
        require(self.local_capabilities()['available'],'SANDBOX_LIMIT_UNSUPPORTED',status=503)
        return ['--pids-limit','32','--memory','256m','--memory-swap','256m','--cpuset-cpus','0']

    def _remote(self):
        return not self.local_capabilities()['available'] and os.environ.get('SEED_VERIFIER_DRIVER')!='1'

    def _temp(self,prefix):
        root=self.store.root/'sandbox-jobs';root.mkdir(parents=True,exist_ok=True);root.chmod(0o755)
        temp=Path(tempfile.mkdtemp(prefix=prefix,dir=root));temp.chmod(0o755)
        return temp

    def _identity(self,prefix):
        return 'seed-rpc-'+self._request_id if self._request_id else prefix+uid()

    def _mount(self,temp):
        require(temp.resolve().is_relative_to((self.store.root/'sandbox-jobs').resolve()),'SANDBOX_MOUNT_OUTSIDE_SCOPE')
        host=os.environ.get('SEED_SANDBOX_HOST_ROOT')
        # A container path is not necessarily a Docker daemon host path.
        if not host and Path('/.dockerenv').exists():
            shutil.rmtree(temp,ignore_errors=True)
            require(False,'SANDBOX_HOST_ROOT_REQUIRED',status=503)
        if host:
            require(Path(host).is_absolute() and ',' not in host and '\n' not in host,'SANDBOX_HOST_ROOT_INVALID')
            source=Path(host)/temp.resolve().relative_to(self.store.root.resolve())
        else:source=temp.resolve()
        require(',' not in str(source),'SANDBOX_HOST_ROOT_INVALID')
        return f'type=bind,source={source},target=/work,readonly'

    def _command(self,temp,language,command):
        """Validate host bind contents using trusted inline code before execution."""
        checks={p.name:digest(p.read_bytes()) for p in temp.iterdir() if p.is_file()}
        for path in temp.iterdir():path.chmod(0o644)
        if language in ('python','sql'):
            guard="import hashlib,pathlib;checks="+repr(checks)+";assert all(hashlib.sha256((pathlib.Path('/work')/n).read_bytes()).hexdigest()==h for n,h in checks.items()),'SANDBOX_BIND_MISMATCH';"
            if command[-2]=='-c':return [*command[:-1],guard+command[-1]]
            return ['python','-I','-c',guard+"exec(compile(pathlib.Path('/work/runner.py').read_text(),'runner.py','exec'),{'__name__':'__main__'})"]
        guard="const fs=require('node:fs'),crypto=require('node:crypto');const checks="+json.dumps(checks)+";for(const [n,h] of Object.entries(checks)){if(crypto.createHash('sha256').update(fs.readFileSync('/work/'+n)).digest('hex')!==h)process.exit(97);}"
        if command[:2]==['node','-e']:return ['node','-e',guard+command[2]]
        return ['node','-e',guard+"process.argv=['node','/work/runner.js','/work/solution.js'];require('/work/runner.js');"]

    def _registered(self,kind,value):
        actual=self.store.get(kind,value.get('id',''))
        require(digest(value)==digest(actual),'VERIFIER_REGISTRATION_CHANGED',status=409)
        return actual

    def _start(self,identity):
        with self._container_lock:
            require(not self._cancelled.is_set(),'VERIFIER_CANCELLED',status=503)
            self._containers.add(identity)

    def _clean(self,identity,temp):
        removed=self._remove(identity)
        if removed:
            with self._container_lock:self._containers.discard(identity)
        if temp is not None:shutil.rmtree(temp,ignore_errors=True)

    def _remove(self,identity):
        try:
            result=subprocess.run(['docker','rm','-f',identity],capture_output=True,timeout=3)
            stderr=result.stderr or b''
            if isinstance(stderr,str):stderr=stderr.encode()
            return result.returncode==0 or b'No such container' in stderr
        except (OSError,subprocess.TimeoutExpired):return False

    def cancel(self):
        self._cancelled.set()
        with self._container_lock:identities=list(self._containers)
        removed=True
        for identity in identities:
            if self._remove(identity):
                with self._container_lock:self._containers.discard(identity)
            else:removed=False
        return removed

    def probe(self,profile):
        """Real kernel + Docker inspection evidence, cached for five minutes."""
        profile=self._registered('verifier_profile',profile)
        require(pinned(profile),'PINNED_SANDBOX_IMAGE_REQUIRED')
        pointer=self.store.get('verifier_probe_cache',profile['id'],{})
        cached=self.store.get('verifier_probe',pointer.get('receiptId',''),{})
        if cached.get('profileHash')==digest(profile) and cached.get('expiresAt',0)>time.time():return cached
        if self._remote():return VerifierBroker(self.store).probe(profile)
        require(self.local_capabilities()['available'],'SANDBOX_UNAVAILABLE',status=503)
        identity=self._identity('seed-probe-')
        args=['docker','create','--pull','never','--name',identity,'--network','none','--read-only','--cap-drop','ALL',
              '--security-opt','no-new-privileges',*self._limits(),'--user','65534:65534',
              '--tmpfs','/tmp:rw,noexec,nosuid,size=32m','--entrypoint','python',profile['image'],'-I','-c',PROBE_SCRIPT]
        limits={};passed=False;runtime_image=None;reasons=['SANDBOX_OS_PROBE_FAILED']
        phase='start_guard';failure=None
        try:
            self._start(identity)
            phase='docker_create'
            created=subprocess.run(args,capture_output=True,timeout=PROBE_CREATE_SECONDS)
            require(created.returncode==0,'SANDBOX_PROBE_CREATE_FAILED',status=503)
            phase='docker_inspect'
            inspected=subprocess.run(['docker','inspect',identity],capture_output=True,timeout=3)
            require(inspected.returncode==0,'SANDBOX_PROBE_INSPECT_FAILED',status=503)
            phase='inspect_validation'
            config=json.loads(inspected.stdout)[0];host=config.get('HostConfig',{});runtime_image=config.get('Image')
            inspection=(host.get('NetworkMode')=='none' and host.get('ReadonlyRootfs') is True
                and host.get('PidsLimit')==32 and host.get('Memory')==268435456
                and host.get('MemorySwap')==268435456 and host.get('CpusetCpus')=='0'
                and 'ALL' in [str(v).upper() for v in host.get('CapDrop',[])]
                and any(str(v).split(':')[0]=='no-new-privileges' for v in host.get('SecurityOpt',[]))
                and host.get('Privileged',False) is False and not host.get('CapAdd') and not host.get('Devices')
                and config.get('Config',{}).get('User')=='65534:65534'
                and all(m.get('Type')=='tmpfs' and m.get('Destination')=='/tmp' for m in config.get('Mounts',[])))
            require(inspection,'SANDBOX_INSPECTION_MISMATCH',status=503)
            phase='kernel_probe_start'
            result=subprocess.run(['docker','start','--attach',identity],capture_output=True,timeout=8)
            require(result.returncode==0 and len(result.stdout)<=8192,'SANDBOX_OS_PROBE_FAILED',status=503)
            phase='kernel_probe_validation'
            body=json.loads(result.stdout);limits=body.get('limits',{})
            passed=(body.get('passed') is True and not self._cancelled.is_set()
                and type(limits.get('memoryBytes')) is int and 0<limits['memoryBytes']<=268435456
                and type(limits.get('pids')) is int and 0<limits['pids']<=32
                and limits.get('uid')==65534 and limits.get('cpuset')=='0'
                and all(limits.get(k) is True for k in ('networkOnlyLoopback','rootReadonly','capEffZero',
                    'noNewPrivileges','dockerSocketAbsent','workDataAbsent')))
            limits={**limits,'dockerInspectPassed':True,'hostDataMountsAbsent':True}
            reasons=[] if passed else ['SANDBOX_OS_LIMIT_UNSUPPORTED']
        except Exception as exc:
            reasons=[getattr(exc,'code','SANDBOX_OS_PROBE_FAILED')]
            # Observe infrastructure failures without recording exception text,
            # command arguments, Docker output, paths or generated source.
            failure={'phase':phase,'exceptionType':type(exc).__name__,
                'errno':getattr(exc,'errno',None) if type(getattr(exc,'errno',None)) is int else None,
                'winerror':getattr(exc,'winerror',None) if type(getattr(exc,'winerror',None)) is int else None}
        finally:self._clean(identity,None)
        receipt=self.store.put('verifier_probe',{'profileId':profile['id'],'profileHash':digest(profile),
            'environmentDigest':profile['image'],'imageDigest':profile['image'],'runtimeImageId':runtime_image,
            'passed':passed,'state':'passed' if passed else 'held','reasonCodes':reasons,'limits':limits,
            'failureObservation':failure,
            'probeCreateDeadlineSeconds':PROBE_CREATE_SECONDS,
            'observedAt':now(),'expiresAt':time.time()+(300 if passed else 5)})
        self.store.put('verifier_probe_cache',{'receiptId':receipt['id']},profile['id'])
        return receipt

    def _require_probe(self,profile):
        receipt=self.probe(profile)
        limits=receipt.get('limits',{})
        stored=self.store.get('verifier_probe',receipt.get('id') or '',{})
        require(receipt.get('passed') is True and receipt.get('state')=='passed' and digest(stored)==digest(receipt)
            and receipt.get('profileId')==profile.get('id') and receipt.get('profileHash')==digest(profile)
            and receipt.get('environmentDigest')==profile.get('image') and receipt.get('expiresAt',0)>time.time()
            and type(limits.get('memoryBytes')) is int and 0<limits['memoryBytes']<=268435456
            and type(limits.get('pids')) is int and 0<limits['pids']<=32 and limits.get('uid')==65534
            and limits.get('cpuset')=='0' and all(limits.get(key) is True for key in
                ('networkOnlyLoopback','rootReadonly','capEffZero','noNewPrivileges','dockerSocketAbsent',
                 'workDataAbsent','dockerInspectPassed','hostDataMountsAbsent')),
            'SANDBOX_OS_PROBE_REQUIRED',status=503,reasonCodes=receipt.get('reasonCodes',[]))
        return receipt

    def _execute(self,args,raw,timeout,limit):
        """Bound stdin, wall time and stdout together, including blocked stdin."""
        with tempfile.TemporaryFile() as output:
            process=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=output,stderr=output)
            def feed():
                try:process.stdin.write(raw)
                except (BrokenPipeError,OSError):pass
                finally:
                    try:process.stdin.close()
                    except OSError:pass
            feeder=threading.Thread(target=feed,daemon=True);feeder.start()
            deadline=time.monotonic()+timeout;oversized=False;timed_out=False
            try:
                while process.poll() is None:
                    if output.tell()>limit:oversized=True;break
                    if time.monotonic()>=deadline or self._cancelled.is_set():timed_out=True;break
                    time.sleep(.02)
                if process.poll() is None:process.kill();process.wait(timeout=3)
                output.seek(0);result=output.read(limit+1)
                return process.returncode,result,oversized or len(result)>limit,timed_out
            finally:
                if process.poll() is None:process.kill();process.wait(timeout=3)
                feeder.join(timeout=1)

    def run_python(self,code,inputs,profile):
        """JSON-only sandbox execution; never imports generated code into the host."""
        if self._remote():return VerifierBroker(self.store).run_python(code,inputs,profile)
        require(self.available(),'SANDBOX_UNAVAILABLE',status=503)
        profile=self._registered('verifier_profile',profile)
        require(profile.get('language')=='python' and pinned(profile),'PINNED_SANDBOX_IMAGE_REQUIRED')
        self._require_probe(profile)
        raw=json.dumps(inputs,ensure_ascii=False,allow_nan=False).encode()
        require(len(raw)<=MAX_JSON and 0<len(code.encode())<=MAX_CODE,'SANDBOX_INPUT_TOO_LARGE')
        identity=self._identity('seed-program-');temp=self._temp('seed-program-')
        (temp/'solution.py').write_text(code,'utf-8')
        (temp/'runner.py').write_text("import sys,json,importlib.util\ns=importlib.util.spec_from_file_location('solution','/work/solution.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\nrows=json.load(sys.stdin);values=[]\nfor row in rows:\n try:values.append({'value':getattr(m,row['function'])(*row.get('args',[]),**row.get('kwargs',{}))})\n except Exception as e:values.append({'error':type(e).__name__})\nprint(json.dumps(values,allow_nan=False))\n",'utf-8')
        command=self._command(temp,'python',['python','-I','/work/runner.py'])
        args=['docker','run','--pull','never','--name',identity,'--rm','-i','--network','none','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges',
            *self._limits(),'--user','65534:65534','--tmpfs','/tmp:rw,noexec,nosuid,size=32m',
            '--mount',self._mount(temp),'--workdir','/work','--entrypoint',command[0],profile['image'],*command[1:]]
        try:
            self._start(identity)
            self_code=profile.get('purpose')=='self_code_regression'
            exit_code,result,oversized,timed_out=self._execute(args,raw,5 if self_code else 15,65536 if self_code else MAX_JSON)
            require(not timed_out,'SANDBOX_PROGRAM_TIMEOUT')
            require(exit_code==0 and not oversized,'SANDBOX_PROGRAM_FAILED')
            try:values=_strict_json(result)
            except ValueError:require(False,'SANDBOX_OUTPUT_INVALID')
            require(isinstance(values,list) and len(values)==len(inputs),'SANDBOX_OUTPUT_INVALID')
            return values
        finally:
            self._clean(identity,temp)

    def verify(self,code,fixture):
        if self._remote():return VerifierBroker(self.store).verify(code,fixture)
        fixture=self._registered('fixture',fixture)
        profile=self.store.get('verifier_profile',fixture.get('profileId',''))
        if fixture.get('verifierProfileHash'):
            require(digest(profile)==fixture['verifierProfileHash'] and profile.get('image')==fixture.get('profileImageDigest'),
                'VERIFIER_PROFILE_CHANGED',status=409)
        require(self.available(),'SANDBOX_UNAVAILABLE',status=503)
        require(pinned(profile),'PINNED_SANDBOX_IMAGE_REQUIRED')
        probe=self._require_probe(profile)
        require(isinstance(code,str) and 0<len(code.encode())<=MAX_CODE,'SANDBOX_INPUT_TOO_LARGE')
        require(len(json.dumps(fixture,allow_nan=False).encode())<=MAX_JSON,'SANDBOX_INPUT_TOO_LARGE')
        if fixture.get('cases'):
            return self.verify_cases(code,fixture,profile,probe)
        require(profile.get('language')=='python','VERIFIER_LANGUAGE_UNAVAILABLE',status=503)
        require(isinstance(fixture.get('tests'),str),'FIXTURE_REQUIRED')
        identity=self._identity('seed-verify-');temp=self._temp('seed-verifier-')
        # Only generated code and an administrator-registered fixture enter this volume.
        (temp/'solution.py').write_text(code,'utf-8');(temp/'test_solution.py').write_text(fixture['tests'],'utf-8')
        command=self._command(temp,'python',['python','-I','-c',"import sys,unittest;sys.path.insert(0,'/work');r=unittest.TextTestRunner().run(unittest.defaultTestLoader.discover('/work'));sys.exit(0 if r.wasSuccessful() and r.testsRun>0 else 1)"])
        args=['docker','run','--pull','never','--name',identity,'--rm','--network','none','--read-only','--cap-drop','ALL',
            '--security-opt','no-new-privileges',*self._limits(),'--user','65534:65534',
            '--tmpfs','/tmp:rw,noexec,nosuid,size=32m','--mount',self._mount(temp),
            '--workdir','/work','--entrypoint',command[0],profile['image'],*command[1:]]
        try:
            self._start(identity)
            exit_code,output,oversized,timed_out=self._execute(args,b'',10,MAX_JSON)
            passed=exit_code==0 and not oversized and not timed_out
            state='passed' if passed else 'timeout' if timed_out else 'failed'
            return self.store.put('verifier_receipt',{'codeHash':digest(code.encode()),'fixtureHash':digest(fixture),
                'environmentDigest':profile['image'],'state':state,'passed':state=='passed','observedAt':now(),
                'testOutputStored':False,'probeReceiptId':probe['id']})
        finally:
            self._clean(identity,temp)

    def verify_cases(self,code,fixture,profile,probe=None):
        """Expected answers stay outside the untrusted container/solution process."""
        cases=fixture['cases'];language=profile.get('language')
        require(language in ('python','javascript','typescript','sql'),'VERIFIER_LANGUAGE_UNAVAILABLE',status=503)
        require(isinstance(cases,list) and 1<=len(cases)<=100 and all(sum(k in c for k in ('expected','expectedError','expectedTruthiness'))==1 for c in cases),'INVALID_VERIFIER_CASES')
        require(all('expectedTruthiness' not in c or language=='python' and type(c['expectedTruthiness']) is bool for c in cases),'INVALID_VERIFIER_CASES')
        fixture=self._registered('fixture',fixture)
        require(fixture.get('profileId')==profile.get('id'),'VERIFIER_PROFILE_CHANGED',status=409)
        actual_probe=self._require_probe(profile)
        if probe is not None:require(digest(probe)==digest(actual_probe),'VERIFIER_PROBE_CHANGED',status=409)
        probe=actual_probe
        inputs=[{k:v for k,v in c.items() if k not in ('expected','expectedError','expectedTruthiness')} for c in cases]
        require(len(json.dumps(inputs,allow_nan=False).encode())<=MAX_JSON,'SANDBOX_INPUT_TOO_LARGE')
        identity=self._identity('seed-cases-');temp=self._temp('seed-cases-')
        if language=='python':
            (temp/'solution.py').write_text(code,'utf-8')
            driver="""import sys,json,importlib.util
spec=importlib.util.spec_from_file_location('solution','/work/solution.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
rows=json.load(sys.stdin);result=[]
for row in rows:
 try: result.append({'value':getattr(m,row['function'])(*row.get('args',[]),**row.get('kwargs',{}))})
 except Exception as e: result.append({'error':type(e).__name__})
print(json.dumps(result,allow_nan=False))
"""
            command=['python','-I','/work/runner.py'];(temp/'runner.py').write_text(driver,'utf-8')
        elif language in ('javascript','typescript'):
            (temp/('solution.ts' if language=='typescript' else 'solution.js')).write_text(code,'utf-8')
            driver="""const fs=require('node:fs');
const m=require(process.argv[2]);const rows=JSON.parse(fs.readFileSync(0,'utf8'));
Promise.all(rows.map(async r=>{try{return {value:await m[r.function](...(r.args||[]))};}catch(e){return {error:e.name};}})).then(result=>process.stdout.write(JSON.stringify(result)));
"""
            (temp/'runner.js').write_text(driver,'utf-8')
            if language=='typescript':
                # Compiler is part of the administrator-pinned image, never fetched.
                command=['node','-e',"const cp=require('node:child_process');let r=cp.spawnSync('tsc',['/work/solution.ts','--module','commonjs','--target','es2020','--outDir','/tmp/build'],{stdio:'inherit'});if(r.status)process.exit(r.status);r=cp.spawnSync('node',['/work/runner.js','/tmp/build/solution.js'],{stdio:'inherit'});process.exit(r.status);"]
            else:command=['node','/work/runner.js','/work/solution.js']
        else:
            (temp/'solution.sql').write_text(code,'utf-8')
            driver="""import sys,json,sqlite3
code=open('/work/solution.sql').read();rows=json.load(sys.stdin);result=[]
for row in rows:
 c=sqlite3.connect(':memory:');c.enable_load_extension(False)
 for statement in row.get('setup',[]): c.execute(statement)
 def authorize(action,*args):
  return sqlite3.SQLITE_DENY if action in (sqlite3.SQLITE_ATTACH,sqlite3.SQLITE_DETACH,sqlite3.SQLITE_PRAGMA) else sqlite3.SQLITE_OK
 c.set_authorizer(authorize)
 try:result.append({'value':[list(x) for x in c.execute(code,row.get('params',[])).fetchall()]})
 except Exception as e:result.append({'error':type(e).__name__})
 c.close()
print(json.dumps(result,allow_nan=False))
"""
            (temp/'runner.py').write_text(driver,'utf-8');command=['python','-I','/work/runner.py']
        command=self._command(temp,language,command)
        args=['docker','run','--pull','never','--name',identity,'--rm','-i','--network','none','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges',
            *self._limits(),'--user','65534:65534','--tmpfs','/tmp:rw,noexec,nosuid,size=32m',
            '--mount',self._mount(temp),'--workdir','/work','--entrypoint',command[0],profile['image'],*command[1:]]
        try:
            self._start(identity)
            try:
                self_code=profile.get('purpose')=='self_code_regression'
                exit_code,raw,oversized,timed_out=self._execute(args,json.dumps(inputs,allow_nan=False).encode(),5 if self_code else 10,65536 if self_code else MAX_JSON)
                require(not oversized,'VERIFIER_OUTPUT_TOO_LARGE')
                values=_strict_json(raw) if exit_code==0 and not timed_out else []
                def matches(actual,case):
                    if 'expectedTruthiness' in case:
                        return isinstance(actual,dict) and set(actual)=={'value'} and bool(actual['value']) is case['expectedTruthiness']
                    expected={'error':case['expectedError']} if 'expectedError' in case else {'value':case['expected']}
                    return actual==expected
                passed=exit_code==0 and not timed_out and isinstance(values,list) and len(values)==len(cases) and all(matches(value,case) for value,case in zip(values,cases))
                state='passed' if passed else 'timeout' if timed_out else 'failed'
            except (subprocess.TimeoutExpired,ValueError):state='failed';passed=False
            return self.store.put('verifier_receipt',{'codeHash':digest(code.encode()),'fixtureHash':digest(fixture),'environmentDigest':profile['image'],
                'state':state,'passed':passed,'testCount':len(cases),'independentCases':True,'expectedAnswersExposed':False,'observedAt':now(),
                'probeReceiptId':probe['id']})
        finally:
            self._clean(identity,temp)
