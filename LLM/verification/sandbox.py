"""Untrusted generated code only runs in an explicitly registered sandbox image."""
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from LLM.core import require,digest,uid,now

class Sandbox:
    def __init__(self,store):self.store=store

    def available(self):
        if not shutil.which('docker'):return False
        try:return subprocess.run(['docker','info','--format','{{.ServerVersion}}'],capture_output=True,timeout=5).returncode==0
        except (OSError,subprocess.TimeoutExpired):return False

    def run_python(self,code,inputs,profile):
        """JSON-only sandbox execution; never imports generated code into the host."""
        require(self.available(),'SANDBOX_UNAVAILABLE',status=503)
        require(profile.get('language')=='python' and '@sha256:' in profile.get('image',''),'PINNED_SANDBOX_IMAGE_REQUIRED')
        raw=json.dumps(inputs,ensure_ascii=False,allow_nan=False).encode()
        require(len(raw)<=1024**2 and len(code.encode())<=64000,'SANDBOX_INPUT_TOO_LARGE')
        identity='seed-program-'+uid();temp=Path(tempfile.mkdtemp(prefix='seed-program-'))
        (temp/'solution.py').write_text(code,'utf-8')
        (temp/'runner.py').write_text("import sys,json,importlib.util\ns=importlib.util.spec_from_file_location('solution','/work/solution.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\nrows=json.load(sys.stdin);values=[]\nfor row in rows:\n try:values.append({'value':getattr(m,row['function'])(*row.get('args',[]),**row.get('kwargs',{}))})\n except Exception as e:values.append({'error':type(e).__name__})\nprint(json.dumps(values,allow_nan=False))\n",'utf-8')
        args=['docker','run','--name',identity,'--rm','-i','--network','none','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges',
            '--pids-limit','32','--memory','256m','--cpus','1','--user','65534:65534','--tmpfs','/tmp:rw,noexec,nosuid,size=32m',
            '--mount',f'type=bind,source={temp},target=/work,readonly','--workdir','/work',profile['image'],'python','-I','/work/runner.py']
        try:
            with tempfile.TemporaryFile() as output:
                process=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=output,stderr=output)
                try:
                    process.stdin.write(raw);process.stdin.close();deadline=time.monotonic()+15
                    while process.poll() is None and time.monotonic()<deadline:
                        if output.tell()>1024**2:break
                        time.sleep(.05)
                    if process.poll() is None:process.kill();process.wait(timeout=5)
                    output.seek(0);result=output.read(1024**2+1)
                    require(process.returncode==0 and len(result)<=1024**2,'SANDBOX_PROGRAM_FAILED')
                    try:values=json.loads(result)
                    except ValueError:require(False,'SANDBOX_OUTPUT_INVALID')
                    require(isinstance(values,list) and len(values)==len(inputs),'SANDBOX_OUTPUT_INVALID')
                    return values
                finally:
                    if process.poll() is None:process.kill();process.wait(timeout=5)
        finally:
            subprocess.run(['docker','rm','-f',identity],capture_output=True,timeout=10);shutil.rmtree(temp,ignore_errors=True)

    def verify(self,code,fixture):
        profile=self.store.get('verifier_profile',fixture.get('profileId',''))
        require(self.available(),'SANDBOX_UNAVAILABLE',status=503)
        require('@sha256:' in profile.get('image',''),'PINNED_SANDBOX_IMAGE_REQUIRED')
        if fixture.get('cases'):
            return self.verify_cases(code,fixture,profile)
        require(profile.get('language')=='python','VERIFIER_LANGUAGE_UNAVAILABLE',status=503)
        require(isinstance(fixture.get('tests'),str),'FIXTURE_REQUIRED')
        identity='seed-verify-'+uid();temp=Path(tempfile.mkdtemp(prefix='seed-verifier-'))
        # Only generated code and an administrator-registered fixture enter this volume.
        (temp/'solution.py').write_text(code,'utf-8');(temp/'test_solution.py').write_text(fixture['tests'],'utf-8')
        args=['docker','run','--name',identity,'--rm','--network','none','--read-only','--cap-drop','ALL',
            '--security-opt','no-new-privileges','--pids-limit','32','--memory','256m','--cpus','1','--user','65534:65534',
            '--tmpfs','/tmp:rw,noexec,nosuid,size=32m','--mount',f'type=bind,source={temp},target=/work,readonly',
            '--workdir','/work',profile['image'],'python','-I','-c',
            "import sys,unittest;sys.path.insert(0,'/work');r=unittest.TextTestRunner().run(unittest.defaultTestLoader.discover('/work'));sys.exit(0 if r.wasSuccessful() and r.testsRun>0 else 1)"]
        try:
            with tempfile.TemporaryFile() as out:
                try:
                    result=subprocess.run(args,stdout=out,stderr=subprocess.STDOUT,timeout=10)
                    size=out.tell();out.seek(0);output=out.read(1024**2).decode('utf-8','replace')
                    passed=result.returncode==0 and size<=1024**2
                    state='passed' if passed else 'failed'
                except subprocess.TimeoutExpired:state='timeout';output=''
            return self.store.put('verifier_receipt',{'codeHash':digest(code.encode()),'fixtureHash':digest(fixture),
                'environmentDigest':profile['image'],'state':state,'passed':state=='passed','observedAt':now(),
                'output':output[-4000:]})
        finally:
            subprocess.run(['docker','rm','-f',identity],capture_output=True,timeout=10)
            shutil.rmtree(temp,ignore_errors=True)

    def verify_cases(self,code,fixture,profile):
        """Expected answers stay outside the untrusted container/solution process."""
        cases=fixture['cases'];language=profile.get('language')
        require(language in ('python','javascript','typescript','sql'),'VERIFIER_LANGUAGE_UNAVAILABLE',status=503)
        require(isinstance(cases,list) and 1<=len(cases)<=100 and all('expected' in c or 'expectedError' in c for c in cases),'INVALID_VERIFIER_CASES')
        inputs=[{k:v for k,v in c.items() if k not in ('expected','expectedError')} for c in cases]
        identity='seed-cases-'+uid();temp=Path(tempfile.mkdtemp(prefix='seed-cases-'))
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
        args=['docker','run','--name',identity,'--rm','-i','--network','none','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges',
            '--pids-limit','32','--memory','256m','--cpus','1','--user','65534:65534','--tmpfs','/tmp:rw,noexec,nosuid,size=32m',
            '--mount',f'type=bind,source={temp},target=/work,readonly','--workdir','/work',profile['image'],*command]
        try:
            try:
                with tempfile.TemporaryFile() as output:
                    process=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=output,stderr=output)
                    process.stdin.write(json.dumps(inputs,allow_nan=False).encode());process.stdin.close()
                    deadline=time.monotonic()+10;oversized=False
                    while process.poll() is None and time.monotonic()<deadline:
                        if output.tell()>1024**2:oversized=True;break
                        time.sleep(.05)
                    if process.poll() is None:process.kill();process.wait(timeout=5)
                    output.seek(0);raw=output.read(1024**2+1)
                    require(not oversized and len(raw)<=1024**2,'VERIFIER_OUTPUT_TOO_LARGE')
                    values=json.loads(raw) if process.returncode==0 else []
                expected=[{'error':r['expectedError']} if 'expectedError' in r else {'value':r['expected']} for r in cases]
                passed=process.returncode==0 and values==expected
                state='passed' if passed else 'failed'
            except (subprocess.TimeoutExpired,ValueError):state='failed';passed=False
            return self.store.put('verifier_receipt',{'codeHash':digest(code.encode()),'fixtureHash':digest(fixture),'environmentDigest':profile['image'],
                'state':state,'passed':passed,'testCount':len(cases),'independentCases':True,'expectedAnswersExposed':False,'observedAt':now()})
        finally:
            subprocess.run(['docker','rm','-f',identity],capture_output=True,timeout=10);shutil.rmtree(temp,ignore_errors=True)
