"""Pre-NAS evidence. Software, data, and model capability never share a pass flag."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from LLM.core import Fault, atomic_json, digest, file_hash, inside, now, require, uid

DEFERRED = frozenset((1, 4, 5, 8, 11))
MODEL_REQUIREMENTS = frozenset((6, 7, 9, 10))
SUITES = ('LLM/tests', 'RAG/tests', 'docker/tests', 'tests')
SOURCE_SUFFIXES = {'.py', '.js', '.mjs', '.jsx', '.ts', '.tsx', '.css', '.sh', '.cmd', '.ps1', '.yaml'}
EXCLUDED = {'.git', '.venv', '__pycache__', '.pytest_cache', 'node_modules', '.state',
            'runtime-data', 'data', 'preparation', 'dist', 'tests', '__tests__', 'test'}
NAMES = {
    1:'nas_deployment', 2:'raw_data', 3:'sft_data', 4:'nas_measurement',
    5:'production_training', 6:'independent_quality', 7:'service_acceptance',
    8:'nas_isolation', 9:'self_code', 10:'specialization', 11:'long_running',
    12:'disaster_recovery', 13:'revocation', 14:'interface_security',
    15:'data_selection', 16:'large_corpus', 17:'curriculum',
    18:'task_domains', 19:'offline_transfer', 20:'formats_distillation',
    21:'architecture_growth', 22:'selective_relearning', 23:'storage_lifecycle',
}


def source_snapshot(root=None):
    root=Path(root or Path(__file__).resolve().parents[2]).resolve()
    files={}
    for directory in ('LLM','RAG','Interface/src','Interface/public/licenses','docker'):
        base=root/directory
        if not base.is_dir():continue
        for parent,directories,names in os.walk(base,followlinks=False):
            directories[:]=[name for name in directories if name not in EXCLUDED and not (Path(parent)/name).is_symlink()]
            for name in names:
                path=Path(parent)/name;relative=path.relative_to(root)
                if path.is_symlink() or not path.is_file():continue
                if path.name.startswith('test_') or '.test.' in path.name or '.spec.' in path.name:continue
                allowed=(path.suffix in SOURCE_SUFFIXES or path.name in ('Dockerfile','requirements.txt','requirements.lock','backend-lock.json')
                    or relative.as_posix().startswith('Interface/src/assets/fonts/')
                    or relative.as_posix().startswith('Interface/public/licenses/')
                    or relative.as_posix() in ('docker/ui/package.json','docker/ui/package-lock.json',
                        'docker/ui/ui-toolchain-lock.json','docker/ui/.dockerignore'))
                if allowed:files[relative.as_posix()]=file_hash(path)
    for name in ('Interface/package.json','Interface/package-lock.json','Interface/vite.config.js','.dockerignore'):
        path=root/name
        if path.is_file() and not path.is_symlink():files[name]=file_hash(path)
    require(files,'ACCEPTANCE_SOURCE_MISSING')
    return {'hash':digest(files),'files':files}


def test_snapshot(root):
    root=Path(root).resolve();files={}
    for name in SUITES:
        base=root/name
        if not base.is_dir():continue
        for parent,directories,names in os.walk(base,followlinks=False):
            directories[:]=[d for d in directories if d not in ('__pycache__','.pytest_cache','.venv') and not (Path(parent)/d).is_symlink()]
            for filename in names:
                path=Path(parent)/filename
                if path.suffix=='.py' and not path.is_symlink():files[path.relative_to(root).as_posix()]=file_hash(path)
    return {'hash':digest(files),'files':files}


def _seal(store, kind, body, identity=None):
    value={**body,'schemaVersion':1,'observedAt':now()}
    value['evidenceHash']=digest(value)
    return store.put(kind,value,identity)


def verify_evidence(record, *, expected_source=None):
    body={k:v for k,v in record.items() if k not in ('id','version','createdAt','updatedAt','evidenceHash')}
    require(record.get('schemaVersion')==1 and record.get('evidenceHash')==digest(body),'ACCEPTANCE_EVIDENCE_INVALID',status=409)
    if expected_source:require(record.get('sourceHash')==expected_source,'ACCEPTANCE_SOURCE_CHANGED',status=409)
    return record


def _xml_summary(path):
    require(path.is_file() and path.stat().st_size<=16*1024**2,'ACCEPTANCE_RESULTS_MISSING')
    root=ET.parse(path).getroot()
    cases=list(root.iter('testcase'))
    failed=sum(any(child.tag in ('failure','error') for child in case) for case in cases)
    skipped=sum(any(child.tag=='skipped' for child in case) for case in cases)
    return {'tests':len(cases),'failures':failed,'skipped':skipped,
            'caseIds':sorted(case.get('classname','')+'::'+case.get('name','') for case in cases)}


def run_software(control, *, source_root=None, timeout_seconds=1800):
    """Only this trusted local runner creates software receipts; no score upload API."""
    require(type(timeout_seconds) is int and 1<=timeout_seconds<=7200,'CONFIG_INVALID')
    root=Path(source_root or Path(__file__).resolve().parents[2]).resolve()
    for name in SUITES:require((root/name).is_dir(),'ACCEPTANCE_SUITE_MISSING',suite=name)
    snapshot=source_snapshot(root);tests_snapshot=test_snapshot(root);identity=uid()
    directory=inside(control.store.root,'reports/pre-nas/'+identity);directory.mkdir(parents=True)
    temporary=inside(control.store.root,'staging/pre-nas-'+identity)
    results=directory/'results.xml'
    command=[sys.executable,'-X','utf8','-m','pytest',*SUITES,'-q',
        '--junitxml='+str(results),'--basetemp='+str(temporary),
        '-o','cache_dir='+str(directory/'pytest-cache')]
    started=time.monotonic();reason=None;exit_code=None
    with (directory/'runner.log').open('wb') as output:
        process=subprocess.Popen(command,cwd=root,stdout=output,stderr=output,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        try:
            while process.poll() is None:
                if time.monotonic()-started>timeout_seconds or output.tell()>16*1024**2:
                    reason='ACCEPTANCE_TIME_LIMIT' if time.monotonic()-started>timeout_seconds else 'ACCEPTANCE_OUTPUT_LIMIT'
                    import psutil
                    try:
                        parent=psutil.Process(process.pid)
                        for child in parent.children(recursive=True):
                            try:child.kill()
                            except psutil.NoSuchProcess:pass
                    except psutil.NoSuchProcess:pass
                    process.kill();break
                time.sleep(.25)
            exit_code=process.wait(timeout=10)
        finally:
            if process.poll() is None:process.kill();process.wait(timeout=10)
    summary=_xml_summary(results) if results.exists() else {'tests':0,'failures':0,'skipped':0,'caseIds':[]}
    reasons=[reason] if reason else []
    if source_snapshot(root)['hash']!=snapshot['hash']:reasons.append('ACCEPTANCE_SOURCE_CHANGED')
    if test_snapshot(root)['hash']!=tests_snapshot['hash']:reasons.append('ACCEPTANCE_TESTS_CHANGED')
    if exit_code!=0:reasons.append('SOFTWARE_CHECK_FAILED')
    if summary['tests']<100 or summary['failures'] or summary['skipped']:reasons.append('INCOMPLETE_SOFTWARE_COVERAGE')
    from .acceptance_checks import RUNNER_VERSION
    body={'kind':'software','runnerVersion':RUNNER_VERSION,'sourceHash':snapshot['hash'],'testSourceHash':tests_snapshot['hash'],
        'suiteHash':digest([summary['caseIds'],tests_snapshot['hash']]),
        'suitePaths':list(SUITES),'resultsPath':results.relative_to(control.store.root).as_posix() if results.exists() else None,
        'resultsHash':file_hash(results) if results.exists() else None,'exitCode':exit_code,
        **summary,'elapsedSeconds':time.monotonic()-started,'passed':not reasons,
        'reasonCodes':sorted(set(reasons)),'actualModelQualityPassed':False,'nasModified':False}
    software=_seal(control.store,'foundation_software_acceptance',body,identity)
    if software['passed']:
        from .acceptance_checks import issue_from_software
        issue_from_software(control,software,source_root=root)
    return software


def local_container_status():
    """Inspect availability only. Never connects to a remote Docker daemon."""
    executable=docker_executable()
    if not executable:return {'available':False,'reasonCodes':['LOCAL_CONTAINER_ENGINE_REQUIRED']}
    endpoint=os.environ.get('DOCKER_HOST','')
    if endpoint and not _local_endpoint(endpoint):
        return {'available':False,'reasonCodes':['REMOTE_DOCKER_FORBIDDEN']}
    try:
        selected=subprocess.run([executable,'context','inspect','--format','{{json .Endpoints.docker.Host}}'],capture_output=True,text=True,timeout=5)
        if selected.returncode:return {'available':False,'reasonCodes':['LOCAL_CONTAINER_ENGINE_REQUIRED']}
        host=json.loads(selected.stdout)
        if not _local_endpoint(host):
            return {'available':False,'reasonCodes':['REMOTE_DOCKER_FORBIDDEN']}
        result=subprocess.run([executable,'info','--format','{{json .}}'],capture_output=True,text=True,timeout=8)
        info=json.loads(result.stdout) if result.returncode==0 else {}
        return {'available':info.get('OSType')=='linux','engineVersion':info.get('ServerVersion'),'_endpoint':endpoint or host,
            'reasonCodes':[] if info.get('OSType')=='linux' else ['LINUX_CONTAINER_ENGINE_REQUIRED']}
    except (OSError,ValueError,subprocess.TimeoutExpired):
        return {'available':False,'reasonCodes':['LOCAL_CONTAINER_ENGINE_REQUIRED']}


def _local_endpoint(value):
    if not isinstance(value,str):return False
    value=value.lower()
    return value.startswith(('npipe:////./pipe/','npipe://./pipe/','unix:///'))


def require_local_path(value,root=None):
    """Reject UNC/mapped drives and network mounts before any PC-only write."""
    path=Path(value).resolve()
    require(not str(path).startswith(('\\\\','//')) and (root is None or path.is_relative_to(Path(root).resolve())),
            'LOCAL_PC_DATA_REQUIRED')
    if os.name=='nt':
        import ctypes
        drive=path.drive+'\\'
        require(ctypes.windll.kernel32.GetDriveTypeW(drive) in (3,6),'LOCAL_PC_DATA_REQUIRED')
    else:
        import psutil
        candidates=[row for row in psutil.disk_partitions(all=True) if path.is_relative_to(Path(row.mountpoint).resolve())]
        require(candidates,'LOCAL_STORAGE_PROBE_REQUIRED')
        mount=max(candidates,key=lambda row:len(row.mountpoint))
        require(mount.fstype.lower() not in ('cifs','smbfs','nfs','nfs4','sshfs','fuse.sshfs','9p','afs','davfs','fuse.rclone','fuse.s3fs','gcsfuse'),
                'LOCAL_PC_DATA_REQUIRED')
    return path


def docker_executable():
    found=shutil.which('docker')
    if found:return found
    if os.name=='nt':
        for base in (Path(os.environ.get('LOCALAPPDATA',''))/'Programs/DockerDesktop',
                     Path(os.environ.get('ProgramFiles','C:/Program Files'))/'Docker/Docker'):
            path=base/'resources/bin/docker.exe'
            if path.is_file():return str(path)
    return None


def run_container(control, *, source_root=None, timeout_seconds=3600):
    """Build and exercise only local Linux images; never uses NAS Compose ports."""
    require(type(timeout_seconds) is int and 60<=timeout_seconds<=7200,'CONFIG_INVALID')
    engine=local_container_status()
    require(engine.get('available'),(engine.get('reasonCodes') or ['LOCAL_CONTAINER_ENGINE_REQUIRED'])[0],status=503)
    executable=docker_executable()
    root=Path(source_root or Path(__file__).resolve().parents[2]).resolve()
    require_local_path(root)
    require_local_path(control.store.root)
    snapshot=source_snapshot(root);identity=uid()
    directory=inside(control.store.root,'reports/pre-nas/container-'+identity)
    directory.mkdir(parents=True)
    reasons=[];checks={};started=time.monotonic();container_id=None;ui_container_id=None
    environment_before={name:os.environ.get(name) for name in ('PATH','DOCKER_HOST','DOCKER_CONTEXT')}
    docker_env={**os.environ,'DOCKER_HOST':engine['_endpoint']}
    docker_env.pop('DOCKER_CONTEXT',None)
    def invoke(arguments, *, limit=60):
        remaining=timeout_seconds-(time.monotonic()-started)
        require(remaining>0,'LOCAL_CONTAINER_TIME_LIMIT')
        result=subprocess.run([executable,*arguments],cwd=root,capture_output=True,env=docker_env,
            timeout=min(limit,remaining),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        with (directory/'docker.log').open('ab') as output:
            output.write(result.stdout[-1024**2:]);output.write(result.stderr[-1024**2:])
        require(result.returncode==0,'LOCAL_CONTAINER_COMMAND_FAILED',status=503)
        return result.stdout.decode('utf-8',errors='replace').strip()
    try:
        images={}
        for target in ('runtime','verifier-runtime'):
            iid=directory/(target+'.iid')
            invoke(['build','--file','docker/Dockerfile','--target',target,
                    '--iidfile',str(iid),'.'],limit=timeout_seconds)
            image=iid.read_text('utf-8').strip()
            import re
            require(re.fullmatch(r'sha256:[0-9a-f]{64}',image),'PINNED_IMAGE_REQUIRED')
            images[target]=image
        checks['actualImagesBuilt']=True
        # The actual sandbox probe checks kernel limits, UID, readonly root,
        # capability removal, no socket/work-data, loopback-only networking.
        os.environ['PATH']=str(Path(executable).parent)+os.pathsep+os.environ.get('PATH','')
        os.environ['DOCKER_HOST']=engine['_endpoint'];os.environ.pop('DOCKER_CONTEXT',None)
        control.sandbox._capabilities=None
        profile=control.store.put('verifier_profile',{'image':images['verifier-runtime'],
            'language':'python','purpose':'code','scope':'local_pc_acceptance'})
        probe=control.sandbox.probe(profile)
        checks['kernelSandboxPassed']=probe.get('passed') is True
        require(checks['kernelSandboxPassed'],'SANDBOX_OS_PROBE_FAILED',status=503)
        fixture=control.store.put('fixture',{'profileId':profile['id'],'cases':[{'function':'solve','args':[3],'expected':4}],
             'independent':True,'split':'acceptance','testOnly':True,'neverTrain':True,'purpose':'software_acceptance'})
        receipt=control.sandbox.verify('def solve(x):\n    return x+1\n',fixture)
        checks['actualVerifierFixtureExecuted']=receipt.get('passed') is True
        require(checks['actualVerifierFixtureExecuted'],'LOCAL_VERIFIER_ACCEPTANCE_FAILED')
        backend=invoke(['run','--rm','--network','none','--read-only',
            '--cap-drop','ALL','--security-opt','no-new-privileges:true',
            '--user','65534:65534','--memory','2g','--pids-limit','64',
            '--tmpfs','/tmp:rw,nosuid,size=128m','--entrypoint','python',images['runtime'],
            '-c','import json; from LLM.foundation.backend import verify; print(json.dumps(verify()))'],limit=60)
        checks['linuxBackendHashLockPassed']=json.loads(backend)['targetCompatible'] is True
        # Fresh disposable data, unrelated to existing local application or NAS.
        data=directory/'service-data'
        for name in ('rag','llm','extensions'):(data/name).mkdir(parents=True)
        import secrets
        from RAG.core import Store as RagStore
        from RAG.security import Auth
        rag=RagStore(data/'rag');password=secrets.token_urlsafe(32)
        try:Auth(rag).set_password(password)
        finally:rag.close()
        if os.name!='nt':
            # This fresh disposable test volume contains no application data.
            # Its nonroot service UID must be able to create its own SQLite WAL.
            for path in (data,*data.rglob('*')):
                require(not path.is_symlink() and path.resolve().is_relative_to(data.resolve()),'LOCAL_CONTAINER_DATA_UNSAFE')
                path.chmod(0o777 if path.is_dir() else 0o666)
        container_id=invoke(['run','--rm','-d','--name','seed-local-'+identity,
            '--read-only','--cap-drop','ALL','--security-opt','no-new-privileges:true',
            '--user','65534:65534','--memory','2g','--pids-limit','128',
            '--tmpfs','/tmp:rw,nosuid,size=256m','-p','127.0.0.1::8000',
            '--mount','type=bind,source='+str(data)+',target=/data',
            '--mount','type=bind,source='+str(data/'extensions')+',target=/app/LLM/extensions',
            '-e','SEED_ACCESS_IP=127.0.0.1','-e','SEED_ENABLE_WORKER=0',images['runtime']])
        bindings=json.loads(invoke(['inspect',container_id]))[0]['NetworkSettings']['Ports']['8000/tcp']
        require(len(bindings)==1 and bindings[0]['HostIp']=='127.0.0.1','LOCAL_BIND_REQUIRED')
        base='http://127.0.0.1:'+bindings[0]['HostPort']
        import httpx
        with httpx.Client(base_url=base,timeout=10,trust_env=False) as client:
            deadline=time.monotonic()+90
            while True:
                try:
                    live=client.get('/api/v1/health/live')
                    if live.status_code==200:break
                except httpx.TransportError:pass
                require(time.monotonic()<deadline,'LOCAL_SERVICE_START_FAILED')
                time.sleep(.5)
            checks['serviceLiveness']=True
            checks['interfaceRendered']=client.get('/').status_code==200
            checks['authenticationRequired']=client.get('/api/v1/llm/foundation/projects').status_code==401
            denied=client.get('/api/v1/health/live',headers={'X-Forwarded-Host':'proxy.invalid'})
            checks['forwardedProxyDenied']=denied.status_code==403
            denied=client.get('/api/v1/health/live',headers={'Host':'funnel.invalid'})
            checks['hostnameDenied']=denied.status_code==403
            login=client.post('/api/v1/auth/login',json={'password':password})
            checks['passwordLogin']=login.status_code==200 and login.json().get('authenticated') is True
            checks['csrfRequired']=client.post('/api/v1/llm/foundation/projects',json={'name':'blocked'}).status_code==403
            require(all(checks.values()),'LOCAL_SERVICE_SECURITY_ACCEPTANCE_FAILED')
        from . import ui_acceptance
        ui_context=root/'docker/ui'
        ui_build=ui_acceptance.build(executable,docker_env,ui_context,directory/'ui-build',
            timeout_seconds=min(1200,int(timeout_seconds-(time.monotonic()-started))))
        images['ui-browser']=ui_build['imageId']
        # The browser reaches only a second disposable service's loopback.
        # No host port, original model store, or personal browser profile is used.
        ui_data=directory/'ui-service-data'
        for name in ('rag','llm','extensions'):(ui_data/name).mkdir(parents=True)
        ui_rag=RagStore(ui_data/'rag')
        try:Auth(ui_rag).set_password(password)
        finally:ui_rag.close()
        if os.name!='nt':
            for path in (ui_data,*ui_data.rglob('*')):
                require(not path.is_symlink() and path.resolve().is_relative_to(ui_data.resolve()),'LOCAL_CONTAINER_DATA_UNSAFE')
                path.chmod(0o777 if path.is_dir() else 0o666)
        ui_container_id=invoke(['run','--rm','-d','--name','seed-ui-service-'+identity,
            '--network','none','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges:true',
            '--user','65534:65534','--memory','2g','--pids-limit','128',
            '--tmpfs','/tmp:rw,nosuid,size=256m',
            '--mount','type=bind,source='+str(ui_data)+',target=/data',
            '--mount','type=bind,source='+str(ui_data/'extensions')+',target=/app/LLM/extensions',
            '-e','SEED_ACCESS_IP=127.0.0.1','-e','SEED_ENABLE_WORKER=0',images['runtime']])
        ui_directory=directory/'ui-execution'
        ui_result=ui_acceptance.run(executable,docker_env,image_id=images['ui-browser'],
            service_id=ui_container_id,service_image_id=images['runtime'],password=password,
            source_hash=snapshot['hash'],run_id=identity,output_directory=ui_directory,
            allowed_output_root=directory,service_data_root=ui_data,context=ui_context,build_receipt=ui_build)
        ui_acceptance.verify_result(ui_result,ui_directory,ui_context,snapshot['hash'])
        checks['actualIsolatedBrowserPassed']=ui_result['passed'] is True and ui_result['actualBrowserExecuted'] is True
        require(checks['actualIsolatedBrowserPassed'],'ACTUAL_BROWSER_UI_ACCEPTANCE_REQUIRED')
        password=None
        if source_snapshot(root)['hash']!=snapshot['hash']:reasons.append('ACCEPTANCE_SOURCE_CHANGED')
        return _seal(control.store,'foundation_container_acceptance',{'kind':'local_linux_container',
            'sourceHash':snapshot['hash'],'passed':not reasons,'checks':checks,'images':images,
            'engine':{key:value for key,value in engine.items() if not key.startswith('_')},'probeReceiptId':probe.get('id'),'verifierReceiptId':receipt.get('id'),
            'uiEvidence':ui_result,'uiBuild':ui_build,'uiEvidencePath':ui_directory.relative_to(control.store.root).as_posix(),
            'elapsedSeconds':time.monotonic()-started,'nasModified':False,'nasMeasured':False,
            'actualModelQualityPassed':False,'reasonCodes':reasons},identity)
    except Exception as error:
        reasons.append(getattr(error,'code','LOCAL_CONTAINER_ACCEPTANCE_FAILED'))
        _seal(control.store,'foundation_container_acceptance',{'kind':'local_linux_container',
            'sourceHash':snapshot['hash'],'passed':False,'checks':checks,'engine':{key:value for key,value in engine.items() if not key.startswith('_')},
            'elapsedSeconds':time.monotonic()-started,'nasModified':False,'nasMeasured':False,
            'actualModelQualityPassed':False,'reasonCodes':sorted(set(reasons))},identity)
        raise
    finally:
        if ui_container_id:
            subprocess.run([executable,'rm','-f',ui_container_id],capture_output=True,timeout=20,env=docker_env)
        if container_id:
            subprocess.run([executable,'rm','-f',container_id],capture_output=True,timeout=20,env=docker_env)
        for name,value in environment_before.items():
            if value is None:os.environ.pop(name,None)
            else:os.environ[name]=value


def _latest_valid(store,kind,source_hash):
    values=sorted(store.list(kind),key=lambda r:r.get('observedAt',''),reverse=True)
    for record in values:
        try:return verify_evidence(record,expected_source=source_hash)
        except Fault:continue
    return None


def _current_sft_runtime(control,project,preparation,corpus,tokenizer):
    """Historical execution proof must still name these exact current inputs."""
    from .prepare_data import ACTOR
    from .corpus import verify,rows
    from .tokenizer import load
    from .sft_runtime_verification import source_runtime_program,verified_receipt,_source_cases
    from .source_context_contracts import runtime_contract,runtime_evidence_fields,matches_fields,verify_runtime_source_parent
    from . import sft_runtime_verification
    from LLM.verification.broker import pinned
    from datetime import datetime
    store=control.store
    require(getattr(control,'execution_scope',None)=='pc_validation' and project.get('executionScope')=='pc_validation',
            'SFT_RUNTIME_SCOPE_UNSUPPORTED')
    require(preparation and preparation.get('state')=='ready' and preparation.get('actor')==ACTOR
            and preparation.get('preparationKind')=='delegated_source_context_review'
            and project.get('activeSftPreparationId')==preparation['id']
            and preparation.get('projectId')==project['id'] and preparation.get('sftCorpusId')==corpus['id']
            and preparation.get('sftCorpusHash')==corpus.get('manifestHash')
            and preparation.get('tokenizerId')==project.get('tokenizerId')==tokenizer['id']
            and preparation.get('tokenizerHash')==tokenizer.get('tokenizerHash'),'SFT_REFERENCE_REQUIRED')
    execution=runtime_contract(preparation)
    verify_runtime_source_parent(store,preparation)
    candidates=sorted((row for row in store.list('foundation_sft_runtime_verification')
        if row.get('preparationId')==preparation['id'] and row.get('corpusId')==corpus['id']),
        key=lambda row:row.get('observedAt',''),reverse=True)
    require(candidates,'SFT_REFERENCE_REQUIRED')
    evidence=candidates[0]
    body={key:value for key,value in evidence.items() if key not in ('id','version','createdAt','updatedAt','reportHash')}
    require(evidence.get('reportHash')==digest(body) and evidence.get('state')=='passed' and evidence.get('passed') is True
            and evidence.get('projectId')==project['id'] and evidence.get('corpusHash')==corpus['manifestHash']
            and evidence.get('tokenizerHash')==tokenizer['tokenizerHash'] and evidence.get('executionScope')=='pc_validation'
            and evidence.get('implementationVersion')==execution['runtimeVersion']
            and matches_fields(evidence,runtime_evidence_fields(execution,preparation))
            and evidence.get('implementationHash')==file_hash(Path(sft_runtime_verification.__file__))
            and evidence.get('purpose')=='source_span_data_execution' and evidence.get('modelQualityPassed') is False
            and evidence.get('servingEligible') is False and evidence.get('definitionLoadOnlyIsNotSemanticCorrectness') is True,
            'SFT_RUNTIME_EVIDENCE_CHANGED')
    claimed=evidence.get('rows',[])
    expected_count=corpus.get('splitCounts',{}).get('train',0)+corpus.get('splitCounts',{}).get('dev',0)
    require(isinstance(claimed,list) and all(isinstance(row,dict) for row in claimed)
            and len(claimed)==evidence.get('exampleCount')==evidence.get('definitionVerified')==expected_count
            and expected_count>0,'SFT_RUNTIME_COVERAGE_INCOMPLETE')
    profile=store.get('verifier_profile',evidence.get('profileId') or '')
    require(profile.get('purpose')=='code' and profile.get('language')=='python' and profile.get('scope')=='pc_validation'
            and pinned(profile) and evidence.get('profileHash')==digest(profile)
            and evidence.get('imageDigest')==profile['image'],'SFT_RUNTIME_PROFILE_CHANGED')
    bridge=getattr(control,'bridge',None);require(bridge is not None,'RAG_POLICY_GATEWAY_REQUIRED')
    current=control.foundation.project(project['id']);control.foundation.validate_policy(current)
    require(all(current.get(key)==project.get(key) for key in ('activeSftCorpusId','activeSftPreparationId','tokenizerId')),
            'SFT_RUNTIME_EVIDENCE_CHANGED')
    project=current
    verify(store,corpus);tok=load(store,tokenizer)
    snapshot=bridge.post.snapshot(preparation['ragSnapshotId'])
    require(snapshot.get('status')=='ready' and evidence.get('ragSnapshotId')==snapshot['id']
            and evidence.get('ragSnapshotHash')==snapshot['manifestHash'],'SFT_RUNTIME_EVIDENCE_CHANGED')
    exported=bridge.foundation_sft_corpus(project['id'],snapshot['id'],project['policy'].get('allowedSourceIds',[]))
    authorized={row['exampleId']:row for row in exported['records']}
    expected=[*rows(store,corpus,'train',purpose='sft_acceptance_evidence'),*rows(store,corpus,'dev',purpose='sft_acceptance_evidence')]
    actual=claimed
    require(len(expected)==expected_count==len(actual)==evidence.get('exampleCount')==evidence.get('definitionVerified') and expected_count>0,
            'SFT_RUNTIME_COVERAGE_INCOMPLETE')
    indexed={row.get('exampleId'):row for row in actual};require(len(indexed)==len(actual),'SFT_RUNTIME_COVERAGE_INCOMPLETE')
    semantic_count=semantic_cases=0
    def proof(item,code,fixture,receipt,probe,*,semantic=False):
        passed,verified_probe=verified_receipt(store,receipt,code,fixture,profile)
        require(passed and probe==verified_probe,'SFT_RUNTIME_EXECUTION_FAILED')
        limits=probe.get('limits',{})
        require(type(limits.get('memoryBytes')) is int and 0<limits['memoryBytes']<=268435456
                and type(limits.get('pids')) is int and 0<limits['pids']<=32 and limits.get('uid')==65534
                and limits.get('cpuset')=='0' and all(limits.get(key) is True for key in
                ('networkOnlyLoopback','rootReadonly','capEffZero','noNewPrivileges','dockerSocketAbsent','workDataAbsent',
                 'dockerInspectPassed','hostDataMountsAbsent')),'SFT_RUNTIME_PROBE_INVALID')
        start=datetime.fromisoformat(probe['observedAt'].replace('Z','+00:00')).timestamp()
        observed=datetime.fromisoformat(receipt['observedAt'].replace('Z','+00:00')).timestamp()
        require(start<=observed<=probe['expiresAt'],'SFT_RUNTIME_PROBE_INVALID')
        prefix='semantic' if semantic else 'definition'
        probe_prefix='semanticProbe' if semantic else 'probe'
        require(item.get(prefix+'FixtureId')==fixture['id'] and item.get(prefix+'FixtureHash')==digest(fixture)
                and item.get(prefix+'ReceiptId')==receipt['id'] and item.get(prefix+'ReceiptHash')==digest(receipt)
                and item.get(probe_prefix+'ReceiptId')==probe['id'] and item.get(probe_prefix+'ReceiptHash')==digest(probe),
                'SFT_RUNTIME_EVIDENCE_CHANGED')
    for row in expected:
        identity=row['permissionReceipt']['exampleId'];item=indexed.get(identity);authority=authorized.get(identity)
        require(item and authority and authority['instruction']==row['instruction'] and authority['output']==row['output']
                and all(authority[key]==row[key] for key in ('sourceId','documentId','versionId'))
                and authority.get('ragSplit')==('train' if row['split']=='train' else 'validation'),'POLICY_REVOKED')
        example=bridge.store.get('training_example',identity);version=bridge.store.get('version',row['versionId'])
        require(example.get('reviewedBy')==ACTOR and example.get('approvalKind')=='delegated_source_context_review'
                and example.get('approvalReceipt',{}).get('contextSha256')==row['approvalReceipt']['contextSha256'],'SFT_APPROVAL_REQUIRED')
        runtime_contract(preparation,{'approvalReceipt':example.get('approvalReceipt',{})})
        source=bridge.pre.content(version['id'])
        require(digest(source.encode())==version['rawBlobHash']==row['sourceSpan']['contentSha256'],'SOURCE_CONTENT_CHANGED')
        code,probe_name,function_name,prepared=source_runtime_program(row,source,tok,preparation=preparation)
        require(item.get('definitionVerified') is True and item.get('exampleHash')==row['exampleHash']
                and item.get('sourceVersionId')==version['id'] and item.get('sourceContentHash')==version['rawBlobHash']
                and item.get('sourceFileSha256')==row['sourceSpan']['sourceFileSha256']
                and item.get('sourceSpanHash')==digest(row['sourceSpan']) and item.get('targetHash')==row['sourceSpan']['targetSha256']
                and item.get('contextHash')==prepared['contextSha256'] and item.get('programHash')==digest(code.encode())
                and item.get('totalTokens')==prepared['totalTokens']<=execution['maximumContextTokens']
                and item.get('targetTokens')==prepared['targetTokens']<=execution['maximumTargetTokens'],
                'SFT_RUNTIME_EVIDENCE_CHANGED')
        fixture=store.get('fixture',item.get('definitionFixtureId') or '')
        require(fixture.get('purpose')=='source_definition_load_only' and fixture.get('profileId')==profile['id']
                and fixture.get('verifierProfileHash')==digest(profile) and fixture.get('sourceCorpusHash')==corpus['manifestHash']
                and fixture.get('sourceExampleId')==identity and fixture.get('sourceTargetHash')==row['sourceSpan']['targetSha256']
                and fixture.get('cases')==[{'function':probe_name,'args':[],'kwargs':{},'expected':True}],
                'SFT_RUNTIME_FIXTURE_CHANGED')
        receipt=store.get('verifier_receipt',item.get('definitionReceiptId') or '')
        probe=store.get('verifier_probe',item.get('probeReceiptId') or '');proof(item,code,fixture,receipt,probe)
        cases,_=_source_cases(control,preparation,example,version,function_name)
        require(item.get('semanticCaseCount')==len(cases),'SFT_RUNTIME_SEMANTIC_EVIDENCE_CHANGED')
        if cases:
            semantic=store.get('fixture',item.get('semanticFixtureId') or '')
            require(item.get('semanticVerified') is True and semantic.get('purpose')=='official_source_scalar_cases'
                    and semantic.get('profileId')==profile['id'] and semantic.get('verifierProfileHash')==digest(profile)
                    and semantic.get('cases')==cases and semantic.get('sourceExampleId')==identity
                    and semantic.get('sourceCorpusHash')==corpus['manifestHash']
                    and semantic.get('sourceFixtureId')==example['independentVerification']['fixtureId']
                    and semantic.get('sourceAssertionHash')==example['independentVerification']['assertionBlobHash'],
                    'SFT_RUNTIME_SEMANTIC_EVIDENCE_CHANGED')
            answer=store.get('verifier_receipt',item.get('semanticReceiptId') or '')
            semantic_probe=store.get('verifier_probe',item.get('semanticProbeReceiptId') or '')
            proof(item,code,semantic,answer,semantic_probe,semantic=True);semantic_count+=1;semantic_cases+=len(cases)
        else:require(item.get('semanticVerified') is False,'SFT_RUNTIME_SEMANTIC_EVIDENCE_CHANGED')
    require(evidence.get('runtimeVerified')==semantic_count and evidence.get('semanticCasesExecuted')==semantic_cases
            and evidence.get('semanticCaseFailures')==0,'SFT_RUNTIME_SEMANTIC_EVIDENCE_CHANGED')
    control.foundation.validate_policy(control.foundation.project(project['id']))
    require(store.get('verifier_profile',profile['id'])==profile,'SFT_RUNTIME_PROFILE_CHANGED')
    return {'runtimeEvidenceId':evidence['id'],'runtimeEvidenceHash':evidence['reportHash'],
            'definitionVerified':len(expected),'semanticRuntimeVerified':semantic_count,'semanticCasesExecuted':semantic_cases,
            'fullSemanticCorrectnessProven':False,'claimScope':'pc_source_context_execution'}


def assess(control, project_id=None, *, source_root=None):
    """A blocked model exam stays blocked even when every software test passes."""
    store=control.store;snapshot=source_snapshot(source_root)
    software=_latest_valid(store,'foundation_software_acceptance',snapshot['hash'])
    if software and software.get('testSourceHash')!=test_snapshot(source_root or Path(__file__).resolve().parents[2])['hash']:software=None
    container=_latest_valid(store,'foundation_container_acceptance',snapshot['hash'])
    project=store.get('foundation_project',project_id,{}) if project_id else {}
    checks={n:{'number':n,'name':name,'state':'deferred' if n in DEFERRED else 'blocked',
        'reasonCodes':['USER_DEFERRED'] if n in DEFERRED else ['EVIDENCE_REQUIRED']}
        for n,name in NAMES.items()}
    if project:
        raw=store.get('foundation_corpus',project.get('activeCorpusId') or '',{})
        token=store.get('foundation_tokenizer',project.get('tokenizerId') or '',{})
        counts=raw.get('groupCounts',{});sizes=raw.get('splitBytes',{})
        enough=(not project.get('testOnly') and not raw.get('fixture') and not token.get('testOnly') and raw.get('state')=='ready' and sizes.get('train',0)>=16*1024**2 and counts.get('train',0)>=20
            and token.get('state')=='ready' and token.get('vocabSize',0)==16384)
        preparation=next((r for r in store.list('foundation_data_preparation') if r.get('projectId')==project_id and r.get('state')=='ready'),{})
        measures=preparation.get('requirements',{})
        dev_ready=(counts.get('dev',0)>=20 and measures.get('devTargetTokens',preparation.get('devTargetTokens',0))>=100_000)
        checks[2].update(state='passed' if enough and dev_ready else 'blocked',
            reasonCodes=[] if enough and dev_ready else ['RAW_DATA_REQUIREMENTS_NOT_MET'],corpusId=raw.get('id'),tokenizerId=token.get('id'))
        sft=store.get('foundation_corpus',project.get('activeSftCorpusId') or '',{})
        sft_ready=(not project.get('testOnly') and not sft.get('fixture') and sft.get('state')=='ready' and sft.get('splitCounts',{}).get('train',0)>=32
            and sft.get('groupCounts',{}).get('train',0)>=3 and sft.get('sftTargetTokens',0)>=4096)
        checks[3].update(state='blocked',reasonCodes=['SFT_REFERENCE_REQUIRED'] if sft_ready else ['SFT_DATA_REQUIREMENTS_NOT_MET'],
                         corpusId=sft.get('id'),dataPrepared=bool(sft_ready),runtimeRequired=True,
                         dataState='prepared' if sft_ready else 'insufficient',runtimeState='required',fullSemanticCorrectnessProven=False)
        if sft_ready:
            try:
                active_preparation=store.get('foundation_data_preparation',project.get('activeSftPreparationId') or '',{})
                proof=_current_sft_runtime(control,project,active_preparation,sft,token)
                checks[3].update(**proof,state='passed',reasonCodes=[],runtimeRequired=False,runtimeState='verified')
            except (Fault,KeyError,ValueError,TypeError,OSError) as error:
                checks[3]['reasonCodes']=['SFT_REFERENCE_REQUIRED',getattr(error,'code','SFT_RUNTIME_EVIDENCE_INVALID')]
        for number in MODEL_REQUIREMENTS:
            checks[number]['reasonCodes']=['TRAINED_CANDIDATE_AND_INDEPENDENT_EXAM_REQUIRED']
        releases=[r for r in store.list('foundation_release') if r.get('projectId',r.get('foundationProjectId'))==project_id]
        for release in releases:
            try:
                control.foundation.runtime._release({'releaseId':release['id'],'mode':'code'})
                checks[6].update(state='passed',reasonCodes=[],releaseId=release['id'])
                # Serving still requires a real end-to-end service receipt.
                break
            except Fault:continue
        eligible=[r for r in store.list('foundation_self_code_eligibility') if r.get('projectId')==project_id and r.get('passed') and r.get('state')=='eligible' and not r.get('testOnly')]
        for eligibility in reversed(eligible):
            try:
                control.evolution._foundation_eligibility(store.get('artifact',eligibility['artifactId']))
                require(project.get('state')!='revoked' and not project.get('revocationPending'),'POLICY_REVOKED')
                checks[9].update(state='passed',reasonCodes=[],evaluationId=eligibility.get('evaluationId'));break
            except Fault:continue
        # These are actual-model local runners. A green generic software suite
        # cannot satisfy either service behavior or seven specialized models.
        from .acceptance_model import latest as latest_model_acceptance
        for number,reason in ((7,'ACTUAL_LOCAL_MODEL_SERVICE_ACCEPTANCE_REQUIRED'),
                              (10,'ACTUAL_SEVEN_MODEL_SPECIALIZATION_ACCEPTANCE_REQUIRED')):
            checks[number]['reasonCodes']=[reason]
            try:
                evidence=latest_model_acceptance(control,project_id,number,snapshot['hash'])
                if evidence:
                    checks[number].update(state='passed',reasonCodes=[],evidenceId=evidence['id'],
                        claimScope=evidence['claimScope'],executionScope=evidence['executionScope'])
            except (Fault,OSError,ValueError,KeyError):pass
    from .acceptance_checks import verify_software,verify_requirement,verify_boundary
    context=None
    if software and software.get('passed'):
        try:context=verify_software(store,software,source_root=source_root)
        except Fault:pass
    for number in (12,13,14,15,16,17,18,19,20,21,22,23):
        # A full software suite is necessary; requirement-specific acceptance
        # records are additionally required so a generic green suite cannot
        # certify unimplemented features.
        evidence=_latest_valid(store,'foundation_requirement_acceptance_'+str(number),snapshot['hash'])
        boundary=_latest_valid(store,'foundation_requirement_software_boundary_'+str(number),snapshot['hash'])
        if context and boundary:
            try:
                verify_boundary(store,boundary,software,source_root=source_root,context=context)
                checks[number].update(softwareBoundaryPassed=boundary['softwareBoundaryPassed'],
                    softwareEvidenceId=software['id'],boundaryEvidenceId=boundary['id'],
                    additionalEvidenceRequired=boundary.get('additionalEvidenceRequired',[]))
            except Fault:pass
        if context and evidence and evidence.get('passed'):
            try:
                verify_requirement(store,evidence,software,source_root=source_root,context=context)
                checks[number].update(state='passed',reasonCodes=[],evidenceId=evidence['id'],
                    claimScope=evidence['claimScope'],softwareEvidenceId=software['id'])
            except Fault:checks[number]['reasonCodes']=['CURRENT_REQUIREMENT_ACCEPTANCE_REQUIRED']
        elif not context:checks[number]['reasonCodes']=['CURRENT_SOFTWARE_ACCEPTANCE_REQUIRED']
    if context and container and container.get('passed'):
        try:
            from .ui_acceptance import verify_result
            ui=container.get('uiEvidence') or {}
            require(container.get('checks',{}).get('actualIsolatedBrowserPassed') is True
                and ui.get('imageId')==container.get('images',{}).get('ui-browser')
                and ui.get('serviceImageId')==container.get('images',{}).get('runtime'),
                'ACTUAL_BROWSER_UI_ACCEPTANCE_REQUIRED')
            verify_result(ui,inside(store.root,container['uiEvidencePath']),
                Path(source_root or Path(__file__).resolve().parents[2])/'docker/ui',snapshot['hash'])
            checks[14].update(state='passed',reasonCodes=[],evidenceId=container['id'],
                claimScope='actual_local_linux_interface_and_security',
                softwareEvidenceId=software['id'],actualBrowserExecuted=True)
        except (Fault,KeyError,ValueError,OSError,RuntimeError) as error:
            checks[14]['reasonCodes']=[getattr(error,'code','ACTUAL_BROWSER_UI_ACCEPTANCE_REQUIRED')]
    if project and context:
        from .acceptance_transitions import assess_requirement
        for number in (17,18,20,21):
            observed=assess_requirement(control,number,project_id,software=software)
            checks[number].update(state=observed['state'],reasonCodes=observed['reasonCodes'],
                claimScope=observed['claimScope'],actualEvidenceIds=observed['evidenceIds'],
                actualComponents=observed['components'],failureDetails=observed['failureDetails'],
                availableExecutionPaths=observed['availableExecutionPaths'],
                missingOperationalPaths=observed['missingOperationalPaths'])
    model_pending=[n for n in sorted(MODEL_REQUIREMENTS) if checks[n]['state']!='passed']
    required=[n for n in checks if n not in DEFERRED]
    pending=[n for n in required if checks[n]['state']!='passed']
    reasons=[]
    if pending:reasons.append('PRE_NAS_REQUIREMENTS_INCOMPLETE')
    if not container or not container.get('passed'):reasons.append('ACTUAL_CONTAINER_ACCEPTANCE_REQUIRED')
    return {'schemaVersion':1,'sourceHash':snapshot['hash'],'projectId':project_id,
        'checks':[checks[n] for n in sorted(checks)],'deferredRequirements':sorted(DEFERRED),
        'pendingRequirements':pending,'pendingModelRequirements':model_pending,
        'softwarePassed':bool(context),
        'containerPassed':bool(container and container.get('passed')),
        'readyForRequestedPreNasAcceptance':not reasons,'reasonCodes':reasons,
        'observedAt':now(),'nasModified':False,'executionScope':project.get('executionScope',getattr(control,'execution_scope','nas'))}
