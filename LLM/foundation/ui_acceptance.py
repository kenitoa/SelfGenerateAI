"""Trusted local Linux UI driver, with no host browser or user profile access.

The parent acceptance runner alone builds/runs this driver and seals its result.
Unit fixtures cannot stand in for an actual browser report.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import time
import uuid

CHECKS=frozenset(('loopbackOnly','unauthenticatedLogin','authenticatedSession','desktopNavigation',
    'foundationStages','configurationOnlyProjectCreation','acceptance23Labels','disabledFoundationTrainingControls',
    'stableEmptyOrUnknownProgress','desktop1440NoOverflow','mobile390NoOverflow','mobileNavigation',
    'noWorkOrActivationCreated','mutationBoundary','actualAppAssetsMeasured','noPageErrors','noConsoleErrors',
    'noUnapprovedMutation','offlineKoreanAndLatinFonts'))
IMAGE=re.compile(r'sha256:[0-9a-f]{64}')
HASH=re.compile(r'[0-9a-f]{64}')
FORMAT='{"id":{{json .Id}},"image":{{json .Image}},"user":{{json .Config.User}},"hostConfig":{{json .HostConfig}},"mounts":{{json .Mounts}},"networks":{{json .NetworkSettings.Networks}},"ports":{{json .NetworkSettings.Ports}},"state":{{json .State}}}'
IMAGE_FORMAT='{"id":{{json .Id}},"os":{{json .Os}},"architecture":{{json .Architecture}},"user":{{json (index .Config "User")}},"entrypoint":{{json (index .Config "Entrypoint")}},"layers":{{json .RootFS.Layers}},"repoDigests":{{json .RepoDigests}},"descriptor":{{json (index . "Descriptor")}}}'


class UIError(RuntimeError):
    def __init__(self,code):super().__init__(code);self.code=code


def need(condition,code):
    if not condition:raise UIError(code)


def canonical(value):return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'))
def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024**2),b''):result.update(block)
    return result.hexdigest()


def safe(path):
    path=Path(path).absolute()
    need(not str(path).startswith(('\\\\','//')) and all(not part.is_symlink() for part in (*path.parents,path)),'UI_LOCAL_PATH_REQUIRED')
    return path.resolve()


def child(root,relative):
    need(isinstance(relative,str) and relative and not Path(relative).is_absolute(),'UI_OUTPUT_PATH_INVALID')
    root=safe(root);value=safe(root/relative)
    need(value.is_relative_to(root),'UI_OUTPUT_PATH_INVALID')
    return value


def _local_env(environment):
    result=dict(environment)
    endpoint=result.get('DOCKER_HOST','')
    need(endpoint in ('npipe:////./pipe/dockerDesktopLinuxEngine','npipe:////./pipe/docker_engine',
        'unix:///var/run/docker.sock','unix:///run/docker.sock'),'UI_LOCAL_DOCKER_ENDPOINT_REQUIRED')
    result.pop('DOCKER_CONTEXT',None)
    return result


def _invoke(executable,arguments,environment,log,*,timeout=60,allow_failure=False):
    """Bound output on disk; argv contains no password or authentication token."""
    log=safe(log);log.parent.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    with log.open('wb') as stream:
        process=subprocess.Popen([str(executable),*arguments],stdout=stream,stderr=subprocess.STDOUT,
            env=environment,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        try:
            while process.poll() is None:
                need(time.monotonic()-started<timeout,'UI_CONTAINER_TIME_LIMIT')
                need(log.stat().st_size<=4*1024**2,'UI_CONTAINER_LOG_LIMIT')
                time.sleep(.1)
        except BaseException:
            process.terminate()
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
            raise
    need(log.stat().st_size<=4*1024**2,'UI_CONTAINER_LOG_LIMIT')
    raw=log.read_bytes()
    need(allow_failure or process.returncode==0,'UI_CONTAINER_COMMAND_FAILED')
    return process.returncode,raw.decode('utf-8',errors='replace').strip()


def toolchain(context):
    context=safe(context);lock=json.loads((context/'ui-toolchain-lock.json').read_text('utf-8'))
    need(lock.get('lockHash')==digest({k:v for k,v in lock.items() if k!='lockHash'}),'UI_TOOLCHAIN_LOCK_INVALID')
    need(lock.get('kind')=='official_playwright_ui_toolchain' and lock.get('playwrightVersion')=='1.65.0'
        and lock.get('image',{}).get('reference')=='mcr.microsoft.com/playwright@sha256:bc4fbb329a25287ed23e9216c97063646abdd1bf2da669a9895fa3e2078b2302',
        'UI_OFFICIAL_PIN_REQUIRED')
    names=('Dockerfile','.dockerignore','package.json','package-lock.json','ui-toolchain-lock.json',
        'verify-package.mjs','run-browser.mjs','browsercheck.mjs')
    files={name:file_hash(child(context,name)) for name in names}
    for item in lock['packages'].values():
        path=child(context,item['file']);need(path.stat().st_size==item['sizeBytes'] and file_hash(path)==item['sha256'],'UI_PACKAGE_HASH_MISMATCH')
        files[item['file']]=item['sha256']
    first=(context/'Dockerfile').read_text('utf-8').splitlines()[0]
    need(first=='FROM '+lock['image']['reference'],'UI_OFFICIAL_PIN_REQUIRED')
    return lock,files


def _base_identity(base,manifest_bytes,lock):
    """Bind classic/config and containerd/manifest IDs to the same pinned content."""
    pinned=lock['image'];manifest_digest=pinned['linuxAmd64Digest']
    need(isinstance(manifest_bytes,bytes) and 0<len(manifest_bytes)<=128*1024
        and 'sha256:'+hashlib.sha256(manifest_bytes).hexdigest()==manifest_digest,
        'UI_BASE_MANIFEST_HASH_MISMATCH')
    try:manifest=json.loads(manifest_bytes)
    except (UnicodeDecodeError,ValueError):raise UIError('UI_BASE_MANIFEST_INVALID') from None
    need(manifest.get('schemaVersion')==2 and manifest.get('mediaType') in
        ('application/vnd.oci.image.manifest.v1+json','application/vnd.docker.distribution.manifest.v2+json')
        and manifest.get('config',{}).get('digest')==pinned['configDigest']
        and isinstance(manifest.get('layers'),list) and bool(manifest['layers']),
        'UI_BASE_CONFIG_BINDING_MISMATCH')
    descriptor=base.get('descriptor') or {}
    classic=base.get('id')==pinned['configDigest']
    containerd=base.get('id')==manifest_digest and descriptor.get('digest')==manifest_digest
    need((classic or containerd) and pinned['reference'] in (base.get('repoDigests') or [])
        and base.get('os')==pinned['os']=='linux' and base.get('architecture')==pinned['architecture']=='amd64'
        and isinstance(base.get('layers'),list) and len(base['layers'])==len(manifest['layers'])
        and all(isinstance(value,str) and IMAGE.fullmatch(value) for value in base['layers']),
        'UI_BASE_IMAGE_IDENTITY_MISMATCH')
    return {'baseManifestDigest':manifest_digest,'baseConfigDigest':manifest['config']['digest'],
        'baseIdentityKind':'config' if classic else 'manifest'}


def build(executable,environment,context,output_directory,*,timeout_seconds=1200):
    """Actual digest-pinned build, all RUN commands offline/ignore-scripts."""
    environment=_local_env(environment);context=safe(context);directory=safe(output_directory)
    need(not directory.exists(),'UI_FRESH_OUTPUT_REQUIRED');directory.mkdir(parents=True)
    need(type(timeout_seconds) is int and 1<=timeout_seconds<=7200,'UI_BUILD_TIME_LIMIT_INVALID')
    lock,files=toolchain(context);iid=directory/'browser-image.iid';started=time.monotonic()
    def invoke(arguments,name,limit=60):
        remaining=timeout_seconds-(time.monotonic()-started)
        need(remaining>0,'UI_CONTAINER_TIME_LIMIT')
        return _invoke(executable,arguments,environment,directory/name,timeout=min(limit,remaining))
    # BuildKit's FROM cache is not a daemon image-store entry on fresh engines.
    # Pull only the exact locked Linux manifest, then verify its raw config link.
    invoke(['pull','--platform','linux/amd64',lock['image']['reference']],'base-pull.log',timeout_seconds)
    invoke(['buildx','imagetools','inspect','--raw',lock['image']['reference']],'base-manifest.json')
    invoke(['build','--platform','linux/amd64','--network','none','--file',str(context/'Dockerfile'),
        '--iidfile',str(iid),str(context)],'build.log',timeout_seconds)
    identity=iid.read_text('utf-8').strip();need(IMAGE.fullmatch(identity),'UI_IMAGE_ID_REQUIRED')
    _,raw=invoke(['image','inspect','--format',IMAGE_FORMAT,identity],'image-identity.json')
    image=json.loads(raw)
    _,raw=invoke(['image','inspect','--format',IMAGE_FORMAT,lock['image']['reference']],'base-image-identity.json')
    base=json.loads(raw)
    binding=_base_identity(base,(directory/'base-manifest.json').read_bytes(),lock)
    need(image.get('id')==identity and image.get('os')=='linux' and image.get('architecture')=='amd64'
        and image.get('user')=='1000:1000' and image.get('entrypoint')==['node','/opt/seed-ui/run-browser.mjs']
        and image.get('layers',[])[:len(base['layers'])]==base['layers'],
        'UI_BUILT_IMAGE_IDENTITY_MISMATCH')
    need(toolchain(context)[1]==files,'UI_TOOLCHAIN_CHANGED_DURING_BUILD')
    record={'schemaVersion':1,'kind':'actual_isolated_ui_image_build','passed':True,'imageId':identity,
        'baseImageId':base['id'],'baseImageReference':lock['image']['reference'],**binding,'lockHash':lock['lockHash'],
        'inputFiles':files,'inputHash':digest(files),'networkDuringRunCommands':'none',
        'npmIgnoreScripts':True,'npmOffline':True,'actualDockerBuild':True,'browserExecuted':False,'nasModified':False}
    record['reportHash']=digest(record)
    (directory/'ui-image-build.json').write_text(json.dumps(record,indent=2)+'\n','utf-8')
    return record


def _paths_equal(actual,expected):
    expected=safe(expected);text=str(expected).replace('\\','/')
    aliases={text.rstrip('/')}
    if re.match(r'^[A-Za-z]:/',text):
        aliases.update('/'+prefix+'/'+text[0].lower()+text[2:] for prefix in ('run/desktop/mnt/host','host_mnt'))
    return str(actual).replace('\\','/').rstrip('/').casefold() in {row.casefold() for row in aliases}


def _restricted(body,*,user,memory,pids):
    host=body.get('hostConfig',{})
    need(body.get('user')==user and host.get('ReadonlyRootfs') is True and not host.get('Privileged')
        and 'ALL' in [str(value).upper() for value in host.get('CapDrop',[])] and not host.get('CapAdd')
        and any(str(value).split(':')[0]=='no-new-privileges' for value in host.get('SecurityOpt',[]))
        and type(host.get('Memory')) is int and 0<host['Memory']<=memory
        and type(host.get('PidsLimit')) is int and 0<host['PidsLimit']<=pids
        and not host.get('Devices') and not host.get('DeviceRequests')
        and host.get('PidMode','') in ('','private') and host.get('IpcMode','private')!='host',
        'UI_CONTAINER_ISOLATION_REQUIRED')


def _browser_restricted(body):
    # Chromium 158 reached 128 cgroup tasks on the first about:blank page.
    # Only this trusted UI browser has the measured, fixed 256-task budget.
    # The application (128) and generated-code verifier (32) are unchanged.
    _restricted(body,user='1000:1000',memory=1024**3,pids=256)
    need(body.get('hostConfig',{}).get('PidsLimit')==256,'UI_BROWSER_PIDS_BUDGET_REQUIRED')


DB_PROBE=r'''import sqlite3,json
db=sqlite3.connect('file:/data/llm/metadata/llm.db?mode=ro',uri=True)
kinds=('foundation_genesis','foundation_checkpoint','foundation_run','foundation_tokenizer','foundation_corpus',
 'foundation_teacher_approval','artifact','release','foundation_release','foundation_certificate','foundation_growth_candidate')
counts={kind:db.execute('SELECT count(*) FROM records WHERE kind=?',(kind,)).fetchone()[0] for kind in kinds}
counts['jobs']=db.execute('SELECT count(*) FROM jobs').fetchone()[0]
projects=[]
for identity,raw in db.execute("SELECT id,body FROM records WHERE kind='foundation_project'"):
 row=json.loads(raw);policy=db.execute("SELECT body FROM records WHERE kind='foundation_policy' AND id=?",(identity,)).fetchone()
 policy=json.loads(policy[0]) if policy else {}
 projects.append({key:row.get(key) for key in ('id','name','origin','stage','genesisId','latestCheckpointId','tokenizerId','activeCorpusId','activeSftCorpusId')})
 projects[-1]['automaticLearningEnabled']=policy.get('enabled',False)
db.close()
print(json.dumps({'counts':counts,'projects':projects},sort_keys=True))'''


def verify_outputs(output,source_hash,run_id,context):
    """Validate files made by the trusted actual execution; never accept a score."""
    output=safe(output);lock,_=toolchain(context)
    def load(name,maximum):
        path=child(output,name);need(path.is_file() and path.stat().st_size<=maximum,'UI_REPORT_REQUIRED')
        row=json.loads(path.read_text('utf-8'))
        need(row.get('reportHash')==digest({k:v for k,v in row.items() if k!='reportHash'}),'UI_REPORT_HASH_MISMATCH')
        need(row.get('runId')==run_id and row.get('sourceHash')==source_hash,'UI_REPORT_SOURCE_MISMATCH')
        return row
    runtime=load('runtime-environment.json',128*1024);report=load('browser-report.json',2*1024**2)
    script_hash=file_hash(Path(context)/'browsercheck.mjs')
    need(runtime.get('kind')=='isolated_linux_browser_environment' and runtime.get('passed') is True
        and runtime.get('platform')=='linux' and runtime.get('architecture')=='x64'
        and runtime.get('uid')==1000 and runtime.get('gid')==1000
        and runtime.get('cgroupPidsLimit')==256
        and runtime.get('lockHash')==lock['lockHash'] and runtime.get('scriptHash')==script_hash
        and all(runtime.get(key) is True for key in ('loopbackOnly','readOnlyRoot','capEffZero','noNewPrivileges','privateTmpfs'))
        and runtime.get('packageVersions')=={'playwright':'1.65.0','playwright-core':'1.65.0'}
        and HASH.fullmatch(runtime.get('browserBinarySha256','')),'UI_ACTUAL_BROWSER_ENVIRONMENT_REQUIRED')
    checks=report.get('checks',[])
    expected_version=next(row['browserVersion'] for row in lock['packages']['playwright-core']['chromium'] if row['name']=='chromium')
    need(report.get('kind')=='isolated_playwright_ui_verification' and report.get('state')=='passed'
        and report.get('passed') is True and report.get('executionScope')=='pc_ui_validation'
        and report.get('scriptHash')==script_hash and report.get('browser',{}).get('name')=='chromium'
        and report.get('browser',{}).get('platform')=='linux'
        and report.get('browser',{}).get('version')==expected_version
        and len({row.get('id') for row in checks})==len(checks)
        and CHECKS<={row.get('id') for row in checks} and all(row.get('passed') is True for row in checks)
        and report.get('consoleErrors')==[] and report.get('pageErrors')==[] and report.get('blockedMutations')==[]
        and report.get('modelQualityMeasured') is False and report.get('nasModified') is False,
        'UI_ACTUAL_WORKFLOW_REQUIRED')
    screenshots=report.get('screenshots',[]);viewports=set();files={}
    need(len(screenshots)>=2 and len(report.get('domSnapshots',[]))>=2,'UI_ACTUAL_SCREENSHOTS_REQUIRED')
    for row in screenshots:
        path=child(output,row['path']);need(path.is_file() and path.suffix=='.png' and path.stat().st_size<=16*1024**2,'UI_SCREENSHOT_INVALID')
        with path.open('rb') as image:header=image.read(24)
        need(header[:8]==b'\x89PNG\r\n\x1a\n' and header[12:16]==b'IHDR','UI_SCREENSHOT_INVALID')
        width,height=struct.unpack('>II',header[16:24])
        need(row.get('width')==width and row.get('height')==height and row.get('afterAuthentication') is True
            and row.get('sha256')==file_hash(path),'UI_SCREENSHOT_INVALID')
        if width>=1200:viewports.add('desktop')
        if width<=480:viewports.add('mobile')
        files[row['path']]=row['sha256']
    need(viewports=={'desktop','mobile'},'UI_RESPONSIVE_SCREENSHOTS_REQUIRED')
    for row in report['domSnapshots']:
        path=child(output,row['path']);need(path.is_file() and path.stat().st_size<=2*1024**2
            and row.get('sha256')==file_hash(path),'UI_DOM_SNAPSHOT_INVALID')
        files[row['path']]=row['sha256']
    assets=report.get('assets',[])
    need(any(row.get('resourceType')=='script' and row.get('status')==200 and HASH.fullmatch(row.get('sha256','')) for row in assets)
        and any(row.get('resourceType')=='stylesheet' and row.get('status')==200 and HASH.fullmatch(row.get('sha256','')) for row in assets),
        'UI_ACTUAL_ASSETS_REQUIRED')
    for name in ('runtime-environment.json','browser-report.json'):files[name]=file_hash(output/name)
    return report,runtime,files


def _database_boundary(before,after,report):
    need(not before.get('projects') and all(value==0 for value in before['counts'].values()),'UI_FRESH_SERVICE_DATA_REQUIRED')
    projects=after.get('projects',[])
    need(all(value==0 for value in after['counts'].values()) and len(projects)==1
        and projects[0].get('id')==report.get('createdProject',{}).get('id')
        and projects[0].get('origin')=='random_init' and projects[0].get('automaticLearningEnabled') is False
        and all(projects[0].get(key) is None for key in ('genesisId','latestCheckpointId','tokenizerId','activeCorpusId','activeSftCorpusId')),
        'UI_CONFIGURATION_ONLY_BOUNDARY_FAILED')


def verify_result(record,directory,context,source_hash):
    """Recheck a result inside the parent's source-bound sealed receipt.

    This is not a substitute for the parent's receipt signature/current source
    check. There is no API for submitting this record or caller-provided scores.
    """
    directory=safe(directory);lock,inputs=toolchain(context)
    need(record.get('reportHash')==digest({k:v for k,v in record.items() if k!='reportHash'})
        and record.get('kind')=='actual_isolated_linux_ui_acceptance' and record.get('passed') is True
        and record.get('actualBrowserExecuted') is True and record.get('containerIsolationMeasured') is True
        and record.get('claimScope')=='actual_local_linux_interface_and_security'
        and record.get('executionScope')=='pc_ui_validation' and record.get('sourceHash')==source_hash
        and record.get('toolchainInputHash')==digest(inputs) and record.get('lockHash')==lock['lockHash']
        and record.get('modelQualityMeasured') is False and record.get('nasModified') is False
        and record.get('windowsProfileAccessed') is False,'UI_ACTUAL_RECEIPT_REQUIRED')
    report,runtime,files=verify_outputs(directory/'out',source_hash,record['runId'],context)
    need(record.get('files')==files and record.get('browserReportHash')==report['reportHash']
        and record.get('runtimeReportHash')==runtime['reportHash']
        and record.get('browserBinarySha256')==runtime['browserBinarySha256'],'UI_ACTUAL_RECEIPT_MISMATCH')
    for name,checksum in record.get('supportFiles',{}).items():
        need(file_hash(child(directory,name))==checksum,'UI_SUPPORT_EVIDENCE_CHANGED')
    required={'service-inspection.json','browser-inspection.json','database-before.json','database-after.json','worker-mode.txt'}
    need(required==set(record.get('supportFiles',{})),'UI_SUPPORT_EVIDENCE_REQUIRED')
    before=json.loads((directory/'database-before.json').read_text('utf-8'));after=json.loads((directory/'database-after.json').read_text('utf-8'))
    need(record.get('databaseBefore')==before and record.get('databaseAfter')==after,'UI_SUPPORT_EVIDENCE_CHANGED')
    _database_boundary(before,after,report)
    service=json.loads((directory/'service-inspection.json').read_text('utf-8'))
    browser=json.loads((directory/'browser-inspection.json').read_text('utf-8'))
    _restricted(service,user='65534:65534',memory=2*1024**3,pids=128)
    _browser_restricted(browser)
    need(service.get('image')==record.get('serviceImageId') and browser.get('image')==record.get('imageId')
        and service['hostConfig'].get('NetworkMode')=='none'
        and set(service.get('networks') or {})<={'none'} and not service['hostConfig'].get('PortBindings')
        and browser['hostConfig'].get('NetworkMode')=='container:'+service['id']
        and not browser['hostConfig'].get('PortBindings')
        and (directory/'worker-mode.txt').read_text('utf-8').strip()=='disabled',
        'UI_BROWSER_NETWORK_REQUIRED')
    return record


def run(executable,environment,*,image_id,service_id,service_image_id,password,source_hash,run_id,
        output_directory,allowed_output_root,service_data_root,context,build_receipt,timeout_seconds=240):
    """Run against a SECOND fresh network-none application prepared by parent."""
    need(HASH.fullmatch(source_hash or '') and re.fullmatch('[A-Za-z0-9_-]{1,100}',run_id or ''),'UI_RUN_BINDING_REQUIRED')
    need(IMAGE.fullmatch(image_id or '') and IMAGE.fullmatch(service_image_id or '')
        and re.fullmatch('[0-9a-f]{64}',service_id or ''),'UI_IMAGE_ID_REQUIRED')
    need(isinstance(password,str) and len(password)>=24,'UI_EPHEMERAL_PASSWORD_REQUIRED')
    environment=_local_env(environment);root=safe(allowed_output_root);directory=safe(output_directory)
    need(directory.is_relative_to(root) and not directory.exists(),'UI_FRESH_OUTPUT_REQUIRED')
    directory.mkdir(parents=True);output=directory/'out';output.mkdir();output.chmod(0o777)
    data=safe(service_data_root);need(data.is_relative_to(root),'UI_EPHEMERAL_SERVICE_REQUIRED')
    lock,files=toolchain(context)
    need(build_receipt.get('passed') is True and build_receipt.get('actualDockerBuild') is True
        and build_receipt.get('imageId')==image_id and build_receipt.get('lockHash')==lock['lockHash']
        and build_receipt.get('inputFiles')==files
        and build_receipt.get('reportHash')==digest({k:v for k,v in build_receipt.items() if k!='reportHash'}),
        'UI_ACTUAL_BUILD_REQUIRED')
    def invoke(arguments,name,**kwargs):return _invoke(executable,arguments,environment,directory/name,**kwargs)
    _,raw=invoke(['inspect','--format',FORMAT,service_id],'service-inspection.json');service=json.loads(raw)
    _restricted(service,user='65534:65534',memory=2*1024**3,pids=128)
    need(service.get('image')==service_image_id and service.get('state',{}).get('Running') is True
        and service['hostConfig'].get('NetworkMode')=='none' and not service['hostConfig'].get('PortBindings')
        and set(service.get('networks') or {})<={'none'},'UI_LOOPBACK_SERVICE_REQUIRED')
    binds=[row for row in service.get('mounts',[]) if row.get('Type')=='bind']
    need(len(binds)==2 and all(row.get('Type') in ('bind','tmpfs') for row in service.get('mounts',[])),'UI_EPHEMERAL_SERVICE_REQUIRED')
    expected={'/data':data,'/app/LLM/extensions':data/'extensions'}
    need({row.get('Destination') for row in binds}==set(expected)
        and all(_paths_equal(row.get('Source'),expected[row['Destination']]) for row in binds),'UI_EPHEMERAL_SERVICE_REQUIRED')
    _,worker=invoke(['inspect','--format','{{range .Config.Env}}{{if eq . "SEED_ENABLE_WORKER=0"}}disabled{{end}}{{end}}',service_id],'worker-mode.txt')
    need(worker=='disabled','UI_WORKER_MUST_BE_DISABLED')
    ready_code="import urllib.request,sys; r=urllib.request.build_opener(urllib.request.ProxyHandler({})).open('http://127.0.0.1:8000/api/v1/health/live',timeout=3); sys.exit(0 if r.status==200 else 1)"
    deadline=time.monotonic()+60
    while True:
        code,_=invoke(['exec',service_id,'python','-c',ready_code],'service-ready.log',timeout=10,allow_failure=True)
        if code==0:break
        need(time.monotonic()<deadline,'UI_EPHEMERAL_SERVICE_NOT_READY');time.sleep(.5)
    _,raw=invoke(['exec',service_id,'python','-c',DB_PROBE],'database-before.json');before=json.loads(raw)
    need(not before.get('projects') and all(value==0 for value in before['counts'].values()),'UI_FRESH_SERVICE_DATA_REQUIRED')
    name='seed-ui-'+uuid.uuid4().hex;container_id=None
    private_environment={**environment,'SEED_UI_PASSWORD':password}
    try:
        args=['create','--name',name,'--init','--network','container:'+service_id,'--read-only',
            '--cap-drop','ALL','--security-opt','no-new-privileges:true','--user','1000:1000',
            '--memory','1g','--memory-swap','1g','--pids-limit','256','--cpus','1','--ipc','private',
            '--tmpfs','/tmp:rw,nosuid,nodev,size=768m,mode=1777','--shm-size','64m',
            '--mount','type=bind,source='+str(output)+',target=/out',
            '-e','SEED_UI_PASSWORD','-e','SEED_UI_SOURCE_HASH='+source_hash,'-e','SEED_UI_RUN_ID='+run_id,
            '-e','SEED_UI_SCRIPT_HASH='+files['browsercheck.mjs'],'-e','SEED_UI_BASE_URL=http://127.0.0.1:8000',image_id]
        _,container_id=_invoke(executable,args,private_environment,directory/'browser-create.log')
        need(re.fullmatch('[0-9a-f]{64}',container_id or ''),'UI_CONTAINER_ID_REQUIRED')
        _,raw=invoke(['inspect','--format',FORMAT,container_id],'browser-inspection.json');inspected=json.loads(raw)
        _browser_restricted(inspected)
        need(inspected.get('image')==image_id and inspected['hostConfig'].get('NetworkMode')=='container:'+service_id
            and not inspected['hostConfig'].get('PortBindings'),'UI_BROWSER_NETWORK_REQUIRED')
        mounts=inspected.get('mounts',[]);binds=[row for row in mounts if row.get('Type')=='bind']
        need(len(binds)==1 and binds[0].get('Destination')=='/out' and _paths_equal(binds[0].get('Source'),output)
            and all(row.get('Type') in ('bind','tmpfs') for row in mounts),'UI_BROWSER_PRIVATE_OUTPUT_REQUIRED')
        exit_code,_=invoke(['start','--attach',container_id],'browser-execution.log',timeout=timeout_seconds,allow_failure=True)
        need(exit_code==0,'UI_BROWSER_WORKFLOW_FAILED')
        report,runtime,output_files=verify_outputs(output,source_hash,run_id,context)
        need(password not in (output/'browser-report.json').read_text('utf-8'),'UI_SECRET_IN_REPORT')
        _,raw=invoke(['exec',service_id,'python','-c',DB_PROBE],'database-after.json');after=json.loads(raw)
        _database_boundary(before,after,report)
        need(toolchain(context)[1]==files,'UI_TOOLCHAIN_CHANGED_DURING_EXECUTION')
        record={'schemaVersion':1,'kind':'actual_isolated_linux_ui_acceptance','passed':True,
            'claimScope':'actual_local_linux_interface_and_security','executionScope':'pc_ui_validation',
            'sourceHash':source_hash,'runId':run_id,'scriptHash':files['browsercheck.mjs'],'toolchainInputHash':digest(files),
            'lockHash':lock['lockHash'],'imageId':image_id,'serviceImageId':service_image_id,
            'browserReportHash':report['reportHash'],'runtimeReportHash':runtime['reportHash'],'files':output_files,
            'supportFiles':{name:file_hash(directory/name) for name in ('service-inspection.json','browser-inspection.json',
                'database-before.json','database-after.json','worker-mode.txt')},
            'checks':{row['id']:row['passed'] for row in report['checks']},'browser':report['browser'],
            'databaseBefore':before,'databaseAfter':after,'browserBinarySha256':runtime['browserBinarySha256'],
            'chromiumSandboxEnabled':False,'containerIsolationMeasured':True,'actualBrowserExecuted':True,
            'modelQualityMeasured':False,'nasMeasured':False,'nasModified':False,'windowsProfileAccessed':False}
        record['reportHash']=digest(record)
        (directory/'ui-acceptance.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n','utf-8')
        return record
    finally:
        private_environment.pop('SEED_UI_PASSWORD',None)
        try:invoke(['rm','-f',container_id or name],'browser-cleanup.log',timeout=20,allow_failure=True)
        except (OSError,UIError,subprocess.SubprocessError):pass
