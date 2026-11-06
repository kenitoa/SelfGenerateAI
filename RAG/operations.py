"""Durable worker, bounded collectors, observability, backups and schedules."""
from __future__ import annotations

import gzip
import hashlib
import http.client
import ipaddress
import json
import logging
import os
import re
import shutil
import socket
import ssl
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse, quote
from zoneinfo import ZoneInfo

import apsw
import psutil

from RAG.core import DomainError, digest, now, uid

log = logging.getLogger("seed.jobs")
KST = timezone(timedelta(hours=9))


def system_status(store):
    memory = psutil.virtual_memory()
    disk = shutil.disk_usage(store.root)
    scope = "host"
    total, available = memory.total, memory.available
    for base in (Path("/sys/fs/cgroup"),):
        try:
            limit = (base / "memory.max").read_text().strip()
            used = int((base / "memory.current").read_text().strip())
            if limit != "max" and int(limit) < total:
                total, available, scope = int(limit), max(0, int(limit)-used), "container"
        except (OSError, ValueError):
            pass
    return {"observedAt": now(), "stale": False, "dataOrigin": "server", "observationScope": scope,
            "cpuObservationScope": "host", "memoryObservationScope": scope,
            "platform": os.name, "device": "NAS" if os.name != "nt" else "로컬 개발 PC",
            "cpuPercent": psutil.cpu_percent(interval=None),
            "memory": {"totalBytes": total, "availableBytes": available, "usedBytes": total-available,
                       "processRssBytes": psutil.Process().memory_info().rss},
            "disk": {"totalBytes": disk.total, "freeBytes": disk.free, "usedBytes": disk.used},
            "configuredMemoryGb": store.setting("settings", {}).get("memory", "2"), "sqliteVersion": store.sqlite_version}


def resource_guard(store, heavy=True):
    status = system_status(store)
    if status["disk"]["freeBytes"] < max(2 * 1024**3, status["disk"]["totalBytes"] * .05):
        raise DomainError("DISK_LOW", "저장 공간이 부족합니다. 새 수집·색인을 중지했습니다.", 503)
    if heavy and status["memory"]["availableBytes"] < 384 * 1024**2:
        raise DomainError("RESOURCE_PRESSURE", "메모리 여유가 부족합니다.", 503)
    try:configured=float(store.setting('settings',{}).get('memory','2'))*1024**3
    except (ValueError,TypeError):configured=2*1024**3
    if heavy and status['memory']['processRssBytes']>configured:
        raise DomainError("RESOURCE_PRESSURE","프로세스 메모리 설정 한도를 초과했습니다. 신규 작업을 보류합니다.",503)
    return status


class PinnedHTTPS(http.client.HTTPSConnection):
    """Connect to a previously validated IP while retaining hostname TLS/SNI."""
    def __init__(self, hostname, ip, port=443, timeout=15):
        super().__init__(hostname, port=port, timeout=timeout, context=ssl.create_default_context())
        self.pinned_ip = ip

    def connect(self):
        raw = socket.create_connection((self.pinned_ip, self.port), self.timeout)
        self.sock = self._context.wrap_socket(raw, server_hostname=self.host)


def validate_public_url(url, allowed_hosts=None):
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
        raise DomainError("SOURCE_BLOCKED", "공개 http/https 주소만 수집할 수 있습니다.", 422)
    if parsed.port not in (None, 80, 443):
        raise DomainError("SOURCE_BLOCKED", "웹 수집 포트는 80/443만 허용합니다.", 403)
    host = parsed.hostname.rstrip(".").lower()
    if allowed_hosts and host not in {str(h).rstrip('.').lower() for h in allowed_hosts}:
        raise DomainError("SOURCE_BLOCKED", "허용되지 않은 수집 호스트입니다.", 403)
    try:
        addresses = list(dict.fromkeys(row[4][0] for row in socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)))
    except OSError as exc:
        raise DomainError("SOURCE_UNAVAILABLE", "수집 주소를 확인할 수 없습니다.", 503) from exc
    if not addresses or any(not ipaddress.ip_address(ip).is_global for ip in addresses):
        raise DomainError("SOURCE_BLOCKED", "사설·로컬·메타데이터 주소는 수집할 수 없습니다.", 403)
    return parsed, addresses[0]


def fetch_public(url, *, headers=None, allowed_hosts=None, max_bytes=5*1024**2):
    current = url
    for redirect in range(4):
        parsed, ip = validate_public_url(current, allowed_hosts)
        if parsed.scheme == "https":
            connection = PinnedHTTPS(parsed.hostname, ip, parsed.port or 443)
        else:
            connection = http.client.HTTPConnection(ip, parsed.port or 80, timeout=15)
        request_headers = {"Host": parsed.netloc, "User-Agent": "SEED-RAG/1.0 (bounded personal indexer)", "Accept-Encoding": "identity", **(headers or {})}
        try:
            connection.request("GET", (parsed.path or "/") + ("?"+parsed.query if parsed.query else ""), headers=request_headers)
            response = connection.getresponse()
            if response.status in (301,302,303,307,308):
                if redirect == 3 or not response.getheader("Location"):
                    raise DomainError("SOURCE_BLOCKED", "리다이렉트 한도를 초과했습니다.", 422)
                current = urljoin(current, response.getheader("Location"))
                continue
            response_headers = {key.lower(): value for key,value in response.getheaders()}
            if response.status == 304:
                return {"status":304, "body":b"", "headers":response_headers, "url":current}
            if response.status != 200:
                raise DomainError("SOURCE_HTTP_ERROR", f"수집 서버 응답 오류 ({response.status})", 503 if response.status in (429,500,502,503,504) else 422, {"httpStatus":response.status})
            if int(response.getheader("Content-Length") or 0) > max_bytes:
                raise DomainError("SOURCE_TOO_LARGE", "수집 문서 크기 상한을 초과했습니다.", 413)
            body = response.read(max_bytes+1)
            if len(body) > max_bytes:
                raise DomainError("SOURCE_TOO_LARGE", "수집 문서 크기 상한을 초과했습니다.", 413)
            encoding = response_headers.get("content-encoding", "identity")
            if encoding != "identity":
                if encoding != "gzip":
                    raise DomainError("UNSUPPORTED_ENCODING", "지원하지 않는 압축 응답입니다.", 422)
                import io
                with gzip.GzipFile(fileobj=io.BytesIO(body)) as compressed:
                    body = compressed.read(max_bytes+1)
                if len(body)>max_bytes:
                    raise DomainError("SOURCE_TOO_LARGE", "압축 해제 후 크기 상한을 초과했습니다.", 413)
            return {"status":200,"body":body,"headers":response_headers,"url":current}
        except DomainError:
            raise
        except (OSError, http.client.HTTPException, ValueError) as exc:
            raise DomainError("SOURCE_UNAVAILABLE", "수집 서버 연결에 실패했습니다.", 503) from exc
        finally:
            connection.close()
    raise DomainError("SOURCE_BLOCKED", "리다이렉트 오류", 422)


class Operations:
    def __init__(self, store, pre, post):
        self.store, self.pre, self.post = store, pre, post

    def sync_source(self, id):
        source = self.store.get("source", id)
        if source.get('kind') == 'local_directory':
            raise DomainError('FOLDER_COLLECTION_REMOVED', 'NAS/로컬 폴더 수집은 지원하지 않습니다.', 422)
        if not source.get("collectionEnabled", True):
            raise DomainError("SOURCE_DISABLED", "수집이 중지된 소스입니다.", 409)
        return self.store.enqueue("source_sync", {"sourceId": id}, key=f"sync:{id}:{uid()}", priority=30)

    def collect(self, job):
        source = self.store.get("source", job["payload"]["sourceId"])
        if source.get('kind') == 'local_directory':
            raise DomainError('FOLDER_COLLECTION_REMOVED', 'NAS/로컬 폴더 수집은 지원하지 않습니다.', 422)
        if not source.get("collectionEnabled", True):
            raise DomainError("SOURCE_DISABLED", "수집이 중지된 소스입니다.", 409)
        if source.get('crawlerManaged'):
            from RAG.Pre.crawling.service import CrawlerService
            return CrawlerService(self.store,self.pre).process_job({**job,'payload':{**job.get('payload',{}),'targetSourceId':source['id']}})
        kind = source.get("kind", "web_document")
        settings = self.store.setting("settings", {})
        allowed = settings.get("allowedSourceHosts") or None
        self.store.update_job(job["id"],stage="fetching", unit="documents")
        locator = source.get("locator", {})
        if isinstance(locator, str):
            locator = {"url":locator}
        if kind == "web_document":
            headers = {}
            if source.get("etag"): headers["If-None-Match"] = source["etag"]
            if source.get("lastModified"): headers["If-Modified-Since"] = source["lastModified"]
            result = fetch_public(locator.get("url") or source.get("url", ""), headers=headers, allowed_hosts=allowed)
            if result["status"] == 304:
                self.store.put("source", {"lastSyncedAt":now()}, id=source["id"])
                return {"unchanged":True}
            content_type = result["headers"].get("content-type", "text/plain").lower()
            if not any(x in content_type for x in ("text/", "json", "xml")):
                raise DomainError("UNSUPPORTED_MEDIA_TYPE", "텍스트 문서만 수집할 수 있습니다.", 422)
            name = Path(urlparse(result["url"]).path).name or "index.html"
            if "." not in name: name += ".html" if "html" in content_type else ".txt"
            files = [(name,result["body"])]
            revision = hashlib.sha256(result["body"]).hexdigest()
            output = self.pre.ingest_source_files(source["id"],files,revision=revision)
            self.store.put("source", {"lastSyncedAt":now(),"etag":result["headers"].get("etag"),"lastModified":result["headers"].get("last-modified"),"lastSuccessfulRevision":revision},id=source["id"])
            return output
        if kind == "git_repository":
            parsed = urlparse(locator.get("url", ""))
            parts = parsed.path.strip("/").removesuffix(".git").split("/")
            if parsed.hostname != "github.com" or len(parts)!=2:
                raise DomainError("SOURCE_BLOCKED", "초기 저장소 수집은 공개 GitHub owner/repo 주소를 지원합니다.", 422)
            owner,repo = map(lambda p: quote(p,safe=""),parts)
            ref = quote(locator.get("ref", "HEAD"),safe="")
            base=f"https://api.github.com/repos/{owner}/{repo}"
            repository=json.loads(fetch_public(base,allowed_hosts=allowed,max_bytes=1024**2)["body"])
            # GitHub's source repository identifies fork families. Derive this
            # from provider metadata rather than trusting an export caller label.
            origin=repository.get('source') or repository.get('parent') or repository
            family=origin.get('html_url') or f"https://github.com/{owner}/{repo}"
            family_url=urlparse(family)
            if family_url.scheme=='https' and family_url.hostname=='github.com' and len(family_url.path.strip('/').split('/'))==2:
                source=self.store.put('source',{'repositoryFamily':family.rstrip('/').removesuffix('.git')},id=source['id'])
            commit=json.loads(fetch_public(base+"/commits/"+ref,allowed_hosts=allowed,max_bytes=1024**2)["body"])
            revision=commit["sha"]
            if source.get("lastSuccessfulRevision")==revision:
                return {"unchanged":True,"revision":revision}
            tree=json.loads(fetch_public(base+"/git/trees/"+revision+"?recursive=1",allowed_hosts=allowed,max_bytes=5*1024**2)["body"])
            if tree.get("truncated"):
                raise DomainError("SOURCE_TOO_LARGE", "저장소 트리가 너무 큽니다. 경로 범위를 줄이세요.", 422)
            import fnmatch
            excluded={"node_modules","vendor","dist",".git","build","__pycache__"}
            extensions={".py",".js",".jsx",".ts",".tsx",".md",".txt",".json",".jsonl",".sql",".html",".css",".toml",".yml",".yaml"}
            files,total=[],0
            for entry in tree.get("tree",[]):
                path=entry["path"]
                if entry.get("type")!="blob" or entry.get("mode")!="100644" or Path(path).suffix.lower() not in extensions: continue
                if any(p in excluded or p.startswith('.env') for p in path.split('/')) or path.endswith(("lock.json","lock.yaml",".lock")): continue
                if source.get("includePaths") and not any(fnmatch.fnmatch(path,p) for p in source["includePaths"]): continue
                if any(fnmatch.fnmatch(path,p) for p in source.get("excludePaths",[])): continue
                if len(files)>=1000: raise DomainError("SOURCE_TOO_LARGE", "저장소 파일 수 상한을 초과했습니다.", 413)
                if self.store.job(job["id"])["state"]=="cancelling": raise DomainError("CANCELLED", "수집이 중단됐습니다.", 409)
                data=fetch_public(f"https://raw.githubusercontent.com/{owner}/{repo}/{revision}/{quote(path,safe='/')}",allowed_hosts=allowed,max_bytes=2*1024**2)["body"]
                total+=len(data)
                if total>50*1024**2: raise DomainError("SOURCE_TOO_LARGE", "저장소 수집 용량 상한을 초과했습니다.", 413)
                files.append((path,data))
                self.store.update_job(job["id"],processedItems=len(files),unit="files")
            output=self.pre.ingest_source_files(source["id"],files,revision=revision)
            self.store.put("source",{"lastSyncedAt":now(),"lastSuccessfulRevision":revision},id=source["id"])
            return output
        raise DomainError("UNSUPPORTED_SOURCE", "직접 업로드 소스는 파일 추가로 갱신하세요.", 422)

    def backup(self, job):
        identity=job["payload"].get("backupId",job['id'])
        if not isinstance(identity,str) or not re.fullmatch(r'[a-fA-F0-9-]{36}',identity):
            raise DomainError("INVALID_BACKUP","잘못된 백업 ID입니다.",422)
        backup_root=(self.store.root/"backups").resolve()
        target=(backup_root/identity).resolve()
        if target.parent!=backup_root or target.is_symlink():
            raise DomainError("INVALID_BACKUP","백업 경로 오류입니다.",422)
        if target.exists():
            if (target/"manifest.json").exists():return self.store.get("backup",identity)
            shutil.rmtree(target)
        target.mkdir(parents=True)
        self.store.put("backup",{"status":"running","path":str(target)},id=identity)
        # Runtime uses one writer worker; RLock quiesces API writes + pinning.
        with self.store.lock:
            destination=apsw.Connection(str(target/"seed.db"))
            source=self.store.connect()
            try:
                with destination.backup("main",source,"main") as backup:
                    while not backup.done:backup.step(128)
            finally:
                source.close();destination.close()
            # No blob/index GC runs while this job is active. The entire immutable
            # file set is pinned by backup operation and copied before release.
            file_hashes={}
            for dirname in ("blobs","quarantine","indexes","exports"):
                source_path=self.store.root/dirname
                for file in source_path.rglob("*"):
                    if not file.is_file() or file.is_symlink():continue
                    relative=file.relative_to(self.store.root)
                    out=target/relative;out.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(file,out)
                    file_hashes[relative.as_posix()]=hashlib.sha256(out.read_bytes()).hexdigest()
            file_hashes["seed.db"]=hashlib.sha256((target/"seed.db").read_bytes()).hexdigest()
            manifest={"id":identity,"createdAt":now(),"schemaVersion":1,"files":file_hashes,"complete":True}
            temp=target/"manifest.tmp";temp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf8");os.replace(temp,target/"manifest.json")
        result=self.store.put("backup",{"status":"completed","completedAt":now(),"fileCount":len(file_hashes),"manifestHash":digest(manifest)},id=identity)
        self.store.activity("backup_completed",identity,"정합 백업 완료")
        completed=sorted([b for b in self.store.list("backup") if b.get("status")=="completed"],key=lambda x:x["createdAt"],reverse=True)
        for old in completed[7:]:
            oldpath=(self.store.root/"backups"/old["id"]).resolve()
            if oldpath.parent == (self.store.root/"backups").resolve():
                shutil.rmtree(oldpath,ignore_errors=True);self.store.delete("backup",old["id"])
        return result

    def restore(self, backup_id, destination):
        source=(self.store.root/"backups"/backup_id).resolve()
        if source.parent != (self.store.root/"backups").resolve():raise DomainError("INVALID_BACKUP","잘못된 백업 ID",422)
        manifest=json.loads((source/"manifest.json").read_text(encoding="utf8"))
        if not manifest.get("complete"):raise DomainError("INCOMPLETE_BACKUP","미완료 백업입니다.",422)
        dest=Path(destination).resolve()
        if dest.exists() and any(dest.iterdir()):raise DomainError("RESTORE_TARGET_NOT_EMPTY","복원 대상은 빈 디렉터리여야 합니다.",409)
        for relative,expected in manifest["files"].items():
            relative_path=Path(relative)
            if relative_path.is_absolute() or '..' in relative_path.parts or '\\' in relative or re.match(r'^[A-Za-z]:',relative):
                raise DomainError("INVALID_BACKUP","백업 상대 경로 오류",422)
            original=(source/relative).resolve()
            if not original.is_relative_to(source) or original.is_symlink():raise DomainError("INVALID_BACKUP","백업 경로 오류",422)
            if hashlib.sha256(original.read_bytes()).hexdigest()!=expected:raise DomainError("BACKUP_CORRUPT","백업 해시가 일치하지 않습니다.",422)
        for relative in manifest["files"]:
            out=dest/("metadata/seed.db" if relative=="seed.db" else relative)
            if not out.resolve().is_relative_to(dest):raise DomainError("INVALID_BACKUP","복원 대상 경로 탈출",422)
            out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/relative,out)
        connection=apsw.Connection(str(dest/"metadata/seed.db"))
        try:
            if connection.execute("PRAGMA integrity_check").fetchone()[0]!="ok":raise DomainError("BACKUP_CORRUPT","DB 무결성 검사 실패",422)
        finally:connection.close()
        return {"restored":True,"destination":str(dest),"backupId":backup_id}

    def evaluate(self, job):
        from RAG.evaluation.service import EvaluationService
        result=EvaluationService(self.store,self.pre).run(job)
        return self.store.put('evaluation',{'observation':system_status(self.store)},id=result['id'])

    def schedules(self):return self.store.list("schedule")

    def ensure_maintenance(self):
        if not self.store.setting('defaultBackupScheduleCreated',False):
            self.save_schedule({'kind':'backup','time':'02:30','enabled':True,'system':True,'payload':{}},id='system-daily-backup')
            self.store.set_setting('defaultBackupScheduleCreated',True)

    def maintenance(self):
        stamp=time.time()
        if stamp-float(self.store.setting('maintenanceLastAt',0))<3600:return
        event_cutoff=(datetime.now(timezone.utc)-timedelta(days=7)).isoformat().replace('+00:00','Z')
        activity_cutoff=(datetime.now(timezone.utc)-timedelta(days=90)).isoformat().replace('+00:00','Z')
        with self.store.tx() as db:
            db.execute('DELETE FROM activities WHERE created_at<?',(activity_cutoff,))
            db.execute("DELETE FROM records WHERE kind='run_event' AND created_at<?",(event_cutoff,))
        temporary_root=(self.store.root/'uploads'/'tmp').resolve()
        for file in temporary_root.rglob('*'):
            resolved=file.resolve()
            if file.is_symlink() or not resolved.is_relative_to(temporary_root) or not file.is_file():continue
            if file.stat().st_mtime<stamp-86400:file.unlink()
        self.store.set_setting('maintenanceLastAt',stamp)

    def save_schedule(self, body, id=None):
        kind=body.get("kind")
        if kind not in ("source_sync","reindex","backup","evaluate"):
            raise DomainError("INVALID_SCHEDULE","지원하는 예약 종류가 아닙니다. 실제 모델 학습은 별도 서버가 담당합니다.",422)
        hour_minute=body.get("time")
        if not hour_minute and body.get("cron"):
            fields=str(body['cron']).split()
            if len(fields)!=5 or fields[2:]!=['*','*','*']:
                raise DomainError("INVALID_SCHEDULE","초기 예약은 매일 실행하는 분 시 * * * 형식만 지원합니다.",422)
            hour_minute=f"{fields[1]}:{fields[0]}"
        hour_minute=hour_minute or "02:00"
        try:
            hour,minute=map(int,hour_minute.split(':'))
            if not (0<=hour<24 and 0<=minute<60):raise ValueError()
        except (ValueError,AttributeError):raise DomainError("INVALID_SCHEDULE","예약 시각은 HH:MM 형식이어야 합니다.",422)
        payload=body.get('payload',{})
        if not isinstance(payload,dict):raise DomainError("INVALID_SCHEDULE","예약 payload는 객체여야 합니다.",422)
        if kind=='source_sync':
            source_id=payload.get('sourceId') or body.get('sourceId')
            source=self.store.get('source',source_id)
            if source.get('kind') not in ('web_document','git_repository'):raise DomainError("INVALID_SCHEDULE","웹 문서·공개 GitHub 저장소만 수집 예약할 수 있습니다.",422)
            payload={**payload,'sourceId':source_id}
        return self.store.put("schedule",{**body,"kind":kind,"payload":payload,"time":f"{hour:02}:{minute:02}","cron":f"{minute} {hour} * * *","timezone":"Asia/Seoul","enabled":body.get("enabled",True),"nextRunAt":self.next_time(hour,minute),"misfirePolicy":"coalesce"},id=id)

    def next_time(self,hour,minute):
        current=datetime.now(KST);following=current.replace(hour=hour,minute=minute,second=0,microsecond=0)
        if following<=current:following+=timedelta(days=1)
        return following.astimezone(timezone.utc).isoformat().replace('+00:00','Z')

    def tick(self):
        self.ensure_maintenance()
        self.maintenance()
        current=now()
        for schedule in self.schedules():
            if not schedule.get("enabled") or schedule.get("nextRunAt",current)>current:continue
            key=f"schedule:{schedule['id']}:{schedule['nextRunAt']}"
            self.store.enqueue(schedule['kind'],schedule.get('payload',{}),key=key,priority=80)
            hour,minute=map(int,schedule['time'].split(':'))
            self.store.put("schedule",{"lastRunAt":current,"nextRunAt":self.next_time(hour,minute)},id=schedule['id'])
        # Actual daily point-in-time counts, never fabricate historical data.
        day=datetime.now(KST).date().isoformat()
        self.store.put("metric_daily",{"date":day,"collectedDocuments":len([d for d in self.pre.documents() if d.get("state") not in ('deleted','excluded')]),"indexedChunks":sum(d.get("chunkCount",0) for d in self.pre.documents() if d.get("state")=="ready")},id=day)


class Runtime:
    def __init__(self, store, pre, post):
        self.store,self.pre,self.post=store,pre,post
        self.operations=Operations(store,pre,post)
        from RAG.Pre.crawling.service import CrawlerService
        self.crawler=CrawlerService(store,pre)
        self.stop_event=threading.Event();self.owner=uid();self.active=None;self.threads=[]
        self.pressure_since=None

    def start(self):
        self.operations.ensure_maintenance()
        self.crawler.tick()
        for target,name in ((self._work,"seed-worker"),(self._supervise,"seed-supervisor")):
            thread=threading.Thread(target=target,name=name,daemon=True);thread.start();self.threads.append(thread)

    def stop(self):
        self.stop_event.set()
        for thread in self.threads:thread.join(timeout=5)

    def _supervise(self):
        last_tick=0
        while not self.stop_event.wait(5):
            try:
                if self.active:self.store.heartbeat(self.active['id'],self.owner,self.active['fencingToken'])
                if time.monotonic()-last_tick>=30:
                    self.operations.tick();self.crawler.tick();last_tick=time.monotonic()
            except Exception:log.exception("supervisor_failed")

    def step(self):
        # Leave low-priority jobs durable/queued while the NAS is under pressure.
        peek=self.store.one("SELECT kind,priority FROM jobs WHERE state IN ('queued','retry_wait') AND next_attempt_at<=? ORDER BY priority,created_at LIMIT 1",(time.time(),))
        if peek and peek['kind']!='generate':
            status=system_status(self.store)
            low_memory=status['memory']['availableBytes']<768*1024**2
            if status['cpuPercent']>80:
                self.pressure_since=self.pressure_since or time.monotonic()
            else:self.pressure_since=None
            sustained=self.pressure_since is not None and time.monotonic()-self.pressure_since>=30
            try:rss_limit=float(self.store.setting('settings',{}).get('memory','2'))*1024**3
            except (ValueError,TypeError):rss_limit=2*1024**3
            if low_memory or sustained or status['memory']['processRssBytes']>rss_limit:return False
        job=self.store.claim(self.owner)
        if not job:return False
        self.active=job
        try:
            resource_guard(self.store)
            if job['kind']=='generate':
                created=datetime.fromisoformat(job['createdAt'].replace('Z','+00:00'))
                if (datetime.now(timezone.utc)-created).total_seconds()>120:
                    raise DomainError('GENERATION_QUEUE_TIMEOUT','생성 대기 시간이 120초를 초과했습니다. 다시 요청하세요.',409)
            if job['kind']=='embedding_install':
                from RAG.Pre.model_setup import install
                def progress(name,count,size):
                    self.store.assert_lease(job['id'],job['fencingToken'])
                    self.store.update_job(job['id'],stage='downloading_'+name,processedItems=count,totalItems=size,unit='bytes')
                self.store.assert_lease(job['id'],job['fencingToken'])
                result=install(self.store,progress)
                self.store.assert_lease(job['id'],job['fencingToken'])
                self.store.enqueue('reindex',{},priority=40)
            elif job['kind']=='crawl_cycle':result=self.crawler.process_job(job)
            elif job['kind'] in ('generate','export'):
                result=self.post.process_job(job)
            elif job['kind']=='source_sync':result=self.operations.collect(job)
            elif job['kind']=='backup':result=self.operations.backup(job)
            elif job['kind']=='evaluate':result=self.operations.evaluate(job)
            else:result=self.pre.process_job(job)
            current=self.store.job(job['id'])
            if current['fencingToken']!=job['fencingToken']:
                log.warning("stale_worker_result_ignored id=%s",job['id'])
                return True
            if current['state']=='cancelling':self.store.update_job(job['id'],state='cancelled')
            elif current['state']=='running':self.store.assert_lease(job['id'],job['fencingToken']);self.store.update_job(job['id'],state='needs_review' if isinstance(result,dict) and result.get('needsReviewCount',0)>0 else 'succeeded',result=result)
            self.store.activity("job_finished",job['id'],"작업 완료" if self.store.job(job['id'])['state']=='succeeded' else "작업 종료")
        except Exception as exc:
            code=getattr(exc,'code','JOB_FAILED')
            error={"code":code,"message":str(exc) if isinstance(exc,DomainError) else "처리 중 오류가 발생했습니다."}
            retryable=isinstance(exc,DomainError) and (exc.status==503 or code in ('INDEX_POLICY_CONFLICT','INDEX_BASE_CONFLICT')) and code not in ('RESOURCE_PRESSURE','DISK_LOW')
            current=self.store.job(job['id'])
            # A recovered old worker must never cancel/fail a newer owner's job.
            if current['fencingToken']!=job['fencingToken']:
                log.warning("stale_worker_result_ignored id=%s",job['id'])
                return True
            if job['kind']=='generate' and code not in ('CANCELLED','JOB_CANCELLED','LEASE_EXPIRED'):
                self.post.fail_run(job['payload']['runId'],code,error['message'])
            if code in ('CANCELLED','JOB_CANCELLED','LEASE_EXPIRED') or current['state']=='cancelling':
                self.store.update_job(job['id'],state='cancelled',error=error)
            elif retryable and job['attempt']<3:
                self.store.update_job(job['id'],state='retry_wait',error=error,nextAttemptAt=time.time()+[10,30,120][job['attempt']-1])
            else:self.store.update_job(job['id'],state='failed',error=error)
            self.store.activity("job_error",job['id'],error['message'])
            if not isinstance(exc,DomainError):log.exception("job_failed id=%s kind=%s",job['id'],job['kind'])
        finally:self.active=None
        return True

    def _work(self):
        while not self.stop_event.is_set():
            try:
                if not self.step():self.stop_event.wait(.3)
            except Exception:
                log.exception("queue_claim_failed");self.stop_event.wait(2)
