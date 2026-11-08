"""Local durable control storage. Never holds transactions across model work."""
from __future__ import annotations
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time
import uuid
from datetime import datetime, timezone
import apsw

HEAVY_JOBS=frozenset({'train','merge','evaluate','quantize','distill','bootstrap','compress','curate','evolve',
    'foundation_corpus','foundation_tokenizer','foundation_initialize','foundation_calibrate','foundation_pretrain','foundation_cpt','foundation_sft','foundation_evaluate','foundation_self_code_evaluate',
    'foundation_distill','foundation_response_distill','foundation_growth','foundation_growth_adopt','foundation_relearn','foundation_format'})

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")

def now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

def uid():
    return str(uuid.uuid4())

def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)

def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else canonical(value).encode()).hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()

class Fault(Exception):
    def __init__(self, code, message=None, status=422, details=None):
        self.code, self.message, self.status, self.details = code, message or code, status, details or {}
        super().__init__(self.message)

def require(condition, code, message=None, status=422, **details):
    if not condition:
        raise Fault(code, message, status, details)

def inside(root, relative):
    root = Path(root).resolve()
    target = (root / relative).resolve()
    require(target.is_relative_to(root) and not Path(relative).is_absolute(), "PATH_OUTSIDE_SCOPE", status=403)
    return target

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp-" + uid())
    with temp.open("w", encoding="utf-8", newline="\n") as f:
        f.write(canonical(value)); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)

class Store:
    @classmethod
    def open_existing(cls, root, *, read_only=False, busy_timeout_ms=5000):
        """Attach to an initialized store without running schema writes.

        Deployment inspection and health probes must not compete for the writer
        lock merely to read a heartbeat. Missing stores are never initialized by
        this entry point; callers that create or migrate stores use Store().
        """
        require(type(read_only) is bool and type(busy_timeout_ms) is int
                and 0 <= busy_timeout_ms <= 300000, "STORE_OPTIONS_INVALID")
        require(tuple(map(int, apsw.sqlitelibversion().split("."))) >= (3,51,3), "SQLITE_PATCH_REQUIRED", status=503)
        value = cls.__new__(cls)
        value.root = Path(root).resolve()
        path = value.root / "metadata" / "llm.db"
        require(path.is_file(), "STORE_NOT_INITIALIZED", status=503)
        value.lock = threading.RLock()
        value.db = apsw.Connection(str(path), flags=apsw.SQLITE_OPEN_READONLY if read_only else apsw.SQLITE_OPEN_READWRITE)
        try:
            value.db.set_busy_timeout(busy_timeout_ms)
            # Both pragmas apply to this connection, without changing the
            # persisted journal mode or running CREATE/INSERT migrations.
            value.db.execute("PRAGMA foreign_keys=ON; PRAGMA synchronous=FULL;")
            require(bool(value.rows("SELECT version FROM migrations WHERE version=1")), "STORE_NOT_INITIALIZED", status=503)
            value.rows("SELECT kind,id,body FROM records LIMIT 0")
        except BaseException:
            value.db.close()
            raise
        return value

    def __init__(self, root):
        self.root = Path(root).resolve()
        for name in ("metadata", "artifacts", "snapshots", "checkpoints", "staging", "backups", "suites", "reports"):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        require(tuple(map(int, apsw.sqlitelibversion().split("."))) >= (3,51,3), "SQLITE_PATCH_REQUIRED", status=503)
        self.lock = threading.RLock()
        self.db = apsw.Connection(str(self.root / "metadata" / "llm.db"))
        self.db.set_busy_timeout(5000)
        self.db.execute("PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; PRAGMA foreign_keys=ON;")
        self.db.execute("""
          CREATE TABLE IF NOT EXISTS records(kind TEXT,id TEXT,body TEXT NOT NULL,PRIMARY KEY(kind,id));
          CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,type TEXT,state TEXT,payload TEXT,result TEXT,
            error TEXT,owner TEXT,lease REAL DEFAULT 0,fence INTEGER DEFAULT 0,attempt INTEGER DEFAULT 0,
            progress TEXT,created TEXT,updated TEXT,request_key TEXT UNIQUE);
          CREATE TABLE IF NOT EXISTS events(sequence INTEGER PRIMARY KEY AUTOINCREMENT,type TEXT,entity TEXT,data TEXT,at TEXT);
          CREATE TABLE IF NOT EXISTS consumed(branch TEXT,hash TEXT,example TEXT,run TEXT,at TEXT,PRIMARY KEY(branch,hash));
          CREATE TABLE IF NOT EXISTS splits(family TEXT,group_id TEXT,split TEXT,PRIMARY KEY(family,group_id));
          CREATE TABLE IF NOT EXISTS idempotency(key TEXT PRIMARY KEY,request_hash TEXT,result TEXT);
          CREATE TABLE IF NOT EXISTS migrations(version INTEGER PRIMARY KEY,at TEXT);
          INSERT OR IGNORE INTO migrations VALUES(1,datetime('now'));
        """)

    @contextlib.contextmanager
    def tx(self):
        with self.lock:
            nested = not self.db.get_autocommit()
            name = "sp_" + uuid.uuid4().hex
            self.db.execute("SAVEPOINT " + name if nested else "BEGIN IMMEDIATE")
            try:
                yield
                self.db.execute("RELEASE " + name if nested else "COMMIT")
            except BaseException:
                if nested:
                    self.db.execute("ROLLBACK TO " + name); self.db.execute("RELEASE " + name)
                elif not self.db.get_autocommit():
                    self.db.execute("ROLLBACK")
                raise

    def rows(self, sql, args=()):
        with self.lock:
            cur = self.db.cursor(); cur.execute(sql, args)
            try: names = [x[0] for x in cur.get_description()]
            except apsw.ExecutionCompleteError: return []
            return [dict(zip(names, row)) for row in cur]

    def get(self, kind, identity, default=None):
        rows = self.rows("SELECT body FROM records WHERE kind=? AND id=?", (kind, identity))
        if not rows:
            if default is not None: return default
            raise Fault("NOT_FOUND", status=404, details={"kind":kind,"id":identity})
        return json.loads(rows[0]["body"])

    def put(self, kind, body, identity=None, expected=None):
        identity = identity or body.get("id") or uid()
        with self.tx():
            old = self.get(kind, identity, {})
            if expected is not None: require(old.get("version",0)==expected, "VERSION_CONFLICT", status=412)
            value = {**old, **body, "id":identity,"version":old.get("version",0)+1,
                     "createdAt":old.get("createdAt", now()),"updatedAt":now()}
            self.db.execute("INSERT OR REPLACE INTO records VALUES(?,?,?)", (kind,identity,canonical(value)))
            return value

    def list(self, kind):
        return [json.loads(r["body"]) for r in self.rows("SELECT body FROM records WHERE kind=? ORDER BY id",(kind,))]

    def event(self, kind, identity, data=None):
        with self.tx():
            self.db.execute("INSERT INTO events(type,entity,data,at) VALUES(?,?,?,?)",(kind,identity,canonical(data or {}),now()))

    def enqueue(self, kind, payload, key=None):
        with self.tx():
            if kind in HEAVY_JOBS:require(not self.get('setting','deploymentMaintenance',{}).get('enabled'),'DEPLOYMENT_MAINTENANCE',status=503)
            if key:
                rows = self.rows("SELECT id,payload FROM jobs WHERE request_key=?",(key,))
                if rows:
                    require(json.loads(rows[0]["payload"]) == payload, "IDEMPOTENCY_CONFLICT",status=409)
                    return self.job(rows[0]["id"])
            identity=uid()
            self.db.execute("INSERT INTO jobs(id,type,state,payload,created,updated,request_key,progress) VALUES(?,?,'queued',?,?,?,?,?)",
                            (identity,kind,canonical(payload),now(),now(),key,canonical({"stage":"queued","processedItems":0,"totalItems":None,"unit":"items","percent":None})))
            self.event(kind+".queued",identity)
            return self.job(identity)

    def job(self, identity):
        rows=self.rows("SELECT * FROM jobs WHERE id=?",(identity,))
        require(bool(rows),"JOB_NOT_FOUND",status=404)
        value=rows[0]
        for name in ("payload","result","error","progress"):
            value[name]=json.loads(value[name]) if value[name] else None
        return {**value,"fencingToken":value["fence"],"updatedAt":value["updated"],**(value.get("progress") or {})}

    def claim(self, owner, kinds=None):
        with self.tx():
            if self.get('setting','deploymentMaintenance',{}).get('enabled'):return None
            rows=self.rows("SELECT id,type FROM jobs WHERE state='queued' OR (state='running' AND lease<?) ORDER BY created",(time.time(),))
            for row in rows:
                if kinds and row["type"] not in kinds: continue
                if row['type'] in HEAVY_JOBS:
                    foreground=self.get('setting','foregroundLease',{})
                    if foreground.get('expires',0)>time.time():continue
                    busy=self.rows("SELECT id,type FROM jobs WHERE id<>? AND state IN ('running','pause_requested','cancelling') AND lease>?",(row['id'],time.time()))
                    if any(item['type'] in HEAVY_JOBS for item in busy):continue
                self.db.execute("UPDATE jobs SET state='running',owner=?,lease=?,fence=fence+1,attempt=attempt+1,updated=? WHERE id=?",(owner,time.time()+60,now(),row["id"]))
                return self.job(row["id"])

    @contextlib.contextmanager
    def foreground_slot(self,timeout=60):
        """Cross-process priority for interactive work, with expiring ownership."""
        identity=uid();deadline=time.monotonic()+timeout;acquired=False
        try:
            while time.monotonic()<deadline:
                with self.tx():
                    require(not self.get('setting','deploymentMaintenance',{}).get('enabled'),'DEPLOYMENT_MAINTENANCE',status=503)
                    old=self.get('setting','foregroundLease',{})
                    if old.get('expires',0)<=time.time() or old.get('owner')==identity:
                        self.put('setting',{'owner':identity,'expires':time.time()+90},'foregroundLease')
                        acquired=True
                        self.db.execute("UPDATE jobs SET state='pause_requested' WHERE type IN ('train','foundation_pretrain','foundation_cpt','foundation_sft') AND state='running' AND lease>?",(time.time(),))
                        for running in self.rows("SELECT id FROM jobs WHERE type IN ('foundation_pretrain','foundation_cpt','foundation_sft') AND state='pause_requested' AND lease>?",(time.time(),)):
                            self.put('foundation_run',{'pauseReason':'FOREGROUND_REQUEST'},running['id'])
                        busy=self.rows("SELECT type FROM jobs WHERE state IN ('running','pause_requested','cancelling') AND lease>?",(time.time(),))
                        if not any(item['type'] in HEAVY_JOBS for item in busy):break
                time.sleep(.1)
            else:raise Fault('BACKGROUND_YIELD_PENDING',status=503)
            ended=threading.Event()
            def keep_alive():
                while not ended.wait(10):
                    with self.tx():
                        value=self.get('setting','foregroundLease',{})
                        if value.get('owner')!=identity:return
                        self.put('setting',{'expires':time.time()+90},'foregroundLease')
            thread=threading.Thread(target=keep_alive,daemon=True);thread.start()
            try:yield
            finally:ended.set();thread.join(timeout=1)
        finally:
            if acquired:
                with self.tx():
                    if self.get('setting','foregroundLease',{}).get('owner')==identity:
                        self.db.execute("DELETE FROM records WHERE kind='setting' AND id='foregroundLease'")

    def assert_lease(self, job):
        current=self.job(job["id"])
        require(current["fence"]==job["fence"] and current["owner"]==job["owner"] and current["lease"]>time.time(),"LEASE_EXPIRED",status=409)
        return current

    def heartbeat(self, job):
        with self.tx():
            self.assert_lease(job)
            self.db.execute("UPDATE jobs SET lease=?,updated=? WHERE id=?",(time.time()+60,now(),job["id"]))

    def progress(self, job, stage, completed=0, total=None, unit="items", **extra):
        with self.tx():
            self.assert_lease(job)
            value={"stage":stage,"processedItems":completed,"totalItems":total,"unit":unit,
                   "percent":math.floor(completed/total*1000)/10 if total else None,**extra}
            self.db.execute("UPDATE jobs SET progress=?,updated=? WHERE id=?",(canonical(value),now(),job["id"]))

    def finish(self, job, state, result=None, error=None):
        with self.tx():
            self.assert_lease(job)
            self.db.execute("UPDATE jobs SET state=?,result=?,error=?,updated=? WHERE id=?",(state,canonical(result),canonical(error),now(),job["id"]))
            self.event(job["type"]+"."+state,job["id"],{"error":error})

    def close(self):
        with self.lock: self.db.close()
