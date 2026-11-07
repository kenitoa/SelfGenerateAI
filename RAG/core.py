"""Small, persistent storage and queue shared by Pre and Post.

APSW deliberately supplies its own patched SQLite instead of using the host
Python's potentially older SQLite. Transactions never wrap network/ML work.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import apsw


def now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def uid():
    return str(uuid.uuid4())


def digest(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(value).hexdigest()


class DomainError(Exception):
    def __init__(self, code, message, status=422, details=None):
        super().__init__(message)
        self.code, self.message, self.status, self.details = code, message, status, details or {}


class Cursor:
    def __init__(self, cursor):
        self.cursor = cursor
        try:
            self.names = [d[0] for d in cursor.get_description()]
        except apsw.ExecutionCompleteError:
            self.names = []

    def fetchone(self):
        row = self.cursor.fetchone()
        return None if row is None else dict(zip(self.names, row))

    def fetchall(self):
        return [dict(zip(self.names, row)) for row in self.cursor.fetchall()]

    def __iter__(self):
        for row in self.cursor:
            yield dict(zip(self.names, row))


class Connection:
    def __init__(self, connection):
        self.raw = connection

    def execute(self, sql, params=()):
        return Cursor(self.raw.execute(sql, params))

    def executemany(self, sql, rows):
        return Cursor(self.raw.executemany(sql, rows))

    def executescript(self, script):
        self.raw.execute(script)


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.blobs = self.root / "blobs"
        self.blobs.mkdir(exist_ok=True)
        for folder in ("metadata", "quarantine", "indexes", "uploads/tmp", "exports", "backups"):
            (self.root / folder).mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "metadata" / "seed.db"
        self.lock = threading.RLock()
        self.local = threading.local()
        self.sqlite_version = apsw.sqlitelibversion()
        version = tuple(map(int, self.sqlite_version.split(".")))
        if version < (3, 51, 3):
            raise RuntimeError("WAL-reset 수정이 포함된 SQLite 3.51.3 이상이 필요합니다.")
        with self.tx() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS records(
                  kind TEXT NOT NULL, id TEXT NOT NULL, body TEXT NOT NULL,
                  created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                  PRIMARY KEY(kind,id));
                CREATE INDEX IF NOT EXISTS records_by_kind_time ON records(kind,created_at,id);
                CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY,value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS jobs(
                  id TEXT PRIMARY KEY,kind TEXT NOT NULL,state TEXT NOT NULL,
                  priority INTEGER NOT NULL,payload TEXT NOT NULL,idempotency_key TEXT UNIQUE,
                  stage TEXT NOT NULL DEFAULT 'queued',processed_items INTEGER NOT NULL DEFAULT 0,
                  total_items INTEGER,unit TEXT NOT NULL DEFAULT 'items',attempt INTEGER NOT NULL DEFAULT 0,
                  lease_owner TEXT,lease_expires_at REAL,fencing_token INTEGER NOT NULL DEFAULT 0,
                  heartbeat_at TEXT,error TEXT,result TEXT,next_attempt_at REAL NOT NULL DEFAULT 0,
                  created_at TEXT NOT NULL,updated_at TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS jobs_claim ON jobs(state,priority,next_attempt_at,created_at);
                CREATE TABLE IF NOT EXISTS activities(
                  sequence INTEGER PRIMARY KEY AUTOINCREMENT,type TEXT NOT NULL,
                  entity_id TEXT,message TEXT NOT NULL,created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS idempotency(
                  key TEXT PRIMARY KEY,request_hash TEXT NOT NULL,status TEXT NOT NULL,
                  resource TEXT,created_at REAL NOT NULL);
                INSERT OR IGNORE INTO schema_migrations VALUES(1,datetime('now'));
            """)
        if not self.setting("settings"):
            self.set_setting("settings", {
                "version": 1, "name": "나의 워크스페이스", "endpoint": "http://localhost:11434",
                "model": "qwen2.5-coder:0.5b", "memory": "2", "schedule": "02:00", "auto": False,
                "training": {"endpoint": "", "epochs": "3", "batch": "1", "rate": "0.0002"},
                "contextTokens": 4096, "maxOutputTokens": 700, "tokenizerPath": "",
                "embedding": {"enabled": False, "directory": "", "revision": "", "dimension": 384},
                "allowedSourceHosts": [],
            })

    def connect(self):
        connection = apsw.Connection(str(self.db_path))
        connection.set_busy_timeout(5000)
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    @contextlib.contextmanager
    def tx(self):
        with self.lock:
            current = getattr(self.local, "connection", None)
            owner = current is None
            connection = current or self.connect()
            savepoint = "nested_" + uuid.uuid4().hex
            if owner:
                self.local.connection = connection
                connection.execute("BEGIN IMMEDIATE")
            else:
                connection.execute("SAVEPOINT " + savepoint)
            try:
                yield Connection(connection)
                connection.execute("COMMIT" if owner else "RELEASE " + savepoint)
            except BaseException:
                if owner:
                    if not connection.get_autocommit():
                        connection.execute("ROLLBACK")
                else:
                    connection.execute("ROLLBACK TO " + savepoint)
                    connection.execute("RELEASE " + savepoint)
                raise
            finally:
                if owner:
                    self.local.connection = None
                    connection.close()

    def read(self, sql, params=()):
        with self.lock:
            connection = getattr(self.local, "connection", None)
            owner = connection is None
            connection = connection or self.connect()
            try:
                return Connection(connection).execute(sql, params).fetchall()
            finally:
                if owner:
                    connection.close()

    def one(self, sql, params=()):
        rows = self.read(sql, params)
        return rows[0] if rows else None

    def put(self, kind, body, id=None):
        identity = str(id or body.get("id") or uid())
        with self.tx() as db:
            old = db.execute("SELECT body FROM records WHERE kind=? AND id=?", (kind, identity)).fetchone()
            previous = json.loads(old["body"]) if old else {}
            value = {**previous, **body, "id": identity, "createdAt": previous.get("createdAt", body.get("createdAt", now())), "updatedAt": now()}
            db.execute("INSERT INTO records(kind,id,body,created_at,updated_at) VALUES(?,?,?,?,?) ON CONFLICT(kind,id) DO UPDATE SET body=excluded.body,updated_at=excluded.updated_at",
                       (kind, identity, json.dumps(value, ensure_ascii=False), value["createdAt"], value["updatedAt"]))
            if kind in ('source','document','version','training_example'):
                db.execute("INSERT INTO settings(key,value) VALUES('trainingDataRevision','1') ON CONFLICT(key) DO UPDATE SET value=CAST(settings.value AS INTEGER)+1")
            return value

    def get(self, kind, id):
        row = self.one("SELECT body FROM records WHERE kind=? AND id=?", (kind, str(id)))
        if row is None:
            raise DomainError("NOT_FOUND", "항목을 찾을 수 없습니다.", 404, {"kind": kind, "id": str(id)})
        return json.loads(row["body"])

    def list(self, kind):
        return [json.loads(row["body"]) for row in self.read("SELECT body FROM records WHERE kind=? ORDER BY created_at DESC,id DESC", (kind,))]

    def delete(self, kind, id):
        with self.tx() as db:
            db.execute("DELETE FROM records WHERE kind=? AND id=?", (kind, str(id)))
            if kind in ('source','document','version','training_example'):
                db.execute("INSERT INTO settings(key,value) VALUES('trainingDataRevision','1') ON CONFLICT(key) DO UPDATE SET value=CAST(settings.value AS INTEGER)+1")

    def setting(self, key, default=None):
        row = self.one("SELECT value FROM settings WHERE key=?", (key,))
        return json.loads(row["value"]) if row else default

    def set_setting(self, key, value):
        with self.tx() as db:
            db.execute("INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, json.dumps(value, ensure_ascii=False)))
            if key in ('policyVersion','evaluationHoldout','trainingStableOrigins','trainingStableSplits'):
                db.execute("INSERT INTO settings(key,value) VALUES('trainingDataRevision','1') ON CONFLICT(key) DO UPDATE SET value=CAST(settings.value AS INTEGER)+1")

    def activity(self, type, entity_id, message):
        with self.tx() as db:
            db.execute("INSERT INTO activities(type,entity_id,message,created_at) VALUES(?,?,?,?)", (type, entity_id, message, now()))

    def enqueue(self, kind, payload, key=None, priority=50):
        with self.tx() as db:
            if key:
                old = db.execute("SELECT id FROM jobs WHERE idempotency_key=?", (key,)).fetchone()
                if old:
                    return self.job(old["id"])
            identity, stamp = uid(), now()
            db.execute("INSERT INTO jobs(id,kind,state,priority,payload,idempotency_key,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)",
                       (identity, kind, "queued", priority, json.dumps(payload, ensure_ascii=False), key, stamp, stamp))
            return self.job(identity)

    def job(self, id):
        row = self.one("SELECT * FROM jobs WHERE id=?", (str(id),))
        if row is None:
            raise DomainError("NOT_FOUND", "작업을 찾을 수 없습니다.", 404)
        return self._job(row)

    def _job(self, row):
        result = {"id": row["id"], "kind": row["kind"], "state": row["state"], "priority": row["priority"],
                  "payload": json.loads(row["payload"]), "stage": row["stage"], "processedItems": row["processed_items"],
                  "totalItems": row["total_items"], "unit": row["unit"], "attempt": row["attempt"],
                  "leaseOwner": row["lease_owner"], "fencingToken": row["fencing_token"],
                  "createdAt": row["created_at"], "updatedAt": row["updated_at"],
                  "error": json.loads(row["error"]) if row["error"] else None,
                  "result": json.loads(row["result"]) if row["result"] else None}
        total = result["totalItems"]
        result["percent"] = min(100, result["processedItems"] / total * 100) if total else None
        result["capabilities"] = {"cancel": row["state"] in ("queued", "running", "retry_wait"), "retry": row["state"] in ("failed", "cancelled")}
        return result

    def update_job(self, id, **fields):
        names = {"state": "state", "stage": "stage", "processedItems": "processed_items", "totalItems": "total_items", "unit": "unit", "error": "error", "result": "result", "nextAttemptAt": "next_attempt_at"}
        updates, params = ["updated_at=?"], [now()]
        for key, value in fields.items():
            if key not in names:
                continue
            updates.append(names[key] + "=?")
            params.append(json.dumps(value, ensure_ascii=False) if key in ("error", "result") and value is not None else value)
        with self.tx() as db:
            db.execute("UPDATE jobs SET " + ",".join(updates) + " WHERE id=?", (*params, str(id)))
        return self.job(id)

    def claim(self, owner):
        with self.tx() as db:
            stamp = time.time()
            db.execute("UPDATE jobs SET state='cancelled',lease_owner=NULL WHERE state='cancelling' AND lease_expires_at<?", (stamp,))
            db.execute("UPDATE jobs SET state='queued',lease_owner=NULL WHERE state='running' AND lease_expires_at<?", (stamp,))
            row = db.execute("SELECT id FROM jobs WHERE state IN ('queued','retry_wait') AND next_attempt_at<=? ORDER BY priority,created_at LIMIT 1", (stamp,)).fetchone()
            if not row:
                return None
            db.execute("UPDATE jobs SET state='running',attempt=attempt+1,lease_owner=?,lease_expires_at=?,heartbeat_at=?,fencing_token=fencing_token+1,updated_at=? WHERE id=?", (owner, stamp+60, now(), now(), row["id"]))
            return self.job(row["id"])

    def heartbeat(self, id, owner, token):
        with self.tx() as db:
            db.execute("UPDATE jobs SET lease_expires_at=?,heartbeat_at=? WHERE id=? AND lease_owner=? AND fencing_token=? AND state IN ('running','cancelling')", (time.time()+60, now(), id, owner, token))

    def assert_lease(self, id, token):
        row = self.one("SELECT state,fencing_token,lease_expires_at FROM jobs WHERE id=?", (id,))
        if not row or row["state"] != "running" or row["fencing_token"] != token or row["lease_expires_at"] < time.time():
            raise DomainError("LEASE_EXPIRED", "작업 소유권이 만료되었습니다.", 409)

    def cancel_job(self, id):
        job = self.job(id)
        if job["state"] in ("queued", "retry_wait"):
            return self.update_job(id, state="cancelled")
        if job["state"] == "running":
            return self.update_job(id, state="cancelling")
        return job

    def retry_job(self, id):
        job = self.job(id)
        if not job["capabilities"]["retry"]:
            raise DomainError("JOB_CONFLICT", "종료된 실패·취소 작업만 다시 실행할 수 있습니다.", 409)
        return self.update_job(id, state="queued", error=None, nextAttemptAt=0)

    def close(self):
        pass  # Every operation owns/returns its connection.
