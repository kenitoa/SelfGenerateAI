"""Conservative per-project quota accounting and explicit checkpoint reclamation.

Raw corpora, tokenizers, Genesis, registry artifacts and unsafe/unowned paths are
never automatic deletion candidates. Metadata tombstones preserve all lineage.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import shutil
import time

from LLM.core import require, now, digest, file_hash
from .checkpoint import verify as verify_checkpoint

DEFAULT_QUOTA_BYTES = 20 * 1024**3
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_VERIFIED = {}


def _project(store, value):
    identity = value if isinstance(value, str) else value["id"]
    record = store.get("foundation_project", identity)
    record["policy"] = store.get("foundation_policy", identity, {}) or (value.get("policy", {}) if isinstance(value, dict) else {})
    return record


def _records(store):
    return [(row["kind"], json.loads(row["body"])) for row in store.rows("SELECT kind,body FROM records")]


def _belongs(record, project_id):
    return record.get("projectId") == project_id or record.get("foundationProjectId") == project_id


def _strings(value, *, root=True):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            if root and key == "id":
                continue
            yield from _strings(item, root=False)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item, root=False)


def _path(store, relative, *, candidate_id=None):
    require(isinstance(relative, str) and relative and "\\" not in relative, "RETENTION_PATH_UNSAFE")
    rel = Path(relative)
    require(not rel.is_absolute() and ".." not in rel.parts and rel.parts[0] in ("foundation", "artifacts"),
            "RETENTION_PATH_UNSAFE")
    if candidate_id is not None:
        require(_SAFE_ID.fullmatch(candidate_id) and rel.parts == ("foundation", "checkpoints", candidate_id),
                "RETENTION_PATH_UNSAFE")
    root = Path(store.root).resolve()
    current = root
    for part in rel.parts:
        current = current / part
        require(not current.is_symlink(), "RETENTION_PATH_UNSAFE")
    resolved=current.resolve()
    require(resolved.is_relative_to(root) and resolved != root, "RETENTION_PATH_UNSAFE")
    return resolved


def _bytes(path):
    """Never follows symlinks; special files make accounting fail closed."""
    require(not path.is_symlink(), "RETENTION_PATH_UNSAFE")
    if path.is_file():
        return path.stat().st_size
    require(path.is_dir(), "RETENTION_PATH_MISSING")
    total = 0
    with os.scandir(path) as entries:
        for entry in entries:
            require(not entry.is_symlink(), "RETENTION_PATH_UNSAFE")
            if entry.is_dir(follow_symlinks=False):
                total += _bytes(Path(entry.path))
            elif entry.is_file(follow_symlinks=False):
                total += entry.stat(follow_symlinks=False).st_size
            else:
                require(False, "RETENTION_PATH_UNSAFE")
    return total


def _verified_candidate(store, record, *, cached=False):
    path=_path(store,record['storagePath'],candidate_id=record['id'])
    signature=tuple(sorted((p.relative_to(path).as_posix(),p.stat().st_size,p.stat().st_mtime_ns)
        for p in path.rglob('*') if p.is_file()))
    key=(str(path),record.get('manifestHash'))
    if cached and _VERIFIED.get(key,{}).get('signature')==signature:
        _bytes(path)
        return path,_VERIFIED[key]['manifest']
    root, manifest = verify_checkpoint(store, record)
    _bytes(root)
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    require(actual == set(manifest["files"]) | {"manifest.json"}, "RETENTION_UNOWNED_FILES")
    _VERIFIED[key]={'signature':signature,'manifest':manifest}
    return root, manifest


def plan(store, project):
    project = _project(store, project)
    identity = project["id"]
    quota = project["policy"].get("quotaBytes", DEFAULT_QUOTA_BYTES)
    require(isinstance(quota, int) and not isinstance(quota, bool) and quota > 0, "QUOTA_INVALID")
    all_records = _records(store)
    candidates = {record["id"]: record for kind, record in all_records
                  if kind == "foundation_checkpoint" and _belongs(record, identity)
                  and record.get("state") != "deleted"}
    protected = {}
    def protect(checkpoint_id, reason):
        if checkpoint_id in candidates:
            protected.setdefault(checkpoint_id, set()).add(reason)
    # Any external reference is retained, not just references recognizable as a release.
    # This deliberately prefers quota-held over silently breaking audit/recovery history.
    for kind, record in all_records:
        if kind == "foundation_checkpoint":
            continue
        if record.get("state") in ("deleted", "archived"):
            continue
        for value in _strings(record):
            protect(value, "referenced:" + kind)
    for row in store.rows("SELECT id,state,payload,result FROM jobs"):
        for field in ("payload", "result"):
            if row.get(field):
                for value in _strings(json.loads(row[field])):
                    protect(value, "job:" + row["state"])
    ordered = sorted(candidates.values(), key=lambda value: (value.get("createdAt", ""), value["id"]), reverse=True)
    for record in ordered[:3]:
        protect(record["id"], "latest_three")
    scores = {}
    for kind, record in all_records:
        if kind == "foundation_checkpoint" and record["id"] in candidates:
            score = record.get("validationNll", record.get("validationLoss"))
            if isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score):
                scores[record["id"]] = float(score)
        if kind == "foundation_evaluation" and record.get("state") == "completed" and record.get("kind") in ("qualification", "lm_dev", "dev"):
            checkpoint_id = record.get("checkpointId", record.get("artifactId"))
            score = record.get("nll", {}).get("candidate") if isinstance(record.get("nll"), dict) else record.get("validationNll")
            if checkpoint_id in candidates and isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score):
                scores[checkpoint_id] = min(scores.get(checkpoint_id, math.inf), float(score))
    for checkpoint_id in sorted(scores, key=lambda key: (scores[key], key))[:2]:
        protect(checkpoint_id, "best_two_dev")
    for checkpoint_id, record in candidates.items():
        if record.get("pinned") or record.get("pinReason"):
            protect(checkpoint_id, "explicit_pin")
        if record.get("state") in ("revoked", "quarantined"):
            protect(checkpoint_id, "policy_investigation")
    # Keep the complete parent chain of every retained checkpoint.
    changed = True
    while changed:
        previous = len(protected)
        for checkpoint_id in list(protected):
            for field in ("parentCheckpointId", "sourceCheckpointId"):
                protect(candidates[checkpoint_id].get(field), "retained_parent_chain")
        changed = len(protected) != previous
    usage = 0
    linked_artifact_bytes = 0
    unsafe = []
    sizes = {}
    scanned = set()
    paths = []
    for kind, record in all_records:
        if kind=='foundation_retention_intent':continue
        if not _belongs(record, identity) or not record.get("storagePath") or record.get("state") == "deleted":
            continue
        try:
            path = _path(store, record["storagePath"],
                         candidate_id=record["id"] if kind == "foundation_checkpoint" else None)
            if record.get('state')=='archived' and not path.exists():continue
            paths.append((path,kind,record))
        except (Exception,) as exc:
            unsafe.append({"kind": kind, "id": record["id"], "reasonCode": getattr(exc, "code", "RETENTION_ACCOUNTING_FAILED")})
            if kind == "foundation_checkpoint":
                protect(record["id"], "unsafe_or_unverified")
    # Canonicalize each object once, then charge ancestors before descendants.
    # Calling Path.resolve() inside an all-prior-path comparison makes n token
    # caches trigger n^2 real filesystem lookups on Windows. Parent membership
    # below is lexical and bounded by path depth, never another disk traversal.
    paths.sort(key=lambda item:(len(item[0].parts),os.path.normcase(str(item[0]))))
    known_sizes={}
    for path,kind,record in paths:
        try:
            canonical=os.path.normcase(str(path))
            covered=canonical in scanned or any(os.path.normcase(str(parent)) in scanned for parent in path.parents)
            if kind == 'foundation_checkpoint':
                # Coverage by another directory must not bypass integrity or
                # remove this checkpoint's true reclaimable-byte measurement.
                size=known_sizes.get(canonical)
                if size is None:size=_bytes(path);known_sizes[canonical]=size
                sizes[record['id']]=size
                _verified_candidate(store,record,cached=True)
            elif not covered:
                size=_bytes(path);known_sizes[canonical]=size
            if not covered:
                scanned.add(canonical)
                usage+=size
                if Path(record['storagePath']).parts[0]=='artifacts':linked_artifact_bytes+=size
        except (Exception,) as exc:
            unsafe.append({'kind':kind,'id':record['id'],
                           'reasonCode':getattr(exc,'code','RETENTION_ACCOUNTING_FAILED')})
            if kind=='foundation_checkpoint':protect(record['id'],'unsafe_or_unverified')
    # An unsafe child's parent cannot be reclaimed indirectly either.
    changed = True
    while changed:
        previous = len(protected)
        for checkpoint_id in list(protected):
            protect(candidates[checkpoint_id].get("parentCheckpointId"), "retained_parent_chain")
        changed = len(protected) != previous
    reclaimable = [{"id": key, "storagePath": record["storagePath"], "bytes": sizes[key],
                    "manifestHash": record["manifestHash"], "recordVersion": record["version"]}
                   for key, record in candidates.items() if key not in protected and key in sizes]
    reclaimable.sort(key=lambda item: (candidates[item["id"]].get("createdAt", ""), item["id"]))
    reclaimable_bytes = sum(item["bytes"] for item in reclaimable)
    reasons = []
    if unsafe:
        reasons.append("QUOTA_ACCOUNTING_UNSAFE")
    if usage > quota:
        reasons.append("QUOTA_EXCEEDED")
    if usage > quota and usage-reclaimable_bytes > quota:
        reasons.append("QUOTA_HELD_BY_REFERENCES")
    result = {"projectId": identity, "usageBytes": usage, "quotaBytes": quota,
              "linkedArtifactBytes": linked_artifact_bytes, "availableBytes": max(0, quota-usage),
              "protected": [{"id": key, "bytes": sizes.get(key), "reasons": sorted(reasons_)}
                            for key, reasons_ in sorted(protected.items())],
              "reclaimable": reclaimable, "reclaimableBytes": reclaimable_bytes,
              "minimumUsageAfterCollection": usage-reclaimable_bytes, "accountingComplete": not unsafe,
              "unsafeObjects": unsafe, "reasonCodes": reasons, "observedAt": now(), "dryRun": True}
    result["planHash"] = digest({k: v for k, v in result.items() if k not in ("observedAt", "planHash")})
    return result


def enforce(store, project, dry_run=True):
    require(isinstance(dry_run, bool), "RETENTION_MODE_INVALID")
    if not dry_run:require(not store.get('setting','deploymentMaintenance',{}).get('enabled'),'DEPLOYMENT_MAINTENANCE',status=409)
    initial = plan(store, project)
    if dry_run:
        return initial
    deleted = []
    # Reclamation intent is committed before touching files. A partial filesystem
    # failure must not roll back metadata for checkpoints already removed.
    with store.lock:
        require(initial["accountingComplete"], "QUOTA_ACCOUNTING_UNSAFE", status=409)
        for item in initial["reclaimable"]:
            fresh_plan = plan(store, project)
            eligible = {value["id"]: value for value in fresh_plan["reclaimable"]}
            require(item["id"] in eligible, "RETENTION_REFERENCES_CHANGED", status=409)
            record = store.get("foundation_checkpoint", item["id"])
            require(record["version"] == item["recordVersion"] and record["manifestHash"] == item["manifestHash"],
                    "RETENTION_REFERENCES_CHANGED", status=409)
            path = _path(store, record["storagePath"], candidate_id=record["id"])
            _, manifest = _verified_candidate(store, record)
            _bytes(path)  # Final symlink/special-file traversal check before deletion.
            with store.tx():
                require(not store.get('setting','deploymentMaintenance',{}).get('enabled'),'DEPLOYMENT_MAINTENANCE',status=409)
                intent = store.put("foundation_retention_intent", {"projectId": initial["projectId"],
                    "checkpointId": record["id"], "storagePath": record["storagePath"], "state": "deleting",
                    "manifestHash": record["manifestHash"], "verifiedManifest": manifest,
                    "bytes": item["bytes"], "planHash": initial["planHash"]}, "reclaim-" + record["id"])
                store.put("foundation_checkpoint", {"state": "reclaiming"}, record["id"], expected=record["version"])
            # Ownership and manifest have been checked against the exact generated directory.
            try:
                shutil.rmtree(path)
            except BaseException:
                store.event("foundation.retention.interrupted", record["id"], {"projectId": initial["projectId"]})
                raise
            tombstone = {"state": "deleted", "deletedAt": now(), "retentionReason": "unreferenced_checkpoint",
                         "deletedBytes": item["bytes"], "formerManifestHash": record["manifestHash"],
                         "deletionPlanHash": initial["planHash"]}
            with store.tx():
                store.put("foundation_checkpoint", tombstone, record["id"])
                store.put("foundation_retention_intent", {"state": "completed", "finishedAt": now()}, intent["id"])
                store.event("foundation.checkpoint.reclaimed", record["id"], {"projectId": initial["projectId"], "bytes": item["bytes"]})
            deleted.append(record["id"])
    result = plan(store, project)
    return {**result, "dryRun": False, "deletedCheckpointIds": deleted, "requestedPlanHash": initial["planHash"]}


def _quota_token(store, identity):
    # Heartbeats/events and in-memory optimizer progress do not change storage
    # allocation. Published paths, reservations and project policy do.
    rows=store.rows("""SELECT kind,id,json_extract(body,'$.version') AS revision FROM records
        WHERE (kind IN ('foundation_project','foundation_policy') AND id=?)
        OR ((json_extract(body,'$.projectId')=? OR json_extract(body,'$.foundationProjectId')=?)
            AND (json_type(body,'$.storagePath')='text' OR kind='foundation_storage_reservation'))
        ORDER BY kind,id""",(identity,identity,identity))
    return digest(rows)


def admit(store, project, additional_bytes=0, *, _allocate=None):
    require(isinstance(additional_bytes, int) and not isinstance(additional_bytes, bool) and additional_bytes >= 0,
            "QUOTA_ESTIMATE_INVALID")
    require(store.db.get_autocommit(),'QUOTA_TRANSACTION_BOUNDARY_REQUIRED',status=409)
    identity=project if isinstance(project,str) else project['id']
    for attempt in range(8):
        token=_quota_token(store,identity)
        # File traversal and immutable checkpoint verification can take seconds.
        # They never run while holding SQLite's single writer transaction.
        result=plan(store,project)
        with store.tx():
            if _quota_token(store,identity)!=token:continue
            reserved=0
            for reservation in store.list('foundation_storage_reservation'):
                if reservation.get('projectId')!=identity or reservation.get('state') not in ('active','held'):continue
                if reservation.get('state')=='active' and not reservation.get('activeWorkId') and reservation.get('expiresAt',0)<=time.time():
                    store.put('foundation_storage_reservation',{'state':'expired','remainingBytes':0,'fence':reservation.get('fence',0)+1},
                        reservation['id'],expected=reservation['version'])
                    continue
                reserved+=reservation.get('remainingBytes',0)
            require(result['accountingComplete'],'QUOTA_ACCOUNTING_UNSAFE',status=409)
            require(result['usageBytes']+reserved+additional_bytes<=result['quotaBytes'],'FOUNDATION_QUOTA_EXCEEDED',
                status=409,usageBytes=result['usageBytes'],reservedBytes=reserved,quotaBytes=result['quotaBytes'],
                additionalBytes=additional_bytes,reclaimableBytes=result['reclaimableBytes'])
            admitted={**result,'allowed':True,'additionalBytes':additional_bytes,'reservedBytes':reserved,
                'availableBytes':max(0,result['quotaBytes']-result['usageBytes']-reserved)}
            # A reservation is created in the same short CAS transaction. Two
            # allocators cannot both spend the same observed free bytes.
            return _allocate(admitted) if _allocate is not None else admitted
    require(False,'QUOTA_SNAPSHOT_CHANGED',status=409)


admission = admit
quota_admit = admit
