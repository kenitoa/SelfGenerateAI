"""Permission-scoped raw corpora. Nothing in this module executes source text."""
from __future__ import annotations

import ipaddress
import hashlib
import heapq
import json
import os
import re
import shutil
import apsw
from contextlib import nullcontext
from pathlib import Path

from LLM.core import Fault, atomic_json, canonical, digest, file_hash, inside, now, require, uid

SPLITS = ("train", "dev", "release", "audit")
SPLIT_BOUNDARIES = ((9000, "train"), (9800, "dev"), (9900, "release"), (10000, "audit"))
MAX_DOCUMENT_BYTES = 2 * 1024**2 - 65536
MAX_SHARD_BYTES = 128 * 1024**2
MAX_LINE_BYTES = 2 * 1024**2
_SECRET = [
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,})\b"),
    re.compile(r"(?i)\b(?:password|passwd|api[_-]?key|access[_-]?token|secret)\s*[:=]\s*['\"]([^'\"\n]{8,})['\"]"),
    re.compile(r"(?i)\b(?:https?|ssh)://[^\s/@:]+:[^\s/@]+@"),
    re.compile(r"(?i)(?:[a-z]:[\\/]users[\\/][^\s\\/'\"]+|/home/[^\s/'\"]+|/Users/[^\s/'\"]+)"),
]
_IP = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
_IPV6 = re.compile(r"(?<![\w:])(?:[a-fA-F0-9]{0,4}:){2,7}[a-fA-F0-9]{0,4}(?![\w:])")


def secret_reasons(text):
    """Conservative quarantine signals, not a claim of complete secret detection."""
    reasons = []
    for pattern in _SECRET:
        for match in pattern.finditer(text):
            sample = match.group(1) if match.lastindex else match.group(0)
            if any(marker in sample.lower() for marker in ("example", "placeholder", "your_", "your-", "changeme", "<", "${")):
                continue
            reasons.append("SECRET_OR_PERSONAL_VALUE")
            break
    for match in [*_IP.finditer(text), *_IPV6.finditer(text)]:
        try:
            address = ipaddress.ip_address(match.group(0))
        except ValueError:
            continue
        if address.is_private and not address.is_loopback and not address.is_unspecified:
            reasons.append("PRIVATE_NETWORK_ADDRESS")
            break
    return sorted(set(reasons))


def _groups(record):
    raw = record.get("groupId", record.get("group", record.get("lineageGroups")))
    values = raw if isinstance(raw, list) else [raw]
    return sorted({str(value) for value in values if value is not None and str(value).strip()})


def _stable_split(group):
    position = int(digest(["foundation-global-split-v1", group])[:12], 16) % 10000
    return next(name for upper, name in SPLIT_BOUNDARIES if position < upper)


def _resolve_split(store, groups, content_hash, preferred=None):
    with store.tx():
        return _resolve_split_locked(store, groups, content_hash, preferred)


def _resolve_split_locked(store, groups, content_hash, preferred=None):
    """One global source-family/content split survives project reset and retries."""
    existing = [store.get("foundation_split", digest(group), {}) for group in groups]
    old_content = store.get("foundation_content_split", content_hash, {})
    splits = {item["split"] for item in existing if item}
    if old_content:
        splits.add(old_content["split"])
    if preferred:
        require(preferred in SPLITS, "INVALID_CORPUS_SPLIT")
        splits.add(preferred)
    require(not (splits - set(SPLITS)), "SPLIT_LEAKAGE", status=409)
    require(len(splits) <= 1, "SPLIT_LEAKAGE", status=409)
    chosen = next(iter(splits)) if splits else _stable_split(groups[0])
    for group in groups:
        store.put("foundation_split", {"groupId": group, "split": chosen}, digest(group))
    store.put("foundation_content_split", {"contentHash": content_hash, "groupIds": groups, "split": chosen}, content_hash)
    return chosen


def _build_memory_legacy(store, project, records, config=None):
    """Build(store, project, approved records, config) -> immutable corpus record.

    The controller supplies authenticated original versions, never a browser's
    permission assertion. We still recheck each record and retain excluded IDs.
    """
    config = config or {}
    sft_mode = config.get("objective") == "sft" or config.get("kind") in ("sft", "foundation_sft_corpus")
    policy = project.get("policy", {})
    allowed = set(policy.get("allowedSourceIds", []))
    require(isinstance(records, (list, tuple)), "INVALID_CORPUS_RECORDS")
    limit = int(config.get("maxCorpusBytes", policy.get("maxCorpusBytes", 512 * 1024**2)))
    require(0 < limit <= 20 * 1024**3, "CORPUS_LIMIT_INVALID")
    accepted, excluded, duplicate, total = [], [], [], 0
    seen = {}
    # Parsing/scanning an entire corpus must not block worker heartbeats. Only
    # one record's split assignment is locked by _resolve_split above.
    with nullcontext():
        for record in records:
            require(isinstance(record, dict), "INVALID_CORPUS_RECORD")
            identity = str(record.get("versionId", record.get("id", "")))
            reason = []
            if record.get("trainingPolicy") != "allowed":
                reason.append("SOURCE_NOT_ALLOWED")
            if allowed and record.get("sourceId") not in allowed:
                reason.append("SOURCE_NOT_ALLOWED")
            if record.get("state") in ("deleted", "revoked", "excluded", "quarantined") or record.get("deletedAt") or record.get("revokedAt"):
                reason.append("POLICY_REVOKED")
            if record.get("legacy_demo") or record.get("legacyDemo") or record.get("fixture"):
                reason.append("FIXTURE_OR_DEMO_NOT_TRAINING_DATA")
            origin = record.get("targetOrigin", record.get("origin", "source_span"))
            if origin in ("teacher", "model_generated", "synthetic_model"):
                reason.append("MODEL_GENERATED_TARGET_DISABLED")
            text = record.get("text")
            if not isinstance(text, str) or not text.strip() or not identity or not record.get("documentId") or not record.get("sourceId"):
                reason.append("INVALID_SOURCE_RECORD")
            groups = _groups(record)
            if not groups:
                reason.append("LINEAGE_REQUIRED")
            if record.get("neverTrain") or record.get("split") in ("test", "holdout"):
                reason.append("EVALUATION_DATA_NOT_FOR_TRAINING")
            if isinstance(text, str):
                try:
                    raw = text.encode("utf-8", errors="strict")
                except UnicodeError:
                    raw = b""
                    reason.append("INVALID_UTF8")
                if len(raw) > MAX_DOCUMENT_BYTES:
                    reason.append("DOCUMENT_REQUIRES_TRUSTED_SPLIT")
                reason.extend(secret_reasons(text))
            else:
                raw = b""
            if reason:
                excluded.append({"versionId": identity, "reasonCodes": sorted(set(reason))})
                continue
            if sft_mode and not (isinstance(record.get("instruction"), str) and record.get("instruction")
                                 and isinstance(record.get("output"), str) and record.get("output")
                                 and record.get("reviewStatus") == "approved"):
                excluded.append({"versionId": identity, "reasonCodes": ["SFT_APPROVAL_REQUIRED"]})
                continue
            content_hash = digest(raw)
            try:
                rag_split = record.get("ragSplit")
                preferred = {"train": "train", "validation": "dev", "dev": "dev", "test": "release", "release": "release", "audit": "audit"}.get(rag_split)
                require(rag_split is None or preferred is not None, "SPLIT_LEAKAGE", status=409)
                split = _resolve_split(store, groups, content_hash, preferred)
            except Fault as exc:
                if exc.code != "SPLIT_LEAKAGE":
                    raise
                previous = store.get("foundation_content_split", content_hash, {})
                affected_groups = sorted(set(groups) | set(previous.get("groupIds", [])))
                store.put("foundation_split_conflict", {"contentHash": content_hash,
                          "groupIds": affected_groups, "reasonCode": "SPLIT_LEAKAGE", "state": "quarantined"}, content_hash)
                for group in affected_groups:
                    prior = store.get("foundation_split", digest(group), {})
                    store.put("foundation_split", {"groupId": group, "split": "contaminated",
                              "previousSplit": prior.get("split"), "reasonCode": "SPLIT_LEAKAGE"}, digest(group))
                excluded.append({"versionId": identity, "reasonCodes": ["SPLIT_LEAKAGE"]})
                continue
            # Records already known to be holdout remain holdout, never train.
            sft_hash = digest([record.get("instruction"), record.get("output")]) if "instruction" in record else None
            key = (content_hash, split, sft_hash)
            lineage = {"documentId": record["documentId"], "versionId": identity,
                       "sourceId": record["sourceId"], "groupIds": groups,
                       "policyVersion": record.get("policyVersion"), "trainingPolicy": "allowed"}
            if key in seen:
                accepted[seen[key]]["lineage"].append(lineage)
                duplicate.append(identity)
                continue
            total += len(raw)
            require(total <= limit, "CORPUS_TOO_LARGE", status=413)
            seen[key] = len(accepted)
            accepted.append({"text": text, "hash": content_hash, "contentHash": content_hash,
                             "groupId": groups[0], "groupIds": groups, "split": split,
                             "documentId": record["documentId"], "versionId": identity,
                             "sourceId": record["sourceId"], "language": record.get("language", "unknown"),
                             "lineage": [lineage], "permissionReceipt": record.get("permissionReceipt", {
                                 "trainingPolicy": "allowed", "policyVersion": record.get("policyVersion")}),
                             "targetOrigin": origin, "rawBytes": len(raw)})
            if isinstance(record.get("instruction"), str) and isinstance(record.get("output"), str):
                require(record.get("reviewStatus") == "approved" and record.get("permissionStatus", "allowed") == "allowed",
                        "SFT_APPROVAL_REQUIRED", status=403)
                require(not secret_reasons(record["instruction"] + "\n" + record["output"]), "SECRET_OR_PERSONAL_VALUE", status=403)
                accepted[-1].update(instruction=record["instruction"], output=record["output"],
                                    reviewStatus="approved", permissionStatus="allowed", exampleHash=sft_hash)
    bounded = []
    for row in accepted:
        if len(canonical(row).encode("utf-8")) + 1 > 2*1024**2:
            excluded.append({"versionId": row["versionId"], "reasonCodes": ["DOCUMENT_REQUIRES_TRUSTED_SPLIT"]})
        else:
            bounded.append(row)
    accepted = bounded
    total = sum(row["rawBytes"] for row in accepted)
    accepted.sort(key=lambda item: (item["split"], item["groupId"], item["hash"]))
    excluded.sort(key=lambda item: (item["versionId"], item["reasonCodes"]))
    duplicate.sort()
    split_counts = {split: sum(row["split"] == split for row in accepted) for split in SPLITS}
    split_bytes = {split: sum(row["rawBytes"] for row in accepted if row["split"] == split) for split in SPLITS}
    group_counts = {split: len({group for row in accepted if row["split"] == split for group in row["groupIds"]}) for split in SPLITS}
    reasons = []
    minimum_bytes = 0 if sft_mode else int(policy.get("minCorpusBytes", 16 * 1024**2))
    minimum_groups = (1 if project.get("testOnly") else 3) if sft_mode else int(policy.get("minTrainGroups", 20))
    sft_target_tokens = None
    if sft_mode:
        from .tokenizer import load
        tok = load(store, project["tokenizerId"])
        sft_target_tokens = sum(len(tok.encode_ordinary(row["output"]))+1 for row in accepted if row["split"] == "train")
        minimum_examples = 1 if project.get("testOnly") else 32
        minimum_tokens = 1 if project.get("testOnly") else 4096
        if split_counts["train"] < minimum_examples or group_counts["train"] < minimum_groups or sft_target_tokens < minimum_tokens:
            reasons.append("INSUFFICIENT_APPROVED_SFT_DATA")
    elif split_bytes["train"] < minimum_bytes or group_counts["train"] < minimum_groups:
        reasons.append("INSUFFICIENT_CORPUS")
    if not accepted:
        reasons.append("NO_ELIGIBLE_SOURCE_VERSIONS")
    membership = [{key: row[key] for key in ("hash", "groupIds", "split", "lineage", "exampleHash") if key in row} for row in accepted]
    policy_version = config.get("policyVersion", project.get("policyVersion", policy.get("version", 0)))
    identity = "corpus-" + digest([project["id"], membership, excluded, duplicate, policy_version, minimum_bytes, minimum_groups, sft_mode])[:24]
    destination = inside(store.root, "foundation/corpora/" + identity)
    staging = inside(store.root, "foundation/staging/" + uid())
    staging.mkdir(parents=True)
    try:
        files = {}
        for split in SPLITS:
            name = split + ".jsonl"
            with (staging / name).open("w", encoding="utf-8", newline="\n") as handle:
                for row in accepted:
                    if row["split"] == split:
                        handle.write(canonical(row) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            files[name] = file_hash(staging / name)
        manifest = {"schemaVersion": 1, "kind": "foundation_corpus", "id": identity,
                    "projectId": project["id"], "files": files, "splitCounts": split_counts,
                    "splitBytes": split_bytes, "groupCounts": group_counts,
                    "documentCount": len(accepted), "eligibleBytes": total,
                    "policyVersion": policy_version,
                    "exclusions": excluded, "duplicateVersionIds": duplicate,
                    "modelGeneratedTargetRatio": 0, "members": membership}
        manifest.update(objective="sft" if sft_mode else "causal_lm", sftTargetTokens=sft_target_tokens)
        atomic_json(staging / "manifest.json", manifest)
        manifest_hash = file_hash(staging / "manifest.json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            require(file_hash(destination / "manifest.json") == manifest_hash, "CORPUS_IDENTITY_CONFLICT", status=409)
            for name, checksum in files.items():
                require(file_hash(destination / name) == checksum, "CORPUS_HASH_MISMATCH", status=409)
        else:
            os.replace(staging, destination)
        result = store.put("foundation_corpus", {**{key: value for key, value in manifest.items() if key not in ("members",)},
                           "storagePath": destination.relative_to(store.root).as_posix(),
                           "manifestHash": manifest_hash, "state": "held" if reasons else "ready",
                           "reasonCodes": reasons, "fixture": project.get("testOnly") is True}, identity)
        store.event("foundation.corpus.ready" if not reasons else "foundation.corpus.held", identity, {"reasonCodes": reasons})
        return result
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _near_sketch(text):
    # Bottom-k token-five-gram MinHash sketch; bounded per-document state.
    tokens=re.findall(r'[A-Za-z_]\w*|\d+|[^\s\w]',text)
    if len(tokens)<80:return []
    signatures={int.from_bytes(hashlib.blake2b(' '.join(tokens[i:i+5]).encode(),digest_size=8).digest(),'big')
                for i in range(len(tokens)-4)}
    return heapq.nsmallest(128,signatures)


def restore_near_fingerprints(store):
    """Rehydrate verified, data-only fingerprint records after offline import."""
    with store.tx():
        store.db.execute('CREATE TABLE IF NOT EXISTS foundation_near_documents(namespace TEXT,hash TEXT,split TEXT,groups_json TEXT,sketch_json TEXT,PRIMARY KEY(namespace,hash))')
        store.db.execute('CREATE TABLE IF NOT EXISTS foundation_near_buckets(namespace TEXT,bucket TEXT,hash TEXT,PRIMARY KEY(namespace,bucket,hash))')
    cursor=''
    while True:
        page=store.rows("SELECT id,body FROM records WHERE kind='foundation_near_fingerprint' AND id>? ORDER BY id LIMIT 256",(cursor,))
        if not page:break
        records=[]
        for row in page:
            body=json.loads(row['body']);sketch=body.get('sketch',[])
            require(body.get('namespace') in ('raw','sft') and body.get('split') in SPLITS and re.fullmatch('[0-9a-f]{64}',body.get('contentHash',''))
                and isinstance(sketch,list) and len(sketch)<=128 and all(type(n) is int and 0<=n<2**64 for n in sketch),'NEAR_FINGERPRINT_INVALID')
            records.append((body,sketch))
        with store.tx():
            for body,sketch in records:
                store.db.execute('INSERT OR IGNORE INTO foundation_near_documents VALUES(?,?,?,?,?)',
                    (body['namespace'],body['contentHash'],body['split'],canonical(body['groupIds']),canonical(sketch)))
                for signature in sketch[:8]:store.db.execute('INSERT OR IGNORE INTO foundation_near_buckets VALUES(?,?,?)',(body['namespace'],str(signature),body['contentHash']))
        cursor=page[-1]['id']


def _near_match(store, content_hash, text, split, groups, namespace):
    # Tokenization and MinHash are proportional to the source body and must
    # never block another connection's worker heartbeat writer.
    sketch=_near_sketch(text)
    if not sketch:return None
    with store.tx():
        store.db.execute('CREATE TABLE IF NOT EXISTS foundation_near_documents(namespace TEXT,hash TEXT,split TEXT,groups_json TEXT,sketch_json TEXT,PRIMARY KEY(namespace,hash))')
        store.db.execute('CREATE TABLE IF NOT EXISTS foundation_near_buckets(namespace TEXT,bucket TEXT,hash TEXT,PRIMARY KEY(namespace,bucket,hash))')
        if store.rows('SELECT hash FROM foundation_near_documents WHERE namespace=? AND hash=?',(namespace,content_hash)):return None
        keys=[str(value) for value in sketch[:8]]
        matches=store.rows('SELECT hash,count(*) AS hits FROM foundation_near_buckets WHERE namespace=? AND bucket IN ('+','.join('?' for _ in keys)+') GROUP BY hash HAVING count(*)>=2 ORDER BY hits DESC,hash LIMIT 32',(namespace,*keys))
        for match in matches:
            other=store.rows('SELECT * FROM foundation_near_documents WHERE namespace=? AND hash=?',(namespace,match['hash']))[0]
            old=set(json.loads(other['sketch_json']));new=set(sketch)
            similarity=len(old&new)/len(old|new)
            if similarity>=.90:
                return {'contentHash':other['hash'],'split':other['split'],'groupIds':json.loads(other['groups_json']),
                        'similarity':similarity,'method':'bottom-k-token5-jaccard-v1'}
        store.db.execute('INSERT OR IGNORE INTO foundation_near_documents VALUES(?,?,?,?,?)',
                         (namespace,content_hash,split,canonical(groups),canonical(sketch)))
        for key in keys:store.db.execute('INSERT OR IGNORE INTO foundation_near_buckets VALUES(?,?,?)',(namespace,key,content_hash))
    return None


def _canonical_groups(store,record):
    values=[]
    for group in _groups(record):
        normalized=group.rstrip('/').removesuffix('.git')
        if normalized.startswith('repository:'):normalized=normalized.lower()
        alias=store.get('foundation_repository_family',digest(normalized),{})
        values.append(alias.get('canonicalGroupId',normalized))
    return sorted(set(values))


def build(store, project, records, config=None,*,teacher_validator=None,record_kind='foundation_corpus'):
    """Disk-backed, bounded construction; iterable inputs never become a text list."""
    config=config or {};policy=project.get('policy',{})
    sampling=config.get('samplingPolicy')
    require(sampling is None or sampling=={'kind':'python-focused-targets-v2','weights':{'python':90,'other_code':5,'docs':5}},'SAMPLER_POLICY_INVALID')
    restore_near_fingerprints(store)
    require(not isinstance(records,(str,bytes,dict)) and hasattr(records,'__iter__'),'INVALID_CORPUS_RECORDS')
    sft=config.get('objective')=='sft' or config.get('kind') in ('sft','foundation_sft_corpus')
    require(record_kind in ('foundation_corpus','foundation_teacher_corpus'),'INVALID_CORPUS_KIND')
    require(record_kind!='foundation_teacher_corpus' or sft and callable(teacher_validator),'TEACHER_POLICY_REQUIRED')
    allowed=set(policy.get('allowedSourceIds',[]));limit=int(config.get('maxCorpusBytes',policy.get('maxCorpusBytes',512*1024**2)))
    require(0<limit<=20*1024**3,'CORPUS_LIMIT_INVALID')
    shard_limit=int(config.get('shardBytes',MAX_SHARD_BYTES));require(4096<=shard_limit<=MAX_SHARD_BYTES,'CORPUS_LIMIT_INVALID')
    staging=inside(store.root,'foundation/staging/'+uid());staging.mkdir(parents=True)
    index=apsw.Connection(str(staging/'rows.sqlite'));list(index.execute('PRAGMA journal_mode=OFF;PRAGMA cache_size=-2048;PRAGMA temp_store=FILE;'))
    index.execute('CREATE TABLE accepted(identity TEXT PRIMARY KEY,hash TEXT,split TEXT,group_id TEXT,language TEXT,body TEXT,raw_bytes INTEGER);CREATE INDEX row_order ON accepted(split,group_id,hash);CREATE TABLE exclusions(version TEXT,reasons TEXT);CREATE TABLE duplicates(version TEXT);')
    total=0;offsets=None
    def exclude(identity,reasons):index.execute('INSERT INTO exclusions VALUES(?,?)',(identity,canonical(sorted(set(reasons)))))
    try:
        for record in records:
            require(isinstance(record,dict),'INVALID_CORPUS_RECORD')
            identity=str(record.get('versionId',record.get('id','')));text=record.get('text');reasons=[]
            groups=_canonical_groups(store,record)
            if record.get('trainingPolicy')!='allowed' or allowed and record.get('sourceId') not in allowed:reasons.append('SOURCE_NOT_ALLOWED')
            if record.get('state') in ('deleted','revoked','excluded','quarantined') or record.get('deletedAt') or record.get('revokedAt'):reasons.append('POLICY_REVOKED')
            if record.get('fixture') or record.get('legacyDemo') or record.get('legacy_demo'):reasons.append('FIXTURE_OR_DEMO_NOT_TRAINING_DATA')
            origin=record.get('targetOrigin',record.get('origin','source_span'))
            if origin in ('teacher','model_generated','synthetic_model'):
                if origin=='teacher' and record_kind=='foundation_teacher_corpus':
                    require(teacher_validator(record) is True,'TEACHER_RECORD_UNVERIFIED')
                else:reasons.append('MODEL_GENERATED_TARGET_DISABLED')
            if not isinstance(text,str) or not text.strip() or not identity or not record.get('documentId') or not record.get('sourceId'):reasons.append('INVALID_SOURCE_RECORD')
            if not groups:reasons.append('LINEAGE_REQUIRED')
            if record.get('neverTrain') or record.get('split') in ('test','holdout'):reasons.append('EVALUATION_DATA_NOT_FOR_TRAINING')
            try:raw=text.encode('utf-8',errors='strict') if isinstance(text,str) else b''
            except UnicodeError:raw=b'';reasons.append('INVALID_UTF8')
            if len(raw)>MAX_DOCUMENT_BYTES:reasons.append('DOCUMENT_REQUIRES_TRUSTED_SPLIT')
            if isinstance(text,str):reasons.extend(secret_reasons(text))
            if sft and not (isinstance(record.get('instruction'),str) and record.get('instruction') and isinstance(record.get('output'),str) and record.get('output') and record.get('reviewStatus')=='approved'):
                reasons.append('SFT_APPROVAL_REQUIRED')
            if reasons:exclude(identity,reasons);continue
            has_sft=isinstance(record.get('instruction'),str) and isinstance(record.get('output'),str)
            if has_sft:require(record.get('reviewStatus')=='approved' and record.get('permissionStatus','allowed')=='allowed','SFT_APPROVAL_REQUIRED',status=403)
            checksum=digest(raw);sft_hash=digest([record['instruction'],record['output']]) if has_sft else None
            try:
                rag=record.get('ragSplit');preferred={'train':'train','validation':'dev','dev':'dev','test':'release','release':'release','audit':'audit'}.get(rag)
                require(rag is None or preferred is not None,'SPLIT_LEAKAGE',status=409)
                split=_resolve_split(store,groups,checksum,preferred)
            except Fault as exc:
                if exc.code!='SPLIT_LEAKAGE':raise
                previous=store.get('foundation_content_split',checksum,{})
                old_conflict=store.get('foundation_split_conflict',checksum,{})
                if old_conflict and old_conflict.get('state')!='candidate_excluded':
                    exclude(identity,['SPLIT_LEAKAGE']);continue
                if previous and not set(groups)&set(previous.get('groupIds',[])):
                    # Identical boilerplate in independent origins is excluded
                    # from the newly encountered split. It does not move either
                    # origin, nor poison unrelated documents in both families.
                    store.put('foundation_split_conflict',{'contentHash':checksum,'groupIds':groups,
                        'previousGroupIds':previous.get('groupIds',[]),'reasonCode':'EXACT_DUPLICATE_CROSS_SPLIT',
                        'state':'candidate_excluded','retainedSplit':previous.get('split')},checksum)
                    exclude(identity,['EXACT_DUPLICATE_CROSS_SPLIT']);continue
                affected=sorted(set(groups)|set(previous.get('groupIds',[])))
                store.put('foundation_split_conflict',{'contentHash':checksum,'groupIds':affected,'reasonCode':'SPLIT_LEAKAGE','state':'quarantined'},checksum)
                for group in affected:
                    prior=store.get('foundation_split',digest(group),{})
                    store.put('foundation_split',{'groupId':group,'split':'contaminated','previousSplit':prior.get('split'),'reasonCode':'SPLIT_LEAKAGE'},digest(group))
                exclude(identity,['SPLIT_LEAKAGE']);continue
            lineage={'documentId':record['documentId'],'versionId':identity,'sourceId':record['sourceId'],'groupIds':groups,
                     'policyVersion':record.get('policyVersion'),'trainingPolicy':'allowed'}
            key=digest([checksum,split,sft_hash]);existing=list(index.execute('SELECT body FROM accepted WHERE identity=?',(key,)))
            if existing:
                old=json.loads(existing[0][0]);old['lineage'].append(lineage);old['lineage'].sort(key=canonical)
                index.execute('UPDATE accepted SET body=? WHERE identity=?',(canonical(old),key));index.execute('INSERT INTO duplicates VALUES(?)',(identity,));continue
            near=None if project.get('testOnly') and config.get('skipNearDedupForStress') is True else _near_match(store,checksum,text,split,groups,'sft' if sft else 'raw')
            if near:
                exclude(identity,['NEAR_DUPLICATE_SPLIT_LEAKAGE' if near['split']!=split else 'NEAR_DUPLICATE'])
                store.put('foundation_near_duplicate',{'contentHash':checksum,'representative':near,'groupIds':groups,'split':split,'state':'excluded'},checksum)
                continue
            receipt={k:v for k,v in record.get('permissionReceipt',{'trainingPolicy':'allowed','policyVersion':record.get('policyVersion')}).items() if k!='checkedAt'}
            row={'text':text,'hash':checksum,'contentHash':checksum,'groupId':groups[0],'groupIds':groups,'split':split,
                 'documentId':record['documentId'],'versionId':identity,'sourceId':record['sourceId'],'language':record.get('language','unknown'),
                 'lineage':[lineage],'permissionReceipt':receipt,'targetOrigin':origin,'rawBytes':len(raw)}
            teacher_fields=('teacherArtifactId','teacherManifestHash','verifierReceiptId','verifierReceiptHash','sourceTaskHash',
                'teacherApprovalId','teacherQueryId','probeReceiptId','teacherPolicyId','teacherPolicyVersion','fixtureId','generationProfileHash','generationPurpose')
            for key_name in ('sourceSpan','provenance','independentVerification','approvalReceipt',*teacher_fields):
                if key_name in record:row[key_name]=record[key_name]
            if has_sft:
                require(record.get('permissionStatus','allowed')=='allowed','SFT_APPROVAL_REQUIRED',status=403)
                require(not secret_reasons(record['instruction']+'\n'+record['output']),'SECRET_OR_PERSONAL_VALUE',status=403)
                row.update(instruction=record['instruction'],output=record['output'],reviewStatus='approved',permissionStatus='allowed',exampleHash=sft_hash)
            if len(canonical(row).encode())+1>MAX_LINE_BYTES:exclude(identity,['DOCUMENT_REQUIRES_TRUSTED_SPLIT']);continue
            total+=len(raw);require(total<=limit,'CORPUS_TOO_LARGE',status=413)
            index.execute('INSERT INTO accepted VALUES(?,?,?,?,?,?,?)',(key,checksum,split,groups[0],row['language'],canonical(row),len(raw)))
        split_counts={s:index.execute('SELECT count(*) FROM accepted WHERE split=?',(s,)).fetchone()[0] for s in SPLITS}
        split_bytes={s:index.execute('SELECT coalesce(sum(raw_bytes),0) FROM accepted WHERE split=?',(s,)).fetchone()[0] for s in SPLITS}
        group_counts={s:index.execute('SELECT count(DISTINCT group_id) FROM accepted WHERE split=?',(s,)).fetchone()[0] for s in SPLITS}
        minimum_bytes=0 if sft else int(policy.get('minCorpusBytes',16*1024**2))
        minimum_groups=(1 if project.get('testOnly') else 3) if sft else int(policy.get('minTrainGroups',20))
        reasons=[];sft_targets=None;teacher_targets=0
        if sft:
            from .tokenizer import load
            tok=load(store,project['tokenizerId']);sft_targets=0
            for (body,) in index.execute("SELECT body FROM accepted WHERE split='train'"):
                row=json.loads(body);count=len(tok.encode_ordinary(row['output']))+1;sft_targets+=count
                if row.get('targetOrigin')=='teacher':teacher_targets+=count
            if split_counts['train']<(1 if project.get('testOnly') else 32) or group_counts['train']<minimum_groups or sft_targets<(1 if project.get('testOnly') else 4096):reasons.append('INSUFFICIENT_APPROVED_SFT_DATA')
        elif split_bytes['train']<minimum_bytes or group_counts['train']<minimum_groups:reasons.append('INSUFFICIENT_CORPUS')
        if not sum(split_counts.values()):reasons.append('NO_ELIGIBLE_SOURCE_VERSIONS')
        files={};split_shards={};inline_members=[];member_size=0
        offsets=apsw.Connection(str(staging/'row-index.sqlite'))
        list(offsets.execute('PRAGMA journal_mode=OFF;PRAGMA synchronous=OFF;CREATE TABLE positions(id INTEGER PRIMARY KEY,split TEXT,group_id TEXT,language TEXT,file_name TEXT,offset INTEGER,length INTEGER,hash TEXT,raw_bytes INTEGER);CREATE INDEX positions_group ON positions(split,group_id,id);'))
        with (staging/'members.jsonl').open('wb') as members:
            for split in SPLITS:
                names=[];handle=None;size=0;part=0
                def open_part(number):
                    name=split+('.jsonl' if number==0 else f'-{number:05d}.jsonl');names.append(name)
                    return (staging/name).open('wb')
                handle=open_part(0)
                try:
                    for (encoded,) in index.execute('SELECT body FROM accepted WHERE split=? ORDER BY group_id,hash,identity',(split,)):
                        data=(encoded+'\n').encode()
                        if size and size+len(data)>shard_limit:
                            handle.flush();os.fsync(handle.fileno());handle.close();part+=1;handle=open_part(part);size=0
                        offset=handle.tell();handle.write(data);size+=len(data)
                        row=json.loads(encoded);member={k:row[k] for k in ('hash','groupIds','split','lineage','exampleHash','targetOrigin',*teacher_fields) if k in row}
                        offsets.execute('INSERT INTO positions(split,group_id,language,file_name,offset,length,hash,raw_bytes) VALUES(?,?,?,?,?,?,?,?)',
                            (split,row['groupId'],row['language'],names[-1],offset,len(data),row['hash'],row['rawBytes']))
                        member_raw=(canonical(member)+'\n').encode();members.write(member_raw);member_size+=len(member_raw)
                        if member_size<=65536:inline_members.append(member)
                        else:inline_members=[]
                finally:
                    handle.flush();os.fsync(handle.fileno());handle.close()
                split_shards[split]=names
                for name in names:files[name]=file_hash(staging/name)
            members.flush();os.fsync(members.fileno())
        files['members.jsonl']=file_hash(staging/'members.jsonl')
        offsets.close();offsets=None;files['row-index.sqlite']=file_hash(staging/'row-index.sqlite')
        exclusions=[{'versionId':r[0],'reasonCodes':json.loads(r[1])} for r in index.execute('SELECT version,reasons FROM exclusions ORDER BY version,reasons LIMIT 1000')]
        duplicates=[r[0] for r in index.execute('SELECT version FROM duplicates ORDER BY version LIMIT 1000')]
        for table,name in (('exclusions','exclusions.jsonl'),('duplicates','duplicates.jsonl')):
            with (staging/name).open('w',encoding='utf-8',newline='\n') as handle:
                for row in index.execute('SELECT * FROM '+table+' ORDER BY version'):handle.write(canonical(list(row))+'\n')
            files[name]=file_hash(staging/name)
        index.close();index=None;(staging/'rows.sqlite').unlink()
        policy_version=config.get('policyVersion',project.get('policyVersion',policy.get('version',0)))
        identity='corpus-'+digest([project['id'],files,policy_version,minimum_bytes,minimum_groups,sft,record_kind,config.get('teacherPolicyHash'),*([sampling] if sampling is not None else [])])[:24]
        ratio=teacher_targets/sft_targets if sft_targets else 0
        require(record_kind!='foundation_teacher_corpus' or ratio<=.10,'TEACHER_FRACTION_EXCEEDED')
        manifest={'schemaVersion':2,'kind':record_kind,'id':identity,'projectId':project['id'],'files':files,
            'splitShards':split_shards,'rowIndex':'row-index.sqlite','maxShardBytes':shard_limit,'splitCounts':split_counts,'splitBytes':split_bytes,'groupCounts':group_counts,
            'documentCount':sum(split_counts.values()),'eligibleBytes':total,'policyVersion':policy_version,'exclusions':exclusions,
            'duplicateVersionIds':duplicates,'modelGeneratedTargetRatio':ratio,'teacherTargetTokens':teacher_targets,'members':inline_members,'membersFile':'members.jsonl',
            'deduplication':{'exact':True,'nearMethod':'bottom-k-token5-jaccard-v1','nearThreshold':.90,'forkFamilyCanonicalization':True},
            'objective':'sft' if sft else 'causal_lm','sftTargetTokens':sft_targets,
            **({'samplingPolicy':sampling} if sampling is not None else {}),
            **({'teacherPolicyHash':config.get('teacherPolicyHash'),'teacherPolicyId':config.get('teacherPolicyId'),
                'teacherPolicyVersion':config.get('teacherPolicyVersion'),'actualTeacherFraction':ratio} if record_kind=='foundation_teacher_corpus' else {})}
        atomic_json(staging/'manifest.json',manifest);manifest_hash=file_hash(staging/'manifest.json')
        destination=inside(store.root,'foundation/corpora/'+identity);destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():require(file_hash(destination/'manifest.json')==manifest_hash,'CORPUS_IDENTITY_CONFLICT',status=409)
        else:os.replace(staging,destination)
        result=store.put('foundation_corpus',{**{k:v for k,v in manifest.items() if k!='members'},'storagePath':destination.relative_to(store.root).as_posix(),
            'manifestHash':manifest_hash,'state':'held' if reasons else 'ready','reasonCodes':reasons,'fixture':project.get('testOnly') is True},identity)
        if record_kind=='foundation_teacher_corpus':store.put(record_kind,result,identity)
        store.event('foundation.corpus.held' if reasons else 'foundation.corpus.ready',identity,{'reasonCodes':reasons})
        return result
    finally:
        if offsets is not None:offsets.close()
        if index is not None:index.close()
        if staging.exists():shutil.rmtree(staging)


def resolve(store,record_or_id):
    identity=record_or_id if isinstance(record_or_id,str) else record_or_id['id']
    record=store.get('foundation_corpus',identity,{}) or store.get('foundation_teacher_corpus',identity,{})
    require(record,'CORPUS_NOT_FOUND',status=404)
    if not isinstance(record_or_id,str):require(record.get('manifestHash')==record_or_id.get('manifestHash'),'CORPUS_HASH_MISMATCH')
    require(record.get('state') not in ('revoked','failed'),'POLICY_REVOKED',status=409)
    if record.get('kind')=='foundation_teacher_corpus':
        from .response_distillation import validate_corpus
        validate_corpus(store,record)
    return record


def verify(store, record_or_id):
    record=resolve(store,record_or_id)
    require(record.get("state") not in ("revoked", "failed"), "POLICY_REVOKED", status=409)
    root = inside(store.root, record["storagePath"])
    require(file_hash(root / "manifest.json") == record["manifestHash"], "CORPUS_HASH_MISMATCH", status=409)
    manifest = json.loads((root / "manifest.json").read_text("utf-8"))
    for name, checksum in manifest["files"].items():
        require(re.fullmatch(r'(?:train|dev|release|audit)(?:-\d{5})?\.jsonl|members\.jsonl|exclusions\.jsonl|duplicates\.jsonl|row-index\.sqlite',name), "CORPUS_FILE_INVALID")
        require(file_hash(inside(root, name)) == checksum, "CORPUS_HASH_MISMATCH", status=409)
    return root, manifest


def rows(store, record_or_id, split="train", *, purpose=None):
    require(split in SPLITS, "INVALID_CORPUS_SPLIT")
    record=resolve(store,record_or_id)
    root, manifest = verify(store, record)
    require(purpose not in ("train", "tokenizer") or split == "train", "SPLIT_LEAKAGE", status=409)
    names=manifest.get('splitShards',{}).get(split,[split+'.jsonl'])
    for name in names:
      require(name in manifest['files'],'CORPUS_FILE_INVALID')
      with (root / name).open("rb") as handle:
        while True:
            line=handle.readline(MAX_LINE_BYTES+1)
            if not line:break
            require(len(line) <= MAX_LINE_BYTES, "CORPUS_RECORD_TOO_LARGE")
            row = json.loads(line)
            if record.get('kind')=='foundation_teacher_corpus':
                from .response_distillation import validate_row
                validate_row(store,record,row)
            require(row.get("split") == split and digest(row["text"].encode()) == row["hash"], "CORPUS_HASH_MISMATCH", status=409)
            require(store.get("foundation_split_conflict", row["hash"], {}).get('state','candidate_excluded')=='candidate_excluded' and all(
                    store.get("foundation_split", digest(group), {}).get("split") != "contaminated"
                    for group in row["groupIds"]), "SPLIT_LEAKAGE", status=409)
            if purpose:
                key = digest([row["hash"], purpose, record["projectId"], record["id"]])
                store.put("foundation_exposure", {"contentHash": row["hash"], "groupIds": row["groupIds"],
                          "split": split, "purpose": purpose, "projectId": record["projectId"], "corpusId": record["id"], "observedAt": now()}, key)
            yield row


def iter_members(store,record_or_id,*,allow_revoked=False):
    record=store.get('foundation_corpus',record_or_id) if isinstance(record_or_id,str) else record_or_id
    require(allow_revoked or record.get('state') not in ('revoked','failed'),'POLICY_REVOKED',status=409)
    root=inside(store.root,record['storagePath'])
    require(file_hash(root/'manifest.json')==record['manifestHash'],'CORPUS_HASH_MISMATCH')
    manifest=json.loads((root/'manifest.json').read_text('utf-8'))
    if manifest.get('membersFile'):
        name=manifest['membersFile'];require(name=='members.jsonl','CORPUS_FILE_INVALID')
        require(file_hash(root/name)==manifest['files'][name],'CORPUS_HASH_MISMATCH')
        with (root/name).open('rb') as handle:
            while line:=handle.readline(MAX_LINE_BYTES+1):
                require(len(line)<=MAX_LINE_BYTES,'CORPUS_RECORD_TOO_LARGE');yield json.loads(line)
    else:yield from manifest.get('members',[])


class BalancedRows:
    """SQL cursors + actual-token feedback; no group holds a source text buffer.

    Quotas are absolute scored targets per pass, not a claim that small corpora
    can magically satisfy percentage caps. Exhaustion and actual shares are
    reported explicitly. Packing applies the returned remaining target budget.
    """
    def __init__(self,store,record,split,*,purpose='train',seed=42,group_limit=1_000_000,language_limit=8_000_000,target_weights=None):
        record=resolve(store,record)
        self.store,self.record,self.split,self.purpose=store,record,split,purpose
        self.root,manifest=verify(store,record);self.seed=seed
        self.db=apsw.Connection(str(self.root/manifest['rowIndex']),flags=apsw.SQLITE_OPEN_READONLY)
        self.cursors={row[0]:0 for row in self.db.execute('SELECT DISTINCT group_id FROM positions WHERE split=?',(split,))}
        require(target_weights is None or target_weights=={'python':90,'other_code':5,'docs':5},'SAMPLER_POLICY_INVALID')
        self.target_weights=target_weights;self.category_tokens={'python':0,'other_code':0,'docs':0}
        self.pending={};self.group_tokens={g:0 for g in self.cursors};self.language_tokens={}
        self.group_limit=group_limit;self.language_limit=language_limit;self.active=None;self.reasons=set();self.closed=False

    def __iter__(self):return self

    @staticmethod
    def _category(language):
        return 'python' if language in ('python','py','pyi') else 'docs' if language in ('md','rst','markdown','text','txt') else 'other_code'

    def feedback(self,count):
        require(type(count) is int and count>=0,'SAMPLER_TARGET_COUNT_INVALID')
        if self.active:
            group,language=self.active
            self.category_tokens[self._category(language)]+=count
            self.group_tokens[group]+=count;self.language_tokens[language]=self.language_tokens.get(language,0)+count;self.active=None

    def __next__(self):
        if self.closed:raise StopIteration
        for group,cursor in list(self.cursors.items()):
            if group not in self.pending:
                found=self.db.execute('SELECT id,language,file_name,offset,length,hash,raw_bytes FROM positions WHERE split=? AND group_id=? AND id>? ORDER BY id LIMIT 1',(self.split,group,cursor)).fetchone()
                if found:self.pending[group]=found
                else:self.cursors.pop(group)
        candidates=[]
        for group,row in self.pending.items():
            if self.group_tokens[group]>=self.group_limit:self.reasons.add('GROUP_TARGET_QUOTA');continue
            if self.language_tokens.get(row[1],0)>=self.language_limit:self.reasons.add('LANGUAGE_TARGET_QUOTA');continue
            candidates.append(group)
        if not candidates:self.close();raise StopIteration
        group=min(candidates,key=lambda g:((self.category_tokens[self._category(self.pending[g][1])]/self.target_weights[self._category(self.pending[g][1])] if self.target_weights else self.language_tokens.get(self.pending[g][1],0)),self.group_tokens[g],digest([self.seed,g])))
        position=self.pending.pop(group);self.cursors[group]=position[0];language=position[1]
        with (self.root/position[2]).open('rb') as handle:handle.seek(position[3]);data=handle.read(position[4])
        require(len(data)==position[4] and len(data)<=MAX_LINE_BYTES,'CORPUS_RECORD_TOO_LARGE')
        row=json.loads(data)
        if self.record.get('kind')=='foundation_teacher_corpus':
            from .response_distillation import validate_row
            validate_row(self.store,self.record,row)
        require(digest(row['text'].encode())==position[5] and row['split']==self.split,'CORPUS_HASH_MISMATCH')
        require(self.store.get('foundation_split_conflict',row['hash'],{}).get('state','candidate_excluded')=='candidate_excluded' and all(
            self.store.get('foundation_split',digest(g),{}).get('split')!='contaminated' for g in row['groupIds']),'SPLIT_LEAKAGE',status=409)
        if self.purpose:
            key=digest([row['hash'],self.purpose,self.record['projectId'],self.record['id']])
            self.store.put('foundation_exposure',{'contentHash':row['hash'],'groupIds':row['groupIds'],'split':self.split,
                'purpose':self.purpose,'projectId':self.record['projectId'],'corpusId':self.record['id'],'observedAt':now()},key)
        self.active=(group,language)
        return {**row,'_samplerTargetBudget':min(self.group_limit-self.group_tokens[group],self.language_limit-self.language_tokens.get(language,0))}

    def close(self):
        if self.closed:return
        self.closed=True;self.db.close()
        total=sum(self.group_tokens.values())
        self.store.put('foundation_sampler_report',{'corpusId':self.record['id'],'projectId':self.record['projectId'],
            'method':'python-focused-targets-v2' if self.target_weights else 'actual-target-fair-round-robin-v1','seed':self.seed,'groupTargetLimit':self.group_limit,
            'languageTargetLimit':self.language_limit,'observedTargetTokens':total,'groupTargetTokens':self.group_tokens,
            'languageTargetTokens':self.language_tokens,'categoryTargetTokens':self.category_tokens,'targetWeights':self.target_weights,'reasonCodes':sorted(self.reasons),'observedAt':now()},digest([self.record['id'],self.purpose,self.seed]))

    def __del__(self):
        try:self.close()
        except Exception:pass


def balanced_rows(store,record,split='train',*,purpose='train',seed=42,group_limit=1_000_000,language_limit=8_000_000,target_weights=None):
    if record.get('rowIndex'):
        return BalancedRows(store,record,split,purpose=purpose,seed=seed,group_limit=group_limit,language_limit=language_limit,target_weights=target_weights)
    return rows(store,record,split,purpose=purpose)


def rows_page(store,record_or_id,split='train',*,cursor=None,limit=128):
    """Bounded immutable SQL offset page, suitable for saved data cursors."""
    record=resolve(store,record_or_id);root,manifest=verify(store,record)
    require(split in SPLITS and type(limit) is int and 1<=limit<=1024,'INVALID_PAGE_LIMIT')
    require(manifest.get('rowIndex')=='row-index.sqlite','CORPUS_SQL_INDEX_REQUIRED')
    binding={'corpusId':record['id'],'manifestHash':record['manifestHash'],'split':split}
    after=0
    if cursor:
        require({k:cursor.get(k) for k in binding}==binding and type(cursor.get('ordinal')) is int and cursor['ordinal']>=0,'STALE_CORPUS_CURSOR')
        after=cursor['ordinal']
    db=apsw.Connection(str(root/manifest['rowIndex']),flags=apsw.SQLITE_OPEN_READONLY)
    try:positions=list(db.execute('SELECT id,file_name,offset,length,hash FROM positions WHERE split=? AND id>? ORDER BY id LIMIT ?',(split,after,limit+1)))
    finally:db.close()
    output=[]
    for ordinal,name,offset,length,checksum in positions[:limit]:
        require(name in manifest['files'] and length<=MAX_LINE_BYTES,'CORPUS_FILE_INVALID')
        with (root/name).open('rb') as handle:handle.seek(offset);row=json.loads(handle.read(length))
        require(digest(row['text'].encode())==checksum and row['split']==split,'CORPUS_HASH_MISMATCH')
        require(all(store.get('foundation_split',digest(g),{}).get('split')!='contaminated' for g in row['groupIds']),'SPLIT_LEAKAGE',status=409)
        if record.get('kind')=='foundation_teacher_corpus':
            from .response_distillation import validate_row
            validate_row(store,record,row)
        output.append(row)
    return {'records':output,'nextCursor':{**binding,'ordinal':positions[limit-1][0]} if len(positions)>limit else None,
        'totalRows':manifest['splitCounts'][split],'manifestHash':record['manifestHash']}


build_corpus = build
iter_rows = rows
