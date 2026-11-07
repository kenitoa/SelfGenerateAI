"""Immutable split-bound caches of ordinary token IDs, stored as LE uint16.

The source iterator still performs policy/exposure checks on cache hits. This
module also rechecks current record state before returning any cached IDs.
"""
from __future__ import annotations

from array import array
import json
import os
import re
import shutil
import sys

from LLM.core import require, digest, file_hash, atomic_json, inside, uid

CHUNK_TOKENS = 4096
MAX_TEXT_BYTES = 2 * 1024**2
MAX_TOKENS = MAX_TEXT_BYTES


def supports(tokenizer):
    return bool(getattr(tokenizer, "tokenizer_hash", None)
                and isinstance(getattr(tokenizer, "record", None), dict)
                and tokenizer.record.get("id") and callable(getattr(tokenizer, "encode_ordinary", None)))


def _policy(store, corpus, tokenizer):
    from .corpus import resolve
    current = resolve(store,corpus['id'])
    require(current.get("state") in ("ready", "verified") and current.get("manifestHash") == corpus.get("manifestHash"),
            "POLICY_REVOKED", status=409)
    token_record = store.get("foundation_tokenizer", tokenizer.record["id"])
    require(token_record.get("state") in ("ready", "verified")
            and token_record.get("tokenizerHash") == tokenizer.tokenizer_hash
            and token_record.get("projectId") == current.get("projectId"),
            "TOKENIZER_HASH_MISMATCH", status=409)
    project = store.get("foundation_project", current["projectId"])
    require(project.get("state") != "revoked" and project.get("stage") != "revoked", "POLICY_REVOKED", status=409)
    return current, project


def _path(store, expected):
    current = store.root
    for part in expected.split("/"):
        current = current / part
        require(not current.is_symlink(), "TOKEN_SHARD_PATH_INVALID")
    return inside(store.root, expected)


def _root(store, record):
    require(isinstance(record.get("id"), str) and re.fullmatch(r"[0-9a-f]{64}", record["id"]),
            "TOKEN_SHARD_PATH_INVALID")
    expected = "foundation/token-shards/" + record["id"]
    require(record.get("storagePath") == expected, "TOKEN_SHARD_PATH_INVALID")
    root = _path(store, expected)
    require(root.is_dir(), "TOKEN_SHARD_MISSING")
    return root


def iter_tokens(store, corpus, tokenizer, record):
    _policy(store, corpus, tokenizer)
    require(record.get("state") == "ready" and record.get("corpusId") == corpus["id"]
            and record.get("tokenizerHash") == tokenizer.tokenizer_hash
            and digest(record.get("identity")) == record.get("id"), "TOKEN_SHARD_IDENTITY_MISMATCH")
    current = store.get("foundation_token_shard", record["id"])
    require(current.get("state") == "ready" and current.get("manifestHash") == record.get("manifestHash")
            and current.get("identity") == record.get("identity"), "TOKEN_SHARD_IDENTITY_MISMATCH")
    root = _root(store, record)
    for name in ("manifest.json", "index.json", "tokens.u16"):
        require((root/name).is_file() and not (root/name).is_symlink(), "TOKEN_SHARD_PATH_INVALID")
    require((root/"manifest.json").stat().st_size <= 4096
            and (root/"index.json").stat().st_size <= 131072, "TOKEN_SHARD_SCHEMA_INVALID")
    require(file_hash(root/"manifest.json") == record["manifestHash"], "TOKEN_SHARD_HASH_MISMATCH")
    manifest = json.loads((root/"manifest.json").read_text("utf-8"))
    require(manifest.get("schemaVersion") == 1 and manifest.get("id") == record["id"], "TOKEN_SHARD_SCHEMA_INVALID")
    require(set(manifest.get("files", {})) == {"index.json", "tokens.u16"}, "TOKEN_SHARD_SCHEMA_INVALID")
    for name, expected in manifest["files"].items():
        require(name in ("index.json", "tokens.u16") and file_hash(root/name) == expected, "TOKEN_SHARD_HASH_MISMATCH")
    index = json.loads((root/"index.json").read_text("utf-8"))
    require(index.get("identity") == record.get("identity") and index.get("dtype") == "uint16"
            and index.get("byteOrder") == "little" and index.get("chunkTokens") == CHUNK_TOKENS,
            "TOKEN_SHARD_IDENTITY_MISMATCH")
    count = index.get("tokenCount")
    require(isinstance(count, int) and not isinstance(count, bool) and 0 <= count <= MAX_TOKENS,
            "TOKEN_SHARD_SCHEMA_INVALID")
    require((root/"tokens.u16").stat().st_size == count*2 and record.get("tokenCount") == count,
            "TOKEN_SHARD_LENGTH_MISMATCH")
    expected_chunks = index.get("chunkHashes")
    require(isinstance(expected_chunks, list) and len(expected_chunks) == (count+CHUNK_TOKENS-1)//CHUNK_TOKENS,
            "TOKEN_SHARD_SCHEMA_INVALID")
    read_count = 0
    with (root/"tokens.u16").open("rb") as handle:
        for checksum in expected_chunks:
            raw = handle.read(CHUNK_TOKENS*2)
            require(raw and len(raw)%2 == 0 and digest(raw) == checksum, "TOKEN_SHARD_HASH_MISMATCH")
            ids = array("H")
            ids.frombytes(raw)
            if sys.byteorder != "little":
                ids.byteswap()
            for value in ids:
                require(12 <= value < tokenizer.vocab_size, "TOKEN_SHARD_RESERVED_OR_INVALID_ID")
                yield value
                read_count += 1
        require(not handle.read(1) and read_count == count, "TOKEN_SHARD_LENGTH_MISMATCH")
    _policy(store, corpus, tokenizer)


def encode(store, corpus, tokenizer, row, text, *, component="raw", split="train"):
    if not supports(tokenizer):
        return tokenizer.encode_ordinary(text)
    require(component in ("raw", "instruction", "target") and split in ("train", "dev", "release", "audit"),
            "TOKEN_SHARD_SCHEMA_INVALID")
    require(row.get("split") == split and isinstance(text, str) and isinstance(row.get("text"), str),
            "TOKEN_SHARD_SPLIT_MISMATCH")
    corpus, project = _policy(store, corpus, tokenizer)
    raw = text.encode("utf-8", errors="strict")
    require(len(raw) <= MAX_TEXT_BYTES, "TOKEN_SHARD_TEXT_TOO_LARGE")
    row_bytes = row["text"].encode("utf-8", errors="strict")
    require(len(row_bytes) <= MAX_TEXT_BYTES, "TOKEN_SHARD_TEXT_TOO_LARGE")
    row_hash = digest(row_bytes)
    require((row.get("hash") is None or row["hash"] == row_hash)
            and (row.get("contentHash") is None or row["contentHash"] == row_hash), "CORPUS_HASH_MISMATCH")
    identity = {"corpusId": corpus["id"], "corpusHash": corpus["manifestHash"],
                "tokenizerId": tokenizer.record["id"], "tokenizerHash": tokenizer.tokenizer_hash,
                "rowContentHash": row_hash, "ordinaryTextHash": digest(raw), "component": component,
                "split": split, "encoderVersion": "ordinary-uint16-v1"}
    key = digest(identity)
    old = store.get("foundation_token_shard", key, {})
    if old:
        require(old.get("identity") == identity, "TOKEN_SHARD_IDENTITY_MISMATCH")
        if old.get('state') == 'ready':
            return list(iter_tokens(store, corpus, tokenizer, old))
        require(old.get('state') in ('evicted','deleted'), 'TOKEN_SHARD_IDENTITY_MISMATCH')
    ids = tokenizer.encode_ordinary(text)
    require(len(ids) <= MAX_TOKENS and all(isinstance(i, int) and not isinstance(i, bool)
            and 12 <= i < tokenizer.vocab_size and i <= 65535 for i in ids), "TOKEN_SHARD_RESERVED_OR_INVALID_ID")
    # A durable batch is charged once, then each immutable publication consumes
    # only its actual bytes. Every write has an expiring publication fence.
    work=None
    if not project.get("testOnly"):
        from .storage_reservations import begin
        work=begin(store,project,len(ids)*2+16384+((len(ids)+CHUNK_TOKENS-1)//CHUNK_TOKENS)*80)
    return _publish(store,corpus,tokenizer,identity,key,project,ids,component,split,work)


def _publish(store,corpus,tokenizer,identity,key,project,ids,component,split,work):
  from .storage_reservations import assert_work,commit,abort
  staging = _path(store, "foundation/staging/token-shard-" + uid())
  destination = _path(store, "foundation/token-shards/" + key)
  try:
    staging.mkdir(parents=True)
    chunk_hashes = []
    with (staging/"tokens.u16").open("wb") as handle:
        for offset in range(0, len(ids), CHUNK_TOKENS):
            values = array("H", ids[offset:offset+CHUNK_TOKENS])
            if sys.byteorder != "little":
                values.byteswap()
            data = values.tobytes()
            chunk_hashes.append(digest(data))
            handle.write(data)
        handle.flush(); os.fsync(handle.fileno())
    index = {"schemaVersion": 1, "identity": identity, "dtype": "uint16", "byteOrder": "little",
             "tokenCount": len(ids), "chunkTokens": CHUNK_TOKENS, "chunkHashes": chunk_hashes}
    atomic_json(staging/"index.json", index)
    manifest = {"schemaVersion": 1, "id": key, "kind": "foundation_token_shard",
                "files": {name: file_hash(staging/name) for name in ("tokens.u16", "index.json")}}
    atomic_json(staging/"manifest.json", manifest)
    destination = _path(store, "foundation/token-shards/" + key)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with store.tx():
        if work:assert_work(store,work)
        _policy(store, corpus, tokenizer)
        old = store.get("foundation_token_shard", key, {})
        require(not old or old.get('identity') == identity and old.get('state') in ('ready','evicted','deleted'),
                'TOKEN_SHARD_IDENTITY_MISMATCH')
        if destination.exists():
            require(not destination.is_symlink() and file_hash(destination/"manifest.json") == file_hash(staging/"manifest.json"),
                    "TOKEN_SHARD_IMMUTABLE", status=409)
            shutil.rmtree(staging)
        else:
            staging.replace(destination)
            if os.name != "nt":
                descriptor = os.open(destination.parent, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        if not old or old.get('state') in ('evicted','deleted'):
            old = store.put("foundation_token_shard", {"projectId": project["id"], "corpusId": corpus["id"],
                "tokenizerId": tokenizer.record["id"], "tokenizerHash": tokenizer.tokenizer_hash,
                "identity": identity, "split": split, "component": component, "state": "ready",
                "tokenCount": len(ids), "storagePath": destination.relative_to(store.root).as_posix(),
                "manifestHash": file_hash(destination/"manifest.json"),
                "sizeBytes": sum(p.stat().st_size for p in destination.iterdir() if p.is_file()),
                "testOnly": project.get("testOnly") is True}, key)
            if work:commit(store,work,old['sizeBytes'])
        elif work:commit(store,work,0)
    return list(iter_tokens(store, corpus, tokenizer, old))
  except BaseException:
    if staging.exists():shutil.rmtree(staging,ignore_errors=True)
    if work:
        registered=store.get('foundation_token_shard',key,{})
        removed=not staging.exists() and (not destination.exists() or registered.get('state')=='ready')
        abort(store,work,files_removed=removed)
    raise
