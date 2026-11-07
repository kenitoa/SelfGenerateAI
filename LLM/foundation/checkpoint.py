"""Atomic, pickle-free model/optimizer/RNG checkpoints with fenced DB publication."""
from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path

from LLM.core import require, digest, file_hash, atomic_json, uid, now, inside
from .schemas import record_path, parameter_count


def _pack(value, tensors):
    import torch
    if isinstance(value, torch.Tensor):
        require(bool(torch.isfinite(value).all()) if value.is_floating_point() else True, "CHECKPOINT_NONFINITE")
        key = "tensor_" + str(len(tensors))
        tensors[key] = value.detach().cpu().contiguous().clone()
        return {"type": "tensor", "key": key}
    if value is None or isinstance(value, (bool, str, int)):
        return {"type": "scalar", "value": value}
    if isinstance(value, float):
        require(math.isfinite(value), "CHECKPOINT_NONFINITE")
        return {"type": "scalar", "value": value}
    if isinstance(value, dict):
        require(all(isinstance(k, (int, str)) and not isinstance(k, bool) for k in value), "CHECKPOINT_SCHEMA_INVALID")
        return {"type": "dict", "items": [[_pack(k, tensors), _pack(v, tensors)] for k, v in value.items()]}
    if isinstance(value, (tuple, list)):
        return {"type": "tuple" if isinstance(value, tuple) else "list", "items": [_pack(v, tensors) for v in value]}
    require(False, "CHECKPOINT_UNSUPPORTED_STATE", pythonType=type(value).__name__)


def _unpack(value, tensors, depth=0):
    require(depth <= 64 and isinstance(value, dict), "CHECKPOINT_SCHEMA_INVALID")
    kind = value.get("type")
    if kind == "scalar":
        scalar = value.get("value")
        require(scalar is None or isinstance(scalar, (str, int, float, bool)), "CHECKPOINT_SCHEMA_INVALID")
        require(not isinstance(scalar, float) or math.isfinite(scalar), "CHECKPOINT_NONFINITE")
        return scalar
    if kind == "tensor":
        require(value.get("key") in tensors, "CHECKPOINT_SCHEMA_INVALID")
        return tensors[value["key"]]
    if kind in ("tuple", "list", "dict"):
        items = value.get("items")
        require(isinstance(items, list), "CHECKPOINT_SCHEMA_INVALID")
        if kind == "dict":
            result = {}
            for item in items:
                require(isinstance(item, list) and len(item) == 2, "CHECKPOINT_SCHEMA_INVALID")
                key = _unpack(item[0], tensors, depth+1)
                require(isinstance(key, (str, int)) and not isinstance(key, bool) and key not in result,
                        "CHECKPOINT_SCHEMA_INVALID")
                result[key] = _unpack(item[1], tensors, depth+1)
            return result
        data = [_unpack(v, tensors, depth+1) for v in items]
        return tuple(data) if kind == "tuple" else data
    require(False, "CHECKPOINT_SCHEMA_INVALID")


def _sync(path):
    with Path(path).open("r+b") as handle:
        handle.flush()
        os.fsync(handle.fileno())


def save(store, model, config, tokenizer_record, state, optimizer=None, *, job=None,
         record_kind="foundation_checkpoint", record_id=None):
    identity=record_id or uid();work=None
    if not state.get('testOnly'):
        from .storage_reservations import begin,abort
        project=store.get('foundation_project',state.get('projectId') or '')
        work=begin(store,project,parameter_count(config)*32+16*1024**2,purpose='checkpoint')
    try:
        return _save(store,model,config,tokenizer_record,state,optimizer,job=job,
            record_kind=record_kind,record_id=identity,storage_work=work)
    except BaseException:
        if work:
            staging=store.root/'staging'/('foundation-checkpoint-'+identity)
            published=store.root/'foundation'/('genesis' if record_kind=='foundation_genesis' else 'checkpoints')/identity
            abort(store,work,files_removed=not staging.exists() and not published.exists())
        raise


def _save(store, model, config, tokenizer_record, state, optimizer=None, *, job=None,
          record_kind="foundation_checkpoint", record_id=None,storage_work=None):
    import torch
    from safetensors.torch import save_file
    require(record_kind in ("foundation_genesis", "foundation_checkpoint"), "CHECKPOINT_SCHEMA_INVALID")
    if job is not None:store.assert_lease(job)
    identity = record_id or uid()
    staging = store.root / "staging" / ("foundation-checkpoint-" + identity)
    require(not staging.exists(), "CHECKPOINT_ALREADY_EXISTS", status=409)
    staging.mkdir(parents=True)
    saved, aliases, owners = {}, {}, {}
    for name, tensor in model.state_dict().items():
        require(bool(torch.isfinite(tensor).all()) if tensor.is_floating_point() else True,
                "CHECKPOINT_NONFINITE")
        key = (tensor.data_ptr(), tensor.storage_offset(), tuple(tensor.shape), tuple(tensor.stride()))
        if key in owners:
            aliases[name] = owners[key]
        else:
            owners[key] = name
            saved[name] = tensor.detach().cpu().contiguous().clone()
    save_file(saved, str(staging / "model.safetensors"))
    atomic_json(staging / "config.json", config)
    tensor_state = {}
    tree = _pack({"state": state, "optimizer": optimizer.state_dict() if optimizer is not None else None,
                  "torchRng": torch.get_rng_state(), "pythonRng": random.getstate()}, tensor_state)
    save_file(tensor_state, str(staging / "state.safetensors"))
    atomic_json(staging / "state.json", tree)
    import transformers
    manifest = {"schemaVersion": 1, "kind": record_kind, "id": identity, "config": config,
                "configHash": digest(config), "architectureHashV2": digest([config, transformers.__version__]),
                "tokenizerId": tokenizer_record["id"], "tokenizerArtifactId": tokenizer_record["id"],
                "tokenizerHash": tokenizer_record["tokenizerHash"], "parameterCount": parameter_count(config),
                "tensorAliases": aliases, "testOnly": state.get("testOnly") is True,
                "origin": state.get('origin','random_init'), "pretrainedWeightArtifactIds": state.get('pretrainedWeightArtifactIds',[]),
                "trustedGrowthReceipt":state.get('trustedGrowthReceipt'),
                "torchVersion": torch.__version__, "transformersVersion": transformers.__version__,
                "files": {p.name: file_hash(p) for p in staging.iterdir() if p.is_file()}}
    atomic_json(staging / "manifest.json", manifest)
    for path in staging.iterdir():
        _sync(path)
    actual_bytes=sum(path.stat().st_size for path in staging.iterdir() if path.is_file())
    if storage_work:
        from .storage_reservations import assert_work
        assert_work(store,storage_work)
        require(actual_bytes<=storage_work['estimate'],'STORAGE_RESERVATION_SIZE_MISMATCH')
    destination = store.root / "foundation" / ("genesis" if record_kind == "foundation_genesis" else "checkpoints") / identity
    destination.parent.mkdir(parents=True, exist_ok=True)
    require(not destination.exists(), "CHECKPOINT_ALREADY_EXISTS", status=409)
    staging.replace(destination)
    # Directory fsync is supported on Linux; Windows directory fsync is unavailable.
    if os.name != "nt":
        descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    record = {**state, "config": config, "configHash": manifest["configHash"],
              "architectureHashV2": manifest["architectureHashV2"],
              "tokenizerId": tokenizer_record["id"], "tokenizerArtifactId": tokenizer_record["id"],
              "tokenizerHash": tokenizer_record["tokenizerHash"], "parameterCount": manifest["parameterCount"],
              "storagePath": destination.relative_to(store.root).as_posix(),
              "manifestHash": file_hash(destination / "manifest.json"), "files": manifest["files"],
              "state": "ready", "servingEligible": False, "committedAt": now()}
    with store.tx():
        project=store.get('foundation_project',state.get('projectId') or '')
        from .origins import assert_project_available
        assert_project_available(store,project)
        require(store.get('foundation_tokenizer',tokenizer_record['id']).get('state') in ('ready','verified'),
                'POLICY_REVOKED',status=409)
        if job is not None:
            current=store.assert_lease(job)
            require(current.get('state') in ('running','pause_requested','cancelling'),'LEASE_EXPIRED',status=409)
        if storage_work:
            from .storage_reservations import assert_work,commit
            assert_work(store,storage_work)
        value = store.put(record_kind, record, identity)
        if record_kind == "foundation_checkpoint" and state.get("runId"):
            run = store.get("foundation_run", state["runId"], {})
            store.put("foundation_run", {"lastCheckpointId": identity, "checkpointId": identity,
                      "completedOptimizerSteps": state.get("completedOptimizerSteps", 0),
                      "committedTargetTokens": state.get("committedTargetTokens", 0),
                      "cursor": state.get("cursor", {}), "consumptionWatermark": state.get("consumptionWatermark", {})},
                      state["runId"])
            store.put("foundation_consumption", {"projectId": state.get("projectId"), "runId": state["runId"],
                      "tokenizerHash": tokenizer_record["tokenizerHash"],
                      "targetNamespace": state.get("targetNamespace", "causal_lm"),
                      "checkpointId": identity, "completedOptimizerSteps": state.get("completedOptimizerSteps", 0),
                      "committedTargetTokens": state.get("committedTargetTokens", 0),
                      "newUniqueTokens": state.get("newUniqueTokens", 0),
                      "replayTargetTokens": state.get("replayTargetTokens", 0),
                      "consumedContentHashes": state.get("consumedContentHashes", []),
                      "consumedTargetRanges": state.get("consumedTargetRanges", {}),
                      "cursor": state.get("cursor", {}), "watermark": state.get("consumptionWatermark", {})},
                      state["runId"])
        store.event("checkpoint.committed" if record_kind == "foundation_checkpoint" else "genesis.committed", identity)
        if storage_work:commit(store,storage_work,actual_bytes)
    return value


def verify(store, record):
    root = record_path(store, record)
    path = root / "manifest.json"
    require(path.is_file() and not path.is_symlink() and file_hash(path) == record.get("manifestHash"),
            "CHECKPOINT_HASH_MISMATCH")
    manifest = json.loads(path.read_text("utf-8"))
    require(manifest.get("schemaVersion") == 1 and manifest.get("id") == record["id"], "CHECKPOINT_SCHEMA_INVALID")
    require(digest(manifest["config"]) == manifest.get("configHash") == record.get("configHash"),
            "CHECKPOINT_CONFIG_MISMATCH")
    require(manifest.get("tokenizerHash") == record.get("tokenizerHash"), "TOKENIZER_HASH_MISMATCH")
    require(record.get("state") in ("ready", "verified", "completed"), "CHECKPOINT_NOT_READY", status=409)
    require(isinstance(manifest.get("files"), dict), "CHECKPOINT_SCHEMA_INVALID")
    for name, expected in manifest["files"].items():
        path = inside(root, name)
        require(path.is_file() and not path.is_symlink() and file_hash(path) == expected,
                "CHECKPOINT_HASH_MISMATCH", file=name)
    require(all(n in manifest["files"] for n in ("model.safetensors", "state.safetensors", "state.json", "config.json")),
            "CHECKPOINT_SCHEMA_INVALID")
    return root, manifest


def load_model_state(root, manifest, model):
    from safetensors.torch import load_file
    import torch
    weights = load_file(str(root / "model.safetensors"), device="cpu")
    require(all(bool(torch.isfinite(v).all()) for v in weights.values() if v.is_floating_point()), "CHECKPOINT_NONFINITE")
    for alias, owner in manifest.get("tensorAliases", {}).items():
        require(owner in weights and alias not in weights, "CHECKPOINT_ALIAS_INVALID")
        weights[alias] = weights[owner]
    require(set(weights) == set(model.state_dict()), "CHECKPOINT_TENSOR_SCHEMA_MISMATCH")
    model.load_state_dict(weights, strict=True)
    model.tie_weights()
    require(model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr(),
            "MODEL_TIED_WEIGHTS_INVALID")


def restore(store, record, model=None, optimizer=None, *, restore_rng=True):
    from safetensors.torch import load_file
    import torch
    root, manifest = verify(store, record)
    if model is not None:
        load_model_state(root, manifest, model)
    data = _unpack(json.loads((root / "state.json").read_text("utf-8")),
                   load_file(str(root / "state.safetensors"), device="cpu"))
    require(isinstance(data, dict) and isinstance(data.get("state"), dict), "CHECKPOINT_SCHEMA_INVALID")
    if optimizer is not None:
        import transformers
        require(manifest.get("torchVersion") == torch.__version__ and
                manifest.get("transformersVersion") == transformers.__version__,
                "CHECKPOINT_TOOLCHAIN_MISMATCH", status=409)
        require(data.get("optimizer") is not None, "CHECKPOINT_OPTIMIZER_MISSING")
        optimizer.load_state_dict(data["optimizer"])
    if restore_rng:
        torch.set_rng_state(data["torchRng"])
        random.setstate(data["pythonRng"])
    return data["state"]


save_checkpoint = save
load_checkpoint = restore
