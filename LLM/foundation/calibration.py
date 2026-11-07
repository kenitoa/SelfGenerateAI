"""Disposable, real CPU training measurements. Never publishes trained weights."""
from __future__ import annotations

import itertools
import math
import os
import platform
import statistics
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from LLM.core import require, digest, now, file_hash, atomic_json
from .schemas import project_architecture, integer, real
from . import architecture, checkpoint, packing, tokenizer as tokenizer_module
from .trainer import training_config, DEFAULTS


@contextmanager
def _checkpoint_directory(store,project):
    work=None;directory=None
    if not project.get('testOnly'):
        from .storage_reservations import begin
        work=begin(store,project,project['parameterCount']*32+16*1024**2,purpose='temporary-checkpoint')
    try:
        with tempfile.TemporaryDirectory(prefix='foundation-calibration-',dir=store.root/'staging') as temporary:
            directory=Path(temporary)
            if work:
                from .storage_reservations import assert_work
                assert_work(store,work)
            yield directory
            if work:
                from .storage_reservations import assert_work
                assert_work(store,work)
                require(sum(path.stat().st_size for path in directory.iterdir() if path.is_file())<=work['estimate'],
                        'STORAGE_RESERVATION_SIZE_MISMATCH')
    finally:
        if work:
            from .storage_reservations import abort
            abort(store,work,files_removed=directory is None or not directory.exists())


def run(store, project, corpus, tokenizer, parent, config, job=None):
    import psutil
    import torch
    import transformers
    from safetensors.torch import save_file

    cfg = training_config({k: v for k, v in (config or {}).items() if k in set(DEFAULTS)|{'cpuThreads'}},
                          project_architecture(project)["max_position_embeddings"])
    stage=(job or {}).get('payload',{}).get('stage',(config or {}).get('stage','pretrain'))
    require(stage in ('pretrain','cpt','sft'),'INVALID_TRAINING_OBJECTIVE')
    teacher_mix = packing.teacher_mix_descriptor(corpus)
    require(not teacher_mix or stage == 'sft','TEACHER_CORPUS_SFT_ONLY')
    measured_limit = integer((config or {}).get("measurementSteps", 20), "measurementSteps", 5, 20)
    warmup_limit = integer((config or {}).get("warmupSteps", 3), "warmupSteps", 3, 3)
    maximum_seconds = real((config or {}).get("maxSeconds", 300), "maxSeconds", .01, 300, positive=True)
    def preparing(phase,completed=0,warmup_steps=0):
        if job:
            store.progress(job,'preparing_calibration',completed,measured_limit,'optimizer_steps',
                preparingPhase=phase,warmupSteps=warmup_steps,measurementOnly=True)
    parent = store.get("foundation_checkpoint", parent, {}) or store.get("foundation_genesis", parent) if isinstance(parent, str) else parent
    tokenizer = store.get("foundation_tokenizer", tokenizer) if isinstance(tokenizer, str) else tokenizer
    require(corpus.get("state") in ("ready", "verified") and parent.get("state") in ("ready", "verified", "completed"),
            "POLICY_REVOKED", status=409)
    preparing('loading_tokenizer')
    tok = tokenizer_module.load(store, tokenizer)
    require(parent.get("tokenizerHash") == tok.tokenizer_hash, "TOKENIZER_HASH_MISMATCH")
    replay_ranges = {}
    if stage == 'cpt':
        preparing('loading_replay_ledger')
        from .trainer import _mark_interval
        for receipt in store.list('foundation_consumption'):
            if (receipt.get('projectId') == project['id'] and receipt.get('tokenizerHash') == tok.tokenizer_hash
                    and receipt.get('targetNamespace','causal_lm') == 'causal_lm'):
                for content_hash,ranges in receipt.get('consumedTargetRanges',{}).items():
                    for begin,end in ranges:_mark_interval(replay_ranges.setdefault(content_hash,[]),begin,end)
    require(project.get('executionScope','nas') == 'pc_validation' or cfg.get('cpuThreads',1) == 1,
        'NAS_CPU_THREADS_FIXED')
    torch.set_num_threads(cfg.get('cpuThreads',1))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    preparing('loading_model')
    model = architecture.load_model(store, parent)
    model.train()
    model.config.use_cache = False
    initial_digest = architecture.model_digest(model)
    parameters = list(model.parameters())
    embedding = model.get_input_embeddings().weight
    preparing('creating_optimizer')
    optimizer = torch.optim.AdamW([
        {"params": [p for p in parameters if p is not embedding and p.ndim >= 2], "weight_decay": cfg["weightDecay"]},
        {"params": [p for p in parameters if p is embedding or p.ndim < 2], "weight_decay": 0}],
        lr=cfg["learningRate"], betas=(.9, .95), eps=1e-8, foreach=False, fused=False)
    process = psutil.Process()
    resource_cap = project.get("policy", {}).get("workerMemoryBytes", 6 * 1024**3)
    integer(resource_cap, "workerMemoryBytes", 1, 2**63-1)
    environment = {"cpu": platform.processor(), "machine": platform.machine(),
        "os": platform.system(), "logicalCpuCount": psutil.cpu_count(),
        "torchVersion": torch.__version__, "transformersVersion": transformers.__version__,
        "intraOpThreads": torch.get_num_threads(), "interOpThreads": torch.get_num_interop_threads(),
        "memoryBudgetBytes": resource_cap,
        "workerScope": "pc-validation" if project.get('executionScope')=='pc_validation' else "nas-local",
        "executionScope":project.get('executionScope','nas')}
    from .resources import machine_fingerprint
    environment['machineFingerprint']=machine_fingerprint()
    config_hash = digest(cfg)
    recipe_hash = digest([parent.get("architectureHashV2", parent["configHash"]), tok.tokenizer_hash,
                          cfg, environment,teacher_mix])
    peak_rss = process.memory_info().rss
    stop = threading.Event()
    background_errors = []
    def monitor():
        nonlocal peak_rss
        last_heartbeat = time.monotonic()
        while not stop.wait(.02):
            peak_rss = max(peak_rss, process.memory_info().rss)
            if job and time.monotonic()-last_heartbeat >= 10:
                try:
                    store.heartbeat(job)
                    last_heartbeat = time.monotonic()
                except BaseException as exc:
                    background_errors.append(exc)
                    return
    thread = threading.Thread(target=monitor, name="foundation-profile-monitor", daemon=True)
    thread.start()
    measured = []
    measured_targets = 0
    warmups = 0
    epoch = 0
    def make_source(epoch):
        return (iter(packing.teacher_groups(store,corpus,tok,cfg,epoch=epoch,purpose='calibration')) if teacher_mix else
            iter(packing.iter_blocks(store,corpus,tok,cfg['sequenceLength'],stage=stage,split='train',
                epoch=epoch,seed=cfg['seed'],shuffle=cfg['shuffle'],purpose='calibration',replay_ranges=replay_ranges)))
    source = make_source(0)
    def get_blocks():
        if teacher_mix:
            envelope = next(source,None)
            return envelope['blocks'] if envelope else []
        return list(itertools.islice(source,cfg['microBatchSize']*cfg['gradientAccumulation']))
    started = time.monotonic()
    reason_codes = []
    checkpoint_seconds = None
    try:
        while len(measured) < measured_limit:
            if time.monotonic()-started >= maximum_seconds:
                reason_codes.append("PROFILE_TIME_LIMIT")
                break
            require(not background_errors, "LEASE_EXPIRED", status=409)
            if job:
                current = store.assert_lease(job)
                require(current["state"] not in ("cancelling", "cancelled", "pause_requested"), "JOB_CANCELLED", status=409)
            from .corpus import resolve
            require(resolve(store,corpus['id'])["state"] in ("ready", "verified"),
                    "POLICY_REVOKED", status=409)
            step_start = time.monotonic()
            preparing('sampling_batches',len(measured),warmups)
            blocks = get_blocks()
            if not blocks:
                # Calibration may recycle a train-only stream; it is NOT production training exposure.
                epoch += 1
                source = make_source(epoch)
                blocks = get_blocks()
            blocks = [block for block in blocks if block["targetCount"]]
            require(blocks, "NO_TRAINABLE_EXAMPLES")
            targets = sum(block["targetCount"] for block in blocks)
            preparing('forward_backward',len(measured),warmups)
            optimizer.zero_grad(set_to_none=True)
            for offset in range(0, len(blocks), cfg["microBatchSize"]):
                batch = blocks[offset:offset+cfg["microBatchSize"]]
                count = sum(block["targetCount"] for block in batch)
                output = model(**packing.tensor_batch(torch, batch, tok.pad_token_id))
                require(bool(torch.isfinite(output.loss)), "NONFINITE_TRAINING")
                (output.loss * count / targets).backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, cfg["gradientClipping"])
            require(bool(torch.isfinite(norm)), "NONFINITE_TRAINING")
            optimizer.step()
            require(all(bool(torch.isfinite(p).all()) for p in parameters), "NONFINITE_TRAINING")
            elapsed = time.monotonic()-step_start
            peak_rss = max(peak_rss, process.memory_info().rss)
            require(peak_rss <= resource_cap, "RESOURCE_PRESSURE", status=503)
            if warmups < warmup_limit:
                warmups += 1
            else:
                measured.append(elapsed)
                measured_targets += targets
            if job:
                store.progress(job, "calibrating", len(measured), measured_limit, "optimizer_steps",
                               warmupSteps=warmups, measurementOnly=True)
        # Measure a complete pickle-free model+optimizer checkpoint without publishing it.
        preparing('measuring_checkpoint',len(measured),warmups)
        with _checkpoint_directory(store,project) as temporary:
            directory = Path(temporary)
            tick = time.monotonic()
            tensors = {}
            tree = checkpoint._pack({"optimizer": optimizer.state_dict(), "torchRng": torch.get_rng_state()}, tensors)
            save_file(tensors, str(directory / "state.safetensors"))
            weights, owners = {}, set()
            for key, value in model.state_dict().items():
                marker = (value.data_ptr(), value.storage_offset(), tuple(value.shape))
                if marker not in owners:
                    owners.add(marker)
                    weights[key] = value.detach().cpu().contiguous().clone()
            save_file(weights, str(directory / "model.safetensors"))
            atomic_json(directory / "state.json", tree)
            for path in directory.iterdir():
                with path.open("r+b") as handle:
                    handle.flush(); os.fsync(handle.fileno())
            checkpoint_seconds = time.monotonic()-tick
        preparing('verifying_parent',len(measured),warmups)
        parent_model = architecture.load_model(store, parent)
        require(architecture.model_digest(parent_model) == initial_digest, "PROFILE_PARENT_MUTATED")
        require(store.get("foundation_tokenizer", tokenizer["id"])["state"] in ("ready", "verified")
                and store.get("foundation_corpus", corpus["id"])["state"] in ("ready", "verified"),
                "POLICY_REVOKED", status=409)
        peak_rss = max(peak_rss, process.memory_info().rss)
        passed = warmups == warmup_limit and len(measured) >= 5 and peak_rss <= resource_cap
        if not passed:
            reason_codes.append("PROFILE_INSUFFICIENT_STEPS")
        sorted_steps = sorted(measured)
        n = sum(p.numel() for p in parameters)
        value = {"projectId": project["id"], "parentCheckpointId": parent["id"], "corpusId": corpus["id"],
                 'executionScope':project.get('executionScope','nas'),
                 "backendReceipt":project.get('backendReceipt'),
                 "tokenizerId": tokenizer["id"], "tokenizerHash": tok.tokenizer_hash,"stage":stage,
                 "architectureHashV2": parent.get("architectureHashV2"), "config": cfg, "configHash": config_hash,
                 'teacherMixHash':digest(teacher_mix) if teacher_mix else None,'teacherMixDescriptor':teacher_mix,
                 'cptReplayRangesHash':digest(replay_ranges) if stage=='cpt' else None,
                 "recipeHash": recipe_hash, "environment": environment, "state": "passed" if passed else "insufficient",
                 "passed": passed, "measurementOnly": True, "productionTrainedTokens": 0,
                 "testOnly": project.get("testOnly") is True, "servingEligible": False,
                 "warmupSteps": warmups, "measuredSteps": len(measured), "measuredTargetTokens": measured_targets,
                 "effectiveTokensPerSecond": measured_targets / sum(measured) if measured else None,
                 "meanStepSeconds": statistics.mean(measured) if measured else None,
                 "p95StepSeconds": sorted_steps[max(0, math.ceil(.95*len(sorted_steps))-1)] if measured else None,
                 "peakRssBytes": max(peak_rss, process.memory_info().rss), "checkpointSeconds": checkpoint_seconds,
                 "totalSeconds": time.monotonic()-started, "reasonCodes": sorted(set(reason_codes)),
                 "fp32StateEstimate": {"parameterCount": n, "weightsBytes": n*4, "gradientsBytes": n*4,
                                       "adamMomentBytes": n*8, "persistentStateBytes": n*16}, "measuredAt": now()}
        if job:
            store.assert_lease(job)
        record = store.put("foundation_profile", value)
        store.event("foundation.profile.completed", record["id"], {"passed": passed})
        return record
    finally:
        stop.set()
        thread.join(timeout=2)
