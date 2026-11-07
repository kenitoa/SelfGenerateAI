"""Real CPU FP32 full-parameter causal training with bounded streaming and resume."""
from __future__ import annotations

import itertools
import math
import random
import copy
import threading
import time

from LLM.core import require, digest, now, Fault
from .schemas import integer, real, project_architecture, corpus_manifest
from . import architecture, checkpoint, packing, tokenizer as tokenizer_module

DEFAULTS = {"sequenceLength": 128, "microBatchSize": 1, "gradientAccumulation": 8,
            "learningRate": 3e-4, "maxTokenExposures": 1_000_000, "maxEpochs": 3,
            "sliceSeconds": 900, "maxActiveSeconds": 14400, "checkpointEvery": 100,
            "checkpointSeconds": 600, "seed": 42, "gradientClipping": 1.0,
            "weightDecay": 0.1, "shuffle": True, "maxSteps": 0, "maxDevTokens": 100_000}


def _mark_interval(ranges, start, end):
    """Union one scored target range and return the number not previously learned."""
    overlap = sum(max(0, min(end, b) - max(start, a)) for a, b in ranges)
    remaining = []
    merged_start, merged_end = start, end
    inserted = False
    for a, b in ranges:
        if b < merged_start:
            remaining.append([a, b])
        elif merged_end < a:
            if not inserted:
                remaining.append([merged_start, merged_end]); inserted = True
            remaining.append([a, b])
        else:
            merged_start, merged_end = min(merged_start, a), max(merged_end, b)
    if not inserted:
        remaining.append([merged_start, merged_end])
    ranges[:] = remaining
    return end - start - overlap


def _scored_ranges(block):
    ranges = []
    for index, origin in enumerate(block.get("targetOrigins", [])):
        if index == 0 or block["labels"][index] == -100 or origin is None:
            continue
        identity, position = origin
        if ranges and ranges[-1][0] == identity and ranges[-1][2] == position:
            ranges[-1][2] += 1
        else:
            ranges.append([identity, position, position+1])
    return ranges


def training_config(value=None, maximum_context=512):
    value = dict(value or {})
    aliases = {"epochs": "maxEpochs", "batchSize": "microBatchSize", "maxSequence": "sequenceLength"}
    for old, new in aliases.items():
        if old in value:
            require(new not in value or value[new] == value[old], "CONFIG_INVALID", field=old)
            value[new] = value.pop(old)
    require(not set(value) - (set(DEFAULTS)|{'cpuThreads'}), "CONFIG_INVALID", fields=sorted(set(value)-set(DEFAULTS)-{'cpuThreads'}))
    cfg = {**DEFAULTS, **value}
    if 'cpuThreads' in cfg:integer(cfg['cpuThreads'],'cpuThreads',1,4)
    for key, minimum, maximum in (("sequenceLength", 8, maximum_context), ("microBatchSize", 1, 8),
            ("gradientAccumulation", 1, 128), ("maxTokenExposures", 1, 10**12), ("maxEpochs", 1, 3),
            ("sliceSeconds", 1, 14400), ("maxActiveSeconds", 1, 86400), ("checkpointEvery", 1, 10000),
            ("checkpointSeconds", 1, 3600), ("seed", 0, 2**32-1), ("maxSteps", 0, 10**8),
            ("maxDevTokens", 1, 10**7)):
        integer(cfg[key], key, minimum, maximum)
    real(cfg["learningRate"], "learningRate", 0, .01, positive=True)
    real(cfg["gradientClipping"], "gradientClipping", .001, 100)
    real(cfg["weightDecay"], "weightDecay", 0, 1)
    require(isinstance(cfg["shuffle"], bool), "CONFIG_INVALID", field="shuffle")
    return cfg


class FoundationTrainer:
    def __init__(self, store, init_fn=None):
        self.store, self.init_fn = store, init_fn
        self.should_yield = lambda: False

    def config(self, body, maximum_context=512):
        return training_config(body, maximum_context)

    def _validate_policy(self, corpus_id, tokenizer_id, parent_id, *, metadata_only=False):
        from .corpus import resolve
        corpus = ((self.store.get('foundation_corpus',corpus_id,{}) or self.store.get('foundation_teacher_corpus',corpus_id))
                  if metadata_only else resolve(self.store,corpus_id))
        if metadata_only and corpus.get('kind')=='foundation_teacher_corpus':
            from .response_distillation import validate_corpus
            validate_corpus(self.store,corpus)  # Current policy/approval/artifact metadata only.
        from .origins import assert_project_available
        assert_project_available(self.store,self.store.get('foundation_project',corpus['projectId']))
        require(corpus.get("state") in ("ready", "verified"), "POLICY_REVOKED", status=409)
        require(self.store.get("foundation_tokenizer", tokenizer_id).get("state") in ("ready", "verified"),
                "POLICY_REVOKED", status=409)
        parent = self.store.get("foundation_checkpoint", parent_id, {}) or self.store.get("foundation_genesis", parent_id)
        require(parent.get("state") in ("ready", "verified", "completed"), "POLICY_REVOKED", status=409)
        teacher_refs=dict(parent.get('teacherManifestHashes') or {})
        if parent.get('teacherArtifactId'):teacher_refs[parent['teacherArtifactId']]=parent.get('teacherManifestHash')
        for identity,checksum in teacher_refs.items():
            teacher=self.store.get('artifact',identity,{}) or self.store.get('foundation_checkpoint',identity,{})
            require(teacher.get('state') in ('ready','verified','completed') and teacher.get('manifestHash')==checksum,
                'POLICY_REVOKED',status=409)
        hook = getattr(self, "policy_validator", None)
        if callable(hook) and not metadata_only:
            hook(corpus)

    def _validation(self, model, corpus, tok, cfg, stage):
        import torch
        model.eval()
        total_loss, total_targets = 0.0, 0
        try:
            with torch.inference_mode():
                for block in packing.iter_blocks(self.store, corpus, tok, cfg["sequenceLength"], stage=stage,
                                                  split="dev", shuffle=False):
                    count = block["targetCount"]
                    if not count:
                        continue
                    result = model(**packing.tensor_batch(torch, [block], tok.pad_token_id))
                    require(bool(torch.isfinite(result.loss)), "NONFINITE_VALIDATION")
                    total_loss += float(result.loss) * count
                    total_targets += count
                    if total_targets >= cfg["maxDevTokens"]:
                        break
        finally:
            model.train()
        require(total_targets > 0, "VALIDATION_DATA_REQUIRED")
        return total_loss / total_targets, total_targets

    def run(self, job):
        import torch
        body = job["payload"]
        project = self.store.get("foundation_project", body["projectId"])
        from .corpus import resolve
        corpus = resolve(self.store,body['corpusId'])
        token_id = body.get("tokenizerId") or project.get("tokenizerId")
        token_record = self.store.get("foundation_tokenizer", token_id)
        tok = tokenizer_module.load(self.store, token_record)
        model_cfg = project_architecture(project)
        stage = body.get("stage", "pretrain")
        require(stage in ("pretrain", "cpt", "sft"), "INVALID_TRAINING_OBJECTIVE")
        teacher_mix = packing.teacher_mix_descriptor(corpus)
        require(not teacher_mix or stage == 'sft','TEACHER_CORPUS_SFT_ONLY')
        supplied_config = body.get("config", {})
        if stage == "sft":
            supplied_config = {"maxEpochs": 2, "learningRate": 5e-5, **supplied_config}
        cfg = self.config(supplied_config, model_cfg["max_position_embeddings"])
        require(project.get('executionScope','nas') == 'pc_validation' or cfg.get('cpuThreads',1) == 1,
            'NAS_CPU_THREADS_FIXED')
        require(stage != "sft" or cfg["maxEpochs"] <= 2, "CONFIG_INVALID", field="maxEpochs")
        require(tok.vocab_size == model_cfg["vocab_size"], "TOKENIZER_HASH_MISMATCH")
        corpus_root, manifest = corpus_manifest(self.store, corpus)
        corpus_hash = corpus.get("manifestHash")
        from LLM.core import file_hash
        corpus_hash = corpus_hash or file_hash(corpus_root / "manifest.json")
        prior = self.store.get("foundation_run", job["id"], {})
        if prior.get("state") == "completed" and prior.get("lastCheckpointId"):
            checkpoint.verify(self.store, self.store.get("foundation_checkpoint", prior["lastCheckpointId"]))
            return {"state": "completed", "runId": job["id"], "checkpointId": prior["lastCheckpointId"]}
        parent_id = body.get("parentCheckpointId") or project.get("latestCheckpointId") or project.get("genesisId")
        require(bool(parent_id), "FOUNDATION_PARENT_REQUIRED")
        parent = self.store.get("foundation_checkpoint", parent_id, {}) or self.store.get("foundation_genesis", parent_id)
        require(parent["configHash"] == digest(model_cfg) and parent["tokenizerHash"] == token_record["tokenizerHash"],
                "CHECKPOINT_INCOMPATIBLE", status=409)
        self._validate_policy(corpus["id"], token_id, parent_id)
        torch.set_num_threads(cfg.get('cpuThreads',1))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass  # A process can configure inter-op threads only before its first parallel call.
        random.seed(cfg["seed"])
        torch.manual_seed(cfg["seed"])
        resume_id = prior.get("lastCheckpointId") or prior.get("checkpointId")
        resumed = self.store.get("foundation_checkpoint", resume_id) if resume_id else None
        selected = resumed or parent
        model = architecture.load_model(self.store, selected, init_fn=self.init_fn)
        model.train()
        model.config.use_cache = False
        parameters = list(model.parameters())
        require(parameters and all(p.requires_grad for p in parameters), "FULL_PARAMETER_TRAINING_REQUIRED")
        # Embedding is tied to LM head: deduplicate parameters before grouping.
        embedding = model.get_input_embeddings().weight
        decay, no_decay = [], []
        for parameter in parameters:
            (no_decay if parameter is embedding or parameter.ndim < 2 else decay).append(parameter)
        optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weightDecay"]},
                                       {"params": no_decay, "weight_decay": 0.0}],
                                      lr=cfg["learningRate"], betas=(.9, .95), eps=1e-8,
                                      amsgrad=False, foreach=False, fused=False)
        input_hash = digest([project["id"], parent_id, corpus_hash, token_record["tokenizerHash"], stage, cfg]
            + ([teacher_mix] if teacher_mix else []))
        state = {"runId": job["id"], "projectId": project["id"], "corpusId": corpus["id"],
                 "origin":project.get('origin','random_init'),'trustedGrowthReceipt':project.get('trustedGrowthReceipt'),
                 'executionScope':project.get('executionScope','nas'),
                 'pretrainedWeightArtifactIds':project.get('pretrainedWeightArtifactIds',[]),
                 "backendReceipt":project.get('backendReceipt'),
                 "corpusHash": corpus_hash, "parentCheckpointId": parent_id, "runConfig": cfg,
                 "runConfigHash": digest(cfg), "inputHash": input_hash, "stage": stage,
                 "completedOptimizerSteps": 0, "committedTargetTokens": 0, "attemptedTargetTokens": 0,
                 "newUniqueTokens": 0, "replayTargetTokens": 0,
                 "targetNamespace": "instruction_output" if stage == "sft" else "causal_lm",
                 "consumedContentHashes": [], "consumedTargetRanges": {},
                 "cursor": {"epoch": 0, "block": 0}, "activeSeconds": 0.0,
                 "trainedContextTokens": parent.get("trainedContextTokens", 0),
                 "initialTensorDigest": architecture.model_digest(model),
                 "consumptionWatermark": {"epoch": 0, "block": 0, "targets": 0},
                 "policyVersion": corpus.get("policyVersion"), "testOnly": project.get("testOnly") is True,
                 "servingEligible": False}
        if not resumed and body.get("continueCorpus") is True:
            parent_state = checkpoint.restore(self.store, parent, restore_rng=False)
            previous_cfg = parent_state.get("runConfig", {})
            require(parent_state.get("corpusHash") == corpus_hash and parent_state.get("stage") == stage
                    and all(previous_cfg.get(k) == cfg[k] for k in ("sequenceLength", "seed", "shuffle")),
                    "CHECKPOINT_CONTINUATION_INCOMPATIBLE", status=409)
            state["cursor"] = dict(parent_state["cursor"])
            state["optimizerReset"] = True
            state["continuedFromCheckpointId"] = parent["id"]
            if "cptReplayRanges" in parent_state:
                state["cptReplayRanges"] = parent_state["cptReplayRanges"]
            require(state["cursor"]["epoch"] < cfg["maxEpochs"], "NO_REMAINING_CORPUS", status=409)
        if resumed:
            saved = checkpoint.restore(self.store, resumed, model, optimizer, restore_rng=True)
            require(saved.get("runId") == job["id"] and saved.get("inputHash") == input_hash,
                    "CHECKPOINT_INCOMPATIBLE", status=409)
            state = saved
        known_ranges = {}
        for receipt in self.store.list("foundation_consumption"):
            if receipt.get("projectId") == project["id"] and receipt.get("tokenizerHash") == token_record["tokenizerHash"] \
                    and receipt.get("targetNamespace", "causal_lm") == state["targetNamespace"]:
                for content_hash, ranges in receipt.get("consumedTargetRanges", {}).items():
                    for begin, end in ranges:
                        _mark_interval(known_ranges.setdefault(content_hash, []), begin, end)
        for content_hash, ranges in state.get("consumedTargetRanges", {}).items():
            for begin, end in ranges:
                _mark_interval(known_ranges.setdefault(content_hash, []), begin, end)
        if stage == "cpt" and "cptReplayRanges" not in state:
            state["cptReplayRanges"] = copy.deepcopy(known_ranges)
        state.setdefault("samplingNewTargetTokens", 0)
        state.setdefault("samplingReplayTargetTokens", 0)
        state.setdefault("replayMix", {})
        state.setdefault('committedTeacherTargetTokens',0)
        state.setdefault('committedHumanTargetTokens',state.get('committedTargetTokens',0)-state['committedTeacherTargetTokens'])
        state['teacherMixDescriptor'] = teacher_mix
        inherited_teachers=dict(parent.get('teacherManifestHashes') or {})
        if parent.get('teacherArtifactId'):
            inherited_teachers[parent['teacherArtifactId']]=parent.get('teacherManifestHash')
        inherited_teachers.update(state.get('teacherManifestHashes') or {})
        state['teacherManifestHashes']=inherited_teachers
        state['teacherArtifactIds']=sorted(inherited_teachers)
        state['teacherApprovalIds']=sorted(set(parent.get('teacherApprovalIds') or [])|set(state.get('teacherApprovalIds') or []))
        if parent.get('distillationMode'):state.setdefault('distillationMode',parent['distillationMode'])
        if teacher_mix:
            from .corpus import iter_members
            teacher_refs=dict(inherited_teachers);approval_refs=set(state['teacherApprovalIds'])
            for member in iter_members(self.store,corpus):
                if member.get('targetOrigin')=='teacher':
                    teacher_refs[member['teacherArtifactId']]=member['teacherManifestHash']
                    approval_refs.add(member['teacherApprovalId'])
            state['teacherArtifactIds']=sorted(teacher_refs)
            state['teacherManifestHashes']=teacher_refs
            state['teacherApprovalIds']=sorted(approval_refs)
            state['distillationMode']='verified_response_sft'
            state['teacherPolicyHash']=corpus['teacherPolicyHash']
            if len(teacher_refs)==1:
                state['teacherArtifactId']=next(iter(teacher_refs))
                state['teacherManifestHash']=teacher_refs[state['teacherArtifactId']]
        with self.store.tx():
            self.store.assert_lease(job)
            self.store.put("foundation_run", {"projectId": project["id"], "corpusId": corpus["id"],
                "parentCheckpointId": parent_id, "tokenizerId": token_id, "stage": stage, "config": cfg,
                "inputHash": input_hash, "state": "running", "startedAt": prior.get("startedAt", now()),
                "trainingMode": "full_parameters", "servingEligible": False}, job["id"])
        stop_heartbeat, heartbeat_errors = threading.Event(), []
        def renew():
            while not stop_heartbeat.wait(10):
                try:
                    self.store.heartbeat(job)
                except BaseException as exc:
                    heartbeat_errors.append(exc)
                    return
        thread = threading.Thread(target=renew, daemon=True, name="foundation-heartbeat")
        thread.start()
        start = time.monotonic()
        start_active = state["activeSeconds"]
        last_checkpoint = start
        last_checkpoint_id = resume_id
        loss_value = None
        ending = "completed"
        termination_reason = "corpus_exhausted"
        initial_targets = state["committedTargetTokens"]
        # A group count estimate is used for LR scheduling only, never fake progress.
        nominal_targets = max(1, (cfg["sequenceLength"]-1)*cfg["microBatchSize"]*cfg["gradientAccumulation"])
        schedule_steps = max(1, math.ceil(cfg["maxTokenExposures"] / nominal_targets))
        warmup = min(200, max(1, math.ceil(schedule_steps*.05)))

        def save_boundary():
            nonlocal last_checkpoint, last_checkpoint_id
            self._validate_policy(corpus["id"], token_id, parent_id)
            state["activeSeconds"] = start_active + time.monotonic() - start
            state["trainedTensorDigest"] = architecture.model_digest(model)
            sampled = state["samplingNewTargetTokens"] + state["samplingReplayTargetTokens"]
            if stage == "cpt":
                state["replayMix"]["selectedStreamNewFraction"] = state["replayMix"].get("actualNewFraction")
                state["replayMix"]["committedNewTargetTokens"] = state["samplingNewTargetTokens"]
                state["replayMix"]["committedReplayTargetTokens"] = state["samplingReplayTargetTokens"]
                state["replayMix"]["actualNewFraction"] = state["samplingNewTargetTokens"] / sampled if sampled else None
            record = checkpoint.save(self.store, model, model_cfg, token_record, state, optimizer, job=job)
            last_checkpoint_id = record["id"]
            last_checkpoint = time.monotonic()
            return record

        def control():
            if heartbeat_errors:
                raise heartbeat_errors[0]
            current = self.store.assert_lease(job)
            if current["state"] in ("cancelling", "cancelled", "cancel_requested"):
                return "cancelled"
            if current["state"] in ("pause_requested", "pausing"):
                return "paused"
            if self.should_yield() or time.monotonic() - start >= cfg["sliceSeconds"]:
                return "yielded"
            return None

        try:
            optimizer.zero_grad(set_to_none=True)
            done = False
            for epoch in range(state["cursor"]["epoch"], cfg["maxEpochs"]):
                first_block = state["cursor"]["block"] if epoch == state["cursor"]["epoch"] else 0
                source = (iter(packing.teacher_groups(self.store,corpus,tok,cfg,epoch=epoch,
                    skip_groups=first_block,report=state['replayMix'])) if teacher_mix else
                    iter(packing.iter_blocks(self.store, corpus, tok, cfg["sequenceLength"], stage=stage,
                    epoch=epoch, seed=cfg["seed"], shuffle=cfg["shuffle"], skip_blocks=first_block,
                    replay_ranges=state.get("cptReplayRanges"), mix_report=state["replayMix"])))
                block_cursor = first_block
                while True:
                    decision = control()
                    if decision:
                        ending, termination_reason = decision, decision
                        done = True
                        break
                    if state["committedTargetTokens"] >= cfg["maxTokenExposures"]:
                        termination_reason, done = "token_budget", True
                        break
                    if cfg["maxSteps"] and state["completedOptimizerSteps"] >= cfg["maxSteps"]:
                        termination_reason, done = "step_budget", True
                        break
                    if start_active + time.monotonic() - start >= cfg["maxActiveSeconds"]:
                        termination_reason, done = "active_time_budget", True
                        break
                    self._validate_policy(corpus["id"], token_id, parent_id)
                    group = []
                    remaining = cfg["maxTokenExposures"] - state["committedTargetTokens"]
                    if teacher_mix:
                        envelope = next(source,None)
                        if envelope is not None:
                            block_cursor += 1
                            require(envelope.get('mixGroup') is True,'TEACHER_ATOMIC_GROUP_REQUIRED')
                            if envelope['targetCount'] > remaining: continue
                            group = envelope['blocks']
                            require(envelope['targetCount'] == sum(b['targetCount'] for b in group), 'TEACHER_GROUP_TARGET_MISMATCH')
                    for _ in range(0 if teacher_mix else cfg["gradientAccumulation"]):
                        blocks = list(itertools.islice(source, cfg["microBatchSize"]))
                        if not blocks:
                            break
                        block_cursor += len(blocks)
                        for block in blocks:
                            if block["targetCount"] > remaining:
                                if stage == "sft":
                                    continue  # Never truncate an instruction's target answer.
                                labels = list(block["labels"])
                                kept = 0
                                for index in range(1, len(labels)):
                                    if labels[index] != -100:
                                        if kept >= remaining:
                                            labels[index] = -100
                                        else:
                                            kept += 1
                                block = {**block, "labels": labels, "targetCount": kept}
                            if block["targetCount"]:
                                group.append(block)
                                remaining -= block["targetCount"]
                        if remaining <= 0:
                            break
                    if not group:
                        state["cursor"] = {"epoch": epoch+1, "block": 0}
                        break
                    count = sum(block["targetCount"] for block in group)
                    teacher_targets = sum(b['targetCount'] for b in group if b.get('targetOrigin') == 'teacher')
                    if teacher_mix:
                        from .response_distillation import validate_corpus
                        current_policy = validate_corpus(self.store,corpus)
                        require(teacher_targets/count <= current_policy['maxTeacherFraction']+1e-12,
                            'TEACHER_ACTUAL_BATCH_FRACTION_EXCEEDED')
                    group_loss = 0.0
                    for offset in range(0, len(group), cfg["microBatchSize"]):
                        if teacher_mix:
                            decision = control()
                            if decision:
                                optimizer.zero_grad(set_to_none=True)
                                ending, termination_reason, done = decision, decision, True
                                break  # No partial envelope reaches optimizer.step.
                        batch = group[offset:offset+cfg["microBatchSize"]]
                        batch_targets = sum(block["targetCount"] for block in batch)
                        state["attemptedTargetTokens"] += batch_targets
                        result = model(**packing.tensor_batch(torch, batch, tok.pad_token_id))
                        require(bool(torch.isfinite(result.loss)), "NONFINITE_TRAINING")
                        weighted = result.loss * (batch_targets / count)
                        weighted.backward()
                        group_loss += float(result.loss.detach()) * batch_targets
                    if done: break
                    if teacher_mix: self._validate_policy(corpus['id'],token_id,parent_id)
                    norm = torch.nn.utils.clip_grad_norm_(parameters, cfg["gradientClipping"])
                    require(bool(torch.isfinite(norm)), "NONFINITE_TRAINING")
                    step = state["completedOptimizerSteps"] + 1
                    progress = min(1., max(0., (step-warmup)/max(1, schedule_steps-warmup)))
                    factor = step / warmup if step <= warmup else .1 + .9*.5*(1+math.cos(math.pi*progress))
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = cfg["learningRate"] * factor
                    optimizer.step()
                    require(all(bool(torch.isfinite(p).all()) for p in parameters), "NONFINITE_TRAINING")
                    optimizer.zero_grad(set_to_none=True)
                    state["completedOptimizerSteps"] += 1
                    state["committedTargetTokens"] += count
                    state['committedTeacherTargetTokens'] += teacher_targets
                    state['committedHumanTargetTokens'] += count-teacher_targets
                    state['actualTeacherFraction'] = state['committedTeacherTargetTokens']/state['committedTargetTokens']
                    state["trainedContextTokens"] = max(state.get("trainedContextTokens", 0),
                                                         max(len(block["inputIds"]) for block in group))
                    learned_new = 0
                    for block in group:
                        pool = block.get("samplingPool", "new")
                        key = "samplingReplayTargetTokens" if pool == "replay" else "samplingNewTargetTokens"
                        state[key] += block["targetCount"]
                        for content_hash, begin, end in _scored_ranges(block):
                            learned_new += _mark_interval(known_ranges.setdefault(content_hash, []), begin, end)
                            _mark_interval(state["consumedTargetRanges"].setdefault(content_hash, []), begin, end)
                    state["newUniqueTokens"] += learned_new
                    state["replayTargetTokens"] += count - learned_new
                    state["consumedContentHashes"] = sorted(state["consumedTargetRanges"])
                    state["cursor"] = {"epoch": epoch, "block": block_cursor}
                    state["consumptionWatermark"] = {"epoch": epoch, "block": block_cursor,
                                                       "targets": state["committedTargetTokens"]}
                    loss_value = group_loss / count
                    elapsed = time.monotonic()-start
                    self.store.progress(job, "training", state["committedTargetTokens"], cfg["maxTokenExposures"],
                        "target_tokens", completedOptimizerSteps=state["completedOptimizerSteps"],
                        observedLoss=loss_value, attemptedTargetTokens=state["attemptedTargetTokens"],
                        effectiveTokensPerSecond=(state["committedTargetTokens"]-initial_targets)/elapsed if elapsed else None)
                    if state["completedOptimizerSteps"] % cfg["checkpointEvery"] == 0 or time.monotonic()-last_checkpoint >= cfg["checkpointSeconds"]:
                        save_boundary()
                if done:
                    break
            if ending == "completed":
                require(state["completedOptimizerSteps"] > 0, "NO_EFFECTIVE_TRAINING")
                require(architecture.model_digest(model) != state["initialTensorDigest"], "NO_EFFECTIVE_TRAINING")
            saved = save_boundary()
            validation = None
            if ending == "completed":
                self.store.progress(job, "validating", 0, None, "target_tokens")
                validation, dev_targets = self._validation(model, corpus, tok, cfg, stage)
            state["activeSeconds"] = start_active + time.monotonic()-start
            trained_digest=architecture.model_digest(model)
            self._validate_policy(corpus['id'],token_id,parent_id)
            with self.store.tx():
                current=self.store.assert_lease(job)
                self._validate_policy(corpus['id'],token_id,parent_id,metadata_only=True)
                if current['state'] in ('cancelling','cancelled','cancel_requested'):
                    ending,termination_reason,validation='cancelled','cancelled',None
                elif ending=='completed' and current['state'] in ('pause_requested','pausing'):
                    ending,termination_reason,validation='paused','paused',None
                self.store.put("foundation_run", {"state": "paused" if ending in ("paused", "yielded") else ending,
                    "lastCheckpointId": saved["id"], "checkpointId": saved["id"],
                    "completedOptimizerSteps": state["completedOptimizerSteps"],
                    "committedTargetTokens": state["committedTargetTokens"],
                    "attemptedTargetTokens": state["attemptedTargetTokens"], "observedLoss": loss_value,
                    "newUniqueTokens": state["newUniqueTokens"], "replayTargetTokens": state["replayTargetTokens"],
                    "consumedContentHashes": state["consumedContentHashes"],
                    "consumedTargetRanges": state["consumedTargetRanges"],
                    "samplingNewTargetTokens": state["samplingNewTargetTokens"],
                    "samplingReplayTargetTokens": state["samplingReplayTargetTokens"],
                    "replayMix": state["replayMix"],
                    'committedTeacherTargetTokens':state['committedTeacherTargetTokens'],
                    'committedHumanTargetTokens':state['committedHumanTargetTokens'],
                    'actualTeacherFraction':state.get('actualTeacherFraction',0),
                    "validationNll": validation, "activeSeconds": state["activeSeconds"],
                    "terminationReason": termination_reason, "trainedTensorDigest": trained_digest,
                    "servingEligible": False, **({"finishedAt": now()} if ending == "completed" else {})}, job["id"])
            return {"state": ending, "runId": job["id"], "checkpointId": saved["id"],
                    "completedOptimizerSteps": state["completedOptimizerSteps"],
                    "committedTargetTokens": state["committedTargetTokens"], "validationNll": validation,
                    "terminationReason": termination_reason}
        except BaseException as exc:
            # Never publish a half optimizer update. Last complete checkpoint stays valid.
            if getattr(exc, "code", None) != "LEASE_EXPIRED":
                try:
                    self.store.assert_lease(job)
                    self.store.put("foundation_run", {"state": "failed", "errorCode": getattr(exc, "code", "TRAINING_FAILED"),
                        "lastCheckpointId": last_checkpoint_id, "servingEligible": False}, job["id"])
                except BaseException:
                    pass
            raise
        finally:
            stop_heartbeat.set()
            thread.join(timeout=2)
