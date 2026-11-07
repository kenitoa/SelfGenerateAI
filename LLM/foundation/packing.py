"""Deterministic bounded streaming blocks; every reported target is actually scored."""
from __future__ import annotations

import random

from LLM.core import require, digest


def _encode(store, corpus, tokenizer, row, text, component, split):
    from .shards import encode
    return encode(store, corpus, tokenizer, row, text, component=component, split=split)


def _source_rows(store, corpus, split, purpose,seed=42):
    from .corpus import rows,balanced_rows
    if split=='train' and purpose=='train':
        policy=corpus.get('samplingPolicy')
        require(policy is None or policy=={'kind':'python-focused-targets-v2','weights':{'python':90,'other_code':5,'docs':5}},'SAMPLER_POLICY_INVALID')
        return balanced_rows(store,corpus,split,purpose=purpose,seed=seed,target_weights=policy['weights'] if policy else None,language_limit=16_000_000 if policy else 8_000_000)
    return rows(store,corpus,split,purpose=purpose)


def _buffered_shuffle(rows, seed, maximum_bytes=8*1024*1024):
    rng = random.Random(seed)
    pending, size = [], 0
    for row in rows:
        weight = len(row["text"].encode("utf-8")) + 2048
        while pending and size + weight > maximum_bytes:
            index = rng.randrange(len(pending))
            selected, selected_size = pending.pop(index)
            size -= selected_size
            yield selected
        require(weight <= maximum_bytes, "CORPUS_RECORD_TOO_LARGE")
        pending.append((row, weight))
        size += weight
    while pending:
        index = rng.randrange(len(pending))
        selected, _ = pending.pop(index)
        yield selected


def _identity(row):
    return row.get("contentHash") or digest(row["text"].encode("utf-8"))


def _block(ids, labels, provenance):
    targets = sum(value != -100 for value in labels[1:])
    return {"inputIds": ids, "labels": labels, "targetCount": targets,
            "contentHashes": sorted(set(p[0] for p in provenance)),
            "groupIds": sorted(set(p[1] for p in provenance if p[1])),
            "targetOrigins": [[p[0], p[2]] if len(p) > 2 and p[2] >= 0 else None for p in provenance],
            "blockHash": digest([ids, labels])}


def _iter_blocks_plain(store, corpus, tokenizer, sequence_length, *, stage="pretrain", split="train",
                epoch=0, seed=42, shuffle=True, skip_blocks=0, purpose=None):
    require(isinstance(sequence_length, int) and sequence_length >= 8, "CONFIG_INVALID")
    purpose = purpose or ("train" if split == "train" else "dev")
    source = _source_rows(store, corpus, split, purpose,seed+epoch)
    if shuffle and split == "train" and not hasattr(source,'feedback'):
        source = _buffered_shuffle(source, seed + epoch)
    if stage == "sft":
        index = 0
        for row in source:
            # SFT corpus must explicitly carry independently approved instruction/output.
            instruction = row.get("instruction")
            target = row.get("output")
            if not isinstance(instruction, str) or not isinstance(target, str) or not target:
                continue
            require(row.get("targetOrigin", "source_span") != "model_generated", "SYNTHETIC_TARGET_FORBIDDEN")
            prefix = [1, 5, *_encode(store, corpus, tokenizer, row, instruction, "instruction", split), 7, 6]
            ending = [*_encode(store, corpus, tokenizer, row, target, "target", split), 7]
            if len(ending)>row.get('_samplerTargetBudget',len(ending)):continue
            if hasattr(source,'feedback'):source.feedback(len(ending))
            require(len(prefix) + len(ending) <= sequence_length, "INPUT_TOO_LARGE_FOR_TRAINING")
            ids = prefix + ending
            labels = [-100] * len(prefix) + ending
            if index >= skip_blocks:
                identity, group = _identity(row), row.get("groupId") or row.get("splitGroup") or ""
                provenance = [(identity, group, -1)] * len(prefix)
                provenance += [(identity, group, offset) for offset in range(len(ending))]
                packed = _block(ids, labels, provenance)
                packed['targetOrigin'] = row.get('targetOrigin','source_span')
                for key in ('teacherArtifactId','teacherManifestHash','teacherApprovalId','verifierReceiptId','teacherPolicyId','teacherPolicyVersion'):
                    if key in row: packed[key] = row[key]
                yield packed
            index += 1
        return
    require(stage in ("pretrain", "cpt"), "INVALID_TRAINING_OBJECTIVE")
    pending_ids, pending_labels, provenance = [], [], []
    index = 0
    for row in source:
        content = _encode(store, corpus, tokenizer, row, row["text"], "raw", split)
        budget=row.get('_samplerTargetBudget',len(content)+1)
        content=list(content)
        targets=content+[2] if len(content)+1<=budget else content[:budget]
        if hasattr(source,'feedback'):source.feedback(len(targets))
        ids = [1, *targets]
        identity = (_identity(row), row.get("groupId") or row.get("splitGroup") or "")
        for offset in range(0, len(ids), sequence_length):
            chunk = ids[offset:offset+sequence_length]
            pending_ids.extend(chunk)
            pending_labels.extend(-100 if token == 1 else token for token in chunk)
            provenance.extend([(*identity, offset+index-1) for index in range(len(chunk))])
            while len(pending_ids) >= sequence_length:
                if index >= skip_blocks:
                    yield _block(pending_ids[:sequence_length], pending_labels[:sequence_length], provenance[:sequence_length])
                index += 1
                # One context token overlaps; no target is counted twice.
                remove = sequence_length - 1
                del pending_ids[:remove]
                del pending_labels[:remove]
                del provenance[:remove]
    if len(pending_ids) >= 2:
        if index >= skip_blocks:
            yield _block(pending_ids, pending_labels, provenance)


def _pool_block(block, learned_ranges, learned):
    labels = list(block["labels"])
    count = 0
    for index, origin in enumerate(block["targetOrigins"]):
        if index == 0 or labels[index] == -100 or origin is None:
            labels[index] = -100
            continue
        content_hash, position = origin
        is_learned = any(start <= position < end for start, end in learned_ranges.get(content_hash, []))
        if is_learned == learned:
            count += 1
        else:
            labels[index] = -100
    return {**block, "labels": labels, "targetCount": count,
            "samplingPool": "replay" if learned else "new"} if count else None


def iter_blocks(store, corpus, tokenizer, sequence_length, *, stage="pretrain", split="train",
                epoch=0, seed=42, shuffle=True, skip_blocks=0, replay_ranges=None, mix_report=None, purpose=None):
    """CPT uses two bounded passes, splitting mixed blocks by actual target offset.

    replay_ranges is pinned at run creation. Resume must not reclassify targets
    using ranges learned by that same run, or its deterministic cursor changes.
    """
    if stage != "cpt" or split != "train":
        yield from _iter_blocks_plain(store, corpus, tokenizer, sequence_length, stage=stage,
            split=split, epoch=epoch, seed=seed, shuffle=shuffle, skip_blocks=skip_blocks, purpose=purpose)
        return
    learned_ranges = replay_ranges or {}
    def pool(learned):
        for block in _iter_blocks_plain(store, corpus, tokenizer, sequence_length, stage=stage,
                split=split, epoch=epoch, seed=seed, shuffle=shuffle, skip_blocks=0,purpose=purpose):
            selected = _pool_block(block, learned_ranges, learned)
            if selected is not None:
                yield selected
    # With no learned offsets, the replay pool is provably empty. Walking the
    # entire corpus to rediscover that fact blocks the first yielded batch.
    fresh, old = iter(pool(False)), iter(pool(True)) if learned_ranges else iter(())
    next_fresh, next_old = next(fresh, None), next(old, None)
    fresh_targets, old_targets, index = 0, 0, 0
    report = mix_report if mix_report is not None else {}
    deficits = set()
    if next_fresh is None:
        deficits.add("NEW_POOL_EMPTY")
    if next_old is None:
        deficits.add("REPLAY_POOL_EMPTY")
    while next_fresh is not None or next_old is not None:
        ratio = fresh_targets / (fresh_targets+old_targets) if fresh_targets+old_targets else 0.0
        choose_fresh = next_fresh is not None and (next_old is None or ratio <= .7)
        if choose_fresh:
            block = next_fresh
            fresh_targets += block["targetCount"]
            next_fresh = next(fresh, None)
            if next_fresh is None and next_old is not None:
                deficits.add("NEW_POOL_EXHAUSTED")
        else:
            block = next_old
            old_targets += block["targetCount"]
            next_old = next(old, None)
            if next_old is None and next_fresh is not None:
                deficits.add("REPLAY_POOL_EXHAUSTED")
        report.update({"targetNewFraction": .7, "sampledNewTargetTokens": fresh_targets,
                       "sampledReplayTargetTokens": old_targets,
                       "actualNewFraction": fresh_targets/(fresh_targets+old_targets),
                       "deficitReasonCodes": sorted(deficits)})
        if index >= skip_blocks:
            yield block
        index += 1


def tensor_batch(torch, blocks, pad_token_id=0):
    require(bool(blocks), "NO_TRAINABLE_EXAMPLES")
    maximum = max(len(block["inputIds"]) for block in blocks)
    ids, labels, masks = [], [], []
    for block in blocks:
        count = len(block["inputIds"])
        require(len(block["labels"]) == count, "PACKING_LABEL_MISMATCH")
        ids.append(block["inputIds"] + [pad_token_id] * (maximum-count))
        labels.append(block["labels"] + [-100] * (maximum-count))
        masks.append([1] * count + [0] * (maximum-count))
    return {"input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long), "use_cache": False}


def teacher_mix_descriptor(corpus):
    if corpus.get('kind') != 'foundation_teacher_corpus': return None
    return {'version':'atomic-teacher-target-v1','corpusHash':corpus['manifestHash'],
        'teacherPolicyHash':corpus['teacherPolicyHash'],'maxGroupBlocks':32,'maxGroupTargetTokens':8192}


def teacher_groups(store,corpus,tokenizer,cfg,*,epoch=0,skip_groups=0,report=None,purpose=None):
    from .response_distillation import validate_corpus, mix_blocks
    policy = validate_corpus(store,corpus)
    def factory():
        yield from iter_blocks(store,corpus,tokenizer,cfg['sequenceLength'],stage='sft',epoch=epoch,
            seed=cfg['seed'],shuffle=cfg['shuffle'],purpose=purpose)
    for index,group in enumerate(mix_blocks(factory,teacher_fraction=policy['maxTeacherFraction'],report=report)):
        if index >= skip_groups: yield group
