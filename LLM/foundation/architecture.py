"""A trusted HF Llama implementation initialized from our local configuration only."""
from __future__ import annotations

import json
import random

from LLM.core import require, digest, uid, now
from .schemas import project_architecture, architecture_config, parameter_count, integer


def create_model(config, *, test_only=False, init_fn=None, growth_receipt=None):
    import torch
    cfg = architecture_config(config, test_only=test_only,growth_receipt=growth_receipt)
    require(init_fn is None or test_only, "TEST_FACTORY_FORBIDDEN", status=403)
    if init_fn is not None:
        model = init_fn(dict(cfg))
    else:
        from transformers import LlamaConfig, LlamaForCausalLM
        native = LlamaConfig(**cfg)
        native._attn_implementation = "eager"
        model = LlamaForCausalLM(native)
    model = model.to(device="cpu", dtype=torch.float32)
    model.config.use_cache = False
    model.tie_weights()
    actual = sum(p.numel() for p in model.parameters())
    require(actual == parameter_count(cfg), "MODEL_PARAMETER_COUNT_MISMATCH",
            expected=parameter_count(cfg), actual=actual)
    require(model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr(),
            "MODEL_TIED_WEIGHTS_INVALID")
    return model


def model_digest(model):
    return digest([(name, digest(p.detach().cpu().contiguous().numpy().tobytes()))
                   for name, p in model.named_parameters()])


def initialize(store, project, tokenizer_record, *, init_fn=None):
    import torch
    from .tokenizer import load
    from .checkpoint import save
    cfg = project_architecture(project)
    tokenizer = load(store, tokenizer_record)
    require(tokenizer.vocab_size == cfg["vocab_size"], "TOKENIZER_HASH_MISMATCH")
    seed = integer(project.get("seed", project.get("config", {}).get("seed", 42)), "seed", 0, 2**32-1)
    # Keep initialization from silently altering inference or another caller's RNG.
    python_state = random.getstate()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = create_model(cfg, test_only=project.get("testOnly") is True, init_fn=init_fn,growth_receipt=project.get('trustedGrowthReceipt'))
    random.setstate(python_state)
    require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), "MODEL_NONFINITE")
    state = {"projectId": project["id"], "origin": "random_init", "pretrainedWeightArtifactIds": [],
             "seed": seed, "completedOptimizerSteps": 0, "committedTargetTokens": 0,
             "attemptedTargetTokens": 0, "cursor": {"epoch": 0, "block": 0},
             "testOnly": project.get("testOnly") is True, "servingEligible": False,
             "initialTensorDigest": model_digest(model)}
    genesis = save(store, model, cfg, tokenizer_record, state, record_kind="foundation_genesis")
    store.event("genesis.ready", genesis["id"], {"projectId": project["id"]})
    return genesis


def load_model(store, record_or_id, *, init_fn=None):
    import torch
    from .checkpoint import verify, load_model_state
    if isinstance(record_or_id, str):
        record = store.get("foundation_checkpoint", record_or_id, {}) or store.get("foundation_genesis", record_or_id)
    else:
        record = record_or_id
    root, manifest = verify(store, record)
    cfg = manifest["config"]
    with torch.random.fork_rng(devices=[]):
        project=store.get('foundation_project',record.get('projectId') or '',{})
        receipt=project.get('trustedGrowthReceipt')
        if cfg!=architecture_config():
            require(project.get('testOnly') or (record.get('trustedGrowthReceipt')==receipt and manifest.get('trustedGrowthReceipt')==receipt),'GROWTH_RECEIPT_REQUIRED')
        model = create_model(cfg, test_only=manifest.get("testOnly") is True, init_fn=init_fn,growth_receipt=receipt)
    load_model_state(root, manifest, model)
    return model
