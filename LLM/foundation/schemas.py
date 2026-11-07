"""Validated contracts for local, from-scratch foundation training.

These helpers deliberately do not import a model or download anything.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from LLM.core import require, digest, file_hash, inside

PRESET = "seed-s0-11m-v1"
FOUNDATION_ORIGINS = frozenset(('random_init','architecture_growth'))
SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>", "<|system|>", "<|user|>",
                  "<|assistant|>", "<|end|>", "<|fim_prefix|>", "<|fim_suffix|>",
                  "<|fim_middle|>", "<|file|>")
ARCHITECTURE = {
    "vocab_size": 16384, "hidden_size": 256, "intermediate_size": 768,
    "num_hidden_layers": 8, "num_attention_heads": 8, "num_key_value_heads": 8,
    "max_position_embeddings": 512, "hidden_act": "silu", "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0, "rope_scaling": None, "tie_word_embeddings": True,
    "attention_bias": False, "mlp_bias": False, "attention_dropout": 0.0,
    "initializer_range": 0.02, "pad_token_id": 0, "bos_token_id": 1, "eos_token_id": 2,
}


def integer(value, field, minimum=0, maximum=2**63-1):
    require(isinstance(value, int) and not isinstance(value, bool) and minimum <= value <= maximum,
            "CONFIG_INVALID", field=field)
    return value


def real(value, field, minimum=0.0, maximum=1.0, positive=False):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and minimum <= value <= maximum and (not positive or value > 0),
            "CONFIG_INVALID", field=field)
    return float(value)


def architecture_config(value=None, *, test_only=False, growth_receipt=None):
    """Return a closed Llama config. Small fixtures require an explicit trusted flag."""
    if value in (None, PRESET):
        return dict(ARCHITECTURE)
    require(isinstance(value, dict), "CONFIG_INVALID", field="architecture")
    if "architecture" in value:
        value = value["architecture"]
    elif "architecturePreset" in value and not any(k in value for k in ARCHITECTURE):
        require(value["architecturePreset"] == PRESET, "CONFIG_INVALID")
        return dict(ARCHITECTURE)
    require(not (set(value) - set(ARCHITECTURE)), "CONFIG_INVALID", field="architecture")
    result = {**ARCHITECTURE, **value}
    if not test_only:
        require(result == ARCHITECTURE or valid_growth_config(result,growth_receipt), "ARCHITECTURE_PRESET_REQUIRED")
    for key in ("vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
                "num_attention_heads", "num_key_value_heads", "max_position_embeddings"):
        integer(result[key], key, 1, 65536)
    require(result["vocab_size"] >= 268 and result["vocab_size"] <= 65535,
            "CONFIG_INVALID", field="vocab_size")
    d, h, kv = result["hidden_size"], result["num_attention_heads"], result["num_key_value_heads"]
    require(d % h == 0 and h % kv == 0 and (d // h) % 2 == 0, "CONFIG_INVALID", field="heads")
    require(result["tie_word_embeddings"] is True and result["attention_bias"] is False
            and result["mlp_bias"] is False and result["hidden_act"] == "silu"
            and result["rope_scaling"] is None, "CONFIG_INVALID", field="architecture")
    for key in ("attention_dropout", "rms_norm_eps", "initializer_range", "rope_theta"):
        real(result[key], key, 0, 1e9, positive=key != "attention_dropout")
    require(result["attention_dropout"] == 0.0 and [result[k] for k in
            ("pad_token_id", "bos_token_id", "eos_token_id")] == [0, 1, 2], "CONFIG_INVALID")
    return result


def valid_growth_config(config, receipt):
    """Validate a server-adopted receipt; public project requests cannot set it."""
    if not isinstance(receipt,dict):return False
    body={k:v for k,v in receipt.items() if k!='receiptHash'}
    if receipt.get('schemaVersion')!=1 or receipt.get('kind')!='verified_growth_recipe' or receipt.get('receiptHash')!=digest(body):return False
    source=receipt.get('sourceConfig');target=receipt.get('targetConfig');recipe=receipt.get('recipe',{})
    if not isinstance(source,dict) or target!=config or receipt.get('recipeHash')!=digest(recipe):return False
    immutable=set(ARCHITECTURE)-{'num_hidden_layers','intermediate_size'}
    if any(source.get(k)!=ARCHITECTURE[k] or config.get(k)!=ARCHITECTURE[k] for k in immutable):return False
    if not (8<=source.get('num_hidden_layers',0)<=32 and 768<=source.get('intermediate_size',0)<=6144):return False
    if not (8<=config.get('num_hidden_layers',0)<=32 and 768<=config.get('intermediate_size',0)<=6144):return False
    if parameter_count(config)>32_000_000:return False
    if recipe.get('sourceConfigHash')!=digest(source) or recipe.get('targetConfigHash')!=digest(config):return False
    if recipe.get('mappingVersion')!='seed-function-preserving-growth-v1':return False
    if not all(isinstance(receipt.get(k),str) and len(receipt[k])==64 for k in
        ('parentTensorDigest','candidateTensorDigest','tokenizerHash','sourceManifestHash','evidenceHash')):return False
    if recipe.get('operation')=='ffn_expand':
        return (config['num_hidden_layers']==source['num_hidden_layers'] and
            source['intermediate_size']<config['intermediate_size']<=source['intermediate_size']*2 and config['intermediate_size']%8==0)
    if recipe.get('operation')=='insert_identity_layer':
        return (config['num_hidden_layers']==source['num_hidden_layers']+1 and config['intermediate_size']==source['intermediate_size']
            and type(recipe.get('position')) is int and 0<=recipe['position']<=source['num_hidden_layers'])
    return False


def project_architecture(project):
    cfg = project.get("config", {})
    if isinstance(cfg, dict) and "architecture" in cfg:
        cfg = cfg["architecture"]
    if not cfg or (isinstance(cfg, dict) and not any(k in cfg for k in ARCHITECTURE)):
        cfg = project.get("architecturePreset", PRESET)
    return architecture_config(cfg, test_only=project.get("testOnly") is True,growth_receipt=project.get('trustedGrowthReceipt'))


def parameter_count(config):
    d, f, layers = config["hidden_size"], config["intermediate_size"], config["num_hidden_layers"]
    kv_width = config["num_key_value_heads"] * (d // config["num_attention_heads"])
    return config["vocab_size"] * d + layers * (2*d*d + 2*d*kv_width + 3*d*f + 2*d) + d


def record_path(store, record):
    require(isinstance(record.get("storagePath"), str), "ARTIFACT_PATH_INVALID")
    path = inside(store.root, record["storagePath"])
    require(path.is_dir(), "ARTIFACT_PATH_INVALID")
    relative = path.relative_to(store.root)
    part = Path(store.root)
    for name in relative.parts:
        part = part / name
        require(not part.is_symlink(), "ARTIFACT_PATH_INVALID")
    return path


def corpus_manifest(store, record):
    require(record.get("state") in ("ready", "verified"), "CORPUS_NOT_READY", status=409)
    root = record_path(store, record)
    path = inside(root, "manifest.json")
    require(path.is_file() and not path.is_symlink(), "CORPUS_MANIFEST_INVALID")
    if record.get("manifestHash"):
        require(file_hash(path) == record["manifestHash"], "CORPUS_HASH_MISMATCH")
    manifest = json.loads(path.read_text("utf-8"))
    require(isinstance(manifest, dict), "CORPUS_MANIFEST_INVALID")
    return root, manifest


def iter_corpus(store, record, split="train"):
    root,manifest=corpus_manifest(store,record)
    if manifest.get('kind') not in ('foundation_corpus','foundation_teacher_corpus') and manifest.get('schemaVersion')!=2:
        filename=split+'.jsonl'
        if split=='dev' and filename not in manifest.get('files',{}) and 'validation.jsonl' in manifest.get('files',{}):filename='validation.jsonl'
        path=inside(root,filename);expected=manifest.get('files',{}).get(filename)
        if isinstance(expected,dict):expected=expected.get('sha256') or expected.get('hash')
        require(path.is_file() and not path.is_symlink(),'CORPUS_SPLIT_MISSING',split=split)
        if expected:require(file_hash(path)==expected,'CORPUS_HASH_MISMATCH')
        with path.open('rb') as handle:
            row_number=0
            while line:=handle.readline(2*1024**2+1):
                require(len(line)<=2*1024**2,'CORPUS_RECORD_TOO_LARGE')
                if not line.strip():continue
                row=json.loads(line);row_number+=1
                require(isinstance(row,dict) and isinstance(row.get('text'),str) and row['text'],'CORPUS_RECORD_INVALID')
                require(not row.get('split') or row['split'] in (split,'validation' if split=='dev' else split),'SPLIT_LEAKAGE',status=409)
                yield {**row,'_rowNumber':row_number}
        return
    from .corpus import rows
    for row_number,row in enumerate(rows(store,record,split),1):
        yield {**row,"_rowNumber":row_number}


validate_architecture = architecture_config
DEFAULT_ARCHITECTURE = ARCHITECTURE
