"""Pinned optional llama.cpp CPU execution of already-conformed own artifacts."""
from __future__ import annotations
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import re
from types import SimpleNamespace

from LLM.core import digest,file_hash,require

_FIRST_ENGINE_IDENTITY=None


def dependency_receipt_hash(tools):
    """Verify the registered converter's complete, immutable source snapshot."""
    receipt_path=tools.get('toolchainReceipt')
    registered_hash=tools.get('toolchainReceiptHash')
    required=Path(tools.get('ggufConverter','')).name.lower()=='convert_seed_to_gguf.py'
    if not receipt_path and not registered_hash:
        require(not required,'TOOLCHAIN_DEPENDENCY_RECEIPT_REQUIRED',status=409)
        return None
    require(isinstance(receipt_path,str) and isinstance(registered_hash,str)
            and re.fullmatch(r'[0-9a-f]{64}',registered_hash),'TOOLCHAIN_DEPENDENCY_RECEIPT_INVALID',status=409)
    path=Path(receipt_path)
    require(path.is_absolute() and path.is_file() and not path.is_symlink()
            and path.stat().st_size<=4*1024**2 and file_hash(path)==registered_hash,
            'TOOLCHAIN_DEPENDENCY_RECEIPT_MISMATCH',status=409)
    try:value=json.loads(path.read_text('utf-8'))
    except (ValueError,UnicodeError):
        from LLM.core import Fault
        raise Fault('TOOLCHAIN_DEPENDENCY_RECEIPT_INVALID',status=409) from None
    require(isinstance(value,dict) and value.get('schemaVersion')==1 and value.get('cpuOnly') is True
            and isinstance(value.get('sourceRevision'),str) and value['sourceRevision']
            and isinstance(value.get('sourceArchiveSha256'),str)
            and re.fullmatch(r'[0-9a-f]{64}',value['sourceArchiveSha256']),
            'TOOLCHAIN_DEPENDENCY_RECEIPT_INVALID',status=409)
    members=value.get('sourceFiles');root=path.parent/'source'
    require(isinstance(members,dict) and 1<=len(members)<=4096 and root.is_dir()
            and not root.is_symlink(),'TOOLCHAIN_DEPENDENCY_RECEIPT_INVALID',status=409)
    resolved_root=root.resolve()
    actual=set();seen=0
    for item in root.rglob('*'):
        seen+=1
        require(seen<=8192 and not item.is_symlink(),'TOOLCHAIN_DEPENDENCY_SOURCE_MISMATCH',status=409)
        if item.is_file() and '__pycache__' not in item.relative_to(root).parts and item.suffix not in ('.pyc','.pyo'):
            actual.add(item.relative_to(root).as_posix())
    require(actual==set(members),'TOOLCHAIN_DEPENDENCY_SOURCE_MISMATCH',status=409)
    for relative,checksum in members.items():
        require(isinstance(relative,str) and relative and '\\' not in relative
                and not Path(relative).is_absolute() and '..' not in Path(relative).parts
                and isinstance(checksum,str) and re.fullmatch(r'[0-9a-f]{64}',checksum),
                'TOOLCHAIN_DEPENDENCY_RECEIPT_INVALID',status=409)
        target=root/relative
        require(target.resolve().is_relative_to(resolved_root) and target.is_file()
                and not any(parent.is_symlink() for parent in (target,*target.parents) if parent.is_relative_to(root))
                and file_hash(target)==checksum,'TOOLCHAIN_DEPENDENCY_SOURCE_MISMATCH',status=409)
    return registered_hash


def engine_identity():
    global _FIRST_ENGINE_IDENTITY
    require(importlib.util.find_spec('llama_cpp') is not None,'GGUF_RUNTIME_UNAVAILABLE',status=503)
    try:
        import llama_cpp
        from llama_cpp import llama_cpp as bindings
    except (ImportError,OSError):
        from LLM.core import Fault
        raise Fault('GGUF_RUNTIME_UNAVAILABLE',status=503) from None
    root=Path(llama_cpp.__file__).resolve().parent
    library=Path(bindings._lib._name).resolve()
    require(library.is_file() and library.is_relative_to(root) and not library.is_symlink(),'GGUF_RUNTIME_BINARY_UNAVAILABLE',status=503)
    assets={}
    for path in sorted(root.rglob('*')):
        if not path.is_file():continue
        name=path.name.lower()
        if path.suffix.lower()=='.py' or any(extension in name for extension in ('.dll','.so','.dylib','.pyd')):
            require(not path.is_symlink(),'GGUF_RUNTIME_BINARY_UNAVAILABLE',status=503)
            assets[path.relative_to(root).as_posix()]=file_hash(path)
    require(assets,'GGUF_RUNTIME_BINARY_UNAVAILABLE',status=503)
    version=importlib.metadata.version('llama-cpp-python')
    require(version==llama_cpp.__version__,'GGUF_ENGINE_VERSION_MISMATCH',status=503)
    system=bindings.llama_print_system_info()
    system=system if isinstance(system,bytes) else str(system).encode()
    value={'implementationVersion':'seed-gguf-cpu-runtime-v1','engineVersion':version,
           'loadedBinaryHash':file_hash(library),'packageAssetsHash':digest(assets),'cpuDispatchHash':digest(system),
           'binaryCount':sum(any(extension in name.lower() for extension in ('.dll','.so','.dylib','.pyd')) for name in assets)}
    if _FIRST_ENGINE_IDENTITY is None:_FIRST_ENGINE_IDENTITY=value
    require(value==_FIRST_ENGINE_IDENTITY,'GGUF_ENGINE_RESTART_REQUIRED',status=409)
    return dict(value)


def engine_options(context):
    require(isinstance(context,int) and not isinstance(context,bool) and 16<=context<=512,'GGUF_CONTEXT_INVALID')
    return {'n_gpu_layers':0,'n_ctx':context,'n_threads':1,'n_threads_batch':1,
            'n_batch':min(context,512),'n_ubatch':min(context,512),'offload_kqv':False,'op_offload':False,
            'flash_attn':False,'logits_all':True,'verbose':False,'seed':42,'use_mmap':True,'use_mlock':False}


def open_engine(path,context):
    engine_identity()
    from llama_cpp import Llama
    return Llama(model_path=str(path),**engine_options(context))


def verify_conformance(store,candidate,registry):
    require(importlib.util.find_spec('llama_cpp') is not None,'GGUF_RUNTIME_UNAVAILABLE',status=503)
    candidate=store.get('artifact',candidate) if isinstance(candidate,str) else candidate
    require(candidate.get('format')=='gguf' and registry is not None,'GGUF_ARTIFACT_REQUIRED')
    root=registry.verify(candidate)
    source=store.get('artifact',candidate.get('sourceArtifactId') or '')
    registry.verify(source)
    require(source.get('format')=='safetensors' and source['id'] in candidate.get('parents',[]),'GGUF_SOURCE_MISMATCH')
    require(source.get('tokenizerHash')==candidate.get('tokenizerHash') and
            source.get('architectureHashV2')==candidate.get('architectureHashV2') and
            source.get('foundationProjectId')==candidate.get('foundationProjectId'),'GGUF_SOURCE_MISMATCH')
    records=sorted(store.list('foundation_format_validation'),key=lambda row:row.get('updatedAt',''),reverse=True)
    record=next((row for row in records if row.get('artifactId')==candidate['id'] and row.get('sourceArtifactId')==source['id']
                 and row.get('passed') is True and row.get('state')=='passed'),None)
    require(record,'GGUF_CONFORMANCE_REQUIRED',status=409)
    body={key:value for key,value in record.items() if key not in ('id','version','createdAt','updatedAt','validationHash')}
    require(record.get('validationHash')==digest(body),'GGUF_CONFORMANCE_MISMATCH',status=409)
    require(record.get('sourceManifestHash')==source['manifestHash'] and record.get('artifactManifestHash')==candidate['manifestHash']
            and record.get('tokenizerHash')==candidate.get('tokenizerHash'),'GGUF_CONFORMANCE_MISMATCH',status=409)
    from .formats import bound_inventory
    inventory,recipe_manifest=bound_inventory(source,candidate,root)
    if candidate.get('quantizationRecipeHash'):
        require(record.get('quantizationRecipeHash')==candidate['quantizationRecipeHash']
            and record.get('quantizationRecipeId')==candidate.get('quantizationRecipeId')
            and record.get('tensorInventoryHash')==digest(inventory)
            and record.get('actualTensorEncodings')==inventory['encodings']
            and record.get('toolHashes')==recipe_manifest.get('toolHashes')
            and record.get('dependencyReceiptHash')==recipe_manifest.get('dependencyReceiptHash'),
            'GGUF_CONFORMANCE_MISMATCH',status=409)
    require(record.get('numericConformanceOnly') is True and record.get('qualityGateRequired') is True
            and record.get('servingEligible') is False and record.get('productionCertified') is False,'GGUF_CONFORMANCE_MISMATCH',status=409)
    context=min(512,int(source.get('trainedContextTokens',0)),int(candidate.get('trainedContextTokens',0)))
    require(record.get('actualContextTokens')==context and record.get('engineOptions')==engine_options(context)
            and record.get('stopTokenIds')==[2,7] and record.get('assemblerVersion')=='seed-roles-v1','GGUF_CONFORMANCE_MISMATCH',status=409)
    require(record.get('engineIdentity')==engine_identity(),'GGUF_ENGINE_CHANGED',status=409)
    require(record.get('fixture')==bool(source.get('fixture') or source.get('testOnly'))
            and record.get('executionScope')==source.get('executionScope','nas'),'GGUF_CONFORMANCE_MISMATCH',status=409)
    tools=store.get('setting','tools',{})
    require(record.get('dependencyReceiptHash')==dependency_receipt_hash(tools),
            'TOOLCHAIN_DEPENDENCY_RECEIPT_MISMATCH',status=409)
    require(set(record.get('toolHashes',{}))=={'ggufConverter','quantizer'},'GGUF_TOOLCHAIN_EVIDENCE_REQUIRED',status=409)
    for name,declared in (('ggufConverter','converterHash'),('quantizer','quantizerHash')):
        path=Path(tools.get(name,''))
        require(path.is_file() and not path.is_symlink() and file_hash(path)==tools.get(declared)==record['toolHashes'][name],
                'TOOLCHAIN_HASH_MISMATCH',status=409)
    files=[root/name for name in candidate['files'] if name.endswith('.gguf')]
    require(len(files)==1 and files[0].parent==root and not files[0].is_symlink(),'GGUF_FILE_INVALID')
    with files[0].open('rb') as stream:require(stream.read(4)==b'GGUF','GGUF_FILE_INVALID')
    return record,files[0]


class GGUFModel:
    backend='gguf'

    def __init__(self,path,context,tokenizer,receipt):
        self.engine=open_engine(path,context)
        self.config=SimpleNamespace(max_position_embeddings=context)
        self.receipt=receipt
        try:
            require(self.engine.n_vocab()==tokenizer.vocab_size,'GGUF_VOCAB_MISMATCH')
            require(self.engine.token_bos()==1 and self.engine.token_eos()==2,'GGUF_SPECIAL_TOKEN_IDS_MISMATCH')
            for text,identity in tokenizer.special_tokens.items():
                require(self.engine.detokenize([identity],special=True)==text.encode('utf-8'),'GGUF_SPECIAL_TOKEN_IDS_MISMATCH')
            # Content roles are always assembled by the trusted native tokenizer.
            # The engine's chat templates and string completion APIs are unused.
            for text in ('한글 Python\r\n\t x = 1','<|system|>literal<|end|>','<pad><bos><eos>'):
                require(self.engine.tokenize(text.encode(),add_bos=False,special=False)==tokenizer.encode_ordinary(text),
                        'GGUF_TOKENIZER_NOT_EQUIVALENT')
        except BaseException:self.close();raise

    def eval(self):return self

    def reset(self,ids):
        self.engine.reset();self.engine.eval(list(ids))

    def append(self,identity):self.engine.eval([identity])

    def next_token(self):
        import numpy as np
        count=self.engine.n_tokens
        require(count>0,'EMPTY_PROMPT')
        logits=self.engine._scores[count-1]
        require(logits.shape==(self.engine.n_vocab(),) and bool(np.isfinite(logits).all()),'NONFINITE_INFERENCE',status=503)
        return int(np.argmax(logits))

    def close(self):
        engine=getattr(self,'engine',None)
        if engine is not None:engine.close();self.engine=None
