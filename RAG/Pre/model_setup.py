"""Explicit, reproducible CPU embedding installation; never runs at server startup."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import httpx
from RAG.core import DomainError, now
from .embeddings import LocalEncoder

REPO='Xenova/multilingual-e5-small'
REVISION='761b726dd34fb83930e26aab4e9ac3899aa1fa78'
ARTIFACTS={
    'model.onnx':('onnx/model_quantized.onnx',118308185,'f80102d3f2a1229f387d3c81909990d8945513e347b0eab049f7de3c6f98c193'),
    'tokenizer.json':('tokenizer.json',17082730,'0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39'),
}

def file_hash(path):
    hashed=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):hashed.update(block)
    return hashed.hexdigest()

def install(store,progress=None):
    directory=store.root/'models'/'multilingual-e5-small-int8';directory.mkdir(parents=True,exist_ok=True)
    for name,(remote,size,checksum) in ARTIFACTS.items():
        target=directory/name
        if target.exists() and file_hash(target)==checksum:continue
        temporary=directory/(name+'.download')
        for attempt in range(3):
            try:
                count=0;hashed=hashlib.sha256()
                with httpx.Client(follow_redirects=True,timeout=httpx.Timeout(120,connect=15)) as client:
                    with client.stream('GET',f'https://huggingface.co/{REPO}/resolve/{REVISION}/{remote}') as response:
                        response.raise_for_status()
                        with temporary.open('wb') as output:
                            for block in response.iter_bytes(1024**2):
                                count+=len(block)
                                if count>size:raise ValueError('artifact exceeds pinned size')
                                output.write(block);hashed.update(block)
                                if progress:progress(name,count,size)
                            output.flush();os.fsync(output.fileno())
                if count!=size or hashed.hexdigest()!=checksum:raise ValueError('artifact checksum mismatch')
                os.replace(temporary,target);break
            except (httpx.HTTPError,ValueError,OSError):
                temporary.unlink(missing_ok=True)
                if attempt==2:raise
                time.sleep(attempt+1)
    profile={'model':'intfloat/multilingual-e5-small','artifactRepository':REPO,'modelRevision':REVISION,'tokenizerRevision':REVISION,
             'dimension':384,'maxTokens':512,'quantization':'int8','license':'MIT','installedAt':now(),
             'checksums':{name:checksum for name,(_,_,checksum) in ARTIFACTS.items()}}
    (directory/'profile.json').write_text(json.dumps(profile,ensure_ascii=False,indent=2),encoding='utf-8')
    config={'enabled':True,'directory':str(directory)}
    encoder=LocalEncoder(config)
    if not encoder.available:raise DomainError('EMBEDDING_INSTALL_FAILED',encoder.reason,503)
    vectors=encoder.encode(['중복 항목을 제거합니다.','Remove duplicate items.'],query=True)
    if vectors.shape!=(2,384):raise DomainError('EMBEDDING_INSTALL_FAILED','임베딩 실물 검증 실패',503)
    with store.tx():
        store.set_setting('embedding',config)
        settings=store.setting('settings',{})
        store.set_setting('settings',{**settings,'embedding':config,'version':settings.get('version',1)+1})
    store.activity('embedding.installed','workspace','다국어 E5 INT8 · CPU 임베딩 설치·검증')
    return {'directory':str(directory),'profile':profile,'verified':True}
