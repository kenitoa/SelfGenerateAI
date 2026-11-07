"""Pinned official external weights for independent comparison, never training."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import urllib.request

from LLM.core import atomic_json, digest, file_hash, require

MODEL_ID = 'Qwen/Qwen2.5-Coder-0.5B-Instruct'
REVISION = 'ea3f2471cf1b1f0db85067f1ef93848e38e88c25'
FILES = ('LICENSE','README.md','config.json','generation_config.json','merges.txt',
         'model.safetensors','tokenizer.json','tokenizer_config.json','vocab.json')
MAX_BYTES = 2 * 1024**3


def _request(url):
    return urllib.request.Request(url,headers={'User-Agent':'SelfGenerateAI-comparison-only/1'})


def collect(destination, revision=REVISION):
    from .acceptance import require_local_path
    root = require_local_path(destination)
    require(re.fullmatch('[0-9a-f]{40}',revision), 'PINNED_MODEL_REVISION_REQUIRED')
    require(not root.is_symlink() and all(not p.is_symlink() for p in root.parents), 'MODEL_PATH_NOT_ALLOWED')
    root.mkdir(parents=True,exist_ok=True)
    api = 'https://huggingface.co/api/models/'+MODEL_ID+'/revision/'+revision+'?blobs=true'
    with urllib.request.urlopen(_request(api),timeout=30) as response:
        raw = response.read(4*1024**2+1)
    require(len(raw) <= 4*1024**2, 'MARKET_MODEL_METADATA_TOO_LARGE')
    metadata = json.loads(raw)
    require(metadata.get('id') == MODEL_ID and metadata.get('sha') == revision
        and metadata.get('cardData',{}).get('license') == 'apache-2.0', 'MARKET_MODEL_LICENSE_REQUIRED')
    inventory = {item['rfilename']:item for item in metadata['siblings']}
    require(all(name in inventory for name in FILES), 'MODEL_BUNDLE_INCOMPLETE')
    expected_bytes = sum(inventory[name]['size'] for name in FILES)
    require(expected_bytes <= MAX_BYTES and shutil.disk_usage(root).free > expected_bytes*2+1024**3,
            'MARKET_MODEL_DISK_BUDGET_REQUIRED')
    bundle = root/revision; bundle.mkdir(exist_ok=True)
    hashes = {}
    for name in FILES:
        entry = inventory[name]; path = bundle/name
        require(not path.is_symlink(), 'MODEL_PATH_NOT_ALLOWED')
        expected = entry.get('lfs',{}).get('sha256')
        if path.is_file():
            data_hash = file_hash(path)
            valid = path.stat().st_size == entry['size']
            if expected: valid = valid and data_hash == expected
            else:
                content=path.read_bytes()
                valid = valid and hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()==entry['blobId']
            require(valid,'MARKET_MODEL_HASH_MISMATCH',file=name)
        else:
            temporary = bundle/(name+'.partial')
            require(not temporary.exists(), 'MARKET_MODEL_PARTIAL_REQUIRES_REVIEW',file=name)
            total=0; sha=hashlib.sha256(); git=hashlib.sha1(b'blob '+str(entry['size']).encode()+b'\0')
            try:
                url='https://huggingface.co/'+MODEL_ID+'/resolve/'+revision+'/'+name+'?download=true'
                with urllib.request.urlopen(_request(url),timeout=120) as response, temporary.open('xb') as target:
                    while block := response.read(1024**2):
                        total += len(block)
                        require(total <= entry['size'], 'MARKET_MODEL_SIZE_MISMATCH')
                        sha.update(block); git.update(block); target.write(block)
                require(total == entry['size'] and (sha.hexdigest()==expected if expected else git.hexdigest()==entry['blobId']),
                        'MARKET_MODEL_HASH_MISMATCH',file=name)
                temporary.replace(path)
            finally:
                if temporary.exists():temporary.unlink()
            data_hash=sha.hexdigest()
        hashes[name]=data_hash
    license_text=(bundle/'LICENSE').read_text('utf-8')
    require('Apache License' in license_text and 'Version 2.0' in license_text
        and 'END OF TERMS AND CONDITIONS' in license_text, 'MARKET_MODEL_LICENSE_REQUIRED')
    configuration=json.loads((bundle/'config.json').read_text('utf-8'))
    require(configuration.get('model_type')=='qwen2' and not configuration.get('auto_map'), 'MARKET_MODEL_CONFIGURATION_INVALID')
    receipt={'schemaVersion':1,'sourceRepository':MODEL_ID,'revision':revision,'license':'Apache-2.0',
        'licenseUrl':'https://huggingface.co/'+MODEL_ID+'/blob/'+revision+'/LICENSE',
        'licenseHash':hashes['LICENSE'],'metadataHash':digest(metadata),'files':hashes,
        'downloadBytes':expected_bytes,'parameterCount':metadata.get('safetensors',{}).get('total'),
        'origin':'external','comparisonOnly':True,'teacherUseAllowed':False,'trainingUseAllowed':False,
        'servingEligible':False,'neverTrain':True,'executionScope':'pc_validation','nasModified':False}
    receipt['receiptHash']=digest(receipt)
    atomic_json(root/'repository.json',metadata)
    atomic_json(root/'comparison-intake.json',receipt)
    return bundle,receipt


def prepare(control, destination, revision=REVISION):
    require(getattr(control,'execution_scope','nas')=='pc_validation','PC_COMPARISON_SCOPE_REQUIRED')
    from .acceptance import require_local_path
    require_local_path(control.store.root)
    before={r['id']:(r.get('genesisId'),r.get('pretrainedWeightArtifactIds',[]))
            for r in control.store.list('foundation_project')}
    bundle,receipt=collect(destination,revision)
    artifact=control.registry.publish(bundle,kind='base',parents=[],license_refs=[receipt['licenseUrl']],metadata={
        'origin':'external','comparisonOnly':True,'comparisonPurpose':'fixed_independent_market_baseline',
        'sourceRepository':MODEL_ID,'sourceRevision':revision,'executionScope':'pc_validation',
        'comparisonReceiptHash':receipt['receiptHash'],'neverTrain':True,'servingEligible':False,'fixture':False})
    require(artifact['parameterCount']==receipt['parameterCount'] and artifact['files']==receipt['files'],
            'MARKET_MODEL_REGISTRATION_MISMATCH')
    registered=control.store.put('foundation_market_baseline',{**receipt,'artifactId':artifact['id'],
        'artifactManifestHash':artifact['manifestHash'],'state':'ready','bundlePath':str(bundle)},
        digest([MODEL_ID,revision,artifact['manifestHash']]))
    require(before=={r['id']:(r.get('genesisId'),r.get('pretrainedWeightArtifactIds',[]))
            for r in control.store.list('foundation_project')},'FOUNDATION_WEIGHT_POLICY_CHANGED')
    verify(control,artifact['id'])
    return {'artifact':artifact,'receipt':registered,'modelTrainingPerformed':False,'nasModified':False}


def verify(control, artifact_id):
    artifact=control.store.get('artifact',artifact_id)
    require(artifact.get('origin')=='external' and artifact.get('comparisonOnly') is True
        and artifact.get('neverTrain') is True and artifact.get('executionScope')=='pc_validation'
        and artifact.get('servingEligible') is False and not artifact.get('foundationProjectId'),
        'MARKET_COMPARISON_ONLY_REQUIRED')
    control.registry.verify(artifact)
    receipts=[r for r in control.store.list('foundation_market_baseline') if r.get('artifactId')==artifact_id]
    require(len(receipts)==1,'MARKET_BASELINE_RECEIPT_REQUIRED')
    receipt=receipts[0]
    original={k:v for k,v in receipt.items() if k not in ('id','version','createdAt','updatedAt','receiptHash',
        'artifactId','artifactManifestHash','state','bundlePath')}
    require(receipt.get('receiptHash')==digest(original) and receipt.get('state')=='ready'
        and receipt.get('sourceRepository')==MODEL_ID and receipt.get('revision')==artifact.get('sourceRevision')
        and receipt.get('files')==artifact.get('files') and receipt.get('licenseHash')==artifact['files'].get('LICENSE')
        and receipt.get('artifactManifestHash')==artifact['manifestHash']
        and artifact.get('comparisonReceiptHash')==receipt['receiptHash'],'MARKET_BASELINE_RECEIPT_INVALID')
    return receipt


def baseline_ids(control):
    records=[r for r in control.store.list('foundation_market_baseline') if r.get('state')=='ready']
    require(records,'MARKET_BASELINE_REQUIRED')
    identities=[]
    for record in records:
        verify(control,record['artifactId']);identities.append(record['artifactId'])
    return sorted(set(identities))
