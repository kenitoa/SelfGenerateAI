"""Real GGUF CPU conformance probes. Missing engines/tools never count as support."""
from __future__ import annotations

import importlib.util
import importlib.metadata
import json
import math
import struct
from pathlib import Path
from LLM.core import Fault, require, file_hash, digest

Q4_RECIPE_ID='seed-q4-k-m-f16-embedding-output-attention-v1'


def quantization_recipe(quantization):
    require(quantization in ('F16','Q4_K_M'),'UNSUPPORTED_QUANTIZATION')
    return {'schemaVersion':1,'id':'seed-f16-v1' if quantization=='F16' else Q4_RECIPE_ID,
        'baseQuantization':quantization,'tokenEmbeddingType':'F16','outputTensorType':'F16',
        'tensorOverrides':{} if quantization=='F16' else {name:'F16' for name in ('attn_q','attn_k','attn_v','attn_output')},
        'threads':1,'description':'F16' if quantization=='F16' else 'Q4_K_M FFN weights with F16 embeddings, output projection and attention projections'}


def inspect_gguf(path):
    """Bounded binary inventory; no label or nominal file type proves a tensor's bits."""
    path=Path(path);size=path.stat().st_size;encodings={0:('F32',1,4),1:('F16',1,2),2:('Q4_0',32,18),
        3:('Q4_1',32,20),6:('Q5_0',32,22),7:('Q5_1',32,24),8:('Q8_0',32,34),9:('Q8_1',32,40),
        10:('Q2_K',256,84),11:('Q3_K',256,110),12:('Q4_K',256,144),13:('Q5_K',256,176),
        14:('Q6_K',256,210),15:('Q8_K',256,292),30:('BF16',1,2)}
    scalars={0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
    with path.open('rb') as handle:
        def read(count):
            require(0<=count<=64*1024**2 and handle.tell()+count<=min(size,64*1024**2),'GGUF_METADATA_LIMIT')
            value=handle.read(count);require(len(value)==count,'GGUF_FILE_INVALID');return value
        def integer(code):return struct.unpack('<'+code,read(struct.calcsize('<'+code)))[0]
        def string():
            length=integer('Q');require(length<=1024**2,'GGUF_METADATA_LIMIT')
            try:return read(length).decode('utf-8')
            except UnicodeError:require(False,'GGUF_FILE_INVALID')
        def skip(kind,depth=0):
            require(depth<=2,'GGUF_METADATA_LIMIT')
            if kind==8:string()
            elif kind==9:
                item=integer('I');count=integer('Q');require(count<=1024**2,'GGUF_METADATA_LIMIT')
                if item in scalars:read(count*scalars[item])
                else:
                    for _ in range(count):skip(item,depth+1)
            else:
                require(kind in scalars,'GGUF_FILE_INVALID');read(scalars[kind])
        require(read(4)==b'GGUF' and integer('I') in (2,3),'GGUF_FILE_INVALID')
        count=integer('Q');metadata_count=integer('Q')
        require(0<count<=100000 and metadata_count<=4096,'GGUF_METADATA_LIMIT')
        alignment=32
        for _ in range(metadata_count):
            key=string();kind=integer('I')
            if key=='general.alignment':
                require(kind==4,'GGUF_FILE_INVALID');alignment=integer('I')
                require(0<alignment<=4096 and not alignment&(alignment-1),'GGUF_FILE_INVALID')
            else:skip(kind)
        rows=[];names=set();types={}
        for _ in range(count):
            name=string();dimensions=integer('I');require(name not in names and 1<=dimensions<=4,'GGUF_TENSOR_INVALID');names.add(name)
            shape=[integer('Q') for _ in range(dimensions)]
            require(all(0<dimension<=2**32 for dimension in shape),'GGUF_TENSOR_INVALID')
            code=integer('I');offset=integer('Q');require(code in encodings,'GGUF_TENSOR_ENCODING_UNSUPPORTED')
            label,block,block_bytes=encodings[code];elements=math.prod(shape)
            require(shape[0]%block==0,'GGUF_TENSOR_INVALID')
            length=elements//block*block_bytes
            row={'name':name,'shape':shape,'encoding':label,'encodingId':code,'elements':elements,'bytes':length,'offset':offset}
            rows.append(row);group=types.setdefault(label,{'tensorCount':0,'elements':0,'bytes':0})
            group['tensorCount']+=1;group['elements']+=elements;group['bytes']+=length
        data_offset=(handle.tell()+alignment-1)//alignment*alignment
        end=0
        for row in sorted(rows,key=lambda item:item['offset']):
            require(row['offset']>=end and row['offset']%alignment==0 and data_offset+row['offset']+row['bytes']<=size,'GGUF_TENSOR_INVALID')
            end=row['offset']+row['bytes']
    return {'schemaVersion':1,'fileSha256':file_hash(path),'fileBytes':size,'tensorCount':len(rows),
        'tensorBytes':sum(row['bytes'] for row in rows),'tensorElements':sum(row['elements'] for row in rows),
        'encodings':types,'tensors':rows}


def validate_recipe(inventory,recipe,*,source_tied_embeddings=False):
    require(recipe==quantization_recipe(recipe.get('baseQuantization')),'QUANTIZATION_RECIPE_MISMATCH')
    rows={row['name']:row for row in inventory['tensors']}
    if recipe['baseQuantization']=='Q4_K_M':
        require(inventory['encodings'].get('Q4_K',{}).get('elements',0)>0,'GGUF_ACTUAL_Q4_TENSORS_REQUIRED')
        require(rows.get('token_embd.weight',{}).get('encoding')=='F16','GGUF_EMBEDDING_PRECISION_MISMATCH')
        require(rows.get('output.weight',{}).get('encoding')=='F16' or 'output.weight' not in rows and source_tied_embeddings is True,
            'GGUF_OUTPUT_PRECISION_MISMATCH')
        for name in recipe['tensorOverrides']:
            retained=[row for row in rows.values() if '.'+name+'.' in row['name']]
            require(retained and all(row['encoding']=='F16' for row in retained),'GGUF_ATTENTION_PRECISION_MISMATCH')
        require(all('.ffn_' in row['name'] for row in rows.values() if row['encoding'] not in ('F16','F32')),
            'GGUF_QUANTIZED_TENSOR_SCOPE_MISMATCH')
    else:require(set(inventory['encodings'])<= {'F16','F32'} and inventory['encodings'].get('F16',{}).get('elements',0)>0,'GGUF_FLOAT_PRECISION_MISMATCH')
    return inventory


def bound_inventory(source,candidate,root):
    files=[root/name for name in candidate['files'] if name.endswith('.gguf')]
    require(len(files)==1 and files[0].parent==root and not files[0].is_symlink(),'GGUF_FILE_INVALID')
    inventory=inspect_gguf(files[0]);recipe_hash=candidate.get('quantizationRecipeHash')
    if recipe_hash:
        values={}
        for name,limit in (('quantization-recipe.json',1024**2),('tensor-inventory.json',16*1024**2)):
            path=root/name
            require(name in candidate['files'] and path.is_file() and not path.is_symlink() and path.stat().st_size<=limit
                and file_hash(path)==candidate['files'][name],'QUANTIZATION_RECIPE_MISMATCH')
            try:values[name]=json.loads(path.read_text('utf-8'))
            except (ValueError,UnicodeError):raise Fault('QUANTIZATION_RECIPE_MISMATCH') from None
        manifest=values['quantization-recipe.json'];recipe=manifest.get('recipe',{})
        require(digest(manifest)==recipe_hash and manifest.get('schemaVersion')==1
            and candidate.get('quantizationRecipeId')==recipe.get('id')
            and recipe.get('baseQuantization')==candidate.get('quantization')
            and manifest.get('sourceArtifactId')==source['id'] and manifest.get('sourceManifestHash')==source['manifestHash']
            and manifest.get('outputGgufSha256')==inventory['fileSha256']
            and digest(inventory)==manifest.get('tensorInventoryHash')==candidate.get('tensorInventoryHash')
            and inventory==values['tensor-inventory.json'] and inventory['encodings']==candidate.get('actualTensorEncodings')
            and inventory['fileBytes']==candidate.get('ggufBytes'),'QUANTIZATION_RECIPE_MISMATCH')
        config=json.loads((root/'config.json').read_text('utf-8'))
        require(manifest.get('sourceTiedEmbeddings')==(config.get('tie_word_embeddings') is True),'QUANTIZATION_RECIPE_MISMATCH')
        validate_recipe(inventory,recipe,source_tied_embeddings=manifest['sourceTiedEmbeddings'])
        float_bytes=manifest.get('sourceFloatGgufBytes')
        require(type(float_bytes) is int and float_bytes==candidate.get('sourceFloatGgufBytes')
            and float_bytes>=inventory['fileBytes'] and float_bytes/inventory['fileBytes']==candidate.get('compressionRatioVsF16Gguf'),
            'QUANTIZATION_RECIPE_MISMATCH')
        if recipe['baseQuantization']=='Q4_K_M':require(float_bytes>inventory['fileBytes'],'GGUF_COMPRESSION_REQUIRED')
        return inventory,manifest
    require(not candidate.get('quantizationRecipeId') and not candidate.get('tensorInventoryHash'),'QUANTIZATION_RECIPE_MISMATCH')
    if candidate.get('quantization','F16')=='F16':validate_recipe(inventory,quantization_recipe('F16'))
    else:require(inventory['encodings'].get('Q4_K',{}).get('elements',0)>0,'GGUF_ACTUAL_Q4_TENSORS_REQUIRED')
    return inventory,None

PROBES = ('def add(a, b):\n    return a + b\n','한글 설명과 Python 코드','  a\t b\r\n',
    '<|assistant|>literal<|end|>','<pad><bos><eos>','x = "é🙂"\n','def empty():\n    pass\n','a+b',
    'def context_probe(x):\n    return x + 1\n'*128)


def engine_fingerprint():
    require(importlib.util.find_spec('llama_cpp') is not None,'GGUF_RUNTIME_UNAVAILABLE',status=503)
    import llama_cpp
    from llama_cpp import llama_cpp as bindings
    root=Path(llama_cpp.__file__).resolve().parent
    files={str(path.relative_to(root)):file_hash(path) for path in root.rglob('*') if path.is_file()
        and any(part in path.name.lower() for part in ('.dll','.so','.dylib'))}
    runtime=Path(bindings._lib._name).resolve()
    require(runtime.is_file(),'GGUF_RUNTIME_BINARY_UNAVAILABLE')
    files['loaded-runtime/'+runtime.name]=file_hash(runtime)
    return {'engineVersion':importlib.metadata.version('llama-cpp-python'),'engineBinaryHash':digest(files),
        'engineBindingsHash':file_hash(Path(bindings.__file__))}


def verify_conformance(store,source,candidate):
    source=store.get('artifact',source) if isinstance(source,str) else source
    candidate=store.get('artifact',candidate) if isinstance(candidate,str) else candidate
    record=next((row for row in store.list('foundation_format_validation') if row.get('sourceArtifactId')==source['id']
        and row.get('artifactId')==candidate['id'] and row.get('sourceManifestHash')==source['manifestHash']
        and row.get('artifactManifestHash')==candidate['manifestHash'] and row.get('passed') is True
        and row.get('state')=='passed'),None)
    require(record,'GGUF_CONFORMANCE_REQUIRED',status=409)
    body={k:v for k,v in record.items() if k not in ('id','version','createdAt','updatedAt','validationHash')}
    require(record.get('validationHash')==digest(body) and record.get('tokenizerHash')==source.get('tokenizerHash'),
        'GGUF_CONFORMANCE_MISMATCH')
    if candidate.get('quantizationRecipeHash'):
        require(record.get('quantizationRecipeHash')==candidate['quantizationRecipeHash']
            and record.get('tensorInventoryHash')==candidate.get('tensorInventoryHash'),'GGUF_CONFORMANCE_MISMATCH')
    current=engine_fingerprint()
    require(all(record.get(key)==value for key,value in current.items()),'GGUF_ENGINE_CHANGED')
    return record


def preflight(control, source_id):
    source = control.store.get('artifact',source_id); root = control.registry.verify(source)
    reasons, hashes = [], {}
    if source.get('format') != 'safetensors': reasons.append('GGUF_FLOAT_SOURCE_REQUIRED')
    if not (root/'tokenizer.json').is_file(): reasons.append('TOKENIZER_REQUIRED')
    tools = control.store.get('setting','tools',{})
    from .gguf_runtime import dependency_receipt_hash
    dependency_hash=None
    try:dependency_hash=dependency_receipt_hash(tools)
    except Exception as error:
        from LLM.core import Fault
        if not isinstance(error,Fault):raise
        reasons.append(error.code)
    for field, hash_field in (('ggufConverter','converterHash'),('quantizer','quantizerHash')):
        path = Path(tools.get(field,''))
        if not path.is_file() or path.is_symlink(): reasons.append('QUANTIZATION_TOOLS_UNAVAILABLE'); continue
        actual = file_hash(path)
        if actual != tools.get(hash_field): reasons.append('TOOLCHAIN_HASH_MISMATCH')
        hashes[field] = actual
    if importlib.util.find_spec('llama_cpp') is None: reasons.append('GGUF_RUNTIME_UNAVAILABLE')
    return {'state':'held' if reasons else 'ready_for_experiment','reasonCodes':sorted(set(reasons)),
        'sourceArtifactId':source_id,'sourceManifestHash':source['manifestHash'],'toolHashes':hashes,
        'dependencyReceiptHash':dependency_hash,
        'supportedQuantizations':['F16','Q4_K_M'],'tokenizerCompatibilityMeasured':False,
        'defaultQuantizationRecipes':{value:quantization_recipe(value) for value in ('F16','Q4_K_M')},
        'servingEligible':False,'actualConformanceRequired':True,'qualityGateRequired':True}


def validate(control, source_id, gguf_id, body=None):
    import torch
    import torch.nn.functional as F
    body = body or {}
    require(importlib.util.find_spec('llama_cpp') is not None, 'GGUF_RUNTIME_UNAVAILABLE', status=503)
    toolchain=preflight(control,source_id)
    require(not toolchain['reasonCodes'],toolchain['reasonCodes'][0] if toolchain['reasonCodes'] else 'GGUF_TOOLCHAIN_UNAVAILABLE',status=503)
    from .gguf_runtime import open_engine,engine_options
    source = control.store.get('artifact',source_id)
    candidate = control.store.get('artifact',gguf_id)
    root = control.registry.verify(source); quantized_root = control.registry.verify(candidate)
    require(candidate.get('format') == 'gguf' and source_id in candidate.get('parents',[])
            and candidate.get('sourceArtifactId') == source_id, 'GGUF_SOURCE_MISMATCH')
    require(candidate.get('tokenizerHash') == source.get('tokenizerHash'), 'TOKENIZER_HASH_MISMATCH')
    inventory,recipe_manifest=bound_inventory(source,candidate,quantized_root)
    if recipe_manifest:
        require(recipe_manifest.get('toolHashes')==toolchain['toolHashes']
            and recipe_manifest.get('dependencyReceiptHash')==toolchain['dependencyReceiptHash'],'TOOLCHAIN_HASH_MISMATCH')
    files=[quantized_root/name for name in candidate['files'] if name.endswith('.gguf')]
    require(source.get('trainedContextTokens',0) >= 16, 'TRAINED_CONTEXT_REQUIRED')
    context = min(512,source['trainedContextTokens'])
    quantization = candidate.get('quantization','F16')
    require(quantization in ('F16','Q4_K_M'), 'UNSUPPORTED_QUANTIZATION')
    tolerance = .005 if quantization == 'F16' else .25
    max_error, divergences, agreements, rows = 0., [], [], []
    reasons=set();engine=None;failure=None
    kl_limit=.001 if quantization=='F16' else .05
    agreement_limit=.99 if quantization=='F16' else .90
    try:
        fp_model,native=control.foundation.runtime.load(source_id,purpose='evaluation')
        engine=open_engine(files[0],context)
        require(engine.token_bos()==1 and engine.token_eos()==2,'GGUF_SPECIAL_TOKEN_IDS_MISMATCH')
        with torch.inference_mode():
            for probe_index,text in enumerate(PROBES):
                expected = native.encode_ordinary(text)
                actual = engine.tokenize(text.encode('utf-8'),add_bos=False,special=False)
                require(actual == expected, 'GGUF_TOKENIZER_NOT_EQUIVALENT')
                ids = [1,*expected][:context if probe_index==len(PROBES)-1 else max(2,context-8)]
                fp = fp_model(input_ids=torch.tensor([ids]),use_cache=False).logits[0].float()
                engine.reset(); engine.eval(ids)
                q = torch.as_tensor(engine._scores[:len(ids)].copy(),dtype=torch.float32)
                require(q.shape == fp.shape and bool(torch.isfinite(q).all()) and bool(torch.isfinite(fp).all()), 'GGUF_LOGITS_INVALID')
                error = float((fp-q).abs().max()); max_error = max(max_error,error)
                kl = float(F.kl_div(F.log_softmax(q,dim=-1),F.softmax(fp,dim=-1),reduction='batchmean'))
                agreement = float((fp.argmax(-1) == q.argmax(-1)).float().mean())
                divergences.append(max(0.,kl)); agreements.append(agreement)
                if error>tolerance or kl>kl_limit:reasons.add('GGUF_LOGITS_NOT_EQUIVALENT')
                # Run each engine's actual greedy loop and compare stop behavior.
                outputs, stops = [], []
                for backend in ('native','gguf'):
                    sequence, generated, stopped = list(ids), [], False
                    for _ in range(min(8,context-len(sequence))):
                        if backend == 'native': logits = fp_model(input_ids=torch.tensor([sequence]),use_cache=False).logits[0,-1]
                        else:
                            engine.reset(); engine.eval(sequence); logits = torch.as_tensor(engine._scores[len(sequence)-1].copy())
                        next_id = int(logits.argmax()); generated.append(next_id); sequence.append(next_id)
                        if next_id in (2,7): stopped=True; break
                    outputs.append(generated); stops.append(stopped)
                if stops[0]!=stops[1]:reasons.add('GGUF_TERMINATION_NOT_EQUIVALENT')
                rows.append({'promptHash':digest(text.encode()),'tokenCount':len(ids),'maxAbsoluteLogitError':error,
                    'klDivergence':max(0.,kl),'argmaxAgreement':agreement,'boundedGeneration':True,
                    'terminationEquivalent':stops[0]==stops[1],'nativeOutputHash':digest(outputs[0]),'ggufOutputHash':digest(outputs[1])})
    except Exception as error:
        failure=error;reasons.add(getattr(error,'code','GGUF_CONFORMANCE_FAILED'))
    finally:
        if engine is not None:engine.close()
        control.foundation.runtime.expire(force=True)
    mean_agreement=sum(agreements)/len(agreements) if agreements else None
    if mean_agreement is None or mean_agreement<agreement_limit:reasons.add('GGUF_ARGMAX_AGREEMENT_LOW')
    if len(rows)!=len(PROBES):reasons.add('GGUF_PROBES_INCOMPLETE')
    passed=not reasons
    from .gguf_runtime import engine_identity
    record = {'projectId':source.get('foundationProjectId',source.get('projectId')),
        'sourceArtifactId':source_id,'sourceManifestHash':source['manifestHash'],
        'artifactId':gguf_id,'artifactManifestHash':candidate['manifestHash'],'tokenizerHash':source['tokenizerHash'],
        **engine_fingerprint(),'implementationVersion':'seed-gguf-conformance-v1','quantization':quantization,'state':'passed' if passed else 'failed',
        'engineIdentity':engine_identity(),'engineOptions':engine_options(context),
        'toolHashes':toolchain['toolHashes'],'dependencyReceiptHash':toolchain['dependencyReceiptHash'],'actualContextTokens':context,
        'stopTokenIds':[2,7],'assemblerVersion':'seed-roles-v1','fixture':bool(source.get('fixture') or source.get('testOnly')),
        'executionScope':source.get('executionScope') or getattr(control,'execution_scope','nas'),
        'passed':passed,'reasonCodes':sorted(reasons),'numericConformanceOnly':True,'rows':rows,'maxAbsoluteLogitError':max_error if agreements else None,
        'meanKlDivergence':sum(divergences)/len(divergences) if divergences else None,'meanArgmaxAgreement':mean_agreement,
        'thresholds':{'maxAbsoluteLogitError':tolerance,'perProbeKlDivergence':kl_limit,'meanArgmaxAgreement':agreement_limit,'terminationEquivalent':True},
        'quantizationRecipeId':candidate.get('quantizationRecipeId'),'quantizationRecipeHash':candidate.get('quantizationRecipeHash'),
        'tensorInventoryHash':digest(inventory),'actualTensorEncodings':inventory['encodings'],'ggufBytes':inventory['fileBytes'],
        'sourceFloatGgufBytes':recipe_manifest.get('sourceFloatGgufBytes') if recipe_manifest else None,
        'compressionRatioVsF16Gguf':candidate.get('compressionRatioVsF16Gguf'),
        'qualityGateRequired':True,'servingEligible':False,'productionCertified':False}
    record['validationHash'] = digest(record)
    measured=control.store.put('foundation_format_validation',record,digest([source['manifestHash'],candidate['manifestHash'],record['engineVersion']]))
    if not passed:
        code=getattr(failure,'code',None) or ('GGUF_LOGITS_NOT_EQUIVALENT' if 'GGUF_LOGITS_NOT_EQUIVALENT' in reasons else
            'GGUF_TERMINATION_NOT_EQUIVALENT' if 'GGUF_TERMINATION_NOT_EQUIVALENT' in reasons else 'GGUF_ARGMAX_AGREEMENT_LOW')
        raise Fault(code,status=409,details={'validationId':measured['id'],'reasonCodes':sorted(reasons),
            'maxAbsoluteLogitError':record['maxAbsoluteLogitError'],'meanKlDivergence':record['meanKlDivergence'],
            'meanArgmaxAgreement':mean_agreement,'thresholds':record['thresholds']}) from failure
    return measured
