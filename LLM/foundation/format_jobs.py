"""Lease-fenced float/GGUF conversion followed by actual numeric conformance."""
from __future__ import annotations
import json
import os
import subprocess
import sys
import time

from LLM.core import Fault,atomic_json,digest,file_hash,inside,require,uid


def run(control,job):
    body=job['payload'];store=control.store
    def guard():
        current=store.assert_lease(job)
        require(current['state'] not in ('cancelled','cancelling','pause_requested','paused'),'JOB_CANCELLED')
        control.foundation.validate_policy(control.foundation._project(body['projectId']))
        return current
    guard()
    source=store.get('artifact',body.get('artifactId') or '',{})
    if not source:
        checkpoint=store.get('foundation_checkpoint',body.get('checkpointId') or '')
        require(checkpoint.get('projectId')==body['projectId']
                and checkpoint.get('completedOptimizerSteps',0)>0,'NO_EFFECTIVE_TRAINING')
        from .graduation import _publish_base
        source=_publish_base(store,store.get('foundation_project',body['projectId']),checkpoint,control.registry)
    require(source.get('foundationProjectId',source.get('projectId'))==body['projectId'],
            'INCOMPATIBLE_BASE')
    root=control.registry.verify(source)
    from .formats import preflight,validate,quantization_recipe,inspect_gguf,validate_recipe
    toolchain=preflight(control,source['id'])
    require(not toolchain['reasonCodes'],toolchain['reasonCodes'][0] if toolchain['reasonCodes'] else 'GGUF_TOOLCHAIN_UNAVAILABLE')
    quantization=body.get('quantization','F16')
    require(quantization in ('F16','Q4_K_M'),'UNSUPPORTED_QUANTIZATION')
    recipe=quantization_recipe(quantization)
    tools=store.get('setting','tools')
    stage=inside(store.root,'staging/foundation-format-'+uid());stage.mkdir(parents=True)
    fp=stage/'float.gguf';output=stage/'model.gguf'
    commands=[[sys.executable,tools['ggufConverter'],str(root),'--outfile',str(fp),'--outtype','f16']]
    if quantization=='Q4_K_M':
        overrides=['--token-embedding-type','f16','--output-tensor-type','f16']
        for name,encoding in recipe['tensorOverrides'].items():overrides.extend(['--tensor-type',name+'='+encoding.lower()])
        commands.append([tools['quantizer'],*overrides,str(fp),str(output),'Q4_K_M','1'])
    logs=inside(store.root,'foundation/job-results/'+job['id']);logs.mkdir(parents=True,exist_ok=True)
    for index,command in enumerate(commands):
        guard()
        if index==0:
            # -B alone still reads old pyc files. A fresh interpreter prefix
            # makes conversion compile the hash-verified source tree itself.
            cache_prefix=inside(store.root,'foundation/job-results/'+job['id']+'/pycache-'+uid())
            command=[command[0],'-B','-X','pycache_prefix='+str(cache_prefix),*command[1:]]
        with (logs/('format-stage-'+str(index)+'.log')).open('wb') as log:
            env=os.environ.copy()
            if tools.get('toolchainReceiptHash'):env['SEED_GGUF_TOOLCHAIN_HASH']=tools['toolchainReceiptHash']
            process=subprocess.Popen(command,stdout=log,stderr=log,
                env=env,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            began=time.monotonic()
            try:
                while process.poll() is None:
                    guard()
                    require(time.monotonic()-began<1800,'FORMAT_TIME_LIMIT')
                    require(log.tell()<16*1024**2,'FORMAT_OUTPUT_LIMIT')
                    time.sleep(.25)
                require(process.wait(timeout=10)==0,'QUANTIZATION_FAILED',stage=index)
            finally:
                if process.poll() is None:
                    import psutil
                    try:
                        parent=psutil.Process(process.pid)
                        for child in parent.children(recursive=True):
                            try:child.kill()
                            except psutil.NoSuchProcess:pass
                    except psutil.NoSuchProcess:pass
                    process.kill();process.wait(timeout=10)
        store.progress(job,'converting' if index==0 else 'quantizing',index+1,len(commands),'stages')
    float_inventory=validate_recipe(inspect_gguf(fp),quantization_recipe('F16'))
    if quantization=='F16':fp.replace(output)
    elif fp.exists():fp.unlink()
    require(output.is_file() and output.stat().st_size>16,'GGUF_FILE_INVALID')
    configuration=json.loads((root/'config.json').read_text('utf-8'))
    inventory=validate_recipe(inspect_gguf(output),recipe,source_tied_embeddings=configuration.get('tie_word_embeddings') is True)
    if quantization=='Q4_K_M':require(inventory['fileBytes']<float_inventory['fileBytes'],'GGUF_COMPRESSION_REQUIRED')
    manifest={'schemaVersion':1,'recipe':recipe,'sourceArtifactId':source['id'],'sourceManifestHash':source['manifestHash'],
        'sourceFloatGgufSha256':float_inventory['fileSha256'],'sourceFloatGgufBytes':float_inventory['fileBytes'],
        'sourceTiedEmbeddings':configuration.get('tie_word_embeddings') is True,'outputGgufSha256':inventory['fileSha256'],
        'tensorInventoryHash':digest(inventory),'toolHashes':toolchain['toolHashes'],'dependencyReceiptHash':toolchain['dependencyReceiptHash']}
    atomic_json(stage/'quantization-recipe.json',manifest);atomic_json(stage/'tensor-inventory.json',inventory)
    from LLM.merging.service import copy_metadata
    copy_metadata(root,stage)
    guard();control.registry.verify(source)
    latest_tools=preflight(control,source['id'])
    require(latest_tools==toolchain,'TOOLCHAIN_HASH_MISMATCH')
    candidate=control.registry.publish(stage,kind='quantized',family_id=source['familyId'],
        parents=[source['id']],license_refs=source.get('licenseRefs',[]),metadata={
            'parameterCount':source['parameterCount'],'sourceArtifactId':source['id'],
            'quantization':quantization,'quantizationRecipeId':recipe['id'],'quantizationRecipeHash':digest(manifest),
            'tensorInventoryHash':digest(inventory),'actualTensorEncodings':inventory['encodings'],
            'sourceFloatGgufBytes':float_inventory['fileBytes'],'ggufBytes':inventory['fileBytes'],
            'compressionRatioVsF16Gguf':float_inventory['fileBytes']/inventory['fileBytes'],
            'precisionDescription':recipe['description'],'servingEligible':False,'numericConformanceRequired':True})
    try:
        validation=validate(control,source['id'],candidate['id'])
        guard();control.registry.verify(source)
    except Exception as error:
        store.put('artifact',{'state':'quarantined','formatReasonCodes':[getattr(error,'code','FORMAT_CONFORMANCE_FAILED')],
            'formatValidationId':getattr(error,'details',{}).get('validationId')},candidate['id'])
        raise
    return {'artifactId':candidate['id'],'sourceArtifactId':source['id'],
            'validationId':validation['id'],'numericConformancePassed':True,'quantization':quantization,
            'quantizationRecipeId':recipe['id'],'quantizationRecipeHash':digest(manifest),'actualTensorEncodings':inventory['encodings'],
            'ggufBytes':inventory['fileBytes'],'sourceFloatGgufBytes':float_inventory['fileBytes'],
            'compressionRatioVsF16Gguf':float_inventory['fileBytes']/inventory['fileBytes'],
            'servingEligible':False,'qualityGateRequired':True,'nasModified':False}
