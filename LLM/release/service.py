import json
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from LLM.core import require, digest, file_hash, atomic_json, uid, now

class Releases:
    def __init__(self,store,registry):
        self.store,self.registry=store,registry;self.rag=None;self.inference=None
        self.activation_lock=threading.RLock()

    def package(self,job):
        body=job['payload'];artifact=self.store.get('artifact',body['artifactId']);root=self.registry.verify(artifact)
        require(not artifact.get('comparisonOnly'),'COMPARISON_ONLY_MODEL',status=409)
        tools=self.store.get('setting','tools',{})
        converter=Path(tools.get('ggufConverter',''));quantizer=Path(tools.get('quantizer',''))
        require(converter.is_file() and quantizer.is_file(),'QUANTIZATION_TOOLS_UNAVAILABLE',status=503)
        require(file_hash(converter)==tools.get('converterHash') and file_hash(quantizer)==tools.get('quantizerHash'),'TOOLCHAIN_HASH_MISMATCH')
        stage=self.store.root/'staging'/('quantized-'+uid());stage.mkdir()
        fp=stage/'float.gguf';q=stage/'model.gguf'
        commands=[[sys.executable,str(converter),str(root),'--outfile',str(fp),'--outtype','f16'],[str(quantizer),str(fp),str(q),'Q4_K_M','1']]
        for i,command in enumerate(commands):
            self.store.assert_lease(job)
            result=subprocess.run(command,capture_output=True,timeout=7200)
            require(result.returncode==0,'QUANTIZATION_FAILED',status=503,stage=i)
            self.store.progress(job,'converting' if i==0 else 'quantizing',i+1,2,'stages')
        require(q.is_file() and q.stat().st_size>16,'QUANTIZATION_FAILED')
        fp.unlink()
        from LLM.merging.service import copy_metadata
        copy_metadata(root,stage)
        return self.registry.publish(stage,kind='quantized',family_id=artifact['familyId'],parents=[artifact['id']],license_refs=artifact['licenseRefs'],
            metadata={'parameterCount':artifact['parameterCount'],'quantization':'Q4_K_M','sourceArtifactId':artifact['id']})

    def prepare(self,body):
        if self.rag:self.rag.refresh_policy()
        artifact=self.store.get('artifact',body['artifactId']);root=self.registry.verify(artifact)
        require(not artifact.get('comparisonOnly'),'COMPARISON_ONLY_MODEL',status=409)
        certificate=self.store.get('certificate',body['certificateId'])
        if artifact.get('origin')=='random_init' or artifact.get('foundationProjectId'):
            require(certificate.get('purpose')=='usable_release' and certificate.get('usable') is True
                    and not certificate.get('bootstrap') and not certificate.get('baselineOnly')
                    and certificate.get('approvedModes') and certificate.get('architectureHashV2')==artifact.get('architectureHashV2'),
                    'FOUNDATION_USABLE_CERTIFICATE_REQUIRED',status=409)
            release=next((r for r in self.store.list('release') if r.get('artifactId')==artifact['id'] and r.get('certificateId')==certificate['id'] and r.get('profile',{}).get('provider')=='foundation'),None)
            require(release,'FOUNDATION_RELEASE_PIPELINE_REQUIRED',status=409)
            return release
        require(certificate['artifactId']==artifact['id'] and certificate['passed'],'RELEASE_NOT_CERTIFIED',status=409)
        require(not artifact.get('fixture'),'FIXTURE_NOT_FOR_PRODUCTION',status=409)
        require((root/'tokenizer.json').is_file(),'TOKENIZER_REQUIRED')
        identity='release-'+digest([artifact['manifestHash'],certificate['id']])[:24]
        alias='seed/'+identity
        engine_digest=artifact['manifestHash']
        if artifact['format']=='gguf':
            require(self.inference is not None,'RUNTIME_BACKEND_REQUIRED')
            engine_digest=self.inference.ollama_register(artifact)['digest']
        # Native HF facade uses the immutable artifact manifest digest.
        profile={'model':alias,'modelRevision':engine_digest,'tokenizerRevision':artifact.get('sourceRevision') or artifact['tokenizerHash'],
            'tokenizerSha256':file_hash(root/'tokenizer.json'),'templateFormat':'chatml','contextLength':int(body.get('contextLength',4096)),'addSpecialTokens':False}
        destination=self.store.root/'releases'/identity;destination.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/'tokenizer.json',destination/'tokenizer.json');atomic_json(destination/'generation-profile.json',profile)
        release=self.store.put('release',{'artifactId':artifact['id'],'certificateId':certificate['id'],'state':'ready','engineAlias':alias,
            'engineDigest':engine_digest,'profile':profile,'tokenizerPath':str(destination/'tokenizer.json')},identity)
        source=artifact.get('sourceArtifactId',artifact['id'])
        for combo in self.store.list('combination'):
            if combo.get('outputArtifactId')==source:self.store.put('combination',{'servingArtifactId':artifact['id'],'releaseId':identity,'state':'ready'},combo['id'])
        return release

    def activate(self,release_id,expected=None):
        with self.activation_lock:return self._activate(release_id,expected)

    def _activate(self,release_id,expected=None):
        require(self.rag is not None,'RAG_DEPLOYMENT_GATEWAY_REQUIRED',status=503)
        self.rag.refresh_policy()
        release=self.store.get('release',release_id)
        require(release['state'] in ('ready','retired','active'),'RELEASE_NOT_READY',status=409)
        self.registry.verify(self.store.get('artifact',release['artifactId']))
        certificate=self.store.get('certificate',release['certificateId']);require(certificate['passed'],'RELEASE_NOT_CERTIFIED',status=409)
        artifact=self.store.get('artifact',release['artifactId'])
        require(not artifact.get('comparisonOnly'),'COMPARISON_ONLY_MODEL',status=409)
        if artifact.get('origin')=='random_init' or artifact.get('foundationProjectId'):
            require(certificate.get('purpose')=='usable_release' and certificate.get('usable') is True
                    and not certificate.get('bootstrap') and not certificate.get('baselineOnly')
                    and certificate.get('profileHash')==release.get('profileHash')
                    and certificate.get('architectureHashV2')==artifact.get('architectureHashV2')
                    and certificate.get('approvedModes'),'FOUNDATION_USABLE_CERTIFICATE_REQUIRED',status=409)
        previous=self.store.get('slot','champion',{})
        if expected is not None:require(previous.get('version',0)==expected,'VERSION_CONFLICT',status=412)
        if previous.get('activeReleaseId')==release_id:return previous
        self.store.put('deployment',{'releaseId':release_id,'previousReleaseId':previous.get('activeReleaseId'),'state':'preparing'},release_id)
        acknowledgment=self.rag.activate_release(release)
        require(acknowledgment.get('activeReleaseId')==release_id,'DEPLOYMENT_RESULT_UNKNOWN',status=503)
        with self.store.tx():
            slot=self.store.put('slot',{'activeReleaseId':release_id,'previousReleaseId':previous.get('activeReleaseId')},'champion',expected=previous.get('version',0))
            if previous.get('activeReleaseId'):self.store.put('release',{'state':'retired'},previous['activeReleaseId'])
            self.store.put('release',{'state':'active','activatedAt':now()},release_id)
            if self.store.get('foundation_release',release_id,{}):
                self.store.put('foundation_release',{'state':'active','activatedAt':now()},release_id)
                self.store.put('foundation_project',{'stage':'usable'},artifact['foundationProjectId'])
            self.store.put('deployment',{'state':'active'},release_id)
        return slot

    def rollback(self,expected=None):
        slot=self.store.get('slot','champion');require(slot.get('previousReleaseId'),'NO_ROLLBACK_RELEASE',status=409)
        if expected is not None:require(slot['version']==expected,'VERSION_CONFLICT',status=412)
        return self.activate(slot['previousReleaseId'],slot['version'])

    def reconcile(self):
        with self.activation_lock:return self._reconcile()

    def _reconcile(self):
        if not self.rag:return {'state':'disconnected'}
        active=self.rag.active_release()
        slot=self.store.get('slot','champion',{})
        if not active:return {'state':'unconfigured' if not slot else 'needs_review','reasonCodes':[] if not slot else ['RAG_ACTIVE_RELEASE_MISSING']}
        release=self.store.get('release',active,{})
        require(release,'RELEASE_MISSING_FROM_REGISTRY',status=503)
        self.registry.verify(self.store.get('artifact',release['artifactId']))
        if slot.get('activeReleaseId')!=active:
            self.store.put('slot',{'activeReleaseId':active,'previousReleaseId':slot.get('activeReleaseId')},'champion',expected=slot.get('version',0))
            self.store.put('release',{'state':'active'},active)
            self.store.put('deployment',{'state':'active','reconciledAt':now()},active)
        return {'state':'consistent','activeReleaseId':active}
