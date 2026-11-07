"""Evidence-bound clean-ancestor recovery; this does not claim unlearning."""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import time

from LLM.core import Fault, digest, file_hash, inside, now, require, uid, atomic_json,canonical

FIELDS={'sourceIds':'sourceId','documentIds':'documentId','versionIds':'versionId'}


def _selectors(value):
    require(isinstance(value,dict) and not set(value)-set(FIELDS)-{'contentHashes','allProjectData','teacherArtifactIds'},'RELEARNING_SELECTORS_INVALID')
    result={}
    for key,items in value.items():
        if key=='allProjectData':require(isinstance(items,bool),'RELEARNING_SELECTORS_INVALID');result[key]=items;continue
        require(isinstance(items,list) and all(isinstance(item,str) and item for item in items),'RELEARNING_SELECTORS_INVALID')
        result[key]=sorted(set(items))
    return result


def _manifest(store,record):
    path=inside(store.root,record['storagePath'])/'manifest.json'
    require(path.is_file() and not path.is_symlink() and file_hash(path)==record.get('manifestHash'),'RELEARNING_EVIDENCE_INVALID')
    value=json.loads(path.read_text('utf-8'))
    for name,checksum in value.get('files',{}).items():
        target=inside(path.parent,name)
        require(target.is_file() and not target.is_symlink() and file_hash(target)==checksum,'RELEARNING_EVIDENCE_INVALID')
    return value


def _matches(member,selected):
    if selected.get('allProjectData') or member.get('hash') in selected.get('contentHashes',[]):return True
    teachers=set(selected.get('teacherArtifactIds',[]))
    if teachers and any((link.get('teacherArtifactId') in teachers or set(link.get('teacherArtifactIds') or [])&teachers)
                        for link in (member,*member.get('lineage',[]))):return True
    return any(link.get(field) in selected.get(key,[]) for link in member.get('lineage',[]) for key,field in FIELDS.items())


def filter_records(control,project,records):
    """Keep operator exclusions permanent across every later raw/SFT refresh."""
    if not project.get('relearningPlanId'):return records
    value=control.store.get('foundation_relearning_plan',project['relearningPlanId'])
    require(value.get('newProjectId')==project['id'] and value.get('proofHash')==project.get('relearningProofHash'),
            'RELEARNING_EVIDENCE_INVALID')
    result=[]
    for row in records:
        member={**row,'hash':digest(row.get('text','').encode()),'lineage':[row,*row.get('lineage',[])]}
        if not _matches(member,value['selectors']):result.append(row)
    return result


def _corpus_record(store,identity):
    record=store.get('foundation_corpus',identity,{}) or store.get('foundation_teacher_corpus',identity,{})
    require(record,'RELEARNING_LINEAGE_INCOMPLETE')
    return record


def _members(store,record):
    from . import corpus
    iterator=getattr(corpus,'iter_members',None)
    if callable(iterator):
        # Private provenance inspection never changes the persisted policy state.
        manifest=_manifest(store,record)
        require('members' in manifest or manifest.get('membersFile'),'RELEARNING_LINEAGE_INCOMPLETE')
        yield from iterator(store,record,allow_revoked=True)
    else:
        manifest=_manifest(store,record)
        require('members' in manifest,'CORPUS_MEMBER_READER_REQUIRED')
        yield from manifest['members']


def _authority(control,selected,corpora):
    bridge=getattr(control,'bridge',None)
    require(bridge is not None,'RAG_POLICY_GATEWAY_REQUIRED',status=503)
    result={key:list(value) if isinstance(value,list) else value for key,value in selected.items()}
    invalid=set(result.get('versionIds',[]))
    for corpus in corpora:
        for member in _members(control.store,corpus):
            for link in member.get('lineage',[]):
                source=bridge.store.get('source',link.get('sourceId',''),{})
                doc=bridge.store.get('document',link.get('documentId',''),{})
                version=bridge.store.get('version',link.get('versionId',''),{})
                if (source.get('trainingPolicy')!='allowed' or source.get('upstreamDeleted') is True or not doc
                    or doc.get('deletedAt') is not None or not doc.get('enabled',True) or doc.get('trainingPolicy')=='denied'
                    or not version or version.get('qualityStatus')!='passed'):
                    if link.get('versionId'):invalid.add(link['versionId'])
    result['versionIds']=sorted(invalid)
    return result,bridge.store.setting('policyVersion',0)


def plan(control,project_id,selectors=None):
    store=control.store;project=store.get('foundation_project',project_id)
    selected=_selectors(selectors or {})
    corpora=list({row['id']:row for kind in ('foundation_corpus','foundation_teacher_corpus') for row in store.list(kind)
                 if row.get('projectId')==project_id and row.get('storagePath') and row.get('state') not in ('deleted','archived')}.values())
    if selected.pop('allProjectData',False):
        versions=set(selected.get('versionIds',[]))
        for corpus in corpora:
            for member in _members(store,corpus):
                versions.update(link['versionId'] for link in member.get('lineage',[]) if link.get('versionId'))
        selected['versionIds']=sorted(versions)
    selected,epoch=_authority(control,selected,corpora)
    require(any(value for value in selected.values()),'RELEARNING_CAUSE_REQUIRED')
    by_corpus={row['id']:row for row in corpora}
    token=store.get('foundation_tokenizer',project.get('tokenizerId') or '',{})
    token_corpus=by_corpus.get(token.get('corpusId'))
    token_tainted=not token or token_corpus is None
    evidence={}
    if token_corpus:
        token_tainted=any(member.get('split')=='train' and _matches(member,selected) for member in _members(store,token_corpus))
        evidence[token_corpus['id']]=token_corpus['manifestHash']
    checkpoints={row['id']:row for row in store.list('foundation_checkpoint') if row.get('projectId')==project_id}
    genesis=store.get('foundation_genesis',project.get('genesisId') or '',{})
    memo={genesis.get('id'):bool(token_tainted)};visiting=set()
    def tainted(identity):
        if identity in memo:return memo[identity]
        require(identity in checkpoints and identity not in visiting,'RELEARNING_ANCESTRY_INCOMPLETE')
        visiting.add(identity);row=checkpoints[identity]
        prior=row.get('parentCheckpointId');bad=not prior or tainted(prior)
        teachers=set(row.get('teacherArtifactIds') or [])|{row.get('teacherArtifactId')}
        bad |= bool(teachers&set(selected.get('teacherArtifactIds',[])))
        corpus=by_corpus.get(row.get('corpusId'))
        if corpus is None:bad=True
        else:
            _manifest(store,corpus);evidence[corpus['id']]=corpus['manifestHash']
            hashes=set(row.get('consumedContentHashes',[]))
            bad |= any(member.get('split')=='train' and (not hashes or member.get('hash') in hashes)
                       and _matches(member,selected) for member in _members(store,corpus))
        visiting.remove(identity);memo[identity]=bool(bad);return bool(bad)
    pointer=project.get('latestCheckpointId');chain=[];clean=None
    while pointer:
        require(pointer not in chain,'RELEARNING_ANCESTRY_INCOMPLETE');chain.append(pointer)
        row=checkpoints.get(pointer)
        if not row:break
        if not tainted(pointer) and row.get('state') in ('ready','verified','completed'):
            _manifest(store,row);clean=row;break
        pointer=row.get('parentCheckpointId')
    value={'projectId':project_id,'state':'planned','selectors':selected,'policyEpoch':epoch,'tokenizerTainted':bool(token_tainted),
           'strategy':'clean_ancestor' if clean and not token_tainted else 'new_genesis','cleanAncestorCheckpointId':clean['id'] if clean else None,
           'sourceTokenizerId':token.get('id'),'sourceGenesisId':genesis.get('id'),'corpusEvidence':evidence,
           'cleanCheckpointManifestHash':clean.get('manifestHash') if clean else None,'tokenizerManifestHash':token.get('manifestHash'),
           'mathematicalUnlearning':False,'automaticAuthorized':store.get('foundation_policy',project_id,{}).get('enabled') is True,
           'reasonCodes':['TOKENIZER_REBUILD_REQUIRED'] if token_tainted else [],'observedAt':now()}
    value['proofHash']=digest({k:v for k,v in value.items() if k not in ('state','observedAt')})
    return store.put('foundation_relearning_plan',value)


def _current(control,value):
    store=control.store
    corpora=[_corpus_record(store,identity) for identity in value['corpusEvidence']]
    for corpus in corpora:require(corpus.get('manifestHash')==value['corpusEvidence'][corpus['id']],'RELEARNING_EVIDENCE_CHANGED')
    selected,epoch=_authority(control,value['selectors'],corpora)
    require(selected==value['selectors'],'RELEARNING_POLICY_CHANGED',status=409)
    return epoch


def start(control,plan_id):
    store=control.store;value=store.get('foundation_relearning_plan',plan_id)
    if value.get('newProjectId'):return value
    require(value['state']=='planned','RELEARNING_NOT_PLANNED');_current(control,value)
    old=store.get('foundation_project',value['projectId']);policy=store.get('foundation_policy',old['id'],{})
    allowed=[identity for identity in policy.get('allowedSourceIds',[]) if identity not in value['selectors'].get('sourceIds',[])
             and control.bridge.store.get('source',identity,{}).get('trainingPolicy')=='allowed']
    require(allowed,'NO_CLEAN_TRAINING_SOURCES',status=409)
    body={'name':old.get('name','foundation')[:54]+' / clean rebuild','architecture':old['config'],'seed':old.get('seed',42),'automation':policy.get('automation','always_on_idle')}
    exported=control.bridge.foundation_corpus(old['id'],None,allowed)
    versions=[row.get('versionId',row.get('id')) for row in exported.get('records',[]) if
              not _matches({'hash':digest(row.get('text','').encode()),'lineage':[row]},value['selectors'])]
    require(versions,'NO_CLEAN_TRAINING_SOURCES',status=409)
    project=control.foundation.create_project(body,trusted_test=old.get('testOnly') is True,
                                             trusted_growth_receipt=old.get('trustedGrowthReceipt'))
    # Preserve the same authorization envelope; exclusions never expand source access.
    store.put('foundation_policy',{**policy,'allowedSourceIds':allowed,'enabled':False},project['id'])
    store.put('foundation_project',{'relearningPlanId':plan_id,'relearningFromProjectId':old['id'],'relearningProofHash':value['proofHash']},project['id'])
    value=store.put('foundation_relearning_plan',{'newProjectId':project['id'],'state':'preparing_corpus','selectedVersionIds':versions},plan_id)
    job=control.foundation.create_corpus(project['id'],{'sourceVersionIds':versions})
    return store.put('foundation_relearning_plan',{'activeJobId':job['id']},plan_id)


def _provenance_corpus(store,value,project,*,tokenizer_only=False):
    """Keep every actually inherited source visible to future RAG withdrawal audits."""
    token=store.get('foundation_tokenizer',value['sourceTokenizerId']);selected={token['corpusId']:None}
    pointer={} if tokenizer_only else store.get('foundation_checkpoint',value['cleanAncestorCheckpointId'])
    while pointer:
        identity=pointer.get('corpusId');hashes=set(pointer.get('consumedContentHashes',[]))
        if identity and identity not in selected:selected[identity]=hashes or None
        elif identity and selected[identity] is not None:selected[identity]|=hashes
        prior=pointer.get('parentCheckpointId');pointer=store.get('foundation_checkpoint',prior,{}) if prior else {}
    members={}
    for identity,hashes in selected.items():
        for member in _members(store,_corpus_record(store,identity)):
            if member.get('split')!='train' or (hashes is not None and member.get('hash') not in hashes):continue
            require(not _matches(member,value['selectors']),'RELEARNING_EVIDENCE_INVALID')
            members[digest(member)]=member
    require(members,'RELEARNING_LINEAGE_INCOMPLETE')
    identity='clean-provenance-'+digest([value['id'],tokenizer_only,sorted(members)])[:24]
    root=inside(store.root,'foundation/corpora/'+identity);root.mkdir(parents=True,exist_ok=True)
    content=''.join(canonical(members[key])+'\n' for key in sorted(members))
    path=root/'members.jsonl'
    if path.exists():require(path.read_text('utf-8')==content,'RELEARNING_EVIDENCE_CHANGED')
    else:path.write_text(content,encoding='utf-8',newline='\n')
    manifest={'id':identity,'schemaVersion':2,'kind':'foundation_corpus','projectId':project['id'],'membersFile':'members.jsonl',
              'files':{'members.jsonl':file_hash(path)},'splitShards':{key:[] for key in ('train','dev','release','audit')},
              'objective':'provenance_only','sourceCorpusHashes':value['corpusEvidence'],'relearningProofHash':value['proofHash']}
    atomic_json(root/'manifest.json',manifest)
    return store.put('foundation_corpus',{'projectId':project['id'],'state':'verified','storagePath':root.relative_to(store.root).as_posix(),
                 'manifestHash':file_hash(root/'manifest.json'),'relearningProvenanceOnly':True,'relearningPlanId':value['id'],
                 'objective':'provenance_only','modelGeneratedTargetRatio':0},identity)


def execute(control,job):
    """Trusted heavy-job handler. Only the recorded, hash-bound clean base is reused."""
    from . import architecture,checkpoint,tokenizer
    store=control.store;value=store.get('foundation_relearning_plan',job['payload']['planId']);_current(control,value)
    require(value['strategy']=='clean_ancestor' and value.get('newProjectId')==job['payload']['projectId'],'RELEARNING_EVIDENCE_INVALID')
    project=store.get('foundation_project',value['newProjectId']);store.assert_lease(job)
    original=store.get('foundation_tokenizer',value['sourceTokenizerId'])
    require(original.get('manifestHash')==value['tokenizerManifestHash'],'RELEARNING_EVIDENCE_CHANGED');_manifest(store,original)
    identity='clean-tokenizer-'+digest([value['id'],original['id']])[:24]
    root=inside(store.root,'foundation/tokenizers/'+identity)
    if not root.exists():shutil.copytree(inside(store.root,original['storagePath']),root)
    provenance=_provenance_corpus(store,value,project,tokenizer_only=True)
    weight_provenance=_provenance_corpus(store,value,project)
    token=store.put('foundation_tokenizer',{**{k:v for k,v in original.items() if k not in ('id','createdAt','updatedAt','version')},
        'projectId':project['id'],'storagePath':root.relative_to(store.root).as_posix(),'state':'ready',
        'corpusId':provenance['id'],'sourceTrainingCorpusId':original['corpusId'],
        'sourceCleanTokenizerId':original['id'],'relearningPlanId':value['id'],'relearningProofHash':value['proofHash']},identity)
    tokenizer.load(store,token)
    old_genesis=store.get('foundation_genesis',value['sourceGenesisId']);_manifest(store,old_genesis)
    model=architecture.load_model(store,{**old_genesis,'state':'ready'})
    genesis=checkpoint.save(store,model,old_genesis['config'],token,{'projectId':project['id'],'testOnly':project.get('testOnly',False),
        'sourceCleanGenesisId':old_genesis['id'],'initialTensorDigest':old_genesis.get('initialTensorDigest'),
        'origin':'random_init','relearningProofHash':value['proofHash']},job=job,record_kind='foundation_genesis')
    old=store.get('foundation_checkpoint',value['cleanAncestorCheckpointId'])
    require(old.get('manifestHash')==value['cleanCheckpointManifestHash'],'RELEARNING_EVIDENCE_CHANGED');_manifest(store,old)
    model=architecture.load_model(store,{**old,'state':'ready'})
    ranges={};pointer=old
    while pointer:
        for checksum,spans in pointer.get('consumedTargetRanges',{}).items():ranges.setdefault(checksum,[]).extend(spans)
        parent_id=pointer.get('parentCheckpointId')
        pointer=store.get('foundation_checkpoint',parent_id,{}) if parent_id else {}
    inherited_hashes=sorted({member['hash'] for member in _members(store,weight_provenance)})
    state={k:old[k] for k in ('completedOptimizerSteps','committedTargetTokens','trainedContextTokens','trainedTensorDigest','initialTensorDigest','consumedContentHashes','consumedTargetRanges') if k in old}
    state.update(projectId=project['id'],parentCheckpointId=genesis['id'],corpusId=weight_provenance['id'],
                 sourceCleanCheckpointId=old['id'],relearningProofHash=value['proofHash'],testOnly=project.get('testOnly',False),
                 optimizerReset=True,origin='random_init',consumedContentHashes=inherited_hashes,consumedTargetRanges=ranges)
    clone=checkpoint.save(store,model,old['config'],token,state,job=job)
    # Inherited clean learning is replay history, not new data in the new namespace.
    store.put('foundation_consumption',{'projectId':project['id'],'runId':None,'tokenizerHash':token['tokenizerHash'],
              'checkpointId':clone['id'],'consumedTargetRanges':ranges,'consumedContentHashes':sorted(ranges),
              'targetNamespace':'causal_lm','newUniqueTokens':0,'replayTargetTokens':0,'inheritedCleanHistory':True},'relearn-'+value['id'])
    store.assert_lease(job)
    store.put('foundation_project',{'tokenizerId':token['id'],'genesisId':genesis['id'],'latestCheckpointId':clone['id'],
              'cleanAncestorReceiptId':value['id'],'stage':'clean_base_ready'},project['id'])
    return store.put('foundation_relearning_plan',{'state':'base_ready','cleanBaseCheckpointId':clone['id'],
                     'newTokenizerId':token['id'],'newGenesisId':genesis['id']},value['id'])


def advance(control,plan_id):
    store=control.store;value=store.get('foundation_relearning_plan',plan_id)
    if value['state'] in ('completed','held','failed','cancelled'):return value
    if value['state']=='planned':return start(control,plan_id)
    identity=value['newProjectId'];project=store.get('foundation_project',identity);_current(control,value)
    if value['state']=='waiting_data':
        if time.time()<value.get('nextDataCheckAt',0):return value
        job=control.foundation.create_corpus(identity,{})
        return store.put('foundation_relearning_plan',{'state':'preparing_corpus','activeJobId':job['id'],
                         'nextDataCheckAt':time.time()+300},plan_id)
    active=store.job(value['activeJobId']) if value.get('activeJobId') else {}
    if active.get('state') in ('queued','running','pause_requested','paused','cancelling'):return value
    if active.get('state') in ('failed','cancelled'):
        return store.put('foundation_relearning_plan',{'state':'held','reasonCodes':[active.get('error',{}).get('code','RELEARNING_JOB_FAILED')]},plan_id)
    if value['state']=='training':
        run=store.get('foundation_run',active.get('id'),{})
        require(run.get('completedOptimizerSteps',0)>0 and run.get('checkpointId',run.get('lastCheckpointId')),'NO_EFFECTIVE_TRAINING')
        store.put('foundation_policy',{'enabled':bool(value.get('automaticAuthorized'))},identity)
        return store.put('foundation_relearning_plan',{'state':'completed','trainedCheckpointId':run.get('checkpointId',run.get('lastCheckpointId')),
                          'usable':False,'mathematicalUnlearning':False,'requiresNewQualityEvaluation':True},plan_id)
    corpus=store.get('foundation_corpus',project.get('activeCorpusId') or '',{})
    if corpus.get('state')!='ready':
        reasons=corpus.get('reasonCodes',['INSUFFICIENT_CORPUS'])
        insufficient=bool(reasons) and set(reasons)<={'INSUFFICIENT_CORPUS','INSUFFICIENT_TRAIN_GROUPS','INSUFFICIENT_TRAIN_BYTES',
                                                      'INSUFFICIENT_DEV_GROUPS','INSUFFICIENT_SFT_EXAMPLES','INSUFFICIENT_SFT_GROUPS'}
        return store.put('foundation_relearning_plan',{'state':'waiting_data' if insufficient else 'held',
                         'nextDataCheckAt':time.time()+300,'reasonCodes':reasons},plan_id)
    if not project.get('tokenizerId'):
        if value['strategy']=='clean_ancestor':job=control.foundation.enqueue('foundation_relearn',identity,{'planId':plan_id});phase='cloning_clean_base'
        else:job=control.foundation.start_tokenizer(identity);phase='tokenizer'
    elif not project.get('genesisId'):job=control.foundation.initialize(identity);phase='initialization'
    else:
        stage='cpt' if value['strategy']=='clean_ancestor' else 'pretrain'
        body={'parentCheckpointId':project.get('latestCheckpointId') or project['genesisId'],
              'config':{'sequenceLength':min(128,project['config']['max_position_embeddings'])}}
        try:job=control.foundation.start_training(identity,body,stage=stage);phase='training'
        except Fault as exc:
            if exc.code!='CALIBRATION_REQUIRED':raise
            job=control.foundation.calibrate(identity,{**body,'stage':stage});phase='calibration'
    return store.put('foundation_relearning_plan',{'state':phase,'activeJobId':job['id']},plan_id)
