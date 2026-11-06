import json
from LLM.core import digest,now,require
from LLM.data.curation import TaskFactory,VERSION,quality


class Curator:
    def __init__(self,bridge):self.b=bridge;self.c=bridge.c;self.store=self.c.store

    def run(self,policy):
        branches=self.store.list('branch');created=[]
        if not branches:return {'createdExampleIds':[],'reasonCodes':['BRANCH_REQUIRED']}
        family=policy.get('familyId') or branches[0]['familyId']
        branches=[b for b in branches if b['familyId']==family]
        roles={b['specialization']:b for b in branches}
        sources={s['id']:s for s in self.b.store.list('source')}
        factory=TaskFactory(self.c)
        for document in self.b.store.list('document'):
            identity=digest([document['id'],document.get('currentVersionId'),family])
            base={'documentId':document['id'],'versionId':document.get('currentVersionId'),'sourceId':document.get('sourceId'),'familyId':family,'builderVersion':VERSION}
            source=sources.get(document.get('sourceId'),{})
            reasons=[]
            if document.get('deletedAt') or not document.get('currentVersionId'):reasons.append('DOCUMENT_UNAVAILABLE')
            if source.get('trainingPolicy')!='allowed' or source.get('id') not in policy.get('allowedSourceIds',[]):reasons.append('SOURCE_TRAINING_NOT_ALLOWED')
            if policy.get('allowedLicenses') and source.get('licenseId') not in policy['allowedLicenses']:reasons.append('LICENSE_NOT_ALLOWED')
            if reasons:
                self.store.put('document_curation',{**base,'state':'blocked','reasonCodes':reasons},identity);continue
            binding=self.store.get('document_verification',document['currentVersionId'],{})
            route=self.store.get('source_purpose',source['id'],{})
            signature=digest([VERSION,policy.get('version'),binding.get('fixtureIds',[]),route.get('specialization'),self.c.evolution.active_hash()])
            previous=self.store.get('document_curation',identity,{})
            if previous.get('signature')==signature and previous.get('state') in ('ready','quarantined'):continue
            try:
                content=self.b.pre.content(document['currentVersionId'])
                text=content['content'] if isinstance(content,dict) else content
                if isinstance(text,bytes):text=text.decode('utf-8')
                require(isinstance(text,str),'UNSUPPORTED_DOCUMENT_CONTENT')
                reasons=quality(text)
                if reasons:
                    self.store.put('document_curation',{**base,'signature':signature,'state':'quarantined','reasonCodes':reasons,'inputHash':digest(text.encode())},identity);continue
                transformed=self.c.evolution.transform({'text':text,'documentId':document['id']})
                text=transformed['text']
                fixtures=[self.store.get('fixture',i) for i in binding.get('fixtureIds',[])]
                offset=previous.get('chunkCursor',0) if previous.get('signature')==signature else 0
                drafts,held,plan=factory.build(text,fixtures,offset=offset,document_role=route.get('specialization'));ids=list(previous.get('exampleIds',[])) if offset else []
                for draft in drafts:
                    draft['instruction']='분야: '+draft['role']+'\n목적: '+draft['objective']+'\n'+draft['instruction']
                    branch=roles.get(draft['role'])
                    if not branch:
                        held.append({'role':draft['role'],'reasonCodes':['SPECIALIST_BRANCH_REQUIRED']});continue
                    candidate_key=digest([family,document['currentVersionId'],branch['id'],draft['instruction'],draft['target'],VERSION])
                    old=self.store.get('task_candidate',candidate_key,{})
                    if old:ids.append(old['exampleId']);continue
                    lineage=[{'sourceId':source['id'],'documentId':document['id'],'versionId':document['currentVersionId']}]
                    item=self.b.post.create_example({'input':draft['instruction'],'target':draft['target'],'lineage':lineage,'objective':draft['objective']})
                    require(not self.b.post._example_reasons(item),'EXAMPLE_POLICY_CHANGED',status=409)
                    reviewed=self.b.post.review_example(item['id'],{'decision':'approve','permissionStatus':'allowed','reason':'정책 검증: '+draft['receipt']['verifier']})
                    receipt={**draft['receipt'],'approvalMode':'policy','policyId':'learning','policyVersion':policy.get('version',0),
                        'targetHash':digest(draft['target'].encode()),'extensionHash':self.c.evolution.active_hash()}
                    self.b.store.put('training_example',{**reviewed,'approvalMode':'policy','policyReceipt':receipt,'branchId':branch['id'],
                        'targetOrigin':'teacher' if source.get('origin')=='model_self_code' else draft['targetOrigin'],'objective':draft['objective'],'executionStatus':receipt['executionStatus']},item['id'])
                    self.store.put('task_candidate',{'exampleId':item['id'],'branchId':branch['id'],'receipt':receipt},candidate_key)
                    ids.append(item['id']);created.append(item['id'])
                self.store.put('document_curation',{**base,'signature':signature,'state':('ready' if plan['complete'] else 'partial') if ids else 'held','exampleIds':list(dict.fromkeys(ids)),'heldTasks':held,
                    'chunkCursor':plan['nextCursor'],'chunkCount':plan['chunkCount'],'reasonCodes':[] if ids else ['NO_VERIFIED_TRAINING_TASK'],'inputHash':digest(text.encode()),'processedAt':now()},identity)
            except Exception as exc:self.store.put('document_curation',{**base,'signature':signature,'state':'held','reasonCodes':[getattr(exc,'code','CURATION_FAILED')]},identity)
        if created:
            # Dedup export requests, including a previous attempt still in progress.
            ids=[e['id'] for e in self.b.post.examples() if e.get('reviewStatus')=='approved' and e.get('permissionStatus')=='allowed' and
                self.store.get('branch',e.get('branchId',''),{}).get('familyId')==family and not self.b.post._example_reasons(e)]
            key=digest(sorted(ids))
            if not self.store.get('curation_export',key,{}):
                try:
                    snapshot=self.b.post.create_snapshot({'name':'목적별 자동학습 '+now(),'exampleIds':ids})
                    self.store.put('curation_export',{'snapshotId':snapshot['id'],'familyId':family},key)
                except Exception as exc:self.store.event('snapshot.blocked','curation',{'code':getattr(exc,'code','SNAPSHOT_EXPORT_FAILED')})
        self.import_exports()
        return {'createdExampleIds':created,'documentCount':len(self.store.list('document_curation'))}

    def import_exports(self):
        for receipt in self.store.list('curation_export'):
            snapshot=self.b.post.snapshot(receipt['snapshotId'])
            if snapshot['status']=='ready' and not self.store.get('snapshot',snapshot['id'],{}):
                try:self.b.import_snapshot(snapshot['id'],receipt['familyId'])
                except Exception as exc:self.store.event('snapshot.import_failed',snapshot['id'],{'code':getattr(exc,'code','IMPORT_FAILED')})
