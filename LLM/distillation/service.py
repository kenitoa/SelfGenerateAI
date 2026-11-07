from LLM.core import require,digest

class Distiller:
    def __init__(self,control):self.c=control
    def run(self,job):
        body=job['payload'];teachers=body.get('teacherArtifactIds',[])
        require(2<=len(teachers)<=3,'DISTILLATION_REQUIRES_TEACHERS')
        require(all(not self.c.store.get('artifact',teacher).get('comparisonOnly') for teacher in teachers),
                'COMPARISON_ONLY_MODEL',status=409)
        require(self.c.bridge is not None,'RAG_POLICY_GATEWAY_REQUIRED',status=503)
        branch=self.c.store.get('branch',body['branchId']);accepted=[]
        tasks=body.get('fixtureIds',[]);require(tasks,'INDEPENDENT_FIXTURES_REQUIRED')
        for index,identity in enumerate(tasks):
            fixture=self.c.store.get('fixture',identity)
            for teacher in teachers:
                self.c.store.assert_lease(job);self.c.registry.verify(self.c.store.get('artifact',teacher))
                text,_=self.c.inference.text(teacher,fixture['prompt'],int(fixture.get('maxTokens',256)))
                receipt=self.c.sandbox.verify(text,fixture)
                if not receipt['passed']:continue
                require(fixture.get('lineage'),'DISTILLATION_LINEAGE_REQUIRED')
                item=self.c.bridge.post.create_example({'input':fixture['prompt'],'target':text,'lineage':fixture['lineage']})
                policy=self.c.scheduler.policy()
                require(policy.get('enabled') and 'fixture_solution' in policy.get('permittedTaskTypes',[]),'POLICY_NOT_APPROVED',status=403)
                require(all(x['sourceId'] in policy['allowedSourceIds'] for x in fixture['lineage']),'POLICY_NOT_APPROVED',status=403)
                reviewed=self.c.bridge.post.review_example(item['id'],{'decision':'approve','permissionStatus':'allowed','reason':'독립 격리 fixture 검증: '+receipt['id']})
                self.c.bridge.store.put('training_example',{**reviewed,'targetOrigin':'teacher','teacherArtifactId':teacher,'verifierReceiptId':receipt['id'],'branchId':branch['id']},item['id'])
                accepted.append(item['id']);break
            self.c.store.progress(job,'distillation_data',index+1,len(tasks),'tasks')
        require(accepted,'NO_VERIFIED_DISTILLATION_EXAMPLES')
        approved=self.c.bridge.post.examples()
        seed_ids=[e['id'] for e in approved if e.get('reviewStatus')=='approved' and e.get('permissionStatus')=='allowed' and e.get('targetOrigin')!='teacher']
        snapshot=self.c.bridge.post.create_snapshot({'exampleIds':list(dict.fromkeys(seed_ids+accepted)),'name':'증류 '+digest(teachers)[:8]})
        self.c.store.put('distillation',{'teacherArtifactIds':teachers,'branchId':branch['id'],'snapshotId':snapshot['id'],
            'state':'waiting_export','singleStudent':True,'exampleIds':accepted},job['id'])
        return {'state':'waiting_export','snapshotId':snapshot['id'],'branchId':branch['id'],'exampleIds':accepted}
