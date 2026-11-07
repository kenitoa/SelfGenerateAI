import json
import math
import re
import random
from LLM.core import require,digest,now

def confidence(candidate,baseline,groups,iterations=10000):
    require(len(candidate)==len(baseline)==len(groups) and bool(candidate),'INVALID_PAIRED_RESULTS')
    grouped={}
    for a,b,g in zip(candidate,baseline,groups):grouped.setdefault(g,[]).append(float(a)-float(b))
    values=[sum(v)/len(v) for v in grouped.values()]
    rng=random.Random(42);samples=sorted(sum(rng.choices(values,k=len(values)))/len(values) for _ in range(iterations))
    return {'lower':samples[int(iterations*.025)],'upper':samples[min(iterations-1,int(iterations*.975))]}

def gate(results,baselines,*,production=True,bootstrap=False,focus=None):
    blockers=[]
    if production and (len(results)<200 or any(sum(r.get('mode')==m for r in results)<50 for m in ('code','review','refactor','test'))):blockers.append('INSUFFICIENT_RELEASE_SUITE')
    if any(r.get('criticalFailure') for r in results):blockers.append('CRITICAL_REGRESSION')
    if not results:blockers.append('NO_EVALUATION_RESULTS')
    comparisons=[]
    for name,baseline in baselines.items():
        require([r['taskId'] for r in results]==[r['taskId'] for r in baseline],'EVALUATION_TASK_MISMATCH')
        scores=[int(r['passed']) for r in results];old=[int(r['passed']) for r in baseline]
        selected=[i for i,r in enumerate(results) if not focus or r.get('mode') in focus]
        if not selected:blockers.append('SPECIALIST_EVALUATION_REQUIRED')
        delta=sum(scores[i]-old[i] for i in selected)/len(selected) if selected else 0
        interval=confidence([scores[i] for i in selected],[old[i] for i in selected],[results[i]['group'] for i in selected]) if selected else {'lower':0,'upper':0}
        comparisons.append({'baseline':name,'delta':delta,'interval':interval,'focusModes':focus or [],'sampleCount':len(selected),
            'overallDelta':(sum(scores)-sum(old))/len(scores) if scores else 0})
        if not bootstrap and (delta<.02 or interval['lower']<=0):blockers.append('EVALUATION_INCONCLUSIVE')
        for key in ('mode','language'):
            for label in set(r.get(key) for r in results):
                ids=[i for i,r in enumerate(results) if r.get(key)==label]
                if ids and sum(scores[i]-old[i] for i in ids)/len(ids)<-.02:blockers.append('REGRESSION_DETECTED')
    if not baselines and not bootstrap:blockers.append('BASELINE_REQUIRED')
    return {'passed':not blockers,'decision':'passed' if not blockers else 'held','reasonCodes':sorted(set(blockers)),'comparisons':comparisons}

class Evaluator:
    def __init__(self,store,registry,inference,sandbox):self.store,self.registry,self.inference,self.sandbox=store,registry,inference,sandbox

    def register_suite(self,body):
        require(body.get('kind') in ('dev','release','regression','evidence','unanswerable','audit'),'INVALID_SUITE_KIND')
        require(isinstance(body.get('tasks'),list) and body['tasks'],'EMPTY_EVALUATION_SUITE')
        require(len({t.get('id') for t in body['tasks']})==len(body['tasks']),'DUPLICATE_TASK')
        for t in body['tasks']:
            require(t.get('id') and t.get('group') and t.get('prompt') and t.get('verifier') in ('exact','contains','python_tests','cases'),'INVALID_EVALUATION_TASK')
            if t['verifier'] in ('python_tests','cases'):self.store.get('fixture',t['fixtureId'])
            elif t['verifier']=='exact':require(isinstance(t.get('expected'),str),'INVALID_EVALUATION_TASK')
            else:require(isinstance(t.get('expected'),list) and bool(t['expected']) and all(isinstance(v,str) and v for v in t['expected']),'INVALID_EVALUATION_TASK')
        return self.store.put('suite',{**body,'suiteHash':digest(body),'fixture':body.get('fixture') is True})

    def baseline_ids(self,artifact):
        family=self.store.get('family',artifact['familyId'])
        ids=[family['anchorArtifactId'],*family.get('marketBaselineIds',[])]
        slot=self.store.get('slot','champion',{})
        if slot:ids.append(self.store.get('release',slot['activeReleaseId'])['artifactId'])
        if artifact['kind']=='merged_checkpoint':ids.extend(artifact.get('parents',[]))
        if artifact.get('sourceArtifactId'):ids.extend(self.baseline_ids(self.store.get('artifact',artifact['sourceArtifactId'])))
        return [i for i in dict.fromkeys(ids) if i!=artifact['id']]

    def focus(self,artifact):
        if artifact.get('branchId'):
            role=self.store.get('branch',artifact['branchId'])['specialization']
            return [role] if role in ('code','review','test','refactor') else None
        if artifact.get('sourceArtifactId'):return self.focus(self.store.get('artifact',artifact['sourceArtifactId']))
        if artifact['kind']=='merged_checkpoint':
            return sorted({role for identity in artifact.get('parents',[]) for role in (self.focus(self.store.get('artifact',identity)) or [])}) or None
        return None

    def progress(self):
        rows=[]
        for result in self.store.list('evaluation'):
            artifact=self.store.get('artifact',result['artifactId']);family=self.store.get('family',artifact['familyId'])
            for comparison in result['gate'].get('comparisons',[]):
                rows.append({'evaluationId':result['id'],'artifactId':artifact['id'],'familyId':family['id'],'branchId':artifact.get('branchId'),
                    'suiteId':result['suiteId'],'suiteHash':result['suiteHash'],'observedAt':result['finishedAt'],'fixture':result.get('fixture',False),
                    'baselineRole':'original' if comparison['baseline']==family['anchorArtifactId'] else 'market_or_previous',**comparison,
                    'improvementConfirmed':comparison['delta']>=.02 and comparison['interval']['lower']>0})
        return sorted(rows,key=lambda r:r['observedAt'])

    def bootstrap(self,job):
        import psutil
        artifact=self.store.get('artifact',job['payload']['artifactId'])
        require(not (artifact.get('origin')=='random_init' or artifact.get('foundationProjectId')),'FOUNDATION_USABLE_CERTIFICATE_REQUIRED',status=409)
        require(artifact['kind']=='base' and not artifact.get('fixture'),'BOOTSTRAP_ONLY_FOR_ANCHOR')
        root=self.registry.verify(artifact)
        config=json.loads((root/'tokenizer_config.json').read_text('utf-8'))
        require('<|im_start|>' in str(config.get('chat_template','')),'UNSUPPORTED_CHAT_TEMPLATE')
        prompts=[('Reply exactly with OK.','OK'),('What is 1 + 1? Reply with the number only.','2'),
            ('Write a Python function identity(x) that returns x.','return x')]
        results=[]
        for i,(instruction,expected) in enumerate(prompts):
            prompt='<|im_start|>user\n'+instruction+'<|im_end|>\n<|im_start|>assistant\n'
            text,timing=self.inference.text(artifact['id'],prompt,64)
            results.append({'expected':expected,'passed':expected in text,'responseHash':digest(text.encode()),'timings':timing})
            self.store.progress(job,'bootstrap',i+1,len(prompts),'tasks')
        from LLM.workers.resources import budget
        memory=psutil.Process().memory_info().rss;resource_budget=budget(self.store);limit=resource_budget['maxRssBytes']
        reasons=[]
        if not all(r['passed'] for r in results):reasons.append('ANCHOR_SMOKE_FAILED')
        if memory>limit:reasons.append('MEMORY_BUDGET_EXCEEDED')
        certificate=self.store.put('certificate',{'artifactId':artifact['id'],'passed':not reasons,'bootstrap':True,'baselineOnly':True,
            'reasonCodes':reasons,'results':results,'rssBytes':memory,'resourceBudget':resource_budget,'claim':'기술 검증·초기 기준; 성능 개선 인증 아님','observedAt':now()})
        if not reasons:self.store.put('artifact',{'state':'ready','certificateId':certificate['id']},artifact['id'])
        return certificate

    def evaluate(self,job):
        import psutil
        body=job['payload'];suite=self.store.get('suite',body['suiteId']);artifact=self.store.get('artifact',body['artifactId'])
        require(suite['kind']!='audit' or body.get('audit') is True,'AUDIT_NOT_FOR_TUNING')
        if body.get('bootstrap'):require(artifact['kind']=='base','BOOTSTRAP_ONLY_FOR_ANCHOR')
        identities=list(dict.fromkeys([artifact['id'],*body.get('baselineIds',[]),*self.baseline_ids(artifact)]));all_results={}
        for identity in identities:
            results=[]
            for i,task in enumerate(suite['tasks']):
                current=self.store.assert_lease(job)
                require(current['state'] not in ('cancelling','cancelled'),'JOB_CANCELLED',status=409)
                text,timing=self.inference.text(identity,task['prompt'],int(task.get('maxTokens',128)))
                if task['verifier'] in ('python_tests','cases'):
                    blocks=re.findall(r'```(?:[\w+-]+)?\s*\n([\s\S]*?)```',text)
                    receipt=self.sandbox.verify(blocks[0] if blocks else text,self.store.get('fixture',task['fixtureId']))
                    passed=receipt['passed']
                elif task['verifier']=='exact':passed=text.strip()==task['expected'].strip()
                else:passed=all(s in text for s in task['expected'])
                results.append({'taskId':task['id'],'group':task['group'],'mode':task.get('mode','code'),'language':task.get('language','unknown'),
                    'passed':passed,'timings':timing,'rssBytes':psutil.Process().memory_info().rss+timing.get('runtimeRssBytes',0),
                    'resourceObserved':timing.get('runtimeResourceObserved',True),'responseHash':digest(text.encode()),'criticalFailure':task.get('critical') is True and not passed})
                self.store.progress(job,'evaluating',i+1,len(suite['tasks']),'tasks',artifactId=identity)
            all_results[identity]=results
        decision=gate(all_results[artifact['id']],{k:v for k,v in all_results.items() if k!=artifact['id']},production=suite['kind']=='release' and not suite['fixture'],bootstrap=body.get('bootstrap') is True,focus=self.focus(artifact))
        # Fixture evaluations can verify plumbing but can never certify production releases.
        return self.store.put('evaluation',{'artifactId':artifact['id'],'suiteId':suite['id'],'suiteHash':suite['suiteHash'],
            'results':all_results,'gate':decision,'productionCertified':False,'fixture':suite['fixture'],'finishedAt':now(),
            'missingCertifications':['regression','evidence','unanswerable','resources']},job['id'])

    def certify(self,body):
        artifact=self.store.get('artifact',body['artifactId']);bootstrap=body.get('bootstrap') is True
        require(not artifact.get('fixture'),'FIXTURE_NOT_FOR_PRODUCTION',status=409)
        require(not bootstrap or artifact['kind']=='base','BOOTSTRAP_ONLY_FOR_ANCHOR')
        records=[self.store.get('evaluation',i) for i in body.get('evaluationIds',[])]
        require(records and all(r['artifactId']==artifact['id'] and not r.get('fixture') for r in records),'INVALID_CERTIFICATION_INPUT')
        kinds={self.store.get('suite',r['suiteId'])['kind']:r for r in records}
        blockers=[]
        for kind,count in [('release',200),('regression',100),('evidence',40),('unanswerable',20)]:
            record=kinds.get(kind)
            if not record or len(record['results'][artifact['id']])<count:blockers.append(kind.upper()+'_SUITE_REQUIRED');continue
            rows=record['results'][artifact['id']]
            if any(not r.get('resourceObserved',True) for r in rows):blockers.append('RUNTIME_RESOURCE_OBSERVATION_UNAVAILABLE')
            if kind=='release' and not record['gate']['passed'] and not bootstrap:blockers.extend(record['gate']['reasonCodes'])
            if any(r['criticalFailure'] for r in rows):blockers.append('CRITICAL_REGRESSION')
            rate=sum(r['passed'] for r in rows)/len(rows)
            if kind in ('evidence','unanswerable') and rate<.95:blockers.append(kind.upper()+'_QUALITY_FAILED')
            baseline_ids=list(record['results']);baseline_ids.remove(artifact['id'])
            if not baseline_ids and not bootstrap:blockers.append('BASELINE_REQUIRED')
            for baseline in baseline_ids:
                old=record['results'][baseline]
                if kind=='regression' and any(b['passed'] and not a['passed'] for a,b in zip(rows,old)):blockers.append('REGRESSION_DETECTED')
                timings=sorted(r['timings']['total_duration'] for r in rows)
                previous=sorted(r['timings']['total_duration'] for r in old)
                if timings[int(.95*(len(timings)-1))]>previous[int(.95*(len(previous)-1))]*1.25:blockers.append('LATENCY_BUDGET_EXCEEDED')
            from LLM.workers.resources import budget
            limit=budget(self.store)['maxRssBytes']
            if max(r.get('rssBytes',limit+1) for r in rows)>limit:blockers.append('MEMORY_BUDGET_EXCEEDED')
        result=self.store.put('certificate',{'artifactId':artifact['id'],'evaluationIds':[r['id'] for r in records],
            'passed':not blockers,'reasonCodes':sorted(set(blockers)),'bootstrap':bootstrap,'policyVersion':1,'observedAt':now()})
        if not blockers:self.store.put('artifact',{'state':'ready','certificateId':result['id']},artifact['id'])
        return result
