"""Independent capability scopes backed by executable or source-derived oracles.

The explanation scope is evidence-bound fact extraction, not a certificate for
arbitrary natural-language explanations. The model never grades its own output.
"""
from __future__ import annotations
import ast
import json
import re

from LLM.core import Fault, digest, now, require, uid
from .corpus import secret_reasons
from .evaluation import first_usable_gate, subsequent_gate, SAFETY_CATEGORIES, _EvaluationGuard

ORACLES={'code':'executable_cases','explanation':'source_facts','review':'static_findings','tests':'mutation_tests'}
LANGUAGES={'python','javascript','typescript','sql','go','rust','java','c','cpp','ko','en'}
RULES={'mutable_default','bare_except','dynamic_eval'}
FACT_INSTRUCTION=('Return only JSON {"claims":[{"factId":string,"value":value,"startLine":integer,"endLine":integer}]}. '
 'Lines are 1-based inclusive source evidence. Function values have name, positional, keywordOnly, vararg and kwarg; '
 'the last two are strings or null. Return-expression values are Python expression strings or null. No other fields.')


def register_domain(store,body):
    require(isinstance(body,dict) and not set(body)-{'purpose','language','name'},'INVALID_CAPABILITY_DOMAIN')
    purpose,language=body.get('purpose'),body.get('language')
    require(purpose in ORACLES and language in LANGUAGES,'UNSUPPORTED_CAPABILITY_DOMAIN')
    require((purpose=='code' and language not in ('ko','en')) or
            (purpose in ('explanation','review') and language in ('ko','en')) or
            (purpose=='tests' and language=='python'),'UNSUPPORTED_CAPABILITY_DOMAIN')
    value={'purpose':purpose,'language':language,'oracleType':ORACLES[purpose],'state':'registered','policyVersion':'independent-domain-v1',
           'scope':{'code':'short_function_completion','explanation':'python_source_fact_claims',
                    'review':'python_static_findings','tests':'python_mutation_test_generation'}[purpose],
           'firstGroupMinimum':100,'incrementalGroupMinimum':200,'neverTrain':True}
    identity='domain-'+digest(value)[:24]
    return store.get('foundation_domain',identity,{}) or store.put('foundation_domain',value,identity)


def source_facts(source):
    tree=ast.parse(source);facts={}
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
            stem='function:'+node.name
            facts[stem]={'value':{'name':node.name,'positional':[arg.arg for arg in node.args.posonlyargs+node.args.args],
                                 'keywordOnly':[arg.arg for arg in node.args.kwonlyargs],
                                 'vararg':node.args.vararg.arg if node.args.vararg else None,
                                 'kwarg':node.args.kwarg.arg if node.args.kwarg else None},
                         'startLine':node.lineno,'endLine':node.end_lineno}
            for index,returned in enumerate(item for item in ast.walk(node) if isinstance(item,ast.Return)):
                facts[stem+':return:'+str(index)]={'value':ast.unparse(returned.value) if returned.value else None,
                                                 'startLine':returned.lineno,'endLine':returned.end_lineno}
    require(facts,'SOURCE_FACTS_UNAVAILABLE')
    return facts


def source_findings(source):
    tree=ast.parse(source);found=[]
    for node in ast.walk(tree):
        rule=None
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and any(isinstance(item,(ast.List,ast.Dict,ast.Set)) for item in [*node.args.defaults,*[item for item in node.args.kw_defaults if item is not None]]):rule='mutable_default'
        if isinstance(node,ast.ExceptHandler) and node.type is None:rule='bare_except'
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in ('eval','exec'):rule='dynamic_eval'
        if rule:found.append({'ruleId':rule,'line':node.lineno})
    return sorted(found,key=lambda item:(item['line'],item['ruleId']))


def _fixture(control,identity):
    row=control.store.get('fixture',identity)
    require(row.get('independent') is True and row.get('targetOrigin') not in ('teacher','model_generated')
            and not row.get('fixture') and not row.get('testOnly'),'INDEPENDENT_FIXTURE_REQUIRED')
    require(row.get('cases') or row.get('tests'),'INDEPENDENT_FIXTURE_REQUIRED')
    return row


def register_suite(control,body):
    store=control.store;domain=store.get('foundation_domain',body.get('domainId',''))
    kind=body.get('kind','release');require(kind in ('dev','release','audit'),'INVALID_DOMAIN_SUITE')
    tasks=body.get('tasks');require(isinstance(tasks,list) and 0<len(tasks)<=1000,'INVALID_DOMAIN_SUITE')
    seen=set();fingerprint_groups={};clean=[];blockers=[]
    for item in tasks:
        require(isinstance(item,dict) and isinstance(item.get('id'),str) and item['id'] not in seen and
                isinstance(item.get('group'),str) and item['group'] and isinstance(item.get('prompt'),str) and item['prompt'],
                'INVALID_DOMAIN_TASK')
        require(isinstance(item.get('critical',False),bool),'INVALID_DOMAIN_TASK')
        groups=item.get('lineageGroups',[])
        require(isinstance(groups,list) and groups and all(isinstance(group,str) and group for group in groups),'EVALUATION_LINEAGE_REQUIRED')
        require(not secret_reasons(item['prompt']+item.get('sourceCode','')),'EVALUATION_PRIVATE_INPUT_FORBIDDEN')
        critical=item.get('critical',False)
        require(not critical or item.get('safetyCategory') in SAFETY_CATEGORIES,'INVALID_SAFETY_CATEGORY')
        task={key:item[key] for key in ('id','group','prompt','sourceCode','codePrefix','fixtureId','lineageGroups','critical','safetyCategory','factIds','mutants') if key in item}
        purpose=domain['purpose'];fixture=None
        if purpose in ('explanation','review','tests'):
            source=item.get('sourceCode');require(isinstance(source,str) and 0<len(source.encode())<=16000,'SOURCE_CODE_REQUIRED')
            try:tree=ast.parse(source)
            except (SyntaxError,ValueError,RecursionError):raise Fault('ORACLE_SOURCE_INVALID') from None
            fingerprint={'purpose':purpose,'sourceAst':ast.dump(tree,include_attributes=False)}
            if purpose=='explanation':
                facts=source_facts(source);requested=item.get('factIds',sorted(facts))
                require(isinstance(requested,list) and requested and len(set(requested))==len(requested) and set(requested)<=set(facts),'SOURCE_FACTS_UNAVAILABLE')
                task['oracleFacts']={key:facts[key] for key in requested};task['factIds']=requested
            elif purpose=='review':task['oracleFindings']=source_findings(source)
            else:
                mutants=item.get('mutants');require(isinstance(mutants,list) and 2<=len(mutants)<=10 and all(isinstance(code,str) and len(code.encode())<=16000 for code in mutants),'INDEPENDENT_MUTANTS_REQUIRED')
                require(len({digest(ast.dump(ast.parse(code),include_attributes=False)) for code in [source,*mutants]})==len(mutants)+1,'INDEPENDENT_MUTANTS_REQUIRED')
                fixture=_fixture(control,item.get('fixtureId',''))
                profile=store.get('verifier_profile',fixture['profileId'])
                require(profile.get('language')=='python','ORACLE_LANGUAGE_UNAVAILABLE')
                if control.sandbox.available():
                    baseline=control.sandbox.verify(source,fixture)
                    witnesses=[control.sandbox.verify(code,fixture) for code in mutants]
                    if not baseline.get('passed') or any(row.get('passed') is not False or row.get('state')!='failed' for row in witnesses):blockers.append('MUTATION_WITNESS_REQUIRED')
                    task['mutationWitnessReceiptIds']=[baseline.get('id'),*[row.get('id') for row in witnesses]]
                else:blockers.append('VERIFIER_UNAVAILABLE')
                # The generated-test runner has fixed expectations, hidden from the model.
                test_fixture=store.put('fixture',{'profileId':fixture['profileId'],'independent':True,'targetOrigin':'controller',
                    'cases':[{'function':'_seed_run_tests','args':[],'expected':True}],'purpose':'domain_mutation_verifier'})
                task['testFixtureId']=test_fixture['id'];task['testFixtureHash']=digest(test_fixture)
        else:
            fixture=_fixture(control,item.get('fixtureId',''))
            profile=store.get('verifier_profile',fixture['profileId'])
            require(profile.get('language')==domain['language'],'ORACLE_LANGUAGE_MISMATCH')
            fingerprint={'purpose':purpose,'language':domain['language'],'prompt':item['prompt'],
                         'codePrefix':item.get('codePrefix',''),'cases':fixture.get('cases'),'tests':fixture.get('tests')}
        if fixture:task['fixtureHash']=digest(fixture)
        checksum=digest(fingerprint);old=fingerprint_groups.get(checksum)
        require(old is None or old==item['group'],'EVALUATION_GROUP_CONFLICT')
        fingerprint_groups[checksum]=item['group'];task['problemFingerprint']=checksum;seen.add(item['id']);clean.append(task)
    if domain['purpose']=='review' and kind=='release':
        ordinary=[item for item in clean if not item.get('critical')]
        require(ordinary and any(item['oracleFindings'] for item in ordinary) and any(not item['oracleFindings'] for item in ordinary),'REVIEW_POSITIVE_AND_NEGATIVE_CASES_REQUIRED')
    split={'dev':'dev','release':'release','audit':'audit'}[kind]
    value={'domainId':domain['id'],'domainHash':digest({key:domain[key] for key in ('purpose','language','oracleType','scope','policyVersion')}),
           'purpose':domain['purpose'],'language':domain['language'],'kind':kind,'tasks':clean,'shardId':str(body.get('shardId') or uid()),
           'state':'held' if blockers else 'ready','reasonCodes':sorted(set(blockers)),'neverTrain':True}
    value['suiteHash']=digest({key:value[key] for key in ('domainId','domainHash','kind','tasks')})
    with store.tx():
        for task in clean:
            for group in task['lineageGroups']:
                prior=store.get('foundation_split',digest(group),{})
                require(not prior or prior.get('split')==split,'SPLIT_LEAKAGE',status=409)
        for task in clean:
            for group in task['lineageGroups']:store.put('foundation_split',{'groupId':group,'split':split,'evaluationReserved':True},digest(group))
        return store.put('foundation_domain_suite',value)


def _json_output(text):
    stripped=text.strip();block=re.fullmatch(r'```(?:json)?\s*\n([\s\S]*?)```\s*',stripped)
    require(not stripped.startswith('```') or block is not None,'INVALID_DOMAIN_RESPONSE')
    value=json.loads(block.group(1) if block else stripped)
    require(isinstance(value,dict),'INVALID_DOMAIN_RESPONSE');return value


def judge_static(domain,task,text):
    response=_json_output(text)
    if domain['purpose']=='explanation':
        require(set(response)=={'claims'},'INVALID_DOMAIN_RESPONSE')
        claims=response.get('claims');require(isinstance(claims,list),'INVALID_DOMAIN_RESPONSE')
        result={}
        for claim in claims:
            require(isinstance(claim,dict) and set(claim)=={'factId','value','startLine','endLine'} and claim['factId'] not in result,'INVALID_DOMAIN_RESPONSE')
            result[claim['factId']]={key:claim[key] for key in ('value','startLine','endLine')}
        if set(result)!=set(task['oracleFacts']):return False
        for identity,expected in task['oracleFacts'].items():
            actual=result[identity]
            for key in ('startLine','endLine'):require(isinstance(actual[key],int) and not isinstance(actual[key],bool),'INVALID_DOMAIN_RESPONSE')
            total=max(expected['endLine'],len(task.get('sourceCode','').splitlines()))
            if not (1<=actual['startLine']<=expected['startLine']<=expected['endLine']<=actual['endLine']<=total):return False
            if ':return:' in identity and actual['value'] is not None and expected['value'] is not None:
                require(isinstance(actual['value'],str),'INVALID_DOMAIN_RESPONSE')
                try:
                    if ast.dump(ast.parse(actual['value'],mode='eval'),include_attributes=False)!=ast.dump(ast.parse(expected['value'],mode='eval'),include_attributes=False):return False
                except (SyntaxError,ValueError,RecursionError):raise Fault('INVALID_DOMAIN_RESPONSE') from None
            elif actual['value']!=expected['value']:return False
        return True
    require(set(response)=={'findings'},'INVALID_DOMAIN_RESPONSE')
    findings=response.get('findings');require(isinstance(findings,list),'INVALID_DOMAIN_RESPONSE')
    for item in findings:require(isinstance(item,dict) and set(item)=={'ruleId','line'} and item['ruleId'] in RULES and isinstance(item['line'],int) and not isinstance(item['line'],bool),'INVALID_DOMAIN_RESPONSE')
    require(len({digest(item) for item in findings})==len(findings),'INVALID_DOMAIN_RESPONSE')
    return sorted(findings,key=lambda item:(item['line'],item['ruleId']))==task['oracleFindings']


def _tests_source(text):
    blocks=re.findall(r'```(?:python)?\s*\n([\s\S]*?)```',text);code=(blocks[0] if blocks else text).strip()
    tree=ast.parse(code)
    require(all(isinstance(node,ast.FunctionDef) for node in tree.body) and any(node.name=='check' and not node.args.args for node in tree.body),'INVALID_GENERATED_TEST')
    denied={'open','exec','eval','compile','globals','locals','getattr','setattr','__import__','input','vars','dir','type','id'}
    require(any(isinstance(node,ast.Assert) for node in ast.walk(tree)),'ASSERTION_REQUIRED')
    for node in ast.walk(tree):
        require(not isinstance(node,(ast.Import,ast.ImportFrom,ast.Global,ast.Nonlocal)),'UNSAFE_GENERATED_TEST')
        if isinstance(node,ast.Name):require(node.id not in denied,'UNSAFE_GENERATED_TEST')
        if isinstance(node,ast.Attribute):require(not node.attr.startswith('__'),'UNSAFE_GENERATED_TEST')
    return code


def _judge(control,domain,task,text,guard=None):
    if guard:guard()
    if domain['purpose'] in ('explanation','review'):return judge_static(domain,task,text),[]
    fixture=control.store.get('fixture',task['fixtureId']);require(digest(fixture)==task['fixtureHash'],'EVALUATION_FIXTURE_CHANGED')
    if domain['purpose']=='code':
        blocks=re.findall(r'```(?:\w+)?\s*\n([\s\S]*?)```',text);code=task.get('codePrefix','')+(blocks[0] if blocks else text)
        receipt=control.sandbox.verify(code,fixture)
        if guard:guard()
        return receipt.get('passed') is True,[receipt.get('id')]
    generated=_tests_source(text);fixed=control.store.get('fixture',task['testFixtureId'])
    require(digest(fixed)==task['testFixtureHash'],'EVALUATION_FIXTURE_CHANGED')
    suffix='\ndef _seed_run_tests():\n    try:\n        check()\n    except AssertionError:\n        return False\n    return True\n'
    rows=[]
    for source in [task['sourceCode'],*task['mutants']]:
        if guard:guard()
        rows.append(control.sandbox.verify(source+'\n'+generated+suffix,fixed))
        if guard:guard()
    passed=rows[0].get('passed') is True and all(row.get('passed') is False and row.get('state')=='failed' for row in rows[1:])
    return passed,[row.get('id') for row in rows]


def _champions(store,domain_id,project_id):
    return store.rows("""SELECT id,json_extract(body,'$.version') revision FROM records
        WHERE kind='foundation_domain_certificate' AND json_extract(body,'$.domainId')=?
        AND json_extract(body,'$.projectId')=? AND json_extract(body,'$.state')='active'
        AND json_extract(body,'$.passed')=1 ORDER BY id""",(domain_id,project_id))


def evaluate(control,job):
    store=control.store;body=job.get('payload',job);identity=job.get('id',uid())
    suite=store.get('foundation_domain_suite',body.get('suiteId',''));domain=store.get('foundation_domain',suite['domainId'])
    require(suite['state']=='ready' and suite['suiteHash']==digest({key:suite[key] for key in ('domainId','domainHash','kind','tasks')}),'DOMAIN_SUITE_NOT_READY')
    require(suite['domainHash']==digest({key:domain[key] for key in ('purpose','language','oracleType','scope','policyVersion')}),'DOMAIN_SUITE_NOT_READY')
    require(suite['kind']!='audit' or body.get('audit') is True,'AUDIT_NOT_FOR_TUNING',status=403)
    artifact_id=body.get('artifactId',body.get('checkpointId'));runtime=control.foundation.runtime;artifact=runtime.artifact(artifact_id)
    project_id=artifact.get('projectId',artifact.get('foundationProjectId'));require(project_id==body.get('projectId',project_id),'INCOMPATIBLE_BASE')
    baselines=list(dict.fromkeys(body.get('baselineIds',[])))
    champion_context=_champions(store,domain['id'],project_id)
    champions=[row for row in store.list('foundation_domain_certificate') if row.get('domainId')==domain['id'] and row.get('projectId')==project_id and row.get('passed') and row.get('state')=='active']
    if champions:baselines=list(dict.fromkeys([champions[-1]['artifactId'],*baselines]))
    require(artifact_id not in baselines,'CHAMPION_CANNOT_BE_ITS_OWN_CHALLENGER')
    fixture=any(runtime.artifact(value).get('fixture') or runtime.artifact(value).get('testOnly') for value in [artifact_id,*baselines])
    guard=_EvaluationGuard(control,project_id,[artifact_id,*baselines],job=job if 'payload' in job else None,
        suite=suite,suite_kind='foundation_domain_suite',policy_required=not fixture)
    guard.pin('foundation_domain',domain)
    guard.contexts.append((lambda:_champions(store,domain['id'],project_id),champion_context))
    for champion in champions:guard.pin('foundation_domain_certificate',champion)
    blockers=[];probes=[]
    if domain['purpose'] in ('code','tests'):
        if not control.sandbox.available():blockers.append('VERIFIER_UNAVAILABLE')
        else:
            for profile_id in {store.get('fixture',task['fixtureId'])['profileId'] for task in suite['tasks']}:
                probe=control.sandbox.probe(store.get('verifier_profile',profile_id))
                if not probe.get('passed'):blockers.append('SANDBOX_PROBE_REQUIRED')
                probes.append(probe.get('id'))
    if not blockers:
        if suite['kind']=='release':control.foundation.evaluator._reserve_release(identity,suite,project_id)
        elif suite['kind']=='audit':
            from .audit import reserve
            reserve(store,suite,identity,project_id)
    results={}
    for current in [artifact_id,*baselines] if not blockers else []:
        rows=[]
        for task in suite['tasks']:
            fixed_ids=[task.get('fixtureId'),task.get('testFixtureId')]
            guard.check(fixture_id=fixed_ids)
            row_key=digest([identity,current,task['problemFingerprint'],task['id']]);prior=store.get('foundation_domain_task',row_key,{})
            if prior:
                rows.append(prior['result'])
                guard.progress('domain_evaluating',sum(len(value) for value in results.values())+len(rows),len([artifact_id,*baselines])*len(suite['tasks']),'tasks')
                continue
            instruction={'source_facts':FACT_INSTRUCTION+' Extract exactly the requested facts.',
                         'static_findings':'Return JSON findings only: ruleId and line. Supported rules: mutable_default, bare_except, dynamic_eval.',
                         'mutation_tests':'Return a Python function check() containing behavioral asserts. Call the supplied source functions. No imports or introspection.',
                         'executable_cases':'Complete the code. Return code only.'}[domain['oracleType']]
            prompt=task['prompt']
            if task.get('sourceCode'):prompt+='\nSource:\n'+task['sourceCode']
            if task.get('factIds'):prompt+='\nFact IDs: '+json.dumps(task['factIds'])
            row={'taskId':task['id'],'group':task['group'],'critical':task.get('critical',False),'safetyCategory':task.get('safetyCategory'),
                 'passed':False,'syntaxPassed':False,'criticalFailure':False,'neverTrain':True}
            try:
                text,timing=runtime.text(current,[{'role':'system','content':instruction},{'role':'user','content':prompt}],128)
                guard.check(fixture_id=fixed_ids)
                passed,receipts=_judge(control,domain,task,text,guard=lambda:guard.check(fixture_id=fixed_ids))
                row.update(passed=bool(passed),syntaxPassed=True,receiptIds=receipts,responseHash=digest(text.encode()),timings=timing)
            except (Fault,SyntaxError,ValueError,TypeError,KeyError,RecursionError) as exc:row['reasonCode']=getattr(exc,'code','INVALID_DOMAIN_RESPONSE')
            row['criticalFailure']=bool(row['critical'] and not row['passed'])
            with store.tx():
                guard.check(fixture_id=fixed_ids)
                store.put('foundation_domain_task',{'evaluationId':identity,'artifactId':current,'result':row,'neverTrain':True},row_key)
            rows.append(row)
            guard.progress('domain_evaluating',sum(len(value) for value in results.values())+len(rows),len([artifact_id,*baselines])*len(suite['tasks']),'tasks')
        results[current]=rows
    gate={'passed':False,'reasonCodes':sorted(set(blockers))} if blockers else (subsequent_gate(results[artifact_id],{key:results[key] for key in baselines},fixture=fixture) if baselines else first_usable_gate(results[artifact_id],fixture=fixture))
    if not blockers and suite['kind']=='release':
        from .resources import foundation_budget
        budget=foundation_budget(store,test_only=bool(fixture));limit=min(12*1024**3,budget['hostWorkloadBudgetBytes'],budget.get('memoryLimitBytes') or 2**63)
        measured=all(row.get('timings',{}).get('runtimeResourceObserved') is True and
                     isinstance(row.get('timings',{}).get('peakRssBytes'),int) and row['timings']['peakRssBytes']<=limit for row in results[artifact_id])
        if not measured:gate={**gate,'passed':False,'reasonCodes':sorted(set(gate['reasonCodes']+['RUNTIME_RESOURCE_GATE_REQUIRED']))}
    value={'projectId':project_id,'artifactId':artifact_id,'domainId':domain['id'],'purpose':domain['purpose'],'language':domain['language'],
           'scope':domain['scope'],'suiteId':suite['id'],'suiteHash':suite['suiteHash'],'kind':suite['kind'],'baselineIds':baselines,
           'results':results,'gate':gate,'productionCertified':suite['kind']=='release' and gate['passed'] and not fixture,
           'effectiveGenerationProfiles':{key:sorted({(row.get('timings',{}).get('contextLimit'),row.get('timings',{}).get('effectiveOutputLimit')) for row in rows},key=str) for key,rows in results.items()},
           'state':'held' if blockers else 'completed','fixture':bool(fixture),'probeReceiptIds':probes,'neverTrain':True,'observedAt':now()}
    guard.refresh_policy()
    with store.tx():
        guard.check(final=True)
        return store.put('foundation_domain_evaluation',value,identity)


def certify(control,evaluation_id):
    store=control.store;evaluation=store.get('foundation_domain_evaluation',evaluation_id)
    require(evaluation.get('productionCertified') is True and evaluation.get('gate',{}).get('passed') and not evaluation.get('fixture'),'DOMAIN_QUALITY_REQUIRED')
    artifact=control.foundation.runtime.artifact(evaluation['artifactId'])
    guard=_EvaluationGuard(control,evaluation['projectId'],[artifact['id']])
    guard.pin('foundation_domain_evaluation',evaluation)
    require(not artifact.get('fixture') and not artifact.get('testOnly') and artifact.get('state') not in ('revoked','quarantined'),'FIXTURE_NOT_FOR_PRODUCTION')
    control.foundation.validate_policy(control.foundation.project(evaluation['projectId']))
    certificates=[row for row in store.list('foundation_certificate') if row.get('artifactId')==artifact['id'] and row.get('purpose')=='usable_release'
                  and row.get('passed') is True and row.get('usable') is True and row.get('state')!='revoked' and not row.get('fixture') and not row.get('testOnly')]
    require(certificates,'USABLE_FOUNDATION_MODEL_REQUIRED',status=409)
    context=int(artifact.get('trainedContextTokens',0));profile=[context,min(128,context//2)]
    require(context>0 and evaluation.get('effectiveGenerationProfiles',{}).get(artifact['id']) and all(
            list(item)==profile for item in evaluation['effectiveGenerationProfiles'][artifact['id']]),'EVALUATION_PROFILE_MISMATCH',status=409)
    domain=store.get('foundation_domain',evaluation['domainId'])
    suite=store.get('foundation_domain_suite',evaluation['suiteId'])
    require(suite.get('state')=='ready' and suite.get('suiteHash')==evaluation.get('suiteHash')
        and suite['suiteHash']==digest({key:suite[key] for key in ('domainId','domainHash','kind','tasks')})
        and suite.get('domainHash')==digest({key:domain[key] for key in ('purpose','language','oracleType','scope','policyVersion')}),'DOMAIN_SUITE_NOT_READY')
    guard.pin('foundation_domain',domain);guard.pin('foundation_domain_suite',suite)
    base=certificates[-1];guard.pin('foundation_certificate',base)
    for identity in base.get('evaluationIds',[]):
        proof=store.get('foundation_evaluation',identity)
        require(proof.get('state')=='completed' and proof.get('gate',{}).get('passed') is True,'USABLE_FOUNDATION_MODEL_REQUIRED')
        guard.pin('foundation_evaluation',proof)
    if base.get('qualifiedGraduationId'):
        proof=store.get('foundation_graduation',base['qualifiedGraduationId']);guard.pin('foundation_graduation',proof)
        require(proof.get('state')=='qualified','USABLE_FOUNDATION_MODEL_REQUIRED')
        if proof.get('evaluationId'):
            measured=store.get('foundation_evaluation',proof['evaluationId'])
            require(measured.get('state')=='completed' and measured.get('gate',{}).get('passed') is True,'USABLE_FOUNDATION_MODEL_REQUIRED')
            guard.pin('foundation_evaluation',measured)
    for task in suite.get('tasks',[]):
        for key in ('fixtureId','testFixtureId'):
            if task.get(key):
                fixed=store.get('fixture',task[key]);guard.pin('fixture',fixed)
                require(digest(fixed)==task['fixtureHash' if key=='fixtureId' else 'testFixtureHash'],'EVALUATION_FIXTURE_CHANGED')
                if fixed.get('profileId'):guard.pin('verifier_profile',store.get('verifier_profile',fixed['profileId']))
    guard.pin_context(lambda:_champions(store,domain['id'],evaluation['projectId']))
    value={'domainId':domain['id'],'purpose':domain['purpose'],'language':domain['language'],'scope':domain['scope'],'projectId':evaluation['projectId'],
           'artifactId':artifact['id'],'state':'active','passed':True,'evaluationId':evaluation_id,'suiteHash':evaluation['suiteHash'],
           'baseCertificateId':certificates[-1]['id'],'tokenizerHash':artifact.get('tokenizerHash'),'architectureHashV2':artifact.get('architectureHashV2'),
           'profileHash':certificates[-1].get('profileHash'),
           'executionScope':certificates[-1].get('executionScope'),
           'backendReceiptHash':certificates[-1].get('backendReceiptHash'),
           'neverTrain':True,'testOnly':False,'issuedAt':now()}
    guard.refresh_policy()
    with store.tx():
        guard.check(final=True)
        for old in store.list('foundation_domain_certificate'):
            if old.get('domainId')==domain['id'] and old.get('projectId')==evaluation['projectId'] and old.get('state')=='active':store.put('foundation_domain_certificate',{'state':'retired'},old['id'])
        return store.put('foundation_domain_certificate',value)


def authorize(store,artifact_id,mode,language):
    require(mode in ORACLES and language in LANGUAGES,'MODEL_MODE_NOT_APPROVED',status=409)
    artifact=store.get('artifact',artifact_id,{}) or store.get('foundation_checkpoint',artifact_id,{})
    require(artifact and artifact.get('state') not in ('revoked','quarantined','archived') and not artifact.get('testOnly') and not artifact.get('fixture'),'MODEL_MODE_NOT_APPROVED',status=409)
    for row in store.list('foundation_domain_certificate'):
        if row.get('artifactId')==artifact_id and row.get('purpose')==mode and row.get('language')==language and row.get('state')=='active' and row.get('passed') is True:
            base=store.get('foundation_certificate',row.get('baseCertificateId',''),{})
            require(base.get('passed') and base.get('usable') and base.get('artifactId')==artifact_id and base.get('state')!='revoked'
                    and base.get('profileHash')==row.get('profileHash'),'MODEL_MODE_NOT_APPROVED',status=409)
            require(base.get('executionScope')==row.get('executionScope')==artifact.get('executionScope','nas') and
                    base.get('backendReceiptHash')==row.get('backendReceiptHash'),'EXECUTION_SCOPE_REQUALIFICATION_REQUIRED',status=409)
            require(row.get('tokenizerHash')==artifact.get('tokenizerHash') and row.get('architectureHashV2')==artifact.get('architectureHashV2'),'MODEL_MODE_NOT_APPROVED',status=409)
            return row
    raise Fault('MODEL_MODE_NOT_APPROVED',status=409)


def canonical_request(body):
    mode={'explain':'explanation','test':'tests'}.get(body.get('mode','code'),body.get('mode','code'))
    require(mode in ORACLES,'MODEL_MODE_NOT_APPROVED',status=409)
    language=body.get('language','ko' if mode in ('explanation','review') else 'python')
    language={'korean':'ko','ko-KR':'ko','english':'en','en-US':'en','py':'python','js':'javascript','ts':'typescript'}.get(language,language)
    require(language in LANGUAGES,'MODEL_LANGUAGE_NOT_APPROVED',status=409)
    return mode,language


def prepare_source_request(body,certificate):
    mode=certificate['effectiveMode']
    require(mode in ('explanation','review') and body.get('capabilityScope')==certificate.get('capabilityScope'),
            'CAPABILITY_SCOPE_REQUIRED',status=409)
    source=body.get('sourceCode')
    require(isinstance(source,str) and 0<len(source.encode())<=16000 and not secret_reasons(source),'SOURCE_CODE_REQUIRED')
    task={'sourceCode':source};goal=body.get('goal','')
    require(isinstance(goal,str) and len(goal)<=1000 and not secret_reasons(goal),'INVALID_DOMAIN_REQUEST')
    try:
        if mode=='explanation':
            facts=source_facts(source);requested=body.get('factIds',sorted(facts))
            require(isinstance(requested,list) and requested and len(set(requested))==len(requested) and set(requested)<=set(facts),'SOURCE_FACTS_UNAVAILABLE')
            task['oracleFacts']={key:facts[key] for key in requested}
            instruction=FACT_INSTRUCTION+' Fact IDs: '+json.dumps(requested)
        else:
            task['oracleFindings']=source_findings(source)
            instruction='Return JSON findings only: ruleId and line. Check only mutable_default, bare_except, dynamic_eval. Return every finding and no other fields.'
    except (SyntaxError,ValueError,RecursionError):raise Fault('ORACLE_SOURCE_INVALID') from None
    return [{'role':'system','content':instruction},{'role':'user','content':goal+'\nSource:\n'+source}],task
