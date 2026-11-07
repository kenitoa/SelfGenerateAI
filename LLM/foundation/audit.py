"""Fresh, non-training audit bank usage and restricted aggregate feedback."""
from datetime import datetime,timezone,timedelta
from LLM.core import digest,now,require


def reserve(store,suite,evaluation_id,project_id):
    require(suite.get('kind')=='audit','AUDIT_SUITE_REQUIRED')
    key=digest([suite['suiteHash'],suite['shardId']]);week=list(datetime.now(timezone(timedelta(hours=9))).isocalendar()[:2])
    with store.tx():
        prior=store.get('foundation_audit_shard',key,{})
        require(not prior or prior.get('evaluationId')==evaluation_id,'AUDIT_SHARD_ALREADY_CONSUMED',status=409)
        if prior:return prior
        require(not any(row.get('weekKst')==week for row in store.list('foundation_audit_shard')),'AUDIT_BUDGET_EXHAUSTED',status=409)
        tasks=[task for task in suite['tasks'] if not task.get('critical')]
        for task in tasks:
            require(not store.get('foundation_eval_problem',task['problemFingerprint'],{}),'AUDIT_PROBLEM_REUSED',status=409)
        for task in tasks:store.put('foundation_eval_problem',{'evaluationId':evaluation_id,'projectId':project_id,'state':'audit_consumed','neverTrain':True},task['problemFingerprint'])
        return store.put('foundation_audit_shard',{'evaluationId':evaluation_id,'suiteHash':suite['suiteHash'],'projectId':project_id,
                         'weekKst':week,'neverTrain':True,'state':'consumed','consumedAt':now()},key)


def aggregate(evaluation):
    require(evaluation.get('kind')=='audit','AUDIT_SUITE_REQUIRED')
    rows=evaluation.get('results',{}).get(evaluation.get('artifactId'),[])
    groups={}
    for row in rows:
        if not row.get('critical'):groups.setdefault(row['group'],[]).append(row.get('passed') is True)
    successes=sum(all(values) for values in groups.values())
    return {'domainId':evaluation.get('domainId'),'purpose':evaluation.get('purpose'),'language':evaluation.get('language'),
            'groupCount':len(groups),'successCount':successes,'passAt1':successes/len(groups) if groups else None,
            'criticalFailureCount':sum(bool(row.get('criticalFailure')) for row in rows),'neverTrain':True,
            'independentFinalReuseAllowed':False,'state':evaluation.get('state'),'reasonCodes':evaluation.get('gate',{}).get('reasonCodes',[])}


def run(control,job):
    from .domains import evaluate
    payload={**job.get('payload',job),'audit':True}
    result=evaluate(control,{**job,'payload':payload} if 'payload' in job else payload)
    summary=aggregate(result)
    control.store.put('foundation_audit_feedback',summary,result['id'])
    return {'evaluationId':result['id'],**summary}
