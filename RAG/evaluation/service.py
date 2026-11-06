"""Actual retrieval measurements with stable corpus IDs and ranking metrics."""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

from RAG.core import DomainError,Store,digest,now,uid
from .benchmark import CORPUS,VERSION,questions as builtin_questions

def metrics(ranked,relevant):
    # One credit per relevant unit even if multiple chunks represent that unit.
    seen=set();gains=[]
    for value in ranked[:5]:
        hit=value in relevant and value not in seen;gains.append(int(hit));seen.add(value)
    hits=sum(gains);ideal=sum(1/math.log2(i+2) for i in range(min(5,len(relevant))))
    return {'recallAt5':hits/len(relevant) if relevant else None,'hitAt5':int(hits>0) if relevant else None,
            'mrrAt5':next((1/(i+1) for i,g in enumerate(gains) if g),0) if relevant else None,
            'ndcgAt5':sum(g/math.log2(i+2) for i,g in enumerate(gains))/ideal if ideal else None,
            'correctRejection':not bool(ranked) if not relevant else None}

def aggregate(rows):
    result={}
    for name in ('recallAt5','hitAt5','mrrAt5','ndcgAt5','correctRejection'):
        values=[row[name] for row in rows if row.get(name) is not None]
        result[name]=sum(values)/len(values) if values else None
    times=sorted(row['latencyMs'] for row in rows)
    result.update(count=len(rows),answerableCount=sum(r.get('recallAt5') is not None for r in rows),
                  noAnswerCount=sum(r.get('correctRejection') is not None for r in rows),
                  p50Ms=times[max(0,math.ceil(len(times)*.50)-1)] if times else None,
                  p95Ms=times[max(0,math.ceil(len(times)*.95)-1)] if times else None)
    return result

def validate_questions(questions):
    if not isinstance(questions,list) or not 1<=len(questions)<=2000:raise DomainError('INVALID_EVALUATION','평가 질문은 1~2000개입니다.',422)
    for q in questions:
        if not isinstance(q,dict) or not isinstance(q.get('query'),str) or not q['query'].strip() or len(q['query'].encode())>32768:raise DomainError('INVALID_EVALUATION','질문 형식을 확인하세요.',422)
        for key in ('expectedChunkIds','expectedDocumentIds','splitGroups'):
            if not isinstance(q.get(key,[]),list) or len(q.get(key,[]))>100 or any(not isinstance(i,str) or len(i)>512 for i in q.get(key,[])):raise DomainError('INVALID_EVALUATION','정답 ID 목록을 확인하세요.',422)
        if not isinstance(q.get('filters',{}),dict) or set(q.get('filters',{}))-{'sourceIds','languages','projectId','revision','mode','retrievalMode','includeRelated','useQuality','minDenseScore'}:raise DomainError('INVALID_EVALUATION','평가 필터를 확인하세요.',422)

class EvaluationService:
    def __init__(self,store,pre):self.store,self.pre=store,pre

    def _check(self,job):
        if job.get('fencingToken') is not None:self.store.assert_lease(job['id'],job['fencingToken'])

    def measure(self,pre,questions,options,job=None):
        generation=pre._active_id();rows=[]
        for index,q in enumerate(questions):
            if job:self._check(job)
            if pre._active_id()!=generation:raise DomainError('EVALUATION_INDEX_CHANGED','평가 중 색인이 바뀌었습니다. 다시 실행하세요.',503)
            start=time.perf_counter();response=pre.search({'query':q['query'],'topK':5,**options,**q.get('filters',{})})
            if q.get('expectedPaths') is not None:
                relevant=set(q['expectedPaths']);ranked=[i['path'] for i in response['items']]
            elif q.get('expectedDocumentIds'):
                relevant=set(q['expectedDocumentIds']);ranked=[i['documentId'] for i in response['items']]
            else:relevant=set(q.get('expectedChunkIds',[]));ranked=[i['chunkId'] for i in response['items']]
            row={'queryId':q.get('id',str(index)),'traceId':response['traceId'],'latencyMs':round((time.perf_counter()-start)*1000,3),
                 'selectedIds':[i['chunkId'] for i in response['items']],'searchMode':response['searchMode'],'cacheHit':response.get('cacheHit',False),
                 'questionLanguage':q.get('questionLanguage','unknown'),'case':q.get('case','custom'),**metrics(ranked,relevant)}
            rows.append(row)
            if job:self.store.update_job(job['id'],processedItems=index+1,totalItems=len(questions),unit='queries')
        return {'indexGenerationId':generation,'metrics':aggregate(rows),'byLanguage':{lang:aggregate([r for r in rows if r['questionLanguage']==lang]) for lang in sorted({r['questionLanguage'] for r in rows})},'results':rows}

    def run(self,job):
        body=job['payload'];identity=body.get('evaluationId') or job['id']
        if body.get('suite')=='builtin-code':return self.builtin(job,identity)
        questions=body.get('questions');validate_questions(questions)
        result=self.measure(self.pre,questions,{},job)
        return self.store.put('evaluation',{**result['metrics'],**result,'questionHash':digest(questions),'split':body.get('split','dev'),'finishedAt':now()},id=identity)

    def builtin(self,job,identity):
        from RAG.Pre.service import PreService
        # No benchmark documents enter the production corpus or training exports.
        root=self.store.root/'evaluations'/job['id']
        child=Store(root);config=self.pre.encoder()._config
        child.set_setting('embedding',config)
        child.set_setting('settings',{**child.setting('settings',{}),'embedding':config})
        pre=PreService(child)
        pre._encoder=self.pre.encoder();pre._encoder_key=json.dumps(config,sort_keys=True)
        if not child.list('document'):
            accepted=pre.upload([(path,code.encode()) for path,code in CORPUS.items()])
            for version in child.list('version'):
                self._check(job);pre._process_version(version['id'])
            pre.rebuild()
        elif pre.encoder().available and pre.readiness().get('searchMode')!='hybrid':pre.rebuild()
        questions=builtin_questions();comparisons=[]
        for mode in ('lexical','dense','hybrid'):
            if mode!='lexical' and not pre.encoder().available:
                comparisons.append({'mode':mode,'available':False,'reason':pre.encoder().reason});continue
            self.store.update_job(job['id'],stage='evaluating_'+mode)
            result=self.measure(pre,questions,{'retrievalMode':mode,'includeRelated':mode=='hybrid','useQuality':mode=='hybrid'},job)
            comparisons.append({'mode':mode,'available':True,**result})
        primary=next((r for r in comparisons if r['mode']=='hybrid' and r['available']),comparisons[0])
        report={'suite':'builtin-code','suiteVersion':VERSION,'split':'dev','relevanceUnit':'document_path','questionHash':digest(questions),'corpusHash':digest(CORPUS),
                'questionCount':len(questions),'corpusDocuments':len(CORPUS),'embeddingProfile':pre.encoder().profile,'comparisons':comparisons,
                **primary['metrics'],'results':primary['results'],'finishedAt':now(),'reportPath':str(root/'report.json'),
                'limitations':['small_authored_dev_set','not_production_quality_claim','not_nas_performance_test']}
        pre._write_json(root/'report.json',report)
        return self.store.put('evaluation',report,id=identity)
