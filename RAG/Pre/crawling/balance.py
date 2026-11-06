"""Soft fair scheduling across hosts, code languages, providers and repositories."""
import json
from pathlib import PurePosixPath
from urllib.parse import urlsplit
from RAG.Pre.processing import LANGUAGES

def labels(row):
    meta=json.loads(row['meta']) if isinstance(row.get('meta'),str) else row.get('meta',row)
    url=row.get('url') or meta['url'];host=urlsplit(url).hostname
    language=LANGUAGES.get(PurePosixPath(urlsplit(url).path).suffix.lower())
    if language in ('html','markdown','json','yaml','toml'):language=None
    language=language or meta.get('language')
    if not language:
        language=next((lang for word,lang in [('python','python'),('javascript','javascript'),('typescript','typescript'),('/rust','rust'),('/go/','go')] if word in url.lower()),'unknown')
    return {'hosts':host,'languages':language,'providers':meta.get('origin',row.get('origin','link')),'repositories':meta.get('repository','web:'+host)}

def record(counts,row):
    for key,value in labels(row).items():counts.setdefault(key,{})[value]=counts.setdefault(key,{}).get(value,0)+1

def select(rows,count,counts,config):
    usage={key:dict(value) for key,value in counts.items()};pool=list(rows);chosen=[]
    categories={id(row):labels(row) for row in pool}
    weights=config.get('languageWeights',{})
    while pool and len(chosen)<count:
        def score(row):
            item=categories[id(row)];penalty=0
            for key,factor in [('hosts',2),('languages',2),('providers',1),('repositories',1)]:
                value=usage.get(key,{}).get(item[key],0)
                denominator=weights.get(item[key],1) if key=='languages' else 1
                penalty+=factor*value/denominator
            return (penalty+row['priority']/100,row.get('created_at',''),row['url'])
        best=min(pool,key=score);pool.remove(best);chosen.append(best);record(usage,best)
    return chosen
