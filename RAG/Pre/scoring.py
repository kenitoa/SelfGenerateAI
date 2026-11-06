"""Explainable quality signals. These are heuristics, never test execution results."""
from datetime import datetime,timezone
import re

def rating(text,language,analysis,source=None):
    source=source or {};repo=source.get('repositorySignals',{})
    syntax=analysis.get('syntaxValid')
    syntax_score=1.0 if syntax is True else .1 if syntax is False else .5
    documented=bool(re.search(r'(?m)^\s*(?:# |// |/\*|///)|"""|\'\'\'|(?m:^#\s)',text))
    tests=bool(re.search(r'(?m)\b(?:def test_|describe\(|it\(|@Test|#\[test\])',text))
    test_signal=True if tests else repo.get('hasTests')
    docs_signal=True if documented else repo.get('hasReadme')
    symbols=analysis.get('symbols',[])
    documented_symbols=0
    for symbol in symbols:
        body=text[symbol['startOffset']:symbol['endOffset']]
        preceding='\n'.join(text[:symbol['startOffset']].splitlines()[-3:])
        if re.search(r'"""|\'\'\'|//|/\*|(?m:^\s*#)',preceding+'\n'+'\n'.join(body.splitlines()[:3])):documented_symbols+=1
    prose=re.sub(r'(?ms)^(`{3,}|~{3,})[^\n]*\n.*?^\1[^\n]*','',text) if language=='markdown' else ''
    references=set(re.findall(r'`([A-Za-z_][\w]*)`',prose))
    defined={s['name'] for s in symbols}
    reference_match=len(references&defined)/len(references) if references and defined else None
    stamp=source.get('repositoryUpdatedAt') or source.get('lastSyncedAt')
    age=None
    if stamp:
        try:age=max(0,(datetime.now(timezone.utc)-datetime.fromisoformat(stamp.replace('Z','+00:00'))).days)
        except (ValueError,TypeError):pass
    freshness=.5 if age is None else max(.1,1-age/730)
    versioned=bool(source.get('lastSuccessfulRevision') or source.get('discoveryRevision'))
    score=round(30*syntax_score+20*(1 if docs_signal else .5 if docs_signal is None else .2)+20*(1 if test_signal else .5 if test_signal is None else .2)+15*freshness+15*(1 if versioned else .5))
    if reference_match is not None:score=max(0,score-round(5*(1-reference_match)))
    return {'score':score,'tier':'high' if score>=75 else 'standard' if score>=45 else 'low','method':'static-heuristic-v1',
            'signals':{'syntaxValid':syntax,'documentationPresent':docs_signal,'testsPresent':test_signal,'ageDays':age,'revisionPinned':versioned,
                       'testExecution':'not_run','documentationConsistency':'not_verified','documentedSymbols':documented_symbols,
                       'documentationCoverage':documented_symbols/len(symbols) if symbols else None,'documentedReferenceMatch':reference_match}}
