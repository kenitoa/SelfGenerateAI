"""Conservative preparation of held-out document-policy modification problems.

These are controller-authored policy specifications, not human-authored tasks
or a model qualification. Reference modules are never executed on the host.
Only an actual local pinned sandbox run can verify a reference implementation.
"""
from __future__ import annotations

import argparse
import ast
import inspect
import json
from pathlib import Path
import textwrap

from LLM.core import Fault,Store,atomic_json,canonical,digest,file_hash,now,require
from LLM.evolution.service import TARGET,SelfEvolution,validate_source,validate_transform
from LLM.verification.broker import pinned

ACTOR='user-delegated-self-code-policy-specification-v1'
MINIMUM=100


def specifications():
    root=Path(__file__).resolve().parents[2]
    path=root/'LLM/runtime-data/preparation/self-code-policy-bank-v1/catalog.json'
    require(path.is_file() and not path.is_symlink() and path.stat().st_size<=1024**2,'SELF_CODE_CATALOG_REQUIRED')
    value=json.loads(path.read_text('utf-8'))
    require(value.get('catalogHash')==digest({k:v for k,v in value.items() if k!='catalogHash'})
        and value.get('kind')=='self_code_policy_specification_catalog'
        and value.get('origin')=='policy_specification' and value.get('humanAuthor') is False
        and value.get('neverTrain') is True,'SELF_CODE_CATALOG_CHANGED')
    records=value.get('specifications')
    require(isinstance(records,list) and 0<len(records)<=200,'SELF_CODE_CATALOG_REQUIRED')
    return records


def regressions():
    # Read the trusted controller's literal list without executing its code.
    source=inspect.getsource(SelfEvolution.setup);tree=ast.parse(textwrap.dedent(source))
    assignment=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='texts' for t in n.targets))
    def literal(node):
        if isinstance(node,ast.Constant) and isinstance(node.value,(str,int)): return node.value
        if isinstance(node,ast.List): return [literal(item) for item in node.elts]
        if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Mult):
            left,right=literal(node.left),literal(node.right)
            require(isinstance(left,str) and type(right) is int and 0<=right<=2048,'CONTROLLER_REGRESSION_CHANGED')
            return left*right
        raise Fault('CONTROLLER_REGRESSION_CHANGED')
    texts=literal(assignment.value)
    require(len(texts)==12,'CONTROLLER_REGRESSION_CHANGED')
    cases=[{'function':'transform','args':[{'text':text,'documentId':str(index)}],
        'expected':{'text':text.replace('\r\n','\n').strip(),'tags':[]}} for index,text in enumerate(texts)]
    return cases,digest(source.encode())


def _reference(spec):
    return spec['predicateSource']+'\ndef transform(document):\n    text=document[\'text\']\n    return {\'text\':text.replace(\'\\r\\n\',\'\\n\').strip(),\'tags\':['+repr(spec['id'])+'] if detect(text) else []}\n'


def _cases(spec,mandatory):
    cases=[]
    for positive,texts in ((True,spec['positive']),(False,spec['negative'])):
        for index,text in enumerate(texts):
            cases.append({'function':'transform','args':[{'text':text,'documentId':spec['id']+':'+str(positive)+':'+str(index)}],
                'expected':{'text':text.replace('\r\n','\n').strip(),'tags':[spec['id']] if positive else []}})
    return cases+mandatory


def _goal(spec):
    return (spec['title']+'. Add only the tag '+repr(spec['id'])+' exactly when this policy holds: '+spec['mechanism']+
        '. Preserve every word and code AST. Preserve the existing CRLF/outer-whitespace normalization and its untagged safety regressions. Never mutate the input. Return only text and tags.')


def _shape(source):
    class WithoutValues(ast.NodeTransformer):
        def __init__(self):self.names={}
        def rename(self,name):
            stable={'transform','json','math','re','unicodedata','isinstance','len','max','min','bool','str','int','dict','list','tuple','set','type','ord','abs','any','zip','map','enumerate','ValueError','TypeError'}
            if name in stable:return name
            if name not in self.names:self.names[name]='local_'+str(len(self.names))
            return self.names[name]
        def visit_Name(self,node):node.id=self.rename(node.id);return node
        def visit_arg(self,node):node.arg=self.rename(node.arg);return node
        def visit_FunctionDef(self,node):node.name=self.rename(node.name);return self.generic_visit(node)
        def visit_Constant(self,node):return ast.copy_location(ast.Constant(value=type(node.value).__name__),node)
    return digest(ast.dump(WithoutValues().visit(ast.parse(source)),include_attributes=False))


def prepare(root):
    from .corpus import secret_reasons
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    project=Path(__file__).resolve().parents[2];base=(project/TARGET).read_text('utf-8')
    mandatory,controller_hash=regressions();validate_source(base)
    problems=[];shapes=set();families=set()
    for spec in specifications():
        require(spec['id'] not in families,'POLICY_FAMILY_REUSED')
        reference=_reference(spec);validate_source(reference)
        shape=_shape(reference);require(shape not in shapes,'POLICY_IMPLEMENTATION_CONSTANT_VARIANT')
        shapes.add(shape);families.add(spec['id']);cases=_cases(spec,mandatory)
        for case in cases:
            require(not secret_reasons(case['args'][0]['text']),'SELF_CODE_PRIVATE_INPUT_FORBIDDEN')
            validate_transform(case['args'][0]['text'],case['expected'])
        goal=_goal(spec)
        specification_hash=digest({'family':spec['id'],'title':spec['title'],'mechanism':spec['mechanism'],'cases':cases})
        problems.append({'id':'document-policy:'+spec['id'],'familyId':'policy:self-code:'+spec['id'],
            'taxonomy':spec['title'],'decisionMechanism':spec['mechanism'],'goal':goal,'baseSource':base,
            'baseSourceHash':digest(base.encode()),'referenceSource':reference,'referenceSourceHash':digest(reference.encode()),
            'implementationShapeHash':shape,'cases':cases,'oracleHash':digest(cases),'specificationHash':specification_hash,
            'problemFingerprint':digest({'baseAst':ast.dump(ast.parse(base),include_attributes=False),'cases':cases}),
            'positiveCaseCount':len(spec['positive']),'negativeCaseCount':len(spec['negative']),
            'mandatoryCaseCount':len(mandatory),'neverTrain':True,'split':'release','referenceState':'pending_reference_execution'})
    body={'schemaVersion':1,'kind':'foundation_self_code_problem_bank','target':TARGET,
        'origin':'policy_specification','actor':ACTOR,'humanAuthor':False,'humanReviewed':False,
        'approval':{'scope':'policy_specification','approvedForPreparation':True,'modelQualificationApproved':False,
            'runtimeReferenceVerified':False,'delegation':'Explicit user request to prepare document-policy self-code examination problems'},
        'sourceReceipt':{'controllerSetupHash':controller_hash,'extensionSha256':file_hash(project/TARGET),
            'authoringModuleSha256':file_hash(Path(__file__)),
            'privateCatalogSha256':file_hash(project/'LLM/runtime-data/preparation/self-code-policy-bank-v1/catalog.json')},
        'familyDefinition':'Distinct predicate/decision mechanism and oracle. Constant, keyword, threshold, extension, whitespace and paraphrase variants share one family; combinations cannot add independent families.',
        'independentPreparedFamilyCount':len(families),'independentVerifiedFamilyCount':0,
        'preparedProblemCount':len(problems),'minimumIndependentProblems':MINIMUM,
        'shortfall':max(0,MINIMUM-len(families)),'mandatoryCases':mandatory,'tasks':problems,
        'neverTrain':True,'holdoutGroups':sorted(p['familyId'] for p in problems),
        'state':'pending_reference_execution','readyForModelExam':False,'modelEvaluationPerformed':False,
        'selfCodeEligible':False,'qualificationSuccessRate':None,
        'reasonCodes':['REFERENCE_EXECUTION_REQUIRED']+(['SELF_CODE_INDEPENDENT_TASKS_REQUIRED'] if len(families)<MINIMUM else []),
        'limitation':'This partial bank does not assert a mathematical upper bound. Only the enumerated independent requirements have been prepared; no constant-only variants or safety regressions are counted toward 100.',
        'nasModified':False,'observedAt':now()}
    body['bankHash']=digest(body)
    atomic_json(root/'bank.json',body)
    return body


def checked_bank(path):
    value=json.loads(Path(path).read_text('utf-8'))
    return validate_bank(value)


def validate_bank(value):
    require(value.get('kind')=='foundation_self_code_problem_bank' and value.get('bankHash')==digest({k:v for k,v in value.items() if k!='bankHash'}),'SELF_CODE_BANK_CHANGED')
    mandatory,controller_hash=regressions();specs={spec['id']:spec for spec in specifications()}
    root=Path(__file__).resolve().parents[2]
    require(value['sourceReceipt']=={'controllerSetupHash':controller_hash,'extensionSha256':file_hash(root/TARGET),
        'authoringModuleSha256':file_hash(Path(__file__)),
        'privateCatalogSha256':file_hash(root/'LLM/runtime-data/preparation/self-code-policy-bank-v1/catalog.json')},'SELF_CODE_BANK_SOURCE_CHANGED')
    require(value.get('origin')=='policy_specification' and value.get('humanAuthor') is False and value.get('humanReviewed') is False
        and value.get('neverTrain') is True and value.get('approval',{}).get('scope')=='policy_specification','SELF_CODE_BANK_PROVENANCE_REQUIRED')
    require(len(value['tasks'])==len(specs)==value['independentPreparedFamilyCount']==value['preparedProblemCount']
        and len({task['familyId'] for task in value['tasks']})==len(specs),'POLICY_FAMILY_REUSED')
    shapes=set()
    for task in value['tasks']:
        identity=task['id'].removeprefix('document-policy:');spec=specs.get(identity)
        require(spec is not None and task['familyId']=='policy:self-code:'+identity,'SELF_CODE_BANK_CHANGED')
        require(task['oracleHash']==digest(task['cases']) and task['referenceSourceHash']==digest(task['referenceSource'].encode())
            and task['referenceSource']==_reference(spec) and task['cases']==_cases(spec,mandatory)
            and task['goal']==_goal(spec) and task['baseSource']==(root/TARGET).read_text('utf-8')
            and task['baseSourceHash']==digest(task['baseSource'].encode())
            and task['problemFingerprint']==digest({'baseAst':ast.dump(ast.parse(task['baseSource']),include_attributes=False),'cases':task['cases']}),'SELF_CODE_BANK_CHANGED')
        shape=_shape(task['referenceSource']);require(shape not in shapes,'POLICY_IMPLEMENTATION_CONSTANT_VARIANT');shapes.add(shape)
        require(task['implementationShapeHash']==shape and task['neverTrain'] is True and task['split']=='release','SELF_CODE_BANK_CHANGED')
    require(value['holdoutGroups']==sorted(task['familyId'] for task in value['tasks']),'SELF_CODE_BANK_CHANGED')
    return value


def verify_references(control,bank,profile_id):
    """Explicit local verification; creates no model eligibility or source patch."""
    from .acceptance import local_container_status,require_local_path
    validate_bank(bank)
    require_local_path(control.store.root)
    engine=local_container_status();results=[];reason=[]
    profile=control.store.get('verifier_profile',profile_id,{})
    if not engine.get('available'):reason.extend(engine.get('reasonCodes') or ['LOCAL_CONTAINER_ENGINE_REQUIRED'])
    if not profile or not pinned(profile) or profile.get('language')!='python' or profile.get('purpose')!='self_code_regression':reason.append('SELF_CODE_SANDBOX_PROFILE_REQUIRED')
    probe=None
    if not reason:
        probe=control.sandbox.probe(profile)
        if probe.get('passed') is not True:reason.append('SANDBOX_PROBE_REQUIRED')
    if not reason:
        for task in bank['tasks']:
            fixture=control.store.put('fixture',{'profileId':profile_id,'cases':task['cases'],'independent':True,
                'targetOrigin':'policy_specification','humanAuthor':False,'humanReviewed':False,'neverTrain':True,
                'split':'release','purpose':'self_code_regression','sourceTaskHash':task['specificationHash'],
                'verifierProfileHash':digest(profile),'profileImageDigest':profile['image'],'bankHash':bank['bankHash']})
            # A sandbox-side wrapper checks input immutability as well. It never
            # receives case expectations; those stay in the outside comparator.
            wrapped='import json\n'+task['referenceSource']+'\n_bank_original=transform\ndef transform(document):\n    before=json.dumps(document,sort_keys=True)\n    result=_bank_original(document)\n    if json.dumps(document,sort_keys=True)!=before: raise ValueError(\'INPUT_MUTATED\')\n    return result\n'
            validate_source(wrapped)
            receipt=control.sandbox.verify(wrapped,fixture)
            verified=(receipt.get('passed') is True and receipt.get('state')=='passed' and receipt.get('codeHash')==digest(wrapped.encode())
                and receipt.get('fixtureHash')==digest(fixture) and receipt.get('environmentDigest')==profile['image']
                and receipt.get('probeReceiptId')==probe['id'])
            results.append({'taskId':task['id'],'familyId':task['familyId'],'passed':verified,
                'fixtureId':fixture['id'],'fixtureHash':digest(fixture),'referenceSourceHash':task['referenceSourceHash'],
                'executedWrapperHash':digest(wrapped.encode()),'verifierReceiptId':receipt['id'],'probeReceiptId':receipt.get('probeReceiptId')})
    passed=bool(results) and len(results)==len(bank['tasks']) and all(row['passed'] for row in results)
    if results and not passed:reason.append('SELF_CODE_REFERENCE_FAILED')
    if bank['independentPreparedFamilyCount']<MINIMUM:reason.append('SELF_CODE_INDEPENDENT_TASKS_REQUIRED')
    body={'bankHash':bank['bankHash'],'state':'references_verified' if passed else 'pending_reference_execution',
        'referencesVerified':passed,'independentVerifiedFamilyCount':len({r['familyId'] for r in results if r['passed']}),
        'results':results,'profileId':profile_id,'profileHash':digest(profile) if profile else None,'probeReceiptId':probe.get('id') if probe else None,
        'reasonCodes':sorted(set(reason)),'readyForModelExam':False,'modelEvaluationPerformed':False,'selfCodeEligible':False,
        'qualificationSuccessRate':None,'neverTrain':True,'nasModified':False,'observedAt':now()}
    return control.store.put('foundation_self_code_bank_verification',body)


def install(control,bank,verification):
    """Existing 100-task gate remains authoritative; partial banks cannot register."""
    from .selfcode import register_suite
    validate_bank(bank)
    verification=control.store.get('foundation_self_code_bank_verification',verification.get('id',''))
    require(bank['independentPreparedFamilyCount']>=MINIMUM and verification.get('referencesVerified') is True
        and verification.get('bankHash')==bank['bankHash'] and verification.get('independentVerifiedFamilyCount',0)>=MINIMUM,
        'SELF_CODE_INDEPENDENT_TASKS_REQUIRED')
    current_fixture=control.evolution.policy().get('fixtureId');require(current_fixture,'SELF_CODE_CURRENT_REGRESSION_REQUIRED')
    existing=control.store.get('fixture',current_fixture)
    require(existing['cases']==bank['mandatoryCases'],'SELF_CODE_CURRENT_REGRESSION_CHANGED')
    require(digest(control.store.get('verifier_profile',verification['profileId']))==verification['profileHash'],
        'SELF_CODE_BANK_PROFILE_CHANGED')
    fixtures={row['taskId']:row for row in verification['results']}
    tasks=[{'id':task['id'],'baseSource':task['baseSource'],'goal':task['goal'],'fixtureId':fixtures[task['id']]['fixtureId'],
        'independent':True,'mandatory':False} for task in bank['tasks']]
    tasks.append({'id':'mandatory:current-controller','baseSource':bank['tasks'][0]['baseSource'],
        'goal':'Preserve current controller document normalization and all safety regressions.',
        'fixtureId':current_fixture,'independent':True,'mandatory':True})
    for group in bank['holdoutGroups']:
        old=control.store.get('foundation_split',digest(group),{})
        require(not old or old.get('split')=='release','SPLIT_LEAKAGE')
        control.store.put('foundation_split',{'groupId':group,'split':'release','neverTrain':True,'bankHash':bank['bankHash']},digest(group))
    return register_suite(control,{'target':TARGET,'tasks':tasks})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','verify'))
    parser.add_argument('--directory',default='LLM/runtime-data/preparation/self-code-policy-bank-v1')
    parser.add_argument('--data-dir');parser.add_argument('--profile-id');args=parser.parse_args()
    if args.action=='prepare':result=prepare(args.directory)
    else:
        require(args.data_dir and args.profile_id,'SELF_CODE_SANDBOX_PROFILE_REQUIRED')
        from LLM.service import Control
        control=Control(args.data_dir,worker=False,execution_scope='pc_validation')
        try:result=verify_references(control,checked_bank(Path(args.directory)/'bank.json'),args.profile_id)
        finally:control.store.close()
        atomic_json(Path(args.directory)/'reference-verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('tasks','mandatoryCases','results')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
