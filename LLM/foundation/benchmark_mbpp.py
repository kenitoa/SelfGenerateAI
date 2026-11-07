"""Official MBPP test IDs, complete literal oracles, fixed before any generation."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import re

from LLM.core import Fault,atomic_json,digest,file_hash,now,require
from .benchmark import _fetch,whole_program_prompt

REVISION='f46ca8374b4cddef97ca4208ad986049d74d296a'
PROTOCOL='whole-python-program-v1'
SOURCE_SCHEMA=2
DATASET_LICENSE_URL='https://creativecommons.org/licenses/by/4.0/'
SOURCE_FILES={'LICENSE','README.md','sanitized-mbpp.json','ROOT_README.md','DATASET_LICENSE_CC-BY-4.0.txt'}


def _license_evidence(root,intake=None):
    require('All datasets in this repository are released under the CC BY 4.0 International'
        in (root/'ROOT_README.md').read_text('utf-8'),'BENCHMARK_DATASET_LICENSE_REQUIRED')
    require('Attribution 4.0 International' in (root/'DATASET_LICENSE_CC-BY-4.0.txt').read_text('utf-8'),
        'BENCHMARK_DATASET_LICENSE_REQUIRED')
    require('Apache License' in (root/'LICENSE').read_text('utf-8'),'BENCHMARK_CODE_LICENSE_REQUIRED')
    if intake is not None:
        require(intake.get('schemaVersion')==SOURCE_SCHEMA and intake.get('license')=='CC-BY-4.0'
            and intake.get('codeLicense')=='Apache-2.0' and intake.get('licenseUrl')==DATASET_LICENSE_URL
            and set(intake.get('files',{}))==SOURCE_FILES
            and intake.get('licenseDeclarationHash')==intake['files']['ROOT_README.md'],
            'BENCHMARK_DATASET_LICENSE_REQUIRED')


def collect(destination,revision=REVISION):
    from .acceptance import require_local_path
    root=require_local_path(destination);root.mkdir(parents=True,exist_ok=True)
    require(re.fullmatch('[0-9a-f]{40}',revision),'INVALID_SOURCE_REVISION')
    existing=json.loads((root/'intake.json').read_text('utf-8')) if (root/'intake.json').exists() else None
    require(existing is None or existing.get('schemaVersion')==SOURCE_SCHEMA,
        'BENCHMARK_SOURCE_VERSION_REQUIRES_NEW_DIRECTORY')
    base='https://raw.githubusercontent.com/google-research/google-research/'+revision+'/'
    files={}
    for upstream,name in (('LICENSE','LICENSE'),('mbpp/README.md','README.md'),('mbpp/sanitized-mbpp.json','sanitized-mbpp.json'),('README.md','ROOT_README.md')):
        path=root/name;raw=_fetch(base+upstream)
        if path.exists():require(path.read_bytes()==raw,'BENCHMARK_HASH_MISMATCH')
        else:path.write_bytes(raw)
        files[name]=file_hash(path)
    license_path=root/'DATASET_LICENSE_CC-BY-4.0.txt'
    raw=_fetch(DATASET_LICENSE_URL+'legalcode.txt')
    if license_path.exists():require(license_path.read_bytes()==raw,'BENCHMARK_HASH_MISMATCH')
    else:license_path.write_bytes(raw)
    files[license_path.name]=file_hash(license_path)
    _license_evidence(root)
    license_text=(root/'LICENSE').read_text('utf-8')
    require('Apache License' in license_text and 'Version 2.0' in license_text,'BENCHMARK_LICENSE_REQUIRED')
    rows=json.loads((root/'sanitized-mbpp.json').read_text('utf-8'))
    require(len(rows)==427 and len({row['task_id'] for row in rows})==427,'BENCHMARK_INVENTORY_CHANGED')
    report={'schemaVersion':SOURCE_SCHEMA,'source':'google-research/google-research/mbpp','revision':revision,
        'license':'CC-BY-4.0','codeLicense':'Apache-2.0',
        'licenseScope':'Pinned repository root README explicitly licenses datasets under CC BY 4.0; LICENSE applies to source files.',
        'licenseUrl':DATASET_LICENSE_URL,'licenseDeclarationUrl':base+'README.md',
        'licenseDeclarationHash':files['ROOT_README.md'],
        'attribution':{'title':'Mostly Basic Python Problems','creator':'Austin et al., Google Research, 2021',
            'sourceUrl':base+'mbpp/sanitized-mbpp.json','licenseUrl':DATASET_LICENSE_URL,
            'changes':'Original dataset bytes preserved. Evaluation subset selection and literal-JSON assertion translation are separately reported.'},
        'files':files,'taskCount':427,'officialTestIdRange':[11,510],
        'neverTrain':True,'referenceAnswersNeverTrain':True,'modelOutputsObserved':False,
        'scope':'sanitized official-test literal-JSON complete-oracle subset; not full official MBPP score',
        'externalPretrainingContaminationUnknown':True,'observedAt':now(),'nasModified':False}
    if existing is not None:
        _license_evidence(root,existing)
        require(existing.get('revision')==revision and existing.get('source')==report['source']
            and existing.get('files')==files,'BENCHMARK_HASH_MISMATCH')
        return existing
    atomic_json(root/'intake.json',report)
    return report


def literal_cases(row):
    """Translate every assertion, or exclude the entire task. Never exec code."""
    require(isinstance(row.get('test_list'),list) and 1<=len(row['test_list'])<=100,'BENCHMARK_ORACLE_UNSUPPORTED')
    cases=[];entry=None
    for source in row['test_list']:
        require(isinstance(source,str) and len(source.encode())<=32768,'BENCHMARK_ORACLE_UNSUPPORTED')
        module=ast.parse(source)
        require(len(module.body)==1 and isinstance(module.body[0],ast.Assert),'BENCHMARK_ORACLE_UNSUPPORTED')
        expression=module.body[0].test;truthiness=None
        if isinstance(expression,ast.Compare) and len(expression.ops)==1 and isinstance(expression.ops[0],ast.Eq):
            call=expression.left;expected=ast.literal_eval(expression.comparators[0])
        elif isinstance(expression,ast.UnaryOp) and isinstance(expression.op,ast.Not):
            call=expression.operand;truthiness=False
        else:call=expression;truthiness=True
        require(isinstance(call,ast.Call) and isinstance(call.func,ast.Name)
            and not call.func.id.startswith('_') and all(k.arg is not None for k in call.keywords),
            'BENCHMARK_ORACLE_UNSUPPORTED')
        require(entry is None or entry==call.func.id,'BENCHMARK_ORACLE_UNSUPPORTED')
        entry=call.func.id
        case={'function':entry,'args':[ast.literal_eval(a) for a in call.args],
              'kwargs':{k.arg:ast.literal_eval(k.value) for k in call.keywords}}
        case.update(expected=expected) if truthiness is None else case.update(expectedTruthiness=truthiness)
        require(json.loads(json.dumps(case,allow_nan=False))==case,'BENCHMARK_ORACLE_UNSUPPORTED')
        cases.append(case)
    tree=ast.parse(row['code'])
    definitions=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name==entry]
    require(len(definitions)==1,'BENCHMARK_ORACLE_UNSUPPORTED')
    signature='def '+entry+'('+ast.unparse(definitions[0].args)+'):'
    return cases,signature


def register(control,directory,profile_id):
    root=Path(directory).resolve();intake=json.loads((root/'intake.json').read_text('utf-8'))
    require(intake.get('source')=='google-research/google-research/mbpp' and intake.get('neverTrain') is True
        and intake.get('referenceAnswersNeverTrain') is True and intake.get('license')=='CC-BY-4.0'
        and intake.get('schemaVersion')==SOURCE_SCHEMA
        and re.fullmatch('[0-9a-f]{40}',intake.get('revision',''))
        and set(intake.get('files',{}))==SOURCE_FILES,'BENCHMARK_SOURCE_REQUIRED')
    for name,expected in intake['files'].items():require(file_hash(root/name)==expected,'BENCHMARK_HASH_MISMATCH')
    _license_evidence(root,intake)
    from LLM.verification.broker import pinned
    profile=control.store.get('verifier_profile',profile_id)
    require(pinned(profile) and profile.get('language')=='python','PINNED_SANDBOX_IMAGE_REQUIRED')
    rows=json.loads((root/'sanitized-mbpp.json').read_text('utf-8'))
    require(len(rows)==427 and len({r['task_id'] for r in rows})==427,'BENCHMARK_INVENTORY_CHANGED')
    tasks=[];excluded=[]
    for row in sorted(rows,key=lambda item:item['task_id']):
        if not 11<=row['task_id']<=510:
            excluded.append({'taskId':row['task_id'],'reasonCode':'NOT_OFFICIAL_TEST_SPLIT'});continue
        try:cases,signature=literal_cases(row)
        except (Fault,ValueError,TypeError,SyntaxError,KeyError,RecursionError) as error:
            excluded.append({'taskId':row['task_id'],'reasonCode':getattr(error,'code','BENCHMARK_ORACLE_UNSUPPORTED')});continue
        task_hash=digest(row)
        identity='mbpp-case-v2-'+digest([SOURCE_SCHEMA,intake['revision'],intake['files'],row['task_id'],cases,profile['id'],profile['image'],PROTOCOL])[:32]
        body={'profileId':profile_id,'cases':cases,'independent':True,'targetOrigin':'human','split':'release',
            'neverTrain':True,'sourceRevision':intake['revision'],'sourceTaskHash':task_hash,'sourceFiles':intake['files'],
            'verifierProfileHash':digest(profile),'profileImageDigest':profile['image'],
            'oracleScope':'all-official-assertions-literal-json-values','pythonTruthinessPreserved':True,
            'benchmarkSource':intake['source'],'protocol':PROTOCOL,
            'datasetLicense':'CC-BY-4.0','licenseDeclarationHash':intake['licenseDeclarationHash'],
            'sourceSchemaVersion':SOURCE_SCHEMA}
        fixture=control.store.get('fixture',identity,{})
        if fixture:require(all(fixture.get(k)==v for k,v in body.items()),'BENCHMARK_FIXTURE_CHANGED')
        else:fixture=control.store.put('fixture',body,identity)
        task_id='MBPP/'+str(row['task_id']);group='benchmark:mbpp:'+task_id
        tasks.append({'id':task_id,'group':group,'lineageGroups':[group,'repository:https://github.com/google-research/google-research'],
            'prompt':whole_program_prompt(row['prompt']+'\nRequired function signature:\n'+signature),
            'codePrefix':'','fixtureId':fixture['id'],'language':'python','mode':'code','maxTokens':128})
    return control.store.put('foundation_benchmark_intake',{'source':intake['source'],'revision':intake['revision'],
        'license':intake['license'],'codeLicense':intake['codeLicense'],
        'sourceSchemaVersion':SOURCE_SCHEMA,'licenseDeclarationHash':intake['licenseDeclarationHash'],
        'licenseUrl':intake['licenseUrl'],'attribution':intake['attribution'],
        'sourceFiles':intake['files'],'compatibleTaskCount':len(tasks),
        'excluded':excluded,'tasks':tasks,'neverTrain':True,'referenceAnswersNeverTrain':True,
        'state':'ready_for_suite' if len(tasks)>=100 else 'held','protocol':PROTOCOL,
        'scope':intake['scope'],'officialTestIdRange':[11,510],'selectionBeforeModelOutputs':True,
        'evaluationPerformed':False,'externalPretrainingContaminationUnknown':True,'nasModified':False},
        'mbpp-intake-v2-'+digest([SOURCE_SCHEMA,intake['revision'],intake['files'],profile['id'],profile['image'],PROTOCOL])[:24])
