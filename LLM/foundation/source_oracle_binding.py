"""Static, portable source-symbol evidence, independent of test outcomes.

Only already registered official witnesses are considered. No collected Python
is executed on the host, and no new expected answer is inferred from a target.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

from LLM.core import digest,file_hash,require
from .prepare_data import ACTOR,POLICY,EXCLUDE_PARTS,identify_license

VERSION='seed-source-symbol-binding-v1'
KIND='foundation_source_oracle_binding'
PERMISSIVE=frozenset(('MIT','Apache-2.0','BSD-2-Clause','BSD-3-Clause'))


def implementation_hash():return file_hash(Path(__file__))


def module_name(path):
    value=PurePosixPath(path)
    require(not value.is_absolute() and '..' not in value.parts and value.suffix=='.py','SOURCE_ORACLE_MODULE_INVALID')
    parts=list(value.with_suffix('').parts)
    if parts and parts[0]=='src':parts=parts[1:]
    if parts and parts[-1]=='__init__':parts=parts[:-1]
    require(parts and all(part.isidentifier() for part in parts),'SOURCE_ORACLE_MODULE_INVALID')
    return '.'.join(parts)


def _qualified(node,environment):
    if isinstance(node,ast.Name):return environment.get(node.id)
    if isinstance(node,ast.Attribute):
        prefix=_qualified(node.value,environment)
        return prefix+'.'+node.attr if prefix else None
    return None


def _writes(node):
    """Lexical bindings, excluding nested function/class bodies."""
    found=set()
    def visit(value):
        if isinstance(value,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
            found.add(value.name);return
        if isinstance(value,ast.Name) and isinstance(value.ctx,(ast.Store,ast.Del)):found.add(value.id)
        if isinstance(value,ast.Import):found.update(alias.asname or alias.name.split('.')[0] for alias in value.names)
        if isinstance(value,ast.ImportFrom):found.update(alias.asname or alias.name for alias in value.names if alias.name!='*')
        if isinstance(value,ast.ExceptHandler) and value.name:found.add(value.name)
        for child in ast.iter_child_nodes(value):visit(child)
    visit(node)
    return found


def _apply(statement,environment,module):
    if isinstance(statement,ast.Import):
        for alias in statement.names:environment[alias.asname or alias.name.split('.')[0]]=alias.name if alias.asname else alias.name.split('.')[0]
    elif isinstance(statement,ast.ImportFrom):
        if statement.level:
            package=module.split('.')[:-1]
            prefix=package[:len(package)-statement.level+1]
            base='.'.join([*prefix,*((statement.module or '').split('.') if statement.module else [])]) if statement.level<=len(package) else ''
        else:base=statement.module or ''
        for alias in statement.names:
            if alias.name=='*':
                for name in list(environment):environment[name]=None
            else:environment[alias.asname or alias.name]=base+'.'+alias.name if base else None
    elif isinstance(statement,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
        environment[statement.name]=module+'.'+statement.name
    elif isinstance(statement,(ast.Assign,ast.AnnAssign)):
        targets=statement.targets if isinstance(statement,ast.Assign) else [statement.target]
        value=_qualified(statement.value,environment) if statement.value else None
        for target in targets:
            for name in _writes(target):environment[name]=value if isinstance(target,ast.Name) else None
    else:
        for name in _writes(statement):environment[name]=None


def _prefix(body,environment,module,line):
    for statement in body:
        if statement.lineno>line:break
        if statement.end_lineno<line:
            _apply(statement,environment,module);continue
        if isinstance(statement,ast.Assert):return
        # Only the lexical branch containing this exact assertion is followed.
        # Earlier conditional writes remain ambiguous rather than being guessed.
        blocks=[]
        for field in ('body','orelse','finalbody'):
            block=getattr(statement,field,None)
            if isinstance(block,list):blocks.append(block)
        for handler in getattr(statement,'handlers',[]):blocks.append(handler.body)
        for block in blocks:
            if any(child.lineno<=line<=child.end_lineno for child in block):
                _prefix(block,environment,module,line);return
        for name in _writes(statement):environment[name]=None


def resolve_binding(target_text,target_path,function_name,target_span,test_text,test_path,reference):
    """Resolve imports/aliases/shadowing without running either source module."""
    target_module=module_name(target_path);test_module=module_name(test_path)
    target_tree=ast.parse(target_text);test_tree=ast.parse(test_text)
    definitions=[node for node in target_tree.body if isinstance(node,ast.FunctionDef) and node.name==function_name]
    require(len(definitions)==1,'SOURCE_ORACLE_TARGET_AMBIGUOUS')
    target=definitions[0]
    require(target.lineno==target_span['startLine'] and target.end_lineno==target_span['endLine']
            and digest(ast.get_source_segment(target_text,target).encode())==target_span['targetSha256'],
            'SOURCE_ORACLE_TARGET_CHANGED')
    expected=target_module+'.'+function_name
    target_environment={}
    for statement in target_tree.body:_apply(statement,target_environment,target_module)
    witness=next((node for node in ast.walk(test_tree) if isinstance(node,ast.Assert)
        and node.lineno==reference['startLine'] and node.end_lineno==reference['endLine']),None)
    require(witness is not None and ast.get_source_segment(test_text,witness)==reference['assertion'],
            'SOURCE_TEST_REFERENCE_MISMATCH')
    environment={}
    for statement in test_tree.body:_apply(statement,environment,test_module)
    parents={child:node for node in ast.walk(test_tree) for child in ast.iter_child_nodes(node)}
    scopes=[];current=parents.get(witness)
    while current is not None:
        if isinstance(current,(ast.FunctionDef,ast.AsyncFunctionDef,ast.Lambda)):scopes.append(current)
        current=parents.get(current)
    for scope in reversed(scopes):
        names=set()
        for statement in getattr(scope,'body',[]):names.update(_writes(statement))
        arguments=scope.args
        names.update(arg.arg for arg in [*arguments.posonlyargs,*arguments.args,*arguments.kwonlyargs])
        if arguments.vararg:names.add(arguments.vararg.arg)
        if arguments.kwarg:names.add(arguments.kwarg.arg)
        globals_={name for node in getattr(scope,'body',[]) if isinstance(node,ast.Global) for name in node.names}
        for name in names-globals_:environment[name]=None
        _prefix(scope.body,environment,test_module,witness.lineno)
    comparison=witness.test
    call=comparison.left if isinstance(comparison,ast.Compare) and len(comparison.ops)==1 and isinstance(comparison.ops[0],ast.Eq) else None
    observed=_qualified(call.func,environment) if isinstance(call,ast.Call) else None
    target_effective=target_environment.get(function_name)==expected and not target.decorator_list
    matched=target_effective and observed==expected
    state='matched' if matched else 'mismatch' if observed else 'unresolved'
    reason=None if matched else 'SOURCE_ORACLE_SYMBOL_MISMATCH' if observed else 'SOURCE_ORACLE_CALLABLE_UNRESOLVED'
    if not target_effective:state='unresolved';reason='SOURCE_ORACLE_TARGET_BINDING_UNRESOLVED'
    return {'state':state,'reasonCode':reason,'expectedQualifiedName':expected,'observedQualifiedName':observed,
        'targetModule':target_module,'targetPath':target_path,'targetStartLine':target.lineno,
        'targetEndLine':target.end_lineno,'testModule':test_module,'testPath':test_path,
        'assertionStartLine':witness.lineno,'assertionEndLine':witness.end_lineno,
        'lexicalScopes':[scope.name for scope in reversed(scopes)],'resolutionMethod':'static_import_alias_lexical_scope_v1',
        'testOutcomeConsulted':False,'targetExecuted':False}


def _authority(control,preparation,example,version):
    rag=control.bridge.store
    source=rag.get('source',version['sourceId']);rights=rag.get('source_rights_receipt',source['rightsReceiptId'])
    document=rag.get('document',version['documentId'])
    provenance=version.get('provenance',{})
    require(source.get('trainingPolicy')=='allowed' and source.get('retrievalPolicy')=='approved'
        and rights.get('state')=='rights_verified' and rights.get('decision')=='allowed' and rights.get('actor')==ACTOR
        and rights.get('policy')==POLICY and rights.get('repository')==provenance.get('repository')
        and rights.get('revision')==provenance.get('revision') and rights.get('archiveSha256')==provenance.get('archiveSha256')
        and version.get('rightsReceiptId')==rights['id'] and version.get('qualityStatus')=='passed'
        and document.get('currentVersionId')==version['id'] and document.get('deletedAt') is None
        and document.get('enabled',True) and document.get('trainingPolicy')!='denied'
        and document.get('retrievalPolicy')=='approved'
        and example.get('reviewedBy')==ACTOR and example.get('reviewStatus')=='approved'
        and example.get('permissionStatus','allowed')=='allowed'
        and example.get('approvalKind')=='delegated_source_context_review','SOURCE_ORACLE_AUTHORITY_REQUIRED')
    span=version.get('sourceSpan',{});target=example.get('sourceSpan',{})
    require(span.get('path')==target.get('path') and span.get('sourceFileSha256')==target.get('sourceFileSha256')
        and span.get('startLine',0)<=target.get('startLine',-1)<=target.get('endLine',-1)<=span.get('endLine',0),
        'SOURCE_ORACLE_TARGET_CHANGED')
    reference=example['independentVerification']['source']
    fixture=control.store.get('fixture',example['independentVerification']['fixtureId'])
    require(fixture.get('sourceEvidence')==reference,'SOURCE_TEST_REFERENCE_MISMATCH')
    return rights,fixture


def _identity(preparation,example,version,rights):
    return digest([VERSION,implementation_hash(),preparation['sftCorpusHash'],example['id'],version['id'],
        digest(example['independentVerification']['source']),digest(rights)])


def _blob(rag,checksum,maximum):
    require(isinstance(checksum,str) and re.fullmatch('[0-9a-f]{64}',checksum),'SOURCE_ORACLE_BLOB_INVALID')
    path=rag.root/'blobs'/checksum[:2]/checksum
    require(path.is_file() and not path.is_symlink() and path.stat().st_size<=maximum,'SOURCE_ORACLE_BLOB_REQUIRED')
    raw=path.read_bytes();require(digest(raw)==checksum,'SOURCE_ORACLE_BLOB_CHANGED')
    return raw


def _validate(record,control,preparation,example,version,function_name,rights,fixture):
    reference=example['independentVerification']['source'];rag=control.bridge.store
    body={k:v for k,v in record.items() if k not in ('id','version','createdAt','updatedAt','receiptHash')}
    require(record.get('version')==1 and record.get('receiptHash')==digest(body)
        and record.get('helperVersion')==VERSION and record.get('helperHash')==implementation_hash()
        and record.get('sourceExampleHash')==digest(example) and record.get('sourceVersionId')==version['id']
        and record.get('sourceExampleId')==example['id'] and record.get('sourceCorpusId')==preparation['sftCorpusId']
        and record.get('sourceCorpusHash')==preparation['sftCorpusHash'] and record.get('preparationId')==preparation['id']
        and record.get('rightsReceiptId')==rights['id'] and record.get('rightsReceiptHash')==digest(rights)
        and record.get('referenceHash')==digest(reference) and record.get('originalFixtureId')==fixture['id']
        and record.get('originalFixtureHash')==digest(fixture)
        and record.get('targetModuleBlobHash')==example['sourceSpan']['sourceFileSha256']
        and record.get('testModuleBlobHash')==reference['sourceSha256']
        and record.get('assertionBlobHash')==example['independentVerification']['assertionBlobHash']
        and record.get('repository')==rights['repository'] and record.get('revision')==rights['revision']
        and record.get('archiveSha256')==rights['archiveSha256'] and record.get('targetPath')==example['sourceSpan']['path']
        and record.get('testPath')==reference['path']
        and record.get('neverTrain') is True and record.get('modelQualityEvidence') is False,
        'SOURCE_ORACLE_BINDING_CHANGED')
    target=_blob(rag,record['targetModuleBlobHash'],2*1024**2).decode('utf-8-sig')
    test=_blob(rag,record['testModuleBlobHash'],512*1024).decode('utf-8')
    assertion=_blob(rag,record['assertionBlobHash'],512*1024).decode('utf-8')
    require(assertion==reference['assertion'],'SOURCE_TEST_REFERENCE_MISMATCH')
    binding=resolve_binding(target,example['sourceSpan']['path'],function_name,example['sourceSpan'],test,reference['path'],reference)
    require(binding==record.get('binding') and record.get('state')==binding['state'],'SOURCE_ORACLE_BINDING_CHANGED')
    return record


def binding_evidence(control,preparation,example,version,function_name):
    rights,fixture=_authority(control,preparation,example,version)
    record=control.store.get(KIND,_identity(preparation,example,version,rights),{})
    require(record,'SOURCE_ORACLE_BINDING_EVIDENCE_REQUIRED',status=409)
    return _validate(record,control,preparation,example,version,function_name,rights,fixture)


def _source_file(package,path,expected,maximum,*,test=False):
    context=getattr(package,'_oracle_context',None)
    if context is None:
        prefix=package.infolist()[0].filename.split('/')[0]+'/'
        members={info.filename.removeprefix(prefix):info for info in package.infolist() if info.filename.startswith(prefix)}
        scopes=[str(PurePosixPath(name).parent)+'/' for name in members
            if len(PurePosixPath(name).parts)>1 and PurePosixPath(name).name.upper().startswith(('LICENSE','COPYING'))]
        context={'members':members,'scopes':scopes};package._oracle_context=context
    value=PurePosixPath(path);blocked=EXCLUDE_PARTS-({'test','tests','testing','fixtures'} if test else set())
    require(not value.is_absolute() and '..' not in value.parts and value.suffix=='.py'
        and not any(part.lower() in blocked for part in value.parts)
        and not any(path.startswith(scope) for scope in context['scopes']),'SOURCE_ORACLE_FILE_SCOPE_INVALID')
    info=context['members'].get(path)
    require(info is not None and not info.is_dir() and info.file_size<=maximum
        and (info.external_attr>>16)&0o170000!=0o120000,'SOURCE_ORACLE_FILE_INVALID')
    raw=package.read(info);require(digest(raw)==expected,'SOURCE_ORACLE_FILE_CHANGED')
    text=raw.decode('utf-8-sig')
    for declaration in re.findall(r'SPDX-License-Identifier:\s*([^\r\n*]+)',text):
        require(all(part in PERMISSIVE|{'OR','AND','(',')'} for part in declaration.replace('(',' ( ').replace(')',' ) ').split()),
            'SOURCE_ORACLE_FOREIGN_FILE_LICENSE')
    return raw


def prepare_oracle_evidence(control,rag,preparation_id):
    """Root-invoked append-only evidence preparation; no model or target runs."""
    from .corpus import rows,verify
    from .source_context_contracts import runtime_contract,verify_runtime_source_parent
    from .acceptance import require_local_path
    require(control.execution_scope=='pc_validation' and control.bridge.store is rag,'PC_VALIDATION_SCOPE_REQUIRED')
    require_local_path(control.store.root);require_local_path(rag.root)
    preparation=control.store.get('foundation_data_preparation',preparation_id)
    runtime_contract(preparation);verify_runtime_source_parent(control.store,preparation)
    control.foundation.validate_policy(control.foundation.project(preparation['projectId']))
    corpus=control.store.get('foundation_corpus',preparation['sftCorpusId']);verify(control.store,corpus)
    require(corpus['manifestHash']==preparation['sftCorpusHash'],'CORPUS_HASH_MISMATCH')
    packages={};records=[];registered=unsupported=0
    try:
        for split in ('train','dev'):
            for row in rows(control.store,corpus,split,purpose=None):
                example=rag.get('training_example',row['permissionReceipt']['exampleId'])
                if not example.get('independentVerification',{}).get('fixtureId'):continue
                registered+=1
                require(example['target']==row['output'] and example['input']==row['instruction'],'SOURCE_CONTEXT_MISMATCH')
                function=ast.parse(example['target']).body[0];require(isinstance(function,ast.FunctionDef),'SOURCE_TARGET_HASH_MISMATCH')
                verification=example['independentVerification'];fixture=control.store.get('fixture',verification['fixtureId'])
                require(fixture.get('sourceEvidence')==verification['source'],'SOURCE_TEST_REFERENCE_MISMATCH')
                from .sft_runtime_verification import literal_cases_from_assertion
                if not literal_cases_from_assertion(verification['source']['assertion'],function.name):
                    unsupported+=1;continue
                version=rag.get('version',row['versionId']);rights,fixture=_authority(control,preparation,example,version)
                identity=_identity(preparation,example,version,rights)
                existing=control.store.get(KIND,identity,{})
                if existing:
                    records.append(_validate(existing,control,preparation,example,version,function.name,rights,fixture));continue
                checksum=rights['archiveSha256']
                if checksum not in packages:
                    archive=control.store.root.parent/'sources'/rights['repository'].replace('/','--')/(rights['revision']+'.zip')
                    require_local_path(archive)
                    require(archive.is_file() and file_hash(archive)==checksum,'SOURCE_ARCHIVE_HASH_MISMATCH')
                    package=zipfile.ZipFile(archive);prefix=package.infolist()[0].filename.split('/')[0]+'/'
                    grants=[]
                    for license in rights.get('licenses',[]):
                        if license.get('licenseId') not in PERMISSIVE:continue
                        path=license['path'];require('/' not in path and '\\' not in path,'SOURCE_ORACLE_LICENSE_INVALID')
                        raw=package.read(prefix+path)
                        require(digest(raw)==license['sha256'] and identify_license(raw.decode('utf-8-sig'))==license['licenseId'],
                            'SOURCE_ORACLE_LICENSE_INVALID')
                        grants.append({k:license[k] for k in ('path','sha256','licenseId')})
                    require(grants,'SOURCE_ORACLE_LICENSE_REQUIRED');packages[checksum]=(package,grants)
                package,grants=packages[checksum];reference=example['independentVerification']['source']
                target=_source_file(package,example['sourceSpan']['path'],example['sourceSpan']['sourceFileSha256'],2*1024**2)
                test=_source_file(package,reference['path'],reference['sourceSha256'],512*1024,test=True)
                binding=resolve_binding(target.decode('utf-8-sig'),example['sourceSpan']['path'],function.name,
                    example['sourceSpan'],test.decode('utf-8'),reference['path'],reference)
                # CAS additions are validation evidence only: no document,
                # source-unit, approval or original example is rewritten.
                target_blob=control.bridge.pre._blob(target.decode('utf-8'))
                test_blob=control.bridge.pre._blob(test.decode('utf-8'))
                body={'helperVersion':VERSION,'helperHash':implementation_hash(),'actor':ACTOR,'state':binding['state'],
                    'preparationId':preparation_id,'sourceCorpusId':corpus['id'],'sourceCorpusHash':corpus['manifestHash'],
                    'sourceExampleId':example['id'],'sourceExampleHash':digest(example),'sourceVersionId':version['id'],
                    'rightsReceiptId':rights['id'],'rightsReceiptHash':digest(rights),'referenceHash':digest(reference),
                    'repository':rights['repository'],'revision':rights['revision'],'archiveSha256':checksum,
                    'targetPath':example['sourceSpan']['path'],'testPath':reference['path'],
                    'targetModuleBlobHash':target_blob,'testModuleBlobHash':test_blob,
                    'assertionBlobHash':example['independentVerification']['assertionBlobHash'],
                    'originalFixtureId':fixture['id'],'originalFixtureHash':digest(fixture),
                    'archiveLicenseEvidence':grants,'binding':binding,'originalNameOnlyReferencePreserved':True,
                    'neverTrain':True,'modelQualityEvidence':False,'hostSourceExecuted':False}
                body['receiptHash']=digest(body)
                value=control.store.put(KIND,body,identity,expected=0)
                records.append(_validate(value,control,preparation,example,version,function.name,rights,fixture))
        return {'helperVersion':VERSION,'helperHash':implementation_hash(),'records':records,
            'registeredFixtureCount':registered,'literalEligibleCount':registered-unsupported,
            'unsupportedLiteralCount':unsupported,
            'counts':{state:sum(record['state']==state for record in records) for state in ('matched','mismatch','unresolved')},
            'originalExamplesChanged':False,'modelExecutionPerformed':False}
    finally:
        for package,_ in packages.values():package.close()
