"""Closed, versioned source-context budgets; never accept a caller token limit."""
from pathlib import Path
import re
from LLM.core import digest,file_hash,require

LEGACY_VERSION='seed-source-context-sft-v1'
V3_VERSION='seed-source-context-sft-v3'


def matches_fields(record,fields):
    # JSON type identity matters: bool/int and float/int equality is not a pin.
    return all(digest(record.get(key))==digest(value) for key,value in fields.items())


def selection(version='v2'):
    require(version in ('v2','v3'),'SOURCE_CONTEXT_VERSION_UNSUPPORTED')
    third=version=='v3'
    return {'version':version,'preparationId':'official-context-'+version,
        'staticVersion':V3_VERSION if third else LEGACY_VERSION,
        'runtimeVersion':'seed-source-context-runtime-v3' if third else 'seed-source-context-runtime-v1',
        'maximumContextTokens':512,'maximumTargetTokens':256 if third else 128,
        'qualityFile':'sft-quality-preparation-v3.json' if third else 'sft-quality-preparation-v1.json',
        'auditFile':'sft-source-context-audit-v3.json' if third else 'sft-source-context-audit-v1.json',
        'rawPreparationId':'official-python-v2' if third else None}


def stamp_fields(plan):
    if plan['version']=='v2':return {}
    contract={'schemaVersion':1,'version':V3_VERSION,'maxContextTokens':512,'maxTargetTokens':256}
    return {'preparationId':'official-context-v3','sourceContextContract':contract,
        'sourceContextContractHash':digest(contract)}


def runtime_contract(preparation=None,row=None):
    """A v3 runtime gets its budget from the stored server preparation + approval.

    The caller selects no numeric budget. verify_runtime obtains the preparation
    from its own store and checks current rights before calling the pure builder.
    Legacy rows retain the original 512/128 contract and metadata shape.
    """
    preparation=preparation or {}
    approval=(row or {}).get('approvalReceipt',{})
    third=(preparation.get('id')=='official-context-v3'
        or preparation.get('implementationVersion')==V3_VERSION
        or 'sourceContextContract' in preparation
        or approval.get('implementationVersion')==V3_VERSION
        or 'sourceContextContract' in approval)
    if third:
        from .prepare_data import ACTOR
        plan=selection('v3');fields=stamp_fields(plan)
        require(preparation.get('id')==plan['preparationId'] and preparation.get('state')=='ready'
            and preparation.get('actor')==ACTOR
            and preparation.get('preparationKind')=='delegated_source_context_review'
            and preparation.get('implementationVersion')==plan['staticVersion']
            and matches_fields(preparation,fields),
            'SOURCE_CONTEXT_PREPARATION_CONTRACT_REQUIRED')
        parent=pinned_source_parent(preparation)
        if row is not None:
            require(approval.get('implementationVersion')==plan['staticVersion']
                and matches_fields(approval,fields) and matches_fields(approval,parent),
                'SOURCE_CONTEXT_APPROVAL_CONTRACT_CHANGED')
        return plan
    require(preparation.get('implementationVersion') in (None,LEGACY_VERSION)
        and approval.get('implementationVersion') in (None,LEGACY_VERSION)
        and not any(key in preparation or key in approval for key in
                    ('sourceContextContractHash','preparationId')),
        'SOURCE_CONTEXT_PREPARATION_CONTRACT_REQUIRED')
    return selection('v2')


def runtime_evidence_fields(plan,preparation=None):
    if plan['version']=='v2':return {}
    return {**stamp_fields(plan),**pinned_source_parent(preparation or {}),
        'sourceContextContractVerifierHash':file_hash(Path(__file__))}


def verify_quality_record(plan,document,authoritative):
    """V3 registration binds actual immutable report fields, not a copied hash."""
    if plan['version']=='v2':return
    pinned_source_parent(document)
    ignored={'id','version','createdAt','updatedAt','reportHash','registeredCorpusId',
        'registeredCorpusHash','ragSnapshotId','dataPreparationId','registrationState'}
    bodies=[{key:value for key,value in record.items() if key not in ignored}
            for record in (document,authoritative)]
    require(document.get('id')==authoritative.get('id')
        and document.get('reportHash')==authoritative.get('reportHash')==digest(bodies[0])==digest(bodies[1])
        and matches_fields(document,stamp_fields(plan))
        and document.get('implementationVersion')==plan['staticVersion']
        and document.get('executionContract')==plan['staticVersion'],
        'QUALITY_PREPARATION_CHANGED')


def project_preparation_fields(preparation):
    if (preparation.get('id')=='official-context-v3'
        or preparation.get('implementationVersion')==V3_VERSION
        or 'sourceContextContract' in preparation):
        plan=runtime_contract(preparation)
        return {'activeSftPreparationId':plan['preparationId'],
            'activeRawPreparationId':preparation['rawPreparationId']}
    return {}


def pinned_source_parent(preparation):
    fields={key:preparation.get(key) for key in
        ('rawPreparationId','rawCorpusId','rawCorpusHash','rawSelectedVersionIdsHash')}
    require(fields['rawPreparationId']=='official-python-v2'
        and isinstance(fields['rawCorpusId'],str) and 1<=len(fields['rawCorpusId'])<=256
        and all(isinstance(fields[key],str) and re.fullmatch('[0-9a-f]{64}',fields[key])
                for key in ('rawCorpusHash','rawSelectedVersionIdsHash')),
        'SOURCE_CONTEXT_RAW_PREPARATION_REQUIRED')
    return fields


def source_parent_fields(record,*,project_id,tokenizer_id,tokenizer_hash):
    from .prepare_data import ACTOR
    ids=record.get('selectedVersionIds')
    require(record.get('id')=='official-python-v2' and record.get('selectionVersion')=='official-python-v2'
        and record.get('state')=='ready' and record.get('actor')==ACTOR
        and record.get('sourcePreparationId')=='official-v1' and record.get('projectId')==project_id
        and record.get('tokenizerId')==tokenizer_id and record.get('tokenizerHash')==tokenizer_hash
        and record.get('holdoutUnchanged') is True and record.get('originalRightsPreserved') is True
        and record.get('originalRagRecordsUnchanged') is True
        and isinstance(ids,list) and bool(ids) and all(isinstance(value,str) and value for value in ids)
        and len(ids)==len(set(ids)) and ids==sorted(ids)
        and record.get('selectedVersionIdsHash')==digest(ids),'SOURCE_CONTEXT_RAW_PREPARATION_REQUIRED')
    fields={'rawPreparationId':record['id'],'rawCorpusId':record.get('rawCorpusId'),
        'rawCorpusHash':record.get('rawCorpusHash'),'rawSelectedVersionIdsHash':record['selectedVersionIdsHash']}
    pinned_source_parent(fields)
    return fields


def verify_runtime_source_parent(store,preparation):
    if runtime_contract(preparation)['version']=='v2':return None
    record=store.get('foundation_data_preparation',preparation['rawPreparationId'],{})
    fields=source_parent_fields(record,project_id=preparation.get('projectId'),
        tokenizer_id=preparation.get('tokenizerId'),tokenizer_hash=preparation.get('tokenizerHash'))
    require(matches_fields(preparation,fields),'SOURCE_CONTEXT_RAW_PREPARATION_CHANGED')
    corpus=store.get('foundation_corpus',fields['rawCorpusId'],{})
    require(corpus.get('state')=='ready' and corpus.get('manifestHash')==fields['rawCorpusHash']
        and corpus.get('projectId')==preparation.get('projectId'),'SOURCE_CONTEXT_RAW_PREPARATION_CHANGED')
    return record
