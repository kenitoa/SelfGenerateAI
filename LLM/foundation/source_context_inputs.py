"""Read-only input selection for source-context preparations, never model work."""
from contextlib import contextmanager
from itertools import groupby
import json
from pathlib import Path
import apsw
from LLM.core import Fault,digest,file_hash,require
from .source_context_contracts import source_parent_fields


class _ReadStore:
    def __init__(self,root):
        self.root=Path(root).resolve()
        self.db=apsw.Connection(str(self.root/'metadata/llm.db'),flags=apsw.SQLITE_OPEN_READONLY)
        self.db.set_busy_timeout(5000)
    def get(self,kind,identity,default=None):
        row=next(iter(self.db.execute('SELECT body FROM records WHERE kind=? AND id=?',(kind,identity))),None)
        if row:return json.loads(row[0])
        if default is not None:return default
        raise Fault('NOT_FOUND',status=404)


@contextmanager
def source_inputs(root,ready,plan,sources):
    """V2 keeps its original files. V3 reads only pinned new raw train/dev rows."""
    root=Path(root)
    receipts={}
    for path in sorted((root/'sources').glob('*/receipt.json')):
        receipt=json.loads(path.read_text('utf-8'))
        group='repository:https://github.com/'+receipt['repositoryFamily']
        if group in sources:receipts[group]=(path,receipt)
    evidence_cache={}
    def evidence(group,parent_fields=None):
        path,receipt=receipts[group]
        archive=path.parent/(receipt['revision']+'.zip')
        key=(group,receipt['revision'],receipt['archiveSha256'],digest(parent_fields))
        signature=tuple((item.stat().st_size,item.stat().st_mtime_ns) for item in (archive,path))
        if parent_fields is not None and key in evidence_cache:
            saved=evidence_cache[key]
            require(saved[0]==signature,'SOURCE_ARCHIVE_CHANGED_DURING_PREPARATION')
            return archive,dict(saved[1])
        require(file_hash(archive)==receipt['archiveSha256'],'SOURCE_ARCHIVE_HASH_MISMATCH')
        proof={'receiptHash':file_hash(path),'archiveSha256':receipt['archiveSha256'],'revision':receipt['revision']}
        if parent_fields is None:
            data=path.parent/'records.jsonl'
            require(file_hash(data)==receipt['recordsSha256'],'SOURCE_RECORDS_HASH_MISMATCH')
            proof['recordsSha256']=receipt['recordsSha256']
        else:
            proof.update(parent_fields)
            evidence_cache[key]=(signature,dict(proof))
        return archive,proof
    if plan['version']=='v2':
        def batches():
            for group,(path,receipt) in receipts.items():
                archive,proof=evidence(group)
                def records(path=path):
                    with (path.parent/'records.jsonl').open(encoding='utf-8') as handle:
                        for line in handle:yield json.loads(line)
                yield group,sources[group],archive,proof,records()
        yield None,batches()
        return
    store=_ReadStore(root/'control')
    try:
        parent=store.get('foundation_data_preparation',plan['rawPreparationId'],{})
        parent_fields=source_parent_fields(parent,project_id=ready['projectId'],tokenizer_id=ready['tokenizerId'],tokenizer_hash=ready['tokenizerHash'])
        from .corpus import resolve,rows
        corpus=resolve(store,parent['rawCorpusId'])
        require(corpus.get('state')=='ready' and corpus.get('manifestHash')==parent['rawCorpusHash']
            and corpus.get('projectId')==ready['projectId'],'SOURCE_CONTEXT_RAW_PREPARATION_CHANGED')
        selected=set(parent['selectedVersionIds']);by_source={value['sourceId']:(group,value) for group,value in sources.items()}
        def batches():
            for split in ('train','dev'):
                for source_id,grouped in groupby(rows(store,corpus,split,purpose=None),key=lambda row:row['sourceId']):
                    require(source_id in by_source,'SOURCE_CONTEXT_SOURCE_FAMILY_CHANGED')
                    group,mapped=by_source[source_id]
                    require(mapped['split']==split and group in receipts,'SOURCE_CONTEXT_SOURCE_FAMILY_CHANGED')
                    archive,proof=evidence(group,parent_fields)
                    def records(grouped=grouped,split=split,source_id=source_id,group=group):
                        for row in grouped:
                            require(row.get('versionId') in selected and row.get('sourceId')==source_id
                                and row.get('split')==split and group.lower() in {g.lower() for g in row.get('groupIds',[])},
                                'SOURCE_CONTEXT_RAW_SELECTION_CHANGED')
                            yield row
                    yield group,mapped,archive,proof,records()
        yield parent,batches()
    finally:store.db.close()
