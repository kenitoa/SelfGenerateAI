"""Persistent generation-before knowledge pipeline.

Public methods are synchronous, transaction bounded and safe for the API's
thread pool or the one durable worker. Models are optional local artifacts.
"""
from __future__ import annotations

import json
import copy
import hashlib
import os
import re
import shutil
import threading
import time
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

from RAG.core import DomainError, now, uid
from .embeddings import LocalEncoder
from .processing import decode, digest, logical_name, make_chunks, normalize, quality, search_terms, text_language, validate_file
from .structure import analyze, chunk_structure, related
from .scoring import rating

PIPELINE_VERSION = 'seed-pre-2'


class PreService:
    def __init__(self, store):
        self.store = store
        self.lock = threading.RLock()
        self._build_lock = threading.Lock()
        self.index_root = Path(store.root) / 'indexes'
        self.index_root.mkdir(parents=True, exist_ok=True)
        (Path(store.root) / 'quarantine').mkdir(parents=True, exist_ok=True)
        self._encoder_key = None
        self._encoder = None
        self._generation_readers = defaultdict(int)
        self._search_cache = OrderedDict()
        with store.tx() as con:
            con.execute("CREATE INDEX IF NOT EXISTS pre_records_document ON records(kind,json_extract(body,'$.documentId'))")
            con.execute("CREATE INDEX IF NOT EXISTS pre_records_version ON records(kind,json_extract(body,'$.versionId'))")
            con.execute('CREATE VIRTUAL TABLE IF NOT EXISTS pre_fts USING fts5(chunk_id UNINDEXED,generation_id UNINDEXED,title,path,symbol,body,search_terms, tokenize="unicode61")')
            con.execute('CREATE TABLE IF NOT EXISTS pre_memberships(generation_id TEXT NOT NULL,chunk_id TEXT NOT NULL,vector_row INTEGER,PRIMARY KEY(generation_id,chunk_id))')
            con.execute('CREATE TABLE IF NOT EXISTS pre_active(workspace_id TEXT PRIMARY KEY,generation_id TEXT)')
            con.execute('INSERT OR IGNORE INTO pre_active(workspace_id,generation_id) VALUES (?,NULL)', ('default',))
            con.execute('CREATE TABLE IF NOT EXISTS pre_fingerprints(version_id TEXT PRIMARY KEY,language TEXT NOT NULL,text_length INTEGER NOT NULL,band0 INTEGER NOT NULL,band1 INTEGER NOT NULL,band2 INTEGER NOT NULL,band3 INTEGER NOT NULL)')
            for i in range(4):
                con.execute(f'CREATE INDEX IF NOT EXISTS pre_fingerprint_band{i} ON pre_fingerprints(language,band{i},text_length)')
        self._retire_folder_sources()

    def _retire_folder_sources(self):
        """Disable removed folder collection without deleting retained source bytes."""
        with self.store.tx():
            settings = self.store.setting('settings', {})
            if 'approvedLocalPaths' in settings:
                settings.pop('approvedLocalPaths', None)
                settings['version'] = settings.get('version', 1) + 1
                self.store.set_setting('settings', settings)
            folder_ids = {source['id'] for source in self.store.list('source') if source.get('kind') == 'local_directory'}
            changed = False
            for identifier in folder_ids:
                source = self.store.get('source', identifier)
                if source.get('collectionEnabled') is not False or source.get('retrievalPolicy') != 'denied' or source.get('trainingPolicy') != 'denied' or not source.get('archivedAt'):
                    self.store.put('source', {'collectionEnabled': False, 'retrievalPolicy': 'denied', 'trainingPolicy': 'denied', 'archivedAt': source.get('archivedAt') or now(), 'removedReason': 'FOLDER_COLLECTION_REMOVED', 'policyVersion': source.get('policyVersion', 1) + 1}, id=identifier)
                    changed = True
            for schedule in self.store.list('schedule'):
                if (schedule.get('sourceId') or schedule.get('payload', {}).get('sourceId')) in folder_ids:
                    self.store.put('schedule', {'enabled': False}, id=schedule['id'])
            for row in self.store.read("SELECT id,payload FROM jobs WHERE state IN ('queued','running','retry_wait')"):
                if json.loads(row['payload']).get('sourceId') in folder_ids:
                    self.store.cancel_job(row['id'])
            if changed:
                self._bump_policy()
                self.store.activity('folder_collection.removed', 'workspace', 'NAS 폴더 수집 제거 · 기존 자료 검색/학습 제외')

    def _all(self, kind):
        if kind == 'job':
            return [self.store.job(row['id']) for row in self.store.read('SELECT id FROM jobs ORDER BY created_at DESC,id DESC')]
        return self.store.list(kind)

    def _get(self, kind, identifier):
        if kind == 'job':
            return self.store.job(identifier)
        return self.store.get(kind, identifier)

    def _save(self, kind, body, identifier=None):
        return self.store.put(kind, body, id=identifier or body.get('id'))

    def _blob(self, raw, quarantine=False):
        raw = raw.encode('utf-8') if isinstance(raw, str) else raw
        hashed = digest(raw)
        base = Path(self.store.root) / 'quarantine' if quarantine else Path(self.store.blobs)
        target = base / hashed[:2] / hashed
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            temporary = target.with_name(target.name + '.' + uid() + '.tmp')
            with temporary.open('wb') as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        return hashed

    def _blob_read(self, hashed, quarantine=False):
        if not re.fullmatch(r'[0-9a-f]{64}', str(hashed)):
            raise DomainError('BLOB_REFERENCE_INVALID', '원문 참조 오류입니다.', 503)
        base = Path(self.store.root) / 'quarantine' if quarantine else Path(self.store.blobs)
        try:
            value = (base / hashed[:2] / hashed).read_bytes()
        except OSError as exc:
            raise DomainError('SOURCE_CONTENT_MISSING', '저장된 원문을 찾을 수 없습니다.', 503) from exc
        if digest(value) != hashed:
            raise DomainError('SOURCE_CONTENT_CORRUPT', '원문 무결성 검사에 실패했습니다.', 503)
        return value

    def encoder(self):
        config = self.store.setting('settings', {}).get('embedding') or self.store.setting('embedding', {})
        key = json.dumps(config, sort_keys=True)
        with self.lock:
            if key != self._encoder_key:
                self._encoder = LocalEncoder(config)
                self._encoder_key = key
            return self._encoder

    def readiness(self):
        encoder = self.encoder()
        active = self.store.one('SELECT generation_id FROM pre_active WHERE workspace_id=?', ('default',))
        result = {'lexicalSearch': True, 'denseSearch': bool(encoder.available), 'searchMode': 'hybrid' if encoder.available else 'lexical_only', 'degradedReason': encoder.reason, 'indexGenerationId': active['generation_id'] if active else None}
        if result['indexGenerationId']:
            try:
                manifest, _ = self._manifest(self._get('generation', result['indexGenerationId']))
            except DomainError as exc:
                result.update({'indexReady': False, 'error': exc.code})
            else:
                result['indexReady'] = True
                result['searchMode'] = manifest['searchMode'] if encoder.available or manifest['searchMode'] == 'lexical_only' else 'lexical_only'
                result['degradedReason'] = manifest.get('degradedReason') if result['searchMode'] == manifest['searchMode'] else encoder.reason
        else:
            result['indexReady'] = True
        return result

    def create_dataset(self, body):
        name = str(body.get('name', '')).strip()
        if not name or len(name) > 120:
            raise DomainError('INVALID_DATASET_NAME', '데이터셋 이름은 1~120자입니다.', 422)
        return self._save('dataset', {'name': name, 'description': str(body.get('description', ''))[:2000], 'state': 'processing', 'deletedAt': None})

    def _dataset_default(self):
        existing = next((item for item in self._all('dataset') if item.get('isDefault') and not item.get('deletedAt')), None)
        return existing or self._save('dataset', {'name': '나의 데이터', 'description': '', 'state': 'processing', 'isDefault': True, 'deletedAt': None})

    def list_datasets(self):
        return [self.dataset(item['id']) for item in self._all('dataset') if not item.get('deletedAt')]

    def dataset(self, identifier):
        item = self._get('dataset', identifier)
        if item.get('deletedAt'):
            raise DomainError('DATASET_DELETED', '삭제된 데이터셋입니다.', 404)
        docs = self.documents(identifier)
        ready = sum(doc['state'] == 'ready' for doc in docs)
        state = 'ready' if docs and ready == len(docs) else 'needs_review' if any(doc['state'] == 'needs_review' for doc in docs) else 'failed' if any(doc['state'] == 'failed' for doc in docs) else 'processing'
        return {**item, 'state': state, 'documentCount': len(docs), 'chunkCount': sum(doc.get('chunkCount', 0) for doc in docs), 'trainingExampleCount': sum(e.get('datasetId') == identifier and e.get('reviewStatus') == 'approved' for e in self._all('training_example')), 'sourceCount': len({doc['sourceId'] for doc in docs}), 'documents': docs, 'language': docs[0]['language'] if docs and len({d['language'] for d in docs}) == 1 else 'mixed', 'latestJob': next((job for job in self._all('job') if job.get('payload', {}).get('datasetId') == identifier), None)}

    def update_dataset(self, identifier, body):
        item = self._get('dataset', identifier)
        if 'name' in body:
            if not str(body['name']).strip() or len(str(body['name'])) > 120:
                raise DomainError('INVALID_DATASET_NAME', '데이터셋 이름은 1~120자입니다.', 422)
            item['name'] = str(body['name']).strip()
        if 'description' in body:
            item['description'] = str(body['description'])[:2000]
        self._save('dataset', item)
        return self.dataset(identifier)

    def delete_dataset(self, identifier):
        with self.store.tx():
            item = self._get('dataset', identifier)
            for document in self.documents(identifier):
                self.delete_document(document['id'])
            for source in self._all('source'):
                if source.get('datasetId') == identifier:
                    self._save('source', {**source, 'collectionEnabled': False, 'retrievalPolicy': 'denied', 'archivedAt': now(), 'policyVersion': source.get('policyVersion', 1) + 1})
            self._save('dataset', {**item, 'deletedAt': now()})
        self.store.activity('dataset.deleted', identifier, '데이터셋 삭제')
        return {'id': identifier, 'deleted': True}

    def documents(self, dataset_id=None):
        values = [item for item in self._all('document') if not item.get('deletedAt') and (dataset_id is None or item.get('datasetId') == dataset_id)]
        catalog = self._catalog()
        return [self._document_dto(value, catalog) for value in values]

    def _catalog(self):
        result = {kind: {item['id']: item for item in self._all(kind)} for kind in ('document', 'version', 'source', 'chunk')}
        result['activeIds'] = {row['chunk_id'] for row in self.store.read('SELECT chunk_id FROM pre_memberships WHERE generation_id=?', (self._active_id(),))}
        return result

    def _allowed_from(self, chunk, catalog):
        doc = catalog['document'].get(chunk['documentId'])
        version = catalog['version'].get(chunk['versionId'])
        source = catalog['source'].get(chunk['sourceId'])
        return bool(chunk['id'] in catalog['activeIds'] and doc and version and source and not source.get('upstreamDeleted') and not doc.get('deletedAt') and doc.get('enabled', True) and doc.get('currentVersionId') == chunk['versionId'] and doc.get('retrievalPolicy') == 'approved' and source.get('retrievalPolicy') == 'approved' and version.get('qualityStatus') == 'passed' and chunk.get('retrievalAllowed', True))

    def _document_dto(self, item, catalog=None, detail=False):
        catalog = catalog or self._catalog()
        try:
            version = catalog['version'][item['currentVersionId']]
            source = catalog['source'][item['sourceId']]
        except KeyError:
            return {**item, 'state': 'failed', 'qualityReasons': ['METADATA_REFERENCE_MISSING']}
        state = item.get('state', 'processing')
        if item.get('retrievalPolicy') == 'denied' or source.get('retrievalPolicy') == 'denied' or source.get('upstreamDeleted'):
            state = 'excluded'
        elif version.get('qualityStatus') != 'passed' or item.get('retrievalPolicy') != 'approved' or source.get('retrievalPolicy') != 'approved':
            state = 'needs_review'
        return {**item, 'state': state, 'language': version.get('language', 'unknown'), 'qualityStatus': version.get('qualityStatus'), 'qualityReasons': version.get('qualityReasons', []), 'qualityRating':version.get('qualityRating'), 'codeStructure':version.get('codeStructure') if detail else None, 'structureSummary':{key:len(version.get('codeStructure',{}).get(key,[])) for key in ('symbols','imports','calls')}, 'revision': version.get('revision'), 'source': source.get('name'), 'sourceUrl': source.get('locator', {}).get('url'), 'permissions': {'retrieval': item.get('retrievalPolicy', 'pending'), 'sourceRetrieval': source.get('retrievalPolicy', 'pending'), 'training': item.get('trainingPolicy', source.get('trainingPolicy', 'unknown'))}, 'policyVersion': item.get('policyVersion', 1), 'chunkCount': sum(chunk.get('versionId') == version['id'] and self._allowed_from(chunk, catalog) for chunk in catalog['chunk'].values())}

    def document(self, identifier):
        item = self._get('document', identifier)
        if item.get('deletedAt'):
            raise DomainError('DOCUMENT_DELETED', '삭제된 문서입니다.', 404)
        version=self._get('version',item['currentVersionId']);source=self._get('source',item['sourceId'])
        chunks=[json.loads(row['body']) for row in self.store.read("SELECT body FROM records WHERE kind='chunk' AND json_extract(body,'$.versionId')=?",(version['id'],))]
        catalog={'document':{item['id']:item},'version':{version['id']:version},'source':{source['id']:source},
            'chunk':{row['id']:row for row in chunks},'activeIds':{row['chunk_id'] for row in self.store.read(
                "SELECT m.chunk_id FROM pre_memberships m JOIN records c ON c.kind='chunk' AND c.id=m.chunk_id WHERE m.generation_id=? AND json_extract(c.body,'$.versionId')=?",(self._active_id(),version['id']))}}
        return self._document_dto(item,catalog=catalog,detail=True)

    def versions(self, identifier):
        item=self._get('document',identifier)
        if item.get('deletedAt'):raise DomainError('DOCUMENT_DELETED','DOCUMENT_DELETED',404)
        return [self._version_dto(json.loads(row['body'])) for row in self.store.read(
            "SELECT body FROM records WHERE kind='version' AND json_extract(body,'$.documentId')=? ORDER BY created_at DESC,id DESC",(identifier,))]

    def _version_dto(self, version):
        return {key: value for key, value in version.items() if key not in {'rawBlobHash', 'normalizedBlobHash', 'rawFileBlobHash', 'quarantineBlobHash'}}

    def content(self, version_id):
        version = self._get('version', version_id)
        doc = self._get('document', version['documentId'])
        source = self._get('source', doc['sourceId'])
        if source.get('upstreamDeleted') or doc.get('deletedAt') or not doc.get('enabled', True) or doc.get('retrievalPolicy') != 'approved' or source.get('retrievalPolicy') != 'approved' or version.get('qualityStatus') != 'passed':
            raise DomainError('SOURCE_NOT_APPROVED', '이 원문은 검색 승인 상태가 아닙니다.', 403)
        return self._blob_read(version['rawBlobHash']).decode('utf-8-sig')

    def chunks(self, version_id):
        self.content(version_id)
        version=self._get('version',version_id)
        return [item for item in self._all('chunk') if item.get('versionId') == version_id and item.get('chunkingVersion')==version.get('chunkingVersion')]

    def allowed_chunk(self, identifier):
        if not self.store.one('SELECT chunk_id FROM pre_memberships WHERE generation_id=? AND chunk_id=?', (self._active_id(), identifier)):
            return False
        try:
            chunk = self._get('chunk', identifier)
            doc = self._get('document', chunk['documentId'])
            source = self._get('source', doc['sourceId'])
            version = self._get('version', chunk['versionId'])
        except DomainError:
            return False
        return bool(not source.get('upstreamDeleted') and not doc.get('deletedAt') and doc.get('enabled', True) and doc.get('currentVersionId') == chunk['versionId'] and doc.get('retrievalPolicy') == 'approved' and source.get('retrievalPolicy') == 'approved' and version.get('qualityStatus') == 'passed' and chunk.get('retrievalAllowed', True))

    def review(self, document_id, body):
        with self.store.tx():
            doc = self._get('document', document_id)
            version = self._get('version', doc['currentVersionId'])
            if body.get('versionId') != version['id'] or body.get('expectedPolicyVersion') != doc.get('policyVersion', 1):
                raise DomainError('REVIEW_CONFLICT', '문서 또는 정책이 변경되었습니다. 다시 확인하세요.', 409)
            if body.get('decision') not in {'approve', 'exclude'}:
                raise DomainError('INVALID_REVIEW', 'approve 또는 exclude 결정이 필요합니다.', 422)
            if body['decision'] == 'approve' and version.get('qualityStatus') == 'quarantined':
                raise DomainError('SECRET_DETECTED', '격리된 원문은 정정본 업로드 후 다시 검사해야 합니다.', 409)
            retrieval = body.get('retrievalPolicy', 'approved' if body['decision'] == 'approve' else 'denied')
            training = body.get('trainingPolicy', doc.get('trainingPolicy', 'unknown'))
            if retrieval not in {'pending', 'approved', 'denied'} or training not in {'unknown', 'allowed', 'denied'}:
                raise DomainError('INVALID_POLICY', '지원하지 않는 권한 정책입니다.', 422)
            doc.update({'retrievalPolicy': retrieval, 'trainingPolicy': training, 'policyVersion': doc.get('policyVersion', 1) + 1, 'reviewReason': str(body.get('reason', ''))[:1000], 'reviewedAt': now(), 'state': 'processing' if retrieval == 'approved' else 'excluded'})
            if body['decision'] == 'approve' and version.get('qualityStatus') == 'needs_review':
                if 'PARSER_SIZE_LIMIT' in version.get('qualityReasons', []):
                    raise DomainError('PARSER_SIZE_LIMIT', '작은 파일로 나눈 뒤 다시 업로드하세요.', 409)
                version.update({'qualityStatus': 'passed', 'reviewedAt': now(), 'qualityReviewApproved': True})
                self._save('version', version)
            self._save('document', doc)
            self._bump_policy()
            job = self.store.enqueue('index', {'documentIds': [document_id], 'datasetId': doc['datasetId']}, key=f"review:{document_id}:{doc['policyVersion']}")
        self.store.activity('document.reviewed', document_id, '문서 검토 반영')
        return {**self.document(document_id), 'jobId': job['id']}

    def delete_document(self, identifier):
        with self.store.tx():
            doc = self._get('document', identifier)
            self._save('document', {**doc, 'deletedAt': now(), 'enabled': False, 'retrievalPolicy': 'denied', 'state': 'deleted', 'policyVersion': doc.get('policyVersion', 1) + 1})
            self._bump_policy()
            # Revocation is immediate; old generation files may remain for recovery.
            for snapshot in self._all('training_snapshot'):
                lineage = json.dumps(snapshot, ensure_ascii=False)
                if identifier in lineage:
                    self._save('training_snapshot', {**snapshot, 'status': 'revoked', 'revokedAt': now()})
        self.store.activity('document.deleted', identifier, '문서 검색 제외')
        return {'id': identifier, 'deleted': True}

    def _bump_policy(self):
        self.store.set_setting('policyVersion', int(self.store.setting('policyVersion', 0)) + 1)

    def sources(self):
        return [s for s in self._all('source') if not s.get('archivedAt')]

    def create_source(self, body):
        kind = body.get('kind', 'web_document')
        if kind == 'local_directory':
            raise DomainError('FOLDER_COLLECTION_REMOVED', 'NAS/로컬 폴더 수집은 지원하지 않습니다.', 422)
        if kind not in {'upload', 'web_document', 'git_repository'}:
            raise DomainError('INVALID_SOURCE_KIND', '지원하지 않는 소스 종류입니다.', 422)
        name = str(body.get('name', '')).strip()
        if not name or len(name) > 120:
            raise DomainError('INVALID_SOURCE_NAME', '소스 이름은 1~120자입니다.', 422)
        dataset = self._get('dataset', body['datasetId']) if body.get('datasetId') else self._dataset_default()
        locator = body.get('locator') or ({'url': body['url']} if body.get('url') else {})
        if kind in {'web_document', 'git_repository'}:
            from urllib.parse import urlsplit
            parsed = urlsplit(str(locator.get('url', '')))
            if parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.username or parsed.password:
                raise DomainError('INVALID_SOURCE_URL', '올바른 http/https 소스 주소가 필요합니다.', 422)
        retrieval, training = body.get('retrievalPolicy', 'pending'), body.get('trainingPolicy', 'unknown')
        if retrieval not in {'pending', 'approved', 'denied'} or training not in {'unknown', 'allowed', 'denied'}:
            raise DomainError('INVALID_POLICY', '지원하지 않는 권한 정책입니다.', 422)
        source = self._save('source', {'kind': kind, 'name': name, 'datasetId': dataset['id'], 'locator': locator, 'url': locator.get('url', ''), 'collectionEnabled': body.get('collectionEnabled', True) is True, 'retrievalPolicy': retrieval, 'trainingPolicy': training, 'policyVersion': 1, 'includePaths': body.get('includePaths', []), 'excludePaths': body.get('excludePaths', []), 'scheduleId': body.get('scheduleId'), 'licenseId': body.get('licenseId'), 'licenseUrl': body.get('licenseUrl'), 'credentialRef': body.get('credentialRef'), 'archivedAt': None, 'lastSyncedAt': None})
        self.store.activity('source.created', source['id'], '소스 등록')
        return source

    def update_source(self, identifier, body):
        with self.store.tx():
            source = self._get('source', identifier)
            if source.get('kind') == 'local_directory':
                raise DomainError('FOLDER_COLLECTION_REMOVED', '제거된 폴더 소스는 다시 활성화할 수 없습니다.', 422)
            allowed = {'name', 'collectionEnabled', 'retrievalPolicy', 'trainingPolicy', 'includePaths', 'excludePaths', 'scheduleId', 'licenseId', 'licenseUrl', 'credentialRef'}
            updates = {key: value for key, value in body.items() if key in allowed}
            if updates.get('retrievalPolicy', source['retrievalPolicy']) not in {'pending', 'approved', 'denied'} or updates.get('trainingPolicy', source['trainingPolicy']) not in {'unknown', 'allowed', 'denied'}:
                raise DomainError('INVALID_POLICY', '지원하지 않는 권한 정책입니다.', 422)
            source.update(updates)
            if 'retrievalPolicy' in updates or 'trainingPolicy' in updates:
                source.update({'reviewReason': str(body.get('reason', ''))[:2000], 'reviewedAt': now(), 'reviewedBy': 'administrator'})
            source['policyVersion'] = source.get('policyVersion', 1) + 1
            self._save('source', source)
            self._bump_policy()
            if 'retrievalPolicy' in updates:
                self.store.enqueue('index', {'sourceId': identifier}, key=f"source-policy:{identifier}:{source['policyVersion']}")
        self.store.activity('source.updated', identifier, '소스 정책 갱신')
        return source

    def delete_source(self, identifier, purge=False):
        with self.store.tx():
            source = self._get('source', identifier)
            self._save('source', {**source, 'collectionEnabled': False, 'archivedAt': now(), 'scheduleId': None})
            for schedule in self._all('schedule'):
                if schedule.get('sourceId') == identifier or schedule.get('payload', {}).get('sourceId') == identifier:
                    self._save('schedule', {**schedule, 'enabled': False})
            if purge:
                for document in self.documents():
                    if document['sourceId'] == identifier:
                        self.delete_document(document['id'])
        self.store.activity('source.archived', identifier, '수집 대상 제거 · 자료 유지' if not purge else '소스 자료 검색 제외')
        return {'id': identifier, 'collectionEnabled': False, 'documentsPurged': bool(purge)}

    def upload(self, files, dataset_id=None, source_id=None, mode='replace_snapshot', retrieval_consent=True, upload_keys=None):
        if mode not in {'replace_snapshot', 'append'}:
            raise DomainError('INVALID_UPLOAD_MODE', 'replace_snapshot 또는 append가 필요합니다.', 422)
        if not files or len(files) > 10 or sum(len(raw) for _, raw in files) > 25 * 1024 * 1024:
            raise DomainError('UPLOAD_TOO_LARGE', '최대 10개 파일, 전체 25 MiB입니다.', 413)
        if upload_keys is not None and len(upload_keys) != len(files):
            raise DomainError('INVALID_UPLOAD_KEYS', '파일과 uploadKey 개수가 같아야 합니다.', 422)
        prepared = [validate_file(name, raw) for name, raw in files]
        return self._register(prepared, dataset_id, source_id, mode, retrieval_consent, upload_keys)

    def ingest_source_files(self, source_id, files, revision=None):
        # Collector quotas differ from interactive upload quotas and are enforced
        # by collectors; each individual parsed document still obeys safety limits.
        source = self._get('source', source_id)
        if source.get('kind') == 'local_directory':
            raise DomainError('FOLDER_COLLECTION_REMOVED', '제거된 폴더 소스에는 자료를 추가할 수 없습니다.', 422)
        prepared = [validate_file(name, raw, allow_json_document=True) for name, raw in files]
        if not prepared:
            with self.store.tx():
                removed = []
                for doc in self.documents(source['datasetId']):
                    if doc['sourceId'] == source_id:
                        self.delete_document(doc['id'])
                        removed.append(doc['id'])
                job = self.store.enqueue('index', {'sourceId': source_id}, key=f'empty-source:{source_id}:{revision or uid()}') if removed else None
            return {'sourceId': source_id, 'documentIds': [], 'removedDocumentIds': removed, 'deduplicated': not removed, 'jobId': job['id'] if job else None}
        return self._register(prepared, source['datasetId'], source_id, 'replace_snapshot', source['retrievalPolicy'] == 'approved', [file['name'] for file in prepared], revision=revision, replace_source=True)

    def _register(self, prepared, dataset_id, source_id, mode, consent, upload_keys, revision=None, replace_source=False):
        with self.lock, self.store.tx():
            source = self._get('source', source_id) if source_id else None
            dataset = self._get('dataset', dataset_id or source['datasetId']) if dataset_id or source else self._dataset_default()
            if dataset.get('deletedAt'):
                raise DomainError('DATASET_DELETED', '삭제된 데이터셋입니다.', 404)
            if source and source['datasetId'] != dataset['id']:
                raise DomainError('SOURCE_DATASET_MISMATCH', '소스와 데이터셋이 다릅니다.', 409)
            if not source:
                source = next((s for s in self._all('source') if s.get('kind') == 'upload' and s.get('datasetId') == dataset['id'] and not s.get('archivedAt')), None)
                if not source:
                    source = self.create_source({'kind': 'upload', 'name': '로컬 업로드', 'datasetId': dataset['id'], 'retrievalPolicy': 'approved' if consent else 'pending'})
            if not source.get('collectionEnabled', True):
                raise DomainError('SOURCE_PAUSED', '수집 중지된 소스입니다.', 409)
            document_ids, version_ids, reused, unit_ids = [], [], [], []
            keys = upload_keys or [Path(item['name']).name for item in prepared]
            if len(set(keys)) != len(keys):
                raise DomainError('DUPLICATE_UPLOAD_KEY', '요청 안의 uploadKey는 고유해야 합니다.', 422)
            for file, unit_key in zip(prepared, keys):
                unit_key = logical_name(unit_key)
                unit = next((u for u in self._all('source_unit') if u['sourceId'] == source['id'] and u['unitKey'] == unit_key), None)
                if unit and unit.get('currentSnapshotHash') == file['hash']:
                    existing_ids = [d['id'] for d in self._all('document') if d.get('sourceUnitId') == unit['id'] and not d.get('deletedAt')]
                    document_ids.extend(existing_ids)
                    reused.extend(existing_ids)
                    continue
                raw_hash = self._blob(file['raw'])
                unit = self._save('source_unit', {**(unit or {}), 'sourceId': source['id'], 'unitKey': unit_key, 'rawBlobHash': raw_hash, 'currentSnapshotHash': file['hash'], 'revision': revision, 'mode': mode})
                unit_ids.append(unit['id'])
                seen_keys = set()
                for entry in file['records']:
                    external_key = unit_key + ('/' + entry['key'] if entry['key'] else '')
                    seen_keys.add(external_key)
                    doc = next((d for d in self._all('document') if d.get('sourceId') == source['id'] and d.get('externalKey') == external_key), None)
                    old_version = None
                    if doc:
                        old_version = self._get('version', doc['currentVersionId'])
                        if old_version['contentHash'] == entry['recordHash'] and not doc.get('deletedAt'):
                            document_ids.append(doc['id'])
                            reused.append(doc['id'])
                            continue
                    raw_record = entry.get('rawRecord', file['text'])
                    status, reasons = quality(entry['text'], file['oversized'])
                    quarantine_hash = self._blob(raw_record, quarantine=True) if status == 'quarantined' else None
                    normalized, extracted_title = normalize(entry['text'], entry['language'])
                    version_id = uid()
                    doc = self._save('document', {**(doc or {}), 'name': extracted_title or entry['title'], 'datasetId': dataset['id'], 'sourceId': source['id'], 'sourceUnitId': unit['id'], 'externalKey': external_key, 'path': file['name'], 'enabled': True, 'deletedAt': None, 'currentVersionId': version_id, 'retrievalPolicy': 'approved' if consent else 'pending', 'trainingPolicy': doc.get('trainingPolicy', source.get('trainingPolicy', 'unknown')) if doc else source.get('trainingPolicy', 'unknown'), 'policyVersion': (doc or {}).get('policyVersion', 0) + 1, 'state': 'needs_review' if status != 'passed' or not consent else 'processing'})
                    previous_id = old_version['id'] if old_version and old_version.get('documentId') == doc['id'] else None
                    version = self._save('version', {'id': version_id, 'documentId': doc['id'], 'sourceId': source['id'], 'rawBlobHash': self._blob(raw_record), 'rawFileBlobHash': raw_hash, 'normalizedBlobHash': self._blob(normalized), 'quarantineBlobHash': quarantine_hash, 'contentHash': entry['recordHash'], 'normalizedHash': digest(normalized), 'textLength': len(normalized), 'revision': revision or file['hash'], 'language': entry['language'], 'textLanguage': text_language(normalized), 'qualityStatus': status, 'qualityReasons': reasons, 'parserVersion': PIPELINE_VERSION, 'licenseId': source.get('licenseId'), 'supersedesId': previous_id, 'recordPositions': entry['positions'], 'htmlExtracted': entry['language'] == 'html'})
                    document_ids.append(doc['id'])
                    version_ids.append(version['id'])
                if mode == 'replace_snapshot':
                    for old_doc in self._all('document'):
                        if old_doc.get('sourceUnitId') == unit['id'] and old_doc.get('externalKey') not in seen_keys and not old_doc.get('deletedAt'):
                            self.delete_document(old_doc['id'])
            if replace_source:
                active_keys = set(keys)
                for unit in self._all('source_unit'):
                    if unit['sourceId'] == source['id'] and unit['unitKey'] not in active_keys:
                        for doc in self._all('document'):
                            if doc.get('sourceUnitId') == unit['id'] and not doc.get('deletedAt'):
                                self.delete_document(doc['id'])
            job = self.store.enqueue('ingest', {'datasetId': dataset['id'], 'sourceId': source['id'], 'documentIds': document_ids, 'versionIds': version_ids}, key='ingest:' + digest('|'.join(sorted(version_ids)))) if version_ids else None
            if version_ids:
                self._bump_policy()
            self._save('source', {**source, 'lastSyncedAt': now(), 'lastSuccessfulRevision': revision})
            result = {'uploadId': uid(), 'datasetId': dataset['id'], 'sourceId': source['id'], 'documentIds': list(dict.fromkeys(document_ids)), 'jobId': job['id'] if job else None, 'acceptedBytes': sum(len(f['raw']) for f in prepared), 'deduplicated': not version_ids, 'reusedDocumentIds': reused}
        self.store.activity('upload.accepted', dataset['id'], '원문 저장 · 처리 대기' if version_ids else '동일 원문 확인 · 재사용')
        return result

    def process_job(self, job):
        payload = job.get('payload', {})
        kind = job.get('kind', job.get('type'))
        if kind in {'index', 'reindex'}:
            for identifier in payload.get('documentIds', [d['id'] for d in self._all('document') if not d.get('deletedAt')]):
                self._check_job(job)
                doc = self._get('document', identifier)
                self._process_version(doc['currentVersionId'])
            return self.rebuild(payload, job)
        if kind not in {'ingest', 'normalize'}:
            raise DomainError('UNSUPPORTED_JOB', 'Pre에서 처리할 수 없는 작업입니다.', 422)
        versions = payload.get('versionIds', [])
        for index, version_id in enumerate(versions):
            self._check_job(job)
            self.store.update_job(job['id'], stage='chunking', processedItems=index, totalItems=len(versions), unit='documents')
            self._process_version(version_id)
        self.store.update_job(job['id'], stage='indexing', processedItems=len(versions), totalItems=len(versions), unit='documents')
        result = self.rebuild({}, job)
        review_count = sum(self._get('version', version_id)['qualityStatus'] != 'passed' for version_id in versions)
        result['needsReviewCount'] = review_count
        return result

    def _check_job(self, job):
        if not job or not job.get('id'):
            return
        current = self._get('job', job['id'])
        if current.get('state') in {'cancelling', 'cancelled'}:
            raise DomainError('JOB_CANCELLED', '작업이 중단되었습니다.', 409)
        if job.get('fencingToken') is not None and hasattr(self.store, 'assert_lease'):
            self.store.assert_lease(job['id'], job['fencingToken'])

    def _process_version(self, version_id):
        version = self._get('version', version_id)
        doc = self._get('document', version['documentId'])
        if doc.get('currentVersionId') != version_id or doc.get('deletedAt'):
            return
        if version.get('qualityStatus') != 'passed':
            self._save('document', {**doc, 'state': 'needs_review'})
            return
        existing = [c for c in self._all('chunk') if c['versionId'] == version_id]
        if existing and version.get('chunkingVersion')==PIPELINE_VERSION:
            return
        text = self._blob_read(version['normalizedBlobHash']).decode('utf-8')
        if not version.get('qualityReviewApproved') and len(text) >= 80:
            near = self._near_duplicates(version, text)
            if near:
                self._save('version', {**version, 'qualityStatus': 'needs_review', 'qualityReasons': list(dict.fromkeys(version.get('qualityReasons', []) + ['NEAR_DUPLICATE'])), 'nearDuplicateOf': near})
                self._save('document', {**doc, 'state': 'needs_review'})
                return
        encoder = self.encoder()
        analysis=analyze(text,'tsx' if doc['path'].endswith('.tsx') else version['language'])
        source=self._get('source',doc['sourceId'])
        score=rating(text,version['language'],analysis,source)
        parts = make_chunks(text, version['language'], counter=(lambda value:encoder.count(doc['path']+'\n'+value)) if encoder.available else None,analysis=analysis)
        with self.store.tx():
            self._save('version', {**self._get('version',version_id),'codeStructure':analysis,'qualityRating':score,'chunkingVersion':PIPELINE_VERSION})
            for ordinal, part in enumerate(parts):
                location = {'kind': 'record', 'positions': version['recordPositions']} if version.get('recordPositions') else {'kind': 'section', 'section': part['section'] or doc['name']} if version.get('htmlExtracted') else {'kind': 'lines', 'startLine': part['startLine'], 'endLine': part['endLine']}
                if location['kind']=='lines' and version.get('sourceSpan',{}).get('startLine'):
                    offset=version['sourceSpan']['startLine']-1
                    location={**location,'startLine':location['startLine']+offset,'endLine':location['endLine']+offset}
                self._save('chunk', {**part, **chunk_structure(part,analysis),'chunkingVersion':PIPELINE_VERSION, 'versionId': version_id, 'documentId': doc['id'], 'sourceId': doc['sourceId'], 'ordinal': ordinal, 'textBlobHash': self._blob(part['text']), 'contentHash': digest(part['text']), 'retrievalAllowed': True, 'generatorTokenCount': None, 'location': location})
            self._save('document', {**doc, 'state': 'processing' if doc.get('retrievalPolicy') == 'approved' else 'needs_review'})

    def _near_duplicates(self, version, text):
        """Bounded SimHash candidate buckets + exact character 5-gram Jaccard.

        Candidate generation is approximate; this is a review detector, never a
        claim to find every duplicate or permission to delete source material.
        """
        normalized = re.sub(r'\s+', ' ', text).strip()
        if len(normalized) < 80:
            return []
        counts = Counter(match.group(0).lower() for match in re.finditer(r'\w+', normalized))
        bits = [0] * 64
        for token, count in counts.items():
            hashed = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), 'little')
            weight = min(count, 5)
            for bit in range(64):
                bits[bit] += weight if hashed & (1 << bit) else -weight
        signature = sum((1 << i) for i, value in enumerate(bits) if value >= 0)
        bands = [(signature >> (16 * i)) & 65535 for i in range(4)]
        with self.store.tx() as con:
            con.execute('INSERT OR REPLACE INTO pre_fingerprints VALUES(?,?,?,?,?,?,?)', (version['id'], version['language'], len(normalized), *bands))
        candidate_ids = {row['version_id'] for row in self.store.read('SELECT version_id FROM pre_fingerprints WHERE language=? AND text_length BETWEEN ? AND ? AND (band0=? OR band1=? OR band2=? OR band3=?) AND version_id<>? LIMIT 100', (version['language'], int(len(normalized) * .90), int(len(normalized) / .90), *bands, version['id']))}
        if not candidate_ids:
            return []
        shingles = None
        candidates = []
        versions = {v['id']: v for v in self._all('version')}
        for doc in self._all('document'):
            other = versions.get(doc.get('currentVersionId'))
            if not other or other['id'] not in candidate_ids or doc.get('deletedAt') or other.get('qualityStatus') != 'passed':
                continue
            if other.get('normalizedHash') == version.get('normalizedHash'):
                continue  # Exact duplicates share blobs and are collapsed at retrieval.
            length = other.get('textLength', 0)
            if length and not .85 * len(text) <= length <= 1.18 * len(text):
                continue
            other_text = re.sub(r'\s+', ' ', self._blob_read(other['normalizedBlobHash']).decode('utf-8')).strip()
            if not .90 * len(normalized) <= len(other_text) <= len(normalized) / .90:
                continue
            if shingles is None:
                shingles = {normalized[i:i + 5] for i in range(len(normalized) - 4)}
            other_shingles = {other_text[i:i + 5] for i in range(len(other_text) - 4)}
            union = len(shingles | other_shingles)
            if union and len(shingles & other_shingles) / union >= .90:
                candidates.append({'documentId': doc['id'], 'versionId': other['id']})
        return candidates

    def _active_id(self):
        row = self.store.one('SELECT generation_id FROM pre_active WHERE workspace_id=?', ('default',))
        return row['generation_id'] if row else None

    def rebuild(self, body=None, job=None):
        body = body or {}
        with self._build_lock:
            base = self._active_id()
            base_policy = self.store.setting('policyVersion', 0)
            self._check_job(job)
            docs = {d['id']: d for d in self._all('document')}
            sources = {s['id']: s for s in self._all('source')}
            versions = {v['id']: v for v in self._all('version')}
            chunks = []
            for chunk in self._all('chunk'):
                doc, version = docs.get(chunk['documentId']), versions.get(chunk['versionId'])
                if doc and version and not sources[doc['sourceId']].get('upstreamDeleted') and chunk.get('chunkingVersion')==version.get('chunkingVersion') and not doc.get('deletedAt') and doc.get('enabled', True) and doc['currentVersionId'] == version['id'] and doc.get('retrievalPolicy') == 'approved' and sources[doc['sourceId']].get('retrievalPolicy') == 'approved' and version.get('qualityStatus') == 'passed':
                    chunks.append(chunk)
            if len(chunks) > 50000:
                raise DomainError('INDEX_CAPACITY_LIMIT', '초기 색인 상한은 50,000 청크입니다.', 409)
            generation_id = uid()
            staging = self.index_root / (generation_id + '.staging')
            staging.mkdir()
            encoder = self.encoder()
            dense = bool(encoder.available)
            degraded = encoder.reason
            vector_hash = None
            if dense and chunks:
                try:
                    import numpy as np
                    target = staging / 'vectors.f32'
                    matrix = np.memmap(target, dtype=np.float32, mode='w+', shape=(len(chunks), 384))
                    for i, chunk in enumerate(chunks):
                        self._check_job(job)
                        if job:
                            self.store.update_job(job['id'], stage='embedding', processedItems=i, totalItems=len(chunks), unit='chunks')
                        passage = f"{docs[chunk['documentId']]['path']}\n{chunk['text']}"
                        cache_id=digest(json.dumps({'profile':encoder.profile['id'],'passage':passage},sort_keys=True))
                        try:
                            cached=self._get('embedding_vector',cache_id)
                            vector=np.frombuffer(self._blob_read(cached['blobHash']),dtype=np.float32)
                            if vector.shape!=(384,) or not np.isfinite(vector).all():raise ValueError('corrupt cached vector')
                        except (DomainError,ValueError):
                            vector=encoder.encode([passage])[0]
                            self._save('embedding_vector',{'profileId':encoder.profile['id'],'blobHash':self._blob(vector.tobytes())},cache_id)
                        matrix[i] = vector
                    matrix.flush()
                    del matrix
                    vector_hash = digest(target.read_bytes())
                except (ValueError, RuntimeError, ImportError, OSError) as exc:
                    dense, degraded = False, f'EMBEDDING_FAILED:{type(exc).__name__}'
                    if (staging / 'vectors.f32').exists():
                        (staging / 'vectors.f32').unlink()
            mapping = [c['id'] for c in chunks]
            self._write_json(staging / 'row-map.json', mapping)
            manifest = {'id': generation_id, 'baseGenerationId': base, 'pipelineVersion': PIPELINE_VERSION, 'embeddingProfileId': encoder.profile['id'] if dense else None, 'embeddingProfile': encoder.profile if dense else None, 'dimension': 384 if dense else None, 'chunkCount': len(chunks), 'searchMode': 'hybrid' if dense else 'lexical_only', 'degradedReason': degraded, 'vectorHash': vector_hash, 'rowMapHash': digest((staging / 'row-map.json').read_bytes()), 'createdAt': now()}
            self._write_json(staging / 'manifest.json', manifest)
            final = self.index_root / generation_id
            os.replace(staging, final)
            generation = {**manifest, 'state': 'ready', 'manifestPath': str(final / 'manifest.json'), 'manifestHash': digest((final / 'manifest.json').read_bytes())}
            with self.store.tx() as con:
                self._check_job(job)
                if self._active_id() != base:
                    raise DomainError('INDEX_BASE_CONFLICT', '색인 기반이 변경되었습니다. 다시 시도하세요.', 409)
                if self.store.setting('policyVersion', 0) != base_policy:
                    raise DomainError('INDEX_POLICY_CONFLICT', '색인 중 자료·권한이 변경되었습니다. 다시 색인하세요.', 409)
                for original in docs.values():
                    current = self._get('document', original['id'])
                    if any(current.get(key) != original.get(key) for key in ('currentVersionId', 'policyVersion', 'retrievalPolicy', 'deletedAt', 'enabled')):
                        raise DomainError('INDEX_POLICY_CONFLICT', '색인 중 문서 버전·권한이 변경되었습니다.', 409)
                for i, chunk in enumerate(chunks):
                    doc = docs[chunk['documentId']]
                    con.execute('INSERT INTO pre_memberships(generation_id,chunk_id,vector_row) VALUES (?,?,?)', (generation_id, chunk['id'], i if dense else None))
                    con.execute('INSERT INTO pre_fts(chunk_id,generation_id,title,path,symbol,body,search_terms) VALUES (?,?,?,?,?,?,?)', (chunk['id'], generation_id, doc['name'], doc['path'], chunk.get('symbol', ''), chunk['text'], search_terms(doc['name'] + ' ' + doc['path'] + ' ' + chunk.get('symbol', '') + ' ' + chunk['text'])))
                self._save('generation', generation)
                con.execute('UPDATE pre_active SET generation_id=? WHERE workspace_id=?', (generation_id, 'default'))
                ready_versions = {chunk['versionId'] for chunk in chunks}
                for doc in docs.values():
                    if not doc.get('deletedAt'):
                        doc = self._get('document', doc['id'])
                        state = 'ready' if doc['currentVersionId'] in ready_versions and self.allowed_document(doc) else 'needs_review' if versions.get(doc['currentVersionId'], {}).get('qualityStatus') != 'passed' or doc.get('retrievalPolicy') == 'pending' else 'excluded' if not self.allowed_document(doc) else 'processing'
                        self._save('document', {**doc, 'state': state, 'indexedGenerationId': generation_id if state == 'ready' else None})
                if dense:
                    self._save('embedding_profile', encoder.profile, encoder.profile['id'])
            self.store.activity('index.published', generation_id, f'검색 색인 게시 · {len(chunks)}개 청크')
            with self.lock:self._gc_indexes()
            return generation

    def _gc_indexes(self):
        """Keep two verified generations; immutable readers and backups are pinned."""
        active = self._active_id()
        generations = sorted(self._all('generation'), key=lambda item: (item['createdAt'], item['id']), reverse=True)
        keep = {active, *(item['id'] for item in generations[:2])}
        keep.update(identifier for identifier, count in self._generation_readers.items() if count)
        # Parent backup copies under store.lock, so GC waits for consistent copy.
        with self.store.lock:
            for generation in generations:
                identifier = generation['id']
                if identifier in keep:
                    continue
                path = (self.index_root / identifier).resolve()
                if path.parent != self.index_root.resolve() or path.is_symlink():
                    continue
                with self.store.tx() as con:
                    con.execute('DELETE FROM pre_memberships WHERE generation_id=?', (identifier,))
                    con.execute('DELETE FROM pre_fts WHERE generation_id=?', (identifier,))
                    self.store.delete('generation', identifier)
                if path.exists():
                    shutil.rmtree(path)
            for path in self.index_root.glob('*.staging'):
                resolved = path.resolve()
                if resolved.parent == self.index_root.resolve() and path.stat().st_mtime < time.time() - 86400:
                    shutil.rmtree(resolved)

    def allowed_document(self, doc):
        try:
            source = self._get('source', doc['sourceId'])
            version = self._get('version', doc['currentVersionId'])
        except DomainError:
            return False
        return not source.get('upstreamDeleted') and not doc.get('deletedAt') and doc.get('enabled', True) and doc.get('retrievalPolicy') == 'approved' and source.get('retrievalPolicy') == 'approved' and version.get('qualityStatus') == 'passed'

    def _write_json(self, target, body):
        with target.open('w', encoding='utf-8') as stream:
            json.dump(body, stream, ensure_ascii=False, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())

    def _manifest(self, generation):
        path = self.index_root / generation['id'] / 'manifest.json'
        try:
            raw = path.read_bytes()
            manifest = json.loads(raw)
            if digest(raw) != generation['manifestHash'] or digest((path.parent / 'row-map.json').read_bytes()) != manifest['rowMapHash']:
                raise ValueError('checksum mismatch')
            mapping = json.loads((path.parent / 'row-map.json').read_text(encoding='utf-8'))
            if len(mapping) != manifest['chunkCount']:
                raise ValueError('row count mismatch')
            if manifest.get('vectorHash') and digest((path.parent / 'vectors.f32').read_bytes()) != manifest['vectorHash']:
                raise ValueError('vector checksum mismatch')
            return manifest, mapping
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise DomainError('INDEX_CORRUPT', '색인 무결성 검사에 실패했습니다.', 503) from exc

    def indexes(self):
        active = self._active_id()
        return [{**g, 'active': g['id'] == active} for g in self._all('generation')]

    def activate_index(self, identifier):
        with self.lock, self.store.tx() as con:
            generation = self._get('generation', identifier)
            self._manifest(generation)
            con.execute('UPDATE pre_active SET generation_id=? WHERE workspace_id=?', (identifier, 'default'))
        self.store.activity('index.activated', identifier, '검색 색인 전환')
        return {**generation, 'active': True}

    def search(self, body):
        if not isinstance(body, dict):
            raise DomainError('INVALID_QUERY', '검색 요청은 객체여야 합니다.', 422)
        for field in ('sourceIds', 'languages'):
            value = body.get(field)
            if value is not None and (not isinstance(value, list) or len(value) > 100 or any(not isinstance(item, str) or len(item) > 512 for item in value)):
                raise DomainError('INVALID_FILTER', '소스와 언어 필터는 문자열 배열이어야 합니다.', 422)
        for field in ('projectId', 'revision'):
            value = body.get(field)
            if value is not None and (not isinstance(value, str) or len(value) > 512):
                raise DomainError('INVALID_FILTER', '프로젝트와 버전은 문자열이어야 합니다.', 422)
        if not isinstance(body.get('mode', 'code'), str):
            raise DomainError('INVALID_WORK_MODE', '작업 모드는 문자열이어야 합니다.', 422)
        if body.get('retrievalMode','hybrid') not in ('hybrid','lexical','dense'):raise DomainError('INVALID_RETRIEVAL_MODE','검색 방식은 hybrid/lexical/dense입니다.',422)
        for key in ('includeRelated','useQuality'):
            if key in body and not isinstance(body[key],bool):raise DomainError('INVALID_QUERY',f'{key}는 boolean입니다.',422)
        threshold=body.get('minDenseScore',.75)
        if not isinstance(threshold,(int,float)) or isinstance(threshold,bool) or not 0<=threshold<=1:raise DomainError('INVALID_QUERY','cosine 하한은 0~1입니다.',422)
        began = time.perf_counter()
        with self.lock:
            identifier = self._active_id()
            if identifier:
                self._generation_readers[identifier] += 1
            policy = self.store.setting('policyVersion', 0)
            key = digest(json.dumps({'workspace': 'default', 'generation': identifier, 'policy': policy, 'embedding': self.store.setting('settings', {}).get('embedding', {}), 'query': body, 'pipeline': PIPELINE_VERSION}, sort_keys=True))
            cached = self._search_cache.get(key)
            if cached and cached[0] > time.monotonic():
                self._search_cache.move_to_end(key)
                cached_value = copy.deepcopy(cached[1])
            else:
                self._search_cache.pop(key, None)
                cached_value = None
        try:
            if cached_value is not None and all(self.allowed_chunk(item['chunkId']) for item in cached_value.get('items', [])):
                if identifier:
                    self._manifest(self._get('generation', identifier))
                trace = self._save('trace', {'indexGenerationId': identifier, 'queryHash': digest(str(body.get('query', ''))), 'cacheOf': cached_value['traceId'], 'selectedIds': [item['chunkId'] for item in cached_value['items']], 'policyVersion': policy, 'cacheHit': True})
                cached_value.update({'traceId': trace['id'], 'cacheHit': True, 'observedAt': now(), 'timingsMs': {'total': round((time.perf_counter() - began) * 1000, 3)}})
                return cached_value
            result = self._search(body, identifier)
            with self.lock:
                self._search_cache[key] = (time.monotonic() + 300, copy.deepcopy(result))
                self._search_cache.move_to_end(key)
                while len(self._search_cache) > 100:
                    self._search_cache.popitem(last=False)
            result['cacheHit'] = False
            return result
        finally:
            with self.lock:
                if identifier:
                    self._generation_readers[identifier] -= 1

    def _search(self, body, pinned_generation_id):
        started = time.perf_counter()
        query = str(body.get('query', '')).strip()
        if not query or len(query.encode('utf-8')) > 32768:
            raise DomainError('INVALID_QUERY', '질의는 1~32 KiB입니다.', 422)
        if body.get('mode', 'code') not in {'code', 'review', 'refactor', 'test'}:
            raise DomainError('INVALID_WORK_MODE', '지원하지 않는 작업 모드입니다.', 422)
        top_k = body.get('topK', 5)
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 20:
            raise DomainError('INVALID_TOP_K', 'topK는 1~20입니다.', 422)
        generation_id = pinned_generation_id
        if not generation_id:
            return self._empty_search(body, started, 'NO_ACTIVE_INDEX')
        generation = self._get('generation', generation_id)
        manifest, mapping = self._manifest(generation)
        filters = {'sourceIds': body.get('sourceIds') or [], 'languages': [str(v).lower() for v in body.get('languages') or []], 'projectId': body.get('projectId'), 'revision': body.get('revision')}
        if not isinstance(filters['sourceIds'], list) or not isinstance(filters['languages'], list):
            raise DomainError('INVALID_FILTER', '소스와 언어 필터는 배열이어야 합니다.', 422)
        catalog = self._catalog()
        docs, versions, chunks_by_id = catalog['document'], catalog['version'], catalog['chunk']
        permitted = {}
        for identifier in mapping:
            chunk = chunks_by_id.get(identifier)
            if not chunk or not self._allowed_from(chunk, catalog):
                continue
            doc, version = docs[chunk['documentId']], versions[chunk['versionId']]
            if filters['sourceIds'] and doc['sourceId'] not in filters['sourceIds']:
                continue
            if filters['languages'] and version['language'].lower() not in filters['languages']:
                continue
            if filters['projectId'] and doc.get('projectId', doc.get('datasetId')) != filters['projectId']:
                continue
            if filters['revision'] and version['revision'] != filters['revision']:
                continue
            permitted[identifier] = chunk
        lexical_start = time.perf_counter()
        tokens = re.findall(r'[\w]+', search_terms(query), re.UNICODE)[:40]
        tokens = list(dict.fromkeys(t for t in tokens if t.strip()))
        match = ' OR '.join('"' + token.replace('"', '""') + '"' for token in tokens)
        lexical_rows = []
        if match and permitted and body.get('retrievalMode')!='dense':
            # Policy filtering is part of the SQL query, before ORDER/LIMIT.
            allowed_json = json.dumps(list(permitted))
            lexical_rows = self.store.read('SELECT chunk_id,bm25(pre_fts,0,0,2,3,4,1,1) AS score FROM pre_fts WHERE pre_fts MATCH ? AND generation_id=? AND chunk_id IN (SELECT value FROM json_each(?)) ORDER BY score LIMIT 30', (match, generation_id, allowed_json))
        lexical_ms = (time.perf_counter() - lexical_start) * 1000
        lexical = [row['chunk_id'] for row in lexical_rows]
        exact_tokens = [t.lower() for t in re.findall(r'[A-Za-z_$][\w$]*(?:[./][\w$.-]+)*', query)]
        required_tokens = [t.lower() for t in re.findall(r'[A-Za-z_$][\w$]*(?:[./][\w$.-]+)*', query) if '_' in t or '/' in t or '.' in t or re.search(r'[a-z][A-Z]', t)]
        exact = sorted((c for c in permitted.values() if c.get('symbol', '').lower() in exact_tokens or any(token in docs[c['documentId']]['path'].lower() for token in exact_tokens if '/' in token or '.' in token)), key=lambda c: c['ordinal'])[:20]
        exact_ids = [c['id'] for c in exact] if body.get('retrievalMode')!='dense' else []
        dense, dense_scores, embedding_ms, dense_ms = [], {}, 0.0, 0.0
        degraded = generation.get('degradedReason')
        encoder = self.encoder()
        use_dense = body.get('retrievalMode')!='lexical' and manifest.get('searchMode') == 'hybrid' and encoder.available and manifest.get('embeddingProfileId') == encoder.profile['id']
        if use_dense and permitted:
            try:
                import numpy as np
                started_embedding = time.perf_counter()
                query_vector = encoder.encode([query], query=True)[0]
                embedding_ms = (time.perf_counter() - started_embedding) * 1000
                started_dense = time.perf_counter()
                matrix = np.memmap(self.index_root / generation_id / 'vectors.f32', dtype=np.float32, mode='r', shape=(len(mapping), 384))
                allowed_rows = [i for i, identifier in enumerate(mapping) if identifier in permitted]
                scores = matrix[allowed_rows] @ query_vector
                best = [i for i in np.argsort(-scores)[:30] if float(scores[i])>=body.get('minDenseScore',.75)]
                dense = [mapping[allowed_rows[int(i)]] for i in best]
                dense_scores = {mapping[allowed_rows[int(i)]]: float(scores[i]) for i in best}
                dense_ms = (time.perf_counter() - started_dense) * 1000
            except (OSError, ValueError, RuntimeError, ImportError):
                dense = []
                use_dense = False
                degraded = 'DENSE_SEARCH_FAILED'
        elif body.get('retrievalMode')!='lexical' and manifest.get('searchMode') == 'hybrid':
            degraded = encoder.reason or 'EMBEDDING_PROFILE_CHANGED'
        fused = defaultdict(float)
        for ranked in (exact_ids, lexical, dense):
            for rank, identifier in enumerate(ranked, 1):
                fused[identifier] += 1 / (60 + rank)
        if body.get('useQuality',True):
            for identifier in fused:
                score=versions[permitted[identifier]['versionId']].get('qualityRating',{}).get('score',50)
                fused[identifier]*=.9+.2*score/100
        anchors=sorted(fused,key=lambda identifier:-fused[identifier])
        if required_tokens:anchors=[i for i in anchors if any(t in (permitted[i]['text']+' '+docs[permitted[i]['documentId']]['path']).lower() for t in required_tokens)]
        edges=related(anchors,permitted,docs,catalog['source']) if body.get('includeRelated',True) else []
        related_ids={edge['to'] for edge in edges}
        for edge in edges:fused[edge['to']]=max(fused[edge['to']],fused[edge['from']]*.65)
        sorted_ids = sorted(fused, key=lambda identifier: (-fused[identifier], identifier))
        selected, seen = [], set()
        for identifier in sorted_ids:
            chunk = permitted[identifier]
            if required_tokens and identifier not in related_ids and not any(token in (chunk['text'] + ' ' + docs[chunk['documentId']]['path']).lower() for token in required_tokens):
                continue
            if chunk['contentHash'] in seen:
                continue
            if not self.allowed_chunk(identifier):
                continue
            seen.add(chunk['contentHash'])
            doc, version = docs[chunk['documentId']], versions[chunk['versionId']]
            source = self._get('source', doc['sourceId'])
            selected.append({'rank': len(selected) + 1, 'chunkId': identifier, 'documentId': doc['id'], 'versionId': version['id'], 'sourceId': source['id'], 'title': doc['name'], 'path': doc['path'], 'sourceUrl': source.get('locator', {}).get('url'), 'revision': version['revision'], 'location': chunk['location'], 'excerpt': chunk['text'],'symbol':chunk.get('qualifiedSymbol') or chunk.get('symbol'),'qualityRating':version.get('qualityRating'),'relatedBy':[e for e in edges if e['to']==identifier], 'evidenceUrl': f"/api/v1/document-versions/{version['id']}/content", 'scores': {'rrf': fused[identifier], 'bm25': next((row['score'] for row in lexical_rows if row['chunk_id'] == identifier), None), 'cosine': dense_scores.get(identifier)}})
            if len(selected) >= top_k:
                break
        status = 'sufficient' if selected else 'insufficient_evidence'
        timings = {'queryEmbedding': round(embedding_ms, 3), 'lexical': round(lexical_ms, 3), 'dense': round(dense_ms, 3), 'total': round((time.perf_counter() - started) * 1000, 3)}
        trace = self._save('trace', {'indexGenerationId': generation_id, 'queryHash': digest(query), 'mode': body.get('mode', 'code'), 'retrievalMode':body.get('retrievalMode','hybrid'),'relationships':edges,'qualityWeighted':body.get('useQuality',True), 'filters': filters, 'lexicalIds': lexical, 'denseIds': dense, 'exactIds': exact_ids, 'selectedIds': [item['chunkId'] for item in selected], 'ranks': {key: fused[key] for key in sorted_ids[:60]}, 'timingsMs': timings, 'degradedReason': degraded, 'policyVersion': self.store.setting('policyVersion', 0)})
        return {'traceId': trace['id'], 'indexGenerationId': generation_id, 'status': status, 'searchMode': 'hybrid' if use_dense else 'lexical_only', 'degradedReason': degraded, 'items': selected, 'reasonCodes': [] if selected else ['NO_MATCHING_APPROVED_EVIDENCE'], 'timingsMs': timings, 'observedAt': now()}

    def _empty_search(self, body, started, reason):
        trace = self._save('trace', {'indexGenerationId': None, 'queryHash': digest(str(body.get('query', ''))), 'filters': {}, 'selectedIds': [], 'degradedReason': reason})
        return {'traceId': trace['id'], 'indexGenerationId': None, 'status': 'insufficient_evidence', 'searchMode': 'lexical_only', 'degradedReason': reason, 'items': [], 'reasonCodes': [reason], 'timingsMs': {'total': round((time.perf_counter() - started) * 1000, 3)}, 'observedAt': now()}
