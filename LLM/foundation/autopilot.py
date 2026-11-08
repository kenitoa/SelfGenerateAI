"""One explicit request prepares and schedules foundation training.

GET is a read-only preview. Source rights remain authoritative in RAG. The user's
explicit start request authorizes approval of eligible unknown sources; explicit
denials remain excluded. Preparation uses the existing persistent scheduler and
its admission/quality boundaries.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from LLM.core import Fault, digest, now, require, uid
from .resources import DEFAULT_POLICY, recommended_training_config

SUPPORTED_SOURCES = frozenset(('upload', 'web_document', 'git_repository'))
ACTIVE_JOBS = frozenset(('queued', 'running', 'pause_requested', 'cancelling'))
ACTIVE_SCHEDULER = frozenset(('preparing_corpus', 'tokenizer_training', 'initializing', 'pretraining',
    'instruction_training', 'evaluating', 'evaluating_release', 'preparing_instruction_data',
    'continued_pretraining', 'resuming', 'running', 'deploying'))
MESSAGES = {
    'ISOLATED_TRAINER_REQUIRED': ('학습 워커 연결을 기다리고 있어요', '설정은 저장되었습니다. NAS 학습 워커가 연결되면 자동으로 계속합니다.'),
    'RESOURCE_PRESSURE': ('NAS의 여유 자원을 기다리고 있어요', '다른 작업이 줄어들면 권장 설정으로 자동 재시도합니다.'),
    'RESOURCE_UNAVAILABLE': ('NAS 자원을 확인하고 있어요', '메모리와 디스크를 확인할 수 있을 때 자동으로 계속합니다.'),
    'AGGREGATE_MEMORY_BUDGET_UNAVAILABLE': ('현재 메모리가 학습 실행 조건보다 부족해요', '자료와 설정은 보존됩니다. NAS의 가용 메모리가 실행 조건을 충족해야 학습할 수 있습니다.'),
    'DISK_LOW': ('학습 결과를 저장할 공간이 부족해요', '저장 공간이 확보되면 자동으로 다시 확인합니다. 기존 자료나 모델을 임의로 삭제하지 않습니다.'),
    'INSUFFICIENT_CORPUS': ('학습 원문을 더 준비하고 있어요', '승인된 원문을 검사하고 수집 한도 안에서 자료를 더 모읍니다. 필요한 양이 준비되면 자동으로 학습을 이어갑니다.'),
    'INSUFFICIENT_NEW_TRAIN_DATA': ('새로운 학습 자료를 기다리고 있어요', '기존 학습 결과를 보존하고, 새 원문이 추가되면 자동으로 계속합니다.'),
    'USER_PAUSED': ('일시정지한 학습이 있어요', '사용자가 중지한 작업은 자동으로 재개하지 않습니다. 학습 제어의 재개 버튼으로 이어갈 수 있습니다.'),
    'DEPLOYMENT_MAINTENANCE': ('NAS 업데이트가 진행 중이에요', '업데이트가 끝나면 준비 상태를 다시 확인합니다.'),
    'DAILY_BUDGET_EXHAUSTED': ('오늘의 학습 시간을 모두 사용했어요', '자료와 체크포인트를 보존하고 다음 날 자동으로 계속합니다.'),
    'CALIBRATION_REQUIRED': ('NAS에서 학습 속도와 메모리를 측정하고 있어요', '측정을 통과한 동일 설정으로 실제 학습을 이어갑니다.'),
    'POLICY_REVOKED': ('이 모델의 학습 이용 권한이 철회되었어요', '철회된 자료가 포함된 모델은 재사용할 수 없습니다. 허용된 자료로 새 모델을 준비해야 합니다.'),
    'VERIFIER_UNAVAILABLE': ('학습 결과를 검증할 환경이 필요해요', '학습 결과는 보존되며, 격리 검증 환경이 준비되기 전에는 서비스에 자동 적용하지 않습니다.'),
    'SANDBOX_LIMIT_UNSUPPORTED': ('NAS에서 코드 격리 검증을 지원하지 않아요', '학습 준비와 검증 환경은 별도입니다. 생성 코드 실행과 서비스 적용은 필요한 격리 조건이 갖춰질 때까지 보류합니다.'),
    'CRAWLER_PAUSED': ('자료 수집이 일시정지되어 있어요', '저장한 학습 요청은 유지됩니다. 수집이 다시 활성화되면 자료 준비를 이어갑니다.'),
    'CRAWL_DAILY_LIMIT': ('오늘의 자료 수집 한도에 도달했어요', '학습 요청을 보존하고 다음 수집 주기에 자동으로 자료를 더 준비합니다.'),
}


def _eligible(source):
    return (source.get('kind') in SUPPORTED_SOURCES and not source.get('archivedAt')
            and not source.get('upstreamDeleted') and source.get('origin') != 'model_self_code')


class AutoTraining:
    def __init__(self, control):
        self.control, self.store, self.foundation = control, control.store, control.foundation

    def _project(self, identity=None):
        if identity is not None:
            require(isinstance(identity, str) and identity, 'CONFIG_INVALID', field='projectId')
            value = self.store.get('foundation_project', identity)
        else:
            selected = self.store.get('setting', 'foundationAutoTrainingProject', {}).get('projectId')
            projects = [row for row in self.store.list('foundation_project')
                        if row.get('state') != 'revoked' and not row.get('revocationPending') and not row.get('testOnly')]
            value = next((row for row in projects if row['id'] == selected), None)
            value = value or next(iter(sorted(projects, key=lambda row: (row.get('createdAt', ''), row['id']), reverse=True)), None)
        if value:
            value = {**value, 'policy': self.store.get('foundation_policy', value['id'], {**DEFAULT_POLICY, 'version': 0})}
        return value

    def _sources(self):
        bridge = self.control.bridge
        if bridge is None:
            return []
        return sorted((row for row in bridge.store.list('source') if not row.get('archivedAt')),
                      key=lambda row: (str(row.get('name') or ''), row['id']))

    def _paused_runs(self, project_id, jobs):
        result = []
        for job in jobs:
            if job.get('state') != 'paused':continue
            run = self.store.get('foundation_run', job['id'], {})
            if run.get('pauseReason') != 'USER_PAUSED':continue
            checkpoint = self.store.get('foundation_checkpoint', run.get('checkpointId') or run.get('lastCheckpointId') or '', {})
            resumable = checkpoint.get('state') in ('ready', 'verified', 'completed') and checkpoint.get('projectId') == project_id and checkpoint.get('runId') == job['id']
            result.append({'jobId': job['id'], 'checkpoint': checkpoint, 'resumable': resumable,
                           'runVersion': run.get('version')})
        return result

    def _steps(self, project, jobs, scheduler):
        project = project or {}
        identity = project.get('id')
        policy = project.get('policy', {})
        ready = lambda kind, key: bool(self.store.get(kind, project.get(key) or '', {}).get('state') in ('ready', 'verified', 'completed', 'passed'))
        statuses = [bool(policy.get('allowedSourceIds')), ready('foundation_corpus', 'activeCorpusId'),
                    ready('foundation_tokenizer', 'tokenizerId'), bool(project.get('genesisId')),
                    any(row.get('projectId') == identity and row.get('passed') is True for row in self.store.list('foundation_profile')) if identity else False,
                    any(row.get('projectId') == identity and row.get('completedOptimizerSteps', 0) > 0 for row in self.store.list('foundation_run')) if identity else False,
                    project.get('stage') == 'usable']
        entries = [('sources', '학습 자료 확인'), ('corpus', '자료 검사'), ('tokenizer', '토큰 학습'),
                   ('initialize', '모델 초기화'), ('calibrate', 'NAS 실측'), ('train', '모델 학습'), ('evaluate', '사용 전 검증')]
        active = next((job['type'].removeprefix('foundation_') for job in jobs if job.get('state') in ACTIVE_JOBS), None)
        active = {'pretrain': 'train', 'cpt': 'train', 'sft': 'train', 'evaluate': 'evaluate', 'tokenizer': 'tokenizer',
                  'corpus': 'corpus', 'initialize': 'initialize', 'calibrate': 'calibrate'}.get(active)
        output = []
        for index, (key, label) in enumerate(entries):
            state = 'running' if active == key else 'completed' if statuses[index] else 'pending'
            if state == 'pending' and not active and policy.get('enabled') and not any(not item for item in statuses[:index]):
                state = 'waiting'
            output.append({'id': key, 'label': label, 'state': state})
        return output

    def plan(self, project_id=None):
        project = self._project(project_id)
        identity = project.get('id') if project else None
        sources = self._sources()
        allowed = [row['id'] for row in sources if _eligible(row) and row.get('trainingPolicy') == 'allowed']
        unknown = [row['id'] for row in sources if _eligible(row) and row.get('trainingPolicy', 'unknown') == 'unknown']
        policy = project.get('policy', {}) if project else {}
        automatic = self.store.get('foundation_auto_training', identity, {}) if identity else {}
        enabled = (automatic.get('enabled') is True if 'enabled' in automatic else policy.get('enabled') is True) and policy.get('automation') != 'manual'
        recipe = recommended_training_config(self.store, project or {'executionScope': self.control.execution_scope})
        environment = {**recipe['environment'], 'workerAvailable': self.foundation.has_worker(),
                       'automaticSettings': True, 'nasKernelModified': False}
        jobs = self.foundation._pending(identity) if identity else []
        paused = self._paused_runs(identity, jobs)
        scheduler = self.store.get('foundation_scheduler', identity, {}) if identity else {}
        if enabled and 'AUTOMATION_DISABLED' in scheduler.get('reasonCodes', []):
            # A durable Start supersedes the last disabled scheduler observation.
            scheduler = {**scheduler, 'reasonCodes': [reason for reason in scheduler['reasonCodes'] if reason != 'AUTOMATION_DISABLED']}
            if scheduler.get('state') == 'paused':
                scheduler['state'] = 'preparing'
        corpus = self.store.get('foundation_corpus', project.get('activeCorpusId') or '', {}) if project else {}
        if identity and not corpus:
            corpus = next(iter(sorted((row for row in self.store.list('foundation_corpus') if row.get('projectId') == identity),
                key=lambda row: (row.get('createdAt', ''), row['id']), reverse=True)), {})
        data = {'state': corpus.get('state', 'unprepared'), 'corpusId': corpus.get('id'),
                'trainingBytes': corpus.get('splitBytes', {}).get('train', 0),
                'trainingGroups': corpus.get('groupCounts', {}).get('train', 0),
                'minTrainingBytes': policy.get('minCorpusBytes', DEFAULT_POLICY['minCorpusBytes']),
                'minTrainingGroups': policy.get('minTrainGroups', DEFAULT_POLICY['minTrainGroups'])}
        blockers = list(dict.fromkeys(recipe['reasonCodes'] + ([] if environment['workerAvailable'] else ['ISOLATED_TRAINER_REQUIRED'])))
        if self.control.bridge is None:
            blockers.insert(0, 'RAG_POLICY_GATEWAY_REQUIRED')
        if project:
            try:self.foundation.project(identity)
            except Fault as exc:blockers.insert(0, exc.code)
        maintenance = self.store.get('setting', 'deploymentMaintenance', {}).get('enabled')
        if maintenance:blockers.insert(0, 'DEPLOYMENT_MAINTENANCE')
        if enabled:
            previous_reasons = [reason for reason in scheduler.get('reasonCodes', []) if reason != 'ISOLATED_TRAINER_REQUIRED']
            blockers = list(dict.fromkeys(previous_reasons
                + automatic.get('sourcePreparation', {}).get('reasonCodes', []) + blockers))
            if paused:
                blockers = ['USER_PAUSED'] + [reason for reason in blockers if reason != 'USER_PAUSED']
        action, state = 'start', 'ready'
        title, detail = '학습하기만 누르면 준비부터 자동으로 진행해요', '현재 자료의 학습 이용 승인과 선택, 자료 수집, NAS에 맞는 설정을 자동으로 처리하고 학습까지 이어갑니다.'
        if 'POLICY_REVOKED' in blockers:
            action, state = 'wait', 'revoked'
            title, detail = MESSAGES['POLICY_REVOKED']
        elif maintenance:
            action, state = 'wait', 'maintenance'
            title, detail = MESSAGES['DEPLOYMENT_MAINTENANCE']
        elif self.control.bridge is None:
            action, state = 'retry', 'disconnected'
            title, detail = '자료 서버 연결이 필요해요', 'RAG와 학습 서버가 연결된 통합 실행 환경에서 다시 확인해 주세요.'
        elif not enabled and not allowed and not unknown:
            title, detail = '학습하기를 누르면 자료 수집부터 시작해요', '등록된 출처와 자동 수집 설정을 확인하고 사용할 자료를 준비합니다. 학습이 금지된 자료와 지원이 종료된 출처는 제외합니다.'
        elif enabled:
            action, state = 'wait', scheduler.get('state', 'preparing')
            if 'USER_PAUSED' in blockers:
                state = 'paused'
                if paused and all(row['resumable'] for row in paused):
                    action = 'start'
                    title, detail = '학습하기를 누르면 이어서 학습해요', '저장된 체크포인트를 확인하고 마지막 학습 상태에서 자동으로 재개합니다.'
                else:
                    blockers = ['CHECKPOINT_REQUIRED'] + blockers
                    title, detail = '이 작업에는 이어갈 체크포인트가 없어요', '일시정지한 작업은 보존되어 있습니다. 유효한 체크포인트 없이 학습을 재개하거나 완료한 것으로 표시하지 않습니다.'
            elif any(job.get('state') in ACTIVE_JOBS for job in jobs) or state in ACTIVE_SCHEDULER:
                title, detail = '학습 준비와 실행을 자동으로 진행하고 있어요', '이 화면을 닫아도 NAS에서 계속합니다. 실제 단계와 대기 이유를 아래에 표시합니다.'
            elif not allowed:
                state = 'collecting_sources'
                title, detail = next((MESSAGES[reason] for reason in blockers if reason in ('CRAWLER_PAUSED', 'CRAWL_DAILY_LIMIT')),
                    ('학습 자료를 자동으로 준비하고 있어요', '자료 수집과 이용 권한을 자동으로 확인합니다. 자료가 준비되면 추가 설정 없이 학습을 이어갑니다.'))
            elif blockers:
                title, detail = next((MESSAGES[reason] for reason in blockers if reason in MESSAGES),
                                     ('실행 조건을 다시 확인하고 있어요', '요청과 자료는 보존되어 있습니다. 현재 대기 이유가 해결되면 자동으로 재확인합니다.'))
            else:
                title, detail = '자동 학습 요청이 저장되어 있어요', 'NAS에서 순서대로 준비하고 실행 상태를 갱신합니다.'
        if enabled and corpus and 'INSUFFICIENT_CORPUS' in blockers:
            detail += (f" 현재 학습용 원문 {data['trainingBytes'] / 1024**2:.1f}/{data['minTrainingBytes'] / 1024**2:g} MiB,"
                       f" 독립 자료 묶음 {data['trainingGroups']}/{data['minTrainingGroups']}개입니다.")
        collection = automatic.get('sourcePreparation', {}).get('collection', automatic.get('collection'))
        if enabled and collection and collection.get('discoveryState') == 'daily_limit':
            limit = collection.get('dailyLimit') or {}
            if type(limit.get('pages')) is int and type(limit.get('limit')) is int:
                detail += f" 오늘 자동 수집 {limit['pages']}/{limit['limit']}페이지를 사용했습니다."
            resume_at = collection.get('discoveryNextAttemptAt') or limit.get('resetAt') or collection.get('nextAttemptAt')
            if isinstance(resume_at, (int, float)):
                stamp = datetime.fromtimestamp(resume_at, timezone(timedelta(hours=9)))
                detail += f" {stamp:%m월 %d일 %H:%M}(한국 시간)에 수집을 다시 확인합니다."
        return {'schemaVersion': 1, 'projectId': identity, 'action': action, 'state': state,
                'title': title, 'detail': detail, 'enabled': enabled, 'policyVersion': policy.get('version'), 'observedAt': now(),
                'sources': [{'id': row['id'], 'name': row.get('name', row['id']), 'kind': row.get('kind'),
                             'trainingPolicy': row.get('trainingPolicy', 'unknown'), 'policyVersion': row.get('policyVersion', 1),
                             'eligible': _eligible(row)} for row in sources],
                'consentSourceIds': [], 'readySourceIds': allowed, 'automaticSourceIds': allowed + unknown,
                'excludedSourceCount': len(sources) - len(allowed) - len(unknown),
                'blockers': list(dict.fromkeys(blockers)), 'environment': environment,
                'recommendation': {'automation': 'always_on_idle', 'dailyActiveSeconds': 14400, 'sliceSeconds': 900,
                                   'config': recipe['config'], 'measurementRequired': True},
                'steps': self._steps(project, jobs, scheduler), 'scheduler': scheduler or None, 'data': data,
                'canResume': bool(paused) and all(row['resumable'] for row in paused),
                'collection': collection,
                'automaticSourceApproval': automatic.get('automaticSourceApproval') is True}

    def _approve(self, consent):
        if consent is None:return
        self._validate_consent(consent)
        selected, versions = consent['sourceIds'], consent['policyVersions']
        bridge = self.control.bridge
        require(bridge is not None, 'RAG_POLICY_GATEWAY_REQUIRED', status=503)
        with bridge.store.tx():
            current = {row['id']: row for row in bridge.store.list('source')}
            # Validate the entire snapshot before changing the first permission.
            for identity in selected:
                source = current.get(identity, {})
                require(_eligible(source) and source.get('trainingPolicy', 'unknown') in ('unknown', 'allowed'),
                        'SOURCE_CONSENT_CHANGED', '자료의 이용 권한이 변경되었습니다. 목록을 확인하고 다시 승인해 주세요.', status=409)
                if source.get('trainingPolicy') == 'allowed':continue
                require(type(versions.get(identity)) is int and versions[identity] == source.get('policyVersion', 1),
                        'SOURCE_CONSENT_CHANGED', '자료의 이용 권한이 변경되었습니다. 목록을 확인하고 다시 승인해 주세요.', status=409)
            for identity in selected:
                if current[identity].get('trainingPolicy') == 'allowed':continue
                bridge.pre.update_source(identity, {'trainingPolicy': 'allowed',
                    'reason': '학습하기에서 현재 자료의 학습 이용 권한을 명시적으로 확인하고 허용함'})
        self.store.event('foundation.auto_training.consent', 'sources', {'sourceIds': selected})

    @staticmethod
    def _validate_consent(consent):
        if consent is None:return
        require(isinstance(consent, dict) and not (set(consent) - {'sourceIds', 'policyVersions'}), 'CONFIG_INVALID', field='consent')
        selected, versions = consent.get('sourceIds'), consent.get('policyVersions')
        require(isinstance(selected, list) and selected
                and all(isinstance(value, str) and value for value in selected)
                and len(selected) == len(set(selected)) and isinstance(versions, dict),
                'CONFIG_INVALID', field='consent')

    def start(self, project_id=None, body=None, key=None):
        """Commit the user's intent without waiting for the RAG preparation lock.

        The scheduler owns approval, collection, checkpoint checks and model
        work. A request receipt and its enabled generation are committed in one
        transaction, so an interrupted HTTP response is safe to retry.
        """
        if project_id is not None:require(isinstance(project_id, str) and project_id, 'CONFIG_INVALID', field='projectId')
        route = 'foundation/projects/' + project_id + '/auto-training' if project_id else 'foundation/auto-training'
        body = {} if body is None else body
        require(isinstance(body, dict) and not (set(body) - {'projectId', 'consent', 'autoApproveSources'}), 'CONFIG_INVALID', field='auto-training')
        require(type(body.get('autoApproveSources', True)) is bool, 'CONFIG_INVALID', field='autoApproveSources')
        self._validate_consent(body.get('consent'))
        auto_approve = body.get('autoApproveSources', True)
        if project_id and body.get('projectId'):require(project_id == body['projectId'], 'PROJECT_ID_MISMATCH')
        project_id = project_id or body.get('projectId')
        if project_id is not None:require(isinstance(project_id, str) and project_id, 'CONFIG_INVALID', field='projectId')
        receipt_id = digest([route, key or uid()])
        request_hash = digest(body)
        with self.store.tx():
            previous = self.store.get('foundation_auto_training_request', receipt_id, {})
            if previous:
                require(previous['requestHash'] == request_hash, 'IDEMPOTENCY_CONFLICT', status=409)
                return self.receipt(previous)
            require(not self.store.get('setting', 'deploymentMaintenance', {}).get('enabled'), 'DEPLOYMENT_MAINTENANCE', status=503)
            require(self.control.bridge is not None, 'RAG_POLICY_GATEWAY_REQUIRED', status=503)
            project = self._project(project_id)
            if project is None:
                project = self.foundation.create_project({'name': 'SEED-S0', 'origin': 'random_init',
                    'architecturePreset': 'seed-s0-11m-v1', 'pretrainedWeightArtifactIds': []}, return_status=False)
            identity = project['id']
            project = self.foundation.project(identity)
            policy = project['policy']
            # Source eligibility is inspected in the background. Preserve its
            # current selection until that inspection can safely replace it.
            self.store.put('foundation_policy', {**policy, 'automation': 'always_on_idle',
                'sliceSeconds': 900, 'dailyActiveSeconds': 14400, 'checkIntervalSeconds': 10}, identity)
            request_id = uid()
            requested_at = now()
            resume_candidates = {row['jobId']: row['runVersion']
                for row in self._paused_runs(identity, self.foundation._pending(identity))}
            self.store.put('foundation_auto_training', {'projectId': identity, 'requestedAt': requested_at,
                'requestId': request_id, 'enabled': True, 'automaticSourceApproval': auto_approve,
                'automaticCollection': auto_approve, 'consent': body.get('consent'),
                'sourceApprovalAuthorization': 'user_training_start', 'preparationState': 'pending',
                'preparationRequestId': request_id, 'preparationError': None,
                'resumeCandidates': resume_candidates, 'resumedJobIds': [],
                'startClaimOwner': None, 'startClaimUntil': 0}, identity)
            self.store.put('setting', {'projectId': identity}, 'foundationAutoTrainingProject')
            saved = self.store.put('foundation_auto_training_request', {'projectId': identity,
                'requestId': request_id, 'requestHash': request_hash, 'requestedAt': requested_at}, receipt_id)
            self.store.event('foundation.auto_training.requested', identity, {'requestId': request_id,
                'executionScope': self.control.execution_scope, 'automaticSettings': True,
                'automaticSourceApproval': auto_approve})
        self.control.foundation_scheduler.last_check.pop(identity, None)
        return self.receipt(saved)

    def receipt(self, saved):
        """Small acknowledgement using current intent, never a cached enabled flag."""
        identity = saved['projectId']
        automatic = self.store.get('foundation_auto_training', identity, {})
        policy = self.store.get('foundation_policy', identity, {})
        project = self.store.get('foundation_project', identity, {})
        enabled = automatic.get('enabled') is True and policy.get('automation') != 'manual' \
            and project.get('state') != 'revoked' and not project.get('revocationPending')
        return {'schemaVersion': 1, 'accepted': True, 'projectId': identity,
            'requestId': saved.get('requestId'), 'currentRequestId': automatic.get('requestId'),
            'requestedAt': saved.get('requestedAt'), 'enabled': enabled,
            'policyVersion': policy.get('version'), 'action': 'wait' if enabled else 'start',
            'state': 'preparing' if enabled else 'paused',
            'title': '자동 학습 요청을 저장했어요' if enabled else '자동 학습이 중지되어 있어요',
            'detail': 'NAS에서 자료 승인과 수집, 권장 설정을 순서대로 준비합니다.' if enabled else '저장된 자료와 학습 결과를 보존합니다.',
            'blockers': [], 'steps': [], 'observedAt': now()}
