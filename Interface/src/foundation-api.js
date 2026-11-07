import { ragRequest } from './rag-api.js';

export const foundationPath = path => `/llm/foundation/${String(path).replace(/^\/+/, '')}`;
export const foundationRequest = (path, options = {}) => ragRequest(foundationPath(path), options);
export const actualNumber = value => typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : null;
export const collection = value => Array.isArray(value) ? value : Array.isArray(value?.items) ? value.items : [];

export function foundationProgress(run) {
  const metrics = { ...(run?.progress ?? {}), ...(run?.metrics ?? {}), ...run };
  const completed = actualNumber(metrics.committedTargetTokens);
  const total = actualNumber(metrics.targetTokenBudget ?? metrics.maxTokenExposures ?? run?.config?.maxTokenExposures);
  return { completed, total, percent: completed !== null && total !== null && total > 0 ? Math.min(100, Math.floor(completed / total * 1000) / 10) : null,
    steps: actualNumber(metrics.completedOptimizerSteps), loss: actualNumber(metrics.observedLoss),
    tokensPerSecond: actualNumber(metrics.effectiveTokensPerSecond), peakRssBytes: actualNumber(metrics.peakRssBytes),
    activeSeconds: actualNumber(metrics.activeSeconds) };
}

const reasons = {
  MODEL_NOT_REGISTERED: '모델 생성 전', BRANCH_REQUIRED: '전문화 전', CORPUS_PERMISSION_REQUIRED: '출처 승인',
  INSUFFICIENT_CORPUS: '자료 대기', TOKENIZER_VOCAB_INCOMPLETE: '토큰 자료 부족',
  DAILY_BUDGET_EXHAUSTED: '일일 예산', RESOURCE_PRESSURE: '자원 대기', RESOURCE_UNAVAILABLE: '실행 자원 없음',
  WORKER_UNAVAILABLE: '작업자 대기', NO_USABLE_MODEL: '기초학습 중', VERIFIER_UNAVAILABLE: '격리 준비',
  INSUFFICIENT_EVAL_COVERAGE: '평가 자료 부족', QUALITY_UNCERTAIN: '개선 불확실', POLICY_REVOKED: '재학습 필요',
  CHECKPOINT_INCOMPATIBLE: '재개 불가', TRAINING_SOURCES_NOT_APPROVED: '학습 출처 승인',
  CALIBRATION_REQUIRED: '실측 필요', PROFILE_INSUFFICIENT_STEPS: '측정 스텝 부족', PROFILE_TIME_LIMIT: '측정 시간 한도',
  ISOLATED_TRAINER_REQUIRED: '격리 학습기 대기',
  MODEL_MODE_NOT_APPROVED: '작업별 검증 필요', MODEL_LANGUAGE_NOT_APPROVED: '언어 검증 필요',
  CAPABILITY_SCOPE_REQUIRED: '검증 범위 선택', SOURCE_CODE_REQUIRED: '분석할 코드 필요',
  DOMAIN_RESPONSE_NOT_VERIFIED: '근거 검증 실패', GGUF_RUNTIME_UNAVAILABLE: '추론 엔진 준비',
  GGUF_CONFORMANCE_REQUIRED: '형식 검증 필요', GGUF_ENGINE_CHANGED: '엔진 재검증',
  EXECUTION_SCOPE_REQUALIFICATION_REQUIRED: '환경 재검증', BACKEND_CERTIFICATION_REQUIRED: '실행 환경 검증',
  CURRENT_SOFTWARE_ACCEPTANCE_REQUIRED: '코드 검증 필요', ACTUAL_CONTAINER_ACCEPTANCE_REQUIRED: '컨테이너 검증 필요',
  RAW_DATA_REQUIREMENTS_NOT_MET: '원문 자료 대기', SFT_DATA_REQUIREMENTS_NOT_MET: '승인 답변 대기',
  SFT_REFERENCE_REQUIRED: '원문 실행 검증', QUALIFICATION_HISTORY_REQUIRED: '평가 이력 대기',
  USABLE_FOUNDATION_MODEL_REQUIRED: '사용 모델 대기', CURRENT_CERTIFIED_MODEL_REQUIRED: '모델 자격 검증',
  CURRENT_USABLE_CERTIFICATE_REQUIRED: '모델 자격 검증', SOURCE_WORKERS_MUST_BE_STOPPED: '작업 종료 대기',
  INDEPENDENT_QUALITY_REQUIRED: '독립 품질 검증', PAIRED_200_GROUPS_REQUIRED: '비교 자료 부족',
  QUALIFIED_GROWTH_TRANSITION_REQUIRED: '성장 검증 대기', DOMAIN_QUALITY_REQUIRED: '작업별 검증 대기',
  FORMATS_AND_DISTILLATION_QUALITY_REQUIRED: '형식 · 증류 검증', ACTUAL_TRAINED_DISTILLATION_REQUIRED: '증류 학습 대기',
  TRAINED_CANDIDATE_AND_INDEPENDENT_EXAM_REQUIRED: '모델 독립 검증', EVIDENCE_REQUIRED: '검증 대기', USER_DEFERRED: '별도 진행',
  SOURCE_NOT_ALLOWED: '학습 이용 미승인', FOUNDATION_QUALITY_NOT_REACHED: '기초 품질 미충족',
  EVALUATION_BUDGET_EXHAUSTED: '평가 예산', OUTSIDE_TRAINING_WINDOW: '야간 대기',
};
export const reasonLabel = reason => reasons[reason] ?? String(reason ?? '').replaceAll('_', ' ');
export function stageLabel(stage) {
  return ({ waiting_corpus: '자료 대기', corpus_ready: '자료 준비', waiting_tokenizer: '토큰 준비 전', tokenizer_training: '토큰 학습', tokenizer_ready: '토큰 준비',
    initializing: '초기화', genesis_ready: '초기화 완료', pretraining: '기초학습', foundation_candidate: '기초 후보',
    foundation_qualified: '기초 검증', instruction_training: '지시학습', assistant_candidate: '작업 후보', usable: '사용 가능',
    held: '보류', paused: '일시정지', failed: '실패', revoked: '철회', queued: '대기', running: '진행 중', completed: '완료',
    cancelled: '중단', waiting_worker: '작업자 대기', waiting_window: '시간 대기' })[stage] ?? stage ?? '준비 전';
}
