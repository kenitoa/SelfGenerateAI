import { foundationRequest } from './foundation-api.js';

export const autoTrainingPath = projectId => projectId ? `projects/${encodeURIComponent(projectId)}/auto-training` : 'auto-training';

export function normalizeAutoTrainingPlan(value) {
  if (!value || !['start', 'consent', 'add_data', 'wait', 'retry'].includes(value.action)
    || typeof value.title !== 'string' || typeof value.detail !== 'string') throw new Error('자동 학습 준비 상태를 확인할 수 없습니다. 다시 확인해 주세요.');
  const list = key => Array.isArray(value[key]) ? value[key] : [];
  return { ...value, sources: list('sources'), steps: list('steps'), blockers: list('blockers'),
    consentSourceIds: list('consentSourceIds'), readySourceIds: list('readySourceIds'), automaticSourceIds: list('automaticSourceIds') };
}

export async function readAutoTrainingPlan(projectId, options) {
  return normalizeAutoTrainingPlan(await foundationRequest(autoTrainingPath(projectId), options));
}

export async function startAutoTraining(projectId, body, idempotencyKey) {
  return normalizeAutoTrainingPlan(await foundationRequest(autoTrainingPath(projectId), { method: 'POST', body, idempotencyKey }));
}

export function autoTrainingCollectionStatus(collection) {
  if (collection?.discoveryState !== 'daily_limit') return '';
  const quota = collection.dailyLimit;
  const count = Number.isFinite(quota?.pages) && Number.isFinite(quota?.limit) ? `오늘 ${quota.pages}/${quota.limit}페이지를 수집했습니다. ` : '오늘의 자료 수집 한도에 도달했습니다. ';
  const reset = collection.discoveryNextAttemptAt ?? quota?.resetAt ?? collection.nextAttemptAt;
  const date = typeof reset === 'number' && Number.isFinite(reset) ? new Date(reset * 1000) : null;
  return count + (date && Number.isFinite(date.getTime())
    ? `${new Intl.DateTimeFormat('ko-KR', { timeZone: 'Asia/Seoul', month: 'long', day: 'numeric', hour: '2-digit', minute: '2-digit', hourCycle: 'h23' }).format(date)} (한국시간)에 자동으로 수집을 재개합니다.`
    : '다음 수집 가능 시점에 자동으로 재개합니다.');
}
