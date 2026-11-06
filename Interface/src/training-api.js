import { getCsrfToken } from './rag-api.js';

export const emptyTrainingStatus = () => ({ job: null, runs: [], readySamples: 0, canStart: false });
const states = new Set(['preparing', 'running', 'paused', 'validating', 'completed', 'cancelled', 'failed']);
const isCount = value => Number.isSafeInteger(value) && value >= 0;

export function normalizeTrainingStatus(payload) {
  if (!payload || typeof payload !== 'object' || !Object.hasOwn(payload, 'job')) throw new Error('학습 상태 응답 형식 오류');
  let job = null;
  if (payload.job !== null) {
    const value = payload.job;
    if (!value || typeof value.id !== 'string' || !value.id || !states.has(value.state)
      || !isCount(value.completedSteps) || !isCount(value.totalSteps)
      || value.completedSteps > value.totalSteps) throw new Error('학습 스텝 정보 오류');
    if (value.state === 'completed' && (!value.totalSteps || value.completedSteps !== value.totalSteps)) throw new Error('학습 완료 정보 불일치');
    job = {
      id: value.id, state: value.state,
      completedSteps: value.completedSteps, totalSteps: value.totalSteps,
      progress: value.totalSteps > 0 ? value.completedSteps / value.totalSteps * 100 : null,
      loss: typeof value.loss === 'number' && Number.isFinite(value.loss) ? value.loss : null,
      epoch: typeof value.epoch === 'number' && Number.isFinite(value.epoch) && value.epoch >= 0 ? value.epoch : null,
      updatedAt: typeof value.updatedAt === 'string' && Number.isFinite(Date.parse(value.updatedAt)) ? value.updatedAt : null,
      capabilities: { pause: value.capabilities?.pause === true, resume: value.capabilities?.resume === true, cancel: value.capabilities?.cancel === true },
    };
  }
  const runs = Array.isArray(payload.runs) ? payload.runs.filter(run => run && typeof run.id === 'string'
    && ['completed', 'cancelled', 'failed'].includes(run.state)
    && typeof run.finishedAt === 'string' && Number.isFinite(Date.parse(run.finishedAt))) : [];
  return { job, runs, readySamples: isCount(payload.readySamples) ? payload.readySamples : 0, canStart: payload.canStart === true,
    ...(typeof payload.connection === 'string' ? { connection: payload.connection } : {}),
    ...(Array.isArray(payload.reasonCodes) ? { reasonCodes: payload.reasonCodes } : {}) };
}

export function trainingUrl(endpoint, path) {
  const base = new URL(endpoint);
  if (!['http:', 'https:'].includes(base.protocol)) throw new Error('http 또는 https 학습 서버 주소가 필요합니다.');
  base.search = ''; base.hash = '';
  return new URL(`${base.pathname.replace(/\/$/, '')}/api/training/${path}`, base.origin).href;
}

export async function trainingRequest(endpoint, path, { method = 'GET', body, signal } = {}) {
  const response = await fetch(trainingUrl(endpoint, path), {
    method, signal, cache: 'no-store', credentials: 'same-origin',
    ...((body !== undefined || method !== 'GET') ? { headers: { ...(body !== undefined ? { 'Content-Type': 'application/json' } : {}), ...(getCsrfToken() && method !== 'GET' ? { 'X-CSRF-Token': getCsrfToken() } : {}) } } : {}),
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  });
  if (!response.ok) throw new Error(`학습 서버 응답 오류 (${response.status})`);
  if (method === 'GET') return normalizeTrainingStatus(await response.json());
}
