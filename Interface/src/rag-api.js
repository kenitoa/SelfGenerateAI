const API_BASE = '/api/v1';
let csrfToken = '';

export class RagError extends Error {
  constructor(message, code = 'REQUEST_FAILED', status = 0, details = null) {
    super(message); this.name = 'RagError'; this.code = code; this.status = status; this.details = details;
  }
}

export function setCsrfToken(value) { csrfToken = typeof value === 'string' ? value : ''; }
export function getCsrfToken() { return csrfToken; }
export function requestId() { return globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(36).slice(2)}`; }

export async function ragRequest(path, { method = 'GET', body, signal, headers = {}, idempotencyKey, raw = false, timeoutMs = 20000 } = {}) {
  const write = !['GET', 'HEAD'].includes(method);
  const form = typeof FormData !== 'undefined' && body instanceof FormData;
  const timeout = !raw && timeoutMs > 0 ? AbortSignal.timeout(timeoutMs) : null;
  const combinedSignal = signal && timeout ? AbortSignal.any([signal, timeout]) : signal ?? timeout ?? undefined;
  const response = await fetch(path.startsWith('/api/') ? path : `${API_BASE}${path}`, {
    method, signal: combinedSignal, credentials: 'same-origin', cache: 'no-store',
    headers: { ...(!form && body !== undefined ? { 'Content-Type': 'application/json' } : {}),
      ...(write && csrfToken ? { 'X-CSRF-Token': csrfToken } : {}),
      ...(write ? { 'Idempotency-Key': idempotencyKey ?? requestId() } : {}), ...headers },
    ...(body === undefined ? {} : { body: form ? body : JSON.stringify(body) }),
  });
  if (!response.ok) {
    let payload; try { payload = await response.json(); } catch {}
    const error = payload?.error ?? payload?.detail;
    throw new RagError(typeof error === 'string' ? error : error?.message ?? `서버 응답 오류 (${response.status})`, error?.code, response.status, error?.details);
  }
  if (raw) return response;
  if (response.status === 204) return null;
  const payload = await response.json();
  if (payload?.csrfToken) setCsrfToken(payload.csrfToken);
  return payload;
}

export async function readSession() { return ragRequest('/auth/me'); }
export async function login(password) { return ragRequest('/auth/login', { method: 'POST', body: { password } }); }
export const items = payload => Array.isArray(payload) ? payload : Array.isArray(payload?.items) ? payload.items : [];
export function datasetRecord(record) {
  const states = { ready: '준비', needs_review: '검토', processing: '분류', failed: '실패', excluded: '제외', deleted: '제외' };
  return { ...record, name: record.name ?? '데이터셋', type: record.language ?? '코드 · 문서',
    count: record.chunkCount ?? record.documentCount ?? 0, source: record.source ?? `${record.documentCount ?? 0}개 문서`,
    status: states[record.state] ?? record.status ?? '분류', date: record.updatedAt ? new Date(record.updatedAt).toLocaleDateString('ko-KR') : '—' };
}
export function normalizeThread(thread) {
  return { ...thread, messages: (thread.messages ?? []).map(message => ({ ...message, text: message.text ?? message.content ?? '' })) };
}
export function metricNumber(value, scale = 1) { return typeof value === 'number' && Number.isFinite(value) ? (value / scale).toLocaleString('ko-KR', { maximumFractionDigits: 2 }) : '—'; }

export async function uploadFiles(files, datasetId) {
  const body = new FormData();
  files.forEach(file => body.append('files[]', file));
  if (datasetId) body.append('datasetId', datasetId);
  body.append('retrievalConsent', 'true'); body.append('mode', 'replace_snapshot');
  return ragRequest('/uploads', { method: 'POST', body });
}

export async function consumeSse(response, onEvent, { signal } = {}) {
  if (!response.body?.getReader) throw new RagError('스트림을 읽을 수 없습니다.', 'STREAM_UNAVAILABLE');
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    while (!signal?.aborted) {
      const { done, value } = await reader.read();
      buffer = (buffer + decoder.decode(value, { stream: !done })).replaceAll('\r\n', '\n');
      let boundary;
      while ((boundary = buffer.indexOf('\n\n')) >= 0) {
        const block = buffer.slice(0, boundary); buffer = buffer.slice(boundary + 2);
        let type = 'message', id = '', data = [];
        for (const line of block.split('\n')) {
          if (line.startsWith('event:')) type = line.slice(6).trim();
          if (line.startsWith('id:')) id = line.slice(3).trim();
          if (line.startsWith('data:')) data.push(line.slice(5).trimStart());
        }
        if (data.length) { const text = data.join('\n'); let payload; try { payload = JSON.parse(text); } catch { payload = { text }; } onEvent({ type, id, data: payload }); }
      }
      if (done) break;
    }
  } finally { await reader.cancel().catch(() => {}); reader.releaseLock(); }
}
