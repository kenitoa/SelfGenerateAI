import React, { useEffect, useState } from 'react';
import { Play, RefreshCw } from 'lucide-react';
import { ragRequest } from './rag-api';

const providers = [
  ['githubEnabled', 'GitHub API'], ['sitemapEnabled', 'Sitemap'], ['feedEnabled', 'RSS · Atom'],
  ['commonCrawlEnabled', 'Common Crawl'], ['browserEnabled', 'Playwright'],
];

export default function CrawlerPanel({ open, notify }) {
  const [status, setStatus] = useState(null);
  const [draft, setDraft] = useState(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  async function load(resetDraft = false) {
    try { const next = await ragRequest('/crawler/status'); setStatus(next); setDraft(old => resetDraft ? next.config : (old ?? next.config)); setError(''); }
    catch (cause) { setError(cause.message); }
  }
  useEffect(() => { if (!open) return; load(); const timer = setInterval(load, 5000); return () => clearInterval(timer); }, [open]);
  async function save(patch = {}) {
    setBusy(true);
    try { const cfg = await ragRequest('/crawler/config', { method: 'PATCH', body: { ...draft, ...patch } }); setDraft(cfg); await load(); notify('자동 수집 설정 저장'); }
    catch (cause) { setError(cause.message); }
    finally { setBusy(false); }
  }
  async function run() {
    setBusy(true);
    try { await ragRequest('/crawler/run', { method: 'POST', body: {} }); await load(); notify('코드 탐색·수집 요청'); }
    catch (cause) { setError(cause.message); }
    finally { setBusy(false); }
  }
  if (!draft) return <div className="rag-tools-group"><h3>코드 자동 수집</h3><p className="muted small">{error || '상태 확인 중'}</p></div>;
  return <div className="rag-tools-group">
    <h3>코드 자동 수집</h3>
    <div className="rag-admin-row"><span>{status?.config.enabled ? '자동 탐색 활성' : '자동 탐색 중지'}<small>Scrapy · HTTPX · selectolax · Trafilatura · robots.txt</small></span><button type="button" disabled={busy} onClick={() => save({ enabled: !draft.enabled })}>{draft.enabled ? '중지' : '재개'}</button></div>
    <p className="muted small">발견 → 코드 추출 → 검토·정제 → 색인 · 가중치 훈련은 별도 엔진</p>
    <div className="rag-admin-row"><span>오늘 방문 {status?.today.pages ?? 0}페이지<small>코드 문서 접수 {status?.today.codeDocuments ?? 0}개</small></span><span>대기 {status?.frontier.queued ?? 0}개</span></div>
    <p className="muted small">검색 준비 {status?.indexedDocuments ?? 0}개 · 검토 대기 {status?.reviewDocuments ?? 0}개</p>
    <div className="rag-admin-row"><span>수집 균형<small>사이트 · 언어 · 제공자 · 저장소</small></span><button type="button" aria-pressed={draft.fairScheduling} onClick={() => setDraft(s => ({ ...s, fairScheduling: !s.fairScheduling }))}>{draft.fairScheduling ? '활성' : '중지'}</button></div>
    <details><summary>언어 가중치 · 수집 분포</summary><div className="settings-fields">{Object.entries(draft.languageWeights ?? {}).map(([key, value]) => <label className="field" key={key}><span>{key}</span><input type="number" min="0.1" max="10" step="0.1" value={value} onChange={event => setDraft(s => ({ ...s, languageWeights: { ...s.languageWeights, [key]: Number(event.target.value) } }))} /></label>)}</div><pre className="rag-tools-status">{JSON.stringify(status?.balance ?? {}, null, 2)}</pre><small>추적 저장소 {status?.trackedRepositories ?? 0}개 · 대체 후보가 있을 때 편중 완화</small></details>
    <div className="settings-fields">
      <label className="field"><span>실행 간격 · 분</span><input type="number" min="5" max="1440" value={draft.intervalMinutes} onChange={e => setDraft(s => ({ ...s, intervalMinutes: Number(e.target.value) }))} /></label>
      <label className="field"><span>하루 페이지 상한</span><input type="number" min="1" max="1000" value={draft.dailyPages} onChange={e => setDraft(s => ({ ...s, dailyPages: Number(e.target.value) }))} /></label>
      <label className="field"><span>실행당 페이지</span><input type="number" min="1" max="20" value={draft.maxPagesPerRun} onChange={e => setDraft(s => ({ ...s, maxPagesPerRun: Number(e.target.value) }))} /></label>
      <label className="field"><span>링크 깊이</span><input type="number" min="0" max="4" value={draft.maxDepth} onChange={e => setDraft(s => ({ ...s, maxDepth: Number(e.target.value) }))} /></label>
    </div>
    <div className="rag-tool-buttons">{providers.map(([key, label]) => <button className="button secondary" type="button" key={key} aria-pressed={draft[key]} onClick={() => setDraft(s => ({ ...s, [key]: !s[key] }))}>{draft[key] ? '✓ ' : ''}{label}</button>)}</div>
    <details><summary>탐색 경로</summary>{[
      ['seedUrls', '웹 시작점'], ['githubQueries', 'GitHub 검색 조건'], ['feedUrls', 'RSS · Atom 주소'], ['commonCrawlPatterns', '보관 자료 URL 패턴'],
    ].map(([key, label]) => <label className="field" key={key}><span>{label}</span><textarea rows={3} value={draft[key].join('\n')} onChange={e => setDraft(s => ({ ...s, [key]: e.target.value.split('\n').filter(v => v.trim()) }))} /><small>한 줄에 하나 · 개별 페이지는 자동 발견</small></label>)}</details>
    <div className="rag-tool-buttons"><button className="button secondary" type="button" disabled={busy} onClick={() => save()}>설정 저장</button><button className="button secondary" type="button" disabled={busy || !status?.config.enabled} onClick={run}><Play size={14} />지금 수집</button><button className="button secondary" type="button" disabled={busy} onClick={() => load(true)}><RefreshCw size={14} />새로고침</button></div>
    {status?.lastRun && <p className="muted small">최근 실행 · 방문 {status.lastRun.visited} / 코드 접수 {status.lastRun.ingested}<br />{new Date(status.lastRun.finishedAt).toLocaleString('ko-KR')}</p>}
    {status?.lastRun?.errors?.length > 0 && <details><summary>수집 사유 · {status.lastRun.errors.length}개</summary><pre className="rag-tools-status">{status.lastRun.errors.map(item => `${item.stage ?? item.url ?? ''} · ${item.code}`).join('\n')}</pre></details>}
    {error && <p className="form-error" role="alert">{error}</p>}
  </div>;
}
