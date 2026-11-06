import React, { useEffect, useState } from 'react';
import { ragRequest } from './rag-api';

const percent = value => value == null ? '—' : `${(value * 100).toFixed(1)}%`;
export default function KnowledgePanel({ open, notify }) {
  const [status, setStatus] = useState(null);
  const [evaluations, setEvaluations] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [detail, setDetail] = useState(null);
  async function load() {
    try {
      const [state, results] = await Promise.all([ragRequest('/knowledge/status'), ragRequest('/evaluations')]);
      setStatus(state); setEvaluations(results.items ?? []); setError('');
    } catch (cause) { setError(cause.message); }
  }
  useEffect(() => { if (!open) return; load(); const timer = setInterval(load, 5000); return () => clearInterval(timer); }, [open]);
  async function action(path, body = {}) {
    setBusy(true);
    try { await ragRequest(path, { method: 'POST', body }); notify('작업 접수'); await load(); }
    catch (cause) { setError(cause.message); }
    finally { setBusy(false); }
  }
  return <div className="rag-tools-group">
    <h3>검색 품질</h3>
    <div className="rag-admin-row"><span>{status?.retrieval.searchMode === 'hybrid' ? '하이브리드 검색' : '키워드 검색'}<small>{status?.embedding.available ? '다국어 E5 · INT8 · CPU' : '임베딩 미연결'}</small></span><span>분석 {status?.analyzedDocuments ?? 0}개<small>함수·클래스 {status?.symbols ?? 0} · 호출 {status?.calls ?? 0}</small></span></div>
    <p className="muted small">품질 평균 {status?.quality.meanScore ?? '—'}점 · 삭제 감지 {status?.upstreamDeletedSources ?? 0}개 · 정적 분석 기준</p>
    <div className="rag-tool-buttons">
      <button className="button secondary" type="button" disabled={busy} onClick={() => action('/evaluations', { suite: 'builtin-code' })}>기본 평가</button>
      <button className="button secondary" type="button" disabled={busy} onClick={() => action('/knowledge/refresh')}>분석·색인 갱신</button>
      <button className="button secondary" type="button" disabled={busy || status?.embedding.available} onClick={() => action('/embedding/install')}>임베딩 설치 · 135 MB</button>
    </div>
    <small>기본 평가 · 10개 코드 문서 · 한국어·영어·심볼·근거 없음 36문항 · 운영 데이터와 분리</small>
    {evaluations.slice(0, 4).map(result => <div className="rag-admin-row" key={result.id}><span>{result.suite === 'builtin-code' ? '기본 평가' : result.split} · {result.questionCount ?? result.count}문항<small>Recall@5 {percent(result.recallAt5)} · MRR {result.mrrAt5?.toFixed(3) ?? '—'} · 거절 {percent(result.correctRejection)}</small></span><button type="button" onClick={() => setDetail(detail?.id === result.id ? null : result)}>비교</button></div>)}
    {detail && <div>{detail.comparisons?.filter(item => item.available).map(item => <p className="muted small" key={item.mode}>{item.mode} · Recall {percent(item.metrics.recallAt5)} · nDCG {item.metrics.ndcgAt5?.toFixed(3)} · p95 {item.metrics.p95Ms?.toFixed(1)} ms</p>)}<pre className="rag-tools-status">{JSON.stringify({ id: detail.id, corpusHash: detail.corpusHash, questionHash: detail.questionHash, byLanguage: detail.comparisons?.find(item => item.mode === 'hybrid')?.byLanguage, limitations: detail.limitations }, null, 2)}</pre></div>}
    {error && <p className="form-error" role="alert">{error}</p>}
  </div>;
}
