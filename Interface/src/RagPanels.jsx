import React, { useEffect, useState } from 'react';
import { Activity, ArrowRight, ArrowUpRight, Check, CircleHelp, Copy, Database, ExternalLink, FileCode2, GitBranch, Loader2, Pause, Play, Plus, RefreshCw, ShieldCheck, Trash2, Upload } from 'lucide-react';
import { items, metricNumber, ragRequest } from './rag-api';
import CrawlerPanel from './CrawlerPanel';
import KnowledgePanel from './KnowledgePanel';
import LlmPanel from '../../LLM/web/Panel.jsx';

export function ActivityRows({ activities }) {
  return activities.slice(0, 5).map(activity => <div className="activity-row" key={activity.id ?? activity.sequence}><span className="activity-icon green-bg"><Activity size={16} /></span><div><strong>{activity.message ?? activity.messageKey ?? activity.type}</strong><span>{activity.entityId ?? activity.type}</span></div><time>{activity.createdAt ? new Date(activity.createdAt).toLocaleTimeString('ko-KR', { timeZone: 'Asia/Seoul', hour: '2-digit', minute: '2-digit' }) : '—'}</time></div>);
}

export function GrowthChart({ series = [] }) {
  const values = series.map(item => ({ ...item, value: item.indexedChunks ?? item.indexed_chunks ?? item.value ?? 0 }));
  if (!values.length) return <div className="chart-empty"><Database size={23} strokeWidth={1.5} /><span>검색 데이터 없음</span></div>;
  const max = Math.max(1, ...values.map(item => item.value));
  const points = values.map((item, index) => `${values.length === 1 ? 300 : index / (values.length - 1) * 600},${130 - item.value / max * 110}`).join(' ');
  return <div className="rag-growth-chart"><svg viewBox="0 0 600 145" role="img" aria-label="검색 준비 청크 추이">{[20, 75, 130].map(y => <line key={y} x1="0" x2="600" y1={y} y2={y} stroke="currentColor" opacity=".12" strokeDasharray="3 5" />)}<polyline points={points} fill="none" stroke="currentColor" strokeWidth="2.5" />{values.length === 1 && <circle cx="300" cy={130 - values[0].value / max * 110} r="3" fill="currentColor" />}</svg><div>{values.filter((_, i) => i === 0 || i === values.length - 1 || i === Math.floor(values.length / 2)).map(item => <span key={item.date ?? item.dateKst}>{item.date ?? item.dateKst} · {metricNumber(item.value)}건</span>)}</div></div>;
}

export function SourceControls({ source, refresh, notify }) {
  const [busy, setBusy] = useState(false);
  const url = source.url ?? source.locator?.url;
  const active = source.collectionEnabled !== false;
  async function action(path, method = 'POST', body = {}) {
    setBusy(true);
    try { await ragRequest(`/sources/${encodeURIComponent(source.id)}${path}`, { method, ...(method === 'DELETE' ? {} : { body }) }); await refresh(); notify(path === '/sync' ? '수집 요청 완료' : '소스 설정 반영'); }
    catch (error) { notify(error.message); } finally { setBusy(false); }
  }
  return <div className="source-row"><span className="quick-icon blue"><Database size={19} /></span><div><strong>{source.name}</strong>{url ? <a href={url} target="_blank" rel="noreferrer">{url}<ExternalLink size={12} /></a> : <small>{source.kind === 'upload' ? '업로드 원문' : source.kind}</small>}</div><span className="status-pill pending">{source.upstreamDeleted ? '원본 삭제' : (source.lastJob?.state ?? (active ? '수집 대기' : '수집 중지'))}</span>{source.kind !== 'upload' && <><button className="icon-button" disabled={busy || !active} aria-label={`${source.name} 수집`} title="수집" onClick={() => action('/sync')}>{busy ? <Loader2 className="spin" size={15} /> : <Play size={15} />}</button><button className="icon-button" disabled={busy} aria-label={`${source.name} ${active ? '수집 중지' : '수집 재개'}`} title={active ? '수집 중지 · 자료 유지' : '수집 재개'} onClick={() => action('', 'PATCH', { collectionEnabled: !active })}>{active ? <Pause size={15} /> : <RefreshCw size={15} />}</button></>}<button className="icon-button" disabled={busy} aria-label={`${source.name} 수집 대상 제거 · 자료 유지`} title="수집 대상 제거 · 자료 유지" onClick={() => action('', 'DELETE')}><Trash2 size={16} /></button></div>;
}

export function EvidenceCards({ message, notify }) {
  const citations = message.citations ?? message.evidence ?? message.retrieval?.items ?? [];
  const [feedback, setFeedback] = useState('');
  async function rate(rating) {
    try { await ragRequest(`/messages/${encodeURIComponent(message.id)}/feedback`, { method: 'POST', body: { rating } }); setFeedback(rating); notify('피드백 저장 · 학습 승인은 별도'); } catch (error) { notify(error.message); }
  }
  return <>
    {citations.length > 0 && <details className="rag-evidence"><summary>근거 <span>{citations.length}</span></summary>{citations.map((citation, i) => <div className="rag-evidence-card" key={citation.chunkId ?? i}><strong>[{citation.label ?? `S${i + 1}`}] {citation.title ?? citation.path ?? '참조 자료'}</strong><small>{citation.path} {citation.revision ? `· ${citation.revision.slice(0, 12)}` : ''}{citation.location?.startLine ? ` · ${citation.location.startLine}–${citation.location.endLine}줄` : ''}</small>{citation.excerpt && <pre>{citation.excerpt}</pre>}{citation.revokedAt || citation.revoked ? <span className="status-pill review">철회됨</span> : <a href={citation.evidenceUrl ?? `/api/v1/document-versions/${encodeURIComponent(citation.versionId)}/content`} target="_blank" rel="noreferrer">저장 원문 <ExternalLink size={12} /></a>}</div>)}</details>}
    {message.role === 'assistant' && message.id && <div className="rag-message-actions"><span>코드 · 테스트 미실행</span><button aria-pressed={feedback === 'helpful'} onClick={() => rate('helpful')}>도움됨</button><button aria-pressed={feedback === 'not_helpful'} onClick={() => rate('not_helpful')}>개선 필요</button></div>}
  </>;
}

export function MessageContent({ text, copy }) {
  const segments = String(text ?? '').split(/(```[\s\S]*?```)/g);
  return segments.map((segment, index) => {
    if (!segment.startsWith('```')) return segment ? <p key={index}>{segment}</p> : null;
    const match = segment.match(/^```([^\n]*)\n?([\s\S]*?)```$/);
    const code = match?.[2] ?? segment;
    return <div className="code-block" key={index}><div><span>{match?.[1] || '코드'} · 미실행</span><button onClick={() => copy(code)}><Copy size={13} />복사</button></div><pre><code>{code}</code></pre></div>;
  });
}

export function DatasetDetails({ record, notify, refresh, close }) {
  const [documents, setDocuments] = useState([]);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [expanded, setExpanded] = useState(null);
  const [chunks, setChunks] = useState([]);
  async function load() {
    try { setDocuments(items(await ragRequest(`/datasets/${encodeURIComponent(record.id)}/documents`))); setError(''); }
    catch (cause) { setError(cause.message); }
  }
  useEffect(() => { load(); const timer = setInterval(load, 2000); return () => clearInterval(timer); }, [record.id]);
  async function review(document, decision) {
    setBusy(true);
    try {
      await ragRequest(`/documents/${encodeURIComponent(document.id)}/review`, { method: 'POST', body: { versionId: document.currentVersionId ?? document.versionId, expectedPolicyVersion: document.policyVersion ?? 1, decision, reason: '사용자 검토', retrievalPolicy: decision === 'approve' ? 'approved' : 'denied', trainingPolicy: 'unknown' } });
      await load(); await refresh(); notify(decision === 'approve' ? '검색 승인 · 색인 요청' : '검색 제외');
    } catch (cause) { setError(cause.message); } finally { setBusy(false); }
  }
  async function inspect(document) {
    if (expanded === document.id) { setExpanded(null); return; }
    setExpanded(document.id); setChunks([]);
    try { setChunks(items(await ragRequest(`/document-versions/${encodeURIComponent(document.currentVersionId ?? document.versionId)}/chunks`))); }
    catch (cause) { setError(cause.message); }
  }
  async function remove() {
    setBusy(true);
    try { await ragRequest(`/datasets/${encodeURIComponent(record.id)}`, { method: 'DELETE' }); await refresh(); close(); notify('자료 제외 · 정리 요청'); }
    catch (cause) { setError(cause.message); } finally { setBusy(false); }
  }
  return <div className="rag-dataset-details"><dl className="detail-list"><div><dt>문서</dt><dd>{metricNumber(record.documentCount)}개</dd></div><div><dt>검색 청크</dt><dd>{metricNumber(record.chunkCount)}개</dd></div><div><dt>학습 샘플</dt><dd>{metricNumber(record.trainingExampleCount)}개</dd></div><div><dt>상태</dt><dd>{record.status}</dd></div></dl>{error && <p className="form-error" role="alert">{error}</p>}<div className="rag-document-list">{documents.map(document => <div className="rag-document" key={document.id}><button className="rag-document-title" onClick={() => inspect(document)}><FileCode2 size={16} /><span>{document.title ?? document.name ?? document.path ?? document.externalKey}</span><span className="status-pill pending">{document.state ?? document.status}</span></button>{(document.qualityReasons ?? []).length > 0 && <p className="form-error">{document.qualityReasons.join(' · ')}</p>}<p className="muted small">품질 {document.qualityRating?.score ?? '—'}점 · 심볼 {document.structureSummary?.symbols ?? 0} · 호출 {document.structureSummary?.calls ?? 0}</p><div className="rag-document-actions"><a target="_blank" rel="noreferrer" href={`/api/v1/document-versions/${encodeURIComponent(document.currentVersionId ?? document.versionId)}/content`}>원문 <ExternalLink size={11} /></a><button disabled={busy || document.qualityStatus === 'quarantine' || document.state === 'quarantine'} onClick={() => review(document, 'approve')}>검색 승인</button><button disabled={busy} onClick={() => review(document, 'exclude')}>제외</button></div>{expanded === document.id && <div className="rag-chunk-list">{chunks.length ? chunks.map(chunk => <pre key={chunk.id ?? chunk.chunkId}><small>{chunk.symbol ?? chunk.section ?? '청크'} · {chunk.startLine ?? chunk.location?.startLine ?? '—'}줄</small>{chunk.text ?? chunk.content ?? chunk.excerpt}</pre>) : <p className="muted small">청크 없음 · 처리 상태 확인</p>}</div>}</div>)}</div><div className="subtle-note"><ShieldCheck size={16} /><span>원문은 NAS 서버에 저장됩니다. 검색 승인과 학습 승인은 별도입니다.</span></div><div className="modal-action-row"><button className="button danger" disabled={busy} onClick={remove}><Trash2 size={15} />자료 삭제</button><button className="button secondary" onClick={close}>닫기</button></div></div>;
}

export function RagAdvanced({ settings, trainingConfig, capabilities, onSave, notify, refresh }) {
  const [open, setOpen] = useState(false);
  const [status, setStatus] = useState('');
  const [busy, setBusy] = useState(false);
  const [models, setModels] = useState([]);
  const [jobs, setJobs] = useState([]);
  const [examples, setExamples] = useState([]);
  const [snapshots, setSnapshots] = useState([]);
  const [backups, setBackups] = useState([]);
  const [evaluations, setEvaluations] = useState([]);
  const [schedules, setSchedules] = useState([]);
  const [selectedSource, setSelectedSource] = useState('');
  const [sources, setSources] = useState([]);
  const [policySource, setPolicySource] = useState('');
  const [retrievalPolicy, setRetrievalPolicy] = useState('pending');
  const [trainingPolicy, setTrainingPolicy] = useState('unknown');
  const [licenseId, setLicenseId] = useState('');
  const [licenseUrl, setLicenseUrl] = useState('');
  const [policyReason, setPolicyReason] = useState('');
  const [evaluationJson, setEvaluationJson] = useState('');
  const [evaluationSplit, setEvaluationSplit] = useState('dev');
  const [scheduleTime, setScheduleTime] = useState('02:00');
  const [exampleInput, setExampleInput] = useState('');
  const [exampleTarget, setExampleTarget] = useState('');
  const [exampleLineage, setExampleLineage] = useState('');
  const [tokenizerPath, setTokenizerPath] = useState(settings.tokenizerPath ?? '');
  const [embeddingDirectory, setEmbeddingDirectory] = useState(settings.embedding?.directory ?? '');
  const [allowedHosts, setAllowedHosts] = useState((settings.allowedSourceHosts ?? []).join(', '));
  const [contextTokens, setContextTokens] = useState(settings.contextTokens ?? 4096);
  const [maxOutputTokens, setMaxOutputTokens] = useState(settings.maxOutputTokens ?? 700);
  useEffect(() => {
    setTokenizerPath(settings.tokenizerPath ?? '');
    setEmbeddingDirectory(settings.embedding?.directory ?? '');
    setAllowedHosts((settings.allowedSourceHosts ?? []).join(', '));
    setContextTokens(settings.contextTokens ?? 4096); setMaxOutputTokens(settings.maxOutputTokens ?? 700);
  }, [settings.version]);
  async function action(path, body = {}, method = 'POST') {
    setBusy(true); setStatus('');
    try { const result = await ragRequest(path, { method, ...(method === 'GET' || method === 'DELETE' ? {} : { body }) }); setStatus(result.message ?? `${result.id ?? result.jobId ?? result.snapshotId ?? '요청'} · ${result.state ?? result.status ?? '접수'}`); await refresh(); return result; }
    catch (cause) { setStatus(cause.message); throw cause; } finally { setBusy(false); }
  }
  async function load() {
    const results = await Promise.allSettled(['/sources', '/schedules', '/training-examples', '/training-snapshots', '/ingestion-jobs', '/backups', '/evaluations'].map(path => ragRequest(path)));
    const setters = [setSources, setSchedules, setExamples, setSnapshots, setJobs, setBackups, setEvaluations];
    results.forEach((result, i) => { if (result.status === 'fulfilled') setters[i](items(result.value)); });
  }
  useEffect(() => {
    if (!open) return;
    load(); const timer = setInterval(() => { if (!document.hidden) load(); }, 2000);
    return () => clearInterval(timer);
  }, [open]);
  async function run(path, body = {}, method = 'POST') { try { await action(path, body, method); await load(); } catch (cause) { setStatus(cause.message); } }
  function pickPolicySource(id) {
    setPolicySource(id);
    const source = sources.find(item => item.id === id);
    setRetrievalPolicy(source?.retrievalPolicy ?? 'pending'); setTrainingPolicy(source?.trainingPolicy ?? 'unknown');
    setLicenseId(source?.licenseId ?? ''); setLicenseUrl(source?.licenseUrl ?? ''); setPolicyReason('');
  }
  return <details className="card settings-section rag-advanced" onToggle={event => setOpen(event.currentTarget.open)}><summary>RAG 관리 <span>연결 · 색인 · 예약 · 백업 · 학습 자료</span></summary><div className="rag-tools-content">
    <div className="rag-tools-group"><h3>연결</h3><div className="rag-tool-buttons"><button className="button secondary" type="button" disabled={busy} onClick={async () => { try { await onSave(); const result = await action('/connections/test', { provider: 'inference' }); setStatus(result.message ?? JSON.stringify(result)); } catch (cause) { setStatus(cause.message); } }}>추론 확인</button><button className="button secondary" type="button" disabled={busy} onClick={async () => { try { setModels(items(await action('/models', undefined, 'GET'))); } catch (cause) { setStatus(cause.message); } }}>설치 모델</button></div>{models.length > 0 && <div className="rag-installed-models">{models.map(model => <span key={model.name ?? model.id}>{model.name ?? model.id}</span>)}</div>}<label className="field"><span>생성 tokenizer.json 경로</span><input value={tokenizerPath} onChange={event => setTokenizerPath(event.target.value)} placeholder="서버에서 읽을 수 있는 로컬 경로" /><small>옆에 generation-profile.json 필요 · model/digest/tokenizer SHA/revision/chatml/contextLength 검증</small></label><button className="button secondary" type="button" disabled={busy} onClick={async () => { try { await onSave({ tokenizerPath }); notify('토크나이저 설정 저장'); } catch (cause) { setStatus(cause.message); } }}>경로 저장</button></div>
    <div className="rag-tools-group"><h3>검색 · 색인</h3><p className="muted small">기본 키워드 검색 · 로컬 임베딩 모델은 선택 사항</p><label className="field"><span>임베딩 아티팩트 디렉터리</span><input value={embeddingDirectory} onChange={event => setEmbeddingDirectory(event.target.value)} placeholder="model.onnx · tokenizer.json · profile.json" /><small>고정 revision·체크섬이 있는 profile.json 필요</small></label><div className="settings-fields"><label className="field"><span>컨텍스트 토큰</span><input type="number" min="512" max="32768" value={contextTokens} onChange={event => setContextTokens(Number(event.target.value))} /></label><label className="field"><span>출력 토큰</span><input type="number" min="128" max="8192" value={maxOutputTokens} onChange={event => setMaxOutputTokens(Number(event.target.value))} /></label></div><label className="field"><span>허용 수집 호스트</span><input value={allowedHosts} onChange={event => setAllowedHosts(event.target.value)} placeholder="docs.python.org, github.com" /><small>쉼표로 구분 · 사설 주소는 수집 차단</small></label><div className="rag-tool-buttons"><button type="button" className="button secondary" disabled={busy} onClick={async () => { try { await onSave({ contextTokens, maxOutputTokens, allowedSourceHosts: allowedHosts.split(',').map(host => host.trim()).filter(Boolean), embedding: { ...(settings.embedding ?? {}), directory: embeddingDirectory, enabled: Boolean(embeddingDirectory) } }); notify('검색 설정 저장 · 임베딩 변경 시 재색인'); } catch (cause) { setStatus(cause.message); } }}>검색 설정 저장</button><button type="button" className="button secondary" disabled={busy} onClick={() => run('/indexes/rebuild')}><RefreshCw size={14} />재색인</button></div></div>
    <CrawlerPanel open={open} notify={notify} />
    <KnowledgePanel open={open} notify={notify} />
    <LlmPanel />
    <div className="rag-tools-group"><h3>작업 상태</h3>{jobs.length ? jobs.slice(0, 8).map(job => <div className="rag-admin-row" key={job.id}><span>{job.kind} · {job.stage ?? job.state}<small>{job.processedItems ?? 0} / {job.totalItems ?? '—'} {job.unit ?? ''}</small></span><span>{job.state}</span>{job.capabilities?.cancel && <button type="button" onClick={() => run(`/ingestion-jobs/${job.id}/cancel`)}>중단</button>}{job.capabilities?.retry && <button type="button" onClick={() => run(`/ingestion-jobs/${job.id}/retry`)}>재시도</button>}</div>) : <p className="muted small">처리 기록 없음</p>}</div>
    <div className="rag-tools-group"><h3>검색 · 학습 권한</h3><label className="field"><span>소스</span><select value={policySource} onChange={event => pickPolicySource(event.target.value)}><option value="">소스 선택</option>{sources.map(source => <option key={source.id} value={source.id}>{source.name}</option>)}</select></label><div className="settings-fields"><label className="field"><span>검색 이용</span><select value={retrievalPolicy} onChange={event => setRetrievalPolicy(event.target.value)}><option value="pending">검토 대기</option><option value="approved">허용</option><option value="denied">제외</option></select></label><label className="field"><span>학습 이용</span><select value={trainingPolicy} onChange={event => setTrainingPolicy(event.target.value)}><option value="unknown">미확인</option><option value="allowed">명시적 허용</option><option value="denied">금지</option></select></label></div><div className="settings-fields"><label className="field"><span>라이선스 ID</span><input value={licenseId} onChange={event => setLicenseId(event.target.value)} placeholder="예: MIT 또는 직접 작성" /></label><label className="field"><span>권한 근거 주소</span><input type="url" value={licenseUrl} onChange={event => setLicenseUrl(event.target.value)} placeholder="https://" /></label></div><label className="field"><span>판단 근거</span><input value={policyReason} onChange={event => setPolicyReason(event.target.value)} placeholder="소유·라이선스·승인 확인 내용" /></label><div className="subtle-note"><ShieldCheck size={16} /><span>검색 허용과 학습 허용은 별도입니다. 자료의 이용 권한을 확인한 뒤 학습 이용을 승인하세요.</span></div><button type="button" className="button secondary" disabled={busy || !policySource || !policyReason.trim()} onClick={() => run(`/sources/${policySource}`, { retrievalPolicy, trainingPolicy, licenseId: licenseId.trim() || null, licenseUrl: licenseUrl.trim() || null, reason: policyReason.trim() }, 'PATCH')}><Check size={14} />권한 저장</button></div>
    <div className="rag-tools-group"><h3>수집 예약</h3><div className="settings-fields"><label className="field"><span>소스</span><select value={selectedSource} onChange={event => setSelectedSource(event.target.value)}><option value="">소스 선택</option>{sources.filter(source => source.kind !== 'upload').map(source => <option key={source.id} value={source.id}>{source.name}</option>)}</select></label><label className="field"><span>매일 · 서울</span><input type="time" value={scheduleTime} onChange={event => setScheduleTime(event.target.value)} /></label></div><button className="button secondary" type="button" disabled={busy || !selectedSource} onClick={() => { const [hour, minute] = scheduleTime.split(':'); run('/schedules', { kind: 'source_sync', sourceId: selectedSource, payload: { sourceId: selectedSource }, cron: `${Number(minute)} ${Number(hour)} * * *`, timezone: 'Asia/Seoul', enabled: true }); }}><Plus size={14} />예약 추가</button>{schedules.map(schedule => <div className="rag-admin-row" key={schedule.id}><span>{schedule.kind} · {schedule.cron}<small>{schedule.nextRunAt ?? '다음 실행 계산'}</small></span><button type="button" onClick={() => run(`/schedules/${schedule.id}`, { enabled: !schedule.enabled }, 'PATCH')}>{schedule.enabled ? '중지' : '재개'}</button><button type="button" onClick={() => run(`/schedules/${schedule.id}`, {}, 'DELETE')}>제거</button></div>)}</div>
    <div className="rag-tools-group"><h3>검색 평가</h3><label className="field"><span>평가 구분</span><select value={evaluationSplit} onChange={event => setEvaluationSplit(event.target.value)}><option value="dev">개발 · 튜닝</option><option value="holdout">고정 검증 · 학습 제외</option></select></label><label className="field"><span>질문 JSON 배열</span><textarea value={evaluationJson} onChange={event => setEvaluationJson(event.target.value)} rows={5} placeholder={'[{"id":"q1","query":"검색 질문","expectedChunkIds":["청크 ID"]}]'} /><small>근거 없는 질문은 expectedChunkIds: [] · 최대 2,000개</small></label><button type="button" className="button secondary" disabled={busy || !evaluationJson.trim()} onClick={async () => { try { const parsed = JSON.parse(evaluationJson); const questions = Array.isArray(parsed) ? parsed : parsed.questions; if (!Array.isArray(questions) || !questions.length) throw new Error('질문 배열이 필요합니다.'); await action('/evaluations', { questions, split: evaluationSplit }); await load(); } catch (cause) { setStatus(cause.message); } }}><Play size={14} />평가 실행</button>{evaluations.slice(0, 5).map(result => <div className="rag-admin-row" key={result.id}><span>{result.split} · {result.count}개<small>Recall@5 {metricNumber(result.recallAt5)} · p95 {metricNumber(result.p95Ms)} ms</small></span><button type="button" onClick={() => setStatus(JSON.stringify({ id: result.id, questionHash: result.questionHash, results: result.results }, null, 2))}>결과</button></div>)}</div>
    <div className="rag-tools-group"><h3>백업</h3><p className="muted small">서버 DB · 원문 · 색인 정합 백업</p><button className="button secondary" type="button" disabled={busy} onClick={() => run('/backups')}><ShieldCheck size={14} />백업 생성</button>{backups.slice(0, 5).map(backup => <div className="rag-admin-row" key={backup.id}><span>{backup.id.slice(0, 14)}<small>{backup.completedAt ? new Date(backup.completedAt).toLocaleString('ko-KR') : backup.createdAt} · {backup.fileCount ?? '—'}개 파일</small></span><span>{backup.status ?? backup.state}</span></div>)}</div>
    <div className="rag-tools-group"><h3>학습 후보</h3><label className="field"><span>입력</span><textarea value={exampleInput} onChange={event => setExampleInput(event.target.value)} rows={3} /></label><label className="field"><span>검토한 정답</span><textarea value={exampleTarget} onChange={event => setExampleTarget(event.target.value)} rows={3} /></label><label className="field"><span>출처 문서 ID · 계보</span><input value={exampleLineage} onChange={event => setExampleLineage(event.target.value)} placeholder="문서 ID, 여러 개는 쉼표로 구분" /></label><button type="button" className="button secondary" disabled={busy || !exampleInput.trim() || !exampleTarget.trim() || !exampleLineage.trim()} onClick={async () => { try { const lineage = await Promise.all(exampleLineage.split(',').map(id => id.trim()).filter(Boolean).map(async id => { const document = await ragRequest(`/documents/${encodeURIComponent(id)}`); return { documentId: id, versionId: document.currentVersionId ?? document.versionId, sourceId: document.sourceId }; })); await action('/training-examples', { input: exampleInput, target: exampleTarget, lineage, splitGroup: lineage[0]?.sourceId }); setExampleInput(''); setExampleTarget(''); await load(); } catch (cause) { setStatus(cause.message); } }}><Plus size={14} />검토 후보 추가</button>{examples.map(example => <div className="rag-admin-row" key={example.id}><span>{(example.input ?? '').slice(0, 50)}<small>{example.reviewStatus ?? example.review_status} · {example.permissionStatus ?? '권한 확인'}</small></span><button type="button" disabled={busy} onClick={() => run(`/training-examples/${example.id}/review`, { decision: 'approve', permissionStatus: 'allowed', trainingPolicy: 'allowed', reason: '사용자 직접 검토·학습 이용 승인' })}>승인</button><button type="button" disabled={busy} onClick={() => run(`/training-examples/${example.id}/review`, { decision: 'reject', reason: '사용자 제외' })}>제외</button></div>)}</div>
    <div className="rag-tools-group"><h3>학습 스냅샷</h3><p className="muted small">승인 자료 · 계보 분할 · 평가셋 제외</p><button className="button secondary" type="button" disabled={busy} onClick={() => run('/training-snapshots', {})}><Upload size={14} />내보내기</button>{snapshots.map(snapshot => <div className="rag-admin-row" key={snapshot.id}><span>{snapshot.id.slice(0, 14)}<small>{snapshot.exampleCount ?? 0}개 · {snapshot.status ?? snapshot.state}</small></span><button type="button" disabled={busy} onClick={async () => { try { await onSave({ trainingSnapshotId: snapshot.id }); notify('학습 스냅샷 선택'); } catch (cause) { setStatus(cause.message); } }}>선택</button></div>)}</div>
    <div className="subtle-note"><CircleHelp size={16} /><span>실제 모델 훈련은 별도 학습 서버가 담당합니다. 수집·색인은 학습 횟수에 포함하지 않습니다.</span></div>{status && <p className="rag-tools-status" role="status">{status}</p>}
  </div></details>;
}
