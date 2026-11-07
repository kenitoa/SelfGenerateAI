import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { Activity, ArrowDownToLine, ArrowRight, ArrowUpRight, Bell, Check, CheckCheck, ChevronDown, ChevronRight, CircleHelp, Code2, Copy, Database, Ellipsis, ExternalLink, FileCode2, FileJson, FolderOpen, GitBranch, Globe2, HardDrive, Layers3, LayoutDashboard, Loader2, Menu, MessageSquare, MoreHorizontal, Pause, Play, Plus, Search, Send, Settings2, ShieldCheck, SlidersHorizontal, Sparkles, Sprout, Terminal, Trash2, Upload, Workflow, X, Zap } from 'lucide-react';
import './styles.css';
import './theme.css';
import { Moon, Sun, Monitor } from 'lucide-react';
import ModePicker from './ModePicker';
import { normalizeWorkMode, workModes } from './work-modes';
import TrainingView from './TrainingView';
import NasModelCatalog from './NasModelCatalog';
import { TrainingBranchSelect } from '../../LLM/web/Panel.jsx';
import { isNasCandidate } from './nas-models';
import useRag from './useRag';
import { consumeSse, login, metricNumber, normalizeThread, ragRequest, requestId, uploadFiles } from './rag-api';
import { ActivityRows, DatasetDetails, EvidenceCards, GrowthChart, MessageContent, RagAdvanced, SourceControls } from './RagPanels';
import './rag.css';

const icons = { overview: LayoutDashboard, workspace: Terminal, data: Database, training: Workflow, settings: Settings2 };
const labels = { overview: '개요', workspace: '작업', data: '데이터', training: '학습', settings: '설정' };
function readLocal(key, fallback) { try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; } }
function useLocal(key, fallback, normalize = value => value) {
  const [value, setValue] = useState(() => normalize(readLocal(key, fallback)));
  useEffect(() => { try { localStorage.setItem(key, JSON.stringify(value)); } catch {} }, [key, value]);
  return [value, setValue];
}
const number = n => metricNumber(n);
const dateToday = () => new Date().toLocaleDateString('ko-KR', { year: 'numeric', month: '2-digit', day: '2-digit' }).replaceAll(' ', '').replace(/\.$/, '');

function App() {
  const [page, setPage] = useState('overview');
  const [theme, setTheme] = useLocal('seed-theme-v1', 'system', value => ['light', 'dark', 'system'].includes(value) ? value : 'system');
  const [resolvedTheme, setResolvedTheme] = useState(() => document.documentElement.dataset.theme || 'light');
  const [menuOpen, setMenuOpen] = useState(false);
  const [modal, setModal] = useState(null);
  const [toast, setToast] = useState('');
  const [notifications, setNotifications] = useState(false);
  const [period, setPeriod] = useState('주간');
  const rag = useRag(period);
  const { records, sources, settings, setSettings, threads, setThreads, trainingConfig, setTrainingConfig } = rag;
  const [nasRam, setNasRam] = useLocal('seed-nas-ram-v2', 32, value => [4, 8, 16, 32].includes(value) ? value : 32);
  const [selectedModel, setSelectedModel] = useState('');
  const [modelEndpoint, setModelEndpoint] = useState('');
  const [activeThread, setActiveThread] = useState(null);
  const [prompt, setPrompt] = useState('');
  const [workMode, setWorkMode] = useLocal('seed-work-mode-v1', 'code', normalizeWorkMode);
  const [responding, setResponding] = useState(false);
  const [query, setQuery] = useState('');
  const [dataFilter, setDataFilter] = useState('전체');
  const [dataTab, setDataTab] = useState('데이터셋');
  const [trainingDraft, setTrainingDraft] = useState(trainingConfig);
  const [sourceName, setSourceName] = useState('');
  const [workspaceName, setWorkspaceName] = useState('');
  const [sourceUrl, setSourceUrl] = useState('');
  const [sourceKind, setSourceKind] = useState('web_document');
  const [formError, setFormError] = useState('');
  const [fileBusy, setFileBusy] = useState(false);
  const [retrievalEnabled, setRetrievalEnabled] = useState(true);
  const [allowUngrounded, setAllowUngrounded] = useState(false);
  const [knowledgeSource, setKnowledgeSource] = useState('');
  const [activeRun, setActiveRun] = useState(null);
  const [streamText, setStreamText] = useState('');
  const [runStatus, setRunStatus] = useState('');
  const [loginPassword, setLoginPassword] = useState('');
  const fileRef = useRef(null);
  const chatEnd = useRef(null);
  const composerRef = useRef(null);
  const toastTimer = useRef(null);
  const runController = useRef(null);
  const pendingMessageRequest = useRef(null);
  const pendingThreadRequest = useRef(null);
  const modalRef = useRef(null);
  const returnFocus = useRef(null);
  const notify = text => { setToast(text); clearTimeout(toastTimer.current); toastTimer.current = setTimeout(() => setToast(''), 3800); };
  const go = next => { setPage(next); setMenuOpen(false); setQuery(''); };
  const openModal = value => {
    setFormError('');
    setNotifications(false);
    setMenuOpen(false);
    if (value === 'profile') setWorkspaceName(settings.name);
    if (value === 'trainingConfig') setTrainingDraft({ ...trainingConfig });
    if (value === 'model') {
      setSelectedModel(settings.model);
      setModelEndpoint(settings.endpoint);
    }
    setModal(value);
  };
  async function applyModel(event) {
    event.preventDefault();
    if (!isNasCandidate(selectedModel, nasRam) && !selectedModel.startsWith('seed/')) { setFormError('기준 RAM에 맞는 모델 또는 검증된 조합을 선택하세요.'); return; }
    try {
      const url = new URL(modelEndpoint);
      if (!['http:', 'https:'].includes(url.protocol)) throw new Error();
    } catch { setFormError('올바른 http 또는 https 서버 주소를 입력하세요.'); return; }
    try {
      await rag.saveSettings({ model: selectedModel, endpoint: modelEndpoint.trim() });
      setModal(null); notify('모델 설정 저장 · 실제 설치·연결은 별도 확인');
    } catch (error) { setFormError(error.message); }
  }
  async function saveTrainingConfig(event) {
    event.preventDefault();
    const endpoint = trainingDraft.endpoint?.trim() || '';
    if (endpoint) {
      try { if (!['http:', 'https:'].includes(new URL(endpoint).protocol)) throw new Error(); }
      catch { setFormError('올바른 http 또는 https 학습 서버 주소를 입력하세요.'); return; }
    }
    try {
      await rag.saveSettings({ training: { ...trainingDraft, endpoint } });
      setTrainingConfig({ ...trainingDraft, endpoint }); setModal(null); notify('학습 서버 설정 저장');
    } catch (error) { setFormError(error.message); }
  }
  const startWork = (text = '', mode = 'code') => { setActiveThread(null); setPrompt(text); setWorkMode(mode); go('workspace'); };
  const selectedMode = workModes.find(mode => mode.id === workMode);
  function openThread(thread) {
    setActiveThread(thread.id);
    rag.loadThread(thread.id).catch(error => notify(error.message));
    const lastRequest = [...thread.messages].reverse().find(message => message.role === 'user');
    setWorkMode(normalizeWorkMode(lastRequest?.mode));
  }
  function selectStarter(mode, text) {
    setWorkMode(mode);
    setPrompt(text);
    requestAnimationFrame(() => composerRef.current?.focus());
  }
  useEffect(() => {
    const media = window.matchMedia('(prefers-color-scheme: dark)');
    const applyTheme = () => {
      const resolved = theme === 'system' ? (media.matches ? 'dark' : 'light') : theme;
      document.documentElement.dataset.theme = resolved;
      document.documentElement.style.colorScheme = resolved;
      document.querySelector('meta[name="theme-color"]')?.setAttribute('content', resolved === 'dark' ? '#10141d' : '#f7f8fa');
      setResolvedTheme(resolved);
    };
    applyTheme();
    if (theme === 'system') {
      media.addEventListener('change', applyTheme);
      return () => media.removeEventListener('change', applyTheme);
    }
  }, [theme]);
  useEffect(() => () => { clearTimeout(toastTimer.current); runController.current?.abort(); }, []);
  useEffect(() => { chatEnd.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' }); }, [threads, responding]);
  useEffect(() => {
    if (!modal) return;
    returnFocus.current = document.activeElement;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    const frame = requestAnimationFrame(() => (modalRef.current?.querySelector('[data-autofocus]') ?? modalRef.current?.querySelector('input, button, select'))?.focus());
    const onKey = e => {
      if (e.key === 'Escape') setModal(null);
      if (e.key === 'Tab') {
        const elements = modalRef.current?.querySelectorAll('button:not(:disabled), input, select, textarea, a[href]');
        if (!elements?.length) return;
        const first = elements[0], last = elements[elements.length - 1];
        if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
        else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
      }
    };
    document.addEventListener('keydown', onKey);
    return () => { cancelAnimationFrame(frame); document.body.style.overflow = previousOverflow; document.removeEventListener('keydown', onKey); returnFocus.current?.focus(); };
  }, [modal]);
  const currentMessages = threads.find(t => t.id === activeThread)?.messages ?? [];
  async function sendMessage(event) {
    event?.preventDefault();
    if (!prompt.trim() || responding) return;
    const text = prompt.trim();
    const requestMode = workMode;
    setResponding(true); setStreamText(''); setRunStatus('요청 중');
    const controller = new AbortController(); runController.current = controller;
    let id = activeThread;
    try {
      if (!id) {
        if (pendingThreadRequest.current?.text !== text) pendingThreadRequest.current = { text, key: requestId() };
        const thread = normalizeThread(await ragRequest('/threads', { method: 'POST', body: { title: text.slice(0, 40) }, idempotencyKey: pendingThreadRequest.current.key }));
        id = thread.id; setActiveThread(id); setThreads(old => [thread, ...old]);
        pendingThreadRequest.current = null;
      }
      const body = { mode: requestMode, content: text, retrieval: { enabled: retrievalEnabled, sourceIds: knowledgeSource ? [knowledgeSource] : [], languages: [] }, allowUngrounded };
      const identity = JSON.stringify({ threadId: id, ...body });
      if (pendingMessageRequest.current?.identity !== identity) pendingMessageRequest.current = { identity, key: requestId() };
      const key = pendingMessageRequest.current.key;
      const run = await ragRequest(`/threads/${encodeURIComponent(id)}/messages`, { method: 'POST', body: { clientRequestId: key, ...body }, idempotencyKey: key, signal: controller.signal });
      pendingMessageRequest.current = null;
      setActiveRun(run.runId); setPrompt(''); await rag.loadThread(id);
      try {
        const response = await ragRequest(run.eventsUrl, { raw: true, signal: controller.signal });
        await consumeSse(response, event => {
          const payload = event.data?.data ?? event.data?.payload ?? event.data;
          const type = event.type === 'message' ? (event.data?.type ?? event.data?.event ?? 'message') : event.type;
          if (type === 'token') setStreamText(old => old + (payload.delta ?? payload.text ?? ''));
          const names = { accepted: '요청 접수', retrieving: '근거 검색', context_ready: '근거 준비', verifying: '근거 검증', completed: '완료', failed: '실패', cancelled: '중단', insufficient_evidence: '자료 부족' };
          if (names[type]) setRunStatus(names[type]);
        }, { signal: controller.signal });
      } catch (streamError) {
        if (controller.signal.aborted) throw streamError;
        setRunStatus('스트림 중단 · 서버 상태 확인');
      }
      const terminal = new Set(['completed', 'failed', 'cancelled', 'insufficient_evidence']);
      let current = await ragRequest(`/generation-runs/${encodeURIComponent(run.runId)}`, { signal: controller.signal });
      for (let attempt = 0; !terminal.has(current.state ?? current.status) && attempt < 300; attempt++) {
        setRunStatus('서버 상태 확인');
        if (current.partialText) setStreamText(current.partialText);
        await new Promise((resolve, reject) => {
          const aborted = () => { clearTimeout(timer); reject(new DOMException('Aborted', 'AbortError')); };
          const timer = setTimeout(() => { controller.signal.removeEventListener('abort', aborted); resolve(); }, 1000);
          controller.signal.addEventListener('abort', aborted, { once: true });
        });
        current = await ragRequest(`/generation-runs/${encodeURIComponent(run.runId)}`, { signal: controller.signal });
      }
      await rag.loadThread(id); await rag.refresh();
    } catch (error) { if (error.name !== 'AbortError') notify(error.message); }
    finally { setResponding(false); setActiveRun(null); setStreamText(''); runController.current = null; }
  }
  async function importFiles(event) {
    const files = [...(event.target.files ?? [])];
    setFileBusy(true);
    try {
      if (!files.length) return;
      await uploadFiles(files); await rag.refresh(); notify(`${files.length}개 원문 저장 · 처리 요청`);
    } catch (error) { setFormError(error.message); setModal('importError'); }
    finally { setFileBusy(false); event.target.value = ''; }
  }
  async function addSource(event) {
    event.preventDefault();
    try { const url = new URL(sourceUrl); if (!['http:', 'https:'].includes(url.protocol)) throw new Error(); }
    catch { setFormError('올바른 http 또는 https 주소를 입력하세요.'); return; }
    try {
      await ragRequest('/sources', { method: 'POST', body: { name: sourceName.trim(), kind: sourceKind, locator: { url: sourceUrl.trim(), ...(sourceKind === 'git_repository' ? { ref: 'HEAD' } : {}) }, retrievalPolicy: 'approved', trainingPolicy: 'unknown', collectionEnabled: true } });
      setSourceName(''); setSourceUrl(''); setModal(null); await rag.refresh(); notify('소스 저장 · 수집 버튼으로 시작');
    } catch (error) { setFormError(error.message); }
  }
  async function copy(text) { try { await navigator.clipboard.writeText(text); notify('코드 복사 완료'); } catch { notify('브라우저에서 클립보드 권한을 확인하세요.'); } }
  const filtered = records.filter(r => (dataFilter === '전체' || r.status === dataFilter) && `${r.name} ${r.type} ${r.source}`.toLowerCase().includes(query.toLowerCase()));
  const importedCount = records.reduce((sum, r) => sum + (r.documentCount ?? 0), 0);
  const currentDate = new Intl.DateTimeFormat('ko-KR', { timeZone: 'Asia/Seoul', month: 'long', day: 'numeric', weekday: 'long' }).format(new Date());
  const metrics = rag.dashboard?.metrics;
  const stats = { collected: metrics?.collectedDocuments ?? null, ready: metrics?.indexedChunks ?? null, trained: metrics?.completedTrainingRuns ?? null, modelSize: metrics?.loadedModelParameters ?? null };
  const ragConnected = rag.connection === 'connected';
  const connectionLabel = ragConnected ? 'RAG 연결' : rag.connection === 'stale' ? '갱신 중단' : rag.connection === 'connecting' ? '연결 확인' : '서버 미연결';
  const inferenceConnected = rag.dashboard?.dependencies?.inference === 'ready' || rag.dashboard?.dependencies?.inference === 'connected';
  async function removeThread(id) { try { await ragRequest(`/threads/${encodeURIComponent(id)}`, { method: 'DELETE' }); if (activeThread === id) setActiveThread(null); await rag.refresh(); } catch (error) { notify(error.message); } }
  async function cancelRun() { if (!activeRun) return; try { await ragRequest(`/generation-runs/${encodeURIComponent(activeRun)}/cancel`, { method: 'POST', body: {} }); setRunStatus('중단 요청'); } catch (error) { notify(error.message); } }
  async function saveAllSettings(event) { event?.preventDefault(); try { await rag.saveSettings(); notify('서버 설정 저장 완료'); } catch (error) { notify(error.message); } }

  return <div className="app-shell">
    {menuOpen && <button className="sidebar-scrim" onClick={() => setMenuOpen(false)} aria-label="메뉴 닫기" />}
    <aside className={`sidebar ${menuOpen ? 'is-open' : ''}`}>
      <a className="brand" href="#" onClick={e => { e.preventDefault(); go('overview'); }}><span className="brand-mark"><Sprout size={23} strokeWidth={2} /></span><span>SEED<span className="brand-dot">.</span></span></a>
      <button className="workspace-switch" onClick={() => openModal('profile')} aria-haspopup="dialog"><span className="workspace-avatar">S</span><span><strong>{settings.name}</strong><small>Personal workspace</small></span><ChevronDown size={14} /></button>
      <span className="nav-label">WORKSPACE</span>
      <nav aria-label="주 메뉴">{Object.entries(labels).filter(([key]) => key !== 'settings').map(([key, label]) => { const Icon = icons[key]; return <button key={key} className={`nav-item ${page === key ? 'active' : ''}`} aria-current={page === key ? 'page' : undefined} onClick={() => go(key)}><Icon size={19} /><span>{label}</span>{key === 'data' && <span className="nav-count">{records.length}</span>}{page === key && <span className="nav-active-dot" />}</button>; })}</nav>
      <div className="sidebar-bottom"><div className="device-card"><div className="device-heading"><span className="device-icon"><HardDrive size={17} /></span><strong>Synology DS923+</strong><span className="dot gray" /></div><div className="device-line"><span>서버 연결</span><span>{connectionLabel}</span></div><div className="device-meter"><i /></div><div className="device-foot"><span>{rag.system?.observationScope === 'host' ? '호스트 관측' : '프로세스 관측'}</span><span>{rag.system?.memory?.processRssBytes ? `${metricNumber(rag.system.memory.processRssBytes / 1048576)} MiB` : '—'}</span></div></div><button className={`nav-item ${page === 'settings' ? 'active' : ''}`} onClick={() => go('settings')}><Settings2 size={19} /><span>설정</span></button><button className="nav-item" onClick={() => openModal('help')}><CircleHelp size={19} /><span>도움말</span><ArrowUpRight size={15} className="nav-end" /></button><div className="profile"><span className="profile-avatar">ME</span><span><strong>나의 공간</strong><small>Local · Private</small></span><span className="profile-status" /></div></div>
    </aside>
    <div className="main-shell">
      <header className="topbar"><div className="breadcrumb"><button className="icon-button mobile-menu" onClick={() => setMenuOpen(true)} aria-label="메뉴 열기"><Menu size={20} /></button><span>워크스페이스</span><ChevronRight size={13} /><strong>{labels[page]}</strong></div><div className="topbar-actions"><button className="preview-badge" onClick={() => openModal('preview')}><span className="dot lavender" />{connectionLabel}</button><button className="icon-button theme-toggle" aria-label={resolvedTheme === 'dark' ? '라이트 모드' : '다크 모드'} title={resolvedTheme === 'dark' ? '라이트 모드' : '다크 모드'} onClick={() => setTheme(resolvedTheme === 'dark' ? 'light' : 'dark')}>{resolvedTheme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}</button><span className="topbar-divider" /><div className="notification-wrap"><button className="icon-button notification-button" aria-label="알림" aria-expanded={notifications} onClick={() => setNotifications(!notifications)}><Bell size={18} /><i /></button>{notifications && <div className="notification-popover"><strong>알림</strong><span><span className="dot lavender" />{connectionLabel}</span><p>{rag.error || (rag.activities[0]?.message ?? '실제 서버 상태 기준')}</p><button onClick={() => { setNotifications(false); go('settings'); }}>연결 설정 <ArrowRight size={14} /></button></div>}</div><button type="button" className="top-avatar profile-trigger" aria-label="마이페이지" title="마이페이지" aria-haspopup="dialog" aria-expanded={modal === 'profile'} onClick={() => openModal('profile')}>S</button></div></header>
      <main className={`main-content ${page === 'workspace' ? 'chat-main' : ''}`}>
        {page !== 'workspace' && rag.connection !== 'connected' && <div className="rag-connection" role="status"><span><span className="dot lavender" />{connectionLabel}{rag.error ? ` · ${rag.error}` : ''}<small>{rag.lastReceivedAt ? `마지막 수신 ${new Date(rag.lastReceivedAt).toLocaleTimeString('ko-KR')}` : ''}</small></span><button onClick={rag.refresh}>다시 연결</button></div>}
        {page === 'overview' && <>
          <div className="page-heading" style={{ justifyContent: 'flex-end' }}><div className="heading-right"><span className="date-label">{currentDate}</span><button className="button primary" onClick={() => startWork()}><Plus size={16} />새 작업</button></div></div>
          <div className="hero-grid">
            <section className="model-card"><div className="model-card-top"><span className="overline"><span className="dot green" />MY LOCAL INTELLIGENCE</span><span className="dark-tag">{inferenceConnected ? '모델 연결' : '모델 미연결'}</span></div><div className="model-copy"><h2>자가학습 AI</h2><button className="hero-link" onClick={() => go('training')}>성장 기록 <ArrowUpRight size={16} /></button></div><div className="orbit-art" aria-hidden="true"><div className="orbit orbit-one" /><div className="orbit orbit-two" /><div className="orbit orbit-three" /><span className="orbit-dot one" /><span className="orbit-dot two" /><div className="seed-core"><Sprout size={53} strokeWidth={1.1} /></div><div className="orbit-coordinate">SEED / 001</div></div><div className="model-card-footer"><span><span className="dot green" />로컬 우선</span><span>CODE SPECIALIST <Code2 size={13} /></span></div></section>
            <section className="card quick-card"><div className="quick-actions"><button onClick={() => startWork('작성할 코드: ')}><span className="quick-icon purple"><Code2 size={18} /></span><span><strong>코드 작성</strong><small>아이디어 → 코드</small></span><ArrowUpRight size={16} /></button><button onClick={() => startWork('아래 코드를 검토해줘.\n\n', 'review')}><span className="quick-icon blue"><ShieldCheck size={18} /></span><span><strong>코드 검토</strong><small>버그 · 개선 · 리팩토링</small></span><ArrowUpRight size={16} /></button><button onClick={() => { go('data'); fileRef.current?.click(); }}><span className="quick-icon green-bg"><Database size={18} /></span><span><strong>데이터 추가</strong><small>파일 → 데이터셋</small></span><ArrowUpRight size={16} /></button></div></section>
          </div>
          <div className="metrics-grid">{[
            { label: '수집 데이터', value: number(stats.collected), unit: '건', text: ragConnected ? '저장 문서' : connectionLabel, Icon: Database, color: 'purple' },
            { label: '검색 준비', value: number(stats.ready), unit: '건', text: rag.dashboard?.searchMode === 'hybrid' ? '하이브리드 검색' : '키워드 검색', Icon: Layers3, color: 'blue' },
            { label: '학습 횟수', value: number(stats.trained), unit: '회', text: rag.dashboard?.dependencies?.trainer === 'ready' ? '실제 완료 기록' : '학습 서버 미연결', Icon: GitBranch, color: 'green-bg' },
            { label: '모델 크기', value: metricNumber(stats.modelSize, 1e9), unit: 'B', text: inferenceConnected ? '연결 모델' : '모델 미연결', Icon: Zap, color: 'orange' },
          ].map(({ label, value, unit, text, Icon, color }) => <section className="card metric" key={label}><div className="metric-top"><span>{label}</span><span className={`metric-icon ${color}`}><Icon size={17} /></span></div><div className="metric-value">{value}<span>{unit}</span></div><div className="metric-foot"><span>{text}</span></div></section>)}</div>
          <div className="middle-grid">
            <section className="card growth-card"><div className="section-heading"><div className="segmented">{['주간', '월간'].map(p => <button key={p} className={period === p ? 'selected' : ''} onClick={() => setPeriod(p)}>{p}</button>)}</div></div><div className="chart-legend"><span><i className="legend-dot" />검색 준비</span><strong>{number(stats.ready)}<small>건</small></strong></div><GrowthChart series={rag.dashboard?.series ?? []} /></section>
            <section className="card pipeline-card"><div className="section-heading" style={{ justifyContent: 'flex-end' }}><button className="icon-button" aria-label="데이터 페이지" onClick={() => go('data')}><ArrowUpRight size={17} /></button></div><div className="pipeline-steps">{[{ title: '수집', desc: '코드 · 문서', value: stats.collected, Icon: Globe2 }, { title: '정제', desc: '중복 · 품질 검토', value: rag.dashboard?.pipeline?.processing ?? 0, Icon: SlidersHorizontal }, { title: '색인', desc: '검색 청크 구성', value: stats.ready, Icon: Layers3 }, { title: '검색 준비', desc: '승인된 자료', value: rag.dashboard?.pipeline?.readyDocuments ?? 0, Icon: Sprout }].map(({ title, desc, value, Icon }, i) => <button className="pipeline-step" key={title} onClick={() => go('data')}><span className="pipeline-icon"><Icon size={17} /></span><span><strong>{title}</strong><small>{desc}</small></span><span className="pipeline-value">{number(value)}건</span></button>)}</div><div className="pipeline-note"><span className="dot gray" />{rag.dashboard?.searchMode === 'hybrid' ? '하이브리드' : '키워드 검색'} <span>{connectionLabel}</span></div></section>
          </div>
          <section className="card activity-card"><div className="section-heading"><h2>최근 활동</h2><button className="text-button" onClick={() => go('data')}>데이터 보기 <ArrowRight size={14} /></button></div>{rag.activities.length ? <ActivityRows activities={rag.activities} /> : <Empty Icon={Activity} title="활동 없음" description="수집·처리 대기" />}</section>
          <footer className="page-footer"><span><Sprout size={13} />작게 시작해, 꾸준히 성장하도록.</span><span>SEED v0.1.0 <span className="footer-dot">·</span> Interface ? RAG</span></footer>
        </>}

        {page === 'workspace' && <div className="work-layout"><aside className="thread-sidebar"><div className="section-heading"><h2>작업 기록</h2><button className="icon-button" onClick={() => startWork()} aria-label="새 작업"><Plus size={18} /></button></div>{threads.length ? threads.map(thread => <div key={thread.id} className={`thread-item ${activeThread === thread.id ? 'selected' : ''}`}><button onClick={() => openThread(thread)}><MessageSquare size={15} /><span>{thread.title}</span></button><button className="icon-button" aria-label={`${thread.title} 삭제`} onClick={() => removeThread(thread.id)}><Trash2 size={13} /></button></div>) : <p className="empty-thread">첫 작업을 시작하세요.</p>}<div className="thread-bottom"><ShieldCheck size={15} />서버에 저장</div></aside><section className="chat-panel"><div className="chat-toolbar"><span><span className="dot lavender" /><span className="chat-model-name" title={settings.model}>{settings.model || '모델 선택'}</span><span className="subtle-label">{inferenceConnected ? '연결' : '미연결'}</span></span><button className="text-button" aria-haspopup="dialog" onClick={() => openModal('model')}>모델 설정 <Settings2 size={14} /></button></div><div className="chat-messages">{currentMessages.length === 0 ? <div className="chat-welcome"><span className="welcome-mark"><Sprout size={36} strokeWidth={1.4} /></span><span className="eyebrow">YOUR PERSONAL CODING SPACE</span><h1>무엇을 만들까요?</h1><p>작은 아이디어에서, 다음 프로젝트까지.</p><div className="prompt-grid">{[{ mode: 'code', Icon: Code2, title: '코드 작성', text: 'Python으로 파일 읽기 함수를 작성해줘.' }, { mode: 'review', Icon: ShieldCheck, title: '코드 검토', text: '아래 코드의 버그를 검토해줘.\n\n' }, { mode: 'refactor', Icon: GitBranch, title: '리팩토링', text: '아래 함수를 리팩토링해줘.\n\n' }, { mode: 'test', Icon: FileCode2, title: '테스트 작성', text: '아래 함수의 테스트를 작성해줘.\n\n' }].map(({ mode, Icon, title, text }) => <button key={title} onClick={() => selectStarter(mode, text)}><Icon size={18} /><strong>{title}</strong><ArrowUpRight size={15} /></button>)}</div></div> : <div className="message-list">{currentMessages.map((message, i) => <div className={`message ${message.role}`} key={i}><span className="message-avatar">{message.role === 'user' ? 'ME' : <Sprout size={18} />}</span><div><strong>{message.role === 'user' ? '나' : 'SEED'}{message.mode && <span className="message-mode">{workModes.find(mode => mode.id === message.mode)?.label}</span>}{message.role === 'assistant' && <span className="sample-inline">{message.status === 'completed' ? '응답' : message.status === 'insufficient_evidence' ? '자료 부족' : message.status === 'failed' ? '생성 실패' : message.status ?? '응답'}</span>}</strong><MessageContent text={message.text} copy={copy} />{message.code && <div className="code-block"><div><span>코드 · 미실행</span><button onClick={() => copy(message.code)}><Copy size={13} />복사</button></div><pre><code>{message.code}</code></pre></div>}<EvidenceCards message={message} notify={notify} /></div></div>)}{streamText && <div className="rag-partial">{streamText}</div>}{responding && <div className="thinking"><Loader2 size={15} className="spin" />{runStatus || '서버 처리 중'}</div>}<div ref={chatEnd} /></div>}</div><div className="composer-area"><form className="composer" onSubmit={sendMessage}><textarea ref={composerRef} aria-label={`${selectedMode.label} 요청`} placeholder={selectedMode.placeholder} value={prompt} onChange={e => setPrompt(e.target.value)} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); sendMessage(); } }} rows={3} /><div className="composer-bottom"><ModePicker value={workMode} onChange={setWorkMode} disabled={responding} onSettings={() => openModal('model')} /><button className="send-button" disabled={!prompt.trim() || responding} aria-label="요청 보내기" type="submit"><ArrowRight size={19} /></button></div></form><div className="rag-composer-options"><label><input type="checkbox" checked={retrievalEnabled} onChange={e => setRetrievalEnabled(e.target.checked)} />RAG</label><select aria-label="검색 소스" value={knowledgeSource} onChange={e => setKnowledgeSource(e.target.value)}><option value="">전체 승인 자료</option>{sources.map(source => <option key={source.id} value={source.id}>{source.name}</option>)}</select><label><input type="checkbox" checked={allowUngrounded} onChange={e => setAllowUngrounded(e.target.checked)} />근거 없이 진행</label>{activeRun && <button type="button" className="text-button" onClick={cancelRun}>중단</button>}</div><p>{rag.error || (inferenceConnected ? '실제 서버 응답 · 코드 실행은 별도' : '모델 미연결 · 검색 근거 확인 가능')}</p></div></section></div>}

        {page === 'data' && <><div className="page-heading" style={{ justifyContent: 'flex-end' }}><div className="heading-buttons"><button className="button secondary" onClick={() => openModal('source')}><Plus size={16} />소스 추가</button><button className="button primary" disabled={fileBusy} onClick={() => fileRef.current?.click()}>{fileBusy ? <Loader2 className="spin" size={16} /> : <Upload size={16} />}파일 추가</button></div></div><div className="info-banner"><ShieldCheck size={17} /><span>원문을 서버에 저장하고 정제·색인을 진행합니다. 검색 승인과 학습 승인은 별도입니다.</span></div><div className="data-summary">{[{ label: '데이터셋', value: `${records.length}개`, Icon: Database }, { label: '저장 문서', value: `${number(importedCount)}건`, Icon: FileJson }, { label: '등록 소스', value: `${sources.length}개`, Icon: Globe2 }].map(({ label, value, Icon }) => <div className="card summary-item" key={label}><span className="quick-icon purple"><Icon size={20} /></span><div><span>{label}</span><strong>{value}</strong></div></div>)}</div><section className="card library-card"><div className="library-tabs" role="tablist">{['데이터셋', '소스 목록'].map(t => <button role="tab" aria-selected={dataTab === t} key={t} className={dataTab === t ? 'selected' : ''} onClick={() => setDataTab(t)}>{t}<span>{t === '데이터셋' ? records.length : sources.length}</span></button>)}</div>{dataTab === '데이터셋' ? <><div className="table-toolbar"><div className="filter-group">{['전체', '준비', '검토', '분류', '실패', '제외'].map(filter => <button key={filter} className={dataFilter === filter ? 'selected' : ''} onClick={() => setDataFilter(filter)}>{filter}</button>)}</div><label className="search-field"><Search size={15} /><input aria-label="데이터 검색" placeholder="데이터 검색" value={query} onChange={e => setQuery(e.target.value)} /></label></div><div className="table-scroll"><table><thead><tr><th>데이터셋</th><th>언어</th><th>검색 청크</th><th>상태</th><th>추가일</th><th><span className="sr-only">작업</span></th></tr></thead><tbody>{filtered.map(record => <tr key={record.id}><td><button className="dataset-name" onClick={() => openModal({ type: 'record', record })}><span className="file-icon"><FileCode2 size={19} /></span><span><strong>{record.name}</strong><small>{record.source}</small></span></button></td><td><span className="language-tag">{record.type}</span></td><td className="numeric">{number(record.count)}</td><td><span className={`status-pill ${record.status === '준비' ? 'ready' : record.status === '검토' ? 'review' : 'pending'}`}><span className="dot" />{record.status}</span></td><td className="table-date">{record.date}</td><td><button className="icon-button" aria-label={`${record.name} 상세`} onClick={() => openModal({ type: 'record', record })}><ChevronRight size={16} /></button></td></tr>)}</tbody></table>{filtered.length === 0 && <Empty Icon={records.length ? Search : Database} title={records.length ? "검색 결과 없음" : "데이터 없음"} description={records.length ? "다른 검색어나 필터를 사용하세요." : "파일이나 소스를 추가하세요."} />}</div><div className="table-footer"><span>{filtered.length}개 데이터셋</span><span>서버 원문 · 실제 처리 상태</span></div></> : <div className="sources-list">{sources.length ? sources.map(source => <SourceControls key={source.id} source={source} refresh={rag.refresh} notify={notify} />) : <Empty Icon={Globe2} title="첫 번째 소스" description="문서나 저장소 주소를 등록하세요."><button className="button secondary" onClick={() => openModal('source')}><Plus size={15} />소스 추가</button></Empty>}</div>}</section><div className="upload-note"><FileJson size={16} />코드 · 문서 · JSON · JSONL<span>파일당 최대 10 MiB</span><span>원본은 서버에 저장됩니다.</span></div></>}

        {page === 'training' && <TrainingView settings={settings} config={trainingConfig} onConfig={() => openModal('trainingConfig')} nasRam={nasRam} onRamChange={setNasRam} onModelSelect={model => rag.saveSettings({ model }).catch(error => notify(error.message))} />}

        {page === 'settings' && <><PageHeading eyebrow="MAKE IT YOURS" title="설정" description="나의 환경에 맞게, 가볍게." /><form className="settings-form" onSubmit={saveAllSettings}><section className="card settings-section"><div className="settings-section-title"><span className="quick-icon purple">{resolvedTheme === 'dark' ? <Moon size={20} /> : <Sun size={20} />}</span><div><h2>화면 테마</h2><p>선택 즉시 적용 · 자동 저장</p></div></div><ThemeOptions theme={theme} onChange={setTheme} /></section><section className="card settings-section"><div className="settings-section-title"><span className="quick-icon blue"><HardDrive size={20} /></span><div><h2>모델 연결</h2><p>RAG 서버에서 추론 엔진 호출</p></div><span className="status-pill pending">{inferenceConnected ? '연결' : '미연결'}</span></div><div className="settings-fields"><label className="field"><span>서버 주소</span><input required type="url" value={settings.endpoint} onChange={e => setSettings(s => ({ ...s, endpoint: e.target.value }))} /><small>예: Ollama · http://NAS-IP:11434</small></label><label className="field"><span>모델</span><input readOnly value={settings.model} /><small>DS923+ CPU 추론 후보</small><button type="button" className="button secondary" onClick={() => openModal('model')}>모델 선택</button></label></div><div className="subtle-note"><CircleHelp size={16} /><span>주소는 NAS의 RAG 서버 기준입니다. 실제 설치·응답은 아래 연결 확인으로 검사합니다.</span></div></section><section className="card settings-section"><div className="settings-section-title"><span className="quick-icon green-bg"><Activity size={20} /></span><div><h2>리소스 · 예약</h2><p>NAS 운영 환경을 위한 설정</p></div></div><div className="settings-fields"><label className="field"><span>메모리 제한</span><select value={settings.memory} onChange={e => setSettings(s => ({ ...s, memory: e.target.value }))}>{['1', '2', '4', '8', '16', '24', '32'].map(n => <option value={n} key={n}>{n} GB</option>)}</select></label><label className="field"><span>학습 시간</span><input type="time" required value={settings.schedule} onChange={e => setSettings(s => ({ ...s, schedule: e.target.value }))} /></label></div><div className="toggle-row"><div><strong>자동 학습 예약</strong><small>서버 예약 정책 · 학습 엔진 연결 시 실행</small></div><button type="button" className={`toggle ${settings.auto ? 'on' : ''}`} role="switch" aria-label="자동 학습 예약" aria-checked={settings.auto} onClick={() => setSettings(s => ({ ...s, auto: !s.auto }))}><span /></button></div></section><RagAdvanced settings={settings} trainingConfig={trainingConfig} capabilities={rag.capabilities} onSave={rag.saveSettings} notify={notify} refresh={rag.refresh} /><div className="settings-actions"><span><ShieldCheck size={14} />서버 저장 · 테마는 브라우저</span><button type="submit" className="button primary"><Check size={16} />설정 저장</button></div></form></>}
      </main>
    </div>
    <input type="file" ref={fileRef} accept=".json,.jsonl,.md,.txt,.py,.js,.jsx,.ts,.tsx,.sql,.html,.css,.yml,.yaml,.toml" multiple hidden onChange={importFiles} />
    {rag.auth?.authenticated === false && <div className="modal-backdrop rag-auth"><section className="modal" role="dialog" aria-modal="true" aria-labelledby="auth-title"><div className="modal-heading"><h2 id="auth-title">SEED 로그인</h2></div>{rag.auth.bootstrapRequired ? <p className="modal-description">서버 관리자 계정이 아직 없습니다. RAG 초기화 명령으로 계정을 만든 뒤 다시 연결하세요.</p> : <form onSubmit={async e => { e.preventDefault(); try { await login(loginPassword); setLoginPassword(''); setFormError(''); await rag.refresh(); } catch (error) { setFormError(error.message); } }}><label className="field"><span>비밀번호</span><input type="password" autoComplete="current-password" required value={loginPassword} onChange={e => setLoginPassword(e.target.value)} /></label>{formError && <p className="form-error" role="alert">{formError}</p>}<button className="button primary modal-submit" type="submit">로그인</button></form>}<button className="text-button" style={{ marginTop: 18 }} onClick={rag.refresh}>다시 연결</button></section></div>}
    {toast && <div className="toast" role="status"><span><Check size={16} /></span>{toast}<button aria-label="알림 닫기" onClick={() => setToast('')}><X size={14} /></button></div>}
    {modal && <div className="modal-backdrop" onClick={e => { if (e.target === e.currentTarget) setModal(null); }}><section ref={modalRef} className={`modal ${modal === 'profile' ? 'profile-modal' : modal === 'model' ? 'model-modal' : ''}`} role="dialog" aria-modal="true" aria-labelledby="modal-title"><div className="modal-heading"><h2 id="modal-title">{modal === 'model' ? '모델 선택' : modal === 'profile' ? '마이페이지' : modal === 'source' ? '소스 추가' : modal === 'trainingConfig' ? '학습 설정' : modal === 'help' ? 'SEED 사용 안내' : modal === 'importError' ? '파일 확인' : modal?.type === 'record' ? '데이터셋 상세' : 'RAG 연결 상태'}</h2><button className="icon-button" onClick={() => setModal(null)} aria-label="닫기"><X size={19} /></button></div>
      {modal === 'model' ? <form className="model-picker-form" onSubmit={applyModel}><NasModelCatalog value={selectedModel} onSelect={model => { setSelectedModel(model); setFormError(''); }} ram={nasRam} onRamChange={setNasRam} /><label className="field model-server"><span>추론 서버</span><input required type="url" value={modelEndpoint} placeholder="http://NAS-IP:11434" onChange={e => { setModelEndpoint(e.target.value); setFormError(''); }} /></label>{formError && <p className="form-error" role="alert">{formError}</p>}<div className="model-panel-note"><CircleHelp size={14} /><span>설정 저장 · 모델 설치·자동 다운로드 없음</span></div><div className="model-panel-actions"><button type="button" className="button secondary" onClick={() => setModal(null)}>취소</button><button className="button primary" type="submit" disabled={!isNasCandidate(selectedModel, nasRam) && !selectedModel.startsWith('seed/')}><Check size={15} />적용</button></div></form>
      : modal === 'profile' ? <><div className="mypage-identity"><span className="mypage-avatar">S</span><div><strong>{settings.name}</strong><small>개인 워크스페이스</small></div><span className="status-pill pending">서버</span></div><form className="mypage-form" onSubmit={async e => { e.preventDefault(); const name = workspaceName.trim(); if (!name) { setFormError('이름을 입력하세요.'); return; } try { await rag.saveSettings({ name }); setModal(null); notify('워크스페이스 저장 완료'); } catch (error) { setFormError(error.message); } }}><label className="field"><span>워크스페이스 이름</span><input data-autofocus required maxLength={24} value={workspaceName} onChange={e => { setWorkspaceName(e.target.value); setFormError(''); }} aria-invalid={Boolean(formError)} aria-describedby={formError ? 'profile-error' : undefined} /></label>{formError && <p id="profile-error" className="form-error" role="alert">{formError}</p>}<div className="mypage-theme"><span className="mypage-label">테마</span><ThemeOptions theme={theme} onChange={setTheme} /></div><button type="button" className="mypage-settings-link" onClick={() => { setModal(null); go('settings'); }}><span className="quick-icon blue"><Settings2 size={17} /></span><span><strong>환경 설정</strong><small>모델 · 리소스 · 예약</small></span><ChevronRight size={16} /></button><div className="mypage-footer"><span><ShieldCheck size={13} />서버 저장</span><button className="button primary" type="submit"><Check size={15} />저장</button></div></form></>
      : modal === 'source' ? <form onSubmit={addSource}><p className="modal-description">승인할 웹 문서·공개 저장소를 등록합니다.</p><label className="field"><span>종류</span><select value={sourceKind} onChange={event => setSourceKind(event.target.value)}><option value="web_document">웹 문서</option><option value="git_repository">Git 저장소</option></select></label><label className="field"><span>소스 이름</span><input autoFocus required maxLength={60} placeholder="예: Python 공식 문서" value={sourceName} onChange={e => setSourceName(e.target.value)} /></label><label className="field"><span>주소</span><input required type="url" placeholder="https://" value={sourceUrl} onChange={e => setSourceUrl(e.target.value)} /></label>{formError && <p className="form-error" role="alert">{formError}</p>}<div className="subtle-note"><CircleHelp size={16} /><span>등록 후 수집을 실행하세요. 검색 사용을 승인하며 학습 이용은 별도 검토합니다.</span></div><button className="button primary modal-submit" type="submit"><Plus size={16} />소스 저장</button></form>
      : modal === 'trainingConfig' ? <form onSubmit={saveTrainingConfig}><TrainingBranchSelect value={trainingDraft.branchId} onChange={branchId => setTrainingDraft(s => ({ ...s, branchId }))} /><label className="field"><span>학습 서버</span><input type="url" placeholder="http://NAS-IP:8000" value={trainingDraft.endpoint || ''} onChange={e => setTrainingDraft(s => ({ ...s, endpoint: e.target.value }))} /><small>학습 API 주소 · 추론 서버와 별도</small></label><label className="field"><span>에포크</span><input type="number" required min="1" max="100" value={trainingDraft.epochs} onChange={e => setTrainingDraft(s => ({ ...s, epochs: e.target.value }))} /></label><label className="field"><span>배치 크기</span><select value={trainingDraft.batch} onChange={e => setTrainingDraft(s => ({ ...s, batch: e.target.value }))}>{['1', '2', '4', '8'].map(n => <option key={n}>{n}</option>)}</select></label><label className="field"><span>학습률</span><input type="number" required min="0.000001" max="1" step="any" value={trainingDraft.rate} onChange={e => setTrainingDraft(s => ({ ...s, rate: e.target.value }))} /></label>{formError && <p className="form-error" role="alert">{formError}</p>}<div className="subtle-note"><CircleHelp size={16} /><span>진행률은 학습 서버에서 수신한 스텝만 반영합니다.</span></div><button className="button primary modal-submit" type="submit"><Check size={16} />저장</button></form>
      : modal?.type === 'record' ? <><div className="record-detail-title"><span className="quick-icon purple"><FileCode2 size={24} /></span><h3>{modal.record.name}</h3></div><DatasetDetails record={records.find(record => record.id === modal.record.id) ?? modal.record} notify={notify} refresh={rag.refresh} close={() => setModal(null)} /></>
      : modal === 'importError' ? <><p className="form-error" role="alert">{formError}</p><p className="modal-description">UTF-8 코드·문서 또는 JSON 배열/JSONL 객체 파일을 사용하세요. 오류 파일은 등록되지 않습니다.</p><button className="button primary modal-submit" onClick={() => setModal(null)}>확인</button></>
      : modal === 'help' ? <><p className="modal-description">작게 시작해, 꾸준히 성장하는 나만의 AI 공간.</p><div className="help-steps">{[['01', '작업', '요청 입력 · 코드 확인 · 기록 저장'], ['02', '데이터', '파일 확인 · 데이터 탐색 · 소스 등록'], ['03', '학습', '학습 설정 · 실제 진행률 · 서버 기록'], ['04', '설정', '모델 주소 · 자원 · 예약 설정']].map(([n, title, desc]) => <div key={n}><span>{n}</span><div><strong>{title}</strong><p>{desc}</p></div></div>)}</div><div className="subtle-note"><CircleHelp size={16} /><span>작업은 RAG 서버와 연결됩니다. 실제 학습 진행률은 별도 학습 서버가 보고합니다.</span></div></>
      : <><div className="preview-modal-icon"><Sprout size={34} /></div><p className="modal-description">독립 웹 인터페이스 · 로컬 실행</p><div className="preview-feature"><Check size={16} /><span>페이지 이동 · 데이터 검색 · 설정 저장</span></div><div className="preview-feature"><Check size={16} /><span>파일 형식 확인 · 대화 기록 · 학습 상태 연결</span></div><div className="preview-feature muted"><Workflow size={16} /><span>Interface ↔ RAG (Pre ↔ Post) · 훈련 엔진은 별도</span></div><div className="subtle-note"><ShieldCheck size={16} /><span>원문·대화·설정은 서버에 저장됩니다. 관측값과 미연결 상태를 구분하며 테마만 브라우저에 저장됩니다.</span></div><button className="button primary modal-submit" onClick={() => setModal(null)}>확인</button></>}
    </section></div>}
  </div>;
}

function ThemeOptions({ theme, onChange }) { return <div className="theme-options" role="group" aria-label="화면 테마">{[{ value: 'light', label: '라이트', Icon: Sun }, { value: 'dark', label: '다크', Icon: Moon }, { value: 'system', label: '시스템', Icon: Monitor }].map(({ value, label, Icon }) => <button key={value} type="button" className={`theme-option ${theme === value ? 'selected' : ''}`} aria-pressed={theme === value} onClick={() => onChange(value)}><Icon size={17} /><span>{label}</span>{theme === value && <Check size={14} className="theme-check" />}</button>)}</div>; }

function PageHeading({ eyebrow, title, description, children }) { return <div className="page-heading"><div><div className="eyebrow">{eyebrow}</div><h1>{title}<span className="title-dot">.</span></h1><p>{description}</p></div>{children && <div className="heading-buttons">{children}</div>}</div>; }
function Empty({ Icon, title, description, children }) { return <div className="empty-state"><span><Icon size={26} strokeWidth={1.5} /></span><strong>{title}</strong><p>{description}</p>{children}</div>; }

createRoot(document.getElementById('root')).render(<App />);
