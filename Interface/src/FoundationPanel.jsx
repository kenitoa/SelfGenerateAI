import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Activity, ArrowRight, Check, Clock3, Cpu, Database, FlaskConical, Layers3, Loader2, Pause, Play, Plus, RefreshCw, Sprout, Square } from 'lucide-react';
import { ragRequest } from './rag-api';
import { actualNumber, collection, foundationProgress, foundationRequest, reasonLabel, stageLabel } from './foundation-api';
import './FoundationPanel.css';
import FoundationSources from './FoundationSources';
import AutoTrainingPanel from './AutoTrainingPanel';

const stages = [['자료', Database], ['토큰', Layers3], ['초기화', Sprout], ['학습', Activity], ['검증', FlaskConical], ['사용', Check]];
const stageIndex = value => {
  if (['usable'].includes(value)) return 5;
  if (['foundation_candidate', 'foundation_qualified', 'assistant_candidate'].includes(value)) return 4;
  if (['pretraining', 'instruction_training', 'paused'].includes(value)) return 3;
  if (['initializing', 'genesis_ready'].includes(value)) return 2;
  if (['tokenizer_training', 'tokenizer_ready'].includes(value)) return 1;
  return 0;
};
const number = (value, fraction = 0) => actualNumber(value) === null ? '—' : value.toLocaleString('ko-KR', { maximumFractionDigits: fraction });
const entity = (project, key, identity) => collection(project?.[key]).find(item => item.id === identity)
  ?? [...collection(project?.[key])].sort((a, b) => String(b.createdAt ?? '').localeCompare(String(a.createdAt ?? '')))[0];
const valid = item => item && ['ready', 'verified', 'completed', 'qualified', 'usable'].includes(item.state ?? item.status);
const policyMode = value => value === 'night' ? 'night_window' : ['always_on_idle', 'night_window', 'manual'].includes(value) ? value : 'always_on_idle';
const acceptanceNames = ['NAS 배포', '원문', 'SFT', 'NAS 실측', '운영 학습', '독립 품질', '서비스', 'NAS 격리', '자가 수정', '전문화', '장기 운영', '장애 복원', '권한 철회', '화면 보안', '자료 선택', '대용량', '교육 과정', '작업별 검증', '오프라인 이관', '포맷 · 증류', '구조 성장', '선택 재학습', '보존 · 암호화'];

function ProjectFacts({ project, metrics }) {
  const config = project.config ?? {};
  const externalCount = Array.isArray(project.pretrainedWeightArtifactIds) ? project.pretrainedWeightArtifactIds.length : null;
  const initial = project.origin === 'architecture_growth' ? '기존 자체 가중치 · 구조 변환'
    : project.origin === 'random_init' ? '무작위' : '출처 확인 필요';
  return <dl className="foundation-facts">
    <div><dt>구조</dt><dd>{number(config.num_hidden_layers)} layers · hidden {number(config.hidden_size)} · vocab {number(config.vocab_size)}</dd></div>
    <div><dt>초기 가중치</dt><dd>{initial} · 외부 {number(externalCount)}</dd></div>
    <div><dt>학습 방식</dt><dd>전체 파라미터 · CPU FP32</dd></div>
    <div><dt>실행 범위</dt><dd>{project.executionScope === 'pc_validation' ? 'PC 검증 · NAS 미측정' : project.executionScope === 'nas' ? 'NAS 대상 · 실측 별도' : '미확인'}</dd></div>
    <div><dt>현재 문맥</dt><dd>{number(project.trainedContextTokens)} tokens · 구조 한도 {number(config.max_position_embeddings)}</dd></div>
    <div><dt>최대 RSS</dt><dd>{metrics.peakRssBytes === null ? '미측정' : `${number(metrics.peakRssBytes / 1024 ** 2, 1)} MiB`}</dd></div>
    <div><dt>실제 학습 시간</dt><dd>{metrics.activeSeconds === null ? '미측정' : `${number(metrics.activeSeconds / 60, 1)}분`}</dd></div>
  </dl>;
}

function CalibrationScope({ profile }) {
  if (!profile) return null;
  return <p className="foundation-policy-hint">{profile.executionScope === 'pc_validation' ? 'PC 검증 · NAS 미측정'
    : profile.executionScope === 'nas' ? 'NAS 대상 · 환경 증거 별도' : '실행 범위 미확인'}
    {' · '}{profile.measurementOnly === true ? '측정 전용 · 운영 학습 아님' : '측정 용도 미확인'}</p>;
}

function AcceptanceSummary({ value, failed, loading, refresh }) {
  const checks = collection(value?.checks), confirmed = value?.schemaVersion === 1 && !failed && checks.length === 23
    && new Set(checks.map(item => item.number)).size === 23 && checks.every(item => Number.isInteger(item.number) && item.number >= 1 && item.number <= 23 && ['passed', 'blocked', 'deferred'].includes(item.state));
  const ready = confirmed && value.readyForRequestedPreNasAcceptance === true && value.softwarePassed === true && value.containerPassed === true && checks.every(item => item.state !== 'blocked');
  const count = ids => ids.filter(id => checks.find(item => item.number === id)?.state === 'passed').length;
  const passed = checks.filter(item => item.state === 'passed').length;
  const deferred = checks.filter(item => item.state === 'deferred').length;
  const cards = [
    ['자료', confirmed ? `${count([2, 3])}/2` : '미확인', confirmed && count([2, 3]) === 2],
    ['코드', confirmed ? value.softwarePassed === true ? '통과' : '대기' : '미확인', confirmed && value.softwarePassed === true],
    ['모델', confirmed ? `${count([6, 7, 9, 10])}/4` : '미확인', confirmed && count([6, 7, 9, 10]) === 4],
    ['컨테이너', confirmed ? value.containerPassed === true ? '통과' : '대기' : '미확인', confirmed && value.containerPassed === true],
  ];
  return <aside className="foundation-acceptance" aria-label="실제 인수 조건">
    <div className="foundation-source-heading"><h3>인수 조건</h3><div className="foundation-acceptance-heading"><span className={`status-pill ${ready ? 'ready' : 'pending'}`}>{failed ? '확인 불가' : loading ? '확인 중' : ready ? '준비' : '대기'}</span><button type="button" className="icon-button" onClick={refresh} disabled={loading} aria-label="인수 조건 새로고침"><RefreshCw size={14} /></button></div></div>
    <div className="foundation-acceptance-cards">{cards.map(([label, state, ready]) => <div key={label}><span>{label}</span><strong className={ready ? 'confirmed' : ''}>{state}</strong></div>)}</div>
    {failed ? <p className="foundation-acceptance-note" role="status">상태 확인 불가</p> : <details><summary>23개 조건 <span>{confirmed ? `통과 ${passed} · 보류 ${deferred}` : '미확인'}</span></summary><ul>{Array.from({ length: 23 }, (_, index) => {
      const check = checks.find(item => item.number === index + 1), state = confirmed ? check?.state : null;
      return <li key={index + 1} title={collection(check?.reasonCodes).map(reasonLabel).join(' · ')}><span><i>{String(index + 1).padStart(2, '0')}</i>{acceptanceNames[index]}</span><strong className={state === 'passed' ? 'confirmed' : ''}>{state === 'passed' ? '통과' : state === 'deferred' ? '보류' : state === 'blocked' ? '대기' : '미확인'}</strong></li>;
    })}</ul></details>}
    {confirmed && <p className="foundation-acceptance-note">{value.executionScope === 'pc_validation' ? 'PC 후보' : '운영 범위'} · NAS 반영 별도</p>}
  </aside>;
}

export default function FoundationPanel({ onNavigate }) {
  const [projects, setProjects] = useState([]), [projectId, setProjectId] = useState(''), [project, setProject] = useState(null);
  const [sources, setSources] = useState([]), [selectedSources, setSelectedSources] = useState([]);
  const [run, setRun] = useState(null), [mode, setMode] = useState('always_on_idle');
  const [loading, setLoading] = useState(true), [busy, setBusy] = useState(false), [error, setError] = useState(''), [notice, setNotice] = useState('');
  const [connected, setConnected] = useState(false), [stale, setStale] = useState(false), [observedAt, setObservedAt] = useState(null);
  const [advanced, setAdvanced] = useState(false), [name, setName] = useState('SEED-S0');
  const [sourceError, setSourceError] = useState('');
  const [sourceLoading, setSourceLoading] = useState(true), [showSources, setShowSources] = useState(false);
  const sourceSection = useRef(null);
  const [policyVersion, setPolicyVersion] = useState(null);
  const [corpusKind, setCorpusKind] = useState('raw'), [snapshots, setSnapshots] = useState([]), [snapshotId, setSnapshotId] = useState('');
  const [profiles, setProfiles] = useState([]);
  const [calibrationStage, setCalibrationStage] = useState('pretrain');
  const [acceptance, setAcceptance] = useState(null), [acceptanceError, setAcceptanceError] = useState(false), [acceptanceLoading, setAcceptanceLoading] = useState(false);
  const inFlight = useRef(false), mounted = useRef(true), lastProject = useRef(null), selectedProject = useRef('');
  selectedProject.current = projectId;
  const refreshAcceptance = useCallback(async signal => {
    const selected = projectId;
    if (!selected) { setAcceptance(null); return; }
    setAcceptanceLoading(true);
    try {
      const value = await foundationRequest(`projects/${encodeURIComponent(selected)}/acceptance`, { signal });
      if (mounted.current && selectedProject.current === selected) { setAcceptance(value); setAcceptanceError(false); }
    } catch (cause) {
      if (cause.name !== 'AbortError' && mounted.current && selectedProject.current === selected) { setAcceptance(null); setAcceptanceError(true); }
    } finally { if (mounted.current && selectedProject.current === selected) setAcceptanceLoading(false); }
  }, [projectId]);
  useEffect(() => {
    const controller = new AbortController(); let disposed = false, timer;
    setAcceptance(null); setAcceptanceError(false);
    const poll = async () => { await refreshAcceptance(controller.signal); if (!disposed) timer = setTimeout(poll, 30000); };
    if (projectId) poll();
    return () => { disposed = true; controller.abort(); clearTimeout(timer); };
  }, [projectId, refreshAcceptance]);
  const load = useCallback(async () => {
    if (inFlight.current) return;
    inFlight.current = true;
    const requestedProject = projectId;
    try {
      const result = await foundationRequest('projects');
      if (!mounted.current || selectedProject.current !== requestedProject) return;
      const listed = collection(result).filter(item => item.state !== 'revoked' && !item.revocationPending);
      setProjects(listed); setConnected(true); setStale(false); setError('');
      try { const stored = await ragRequest('/training-snapshots'); if (mounted.current) setSnapshots(collection(stored)); } catch {}
      try { const measured = await foundationRequest('profiles'); if (mounted.current) setProfiles(collection(measured)); } catch {}
      let preferred = projectId;
      if (!preferred && listed.length) {
        try { preferred = (await foundationRequest('auto-training')).projectId; } catch {}
      }
      const selected = listed.find(item => item.id === preferred) ?? listed[0];
      if (selected) {
        if (selected.id !== projectId) setProjectId(selected.id);
        const detail = await foundationRequest(`projects/${encodeURIComponent(selected.id)}`);
        if (!mounted.current || (selectedProject.current !== requestedProject && selectedProject.current !== selected.id)) return;
        setProject(detail);
        if (lastProject.current !== detail.id) {
          setSelectedSources(detail.policy?.allowedSourceIds ?? []); setMode(policyMode(detail.policy?.automation)); lastProject.current = detail.id;
          setPolicyVersion(detail.policy?.version ?? null);
        }
        const latestRun = entity(detail, 'runs', detail.latestRunId);
        if (latestRun?.id) {
          try { const latest = await foundationRequest(`runs/${encodeURIComponent(latestRun.id)}`); if (mounted.current && selectedProject.current === selected.id) setRun(latest); }
          catch (cause) { if (mounted.current) { setRun(latestRun); setError(cause.message); setStale(true); } }
        } else setRun(null);
      } else { setProject(null); setRun(null); if (projectId) setProjectId(''); }
      setObservedAt(new Date().toISOString());
    } catch (cause) {
      if (mounted.current) { setError(cause.status === 404 ? '자체 모델 서비스 미연결' : cause.message); setStale(true); }
    } finally { inFlight.current = false; if (mounted.current) setLoading(false); }
  }, [projectId]);
  useEffect(() => {
    mounted.current = true; let disposed = false, timer;
    const poll = async () => { await load(); if (!disposed) timer = setTimeout(poll, document.hidden ? 30000 : 5000); };
    poll();
    return () => { disposed = true; clearTimeout(timer); mounted.current = false; };
  }, [load]);
  const refreshSources = useCallback(async signal => {
    try {
      const result = await ragRequest('/sources', { signal });
      if (mounted.current) { setSources(collection(result)); setSourceError(''); }
    } catch (cause) {
      if (cause.name !== 'AbortError' && mounted.current) setSourceError(cause.message);
    } finally { if (mounted.current && !signal?.aborted) setSourceLoading(false); }
  }, []);
  useEffect(() => {
    const controller = new AbortController(); let timer;
    const poll = async () => { await refreshSources(controller.signal); if (!controller.signal.aborted) timer = setTimeout(poll, 5000); };
    poll();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [refreshSources]);
  useEffect(() => {
    if (showSources) {
      sourceSection.current?.scrollIntoView({ behavior: 'smooth', block: 'center' });
      sourceSection.current?.focus({ preventScroll: true });
    }
  }, [showSources]);
  useEffect(() => {
    if (!notice) return;
    const timer = setTimeout(() => setNotice(''), 6000);
    return () => clearTimeout(timer);
  }, [notice]);

  function openSources() {
    setShowSources(true);
    sourceSection.current?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    sourceSection.current?.focus({ preventScroll: true });
  }
  async function approveSource(source, body) {
    if (busy) return;
    setBusy(true); setError('');
    try {
      await ragRequest(`/sources/${encodeURIComponent(source.id)}`, { method: 'PATCH', body: { ...body, trainingPolicy: 'allowed' } });
      setSources(previous => previous.map(item => item.id === source.id ? { ...item, ...body, trainingPolicy: 'allowed' } : item));
      setNotice('학습 이용을 허용했습니다. 사용할 자료를 체크해 주세요.');
    } finally { if (mounted.current) setBusy(false); }
  }

  async function mutate(path, body = {}, options = {}) {
    if (busy) return null;
    setBusy(true); setError(''); setNotice('');
    try {
      const result = await foundationRequest(path, { method: 'POST', body, ...options });
      setNotice('요청을 접수했습니다. 실행 상태는 아래에 표시됩니다.');
      await load(); return result;
    } catch (cause) { setError(cause.message); return null; }
    finally { if (mounted.current) setBusy(false); }
  }
  async function createProject(event) {
    event.preventDefault();
    const result = await mutate('projects', { name: name.trim(), origin: 'random_init', architecturePreset: 'seed-s0-11m-v1',
      pretrainedWeightArtifactIds: [], policyId: 'foundation-local-v1', automation: mode });
    if (result?.id) { lastProject.current = null; setProjectId(result.id); setNotice('프로젝트를 만들었습니다. 이제 학습할 자료를 준비해 주세요.'); }
  }
  async function savePolicy(enabled = project?.policy?.enabled ?? false) {
    if (!project) return;
    const version = policyVersion;
    if (!Number.isSafeInteger(version)) { setError('정책 버전을 확인할 수 없습니다. 새로고침하세요.'); return; }
    const result = await mutate(`projects/${project.id}/policy`, { enabled, automation: mode, allowedSourceIds: selectedSources,
      sliceSeconds: 900, dailyActiveSeconds: 14400 }, { method: 'PATCH', headers: { 'If-Match': String(version) } });
    if (result) {
      setPolicyVersion(result.policy?.version ?? result.version ?? version + 1);
      setNotice(enabled && mode !== 'manual' ? '자동 학습 설정을 저장했습니다. 실행 조건에 따라 순서대로 진행합니다.' : mode === 'manual' ? '수동 설정을 저장했습니다. 단계별 제어에서 실행해 주세요.' : '설정을 저장하고 자동 학습을 껐습니다.');
    }
    return result;
  }
  const rawCorpus = entity({ corpora: collection(project?.corpora).filter(item => item.purpose !== 'sft' && item.corpusKind !== 'sft') }, 'corpora', project?.activeCorpusId);
  const approvedCorpus = entity({ corpora: collection(project?.corpora).filter(item => item.purpose === 'sft' || item.corpusKind === 'sft') }, 'corpora', project?.activeSftCorpusId);
  const corpus = corpusKind === 'sft' ? approvedCorpus : rawCorpus;
  const tokenizer = entity(project, 'tokenizers', project?.tokenizerId);
  const genesis = entity(project, 'initializations', project?.genesisId);
  const checkpointId = run?.lastCheckpointId ?? project?.latestCheckpointId ?? genesis?.id ?? project?.genesisId;
  const metrics = foundationProgress(run);
  const runState = run?.state;
  const running = ['running', 'training', 'preparing', 'queued', 'pretraining', 'pause_requested'].includes(runState);
  const phase = project?.stage ?? 'waiting_corpus';
  const reasons = [...new Set([...(project?.reasonCodes ?? []), ...(project?.readiness?.reasonCodes ?? []), ...(run?.reasonCodes ?? [])])];
  const usable = phase === 'usable' && (project?.capabilityFlags?.generation === true || project?.capabilityFlags?.canServe === true);
  const disabled = busy || stale || !connected;
  const startBlocked = disabled || sourceLoading || Boolean(sourceError) || !selectedSources.length
    || selectedSources.some(id => !sources.some(source => source.id === id && source.trainingPolicy === 'allowed' && source.kind !== 'local_directory'));
  function autoStarted(result) {
    lastProject.current = null;
    if (result.projectId && result.projectId !== projectId) setProjectId(result.projectId);
    else void load();
    void refreshSources();
  }
  const policyChanged = JSON.stringify([...(project?.policy?.allowedSourceIds ?? [])].sort()) !== JSON.stringify([...selectedSources].sort())
    || policyMode(project?.policy?.automation) !== mode;
  const createCorpus = () => mutate(`projects/${project.id}/corpora`, { allowedSourceIds: selectedSources,
    ...(corpusKind === 'sft' ? { kind: 'sft', snapshotId } : {}) });
  const createTokenizer = () => mutate(`projects/${project.id}/tokenizers`, { corpusSnapshotId: corpus.id });
  const initialize = () => mutate(`projects/${project.id}/initializations`, { tokenizerArtifactId: tokenizer.id, architecturePreset: 'seed-s0-11m-v1', seed: 42 });
  const recipe = stage => ({ sequenceLength: 128, microBatchSize: 1, gradientAccumulation: 8,
    learningRate: stage === 'sft' ? 0.00005 : 0.0003, maxTokenExposures: 1000000, maxEpochs: stage === 'sft' ? 2 : 3, sliceSeconds: 900 });
  const train = (stage = 'pretrain') => mutate(`projects/${project.id}/${stage === 'pretrain' ? 'pretraining' : stage}-runs`, { corpusSnapshotId: corpus?.id, tokenizerArtifactId: tokenizer?.id,
    parentCheckpointId: checkpointId, trainingMode: 'full_parameters', objective: 'causal_lm', config: recipe(stage) });
  const evaluate = () => mutate('evaluations', { projectId: project.id, artifactId: checkpointId, checkpointId, kind: 'qualification', corpusId: rawCorpus?.id });
  const qualification = [...collection(project?.evaluations)].reverse().find(item => item.kind === 'qualification' && item.gate?.passed === true && item.artifactId === checkpointId && !item.fixture);
  const scope = project?.capabilityFlags?.approvedModes ?? [];
  const sftCorpus = corpus?.purpose === 'sft' || corpus?.corpusKind === 'sft' || (corpus?.sftExampleCount ?? 0) > 0;
  const projectProfiles = profiles.filter(item => item.projectId === project?.id && !item.testOnly);
  const measuredProfile = [...projectProfiles].sort((a, b) => String(b.measuredAt ?? b.createdAt ?? '').localeCompare(String(a.measuredAt ?? a.createdAt ?? '')))[0];
  const matchesRecipe = stage => projectProfiles.some(item => item.stage === stage && item.passed === true && item.state === 'passed' && item.measuredSteps >= 5
    && item.tokenizerId === tokenizer?.id && Object.entries(recipe(stage)).every(([key, value]) => item.config?.[key] === value));
  const selectedCalibrationStage = sftCorpus ? 'sft' : calibrationStage;
  const calibrate = () => mutate(`projects/${project.id}/calibrate`, { corpusSnapshotId: corpus?.id, tokenizerArtifactId: tokenizer?.id,
    parentCheckpointId: checkpointId, stage: selectedCalibrationStage, config: recipe(selectedCalibrationStage),
    measurement: { measurementSteps: 20, warmupSteps: 3, maxSeconds: 300 } });

  return <section className="card foundation-panel" aria-label="자체 기초 모델">
    <div className="foundation-header"><div className="foundation-brand"><span><Sprout size={22} strokeWidth={1.6} /></span><div><div className="foundation-eyebrow">SELF-MADE FOUNDATION</div><h2>나만의 기초 모델</h2></div></div><div className="foundation-header-actions"><span className={`status-pill ${usable ? 'ready' : 'pending'}`}><span className="dot" />{stale ? '갱신 중단' : loading ? '연결 확인' : !connected ? '미연결' : project ? stageLabel(phase) : '모델 생성 전'}</span><button className="icon-button" type="button" onClick={() => { lastProject.current = null; load(); }} disabled={loading || busy} aria-label="자체 모델 상태 새로고침"><RefreshCw size={16} /></button></div></div>
    <div className="foundation-intro"><span>{project?.origin === 'architecture_growth' ? '자체 모델 · 구조 변환' : '외부 초기값 없이 · 데이터에서 시작'}</span><span>{project ? `${number(project.parameterCount / 1000000, 2)}M` : 'S0'} <i>·</i> CPU <i>·</i> 최대 문맥 {number(project?.config?.max_position_embeddings ?? 512)}</span></div>
    {error && <div className="foundation-feedback error" role="alert">{error}</div>}
    {notice && <div className="foundation-feedback" role="status"><Check size={14} />{notice}</div>}
    <AutoTrainingPanel projectId={projectId} onStarted={autoStarted} onNavigate={onNavigate} onBusyChange={setBusy} disabled={busy} />
    {!project ? <details className="foundation-manual-project"><summary>직접 프로젝트 만들기 · 고급 설정</summary><div className="foundation-empty"><span><Sprout size={35} strokeWidth={1.3} /></span><h3>첫 번째 시작</h3><p>토큰 학습 · 무작위 초기화 · 전체 가중치 학습</p><form onSubmit={createProject}><label className="field"><span>모델 이름</span><input aria-label="자체 모델 이름" maxLength={50} required value={name} onChange={event => setName(event.target.value)} /></label><button className="button primary" disabled={disabled || !name.trim()} type="submit">{busy ? <Loader2 size={15} className="spin" /> : <Plus size={15} />}프로젝트 만들기</button></form><small>초기화는 사용 가능 판정과 다릅니다.</small></div></details>
      : <>
        <div className="foundation-project-heading"><label><span className="sr-only">자체 모델 프로젝트</span><select value={projectId} onChange={event => { setProject(null); setRun(null); setProjectId(event.target.value); }} disabled={busy}>{projects.map(item => <option key={item.id} value={item.id}>{item.name ?? item.id}</option>)}</select></label><span>{usable ? `${scope.length ? scope.join(' · ') : '승인 범위'} · 사용 가능` : '일반 생성 잠금 · 독립 검증 필요'}</span></div>
        {run && <div className="foundation-metrics">{[
          ['학습 토큰', number(metrics.completed), 'target tokens'], ['완료 스텝', number(metrics.steps), 'optimizer steps'],
          ['Loss', number(metrics.loss, 5), '실제 관측'], ['처리량', number(metrics.tokensPerSecond, 2), 'tokens / sec'],
        ].map(([label, value, unit]) => <div key={label}><span>{label}</span><strong>{value}</strong><small>{unit}</small></div>)}</div>}
        {run && <div className="foundation-run-progress"><div><span>{stageLabel(run.phase ?? run.state)}</span><strong>{metrics.percent === null ? '분모 대기' : `${metrics.percent}%`}</strong></div><div className="progress-track" role="progressbar" aria-label="실제 학습 노출 토큰 진행률" aria-valuemin={0} aria-valuemax={100} aria-valuenow={metrics.percent ?? undefined}><i style={{ width: `${metrics.percent ?? 0}%` }} /></div><small>{number(metrics.completed)} / {number(metrics.total)} target tokens · 사용 가능 여부는 별도 검증</small></div>}
        {reasons.length > 0 && <div className="foundation-blockers">{reasons.map(reason => <span key={reason} title={reason}>{reasonLabel(reason)}</span>)}</div>}
        <div className="foundation-controls">{run && <>{runState === 'paused' ? <button className="button secondary" type="button" disabled={disabled || run.capabilities?.resume !== true} onClick={() => mutate(`runs/${run.id}/resume`)}><Play size={14} />재개</button> : running && <button className="button secondary" type="button" disabled={disabled || run.capabilities?.pause !== true} onClick={() => mutate(`runs/${run.id}/pause`)}><Pause size={14} />일시정지</button>}{(running || runState === 'paused') && <button className="button secondary" type="button" disabled={disabled || run.capabilities?.cancel !== true} onClick={() => mutate(`runs/${run.id}/cancel`)}>학습 중지</button>}</>}</div>
        <details className="foundation-details" open={advanced} onToggle={event => setAdvanced(event.currentTarget.open)}><summary>단계별 수동 제어 · 상세 정보 <span>고급</span></summary><div className="foundation-detail-content">
    <ol className="foundation-stages">{stages.map(([label, Icon], index) => <li key={label} className={project && index === stageIndex(phase) ? 'current' : ''}><span><Icon size={16} /></span><strong>{label}</strong>{index < stages.length - 1 && <ArrowRight size={12} />}</li>)}</ol>
        <div className="foundation-policy-row"><span><Clock3 size={15} />학습 방식</span><div className="foundation-mode-picker" role="group" aria-label="자가학습 방식">{[['always_on_idle', '유휴 시간 자동'], ['night_window', '야간 자동'], ['manual', '수동']].map(([value, label]) => <button type="button" key={value} aria-pressed={mode === value} className={mode === value ? 'selected' : ''} disabled={disabled} onClick={() => setMode(value)}>{label}</button>)}</div><span className="foundation-policy-hint">{mode === 'night_window' ? '02:00–06:00' : mode === 'manual' ? '요청한 학습만' : '작업 우선 · 여유 자원'} · 하루 4시간</span></div><button className="button secondary" type="button" onClick={() => showSources ? setShowSources(false) : openSources()} aria-expanded={showSources}><Database size={14} />학습 자료 직접 관리</button>        <section className="foundation-source-section" ref={sourceSection} tabIndex={-1} aria-label="학습 자료 설정" hidden={!showSources}>
          <div className="foundation-source-heading"><h3>학습할 자료 선택</h3><button className="text-button" type="button" disabled={busy || sourceLoading} onClick={() => refreshSources()}><RefreshCw size={13} />목록 새로고침</button></div>
          {sourceError && <p className="foundation-feedback error" role="alert">{sourceError}</p>}
          {sourceLoading ? <p className="foundation-setup-copy">자료 목록 확인 중…</p> : <FoundationSources sources={sources} selectedSources={selectedSources} onSelection={setSelectedSources} disabled={disabled || Boolean(sourceError)} onApprove={approveSource} onAddData={() => onNavigate('data')} />}
        </section>
<p className="foundation-setup-copy">수동 실행 순서: 자료 검사 → 토큰 학습 → 초기화 → 실측 → 기초학습. 기본 화면의 학습하기는 이 과정을 자동으로 진행합니다.</p><button className="button secondary" type="button" disabled={startBlocked || !policyChanged} onClick={() => savePolicy()}><Check size={14} />출처 · 방식 저장</button><div className="foundation-corpus-type"><label className="field"><span>학습 자료</span><select value={corpusKind} onChange={event => setCorpusKind(event.target.value)} disabled={disabled}><option value="raw">원문 · 기초학습 / CPT</option><option value="sft">승인 응답 · SFT</option></select></label>{corpusKind === 'sft' && <label className="field"><span>승인 스냅샷</span><select value={snapshotId} onChange={event => setSnapshotId(event.target.value)} disabled={disabled}><option value="">스냅샷 선택</option>{snapshots.filter(item => item.status === 'ready').map(item => <option key={item.id} value={item.id}>{item.name ?? item.id} · {item.exampleCount ?? '—'}개</option>)}</select></label>}</div><div className="foundation-stage-actions">
          <button type="button" disabled={disabled || !selectedSources.length || policyChanged || (corpusKind === 'sft' && !snapshotId)} onClick={createCorpus}><Database size={15} />자료 검사</button>
          <button type="button" disabled={disabled || !valid(corpus) || sftCorpus || Boolean(project.genesisId)} onClick={createTokenizer}><Layers3 size={15} />토큰 학습</button>
          <button type="button" disabled={disabled || !valid(tokenizer) || Boolean(project.genesisId)} onClick={initialize}><Sprout size={15} />초기화</button>
          <label className="foundation-calibration-stage"><span className="sr-only">실측 단계</span><select aria-label="실측 단계" value={selectedCalibrationStage} onChange={event => setCalibrationStage(event.target.value)} disabled={disabled || running || sftCorpus}><option value="pretrain">Pretrain</option><option value="cpt">CPT</option>{sftCorpus && <option value="sft">SFT</option>}</select></label><button type="button" disabled={disabled || !checkpointId || !valid(corpus) || !valid(tokenizer) || running} onClick={calibrate}><Cpu size={15} />실측</button>
          <button type="button" disabled={disabled || !checkpointId || !valid(corpus) || !valid(tokenizer) || running || sftCorpus || !matchesRecipe('pretrain')} onClick={() => train('pretrain')}><Activity size={15} />기초학습</button>
          <button type="button" disabled={disabled || !checkpointId || !valid(corpus) || running || !metrics.steps || sftCorpus || !matchesRecipe('cpt')} onClick={() => train('cpt')}><Activity size={15} />지속학습</button>
          <button type="button" disabled={disabled || !checkpointId || !valid(corpus) || running || !sftCorpus || !matchesRecipe('sft')} onClick={() => train('sft')}><Activity size={15} />지시학습</button>
          <button type="button" disabled={disabled || !checkpointId || !metrics.steps} onClick={evaluate}><FlaskConical size={15} />기초 검증</button>
          <button type="button" disabled={disabled || !qualification} onClick={() => mutate('graduations', { projectId: project.id, checkpointId, evaluationId: qualification.id })}><Check size={15} />기초 자격</button>
        </div><div className="foundation-calibration"><div className="foundation-source-heading"><h3>실측 프로파일</h3><span className={`status-pill ${measuredProfile?.passed === true ? 'ready' : 'pending'}`}>{measuredProfile ? measuredProfile.passed ? '실측 통과' : '측정 불충분' : '실측 대기'}</span></div><p>동일 학습 설정의 실측을 통과한 후 학습합니다. 시험 가중치는 보존하지 않습니다.</p><CalibrationScope profile={measuredProfile} /><dl className="foundation-facts"><div><dt>처리량</dt><dd>{number(measuredProfile?.effectiveTokensPerSecond, 2)} tokens/s</dd></div><div><dt>최대 RSS</dt><dd>{actualNumber(measuredProfile?.peakRssBytes) === null ? '미측정' : `${number(measuredProfile.peakRssBytes / 1024 ** 2, 1)} MiB`}</dd></div><div><dt>저장 시간</dt><dd>{number(measuredProfile?.checkpointSeconds, 2)}초</dd></div><div><dt>관측 스텝</dt><dd>{number(measuredProfile?.measuredSteps)} · 워밍업 제외</dd></div><div><dt>Sequence / batch</dt><dd>{number(measuredProfile?.config?.sequenceLength)} / {number(measuredProfile?.config?.microBatchSize)}</dd></div><div><dt>학습 실적</dt><dd>0 · 실측 전용</dd></div></dl>{measuredProfile?.reasonCodes?.length > 0 && <div className="foundation-blockers">{measuredProfile.reasonCodes.map(reason => <span key={reason}>{reasonLabel(reason)}</span>)}</div>}</div><ProjectFacts project={project} metrics={metrics} /></div></details>
        <details className="foundation-verification"><summary>모델 사용 전 검증 <span>23개 조건 · 상세 보기</span></summary><p className="foundation-setup-copy">학습 결과를 실제 서비스에 사용하기 위한 검증 현황입니다. 지금 해야 할 학습 준비는 위에서 확인하세요.</p><AcceptanceSummary value={acceptance} failed={acceptanceError} loading={acceptanceLoading} refresh={() => refreshAcceptance()} /></details>
        {observedAt && <div className="foundation-observed"><Cpu size={13} /><span>서버 실제 상태</span><time>{new Date(observedAt).toLocaleTimeString('ko-KR', { timeZone: 'Asia/Seoul' })}</time></div>}
      </>}
  </section>;
}
