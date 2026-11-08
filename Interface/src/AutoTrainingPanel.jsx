import React, { useCallback, useEffect, useId, useRef, useState } from 'react';
import { Check, CircleHelp, Cpu, Database, Loader2, Play, RefreshCw, Sprout, Square } from 'lucide-react';
import { foundationRequest, reasonLabel } from './foundation-api';
import { requestId } from './rag-api';
import { autoTrainingCollectionStatus, readAutoTrainingPlan, startAutoTraining } from './auto-training-api';
import './AutoTrainingPanel.css';

export default function AutoTrainingPanel({ projectId, onStarted, onNavigate, onBusyChange, disabled }) {
  const [plan, setPlan] = useState(null), [loading, setLoading] = useState(true), [pending, setPending] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [showDetails, setShowDetails] = useState(false);
  const generation = useRef(0), mounted = useRef(false), requestBusy = useRef(false), attempt = useRef(null);
  const errorKind = useRef(null);
  const descriptionId = useId();

  const refresh = useCallback(async signal => {
    const current = generation.current;
    try {
      const next = await readAutoTrainingPlan(projectId, { signal });
      if (mounted.current && generation.current === current && !requestBusy.current) {
        setPlan(next);
        if (errorKind.current === 'refresh' || next.enabled) setError('');
        if (next.enabled) { setNotice(''); attempt.current = null; }
      }
      return next;
    } catch (cause) {
      if (cause.name !== 'AbortError' && mounted.current && generation.current === current) { errorKind.current = 'refresh'; setError(cause.message); }
      return null;
    } finally { if (mounted.current && generation.current === current) setLoading(false); }
  }, [projectId]);
  useEffect(() => {
    mounted.current = true; generation.current += 1;
    const controller = new AbortController(); let timer;
    setPlan(null); setLoading(true); setError(''); setNotice('');
    const poll = async () => { await refresh(controller.signal); if (!controller.signal.aborted) timer = setTimeout(poll, document.hidden ? 30000 : 5000); };
    poll();
    return () => { mounted.current = false; controller.abort(); clearTimeout(timer); };
  }, [refresh]);
  async function start() {
    if (requestBusy.current || disabled) return;
    requestBusy.current = true; setPending(true); onBusyChange(true); setError(''); setNotice('');
    const current = generation.current;
    const body = { autoApproveSources: true };
    const signature = JSON.stringify([projectId, body]);
    if (attempt.current?.signature !== signature) attempt.current = { signature, key: requestId() };
    try {
      const result = await startAutoTraining(projectId, body, attempt.current.key);
      if (!mounted.current || generation.current !== current) return;
      attempt.current = null; setPlan(result);
      await onStarted(result);
    } catch (cause) {
      if (mounted.current && generation.current === current) {
        if (cause.code === 'REQUEST_IN_PROGRESS' || cause.code === 'REQUEST_TIMEOUT' || ['TimeoutError', 'AbortError'].includes(cause.name)) {
          setNotice(cause.code === 'REQUEST_IN_PROGRESS' ? '서버가 같은 학습 요청을 처리하고 있습니다. 접수 상태를 자동으로 확인합니다.' : '서버 응답이 늦어 접수 상태를 자동으로 확인하고 있습니다. 다시 누르더라도 같은 요청으로 확인합니다.');
        } else { errorKind.current = 'start'; setError(cause.message); }
      }
    } finally {
      requestBusy.current = false;
      if (mounted.current) { setPending(false); onBusyChange(false); }
    }
    if (mounted.current && generation.current === current) await refresh();
  }
  async function stop() {
    if (requestBusy.current || disabled || !plan?.projectId) return;
    requestBusy.current = true; setPending(true); onBusyChange(true); setError(''); setNotice('');
    try {
      let latest = plan;
      if (!Number.isSafeInteger(latest.policyVersion)) latest = await readAutoTrainingPlan(plan.projectId);
      for (let retry = 0; retry < 2; retry += 1) {
        if (!Number.isSafeInteger(latest.policyVersion)) throw new Error('서버에서 중지에 필요한 설정 상태를 확인할 수 없습니다.');
        try {
          await foundationRequest(`projects/${encodeURIComponent(plan.projectId)}/policy`, { method: 'PATCH', body: { enabled: false }, headers: { 'If-Match': String(latest.policyVersion) } });
          break;
        } catch (cause) {
          if (cause.status !== 412 || retry > 0) throw cause;
          latest = await readAutoTrainingPlan(plan.projectId);
        }
      }
      const next = await readAutoTrainingPlan(plan.projectId);
      if (mounted.current) { setPlan(next); await onStarted(next); }
    } catch (cause) { if (mounted.current) { errorKind.current = 'stop'; setError(cause.message); } }
    finally { requestBusy.current = false; if (mounted.current) { setPending(false); onBusyChange(false); } }
  }
  async function primaryAction() {
    setError('');
    if (!plan || plan.action === 'retry' || plan.action === 'wait') { await refresh(); return; }
    await start();
  }
  const active = plan?.action === 'wait';
  const data = plan?.data;
  const dataReady = data && Number.isFinite(data.trainingBytes) && Number.isFinite(data.minTrainingBytes) && data.minTrainingBytes > 0;
  const collectionStatus = autoTrainingCollectionStatus(plan?.collection);
  const label = pending ? '자동으로 준비하는 중' : loading ? '준비 상태 확인 중' : !plan || plan.action === 'retry' ? '다시 확인하기' : active ? '진행 상태 새로고침' : '학습하기';
  return <section className="auto-training" aria-label="자동 학습" aria-busy={pending}>
    <div className="auto-training-heading"><span className="auto-training-mark"><Sprout size={25} /></span><div><span className="auto-training-eyebrow">버튼 하나로 학습 준비부터 시작까지</span><h3>{loading ? '학습 준비 상태를 확인하고 있어요' : plan?.title ?? '서버 연결을 확인해 주세요'}</h3></div></div>
    <p id={descriptionId} className="auto-training-description">{loading ? '등록된 자료와 실행 환경에 맞는 설정을 확인합니다.' : plan?.detail ?? '서버에서 준비 상태를 불러오지 못했습니다. 다시 확인해 주세요.'}</p>
    <div className="auto-training-benefits"><span><Database size={14} />자료 자동 승인·선택</span><span><Cpu size={14} />환경에 맞게 자동 설정</span><span><Check size={14} />준비 단계 자동 진행</span></div>
    <div className="auto-training-actions"><button className="button primary" type="button" disabled={pending || disabled || loading} onClick={primaryAction} aria-describedby={descriptionId}>{pending || loading ? <Loader2 size={17} className="spin" /> : active ? <RefreshCw size={17} /> : <Play size={17} />}{label}</button>{plan?.enabled && <button className="button secondary" type="button" disabled={pending || disabled} onClick={stop}><Square size={14} />자동 학습 중지</button>}<button className="auto-training-details-toggle" type="button" onClick={() => setShowDetails(previous => !previous)} aria-expanded={showDetails}><CircleHelp size={14} />자동 설정 내용</button></div>
    <p className="auto-training-footnote">학습하기를 누르면 등록 자료의 학습 이용 승인과 실행 설정을 자동으로 적용합니다.</p>
    {plan?.action === 'add_data' && <button className="text-button" type="button" onClick={() => onNavigate('data')}>내 자료 직접 추가하기</button>}
    {error && <p className="auto-training-error" role="alert">{error}</p>}
    {notice && <p className="auto-training-notice" role="status">{notice}</p>}
    {dataReady && <div className="auto-training-data" aria-label="학습 자료 확보 현황"><div><strong>학습용 원문</strong><span>{(data.trainingBytes / 1024 ** 2).toLocaleString('ko-KR', { maximumFractionDigits: 2 })} / {(data.minTrainingBytes / 1024 ** 2).toLocaleString('ko-KR', { maximumFractionDigits: 2 })} MiB</span></div><progress aria-label="학습용 원문 확보량" max={data.minTrainingBytes} value={Math.min(data.trainingBytes, data.minTrainingBytes)} /><p>{data.state === 'held' ? '자료가 기준을 충족하면 토큰 생성과 학습을 자동으로 이어갑니다.' : '검사를 통과한 학습용 원문 기준입니다.'}</p></div>}
    {collectionStatus && <p className="auto-training-notice" role="status">{collectionStatus}</p>}
    {plan?.steps.length > 0 && <ol className="auto-training-steps" aria-label="자동 학습 진행 단계">{plan.steps.map(step => <li key={step.id} className={`step-${step.state}`}><span>{['completed', 'ready', 'passed'].includes(step.state) ? <Check size={13} /> : ['running', 'preparing', 'active'].includes(step.state) ? <Loader2 size={13} className="spin" /> : <span className="auto-training-step-dot" />}</span><strong>{step.label}</strong><small>{step.detail ?? ({ completed: '완료', ready: '준비됨', passed: '완료', running: '진행 중', preparing: '준비 중', active: '진행 중', waiting: '대기', pending: '대기', blocked: '확인 필요' })[step.state] ?? '대기'}</small></li>)}</ol>}
    {showDetails && <div className="auto-training-details"><h4>자동으로 처리하는 항목</h4><ul><li>현재 프로젝트를 이어서 사용하고, 처음 시작하면 기초 모델을 만듭니다.</li><li>학습하기 요청에 따라 미승인 자료의 이용을 승인하고, 사용할 출처를 자동으로 선택합니다. 기존에 금지한 자료는 제외합니다.</li><li>CPU·메모리·학습 워커 상태를 확인하고 권장 설정을 적용합니다.</li><li>자료 검사부터 토큰 학습·초기화·실측·기초학습까지 순서대로 진행합니다.</li></ul>{plan && <dl><div><dt>학습 출처</dt><dd>{plan.readySourceIds.length}개 허용 · {plan.automaticSourceIds.filter(id => !plan.readySourceIds.includes(id)).length}개 자동 승인 대상</dd></div><div><dt>실행 환경</dt><dd>{plan.environment?.executionScope === 'nas' ? 'NAS' : plan.environment?.executionScope === 'pc_validation' ? 'PC' : '서버에서 확인'} · CPU</dd></div><div><dt>학습 워커</dt><dd>{plan.environment?.workerAvailable === true ? '연결됨' : '연결 대기'}</dd></div></dl>}{plan?.blockers.length > 0 && <p className="auto-training-diagnostics">현재 확인 사항: {plan.blockers.map(reasonLabel).join(' · ')}</p>}<p className="auto-training-footnote">자료 부족이나 서버 기능 제한이 있으면 이유를 표시하고 대기합니다. 모델의 서비스 사용은 학습 후 검증을 거칩니다.</p></div>}

  </section>;
}
