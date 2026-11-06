import React from 'react';
import { Activity, Check, CircleHelp, Code2, GitBranch, HardDrive, Loader2, Pause, Play, RefreshCw, SlidersHorizontal, Sprout, Square } from 'lucide-react';
import useTrainingStatus from './useTrainingStatus';
import './training.css';
import NasModelCatalog from './NasModelCatalog';

const stateLabels = { preparing: '학습 준비 중', running: '학습 중', paused: '일시정지', validating: '검증 중', completed: '학습 완료', cancelled: '학습 중단', failed: '학습 실패' };
const activeStates = ['preparing', 'running', 'paused', 'validating'];
const count = value => value.toLocaleString('ko-KR');
const timestamp = value => new Date(value).toLocaleString('ko-KR', { timeZone: 'Asia/Seoul' });

export default function TrainingView({ settings, config, onConfig, nasRam, onRamChange, onModelSelect }) {
  const { status, connection, lastReceivedAt, error, pending, refresh, command } = useTrainingStatus(globalThis.location?.origin ?? '');
  const { job, runs } = status;
  const connected = connection === 'connected' && (!status.connection || ['connected', 'ready'].includes(status.connection));
  const active = job && activeStates.includes(job.state);
  const progress = job?.progress ?? 0;
  const progressLabel = job && job.progress === null ? '계산 대기' : `${Math.floor(progress * 10) / 10}%`;
  const blocked = !connected || pending;
  const connectionLabel = connected ? '실시간 연결' : connection === 'stale' ? '갱신 중단' : connection === 'connecting' ? '연결 확인 중' : '서버 미연결';
  return <>
    <div className="page-heading" style={{ justifyContent: 'flex-end' }}><button className="button secondary" onClick={onConfig}><SlidersHorizontal size={16} />학습 설정</button></div>
    <div className="info-banner"><CircleHelp size={17} /><span>{connected ? '진행률 = 완료 스텝 ÷ 전체 스텝 · 서버 응답 기준' : connection === 'stale' ? '연결 끊김 · 마지막 수신값 유지 · 진행률 갱신 중단' : '학습 서버 연결 전 · 실행 및 진행률 측정 불가'}</span></div>
    <div className="training-layout">
      <section className="card training-main">
        <div className="section-heading"><h2>학습 현황</h2><span className={`status-pill ${connected ? 'ready' : 'pending'}`}><span className="dot" />{connectionLabel}</span></div>
        <div className="training-orb"><Sprout size={55} strokeWidth={1.2} /></div>
        <h2>{connection === 'stale' ? '연결 확인 필요' : job ? stateLabels[job.state] : connected ? '학습 대기' : '학습 서버 미연결'}</h2>
        <p className="muted">{job ? `작업 ${job.id}` : connected ? `학습 준비 ${count(status.readySamples)}개 샘플` : '학습 설정에서 서버 주소 등록'}</p>
        <div className="training-progress"><div><span>{job ? '학습 스텝' : '진행률'}</span><strong>{progressLabel}</strong></div><div className="progress-track" role="progressbar" aria-label="실제 학습 진행률" aria-valuemin={0} aria-valuemax={100} aria-valuenow={job?.progress === null ? undefined : progress} aria-valuetext={connection === 'stale' ? `${progressLabel} · 마지막 수신값, 갱신 중단` : progressLabel}><i style={{ width: `${progress}%` }} /></div><div className="training-step-count"><span>{job ? `${count(job.completedSteps)} / ${count(job.totalSteps)} 스텝` : '0 / 0 스텝'}</span>{job?.totalSteps === 0 && <span>전체 스텝 미수신</span>}</div></div>
        {job && <div className="training-live-metrics"><span>에포크 <strong>{job.epoch ?? '—'}</strong></span><span>Loss <strong>{job.loss === null ? '—' : job.loss.toFixed(4)}</strong></span></div>}
        <div className="training-buttons">
          {active ? <>
            {job.state === 'paused' ? <button className="button primary" disabled={blocked || !job.capabilities.resume} onClick={() => command('resume')}><Play size={16} />계속하기</button> : <button className="button secondary" disabled={blocked || job.state !== 'running' || !job.capabilities.pause} onClick={() => command('pause')}><Pause size={16} />일시정지</button>}
            <button className="button secondary" disabled={blocked || !job.capabilities.cancel} onClick={() => command('cancel')}><Square size={14} />중단</button>
          </> : <button className="button primary" disabled={blocked || !status.canStart || status.readySamples === 0} onClick={() => command('start', { model: settings.model, branchId: config.branchId, epochs: Number(config.epochs), batchSize: Number(config.batch), learningRate: Number(config.rate) })}>{pending ? <Loader2 className="spin" size={16} /> : <Play size={16} />}학습 시작</button>}
          {!connected && config.endpoint && <button className="button secondary" disabled={connection === 'connecting' || pending} onClick={refresh}><RefreshCw size={14} />다시 연결</button>}
          {!config.endpoint && <button className="button secondary" onClick={onConfig}><HardDrive size={15} />서버 설정</button>}
        </div>
        {pending && <p className="training-feedback" role="status">요청 중 · 서버 확인 대기</p>}
        {!active && connected && !status.readySamples && <p className="training-feedback">학습 가능한 데이터 없음</p>}
        {lastReceivedAt && <p className="training-last-update">마지막 수신 {timestamp(lastReceivedAt)}{job?.updatedAt && <span>서버 갱신 {timestamp(job.updatedAt)}</span>}</p>}
        {error && <p className="training-error" role="alert">{error}</p>}
      </section>
      <section className="card training-details"><h2>모델 · 학습 구성</h2><NasModelCatalog value={settings.model} onSelect={onModelSelect} ram={nasRam} onRamChange={onRamChange} disabled={Boolean(active) || pending} /><dl className="detail-list"><div><dt>에포크</dt><dd>{config.epochs}</dd></div><div><dt>배치 크기</dt><dd>{config.batch}</dd></div><div><dt>학습률</dt><dd>{config.rate}</dd></div><div><dt>프로세스 메모리 설정</dt><dd>{settings.memory} GB</dd></div><div><dt>학습 서버</dt><dd>{config.endpoint || '미설정'}</dd></div></dl><div className="subtle-note"><Activity size={16} /><span>후보 선택만 저장 · 모델 설치·학습·병합 미실행</span></div></section>
    </div>
    <section className="card history-card"><div className="section-heading"><h2>학습 기록</h2><span className="muted small">{runs.length}건</span></div>{runs.length ? <div className="run-list">{runs.map(run => <div className="run-row" key={run.id}><span className="activity-icon green-bg"><Check size={16} /></span><div><strong>{run.id}</strong><span>{timestamp(run.finishedAt)}</span></div><span className={`status-pill ${run.state === 'completed' ? 'ready' : 'pending'}`}>{stateLabels[run.state]}</span></div>)}</div> : <div className="empty-state"><span><GitBranch size={26} strokeWidth={1.5} /></span><strong>학습 기록 없음</strong><p>{connected ? '서버에서 확인한 기록만 표시' : '서버 연결 후 기록 확인'}</p></div>}</section>
  </>;
}
