import React, { useEffect, useState } from 'react';
import { ragRequest } from '../../Interface/src/rag-api.js';
const states={waiting_data:'자료 대기',waiting_worker:'자원 대기',waiting_window:'예약 대기',training:'학습',building:'병합',evaluating:'검증',release_evaluating:'배포 검증',packaging:'변환',ready:'준비',held:'보류',completed:'완료',failed:'실패',bootstrap:'초기 확인',active:'사용 중',retired:'이전 버전'};

async function request(path,body,method=body===undefined?'GET':'POST') {
  const headers={};
  if(method==='PATCH')headers['If-Match']=String((await ragRequest('/llm/'+path)).version);
  if(path.endsWith('/activate')||path==='deployments/champion/rollback')headers['If-Match']=String((await ragRequest('/llm/deployments/champion')).version);
  return ragRequest('/llm/'+path,body===undefined?{method}:{method,body,headers});
}

export default function LlmPanel() {
  const [data,setData]=useState(null),[error,setError]=useState(''),[busy,setBusy]=useState(false);
  const [directory,setDirectory]=useState(''),[license,setLicense]=useState('Apache-2.0'),[snapshot,setSnapshot]=useState(''),[sourceIds,setSourceIds]=useState('');
  const [evaluationModel,setEvaluationModel]=useState('');
  const [verifierProfile,setVerifierProfile]=useState('');
  async function refresh(){
    try {
      const [cap,models,families,branches,learning,cycles,releases,curation,evolution,progress]=await Promise.all(['capabilities','models','families','branches','learning/status','cycles','releases','curation','self-code','progress'].map(p=>request(p)));
      setData({cap,models:models.items,families:families.items,branches:branches.items,learning,cycles:cycles.items,releases:releases.items,curation:curation.items,evolution,progress:progress.items.filter(p=>!p.fixture&&p.baselineRole==='original')});setError('');
    }catch(e){setError(e.message);}
  }
  useEffect(()=>{refresh();const timer=setInterval(refresh,10000);return()=>clearInterval(timer);},[]);
  async function action(path,body,method){setBusy(true);try{const result=await request(path,body,method);window.dispatchEvent(new Event('seed-llm-settings'));await refresh();return result;}catch(e){setError(e.message);}finally{setBusy(false);}}
  return <details className="rag-tools-group"><summary>LLM · 학습 · 조합</summary>
    {error&&<p className="training-error" role="alert">{error}</p>}
    {!data?<p className="muted small">LLM 연결 대기 · 통합 서버로 실행</p>:<>
      <p className="muted small">{data.cap.training?'학습 엔진 연결':'학습 자원 대기'} · {data.models.length}개 모델 · {data.branches.length}개 전문 모델</p>
      <p className="muted small">NAS {Math.round(data.cap.resources.nasRamBytes/1024**3)} GB · LLM 상한 {(data.cap.resources.maxRssBytes/1024**3).toFixed(1)} GB · 시스템 여유 {(data.cap.resources.systemReserveBytes/1024**3).toFixed(1)} GB</p>
      <button type="button" className="button secondary" disabled={busy} onClick={()=>action('resource-policy',{mode:'adaptive',nasRamBytes:32*1024**3,systemReserveBytes:8*1024**3},'PATCH')}>32 GB 기준</button>
      <h3>자료 · 자기 코드</h3>
      <label className="field"><span>코드 검증기 ID</span><input value={verifierProfile} onChange={e=>setVerifierProfile(e.target.value)} placeholder="등록된 Python 검증기" /></label><button type="button" className="button secondary" disabled={busy||!verifierProfile} onClick={()=>action('self-code/setup',{profileId:verifierProfile})}>코드 검증 연결</button>
      <div className="rag-tool-buttons"><button type="button" className="button secondary" disabled={busy} onClick={()=>action('curation/run',{})}>자료 변환</button><button type="button" className="button secondary" disabled={busy||!evaluationModel} onClick={()=>action('self-code/run',{artifactId:evaluationModel})}>코드 개선</button><button type="button" className="button secondary" disabled={busy} onClick={()=>action('self-code/rollback',{})}>코드 복원</button></div>
      <p className="muted small">자료 {data.curation.filter(d=>d.state==='ready').length}/{data.curation.length} · 자기 코드 {states[data.evolution.status.state]??data.evolution.status.state??'검증기 대기'}</p>
      {data.curation.slice(-4).map(d=><p className="muted small" key={d.id}>{d.state==='ready'?'변환':d.state==='partial'?'분할 처리':d.state==='blocked'?'권한 대기':'검토'} · {d.chunkCursor??0}/{d.chunkCount??0} · {(d.reasonCodes??[]).join(' · ')}</p>)}
      <p className="muted small">원본 대비 {data.progress.length?data.progress.slice(-1).map(p=>`${(p.delta*100).toFixed(1)}%p · ${p.improvementConfirmed?'개선 확인':'개선 미확인'}`):'평가 대기'}</p>
      <div className="rag-tool-buttons"><button type="button" className="button secondary" disabled={busy} onClick={refresh}>새로고침</button><button type="button" className="button secondary" disabled={busy} onClick={()=>action('learning/run',{})}>학습 조건 확인</button><button type="button" className="button secondary" disabled={busy} onClick={()=>action('learning/pause',{})}>자동학습 중지</button><button type="button" className="button secondary" disabled={busy} onClick={()=>action('deployments/champion/rollback',{})}>이전 모델 복원</button></div>
      <p className="muted small">{data.learning.status?.state??'대기'} · {(data.cap.activationBlockers??[]).join(' · ')}</p>
      <label className="field"><span>원본 모델 경로</span><input value={directory} onChange={e=>setDirectory(e.target.value)} placeholder="서버의 허용된 imports 경로" /></label>
      <label className="field"><span>모델 라이선스</span><input value={license} onChange={e=>setLicense(e.target.value)} /></label>
      <button type="button" className="button secondary" disabled={busy||!directory} onClick={()=>action('models',{directory,licenseRefs:[license]})}>모델 등록</button>
      {data.families.map(f=><div key={f.id}><p className="muted small">{f.name??f.id}</p><div className="rag-tool-buttons">{[['A','code'],['B','review'],['C','test']].map(([label,specialization])=><button type="button" className="button secondary" key={label} disabled={busy||data.branches.some(b=>b.familyId===f.id&&b.label===label)} onClick={()=>action('branches',{familyId:f.id,label,specialization})}>{label} 전문 모델</button>)}</div></div>)}
      {data.branches.map(b=><div className="rag-admin-row" key={b.id}><strong>{b.label}</strong><span>{b.currentArtifactId?'학습 완료':b.candidateArtifactId?'검증 대기':'학습 전'}</span></div>)}
      <label className="field"><span>학습 snapshot ID</span><input value={snapshot} onChange={e=>setSnapshot(e.target.value)} /></label>
      <button type="button" className="button secondary" disabled={busy||!snapshot||data.families.length!==1} onClick={()=>action('snapshot-imports',{snapshotId:snapshot,familyId:data.families[0].id})}>자료 검증·연결</button>
      <label className="field"><span>자동학습 허용 소스 ID</span><input value={sourceIds} onChange={e=>setSourceIds(e.target.value)} placeholder="쉼표로 구분 · 소스의 학습 허용 필요" /></label>
      <button type="button" className="button secondary" disabled={busy||!sourceIds} onClick={()=>action('learning/policy',{enabled:true,allowedSourceIds:sourceIds.split(',').map(x=>x.trim()).filter(Boolean)},'PATCH')}>자동학습 활성화</button>
      <h3>검증 · 배포</h3>
      <label className="field"><span>평가셋 추가</span><input type="file" accept=".json" disabled={busy} onChange={async e=>{const file=e.target.files?.[0];if(!file)return;try{if(file.size>1024*1024)throw new Error('평가셋은 1 MB 이하입니다.');await action('suites',JSON.parse(await file.text()));}catch(cause){setError(cause.message);}e.target.value='';}} /></label>
      <label className="field"><span>검증 모델</span><select value={evaluationModel} onChange={e=>setEvaluationModel(e.target.value)}><option value="">선택</option>{data.models.filter(m=>['base','specialist_checkpoint','merged_checkpoint','quantized'].includes(m.kind)).map(m=><option value={m.id} key={m.id}>{m.kind} · {m.id.slice(-8)}</option>)}</select></label>
      <button type="button" className="button secondary" disabled={busy||!evaluationModel} onClick={()=>action('certification-runs',{artifactId:evaluationModel})}>전체 검증</button>
      {data.releases.map(r=><div className="rag-admin-row" key={r.id}><span>{r.engineAlias}<small>{states[r.state]??r.state}</small></span><button className="button secondary" type="button" disabled={busy||r.state==='active'} onClick={()=>action('releases/'+r.id+'/activate',{})}>사용</button></div>)}
      <div className="run-list">{data.cycles.slice(-5).map(c=><div className="run-row" key={c.id}><strong>{states[c.state]??c.state}</strong><span className="muted small">{(c.reasonCodes??[]).join(' · ')}</span></div>)}</div>
      <p className="muted small">실제 학습·병합 결과만 표시 · 평가 미통과 모델은 자동 적용되지 않습니다.</p>
    </>}
  </details>;
}

export function RegistryPicker({onSelect,disabled=false}) {
  const [models,setModels]=useState([]),[branches,setBranches]=useState([]),[selected,setSelected]=useState([]),[check,setCheck]=useState(null),[status,setStatus]=useState(''),[busy,setBusy]=useState(false);
  useEffect(()=>{Promise.all([request('models'),request('branches')]).then(([r,b])=>{setModels(r.items.filter(a=>a.kind==='specialist_checkpoint'&&a.state!=='revoked'));setBranches(b.items);}).catch(()=>{});},[]);
  async function choose(ids){setSelected(ids);setCheck(null);if(ids.length<1)return;try{setCheck(await request('combinations/preflight',{artifactIds:ids}));}catch(e){setStatus(e.message);}}
  async function build(){setBusy(true);try{const recipe=await request('combinations',{artifactIds:selected});const job=await request('combinations/'+recipe.id+'/build',{});setStatus('병합 대기 · '+job.id);}catch(e){setStatus(e.message);}finally{setBusy(false);}}
  if(!models.length)return null;
  return <div className="rag-tools-group"><h3>전문 모델 · 조합</h3>{models.map(a=><label className="nas-candidate" key={a.id}><input type="checkbox" disabled={disabled||busy} checked={selected.includes(a.id)} onChange={e=>choose(e.target.checked?[...selected,a.id]:selected.filter(id=>id!==a.id))} /><span className="nas-candidate-body"><strong>{branches.find(b=>b.id===a.branchId)?.label??'전문 모델'} · {a.id.slice(-8)}</strong><small>{a.state==='ready'?'사용 준비':'검증 전'}</small></span></label>)}
    {check&&<p className="muted small">{check.displayName} · {check.compatible?'호환':'병합 불가'} · {check.reasonCodes.join(' · ')}</p>}
    <div className="rag-tool-buttons"><button type="button" className="button secondary" disabled={disabled||busy||!check?.actions.build} onClick={build}>조합 만들기</button><button type="button" className="button secondary" disabled={!check?.actions.use||disabled} onClick={async()=>{try{const releases=await request('releases');const release=releases.items.find(r=>r.artifactId===check.cachedArtifactId);if(release)onSelect(release.engineAlias);else setStatus('배포 인증 필요');}catch(e){setStatus(e.message);}}}>조합 사용</button></div><p className="muted small">{status}</p>
  </div>;
}

export function TrainingBranchSelect({value,onChange}) {
  const [branches,setBranches]=useState([]);
  useEffect(()=>{request('branches').then(r=>setBranches(r.items)).catch(()=>{});},[]);
  if(!branches.length)return null;
  return <label className="field"><span>학습 전문 모델</span><select value={value??''} onChange={e=>onChange(e.target.value)}><option value="">선택</option>{branches.map(b=><option key={b.id} value={b.id}>{b.label} · {b.specialization}</option>)}</select></label>;
}
