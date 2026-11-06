import React from 'react';
import { Check, ChevronDown, Cpu, ExternalLink, GitBranch } from 'lucide-react';
import { isNasCandidate, nasModels, nasSources } from './nas-models';
import './nas-models.css';
import { RegistryPicker } from '../../LLM/web/Panel.jsx';

export default function NasModelCatalog({ value, onSelect, ram, onRamChange, disabled = false }) {
  const id = React.useId();
  return <div className="nas-catalog">
    <div className="nas-hardware"><span><Cpu size={14} />DS923+ · CPU</span><label><span>기준 RAM</span><select aria-label="DS923+ 기준 RAM" value={ram} onChange={event => onRamChange(Number(event.target.value))} disabled={disabled}>{[4, 8, 16, 32].map(size => <option key={size} value={size}>{size} GB{size === 4 ? ' · 기본' : ''}</option>)}</select></label></div>
    <div className="nas-catalog-caption"><span>CPU 추론 후보</span><span>설치·실측 전</span></div>
    <div className="nas-candidates" role="radiogroup" aria-label="DS923+ 추론 모델 후보">
      {nasModels.map(model => {
        const eligible = isNasCandidate(model.id, ram);
        return <label className={`nas-candidate ${value === model.id ? 'selected' : ''} ${!eligible ? 'needs-upgrade' : ''}`} key={model.id}>
          <input type="radio" name={`nas-model-${id}`} checked={value === model.id} value={model.id} onChange={() => onSelect(model.id)} disabled={disabled || !eligible} />
          <span className="nas-candidate-body"><span className="nas-candidate-title"><strong>{model.name} <b>{model.size}</b></strong><span className={`nas-tier ${model.size === '0.5B' ? 'preferred' : ''}`}>{!eligible ? `${model.reviewRam} GB+ 검토` : model.tier}</span></span><span className="nas-candidate-meta">Q4_K_M · {model.fileSize} 파일</span><span className="nas-candidate-note">{model.note}</span></span>
          {value === model.id && <Check size={14} />}
        </label>;
      })}
    </div>
    <RegistryPicker onSelect={onSelect} disabled={disabled} />
    {!isNasCandidate(value, ram) && !value?.startsWith('seed/') && <p className="nas-selection-warning">현재 선택 모델 · 기준 RAM 또는 호환성 재확인</p>}
    <p className="nas-inference-note">짧은 컨텍스트 · 1개 모델 로드 기준<br />파일 크기 ≠ 실행 RAM · 가용 메모리·속도 실측 필요</p>
    <details className="nas-merge-notes"><summary><GitBranch size={13} />학습·병합 조건<ChevronDown size={12} /></summary><div><p>Q4는 추론용 후보 · 학습·병합 자원은 별도</p><p>0.5B·1.5B·3B 간 직접 가중치 합·곱 불가</p><p>같은 베이스·텐서 크기·토크나이저 호환성 검토</p><p>단순 합·곱으로 성능 향상 보장 안 됨</p><p>3B: Qwen Research 라이선스 확인</p><a href={nasSources.smallConfig} target="_blank" rel="noreferrer">0.5B 구조<ExternalLink size={10} /></a><a href={nasSources.mediumConfig} target="_blank" rel="noreferrer">1.5B 구조<ExternalLink size={10} /></a></div></details>
  </div>;
}
