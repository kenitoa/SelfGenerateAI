import React, { useState } from 'react';
import { Check, Database, ShieldCheck } from 'lucide-react';

export default function FoundationSources({ sources, selectedSources, onSelection, disabled, onApprove, onAddData }) {
  const [reviewId, setReviewId] = useState('');
  const [reason, setReason] = useState('');
  const [license, setLicense] = useState('');
  const [error, setError] = useState('');
  const missing = selectedSources.filter(id => !sources.some(source => source.id === id));
  async function approve(source) {
    if (!reason.trim() || disabled) return;
    setError('');
    try {
      await onApprove(source, { reason: reason.trim(), licenseId: license.trim() || null });
      setReviewId(''); setReason(''); setLicense('');
    } catch (cause) { setError(cause.message); }
  }
  return <div className="foundation-source-setup">
    <p className="foundation-setup-copy">학습 이용을 승인한 출처만 선택할 수 있습니다. 직접 선택한 자료와 학습 방식은 아래 ‘출처 · 방식 저장’으로 저장하세요.</p>
    {sources.length || missing.length ? <div className="foundation-source-options">{sources.map(source => {
      const supported = source.kind !== 'local_directory';
      const allowed = supported && source.trainingPolicy === 'allowed';
      const selected = selectedSources.includes(source.id);
      const review = reviewId === source.id;
      return <div className="foundation-source-option" key={source.id}>
        <div className="foundation-source-option-row"><label><input type="checkbox" checked={selected} disabled={disabled || (!allowed && !selected)} onChange={event => onSelection(event.target.checked ? [...new Set([...selectedSources, source.id])] : selectedSources.filter(id => id !== source.id))} /><span><strong>{source.name}</strong><small>{source.kind === 'upload' ? '업로드 원문' : source.kind === 'git_repository' ? 'Git 저장소' : source.kind === 'web_document' ? '웹 문서' : '이전 폴더 자료'}</small></span></label><span className={`status-pill ${allowed ? 'ready' : 'pending'}`}>{allowed ? '학습 허용' : supported ? '학습 이용 미승인' : '지원 종료'}</span>{supported && !allowed && <button className="button secondary" type="button" disabled={disabled} aria-expanded={review} onClick={() => { setReviewId(review ? '' : source.id); setReason(''); setLicense(source.licenseId ?? ''); setError(''); }}>이용 권한 확인</button>}</div>
        {!supported && <p className="foundation-setup-copy">폴더 소스는 지원이 종료되었습니다. 데이터 화면에서 파일을 다시 추가해 주세요.</p>}
        {review && <div className="foundation-source-review"><p><ShieldCheck size={15} />이 자료를 모델 학습에 사용할 권한이 있는지 확인해 주세요.</p><label className="field"><span>라이선스 · 권한 유형</span><input value={license} onChange={event => setLicense(event.target.value)} disabled={disabled} placeholder="예: MIT 또는 직접 작성" maxLength={200} /></label><label className="field"><span>학습 이용을 허용하는 근거 (필수)</span><input value={reason} onChange={event => setReason(event.target.value)} disabled={disabled} placeholder="예: 직접 작성한 코드로 학습 이용을 허용합니다." maxLength={2000} /></label>{error && <p className="form-error" role="alert">{error}</p>}<div className="foundation-controls"><button className="button primary" type="button" disabled={disabled || !reason.trim()} onClick={() => approve(source)}><Check size={14} />학습 이용 허용</button><button className="button secondary" type="button" disabled={disabled} onClick={() => setReviewId('')}>취소</button></div></div>}
      </div>;
    })}{missing.map(id => <div className="foundation-source-option" key={id}><p className="foundation-setup-copy">목록에 없는 출처가 선택되어 있습니다.</p><button className="button secondary" type="button" disabled={disabled} onClick={() => onSelection(selectedSources.filter(value => value !== id))}>선택 해제</button></div>)}</div> : <p className="foundation-source-empty">아직 등록된 자료가 없습니다. 파일이나 웹 문서·저장소를 추가해 주세요.</p>}
    <button className="button secondary" type="button" onClick={onAddData} disabled={disabled}><Database size={14} />데이터에서 자료 추가</button>
  </div>;
}
