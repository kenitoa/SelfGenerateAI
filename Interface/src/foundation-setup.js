export function trainingSetup({ connected, stale, loading, sourceLoading, sourceError, sources, selectedSources, mode, enabled, running }) {
  if (loading || sourceLoading) return { action: 'wait', title: '학습 준비 상태를 확인하고 있어요', description: '서버 연결과 학습 자료의 이용 권한을 확인합니다.', label: '확인 중' };
  if (!connected || stale) return { action: 'refresh', title: '서버 연결을 확인해 주세요', description: '최신 상태를 받을 수 없어 학습 설정을 저장할 수 없습니다.', label: '다시 연결' };
  if (sourceError) return { action: 'refresh', title: '학습 자료 목록을 불러오지 못했어요', description: '자료 목록을 다시 불러온 뒤 이용 권한과 선택 상태를 확인해 주세요.', label: '자료 다시 불러오기' };
  if (!sources.length) return { action: 'data', title: '먼저 학습할 자료를 추가해 주세요', description: '파일이나 소스를 추가한 뒤, 이 화면에서 학습 이용을 승인하고 자료를 선택하세요.', label: '자료 추가하기' };
  const allowed = sources.filter(source => source.trainingPolicy === 'allowed' && source.kind !== 'local_directory');
  if (!allowed.length) return { action: 'sources', title: '자료의 학습 이용 승인이 필요해요', description: '자료를 등록해도 학습 이용은 따로 승인해야 합니다. 아래에서 권한을 확인하고 승인해 주세요.', label: '학습 이용 승인하기' };
  if (!selectedSources.length) return { action: 'sources', title: '학습에 사용할 자료를 선택해 주세요', description: `학습이 허용된 출처가 ${allowed.length}개 있습니다. 아래 목록에서 사용할 자료를 체크하세요.`, label: '학습 자료 선택하기' };
  if (selectedSources.some(id => !allowed.some(source => source.id === id))) return { action: 'sources', title: '선택한 자료의 이용 권한을 다시 확인해 주세요', description: '학습이 허용되지 않거나 삭제된 출처가 포함되어 있습니다. 아래에서 선택을 해제하거나 이용 권한을 확인하세요.', label: '선택한 자료 확인하기' };
  return { action: 'save', title: running ? '학습 작업이 진행 중이에요' : enabled ? '학습 설정이 저장되어 있어요' : '학습할 자료가 선택되었어요',
    description: mode === 'manual' ? '수동 모드에서는 설정을 저장한 뒤, 아래 단계별 제어에서 학습을 실행하세요.' : '자동 학습을 켜면 자료 검사 → 토큰 학습 → 초기화 → 실측 → 기초학습을 순서대로 진행합니다. 실행 조건이 부족하면 대기 이유를 표시합니다.',
    label: mode === 'manual' ? '수동 설정 저장' : enabled ? '자동 학습 설정 갱신' : '자동 학습 켜기' };
}
