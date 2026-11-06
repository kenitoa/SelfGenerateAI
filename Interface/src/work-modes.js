export const workModes = [
  { id: 'code', label: '코드 작성', description: '기능 · 구현', placeholder: '만들 기능과 언어를 입력하세요.' },
  { id: 'review', label: '코드 검토', description: '버그 · 예외 · 보안', placeholder: '검토할 코드와 확인할 부분을 입력하세요.' },
  { id: 'refactor', label: '리팩토링', description: '구조 · 가독성', placeholder: '개선할 코드와 유지할 동작을 입력하세요.' },
  { id: 'test', label: '테스트', description: '정상 · 경계 · 오류', placeholder: '테스트할 코드와 사용할 도구를 입력하세요.' },
];

export const normalizeWorkMode = value => workModes.some(mode => mode.id === value) ? value : 'code';

export function createDemoReply(mode, prompt) {
  const selected = normalizeWorkMode(mode);
  if (selected === 'review') return {
    text: '검토 형식 예시 · 입력 코드의 실제 분석은 모델 연결 후 제공됩니다.\n\n확인 항목\n• 입력값과 타입\n• 예외 처리와 경계 조건\n• 외부 입력과 권한\n• 개선 우선순위',
  };
  if (selected === 'refactor') return {
    text: '리팩토링 예시 · 입력 코드의 실제 수정안은 모델 연결 후 제공됩니다.\n\n중복 제거 함수를 분리한 예시입니다.',
    code: '// 변경 전\nconst values = [1, 2, 2, 3];\nconst result = [...new Set(values)];\n\n// 변경 후: 재사용 가능한 함수\nconst unique = (items) => [...new Set(items)];\nconst result = unique(values);',
  };
  if (selected === 'test') return {
    text: '테스트 형식 예시 · 입력 코드에 대한 테스트 생성과 실행은 모델 연결 후 제공됩니다.',
    code: '// 예시: Node.js 내장 테스트\nimport test from "node:test";\nimport assert from "node:assert/strict";\n\nconst unique = (items) => [...new Set(items)];\n\ntest("중복 제거", () => {\n  assert.deepEqual(unique([1, 2, 2]), [1, 2]);\n});\n\ntest("빈 배열", () => {\n  assert.deepEqual(unique([]), []);\n});',
  };
  return /파이썬|python|파일/i.test(prompt)
    ? { text: '코드 작성 예시 · 실제 요청에 맞는 코드 생성은 모델 연결 후 제공됩니다.', code: 'from pathlib import Path\n\ndef read_text(path: str) -> str:\n    file = Path(path)\n    if not file.is_file():\n        raise FileNotFoundError(path)\n    return file.read_text(encoding="utf-8")' }
    : { text: '코드 작성 예시 · 실제 요청에 맞는 코드 생성은 모델 연결 후 제공됩니다.', code: '// 예시: 중복 제거\nconst unique = (items) => [...new Set(items)];\n\nconsole.log(unique([1, 2, 2, 3]));\n// [1, 2, 3]' };
}
