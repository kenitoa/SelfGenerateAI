// Candidate tiers are conservative UI guidance, not measured NAS compatibility.
export const nasModels = [
  { id: 'qwen2.5-coder:0.5b', name: 'Qwen2.5 Coder', size: '0.5B', fileSize: '398 MB', reviewRam: 4, tier: '우선 후보', note: '가벼운 코드 작성 · 단일 모델', license: 'Apache 2.0' },
  { id: 'qwen2.5-coder:1.5b', name: 'Qwen2.5 Coder', size: '1.5B', fileSize: '986 MB', reviewRam: 4, tier: '조건부', note: '짧은 컨텍스트 · 여유 RAM 확인', license: 'Apache 2.0' },
  { id: 'qwen2.5-coder:3b', name: 'Qwen2.5 Coder', size: '3B', fileSize: '1.9 GB', reviewRam: 8, tier: '증설 후 검토', note: '8 GB+ 기준 · CPU 지연 확인', license: 'Qwen Research' },
];

export const nasSources = {
  hardware: 'https://global.download.synology.com/download/Document/Hardware/ProductSpec/DiskStation/23-year/DS923%2B/enu/Product_Spec_DS923%2B_enu.pdf',
  models: 'https://ollama.com/library/qwen2.5-coder',
  smallConfig: 'https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct/raw/main/config.json',
  mediumConfig: 'https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct/raw/main/config.json',
};

export function findNasModel(id) { return nasModels.find(model => model.id === id); }
export function isNasCandidate(id, ram) {
  const model = findNasModel(id);
  return Boolean(model && Number(ram) >= model.reviewRam);
}
