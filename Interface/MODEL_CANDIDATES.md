# DS923+ 인터페이스 모델 후보 근거

확인일: 2026-11-06. 모델 다운로드·추론·학습·병합은 수행하지 않았습니다.

## NAS 기준

Synology 공식 사양: AMD Ryzen R1600, 2코어, 기본 4 GB DDR4 ECC, 최대 32 GB. 화면의 기준 RAM은 사용자가 지정하는 가정값이며, NAS에서 감지한 가용 RAM이 아닙니다. 프로세스 메모리 설정과도 구분합니다.

출처: [DS923+ 공식 하드웨어 사양](https://global.download.synology.com/download/Document/Hardware/ProductSpec/DiskStation/23-year/DS923%2B/enu/Product_Spec_DS923%2B_enu.pdf)

## CPU 추론 후보

| 모델 태그 | 양자화 | 배포 파일 크기 | UI 분류 |
| --- | --- | --- | --- |
| qwen2.5-coder:0.5b | Q4_K_M | 398 MB | 4 GB 기준 우선 후보 |
| qwen2.5-coder:1.5b | Q4_K_M | 986 MB | 4 GB 기준 조건부 후보 |
| qwen2.5-coder:3b | Q4_K_M | 1.9 GB | 8 GB 이상 설정 시 검토 후보 |

파일 크기와 양자화 정보 출처: [0.5B](https://ollama.com/library/qwen2.5-coder:0.5b), [1.5B](https://ollama.com/library/qwen2.5-coder:1.5b), [3B](https://ollama.com/library/qwen2.5-coder:3b).

RAM별 분류는 위 배포 파일 크기와 DSM 등 다른 프로세스의 운영 여유를 고려한 보수적인 판단입니다. 공식 최소 요구 RAM 또는 DS923+ 실측 결과가 아닙니다. 짧은 컨텍스트, 동시 모델 로드 1개를 가정합니다. 파일 크기 외에도 런타임, KV 캐시, 컨텍스트, 동시 요청, DSM 서비스 사용량이 메모리에 영향을 줍니다. 처리량이나 응답 속도를 임의 수치로 표시하지 않습니다.

3B의 공식 배포 페이지에는 Qwen Research 라이선스가 연결되어 있습니다. 0.5B·1.5B의 Apache 2.0 표기를 일괄 적용하지 않습니다.

## 가중치 합·곱과 학습

이 목록은 추론용 양자화 모델 후보입니다. 모델 설치 여부, CPU 추론 실측, 학습 및 병합 가능 여부는 별도로 확인해야 합니다.

같은 Qwen2 계열이라도 크기가 다른 모델은 대응하는 텐서 모양이 다릅니다. 예를 들어 공식 설정에서 0.5B는 hidden size 896 / 24 layers, 1.5B는 hidden size 1536 / 28 layers입니다. 따라서 이 둘을 같은 모양의 가중치처럼 직접 더하거나 원소별로 곱하는 방식으로 병합할 수 없습니다.

출처: [0.5B 원본 구성](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct/raw/main/config.json), [1.5B 원본 구성](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct/raw/main/config.json).

같은 베이스의 파인튜닝 체크포인트 또는 호환 어댑터를 대상으로 텐서 모양·토크나이저·병합 방법을 검증하는 단계가 필요합니다. 단순 합·곱은 학습을 대체하지 않으며 성능 향상도 보장하지 않습니다. 합산 외 다양한 병합 기법과 전처리는 [mergekit 원본 저장소](https://github.com/arcee-ai/mergekit)를 참고할 수 있습니다.
