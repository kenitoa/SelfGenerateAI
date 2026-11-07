# SEED LLM

`Interface ↔ RAG ↔ LLM` 구조의 실제 모델 관리·LoRA·단일 가중치 병합·자동학습 제어 구현입니다. 모델을 번갈아 호출하는 앙상블은 사용하지 않습니다. 기존 Interface 디자인은 유지합니다.

자체 모델은 `LLM/foundation/`에 구현했습니다. 승인 원문 → 자체 byte-BPE → 11,014,400개 무작위 가중치 → CPU 전체 파라미터 사전학습/CPT/SFT → 독립 평가 → 인증된 release 순서입니다. 학습 화면에서 외부 모델 없이 프로젝트를 만들 수 있습니다. Genesis와 테스트 가중치는 작업용 모델로 사용할 수 없습니다. [자체 모델 구현·남은 인수 항목](../docs/SelfMadeLLM_Implementation_Status.md)을 참고하세요.

2026-11-07: 사용자 DS923+ RAM 32 GB 기준으로 LLM 목표 상한 24 GiB·시스템 여유 8 GiB를 적용했습니다. 모든 문서의 처리 이력·SFT/CPT·목적별 공급, 모델이 작성한 전처리 코드의 격리 검증·실제 파일 적용·복원, 원본 대비 전문 분야 평가를 추가했습니다. [설정·자가발전 계약](../docs/LLM_Self_Evolution.md)을 참조하세요.

최초 실제 PC 모델 비교는 두 모델 합계 572회 실행을 마쳤습니다. 자체 S0는 기능 0/282·문법 0/282·필수 안전 0/4, 비교 전용 Qwen2.5-Coder-0.5B-Instruct는 기능 110/282(약 39.0%)·문법 195/282·필수 안전 0/4였습니다. 고정한 부분 문제와 전체 코드 응답 규칙의 결과이며 공식 benchmark 전체 점수가 아닙니다. 자체 모델의 작업용 품질은 아직 미달입니다. Qwen은 외부 비교 전용이며 자체 초기화·학습·teacher로 사용하지 않았습니다.

## 설치

프로젝트 루트, Python 3.12 이상:

```powershell
python -m venv LLM/.venv
LLM/.venv/Scripts/python.exe -m pip install -r RAG/requirements.txt -r LLM/requirements.txt
cd Interface
npm.cmd install
npm.cmd run build
cd ..
```

Windows는 위 명령, NAS/Linux는 `LLM/.venv/bin/python`을 사용합니다. 기존 RAG lock 전체를 덮어 설치하면 Transformers의 hub/tokenizer 요구와 충돌할 수 있으므로 함께 resolver에 전달한 requirements 또는 검증한 LLM lock을 사용합니다.

## 실행

```powershell
LLM/.venv/Scripts/python.exe -m LLM serve --integrated --local --worker --port 8000
```

접속은 http://127.0.0.1:8000 입니다. 기존 RAG 자료는 `RAG/data`, LLM 모델·registry·작업은 `LLM/runtime-data`에 저장합니다. 내부 raw-generate facade는 loopback 8100입니다. 이미 같은 포트를 쓰면 종료하지 말고 `--port 8001` 등으로 분리합니다. `--local`은 loopback 개발 전용입니다.

NAS에서는 RAG 관리자 초기화와 HTTPS reverse proxy를 사용하고 `--local`을 제외합니다. 기본 4 GB에서 RAG+생성+훈련 동시 동작을 보장하지 않습니다. worker를 켜지 않으면 자가학습은 자원 대기이며 실제 학습을 가장하지 않습니다.

## 외부 비교 모델 준비

```powershell
LLM/.venv/Scripts/python.exe -m LLM install-model
LLM/.venv/Scripts/python.exe -m LLM check
```

이 명령의 기본 외부 모델은 Qwen2.5-Coder-0.5B-Instruct입니다. 자체 모델 생성에는 이 명령을 사용하지 않습니다. Hub에서 실제 revision을 조회해 고정하고 safetensors/config/tokenizer만 가져옵니다. 등록 모델은 `runtime-data/imports` 및 불변 `artifacts`에 보관합니다. 외부 모델 경로는 `SEED_LLM_MODEL_ROOTS`로 관리자가 추가합니다. 모델 파일은 Git에 넣지 않습니다.

다른 설치 경로:

```powershell
LLM/.venv/Scripts/python.exe -m LLM register C:/approved/model --license Apache-2.0 --name MyAnchor
```

CLI 경로 등록은 관리자 작업입니다. 웹 등록은 서버 allowlist 밖 경로를 거절합니다. pickle 모델·remote code는 사용하지 않습니다.

## 화면과 작업

- **설정 → RAG 관리 → LLM · 학습 · 조합**: 등록·A/B/C branch·snapshot import·자동화 정책·상태·복원.
- **모델 선택 → 전문 모델 · 조합**: 실제 학습 산출물을 복수 선택하고 preflight 후 조합을 만듭니다.
- 조합이 아직 없으면 병합 대기, gate를 통과하지 못했으면 검증 전입니다. 준비 전 부모 모델로 대체 응답하지 않습니다.
- A/B/C는 같은 anchor에서 학습한 specialist입니다. 0.5B/1.5B/3B를 직접 합산하지 않습니다.
- `A+B`, `A+C`, `B+C`, `A+B+C`는 각 recipe에서 계산한 full safetensors 산출물 하나입니다. 선택 순서를 바꿔도 같은 버전·기여도면 캐시를 재사용합니다.

## 실제 학습

RAG의 승인 snapshot을 가져와 manifest·JSONL 파일·행별 canonical hash·계보·분할을 확인합니다. legacy readySamples는 전체 train+validation+test 수이며 optimizer에는 train만 들어갑니다.

학습은 standard LoRA, 실제 optimizer update·loss·validation·checkpoint·재개를 제공합니다. 기존 training API 상태·제어를 유지합니다. 학습 완료와 모델 배포 완료는 별개입니다.

기본 목적은 SFT입니다. 명시적인 `config.objective`로 DPO·CPT를 선택할 수 있으며, 데이터 형식과 API 예시는 [구현 현황](../docs/LLM_Implementation_Status.md#4-실행데이터-계약)에 있습니다. 일반 자료를 선호쌍으로 자동 변환하지 않습니다. 대화 요청이 기다리면 로컬 학습을 optimizer 경계에서 저장·양보하고 이어서 재개합니다.

정책은 허용 source ID와 source의 trainingPolicy=allowed를 모두 요구합니다. Python 승인 원문에서 AST/span 기반 완성 과제를 생산할 수 있습니다. 실행 증거 없는 자료를 테스트 통과로 표시하지 않습니다. 자유형 정답 자동승인은 하지 않습니다.

주기는 신규량·시간대·cooldown을 검사하고 snapshot→train→dev→선택 조합 build→배포 형태→release 평가→인증→활성화를 이어 갑니다. 필요한 suite·도구·권한이 없으면 held/대기 이유를 기록합니다. 실패 모델은 운영 champion을 바꾸지 않습니다.

## 평가와 양자화

`/api/v1/llm/suites`로 dev/release/regression/evidence/unanswerable/audit suite를 등록합니다. 고정 답·근거 체크, 격리 Python unittest와 Python·JavaScript·TypeScript·SQL case fixture를 지원합니다. 임의 생성 코드를 호스트에서 실행하지 않습니다.

Docker 검증기는 관리자가 등록한 digest 고정 이미지가 필요합니다. Docker가 없으면 SANDBOX_UNAVAILABLE입니다. 재부팅 후 로컬 Linux Docker에서 실제 격리·인증·Chromium 화면 검사를 수행했습니다. 이 증거는 해당 source hash와 PC 환경에만 유효하며 NAS 시험으로 재사용하지 않습니다.

GGUF 변환은 검증한 `convert_hf_to_gguf.py`와 `llama-quantize` 경로·SHA256을 `tools` 설정에 등록해야 합니다. F16·Q4_K_M 파일은 각각 별도 artifact입니다. 자체 Foundation GGUF 경로는 고정한 llama-cpp-python 엔진·실제 라이브러리 hash·CPU 옵션·tokenizer 적합성 receipt를 검증합니다. 파일 생성이나 수치 검증 성공만으로 실제 기능 품질·사용 자격을 부여하지 않습니다.

```powershell
LLM/.venv/Scripts/python.exe -m LLM configure-tools --converter C:/approved/convert_hf_to_gguf.py --quantizer C:/approved/llama-quantize.exe
LLM/.venv/Scripts/python.exe -m LLM register-verifier --language python --image YOUR_IMAGE@sha256:ACTUAL_DIGEST
LLM/.venv/Scripts/python.exe -m LLM register-fixture C:/approved/fixture.json
```

위 경로와 digest는 실제 검증한 값으로 지정합니다. 도구나 이미지를 자동으로 설치·실행했다고 가정하지 않습니다. 고급 병합은 TIES·DARE·2개 입력 SLERP를 지원합니다. `/recipe-searches`는 dev에서 최대 3개 recipe만 실제 평가합니다. `/adapter-compressions`의 SVD는 근사 오차를 기록하며 압축만으로 품질을 인증하지 않습니다.

고정밀 HF native runtime도 지원합니다. 자동 정책의 기본 배포 형태는 gguf이며 다른 형태는 자원 gate 통과가 필요합니다. 이 저장소에는 프로젝트 전용 정답 360개나 안전한 sandbox 이미지가 임의로 마련돼 있다고 가장하지 않습니다. 해당 실제 suite를 등록하기 전 자동 승격은 차단됩니다.

## 전용 worker

NAS 밖 worker는 DB를 열지 않고 scope 제한 HTTP로 입력 파일과 결과만 교환합니다. 관리자 서비스 인증으로 `/internal/v1/workers/register`에 등록하면 전용 token을 한 번 반환합니다. token은 파일로 보관하고 로그·명령행에 값 자체를 넣지 않습니다.

```powershell
LLM/.venv/Scripts/python.exe -m LLM.workers.agent --endpoint https://private-controller.example --worker-id ACTUAL_ID --token-file C:/private/worker-token --data-dir C:/seed-worker
```

HTTPS 또는 loopback만 허용합니다. GPU/worker는 자동 구매·등록하지 않습니다. remote pause/resume은 체크포인트 전송 범위가 검증되기 전 capability=false이며 취소·완료·fencing은 지원합니다. worker 입력 전송은 별도 만료 lease를 사용하고 실행 heartbeat로 연장합니다.

## 백업·보존

`python -m LLM backup`은 기본 AES-256-GCM 암호화 백업을 만듭니다. 키는 패키지 밖의 제한된 저장소에 보관합니다. `restore --backup 실제_백업_경로 --destination 비어있는_새_경로 --key-store 원본_키_저장소`로 철회·만료·태그·해시를 확인한 뒤 유지보수 상태로 복원합니다. 평문 이전 백업은 명시적인 `--allow-legacy`가 필요합니다. 진행 작업은 백업을 보류하며 계보와 현재 모델 참조를 보존합니다. 만료·철회 백업과 개인 키는 별도 수명주기 정책으로 처리합니다. `python -m LLM gc`는 dry-run, `gc --apply`는 실제 정리입니다.

## 검증

```powershell
LLM/.venv/Scripts/python.exe -m pytest RAG/tests LLM/tests -q
cd Interface
node --test src/rag-api.test.js src/training-api.test.js
npm.cmd run build
```

`LLM/tests/live_smoke.py`는 실제 가중치의 fixture 학습을 위한 선택적 기술 검증 스크립트입니다. 파일의 존재를 실행 실적으로 세지 않습니다. 현재 등록한 시장 baseline은 `comparisonOnly`이므로 이 학습에 사용할 수 없습니다. 일반 테스트의 작은 무작위 Qwen2 구조·LoRA·네 조합 텐서 검사는 자체 S0의 외부 초기 가중치 사용이나 실제 품질 향상 증거가 아닙니다.

## 관련 문서

- [PRD](../docs/LLM_PRD.md)
- [기능명세](../docs/LLM_Functional_Specification.md)
- [구현·검증 현황](../docs/LLM_Implementation_Status.md)
- [자체 모델 구현·검증](../docs/SelfMadeLLM_Implementation_Status.md)
- [자체 모델 PRD](../docs/SelfMadeLLM_PRD.md)
- [자체 모델 기능명세](../docs/SelfMadeLLM_functional_Specification.md)
- [23개 배포 전 인수 결과](../docs/PreNAS_Acceptance_Result.md)

## 로컬 후보와 배포 전 검사

NAS 운영 학습과 별개인 실제 후보는 저장소 루트에서 `docker\train_pc_candidate.cmd`로 실행합니다. 승인된 자료의 준비 보고서가 있어야 하며, 초기화·실측·실제 전체 가중치 학습·독립 NLL을 checkpoint와 기록합니다. `testOnly=false`, `executionScope=pc_validation`이며 NAS 자격으로 재사용할 수 없습니다. 같은 run의 재개는 기존 구성과 optimizer/RNG를 보존합니다. 새 PC run만 `--cpu-threads 4 --micro-batch-size 4`로 별도 실측할 수 있습니다. 기본 NAS run은 1스레드입니다.

`docker\setup_local_container.cmd`는 검증된 공식 Docker/WSL 설치와 재부팅 상태를 확인합니다. `docker\check_before_NAS.cmd`는 실제 소스·시험 코드 해시에 묶인 통합 검사와 로컬 Linux 이미지·격리·인증 검사를 수행합니다. 모델 능력 합격은 별도이며 점수 입력으로 해제할 수 없습니다. 준비 자료는 `export_prepared_training_data.cmd`, `import_prepared_training_data.cmd`로 검증합니다. `upload_training_data.cmd`는 기본 dry-run이며 실제 전송에는 `--send`가 필요합니다. 데이터 전송은 Docker 배포나 학습 활성화를 수행하지 않습니다.

## 다음 PC 후보: 최대 4시간·최대 4스레드

최종 source `c02e8f844a0669233de6e192229c82497f9ecf136b30af43f29de73f1492bfc7`에서 프로그램 709개·추가 14 subtests·실패 0·skip 0을 통과했습니다. pytest 651.08초·전체 runner 655.87초, software 영수증 `9e83ceeb-4970-415a-b492-972d27c5984d`, testSourceHash `53b700d3bd8677b2d0a89f512eb98f572792877c1bb8854ab40fad5d70fc2403`입니다. 같은 source의 Docker 인수 `16990bc0-ff45-4855-8005-6c2049144a1d`도 87.22초에 12조건과 Chromium UI 19검사를 통과했습니다. 종료 guard 보강 전 `d0f4e6a…` 결과는 과거 이력입니다.

23항목은 통과 10개(2·3·12·13·14·15·16·19·22·23), 미완료 8개(6·7·9·10·17·18·20·21), 연기 5개(1·4·5·8·11)로 `ready=false`입니다. 로컬 프로그램·화면 검사 통과가 모델 능력 합격이나 NAS 배포 완료를 뜻하지 않습니다.

| 준비 자료 | 현재 확인값 |
| --- | --- |
| `official-python-v2` | ready, train 68,763,876 bytes·18,109,545 targets·20그룹 |
| 독립 raw dev | 기존 2,330,573 tokens·20그룹 불변 |
| `official-context-v3` | ready, train 121예시·17그룹·13,166 assistant targets, dev 10예시, 총 문맥 512·target 최대 256 |
| 현재 실제 source 실행 검증 | `3072e14a-c3c1-45ee-bc59-3155fe32653a`, 정의 131개 로드·의미 검사 대상 7개 전부 통과·124개 의미 검사 대기 |

이전 `be2…` 검증의 9개 의미 사례 중 7개 통과·2개 실패 기록은 보존했습니다. 자유 함수와 메서드인 `marquee`, branch/core의 `chain`과 parser-local 별칭을 현재 원문 심벌로 구분합니다. 9개 연결의 판정은 일치 7·불일치 1·미해결 1이며 예상값·원문·학습 target은 바꾸지 않았습니다. 추가 패키지 검증 뒤 2026-11-07 13:59:35 UTC에 PC active raw v2·SFT v3를 CAS로 선택했고, 최종 검사 후 아래 실제 실행을 시작했습니다.

operation `python-v2-1ab5cd43-9557-4cef-b3b9-b7fe57782568`의 시작은 2026-11-07 14:21:40.587 UTC(한국 11월 7일 23:21:40), 절대 마감은 18:21:40.587 UTC(한국 11월 8일 03:21:40)입니다. 시작 epoch `1794061300.587217`, deadline epoch `1794075700.587217`을 고정했습니다. guardian binding은 `e1b76fa6…`입니다.

실제 사전시험 `3afe13c0-c6a0-4bb4-a179-884fea1515b2`는 completed/passed이며 profile은 `61439508-b462-433a-9490-e72bfa70e69a`입니다. warmup 3·측정 20 steps, intra-op 4·inter-op 1에서 effective 1,246.221969 targets/s·peak RSS 1,951,551,488 bytes(약 1.82 GiB)를 관측했습니다. 해당 PC recipe의 복사 모델 측정이므로 생산 학습 토큰 0이며 NAS 처리량 측정이 아닙니다.

첫 CPT `c304066d-a3ad-42ce-a48e-0eb239cfb8ba`는 실제 running입니다. 2026-11-07 14:26:31 UTC에 checkpoint `0b5fb1a6-1ed4-4996-851b-e40d08faac27`의 확정 132,730 targets·35 optimizer steps를 전체 검증했습니다. manifest hash는 `91690fb0d34fb8ace50f91162b94e5cf44fa19ff899390fb777555924829cca1`입니다. 원래 ledger·checkpoint·572개 평가 답변·slot·extension 불변과 같은 최종 source도 확인했고 증거는 `reports/pc-next-first-checkpoint.json`에 보관합니다. 학습은 계속 중이며 `quality=false`·NAS 변경 0입니다. 첫 checkpoint를 CPT 단계 완료나 모델 품질 합격으로 표시하지 않으며 한국 시간 11월 8일 03:21:40 절대 마감도 유지합니다.

최종 자료·검사 확인 후 저장소 루트에서 실행합니다.

```bat
docker\run_pc_candidate_next.cmd
```

명령은 원래 SFT 직전 CPT512 부모에서 분기한 **같은 operation**을 실행·재개합니다. raw v2의 CPT를 최대 3단계·각 4M targets까지 요청하고 각 checkpoint를 동일 v2 dev에서 새로 qualification한 뒤, 정상 자격을 충족하면 SFT v3를 최대 2 epoch 수행합니다. CPT는 문맥 512·microbatch 4·누적 2·learning rate 0.0001, SFT는 learning rate 0.00005입니다. CPU 연산 설정은 최대 intra-op 4이며 전체 OS 스레드 수와 구분합니다. 같은 SFT config라도 다른 corpus의 사전시험을 재사용하지 않습니다.

자료 준비가 끝난 뒤 한 번 시작한 **절대 14,400초**에 calibration·학습·qualification·정리를 포함합니다. 재실행이나 재부팅으로 시간을 늘리지 않습니다. 종료 전 안전 정리 시간을 남기고 마지막 검증 checkpoint·실제 완료 targets·부족량을 보고합니다. 요청 상한을 4시간 내 완료하거나 품질이 개선된다고 보장하지 않습니다.

자동 학습은 계속 비활성입니다. 해당 PC 프로젝트의 일일 시간 상한만 원래 14,400초에서 21,198초로 임시 변경했고 `enabled=false`·manual과 사용 카운터·권한·다른 설정은 유지합니다. 종료 시 자신의 임시 값이 그대로일 때만 원래 상한을 복구합니다. NAS 접속·쓰기·전송·배포·모델 활성화·자기 코드 적용은 수행하지 않습니다. 최초 후보 checkpoint와 평가 답변도 보존합니다.

별도 `pc_budget_guard`는 현재 source·입력 ID·불변 권한·owner/fence·payloadHash·정확한 PID/생성 시각을 확인합니다. DB 잠금과 독립적으로 벽시계·monotonic 마감을 감시하고 180초 전 안전 중지를 요청합니다. 부모 소멸 뒤 최대 120초의 정리 유예가 원래 기한을 늘리지는 않습니다. 다음 실행에서 남은 자신의 임시 cap을 CAS로 회수하면 `review-required`로 멈추며 새 예산이나 자동 재개를 허용하지 않습니다.

- 진행 로그: `docker/.state/local-container/pc-next-candidate-console.log`.
- 결과: 준비 루트 `reports/pc-next-candidate-<operationId>.json` 및 자동 생성 `.md`.
- 계획 전용: 준비된 Python으로 `docker/run_pc_candidate_next.py --plan`을 실행합니다. `.cmd`는 실제 실행용입니다.
- 직접 CLI: `python -X utf8 -m LLM.foundation.pc_next_candidate --preparation-root PREP_ROOT --operation-id STABLE_OPERATION_ID --runtime-verification-id CURRENT_RUNTIME_RECORD_ID`. 기본은 계획 확인이며 `--execute`가 있어야 실행합니다.

현재 `reports/pc-next-candidate-python-v2-1ab5cd43-9557-4cef-b3b9-b7fe57782568.json`은 실제 생성됐으며 종료 시 같은 이름의 Markdown 결과를 자동 저장합니다. 문서의 CPT 진행은 전달된 스냅샷이므로 최신 상태와 확정 완료량은 JSON·job·검증 checkpoint로 확인합니다. 준비 당시의 권한 0·미시작 기록은 현재 실행 상태와 구분합니다.

raw/SFT manifest hash·현재 helper·심벌 연결·fixture/실제 격리 receipt가 일치하지 않으면 시작 전에 보류합니다. 최신 추가 패키지는 47,015 records·9,685 files·390,055,368 bytes로 실제 로컬 검증을 통과했습니다. SHA256은 `1bc00d890d27638f163c0c292b9c8621f33d3192b60aed9fe7042f251f1e2194`, 인수 reportHash는 `fe3a6bb2cd754bf5376d3b6e3140722b64f16c5da5f3598eba20580754a52d25`입니다. 원본 archive를 제외하고 최신 binding 9개를 보존했으며 패키지 검증 때 주 준비 저장소의 원본·정책·모델과 NAS는 변경하지 않았습니다. 이전 `f477…` 패키지의 통과를 새 파일에 재사용하지 않았습니다. 이후 PC 자료 선택·임시 시간 상한·실행 시작은 각각 별도 기록입니다.

모델 인수 6·7·9·10·17·18·20·21은 미완료이며, NAS 항목 1·4·5·8·11은 사용자 지시로 미뤘습니다. 자세한 시간 권한·재개·정책 복구·현재 실행 증거 계약은 [기능명세 27절](../docs/SelfMadeLLM_functional_Specification.md#27-sf-24-다음-pc-후보의-절대-4시간-실행)에 있습니다.
