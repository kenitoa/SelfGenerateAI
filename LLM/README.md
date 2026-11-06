# SEED LLM

`Interface ↔ RAG ↔ LLM` 구조의 실제 모델 관리·LoRA·단일 가중치 병합·자동학습 제어 구현입니다. 모델을 번갈아 호출하는 앙상블은 사용하지 않습니다. 기존 Interface 디자인은 유지합니다.

2026-11-07: 사용자 DS923+ RAM 32 GB 기준으로 LLM 목표 상한 24 GiB·시스템 여유 8 GiB를 적용했습니다. 모든 문서의 처리 이력·SFT/CPT·목적별 공급, 모델이 작성한 전처리 코드의 격리 검증·실제 파일 적용·복원, 원본 대비 전문 분야 평가를 추가했습니다. [설정·자가발전 계약](../docs/LLM_Self_Evolution.md)을 참조하세요.

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

## 모델 준비

```powershell
LLM/.venv/Scripts/python.exe -m LLM install-model
LLM/.venv/Scripts/python.exe -m LLM check
```

기본은 Qwen2.5-Coder-0.5B-Instruct이며 Hub에서 실제 revision을 조회해 고정하고 safetensors/config/tokenizer만 가져옵니다. 등록 모델은 `runtime-data/imports` 및 불변 `artifacts`에 보관합니다. 외부 모델 경로는 `SEED_LLM_MODEL_ROOTS`로 관리자가 추가합니다. 모델 파일은 Git에 넣지 않습니다.

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

Docker 검증기는 관리자가 등록한 digest 고정 이미지가 필요합니다. Docker가 없으면 SANDBOX_UNAVAILABLE입니다. 격리 실행과 실제 NAS 시험은 현재 Windows 환경에서 검증한 것으로 보고하지 않습니다.

GGUF 변환은 검증한 `convert_hf_to_gguf.py`와 `llama-quantize` 경로·SHA256을 `tools` 설정에 등록해야 합니다. Q4_K_M으로 변환한 실제 파일을 별도 artifact로 관리합니다. GGUF 추론은 설치된 Ollama runtime을 사용하며 파일 준비 성공만으로 품질 통과를 표시하지 않습니다.

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

`python -m LLM backup`으로 DB·모델 파일의 checksum 백업을 만들고, `restore --backup 실제_백업_경로 --destination 비어있는_새_경로`로 복원합니다. 하루 간격 보존 정리는 현재 모델·최근 정상 버전·90일 이내 artifact를 보호합니다. 진행 job이 있으면 보류하며 계보 메타데이터는 삭제하지 않습니다. `python -m LLM gc`로 dry-run, `gc --apply`로 적용합니다. snapshot·백업은 자동 삭제 대상에서 제외합니다.

## 검증

```powershell
LLM/.venv/Scripts/python.exe -m pytest RAG/tests LLM/tests -q
cd Interface
node --test src/rag-api.test.js src/training-api.test.js
npm.cmd run build
```

`LLM/tests/live_smoke.py`는 명시적 기술 검증입니다. 실제 설치한 Qwen 0.5B에서 fixture 자료로 8 optimizer step을 실행하고 보고서를 남깁니다. 이 자료·산출물은 fixture 계보로 표시해 운영 인증을 금지합니다. 일반 테스트는 작은 실제 Qwen2 Transformer에서 LoRA와 네 조합의 텐서 결과를 검사합니다. 품질 향상을 검증한 시험과 다릅니다.

## 관련 문서

- [PRD](../docs/LLM_PRD.md)
- [기능명세](../docs/LLM_Functional_Specification.md)
- [구현·검증 현황](../docs/LLM_Implementation_Status.md)
