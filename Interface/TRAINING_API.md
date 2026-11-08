# 학습 상태 인터페이스 계약

기본 화면은 자체 기초 모델의 자동 학습 API를 사용합니다. 기존 전문 모델·LoRA 영역은 아래 `/api/training/*` 게이트웨이 계약을 유지합니다. 예시 JSON은 문서용이며 화면에 주입하지 않습니다.

## 자체 기초 모델 자동 학습

- `GET /api/v1/llm/foundation/auto-training`: 자료·환경·권장 설정·진행 단계의 읽기 전용 확인. 프로젝트·승인·수집 작업을 만들지 않습니다.
- `POST /api/v1/llm/foundation/auto-training`: `{ "autoApproveSources": true }` 한 번으로 프로젝트 생성·재사용과 영속 자동화 요청을 저장하고 `202`로 접수를 응답합니다. 자료 승인·선택·수집, 권장 설정 적용과 모델 준비·학습은 서버 supervisor가 이어서 진행합니다. 접수 응답은 학습 시작이나 완료를 뜻하지 않습니다.
- 선택한 프로젝트는 `/projects/{projectId}/auto-training`의 동일 GET/POST를 사용합니다.
- 응답은 `projectId`, `enabled`, `policyVersion`, `action`, `state`, `title`, `detail`, `steps`, `environment`, `recommendation`, `data`, `collection`, `blockers`를 포함합니다. `wait`는 실제 실행 조건 대기 또는 준비 중을 뜻하며, 학습 완료와 다릅니다.
- 빠른 접수 응답은 `accepted`, `requestId`와 현재 요청 상태를 포함합니다. 자료·환경의 자세한 상태는 후속 GET으로 조회합니다. 같은 키를 다시 보내도 준비 작업을 중복 실행하지 않으며, 중지한 요청을 재전송해도 자동으로 다시 켜지지 않습니다.
- `data.trainingBytes`와 `minTrainingBytes`는 학습 분할의 검사된 원문 크기입니다. 최소량에 못 미친 코퍼스는 자료 대기로 유지하며 토큰 생성 대기로 표시하지 않습니다. 일일 수집 한도에 도달하면 `collection.dailyLimit.resetAt`은 다음 한국시간 자정, `discoveryNextAttemptAt`은 수집 재개 예약을 epoch 초로 제공합니다. `nextAttemptAt`은 개별 출처를 포함한 가장 이른 확인 시점입니다. 기존 수집 한도는 유지합니다.
- 요청 처리 중 응답이나 타임아웃 뒤에는 GET으로 접수·진행 상태를 다시 확인합니다. 정상 상태 수신 시 일시적인 갱신 오류를 해제하고, 재시도에는 같은 멱등 키를 유지합니다.
- 중지는 `/projects/{projectId}/policy`에 PATCH `{ "enabled": false }`와 `If-Match: {policyVersion}`을 보냅니다. 출처를 모으는 중이라 core policy가 아직 꺼져 있어도 자동 요청을 중지할 수 있습니다.

모든 쓰기는 기존 세션 인증·CSRF와 멱등 키를 사용합니다. 같은 요청의 재시도는 같은 키를 유지합니다. 기존 금지·보관·삭제 출처와 문서의 품질 판정은 유지하며, 메모리·디스크·코퍼스 최소량·격리 검증 요건을 낮추지 않습니다. 준비 단계와 실행 작업은 중복 제어를 거치고, 실측 설정과 실제 학습 설정은 일치시킵니다.

학습 설정의 서버 주소는 RAG 서버가 호출할 실제 trainer 주소입니다. 브라우저는 동일 출처의 `/api/training/*` 게이트웨이를 호출하므로 NAS 내부 trainer에 직접 접근하지 않습니다. 학습 화면이 열려 있는 동안 정상 응답은 1초, 실패 후 재시도는 5초 간격입니다. 요청은 중첩하지 않습니다. 상태 변경에는 현재 세션의 CSRF 토큰을 전달합니다.

## 상태

`GET {학습 서버}/api/training/status`

작업이 없을 때도 `job: null`을 명시해야 합니다.

```json
{
  "job": null,
  "runs": [],
  "readySamples": 0,
  "canStart": false
}
```

실제 학습 작업이 존재할 때의 응답 형식:

```json
{
  "job": {
    "id": "actual-job-id",
    "state": "running",
    "completedSteps": 120,
    "totalSteps": 1000,
    "epoch": 0.36,
    "loss": 1.2345,
    "updatedAt": "2026-11-06T05:00:00Z",
    "capabilities": { "pause": true, "resume": false, "cancel": true }
  },
  "runs": [],
  "readySamples": 500,
  "canStart": false
}
```

- `completedSteps`와 `totalSteps`는 학습 엔진의 실제 옵티마이저 스텝 기준 정수입니다. 타이머나 예상 시간으로 생성하지 않습니다.
- 진행률은 `completedSteps / totalSteps * 100`입니다. 표시값은 소수점 한 자리에서 버림하여 미완료 작업이 반올림으로 100%가 되지 않도록 합니다.
- 전체 스텝이 아직 알려지지 않았다면 두 스텝 값을 0으로 보내며, 화면에는 ‘계산 대기’를 표시합니다. 작업 자체가 없고 미연결인 초기 상태는 0%입니다.
- `state`: `preparing`, `running`, `paused`, `validating`, `completed`, `cancelled`, `failed`.
- 100%에 도달해도 프런트엔드가 완료로 바꾸지 않습니다. 서버가 `completed`를 보고해야 완료입니다. `completed` 응답은 전체 스텝이 양수이며 완료 스텝과 같아야 합니다.
- `loss`, `epoch`, `updatedAt`은 선택 사항입니다. 누락된 지표는 계산하거나 생성하지 않습니다.
- 연결 오류·타임아웃·잘못된 응답에서는 마지막 정상 수치를 유지하고 ‘갱신 중단’을 표시합니다. 서버 변경·연결 해제 시에는 기존 서버 상태를 초기화합니다.
- `canStart: true`와 양수 `readySamples`가 확인되어야 시작 버튼을 활성화합니다. 파일의 로컬 메타데이터를 학습 가능 데이터로 간주하지 않습니다.
- `runs`는 서버의 실제 종료 기록입니다. 항목 형식은 `{ "id": "actual-job-id", "state": "completed", "finishedAt": "2026-11-06T05:10:00Z" }`이며 종료 상태만 허용합니다. 기존 시연 기록은 읽지 않습니다.

## 시작 및 제어

`POST /api/training/jobs`

```json
{
  "model": "configured-model-name",
  "epochs": 3,
  "batchSize": 1,
  "learningRate": 0.0002
}
```

`POST /api/training/jobs/{id}/pause`

`POST /api/training/jobs/{id}/resume`

`POST /api/training/jobs/{id}/cancel`

성공 시 2xx를 반환합니다. 프런트엔드는 응답 직후 상태 API를 다시 조회합니다. 버튼 클릭만으로 학습 중·중단·완료 상태를 만들지 않습니다. 각 제어는 서버가 보낸 capability가 true인 경우에만 활성화합니다. POST는 자동 재전송하지 않습니다. 요청 결과를 확인하지 못했다면 상태를 재조회하고 오류를 표시합니다.

학습 데이터 선정, 모델 설치, 실제 훈련, 체크포인트 저장 및 제어 구현은 학습 서버가 담당합니다. 이 변경에는 해당 백엔드를 포함하지 않습니다.
