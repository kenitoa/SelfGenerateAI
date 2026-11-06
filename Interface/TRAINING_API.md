# 학습 상태 인터페이스 계약

실제 가중치 학습 엔진은 별도 범위입니다. RAG의 학습 게이트웨이가 아래 계약을 제공하며, 엔진 미설정 시 시작을 비활성화합니다. 예시 JSON은 문서용이며 화면에 주입하지 않습니다.

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
