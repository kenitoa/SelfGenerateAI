# SEED RAG — Interface ↔ RAG (Pre ↔ Post)

기존 Interface의 개요·작업·데이터·학습·설정 페이지와 디자인을 유지하고 실제 서버 상태·데이터를 연결합니다. `LLM` 폴더는 사용하거나 수정하지 않습니다. 모델 엔진은 설정한 외부 프로세스에 어댑터로 연결합니다.

## 역할

```text
Interface의 기존 페이지
       ↕ same-origin /api
RAG API + 인증 + 영속 작업 큐
       ↕
Pre: 원문 → 검증/격리 → 정제 → 청킹 → 색인 → 검색/근거
       ↕ 근거·버전·정책 재확인
Post: 컨텍스트 → 추론 어댑터 → 인용 검증 → 대화/피드백 → 승인 자료 export
```

`Pre`와 `Post`는 Python 모듈이며 별도 서버 두 개가 아닙니다. 하나의 RAG API가 두 파트와 공유 저장소를 조합합니다. 생성 답변을 Pre 원문으로 자동 재투입하지 않습니다.

## 로컬 실행

프로젝트 루트에서 실행합니다. Python 3.12 이상과 Node.js가 필요합니다.

```powershell
python -m venv RAG/.venv
RAG/.venv/Scripts/python.exe -m pip install -r RAG/requirements.lock.txt
RAG/.venv/Scripts/python.exe -m playwright install chromium
cd Interface
npm.cmd install
npm.cmd run build
cd ..
RAG/.venv/Scripts/python.exe -m RAG serve --local --host 127.0.0.1 --port 8000
```

접속: **http://127.0.0.1:8000**. RAG가 현재 Interface 빌드를 직접 제공합니다. 재실행은 `start-local.ps1` 또는 마지막 명령을 사용합니다. PowerShell 실행 정책이 스크립트를 막으면 Python 명령을 직접 실행하면 됩니다. 실행 정책을 변경할 필요는 없습니다.

Interface 개발 서버가 필요하면 별도 터미널에서 `Interface`의 `npm.cmd run dev -- --port 5173`을 실행합니다. `http://127.0.0.1:5173`의 `/api` 요청은 8000으로 전달됩니다. 두 개발 서버는 loopback에만 바인딩합니다.

`--local`은 loopback에서만 허용하는 개발용 세션입니다. 관리자 비밀번호가 설정되면 개발 자동 세션도 사용하지 않습니다. 실제 NAS/LAN 운영에서는 아래 관리자 초기화와 일반 serve를 사용합니다.

```powershell
RAG/.venv/Scripts/python.exe -m RAG init-admin
RAG/.venv/Scripts/python.exe -m RAG serve --host 0.0.0.0 --port 8000
```

비밀번호는 대화형 입력으로만 받고 파일·명령행·로그에 출력하지 않습니다. NAS에서는 HTTPS reverse proxy를 구성합니다. UI에서 관리자 로그인을 제공하며 CSRF·same-origin·HttpOnly 세션을 적용합니다.

## 사용 흐름

1. **데이터 → 파일 추가**: JSON/JSONL뿐 아니라 Markdown·텍스트·Python·JS/TS·SQL 등 텍스트 코드를 실제 서버에 저장합니다.
2. 처리 결과에서 **문서·청크·검토 사유**를 확인합니다. 비밀정보 의심 자료는 격리되며 검색 승인을 눌러 우회할 수 없습니다.
3. **작업**에서 기존 작성·검토·리팩토링·테스트 모드를 선택합니다. RAG 근거를 검색하며 답변의 근거에서 저장 원문 버전을 확인할 수 있습니다.
4. **소스 목록**에서는 수집 실행·중지·재개를 조작합니다. 수집 대상 제거는 자료를 보존합니다. 문서 삭제와 검색 제외는 별도 동작입니다.
5. **설정 → RAG 관리**에서 연결, 임베딩, 재색인, 예약, 작업 재시도·취소, 백업, 학습 자료를 관리합니다. 새로운 최상위 페이지를 만들지 않습니다.

NAS/로컬 폴더 자료 수집은 제거했습니다. 기존 폴더 소스는 시작 시 수집·검색·학습 이용과 예약을 비활성화하며 원문은 보존합니다. 공개 웹의 새 코드 자료는 GitHub API·Sitemap·RSS/Atom·Common Crawl과 링크 탐색으로 자동 발견합니다. 서버가 실행 중이면 브라우저를 닫아도 작동합니다. 기본값은 활성, 60분 간격, 실행당 6페이지·하루 60페이지·깊이 2입니다. **설정 → RAG 관리 → 코드 자동 수집**에서 중지·재개·설정·즉시 수집·실제 결과를 확인합니다. Playwright는 정적 HTML에 코드가 없고 동적 앱 신호가 있을 때만 실행당 1페이지 사용합니다. Docker 빌드는 Chromium headless shell도 설치합니다.

수집은 검색 지식을 자동 갱신합니다. 가중치 훈련을 실행하지 않으며 수집 자료의 trainingPolicy는 unknown입니다. 라이선스·권한 검토 후 기존 학습 export와 외부 trainer 계약을 사용합니다. [10개 기술과 상세 흐름](../docs/RAG_Crawling_System.md)을 참고하세요. GitHub 공개 검색은 토큰 없이 시작하며 한도 초과 시 쉬었다 재시도합니다. 필요하면 서버의 SEED_GITHUB_TOKEN_FILE에 토큰 파일 경로를 지정합니다.

**설정 → RAG 관리 → 검색 품질**에서 고정 36문항 평가, 검색 방식 비교, 기존 자료 분석·색인 갱신, 다국어 E5 INT8 임베딩 설치를 실행합니다. 이 로컬 작업공간에는 모델을 설치하고 실제 하이브리드 색인을 게시했습니다. 다른 NAS의 빈 데이터 볼륨에는 자동 복사되지 않으며 설치 버튼 또는 `python -m RAG install-embedding`이 필요합니다. 다운로드는 명시적 실행 때만 이뤄집니다. 임베딩 사용 시 재색인으로 tokenizer 예산과 벡터를 갱신합니다. [1~6 구현과 실측](../docs/RAG_Quality_Enhancements.md)을 참고하세요.

코드 구조는 구문 트리로 분석하고 같은 저장소의 호출/import 후보를 함께 검색합니다. 자동 수집은 저장소 기본 브랜치·완전한 tree의 삭제·HTTP 304/404/410을 추적하며, 기존 자동 수집 설정에서 사이트·언어·제공자·저장소 분포와 언어별 상대 가중치를 확인합니다. 품질 점수는 정적 신호이며 테스트를 실행한 결과가 아닙니다.

현재 브라우저에만 있던 예전 파일 메타데이터는 원문이 아니므로 다시 업로드해야 합니다. 데모 대화도 실제 AI 결과나 학습 자료로 자동 이관하지 않습니다.

## 실제 동작과 엔진 의존 기능

| 기능 | 동작 조건 |
| --- | --- |
| 원문·출처·버전·작업·대화·설정 저장 | 기본 제공, SQLite·로컬 파일 |
| JSON/JSONL 검증·청킹·비밀정보 격리·승인 | 기본 제공 |
| 키워드 검색·식별자 검색·근거 원문 | 기본 제공, FTS5 |
| 벡터·하이브리드 검색 | 로컬 ONNX 임베딩 자료 설정 시 활성 |
| 실제 코드 생성 | 설치된 Ollama 모델 + 일치하는 tokenizer/profile 필요 |
| URL 수집 | 선택한 공개 URL, bounded fetch·DNS/IP/redirect 검사 |
| 저장소 수집 | 공개 GitHub 저장소의 텍스트 경로, hooks/설치/실행 안 함 |
| NAS/로컬 디렉터리 수집 | 제거됨, 기존 소스 비활성·신규 등록 거절 |
| 예약·작업 복구·백업·평가 | API 프로세스 상주 중 영속 큐·스케줄러로 실행 |
| 학습 데이터 export | 출처별 학습 허용 + 사람 검토 + 독립 계보 그룹 필요 |
| 실제 가중치 학습·모델 배포 | 이 저장소에 구현하지 않음. 기존 trainer API로 외부 엔진 연동 |

모델이 없으면 **근거 검색 결과와 미연결 사유만** 제공합니다. 고정 예시 코드를 생성 결과로 표시하거나 진행률을 타이머로 늘리지 않습니다. `readySamples`에는 RAG 청크 수를 넣지 않습니다.

## 임베딩: 선택 설정

자동 모델 다운로드는 없습니다. CPU 런타임이 필요하면 `requirements-embedding.txt`를 설치하고, 검증한 ONNX 모델과 tokenizer를 읽기 전용 디렉터리에 준비합니다.

```text
<embedding-directory>/
  model.onnx
  tokenizer.json
  profile.json
```

```json
{
  "modelId": "intfloat/multilingual-e5-small",
  "modelRevision": "실제 고정 commit SHA",
  "tokenizerRevision": "실제 고정 commit SHA",
  "dimension": 384,
  "quantization": "fp32 또는 검증한 int8",
  "checksums": {
    "model.onnx": "실제 SHA256",
    "tokenizer.json": "실제 SHA256"
  }
}
```

설정의 `embedding.directory`에 해당 폴더를 저장하고 활성화한 다음 재색인합니다. prefix·mean pooling·L2 정규화를 적용합니다. 프로필이 바뀌면 새 generation을 만들며 기존 벡터와 섞지 않습니다. 런타임이나 자료가 없으면 키워드 모드로 유지합니다. 이 후보는 코드 전용 모델이 아니며 실제 코퍼스 평가가 필요합니다.

## 실제 생성 연결

추론 주소와 모델명을 저장한 다음 설치 모델 목록·연결을 확인합니다. 주소 저장이나 모델 후보 선택은 다운로드·설치를 실행하지 않습니다.

정확한 전체 프롬프트 길이를 측정하기 위해 초기 생성 어댑터는 검증한 Qwen 계열 **ChatML raw prompt**와 Ollama `/api/generate`를 사용합니다. 설계 초안의 `/api/chat`에서 이 구현 방식으로 정밀화했습니다. 숨겨진 서버 chat template로 인한 토큰 계산 오차를 피하려는 선택이며 화면 계약은 같습니다.

생성 tokenizer 옆에 다음 `generation-profile.json`을 둡니다.

```json
{
  "model": "qwen2.5-coder:0.5b",
  "modelRevision": "설치 모델 목록에서 확인한 실제 digest",
  "tokenizerRevision": "실제 원본 tokenizer commit SHA",
  "tokenizerSha256": "tokenizer.json의 실제 SHA256",
  "templateFormat": "chatml",
  "contextLength": 4096,
  "addSpecialTokens": false
}
```

`tokenizerPath`에 tokenizer.json 경로를 저장합니다. 모델 digest·프로필·체크섬이 맞아야 생성합니다. 프로필 없이 길이를 추정해 생성하지 않습니다. 다른 템플릿·모델 계열은 별도 검증된 어댑터를 추가해야 합니다.

## 저장·복구

기본 데이터 위치는 `RAG/data`이며 `--data-dir` 또는 `SEED_DATA_DIR`로 변경합니다. SQLite는 APSW에 포함된 수정 버전을 사용합니다. 현재 개발 Python의 기본 SQLite는 3.50.4였으므로 그대로 WAL에 쓰지 않았습니다. DB·원문을 SMB/NFS 공유 DB처럼 다른 PC에서 직접 열지 않습니다.

스키마의 도메인 항목은 `records(kind,id,body,created_at,updated_at)`에 JSON 문서로 저장하고, 작업·멱등 키·설정·활동·FTS·색인 membership은 전용 SQL 테이블로 저장합니다. 기능명세의 논리 데이터 모델을 단일 사용자 MVP에 맞춰 이 방식으로 매핑했습니다. DB migration 버전도 기록합니다.

백업은 Online Backup API와 불변 원문·색인·export의 해시 manifest를 함께 저장합니다. 복원은 빈 디렉터리로만 수행합니다.

```powershell
RAG/.venv/Scripts/python.exe -m RAG restore --backup-id <실제 백업 ID> --destination <빈 복원 디렉터리>
```

복원 검증 후 해당 디렉터리를 `--data-dir`로 사용합니다. 서비스가 사용하는 데이터 디렉터리에 덮어쓰지 않습니다. 백업의 관리자 정보·세션도 민감한 데이터이므로 백업 위치를 보호해야 합니다.

## NAS 배포 자료

루트에서 Interface를 빌드한 뒤:

```text
docker compose -f RAG/compose.yaml build
docker compose -f RAG/compose.yaml run --rm seed python -m RAG init-admin
docker compose -f RAG/compose.yaml up -d
```

bind mount의 소유권을 container UID 10001에 맞춥니다. 기본 compose는 RAG 컨테이너 1 GiB·CPU 1개 상한을 둡니다. UI의 메모리 설정과 컨테이너 실제 상한은 다른 값이며 운영자가 일치 여부를 검증해야 합니다. 생성 엔진 메모리는 별도입니다. 실제 NAS 배포·성능 측정은 이 로컬 구현 과정에서 수행하지 않았습니다.

## 검증

```powershell
RAG/.venv/Scripts/python.exe -m pytest RAG/tests -q
cd Interface
node --test src/rag-api.test.js src/training-api.test.js
npm.cmd run build
```

검색 평가 API는 자체 질문·정답 근거 ID를 받아 실제 trace·Recall@5·지연을 산출합니다. holdout으로 등록한 계보는 학습 내보내기에서 제외합니다. 문서의 검색 지연·Recall 목표를 이미 달성한 수치로 표시하지 않습니다.
