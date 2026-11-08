"""Persistent, fail-closed orchestration from an empty foundation registry."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
import threading
import time

from LLM.core import Fault, digest, now, require, uid

KST = timezone(timedelta(hours=9))
ACTIVE = {"queued", "running", "pause_requested", "cancelling"}
PREPARING = frozenset(('preparing_corpus', 'tokenizer_training', 'initializing', 'running', 'resuming',
                      'pretraining', 'preparing_instruction_data', 'instruction_training'))


class FoundationScheduler:
    def __init__(self, control):
        self.control, self.store = control, control.store
        self.last_check = {}
        self._check_lock = threading.RLock()
        self._check_owner = 'foundation-scheduler-' + uid()

    def _status(self, project, state, reasons, **extra):
        delay = project.get('policy', {}).get('checkIntervalSeconds', 300)
        if self.store.get('foundation_auto_training', project['id'], {}).get('enabled') is True:
            delay = min(delay, 30)
        if state in PREPARING:
            delay = min(delay, 5)
        value = {"projectId": project["id"], "state": state, "reasonCodes": reasons,
                 "checkedAt": now(), "nextAttemptAt": (datetime.now(timezone.utc)+timedelta(seconds=delay)).isoformat(), **extra}
        return self.store.put("foundation_scheduler", value, project["id"])

    def _foundation(self):
        service = getattr(self.control, "foundation", None)
        require(service is not None, "FOUNDATION_CONTROLLER_REQUIRED", status=503)
        return service

    def _start_training(self, project, body, stage):
        foundation = self._foundation()
        if not body.get('config'):
            from .resources import save_recommended_training_config
            recommendation = save_recommended_training_config(self.store, project, stage)
            require(not recommendation['reasonCodes'], recommendation['reasonCodes'][0] if recommendation['reasonCodes'] else 'RESOURCE_UNAVAILABLE', status=503)
            body = {**body, 'config': recommendation['config']}
        try:
            result = foundation.start_training(project["id"], body, stage=stage)
            self.store.put("foundation_automation_job", {"projectId": project["id"], "jobId": result["id"]}, result["id"])
            return result
        except Fault as exc:
            if exc.code != "CALIBRATION_REQUIRED":
                raise
            calibrate = getattr(foundation, "calibrate", None)
            require(callable(calibrate), "CALIBRATION_REQUIRED", status=409)
            return calibrate(project["id"], {**body, "stage": stage})

    def _pause_active(self, project, reason):
        foundation = self._foundation()
        paused = []
        for job in self._jobs(project):
            if job["type"] not in ("foundation_pretrain", "foundation_cpt", "foundation_sft"):
                continue
            if reason in ("AUTOMATION_DISABLED", "OUTSIDE_TRAINING_WINDOW") and not self.store.get("foundation_automation_job", job["id"], {}):
                continue
            if job["state"] == "running":
                foundation.command(job["id"], "pause")
                self.store.put("foundation_run", {"pauseReason": reason}, job["id"])
                paused.append(job["id"])
            elif job["state"] == "queued":
                with self.store.tx():
                    current = self.store.job(job["id"])
                    if current["state"] == "queued":
                        self.store.db.execute("UPDATE jobs SET state='paused' WHERE id=? AND state='queued'", (job["id"],))
                        self.store.put("foundation_run", {"state": "paused", "deferredBeforeStart": True,
                                       "pauseReason": reason}, job["id"])
                        paused.append(job["id"])
        return paused

    def _daily_seconds(self, project):
        from .resources import used_today
        return used_today(self.store, project["id"])

    def _jobs(self, project):
        output = []
        for row in self.store.rows("SELECT id FROM jobs WHERE state IN ('queued','running','pause_requested','cancelling')"):
            job = self.store.job(row["id"])
            if job["type"].startswith("foundation_") and job.get("payload", {}).get("projectId") == project["id"]:
                output.append(job)
        return output

    def _window(self, policy):
        if policy.get("automation", "always_on_idle") != "night_window":
            return True
        window = policy.get("trainingWindow", {"start": "02:00", "end": "06:00"})
        clock = datetime.now(KST).strftime("%H:%M")
        return window["start"] <= clock < window["end"] if window["start"] < window["end"] else clock >= window["start"] or clock < window["end"]

    def _new_tokens(self, project, corpus):
        from .tokenizer import load
        from .corpus import rows
        tokenizer = load(self.store, project["tokenizerId"])
        intervals = {}
        for value in self.store.list("foundation_consumption"):
            if value.get("projectId") != project["id"] or value.get("tokenizerHash") != tokenizer.tokenizer_hash or value.get("targetNamespace", "causal_lm") != "causal_lm":
                continue
            for checksum, ranges in value.get("consumedTargetRanges", {}).items():
                intervals.setdefault(checksum, []).extend(ranges)
        count = 0
        for row in rows(self.store, corpus, "train", purpose=None):
            length = len(tokenizer.encode_ordinary(row["text"]))+1
            merged = []
            for begin, end in sorted(intervals.get(row["hash"], [])):
                begin, end = max(0, begin), min(length, end)
                if end <= begin:
                    continue
                if merged and begin <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], end)
                else:
                    merged.append([begin, end])
            count += length-sum(end-begin for begin, end in merged)
        return count

    def check(self, project_id=None, manual=False):
        with self._check_lock:
            return self._check(project_id, manual)

    def _check(self, project_id=None, manual=False):
        projects = [self.store.get("foundation_project", project_id)] if project_id else self.store.list("foundation_project")
        results = []
        for project in projects:
            claimed = False
            try:
                project = self._foundation().project(project["id"])
                interval = project.get("policy", {}).get("checkIntervalSeconds", 300)
                if self.store.get('foundation_auto_training', project['id'], {}).get('enabled') is True:
                    interval = min(interval, 30)
                previous = self.store.get('foundation_scheduler', project['id'], {})
                if self._jobs(project) or previous.get('state') in PREPARING:
                    interval = min(interval, 5)
                if not manual and time.monotonic()-self.last_check.get(project["id"], -1e20) < interval:
                    results.append(self.store.get("foundation_scheduler", project["id"], {"projectId": project["id"], "state": "waiting"}))
                    continue
                # API and isolated worker Controllers share this database but
                # have different Python locks. Claim a short-lived scheduling
                # slot before inspecting sources or queueing a prerequisite.
                with self.store.tx():
                    slot = self.store.get('foundation_scheduler_claim', project['id'], {})
                    if slot.get('owner') and slot.get('until', 0) > time.time():
                        results.append(self.store.get('foundation_scheduler', project['id'],
                            {'projectId': project['id'], 'state': 'waiting', 'reasonCodes': []}))
                        continue
                    self.store.put('foundation_scheduler_claim', {'owner': self._check_owner,
                        'until': time.time() + 300}, project['id'])
                    claimed = True
                self.last_check[project["id"]] = time.monotonic()
                results.append(self._check_project(project, manual))
            except Fault as exc:
                results.append(self._status(project, "held", [exc.code]))
            finally:
                if claimed:
                    with self.store.tx():
                        if self.store.get('foundation_scheduler_claim', project['id'], {}).get('owner') == self._check_owner:
                            self.store.db.execute("DELETE FROM records WHERE kind='foundation_scheduler_claim' AND id=?", (project['id'],))
        return results[0] if project_id else results

    def _refresh(self, project, corpus):
        """Inspect version hashes before deciding that an old held corpus is final."""
        bridge = getattr(self.control, "bridge", None)
        if bridge is None:
            require(project.get("testOnly") and corpus, "RAG_POLICY_GATEWAY_REQUIRED", status=503)
            return None
        payload = bridge.foundation_corpus(project["id"], None, project["policy"]["allowedSourceIds"])
        records = payload.get("records", [])
        signature = digest([payload.get("policyVersion"), sorted((row.get("versionId", row.get("id")), digest(row.get("text", "").encode()),
                           row.get("trainingPolicy")) for row in records)])
        old = self.store.get("foundation_scheduler", project["id"], {})
        if signature != old.get("sourceSignature"):
            if not records:
                return self._status(project, "waiting_corpus", ["INSUFFICIENT_CORPUS"], sourceSignature=signature)
            job = self._foundation().create_corpus(project["id"], {"sourceVersionIds": [row.get("versionId", row.get("id")) for row in records]})
            return self._status(project, "preparing_corpus", [], job=job, sourceSignature=signature)
        return None

    def _prepare_request(self, project):
        """Consume one explicit Start generation; later user pauses stay paused."""
        from .autopilot import AutoTraining
        automatic = self.store.get('foundation_auto_training', project['id'], {})
        request_id = automatic.get('requestId')
        if automatic.get('enabled') is not True or not request_id \
                or automatic.get('preparationState') == 'completed':
            return
        if automatic.get('preparationState') == 'failed':
            error = automatic.get('preparationError') or {}
            raise Fault(error.get('code', 'AUTOMATIC_PREPARATION_FAILED'), error.get('message'), status=409)
        helper = AutoTraining(self.control)
        def current():
            saved = self.store.get('foundation_auto_training', project['id'], {})
            policy = self.store.get('foundation_policy', project['id'], {})
            return saved.get('enabled') is True and saved.get('requestId') == request_id and policy.get('automation') != 'manual'
        try:
            if automatic.get('consent') is not None and automatic.get('consentPreparedRequestId') != request_id:
                if not current():return
                helper._approve(automatic['consent'])
                with self.store.tx():
                    if not current():return
                    self.store.put('foundation_auto_training', {'consentPreparedRequestId': request_id}, project['id'])
            if not current():return
            if self.control.bridge is not None:self.control.bridge.refresh_policy()
            if not current():return
            # Reconcile allowed sources before making a paused training job
            # claimable again. The worker must see the selected policy first.
            from .autopilot_sources import refresh_project_sources
            project = refresh_project_sources(self.control, self._foundation().project(project['id']))
            if not current():return True
            from .resources import save_recommended_training_config
            save_recommended_training_config(self.store, self._foundation().project(project['id']), 'pretrain')
            candidates = automatic.get('resumeCandidates', {})
            paused_runs = [row for row in helper._paused_runs(project['id'], self._foundation()._pending(project['id']))
                if row['jobId'] in candidates and row['runVersion'] == candidates[row['jobId']]
                and row['jobId'] not in automatic.get('resumedJobIds', [])]
            if paused_runs and (not project['policy'].get('enabled') or not project['policy'].get('allowedSourceIds')):
                return True
            for paused in paused_runs:
                require(paused['resumable'], 'CHECKPOINT_REQUIRED', '이어서 학습할 유효한 체크포인트가 없습니다.', status=409)
                from .checkpoint import verify
                try:verify(self.store, paused['checkpoint'])
                except Fault as exc:
                    raise Fault(exc.code, '저장된 체크포인트 검증에 실패해 학습을 재개하지 않았습니다.', status=409) from exc
                except (OSError, ValueError, TypeError) as exc:
                    raise Fault('CHECKPOINT_INVALID', '저장된 체크포인트를 읽을 수 없어 학습을 재개하지 않았습니다.', status=409) from exc
                with self.store.tx():
                    if not current():return
                    # A user can pause again while a checkpoint is being read.
                    # Only the exact paused version authorized by Start resumes.
                    run = self.store.get('foundation_run', paused['jobId'], {})
                    if run.get('version') != candidates[paused['jobId']]:continue
                    self._foundation().command(paused['jobId'], 'resume')
                    requested = self.store.get('foundation_auto_training', project['id'])
                    self.store.put('foundation_auto_training', {'resumedJobIds':
                        sorted(set(requested.get('resumedJobIds', [])) | {paused['jobId']})}, project['id'])
            with self.store.tx():
                if current():
                    self.store.put('foundation_auto_training', {'preparationState': 'completed',
                        'preparationError': None, 'preparedAt': now()}, project['id'])
            return True
        except Fault as exc:
            with self.store.tx():
                if current():
                    self.store.put('foundation_auto_training', {'preparationState': 'failed',
                        'preparationError': {'code': exc.code, 'message': exc.message}}, project['id'])
            raise

    def _check_project(self, project, manual):
        foundation = self._foundation()
        policy = project.get("policy", {})
        if self.store.get("setting", "deploymentMaintenance", {}).get("enabled"):
            self._pause_active(project, "DEPLOYMENT_MAINTENANCE")
            return self._status(project, "waiting_maintenance", ["DEPLOYMENT_MAINTENANCE"])
        mode = policy.get("automation", project.get("automation", "always_on_idle"))
        if project.get("state") == "revoked" or project.get("stage") == "revoked":
            return self._status(project, "revoked", ["POLICY_REVOKED"])
        automatic = self.store.get('foundation_auto_training', project['id'], {})
        if automatic.get('enabled') is True:
            sources_prepared = self._prepare_request(project)
            project = foundation.project(project['id'])
            from .autopilot_sources import refresh_project_sources
            if not sources_prepared:project = refresh_project_sources(self.control, project)
            policy = project.get('policy', {})
            mode = policy.get('automation', project.get('automation', 'always_on_idle'))
            automatic = self.store.get('foundation_auto_training', project['id'], {})
            if automatic.get('enabled') is True and mode != 'manual' and not policy.get('enabled') \
                    and not policy.get('allowedSourceIds'):
                prepared = automatic.get('sourcePreparation', {})
                return self._status(project, 'waiting_corpus', prepared.get('reasonCodes') or ['INSUFFICIENT_CORPUS'])
        if not manual and (not policy.get("enabled", False) or mode == "manual"):
            self._pause_active(project, "AUTOMATION_DISABLED")
            return self._status(project, "paused", ["AUTOMATION_DISABLED"])
        if not policy.get("allowedSourceIds"):
            return self._status(project, "waiting_corpus", ["CORPUS_PERMISSION_REQUIRED"])
        if not manual and not self._window({**policy, "automation": mode}):
            self._pause_active(project, "OUTSIDE_TRAINING_WINDOW")
            return self._status(project, "waiting_window", ["OUTSIDE_TRAINING_WINDOW"])
        daily = self._daily_seconds(project)
        if daily >= policy.get("dailyActiveSeconds", 14400):
            self._pause_active(project, "DAILY_BUDGET_EXHAUSTED")
            return self._status(project, "waiting_budget", ["DAILY_BUDGET_EXHAUSTED"], activeSecondsToday=daily)
        if self.store.get('foundation_growth_policy',project['id'],{}).get('enabled'):
            from .growth import advance as advance_growth
            candidate=advance_growth(self.control,project['id'])
            if candidate.get('blocksDefault'):
                return self._status(project,'growth_experiment',candidate.get('reasonCodes',[]),growthState=candidate['state'])
        curriculum=self.store.get('foundation_curriculum',project['id'],{})
        if curriculum.get('enabled') and curriculum.get('state') not in ('completed','disabled'):
            from .curriculum import Curriculum
            curriculum=Curriculum(self.control).advance(project['id'])
            return self._status(project,'curriculum',curriculum.get('reasonCodes',[]),curriculumState=curriculum.get('state'))
        jobs = self._jobs(project)
        if jobs:
            return self._status(project, "running", [], jobIds=[job["id"] for job in jobs])
        run = self.store.get("foundation_run", project.get("latestRunId", ""), {})
        if run and run.get("state") in ("paused", "completed"):
            elapsed = (datetime.now(timezone.utc)-datetime.fromisoformat(run["updatedAt"].replace("Z", "+00:00"))).total_seconds()
            if not manual and elapsed < policy.get("restSeconds", 120):
                return self._status(project, "resting", ["SLICE_REST_REQUIRED"])
        if run.get("state") in ("paused", "waiting", "waiting_resource", "waiting_budget"):
            if run.get("state") == "paused" and not run.get("deferredBeforeStart") and run.get("terminationReason") != "yielded" and run.get("pauseReason") not in (
                    "OUTSIDE_TRAINING_WINDOW", "DAILY_BUDGET_EXHAUSTED", "DEPLOYMENT_MAINTENANCE", "RESOURCE_PRESSURE", "AUTOMATION_DISABLED",
                    "FOREGROUND_REQUEST", "SLICE_LIMIT", "WORKER_SHUTDOWN"):
                return self._status(project, "paused", ["USER_PAUSED"])
            if run.get("deferredBeforeStart"):
                with self.store.tx():
                    self.store.db.execute("UPDATE jobs SET state='queued',owner=NULL,lease=0 WHERE id=? AND state='paused'", (run["id"],))
                    self.store.put("foundation_run", {"state": "queued", "deferredBeforeStart": False, "pauseReason": None}, run["id"])
                return self._status(project, "resuming", [], job=self.store.job(run["id"]))
            action = getattr(foundation, "control_run", None)
            require(callable(action), "CHECKPOINT_RESUME_CONTROLLER_REQUIRED", status=503)
            result = action(run["id"], "resume")
            self.store.put("foundation_run", {"pauseReason": None}, run["id"])
            return self._status(project, "resuming", [], job=result)
        failures = sorted((run for run in self.store.list("foundation_run") if run.get("projectId") == project["id"]
                           and run.get("state") in ("completed", "failed", "held")), key=lambda value: value.get("updatedAt", ""), reverse=True)
        if len(failures) >= 3 and all(run.get("state") == "failed" for run in failures[:3]):
            return self._status(project, "paused", ["CONSECUTIVE_FAILURE_CIRCUIT_OPEN"])
        corpus = self.store.get("foundation_corpus", project.get("activeCorpusId", ""), {})
        refreshed = self._refresh(project, corpus)
        if refreshed:
            return refreshed
        if not corpus:
            job = foundation.create_corpus(project["id"], {})
            return self._status(project, "preparing_corpus", [], job=job)
        if corpus.get("state") != "ready":
            return self._status(project, "waiting_corpus", corpus.get("reasonCodes", ["INSUFFICIENT_CORPUS"]))
        if not project.get("tokenizerId"):
            job = foundation.start_tokenizer(project["id"], {"corpusSnapshotId": corpus["id"]})
            return self._status(project, "tokenizer_training", [], job=job)
        if not project.get("genesisId"):
            job = foundation.initialize(project["id"], {"tokenizerArtifactId": project["tokenizerId"]})
            return self._status(project, "initializing", [], job=job)
        checkpoint_id = project.get("latestCheckpointId")
        if not checkpoint_id:
            job = self._start_training(project, {"corpusSnapshotId": corpus["id"],
                "parentCheckpointId": project["genesisId"]}, "pretrain")
            return self._status(project, "pretraining", [], job=job)
        evaluations = sorted((record for record in self.store.list("foundation_evaluation")
                              if record.get("projectId") == project["id"] and record.get("artifactId") == checkpoint_id),
                             key=lambda row: row.get("createdAt", ""), reverse=True)
        if run.get("stage") == "sft" and run.get("state") == "completed":
            return self._release_candidate(project, checkpoint_id, evaluations)
        qualified = any(row.get("kind") == "qualification" and row.get("gate", {}).get("passed") for row in evaluations)
        if not evaluations:
            job = foundation.evaluate({"projectId": project["id"], "artifactId": checkpoint_id,
                                       "corpusId": corpus["id"], "kind": "qualification"})
            return self._status(project, "evaluating", [], job=job)
        if qualified:
            graduations = [row for row in self.store.list("foundation_graduation") if row.get("checkpointId") == checkpoint_id]
            if not graduations:
                evaluation = next(row for row in evaluations if row.get("kind") == "qualification" and row.get("gate", {}).get("passed"))
                result = foundation.graduate({"projectId": project["id"], "checkpointId": checkpoint_id, "evaluationId": evaluation["id"]})
                return self._status(project, "foundation_qualified", [], graduation=result)
            sft_id = project.get("activeSftCorpusId")
            sft_corpus = self.store.get("foundation_corpus", sft_id, {}) if sft_id else {}
            if not sft_corpus or sft_corpus.get("state") != "ready":
                prepare = getattr(foundation, "prepare_sft_corpus", None)
                require(callable(prepare), "APPROVED_SFT_CORPUS_REQUIRED", status=409)
                prepared = prepare(project["id"])
                return self._status(project, "preparing_instruction_data", [], job=prepared)
            job = self._start_training(project, {"corpusSnapshotId": sft_id, "parentCheckpointId": checkpoint_id}, "sft")
            return self._status(project, "instruction_training", [], job=job)
        # Failed quality is not an infrastructure failure and cannot fabricate a release.
        new_tokens = self._new_tokens(project, corpus)
        initial_runs = [row for row in self.store.list("foundation_run") if row.get("projectId") == project["id"]
                        and row.get("stage") == "pretrain" and row.get("state") == "completed"]
        initial_continuation = len(initial_runs) < 3 and not any(row.get("gate", {}).get("passed") for row in
            self.store.list("foundation_evaluation") if row.get("projectId") == project["id"] and row.get("kind") == "qualification")
        if new_tokens < policy.get("minNewTargetTokens", 65536) and not initial_continuation:
            return self._status(project, "waiting_data", ["INSUFFICIENT_NEW_TRAIN_DATA"], newUniqueTokens=new_tokens)
        job = self._start_training(project, {"corpusSnapshotId": corpus["id"],
            "parentCheckpointId": checkpoint_id, "continueCorpus": initial_continuation}, "pretrain" if initial_continuation else "cpt")
        return self._status(project, "pretraining", [], job=job)

    def _release_candidate(self, project, checkpoint_id, evaluations):
        foundation = self._foundation()
        releases = [row for row in self.store.list("foundation_release") if row.get("checkpointId") == checkpoint_id]
        if releases:
            release = releases[-1]
            mirror = self.store.get("release", release["id"], {})
            if mirror.get("state") == "active" or release.get("state") == "active":
                corpus = self.store.get("foundation_corpus", project.get("activeCorpusId", ""), {})
                if corpus and corpus.get("state") == "ready":
                    fresh = self._new_tokens(project, corpus)
                    if fresh >= project["policy"].get("minNewTargetTokens", 65536):
                        result = self._start_training(project, {"corpusSnapshotId": corpus["id"],
                                                           "parentCheckpointId": checkpoint_id}, "cpt")
                        return self._status(project, "continued_pretraining", [], job=result, newUniqueTokens=fresh)
                return self._status(project, "completed", [], releaseId=release["id"])
            activate = getattr(foundation, "activate_release", None)
            require(callable(activate), "DEPLOYMENT_CONTROLLER_REQUIRED", status=503)
            result = activate(release["id"])
            return self._status(project, "deploying", [], result=result)
        release_evaluations = [row for row in evaluations if row.get("kind") == "release"]
        if release_evaluations:
            evaluation = release_evaluations[0]
            if not evaluation.get("productionCertified"):
                corpus = self.store.get("foundation_corpus", project.get("activeCorpusId", ""), {})
                fresh = self._new_tokens(project, corpus) if corpus.get("state") == "ready" else 0
                if fresh >= project["policy"].get("minNewTargetTokens", 65536):
                    result = self._start_training(project, {"corpusSnapshotId": corpus["id"],
                                                           "parentCheckpointId": checkpoint_id}, "cpt")
                    return self._status(project, "continued_pretraining", [], job=result, newUniqueTokens=fresh)
                spent = any(row.get("evaluationId") == evaluation["id"] for row in self.store.list("foundation_eval_shard"))
                reasons = evaluation.get("gate", {}).get("reasonCodes", [])
                verifier = getattr(self.control, "sandbox", None)
                if not spent and set(reasons) <= {"VERIFIER_UNAVAILABLE", "SANDBOX_PROBE_REQUIRED"} and reasons \
                        and verifier is not None and verifier.available():
                    result = foundation.evaluate({"projectId": project["id"], "artifactId": checkpoint_id,
                                                  "suiteId": evaluation["suiteId"]})
                    return self._status(project, "evaluating_release", [], job=result)
                return self._status(project, "held", evaluation.get("gate", {}).get("reasonCodes", ["QUALITY_UNCERTAIN"]))
            result = foundation.prepare_release({"projectId": project["id"], "checkpointId": checkpoint_id, "evaluationId": evaluation["id"]})
            return self._status(project, "release_ready", [], releaseId=result["release"]["id"])
        used = {row.get("suiteHash") for row in self.store.list("foundation_eval_shard")}
        suites = [row for row in self.store.list("foundation_suite") if row.get("kind") == "release"
                  and row.get("state") == "ready" and not row.get("fixture") and row.get("suiteHash") not in used]
        require(suites, "INSUFFICIENT_EVAL_COVERAGE", status=409)
        baselines = [row["artifactId"] for row in self.store.list("foundation_release")
                     if row.get("projectId") == project["id"] and
                     self.store.get("release", row["id"], row).get("state") == "active"]
        result = foundation.evaluate({"projectId": project["id"], "artifactId": checkpoint_id, "suiteId": suites[0]["id"],
                                      "baselineIds": list(dict.fromkeys(baselines))})
        return self._status(project, "evaluating_release", [], job=result)

    def advance(self, project_id=None):
        # The NAS shared-volume worker executes queued prerequisites, but only
        # the integrated controller has the RAG bridge needed to schedule them.
        # Do not let its supervisor acquire a scheduling slot and replace the
        # controller's real progress with RAG_POLICY_GATEWAY_REQUIRED.
        offline_worker = (os.environ.get('SEED_FOUNDATION_WORKER') == '1'
            and getattr(self.control, 'worker_enabled', False)
            and getattr(self.control, 'execution_scope', 'nas') == 'nas'
            and getattr(self.control, 'bridge', None) is None)
        if offline_worker:
            if project_id is not None:
                project = self.store.get('foundation_project', project_id)
                if project.get('testOnly') is not True:
                    return self.store.get('foundation_scheduler', project_id,
                        {'projectId': project_id, 'state': 'waiting', 'reasonCodes': []})
            else:
                return [self.check(project['id']) for project in self.store.list('foundation_project')
                        if project.get('testOnly') is True]
        return self.check(project_id)
