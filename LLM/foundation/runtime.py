"""Single-artifact, structured-message CPU inference for locally trained models."""
from __future__ import annotations

import json
import math
import threading
import time

from LLM.core import Fault, digest, now, require


def _find(store, identity, kinds):
    for kind in kinds:
        value = store.get(kind, identity, {})
        if value:
            return value
    raise Fault("FOUNDATION_ARTIFACT_NOT_FOUND", status=404)


class FoundationRuntime:
    def __init__(self, store, artifacts=None):
        self.store, self.artifacts = store, artifacts
        self.lock = threading.RLock()
        self.loaded_id = None
        self.model = None
        self.tokenizer = None
        self.foreground_waiters = 0
        self.policy_validator = None

    def artifact(self, identity):
        value = _find(self.store, identity, ("foundation_checkpoint", "foundation_genesis", "artifact"))
        require(value.get("state") not in ("revoked", "quarantined", "failed", "archived"), "POLICY_REVOKED", status=403)
        return value

    def load(self, identity, *, purpose="evaluation"):
        """Private evaluation may load Genesis; serving must resolve a certificate first."""
        from .architecture import load_model
        from .tokenizer import load as load_tokenizer
        artifact = self.artifact(identity)
        require(purpose in ("evaluation", "serve", "distillation_train"), "INVALID_INFERENCE_PURPOSE")
        conformance=None
        if artifact.get('format')=='gguf':
            from .gguf_runtime import verify_conformance
            conformance,gguf_path=verify_conformance(self.store,artifact,self.artifacts)
        if self.loaded_id != identity:
            previous=self.model
            if getattr(previous,'backend',None)=='gguf':previous.close()
            self.model = self.tokenizer = None
            self.loaded_id = None
            import gc
            gc.collect()
            tokenizer_id = artifact.get("tokenizerId") or artifact.get("tokenizerArtifactId")
            require(tokenizer_id, "TOKENIZER_REQUIRED", status=409)
            tokenizer = load_tokenizer(self.store, tokenizer_id)
            require(not artifact.get("tokenizerHash") or artifact["tokenizerHash"] == tokenizer.tokenizer_hash,
                    "TOKENIZER_HASH_MISMATCH", status=409)
            if self.store.get("artifact", identity, {}):
                require(self.artifacts is not None, "MODEL_REGISTRY_REQUIRED", status=503)
                root = self.artifacts.verify(artifact)
                if artifact.get('format')=='gguf':
                    from .gguf_runtime import GGUFModel
                    model=GGUFModel(gguf_path,conformance['actualContextTokens'],tokenizer,conformance)
                else:
                    import torch
                    from transformers import AutoModelForCausalLM
                    model = AutoModelForCausalLM.from_pretrained(root, local_files_only=True, trust_remote_code=False,
                                                               dtype=torch.float32, attn_implementation="eager")
                    require(sum(value.numel() for value in model.parameters()) == artifact["parameterCount"], "MODEL_PARAMETER_COUNT_MISMATCH")
                    require(all(bool(torch.isfinite(value).all()) for value in model.parameters()), "NONFINITE_INFERENCE")
            else:
                model = load_model(self.store, artifact)
            model.eval()
            self.model, self.tokenizer, self.loaded_id = model, tokenizer, identity
        return self.model, self.tokenizer

    def expire(self, force=False):
        if not self.lock.acquire(blocking=False):
            return False
        try:
            if getattr(self.model,'backend',None)=='gguf':self.model.close()
            self.model = self.tokenizer = self.loaded_id = None
            import gc
            gc.collect()
            return True
        finally:
            self.lock.release()

    def _release(self, body):
        identity = body.get("releaseId")
        release = None
        if identity:
            release = self.store.get("foundation_release", identity, {}) or self.store.get("release", identity, {})
        elif body.get("model"):
            release = next((row for kind in ("foundation_release", "release") for row in self.store.list(kind)
                            if row.get("engineAlias", row.get("model")) == body["model"] and row.get("foundationProjectId", row.get("projectId"))), None)
        if release:
            mirror = self.store.get("release", release["id"], {})
            if mirror:
                release = {**release, **mirror}
        require(release and release.get("state") in ("ready", "active", "retired"), "NO_USABLE_MODEL", status=503)
        alias = release.get("engineAlias", release.get("model"))
        require(not body.get("model") or body["model"] == alias, "RELEASE_PROFILE_MISMATCH", status=409)
        certificate_id = release.get("certificateId")
        certificate = (self.store.get("certificate", certificate_id, {}) or self.store.get("foundation_certificate", certificate_id, {})) if certificate_id else {}
        require(certificate.get("passed") is True and certificate.get("usable") is True and certificate.get("purpose") == "usable_release"
                and certificate.get("state") != "revoked"
                and not certificate.get("fixture") and not certificate.get("testOnly"), "NO_USABLE_MODEL", status=503)
        artifact_id = release.get("artifactId", release.get("checkpointId"))
        require(artifact_id and certificate.get("artifactId", certificate.get("checkpointId")) == artifact_id,
                "RELEASE_PROFILE_MISMATCH", status=409)
        artifact = self.artifact(artifact_id)
        from .schemas import FOUNDATION_ORIGINS
        require(not artifact.get("fixture") and not artifact.get("testOnly") and artifact.get("origin", "random_init") in FOUNDATION_ORIGINS,
                "NO_USABLE_MODEL", status=503)
        project_id = release.get("foundationProjectId", release.get("projectId", artifact.get("projectId")))
        if project_id:
            project = self.store.get("foundation_project", project_id, {})
            require(project and project.get("state") != "revoked" and project.get("stage") != "revoked" and not project.get('revocationPending')
                    and not project.get("testOnly"), "POLICY_REVOKED", status=403)
            if callable(self.policy_validator):
                self.policy_validator(project)
        from .domains import canonical_request, authorize
        mode, language = canonical_request(body)
        allowed = certificate.get("approvedModes", release.get("approvedModes", []))
        languages = certificate.get('approvedLanguages', release.get('approvedLanguages', ['python']))
        ordinary = mode == 'code' and language == 'python' and language in languages and ('code' in allowed or 'python_completion' in allowed)
        if not ordinary:
            capability = authorize(self.store, artifact_id, mode, language)
            require(capability.get('baseCertificateId') == certificate['id'] and capability.get('profileHash') == certificate.get('profileHash'),
                    'RELEASE_PROFILE_MISMATCH', status=409)
            certificate = {**certificate, 'domainCertificateId':capability['id'], 'capabilityScope':capability['scope']}
        certificate = {**certificate, 'effectiveMode':mode, 'effectiveLanguage':language}
        profile = release.get("profile", certificate.get("profile", {}))
        require(profile and profile.get("templateFormat") in ("foundation_structured_v1", "seed-structured-v1"),
                "GENERATION_PROFILE_INCOMPATIBLE", status=409)
        canonical_profile = {key: value for key, value in profile.items() if key != "profileHash"}
        profile_hash = release.get("profileHash") or digest(canonical_profile)
        require(profile_hash == digest(canonical_profile), "RELEASE_PROFILE_MISMATCH", status=409)
        require(not body.get("profileHash") or body["profileHash"] == profile_hash, "RELEASE_PROFILE_MISMATCH", status=409)
        require(not profile.get("model") or profile["model"] == alias, "RELEASE_PROFILE_MISMATCH", status=409)
        if project_id:
            execution_scope=project.get('executionScope','nas')
            backend=project.get('backendReceipt') or {}
            require(certificate.get('executionScope')==profile.get('executionScope')==release.get('executionScope')==execution_scope
                    and artifact.get('executionScope','nas')==execution_scope,'EXECUTION_SCOPE_REQUALIFICATION_REQUIRED',status=409)
            require(backend.get('receiptHash') and certificate.get('backendReceiptHash')==profile.get('backendReceiptHash')
                    ==release.get('backendReceiptHash')==backend['receiptHash'],'BACKEND_CERTIFICATION_REQUIRED',status=409)
            require(execution_scope!='nas' or backend.get('targetCompatible') is True,'BACKEND_TARGET_MISMATCH',status=409)
            require(certificate.get('modelFormat')==profile.get('modelFormat')==artifact.get('format','safetensors')
                    and certificate.get('artifactManifestHash')==profile.get('artifactManifestHash')==artifact.get('manifestHash'),
                    'RELEASE_PROFILE_MISMATCH',status=409)
        return release, certificate, artifact, profile, profile_hash

    @staticmethod
    def assemble(tokenizer, messages, max_context=512, max_output=128):
        require(isinstance(messages, list) and 0 < len(messages) <= 32, "INVALID_MESSAGES")
        require(all(isinstance(row, dict) and row.get("role") in ("system", "user", "assistant")
                    and isinstance(row.get("content"), str) for row in messages), "INVALID_MESSAGES")
        require(sum(len(row["content"].encode("utf-8")) for row in messages) <= 32768, "INPUT_TOO_LARGE_FOR_MODEL", status=413)
        tokens = tokenizer.assemble(messages, assistant_prefix=True)
        require(len(tokens) + max_output <= max_context, "CONTEXT_BUDGET_EXCEEDED", status=422,
                inputTokens=len(tokens), outputTokens=max_output, contextLimit=max_context)
        return tokens

    def prepare(self, body):
        require(isinstance(body, dict), "INVALID_GENERATION_REQUEST")
        release, certificate, artifact, profile, profile_hash = self._release(body)
        if artifact.get('format')=='gguf':
            from .gguf_runtime import verify_conformance
            verify_conformance(self.store,artifact,self.artifacts)
        from .tokenizer import load as load_tokenizer
        tokenizer_id = release.get("tokenizerId") or artifact.get("tokenizerId") or artifact.get("tokenizerArtifactId")
        tokenizer = load_tokenizer(self.store, tokenizer_id)
        require(profile.get("tokenizerHash") == tokenizer.tokenizer_hash, "TOKENIZER_HASH_MISMATCH", status=409)
        assembler = getattr(tokenizer, "assembler_version", tokenizer.record.get("assemblerVersion", "seed-roles-v1"))
        require(profile.get("assemblerVersion") == assembler and profile.get("stopTokenIds") == [2, 7],
                "GENERATION_PROFILE_INCOMPATIBLE", status=409)
        require(profile.get("architectureHashV2") is not None and profile.get("architectureHashV2") == artifact.get("architectureHashV2"),
                "RELEASE_PROFILE_MISMATCH", status=409)
        require(not body.get("profileId") or body["profileId"] == profile.get("profileId", assembler),
                "GENERATION_PROFILE_INCOMPATIBLE", status=409)
        require(certificate.get("profileHash") == profile_hash, "RELEASE_PROFILE_MISMATCH", status=409)
        options = body.get("options", {})
        maximum = options.get("num_predict", body.get("maxOutputTokens", 128))
        require(isinstance(maximum, int) and not isinstance(maximum, bool) and 1 <= maximum <= 128,
                "INVALID_TOKEN_BUDGET")
        maximum = min(maximum, int(profile.get("maxOutputTokens", 128)))
        require(maximum > 0, "INVALID_TOKEN_BUDGET")
        context = min(512, int(profile.get("maxContextTokens", 512)), int(profile.get("trainedContextTokens", 512)))
        if options.get("num_ctx") is not None:
            require(isinstance(options["num_ctx"], int) and not isinstance(options["num_ctx"], bool) and options["num_ctx"] > 0, "INVALID_TOKEN_BUDGET")
            context = min(context, options["num_ctx"])
        messages = body.get("messages")
        domain_task = None
        if certificate.get('effectiveMode') in ('explanation','review'):
            from .domains import prepare_source_request
            messages, domain_task = prepare_source_request(body,certificate)
        tokens = self.assemble(tokenizer, messages, context, maximum)
        return {"artifactId": artifact["id"], "releaseId": release["id"],
                "model": release.get("engineAlias", release.get("model")), "tokenIds": tokens,
                "inputTokenCount": len(tokens), "promptTokenCount": len(tokens),
                "contextLimit": context, "contextLength": context, "outputLimit": maximum,
                "maxOutputTokens": maximum, "profileHash": profile_hash, "profile": profile,
                "messages": messages, "certificateId": certificate["id"], 'mode':certificate['effectiveMode'],
                'language':certificate['effectiveLanguage'], 'capabilityScope':certificate.get('capabilityScope',certificate.get('scope')),
                'domainTask':domain_task}

    def _tokens(self, identity, token_ids, maximum, *, purpose="evaluation", cancel=None):
        model, tokenizer = self.load(identity, purpose=purpose)
        context = min(512, int(model.config.max_position_embeddings))
        require(len(token_ids) + maximum <= context, "CONTEXT_BUDGET_EXCEEDED", status=422)
        require(token_ids, "EMPTY_PROMPT")
        generated, past = [], None
        import psutil
        process = psutil.Process()
        self.last_peak_rss_bytes = process.memory_info().rss
        started = time.monotonic()
        if getattr(model,'backend',None)=='gguf':
            model.reset(token_ids)
            for _ in range(maximum):
                require(cancel is None or not cancel(),'GENERATION_CANCELLED',status=409)
                require(time.monotonic()-started<300,'INFERENCE_TIMEOUT',status=503)
                self.artifact(identity)
                value=model.next_token()
                self.last_peak_rss_bytes=max(self.last_peak_rss_bytes,process.memory_info().rss)
                yield value,tokenizer
                if value in (2,7):break
                model.append(value)
            return
        import torch
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        current = input_ids
        attention = torch.ones_like(input_ids)
        for _ in range(maximum):
            require(cancel is None or not cancel(), "GENERATION_CANCELLED", status=409)
            require(time.monotonic() - started < 300, "INFERENCE_TIMEOUT", status=503)
            self.artifact(identity)  # Revocation is checked before every next token.
            # Never suspend a generator while a thread-local autograd context
            # is active: a shared worker thread may run another request.
            with torch.inference_mode():
                output = model(input_ids=current, attention_mask=attention, past_key_values=past, use_cache=True)
                self.last_peak_rss_bytes = max(self.last_peak_rss_bytes, process.memory_info().rss)
                logits = output.logits[:, -1, :]
                require(bool(torch.isfinite(logits).all()), "NONFINITE_INFERENCE", status=503)
                value = int(torch.argmax(logits, dim=-1).item())
            generated.append(value)
            yield value, tokenizer
            if value in (2, 7):
                break
            past = output.past_key_values
            current = torch.tensor([[value]], dtype=torch.long)
            attention = torch.cat([attention, torch.ones((1, 1), dtype=torch.long)], dim=-1)

    def text(self, artifact_id, prompt, max_tokens=128, *, purpose="evaluation"):
        require(purpose in ('evaluation','distillation_train'), "SERVING_REQUIRES_USABLE_RELEASE", status=403)
        with self.lock:
            _, tokenizer = self.load(artifact_id,purpose=purpose)
            artifact = self.artifact(artifact_id)
            context = min(512, int(self.model.config.max_position_embeddings), int(artifact.get("trainedContextTokens") or self.model.config.max_position_embeddings))
            effective_output = min(max_tokens, 128, context//2)
            ids = tokenizer.assemble(prompt) if isinstance(prompt, list) else [1, *tokenizer.encode_ordinary(prompt)]
            require(isinstance(max_tokens, int) and not isinstance(max_tokens, bool) and 1 <= max_tokens <= 128, "INVALID_TOKEN_BUDGET")
            started = time.monotonic()
            require(len(ids)+effective_output <= context, "CONTEXT_BUDGET_EXCEEDED", status=422)
            generated = [token for token, _ in self._tokens(artifact_id, ids, effective_output,purpose=purpose)]
            return tokenizer.decode(generated, skip_special_tokens=True), {
                "prompt_eval_count": len(ids), "eval_count": len(generated),
                "total_duration": int((time.monotonic() - started) * 1e9), "runtimeResourceObserved": True,
                "peakRssBytes": getattr(self, "last_peak_rss_bytes", None), "contextLimit": context,
                "effectiveOutputLimit": effective_output}

    def generate(self, artifact_id, messages=None, max_new_tokens=128, *, purpose="evaluation"):
        if isinstance(artifact_id, dict):
            chunks, finished = [], None
            for line in self.stream(artifact_id):
                item = json.loads(line)
                chunks.append(item.get("response", ""))
                if item.get("done"):
                    finished = item
            require(finished is not None, "INCOMPLETE_ENGINE_RESPONSE", status=503)
            return {**finished, "response": "".join(chunks)}
        require(purpose == "evaluation", "SERVING_REQUIRES_USABLE_RELEASE", status=403)
        text, timings = self.text(artifact_id, messages, max_new_tokens)
        return {"text": text, "usage": timings, "observedAt": now()}

    def stream(self, body):
        prepared = self.prepare(body)
        with self.lock:
            started = time.monotonic()
            generated, emitted = [], ""
            for token, tokenizer in self._tokens(prepared["artifactId"], prepared["tokenIds"], prepared["outputLimit"], purpose="serve"):
                # Certificate/policy revocation is independent from artifact state.
                self._release({**body, "releaseId": prepared["releaseId"]})
                generated.append(token)
                full = tokenizer.decode(generated, skip_special_tokens=True)
                # Byte BPE can decode a partial Unicode codepoint as U+FFFD.
                safe = full.rstrip("\ufffd")
                if not prepared.get('domainTask') and safe.startswith(emitted) and len(safe) > len(emitted):
                    delta = safe[len(emitted):]
                    emitted = safe
                    yield json.dumps({"model": prepared["model"], "response": delta, "done": False}, ensure_ascii=False) + "\n"
            full = tokenizer.decode(generated, skip_special_tokens=True)
            self._release({**body, "releaseId": prepared["releaseId"]})
            if prepared.get('domainTask'):
                from .domains import judge_static
                try:
                    require(judge_static({'purpose':prepared['mode']},prepared['domainTask'],full),'DOMAIN_RESPONSE_NOT_VERIFIED',status=409)
                except (ValueError,TypeError,KeyError,RecursionError):raise Fault('DOMAIN_RESPONSE_NOT_VERIFIED',status=409) from None
            if full.startswith(emitted) and full != emitted:
                yield json.dumps({"model": prepared["model"], "response": full[len(emitted):], "done": False}, ensure_ascii=False) + "\n"
            yield json.dumps({"model": prepared["model"], "response": "", "done": True,
                              "done_reason": "stop" if generated and generated[-1] in (2, 7) else "length",
                              "prompt_eval_count": prepared["inputTokenCount"], "eval_count": len(generated),
                              "total_duration": int((time.monotonic() - started) * 1e9)}, ensure_ascii=False) + "\n"

    def tags(self):
        output = []
        for release in self.store.list("foundation_release"):
            try:
                self._release({"releaseId": release["id"], "mode": "code"})
                artifact = self.artifact(release["artifactId"])
                if artifact.get('format')=='gguf':
                    from .gguf_runtime import verify_conformance
                    verify_conformance(self.store,artifact,self.artifacts)
            except Fault:
                continue
            output.append({"name": release["engineAlias"], "model": release["engineAlias"],
                           "digest": release["engineDigest"], "size": artifact.get("sizeBytes", 0),
                           "details": {"parameter_size": str(artifact.get("parameterCount", ""))}})
        return {"models": output}
