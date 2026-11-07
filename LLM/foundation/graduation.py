"""Quality-qualified registration and usable certificates, never raw score submission."""
from __future__ import annotations

import json
import shutil

from LLM.core import atomic_json, digest, file_hash, inside, now, require, uid
from .schemas import FOUNDATION_ORIGINS


def _stored(store, kind, value):
    return store.get(kind, value if isinstance(value, str) else value["id"])


def _checkpoint(store, value):
    identity = value if isinstance(value, str) else value["id"]
    checkpoint = store.get("foundation_checkpoint", identity, {}) or store.get("artifact", identity, {})
    require(checkpoint and checkpoint.get("state") not in ("revoked", "failed", "quarantined"), "CHECKPOINT_NOT_READY", status=409)
    steps = checkpoint.get("completedOptimizerSteps", checkpoint.get("completedSteps", 0))
    require((isinstance(steps, int) and not isinstance(steps, bool) and steps > 0)
            or checkpoint.get("sourceCheckpointId"), "NO_EFFECTIVE_TRAINING", status=409)
    return checkpoint


def _publish_base(store, project, checkpoint, registry=None):
    from .checkpoint import verify
    from .tokenizer import load as load_tokenizer
    from LLM.registry import Registry
    registry = registry or Registry(store)
    # An existing registered descendant keeps its model identity.
    if store.get("artifact", checkpoint["id"], {}):
        registry.verify(checkpoint)
        require(checkpoint.get("origin") in FOUNDATION_ORIGINS, "FOUNDATION_ORIGIN_REQUIRED")
        return checkpoint
    root, _ = verify(store, checkpoint)
    tokenizer_id = checkpoint.get("tokenizerId", checkpoint.get("tokenizerArtifactId", project.get("tokenizerId")))
    tokenizer_record = store.get("foundation_tokenizer", tokenizer_id)
    load_tokenizer(store, tokenizer_record)
    tokenizer_root = inside(store.root, tokenizer_record["storagePath"])
    temporary = inside(store.root, "foundation/staging/graduate-" + uid())
    temporary.mkdir(parents=True)
    try:
        # Optimizer files are not model tensors and must never enter Registry.inspect.
        from .architecture import load_model
        model = load_model(store, checkpoint)
        model.save_pretrained(temporary, safe_serialization=True)
        del model
        for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "metadata.json"):
            if (tokenizer_root / name).is_file():
                shutil.copy2(tokenizer_root / name, temporary / name)
        value = registry.publish(temporary, kind="base", parents=[], metadata={
            "origin": project.get('origin','random_init'), "pretrainedWeightArtifactIds": project.get('pretrainedWeightArtifactIds',[]), "foundationProjectId": project["id"],
            'trustedGrowthReceipt':project.get('trustedGrowthReceipt'),'executionScope':project.get('executionScope','nas'),
            'sourceProjectId':project.get('sourceProjectId'),'sourceGenesisId':project.get('sourceGenesisId'),
            "projectId": project["id"], "sourceCheckpointId": checkpoint["id"],
            "genesisId": project.get("genesisId"), "tokenizerId": tokenizer_id,
            "tokenizerArtifactId": tokenizer_id, "tokenizerHash": tokenizer_record["tokenizerHash"],
            "architectureHashV2": checkpoint.get("architectureHashV2", checkpoint.get("configHash")),
            "config": checkpoint.get("config", project.get("config", {})),
            "trainedContextTokens": checkpoint.get("trainedContextTokens", checkpoint.get("config", {}).get("maxSequence", 128)),
            "servingEligible": False, "testOnly": project.get("testOnly") is True,
            "fixture": project.get("testOnly") is True or checkpoint.get("fixture") is True})
        family_id = project.get('plannedFamilyId') or "foundation-family-" + digest([project["id"], value["id"], value["tokenizerHash"]])[:20]
        family = store.get("family", family_id, {}) or store.put("family", {
            "anchorArtifactId": value["id"], "name": project.get("name", "SEED foundation"),
            "architectureHash": value.get("architectureHash"), "architectureHashV2": value.get("architectureHashV2"),
            "tensorSchemaHash": value["tensorSchemaHash"], "tokenizerHash": value["tokenizerHash"],
            "templateHash": value["templateHash"], "origin": project.get('origin','random_init'), "foundationProjectId": project["id"]}, family_id)
        return store.put("artifact", {"familyId": family["id"]}, value["id"])
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def graduate(store, project, checkpoint, evaluation, registry=None):
    project = _stored(store, "foundation_project", project)
    checkpoint = _checkpoint(store, checkpoint)
    evaluation = _stored(store, "foundation_evaluation", evaluation)
    require(project.get("stage") != "revoked" and project.get("state") != "revoked", "POLICY_REVOKED", status=403)
    require(evaluation.get("kind") == "qualification" and evaluation.get("state") == "completed"
            and evaluation.get("artifactId", evaluation.get("checkpointId")) == checkpoint["id"]
            and evaluation.get("projectId") == project["id"] and evaluation.get("gate", {}).get("passed") is True,
            "FOUNDATION_QUALIFICATION_REQUIRED", status=409)
    require(not evaluation.get("fixture") and not project.get("testOnly"), "FIXTURE_NOT_FOR_PRODUCTION", status=409)
    artifact = _publish_base(store, project, checkpoint, registry)
    result = store.put("foundation_graduation", {"projectId": project["id"], "checkpointId": checkpoint["id"],
                       "evaluationId": evaluation["id"], "artifactId": artifact["id"], "familyId": artifact["familyId"],
                       "state": "qualified", "servingEligible": False, "approvedModes": [], "origin": project.get('origin','random_init')})
    store.event("graduation.completed", result["id"], {"projectId": project["id"], "artifactId": artifact["id"]})
    return result


def certify_usable(store, project, checkpoint, evaluation, registry=None):
    project = _stored(store, "foundation_project", project)
    checkpoint = _checkpoint(store, checkpoint)
    evaluation = _stored(store, "foundation_evaluation", evaluation)
    require(not project.get("testOnly") and project.get("state") != "revoked" and project.get("stage") != "revoked", "POLICY_REVOKED", status=403)
    require(evaluation.get("kind") == "release" and evaluation.get("state") == "completed"
            and evaluation.get("productionCertified") is True and evaluation.get("gate", {}).get("passed") is True
            and not evaluation.get("fixture") and evaluation.get("projectId") == project["id"]
            and evaluation.get("artifactId") == checkpoint["id"], "USABLE_RELEASE_EVALUATION_REQUIRED", status=409)
    if project.get('origin')=='architecture_growth':
        require(project.get('sourceCheckpointId') in evaluation.get('baselineIds',[]),
            'GROWTH_SOURCE_BASELINE_REQUIRED',status=409)
    # A real foundation qualification must precede instruction/release certification.
    ancestors, pointer = set(), checkpoint
    while pointer and pointer["id"] not in ancestors:
        ancestors.add(pointer["id"])
        parent_id = pointer.get("parentCheckpointId", pointer.get("sourceCheckpointId"))
        pointer = (store.get("foundation_checkpoint", parent_id, {}) or store.get("foundation_genesis", parent_id, {})) if parent_id else {}
    qualifications = [record for record in store.list("foundation_graduation") if record.get("projectId") == project["id"]
                      and record.get("state") == "qualified" and record.get("checkpointId") in ancestors]
    require(qualifications, "FOUNDATION_QUALIFICATION_REQUIRED", status=409)
    artifact = _publish_base(store, project, checkpoint, registry)
    trained_context = int(checkpoint.get("trainedContextTokens", artifact.get("trainedContextTokens", 128)))
    require(16 <= trained_context <= 512, "GENERATION_PROFILE_INCOMPATIBLE")
    measured = evaluation.get("effectiveGenerationProfiles", {}).get(checkpoint["id"])
    expected_profile = [trained_context, min(128, trained_context//2)]
    require(measured and all(list(item) == expected_profile for item in measured),
            "EVALUATION_PROFILE_MISMATCH", status=409)
    release_id = "foundation-release-" + digest([artifact["id"], evaluation["id"], "structured-v1"])[:24]
    alias = "seed/foundation:" + release_id
    engine_digest = artifact["manifestHash"]
    profile = {"model": alias, "modelRevision": engine_digest, "templateFormat": "seed-structured-v1", "provider": "foundation",
               'executionScope':project.get('executionScope','nas'),
               'backendReceiptHash':(project.get('backendReceipt') or {}).get('receiptHash'),
               'modelFormat':artifact.get('format','safetensors'),'artifactManifestHash':artifact['manifestHash'],
               "tokenizerHash": artifact["tokenizerHash"], "assemblerVersion": "seed-roles-v1", "profileId": "seed-structured-v1", "contextLength": 512,
               "maxContextTokens": 512, "trainedContextTokens": trained_context,
               "maxOutputTokens": min(128, trained_context//2), "stopTokenIds": [2, 7],
               "architectureHashV2": artifact.get("architectureHashV2")}
    certificate = store.put("foundation_certificate", {"purpose": "usable_release", "usable": True, "passed": True,
                "projectId": project["id"], "artifactId": artifact["id"], "checkpointId": checkpoint["id"],
                "evaluationIds": [evaluation["id"]], "qualifiedGraduationId": qualifications[-1]["id"],
                "approvedModes": ["code"], "approvedLanguages": ["python"], "scope": "short_python_function_completion",
                "profileHash": digest(profile), "architectureHashV2": artifact.get("architectureHashV2"),
                'executionScope':project.get('executionScope','nas'),
                'backendReceiptHash':(project.get('backendReceipt') or {}).get('receiptHash'),
                'modelFormat':artifact.get('format','safetensors'),'artifactManifestHash':artifact['manifestHash'],
                "policyVersion": project.get("policyVersion", project.get("policy", {}).get("version", 0)),
                "fixture": False, "testOnly": False, "observedAt": now()})
    artifact = store.put("artifact", {"state": "ready", "servingEligible": True, "certificateId": certificate["id"]}, artifact["id"])
    release = store.put("foundation_release", {"foundationProjectId": project["id"], "projectId": project["id"],
               "artifactId": artifact["id"], "checkpointId": checkpoint["id"], "tokenizerId": artifact["tokenizerId"],
               "certificateId": certificate["id"], "engineAlias": alias, "engineDigest": engine_digest,
               "profile": profile, "profileHash": digest(profile), "approvedModes": ["code"],
               "approvedLanguages": ["python"], "state": "ready", "origin": project.get('origin','random_init'), "verifiedAt": now()}, release_id)
    release=store.put('foundation_release',{'executionScope':project.get('executionScope','nas'),
        'backendReceiptHash':(project.get('backendReceipt') or {}).get('receiptHash'),
        'modelFormat':artifact.get('format','safetensors'),'artifactManifestHash':artifact['manifestHash']},release_id)
    return {"certificate": certificate, "release": release, "artifact": artifact}


certify = certify_usable
