"""Derive narrow local boundary receipts from a trusted four-suite pytest run.

This is an internal runner hook, not a score/receipt upload endpoint. A green
suite can attest only the listed mechanical boundaries, never model ability,
an actual curriculum transition, a container, or a NAS measurement.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from LLM.core import Fault, digest, file_hash, inside, require


RUNNER_VERSION = 'seed-pytest-four-suites-v1'
BOUNDARY_ONLY_REQUIREMENTS = frozenset((12, 13, 15, 19, 22, 23))
CLAIM_SCOPE = 'local_software_boundary'
IO_CLAIM_SCOPE = 'local_io_and_software_boundary'
IO_RUNNER_VERSION = 'seed-raw-io-136mib-v1'
IO_BYTES = 136 * 1024**2
IO_MAX_SHARD_BYTES = 128 * 1024**2
IO_TIMEOUT_SECONDS = 300


def _cases(filename, *names):
    return tuple(filename + '::' + name for name in names)


# These are source-owned case selectors, not configurable request parameters.
# JUnit records retain their complete class/parameter IDs in the issued proof.
REQUIRED_CASES = {
    12: _cases('LLM/tests/test_foundation_backup.py',
        'test_real_foundation_backup_restores_weights_tokenizer_shards_and_policy',
        'test_active_jobs_uncheckpointed_pending_inputs_and_gc_are_held',
        'test_cross_process_copy_gate_tamper_and_symlink_fail_closed')
        + _cases('LLM/tests/test_foundation_training.py',
        'test_pause_resume_reproduces_uninterrupted_optimizer_result',
        'test_tampering_and_lost_lease_cannot_publish_checkpoint')
        + _cases('LLM/tests/test_foundation_scope_lifecycle.py',
        'test_authenticated_chunks_reject_truncation_and_wrong_binding',
        'test_encrypted_backup_has_no_plain_database_and_revocation_blocks_copied_package')
        + _cases('LLM/tests/test_foundation_recovery_boundaries.py',
        'test_actual_commit_failure_retry_restores_exact_optimizer_rng_cursor_and_weights',
        'test_full_s0_checkpoint_large_reservation_and_file_io_stay_outside_write_tx',
        'test_revocation_during_checkpoint_file_sync_cannot_publish_ready_metadata')
        + _cases('LLM/tests/test_foundation_data_pipeline.py',
        'test_slow_quota_scan_does_not_block_real_second_connection_writer',
        'test_large_atomic_reservation_cannot_be_double_allocated'),
    13: _cases('RAG/tests/test_foundation_bridge.py',
        'test_no_caller_permission_can_make_unknown_source_trainable',
        'test_holdout_and_revocation_are_rechecked_for_each_export',
        'test_policy_refresh_calls_foundation_hook_after_rag_epoch_without_recursion')
        + _cases('LLM/tests/test_foundation_scope_lifecycle.py',
        'test_external_teacher_withdrawal_revokes_actual_scalar_and_array_students_only',
        'test_withdrawal_interrupts_real_token_stream_optimizer_and_applied_extension_slot')
        + _cases('LLM/tests/test_foundation_registry_namespace.py',
        'test_revocation_pending_blocks_registry_verification_and_publication'),
    15: _cases('LLM/tests/test_foundation_data_pipeline.py',
        'test_independent_exact_cross_split_copy_is_excluded_without_poisoning_families',
        'test_near_copy_and_verified_fork_do_not_inflate_independent_group_count',
        'test_balancer_uses_actual_target_feedback_and_honest_absolute_caps')
        + _cases('LLM/tests/test_foundation_policy.py',
        'test_dedup_and_global_split_survive_new_project',
        'test_rag_split_is_authoritative_and_conflict_is_excluded',
        'test_secret_or_disallowed_source_never_enters_training_file'),
    19: _cases('LLM/tests/test_foundation_prepared_transfer.py',
        'test_rights_blob_corpus_tokenizer_fixed_ids_roundtrip_and_read_only_dry_run',
        'test_missing_license_evidence_and_revoked_existing_source_cannot_be_overridden')
        + _cases('LLM/tests/test_foundation_growth_transfer.py',
        'test_offline_package_dry_run_preserves_nas_store_and_excludes_holdouts',
        'test_offline_transfer_tampering_path_injection_and_wrong_project_fail_closed'),
    22: _cases('LLM/tests/test_foundation_scope_lifecycle.py',
        'test_tainted_tokenizer_plan_requires_new_genesis_and_original_stays_revoked',
        'test_actual_clean_ancestor_is_reused_without_the_tainted_child_or_its_authority',
        'test_external_teacher_withdrawal_revokes_actual_scalar_and_array_students_only')
        + _cases('LLM/tests/test_foundation_registry_namespace.py',
        'test_equal_bytes_in_new_verified_namespace_do_not_rebind_revoked_old_artifact'),
    23: _cases('LLM/tests/test_foundation_scope_lifecycle.py',
        'test_authenticated_chunks_reject_truncation_and_wrong_binding',
        'test_encrypted_backup_has_no_plain_database_and_revocation_blocks_copied_package',
        'test_backup_expiry_removes_cipher_and_preserves_tombstone',
        'test_long_lived_worker_logs_and_events_are_bounded_without_removing_lineage',
        'test_old_corpus_cache_can_expire_while_the_same_frozen_tokenizer_remains_active')
        + _cases('LLM/tests/test_foundation_retention.py',
        'test_dry_run_and_explicit_deletion_preserve_latest_and_tombstones',
        'test_release_pin_best_and_parent_chain_never_reclaimed',
        'test_manifest_tampering_blocks_collection_and_quota_admission',
        'test_outside_scope_and_corpus_are_never_automatic_delete_candidates',
        'test_apply_rechecks_new_references_and_admission_counts_linked_artifacts',
        'test_unowned_file_inside_checkpoint_is_not_removed_as_generated_data'),
}

# These may report softwareBoundaryPassed but cannot issue requirement passes.
# Their additional real-model/OS/actual-transition evidence remains mandatory.
METADATA_CASES = {
    14: _cases('RAG/tests/test_api.py', 'test_auth_csrf_and_cross_origin_boundaries',
        'test_nas_policy_wraps_health_login_and_interface'),
    16: _cases('LLM/tests/test_foundation_data_pipeline.py',
        'test_disk_streaming_shards_and_sql_index_round_trip')
        + _cases('RAG/tests/test_foundation_bridge.py',
        'test_external_sft_metadata_and_shards_use_bounded_indexed_pages'),
    17: _cases('LLM/tests/test_foundation_curriculum_formats.py',
        'test_curriculum_runs_actual_cpu_slice_then_holds_insufficient_dev_evidence',
        'test_curriculum_requires_matching_baseline_groups_and_rejects_regression'),
    18: _cases('LLM/tests/test_foundation_scope_lifecycle.py',
        'test_source_claims_require_actual_fact_and_exact_source_evidence',
        'test_capability_registration_never_accepts_a_claimed_pass_or_unlocks_language'),
    20: _cases('LLM/tests/test_foundation_growth_transfer.py',
        'test_actual_same_tokenizer_distillation_updates_student_and_records_pinned_teacher',
        'test_logits_distillation_objective_has_correct_temperature_and_shifted_mask')
        + _cases('LLM/tests/test_foundation_response_mixing.py',
        'test_each_optimizer_envelope_is_at_most_ten_percent_teacher_without_duplication',
        'test_insufficient_human_pool_drops_whole_teacher_target_and_reports_deficit',
        'test_release_fixture_or_group_cannot_be_reused_for_teacher_training'),
    21: _cases('LLM/tests/test_foundation_growth_adoption.py',
        'test_growth_adoption_has_new_owner_genesis_random_control_and_no_prior_certificates',
        'test_actual_grown_training_lora_pair_and_merge_preserve_derived_origin',
        'test_growth_adoption_rejects_tampered_candidate_before_new_project'),
}


def verify_software(store, software, *, source_root=None):
    """Reread immutable result bytes and current program/test snapshots."""
    from .acceptance import (SUITES, source_snapshot, test_snapshot,
                             verify_evidence, _xml_summary)
    root = Path(source_root or Path(__file__).resolve().parents[2]).resolve()
    source, tests = source_snapshot(root), test_snapshot(root)
    authoritative = store.get('foundation_software_acceptance', software.get('id') or '', {})
    require(authoritative and authoritative.get('evidenceHash') == software.get('evidenceHash'),
            'ACCEPTANCE_SOFTWARE_BINDING_INVALID', status=409)
    verify_evidence(authoritative, expected_source=source['hash'])
    require(authoritative.get('runnerVersion') == RUNNER_VERSION
        and authoritative.get('passed') is True and authoritative.get('exitCode') == 0
        and authoritative.get('testSourceHash') == tests['hash']
        and authoritative.get('suitePaths') == list(SUITES),
        'ACCEPTANCE_TRUSTED_SUITE_REQUIRED', status=409)
    expected = 'reports/pre-nas/' + authoritative['id'] + '/results.xml'
    require(authoritative.get('resultsPath') == expected, 'ACCEPTANCE_RESULTS_MISSING', status=409)
    path = inside(store.root, expected)
    require(path.is_file() and not path.is_symlink()
        and all(not parent.is_symlink() for parent in path.parents)
        and file_hash(path) == authoritative.get('resultsHash'),
        'ACCEPTANCE_RESULTS_CHANGED', status=409)
    summary = _xml_summary(path)
    require(summary['tests'] >= 100 and not summary['failures'] and not summary['skipped']
        and len(set(summary['caseIds'])) == summary['tests']
        and all(authoritative.get(key) == value for key, value in summary.items())
        and authoritative.get('suiteHash') == digest([summary['caseIds'], tests['hash']]),
        'ACCEPTANCE_SUITE_EVIDENCE_INVALID', status=409)
    # Every required suite must have actually collected cases in this result.
    xml = ET.parse(path).getroot()
    cases = [(case.get('classname', ''), case.get('name', '')) for case in xml.iter('testcase')]
    for suite in SUITES:
        module = suite.replace('/', '.')
        require(any(name == module or name.startswith(module + '.') for name, _ in cases),
                'ACCEPTANCE_SUITE_NOT_EXECUTED', status=409, suite=suite)
    return {'software': authoritative, 'source': source, 'tests': tests,
            'cases': cases, 'summary': summary, 'root': root}


def _coverage(context, selectors):
    matched, missing = [], []
    for selector in selectors:
        filename, method = selector.split('::')
        module = filename[:-3].replace('/', '.')
        source_hash = context['tests']['files'].get(filename)
        ids = [name + '::' + case for name, case in context['cases']
               if source_hash and (name == module or name.startswith(module + '.'))
               and case.split('[', 1)[0] == method]
        if ids: matched.extend(ids)
        else: missing.append(selector)
    return sorted(set(matched)), missing


def _body(context, number, selectors):
    cases, missing = _coverage(context, selectors)
    return {'kind': 'local_software_requirement', 'requirementNumber': number,
        'claimScope': CLAIM_SCOPE, 'sourceHash': context['source']['hash'],
        'testSourceHash': context['tests']['hash'],
        'softwareEvidenceId': context['software']['id'],
        'softwareEvidenceHash': context['software']['evidenceHash'],
        'junitHash': context['software']['resultsHash'],
        'suiteHash': context['software']['suiteHash'],
        'requiredCaseIds': list(selectors), 'caseIds': cases,
        'softwareBoundaryPassed': not missing,
        'reasonCodes': ['ACCEPTANCE_REQUIRED_CASE_MISSING'] if missing else [],
        'missingCaseIds': missing, 'actualModelQualityPassed': False,
        'actualContainerPassed': False, 'nasModified': False, 'nasMeasured': False,
        'productionModelCertified': False}


def _safe_local(root, relative):
    raw = Path(root)/relative
    require(not raw.is_symlink() and all(not parent.is_symlink() for parent in raw.parents),
            'ACCEPTANCE_IO_PATH_UNSAFE', status=409)
    return inside(root, relative)


def _io_paths(context):
    identity = context['software']['id']
    require(re.fullmatch(r'[A-Za-z0-9_-]{1,128}', identity), 'ACCEPTANCE_IO_PATH_UNSAFE')
    relative = 'LLM/runtime-data/verification/raw-io-software-' + identity
    return relative, _safe_local(context['root'], relative)


def _verify_io_report(context, *, expected_file_hash=None):
    """Read the actual fixture, immutable files and close/reopen SQL pages."""
    import apsw
    relative, output = _io_paths(context)
    report_path = _safe_local(output, 'report.json')
    require(report_path.is_file() and report_path.stat().st_size <= 1024**2,
            'ACCEPTANCE_IO_REPORT_REQUIRED', status=409)
    checksum = file_hash(report_path)
    require(expected_file_hash is None or checksum == expected_file_hash,
            'ACCEPTANCE_IO_REPORT_CHANGED', status=409)
    report = json.loads(report_path.read_text('utf-8'))
    require(report.get('reportHash') == digest({k:v for k,v in report.items() if k != 'reportHash'})
        and report.get('schemaVersion') == 1 and report.get('runnerVersion') == IO_RUNNER_VERSION
        and report.get('kind') == 'foundation_raw_io_stress' and report.get('passed') is True
        and report.get('sourceHash') == context['source']['hash'] and report.get('reasonCodes') == [],
        'ACCEPTANCE_IO_REPORT_INVALID', status=409)
    require(report.get('rawBytes') == IO_BYTES and report.get('rowCount') == 136
        and report.get('resumeRowCount') == 136 and report.get('resumeVerified') is True
        and report.get('maxShardBytes') == IO_MAX_SHARD_BYTES
        and report.get('fixtureOnly') is True and report.get('productionReady') is False
        and report.get('modelTrainingPerformed') is False and report.get('nasModified') is False
        and report.get('nearDuplicateDetectionEnabled') is False,
        'ACCEPTANCE_IO_BOUNDARY_MISMATCH', status=409)
    rss, elapsed = report.get('peakRssBytes'), report.get('elapsedSeconds')
    require(type(rss) is int and 0 < rss < 1024**3
        and type(elapsed) in (int,float) and math.isfinite(elapsed) and 0 < elapsed <= IO_TIMEOUT_SECONDS,
        'ACCEPTANCE_IO_MEASUREMENT_INVALID', status=409)
    require(report.get('fixtureStorePath') == relative + '/store', 'ACCEPTANCE_IO_PATH_UNSAFE')
    fixture = _safe_local(output, 'store')
    database = _safe_local(fixture, 'metadata/llm.db')
    require(database.is_file(), 'ACCEPTANCE_IO_FIXTURE_MISSING', status=409)
    db = apsw.Connection(str(database), flags=apsw.SQLITE_OPEN_READONLY)
    try:
        def record(kind, identity):
            row = db.execute('SELECT body FROM records WHERE kind=? AND id=?', (kind,identity)).fetchone()
            require(row, 'ACCEPTANCE_IO_FIXTURE_MISSING', status=409)
            return json.loads(row[0])
        project = record('foundation_project', report.get('projectId'))
        item = record('foundation_corpus', report.get('corpusId'))
        require(project.get('testOnly') is True and item.get('fixture') is True
            and item.get('projectId') == project['id'] and item.get('state') == 'ready'
            and item.get('manifestHash') == report.get('manifestHash'), 'ACCEPTANCE_IO_FIXTURE_INVALID')
        forbidden = ('foundation_genesis','foundation_checkpoint','foundation_run','artifact','foundation_certificate','foundation_release')
        require(not any(db.execute('SELECT 1 FROM records WHERE kind=? LIMIT 1',(kind,)).fetchone() for kind in forbidden)
            and not db.execute('SELECT 1 FROM jobs LIMIT 1').fetchone(), 'ACCEPTANCE_IO_MODEL_ACTIVITY_FORBIDDEN')
    finally:
        db.close()
    corpus_root = _safe_local(fixture, item['storagePath'])
    manifest_path = _safe_local(corpus_root, 'manifest.json')
    require(manifest_path.is_file() and manifest_path.stat().st_size <= 16*1024**2
        and file_hash(manifest_path) == report['manifestHash'], 'ACCEPTANCE_IO_MANIFEST_CHANGED')
    manifest = json.loads(manifest_path.read_text('utf-8'))
    require(manifest.get('schemaVersion') == 2 and manifest.get('id') == item['id']
        and manifest.get('projectId') == project['id'] and manifest.get('rowIndex') == 'row-index.sqlite'
        and manifest.get('splitCounts',{}).get('train') == 136
        and manifest.get('splitBytes',{}).get('train') == IO_BYTES,
        'ACCEPTANCE_IO_MANIFEST_INVALID')
    names = manifest.get('splitShards',{}).get('train',[])
    require(len(names) == len(set(names)) == 2 and 1 <= len(manifest.get('files',{})) <= 64,
            'ACCEPTANCE_IO_SHARD_MISMATCH')
    for name, actual_hash in manifest['files'].items():
        require(re.fullmatch(r'(?:train|dev|release|audit)(?:-\d{5})?\.jsonl|members\.jsonl|exclusions\.jsonl|duplicates\.jsonl|row-index\.sqlite',name),
                'ACCEPTANCE_IO_PATH_UNSAFE')
        path = _safe_local(corpus_root, name)
        require(path.is_file() and file_hash(path) == actual_hash, 'ACCEPTANCE_IO_FILE_CHANGED')
    sizes = {name:_safe_local(corpus_root,name).stat().st_size for name in names}
    require(all(name in manifest['files'] and 0 < length <= IO_MAX_SHARD_BYTES for name,length in sizes.items())
        and sizes == report.get('shards') and sum(sizes.values()) == report.get('serializedBytes'),
        'ACCEPTANCE_IO_SHARD_MISMATCH')
    sequential = hashlib.sha256(); raw_bytes = 0; count = 0
    for name in names:
        with _safe_local(corpus_root,name).open('rb') as handle:
            while line := handle.readline(2*1024**2+1):
                require(len(line) <= 2*1024**2, 'ACCEPTANCE_IO_ROW_INVALID')
                row = json.loads(line); data = row['text'].encode('utf-8')
                require(row.get('split') == 'train' and len(data) == 1024**2
                    and digest(data) == row.get('hash'), 'ACCEPTANCE_IO_ROW_INVALID')
                sequential.update(row['hash'].encode()); raw_bytes += len(data); count += 1
    index = _safe_local(corpus_root, 'row-index.sqlite')
    resumed = hashlib.sha256(); cursor = 0; resumed_count = 0
    while True:
        # Each bounded 17-row page uses a distinct real read-only connection.
        db = apsw.Connection(str(index), flags=apsw.SQLITE_OPEN_READONLY)
        try:
            page = list(db.execute('SELECT id,file_name,offset,length,hash FROM positions WHERE split=? AND id>? ORDER BY id LIMIT 17',('train',cursor)))
        finally:
            db.close()
        if not page: break
        for ordinal,name,offset,length,row_hash in page:
            require(name in sizes and type(offset) is int and type(length) is int
                and offset >= 0 and 0 < length <= 2*1024**2 and offset+length <= sizes[name],
                'ACCEPTANCE_IO_INDEX_INVALID')
            with _safe_local(corpus_root,name).open('rb') as handle:
                handle.seek(offset); row = json.loads(handle.read(length))
            require(row.get('split') == 'train' and digest(row['text'].encode()) == row_hash == row.get('hash'),
                    'ACCEPTANCE_IO_INDEX_INVALID')
            resumed.update(row_hash.encode()); resumed_count += 1; cursor = ordinal
    require(count == resumed_count == 136 and raw_bytes == IO_BYTES
        and sequential.hexdigest() == resumed.hexdigest() == report.get('rowHashDigest')
        and file_hash(report_path) == checksum and file_hash(manifest_path) == report['manifestHash']
        and file_hash(index) == manifest['files']['row-index.sqlite'], 'ACCEPTANCE_IO_RESUME_INVALID')
    driver = _safe_local(context['root'], 'docker/check_large_corpus.py')
    driver_hash = context['source']['files'].get('docker/check_large_corpus.py')
    require(driver.is_file() and file_hash(driver) == driver_hash, 'ACCEPTANCE_IO_RUNNER_CHANGED')
    return {'ioRunnerVersion':IO_RUNNER_VERSION, 'ioDriverHash':driver_hash,
        'ioReportPath':relative+'/report.json', 'ioReportFileHash':checksum,
        'ioReportHash':report['reportHash'], 'ioManifestHash':report['manifestHash'],
        'ioRawBytes':raw_bytes, 'ioShardCount':len(sizes), 'ioRowCount':count,
        'ioResumeRowCount':resumed_count, 'ioRowHashDigest':sequential.hexdigest(),
        'ioPeakRssBytes':rss, 'ioElapsedSeconds':elapsed, 'fixtureOnly':True,
        'modelTrainingPerformed':False, 'nearDuplicateDetectionCertified':False}


def _run_large_io(context, store):
    """Fixed local driver only; no request-supplied command, size, or score."""
    from .acceptance import require_local_path
    driver = _safe_local(context['root'], 'docker/check_large_corpus.py')
    if not driver.is_file(): return None, ['CURRENT_SOURCE_BOUND_136_MIB_IO_REPORT']
    require_local_path(context['root'])
    relative, output = _io_paths(context)
    require_local_path(output, context['root']/'LLM/runtime-data/verification')
    require(not output.exists(), 'FRESH_VERIFICATION_DIRECTORY_REQUIRED')
    require(file_hash(driver) == context['source']['files'].get('docker/check_large_corpus.py'),
            'ACCEPTANCE_IO_RUNNER_CHANGED')
    log = _safe_local(store.root, 'reports/pre-nas/'+context['software']['id']+'/large-io.log')
    log.parent.mkdir(parents=True,exist_ok=True)
    prefix = output.parent/('raw-io-no-bytecode-'+context['software']['id'])
    command = [sys.executable,'-B','-X','utf8','-X','pycache_prefix='+str(prefix),str(driver),'--output',str(output)]
    process = None; started = time.monotonic()
    try:
        with log.open('xb') as stream:
            process = subprocess.Popen(command,cwd=context['root'],stdout=stream,stderr=stream,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            import psutil
            child = psutil.Process(process.pid)
            while process.poll() is None:
                require(time.monotonic()-started <= IO_TIMEOUT_SECONDS, 'ACCEPTANCE_IO_TIME_LIMIT')
                require(stream.tell() <= 2*1024**2, 'ACCEPTANCE_IO_OUTPUT_LIMIT')
                try: require(child.memory_info().rss < 1024**3, 'ACCEPTANCE_IO_MEMORY_LIMIT')
                except psutil.NoSuchProcess: pass
                time.sleep(.25)
            require(process.wait(timeout=10) == 0, 'ACCEPTANCE_IO_RUN_FAILED')
        return _verify_io_report(context), []
    except (Fault,OSError,ValueError) as error:
        return None, [getattr(error,'code','ACCEPTANCE_IO_RUN_FAILED')]
    finally:
        if process is not None and process.poll() is None:
            import psutil
            try:
                for child in psutil.Process(process.pid).children(recursive=True):
                    try: child.kill()
                    except psutil.NoSuchProcess: pass
            except psutil.NoSuchProcess: pass
            process.kill(); process.wait(timeout=10)


def issue_from_software(control, software, *, source_root=None):
    """Called only after run_software seals an executed, unchanged four-suite run."""
    from .acceptance import _seal
    context = verify_software(control.store, software, source_root=source_root)
    io_fields, io_reasons = None, ['CURRENT_SOURCE_BOUND_136_MIB_IO_REPORT']
    if not _coverage(context, METADATA_CASES[16])[1]:
        try: io_fields, io_reasons = _run_large_io(context, control.store)
        except (Fault,OSError,ValueError) as error:
            io_reasons = [getattr(error,'code','ACCEPTANCE_IO_RUN_FAILED')]
        # A successful subprocess cannot bless source/tests edited while it ran.
        context = verify_software(control.store, software, source_root=source_root)
    issued, boundaries = [], []
    for number, selectors in {**REQUIRED_CASES, **METADATA_CASES}.items():
        body = _body(context, number, selectors)
        body['passed'] = number in BOUNDARY_ONLY_REQUIREMENTS and body['softwareBoundaryPassed']
        if number == 16:
            if io_fields and body['softwareBoundaryPassed']:
                body.update(**io_fields, kind='local_io_and_software_requirement',
                    claimScope=IO_CLAIM_SCOPE, passed=True, additionalEvidenceRequired=[])
            else:
                body['additionalEvidenceRequired'] = io_reasons
        elif number not in BOUNDARY_ONLY_REQUIREMENTS:
            body['additionalEvidenceRequired'] = ['ACTUAL_MODEL_OS_OR_TRANSITION_EVIDENCE']
        boundary = _seal(control.store, 'foundation_requirement_software_boundary_' + str(number),
                         body, digest([software['id'], number, CLAIM_SCOPE]))
        boundaries.append(boundary['id'])
        if body['passed']:
            record = _seal(control.store, 'foundation_requirement_acceptance_' + str(number),
                           body, digest([software['id'], number, 'requirement']))
            issued.append(record['id'])
    return {'softwareEvidenceId': software['id'], 'issuedEvidenceIds': issued,
            'boundaryEvidenceIds': boundaries,
            'actualModelQualityPassed': False, 'nasModified': False}


def verify_requirement(store, record, software, *, source_root=None, context=None):
    """No generic green suite or unrelated sealed record can satisfy a boundary."""
    from .acceptance import verify_evidence
    context = context or verify_software(store, software, source_root=source_root)
    verify_evidence(record, expected_source=context['source']['hash'])
    number = record.get('requirementNumber')
    if number == 16:
        expected = _io_body(context, record)
        require(record.get('passed') is True and expected['softwareBoundaryPassed']
            and all(record.get(key) == value for key,value in expected.items()),
            'ACCEPTANCE_REQUIREMENT_BINDING_INVALID', status=409)
        return record
    require(number in BOUNDARY_ONLY_REQUIREMENTS, 'ACCEPTANCE_ADDITIONAL_EVIDENCE_REQUIRED', status=409)
    expected = _body(context, number, REQUIRED_CASES[number])
    require(record.get('passed') is True and expected['softwareBoundaryPassed']
        and all(record.get(key) == value for key, value in expected.items()),
        'ACCEPTANCE_REQUIREMENT_BINDING_INVALID', status=409)
    return record


def verify_boundary(store, record, software, *, source_root=None, context=None):
    from .acceptance import verify_evidence
    context = context or verify_software(store, software, source_root=source_root)
    verify_evidence(record, expected_source=context['source']['hash'])
    number = record.get('requirementNumber')
    selectors = {**REQUIRED_CASES, **METADATA_CASES}.get(number)
    require(selectors, 'ACCEPTANCE_REQUIREMENT_BINDING_INVALID', status=409)
    expected = _io_body(context, record) if number == 16 and record.get('passed') is True else _body(context, number, selectors)
    require(all(record.get(key) == value for key, value in expected.items()),
            'ACCEPTANCE_REQUIREMENT_BINDING_INVALID', status=409)
    return record


def _io_body(context, record):
    require(record.get('claimScope') == IO_CLAIM_SCOPE, 'ACCEPTANCE_IO_REPORT_REQUIRED', status=409)
    expected = _body(context,16,METADATA_CASES[16])
    checksum = record.get('ioReportFileHash')
    require(isinstance(checksum,str) and re.fullmatch('[0-9a-f]{64}',checksum), 'ACCEPTANCE_IO_REPORT_REQUIRED')
    cache = context.get('_validatedIo')
    if not cache or cache.get('ioReportFileHash') != checksum:
        cache = _verify_io_report(context, expected_file_hash=checksum)
        context['_validatedIo'] = cache
    expected.update(**cache,kind='local_io_and_software_requirement',claimScope=IO_CLAIM_SCOPE,
                    passed=True,additionalEvidenceRequired=[])
    return expected
