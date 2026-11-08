"""Measured, shared-host admission for from-scratch workers."""
from __future__ import annotations
import os
import platform
from pathlib import Path
import shutil
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import psutil
from LLM.core import Fault, require, now, digest
from LLM.workers.resources import observe

GIB = 1024 ** 3
DEFAULT_POLICY = {
    'enabled': False, 'automation': 'always_on_idle', 'allowedSourceIds': [],
    'dailyActiveSeconds': 14400, 'sliceSeconds': 900, 'restSeconds': 120,
    'minNewTargetTokens': 65536, 'minCorpusBytes': 16 * 1024**2, 'minTrainGroups': 20,
    'maxConsecutiveFailures': 3, 'checkIntervalSeconds': 300,
    'maxSyntheticRatio': 0.0, 'quotaBytes': 20 * GIB,
    'trainingWindow': {'start': '02:00', 'end': '06:00'},
}


def machine_fingerprint():
    """Role-independent hardware identity; contains no host name or address."""
    features=[]
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.split(':',1)[0].strip() in ('model name','vendor_id','flags','Hardware'):
                features.append(line.strip())
    except OSError:features=[platform.processor()]
    try:affinity=psutil.Process().cpu_affinity()
    except (AttributeError,OSError):affinity=None
    return digest({'machine':platform.machine().lower(),'system':platform.system(),'kernel':platform.release(),
        'cpuFeatures':sorted(set(features)),'logicalCpuCount':psutil.cpu_count(),
        'hostRamBytes':psutil.virtual_memory().total,'cpuAffinity':affinity})


def isolated_network():
    """Offline library flags do not count as OS network isolation."""
    from LLM.verification.network import evidence
    return evidence()['isolated']


def foundation_budget(store, *, test_only=False):
    try:
        measured = observe(store.root)
    except (OSError, ValueError, TypeError):
        measured = {}
    measured = measured if isinstance(measured, dict) else {}
    known = all(type(measured.get(key)) is int and measured[key] >= minimum for key, minimum in
                (('totalRamBytes', 1), ('availableRamBytes', 0), ('freeDiskBytes', 0)))
    total = measured.get('totalRamBytes', 0) if known else 0
    ceiling = max(0, min(24 * GIB, total - 8 * GIB))
    cgroup = measured.get('memoryLimitBytes')
    available = min(measured.get('availableRamBytes', 0), cgroup or total) if known else 0
    reasons = [] if known else ['RESOURCE_UNAVAILABLE']
    if not test_only and known:
        if ceiling < 20 * GIB:reasons.append('AGGREGATE_MEMORY_BUDGET_UNAVAILABLE')
        if measured['availableRamBytes'] < 2 * GIB or measured.get('hostAvailableRamBytes',measured['availableRamBytes']) < 4 * GIB:reasons.append('RESOURCE_PRESSURE')
        if measured['freeDiskBytes'] < 5 * GIB:reasons.append('DISK_LOW')
    return {**measured, 'hostWorkloadBudgetBytes': ceiling, 'hostReserveBytes': 8 * GIB,
            'roleCapsBytes': {'controller': 12*GIB, 'trainer': 6*GIB, 'verifier': 2*GIB},
            'networkIsolated': isolated_network(), 'reasonCodes': reasons,
            'nasMeasured': False, 'observedAt': now(), 'availableForJobBytes': available}


def _automatic_recipe_context(project, stage, threads=1, worker_cap=6 * GIB):
    return digest({'schemaVersion': 1, 'architecture': project.get('config', 'seed-s0-11m-v1'),
                   'executionScope': project.get('executionScope', 'nas'), 'stage': stage,
                   'cpuThreads': threads, 'workerMemoryBudgetBytes': worker_cap,
                   'sliceSeconds': project.get('policy', {}).get('sliceSeconds', 900),
                   'dailyActiveSeconds': project.get('policy', {}).get('dailyActiveSeconds', 14400)})


def recommended_training_config(store, project, stage='pretrain'):
    """Read-only CPU recommendation. Only calibration can establish real fitness.

    A recipe selected at start remains pinned through calibration and training;
    fluctuating free RAM must not create an endless remeasurement loop.
    """
    from .trainer import training_config
    require(stage in ('pretrain', 'cpt', 'sft'), 'INVALID_TRAINING_OBJECTIVE')
    project = project or {}
    scope = project.get('executionScope', 'nas')
    budget = foundation_budget(store, test_only=project.get('testOnly') is True)
    worker_cap = min(6 * GIB, project.get('policy', {}).get('workerMemoryBytes', 6 * GIB))
    available = min(worker_cap, budget.get('availableForJobBytes', 0))
    try:
        logical = psutil.cpu_count()
        affinity = psutil.Process().cpu_affinity()
    except (AttributeError, OSError, ValueError):
        logical, affinity = None, None
    cpus = min(value for value in (logical, len(affinity) if affinity else None, 4)
               if type(value) is int and value > 0)
    threads = min(4, cpus) if scope == 'pc_validation' else 1
    maximum = project.get('config', {}).get('max_position_embeddings', 512)
    sequence = min(maximum, 128 if available >= 4 * GIB else 64)
    policy = project.get('policy', {})
    requested = {'sequenceLength': sequence, 'microBatchSize': 1,
                 'gradientAccumulation': 4 if sequence >= 128 else 8,
                 'learningRate': 5e-5 if stage == 'sft' else 3e-4,
                 'maxEpochs': 2 if stage == 'sft' else 3, 'cpuThreads': threads,
                 'sliceSeconds': min(900, policy.get('sliceSeconds', 900)),
                 'maxActiveSeconds': min(14400, policy.get('dailyActiveSeconds', 14400))}
    context = _automatic_recipe_context(project, stage, threads, worker_cap)
    saved = store.get('foundation_auto_recipe', str(project.get('id', '')) + ':' + stage, {}) if project.get('id') else {}
    source = 'environment_recommendation'
    if saved.get('contextHash') == context:
        try:
            candidate = training_config(saved['config'], maximum)
            # A stored PC recommendation must never increase NAS thread limits.
            require(scope == 'pc_validation' or candidate.get('cpuThreads', 1) == 1, 'NAS_CPU_THREADS_FIXED')
            requested = candidate
            source = 'saved_recommendation'
        except (Fault, KeyError, TypeError):
            pass
    config = training_config(requested, maximum)
    environment = {key: budget.get(key) for key in ('observationScope', 'totalRamBytes', 'availableRamBytes',
        'hostAvailableRamBytes', 'memoryLimitBytes', 'freeDiskBytes', 'hostWorkloadBudgetBytes', 'observedAt')}
    environment.update({'executionScope': scope, 'compute': 'cpu', 'cpuThreads': config['cpuThreads'],
        'logicalCpuCount': logical, 'workerMemoryBudgetBytes': worker_cap, 'measurementRequired': True,
        'recommendationSource': source})
    return {'config': config, 'environment': environment, 'reasonCodes': list(budget['reasonCodes']),
            'contextHash': context, 'stage': stage}


def save_recommended_training_config(store, project, stage='pretrain'):
    """Called only by an explicit start request or the already-enabled scheduler."""
    value = recommended_training_config(store, project, stage)
    if project.get('id') and 'RESOURCE_UNAVAILABLE' not in value['reasonCodes']:
        store.put('foundation_auto_recipe', {'schemaVersion': 1, 'projectId': project['id'], 'stage': stage,
            'contextHash': value['contextHash'], 'config': value['config'],
            'executionScope': project.get('executionScope', 'nas'), 'selectedAt': now()}, project['id'] + ':' + stage)
    return value


def used_today(store, project_id=None):
    day = datetime.now(ZoneInfo('Asia/Seoul')).date().isoformat()
    rows = store.rows('SELECT seconds FROM foundation_active_time WHERE day=?', (day,))
    return sum(float(row['seconds']) for row in rows)


def charge_time(store, identity, start_wall, elapsed):
    """Split a duration across Seoul midnight; retries charge actual work again."""
    remaining = max(0., float(elapsed)); cursor = float(start_wall)
    with store.tx():
        while remaining > 0:
            local = datetime.fromtimestamp(cursor, ZoneInfo('Asia/Seoul'))
            from datetime import timedelta
            following = datetime.combine(local.date()+timedelta(days=1), datetime.min.time(), ZoneInfo('Asia/Seoul')).timestamp()
            portion = min(remaining, max(.001, following-cursor))
            store.db.execute('INSERT INTO foundation_active_time(run,day,seconds) VALUES(?,?,?) ON CONFLICT(run,day) DO UPDATE SET seconds=seconds+excluded.seconds',
                             (identity, local.date().isoformat(), portion))
            cursor += portion; remaining -= portion


def admit(store, project):
    status = foundation_budget(store, test_only=project.get('testOnly') is True)
    require(not status['reasonCodes'], status['reasonCodes'][0] if status['reasonCodes'] else 'RESOURCE_PRESSURE', status=503)
    policy = project.get('policy', DEFAULT_POLICY)
    require(used_today(store) < policy.get('dailyActiveSeconds', 14400), 'DAILY_BUDGET_EXHAUSTED', status=409)
    return status
