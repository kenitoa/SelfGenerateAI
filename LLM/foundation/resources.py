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
from LLM.core import require, now, digest
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
    measured = observe(store.root)
    total = measured['totalRamBytes']
    ceiling = max(0, min(24 * GIB, total - 8 * GIB))
    cgroup = measured.get('memoryLimitBytes')
    available = min(measured['availableRamBytes'], cgroup or total)
    reasons = []
    if not test_only:
        if ceiling < 20 * GIB:reasons.append('AGGREGATE_MEMORY_BUDGET_UNAVAILABLE')
        if measured['availableRamBytes'] < 2 * GIB or measured.get('hostAvailableRamBytes',measured['availableRamBytes']) < 4 * GIB:reasons.append('RESOURCE_PRESSURE')
        if measured['freeDiskBytes'] < 5 * GIB:reasons.append('DISK_LOW')
    return {**measured, 'hostWorkloadBudgetBytes': ceiling, 'hostReserveBytes': 8 * GIB,
            'roleCapsBytes': {'controller': 12*GIB, 'trainer': 6*GIB, 'verifier': 2*GIB},
            'networkIsolated': isolated_network(), 'reasonCodes': reasons,
            'nasMeasured': False, 'observedAt': now(), 'availableForJobBytes': available}


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
