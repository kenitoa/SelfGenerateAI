"""Opted-in source approval and bounded preparation on existing RAG queues.

Nothing here fetches a URL, executes source text, changes a quality decision, or
weakens corpus minimums. RAG's existing workers own collection and retry limits.
"""
from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime, timezone, timedelta
import time

from LLM.core import digest, now

SUPPORTED = ('upload', 'web_document', 'git_repository')
AUTHORIZATION_REASON = '사용자가 학습하기에서 활성화한 자동 자료 승인 정책에 따라 학습 이용을 허용합니다.'


def eligible_source(source):
    """Explicit exclusions remain authoritative even under automatic approval."""
    return (source.get('kind') in SUPPORTED and not source.get('archivedAt')
            and not source.get('upstreamDeleted') and source.get('origin') != 'model_self_code'
            and source.get('trainingPolicy', 'unknown') in ('unknown', 'allowed'))


def _epoch(value):
    try:
        stamp = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        if stamp.tzinfo is None:stamp = stamp.replace(tzinfo=timezone.utc)
        return stamp.timestamp()
    except (ValueError, TypeError, OverflowError):return 0.


def _interval(store):
    from RAG.Pre.crawling.service import DEFAULTS
    config = store.setting('crawler', {}) or {}
    minutes = config.get('intervalMinutes', DEFAULTS['intervalMinutes'])
    if type(minutes) is not int:minutes = DEFAULTS['intervalMinutes']
    return max(300, min(86400, minutes * 60))


def _daily_limit(store, stamp, config):
    # CrawlerService.process_job accounts pages by Korean calendar date, not a
    # rolling 24-hour window. Use that exact boundary for automatic retries.
    local = datetime.fromtimestamp(stamp, timezone(timedelta(hours=9)))
    totals = store.setting('crawlerDay:' + local.date().isoformat(), {}) or {}
    reset = datetime.combine(local.date() + timedelta(days=1), datetime.min.time(), local.tzinfo).timestamp()
    pages, limit = totals.get('pages', 0), config['dailyPages']
    return {'pages':pages, 'limit':limit, 'remaining':max(0, limit-pages),
            'resetAt':reset, 'timezone':'Asia/Seoul'}


def _source_collection(store, source, stamp):
    """Queue at most one refresh per locator/time window; preserve active retries."""
    if source['kind'] == 'upload' or source.get('crawlerManaged'):
        return None, None, None
    if source.get('collectionEnabled') is False:
        return None, None, 'SOURCE_COLLECTION_DISABLED'
    # A user-created schedule owns this source's cadence, including a pause.
    schedules = [row for row in store.list('schedule') if row.get('kind') == 'source_sync'
                 and row.get('payload', {}).get('sourceId', row.get('sourceId')) == source['id']]
    if schedules:
        due = [_epoch(row.get('nextRunAt')) for row in schedules if row.get('enabled')]
        return None, min(due) if due else None, None
    signature = digest({key:source.get(key) for key in ('kind', 'locator', 'url', 'includePaths', 'excludePaths')})
    setting = 'foundationAutoCollection:' + source['id']
    interval = _interval(store)
    with store.tx() as db:
        active = db.execute("SELECT id FROM jobs WHERE kind='source_sync' AND json_extract(payload,'$.sourceId')=? "
                            "AND state IN ('queued','running','retry_wait','cancelling') LIMIT 1", (source['id'],)).fetchone()
        if active:return store.job(active['id']), None, None
        previous = store.setting(setting, {}) or {}
        if previous.get('signature') == signature and previous.get('nextAttemptAt', 0) > stamp:
            return None, previous['nextAttemptAt'], None
        latest = db.execute("SELECT id,state,updated_at,error FROM jobs WHERE kind='source_sync' "
                            "AND json_extract(payload,'$.sourceId')=? ORDER BY created_at DESC,id DESC LIMIT 1", (source['id'],)).fetchone()
        if latest:
            job = store.job(latest['id'])
            error = (job.get('error') or {}).get('code', '')
            cooldown = max(interval, 3600) if error in ('CRAWL_RATE_LIMIT', 'SOURCE_RATE_LIMIT', 'RATE_LIMIT') else interval
            # Existing manual and scheduled jobs are also respected; a learner
            # must not turn a completed/failed collect into a tight retry loop.
            due = _epoch(job.get('updatedAt')) + cooldown
            if due > stamp:return None, due, None
        due = max(stamp + interval, (int(stamp // interval) + 1) * interval)
        key = 'foundation-auto-source:' + source['id'] + ':' + signature[:20] + ':' + str(int(stamp // interval))
        job = store.enqueue('source_sync', {'sourceId':source['id'], 'requestedBy':'foundation_auto_training'}, key=key, priority=30)
        store.set_setting(setting, {'signature':signature, 'jobId':job['id'], 'nextAttemptAt':due})
        return job, due, None


def _initialize_collection(store, pre):
    """An explicit start applies public collection defaults once, never on polls."""
    from RAG.Pre.crawling.service import CrawlerService, DEFAULTS
    crawler = CrawlerService(store, pre)
    config = crawler.config()
    patch = {'version':config['version']}
    if config.get('enabled') is not True:patch['enabled'] = True
    channels = bool(config['seedUrls'] or (config['feedEnabled'] and config['feedUrls'])
                    or (config['githubEnabled'] and config['githubQueries'])
                    or (config['commonCrawlEnabled'] and config['commonCrawlPatterns']))
    if not channels:patch['seedUrls'] = list(DEFAULTS['seedUrls'])
    if len(patch) > 1:
        crawler.configure(patch)
        # Re-enabling after an explicit start should not retain a stale paused
        # deadline. Discovery still checks daily quotas and failed-job cooldowns.
        if patch.get('enabled') is True:store.set_setting('crawlerNextAt', 0)


def needs_training_data(control, project=None):
    """A registered source is not evidence of a sufficiently large raw corpus."""
    if not project:return True
    identity = project['id']
    scheduler = control.store.get('foundation_scheduler', identity, {})
    if 'INSUFFICIENT_NEW_TRAIN_DATA' in scheduler.get('reasonCodes', []):return True
    corpus = control.store.get('foundation_corpus', project.get('activeCorpusId') or '', {})
    if not corpus:
        corpora = [row for row in control.store.list('foundation_corpus') if row.get('projectId') == identity
                   and row.get('purpose') != 'sft' and row.get('corpusKind') != 'sft']
        corpus = next(iter(sorted(corpora, key=lambda row:(row.get('createdAt', ''), row['id']), reverse=True)), {})
    return corpus.get('state') not in ('ready', 'verified', 'completed') or any(reason in
        ('INSUFFICIENT_CORPUS', 'NO_ELIGIBLE_SOURCE_VERSIONS') for reason in corpus.get('reasonCodes', []))


def _discover(store, stamp):
    """Use bounded public collection when existing raw data cannot train yet."""
    from RAG.Pre.crawling.service import DEFAULTS
    config = {**DEFAULTS, **(store.setting('crawler', {}) or {})}
    if config.get('enabled') is not True:return None, None, 'paused'
    interval = _interval(store)
    with store.tx() as db:
        active = db.execute("SELECT id FROM jobs WHERE kind='crawl_cycle' AND state IN ('queued','running','retry_wait','cancelling') LIMIT 1").fetchone()
        if active:return store.job(active['id']), None, 'pending'
        daily = _daily_limit(store, stamp, config)
        if daily['remaining'] == 0:
            return None, max(daily['resetAt'], store.setting('crawlerNextAt', 0) or 0), 'daily_limit'
        due = store.setting('crawlerNextAt', 0) or 0
        if due > stamp:return None, due, 'scheduled'
        # Existing failed crawl jobs retain a full configured cooldown. Their
        # own retry_wait state is handled above and must never be accelerated.
        latest = db.execute("SELECT id FROM jobs WHERE kind='crawl_cycle' ORDER BY created_at DESC,id DESC LIMIT 1").fetchone()
        if latest:
            job = store.job(latest['id'])
            if job['state'] in ('failed', 'cancelled'):
                due = _epoch(job.get('updatedAt')) + max(interval, 3600)
                if due > stamp:return None, due, 'retry_wait'
        key = 'foundation-auto-discovery:' + str(config['version']) + ':' + str(int(stamp // interval))
        job = store.enqueue('crawl_cycle', {}, key=key, priority=60)
        due = stamp + interval
        store.set_setting('crawlerNextAt', due)
        return job, due, 'queued' if job['state'] == 'queued' else job['state']


def prepare_sources(control, *, enabled=True, collect=True, needs_data=False, initialize_collection=False, approve=True):
    """Apply an explicitly enabled automatic policy, without inline network I/O."""
    result = {'sourceIds':[], 'approvedSourceIds':[], 'collectionJobIds':[], 'blockedSourceIds':[], 'reasonCodes':[],
              'collection':{'queued':0, 'pending':0, 'discoveryState':'not_needed', 'nextAttemptAt':None}}
    if enabled is not True:return result
    bridge = getattr(control, 'bridge', None)
    if bridge is None:
        result['reasonCodes'] = ['RAG_POLICY_GATEWAY_REQUIRED']
        return result
    store, pre = bridge.store, bridge.pre
    lock = getattr(bridge, 'deployment_lock', None)
    with lock if lock is not None else nullcontext():
        if collect and initialize_collection is True:_initialize_collection(store, pre)
        sources = store.list('source')
        for source in sources:
            if not eligible_source(source):
                result['blockedSourceIds'].append(source['id'])
                continue
            if not approve or source.get('trainingPolicy', 'unknown') != 'unknown':continue
            # Approve one source per transaction so unrelated authenticated
            # reads can run between approvals. Re-read inside each transaction
            # so an explicit concurrent denial/archive always stays excluded.
            with store.tx():
                source = store.get('source', source['id'])
                if not eligible_source(source):
                    result['blockedSourceIds'].append(source['id'])
                    continue
                if source.get('trainingPolicy', 'unknown') == 'unknown':
                    pre.update_source(source['id'], {'trainingPolicy':'allowed', 'reason':AUTHORIZATION_REASON})
                    result['approvedSourceIds'].append(source['id'])
        selected = [source for source in store.list('source') if eligible_source(source) and source.get('trainingPolicy') == 'allowed']
        # Export and policy auditing require the RAG writer transaction closed.
        if result['approvedSourceIds']:bridge.refresh_policy()
        result['sourceIds'] = sorted(source['id'] for source in selected)
        if collect:
            stamp = time.time(); next_attempts = []
            for source in selected:
                job, due, reason = _source_collection(store, source, stamp)
                if due:next_attempts.append(due)
                if reason:result['reasonCodes'].append(reason)
                if job:
                    result['collectionJobIds'].append(job['id'])
                    result['collection']['queued' if job['state'] == 'queued' else 'pending'] += 1
            if not selected or needs_data is True:
                from RAG.Pre.crawling.service import DEFAULTS
                result['collection']['dailyLimit'] = _daily_limit(store, stamp, {**DEFAULTS, **(store.setting('crawler', {}) or {})})
                job, due, state = _discover(store, stamp)
                result['collection']['discoveryState'] = state
                result['collection']['discoveryNextAttemptAt'] = due
                if due:next_attempts.append(due)
                if job:
                    result['collectionJobIds'].append(job['id'])
                    result['collection']['queued' if job['state'] == 'queued' else 'pending'] += 1
                if state == 'paused':result['reasonCodes'].append('CRAWLER_PAUSED')
                if state == 'daily_limit':result['reasonCodes'].append('CRAWL_DAILY_LIMIT')
            if next_attempts:result['collection']['nextAttemptAt'] = min(next_attempts)
    result['collectionJobIds'] = sorted(set(result['collectionJobIds']))
    result['blockedSourceIds'] = sorted(set(result['blockedSourceIds']))
    result['reasonCodes'] = sorted(set(result['reasonCodes']))
    return result


def refresh_project_sources(control, project):
    """Continue automatic preparation as data arrives and more raw data is needed."""
    automatic = control.store.get('foundation_auto_training', project['id'], {})
    if automatic.get('enabled') is not True:
        return project
    if project.get('policy', {}).get('automation') == 'manual':return project
    request_id = automatic.get('requestId')
    initialize = bool(request_id and request_id != automatic.get('collectionInitializedRequestId'))
    prepared = prepare_sources(control, enabled=True, collect=automatic.get('automaticCollection', True) is True,
                               needs_data=needs_training_data(control, project), initialize_collection=initialize,
                               approve=automatic.get('automaticSourceApproval') is True)
    # Approval/collection can take long enough for a concurrent Stop. Serialize
    # this short metadata-only policy update with that stop and recheck opt-in.
    # No RAG transaction, document export, network or model work is held here.
    with control.store.tx():
        automatic = control.store.get('foundation_auto_training', project['id'], {})
        fresh = control.foundation.project(project['id'])
        if automatic.get('enabled') is not True or automatic.get('requestId') != request_id:
            return fresh
        updates = {'sourcePreparation':prepared, 'sourcesCheckedAt':now()}
        if initialize:updates['collectionInitializedRequestId'] = request_id
        control.store.put('foundation_auto_training', updates, project['id'])
        if fresh.get('policy', {}).get('automation') == 'manual' or not prepared['sourceIds']:return fresh
        old = fresh['policy']
        selected = sorted(set(old.get('allowedSourceIds', [])) | set(prepared['sourceIds']))
        if selected != sorted(old.get('allowedSourceIds', [])) or old.get('enabled') is not True:
            control.foundation.configure(project['id'], {'enabled':True, 'allowedSourceIds':selected}, expected=old['version'])
            return control.foundation.project(project['id'])
        return fresh
