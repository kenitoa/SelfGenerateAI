"""Read-only shared-volume trainer heartbeat validation."""
from datetime import datetime, timezone


def has_shared_worker(store, execution_scope='nas', max_age=60):
    current = datetime.now(timezone.utc)
    for record in store.list('worker'):
        try:
            age = (current-datetime.fromisoformat(record['observedAt'].replace('Z','+00:00'))).total_seconds()
        except (KeyError, ValueError, TypeError, AttributeError):
            continue
        capabilities = record.get('capabilities')
        if (0 <= age < max_age and record.get('executionScope','nas') == execution_scope
                and record.get('state') == 'online' and record.get('foundationSharedVolume') is True
                and isinstance(capabilities, list) and 'foundation_pretrain' in capabilities):
            return True
    return False
