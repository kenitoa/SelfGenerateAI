"""Explicit local foundation origins; derived growth never becomes random-init."""
from .schemas import FOUNDATION_ORIGINS


def assert_project_available(store,project):
    from LLM.core import require
    seen=set();current=project
    while current:
        require(current.get('id') not in seen,'FOUNDATION_PROVENANCE_MISMATCH',status=409)
        seen.add(current['id'])
        require(current.get('state')!='revoked' and not current.get('revocationPending'),'POLICY_REVOKED',status=409)
        parent=current.get('sourceProjectId')
        current=store.get('foundation_project',parent) if parent else None
    return project
