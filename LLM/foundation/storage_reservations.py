"""Durable token-cache quota batches; expired writers can never publish.

An expired idle reservation is reclaimable. Expired in-flight work stays charged
until its staging files are explicitly reconciled, so a crashed process cannot
silently free bytes which still exist on disk.
"""
import os
import threading
import time
from LLM.core import require,uid

_BATCHES={}
_OWNER=uid()
BATCH_BYTES=32*1024**2
LEASE_SECONDS=120


def begin(store,project,required_bytes,purpose='token-cache'):
    require(type(required_bytes) is int and required_bytes>0,'QUOTA_ESTIMATE_INVALID')
    owner=f'{_OWNER}:{os.getpid()}:{threading.get_ident()}'
    key=(str(store.root),project['id'],owner,purpose)
    with store.tx():
        old=store.get('foundation_storage_reservation',_BATCHES.get(key,''),{})
        policy_version=store.get('foundation_policy',project['id'],{}).get('version',0)
        if old and old.get('state')=='active' and not old.get('activeWorkId') and old.get('expiresAt',0)>time.time() and old.get('remainingBytes',0)>=required_bytes and old.get('policyVersion',0)==policy_version:
            return _work(store,old,owner,required_bytes)
        if old and old.get('state')=='active' and not old.get('activeWorkId'):
            store.put('foundation_storage_reservation',{'state':'released','remainingBytes':0,'fence':old.get('fence',0)+1},old['id'],expected=old['version'])
    from .retention import admit
    def allocate(quota):
        amount=min(max(BATCH_BYTES,required_bytes),quota['availableBytes'])
        reservation=store.put('foundation_storage_reservation',{'projectId':project['id'],'owner':owner,'purpose':purpose,
            'policyVersion':store.get('foundation_policy',project['id'],{}).get('version',0),
            'remainingBytes':amount,'state':'active','fence':1,'expiresAt':time.time()+LEASE_SECONDS,'activeWorkId':None})
        _BATCHES[key]=reservation['id']
        return _work(store,reservation,owner,required_bytes)
    return admit(store,project,required_bytes,_allocate=allocate)


def _work(store,reservation,owner,required_bytes):
    work_id=uid()
    current=store.put('foundation_storage_reservation',{'activeWorkId':work_id,'activeEstimateBytes':required_bytes,
        'expiresAt':time.time()+LEASE_SECONDS},reservation['id'],expected=reservation['version'])
    return {'reservationId':current['id'],'owner':owner,'fence':current['fence'],'workId':work_id,'estimate':required_bytes}


def assert_work(store,work):
    current=store.get('foundation_storage_reservation',work['reservationId'])
    require(current.get('state')=='active' and current.get('owner')==work['owner'] and current.get('fence')==work['fence']
        and current.get('activeWorkId')==work['workId'] and current.get('expiresAt',0)>time.time(),
        'STORAGE_RESERVATION_EXPIRED',status=409)
    return current


def commit(store,work,actual_bytes):
    current=assert_work(store,work)
    require(type(actual_bytes) is int and 0<=actual_bytes<=work['estimate'] and actual_bytes<=current['remainingBytes'],
        'STORAGE_RESERVATION_SIZE_MISMATCH')
    return store.put('foundation_storage_reservation',{'remainingBytes':current['remainingBytes']-actual_bytes,
        'activeWorkId':None,'activeEstimateBytes':0,'expiresAt':time.time()+LEASE_SECONDS},current['id'],expected=current['version'])


def abort(store,work,*,files_removed):
    with store.tx():
        current=store.get('foundation_storage_reservation',work['reservationId'],{})
        if current.get('owner')!=work['owner'] or current.get('activeWorkId')!=work['workId']:return
        store.put('foundation_storage_reservation',{'state':'active' if files_removed else 'held',
            'activeWorkId':None if files_removed else work['workId'],'reasonCodes':[] if files_removed else ['STORAGE_RECONCILIATION_REQUIRED']},
            current['id'],expected=current['version'])
