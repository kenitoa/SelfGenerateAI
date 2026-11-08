"""Bounded deployment checks on an already initialized live control database."""
from __future__ import annotations

import sys
import time
from types import SimpleNamespace

import apsw

from LLM.core import Fault, Store, require
from LLM.verification.broker import pinned

PURPOSES = ('code', 'self_code_regression')


def check_worker(root):
    from LLM.foundation.worker_status import has_shared_worker
    store = Store.open_existing(root, read_only=True)
    try:
        require(has_shared_worker(store), 'ISOLATED_TRAINER_REQUIRED',
                'No isolated shared-volume trainer heartbeat', status=503)
    finally:
        store.close()


def _profile(store, identity):
    profile = store.get('verifier_profile', identity)
    require(profile.get('language') == 'python' and pinned(profile),
            'PINNED_SANDBOX_IMAGE_REQUIRED', status=503)
    return profile


def _profiles_ready(store):
    for purpose in PURPOSES:
        identity = store.get('setting', 'defaultVerifier:' + purpose, {}).get('profileId')
        if not identity:
            return False
        _profile(store, identity)
    fixture_id = store.get('policy', 'self-code', {}).get('fixtureId')
    if not fixture_id:
        return False
    fixture = store.get('fixture', fixture_id)
    _profile(store, fixture.get('profileId', ''))
    require(fixture.get('cases') and all(case.get('function') == 'transform' for case in fixture['cases']),
            'INVALID_SELF_CODE_FIXTURE', status=503)
    return True


def _provision(store, image):
    # The caller owns one short metadata transaction. Partial profile/settings/
    # fixture creation is rolled back together before a busy retry can occur.
    from LLM.evolution.service import SelfEvolution
    evolution = SelfEvolution(SimpleNamespace(store=store))
    for purpose in PURPOSES:
        setting = 'defaultVerifier:' + purpose
        identity = store.get('setting', setting, {}).get('profileId')
        if identity:
            profile = _profile(store, identity)
        else:
            profile = store.put('verifier_profile', {'language': 'python', 'image': image,
                'purpose': purpose, 'provisionedBy': 'nas_image_digest', 'autoPull': False})
            store.put('setting', {'profileId': profile['id']}, setting)
        # An older interrupted preparation may have saved the default profile
        # before creating its fixture. Complete that missing step as well.
        if purpose == 'self_code_regression' and not evolution.policy().get('fixtureId'):
            evolution.setup(profile['id'])
    require(_profiles_ready(store), 'VERIFIER_PROFILE_PREPARATION_FAILED', status=503)


def provision_verifier_profiles(root, image, *, timeout_seconds=60):
    require(pinned({'image': image}), 'PINNED_SANDBOX_IMAGE_REQUIRED', status=503)
    require(isinstance(timeout_seconds, (int, float)) and 0 < timeout_seconds <= 300,
            'CONFIG_INVALID', field='timeout_seconds')
    # Most deployments already have profiles. WAL readers do not need to join
    # the production writer queue merely to confirm those existing records.
    reader = Store.open_existing(root, read_only=True)
    try:
        if _profiles_ready(reader):
            return {'state': 'ready', 'changed': False}
    finally:
        reader.close()
    store = Store.open_existing(root, busy_timeout_ms=1000)
    deadline = time.monotonic() + timeout_seconds
    next_notice = time.monotonic()
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise Fault('DEPLOYMENT_DATABASE_BUSY', 'Verifier profile preparation timed out waiting for the database writer.', status=503)
            store.db.set_busy_timeout(max(1, min(1000, int(remaining * 1000))))
            try:
                with store.tx():
                    ready = _profiles_ready(store)
                    if not ready:
                        _provision(store, image)
                return {'state': 'ready', 'changed': not ready}
            except apsw.BusyError as exc:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise Fault('DEPLOYMENT_DATABASE_BUSY', 'Verifier profile preparation timed out waiting for the database writer.', status=503) from exc
                if time.monotonic() >= next_notice:
                    print('Verifier profile preparation is waiting for the database writer; retrying.', file=sys.stderr, flush=True)
                    next_notice = time.monotonic() + 5
                time.sleep(min(.2, remaining))
    finally:
        store.close()
