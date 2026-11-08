"""Private, bounded shared-store RPC for the isolated verifier driver.

These tables deliberately do not use public records/jobs/events: generated source
and private execution results are never included in control-plane responses.
"""
import json
import re
import time
from LLM.core import canonical, digest, require, uid

MAX_JSON = 1024 * 1024
MAX_CODE = 64 * 1024
HEARTBEAT_SECONDS = 45
MAX_WAIT_SECONDS = 30


def pinned(profile):
    image=profile.get('image','')
    return isinstance(image,str) and bool(re.fullmatch(r'(?:[^\s]+@)?sha256:[0-9a-f]{64}', image))


class VerifierBroker:
    def __init__(self, store):
        self.store = store
        with store.tx():
            store.db.execute("""
              CREATE TABLE IF NOT EXISTS verifier_driver(
                id INTEGER PRIMARY KEY CHECK(id=1),owner TEXT,seen REAL,capabilities TEXT);
              CREATE TABLE IF NOT EXISTS verifier_requests(
                id TEXT PRIMARY KEY,state TEXT NOT NULL,created REAL,deadline REAL,
                owner TEXT,fence INTEGER NOT NULL DEFAULT 0,payload TEXT,result TEXT,error TEXT);
              CREATE INDEX IF NOT EXISTS verifier_pending ON verifier_requests(state,created);
            """)

    def heartbeat(self, owner, capabilities):
        # Only the driver calls this method; no route exposes it.
        body = {k: capabilities.get(k) for k in ('available','reasonCodes','missingLimits','engineVersion','cpuMode','observedAt')}
        with self.store.tx():
            rows=self.store.rows('SELECT owner,seen FROM verifier_driver WHERE id=1')
            if rows and rows[0]['owner']!=owner and time.time()-rows[0]['seen']<=HEARTBEAT_SECONDS:return False
            self.store.db.execute('INSERT OR REPLACE INTO verifier_driver VALUES(1,?,?,?)',
                                  (owner, time.time(), canonical(body)))
            return True

    def orphans(self, owner):
        return self.store.rows('SELECT id,owner FROM verifier_requests WHERE owner IS NOT NULL AND owner<>?', (owner,))

    def abandon(self, identity, owner):
        with self.store.tx():
            self.store.db.execute("UPDATE verifier_requests SET state='expired',owner=NULL,payload=NULL,result=NULL,error='VERIFIER_DRIVER_RESTARTED',fence=fence+1 WHERE id=? AND owner=?", (identity,owner))

    def status(self):
        return self.read_status(self.store)

    @staticmethod
    def read_status(store):
        """Observe liveness without constructing the schema-writing broker."""
        rows = store.rows('SELECT owner,seen,capabilities FROM verifier_driver WHERE id=1')
        live = bool(rows) and 0 <= time.time() - rows[0]['seen'] <= HEARTBEAT_SECONDS
        caps = json.loads(rows[0]['capabilities']) if live else {}
        return {**caps, 'alive': live, 'available': live and caps.get('available') is True,
                'transport': 'shared_store', 'reasonCodes': caps.get('reasonCodes', ['SANDBOX_UNAVAILABLE'])}

    def validate(self, payload):
        require(isinstance(payload, dict) and payload.get('kind') in ('verify','run_python','probe'), 'INVALID_VERIFIER_REQUEST')
        allowed={'kind','profileId','profileHash'}
        if payload['kind']=='verify':allowed|={'code','fixtureId','fixtureHash'}
        elif payload['kind']=='run_python':allowed|={'code','inputs'}
        require(not set(payload)-allowed,'INVALID_VERIFIER_REQUEST')
        if payload['kind']!='probe':
            code = payload.get('code')
            require(isinstance(code, str) and 0 < len(code.encode('utf-8')) <= MAX_CODE, 'SANDBOX_INPUT_TOO_LARGE')
        require(len(canonical(payload).encode()) <= MAX_JSON, 'SANDBOX_INPUT_TOO_LARGE')
        profile = self.store.get('verifier_profile', payload.get('profileId',''))
        require(pinned(profile), 'PINNED_SANDBOX_IMAGE_REQUIRED')
        require(digest(profile) == payload.get('profileHash'), 'VERIFIER_PROFILE_CHANGED', status=409)
        if payload['kind'] == 'verify':
            fixture = self.store.get('fixture', payload.get('fixtureId',''))
            require(fixture.get('profileId') == profile['id'] and digest(fixture) == payload.get('fixtureHash'),
                    'VERIFIER_FIXTURE_CHANGED', status=409)
        elif payload['kind']=='run_python':
            require(profile.get('language') == 'python' and isinstance(payload.get('inputs'),list), 'INVALID_VERIFIER_REQUEST')
        return profile

    def enqueue(self, payload, timeout=MAX_WAIT_SECONDS):
        self.validate(payload)
        require(self.status()['available'], 'SANDBOX_UNAVAILABLE', status=503)
        timeout = min(MAX_WAIT_SECONDS, max(.01, float(timeout)))
        identity = uid()
        with self.store.tx():
            self.store.db.execute("INSERT INTO verifier_requests(id,state,created,deadline,payload) VALUES(?,'queued',?,?,?)",
                                  (identity, time.time(), time.time()+timeout, canonical(payload)))
        return identity

    def _expire(self):
        self.store.db.execute("UPDATE verifier_requests SET state='expired',payload=NULL,result=NULL,error='VERIFIER_TIMEOUT',fence=fence+1 WHERE state IN ('queued','running') AND deadline<=?", (time.time(),))
        self.store.db.execute("UPDATE verifier_requests SET payload=NULL,result=NULL WHERE deadline<=? AND state NOT IN ('queued','running')", (time.time(),))

    def claim(self, owner):
        with self.store.tx():
            drivers=self.store.rows('SELECT owner,seen FROM verifier_driver WHERE id=1')
            if not drivers or drivers[0]['owner']!=owner or time.time()-drivers[0]['seen']>HEARTBEAT_SECONDS:return None
            self._expire()
            # Keep ownership until Docker cleanup acknowledges completion. A
            # restarted driver explicitly reaps old containers before release.
            if self.store.rows("SELECT id FROM verifier_requests WHERE owner IS NOT NULL"): return None
            rows = self.store.rows("SELECT * FROM verifier_requests WHERE state='queued' ORDER BY created,id LIMIT 1")
            if not rows: return None
            row = rows[0]
            self.store.db.execute("UPDATE verifier_requests SET state='running',owner=?,fence=fence+1 WHERE id=? AND state='queued'", (owner,row['id']))
            return {**row, 'owner':owner, 'fence':row['fence']+1, 'state':'running', 'payload':json.loads(row['payload'])}

    def active(self, request):
        rows = self.store.rows('SELECT state,owner,fence,deadline FROM verifier_requests WHERE id=?', (request['id'],))
        return bool(rows) and rows[0]['state']=='running' and rows[0]['owner']==request['owner'] and rows[0]['fence']==request['fence'] and rows[0]['deadline']>time.time()

    def release(self, request):
        with self.store.tx():
            self.store.db.execute('UPDATE verifier_requests SET owner=NULL WHERE id=? AND owner=?', (request['id'],request['owner']))

    def complete(self, request, result=None, error=None):
        # A timed-out client can never accept a late result, including receipts
        # produced just before cancellation reaches Docker.
        with self.store.tx():
            self._expire()
            if not self.active(request): return False
            require(result is None or len(canonical(result).encode())<=MAX_JSON, 'VERIFIER_OUTPUT_TOO_LARGE')
            self.store.db.execute("UPDATE verifier_requests SET state=?,payload=NULL,result=?,error=? WHERE id=? AND state='running' AND owner=? AND fence=?",
                ('failed' if error else 'complete', canonical(result) if result is not None else None,
                 error, request['id'],request['owner'],request['fence']))
            return True

    def cancel(self, identity):
        with self.store.tx():
            self.store.db.execute("UPDATE verifier_requests SET state='cancelled',payload=NULL,result=NULL,error='VERIFIER_TIMEOUT',fence=fence+1 WHERE id=? AND state IN ('queued','running')", (identity,))

    def wait(self, identity, timeout=MAX_WAIT_SECONDS):
        deadline = time.monotonic()+min(MAX_WAIT_SECONDS,max(.01,float(timeout)))
        try:
            while time.monotonic()<deadline:
                with self.store.tx():
                    self._expire()
                    rows = self.store.rows('SELECT state,result,error,deadline FROM verifier_requests WHERE id=?', (identity,))
                    require(bool(rows), 'VERIFIER_REQUEST_NOT_FOUND')
                    row = rows[0]
                    if row['state']=='complete':
                        require(row['deadline']>time.time(), 'VERIFIER_TIMEOUT', status=503)
                        result = json.loads(row['result'])
                        self.store.db.execute('UPDATE verifier_requests SET result=NULL WHERE id=?', (identity,))
                        return result
                    if row['state'] in ('failed','expired','cancelled'):
                        require(False, row['error'] or 'VERIFIER_FAILED', status=503)
                time.sleep(.05)
            require(False, 'VERIFIER_TIMEOUT', status=503)
        finally:
            self.cancel(identity)

    def request(self, payload, timeout=MAX_WAIT_SECONDS):
        return self.wait(self.enqueue(payload,timeout),timeout)

    def verify(self, code, fixture):
        profile=self.store.get('verifier_profile',fixture.get('profileId',''))
        payload={'kind':'verify','code':code,'fixtureId':fixture.get('id'), 'fixtureHash':digest(fixture),
                 'profileId':profile['id'],'profileHash':digest(profile)}
        result=self.request(payload)
        receipt=self.store.get('verifier_receipt',result.get('receiptId',''))
        require(receipt.get('codeHash')==digest(code.encode()) and receipt.get('fixtureHash')==digest(fixture)
                and receipt.get('environmentDigest')==profile['image'] and 'output' not in receipt, 'VERIFIER_RECEIPT_MISMATCH')
        return receipt

    def run_python(self, code, inputs, profile):
        result=self.request({'kind':'run_python','code':code,'inputs':inputs,'profileId':profile.get('id'),'profileHash':digest(profile)})
        values=result.get('values')
        require(isinstance(values,list) and len(values)==len(inputs), 'SANDBOX_OUTPUT_INVALID')
        return values

    def probe(self, profile):
        result=self.request({'kind':'probe','profileId':profile.get('id'),'profileHash':digest(profile)})
        receipt=self.store.get('verifier_probe',result.get('receiptId',''))
        require(receipt.get('profileHash')==digest(profile) and receipt.get('environmentDigest')==profile['image']
                and receipt.get('expiresAt',0)>time.time(),'VERIFIER_PROBE_MISMATCH')
        return receipt
