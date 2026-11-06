"""Single-owner sessions and same-origin request protections."""
import hashlib
import ipaddress
import secrets
import time
from urllib.parse import urlparse

from argon2 import PasswordHasher
from argon2.exceptions import VerificationError

from RAG.core import DomainError, now


class Auth:
    def __init__(self, store, local_dev=False):
        self.store, self.local_dev = store, local_dev
        self.passwords = PasswordHasher()
        self.attempts = {}

    def set_password(self, password):
        if len(password) < 12:
            raise DomainError("WEAK_PASSWORD", "관리자 비밀번호는 12자 이상이어야 합니다.", 422)
        self.store.set_setting("admin_password_hash", self.passwords.hash(password))
        for session in self.store.list("session"):
            self.store.delete("session", session["id"])

    def is_local(self, request):
        try:
            peer = ipaddress.ip_address(request.client.host)
            host = urlparse("http://" + request.headers.get("host", "")).hostname
            return self.local_dev and peer.is_loopback and host in ("localhost", "127.0.0.1", "::1")
        except (ValueError, AttributeError):
            return False

    def session(self, request):
        token = request.cookies.get("seed_session", "")
        if not token:
            return None
        identity = hashlib.sha256(token.encode()).hexdigest()
        try:
            value = self.store.get("session", identity)
        except DomainError:
            return None
        if value["expiresAt"] < time.time():
            self.store.delete("session", identity)
            return None
        if value.get("local") and not self.is_local(request):
            return None
        return value

    def create_session(self, response, *, local=False, secure=False):
        token, csrf = secrets.token_urlsafe(40), secrets.token_urlsafe(32)
        identity = hashlib.sha256(token.encode()).hexdigest()
        result = self.store.put("session", {"csrfToken": csrf, "expiresAt": time.time()+12*3600, "local": local}, id=identity)
        response.set_cookie("seed_session", token, httponly=True, secure=secure, samesite="strict", max_age=12*3600, path="/")
        return result

    def origin(self, request):
        origin = request.headers.get("origin")
        if not origin:
            # Non-browser clients still need the unpredictable CSRF token.
            return
        parsed = urlparse(origin)
        expected = request.headers.get("host", "")
        allowed = self.store.setting("settings", {}).get("allowedOrigins", [])
        local = {"http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:8000", "http://127.0.0.1:8000"}
        if parsed.scheme not in ("http", "https") or (parsed.netloc != expected and origin not in allowed and not (self.is_local(request) and origin in local)):
            raise DomainError("ORIGIN_FORBIDDEN", "허용되지 않은 요청 출처입니다.", 403)

    def authorize(self, request):
        session = self.session(request)
        if not session:
            raise DomainError("AUTH_REQUIRED", "관리자 로그인이 필요합니다.", 401)
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            self.origin(request)
            if not secrets.compare_digest(request.headers.get("x-csrf-token", ""), session["csrfToken"]):
                raise DomainError("CSRF_INVALID", "보안 토큰을 갱신하세요.", 403)
        return session

    def login(self, request, response, password):
        self.origin(request)
        peer=request.client.host
        recent=[t for t in self.attempts.get(peer,[]) if t>time.time()-300]
        if len(recent)>=8:
            raise DomainError("LOGIN_RATE_LIMIT", "잠시 후 다시 로그인하세요.", 429)
        stored=self.store.setting("admin_password_hash")
        if not stored:
            raise DomainError("ADMIN_NOT_CONFIGURED", "서버에서 관리자 계정을 먼저 초기화하세요.", 409)
        try:
            self.passwords.verify(stored,password)
        except (VerificationError, TypeError):
            self.attempts[peer]=recent+[time.time()]
            raise DomainError("INVALID_CREDENTIALS", "비밀번호를 확인하세요.", 401)
        self.attempts.pop(peer,None)
        session=self.create_session(response,secure=request.url.scheme=="https")
        return {"authenticated":True,"csrfToken":session["csrfToken"],"localAccess":False,"bootstrapRequired":False}
