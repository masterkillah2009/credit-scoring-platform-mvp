"""Authentication, sessions and role-based access for the hosted demonstration.

Scope note. This is demonstration-grade authentication, deliberately built on
the standard library so the service deploys with no dependencies. It is
sufficient for a hosted demo carrying synthetic data on the public internet.
It is **not** the production design: IPSRS FR-ADM-02 requires OAuth 2.0/OIDC
with MFA and an external identity provider (Keycloak), and clause §11 of the
technical specification says so explicitly. The gap is listed in DEPLOY.md
under "what this is not".

What it does provide:

  * passwords stored as PBKDF2-HMAC-SHA256 with a per-user salt, never plainly
  * stateless HMAC-signed session tokens with an expiry and a revocation list
  * constant-time comparison on every secret check
  * per-account throttling with lockout after repeated failures
  * roles mapped to permissions, and every user bound to exactly one tenant
  * tokens carried in an Authorization header rather than a cookie, so the
    demo has no cross-site request forgery surface at all

Configuration is by environment variable so that no credential is committed:

  DEMO_SECRET_KEY   signing key for session tokens (generated if absent)
  DEMO_USERS        JSON list of users (defaults used if absent, with a warning)
  DEMO_SESSION_TTL  session lifetime in seconds (default 8 hours)
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

#: Password hashing work factor. Deliberately expensive in production; a test
#: suite that constructs many services would otherwise spend its whole runtime
#: hashing, so it is overridable. Lowering it in a deployment would be a
#: security defect, which is why the default is the production value and the
#: override is by explicit environment variable.
PBKDF2_ITERATIONS = int(os.environ.get("DEMO_PBKDF2_ITERATIONS", 240_000))
SESSION_TTL_SECONDS = int(os.environ.get("DEMO_SESSION_TTL", 8 * 60 * 60))
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_SECONDS = 300

#: Role -> permissions. A permission is checked by the API before a handler
#: runs; a role without the permission never reaches the handler.
ROLE_PERMISSIONS: dict[str, set[str]] = {
    "PLATFORM_ADMIN": {"decide", "read_decisions", "read_audit", "read_usage",
                       "read_monitoring", "read_partners", "run_batch",
                       "manage_config"},
    "RISK_MANAGER":   {"decide", "read_decisions", "read_monitoring",
                       "read_partners", "run_batch"},
    "UNDERWRITER":    {"decide", "read_decisions"},
    "COMPLIANCE":     {"read_decisions", "read_audit", "read_monitoring"},
    "FINANCE":        {"read_usage", "read_monitoring"},
    "VIEWER":         {"read_decisions", "read_monitoring"},
}

#: Default demonstration accounts. Passwords are intentionally obvious: this
#: account set exists to be shown to a prospect, and a warning is emitted at
#: startup if it is still in use. Override with DEMO_USERS in any deployment
#: that is not a throwaway.
DEFAULT_USERS = [
    {"username": "admin", "password": "demo-admin-2026",
     "role": "PLATFORM_ADMIN", "tenant": "ZAM-PAY", "name": "Platform administrator"},
    {"username": "risk", "password": "demo-risk-2026",
     "role": "RISK_MANAGER", "tenant": "ZAM-PAY", "name": "Kabwe (credit risk)"},
    {"username": "underwriter", "password": "demo-uw-2026",
     "role": "UNDERWRITER", "tenant": "ZAM-PAY", "name": "Grace (underwriter)"},
    {"username": "compliance", "password": "demo-comp-2026",
     "role": "COMPLIANCE", "tenant": "ZAM-PAY", "name": "Chileshe (compliance)"},
    {"username": "mfi", "password": "demo-mfi-2026",
     "role": "RISK_MANAGER", "tenant": "ZAM-MFI", "name": "Kabwata MFI risk"},
]


@dataclass(frozen=True)
class User:
    username: str
    display_name: str
    role: str
    tenant: str
    salt: bytes
    password_hash: bytes

    @property
    def permissions(self) -> set[str]:
        return ROLE_PERMISSIONS.get(self.role, set())


@dataclass
class Principal:
    """An authenticated caller, however they authenticated."""

    username: str
    display_name: str
    role: str
    tenant: str
    method: str                      # session | api_key
    permissions: set[str] = field(default_factory=set)

    def may(self, permission: str) -> bool:
        return permission in self.permissions


def hash_password(password: str, salt: Optional[bytes] = None) -> tuple[bytes, bytes]:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt,
                                 PBKDF2_ITERATIONS)
    return salt, digest


class AuthService:
    """User directory, password verification, sessions and throttling."""

    def __init__(self, *, secret_key: Optional[bytes] = None,
                 users: Optional[list[dict]] = None):
        self.warnings: list[str] = []

        key = secret_key or (os.environ.get("DEMO_SECRET_KEY") or "").encode()
        if not key:
            key = secrets.token_bytes(32)
            self.warnings.append(
                "DEMO_SECRET_KEY is not set: a random key was generated, so "
                "sessions will not survive a restart and will not work across "
                "multiple instances.")
        self._secret = key

        configured = users
        if configured is None:
            raw = os.environ.get("DEMO_USERS")
            if raw:
                try:
                    configured = json.loads(raw)
                except json.JSONDecodeError:
                    configured = None
                    self.warnings.append(
                        "DEMO_USERS could not be parsed as JSON; the default "
                        "demonstration accounts are in use.")
        if configured is None:
            configured = DEFAULT_USERS
            self.warnings.append(
                "Default demonstration accounts are in use. Set DEMO_USERS "
                "before exposing this service to anyone outside a demo.")

        self._users: dict[str, User] = {}
        for entry in configured:
            salt, digest = hash_password(entry["password"])
            username = entry["username"].lower()
            self._users[username] = User(
                username=username,
                display_name=entry.get("name", entry["username"]),
                role=entry.get("role", "VIEWER"),
                tenant=entry["tenant"],
                salt=salt, password_hash=digest)

        self._failures: dict[str, tuple[int, float]] = {}
        self._revoked: set[str] = set()
        self._lock = threading.Lock()

    # -- passwords --------------------------------------------------------- #
    def authenticate(self, username: str, password: str) -> Optional[User]:
        """Verify credentials, with throttling and constant-time comparison."""
        username = (username or "").strip().lower()
        with self._lock:
            attempts, locked_until = self._failures.get(username, (0, 0.0))
            if locked_until > time.time():
                return None

        user = self._users.get(username)
        # Always perform a hash so that a missing user costs the same time as a
        # wrong password; otherwise response timing enumerates valid usernames.
        salt = user.salt if user else b"\x00" * 16
        _, candidate = hash_password(password or "", salt)
        ok = bool(user) and hmac.compare_digest(candidate, user.password_hash)

        with self._lock:
            if ok:
                self._failures.pop(username, None)
            else:
                attempts += 1
                until = (time.time() + LOCKOUT_SECONDS
                         if attempts >= MAX_FAILED_ATTEMPTS else 0.0)
                self._failures[username] = (attempts, until)
        return user if ok else None

    def is_locked(self, username: str) -> bool:
        with self._lock:
            _, until = self._failures.get((username or "").lower(), (0, 0.0))
        return until > time.time()

    # -- sessions ---------------------------------------------------------- #
    def issue_session(self, user: User) -> dict[str, Any]:
        expires_at = int(time.time()) + SESSION_TTL_SECONDS
        payload = {
            "u": user.username, "r": user.role, "t": user.tenant,
            "exp": expires_at, "n": secrets.token_urlsafe(8),
        }
        body = base64.urlsafe_b64encode(
            json.dumps(payload, separators=(",", ":")).encode()).decode().rstrip("=")
        signature = hmac.new(self._secret, body.encode(), hashlib.sha256).hexdigest()
        return {
            "token": f"{body}.{signature}",
            "expires_at": expires_at,
            "user": {"username": user.username, "name": user.display_name,
                     "role": user.role, "tenant": user.tenant,
                     "permissions": sorted(user.permissions)},
        }

    def validate_session(self, token: Optional[str]) -> Optional[Principal]:
        if not token or "." not in token:
            return None
        body, _, signature = token.rpartition(".")
        expected = hmac.new(self._secret, body.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            return None
        try:
            padding = "=" * (-len(body) % 4)
            payload = json.loads(base64.urlsafe_b64decode(body + padding))
        except Exception:
            return None
        if payload.get("exp", 0) < time.time():
            return None
        with self._lock:
            if token in self._revoked:
                return None
        user = self._users.get(payload.get("u", ""))
        if user is None:
            return None
        return Principal(username=user.username, display_name=user.display_name,
                         role=user.role, tenant=user.tenant, method="session",
                         permissions=set(user.permissions))

    def revoke(self, token: str) -> None:
        with self._lock:
            self._revoked.add(token)
            if len(self._revoked) > 10_000:      # bounded; tokens expire anyway
                self._revoked.clear()

    # -- introspection ------------------------------------------------------ #
    def demo_accounts(self) -> list[dict[str, str]]:
        """Account list for the login screen - never includes passwords."""
        return [{"username": u.username, "name": u.display_name,
                 "role": u.role, "tenant": u.tenant}
                for u in self._users.values()]

    def using_default_accounts(self) -> bool:
        return any("Default demonstration accounts" in w for w in self.warnings)
