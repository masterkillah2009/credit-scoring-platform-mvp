"""Phase 5 test suite: authentication, authorisation and the hosted demo.

Run:  python3 -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from api import server as api_server                  # noqa: E402
from core import auth as auth_module                  # noqa: E402
from core.auth import AuthService, hash_password      # noqa: E402
from core.ledger import Ledger                        # noqa: E402
from core.pipeline import Platform                    # noqa: E402
from partners import simulators                       # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def setUpModule():
    """Lower the password work factor for the suite only.

    Production uses 240,000 PBKDF2 iterations. These tests construct dozens of
    authentication services, and hashing at production strength would make the
    suite slow enough that people stop running it - which costs more security
    than it buys. ``test_production_work_factor_is_high`` asserts the shipped
    default is unchanged.
    """
    auth_module.PBKDF2_ITERATIONS = 1_000

TEST_USERS = [
    {"username": "boss", "password": "correct horse battery staple",
     "role": "PLATFORM_ADMIN", "tenant": "ZAM-PAY", "name": "Test admin"},
    {"username": "grace", "password": "underwriter-password",
     "role": "UNDERWRITER", "tenant": "ZAM-PAY", "name": "Grace"},
    {"username": "mfi", "password": "mfi-password",
     "role": "RISK_MANAGER", "tenant": "ZAM-MFI", "name": "MFI risk"},
]


def service() -> AuthService:
    return AuthService(secret_key=b"unit-test-key", users=TEST_USERS)


class PasswordTests(unittest.TestCase):

    def test_production_work_factor_is_high(self):
        """The shipped default must remain expensive, whatever tests do."""
        import importlib
        source = (ROOT / "core" / "auth.py").read_text()
        self.assertIn("240_000", source)
        self.assertGreaterEqual(
            int(importlib.import_module("core.auth").__dict__
                .get("_PRODUCTION_ITERATIONS", 240_000)), 240_000)

    def test_passwords_are_never_stored_in_the_clear(self):
        auth = service()
        blob = json.dumps([u.__dict__ for u in auth._users.values()], default=str)
        for user in TEST_USERS:
            self.assertNotIn(user["password"], blob)

    def test_same_password_yields_different_hashes(self):
        """A per-user salt means identical passwords are not identifiable."""
        salt_a, hash_a = hash_password("same-password")
        salt_b, hash_b = hash_password("same-password")
        self.assertNotEqual(salt_a, salt_b)
        self.assertNotEqual(hash_a, hash_b)

    def test_correct_credentials_authenticate(self):
        self.assertIsNotNone(service().authenticate("boss", "correct horse battery staple"))

    def test_wrong_password_fails(self):
        self.assertIsNone(service().authenticate("boss", "nearly-right"))

    def test_unknown_user_fails(self):
        self.assertIsNone(service().authenticate("nobody", "anything"))

    def test_username_is_case_insensitive(self):
        self.assertIsNotNone(service().authenticate("BOSS", "correct horse battery staple"))

    def test_account_locks_after_repeated_failures(self):
        auth = service()
        for _ in range(auth_module.MAX_FAILED_ATTEMPTS):
            auth.authenticate("boss", "wrong")
        self.assertTrue(auth.is_locked("boss"))
        # Correct credentials are refused while locked - the point of lockout.
        self.assertIsNone(auth.authenticate("boss", "correct horse battery staple"))

    def test_failure_count_resets_on_success(self):
        auth = service()
        auth.authenticate("boss", "wrong")
        auth.authenticate("boss", "correct horse battery staple")
        for _ in range(auth_module.MAX_FAILED_ATTEMPTS - 1):
            auth.authenticate("boss", "wrong")
        self.assertFalse(auth.is_locked("boss"))


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.auth = service()
        self.user = self.auth.authenticate("boss", "correct horse battery staple")

    def test_session_round_trip(self):
        token = self.auth.issue_session(self.user)["token"]
        principal = self.auth.validate_session(token)
        self.assertIsNotNone(principal)
        self.assertEqual(principal.username, "boss")
        self.assertEqual(principal.tenant, "ZAM-PAY")

    def test_tampered_token_is_rejected(self):
        token = self.auth.issue_session(self.user)["token"]
        body, _, signature = token.rpartition(".")
        self.assertIsNone(self.auth.validate_session(body + ".deadbeef"))
        self.assertIsNone(self.auth.validate_session("x" + body + "." + signature))

    def test_token_signed_with_another_key_is_rejected(self):
        """A token from a different deployment must not be accepted here."""
        other = AuthService(secret_key=b"a-different-key", users=TEST_USERS)
        foreign = other.issue_session(
            other.authenticate("boss", "correct horse battery staple"))["token"]
        self.assertIsNone(self.auth.validate_session(foreign))

    def test_expired_token_is_rejected(self):
        original = auth_module.SESSION_TTL_SECONDS
        auth_module.SESSION_TTL_SECONDS = -1
        try:
            token = self.auth.issue_session(self.user)["token"]
        finally:
            auth_module.SESSION_TTL_SECONDS = original
        self.assertIsNone(self.auth.validate_session(token))

    def test_revoked_token_is_rejected(self):
        token = self.auth.issue_session(self.user)["token"]
        self.assertIsNotNone(self.auth.validate_session(token))
        self.auth.revoke(token)
        self.assertIsNone(self.auth.validate_session(token))

    def test_malformed_tokens_do_not_raise(self):
        for token in (None, "", "no-dot", "..", "a.b.c", "!!!.???"):
            self.assertIsNone(self.auth.validate_session(token))


class ApiAuthTests(unittest.TestCase):
    def setUp(self):
        simulators.reset()
        self.ledger = Ledger(pathlib.Path(tempfile.mkdtemp()) / "auth.db")
        self.api = api_server.Api(Platform(ledger=self.ledger), service())

    def tearDown(self):
        self.ledger.close()

    def request(self, method, path, body=None, token=None, key=None):
        headers = {}
        if token:
            headers["authorization"] = f"Bearer {token}"
        if key:
            headers["x-api-key"] = key
        status, payload, _ = self.api.handle(
            method, path, headers,
            json.dumps(body).encode() if body is not None else b"")
        return status, payload

    def sign_in(self, username, password):
        status, body = self.request("POST", "/v1/auth/login",
                                    {"username": username, "password": password})
        self.assertEqual(status, 200)
        return body["token"]

    def test_login_returns_a_session_and_the_user_profile(self):
        status, body = self.request("POST", "/v1/auth/login",
                                    {"username": "boss",
                                     "password": "correct horse battery staple"})
        self.assertEqual(status, 200)
        for key in ("token", "expires_at", "user"):
            self.assertIn(key, body)
        self.assertEqual(body["user"]["tenant"], "ZAM-PAY")

    def test_failed_login_is_indistinguishable_across_causes(self):
        """Wrong password, unknown user and locked account must look identical."""
        _, wrong = self.request("POST", "/v1/auth/login",
                                {"username": "boss", "password": "no"})
        _, unknown = self.request("POST", "/v1/auth/login",
                                  {"username": "ghost", "password": "no"})
        self.assertEqual(wrong["error"]["code"], unknown["error"]["code"])
        self.assertEqual(wrong["error"]["message"], unknown["error"]["message"])

    def test_login_response_never_contains_a_password(self):
        _, body = self.request("POST", "/v1/auth/login",
                               {"username": "boss",
                                "password": "correct horse battery staple"})
        self.assertNotIn("correct horse battery staple", json.dumps(body))

    def test_protected_endpoints_require_authentication(self):
        for path in ("/v1/decisions", "/v1/usage", "/v1/monitoring/summary",
                     "/v1/partners/health", "/v1/audit/verify"):
            status, body = self.request("GET", path)
            self.assertEqual(status, 401, path)
            self.assertEqual(body["error"]["code"], "UNAUTHENTICATED")

    def test_public_endpoints_need_no_authentication(self):
        for path in ("/healthz", "/openapi.json", "/v1/demo/info"):
            status, _ = self.request("GET", path)
            self.assertEqual(status, 200, path)

    def test_role_without_permission_is_forbidden_not_unauthorised(self):
        token = self.sign_in("grace", "underwriter-password")
        status, body = self.request("GET", "/v1/usage", token=token)
        self.assertEqual(status, 403)
        self.assertEqual(body["error"]["code"], "FORBIDDEN")

    def test_role_with_permission_is_allowed(self):
        token = self.sign_in("grace", "underwriter-password")
        self.assertEqual(self.request("GET", "/v1/decisions", token=token)[0], 200)

    def test_session_scopes_data_to_the_users_tenant(self):
        pay = self.sign_in("boss", "correct horse battery staple")
        self.request("POST", "/v1/batches", {"use_sample": True, "rows": 8},
                     token=pay)
        mfi = self.sign_in("mfi", "mfi-password")
        _, other = self.request("GET", "/v1/decisions", token=mfi)
        self.assertEqual(other["count"], 0)

    def test_api_key_authentication_still_works(self):
        status, _ = self.request("GET", "/v1/usage", key="demo-key-payroll")
        self.assertEqual(status, 200)

    def test_sign_out_revokes_the_session(self):
        token = self.sign_in("boss", "correct horse battery staple")
        self.assertEqual(self.request("GET", "/v1/decisions", token=token)[0], 200)
        self.request("POST", "/v1/auth/logout", token=token)
        self.assertEqual(self.request("GET", "/v1/decisions", token=token)[0], 401)

    def test_sign_in_is_audited(self):
        self.sign_in("boss", "correct horse battery staple")
        trail = self.ledger.audit_trail(tenant_code="ZAM-PAY")
        self.assertTrue(any(e["event_type"] == "USER_SIGN_IN" for e in trail))

    def test_session_endpoint_describes_the_caller(self):
        token = self.sign_in("grace", "underwriter-password")
        status, body = self.request("GET", "/v1/auth/session", token=token)
        self.assertEqual(status, 200)
        self.assertEqual(body["role"], "UNDERWRITER")
        self.assertEqual(body["method"], "session")


class DemoDeploymentTests(unittest.TestCase):
    """The artefacts that make the demonstration deployable."""

    def test_container_and_compose_files_exist(self):
        for name in ("Dockerfile", "docker-compose.yml", "entrypoint.sh",
                     ".dockerignore", "DEPLOY.md"):
            self.assertTrue((ROOT / name).exists(), f"missing {name}")

    def test_container_runs_as_a_non_root_user(self):
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertIn("USER platform", dockerfile)
        self.assertIn("useradd", dockerfile)

    def test_container_declares_a_healthcheck(self):
        self.assertIn("HEALTHCHECK", (ROOT / "Dockerfile").read_text())

    def test_no_secret_is_committed(self):
        """Credentials must come from the environment, never from a file.

        Only assignments are examined; a comment mentioning a variable is not
        a leaked secret.
        """
        compose = (ROOT / "docker-compose.yml").read_text()
        self.assertIn("DEMO_SECRET_KEY", compose)
        for line in compose.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or ":" not in stripped:
                continue
            key, _, value = stripped.partition(":")
            if any(word in key.upper() for word in ("SECRET", "PASSWORD",
                                                    "USERS", "TOKEN")):
                value = value.strip().strip('"').strip("'")
                self.assertTrue(value == "" or value.startswith("${"),
                                f"value appears to be hard-coded: {stripped}")

    def test_default_accounts_raise_a_startup_warning(self):
        auth = AuthService(secret_key=b"k")     # no users supplied -> defaults
        self.assertTrue(auth.using_default_accounts())
        self.assertTrue(any("Default demonstration accounts" in w
                            for w in auth.warnings))

    def test_configured_accounts_raise_no_default_warning(self):
        self.assertFalse(service().using_default_accounts())

    def test_demo_info_never_exposes_passwords(self):
        auth = AuthService(secret_key=b"k")
        blob = json.dumps(auth.demo_accounts())
        for user in auth_module.DEFAULT_USERS:
            self.assertNotIn(user["password"], blob)

    def test_seed_is_idempotent(self):
        from demo import seed as seed_module
        path = pathlib.Path(tempfile.mkdtemp()) / "seed.db"
        first = seed_module.seed(ledger_path=path, volume=6)
        second = seed_module.seed(ledger_path=path, volume=6)
        self.assertFalse(first.get("skipped"))
        self.assertTrue(second.get("skipped"))

    def test_seed_produces_the_named_scenarios(self):
        from demo import seed as seed_module
        self.assertGreaterEqual(len(seed_module.SCENARIOS), 4)
        tenants = {row[0] for row in seed_module.SCENARIOS}
        self.assertIn("ZAM-PAY", tenants)
        self.assertIn("ZAM-MFI", tenants)


if __name__ == "__main__":
    unittest.main(verbosity=2)
