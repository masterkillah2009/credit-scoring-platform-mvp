"""Phase 3 test suite: connectors, ledger, API.

Run:  python3 -m unittest discover -s tests -v

Most tests drive the ``Api`` object directly (fast, no sockets); one drives a
real HTTP server end to end to prove the transport works.
"""
from __future__ import annotations

import json
import pathlib
import socket
import sqlite3
import sys
import tempfile
import threading
import unittest
import urllib.request
from decimal import Decimal

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from api import server as api_server                     # noqa: E402
from core import config                                   # noqa: E402
from core.ledger import Ledger                            # noqa: E402
from core.pipeline import Platform                        # noqa: E402
from partners import simulators                           # noqa: E402
from partners.connector import Connector, ConnectorRegistry  # noqa: E402

APPLICATION = {
    "national_id": "384756/61/1",
    "full_name": "Chanda Mwale",
    "date_of_birth": "1990-03-14",
    "application_date": "2026-07-19",
    "employer_code": "MOE-LSK-01",
    "device_id": "device-77af",
    "requested_amount": 20000.0,
    "tenor_months": 18,
    "declared_monthly_income": 12800.0,
    "declared_monthly_expenses": 4200.0,
    "existing_monthly_debt_service": 700.0,
    "dependants": 1,
    "consent": {"credit_bureau_enquiry": True, "automated_decisioning": True,
                "payroll_verification": True},
}


def temp_ledger() -> Ledger:
    return Ledger(pathlib.Path(tempfile.mkdtemp()) / "test_ledger.db")


class ApiTestCase(unittest.TestCase):
    """Base: a fresh platform and ledger per test, simulators reset."""

    def setUp(self):
        simulators.reset()
        self.ledger = temp_ledger()
        self.platform = Platform(ledger=self.ledger)
        self.api = api_server.Api(self.platform)

    def tearDown(self):
        simulators.reset()
        self.ledger.close()

    def request(self, method: str, path: str, *, body: dict | None = None,
                key: str = "demo-key-payroll",
                headers: dict | None = None) -> tuple[int, dict]:
        request_headers = {"x-api-key": key} if key else {}
        request_headers.update({k.lower(): v for k, v in (headers or {}).items()})
        status, payload, _ = self.api.handle(
            method, path, request_headers,
            json.dumps(body).encode() if body is not None else b"")
        return status, payload

    def decide(self, application: dict | None = None, **kwargs):
        return self.request("POST", "/v1/applications/decision",
                            body={"application": application or dict(APPLICATION)},
                            **kwargs)


class AuthenticationTests(ApiTestCase):
    """IPSRS FR-API-02, FR-ADM-06."""

    def test_missing_api_key_is_rejected(self):
        status, payload = self.decide(key="")
        self.assertEqual(status, 401)
        self.assertEqual(payload["error"]["code"], "UNAUTHENTICATED")

    def test_unknown_api_key_is_rejected(self):
        status, payload = self.decide(key="not-a-real-key")
        self.assertEqual(status, 401)

    def test_public_endpoints_need_no_key(self):
        for path in ("/healthz", "/openapi.json"):
            status, _ = self.request("GET", path, key="")
            self.assertEqual(status, 200, path)

    def test_errors_never_leak_internals(self):
        status, payload = self.decide(key="")
        serialised = json.dumps(payload)
        for leak in ("Traceback", "/sessions/", "sqlite", "File \""):
            self.assertNotIn(leak, serialised)
        self.assertIn("correlation_id", payload)


class ValidationTests(ApiTestCase):
    """IPSRS FR-INT-02 and the VAL-* catalogue."""

    def test_malformed_nrc_is_rejected(self):
        status, payload = self.decide(dict(APPLICATION, national_id="12345"))
        self.assertEqual(status, 400)
        fields = {item["field"] for item in payload["error"]["details"]["errors"]}
        self.assertIn("application.national_id", fields)

    def test_consent_is_required_before_processing(self):
        application = dict(APPLICATION)
        application.pop("consent")
        status, payload = self.decide(application)
        self.assertEqual(status, 400)
        fields = {item["field"] for item in payload["error"]["details"]["errors"]}
        self.assertIn("application.consent.credit_bureau_enquiry", fields)

    def test_negative_amount_is_rejected(self):
        status, _ = self.decide(dict(APPLICATION, requested_amount=-100))
        self.assertEqual(status, 400)

    def test_failed_validation_calls_no_partner_and_bills_nothing(self):
        """FR-INT-02: nothing is billed for an application that never ran."""
        self.decide(dict(APPLICATION, national_id="bad"))
        usage = self.ledger.usage(tenant_code="ZAM-PAY")
        self.assertEqual(usage["total"], "0.00")
        health = self.platform.connectors.health()
        self.assertEqual(sum(p["calls"] for p in health.values()), 0)

    def test_malformed_json_is_reported_as_such(self):
        status, payload, _ = self.api.handle(
            "POST", "/v1/applications/decision",
            {"x-api-key": "demo-key-payroll"}, b"{not json")
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "MALFORMED_JSON")

    def test_unknown_product_is_reported(self):
        status, payload = self.request(
            "POST", "/v1/applications/decision",
            body={"application": dict(APPLICATION), "product_code": "NOPE"})
        self.assertEqual(status, 404)
        self.assertEqual(payload["error"]["code"], "PRODUCT_NOT_FOUND")


class DecisionEndpointTests(ApiTestCase):
    """The core endpoint and its contract."""

    def test_decision_returns_the_full_contract(self):
        status, payload = self.decide()
        self.assertEqual(status, 200)
        for section in ("outcome", "reason_codes", "identifiers", "assessment",
                        "versions", "offer", "affordability", "trace",
                        "decided_at"):
            self.assertIn(section, payload)
        self.assertIn(payload["outcome"],
                      ("APPROVE", "DECLINE", "REFER", "INSUFFICIENT_INFORMATION"))

    def test_decision_is_persisted_and_retrievable(self):
        _, decision = self.decide()
        decision_id = decision["identifiers"]["decision_id"]
        status, stored = self.request("GET", f"/v1/decisions/{decision_id}")
        self.assertEqual(status, 200)
        self.assertEqual(stored["identifiers"]["decision_id"], decision_id)

    def test_application_history_is_returned(self):
        _, first = self.decide()
        application_id = first["identifiers"]["application_id"]
        status, history = self.request(
            "GET", f"/v1/applications/{application_id}/decisions")
        self.assertEqual(status, 200)
        self.assertGreaterEqual(history["count"], 1)

    def test_prequalification_makes_no_credit_decision(self):
        status, payload = self.request(
            "POST", "/v1/prequalification",
            body={"application": dict(APPLICATION)})
        self.assertEqual(status, 200)
        self.assertIn("eligible", payload)
        self.assertNotIn("outcome", payload)
        self.assertIn("no credit bureau enquiry", payload["basis"])

    def test_telemetry_reports_partner_latency(self):
        _, decision = self.decide()
        telemetry = decision["telemetry"]
        self.assertIn("retrieval_ms", telemetry)
        self.assertGreaterEqual(len(telemetry["partners"]), 4)


class IdempotencyTests(ApiTestCase):
    """FR-API-02: a replayed key must not create or bill a second decision."""

    def test_replay_returns_the_original_decision(self):
        _, first = self.decide(headers={"Idempotency-Key": "abc-123"})
        _, second = self.decide(headers={"Idempotency-Key": "abc-123"})
        self.assertTrue(second.get("replayed"))
        self.assertEqual(first["identifiers"]["decision_id"],
                         second["identifiers"]["decision_id"])

    def test_replay_is_not_billed_twice(self):
        self.decide(headers={"Idempotency-Key": "abc-123"})
        self.decide(headers={"Idempotency-Key": "abc-123"})
        usage = self.ledger.usage(tenant_code="ZAM-PAY")
        decisions = next(line for line in usage["lines"]
                         if line["event_type"] == "APPLICATION_DECISION")
        self.assertEqual(decisions["quantity"], 1)

    def test_different_keys_create_different_decisions(self):
        _, first = self.decide(headers={"Idempotency-Key": "key-1"})
        _, second = self.decide(headers={"Idempotency-Key": "key-2"})
        self.assertNotEqual(first["identifiers"]["decision_id"],
                            second["identifiers"]["decision_id"])


class TenantIsolationTests(ApiTestCase):
    """BR-TEN-01: absolute isolation, enforced at the API boundary."""

    def test_one_tenant_cannot_read_another_tenants_decision(self):
        _, decision = self.decide()
        decision_id = decision["identifiers"]["decision_id"]
        status, payload = self.request("GET", f"/v1/decisions/{decision_id}",
                                       key="demo-key-micro")
        self.assertEqual(status, 404)
        self.assertEqual(payload["error"]["code"], "DECISION_NOT_FOUND")

    def test_not_found_and_forbidden_are_indistinguishable(self):
        """Existence is information: both must look identical to a caller."""
        _, decision = self.decide()
        real_id = decision["identifiers"]["decision_id"]
        _, foreign = self.request("GET", f"/v1/decisions/{real_id}",
                                  key="demo-key-micro")
        _, absent = self.request("GET", "/v1/decisions/DEC-DOESNOTEXIST",
                                 key="demo-key-micro")
        self.assertEqual(foreign["error"]["code"], absent["error"]["code"])

    def test_audit_trail_is_scoped_to_the_calling_tenant(self):
        _, decision = self.decide()
        correlation_id = decision["identifiers"]["correlation_id"]
        _, own = self.request("GET", f"/v1/audit/{correlation_id}")
        _, other = self.request("GET", f"/v1/audit/{correlation_id}",
                                key="demo-key-micro")
        self.assertGreaterEqual(own["count"], 1)
        self.assertEqual(other["count"], 0)

    def test_usage_is_scoped_to_the_calling_tenant(self):
        self.decide()
        _, other = self.request("GET", "/v1/usage", key="demo-key-micro")
        self.assertEqual(other["total"], "0.00")


class ConnectorTests(ApiTestCase):
    """FR-CNX-01..04 / BR-DAT-02, BR-DAT-05."""

    def test_partner_failure_does_not_fabricate_data(self):
        simulators.configure("bureau", unavailable=True)
        _, decision = self.decide()
        self.assertFalse(decision["telemetry"]["partners"]["bureau"]["ok"])
        codes = {reason["code"] for reason in decision["reason_codes"]}
        self.assertIn("PARTNER_DATA_UNAVAILABLE", codes)

    def test_refer_degradation_policy_downgrades_an_approval(self):
        simulators.configure("bureau", unavailable=True)
        _, decision = self.decide()
        self.assertEqual(config.get_tenant("ZAM-PAY").degradation_policy, "refer")
        self.assertNotEqual(decision["outcome"], "APPROVE")

    def test_partial_degradation_policy_still_decides(self):
        """ZAM-MFI is configured to score on partial data rather than refer."""
        simulators.configure("bureau", unavailable=True)
        status, decision = self.request(
            "POST", "/v1/applications/decision",
            body={"application": dict(APPLICATION), "product_code": "MICRO_LOAN"},
            key="demo-key-micro")
        self.assertEqual(status, 200)
        self.assertIsNotNone(decision["assessment"]["score"])
        codes = {reason["code"] for reason in decision["reason_codes"]}
        self.assertIn("PARTNER_DATA_UNAVAILABLE", codes)

    def test_retries_are_attempted_then_the_circuit_opens(self):
        simulators.configure("bureau", unavailable=True)
        states = []
        for _ in range(4):
            _, decision = self.decide()
            states.append(decision["telemetry"]["partners"]["bureau"])
        self.assertGreater(states[0]["attempts"], 1)          # retried
        self.assertEqual(states[-1]["circuit_state"], "OPEN")
        self.assertEqual(states[-1]["attempts"], 0)           # short-circuited

    def test_timeout_is_enforced_not_merely_measured(self):
        connector = Connector("slow", lambda **_: __import__("time").sleep(2),
                              timeout_ms=100, retries=0)
        result = connector.invoke()
        self.assertFalse(result.ok)
        self.assertIn("timeout", result.error.lower())
        self.assertLess(result.latency_ms, 1500)

    def test_partners_are_called_in_parallel(self):
        """Retrieval must cost the slowest partner, not the sum."""
        for partner in ("bureau", "ekyc", "aml", "fraud", "payroll"):
            simulators.configure(partner, latency_ms=120)
        _, decision = self.decide()
        retrieval = decision["telemetry"]["retrieval_ms"]
        self.assertLess(retrieval, 5 * 120,
                        "partner calls appear to be sequential")

    def test_partner_health_is_reported(self):
        self.decide()
        status, payload = self.request("GET", "/v1/partners/health")
        self.assertEqual(status, 200)
        bureau = payload["partners"]["bureau"]
        for key in ("calls", "successes", "failures", "availability",
                    "latency_p95_ms", "circuit_state"):
            self.assertIn(key, bureau)


class LedgerTests(ApiTestCase):
    """FR-ADM-05 (tamper-evident audit) and FR-BIL-01 (metering at source)."""

    def test_every_decision_writes_an_audit_event(self):
        _, decision = self.decide()
        trail = self.ledger.audit_trail(
            tenant_code="ZAM-PAY",
            correlation_id=decision["identifiers"]["correlation_id"])
        self.assertTrue(trail)
        self.assertEqual(trail[0]["event_type"], "DECISION_ISSUED")

    def test_audit_chain_is_intact_after_many_events(self):
        for _ in range(5):
            self.decide()
        verification = self.ledger.verify_chain(tenant_code="ZAM-PAY")
        self.assertTrue(verification["intact"])
        self.assertGreaterEqual(verification["events"], 5)

    def test_tampering_with_history_is_detected(self):
        """The property that makes the trail worth having."""
        for _ in range(3):
            self.decide()
        connection = sqlite3.connect(self.ledger.path)
        connection.execute(
            "UPDATE audit_events SET payload = ? WHERE sequence = 2",
            (json.dumps({"outcome": "APPROVE", "tampered": True}),))
        connection.commit()
        connection.close()
        verification = self.ledger.verify_chain(tenant_code="ZAM-PAY")
        self.assertFalse(verification["intact"])
        self.assertEqual(verification["broken_at_sequence"], 2)

    def test_metering_reconciles_one_to_one_with_decisions(self):
        for _ in range(3):
            self.decide()
        reconciliation = self.ledger.reconcile(tenant_code="ZAM-PAY")
        self.assertTrue(reconciliation["balanced"])
        self.assertEqual(reconciliation["unmetered"], [])
        self.assertEqual(reconciliation["orphan_meters"], [])

    def test_usage_totals_are_exact_decimals(self):
        self.decide()
        usage = self.ledger.usage(tenant_code="ZAM-PAY")
        self.assertEqual(Decimal(usage["total"]), Decimal("0.35"))

    def test_external_data_calls_are_metered_separately(self):
        self.decide()
        usage = self.ledger.usage(tenant_code="ZAM-PAY")
        types = {line["event_type"] for line in usage["lines"]}
        self.assertIn("EXTERNAL_DATA_CALL", types)
        self.assertIn("APPLICATION_DECISION", types)

    def test_audit_chains_are_independent_per_tenant(self):
        self.decide()
        self.request("POST", "/v1/applications/decision",
                     body={"application": dict(APPLICATION),
                           "product_code": "MICRO_LOAN"},
                     key="demo-key-micro")
        pay = self.ledger.verify_chain(tenant_code="ZAM-PAY")
        mfi = self.ledger.verify_chain(tenant_code="ZAM-MFI")
        self.assertTrue(pay["intact"] and mfi["intact"])
        self.assertEqual(pay["events"], 1)
        self.assertEqual(mfi["events"], 1)


class OpenApiTests(ApiTestCase):
    """FR-API-01: a published, machine-readable specification."""

    def test_document_is_structurally_valid(self):
        _, spec = self.request("GET", "/openapi.json", key="")
        self.assertEqual(spec["openapi"], "3.0.3")
        for section in ("info", "paths", "components", "security"):
            self.assertIn(section, spec)
        self.assertIn("ApiKeyAuth", spec["components"]["securitySchemes"])

    def test_every_implemented_route_is_documented(self):
        _, spec = self.request("GET", "/openapi.json", key="")
        documented = set(spec["paths"])
        undocumented_by_design = {"/v1/demo/info"}
        for _, pattern, _ in self.api.routes:
            template = (pattern.pattern.strip("^$")
                        .replace(r"(?P<decision_id>[A-Za-z0-9\-]+)", "{decision_id}")
                        .replace(r"(?P<application_id>[A-Za-z0-9\-]+)", "{application_id}")
                        .replace(r"(?P<correlation_id>[A-Za-z0-9\-]+)", "{correlation_id}"))
            if template in undocumented_by_design:
                continue
            self.assertIn(template, documented, f"undocumented route {template}")

    def test_error_responses_are_documented_on_business_endpoints(self):
        _, spec = self.request("GET", "/openapi.json", key="")
        operation = spec["paths"]["/v1/applications/decision"]["post"]
        for status in ("400", "401", "429"):
            self.assertIn(status, operation["responses"])


class RateLimitTests(ApiTestCase):
    def test_rate_limit_returns_429(self):
        original = api_server.RATE_LIMIT_PER_MINUTE
        api_server.RATE_LIMIT_PER_MINUTE = 2
        api_server._rate_state.clear()
        try:
            statuses = [self.request("GET", "/v1/usage")[0] for _ in range(4)]
        finally:
            api_server.RATE_LIMIT_PER_MINUTE = original
            api_server._rate_state.clear()
        self.assertIn(429, statuses)


class HttpTransportTests(unittest.TestCase):
    """One end-to-end test over real HTTP to prove the transport works."""

    @classmethod
    def setUpClass(cls):
        simulators.reset()
        cls.ledger = temp_ledger()
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            cls.port = probe.getsockname()[1]
        cls.server = api_server.serve(cls.port,
                                      platform=Platform(ledger=cls.ledger))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.ledger.close()

    def test_decision_over_http(self):
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/applications/decision",
            method="POST",
            data=json.dumps({"application": dict(APPLICATION)}).encode())
        request.add_header("Content-Type", "application/json")
        request.add_header("X-API-Key", "demo-key-payroll")
        request.add_header("X-Correlation-Id", "COR-HTTP-TEST")
        with urllib.request.urlopen(request, timeout=15) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(response.headers["X-Correlation-Id"],
                             "COR-HTTP-TEST")
            payload = json.loads(response.read())
        self.assertIn(payload["outcome"],
                      ("APPROVE", "DECLINE", "REFER", "INSUFFICIENT_INFORMATION"))
        self.assertEqual(payload["identifiers"]["correlation_id"],
                         "COR-HTTP-TEST")


if __name__ == "__main__":
    unittest.main(verbosity=2)
