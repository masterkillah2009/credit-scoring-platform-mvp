"""Phase 3 demonstration: the API, partner failure, audit and metering.

Run:  python3 demo_phase3.py

Starts the API on a free port, then exercises it over real HTTP:

  1. a decision with all partners healthy
  2. intake validation rejecting a malformed application before any partner call
  3. idempotent replay - the same key returns the original decision, unbilled
  4. bureau outage: circuit opens, tenant degradation policy applies
  5. tenant isolation: one tenant cannot read another's decision
  6. audit reconstruction and tamper-evidence verification
  7. metered usage and decision reconciliation
"""
from __future__ import annotations

import json
import pathlib
import socket
import tempfile
import threading
import urllib.error
import urllib.request

from api.server import serve
from core.ledger import Ledger
from core.pipeline import Platform
from partners import simulators

BASE_APPLICATION = {
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


def call(port: int, method: str, path: str, *, key: str = "demo-key-payroll",
         body: dict | None = None, headers: dict | None = None) -> tuple[int, dict]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", method=method,
        data=json.dumps(body).encode() if body is not None else None)
    request.add_header("Content-Type", "application/json")
    if key:
        request.add_header("X-API-Key", key)
    for name, value in (headers or {}).items():
        request.add_header(name, value)
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def main() -> None:
    simulators.reset()
    ledger_path = pathlib.Path(tempfile.mkdtemp()) / "demo_ledger.db"
    platform = Platform(ledger=Ledger(ledger_path))
    port = free_port()
    server = serve(port, platform=platform)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    print("=" * 78)
    print(f"API listening on http://127.0.0.1:{port}")
    status, health = call(port, "GET", "/healthz", key="")
    print(f"  /healthz -> {status} {health}")
    _, spec = call(port, "GET", "/openapi.json", key="")
    print(f"  /openapi.json -> {len(spec['paths'])} paths, "
          f"OpenAPI {spec['openapi']}")
    print("=" * 78)

    # 1 -- healthy decision --------------------------------------------------
    print("\n1. Decision with all partners healthy")
    status, decision = call(port, "POST", "/v1/applications/decision",
                            body={"application": BASE_APPLICATION},
                            headers={"Idempotency-Key": "demo-key-0001"})
    telemetry = decision["telemetry"]
    print(f"   HTTP {status} -> {decision['outcome']}"
          f"{' (counteroffer)' if decision['offer']['is_counteroffer'] else ''}")
    print(f"   score {decision['assessment']['score']} "
          f"grade {decision['assessment']['risk_grade']} "
          f"segment {decision['versions']['model_segment']}")
    print(f"   offer {decision['offer']['currency']} "
          f"{decision['offer']['recommended_amount']} @ instalment "
          f"{decision['offer']['monthly_instalment']}")
    print(f"   partners retrieved in {telemetry['retrieval_ms']}ms, "
          f"total {telemetry['total_ms']}ms")
    for name, partner in sorted(telemetry["partners"].items()):
        print(f"     {name:<8} ok={partner['ok']!s:<5} "
              f"{partner['latency_ms']}ms attempts={partner['attempts']} "
              f"circuit={partner['circuit_state']}")
    decision_id = decision["identifiers"]["decision_id"]
    correlation_id = decision["identifiers"]["correlation_id"]

    # 2 -- validation --------------------------------------------------------
    print("\n2. Intake validation rejects before any partner is called")
    bad = dict(BASE_APPLICATION, national_id="not-an-nrc", requested_amount=-5)
    bad.pop("consent")
    status, error = call(port, "POST", "/v1/applications/decision",
                         body={"application": bad})
    print(f"   HTTP {status} {error['error']['code']}")
    for item in error["error"]["details"]["errors"]:
        print(f"     {item['field']}: {item['error']}")

    # 3 -- idempotency -------------------------------------------------------
    print("\n3. Idempotent replay")
    status, replay = call(port, "POST", "/v1/applications/decision",
                          body={"application": BASE_APPLICATION},
                          headers={"Idempotency-Key": "demo-key-0001"})
    print(f"   HTTP {status} replayed={replay.get('replayed')} "
          f"same decision_id={replay['identifiers']['decision_id'] == decision_id}")

    # 4 -- partner outage and degradation ------------------------------------
    print("\n4. Bureau outage: circuit breaker and tenant degradation policy")
    simulators.configure("bureau", unavailable=True)
    for attempt in range(1, 5):
        status, degraded = call(
            port, "POST", "/v1/applications/decision",
            body={"application": dict(BASE_APPLICATION,
                                      national_id="384756/61/1")})
        bureau = degraded["telemetry"]["partners"]["bureau"]
        print(f"   attempt {attempt}: {degraded['outcome']:<22} "
              f"bureau ok={bureau['ok']!s:<5} attempts={bureau['attempts']} "
              f"circuit={bureau['circuit_state']}")
    print(f"   reasons: {[r['code'] for r in degraded['reason_codes']]}")
    print(f"   gate:    {degraded['trace']['gates'][-1]['detail']}")
    simulators.reset()

    # 5 -- tenant isolation ---------------------------------------------------
    print("\n5. Tenant isolation")
    status, own = call(port, "GET", f"/v1/decisions/{decision_id}")
    print(f"   owning tenant   -> HTTP {status}")
    status, other = call(port, "GET", f"/v1/decisions/{decision_id}",
                         key="demo-key-micro")
    print(f"   other tenant    -> HTTP {status} {other['error']['code']}")
    status, anonymous = call(port, "GET", f"/v1/decisions/{decision_id}", key="")
    print(f"   no credentials  -> HTTP {status} {anonymous['error']['code']}")

    # 6 -- audit --------------------------------------------------------------
    print("\n6. Audit trail and tamper evidence")
    status, trail = call(port, "GET", f"/v1/audit/{correlation_id}")
    print(f"   events for {correlation_id}: {trail['count']}")
    for event in trail["events"]:
        print(f"     #{event['sequence']} {event['event_type']} by "
              f"{event['actor']} hash {event['entry_hash'][:12]}...")
    status, verification = call(port, "GET", "/v1/audit/verify")
    print(f"   chain: intact={verification['intact']} "
          f"events={verification['events']}")

    # 7 -- usage and reconciliation -------------------------------------------
    print("\n7. Metered usage and reconciliation")
    status, usage = call(port, "GET", "/v1/usage")
    for line in usage["lines"]:
        print(f"   {line['event_type']:<22} qty {line['quantity']:<4} "
              f"@ {line['unit_price']:<6} = {line['amount']}")
    print(f"   total {usage['currency']} {usage['total']}")
    reconciliation = usage["reconciliation"]
    print(f"   reconciliation: decisions={reconciliation['decisions']} "
          f"metered={reconciliation['metered_decisions']} "
          f"balanced={reconciliation['balanced']}")

    server.shutdown()
    print("\n" + "=" * 78)
    print(f"ledger: {ledger_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
