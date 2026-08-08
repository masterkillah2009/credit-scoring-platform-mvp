"""Check whether a deployed instance matches this reference implementation.

Point it at any URL. It exercises the public surface, then - if credentials are
supplied - the decisioning behaviour, and reports what conforms, what differs
and what could not be checked.

    python3 -m tools.conformance https://example.onrender.com
    python3 -m tools.conformance https://example.onrender.com risk demo-risk-2026

The checks are not stylistic. Each one corresponds to a property the
specification requires and this codebase implements and tests: the decision
contract, the version quartet stamped on every decision, the counteroffer
arithmetic, thin-file reason-code safety, tamper-evident audit, metering
reconciliation, tenant isolation and determinism. A deployment that fails them
may still be useful software - but it is not this software, and the evidence
this codebase produces for a credit committee does not transfer to it.

Uses only the standard library, so it runs anywhere the demo does.
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from typing import Any, Optional

TIMEOUT = 60

PASS, FAIL, WARN, SKIP = "PASS", "FAIL", "WARN", "SKIP"
_results: list[tuple[str, str, str]] = []


def record(status: str, check: str, detail: str = "") -> None:
    _results.append((status, check, detail))
    symbol = {PASS: "  ok  ", FAIL: " FAIL ", WARN: " warn ", SKIP: " skip "}[status]
    print(f"[{symbol}] {check}")
    if detail:
        print(f"          {detail}")


def call(base: str, method: str, path: str, *, body: Optional[dict] = None,
         token: Optional[str] = None, api_key: Optional[str] = None
         ) -> tuple[int, Any]:
    request = urllib.request.Request(
        base.rstrip("/") + path, method=method,
        data=json.dumps(body).encode() if body is not None else None)
    request.add_header("Content-Type", "application/json")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    if api_key:
        request.add_header("X-API-Key", api_key)
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            raw = response.read()
            try:
                return response.status, json.loads(raw)
            except json.JSONDecodeError:
                return response.status, raw.decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        raw = error.read()
        try:
            return error.code, json.loads(raw)
        except json.JSONDecodeError:
            return error.code, raw.decode("utf-8", "replace")
    except Exception as error:                      # network, TLS, DNS
        return 0, f"{error.__class__.__name__}: {error}"


# --------------------------------------------------------------------------- #
# Public surface
# --------------------------------------------------------------------------- #
def check_public(base: str) -> dict:
    status, health = call(base, "GET", "/healthz")
    if status != 200:
        record(FAIL, "Service is reachable and healthy", f"/healthz returned {status}")
        return {}
    record(PASS, "Service is reachable and healthy", json.dumps(health))

    status, spec = call(base, "GET", "/openapi.json")
    if status != 200 or not isinstance(spec, dict):
        record(FAIL, "OpenAPI specification is published", f"returned {status}")
        return {}

    paths = set(spec.get("paths", {}))
    expected = {
        "/v1/applications/decision", "/v1/prequalification",
        "/v1/decisions/{decision_id}", "/v1/decisions",
        "/v1/applications/{application_id}/decisions",
        "/v1/audit/{correlation_id}", "/v1/audit/verify", "/v1/usage",
        "/v1/partners/health", "/v1/monitoring/summary", "/v1/batches",
        "/v1/auth/login", "/v1/auth/logout", "/v1/auth/session",
        "/v1/demo/info", "/healthz", "/openapi.json",
    }
    missing = sorted(expected - paths)
    extra = sorted(paths - expected)
    if missing:
        record(FAIL, "OpenAPI documents every reference endpoint",
               f"missing: {', '.join(missing)}")
    else:
        record(PASS, "OpenAPI documents every reference endpoint",
               f"{len(paths)} paths")
    if extra:
        record(WARN, "No undocumented additional endpoints", f"extra: {', '.join(extra)}")

    schemes = set((spec.get("components", {}).get("securitySchemes") or {}))
    if {"ApiKeyAuth", "SessionToken"} <= schemes:
        record(PASS, "Both authentication schemes are documented", str(sorted(schemes)))
    else:
        record(FAIL, "Both authentication schemes are documented",
               f"found {sorted(schemes) or 'none'}; expected ApiKeyAuth and SessionToken")

    schemas = set((spec.get("components", {}).get("schemas") or {}))
    if {"Decision", "Application", "Error"} <= schemas:
        record(PASS, "Request and response schemas are defined", str(sorted(schemas)))
    else:
        record(FAIL, "Request and response schemas are defined",
               f"found {sorted(schemas) or 'none'}; a spec without schemas cannot "
               f"be contract-tested")

    operation = (spec.get("paths", {}).get("/v1/applications/decision", {})
                 .get("post", {}))
    documented = set(operation.get("responses", {}))
    if {"400", "401"} <= documented:
        record(PASS, "Error responses are documented on the decision endpoint",
               str(sorted(documented)))
    else:
        record(FAIL, "Error responses are documented on the decision endpoint",
               f"found {sorted(documented)}")

    status, info = call(base, "GET", "/v1/demo/info")
    if status == 200 and isinstance(info, dict) and "banner" in info:
        record(PASS, "Demonstration banner endpoint present")
        if info.get("default_accounts_in_use"):
            record(WARN, "Default demonstration accounts are not in use",
                   "the deployment still uses the built-in accounts; set "
                   "DEMO_USERS before sharing the URL")
        else:
            record(PASS, "Default demonstration accounts are not in use")
    else:
        record(FAIL, "Demonstration banner endpoint present", f"returned {status}")

    status, _ = call(base, "GET", "/v1/decisions")
    if status in (401, 403):
        record(PASS, "Protected endpoints refuse unauthenticated callers",
               f"/v1/decisions returned {status}")
    else:
        record(FAIL, "Protected endpoints refuse unauthenticated callers",
               f"/v1/decisions returned {status} without credentials")

    home = call(base, "GET", "/")[1]
    if isinstance(home, str) and "not approved for production" in home.lower():
        record(PASS, "Console carries a static prototype notice")
    else:
        record(FAIL, "Console carries a static prototype notice",
               "the disclaimer must be in the page, not only in configuration")
    return spec


# --------------------------------------------------------------------------- #
# Behaviour (requires credentials)
# --------------------------------------------------------------------------- #
APPLICATION = {
    "national_id": "749078/36/8", "full_name": "Bwalya Phiri",
    "date_of_birth": "1988-06-02", "application_date": "2026-08-03",
    "employer_code": "GRZ-HR-22", "requested_amount": 85000,
    "tenor_months": 24, "declared_monthly_income": 9500,
    "declared_monthly_expenses": 3400, "existing_monthly_debt_service": 1500,
    "dependants": 3,
    "consent": {"credit_bureau_enquiry": True, "automated_decisioning": True},
}


def check_behaviour(base: str, username: str, password: str) -> None:
    status, session = call(base, "POST", "/v1/auth/login",
                           body={"username": username, "password": password})
    if status != 200 or not isinstance(session, dict) or "token" not in session:
        record(FAIL, "Sign-in returns a session token", f"returned {status}")
        return
    token = session["token"]
    record(PASS, "Sign-in returns a session token",
           f"role={session.get('user', {}).get('role')}")

    status, bad = call(base, "POST", "/v1/auth/login",
                       body={"username": username, "password": "definitely-wrong"})
    status2, unknown = call(base, "POST", "/v1/auth/login",
                            body={"username": "no-such-user", "password": "x"})
    same = (isinstance(bad, dict) and isinstance(unknown, dict)
            and bad.get("error", {}).get("message")
            == unknown.get("error", {}).get("message"))
    record(PASS if same else FAIL,
           "Failed sign-in does not reveal whether a username exists",
           "" if same else "wrong-password and unknown-user responses differ")

    # -- validation before any partner call --------------------------------- #
    invalid = dict(APPLICATION, national_id="not-an-nrc")
    status, error = call(base, "POST", "/v1/applications/decision",
                         body={"application": invalid}, token=token)
    if status == 400 and isinstance(error, dict) and error.get("error", {}).get("details"):
        record(PASS, "Intake validation rejects a malformed NRC with field detail")
    else:
        record(FAIL, "Intake validation rejects a malformed NRC with field detail",
               f"returned {status}")

    no_consent = dict(APPLICATION)
    no_consent.pop("consent")
    status, _ = call(base, "POST", "/v1/applications/decision",
                     body={"application": no_consent}, token=token)
    record(PASS if status == 400 else FAIL,
           "Consent is required before processing", f"returned {status}")

    # -- the decision contract ---------------------------------------------- #
    status, decision = call(base, "POST", "/v1/applications/decision",
                            body={"application": dict(APPLICATION)}, token=token)
    if status != 200 or not isinstance(decision, dict):
        record(FAIL, "Decision endpoint returns a decision", f"returned {status}")
        return

    required_sections = ("outcome", "reason_codes", "identifiers", "assessment",
                         "versions", "offer", "affordability", "trace")
    missing = [s for s in required_sections if s not in decision]
    record(PASS if not missing else FAIL, "Decision contract is complete",
           "" if not missing else f"missing: {', '.join(missing)}")

    versions = decision.get("versions", {}) or {}
    required_versions = ("model_version", "feature_set_version", "policy_version",
                         "reason_code_library")
    missing = [v for v in required_versions if not versions.get(v)]
    record(PASS if not missing else FAIL,
           "Every decision is stamped with its model, feature, policy and "
           "reason-code versions",
           "" if not missing else f"missing: {', '.join(missing)}")

    trace = decision.get("trace", {}) or {}
    if trace.get("rules") and trace.get("gates"):
        fired = [r for r in trace["rules"] if r.get("matched")]
        record(PASS, "Rule-by-rule trace is returned",
               f"{trace.get('rules_evaluated')} rules evaluated, {len(fired)} fired, "
               f"{len(trace['gates'])} gates")
    else:
        record(FAIL, "Rule-by-rule trace is returned",
               "no rule/gate trace - the decision cannot be explained to a regulator")

    affordability = decision.get("affordability") or {}
    if affordability.get("binding_constraint") and affordability.get("max_affordable_instalment"):
        record(PASS, "Affordability assessment is returned separately from the score",
               f"capacity {affordability['max_affordable_instalment']} "
               f"({affordability['binding_constraint']} binding)")
    else:
        record(FAIL, "Affordability assessment is returned separately from the score")

    offer = decision.get("offer") or {}
    if decision.get("outcome") == "APPROVE" and offer.get("is_counteroffer"):
        try:
            capacity = float(affordability["max_affordable_instalment"])
            instalment = float(offer["monthly_instalment"])
            within = instalment <= capacity + 0.01
            record(PASS if within else FAIL,
                   "Counteroffer instalment stays within assessed capacity",
                   f"instalment {instalment} vs capacity {capacity}")
        except (KeyError, TypeError, ValueError):
            record(WARN, "Counteroffer instalment stays within assessed capacity",
                   "could not parse the amounts")
    else:
        record(WARN, "Counteroffer path exercised",
               f"this applicant returned {decision.get('outcome')}; the reference "
               f"build returns APPROVE with a counteroffer")

    # -- determinism --------------------------------------------------------- #
    scores = set()
    for _ in range(3):
        _, repeat = call(base, "POST", "/v1/applications/decision",
                         body={"application": dict(APPLICATION)}, token=token)
        if isinstance(repeat, dict):
            scores.add(repeat.get("assessment", {}).get("score"))
    record(PASS if len(scores) == 1 else FAIL, "Scoring is deterministic",
           f"scores observed: {sorted(s for s in scores if s is not None)}")

    # -- thin file must not attract bureau reasons --------------------------- #
    thin = dict(APPLICATION, national_id="414328/41/3", full_name="Mutinta Banda")
    thin.pop("employer_code", None)
    status, thin_decision = call(base, "POST", "/v1/applications/decision",
                                 body={"application": thin}, token=token)
    if status == 200 and isinstance(thin_decision, dict):
        codes = {r.get("code") for r in thin_decision.get("reason_codes", [])
                 if isinstance(r, dict)}
        forbidden = {"RECENT_DELINQUENCY", "HIGH_UTILISATION",
                     "SHORT_CREDIT_HISTORY", "MANY_RECENT_ENQUIRIES",
                     "PRIOR_DEFAULT_RECORD"} & codes
        record(PASS if not forbidden else FAIL,
               "A thin-file applicant receives no bureau-based reason codes",
               "" if not forbidden else f"returned {sorted(forbidden)} for an "
                                        f"applicant with no bureau record")
    else:
        record(SKIP, "A thin-file applicant receives no bureau-based reason codes",
               f"thin-file request returned {status}")

    # -- audit and metering --------------------------------------------------- #
    correlation = (decision.get("identifiers") or {}).get("correlation_id")
    if correlation:
        status, trail = call(base, "GET", f"/v1/audit/{correlation}", token=token)
        if status == 200 and isinstance(trail, dict) and trail.get("count"):
            entry = (trail.get("events") or [{}])[0]
            has_chain = bool(entry.get("entry_hash") and entry.get("previous_hash"))
            record(PASS if has_chain else FAIL,
                   "Audit entries are hash-chained",
                   "" if has_chain else "no entry_hash/previous_hash - the trail "
                                        "is not tamper-evident")
        else:
            record(WARN, "Audit trail is retrievable by correlation id",
                   f"returned {status} (may be a permissions restriction)")

    status, verify = call(base, "GET", "/v1/audit/verify", token=token)
    if status == 200 and isinstance(verify, dict) and "intact" in verify:
        record(PASS if verify["intact"] else FAIL, "Audit chain verifies as intact",
               json.dumps(verify))
    else:
        record(WARN, "Audit chain verification endpoint", f"returned {status}")

    status, usage = call(base, "GET", "/v1/usage", token=token)
    if status == 200 and isinstance(usage, dict):
        reconciliation = usage.get("reconciliation") or {}
        if "balanced" in reconciliation:
            record(PASS if reconciliation["balanced"] else FAIL,
                   "Metering reconciles one-to-one with decisions",
                   json.dumps(reconciliation))
        else:
            record(FAIL, "Metering reconciles one-to-one with decisions",
                   "no reconciliation block - billing cannot be evidenced")
    elif status == 403:
        record(SKIP, "Metering reconciliation", "this role may not read usage")
    else:
        record(WARN, "Metering reconciliation", f"returned {status}")

    # -- monitoring ----------------------------------------------------------- #
    status, monitoring = call(base, "GET", "/v1/monitoring/summary", token=token)
    if status == 200 and isinstance(monitoring, dict):
        psi = monitoring.get("population_stability") or {}
        if psi.get("status") and "thresholds" in psi:
            record(PASS, "Monitoring reports PSI with its status and thresholds",
                   f"PSI {psi.get('index')} ({psi['status']})")
        else:
            record(FAIL, "Monitoring reports PSI with its status and thresholds",
                   "a metric without a disclosed threshold cannot be acted on")
    elif status == 403:
        record(SKIP, "Monitoring pack", "this role may not read monitoring")
    else:
        record(WARN, "Monitoring pack", f"returned {status}")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    base = sys.argv[1]
    print(f"\nConformance check against {base}\n" + "=" * 68)
    check_public(base)

    if len(sys.argv) >= 4:
        print("-" * 68)
        check_behaviour(base, sys.argv[2], sys.argv[3])
    else:
        record(SKIP, "Decisioning behaviour",
               "supply a username and password to check the decision contract, "
               "determinism, audit chain and metering")

    print("=" * 68)
    counts = {status: sum(1 for s, _, _ in _results if s == status)
              for status in (PASS, FAIL, WARN, SKIP)}
    print(f"  {counts[PASS]} passed · {counts[FAIL]} failed · "
          f"{counts[WARN]} warnings · {counts[SKIP]} skipped")
    if counts[FAIL]:
        print("\n  This deployment does not match the reference implementation.")
        raise SystemExit(1)
    print("\n  Conforms to the reference implementation on every check performed.")


if __name__ == "__main__":
    main()
