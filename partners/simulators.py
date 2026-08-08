"""Partner simulators.

The platform integrates with credit bureaus, eKYC providers, AML screening,
payroll systems and core banking. None of those specifications is available to
this prototype, and IPSRS CST-02 forbids inventing them: "no partner API
specification may be invented; connectors build only from confirmed
documentation."

These simulators therefore make no claim to mirror any real provider's contract.
They exist to exercise the platform's own behaviour - parallel retrieval,
timeouts, retries, circuit breaking and the tenant's degradation policy - so
that when a real specification arrives only the adapter changes.

Responses are deterministic: the same national ID always produces the same
bureau file, so demonstrations and tests are reproducible. Failure injection is
explicit and global (``configure``), never random, so a test can assert exactly
what the platform does when a partner is unavailable.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Optional


class PartnerError(RuntimeError):
    """Partner call failed in a way the connector should treat as a failure."""


class PartnerTimeout(PartnerError):
    """Partner exceeded its latency budget."""


@dataclass
class FailureMode:
    """Explicit, deterministic failure injection for one partner."""

    unavailable: bool = False        # raise PartnerError on every call
    latency_ms: int = 0              # artificial delay before responding
    empty_response: bool = False     # respond successfully with no record


_MODES: dict[str, FailureMode] = {}


def configure(partner: str, **kwargs: Any) -> None:
    _MODES[partner] = FailureMode(**kwargs)


def reset() -> None:
    _MODES.clear()


def _mode(partner: str) -> FailureMode:
    return _MODES.get(partner, FailureMode())


def _seed(value: str) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest()[:8], 16)


def _enter(partner: str) -> None:
    mode = _mode(partner)
    if mode.latency_ms:
        time.sleep(mode.latency_ms / 1000.0)
    if mode.unavailable:
        raise PartnerError(f"{partner} unavailable (simulated)")


# --------------------------------------------------------------------------- #
# Simulated providers
# --------------------------------------------------------------------------- #
def credit_bureau(*, national_id: str, **_: Any) -> Optional[dict]:
    """Simulated credit-reference-bureau enquiry.

    Roughly a quarter of identities return no record at all, which is the
    thin-file population the THIN scorecard segment exists to serve.
    """
    _enter("bureau")
    if _mode("bureau").empty_response:
        return None

    seed = _seed(national_id)
    if seed % 100 < 25:
        return None                                  # no bureau record

    worst_dpd = [0, 0, 0, 15, 30, 60, 90, 120][seed % 8]
    return {
        "worst_dpd_12m": worst_dpd,
        "open_facilities": (seed >> 3) % 6,
        "enquiries_6m": (seed >> 5) % 5,
        "history_months": 12 + (seed >> 7) % 120,
        "revolving_utilisation": round(((seed >> 9) % 100) / 100, 2),
        "prior_default": worst_dpd >= 120,
        "provider": "SIMULATED_BUREAU",
        "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def ekyc(*, national_id: str, full_name: str = "", **_: Any) -> dict:
    """Simulated identity verification."""
    _enter("ekyc")
    seed = _seed(national_id + full_name)
    verified = seed % 100 >= 5                        # 5% fail verification
    return {
        "verified": verified,
        "method": "SIMULATED_NRC_LOOKUP",
        "confidence": "HIGH" if verified else "LOW",
        "provider": "SIMULATED_EKYC",
    }


def aml_screening(*, national_id: str, full_name: str = "", **_: Any) -> dict:
    """Simulated sanctions and PEP screening."""
    _enter("aml")
    seed = _seed("aml" + national_id + full_name)
    return {
        "sanctions_hit": seed % 500 == 0,             # deliberately rare
        "pep_hit": seed % 200 == 0,
        "provider": "SIMULATED_SCREENING",
    }


def payroll(*, employer_code: str = "", national_id: str = "", **_: Any) -> Optional[dict]:
    """Simulated payroll / employer verification."""
    _enter("payroll")
    if _mode("payroll").empty_response or not employer_code:
        return None
    seed = _seed(employer_code + national_id)
    return {
        "verified": True,
        "employment_months": 6 + (seed % 180),
        "net_monthly_income": None,        # caller may supply a verified figure
        "employer_code": employer_code,
        "provider": "SIMULATED_PAYROLL",
    }


def device_fraud(*, device_id: str = "", national_id: str = "", **_: Any) -> dict:
    """Simulated device and application-fraud signals."""
    _enter("fraud")
    seed = _seed("fraud" + device_id + national_id)
    risk = "HIGH" if seed % 50 == 0 else "MEDIUM" if seed % 11 == 0 else "LOW"
    return {
        "confirmed": seed % 997 == 0,
        "risk_level": risk,
        "device_id": device_id or None,
        "provider": "SIMULATED_DEVICE_INTEL",
    }


#: Partner registry: name -> (callable, timeout budget in milliseconds).
#: Budgets are deliberately tight; the platform's own latency target is under
#: 500 ms excluding partner time (IPSRS NFR-02), and partner calls run in
#: parallel so the slowest budget dominates rather than their sum.
REGISTRY: dict[str, tuple[Any, int]] = {
    "bureau": (credit_bureau, 800),
    "ekyc": (ekyc, 500),
    "aml": (aml_screening, 500),
    "payroll": (payroll, 800),
    "fraud": (device_fraud, 300),
}
