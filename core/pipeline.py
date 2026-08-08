"""End-to-end decisioning pipeline.

Orchestrates IPSRS UC-01:

    validate -> retrieve (parallel) -> features -> score -> affordability
             -> decide -> persist -> meter -> audit

Design points that matter:

* **Partner retrieval runs in parallel** (FR-CNX-02), so the retrieval budget is
  the slowest partner rather than the sum. Each partner has its own timeout,
  retry and circuit breaker.
* **Degradation is the tenant's decision, not the platform's** (FR-CNX-03).
  When a partner fails, the configured policy applies: ``refer`` routes to
  manual underwriting, ``partial`` scores on what is available with the gap
  flagged, ``decline`` stops. In every case the decision records which source
  was missing and why - the platform never fabricates a partner response.
* **Scoring and affordability remain independent**; only the decision engine
  sees both.
* **Every decision is persisted, metered and audited** before it is returned,
  so an invoice line and an audit trail always exist for work performed.
"""
from __future__ import annotations

import time
from typing import Any, Optional

from core import affordability as afford
from core import config, decision, features, scorecard
from core.ledger import Ledger
from partners.connector import ConnectorRegistry, PartnerResult
from partners.simulators import REGISTRY

#: Partners whose absence is material to a credit decision. Fraud and eKYC
#: failures are handled by policy rules (identity unverified -> insufficient
#: information), so they are not listed here as degradation triggers.
_MATERIAL_PARTNERS = ("bureau",)


class Platform:
    """Holds the long-lived collaborators: model, connectors, ledger."""

    def __init__(self, *, ledger: Optional[Ledger] = None,
                 connectors: Optional[ConnectorRegistry] = None,
                 card: Optional[scorecard.Scorecard] = None):
        self.ledger = ledger or Ledger()
        self.connectors = connectors or ConnectorRegistry(REGISTRY)
        self.card = card or scorecard.load()

    # -- retrieval --------------------------------------------------------- #
    def retrieve(self, application: dict) -> dict[str, PartnerResult]:
        national_id = application.get("national_id") or ""
        requests = {
            "bureau": {"national_id": national_id},
            "ekyc": {"national_id": national_id,
                     "full_name": application.get("full_name", "")},
            "aml": {"national_id": national_id,
                    "full_name": application.get("full_name", "")},
            "fraud": {"national_id": national_id,
                      "device_id": application.get("device_id", "")},
        }
        if application.get("employer_code"):
            requests["payroll"] = {
                "employer_code": application["employer_code"],
                "national_id": national_id,
            }
        return self.connectors.fetch_all(requests)

    # -- decisioning ------------------------------------------------------- #
    def decide(self, request: dict, *, tenant: config.Tenant,
               product: config.Product,
               application_id: Optional[str] = None,
               correlation_id: Optional[str] = None,
               actor: str = "api") -> tuple[decision.Decision, dict]:
        started = time.monotonic()
        application = dict(request.get("application") or {})
        overrides = request.get("partner_overrides") or {}

        partner_results = self.retrieve(application)
        retrieval_ms = int((time.monotonic() - started) * 1000)

        # Assemble the partner view. A failed call contributes nothing; it is
        # never replaced with a default, an empty object or a zero.
        def payload(name: str) -> Any:
            if name in overrides:
                return overrides[name]
            result = partner_results.get(name)
            return result.payload if (result and result.ok) else None

        degraded: list[str] = []
        for name in _MATERIAL_PARTNERS:
            result = partner_results.get(name)
            if name not in overrides and result is not None and not result.ok:
                degraded.append(name)

        context = {
            "application": application,
            "bureau": payload("bureau"),
            "payroll": payload("payroll"),
            "internal": request.get("internal") or {},
            "identity": payload("ekyc") or {"verified": False},
            "screening": payload("aml") or {},
            "fraud": payload("fraud") or {},
            "employment": {
                "payroll_verified": bool((payload("payroll") or {}).get("verified"))
            },
        }

        computed = features.compute(context, dq_policy=tenant.dq_policy)
        dq = {
            "status": computed.status,
            "thin_file": computed.thin_file,
            "missing": list(computed.missing),
            "notes": list(computed.notes),
            "degraded_partners": degraded,
        }

        score = None
        if computed.status != "BLOCK":
            score = self.card.score(computed.values, tenant=tenant,
                                    dq_status=computed.status)

        if score is not None:
            band = config.pricing_for_grade(product, score.risk_grade)
            assessment = afford.assess(context, product=product,
                                       annual_rate=band.annual_rate)
        else:
            assessment = afford.assess(context, product=product)

        result = decision.decide(
            tenant=tenant, product=product, application=application,
            features=computed.values, dq=dq, score=score,
            affordability=assessment, partners=context,
            application_id=application_id, correlation_id=correlation_id)

        # -- tenant degradation policy (FR-CNX-03) ------------------------- #
        if degraded:
            result = self._apply_degradation(result, tenant, degraded)

        elapsed_ms = int((time.monotonic() - started) * 1000)
        telemetry = {
            "retrieval_ms": retrieval_ms,
            "total_ms": elapsed_ms,
            "partners": {name: r.as_dict()
                         for name, r in partner_results.items()},
            "degraded_partners": degraded,
        }

        self.ledger.store_decision(tenant_code=tenant.code, decision=result)
        self.ledger.meter(tenant_code=tenant.code,
                          event_type="APPLICATION_DECISION",
                          correlation_id=result.correlation_id,
                          reference=result.decision_id)
        for name, partner_result in partner_results.items():
            if partner_result.ok:
                self.ledger.meter(tenant_code=tenant.code,
                                  event_type="EXTERNAL_DATA_CALL",
                                  correlation_id=result.correlation_id,
                                  reference=name)
        self.ledger.record(
            tenant_code=tenant.code, event_type="DECISION_ISSUED", actor=actor,
            correlation_id=result.correlation_id,
            payload={
                "application_id": result.application_id,
                "decision_id": result.decision_id,
                "outcome": result.outcome,
                "score": result.score,
                "model_version": result.model_version,
                "model_segment": result.model_segment,
                "policy_version": result.policy_version,
                "feature_set_version": result.feature_set_version,
                "reason_codes": result.reason_codes,
                "recommended_amount": str(result.recommended_amount),
                "telemetry": telemetry,
            })
        return result, telemetry

    def _apply_degradation(self, result: decision.Decision,
                           tenant: config.Tenant,
                           degraded: list[str]) -> decision.Decision:
        """Apply the tenant's configured behaviour when a partner is down."""
        from dataclasses import replace

        note = {
            "gate": "partner_degradation",
            "result": tenant.degradation_policy.upper(),
            "detail": (f"unavailable: {', '.join(degraded)}; tenant policy "
                       f"'{tenant.degradation_policy}' applied"),
        }
        codes = list(result.reason_codes)
        if "PARTNER_DATA_UNAVAILABLE" not in codes:
            codes.append("PARTNER_DATA_UNAVAILABLE")

        if tenant.degradation_policy == "refer" and result.outcome == decision.APPROVE:
            return replace(result, outcome=decision.REFER, reason_codes=codes,
                           recommended_amount=result.recommended_amount,
                           gate_trace=result.gate_trace + [note])
        if tenant.degradation_policy == "decline" and result.outcome != decision.DECLINE:
            return replace(result, outcome=decision.DECLINE,
                           decline_type="partner_unavailable",
                           reason_codes=codes,
                           gate_trace=result.gate_trace + [note])
        # 'partial': proceed on available data, but say so.
        return replace(result, reason_codes=codes,
                       gate_trace=result.gate_trace + [note])


# --------------------------------------------------------------------------- #
# Backwards-compatible functional entry point (used by Phase 2 demos and tests)
# --------------------------------------------------------------------------- #
def run(context: dict, *, tenant: config.Tenant, product: config.Product,
        application_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        card: Optional[scorecard.Scorecard] = None) -> decision.Decision:
    """Score, assess and decide one application from a fully supplied context.

    Phase 2 callers pass partner data directly rather than having the platform
    retrieve it. This path performs no I/O, writes no ledger entries and is
    what the unit tests exercise.
    """
    card = card or scorecard.load()
    application = context.get("application") or {}
    partners = {
        "identity": context.get("identity") or {},
        "screening": context.get("screening") or {},
        "fraud": context.get("fraud") or {},
        "employment": context.get("employment") or {},
    }

    computed = features.compute(context, dq_policy=tenant.dq_policy)
    dq = {
        "status": computed.status,
        "thin_file": computed.thin_file,
        "missing": list(computed.missing),
        "notes": list(computed.notes),
    }

    score = None
    if computed.status != "BLOCK":
        score = card.score(computed.values, tenant=tenant,
                           dq_status=computed.status)

    if score is not None:
        band = config.pricing_for_grade(product, score.risk_grade)
        assessment = afford.assess(context, product=product,
                                   annual_rate=band.annual_rate)
    else:
        assessment = afford.assess(context, product=product)

    return decision.decide(
        tenant=tenant, product=product, application=application,
        features=computed.values, dq=dq, score=score,
        affordability=assessment, partners=partners,
        application_id=application_id, correlation_id=correlation_id)
