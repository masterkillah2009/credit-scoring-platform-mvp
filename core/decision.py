"""Decision and policy engine.

Implements IPSRS FR-DEC-01..06:

  FR-DEC-01  evaluate versioned declarative policy rule sets combining score,
             affordability, fraud, AML, eligibility and exposure, producing a
             full decision trace
  FR-DEC-02  outcome types: approve, decline (hard/soft), refer, insufficient
             information, conditional approval, counteroffer
  FR-DEC-03  compute risk-based price, recommended and maximum amount, tenor
  FR-DEC-04  apply policy by version; record model, feature and policy versions
  FR-DEC-05  return the complete decision contract
  FR-DEC-06  enforce decision expiry

This is the only component that combines model output with policy. The
scorecard never sees affordability; the affordability engine never sees the
score; neither sees the fraud result. They meet here, once, in the open.

Rules are data, not code. Conditions are evaluated by a small dispatch table
over a read-only context - there is no ``eval`` anywhere - so a tenant can
change policy without a release, and every rule evaluated is recorded whether
it fired or not.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Optional

from core import money as m
from core import reason_codes
from core.affordability import AffordabilityResult
from core.config import Product, Rule, Tenant, pricing_for_grade
from core.scorecard import ScoreResult

POLICY_ENGINE_VERSION = "DE-1.0.0"

# Outcomes
APPROVE = "APPROVE"
DECLINE = "DECLINE"
REFER = "REFER"
INSUFFICIENT = "INSUFFICIENT_INFORMATION"

#: Precedence when several rule kinds fire. Verification failures outrank
#: credit declines: if identity cannot be established there is no lawful basis
#: to record an adverse credit decision about that person at all.
_KIND_PRECEDENCE = ("hard_decline", "insufficient", "soft_decline", "refer")

_OPERATORS = {
    "eq": lambda actual, expected: actual == expected,
    "ne": lambda actual, expected: actual != expected,
    "lt": lambda actual, expected: actual is not None and actual < expected,
    "lte": lambda actual, expected: actual is not None and actual <= expected,
    "gt": lambda actual, expected: actual is not None and actual > expected,
    "gte": lambda actual, expected: actual is not None and actual >= expected,
    "in": lambda actual, expected: actual in (expected or ()),
    "is_true": lambda actual, _: actual is True,
    "is_false": lambda actual, _: actual is False,
    "is_missing": lambda actual, _: actual is None,
    "is_present": lambda actual, _: actual is not None,
}


@dataclass(frozen=True)
class RuleEvaluation:
    rule_id: str
    kind: str
    field: str
    operator: str
    expected: Any
    actual: Any
    matched: bool
    reason_code: Optional[str]
    description: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "kind": self.kind,
            "field": self.field,
            "operator": self.operator,
            "expected": _plain(self.expected),
            "actual": _plain(self.actual),
            "matched": self.matched,
            "reason_code": self.reason_code,
            "description": self.description,
        }


@dataclass(frozen=True)
class Decision:
    """The decision contract (IPSRS FR-DEC-05)."""

    outcome: str
    decline_type: Optional[str]
    reason_codes: list[str]
    additional_information_required: list[str]

    application_id: str
    decision_id: str
    correlation_id: str
    tenant_code: str
    product_code: str

    score: Optional[int]
    probability_of_default: Optional[float]
    risk_grade: Optional[str]
    model_id: Optional[str]
    model_version: Optional[str]
    model_segment: Optional[str]
    feature_set_version: Optional[str]
    policy_version: str
    policy_engine_version: str
    reason_code_library: str

    currency: str
    requested_amount: Decimal
    recommended_amount: Decimal
    maximum_amount: Decimal
    recommended_tenor_months: int
    annual_rate: Optional[Decimal]
    origination_fee: Optional[Decimal]
    monthly_instalment: Optional[Decimal]
    total_repayable: Optional[Decimal]
    total_cost_of_credit: Optional[Decimal]
    collateral_required: bool
    deposit_required: Decimal

    is_counteroffer: bool
    affordability: Optional[dict]
    data_quality_status: str
    confidence: Optional[str]

    decided_at: str
    expires_at: Optional[str]
    rule_trace: list[dict] = field(default_factory=list)
    gate_trace: list[dict] = field(default_factory=list)

    def as_dict(self, *, audience: str = "customer",
                include_trace: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "outcome": self.outcome,
            "decline_type": self.decline_type,
            "reason_codes": reason_codes.render(self.reason_codes,
                                                audience=audience),
            "additional_information_required": self.additional_information_required,
            "identifiers": {
                "application_id": self.application_id,
                "decision_id": self.decision_id,
                "correlation_id": self.correlation_id,
                "tenant": self.tenant_code,
                "product": self.product_code,
            },
            "assessment": {
                "score": self.score,
                "probability_of_default": self.probability_of_default,
                "risk_grade": self.risk_grade,
                "data_quality_status": self.data_quality_status,
                "confidence": self.confidence,
            },
            "versions": {
                "model_id": self.model_id,
                "model_version": self.model_version,
                "model_segment": self.model_segment,
                "feature_set_version": self.feature_set_version,
                "policy_version": self.policy_version,
                "policy_engine_version": self.policy_engine_version,
                "reason_code_library": self.reason_code_library,
            },
            "offer": {
                "currency": self.currency,
                "requested_amount": str(self.requested_amount),
                "recommended_amount": str(self.recommended_amount),
                "maximum_amount": str(self.maximum_amount),
                "recommended_tenor_months": self.recommended_tenor_months,
                "annual_rate": (None if self.annual_rate is None
                                else str(self.annual_rate)),
                "origination_fee": (None if self.origination_fee is None
                                    else str(self.origination_fee)),
                "monthly_instalment": (None if self.monthly_instalment is None
                                       else str(self.monthly_instalment)),
                "total_repayable": (None if self.total_repayable is None
                                    else str(self.total_repayable)),
                "total_cost_of_credit": (None if self.total_cost_of_credit is None
                                         else str(self.total_cost_of_credit)),
                "is_counteroffer": self.is_counteroffer,
                "collateral_required": self.collateral_required,
                "deposit_required": str(self.deposit_required),
            },
            "affordability": self.affordability,
            "decided_at": self.decided_at,
            "expires_at": self.expires_at,
        }
        if include_trace:
            payload["trace"] = {
                "rules_evaluated": len(self.rule_trace),
                "rules_matched": sum(1 for r in self.rule_trace if r["matched"]),
                "rules": self.rule_trace,
                "gates": self.gate_trace,
            }
        return payload


def _round_to_increment(amount: Decimal, increment: Decimal,
                        *, floor: Decimal) -> Decimal:
    """Round an offer DOWN to a saleable increment, never below the product floor.

    Rounding down matters: the amount came from an affordability ceiling, so
    rounding up would offer a facility the assessment has not approved.
    """
    if increment <= m.ZERO:
        return m.money(amount)
    steps = (m.money(amount) / increment).to_integral_value(rounding="ROUND_DOWN")
    rounded = m.money(steps * increment)
    return rounded if rounded >= floor else m.money(floor)


def _plain(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _resolve(context: dict, path: str) -> Any:
    """Read ``namespace.field`` from the evaluation context, or None."""
    node: Any = context
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def build_context(*, application: dict, features: dict, dq: dict,
                  score: Optional[ScoreResult],
                  affordability: Optional[AffordabilityResult],
                  partners: dict) -> dict:
    """Assemble the read-only namespace the rules evaluate against."""
    return {
        "application": application,
        "features": features,
        "dq": dq,
        "identity": partners.get("identity") or {},
        "screening": partners.get("screening") or {},
        "fraud": partners.get("fraud") or {},
        "employment": partners.get("employment") or {},
        "score": ({} if score is None else {
            "value": score.score,
            "probability_of_default": score.probability_of_default,
            "grade": score.risk_grade,
            "segment": score.segment,
            "confidence": score.confidence,
        }),
        "affordability": ({} if affordability is None else {
            "affordable": affordability.affordable,
            "max_instalment": affordability.max_affordable_instalment,
            "max_amount": affordability.max_affordable_amount,
            "dsr_after": affordability.dsr_after_requested,
            "verification_level": affordability.verification_level,
            "disposable_income": affordability.disposable_income,
        }),
    }


def evaluate_rules(rules: tuple[Rule, ...], context: dict) -> list[RuleEvaluation]:
    """Evaluate every rule and record the outcome, fired or not (FR-DEC-01)."""
    evaluations: list[RuleEvaluation] = []
    for rule in rules:
        actual = _resolve(context, rule.field)
        operator = _OPERATORS.get(rule.op)
        if operator is None:
            raise ValueError(f"rule {rule.id}: unknown operator {rule.op!r}")
        try:
            matched = bool(operator(actual, rule.value))
        except TypeError:
            # Incomparable types (e.g. a missing value against a numeric
            # threshold) never fire a rule silently; they are recorded as
            # not matched so the trace shows the engine considered them.
            matched = False
        evaluations.append(RuleEvaluation(
            rule_id=rule.id, kind=rule.kind, field=rule.field,
            operator=rule.op, expected=rule.value, actual=actual,
            matched=matched, reason_code=rule.reason_code,
            description=rule.description))
    return evaluations


def _eligibility_rules(application: dict, product: Product,
                       features: dict) -> list[RuleEvaluation]:
    """Structural product eligibility, evaluated alongside tenant policy."""
    amount = m.money(application.get("requested_amount") or 0)
    tenor = int(application.get("tenor_months") or 0)
    age = features.get("age_years")
    age_at_maturity = None if age is None else age + tenor / 12

    checks = [
        ("E-AMT-01", "hard_decline", "application.requested_amount",
         "within", f"{product.min_amount}-{product.max_amount}", str(amount),
         not (product.min_amount <= amount <= product.max_amount),
         "AMOUNT_OUTSIDE_PRODUCT_RANGE",
         "Requested amount outside the product range"),
        ("E-TEN-01", "hard_decline", "application.tenor_months",
         "within", f"{product.min_tenor_months}-{product.max_tenor_months}",
         tenor, not (product.min_tenor_months <= tenor <= product.max_tenor_months),
         "TENOR_OUTSIDE_PRODUCT_RANGE",
         "Requested tenor outside the product range"),
        ("E-AGE-02", "hard_decline", "features.age_years",
         "age_at_maturity_lte", product.max_age_at_maturity,
         None if age_at_maturity is None else round(age_at_maturity, 2),
         age_at_maturity is not None and age_at_maturity > product.max_age_at_maturity,
         "AGE_AT_MATURITY",
         "Age at maturity exceeds the product limit"),
    ]
    return [RuleEvaluation(rule_id=rid, kind=kind, field=fld, operator=op,
                           expected=expected, actual=actual, matched=matched,
                           reason_code=code, description=description)
            for rid, kind, fld, op, expected, actual, matched, code, description
            in checks]


def decide(*, tenant: Tenant, product: Product, application: dict,
           features: dict, dq: dict, score: Optional[ScoreResult],
           affordability: Optional[AffordabilityResult], partners: dict,
           application_id: Optional[str] = None,
           correlation_id: Optional[str] = None,
           now: Optional[datetime] = None) -> Decision:
    """Combine model output, affordability and policy into one decision."""
    now = now or datetime.now(timezone.utc)
    application_id = application_id or f"APP-{uuid.uuid4().hex[:12].upper()}"
    correlation_id = correlation_id or application_id
    decision_id = f"DEC-{uuid.uuid4().hex[:12].upper()}"

    context = build_context(application=application, features=features, dq=dq,
                            score=score, affordability=affordability,
                            partners=partners)

    evaluations = _eligibility_rules(application, product, features)
    evaluations += evaluate_rules(product.rules, context)

    outcome: Optional[str] = None
    decline_type: Optional[str] = None
    codes: list[str] = []
    information_required: list[str] = []
    gates: list[dict] = []

    # --- gate 1: policy rules, in precedence order ------------------------ #
    fired = {kind: [e for e in evaluations if e.matched and e.kind == kind]
             for kind in _KIND_PRECEDENCE}
    for kind in _KIND_PRECEDENCE:
        if not fired[kind]:
            continue
        codes = [e.reason_code for e in fired[kind] if e.reason_code]
        if kind == "hard_decline":
            outcome, decline_type = DECLINE, "hard"
        elif kind == "insufficient":
            outcome = INSUFFICIENT
            information_required = [e.description for e in fired[kind]]
        elif kind == "soft_decline":
            outcome, decline_type = DECLINE, "soft"
        else:
            outcome = REFER
        gates.append({
            "gate": "policy_rules", "result": outcome,
            "kind": kind,
            "rules_fired": [e.rule_id for e in fired[kind]],
            "detail": f"{kind} rule(s) matched",
        })
        break
    else:
        gates.append({"gate": "policy_rules", "result": "PASS",
                      "detail": f"{len(evaluations)} rules evaluated, none decisive"})

    # --- gate 2: score band ----------------------------------------------- #
    if outcome is None:
        if score is None:
            outcome = INSUFFICIENT
            codes = ["INSUFFICIENT_INFORMATION"]
            gates.append({"gate": "score_band", "result": INSUFFICIENT,
                          "detail": "no score available"})
        elif score.score >= product.accept_cutoff:
            gates.append({"gate": "score_band", "result": "PASS",
                          "detail": f"score {score.score} >= accept cut-off "
                                    f"{product.accept_cutoff}"})
        elif score.score >= product.refer_floor:
            outcome = REFER
            codes = list(score.reason_codes) + ["SCORE_IN_REFERRAL_BAND"]
            gates.append({"gate": "score_band", "result": REFER,
                          "detail": f"score {score.score} in referral band "
                                    f"[{product.refer_floor}, "
                                    f"{product.accept_cutoff})"})
        else:
            outcome, decline_type = DECLINE, "score"
            codes = list(score.reason_codes) + ["SCORE_BELOW_CUTOFF"]
            gates.append({"gate": "score_band", "result": DECLINE,
                          "detail": f"score {score.score} < referral floor "
                                    f"{product.refer_floor}"})

    # --- gate 3: affordability and counteroffer --------------------------- #
    grade = score.risk_grade if score else product.pricing[-1].grade
    band = pricing_for_grade(product, grade)
    requested_amount = m.money(application.get("requested_amount") or 0)
    tenor = int(application.get("tenor_months") or product.min_tenor_months)
    recommended_amount = requested_amount
    is_counteroffer = False

    if outcome is None:
        if affordability is None:
            outcome = INSUFFICIENT
            codes = ["INSUFFICIENT_INFORMATION"]
            gates.append({"gate": "affordability", "result": INSUFFICIENT,
                          "detail": "affordability not assessed"})
        elif affordability.affordable:
            gates.append({
                "gate": "affordability", "result": "PASS",
                "detail": (f"instalment {affordability.requested_instalment} "
                           f"within capacity "
                           f"{affordability.max_affordable_instalment} "
                           f"({affordability.binding_constraint} binding)")})
        elif affordability.max_affordable_amount >= product.min_amount:
            # Counteroffer rather than decline: the customer can afford
            # something, just not what was asked for (FR-DEC-02).
            capped = min(affordability.max_affordable_amount,
                         product.max_amount)
            recommended_amount = _round_to_increment(capped,
                                                     product.offer_increment,
                                                     floor=product.min_amount)
            is_counteroffer = True
            codes = ["COUNTEROFFER_REDUCED_AMOUNT"] + list(affordability.reason_codes)
            gates.append({
                "gate": "affordability", "result": "COUNTEROFFER",
                "detail": (f"requested {requested_amount} exceeds capacity; "
                           f"reduced to {recommended_amount}")})
        else:
            outcome, decline_type = DECLINE, "affordability"
            codes = list(affordability.reason_codes)
            gates.append({
                "gate": "affordability", "result": DECLINE,
                "detail": (f"maximum affordable amount "
                           f"{affordability.max_affordable_amount} below "
                           f"product minimum {product.min_amount}")})

    if outcome is None:
        outcome = APPROVE

    # --- pricing and offer construction ----------------------------------- #
    annual_rate = fee = instalment = repayable = cost = None
    deposit = m.ZERO
    if outcome == APPROVE:
        annual_rate = band.annual_rate
        fee = m.money(recommended_amount * band.fee_ratio)
        instalment = m.instalment_for(recommended_amount, annual_rate, tenor)
        repayable = m.total_repayable(instalment, tenor)
        cost = m.total_cost_of_credit(recommended_amount, instalment, tenor, fee)
    else:
        recommended_amount = m.ZERO

    # Deduplicate while preserving order, and never emit an unmapped code.
    seen: set[str] = set()
    final_codes: list[str] = []
    for code in codes:
        if code and code not in seen and code in reason_codes.BY_CODE:
            seen.add(code)
            final_codes.append(code)

    expires_at = None
    if outcome == APPROVE:
        expires_at = (now + timedelta(days=product.offer_validity_days)) \
            .isoformat(timespec="seconds")
    elif outcome in (REFER, INSUFFICIENT):
        expires_at = (now + timedelta(days=30)).isoformat(timespec="seconds")

    maximum_amount = (affordability.max_affordable_amount if affordability
                      else product.max_amount)

    return Decision(
        outcome=outcome,
        decline_type=decline_type,
        reason_codes=final_codes,
        additional_information_required=information_required,
        application_id=application_id,
        decision_id=decision_id,
        correlation_id=correlation_id,
        tenant_code=tenant.code,
        product_code=product.code,
        score=score.score if score else None,
        probability_of_default=(round(score.probability_of_default, 6)
                                if score else None),
        risk_grade=score.risk_grade if score else None,
        model_id=score.model_id if score else None,
        model_version=score.model_version if score else None,
        model_segment=score.segment if score else None,
        feature_set_version=score.feature_set_version if score else None,
        policy_version=tenant.policy_version,
        policy_engine_version=POLICY_ENGINE_VERSION,
        reason_code_library=reason_codes.LIBRARY_VERSION,
        currency=product.currency,
        requested_amount=requested_amount,
        recommended_amount=recommended_amount,
        maximum_amount=maximum_amount,
        recommended_tenor_months=tenor,
        annual_rate=annual_rate,
        origination_fee=fee,
        monthly_instalment=instalment,
        total_repayable=repayable,
        total_cost_of_credit=cost,
        collateral_required=False,
        deposit_required=deposit,
        is_counteroffer=is_counteroffer,
        affordability=affordability.as_dict() if affordability else None,
        data_quality_status=dq.get("status", "OK"),
        confidence=score.confidence if score else None,
        decided_at=now.isoformat(timespec="seconds"),
        expires_at=expires_at,
        rule_trace=[e.as_dict() for e in evaluations],
        gate_trace=gates,
    )
