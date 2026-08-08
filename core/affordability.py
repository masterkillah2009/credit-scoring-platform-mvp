"""Affordability engine.

Implements IPSRS FR-AFD-01..04:

  FR-AFD-01  compute DSR, disposable income, maximum affordable instalment and
             maximum amount from verified income, expense models and existing
             obligations, per tenant-configured rules
  FR-AFD-02  apply income haircuts and expense floors by verification level
  FR-AFD-03  incorporate the proposed facility's terms (amount, tenor, rate,
             fees) into the instalment computation
  FR-AFD-04  return the affordability object separately from the score

The engine answers one question - "can this customer service this facility?" -
and answers it without ever seeing the credit score. Combining the two is the
decision engine's job alone (BRD BR-AFF-05).

Capacity is the binding minimum of two independent tests:

  DSR test       (existing + proposed debt service) <= max_dsr x verified income
  cash-flow test verified income - expenses - existing service - required
                 disposable-income buffer

Both are computed in exact decimal arithmetic (core.money).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Optional

from core import money as m
from core.config import AffordabilityPolicy, Product

#: Verification levels in ascending order of evidential strength. The level
#: drives the haircut applied to income (FR-AFD-02) and is recorded on the
#: decision so an auditor can see what the assessment relied on.
VERIFICATION_LEVELS = ("DECLARED", "DOCUMENTED", "PAYROLL_VERIFIED",
                       "TRANSACTION_VERIFIED")


@dataclass(frozen=True)
class AffordabilityResult:
    """The affordability object returned alongside - never merged into - the score."""

    currency: str
    verification_level: str
    haircut_applied: Decimal
    declared_income: Decimal
    verified_income: Decimal
    modelled_expenses: Decimal
    expense_basis: str
    existing_debt_service: Decimal
    required_buffer: Decimal
    disposable_income: Decimal
    dsr_before: Decimal
    dsr_after_requested: Optional[Decimal]
    dsr_capacity_instalment: Decimal
    cashflow_capacity_instalment: Decimal
    max_affordable_instalment: Decimal
    binding_constraint: str
    annual_rate: Decimal
    tenor_months: int
    requested_amount: Decimal
    requested_instalment: Decimal
    max_affordable_amount: Decimal
    affordable: bool
    shortfall_instalment: Decimal
    reason_codes: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        def s(value) -> Optional[str]:
            return None if value is None else str(value)
        return {
            "currency": self.currency,
            "verification_level": self.verification_level,
            "haircut_applied": s(self.haircut_applied),
            "declared_income": s(self.declared_income),
            "verified_income": s(self.verified_income),
            "modelled_expenses": s(self.modelled_expenses),
            "expense_basis": self.expense_basis,
            "existing_debt_service": s(self.existing_debt_service),
            "required_buffer": s(self.required_buffer),
            "disposable_income": s(self.disposable_income),
            "dsr_before": s(self.dsr_before),
            "dsr_after_requested": s(self.dsr_after_requested),
            "max_affordable_instalment": s(self.max_affordable_instalment),
            "binding_constraint": self.binding_constraint,
            "requested_amount": s(self.requested_amount),
            "requested_instalment": s(self.requested_instalment),
            "max_affordable_amount": s(self.max_affordable_amount),
            "affordable": self.affordable,
            "shortfall_instalment": s(self.shortfall_instalment),
            "reason_codes": list(self.reason_codes),
            "notes": list(self.notes),
        }


def verification_level(context: dict) -> str:
    """Derive the evidential level from the partner data actually obtained.

    Deliberately conservative: absent evidence means DECLARED, never an assumed
    upgrade. The level is evidence-driven, not self-asserted by the applicant.
    """
    payroll = context.get("payroll") or {}
    if payroll.get("verified") and payroll.get("net_monthly_income") is not None:
        return "PAYROLL_VERIFIED"
    if payroll.get("verified"):
        return "PAYROLL_VERIFIED"
    transactions = context.get("transactions") or {}
    if transactions.get("verified_recurring_income") is not None:
        return "TRANSACTION_VERIFIED"
    application = context.get("application") or {}
    if application.get("income_documents_provided"):
        return "DOCUMENTED"
    return "DECLARED"


def assess(context: dict, *, product: Product,
           annual_rate: Optional[Decimal] = None,
           amount: Optional[Decimal] = None,
           tenor_months: Optional[int] = None) -> AffordabilityResult:
    """Assess repayment capacity for a proposed facility.

    ``annual_rate`` defaults to the product's worst-priced band, so affordability
    is never assessed on a keener rate than the applicant might ultimately be
    offered. The decision engine re-assesses at the priced rate once a grade is
    known.
    """
    policy: AffordabilityPolicy = product.affordability
    application = context.get("application") or {}

    declared = m.money(application.get("declared_monthly_income") or 0)
    requested_amount = m.money(
        amount if amount is not None else (application.get("requested_amount") or 0))
    tenor = int(tenor_months if tenor_months is not None
                else (application.get("tenor_months") or product.min_tenor_months))
    rate = (Decimal(annual_rate) if annual_rate is not None
            else max(band.annual_rate for band in product.pricing))

    level = verification_level(context)
    haircut = policy.haircuts.get(level, Decimal("0.5"))

    # Prefer an independently verified income figure where one exists; fall
    # back to the declared figure with the level's haircut applied.
    payroll = context.get("payroll") or {}
    payroll_income = payroll.get("net_monthly_income")
    notes: list[str] = []
    if payroll_income is not None:
        verified = m.money(payroll_income)
        notes.append("income taken from payroll record, not applicant declaration")
        if verified < declared:
            notes.append("payroll income is below the declared figure; the "
                         "lower verified figure is used")
    else:
        verified = m.money(declared * haircut)
        if haircut < 1:
            notes.append(f"declared income reduced by "
                         f"{(1 - haircut) * 100:.0f}% for verification level "
                         f"{level}")

    # Expenses: the greater of declared expenses and the policy floor, plus a
    # per-dependant allowance. A floor prevents an applicant understating
    # living costs from manufacturing capacity that does not exist.
    dependants = int(application.get("dependants") or 0)
    declared_expenses = application.get("declared_monthly_expenses")
    floor = m.money(verified * policy.expense_floor_ratio)
    dependant_allowance = m.money(policy.dependant_expense * Decimal(dependants))
    if declared_expenses is None:
        modelled = m.money(floor + dependant_allowance)
        basis = "policy floor (no declared expenses provided)"
    else:
        declared_expenses = m.money(declared_expenses)
        if declared_expenses >= floor:
            modelled = m.money(declared_expenses + dependant_allowance)
            basis = "declared expenses (above policy floor)"
        else:
            modelled = m.money(floor + dependant_allowance)
            basis = "policy floor (declared expenses below floor)"

    existing = m.money(application.get("existing_monthly_debt_service") or 0)
    buffer_required = m.money(policy.min_disposable_income)

    # --- the two capacity tests ------------------------------------------- #
    dsr_capacity = m.money(verified * policy.max_dsr - existing)
    cashflow_capacity = m.money(verified - modelled - existing - buffer_required)
    dsr_capacity = max(dsr_capacity, m.ZERO)
    cashflow_capacity = max(cashflow_capacity, m.ZERO)

    if cashflow_capacity < dsr_capacity:
        binding = "cash_flow"
    elif dsr_capacity < cashflow_capacity:
        binding = "debt_service_ratio"
    else:
        binding = "equal"
    max_instalment = min(dsr_capacity, cashflow_capacity)

    max_amount = m.principal_for(max_instalment, rate, tenor)
    max_amount = min(max_amount, product.max_amount)

    requested_instalment = (m.instalment_for(requested_amount, rate, tenor)
                            if requested_amount > m.ZERO else m.ZERO)

    disposable = m.money(verified - modelled - existing)
    dsr_before = (m.ratio(existing / verified) if verified > m.ZERO
                  else m.ratio(0))
    dsr_after = (m.ratio((existing + requested_instalment) / verified)
                 if verified > m.ZERO else None)

    affordable = requested_instalment <= max_instalment
    shortfall = m.money(max(requested_instalment - max_instalment, m.ZERO))

    reasons: list[str] = []
    if not affordable:
        reasons.append("AMOUNT_EXCEEDS_AFFORDABILITY")
        if binding == "debt_service_ratio":
            reasons.append("DSR_LIMIT_EXCEEDED")
        else:
            reasons.append("INSUFFICIENT_DISPOSABLE_INCOME")
    if max_instalment <= m.ZERO:
        if "INSUFFICIENT_DISPOSABLE_INCOME" not in reasons:
            reasons.append("INSUFFICIENT_DISPOSABLE_INCOME")
        notes.append("no repayment capacity after expenses, existing "
                     "commitments and the required buffer")

    return AffordabilityResult(
        currency=product.currency,
        verification_level=level,
        haircut_applied=haircut,
        declared_income=declared,
        verified_income=verified,
        modelled_expenses=modelled,
        expense_basis=basis,
        existing_debt_service=existing,
        required_buffer=buffer_required,
        disposable_income=disposable,
        dsr_before=dsr_before,
        dsr_after_requested=dsr_after,
        dsr_capacity_instalment=dsr_capacity,
        cashflow_capacity_instalment=cashflow_capacity,
        max_affordable_instalment=max_instalment,
        binding_constraint=binding,
        annual_rate=Decimal(rate),
        tenor_months=tenor,
        requested_amount=requested_amount,
        requested_instalment=requested_instalment,
        max_affordable_amount=max_amount,
        affordable=affordable,
        shortfall_instalment=shortfall,
        reason_codes=tuple(reasons),
        notes=tuple(notes),
    )
