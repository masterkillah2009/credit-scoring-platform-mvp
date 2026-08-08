"""Tenant, product and credit-policy configuration.

Implements IPSRS FR-ADM-01/04 (tenant configuration, versioned) and FR-DEC-04
(policy applied by version and recorded on every decision) at prototype
fidelity: configuration is held in immutable dataclasses rather than a database
so the vertical slice runs with no external dependencies.

Two tenants are configured deliberately:
  * ZAM-PAY  payroll lender      score scale 300-850, DSR ceiling 40%
  * ZAM-MFI  microfinance lender score scale 0-1000,  DSR ceiling 50%

They share one PD model but present different scales, grades, cut-offs and
policy - demonstrating configuration-over-code (BRD BR-TEN-03, BR-SCR-02).
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any


# --------------------------------------------------------------------------- #
# Score scaling and grades
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ScoreScale:
    """Points-based scaling of a probability of default (IPSRS FR-SCO-02)."""

    base_score: int
    base_odds: float          # good:bad odds at base_score
    pdo: int                  # points to double the odds
    min_score: int
    max_score: int


@dataclass(frozen=True)
class RiskGrade:
    code: str
    min_score: int            # inclusive
    max_score: int            # inclusive


# --------------------------------------------------------------------------- #
# Affordability policy
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AffordabilityPolicy:
    """BRD BR-AFF-02: haircuts and floors configurable per tenant/product."""

    haircuts: dict[str, Decimal]      # income haircut by verification level
    max_dsr: Decimal                  # total debt service / net income ceiling
    min_disposable_income: Decimal    # absolute floor, tenant currency
    expense_floor_ratio: Decimal      # minimum living cost as share of income
    dependant_expense: Decimal        # per dependant, per month


# --------------------------------------------------------------------------- #
# Declarative policy rules (IPSRS FR-DEC-01/04)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Rule:
    """A single declarative policy rule.

    Rules are data, never code: the engine interprets field/op/value so policy
    can change without a release. ``kind`` determines the effect of a match:

      hard_decline  immediate decline, not overridable by score
      soft_decline  decline unless a referral rule fires first
      refer         route to manual underwriting
      insufficient  insufficient information (verification or data quality)
    """

    id: str
    kind: str
    field: str
    op: str
    value: Any
    reason_code: str
    description: str


@dataclass(frozen=True)
class PricingBand:
    grade: str
    annual_rate: Decimal      # nominal annual interest rate
    fee_ratio: Decimal        # origination fee as share of principal


@dataclass(frozen=True)
class Product:
    code: str
    name: str
    currency: str
    min_amount: Decimal
    max_amount: Decimal
    min_tenor_months: int
    max_tenor_months: int
    min_age: int
    max_age_at_maturity: int
    offer_validity_days: int
    # Counteroffers are rounded down to this increment. Lenders do not offer
    # ZMW 41,903.13; they offer 41,900. Rounding down never breaches the
    # affordability ceiling that produced the figure.
    offer_increment: Decimal
    requires_payroll: bool
    scorecard_id: str
    accept_cutoff: int        # score at/above this passes the score gate
    refer_floor: int          # score in [refer_floor, accept_cutoff) refers
    pricing: tuple[PricingBand, ...]
    affordability: AffordabilityPolicy
    rules: tuple[Rule, ...]


@dataclass(frozen=True)
class Tenant:
    code: str
    name: str
    api_key: str
    country: str
    currency: str
    isolation_tier: str
    policy_version: str
    score_scale: ScoreScale
    grades: tuple[RiskGrade, ...]
    products: dict[str, Product]
    # Behaviour when an external partner is unavailable
    # (BRD BR-DAT-05 / IPSRS FR-CNX-03): refer | partial | decline
    degradation_policy: str = "refer"
    # Data-quality breach behaviour (IPSRS FR-FST-04): block | flag
    dq_policy: str = "block"


# --------------------------------------------------------------------------- #
# Shared rule sets
# --------------------------------------------------------------------------- #
def _common_rules() -> tuple[Rule, ...]:
    """Rules every product inherits. Platform invariants (BRL-02/03) first."""
    return (
        Rule("R-KYC-01", "insufficient", "identity.verified", "is_false",
             None, "UNABLE_TO_VERIFY_IDENTITY",
             "Identity could not be verified through the configured route"),
        Rule("R-AML-01", "hard_decline", "screening.sanctions_hit", "is_true",
             None, "SANCTIONS_MATCH",
             "Confirmed sanctions match (not overridable)"),
        Rule("R-FRD-01", "hard_decline", "fraud.confirmed", "is_true",
             None, "CONFIRMED_FRAUD",
             "Confirmed fraud indicator (not overridable)"),
        Rule("R-FRD-02", "refer", "fraud.risk_level", "eq", "HIGH",
             "FRAUD_RISK_REVIEW", "Elevated fraud risk requires manual review"),
        Rule("R-AGE-01", "hard_decline", "features.age_years", "lt", 18,
             "BELOW_MINIMUM_AGE", "Applicant below statutory minimum age"),
        Rule("R-BUR-01", "soft_decline", "features.bureau_worst_dpd", "gte", 90,
             "RECENT_SERIOUS_DELINQUENCY",
             "Serious delinquency (90+ days) on an existing facility"),
        Rule("R-BUR-02", "soft_decline", "features.prior_default", "is_true",
             None, "PRIOR_DEFAULT_OR_WRITE_OFF",
             "Previous default or write-off recorded"),
        Rule("R-BUR-03", "refer", "features.bureau_enquiries_6m", "gte", 6,
             "TOO_MANY_RECENT_ENQUIRIES",
             "High number of recent credit enquiries"),
        Rule("R-THN-01", "refer", "dq.thin_file", "is_true", None,
             "THIN_CREDIT_FILE",
             "Limited credit history: manual assessment required"),
    )


def _payroll_product() -> Product:
    return Product(
        code="PAYROLL_LOAN",
        name="Payroll-deduction loan",
        currency="ZMW",
        min_amount=Decimal("1000"),
        max_amount=Decimal("150000"),
        min_tenor_months=3,
        max_tenor_months=48,
        min_age=18,
        max_age_at_maturity=60,
        offer_validity_days=14,
        offer_increment=Decimal("100"),
        requires_payroll=True,
        scorecard_id="APPLICATION_LR_V1",
        # Derived by model.calibrate_cutoffs from the validation + out-of-time
        # score distribution against an 8% target bad rate, not set by eye.
        accept_cutoff=649,   # approval 22.5%, approved bad rate 6.5%
        refer_floor=619,     # referral band = 1.5 x PDO
        pricing=(
            PricingBand("A", Decimal("0.22"), Decimal("0.010")),
            PricingBand("B", Decimal("0.28"), Decimal("0.015")),
            PricingBand("C", Decimal("0.34"), Decimal("0.020")),
            PricingBand("D", Decimal("0.42"), Decimal("0.025")),
            PricingBand("E", Decimal("0.50"), Decimal("0.030")),
        ),
        affordability=AffordabilityPolicy(
            haircuts={
                "DECLARED": Decimal("0.50"),
                "DOCUMENTED": Decimal("0.75"),
                "PAYROLL_VERIFIED": Decimal("1.00"),
                "TRANSACTION_VERIFIED": Decimal("1.00"),
            },
            max_dsr=Decimal("0.40"),
            min_disposable_income=Decimal("1200"),
            expense_floor_ratio=Decimal("0.35"),
            dependant_expense=Decimal("350"),
        ),
        rules=_common_rules() + (
            Rule("R-PAY-01", "insufficient", "employment.payroll_verified",
                 "is_false", None, "UNABLE_TO_VERIFY_EMPLOYMENT",
                 "Payroll verification required for this product"),
            Rule("R-PAY-02", "refer", "features.employment_months", "lt", 6,
                 "SHORT_EMPLOYMENT_TENURE",
                 "Employment tenure below product threshold"),
        ),
    )


def _micro_product() -> Product:
    return Product(
        code="MICRO_LOAN",
        name="Microfinance instalment loan",
        currency="ZMW",
        min_amount=Decimal("500"),
        max_amount=Decimal("25000"),
        min_tenor_months=3,
        max_tenor_months=18,
        min_age=18,
        max_age_at_maturity=70,
        offer_validity_days=7,
        offer_increment=Decimal("50"),
        requires_payroll=False,
        scorecard_id="APPLICATION_LR_V1",
        # Derived by model.calibrate_cutoffs against an 18% target bad rate.
        accept_cutoff=447,   # approval 64.0%, approved bad rate 11.1%
        refer_floor=387,     # referral band = 1.5 x PDO
        pricing=(
            PricingBand("A", Decimal("0.36"), Decimal("0.02")),
            PricingBand("B", Decimal("0.44"), Decimal("0.02")),
            PricingBand("C", Decimal("0.52"), Decimal("0.03")),
            PricingBand("D", Decimal("0.60"), Decimal("0.03")),
            PricingBand("E", Decimal("0.68"), Decimal("0.04")),
        ),
        affordability=AffordabilityPolicy(
            haircuts={
                "DECLARED": Decimal("0.40"),
                "DOCUMENTED": Decimal("0.70"),
                "PAYROLL_VERIFIED": Decimal("1.00"),
                "TRANSACTION_VERIFIED": Decimal("0.90"),
            },
            max_dsr=Decimal("0.50"),
            min_disposable_income=Decimal("600"),
            expense_floor_ratio=Decimal("0.40"),
            dependant_expense=Decimal("250"),
        ),
        rules=_common_rules(),
    )


# --------------------------------------------------------------------------- #
# Tenant registry
# --------------------------------------------------------------------------- #
_TENANTS: dict[str, Tenant] = {
    "ZAM-PAY": Tenant(
        code="ZAM-PAY",
        name="Zambezi Payroll Finance (demo)",
        api_key="demo-key-payroll",
        country="ZM",
        currency="ZMW",
        isolation_tier="SHARED_SCHEMA",
        policy_version="POL-2026.07-A",
        score_scale=ScoreScale(base_score=660, base_odds=15.0, pdo=20,
                               min_score=300, max_score=850),
        grades=(
            RiskGrade("A", 720, 850),
            RiskGrade("B", 660, 719),
            RiskGrade("C", 620, 659),
            RiskGrade("D", 560, 619),
            RiskGrade("E", 300, 559),
        ),
        products={"PAYROLL_LOAN": _payroll_product()},
        degradation_policy="refer",
        dq_policy="block",
    ),
    "ZAM-MFI": Tenant(
        code="ZAM-MFI",
        name="Kabwata Microfinance (demo)",
        api_key="demo-key-micro",
        country="ZM",
        currency="ZMW",
        isolation_tier="DEDICATED_SCHEMA",
        policy_version="POL-2026.07-M",
        score_scale=ScoreScale(base_score=500, base_odds=10.0, pdo=40,
                               min_score=0, max_score=1000),
        grades=(
            RiskGrade("A", 620, 1000),
            RiskGrade("B", 560, 619),
            RiskGrade("C", 520, 559),
            RiskGrade("D", 460, 519),
            RiskGrade("E", 0, 459),
        ),
        products={"MICRO_LOAN": _micro_product()},
        degradation_policy="partial",
        dq_policy="flag",
    ),
}


def get_tenant(code: str) -> Tenant:
    if code not in _TENANTS:
        raise KeyError(f"unknown tenant {code!r}")
    return _TENANTS[code]


def tenant_for_api_key(api_key: str) -> Tenant | None:
    for tenant in _TENANTS.values():
        if tenant.api_key == api_key:
            return tenant
    return None


def all_tenants() -> tuple[Tenant, ...]:
    return tuple(_TENANTS.values())


def grade_for_score(tenant: Tenant, score: int) -> str:
    for grade in tenant.grades:
        if grade.min_score <= score <= grade.max_score:
            return grade.code
    return tenant.grades[-1].code


def pricing_for_grade(product: Product, grade: str) -> PricingBand:
    for band in product.pricing:
        if band.grade == grade:
            return band
    return product.pricing[-1]
