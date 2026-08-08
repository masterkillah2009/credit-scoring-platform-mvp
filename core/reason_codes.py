"""Governed reason-code library (IPSRS FR-EXP-01).

Every score and decision must return plain-language reasons appropriate to five
audiences: credit officers, model validators, compliance, regulators and
customers. Vague explanations such as "failed internal policy" are prohibited
where a more meaningful reason can lawfully be given (BRD BR-EXP-02).

Each entry carries a customer-facing sentence and an internal explanation. The
library is versioned: changing wording is a change-controlled act because the
text reaches borrowers.
"""
from __future__ import annotations

from dataclasses import dataclass

LIBRARY_VERSION = "RCL-1.0.0"


@dataclass(frozen=True)
class ReasonCode:
    code: str
    customer_text: str
    internal_text: str
    category: str          # affordability | credit_history | verification |
                           # policy | fraud_aml | data_quality


_CODES: tuple[ReasonCode, ...] = (
    # --- scorecard characteristics ------------------------------------------
    ReasonCode("HIGH_EXISTING_DEBT",
               "Your existing loan repayments are high compared with your verified income.",
               "existing_dsr in an unfavourable band relative to the neutral bin",
               "affordability"),
    ReasonCode("HIGH_REQUESTED_AMOUNT",
               "The amount requested is large relative to your monthly income.",
               "requested_to_income in an unfavourable band",
               "affordability"),
    ReasonCode("RECENT_DELINQUENCY",
               "Recent missed or late payments were found on your credit record.",
               "bureau_worst_dpd in an unfavourable band",
               "credit_history"),
    ReasonCode("MANY_RECENT_ENQUIRIES",
               "There have been several recent credit applications on your record.",
               "bureau_enquiries_6m in an unfavourable band",
               "credit_history"),
    ReasonCode("SHORT_CREDIT_HISTORY",
               "Your credit history is shorter than we typically require.",
               "credit_history_months in an unfavourable band",
               "credit_history"),
    ReasonCode("HIGH_UTILISATION",
               "You are using a high proportion of your available credit.",
               "revolving_utilisation in an unfavourable band",
               "credit_history"),
    ReasonCode("PRIOR_DEFAULT_RECORD",
               "A previous default or written-off account appears on your credit record.",
               "prior_default flag set on the bureau file",
               "credit_history"),
    ReasonCode("SHORT_EMPLOYMENT",
               "Your time with your current employer is shorter than we typically require.",
               "employment_months in an unfavourable band",
               "policy"),
    ReasonCode("AGE_RISK_BAND",
               "Your application falls outside the age profile for this product's best pricing.",
               "age_years in an unfavourable band",
               "policy"),
    ReasonCode("NO_BUREAU_RECORD",
               "We could not find a credit record for you, so we had less information to work with.",
               "no_bureau_record set: thin or no file",
               "data_quality"),
    ReasonCode("LIMITED_RELATIONSHIP",
               "You are new to us, so we have limited history with you.",
               "relationship_months in an unfavourable band",
               "policy"),

    # --- policy and process outcomes ---------------------------------------
    ReasonCode("AMOUNT_EXCEEDS_AFFORDABILITY",
               "The repayment on the amount requested is more than your assessed affordability allows.",
               "requested instalment exceeds max_affordable_instalment",
               "affordability"),
    ReasonCode("INSUFFICIENT_DISPOSABLE_INCOME",
               "After your expenses and existing commitments, too little income remains to support this loan.",
               "disposable income below product floor",
               "affordability"),
    ReasonCode("DSR_LIMIT_EXCEEDED",
               "Your total loan repayments would exceed the limit we can responsibly allow.",
               "post-facility DSR above product ceiling",
               "affordability"),
    ReasonCode("BELOW_MINIMUM_AGE",
               "Applicants must be at least 18 years old.",
               "statutory minimum age not met",
               "policy"),
    ReasonCode("AGE_AT_MATURITY",
               "The loan would end after the maximum age allowed for this product.",
               "age at maturity above product limit",
               "policy"),
    ReasonCode("AMOUNT_OUTSIDE_PRODUCT_RANGE",
               "The amount requested is outside the range offered for this product.",
               "requested amount outside product min/max",
               "policy"),
    ReasonCode("TENOR_OUTSIDE_PRODUCT_RANGE",
               "The repayment period requested is not available for this product.",
               "tenor outside product min/max",
               "policy"),
    ReasonCode("RECENT_SERIOUS_DELINQUENCY",
               "Your credit record shows a serious recent arrears position.",
               "rule R-BUR-01: worst DPD 90+",
               "credit_history"),
    ReasonCode("PRIOR_DEFAULT_OR_WRITE_OFF",
               "A previous default or write-off is recorded against you.",
               "rule R-BUR-02",
               "credit_history"),
    ReasonCode("TOO_MANY_RECENT_ENQUIRIES",
               "There have been many recent credit applications in your name.",
               "rule R-BUR-03",
               "credit_history"),
    ReasonCode("THIN_CREDIT_FILE",
               "We had limited credit information, so a person will review your application.",
               "rule R-THN-01: thin file referral",
               "data_quality"),
    ReasonCode("SHORT_EMPLOYMENT_TENURE",
               "Your employment record is shorter than this product requires.",
               "rule R-PAY-02",
               "policy"),
    ReasonCode("UNABLE_TO_VERIFY_IDENTITY",
               "We could not verify your identity with the details provided.",
               "rule R-KYC-01: identity verification failed",
               "verification"),
    ReasonCode("UNABLE_TO_VERIFY_EMPLOYMENT",
               "We could not verify your employment or salary.",
               "rule R-PAY-01: payroll verification unavailable",
               "verification"),
    ReasonCode("SANCTIONS_MATCH",
               "We are unable to proceed with this application.",
               "rule R-AML-01: sanctions match (customer text deliberately "
               "non-specific; do not tip off)",
               "fraud_aml"),
    ReasonCode("CONFIRMED_FRAUD",
               "We are unable to proceed with this application.",
               "rule R-FRD-01: confirmed fraud (customer text deliberately "
               "non-specific)",
               "fraud_aml"),
    ReasonCode("FRAUD_RISK_REVIEW",
               "Your application needs additional checks before we can decide.",
               "rule R-FRD-02: elevated fraud risk",
               "fraud_aml"),
    ReasonCode("SCORE_BELOW_CUTOFF",
               "Based on the overall assessment, your application did not meet our current lending criteria.",
               "score below product accept cut-off",
               "policy"),
    ReasonCode("SCORE_IN_REFERRAL_BAND",
               "Your application needs a person to review it before we can decide.",
               "score between refer floor and accept cut-off",
               "policy"),
    ReasonCode("INSUFFICIENT_INFORMATION",
               "We need more information before we can assess your application.",
               "mandatory data missing or data-quality breach",
               "data_quality"),
    ReasonCode("PARTNER_DATA_UNAVAILABLE",
               "One of the checks we need could not be completed just now.",
               "external partner unavailable; degradation policy applied",
               "data_quality"),
    ReasonCode("COUNTEROFFER_REDUCED_AMOUNT",
               "We can offer a smaller amount than you requested, based on affordability.",
               "amount capped to affordability maximum",
               "affordability"),
)

BY_CODE: dict[str, ReasonCode] = {c.code: c for c in _CODES}

#: Maps a scorecard characteristic to the reason code used when that
#: characteristic costs the applicant the most points.
CHARACTERISTIC_CODES: dict[str, str] = {
    "existing_dsr": "HIGH_EXISTING_DEBT",
    "requested_to_income": "HIGH_REQUESTED_AMOUNT",
    "bureau_worst_dpd": "RECENT_DELINQUENCY",
    "bureau_enquiries_6m": "MANY_RECENT_ENQUIRIES",
    "credit_history_months": "SHORT_CREDIT_HISTORY",
    "revolving_utilisation": "HIGH_UTILISATION",
    "prior_default": "PRIOR_DEFAULT_RECORD",
    "employment_months": "SHORT_EMPLOYMENT",
    "age_years": "AGE_RISK_BAND",
    "no_bureau_record": "NO_BUREAU_RECORD",
    "relationship_months": "LIMITED_RELATIONSHIP",
}


def describe(code: str, *, audience: str = "customer") -> str:
    reason = BY_CODE.get(code)
    if reason is None:
        # Never emit an unmapped code to a customer.
        return ("We need more information before we can assess your application."
                if audience == "customer" else f"unmapped reason code {code!r}")
    return reason.customer_text if audience == "customer" else reason.internal_text


def render(codes: list[str], *, audience: str = "customer") -> list[dict]:
    out = []
    for code in codes:
        reason = BY_CODE.get(code)
        out.append({
            "code": code,
            "text": describe(code, audience=audience),
            "category": reason.category if reason else "unknown",
        })
    return out
