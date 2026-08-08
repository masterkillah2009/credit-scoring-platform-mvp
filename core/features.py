"""Versioned feature store.

Implements the IPSRS FR-FST module at prototype fidelity:

  FR-FST-01  one definition per feature serves real-time scoring, batch and
             training - this module is the only place features are computed
  FR-FST-02  a feature cannot be activated without complete metadata
  FR-FST-03  documented missing-value treatment; missing NEVER defaults to zero
  FR-FST-04  data-quality thresholds evaluated at scoring time
  FR-FST-06  sensitive attributes are excluded from production scoring

The feature-set version is a content hash of the active definitions, so any
change to a formula, window or missing treatment changes the version recorded
on every score (BRD BR-SCR-03, IPSRS FR-SCO-01).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Any, Callable, Optional

FEATURE_SET_ID = "FS_APPLICATION"

# Treatments for a missing input. Silent zero is deliberately absent.
MISSING_TREATMENTS = ("dedicated_bin", "policy_referral", "block")


@dataclass(frozen=True)
class FeatureDefinition:
    """Full metadata contract required by IPSRS FR-FST-02."""

    name: str
    business_definition: str
    source_system: str
    owner: str
    formula: str                 # human-readable formula, mirrors the callable
    observation_window: str
    data_type: str               # numeric | boolean
    min_value: Optional[float]
    max_value: Optional[float]
    missing_treatment: str       # one of MISSING_TREATMENTS
    permitted_purpose: str
    jurisdictions: tuple[str, ...]
    refresh: str
    used_by_models: tuple[str, ...]
    retention: str
    sensitive: bool = False      # sensitive attributes never reach scoring

    def validate(self) -> None:
        missing_meta = [
            field for field, value in asdict(self).items()
            if value in (None, "", ()) and field not in ("min_value", "max_value")
        ]
        if missing_meta:
            raise ValueError(
                f"feature {self.name!r} cannot be activated: incomplete "
                f"metadata for {sorted(missing_meta)}"
            )
        if self.missing_treatment not in MISSING_TREATMENTS:
            raise ValueError(
                f"feature {self.name!r}: unknown missing treatment "
                f"{self.missing_treatment!r} (silent zero is not permitted)"
            )


@dataclass
class FeatureValues:
    """Computed features plus the data-quality verdict for one application."""

    values: dict[str, Any]
    missing: tuple[str, ...]
    thin_file: bool
    status: str                  # OK | DEGRADED | BLOCK
    feature_set_version: str
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_set_version": self.feature_set_version,
            "values": dict(self.values),
            "missing": list(self.missing),
            "thin_file": self.thin_file,
            "status": self.status,
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------------- #
# Definitions
# --------------------------------------------------------------------------- #
def _d(**kwargs: Any) -> FeatureDefinition:
    definition = FeatureDefinition(**kwargs)
    definition.validate()
    return definition


_MODELS = ("APPLICATION_LR_V1",)
_ZM = ("ZM",)

DEFINITIONS: tuple[FeatureDefinition, ...] = (
    _d(name="age_years",
       business_definition="Applicant age in whole years at application date",
       source_system="application", owner="credit_risk",
       formula="floor((application_date - date_of_birth) / 365.25)",
       observation_window="point-in-time", data_type="numeric",
       min_value=18, max_value=100, missing_treatment="block",
       permitted_purpose="credit_risk_and_eligibility", jurisdictions=_ZM,
       refresh="per_application", used_by_models=_MODELS,
       retention="7y_after_closure"),
    _d(name="employment_months",
       business_definition="Months of continuous service with current employer",
       source_system="payroll_or_application", owner="credit_risk",
       formula="months_between(application_date, employment_start_date)",
       observation_window="point-in-time", data_type="numeric",
       min_value=0, max_value=600, missing_treatment="dedicated_bin",
       permitted_purpose="credit_risk", jurisdictions=_ZM,
       refresh="per_application", used_by_models=_MODELS,
       retention="7y_after_closure"),
    _d(name="existing_dsr",
       business_definition=("Existing monthly debt service divided by verified "
                            "monthly net income"),
       source_system="bureau_and_payroll", owner="credit_risk",
       formula="existing_monthly_debt_service / verified_monthly_income",
       observation_window="current", data_type="numeric",
       min_value=0, max_value=3, missing_treatment="dedicated_bin",
       permitted_purpose="credit_risk", jurisdictions=_ZM,
       refresh="per_application", used_by_models=_MODELS,
       retention="7y_after_closure"),
    _d(name="requested_to_income",
       business_definition="Requested principal divided by monthly net income",
       source_system="application", owner="credit_risk",
       formula="requested_amount / verified_monthly_income",
       observation_window="point-in-time", data_type="numeric",
       min_value=0, max_value=60, missing_treatment="block",
       permitted_purpose="credit_risk", jurisdictions=_ZM,
       refresh="per_application", used_by_models=_MODELS,
       retention="7y_after_closure"),
    _d(name="bureau_worst_dpd",
       business_definition=("Worst days-past-due status recorded on any bureau "
                            "facility in the last 12 months"),
       source_system="credit_bureau", owner="credit_risk",
       formula="max(days_past_due) over bureau facilities, 12m window",
       observation_window="12 months", data_type="numeric",
       min_value=0, max_value=999, missing_treatment="dedicated_bin",
       permitted_purpose="credit_risk", jurisdictions=_ZM,
       refresh="per_application", used_by_models=_MODELS,
       retention="per_bureau_terms"),
    _d(name="bureau_open_facilities",
       business_definition="Count of open credit facilities at the bureau",
       source_system="credit_bureau", owner="credit_risk",
       formula="count(facilities where status = OPEN)",
       observation_window="current", data_type="numeric",
       min_value=0, max_value=50, missing_treatment="dedicated_bin",
       permitted_purpose="credit_risk", jurisdictions=_ZM,
       refresh="per_application", used_by_models=_MODELS,
       retention="per_bureau_terms"),
    _d(name="bureau_enquiries_6m",
       business_definition="Credit enquiries recorded in the last 6 months",
       source_system="credit_bureau", owner="credit_risk",
       formula="count(enquiries in trailing 6 months)",
       observation_window="6 months", data_type="numeric",
       min_value=0, max_value=50, missing_treatment="dedicated_bin",
       permitted_purpose="credit_risk", jurisdictions=_ZM,
       refresh="per_application", used_by_models=_MODELS,
       retention="per_bureau_terms"),
    _d(name="credit_history_months",
       business_definition="Months since the oldest bureau facility opened",
       source_system="credit_bureau", owner="credit_risk",
       formula="months_between(application_date, min(facility_open_date))",
       observation_window="lifetime", data_type="numeric",
       min_value=0, max_value=600, missing_treatment="dedicated_bin",
       permitted_purpose="credit_risk", jurisdictions=_ZM,
       refresh="per_application", used_by_models=_MODELS,
       retention="per_bureau_terms"),
    _d(name="revolving_utilisation",
       business_definition=("Revolving balances divided by revolving limits "
                            "across bureau facilities"),
       source_system="credit_bureau", owner="credit_risk",
       formula="sum(revolving_balance) / sum(revolving_limit)",
       observation_window="current", data_type="numeric",
       min_value=0, max_value=2, missing_treatment="dedicated_bin",
       permitted_purpose="credit_risk", jurisdictions=_ZM,
       refresh="per_application", used_by_models=_MODELS,
       retention="per_bureau_terms"),
    _d(name="prior_default",
       business_definition=("Whether any default, write-off or restructure is "
                            "recorded on the bureau file"),
       source_system="credit_bureau", owner="credit_risk",
       formula="any(facility.status in {DEFAULT, WRITTEN_OFF, RESTRUCTURED})",
       observation_window="lifetime", data_type="boolean",
       min_value=None, max_value=None, missing_treatment="dedicated_bin",
       permitted_purpose="credit_risk", jurisdictions=_ZM,
       refresh="per_application", used_by_models=_MODELS,
       retention="per_bureau_terms"),
    _d(name="no_bureau_record",
       business_definition=("Whether the applicant has no retrievable credit "
                            "bureau record (thin or no file)"),
       source_system="derived", owner="credit_risk",
       formula="not any(bureau field retrieved)",
       observation_window="point-in-time", data_type="boolean",
       min_value=None, max_value=None, missing_treatment="dedicated_bin",
       permitted_purpose="credit_risk", jurisdictions=_ZM,
       refresh="per_application", used_by_models=_MODELS,
       retention="7y_after_closure"),
    _d(name="relationship_months",
       business_definition="Months since the customer's first product with the lender",
       source_system="core_banking", owner="credit_risk",
       formula="months_between(application_date, first_relationship_date)",
       observation_window="lifetime", data_type="numeric",
       min_value=0, max_value=600, missing_treatment="dedicated_bin",
       permitted_purpose="credit_risk", jurisdictions=_ZM,
       refresh="per_application", used_by_models=_MODELS,
       retention="7y_after_closure"),
    # Held for fairness testing only - excluded from scoring (FR-FST-06)
    _d(name="gender",
       business_definition="Applicant gender as recorded for regulatory reporting",
       source_system="application", owner="compliance",
       formula="application.gender",
       observation_window="point-in-time", data_type="boolean",
       min_value=None, max_value=None, missing_treatment="dedicated_bin",
       permitted_purpose="fairness_testing_and_regulatory_reporting_only",
       jurisdictions=_ZM, refresh="per_application",
       used_by_models=("NONE",), retention="7y_after_closure",
       sensitive=True),
)

BY_NAME: dict[str, FeatureDefinition] = {d.name: d for d in DEFINITIONS}

#: Features permitted in production scoring - sensitive attributes removed.
SCORING_FEATURES: tuple[str, ...] = tuple(
    d.name for d in DEFINITIONS if not d.sensitive
)

# Bureau-sourced features; all missing together implies a thin/no file.
_BUREAU_FEATURES = tuple(
    d.name for d in DEFINITIONS if d.source_system == "credit_bureau"
)


def feature_set_version() -> str:
    """Content hash of the active definitions (changes if any metadata changes)."""
    payload = json.dumps([asdict(d) for d in DEFINITIONS], sort_keys=True,
                         default=str)
    digest = hashlib.sha256(payload.encode()).hexdigest()[:12]
    return f"{FEATURE_SET_ID}@{digest}"


# --------------------------------------------------------------------------- #
# Computation
# --------------------------------------------------------------------------- #
def _months_between(later: str, earlier: Optional[str]) -> Optional[int]:
    if not earlier or not later:
        return None
    ly, lm, ld = (int(p) for p in later.split("-"))
    ey, em, ed = (int(p) for p in earlier.split("-"))
    months = (ly - ey) * 12 + (lm - em) - (1 if ld < ed else 0)
    return max(months, 0)


def _years_between(later: str, earlier: Optional[str]) -> Optional[int]:
    months = _months_between(later, earlier)
    return None if months is None else months // 12


def _safe_ratio(numerator: Optional[float],
                denominator: Optional[float]) -> Optional[float]:
    """Ratio that returns None - not zero - when inputs are absent."""
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


_COMPUTERS: dict[str, Callable[[dict], Any]] = {
    "age_years": lambda ctx: _years_between(
        ctx["application"]["application_date"],
        ctx["application"].get("date_of_birth")),
    "employment_months": lambda ctx: (
        ctx["payroll"].get("employment_months")
        if ctx.get("payroll") else
        _months_between(ctx["application"]["application_date"],
                        ctx["application"].get("employment_start_date"))),
    "existing_dsr": lambda ctx: _safe_ratio(
        ctx["application"].get("existing_monthly_debt_service"),
        ctx["application"].get("declared_monthly_income")),
    "requested_to_income": lambda ctx: _safe_ratio(
        ctx["application"].get("requested_amount"),
        ctx["application"].get("declared_monthly_income")),
    "bureau_worst_dpd": lambda ctx: (ctx.get("bureau") or {}).get("worst_dpd_12m"),
    "bureau_open_facilities": lambda ctx: (ctx.get("bureau") or {}).get("open_facilities"),
    "bureau_enquiries_6m": lambda ctx: (ctx.get("bureau") or {}).get("enquiries_6m"),
    "credit_history_months": lambda ctx: (ctx.get("bureau") or {}).get("history_months"),
    "revolving_utilisation": lambda ctx: (ctx.get("bureau") or {}).get("revolving_utilisation"),
    "prior_default": lambda ctx: (ctx.get("bureau") or {}).get("prior_default"),
    # Derived: the absence of a bureau record is itself information, carried
    # once by this characteristic so bureau features do not double-count it.
    "no_bureau_record": lambda ctx: not bool(ctx.get("bureau")),
    "relationship_months": lambda ctx: (ctx.get("internal") or {}).get("relationship_months"),
    "gender": lambda ctx: ctx["application"].get("gender"),
}


def compute(context: dict, *, dq_policy: str = "block") -> FeatureValues:
    """Compute the scoring feature vector and its data-quality verdict.

    ``context`` carries the application payload plus whatever partner data the
    orchestrator obtained (bureau, payroll, internal). Absent partner data
    yields ``None`` values, never zeros: each feature's documented treatment
    then decides whether scoring proceeds (FR-FST-03).
    """
    values: dict[str, Any] = {}
    missing: list[str] = []
    notes: list[str] = []

    for name in SCORING_FEATURES:
        definition = BY_NAME[name]
        try:
            raw = _COMPUTERS[name](context)
        except (KeyError, TypeError, ValueError):
            raw = None

        if raw is None:
            missing.append(name)
            values[name] = None
            continue

        if definition.data_type == "numeric":
            raw = float(raw)
            if definition.min_value is not None and raw < definition.min_value:
                notes.append(f"{name} below expected range, clamped for scoring")
                raw = float(definition.min_value)
            if definition.max_value is not None and raw > definition.max_value:
                notes.append(f"{name} above expected range, clamped for scoring")
                raw = float(definition.max_value)
        elif definition.data_type == "boolean":
            raw = bool(raw)
        values[name] = raw

    thin_file = all(values.get(f) is None for f in _BUREAU_FEATURES)
    blocking = [m for m in missing if BY_NAME[m].missing_treatment == "block"]

    if blocking:
        status = "BLOCK"
        notes.append("mandatory features missing: " + ", ".join(sorted(blocking)))
    elif thin_file:
        # A thin file is a referral condition (policy rule R-THN-01), not a
        # scoring failure: the dedicated missing bin carries its own weight.
        status = "DEGRADED"
        notes.append("no bureau record available: thin-file treatment applied")
    elif missing:
        status = "DEGRADED"
    else:
        status = "OK"

    return FeatureValues(
        values=values,
        missing=tuple(sorted(missing)),
        thin_file=thin_file,
        status=status,
        feature_set_version=feature_set_version(),
        notes=tuple(notes),
    )


def dictionary() -> list[dict]:
    """Feature dictionary export (BRD BR-FEA-02 deliverable)."""
    return [asdict(d) for d in DEFINITIONS]
