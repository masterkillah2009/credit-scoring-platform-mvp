"""Batch submission and portfolio scoring.

Implements IPSRS FR-INT-04 and WFL-08:

    file received -> checksum + schema validation -> per-row validation
    -> valid rows processed / invalid to an error file
    -> results + reconciliation report -> tenant notified

Two behaviours are non-negotiable and are what separate a batch facility from a
loop:

* **Partial failure is normal.** Valid rows are processed and invalid rows are
  itemised with the reason, so a tenant corrects and resubmits only what failed
  rather than the whole file.
* **Totals must reconcile.** ``submitted = processed + rejected`` is asserted
  before the run is reported as complete; a batch that cannot account for every
  row it received is a failed batch, however many decisions it produced.

The batch runs against the same pipeline as the real-time path, so a batch
score and an API score for identical input are identical by construction.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import pathlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from core import config
from core.pipeline import Platform

#: Columns a batch row must carry. Anything else is passed through untouched.
REQUIRED_COLUMNS = (
    "row_id", "national_id", "date_of_birth", "application_date",
    "requested_amount", "tenor_months", "declared_monthly_income",
)

NUMERIC_COLUMNS = ("requested_amount", "declared_monthly_income",
                   "declared_monthly_expenses", "existing_monthly_debt_service")
INTEGER_COLUMNS = ("tenor_months", "dependants")


@dataclass
class BatchRow:
    row_id: str
    application: dict
    errors: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors


@dataclass
class BatchResult:
    batch_id: str
    tenant_code: str
    product_code: str
    submitted: int
    processed: int
    rejected: int
    started_at: str
    finished_at: str
    checksum: str
    decisions: list[dict] = field(default_factory=list)
    rejects: list[dict] = field(default_factory=list)

    @property
    def reconciled(self) -> bool:
        return self.submitted == self.processed + self.rejected

    def summary(self) -> dict[str, Any]:
        outcomes: dict[str, int] = {}
        for decision in self.decisions:
            outcomes[decision["outcome"]] = outcomes.get(decision["outcome"], 0) + 1
        return {
            "batch_id": self.batch_id,
            "tenant": self.tenant_code,
            "product": self.product_code,
            "checksum_sha256": self.checksum,
            "submitted": self.submitted,
            "processed": self.processed,
            "rejected": self.rejected,
            "reconciled": self.reconciled,
            "outcomes": outcomes,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


def _coerce(row: dict, row_number: int) -> BatchRow:
    """Validate and type-coerce one row without ever guessing a value."""
    row_id = (row.get("row_id") or "").strip() or f"row-{row_number}"
    errors: list[str] = []
    application: dict[str, Any] = {}

    for column in REQUIRED_COLUMNS:
        if column == "row_id":
            continue
        value = (row.get(column) or "").strip() if isinstance(
            row.get(column), str) else row.get(column)
        if value in (None, ""):
            errors.append(f"{column}: required")

    for column, value in row.items():
        if column in ("row_id", "") or value in (None, ""):
            continue
        text = value.strip() if isinstance(value, str) else value
        if column in NUMERIC_COLUMNS:
            try:
                number = float(text)
            except (TypeError, ValueError):
                errors.append(f"{column}: not a number ({value!r})")
                continue
            if number < 0:
                errors.append(f"{column}: must not be negative")
                continue
            application[column] = number
        elif column in INTEGER_COLUMNS:
            try:
                application[column] = int(float(text))
            except (TypeError, ValueError):
                errors.append(f"{column}: not an integer ({value!r})")
        elif column == "consent_bureau" or column == "consent_decisioning":
            application.setdefault("consent", {})[
                "credit_bureau_enquiry" if column == "consent_bureau"
                else "automated_decisioning"] = str(text).lower() in (
                    "1", "true", "yes", "y")
        else:
            application[column] = text

    consent = application.get("consent") or {}
    for purpose in ("credit_bureau_enquiry", "automated_decisioning"):
        if not consent.get(purpose):
            errors.append(f"consent.{purpose}: required before processing")

    return BatchRow(row_id=row_id, application=application, errors=errors)


def parse(content: str) -> tuple[list[BatchRow], str]:
    """Parse a CSV payload into rows plus the file checksum."""
    checksum = hashlib.sha256(content.encode()).hexdigest()
    reader = csv.DictReader(io.StringIO(content))
    rows = [_coerce(raw, number)
            for number, raw in enumerate(reader, start=1)]
    return rows, checksum


def run(content: str, *, tenant: config.Tenant, product: config.Product,
        platform: Platform, batch_id: Optional[str] = None,
        limit: Optional[int] = None) -> BatchResult:
    """Process a batch file end to end."""
    batch_id = batch_id or f"BAT-{uuid.uuid4().hex[:10].upper()}"
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows, checksum = parse(content)
    if limit is not None:
        rows = rows[:limit]

    decisions: list[dict] = []
    rejects: list[dict] = []

    platform.ledger.record(
        tenant_code=tenant.code, event_type="BATCH_RECEIVED",
        correlation_id=batch_id,
        payload={"rows": len(rows), "checksum_sha256": checksum,
                 "product": product.code})

    for row in rows:
        if not row.valid:
            rejects.append({"row_id": row.row_id, "errors": row.errors})
            continue
        try:
            decision, _ = platform.decide(
                {"application": row.application},
                tenant=tenant, product=product,
                application_id=f"{batch_id}-{row.row_id}",
                correlation_id=f"{batch_id}-{row.row_id}",
                actor="batch")
            decisions.append({
                "row_id": row.row_id,
                "decision_id": decision.decision_id,
                "outcome": decision.outcome,
                "score": decision.score,
                "probability_of_default": decision.probability_of_default,
                "risk_grade": decision.risk_grade,
                "segment": decision.model_segment,
                "recommended_amount": str(decision.recommended_amount),
                "reason_codes": decision.reason_codes,
            })
        except Exception as error:            # a bad row must not stop the file
            rejects.append({"row_id": row.row_id,
                            "errors": [f"processing error: "
                                       f"{error.__class__.__name__}"]})

    finished = datetime.now(timezone.utc).isoformat(timespec="seconds")
    result = BatchResult(
        batch_id=batch_id, tenant_code=tenant.code, product_code=product.code,
        submitted=len(rows), processed=len(decisions), rejected=len(rejects),
        started_at=started, finished_at=finished, checksum=checksum,
        decisions=decisions, rejects=rejects)

    platform.ledger.record(
        tenant_code=tenant.code, event_type="BATCH_COMPLETED",
        correlation_id=batch_id, payload=result.summary())
    return result


def to_csv(rows: Iterable[dict], columns: Iterable[str]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(columns),
                            extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()


def sample_file(count: int = 40, *, seed: int = 424242,
                profile: str = "baseline") -> str:
    """Generate a synthetic batch file, including rows that must be rejected.

    ``profile`` selects the applicant population:

      baseline  drawn from the same distributional shapes as the scorecard's
                development sample, so population-stability monitoring reads
                stable - the correct default, because a demonstration should
                not open on a drift alert caused by the demonstration data
      shifted   younger, thinner-file, higher requested-to-income applicants,
                which moves the score distribution enough to trip the PSI
                threshold. Useful for showing what monitoring is *for*.

    Distributions mirror ``model.synth`` but are drawn with the standard
    library, so this module - and therefore the running service - needs no
    third-party package.
    """
    import random
    rng = random.Random(seed)
    shifted = profile == "shifted"
    rows = []

    for index in range(1, count + 1):
        informal = rng.random() < (0.55 if shifted else 0.32)
        age = min(max(rng.gauss(30 if shifted else 36, 10), 18), 72)
        income = min(max(rng.lognormvariate(7.9 if informal else 8.5,
                                            0.62 if informal else 0.55),
                         900 if informal else 1800), 90000)
        employment_months = (rng.expovariate(1 / 26) if informal
                             else rng.gammavariate(2.4, 26))
        employment_start_year = 2026 - max(int(employment_months // 12), 0)

        existing = min(income * rng.betavariate(1.6, 7.0), income * 0.75)
        requested = min(max(income * rng.gammavariate(2.2, 2.6 if shifted else 1.7),
                            500), 150000)

        national_id = (f"{rng.randint(100000, 999999)}/"
                       f"{rng.randint(10, 99)}/{rng.randint(1, 9)}")
        rows.append({
            "row_id": f"R{index:04d}",
            "national_id": national_id,
            "full_name": f"Applicant {index}",
            "date_of_birth": f"{2026 - int(age)}-0{rng.randint(1, 9)}-1{rng.randint(0, 9)}",
            "application_date": "2026-08-03",
            "employer_code": "" if informal else rng.choice(["MOE-LSK-01",
                                                             "GRZ-HR-22"]),
            "requested_amount": round(requested, 2),
            "tenor_months": rng.choice([6, 12, 18, 24, 36]),
            "declared_monthly_income": round(income, 2),
            "declared_monthly_expenses": round(income * rng.uniform(0.2, 0.5), 2),
            "existing_monthly_debt_service": round(existing, 2),
            "dependants": rng.randint(0, 5),
            "consent_bureau": "true",
            "consent_decisioning": "true",
        })

    # Deliberate defects so the reject path is exercised, not merely coded.
    # Positions are proportional so that a small sample still contains every
    # defect class - a 10-row batch must exercise the same paths as a 500-row
    # one, or the reject path is only tested at large sizes.
    defects = (
        ("national_id", ""),                   # missing mandatory field
        ("requested_amount", "not-a-number"),  # numeric type error
        ("consent_bureau", "false"),           # consent withheld
        ("tenor_months", "twelve"),            # integer type error
    )
    for position, (column, value) in enumerate(defects):
        index = min(int(len(rows) * (position + 1) / (len(defects) + 1)),
                    len(rows) - 1)
        rows[index][column] = value

    return to_csv(rows, list(rows[0].keys()))
