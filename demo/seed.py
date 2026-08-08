"""Seed the demonstration environment with a plausible history.

A console with empty tables is a poor demonstration: the monitoring view has
nothing to show, the queue is blank and the usage statement reads zero. This
populates the ledger so the first screen a prospect sees is a working service
with a week of activity behind it.

Everything created here is synthetic and labelled as such. The named applicants
are the personas from the requirements specification, so the demonstration and
the documentation tell the same story.

Run:  python3 -m demo.seed            (idempotent: skips if already seeded)
      python3 -m demo.seed --force    (re-seed from scratch)
      python3 -m demo.seed --force --drift

The background portfolio is drawn from the same distributional shapes as the
scorecard's development sample, so population-stability monitoring opens in a
sensible state rather than alarming at a drift caused by the demonstration
data itself. ``--drift`` seeds a deliberately shifted population - younger,
thinner-file, borrowing more relative to income - which trips the PSI
threshold, so a presenter can show what the monitoring is actually for.
"""
from __future__ import annotations

import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from core import batch, config                      # noqa: E402
from core.ledger import DEFAULT_PATH as DEFAULT_LEDGER  # noqa: E402
from core.ledger import Ledger                       # noqa: E402
from core.pipeline import Platform                   # noqa: E402
from partners import simulators                      # noqa: E402

CONSENT = {"credit_bureau_enquiry": True, "automated_decisioning": True,
           "payroll_verification": True}

#: Named scenarios a presenter walks through, in order. The national
#: registration numbers were selected so that the deterministic partner
#: simulators return the intended credit profile for each - a clean file, a
#: thin file, serious arrears - so the demonstration is repeatable without
#: overriding any partner response. The tenant is chosen per scenario, which
#: also shows the same engine applying two different credit policies.
SCENARIOS = [
    ("ZAM-PAY", "PAYROLL_LOAN", "Chanda Mwale - salaried teacher, clean file", {
        "national_id": "632084/37/1", "full_name": "Chanda Mwale",
        "date_of_birth": "1990-03-14", "application_date": "2026-08-03",
        "employer_code": "MOE-LSK-01", "device_id": "demo-seed-01",
        "requested_amount": 20000.0, "tenor_months": 18,
        "declared_monthly_income": 12800.0, "declared_monthly_expenses": 4200.0,
        "existing_monthly_debt_service": 700.0, "dependants": 1,
        "consent": CONSENT}),
    ("ZAM-PAY", "PAYROLL_LOAN", "Bwalya Phiri - creditworthy, asked for too much", {
        "national_id": "749078/36/8", "full_name": "Bwalya Phiri",
        "date_of_birth": "1988-06-02", "application_date": "2026-08-03",
        "employer_code": "GRZ-HR-22", "device_id": "demo-seed-02",
        "requested_amount": 85000.0, "tenor_months": 24,
        "declared_monthly_income": 9500.0, "declared_monthly_expenses": 3400.0,
        "existing_monthly_debt_service": 1500.0, "dependants": 3,
        "consent": CONSENT}),
    ("ZAM-MFI", "MICRO_LOAN", "Mutinta Banda - market trader, no bureau record", {
        "national_id": "414328/41/3", "full_name": "Mutinta Banda",
        "date_of_birth": "1985-09-02", "application_date": "2026-08-03",
        "device_id": "demo-seed-03",
        "requested_amount": 8000.0, "tenor_months": 12,
        "declared_monthly_income": 6200.0,
        "existing_monthly_debt_service": 300.0, "dependants": 4,
        "consent": CONSENT}),
    ("ZAM-PAY", "PAYROLL_LOAN", "Joseph Tembo - arrears on file, declined", {
        "national_id": "702326/48/9", "full_name": "Joseph Tembo",
        "date_of_birth": "1992-11-20", "application_date": "2026-08-03",
        "employer_code": "MOE-LSK-01", "device_id": "demo-seed-04",
        "requested_amount": 25000.0, "tenor_months": 24,
        "declared_monthly_income": 8100.0,
        "existing_monthly_debt_service": 2200.0, "dependants": 2,
        "consent": CONSENT}),
]


def already_seeded(ledger: Ledger) -> bool:
    return ledger.decision_count(tenant_code="ZAM-PAY") > 0


#: Background volume per tenant. Sized so population stability is actually
#: computable: PSI over ten deciles needs roughly fifty observations at an
#: absolute minimum and several hundred to be trustworthy. The earlier value of
#: sixty produced a ten-band index over twenty-seven microfinance decisions,
#: which reported drift that was pure sampling noise. Seeding a real sample is
#: the honest fix; suppressing the metric is only the safety net.
DEFAULT_VOLUME = 400


def seed(*, force: bool = False, volume: int = DEFAULT_VOLUME,
         drift: bool = False, ledger_path=None) -> dict:
    simulators.reset()

    # "Re-seed from scratch" must mean from scratch. Appending to an existing
    # ledger would leave two populations mixed together, which is precisely
    # what makes a monitoring demonstration unreadable.
    target = pathlib.Path(ledger_path) if ledger_path else DEFAULT_LEDGER
    if force and target.exists():
        target.unlink()

    ledger = Ledger(ledger_path) if ledger_path else Ledger()
    platform = Platform(ledger=ledger)

    if already_seeded(ledger) and not force:
        count = ledger.decision_count(tenant_code="ZAM-PAY")
        ledger.close()
        return {"skipped": True, "decisions": count}

    created: dict = {"scenarios": 0, "batch_rows": 0, "tenants": []}

    # 1. The named walkthrough scenarios, each against its own tenant.
    for tenant_code, product_code, label, application in SCENARIOS:
        tenant = config.get_tenant(tenant_code)
        decision, _ = platform.decide(
            {"application": dict(application)}, tenant=tenant,
            product=tenant.products[product_code], actor="demo-seed")
        created["scenarios"] += 1
        marker = decision.outcome + (" (counteroffer)" if decision.is_counteroffer
                                     else "")
        print(f"  [{tenant_code}] {label:<48} {marker}")

    # 2. A background portfolio so monitoring and the queue have shape.
    for tenant_code, product_code, rows in (("ZAM-PAY", "PAYROLL_LOAN", volume),
                                            ("ZAM-MFI", "MICRO_LOAN", volume // 2)):
        tenant = config.get_tenant(tenant_code)
        result = batch.run(
            batch.sample_file(rows, seed=random.randint(1, 10**6),
                              profile="shifted" if drift else "baseline"),
            tenant=tenant, product=tenant.products[product_code],
            platform=platform)
        created["batch_rows"] += result.processed
        created["tenants"].append({
            "tenant": tenant_code, "processed": result.processed,
            "rejected": result.rejected, "outcomes": result.summary()["outcomes"]})
        print(f"  background portfolio {tenant_code}: {result.processed} decided, "
              f"{result.rejected} rejected")

    created["total_decisions"] = sum(
        ledger.decision_count(tenant_code=t.code) for t in config.all_tenants())
    ledger.close()
    return created


def main() -> None:
    print("Seeding demonstration data (synthetic)...")
    result = seed(force="--force" in sys.argv, drift="--drift" in sys.argv)
    if result.get("skipped"):
        print(f"Already seeded ({result['decisions']} decisions). "
              f"Use --force to re-seed.")
        return
    print(f"\nSeeded {result['total_decisions']} decisions across "
          f"{len(config.all_tenants())} tenants.")
    print("Start the service with:  python3 -m api.server")


if __name__ == "__main__":
    main()
