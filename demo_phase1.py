"""Phase 1 demonstration: feature store -> scorecard -> score, grade, reasons.

Run:  python3 demo_phase1.py

Shows three illustrative applicants scored for two tenants that share one PD
model but present different score scales, grades and cut-offs.
"""
from __future__ import annotations

from core import config, features, scorecard

APPLICANTS: dict[str, dict] = {
    "Chanda - salaried teacher, clean file": {
        "application": {
            "application_date": "2026-07-19",
            "date_of_birth": "1997-03-14",
            "employment_start_date": "2021-02-01",
            "declared_monthly_income": 9800.0,
            "existing_monthly_debt_service": 900.0,
            "requested_amount": 15000.0,
            "tenor_months": 12,
            "dependants": 1,
            "gender": "F",
        },
        "bureau": {
            "worst_dpd_12m": 0, "open_facilities": 2, "enquiries_6m": 1,
            "history_months": 74, "revolving_utilisation": 0.22,
            "prior_default": False,
        },
        "payroll": {"employment_months": 65, "verified": True},
        "internal": {"relationship_months": 40},
    },
    "Mutinta - market trader, no bureau record": {
        "application": {
            "application_date": "2026-07-19",
            "date_of_birth": "1985-09-02",
            "employment_start_date": None,
            "declared_monthly_income": 4200.0,
            "existing_monthly_debt_service": 300.0,
            "requested_amount": 6000.0,
            "tenor_months": 9,
            "dependants": 4,
            "gender": "F",
        },
        "bureau": None,                      # thin file: no record at all
        "payroll": None,
        "internal": {"relationship_months": 8},
    },
    "Joseph - salaried, recent arrears and high utilisation": {
        "application": {
            "application_date": "2026-07-19",
            "date_of_birth": "1990-11-20",
            "employment_start_date": "2025-10-01",
            "declared_monthly_income": 7300.0,
            "existing_monthly_debt_service": 2600.0,
            "requested_amount": 30000.0,
            "tenor_months": 24,
            "dependants": 3,
            "gender": "M",
        },
        "bureau": {
            "worst_dpd_12m": 60, "open_facilities": 5, "enquiries_6m": 4,
            "history_months": 26, "revolving_utilisation": 0.94,
            "prior_default": False,
        },
        "payroll": {"employment_months": 9, "verified": True},
        "internal": {"relationship_months": 5},
    },
}


def main() -> None:
    card = scorecard.load()
    print("=" * 78)
    print(f"Scorecard {card.model_id} v{card.model_version}")
    print(f"  feature set : {card.feature_set_version}")
    print(f"  artefact    : sha256 {card.sha256[:16]}...")
    print(f"  governance  : {card.governance.get('status')}")
    print(f"  type        : {card.model_type}")
    print(f"  segments    : {card.segmentation['rule']}")
    for name in card.segments:
        metrics = card.segment_performance(name)
        print(f"    {name:<7} out-of-time Gini {metrics['gini']:.3f}  "
              f"KS {metrics['ks']:.3f}  bad rate {metrics['bad_rate']:.1%}  "
              f"({len(card.segments[name].characteristics)} characteristics)")
    print("=" * 78)

    for name, context in APPLICANTS.items():
        print(f"\n### {name}")
        computed = features.compute(context)
        print(f"  data quality : {computed.status}"
              f"{'  (thin file)' if computed.thin_file else ''}")
        if computed.missing:
            print(f"  missing      : {', '.join(computed.missing)}")
        for note in computed.notes:
            print(f"  note         : {note}")

        for tenant in config.all_tenants():
            product = next(iter(tenant.products.values()))
            result = card.score(computed.values, tenant=tenant,
                               dq_status=computed.status)
            gate = ("PASS" if result.score >= product.accept_cutoff
                    else "REFER" if result.score >= product.refer_floor
                    else "FAIL")
            print(f"  {tenant.code:<8} [{result.segment}] "
                  f"PD={result.probability_of_default:6.2%}  "
                  f"score={result.score:<5} grade={result.risk_grade}  "
                  f"scale[{tenant.score_scale.min_score}-"
                  f"{tenant.score_scale.max_score}]  "
                  f"cutoff={product.accept_cutoff} -> score gate {gate}  "
                  f"confidence={result.confidence}")
            if tenant.code == "ZAM-PAY":
                for reason in result.as_dict()["reason_codes"]:
                    print(f"           reason: [{reason['code']}] {reason['text']}")

    # One PD, many scales - proves configuration-over-code (BR-SCR-02)
    print("\n" + "=" * 78)
    print("Same PD presented on each tenant's configured scale:")
    for pd in (0.01, 0.03, 0.08, 0.20, 0.40):
        row = "  ".join(
            f"{t.code}={scorecard.pd_to_score(pd, t.score_scale):>4}"
            for t in config.all_tenants())
        print(f"  PD {pd:6.2%} -> {row}")
    print("=" * 78)


if __name__ == "__main__":
    main()
