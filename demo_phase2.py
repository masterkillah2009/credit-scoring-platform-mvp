"""Phase 2 demonstration: score + affordability -> decision, with full trace.

Run:  python3 demo_phase2.py
      python3 demo_phase2.py --json     (full decision contract for one case)

Six scenarios exercise every outcome type the decision engine can produce:
approve, counteroffer, refer (thin file), refer (score band), hard decline
(sanctions), soft decline (delinquency) and insufficient information.
"""
from __future__ import annotations

import json
import sys
from decimal import Decimal

from core import config, pipeline

BASE_PARTNERS = {
    "identity": {"verified": True},
    "screening": {"sanctions_hit": False, "pep_hit": False},
    "fraud": {"confirmed": False, "risk_level": "LOW"},
    "employment": {"payroll_verified": True},
}


def case(name: str, *, application: dict, bureau=None, payroll=None,
         internal=None, **overrides) -> tuple[str, dict]:
    context = {
        "application": application,
        "bureau": bureau,
        "payroll": payroll,
        "internal": internal or {"relationship_months": 24},
        **{k: dict(v) for k, v in BASE_PARTNERS.items()},
    }
    for key, value in overrides.items():
        context[key] = {**context.get(key, {}), **value}
    return name, context


CLEAN_BUREAU = {"worst_dpd_12m": 0, "open_facilities": 2, "enquiries_6m": 1,
                "history_months": 74, "revolving_utilisation": 0.22,
                "prior_default": False}

SCENARIOS = [
    case("1. Approve - salaried, clean file, affordable",
         application={
             "application_date": "2026-07-19", "date_of_birth": "1990-03-14",
             "employment_start_date": "2018-02-01",
             "declared_monthly_income": 12800.0,
             "declared_monthly_expenses": 4200.0,
             "existing_monthly_debt_service": 700.0,
             "requested_amount": 20000.0, "tenor_months": 18, "dependants": 1,
         },
         bureau=CLEAN_BUREAU,
         payroll={"employment_months": 101, "verified": True,
                  "net_monthly_income": 12800.0},
         internal={"relationship_months": 55}),

    case("2. Counteroffer - creditworthy but asked for too much",
         application={
             "application_date": "2026-07-19", "date_of_birth": "1988-06-02",
             "employment_start_date": "2017-01-01",
             "declared_monthly_income": 9500.0,
             "declared_monthly_expenses": 3400.0,
             "existing_monthly_debt_service": 1500.0,
             "requested_amount": 90000.0, "tenor_months": 24, "dependants": 3,
         },
         bureau=CLEAN_BUREAU,
         payroll={"employment_months": 114, "verified": True,
                  "net_monthly_income": 9500.0},
         internal={"relationship_months": 60}),

    case("3. Refer - thin file, no bureau record",
         application={
             "application_date": "2026-07-19", "date_of_birth": "1985-09-02",
             "declared_monthly_income": 6200.0,
             "existing_monthly_debt_service": 300.0,
             "requested_amount": 8000.0, "tenor_months": 12, "dependants": 4,
         },
         bureau=None, payroll={"employment_months": 30, "verified": True,
                               "net_monthly_income": 6200.0},
         internal={"relationship_months": 8}),

    case("4. Decline (soft) - serious recent delinquency",
         application={
             "application_date": "2026-07-19", "date_of_birth": "1992-11-20",
             "employment_start_date": "2024-01-01",
             "declared_monthly_income": 8100.0,
             "existing_monthly_debt_service": 2200.0,
             "requested_amount": 25000.0, "tenor_months": 24, "dependants": 2,
         },
         bureau={"worst_dpd_12m": 120, "open_facilities": 5, "enquiries_6m": 3,
                 "history_months": 40, "revolving_utilisation": 0.88,
                 "prior_default": False},
         payroll={"employment_months": 30, "verified": True,
                  "net_monthly_income": 8100.0}),

    case("5. Decline (hard) - sanctions match, non-overridable",
         application={
             "application_date": "2026-07-19", "date_of_birth": "1980-01-15",
             "employment_start_date": "2010-01-01",
             "declared_monthly_income": 30000.0,
             "existing_monthly_debt_service": 0.0,
             "requested_amount": 15000.0, "tenor_months": 12, "dependants": 0,
         },
         bureau=CLEAN_BUREAU,
         payroll={"employment_months": 198, "verified": True,
                  "net_monthly_income": 30000.0},
         screening={"sanctions_hit": True}),

    case("6. Insufficient information - payroll unverifiable",
         application={
             "application_date": "2026-07-19", "date_of_birth": "1995-04-04",
             "declared_monthly_income": 7400.0,
             "existing_monthly_debt_service": 400.0,
             "requested_amount": 12000.0, "tenor_months": 12, "dependants": 1,
         },
         bureau=CLEAN_BUREAU, payroll=None,
         employment={"payroll_verified": False}),
]


def show(decision, *, verbose: bool = True) -> None:
    payload = decision.as_dict()
    offer = payload["offer"]
    assessment = payload["assessment"]

    marker = {"APPROVE": "APPROVED", "DECLINE": "DECLINED", "REFER": "REFERRED",
              "INSUFFICIENT_INFORMATION": "INSUFFICIENT INFO"}[decision.outcome]
    if decision.is_counteroffer:
        marker += " (COUNTEROFFER)"
    if decision.decline_type:
        marker += f" [{decision.decline_type}]"
    print(f"  -> {marker}")

    if assessment["score"] is not None:
        print(f"     score {assessment['score']} grade {assessment['risk_grade']} "
              f"PD {assessment['probability_of_default']:.2%} "
              f"segment {payload['versions']['model_segment']} "
              f"confidence {assessment['confidence']}")

    afford = payload["affordability"]
    if afford:
        print(f"     income  declared {afford['declared_income']} -> verified "
              f"{afford['verified_income']} ({afford['verification_level']}, "
              f"haircut {afford['haircut_applied']})")
        print(f"     capacity instalment {afford['max_affordable_instalment']} "
              f"({afford['binding_constraint']} binding), requested "
              f"{afford['requested_instalment']}, DSR "
              f"{afford['dsr_before']} -> {afford['dsr_after_requested']}")

    if decision.outcome == "APPROVE":
        print(f"     offer   {offer['currency']} {offer['recommended_amount']} "
              f"over {offer['recommended_tenor_months']}m at "
              f"{Decimal(offer['annual_rate']) * 100:.0f}% -> instalment "
              f"{offer['monthly_instalment']}, total repayable "
              f"{offer['total_repayable']}, cost of credit "
              f"{offer['total_cost_of_credit']}")
        if decision.is_counteroffer:
            print(f"     (requested {offer['requested_amount']}, "
                  f"offered {offer['recommended_amount']})")
        print(f"     expires {payload['expires_at']}")

    for reason in payload["reason_codes"]:
        print(f"     reason  [{reason['code']}] {reason['text']}")
    for item in payload["additional_information_required"]:
        print(f"     needs   {item}")

    if verbose:
        trace = payload["trace"]
        print(f"     trace   {trace['rules_evaluated']} rules evaluated, "
              f"{trace['rules_matched']} matched")
        for rule in trace["rules"]:
            if rule["matched"]:
                print(f"               [{rule['rule_id']}] {rule['kind']}: "
                      f"{rule['field']} {rule['operator']} "
                      f"{rule['expected']} (actual {rule['actual']})")
        for gate in trace["gates"]:
            print(f"               gate {gate['gate']}: {gate['result']} - "
                  f"{gate['detail']}")


def main() -> None:
    tenant = config.get_tenant("ZAM-PAY")
    product = tenant.products["PAYROLL_LOAN"]

    if "--json" in sys.argv:
        _, context = SCENARIOS[1]
        result = pipeline.run(context, tenant=tenant, product=product)
        print(json.dumps(result.as_dict(audience="internal"), indent=2))
        return

    print("=" * 78)
    print(f"Decision engine demonstration - {tenant.name}")
    print(f"  product {product.code}: cut-off {product.accept_cutoff}, "
          f"refer floor {product.refer_floor}, DSR ceiling "
          f"{product.affordability.max_dsr}, policy {tenant.policy_version}")
    print("=" * 78)

    for name, context in SCENARIOS:
        print(f"\n{name}")
        show(pipeline.run(context, tenant=tenant, product=product))

    print("\n" + "=" * 78)
    print("Same applicant, both tenants: policy - not code - drives the outcome")
    _, context = SCENARIOS[2]
    for tenant_code, product_code in (("ZAM-PAY", "PAYROLL_LOAN"),
                                      ("ZAM-MFI", "MICRO_LOAN")):
        other = config.get_tenant(tenant_code)
        result = pipeline.run(context, tenant=other,
                              product=other.products[product_code])
        print(f"  {tenant_code}: {result.outcome}"
              f"{' (counteroffer)' if result.is_counteroffer else ''} - "
              f"score {result.score}, max affordable "
              f"{result.affordability['max_affordable_amount']}, "
              f"offered {result.recommended_amount}")
    print("=" * 78)


if __name__ == "__main__":
    main()
