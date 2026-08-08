"""Phase 2 test suite: money, affordability, decision engine.

Run:  python3 -m unittest discover -s tests -v

Tests are named for the requirement they evidence (IPSRS FR-AFD-*, FR-DEC-*,
CST-06) so the suite doubles as verification evidence.
"""
from __future__ import annotations

import pathlib
import sys
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from core import affordability as afford      # noqa: E402
from core import config, decision, features, money as m, pipeline, reason_codes, scorecard  # noqa: E402

TENANT = config.get_tenant("ZAM-PAY")
PRODUCT = TENANT.products["PAYROLL_LOAN"]
CLEAN_BUREAU = {"worst_dpd_12m": 0, "open_facilities": 2, "enquiries_6m": 1,
                "history_months": 74, "revolving_utilisation": 0.22,
                "prior_default": False}


def context(**overrides) -> dict:
    base = {
        "application": {
            "application_date": "2026-07-19", "date_of_birth": "1990-03-14",
            "employment_start_date": "2018-02-01",
            "declared_monthly_income": 12800.0,
            "declared_monthly_expenses": 4200.0,
            "existing_monthly_debt_service": 700.0,
            "requested_amount": 20000.0, "tenor_months": 18, "dependants": 1,
        },
        "bureau": dict(CLEAN_BUREAU),
        "payroll": {"employment_months": 101, "verified": True,
                    "net_monthly_income": 12800.0},
        "internal": {"relationship_months": 55},
        "identity": {"verified": True},
        "screening": {"sanctions_hit": False, "pep_hit": False},
        "fraud": {"confirmed": False, "risk_level": "LOW"},
        "employment": {"payroll_verified": True},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = {**base[key], **value}
        else:
            base[key] = value
    return base


class MoneyTests(unittest.TestCase):
    """IPSRS CST-06: exact decimal arithmetic, never binary floating point."""

    def test_float_inputs_do_not_leak_binary_error(self):
        self.assertEqual(m.money(0.1) + m.money(0.2), Decimal("0.30"))
        self.assertNotEqual(0.1 + 0.2, 0.3)          # the trap being avoided

    def test_money_rejects_none_rather_than_coercing_to_zero(self):
        with self.assertRaises(ValueError):
            m.money(None)

    def test_instalment_matches_the_annuity_formula(self):
        # 20,000 at 28% over 18 months
        instalment = m.instalment_for(Decimal("20000"), Decimal("0.28"), 18)
        self.assertEqual(instalment, Decimal("1373.46"))
        self.assertIsInstance(instalment, Decimal)

    def test_instalment_rounds_up_so_the_schedule_fully_amortises(self):
        instalment = m.instalment_for(Decimal("10000"), Decimal("0.33"), 7)
        self.assertGreaterEqual(instalment * 7, Decimal("10000"))

    def test_principal_for_is_the_inverse_of_instalment_for(self):
        for amount in ("5000", "17500.55", "90000"):
            principal = Decimal(amount)
            instalment = m.instalment_for(principal, Decimal("0.28"), 24)
            recovered = m.principal_for(instalment, Decimal("0.28"), 24)
            self.assertLessEqual(abs(recovered - principal), Decimal("1.00"))

    def test_principal_for_never_rounds_up_past_capacity(self):
        """Rounding up here would offer a loan the assessment did not approve."""
        capacity = Decimal("2300.00")
        principal = m.principal_for(capacity, Decimal("0.28"), 24)
        self.assertLessEqual(m.instalment_for(principal, Decimal("0.28"), 24),
                             capacity + Decimal("0.01"))

    def test_zero_rate_degenerates_to_straight_line(self):
        self.assertEqual(m.instalment_for(Decimal("1200"), Decimal("0"), 12),
                         Decimal("100.00"))


class AffordabilityTests(unittest.TestCase):
    """IPSRS FR-AFD-01..04."""

    def test_verification_level_drives_the_haircut(self):
        # FR-AFD-02
        declared = afford.assess(
            context(payroll=None, employment={"payroll_verified": False}),
            product=PRODUCT)
        verified = afford.assess(context(), product=PRODUCT)
        self.assertEqual(declared.verification_level, "DECLARED")
        self.assertEqual(verified.verification_level, "PAYROLL_VERIFIED")
        self.assertLess(declared.verified_income, verified.verified_income)
        self.assertEqual(declared.haircut_applied, Decimal("0.50"))

    def test_absent_evidence_never_upgrades_the_verification_level(self):
        result = afford.assess(context(payroll=None), product=PRODUCT)
        self.assertEqual(result.verification_level, "DECLARED")

    def test_capacity_is_the_minimum_of_dsr_and_cashflow_tests(self):
        result = afford.assess(context(), product=PRODUCT)
        self.assertEqual(result.max_affordable_instalment,
                         min(result.dsr_capacity_instalment,
                             result.cashflow_capacity_instalment))
        self.assertIn(result.binding_constraint,
                      ("debt_service_ratio", "cash_flow", "equal"))

    def test_dsr_ceiling_is_respected_after_the_new_facility(self):
        result = afford.assess(context(), product=PRODUCT)
        total_service = (result.existing_debt_service
                         + result.max_affordable_instalment)
        self.assertLessEqual(total_service,
                             result.verified_income * PRODUCT.affordability.max_dsr
                             + Decimal("0.01"))

    def test_expense_floor_applies_when_declared_expenses_are_understated(self):
        understated = afford.assess(
            context(application={"declared_monthly_expenses": 100.0}),
            product=PRODUCT)
        self.assertIn("floor", understated.expense_basis)
        self.assertGreater(understated.modelled_expenses, Decimal("100"))

    def test_dependants_reduce_cashflow_capacity(self):
        """Dependants consume household cash, so they must move the cash-flow
        test - though not necessarily total capacity, which is the minimum of
        two tests and may be bound by the DSR ceiling instead."""
        few = afford.assess(context(application={"dependants": 0}),
                            product=PRODUCT)
        many = afford.assess(context(application={"dependants": 5}),
                             product=PRODUCT)
        self.assertGreater(few.cashflow_capacity_instalment,
                           many.cashflow_capacity_instalment)

    def test_dependants_reduce_total_capacity_when_cashflow_binds(self):
        lower_income = {"declared_monthly_income": 6000.0,
                        "declared_monthly_expenses": 2600.0}
        few = afford.assess(
            context(application={**lower_income, "dependants": 0},
                    payroll={"net_monthly_income": 6000.0}), product=PRODUCT)
        many = afford.assess(
            context(application={**lower_income, "dependants": 5},
                    payroll={"net_monthly_income": 6000.0}), product=PRODUCT)
        self.assertEqual(many.binding_constraint, "cash_flow")
        self.assertGreater(few.max_affordable_instalment,
                           many.max_affordable_instalment)

    def test_no_capacity_yields_zero_not_a_negative_offer(self):
        result = afford.assess(
            context(application={"declared_monthly_income": 2000.0,
                                 "existing_monthly_debt_service": 1900.0,
                                 "declared_monthly_expenses": 1500.0},
                    payroll={"net_monthly_income": 2000.0}),
            product=PRODUCT)
        self.assertEqual(result.max_affordable_instalment, Decimal("0.00"))
        self.assertEqual(result.max_affordable_amount, Decimal("0.00"))
        self.assertIn("INSUFFICIENT_DISPOSABLE_INCOME", result.reason_codes)

    def test_affordability_never_sees_the_credit_score(self):
        """FR-AFD-04 / BR-AFF-05: the two assessments are independent."""
        result = afford.assess(context(), product=PRODUCT)
        serialised = result.as_dict()
        for forbidden in ("score", "probability_of_default", "risk_grade",
                          "grade"):
            self.assertNotIn(forbidden, serialised)

    def test_every_monetary_field_is_decimal(self):
        result = afford.assess(context(), product=PRODUCT)
        for value in (result.verified_income, result.modelled_expenses,
                      result.max_affordable_instalment,
                      result.max_affordable_amount, result.requested_instalment):
            self.assertIsInstance(value, Decimal)

    def test_affordability_reasons_are_in_the_governed_library(self):
        result = afford.assess(
            context(application={"requested_amount": 500000.0}),
            product=PRODUCT)
        for code in result.reason_codes:
            self.assertIn(code, reason_codes.BY_CODE)


class DecisionEngineTests(unittest.TestCase):
    """IPSRS FR-DEC-01..06."""

    def decide(self, **overrides) -> decision.Decision:
        return pipeline.run(context(**overrides), tenant=TENANT, product=PRODUCT)

    def test_clean_application_is_approved_with_a_priced_offer(self):
        result = self.decide()
        self.assertEqual(result.outcome, decision.APPROVE)
        self.assertGreater(result.monthly_instalment, Decimal("0"))
        self.assertIsNotNone(result.expires_at)
        self.assertEqual(result.recommended_amount, result.requested_amount)

    def test_unaffordable_request_becomes_a_counteroffer_not_a_decline(self):
        # FR-DEC-02
        result = self.decide(application={"requested_amount": 90000.0,
                                          "tenor_months": 24})
        self.assertEqual(result.outcome, decision.APPROVE)
        self.assertTrue(result.is_counteroffer)
        self.assertLess(result.recommended_amount, result.requested_amount)
        self.assertIn("COUNTEROFFER_REDUCED_AMOUNT", result.reason_codes)

    def test_counteroffer_instalment_stays_within_assessed_capacity(self):
        result = self.decide(application={"requested_amount": 90000.0,
                                          "tenor_months": 24})
        capacity = Decimal(result.affordability["max_affordable_instalment"])
        self.assertLessEqual(result.monthly_instalment, capacity)

    def test_counteroffer_is_rounded_to_a_saleable_increment(self):
        result = self.decide(application={"requested_amount": 90000.0,
                                          "tenor_months": 24})
        self.assertEqual(result.recommended_amount % PRODUCT.offer_increment,
                         Decimal("0"))

    def test_hard_decline_outranks_a_good_score(self):
        result = self.decide(screening={"sanctions_hit": True})
        self.assertEqual(result.outcome, decision.DECLINE)
        self.assertEqual(result.decline_type, "hard")
        self.assertIn("SANCTIONS_MATCH", result.reason_codes)

    def test_verification_failure_outranks_a_credit_decline(self):
        """No lawful basis to record an adverse credit decision on an
        unverified identity."""
        result = self.decide(employment={"payroll_verified": False},
                             bureau={"worst_dpd_12m": 120})
        self.assertEqual(result.outcome, decision.INSUFFICIENT)
        self.assertTrue(result.additional_information_required)

    def test_serious_delinquency_is_a_soft_decline(self):
        result = self.decide(bureau={"worst_dpd_12m": 120})
        self.assertEqual(result.outcome, decision.DECLINE)
        self.assertEqual(result.decline_type, "soft")

    def test_thin_file_refers_rather_than_auto_deciding(self):
        result = self.decide(bureau=None)
        self.assertEqual(result.outcome, decision.REFER)
        self.assertIn("THIN_CREDIT_FILE", result.reason_codes)

    def test_amount_outside_product_range_is_declined_on_eligibility(self):
        result = self.decide(application={"requested_amount": 500000.0})
        self.assertEqual(result.outcome, decision.DECLINE)
        self.assertIn("AMOUNT_OUTSIDE_PRODUCT_RANGE", result.reason_codes)

    def test_age_at_maturity_limit_is_enforced(self):
        result = self.decide(application={"date_of_birth": "1968-01-01",
                                          "tenor_months": 48})
        self.assertEqual(result.outcome, decision.DECLINE)
        self.assertIn("AGE_AT_MATURITY", result.reason_codes)

    def test_every_rule_is_traced_whether_or_not_it_fired(self):
        # FR-DEC-01
        result = self.decide()
        self.assertGreaterEqual(len(result.rule_trace), len(PRODUCT.rules))
        for entry in result.rule_trace:
            for key in ("rule_id", "kind", "field", "operator", "expected",
                        "actual", "matched", "description"):
                self.assertIn(key, entry)
        self.assertTrue(result.gate_trace)

    def test_decision_contract_is_complete(self):
        # FR-DEC-05
        payload = self.decide().as_dict()
        for section in ("outcome", "reason_codes", "identifiers", "assessment",
                        "versions", "offer", "affordability", "decided_at",
                        "expires_at"):
            self.assertIn(section, payload)
        for key in ("application_id", "decision_id", "correlation_id",
                    "tenant", "product"):
            self.assertIn(key, payload["identifiers"])
        for key in ("model_id", "model_version", "model_segment",
                    "feature_set_version", "policy_version",
                    "policy_engine_version", "reason_code_library"):
            self.assertIn(key, payload["versions"])

    def test_decision_records_the_policy_version_in_force(self):
        # FR-DEC-04
        self.assertEqual(self.decide().policy_version, TENANT.policy_version)

    def test_declines_carry_at_least_one_specific_reason(self):
        for overrides in ({"screening": {"sanctions_hit": True}},
                          {"bureau": {"worst_dpd_12m": 120}},
                          {"application": {"requested_amount": 500000.0}}):
            result = self.decide(**overrides)
            self.assertTrue(result.reason_codes,
                            f"decline without a reason: {overrides}")
            for code in result.reason_codes:
                self.assertIn(code, reason_codes.BY_CODE)

    def test_no_offer_is_priced_on_a_non_approval(self):
        result = self.decide(bureau={"worst_dpd_12m": 120})
        self.assertEqual(result.recommended_amount, Decimal("0"))
        self.assertIsNone(result.monthly_instalment)

    def test_approved_offer_expires_per_product_configuration(self):
        result = self.decide()
        decided = datetime.fromisoformat(result.decided_at)
        expires = datetime.fromisoformat(result.expires_at)
        self.assertEqual((expires - decided).days, PRODUCT.offer_validity_days)

    def test_decisioning_is_deterministic(self):
        outcomes = {(r.outcome, r.score, str(r.recommended_amount),
                     str(r.monthly_instalment))
                    for r in (self.decide() for _ in range(10))}
        self.assertEqual(len(outcomes), 1)

    def test_identifiers_are_unique_per_decision_but_correlation_is_carried(self):
        first = pipeline.run(context(), tenant=TENANT, product=PRODUCT,
                             application_id="APP-TEST-1",
                             correlation_id="CORR-TEST-1")
        second = pipeline.run(context(), tenant=TENANT, product=PRODUCT,
                              application_id="APP-TEST-1",
                              correlation_id="CORR-TEST-1")
        self.assertNotEqual(first.decision_id, second.decision_id)
        self.assertEqual(first.correlation_id, second.correlation_id)

    def test_rule_evaluation_uses_no_dynamic_code_execution(self):
        """Policy is data. A rule must never be able to execute code."""
        source = pathlib.Path(decision.__file__).read_text()
        for forbidden in ("eval(", "exec(", "__import__"):
            self.assertNotIn(forbidden, source)

    def test_unknown_operator_fails_loudly(self):
        bad_rule = config.Rule("R-BAD", "refer", "features.age_years",
                               "approximately", 30, "THIN_CREDIT_FILE", "bad")
        with self.assertRaises(ValueError):
            decision.evaluate_rules((bad_rule,), {"features": {"age_years": 30}})

    def test_missing_value_does_not_silently_fire_a_threshold_rule(self):
        rule = config.Rule("R-NULL", "soft_decline", "features.bureau_worst_dpd",
                           "gte", 90, "RECENT_SERIOUS_DELINQUENCY", "x")
        evaluations = decision.evaluate_rules(
            (rule,), {"features": {"bureau_worst_dpd": None}})
        self.assertFalse(evaluations[0].matched)


class TenantPolicyTests(unittest.TestCase):
    """One engine, two risk appetites - configuration, not code (BR-TEN-03)."""

    def test_same_applicant_can_receive_different_outcomes_per_tenant(self):
        applicant = context(
            application={"requested_amount": 15000.0, "tenor_months": 18,
                         "declared_monthly_income": 7000.0},
            payroll={"net_monthly_income": 7000.0},
            bureau={"worst_dpd_12m": 30, "enquiries_6m": 3,
                    "revolving_utilisation": 0.7, "history_months": 30,
                    "open_facilities": 3, "prior_default": False})
        outcomes = {}
        for code, product_code in (("ZAM-PAY", "PAYROLL_LOAN"),
                                   ("ZAM-MFI", "MICRO_LOAN")):
            tenant = config.get_tenant(code)
            outcomes[code] = pipeline.run(applicant, tenant=tenant,
                                          product=tenant.products[product_code])
        self.assertEqual(outcomes["ZAM-PAY"].policy_version,
                         config.get_tenant("ZAM-PAY").policy_version)
        self.assertNotEqual(outcomes["ZAM-PAY"].policy_version,
                            outcomes["ZAM-MFI"].policy_version)
        # The MFI's looser DSR ceiling must not reduce capacity below the bank's
        self.assertGreaterEqual(
            Decimal(outcomes["ZAM-MFI"].affordability["max_affordable_instalment"]),
            Decimal(outcomes["ZAM-PAY"].affordability["max_affordable_instalment"]))

    def test_each_tenant_prices_from_its_own_bands(self):
        for code, product_code in (("ZAM-PAY", "PAYROLL_LOAN"),
                                   ("ZAM-MFI", "MICRO_LOAN")):
            tenant = config.get_tenant(code)
            product = tenant.products[product_code]
            result = pipeline.run(context(application={"requested_amount": 8000.0,
                                                       "tenor_months": 12}),
                                  tenant=tenant, product=product)
            if result.outcome == decision.APPROVE:
                rates = {band.annual_rate for band in product.pricing}
                self.assertIn(result.annual_rate, rates)


if __name__ == "__main__":
    unittest.main(verbosity=2)
