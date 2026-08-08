"""Phase 1 test suite - stdlib unittest, no external dependencies.

Run:  python3 -m unittest discover -s tests -v

Each test names the requirement it verifies so the suite doubles as evidence
against the IPSRS.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from core import config, features, reason_codes, scorecard  # noqa: E402

GOLDEN = pathlib.Path(__file__).resolve().parent / "golden_scores.json"


def clean_context() -> dict:
    return {
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
    }


def thin_file_context() -> dict:
    context = clean_context()
    context["bureau"] = None
    context["payroll"] = None
    return context


class FeatureStoreTests(unittest.TestCase):
    """IPSRS FR-FST-01..06."""

    def test_feature_metadata_is_complete(self):
        # FR-FST-02: incomplete metadata must prevent activation
        for definition in features.DEFINITIONS:
            definition.validate()   # raises if metadata incomplete

    def test_sensitive_attributes_excluded_from_scoring(self):
        # FR-FST-06
        self.assertIn("gender", features.BY_NAME)
        self.assertTrue(features.BY_NAME["gender"].sensitive)
        self.assertNotIn("gender", features.SCORING_FEATURES)
        computed = features.compute(clean_context())
        self.assertNotIn("gender", computed.values)

    def test_missing_values_are_none_never_zero(self):
        # FR-FST-03: the cardinal rule
        computed = features.compute(thin_file_context())
        for name in ("bureau_worst_dpd", "revolving_utilisation",
                     "credit_history_months"):
            self.assertIsNone(computed.values[name],
                              f"{name} must be None when unavailable, not 0")
            self.assertIn(name, computed.missing)

    def test_thin_file_is_flagged_and_degraded(self):
        computed = features.compute(thin_file_context())
        self.assertTrue(computed.thin_file)
        self.assertEqual(computed.status, "DEGRADED")

    def test_mandatory_missing_feature_blocks_scoring(self):
        # FR-FST-04 with a 'block' treatment feature (age_years)
        context = clean_context()
        context["application"]["date_of_birth"] = None
        computed = features.compute(context)
        self.assertEqual(computed.status, "BLOCK")
        self.assertIn("age_years", computed.missing)

    def test_out_of_range_values_are_clamped_and_noted(self):
        context = clean_context()
        context["bureau"]["revolving_utilisation"] = 9.9      # max is 2
        computed = features.compute(context)
        self.assertLessEqual(computed.values["revolving_utilisation"], 2.0)
        self.assertTrue(any("revolving_utilisation" in n for n in computed.notes))

    def test_feature_set_version_is_stable_and_content_addressed(self):
        self.assertEqual(features.feature_set_version(),
                         features.feature_set_version())
        self.assertTrue(features.feature_set_version().startswith("FS_APPLICATION@"))


class ScoringTests(unittest.TestCase):
    """IPSRS FR-SCO-01..03."""

    @classmethod
    def setUpClass(cls):
        cls.card = scorecard.load()
        cls.tenant = config.get_tenant("ZAM-PAY")

    def test_score_response_contract(self):
        # FR-SCO-01: every mandated field present
        computed = features.compute(clean_context())
        payload = self.card.score(computed.values, tenant=self.tenant,
                                  dq_status=computed.status).as_dict()
        for key in ("probability_of_default", "score", "risk_grade",
                    "reason_codes", "model_id", "model_version",
                    "feature_set_version", "scored_at",
                    "data_quality_status", "confidence"):
            self.assertIn(key, payload)

    def test_scoring_is_deterministic(self):
        # FR-SCO-03: identical inputs -> identical outputs, always
        computed = features.compute(clean_context())
        results = [self.card.score(computed.values, tenant=self.tenant)
                   for _ in range(25)]
        pds = {round(r.probability_of_default, 12) for r in results}
        scores = {r.score for r in results}
        self.assertEqual(len(pds), 1)
        self.assertEqual(len(scores), 1)

    def test_golden_file_regression(self):
        """Scores for fixed inputs must not drift across releases."""
        cases = {
            "clean": features.compute(clean_context()).values,
            "thin_file": features.compute(thin_file_context()).values,
        }
        actual = {
            name: {
                tenant.code: self.card.score(values, tenant=tenant).score
                for tenant in config.all_tenants()
            }
            for name, values in cases.items()
        }
        if not GOLDEN.exists():
            GOLDEN.write_text(json.dumps(
                {"artefact_sha256": self.card.sha256, "scores": actual},
                indent=2, sort_keys=True))
            self.skipTest("golden file created on first run")
        golden = json.loads(GOLDEN.read_text())
        if golden["artefact_sha256"] != self.card.sha256:
            self.skipTest(
                "scorecard artefact changed: regenerate the golden file "
                "deliberately and record the model-version change")
        self.assertEqual(golden["scores"], actual)

    def test_pd_is_scale_invariant_but_score_is_not(self):
        # BR-SCR-02: one PD model, many tenant presentation scales
        computed = features.compute(clean_context())
        pay = self.card.score(computed.values, tenant=config.get_tenant("ZAM-PAY"))
        mfi = self.card.score(computed.values, tenant=config.get_tenant("ZAM-MFI"))
        self.assertAlmostEqual(pay.probability_of_default,
                               mfi.probability_of_default, places=12)
        self.assertNotEqual(pay.score, mfi.score)

    def test_score_respects_configured_bounds(self):
        scale = self.tenant.score_scale
        for pd in (1e-9, 0.001, 0.5, 0.999, 1 - 1e-9):
            score = scorecard.pd_to_score(pd, scale)
            self.assertGreaterEqual(score, scale.min_score)
            self.assertLessEqual(score, scale.max_score)

    def test_scaling_matches_points_to_double_the_odds(self):
        """Halving the odds of default must move the score by exactly one PDO."""
        scale = self.tenant.score_scale
        pd_a = 0.10
        odds_a = (1 - pd_a) / pd_a
        pd_b = 1 / (1 + 2 * odds_a)          # good:bad odds doubled
        delta = scorecard.pd_to_score(pd_b, scale) - scorecard.pd_to_score(pd_a, scale)
        self.assertEqual(abs(delta), scale.pdo)

    def test_worse_applicant_scores_lower(self):
        good = features.compute(clean_context())
        bad_context = clean_context()
        bad_context["bureau"].update({"worst_dpd_12m": 90, "enquiries_6m": 5,
                                      "revolving_utilisation": 1.1,
                                      "prior_default": True})
        bad = features.compute(bad_context)
        self.assertLess(self.card.score(bad.values, tenant=self.tenant).score,
                        self.card.score(good.values, tenant=self.tenant).score)

    def test_all_coefficients_carry_the_expected_sign(self):
        """WoE is oriented so higher = safer, so every coefficient must be < 0."""
        for name, segment in self.card.segments.items():
            for characteristic, coefficient in segment.coefficients.items():
                self.assertLess(coefficient, 0,
                                f"{name}/{characteristic} sign violation")


class ReasonCodeTests(unittest.TestCase):
    """IPSRS FR-EXP-01/02 and BRD BR-EXP-02."""

    @classmethod
    def setUpClass(cls):
        cls.card = scorecard.load()
        cls.tenant = config.get_tenant("ZAM-PAY")

    def test_every_emitted_code_exists_in_the_library(self):
        for context in (clean_context(), thin_file_context()):
            computed = features.compute(context)
            result = self.card.score(computed.values, tenant=self.tenant,
                                     dq_status=computed.status)
            for code in result.reason_codes:
                self.assertIn(code, reason_codes.BY_CODE,
                              f"{code} is not in the governed library")

    def test_thin_file_gets_no_bureau_based_reasons(self):
        """A customer with no credit record must never be told arrears were found."""
        computed = features.compute(thin_file_context())
        result = self.card.score(computed.values, tenant=self.tenant,
                                 dq_status=computed.status)
        forbidden = {"RECENT_DELINQUENCY", "HIGH_UTILISATION",
                     "SHORT_CREDIT_HISTORY", "MANY_RECENT_ENQUIRIES",
                     "PRIOR_DEFAULT_RECORD"}
        self.assertEqual(forbidden & set(result.reason_codes), set())
        self.assertIn("NO_BUREAU_RECORD", result.reason_codes)

    def test_weak_applicant_receives_specific_reasons(self):
        context = clean_context()
        context["bureau"].update({"worst_dpd_12m": 60, "enquiries_6m": 4,
                                  "revolving_utilisation": 0.95})
        computed = features.compute(context)
        result = self.card.score(computed.values, tenant=self.tenant,
                                 dq_status=computed.status)
        self.assertTrue(result.reason_codes)
        self.assertIn("RECENT_DELINQUENCY", result.reason_codes)

    def test_customer_text_is_never_a_bare_code(self):
        for code, reason in reason_codes.BY_CODE.items():
            self.assertGreater(len(reason.customer_text.split()), 4,
                               f"{code} customer text is not a sentence")
            self.assertNotIn("internal policy", reason.customer_text.lower())


class TenantIsolationTests(unittest.TestCase):
    """BRD BR-TEN-01 (configuration-level checks; storage isolation in Phase 3)."""

    def test_api_key_resolves_to_exactly_one_tenant(self):
        for tenant in config.all_tenants():
            resolved = config.tenant_for_api_key(tenant.api_key)
            self.assertIsNotNone(resolved)
            self.assertEqual(resolved.code, tenant.code)

    def test_unknown_api_key_resolves_to_nothing(self):
        self.assertIsNone(config.tenant_for_api_key("not-a-key"))

    def test_tenants_do_not_share_policy_or_scale_objects(self):
        pay, mfi = config.get_tenant("ZAM-PAY"), config.get_tenant("ZAM-MFI")
        self.assertNotEqual(pay.policy_version, mfi.policy_version)
        self.assertIsNot(pay.score_scale, mfi.score_scale)
        self.assertEqual(set(pay.products) & set(mfi.products), set())

    def test_grade_bands_cover_the_whole_scale_without_gaps(self):
        for tenant in config.all_tenants():
            bands = sorted(tenant.grades, key=lambda g: g.min_score)
            self.assertEqual(bands[0].min_score, tenant.score_scale.min_score)
            self.assertEqual(bands[-1].max_score, tenant.score_scale.max_score)
            for lower, upper in zip(bands, bands[1:]):
                self.assertEqual(upper.min_score, lower.max_score + 1)


class SegmentationTests(unittest.TestCase):
    """Segmented scorecard: the structural fix for the thin-file problem."""

    @classmethod
    def setUpClass(cls):
        cls.card = scorecard.load()
        cls.tenant = config.get_tenant("ZAM-PAY")

    def test_segment_selected_from_available_information(self):
        clean = features.compute(clean_context())
        thin = features.compute(thin_file_context())
        self.assertEqual(self.card.select_segment(clean.values), "BUREAU")
        self.assertEqual(self.card.select_segment(thin.values), "THIN")

    def test_thin_segment_contains_no_bureau_characteristic(self):
        """The guarantee is structural: the model cannot see what does not exist."""
        forbidden = {"bureau_worst_dpd", "bureau_enquiries_6m",
                     "bureau_open_facilities", "credit_history_months",
                     "revolving_utilisation", "prior_default"}
        thin_characteristics = set(self.card.segments["THIN"].characteristics)
        self.assertEqual(forbidden & thin_characteristics, set())

    def test_no_characteristic_is_collinear(self):
        """VIF above 5 would signal the defect segmentation was built to remove."""
        for name, segment in self.card.segments.items():
            for characteristic, vif in segment.vif.items():
                self.assertLess(vif, 5.0,
                                f"{name}/{characteristic} VIF {vif} too high")

    def test_thin_file_score_never_reports_high_confidence(self):
        thin = features.compute(thin_file_context())
        result = self.card.score(thin.values, tenant=self.tenant,
                                 dq_status=thin.status)
        self.assertEqual(result.segment, "THIN")
        self.assertNotEqual(result.confidence, "HIGH")

    def test_both_segments_are_reported_in_performance(self):
        for segment in ("BUREAU", "THIN"):
            metrics = self.card.segment_performance(segment)
            self.assertIn("gini", metrics)
            self.assertGreater(metrics["gini"], 0.0)

    def test_score_response_names_the_segment_used(self):
        thin = features.compute(thin_file_context())
        payload = self.card.score(thin.values, tenant=self.tenant,
                                  dq_status=thin.status).as_dict()
        self.assertEqual(payload["model_segment"], "THIN")


class CutoffCalibrationTests(unittest.TestCase):
    """Cut-offs must be traceable to a calibration, not chosen by eye."""

    @classmethod
    def setUpClass(cls):
        path = (pathlib.Path(__file__).resolve().parents[1]
                / "artefacts" / "cutoff_calibration.json")
        cls.calibration = json.loads(path.read_text()) if path.exists() else None

    def test_calibration_artefact_exists(self):
        if self.calibration is None:
            self.skipTest("run 'python3 -m model.calibrate_cutoffs' first")
        self.assertIn("tenants", self.calibration)

    def test_calibration_excludes_the_development_sample(self):
        if self.calibration is None:
            self.skipTest("calibration not generated")
        self.assertIn("development sample excluded",
                      self.calibration["sample"]["basis"])

    def test_configured_cutoffs_match_the_calibration(self):
        if self.calibration is None:
            self.skipTest("calibration not generated")
        for tenant in config.all_tenants():
            product = next(iter(tenant.products.values()))
            recommended = self.calibration["tenants"][tenant.code]["recommended"]
            self.assertEqual(product.accept_cutoff, recommended["accept_cutoff"],
                             f"{tenant.code} cut-off has drifted from calibration")
            self.assertEqual(product.refer_floor, recommended["refer_floor"])

    def test_marginal_risk_is_within_appetite_at_the_cutoff(self):
        """Guards the degenerate 'approve everyone' recommendation."""
        if self.calibration is None:
            self.skipTest("calibration not generated")
        for code, block in self.calibration["tenants"].items():
            recommended, target = block["recommended"], block["target_bad_rate"]
            self.assertLessEqual(recommended["bad_rate_approved"], target, code)
            self.assertLessEqual(recommended["bad_rate_marginal"], target, code)
            self.assertLess(recommended["approval_rate"], 1.0,
                            f"{code}: approving the entire population is not a "
                            f"cut-off")


class ModelGovernanceTests(unittest.TestCase):
    """BRD BR-GOV-01/03: the artefact must carry its own provenance."""

    @classmethod
    def setUpClass(cls):
        cls.card = scorecard.load()

    def test_artefact_declares_prototype_status_and_limitations(self):
        governance = self.card.governance
        self.assertIn("PROTOTYPE", governance.get("status", ""))
        self.assertTrue(governance.get("limitations"))

    def test_artefact_declares_its_segmentation_rule(self):
        self.assertIn("rule", self.card.segmentation)
        self.assertIn("rationale", self.card.segmentation)

    def test_artefact_records_training_provenance(self):
        training = self.card._a["training_data"]
        self.assertIn("SYNTHETIC", training["source"])
        for key in ("seed", "n_development", "n_validation", "n_out_of_time",
                    "default_definition"):
            self.assertIn(key, training)

    def test_out_of_time_performance_is_reported_and_plausible(self):
        oot = self.card.performance["out_of_time"]
        self.assertGreater(oot["gini"], 0.0)
        self.assertLess(oot["gini"], 0.95, "implausibly high: check for leakage")
        self.assertGreater(oot["ks"], 0.0)

    def test_feature_set_version_matches_the_live_feature_store(self):
        """Guards against scoring with an artefact built on stale definitions."""
        self.assertEqual(self.card.feature_set_version,
                         features.feature_set_version())


if __name__ == "__main__":
    unittest.main(verbosity=2)
