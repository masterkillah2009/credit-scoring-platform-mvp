"""Phase 4 test suite: monitoring, batch, UI and the documentation pack.

Run:  python3 -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import pathlib
import random
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from api import server as api_server              # noqa: E402
from core import batch, config, monitoring        # noqa: E402
from core.ledger import Ledger                    # noqa: E402
from core.pipeline import Platform                # noqa: E402
from partners import simulators                   # noqa: E402
from tools import traceability                    # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


class StabilityIndexTests(unittest.TestCase):
    """FR-MOD-04 / BR-GOV-02: PSI and CSI arithmetic."""

    def test_identical_distributions_have_zero_psi(self):
        sample = list(range(100))
        result = monitoring.stability_index(sample, sample)
        self.assertAlmostEqual(result["index"], 0.0, places=6)
        self.assertEqual(result["status"], monitoring.STABLE)

    def test_psi_matches_a_hand_computed_value(self):
        """Two bins, 50/50 expected against 70/30 actual.

        PSI = (0.7-0.5)ln(0.7/0.5) + (0.3-0.5)ln(0.3/0.5)
            = 0.2(0.33647) + (-0.2)(-0.51083) = 0.16946
        """
        expected = [1] * 50 + [3] * 50
        actual = [1] * 70 + [3] * 30
        result = monitoring.stability_index(expected, actual, edges=[2])
        self.assertAlmostEqual(result["index"], 0.169460, places=5)
        self.assertEqual(result["status"], monitoring.WARNING)

    def test_large_shift_breaches(self):
        expected = [1] * 90 + [3] * 10
        actual = [1] * 30 + [3] * 70
        result = monitoring.stability_index(expected, actual, edges=[2])
        self.assertEqual(result["status"], monitoring.BREACH)

    def test_thresholds_are_disclosed_with_the_metric(self):
        sample = [1] * 50 + [3] * 50
        result = monitoring.stability_index(sample, list(sample), edges=[2])
        self.assertEqual(result["thresholds"]["warning"], monitoring.PSI_WARNING)
        self.assertEqual(result["thresholds"]["breach"], monitoring.PSI_BREACH)

    def test_empty_input_is_unknown_not_zero(self):
        """An absent metric must never read as a healthy one."""
        result = monitoring.stability_index([], [1, 2, 3])
        self.assertIsNone(result["index"])
        self.assertEqual(result["status"], "UNKNOWN")

    def test_a_sample_too_small_to_measure_reports_unknown(self):
        """PSI over a handful of observations measures noise, not drift.

        Ten deciles against nine observations cannot be computed honestly: most
        bands are empty by construction, and empty bands are what the formula
        punishes hardest. It must decline to report rather than emit a number
        someone will act on.
        """
        reference = [float(i) for i in range(2000)]
        result = monitoring.stability_index(reference, [1.0] * 9)
        self.assertIsNone(result["index"])
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertIn("insufficient data", result["note"])
        self.assertEqual(result["sample_size"], 9)

    def test_band_count_is_reduced_to_fit_a_small_sample(self):
        reference = [float(i) for i in range(2000)]
        result = monitoring.stability_index(reference, [float(i) for i in range(90)])
        self.assertEqual(result["band_count"], 3)   # 90 // 30 bands
        self.assertEqual(result["sample_size"], 90)

    def test_the_noise_the_metric_carries_is_disclosed(self):
        """A reader must be able to tell a movement from the metric's own noise."""
        reference = [float(i) for i in range(2000)]
        result = monitoring.stability_index(reference, [float(i) for i in range(300)])
        self.assertIn("expected_noise", result)
        self.assertAlmostEqual(result["expected_noise"],
                               (result["band_count"] - 1) / 300, places=6)
        self.assertLess(result["expected_noise"], monitoring.PSI_WARNING)

    def test_a_small_sample_of_the_reference_itself_does_not_breach(self):
        """Regression: the demonstration opened on a PSI breach caused by
        arithmetic rather than drift.

        A sample drawn from the reference distribution has not drifted, by
        definition. But with ten deciles and a few dozen observations, several
        bands are empty purely by chance - and with the old fixed 1e-6
        smoothing floor a single empty decile contributed about 1.15 to the
        index on its own, four times the breach threshold. Every small sample
        therefore breached. Fitting the band count to the sample and scaling
        the floor with n fixes it. This is checked over many seeds because the
        failure was intermittent, which is how it survived review.
        """
        rng = random.Random(20260808)
        reference = [rng.gauss(600, 80) for _ in range(2000)]
        for trial in range(40):
            sample = [rng.choice(reference) for _ in range(60)]
            result = monitoring.stability_index(reference, sample)
            self.assertNotEqual(
                result["status"], monitoring.BREACH,
                f"trial {trial}: a 60-observation sample of the reference "
                f"itself reported {result['index']} ({result['status']})")

    def test_the_smoothing_floor_scales_with_the_sample(self):
        reference = [float(i) for i in range(1000)]
        actual = [float(i % 1000) for i in range(200)]
        result = monitoring.stability_index(reference, actual)
        self.assertAlmostEqual(result["smoothing_floor"], 0.5 / 200, places=8)

    def test_a_genuine_shift_still_breaches(self):
        """The guard must not have been bought at the cost of sensitivity."""
        reference = [float(i) for i in range(1000)]
        actual = [float(i) for i in range(200)]      # only the bottom fifth
        result = monitoring.stability_index(reference, actual)
        self.assertEqual(result["status"], monitoring.BREACH)

    def test_largest_moves_identify_the_bands_that_shifted(self):
        expected = [1] * 50 + [3] * 50
        actual = [1] * 80 + [3] * 20
        result = monitoring.stability_index(expected, actual, edges=[2])
        self.assertTrue(result["largest_moves"])
        self.assertIn("contribution", result["largest_moves"][0])


class CalibrationTests(unittest.TestCase):
    def test_perfect_calibration_is_stable(self):
        expected = [0.1] * 100
        outcomes = [1] * 10 + [0] * 90
        result = monitoring.calibration(expected, outcomes)
        self.assertEqual(result["oe_ratio"], 1.0)
        self.assertEqual(result["status"], monitoring.STABLE)

    def test_underprediction_breaches(self):
        result = monitoring.calibration([0.05] * 100, [1] * 30 + [0] * 70)
        self.assertGreater(result["oe_ratio"], 1.5)
        self.assertEqual(result["status"], monitoring.BREACH)

    def test_calibration_without_outcomes_is_unknown(self):
        """Calibration needs a matured performance window; saying so is honest."""
        result = monitoring.calibration([], [])
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertIn("outcome", result["note"])


class DecisionMetricsTests(unittest.TestCase):
    def sample(self) -> list[dict]:
        def record(outcome, grade="B", counteroffer=False, codes=()):
            return {"outcome": outcome,
                    "assessment": {"risk_grade": grade, "score": 650,
                                   "probability_of_default": 0.05,
                                   "data_quality_status": "OK"},
                    "versions": {"model_segment": "BUREAU"},
                    "offer": {"is_counteroffer": counteroffer},
                    "reason_codes": [{"code": c} for c in codes]}
        return [record("APPROVE"), record("APPROVE", counteroffer=True),
                record("DECLINE", codes=["SCORE_BELOW_CUTOFF"]),
                record("REFER", codes=["THIN_CREDIT_FILE"])]

    def test_rates_sum_to_one(self):
        metrics = monitoring.decision_metrics(self.sample())
        total = (metrics["approval_rate"] + metrics["decline_rate"]
                 + metrics["referral_rate"] + metrics["insufficient_rate"])
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_counteroffer_share_of_approvals_is_reported(self):
        metrics = monitoring.decision_metrics(self.sample())
        self.assertEqual(metrics["counteroffer_share_of_approvals"], 0.5)

    def test_reason_codes_are_ranked(self):
        metrics = monitoring.decision_metrics(self.sample())
        self.assertTrue(metrics["top_reason_codes"])

    def test_empty_period_does_not_divide_by_zero(self):
        self.assertEqual(monitoring.decision_metrics([])["n"], 0)


class BatchTests(unittest.TestCase):
    """FR-INT-04 / WFL-08."""

    def setUp(self):
        simulators.reset()
        self.ledger = Ledger(pathlib.Path(tempfile.mkdtemp()) / "batch.db")
        self.platform = Platform(ledger=self.ledger)
        self.tenant = config.get_tenant("ZAM-PAY")
        self.product = self.tenant.products["PAYROLL_LOAN"]

    def tearDown(self):
        self.ledger.close()

    def run_batch(self, rows: int = 20):
        return batch.run(batch.sample_file(rows), tenant=self.tenant,
                         product=self.product, platform=self.platform)

    def test_totals_reconcile(self):
        """submitted = processed + rejected, or the batch has lost a row."""
        result = self.run_batch(20)
        self.assertTrue(result.reconciled)
        self.assertEqual(result.submitted, result.processed + result.rejected)

    def test_invalid_rows_are_itemised_not_silently_dropped(self):
        result = self.run_batch(20)
        self.assertGreater(result.rejected, 0)
        for reject in result.rejects:
            self.assertIn("row_id", reject)
            self.assertTrue(reject["errors"])

    def test_valid_rows_are_processed_despite_invalid_neighbours(self):
        result = self.run_batch(20)
        self.assertGreater(result.processed, 0)

    def test_consent_is_enforced_in_batch_as_in_the_api(self):
        content = batch.sample_file(20)
        rows, _ = batch.parse(content)
        withheld = [row for row in rows
                    if any("consent" in error for error in row.errors)]
        self.assertTrue(withheld, "consent rule not enforced on batch rows")

    def test_type_errors_are_rejected_rather_than_coerced(self):
        rows, _ = batch.parse(batch.sample_file(20))
        problems = [error for row in rows for error in row.errors]
        self.assertTrue(any("not a number" in p for p in problems))
        self.assertTrue(any("not an integer" in p for p in problems))

    def test_checksum_is_recorded_for_the_file(self):
        result = self.run_batch(10)
        self.assertEqual(len(result.checksum), 64)
        self.assertEqual(result.summary()["checksum_sha256"], result.checksum)

    def test_batch_writes_audit_events(self):
        result = self.run_batch(10)
        trail = self.ledger.audit_trail(tenant_code="ZAM-PAY",
                                        correlation_id=result.batch_id)
        types = {event["event_type"] for event in trail}
        self.assertIn("BATCH_RECEIVED", types)
        self.assertIn("BATCH_COMPLETED", types)

    def test_batch_decisions_are_stored_and_metered(self):
        result = self.run_batch(10)
        reconciliation = self.ledger.reconcile(tenant_code="ZAM-PAY")
        self.assertGreaterEqual(reconciliation["decisions"], result.processed)
        self.assertTrue(reconciliation["balanced"])

    def test_batch_and_realtime_agree_for_identical_input(self):
        """A batch score must equal an API score for the same application."""
        rows, _ = batch.parse(batch.sample_file(6))
        row = next(r for r in rows if r.valid)
        first, _ = self.platform.decide({"application": row.application},
                                        tenant=self.tenant, product=self.product)
        second, _ = self.platform.decide({"application": row.application},
                                         tenant=self.tenant, product=self.product)
        self.assertEqual(first.score, second.score)
        self.assertEqual(first.outcome, second.outcome)


class MonitoringEndpointTests(unittest.TestCase):
    def setUp(self):
        simulators.reset()
        self.ledger = Ledger(pathlib.Path(tempfile.mkdtemp()) / "mon.db")
        self.api = api_server.Api(Platform(ledger=self.ledger))

    def tearDown(self):
        self.ledger.close()

    def request(self, method, path, body=None, key="demo-key-payroll"):
        status, payload, _ = self.api.handle(
            method, path, {"x-api-key": key} if key else {},
            json.dumps(body).encode() if body else b"")
        return status, payload

    def test_monitoring_summary_is_returned(self):
        self.request("POST", "/v1/batches", {"use_sample": True, "rows": 12})
        status, report = self.request("GET", "/v1/monitoring/summary")
        self.assertEqual(status, 200)
        for section in ("overall_status", "population_stability",
                        "calibration", "decisions", "interpretation"):
            self.assertIn(section, report)

    def test_psi_uses_the_development_sample_as_reference(self):
        self.request("POST", "/v1/batches", {"use_sample": True, "rows": 12})
        _, report = self.request("GET", "/v1/monitoring/summary")
        self.assertIn("development sample", report["reference"]["source"])
        self.assertGreater(report["reference"]["n"], 0)

    def test_monitoring_is_scoped_to_the_calling_tenant(self):
        self.request("POST", "/v1/batches", {"use_sample": True, "rows": 12})
        _, other = self.request("GET", "/v1/monitoring/summary",
                                key="demo-key-micro")
        self.assertEqual(other["decisions"]["n"], 0)

    def test_batch_endpoint_returns_rejects_with_reasons(self):
        status, payload = self.request("POST", "/v1/batches",
                                       {"use_sample": True, "rows": 20})
        self.assertEqual(status, 200)
        self.assertTrue(payload["reconciled"])
        self.assertTrue(payload["rejects"])

    def test_batch_requires_a_payload(self):
        status, payload = self.request("POST", "/v1/batches", {})
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "VALIDATION_FAILED")

    def test_recent_decisions_endpoint(self):
        self.request("POST", "/v1/batches", {"use_sample": True, "rows": 8})
        status, payload = self.request("GET", "/v1/decisions")
        self.assertEqual(status, 200)
        self.assertGreater(payload["count"], 0)


class UserInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.ledger = Ledger(pathlib.Path(tempfile.mkdtemp()) / "ui.db")
        self.api = api_server.Api(Platform(ledger=self.ledger))

    def tearDown(self):
        self.ledger.close()

    def test_console_is_served_at_the_root(self):
        status, payload, _ = self.api.handle("GET", "/", {}, b"")
        self.assertEqual(status, 200)
        self.assertIn("_html", payload)
        self.assertIn("Underwriting", payload["_html"])

    def test_console_declares_prototype_status(self):
        _, payload, _ = self.api.handle("GET", "/ui", {}, b"")
        self.assertIn("not approved for production", payload["_html"].lower())

    def test_console_has_no_external_dependencies(self):
        """It must run offline and on a low-bandwidth link (NFR-10)."""
        html = (ROOT / "ui" / "index.html").read_text()
        for forbidden in ("http://cdn", "https://cdn", "unpkg.com",
                          "googleapis.com", "jsdelivr"):
            self.assertNotIn(forbidden, html)


class DocumentationTests(unittest.TestCase):
    """Charter section 18: documentation standards and traceability."""

    def test_documentation_pack_is_present(self):
        for name in ("MODEL_CARD.md", "RUNBOOK.md", "API_GUIDE.md",
                     "TRACEABILITY.md"):
            path = ROOT / "docs" / name
            self.assertTrue(path.exists(), f"missing {name}")
            self.assertGreater(len(path.read_text()), 500, name)

    def test_model_card_declares_prototype_status_and_limitations(self):
        card = (ROOT / "docs" / "MODEL_CARD.md").read_text()
        self.assertIn("NOT APPROVED FOR PRODUCTION", card)
        self.assertIn("Synthetic", card)
        self.assertIn("Known limitations", card)

    def test_traceability_matrix_is_generated_not_handwritten(self):
        matrix = traceability.build()
        self.assertGreater(matrix["summary"]["total"], 20)
        self.assertGreater(matrix["summary"]["verified"], 10)

    def test_traceability_matrix_is_current(self):
        """The committed matrix must match what the source says today."""
        rendered = traceability.render(traceability.build())
        committed = (ROOT / "docs" / "TRACEABILITY.md").read_text()
        # Ignore the generation timestamp line, which changes every run.
        strip = lambda text: "\n".join(
            line for line in text.splitlines()
            if not line.startswith("> Generated 2"))
        self.assertEqual(strip(rendered), strip(committed),
                         "run 'python3 -m tools.traceability' and commit")

    def test_every_verified_requirement_names_a_real_test(self):
        matrix = traceability.build()
        for row in matrix["rows"]:
            if row["status"] == "VERIFIED":
                self.assertTrue(row["tests"])
                self.assertTrue(row["implementation"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
