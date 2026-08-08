"""Model and portfolio monitoring.

Implements the subset of IPSRS FR-MOD-04 / BRD BR-GOV-02 that can be computed
without production outcome data:

  * Population Stability Index (PSI) on the score distribution
  * Characteristic Stability Index (CSI) per scorecard characteristic
  * calibration: observed versus expected default rates, where outcomes exist
  * approval, decline, referral and counteroffer rates, overall and by grade
  * data-quality and thin-file mix
  * reason-code frequency, which is how an operations team notices a policy or
    data problem before the risk metrics move

Every metric carries a **status** against configured thresholds so a dashboard
can show red, amber or green rather than requiring the reader to know what a
PSI of 0.19 means. Thresholds follow common industry practice and are stated
explicitly rather than buried:

    PSI / CSI   < 0.10 stable · 0.10-0.25 warning · > 0.25 breach
    O/E ratio   0.8-1.25 acceptable, outside that a calibration concern

A breach is not an alarm by itself; it is an instruction to investigate, and
the report says which characteristic moved so the investigation has a starting
point.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Optional, Sequence

STABLE, WARNING, BREACH = "STABLE", "WARNING", "BREACH"

PSI_WARNING, PSI_BREACH = 0.10, 0.25
OE_LOWER, OE_UPPER = 0.80, 1.25

#: PSI needs enough observations per band to mean anything. Below roughly five
#: expected observations per band the index measures sampling noise rather than
#: drift: with ten deciles and thirty observations, empty bands are near
#: certain by chance alone, and an empty band is exactly what the formula
#: punishes hardest. So the number of bands is reduced to fit the sample, and
#: if even a two-band comparison cannot be supported the index is not reported
#: at all.
#:
#: This is the same discipline the calibration block already follows: a metric
#: that cannot be computed reads UNKNOWN, and UNKNOWN never reads as healthy.
#: Printing a number that is really a sample-size artefact is worse than
#: printing nothing, because someone will act on it.
#: Under the null hypothesis of no drift, PSI is not zero - it is roughly
#: (bands - 1) / n, because a finite sample never reproduces the reference
#: shares exactly. Ten bands over sixty observations therefore has an expected
#: index of about 0.15 from sampling alone, which is already past the 0.10
#: warning threshold before any drift exists. Thirty observations per band
#: keeps that noise term near 0.015 - an order of magnitude below the warning
#: line - so a reading above the threshold means something.
MIN_OBSERVATIONS_PER_BAND = 30
MIN_BANDS = 2

#: Empty-band smoothing. The floor is tied to the sample size (the standard
#: additive-smoothing choice of half an observation) rather than being a fixed
#: tiny constant. A fixed 1e-6 floor makes one empty decile contribute
#: (0.1 - 1e-6) * ln(0.1 / 1e-6) ~ 1.15 to the index on its own - five times
#: the breach threshold, from a single empty band. Scaling the floor with n
#: keeps an empty band's contribution proportionate to how surprising it
#: actually is.
def _floor(sample_size: int) -> float:
    return 0.5 / max(sample_size, 1)


def _status(value: Optional[float], warning: float, breach: float) -> str:
    if value is None:
        return "UNKNOWN"
    if value >= breach:
        return BREACH
    if value >= warning:
        return WARNING
    return STABLE


def _distribution(values: Sequence[float], edges: Sequence[float]) -> list[float]:
    counts = [0] * (len(edges) + 1)
    for value in values:
        placed = False
        for index, edge in enumerate(edges):
            if value <= edge:
                counts[index] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1
    total = max(len(values), 1)
    return [count / total for count in counts]


def quantile_edges(values: Sequence[float], bands: int = 10) -> list[float]:
    """Band edges from a reference distribution (deciles by default)."""
    if not values:
        return []
    ordered = sorted(values)
    edges = []
    for index in range(1, bands):
        position = int(len(ordered) * index / bands)
        edge = ordered[min(position, len(ordered) - 1)]
        if not edges or edge > edges[-1]:
            edges.append(float(edge))
    return edges


def stability_index(expected: Sequence[float], actual: Sequence[float],
                    *, edges: Optional[Sequence[float]] = None,
                    bands: int = 10) -> dict[str, Any]:
    """PSI (or CSI when applied to one characteristic).

        PSI = sum over bands of (actual% - expected%) * ln(actual% / expected%)
    """
    if not expected or not actual:
        return {"index": None, "status": "UNKNOWN", "bands": [],
                "sample_size": len(actual),
                "note": "insufficient data: no observations to compare"}

    # Fit the number of bands to the smaller sample. Ten deciles against
    # twenty-seven observations is not a drift measurement.
    supportable = min(len(actual), len(expected)) // MIN_OBSERVATIONS_PER_BAND
    if edges is None:
        bands = min(bands, supportable)
    if (edges is None and bands < MIN_BANDS) or supportable < MIN_BANDS:
        return {
            "index": None,
            "status": "UNKNOWN",
            "bands": [],
            "sample_size": len(actual),
            "minimum_sample": MIN_BANDS * MIN_OBSERVATIONS_PER_BAND,
            "note": (f"insufficient data: {len(actual)} observations cannot "
                     f"support a stability index. At least "
                     f"{MIN_BANDS * MIN_OBSERVATIONS_PER_BAND} are needed, and "
                     f"{10 * MIN_OBSERVATIONS_PER_BAND} for a ten-band "
                     f"comparison. This is not a stable reading - it is no "
                     f"reading."),
        }

    edges = list(edges) if edges is not None else quantile_edges(expected, bands)
    expected_shares = _distribution(expected, edges)
    actual_shares = _distribution(actual, edges)

    floor = max(_floor(len(actual)), _floor(len(expected)))
    total = 0.0
    rows = []
    boundaries = [None] + list(edges) + [None]
    for index, (expected_share, actual_share) in enumerate(
            zip(expected_shares, actual_shares)):
        e = max(expected_share, floor)
        a = max(actual_share, floor)
        contribution = (a - e) * math.log(a / e)
        total += contribution
        rows.append({
            "band": index,
            "from": boundaries[index],
            "to": boundaries[index + 1],
            "expected_share": round(expected_share, 6),
            "actual_share": round(actual_share, 6),
            "contribution": round(contribution, 6),
        })

    return {
        "index": round(total, 6),
        "status": _status(total, PSI_WARNING, PSI_BREACH),
        "thresholds": {"warning": PSI_WARNING, "breach": PSI_BREACH},
        "bands": rows,
        "band_count": len(rows),
        "sample_size": len(actual),
        "reference_size": len(expected),
        # What the index would read on average with no drift at all, given
        # this sample size and band count. Disclosed so a reader can tell a
        # real movement from the noise the metric always carries.
        "expected_noise": round((len(rows) - 1) / max(len(actual), 1), 6),
        "largest_moves": sorted(rows, key=lambda row: row["contribution"],
                                reverse=True)[:3],
        "smoothing_floor": round(floor, 8),
    }


def characteristic_stability(expected: dict[str, Sequence[float]],
                             actual: dict[str, Sequence[float]]
                             ) -> dict[str, Any]:
    """CSI per characteristic: which input moved, not merely that the score did."""
    per_characteristic = {}
    for name, expected_values in expected.items():
        actual_values = actual.get(name) or []
        clean_expected = [v for v in expected_values if v is not None]
        clean_actual = [v for v in actual_values if v is not None]
        per_characteristic[name] = stability_index(clean_expected, clean_actual)
    worst = max(
        (item for item in per_characteristic.items()
         if item[1]["index"] is not None),
        key=lambda item: item[1]["index"], default=(None, None))
    return {
        "characteristics": {
            name: {"csi": result["index"], "status": result["status"]}
            for name, result in per_characteristic.items()
        },
        "worst_characteristic": worst[0],
        "worst_csi": worst[1]["index"] if worst[1] else None,
        "detail": per_characteristic,
    }


def calibration(expected_pd: Sequence[float],
                outcomes: Sequence[int]) -> dict[str, Any]:
    """Observed versus expected default rates.

    Discrimination (Gini, KS) says whether the model ranks; calibration says
    whether the probabilities are true. A scorecard can rank perfectly and
    still price every loan wrongly, which is why both are reported.
    """
    if not expected_pd or len(expected_pd) != len(outcomes):
        return {"observed": None, "expected": None, "oe_ratio": None,
                "status": "UNKNOWN", "note": "no outcome data available"}

    expected_mean = sum(expected_pd) / len(expected_pd)
    observed_mean = sum(outcomes) / len(outcomes)
    oe_ratio = (observed_mean / expected_mean) if expected_mean > 0 else None
    brier = sum((p - y) ** 2 for p, y in zip(expected_pd, outcomes)) / len(outcomes)

    if oe_ratio is None:
        status = "UNKNOWN"
    elif OE_LOWER <= oe_ratio <= OE_UPPER:
        status = STABLE
    elif 0.6 <= oe_ratio <= 1.5:
        status = WARNING
    else:
        status = BREACH

    return {
        "n": len(outcomes),
        "expected_default_rate": round(expected_mean, 6),
        "observed_default_rate": round(observed_mean, 6),
        "oe_ratio": None if oe_ratio is None else round(oe_ratio, 4),
        "brier_score": round(brier, 6),
        "status": status,
        "thresholds": {"acceptable_range": [OE_LOWER, OE_UPPER]},
    }


def decision_metrics(decisions: Iterable[dict]) -> dict[str, Any]:
    """Approval, decline, referral and counteroffer rates, and their drivers."""
    decisions = list(decisions)
    if not decisions:
        return {"n": 0, "note": "no decisions in period"}

    outcomes: dict[str, int] = {}
    decline_types: dict[str, int] = {}
    by_grade: dict[str, dict[str, int]] = {}
    reasons: dict[str, int] = {}
    segments: dict[str, int] = {}
    dq: dict[str, int] = {}
    counteroffers = 0

    for record in decisions:
        outcome = record.get("outcome", "UNKNOWN")
        outcomes[outcome] = outcomes.get(outcome, 0) + 1

        if record.get("decline_type"):
            key = record["decline_type"]
            decline_types[key] = decline_types.get(key, 0) + 1

        assessment = record.get("assessment") or {}
        grade = assessment.get("risk_grade") or "UNSCORED"
        bucket = by_grade.setdefault(grade, {})
        bucket[outcome] = bucket.get(outcome, 0) + 1

        status = assessment.get("data_quality_status") or "UNKNOWN"
        dq[status] = dq.get(status, 0) + 1

        segment = (record.get("versions") or {}).get("model_segment") or "NONE"
        segments[segment] = segments.get(segment, 0) + 1

        if (record.get("offer") or {}).get("is_counteroffer"):
            counteroffers += 1

        for reason in record.get("reason_codes") or []:
            code = reason.get("code") if isinstance(reason, dict) else reason
            if code:
                reasons[code] = reasons.get(code, 0) + 1

    total = len(decisions)
    approvals = outcomes.get("APPROVE", 0)
    return {
        "n": total,
        "approval_rate": round(approvals / total, 4),
        "decline_rate": round(outcomes.get("DECLINE", 0) / total, 4),
        "referral_rate": round(outcomes.get("REFER", 0) / total, 4),
        "insufficient_rate": round(
            outcomes.get("INSUFFICIENT_INFORMATION", 0) / total, 4),
        "counteroffer_rate": round(counteroffers / total, 4),
        "counteroffer_share_of_approvals": (
            round(counteroffers / approvals, 4) if approvals else None),
        "outcomes": outcomes,
        "decline_types": decline_types,
        "by_grade": by_grade,
        "by_segment": segments,
        "data_quality": dq,
        "top_reason_codes": sorted(reasons.items(), key=lambda kv: kv[1],
                                   reverse=True)[:10],
    }


def report(*, decisions: Sequence[dict],
           reference_scores: Optional[Sequence[float]] = None,
           reference_features: Optional[dict[str, Sequence[float]]] = None,
           outcomes: Optional[Sequence[int]] = None,
           model_version: Optional[str] = None) -> dict[str, Any]:
    """Assemble the monitoring pack an operations or risk team would read."""
    actual_scores = [d["assessment"]["score"] for d in decisions
                     if (d.get("assessment") or {}).get("score") is not None]
    expected_pd = [d["assessment"]["probability_of_default"] for d in decisions
                   if (d.get("assessment") or {}).get("probability_of_default")
                   is not None]

    psi = (stability_index(list(reference_scores), actual_scores)
           if reference_scores else
           {"index": None, "status": "UNKNOWN",
            "note": "no reference distribution supplied"})

    csi = ({"characteristics": {}, "note": "no reference features supplied"}
           if not reference_features else
           characteristic_stability(reference_features, {}))

    calibration_block = (calibration(expected_pd, list(outcomes))
                         if outcomes else
                         calibration([], []))

    metrics = decision_metrics(decisions)
    statuses = [psi.get("status"), calibration_block.get("status")]
    overall = (BREACH if BREACH in statuses else
               WARNING if WARNING in statuses else
               STABLE if STABLE in statuses else "UNKNOWN")

    return {
        "model_version": model_version,
        "overall_status": overall,
        "population_stability": psi,
        "characteristic_stability": csi,
        "calibration": calibration_block,
        "decisions": metrics,
        "interpretation": {
            "psi": ("PSI compares the current score distribution with the "
                    "reference. Movement means the applicant mix has changed, "
                    "which may be marketing, seasonality or a data fault - not "
                    "necessarily model decay."),
            "calibration": ("O/E compares observed defaults with predicted. "
                            "It requires outcome data, so it is UNKNOWN until "
                            "a performance window has elapsed."),
            "action": ("A BREACH is an instruction to investigate, not to "
                       "switch the model off. Model changes follow the "
                       "governance process (BR-GOV-05)."),
        },
    }
