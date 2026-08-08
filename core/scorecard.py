"""Runtime scoring engine (segmented scorecard).

Implements IPSRS FR-SCO-01..03:

  * loads an approved, versioned scorecard artefact and never re-fits at runtime
  * selects the applicable segment (BUREAU / THIN) from the data actually
    available, then scores with that segment's characteristics only
  * returns PD, scaled score, grade, reason codes, model version, feature-set
    version, timestamp, data-quality status and a confidence indicator
  * scaling is tenant-configurable (base score, base odds, points to double the
    odds, bounds) so one PD model serves many presentation scales
  * scoring is deterministic: identical artefact + feature values produce
    identical output, verified by a golden-file test

Segmentation matters for explanation as well as for statistics: a thin-file
applicant is scored by a model that contains no bureau characteristic, so the
engine cannot construct a bureau-based reason for someone who has no bureau
record. That guarantee is structural here, not a filter applied afterwards.

Points arithmetic follows standard scorecard practice. The model estimates
P(default), so::

    log_odds_bad = intercept + sum(coef_c * woe_c)
    score        = offset - factor * log_odds_bad
    factor       = pdo / ln(2)
    offset       = base_score - factor * ln(base_odds)

WoE is oriented so that higher means safer and every coefficient is negative,
so each characteristic contributes positive points; the points a characteristic
costs relative to its best bin drive the reason codes.
"""
from __future__ import annotations

import json
import math
import pathlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from core import reason_codes
from core.config import ScoreScale, Tenant, grade_for_score

ARTEFACT_DIR = pathlib.Path(__file__).resolve().parents[1] / "artefacts"
MISSING_BIN = "MISSING"

_CACHE: dict[str, "Scorecard"] = {}


@dataclass(frozen=True)
class ScoreResult:
    """The score response contract (IPSRS FR-SCO-01)."""

    probability_of_default: float
    score: int
    risk_grade: str
    reason_codes: list[str]
    model_id: str
    model_version: str
    segment: str
    feature_set_version: str
    scored_at: str
    data_quality_status: str
    confidence: str
    points: dict[str, int] = field(default_factory=dict)
    points_lost: dict[str, int] = field(default_factory=dict)
    bins_used: dict[str, str] = field(default_factory=dict)

    def as_dict(self, *, include_internals: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "probability_of_default": round(self.probability_of_default, 6),
            "score": self.score,
            "risk_grade": self.risk_grade,
            "reason_codes": reason_codes.render(self.reason_codes),
            "model_id": self.model_id,
            "model_version": self.model_version,
            "model_segment": self.segment,
            "feature_set_version": self.feature_set_version,
            "reason_code_library": reason_codes.LIBRARY_VERSION,
            "scored_at": self.scored_at,
            "data_quality_status": self.data_quality_status,
            "confidence": self.confidence,
        }
        if include_internals:
            payload["attribution"] = {
                "points": self.points,
                "points_lost_vs_best_bin": self.points_lost,
                "bins_used": self.bins_used,
            }
        return payload


class Segment:
    """One fitted scorecard within a segmented model."""

    def __init__(self, name: str, payload: dict):
        self.name = name
        self.characteristics: list[str] = list(payload["characteristics"])
        self.intercept: float = float(payload["intercept"])
        self.coefficients: dict[str, float] = {
            k: float(v) for k, v in payload["coefficients"].items()}
        self.binning: dict[str, dict] = payload["binning"]
        self.neutral_woe: dict[str, float] = {
            k: float(v) for k, v in payload["neutral_woe"].items()}
        self.vif: dict[str, float] = payload.get("vif", {})

    def bin_for(self, characteristic: str, value: Optional[Any]) -> str:
        if value is None:
            return MISSING_BIN
        if isinstance(value, bool):
            value = 1.0 if value else 0.0
        try:
            value = float(value)
        except (TypeError, ValueError):
            return MISSING_BIN
        if math.isnan(value):
            return MISSING_BIN
        edges = self.binning[characteristic]["edges"]
        for index, edge in enumerate(edges):
            if value <= float(edge):
                return f"B{index}"
        return f"B{len(edges)}"

    def woe(self, characteristic: str, bin_label: str) -> float:
        bins = self.binning[characteristic]["bins"]
        if bin_label in bins:
            return float(bins[bin_label]["woe"])
        # An unseen bin is neutral rather than silently favourable or punitive.
        return 0.0


class Scorecard:
    """An immutable, approved segmented scorecard artefact ready for scoring."""

    def __init__(self, artefact: dict):
        self._a = artefact
        self.model_id: str = artefact["model_id"]
        self.model_version: str = artefact["model_version"]
        self.model_type: str = artefact.get("model_type", "")
        self.feature_set_version: str = artefact["feature_set_version"]
        self.governance: dict = artefact.get("governance", {})
        self.segmentation: dict = artefact.get("segmentation", {})
        self.performance: dict = artefact.get("performance", {})
        self.warnings: list[str] = list(artefact.get("warnings", []))
        self.sha256: str = artefact.get("artefact_sha256", "")
        self.segments: dict[str, Segment] = {
            name: Segment(name, payload)
            for name, payload in artefact["segments"].items()
        }

    # -- segment selection ------------------------------------------------- #
    def select_segment(self, values: dict[str, Any]) -> str:
        """Choose the segment from the information actually available.

        The rule mirrors ``segmentation.rule`` in the artefact: an applicant
        with no retrievable bureau characteristic is scored by the THIN model.
        """
        bureau_characteristics = [
            name for name in self.segments["BUREAU"].characteristics
            if name.startswith("bureau_") or name in
            ("credit_history_months", "revolving_utilisation", "prior_default")
        ]
        has_bureau = any(values.get(name) is not None
                         for name in bureau_characteristics)
        return "BUREAU" if has_bureau else "THIN"

    # -- scoring ----------------------------------------------------------- #
    def score(self, values: dict[str, Any], *, tenant: Tenant,
              dq_status: str = "OK", max_reasons: int = 4,
              materiality_points: int = 8,
              segment: Optional[str] = None) -> ScoreResult:
        segment_name = segment or self.select_segment(values)
        model = self.segments[segment_name]

        scale = tenant.score_scale
        factor = scale.pdo / math.log(2)
        offset = scale.base_score - factor * math.log(scale.base_odds)

        log_odds = model.intercept
        bins_used: dict[str, str] = {}
        points: dict[str, int] = {}
        points_lost: dict[str, int] = {}

        for characteristic in model.characteristics:
            coefficient = model.coefficients[characteristic]
            bin_label = model.bin_for(characteristic, values.get(characteristic))
            woe = model.woe(characteristic, bin_label)
            log_odds += coefficient * woe

            bins_used[characteristic] = bin_label
            contribution = -factor * coefficient * woe
            best = -factor * coefficient * model.neutral_woe[characteristic]
            points[characteristic] = int(round(contribution))
            points_lost[characteristic] = int(round(max(best - contribution, 0.0)))

        pd = 1.0 / (1.0 + math.exp(-max(min(log_odds, 35.0), -35.0)))
        raw_score = offset - factor * log_odds
        score = int(round(max(min(raw_score, scale.max_score), scale.min_score)))

        # Reason codes: characteristics costing the most points, worst first.
        # A characteristic scored from its MISSING bin never produces a
        # substantive reason, and immaterial point losses are suppressed so a
        # strong applicant is not handed a list of trivial adverse factors.
        codes: list[str] = []
        for characteristic, lost in sorted(points_lost.items(),
                                          key=lambda kv: kv[1], reverse=True):
            if len(codes) >= max_reasons:
                break
            if lost < materiality_points:
                continue
            if bins_used.get(characteristic) == MISSING_BIN:
                continue
            code = reason_codes.CHARACTERISTIC_CODES.get(characteristic)
            if code and code not in codes:
                codes.append(code)

        if dq_status == "BLOCK":
            codes = ["INSUFFICIENT_INFORMATION"]
        elif segment_name == "THIN" and "NO_BUREAU_RECORD" not in codes:
            codes.append("NO_BUREAU_RECORD")

        confidence = {"OK": "HIGH", "DEGRADED": "MEDIUM", "BLOCK": "LOW"}.get(
            dq_status, "MEDIUM")
        # A thin-file score is inherently less reliable: the segment's own
        # discrimination is materially weaker, so never report HIGH confidence.
        if segment_name == "THIN" and confidence == "HIGH":
            confidence = "MEDIUM"

        return ScoreResult(
            probability_of_default=pd,
            score=score,
            risk_grade=grade_for_score(tenant, score),
            reason_codes=codes,
            model_id=self.model_id,
            model_version=self.model_version,
            segment=segment_name,
            feature_set_version=self.feature_set_version,
            scored_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            data_quality_status=dq_status,
            confidence=confidence,
            points=points,
            points_lost=points_lost,
            bins_used=bins_used,
        )

    # -- diagnostics ------------------------------------------------------- #
    def segment_performance(self, segment_name: str,
                            split: str = "out_of_time") -> dict:
        return self.performance.get(f"{split}_{segment_name.lower()}", {})

    def scale_reference(self, scale: ScoreScale) -> dict[str, Any]:
        factor = scale.pdo / math.log(2)
        offset = scale.base_score - factor * math.log(scale.base_odds)
        return {"factor": round(factor, 4), "offset": round(offset, 4),
                "min_score": scale.min_score, "max_score": scale.max_score}


def load(model_id: str = "APPLICATION_LR_V1", *, refresh: bool = False) -> Scorecard:
    """Load (and cache) an approved scorecard artefact."""
    if refresh or model_id not in _CACHE:
        path = ARTEFACT_DIR / f"scorecard_{model_id}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"scorecard artefact missing: {path}. Run "
                f"'python3 -m model.train_scorecard' first.")
        _CACHE[model_id] = Scorecard(json.loads(path.read_text()))
    return _CACHE[model_id]


def pd_to_score(pd: float, scale: ScoreScale) -> int:
    """Expose the scaling function for reporting and cut-off calibration."""
    pd = min(max(pd, 1e-9), 1 - 1e-9)
    factor = scale.pdo / math.log(2)
    offset = scale.base_score - factor * math.log(scale.base_odds)
    raw = offset - factor * math.log(pd / (1 - pd))
    return int(round(max(min(raw, scale.max_score), scale.min_score)))
