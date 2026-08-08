"""Scorecard development pipeline: segmented WoE logistic regression.

Implements the methodology of IPSRS FR-SCO-01 / master brief section 9 at
prototype fidelity.

SEGMENTATION
------------
The population is split into two segments with genuinely different information
sets, and a separate scorecard is fitted to each:

  BUREAU  applicants with a retrievable credit-bureau record
  THIN    applicants with no bureau record at all

This replaces an earlier single-model design in which every bureau
characteristic shared one MISSING bin. That design counted the same
"information absent" effect once per bureau characteristic, produced severe
collinearity, and flipped two coefficients to the wrong sign. Segmentation is
the standard remedy: each model sees only characteristics that exist for its
population, so no characteristic carries a missing-information proxy, and the
thin-file population gets a model actually fitted to it rather than a
bureau model with holes punched in it.

Per segment the pipeline performs:
  * supervised binning: quantile candidates, then merging until bad rates are
    monotonic and every bin holds at least a minimum population share
  * weight-of-evidence transformation and information value screening
  * variance-inflation diagnostics on the WoE design matrix
  * logistic regression by IRLS with light L2 regularisation
  * sign-convention enforcement (all coefficients must be negative)
  * development / validation / out-of-time metrics: AUC, Gini, KS, Brier and
    Cox calibration intercept and slope

Run:  python3 -m model.train_scorecard
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from core import features as fstore          # noqa: E402
from model import synth                      # noqa: E402

MODEL_ID = "APPLICATION_LR_V1"
MODEL_VERSION = "2.0.0"                      # 2.x = segmented
ARTEFACTS = pathlib.Path(__file__).resolve().parents[1] / "artefacts"
MISSING_BIN = "MISSING"
IV_FLOOR = 0.02

#: Characteristics available to each segment, with candidate bin counts.
#: The THIN segment deliberately contains no bureau-sourced characteristic.
SEGMENTS: dict[str, dict[str, int]] = {
    "BUREAU": {
        "age_years": 5,
        "employment_months": 5,
        "existing_dsr": 5,
        "requested_to_income": 5,
        "bureau_worst_dpd": 4,
        "bureau_enquiries_6m": 4,
        "credit_history_months": 4,
        "revolving_utilisation": 4,
        "prior_default": 2,
        "relationship_months": 4,
    },
    "THIN": {
        "age_years": 5,
        "employment_months": 4,
        "existing_dsr": 4,
        "requested_to_income": 4,
        "relationship_months": 4,
    },
}


# --------------------------------------------------------------------------- #
# Binning and weight of evidence
# --------------------------------------------------------------------------- #
def _bin_edges(values: np.ndarray, bins: int) -> list[float]:
    observed = values[~np.isnan(values)]
    if len(observed) == 0:
        return []
    quantiles = np.linspace(0, 100, bins + 1)[1:-1]
    return [float(e) for e in
            sorted(set(np.percentile(observed, quantiles).round(4).tolist()))]


def _assign(value, edges: list[float]) -> str:
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
    for index, edge in enumerate(edges):
        if value <= edge:
            return f"B{index}"
    return f"B{len(edges)}"


def _woe_table(bin_labels: np.ndarray, target: np.ndarray) -> dict:
    """WoE and IV per bin with Laplace smoothing (avoids infinite WoE)."""
    total_bad = float(target.sum())
    total_good = float(len(target) - total_bad)
    table: dict[str, dict] = {}
    iv = 0.0
    for label in sorted(set(bin_labels.tolist())):
        mask = bin_labels == label
        bad = float(target[mask].sum())
        good = float(mask.sum() - bad)
        bad_rate = (bad + 0.5) / (total_bad + 1.0)
        good_rate = (good + 0.5) / (total_good + 1.0)
        woe = math.log(good_rate / bad_rate)
        iv += (good_rate - bad_rate) * woe
        table[label] = {
            "woe": round(woe, 6),
            "count": int(mask.sum()),
            "bad": int(bad),
            "bad_rate": round(bad / max(mask.sum(), 1), 6),
        }
    return {"bins": table, "iv": round(iv, 6)}


def _ordered_bad_rates(edges: list[float], table: dict) -> list[float]:
    return [table["bins"][f"B{i}"]["bad_rate"]
            for i in range(len(edges) + 1)
            if f"B{i}" in table["bins"]]


def _enforce_monotonic(values: np.ndarray, target: np.ndarray,
                       edges: list[float], *, min_share: float = 0.05,
                       min_bins: int = 3) -> tuple[list[float], dict]:
    """Merge adjacent bins until bad rates are monotonic and bins are populated."""
    n = len(values)
    while True:
        labels = np.array([_assign(v, edges) for v in values])
        table = _woe_table(labels, target)

        counts = [(i, table["bins"].get(f"B{i}", {"count": 0})["count"])
                  for i in range(len(edges) + 1)]
        small = [i for i, c in counts if c < min_share * n]
        if small and edges and len(edges) + 1 > min_bins:
            edges.pop(min(small[0], len(edges) - 1))
            continue

        rates = _ordered_bad_rates(edges, table)
        if not edges or len(rates) <= min_bins:
            return edges, table
        deltas = [b - a for a, b in zip(rates, rates[1:])]
        direction = 1.0 if sum(deltas) >= 0 else -1.0
        violations = [i for i, d in enumerate(deltas) if d * direction < -1e-9]
        if not violations:
            return edges, table
        worst = max(violations, key=lambda i: abs(deltas[i]))
        edges.pop(min(worst, len(edges) - 1))


def _is_monotonic(edges: list[float], table: dict) -> bool:
    ordered = _ordered_bad_rates(edges, table)
    if len(ordered) < 3:
        return True
    up = all(b >= a - 1e-9 for a, b in zip(ordered, ordered[1:]))
    down = all(b <= a + 1e-9 for a, b in zip(ordered, ordered[1:]))
    return up or down


# --------------------------------------------------------------------------- #
# Logistic regression and diagnostics
# --------------------------------------------------------------------------- #
def _fit_logistic(X: np.ndarray, y: np.ndarray, *, l2: float = 1e-3,
                  iterations: int = 60) -> np.ndarray:
    design = np.column_stack([np.ones(len(X)), X])
    beta = np.zeros(design.shape[1])
    penalty = np.eye(design.shape[1]) * l2
    penalty[0, 0] = 0.0
    for _ in range(iterations):
        eta = design @ beta
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -35, 35)))
        w = np.clip(p * (1 - p), 1e-8, None)
        gradient = design.T @ (y - p) - penalty @ beta
        hessian = design.T @ (design * w[:, None]) + penalty
        beta = beta + np.linalg.solve(hessian, gradient)
    return beta


def _vif(X: np.ndarray, names: list[str]) -> dict[str, float]:
    """Variance inflation factor per characteristic on the WoE design matrix.

    VIF above roughly 5 indicates the characteristic is largely explained by
    the others - the diagnostic that would have caught the single-model
    collinearity problem before fitting rather than after.
    """
    out: dict[str, float] = {}
    for index, name in enumerate(names):
        others = np.delete(X, index, axis=1)
        if others.shape[1] == 0:
            out[name] = 1.0
            continue
        design = np.column_stack([np.ones(len(others)), others])
        coefficients, *_ = np.linalg.lstsq(design, X[:, index], rcond=None)
        residual = X[:, index] - design @ coefficients
        total = float(np.sum((X[:, index] - X[:, index].mean()) ** 2))
        r_squared = 0.0 if total == 0 else 1.0 - float(np.sum(residual ** 2)) / total
        out[name] = round(1.0 / max(1.0 - r_squared, 1e-6), 3)
    return out


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _auc(y: np.ndarray, score: np.ndarray) -> float:
    if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    _, inverse, counts = np.unique(score, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    positives = y == 1
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _ks(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) == 0:
        return float("nan")
    order = np.argsort(p)
    y_sorted = y[order]
    cum_bad = np.cumsum(y_sorted) / max(y_sorted.sum(), 1)
    cum_good = np.cumsum(1 - y_sorted) / max((1 - y_sorted).sum(), 1)
    return float(np.max(np.abs(cum_bad - cum_good)))


def _calibration(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    clipped = np.clip(p, 1e-9, 1 - 1e-9)
    logit_p = np.log(clipped / (1 - clipped))
    beta = _fit_logistic(logit_p.reshape(-1, 1), y.astype(float), l2=0.0)
    return {"intercept": round(float(beta[0]), 6), "slope": round(float(beta[1]), 6)}


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    auc = _auc(y, p)
    return {
        "n": int(len(y)),
        "bad_rate": round(float(y.mean()), 6) if len(y) else None,
        "auc": round(auc, 6),
        "gini": round(2 * auc - 1, 6),
        "ks": round(_ks(y, p), 6),
        "brier": round(float(np.mean((p - y) ** 2)), 6) if len(y) else None,
        "calibration": _calibration(y, p) if len(y) else None,
    }


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def build() -> dict:
    data = synth.generate()
    y_all = data["default"].astype(float)
    thin = data["thin_file"].astype(bool)
    dev_idx, val_idx, oot_idx = synth.split(data)

    segments: dict[str, dict] = {}
    warnings: list[str] = []

    for segment_name, candidates in SEGMENTS.items():
        in_segment = thin if segment_name == "THIN" else ~thin
        seg_dev = np.array([i for i in dev_idx if in_segment[i]])

        binning: dict[str, dict] = {}
        for name, bins in candidates.items():
            raw = np.asarray(data[name], dtype=float)[seg_dev]
            if fstore.BY_NAME[name].data_type == "boolean":
                edges: list[float] = [0.5]
                table = _woe_table(np.array([_assign(v, edges) for v in raw]),
                                   y_all[seg_dev])
            else:
                edges, table = _enforce_monotonic(
                    raw, y_all[seg_dev], _bin_edges(raw, bins))
            if edges and not _is_monotonic(edges, table):
                warnings.append(f"{segment_name}/{name}: bad rate not monotonic "
                                f"after merging - flagged for expert re-binning")
            binning[name] = {"edges": edges, **table}

        selected = [n for n in candidates if binning[n]["iv"] >= IV_FLOOR]
        for name in candidates:
            if name not in selected:
                warnings.append(f"{segment_name}/{name}: dropped, IV "
                                f"{binning[name]['iv']:.4f} below {IV_FLOOR}")

        def woe_matrix(indices: np.ndarray, names: list[str]) -> np.ndarray:
            columns = []
            for name in names:
                raw = np.asarray(data[name], dtype=float)[indices]
                edges = binning[name]["edges"]
                table = binning[name]["bins"]
                columns.append([table.get(_assign(v, edges),
                                          {"woe": 0.0})["woe"] for v in raw])
            return np.array(columns, dtype=float).T

        vif = _vif(woe_matrix(seg_dev, selected), selected) if selected else {}
        for name, value in vif.items():
            if value > 5.0:
                warnings.append(f"{segment_name}/{name}: VIF {value} indicates "
                                f"collinearity with other characteristics")

        beta = _fit_logistic(woe_matrix(seg_dev, selected), y_all[seg_dev])
        while True:
            offenders = [(n, float(c)) for n, c in zip(selected, beta[1:]) if c > 0]
            if not offenders or len(selected) <= 3:
                break
            worst = max(offenders, key=lambda item: item[1])[0]
            warnings.append(f"{segment_name}/{worst}: removed after fitting - "
                            f"coefficient sign violation")
            selected = [n for n in selected if n != worst]
            beta = _fit_logistic(woe_matrix(seg_dev, selected), y_all[seg_dev])

        segments[segment_name] = {
            "characteristics": selected,
            "intercept": round(float(beta[0]), 6),
            "coefficients": {n: round(float(c), 6)
                             for n, c in zip(selected, beta[1:])},
            "binning": {n: binning[n] for n in selected},
            "neutral_woe": {
                n: round(max(b["woe"] for b in binning[n]["bins"].values()), 6)
                for n in selected},
            "vif": vif,
            "n_development": int(len(seg_dev)),
            "development_bad_rate": round(float(y_all[seg_dev].mean()), 6),
        }

    # ---- evaluation: every row scored by its own segment model ------------ #
    def predict(indices: np.ndarray) -> np.ndarray:
        out = np.zeros(len(indices))
        for position, row in enumerate(indices):
            segment = segments["THIN" if thin[row] else "BUREAU"]
            log_odds = segment["intercept"]
            for name in segment["characteristics"]:
                value = float(np.asarray(data[name], dtype=float)[row])
                label = _assign(value, segment["binning"][name]["edges"])
                woe = segment["binning"][name]["bins"].get(
                    label, {"woe": 0.0})["woe"]
                log_odds += segment["coefficients"][name] * woe
            out[position] = 1.0 / (1.0 + math.exp(-max(min(log_odds, 35.0), -35.0)))
        return out

    performance: dict[str, dict] = {}
    for split_name, indices in (("development", dev_idx), ("validation", val_idx),
                                ("out_of_time", oot_idx)):
        performance[split_name] = _metrics(y_all[indices], predict(indices))
        for segment_name in SEGMENTS:
            subset = np.array([i for i in indices
                               if (thin[i] if segment_name == "THIN" else not thin[i])])
            performance[f"{split_name}_{segment_name.lower()}"] = _metrics(
                y_all[subset], predict(subset))

    artefact = {
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "model_type": "segmented_logistic_regression_woe_scorecard",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feature_set_version": fstore.feature_set_version(),
        "segmentation": {
            "rule": "THIN if no credit-bureau record retrieved, else BUREAU",
            "rationale": ("separate information sets; avoids counting the "
                          "information-absent effect once per bureau "
                          "characteristic, which caused collinearity and sign "
                          "reversals in the single-model design"),
            "segments": list(SEGMENTS),
        },
        "training_data": {
            "source": "SYNTHETIC - prototype only, not a real development sample",
            "generator": "model.synth.generate",
            "seed": 20260719,
            "n_total": int(len(y_all)),
            "n_development": int(len(dev_idx)),
            "n_validation": int(len(val_idx)),
            "n_out_of_time": int(len(oot_idx)),
            "thin_file_share": round(float(thin.mean()), 6),
            "default_definition": ("90+ days past due within a 12-month "
                                   "performance window (synthetic proxy)"),
        },
        "segments": segments,
        # Reference distribution for population-stability monitoring. Storing
        # PDs (not scores) keeps it tenant-agnostic: each tenant's scale is
        # applied at read time, so one reference serves every score scale.
        "reference_pd_distribution": [
            round(float(p), 6) for p in predict(dev_idx)[:2000]
        ],
        "performance": performance,
        "warnings": warnings,
        "governance": {
            "status": "PROTOTYPE - NOT APPROVED FOR PRODUCTION",
            "independent_validation": "not performed",
            "approval_reference": None,
            "limitations": [
                "Trained on synthetic data; no inference about real borrowers.",
                "No reject inference performed.",
                "No fairness testing performed (no protected attributes in "
                "the synthetic sample).",
                "Cut-offs are calibrated separately: see "
                "model.calibrate_cutoffs.",
            ],
        },
    }
    artefact["artefact_sha256"] = hashlib.sha256(
        json.dumps(artefact, sort_keys=True).encode()).hexdigest()
    return artefact


def main() -> None:
    ARTEFACTS.mkdir(parents=True, exist_ok=True)
    artefact = build()

    (ARTEFACTS / f"scorecard_{MODEL_ID}.json").write_text(
        json.dumps(artefact, indent=2))
    (ARTEFACTS / "model_development_report.json").write_text(json.dumps({
        "model_id": artefact["model_id"],
        "model_version": artefact["model_version"],
        "model_type": artefact["model_type"],
        "trained_at": artefact["trained_at"],
        "segmentation": artefact["segmentation"],
        "training_data": artefact["training_data"],
        "segments": {
            name: {
                "characteristics": segment["characteristics"],
                "information_values": {c: segment["binning"][c]["iv"]
                                       for c in segment["characteristics"]},
                "coefficients": segment["coefficients"],
                "vif": segment["vif"],
                "n_development": segment["n_development"],
                "development_bad_rate": segment["development_bad_rate"],
            }
            for name, segment in artefact["segments"].items()
        },
        "performance": artefact["performance"],
        "warnings": artefact["warnings"],
        "governance": artefact["governance"],
    }, indent=2))
    (ARTEFACTS / "feature_dictionary.json").write_text(
        json.dumps(fstore.dictionary(), indent=2, default=str))

    print(f"model    : {artefact['model_id']} v{artefact['model_version']} "
          f"({artefact['model_type']})")
    for name, segment in artefact["segments"].items():
        max_vif = max(segment["vif"].values()) if segment["vif"] else 0
        print(f"  segment {name:<7} n={segment['n_development']:<5} "
              f"bad={segment['development_bad_rate']:.3f}  "
              f"chars={len(segment['characteristics'])}  max VIF={max_vif:.2f}")
    print()
    for split_name in ("development", "validation", "out_of_time"):
        for suffix, label in (("", "all"), ("_bureau", "bureau"), ("_thin", "thin")):
            m = artefact["performance"][f"{split_name}{suffix}"]
            print(f"  {split_name:<12} {label:<7} n={m['n']:<5} "
                  f"bad={m['bad_rate']:.3f}  AUC={m['auc']:.3f}  "
                  f"Gini={m['gini']:.3f}  KS={m['ks']:.3f}  "
                  f"Brier={m['brier']:.4f}")
        print()
    if artefact["warnings"]:
        print("warnings :")
        for warning in artefact["warnings"]:
            print("  -", warning)
    print(f"sha256   : {artefact['artefact_sha256'][:16]}...")


if __name__ == "__main__":
    main()
