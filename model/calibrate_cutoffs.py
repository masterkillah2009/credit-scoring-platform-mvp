"""Cut-off calibration from the score distribution.

Cut-offs must not be chosen by eye. This module derives them from the observed
score distribution against each tenant's stated risk appetite, and reports the
trade-off a credit committee actually needs to see:

  * bad rate by score band, with population share
  * for each candidate cut-off: approval rate, bad rate among approved, and bad
    rate among declined (the opportunity cost of the cut-off)
  * the lowest cut-off that satisfies the target bad rate - the most inclusive
    threshold consistent with the appetite
  * a swap-set analysis against the currently configured cut-off: who would be
    approved that previously was not, and how those applicants perform

Calibration deliberately uses the validation and out-of-time samples, never the
development sample, so the reported trade-off is not optimistic.

Run:  python3 -m model.calibrate_cutoffs
Writes: artefacts/cutoff_calibration.json
"""
from __future__ import annotations

import json
import pathlib
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from core import config, scorecard             # noqa: E402
from model import synth                        # noqa: E402

ARTEFACTS = pathlib.Path(__file__).resolve().parents[1] / "artefacts"

#: Risk appetite per tenant product: the maximum acceptable bad rate among
#: approved applications. In production this belongs in tenant configuration
#: under maker-checker control (IPSRS FR-ADM-04), not in a script.
TARGET_BAD_RATE: dict[str, float] = {
    "ZAM-PAY": 0.08,     # payroll lender: secured by salary deduction
    "ZAM-MFI": 0.18,     # microfinance: higher risk carried at higher price
}

#: Referral band width below the accept cut-off, in score points, expressed as
#: a share of the tenant's points-to-double-the-odds.
REFERRAL_BAND_PDO_MULTIPLE = 1.5


def _score_population(card: scorecard.Scorecard, tenant: config.Tenant,
                      data: dict, indices: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (scores, defaults, thin_flags) for the calibration sample."""
    scores, defaults, thin = [], [], []
    characteristics = sorted({
        name for segment in card.segments.values()
        for name in segment.characteristics
    })
    for row in indices:
        values = {}
        for name in characteristics:
            raw = float(np.asarray(data[name], dtype=float)[row])
            values[name] = None if np.isnan(raw) else raw
        result = card.score(values, tenant=tenant)
        scores.append(result.score)
        defaults.append(int(data["default"][row]))
        thin.append(result.segment == "THIN")
    return np.array(scores), np.array(defaults), np.array(thin, dtype=bool)


def _band_table(scores: np.ndarray, defaults: np.ndarray, *,
                bands: int = 10) -> list[dict]:
    """Bad rate by score band, coarsest useful view for a credit committee."""
    edges = np.quantile(scores, np.linspace(0, 1, bands + 1))
    edges = np.unique(np.round(edges).astype(int))
    table = []
    for lower, upper in zip(edges, edges[1:]):
        mask = (scores >= lower) & (scores < upper)
        if not mask.any():
            continue
        table.append({
            "score_from": int(lower),
            "score_to": int(upper) - 1,
            "population": int(mask.sum()),
            "population_share": round(float(mask.mean()), 4),
            "bad_rate": round(float(defaults[mask].mean()), 4),
        })
    # top band closed at the maximum observed score
    mask = scores >= edges[-1]
    if mask.any():
        table.append({
            "score_from": int(edges[-1]),
            "score_to": int(scores.max()),
            "population": int(mask.sum()),
            "population_share": round(float(mask.mean()), 4),
            "bad_rate": round(float(defaults[mask].mean()), 4),
        })
    return table


def _cutoff_curve(scores: np.ndarray, defaults: np.ndarray, *,
                  step: int = 5, marginal_window: int = 20) -> list[dict]:
    curve = []
    for cutoff in range(int(scores.min()), int(scores.max()) + 1, step):
        approved = scores >= cutoff
        declined = ~approved
        if approved.sum() == 0:
            continue
        # Marginal risk: the applicants this cut-off admits at the margin,
        # i.e. the slice just above it. Cumulative bad rate alone is not a
        # sufficient test - a lenient cut-off can hide bad marginal business
        # behind a large mass of good business above it.
        marginal = approved & (scores < cutoff + marginal_window)
        curve.append({
            "cutoff": cutoff,
            "approval_rate": round(float(approved.mean()), 4),
            "bad_rate_approved": round(float(defaults[approved].mean()), 4),
            "bad_rate_declined": (round(float(defaults[declined].mean()), 4)
                                  if declined.any() else None),
            "bad_rate_marginal": (round(float(defaults[marginal].mean()), 4)
                                  if marginal.any() else None),
            "marginal_population": int(marginal.sum()),
        })
    return curve


def _recommend(curve: list[dict], target: float) -> dict | None:
    """Most inclusive cut-off that satisfies appetite both overall and at the margin.

    Two conditions must hold:
      1. the approved population's bad rate is within appetite, and
      2. the marginal business admitted at the cut-off is itself within
         appetite.

    Condition 2 is what stops a degenerate recommendation. Without it, a
    portfolio whose overall bad rate happens to sit just under the target
    "satisfies" the constraint by approving everyone - a cut-off that uses the
    model for nothing and prices no risk. Lending down to the point where
    marginal risk breaches appetite is the economically meaningful rule.
    """
    feasible = [point for point in curve
                if point["bad_rate_approved"] <= target
                and point["bad_rate_marginal"] is not None
                and point["marginal_population"] >= 30
                and point["bad_rate_marginal"] <= target]
    if not feasible:
        return None
    return min(feasible, key=lambda point: point["cutoff"])


def _swap_set(scores: np.ndarray, defaults: np.ndarray,
              current: int, proposed: int) -> dict:
    """Who moves between decisions, and how they perform."""
    low, high = min(current, proposed), max(current, proposed)
    moved = (scores >= low) & (scores < high)
    direction = ("more inclusive" if proposed < current
                 else "more conservative" if proposed > current else "unchanged")
    return {
        "current_cutoff": current,
        "proposed_cutoff": proposed,
        "direction": direction,
        "applicants_affected": int(moved.sum()),
        "share_of_population": round(float(moved.mean()), 4),
        "bad_rate_of_swap_set": (round(float(defaults[moved].mean()), 4)
                                 if moved.any() else None),
        "approval_rate_current": round(float((scores >= current).mean()), 4),
        "approval_rate_proposed": round(float((scores >= proposed).mean()), 4),
    }


def calibrate() -> dict:
    card = scorecard.load(refresh=True)
    data = synth.generate()
    _, val_idx, oot_idx = synth.split(data)
    calibration_idx = np.concatenate([val_idx, oot_idx])

    out: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_id": card.model_id,
        "model_version": card.model_version,
        "artefact_sha256": card.sha256,
        "sample": {
            "basis": "validation + out-of-time (development sample excluded)",
            "n": int(len(calibration_idx)),
            "source": "SYNTHETIC - prototype only",
        },
        "tenants": {},
    }

    for tenant in config.all_tenants():
        product = next(iter(tenant.products.values()))
        target = TARGET_BAD_RATE[tenant.code]
        scores, defaults, thin = _score_population(card, tenant, data,
                                                   calibration_idx)
        curve = _cutoff_curve(scores, defaults)
        recommendation = _recommend(curve, target)
        proposed_accept = recommendation["cutoff"] if recommendation else None
        referral_width = int(round(REFERRAL_BAND_PDO_MULTIPLE
                                   * tenant.score_scale.pdo))
        proposed_refer = (max(proposed_accept - referral_width,
                              tenant.score_scale.min_score)
                          if proposed_accept is not None else None)

        out["tenants"][tenant.code] = {
            "product": product.code,
            "target_bad_rate": target,
            "score_distribution": {
                "min": int(scores.min()), "max": int(scores.max()),
                "p10": int(np.percentile(scores, 10)),
                "median": int(np.median(scores)),
                "p90": int(np.percentile(scores, 90)),
            },
            "bad_rate_by_band": _band_table(scores, defaults),
            "cutoff_curve": curve,
            "recommended": {
                "accept_cutoff": proposed_accept,
                "refer_floor": proposed_refer,
                "approval_rate": recommendation["approval_rate"] if recommendation else None,
                "bad_rate_approved": recommendation["bad_rate_approved"] if recommendation else None,
                "bad_rate_marginal": recommendation["bad_rate_marginal"] if recommendation else None,
                "basis": (f"lowest cut-off with BOTH approved and marginal bad "
                          f"rate <= {target:.0%}; referral band = "
                          f"{REFERRAL_BAND_PDO_MULTIPLE} x PDO "
                          f"({referral_width} points)"),
            },
            "configured": {
                "accept_cutoff": product.accept_cutoff,
                "refer_floor": product.refer_floor,
                "approval_rate": round(float((scores >= product.accept_cutoff).mean()), 4),
                "bad_rate_approved": (
                    round(float(defaults[scores >= product.accept_cutoff].mean()), 4)
                    if (scores >= product.accept_cutoff).any() else None),
            },
            "swap_set": (_swap_set(scores, defaults, product.accept_cutoff,
                                   proposed_accept)
                         if proposed_accept is not None else None),
            "thin_file_note": {
                "share_of_population": round(float(thin.mean()), 4),
                "bad_rate": round(float(defaults[thin].mean()), 4) if thin.any() else None,
                "segment_gini_out_of_time":
                    card.segment_performance("THIN").get("gini"),
                "policy_implication": (
                    "Thin-file discrimination is materially weaker than the "
                    "bureau segment, so thin-file applicants should refer to "
                    "manual underwriting rather than be auto-decided on score "
                    "alone (policy rule R-THN-01)."),
            },
        }
    return out


def main() -> None:
    result = calibrate()
    ARTEFACTS.mkdir(parents=True, exist_ok=True)
    (ARTEFACTS / "cutoff_calibration.json").write_text(json.dumps(result, indent=2))

    print(f"Cut-off calibration - {result['model_id']} "
          f"v{result['model_version']}")
    print(f"sample: {result['sample']['basis']}, n={result['sample']['n']}\n")

    for code, block in result["tenants"].items():
        distribution = block["score_distribution"]
        print(f"{code} ({block['product']}) target bad rate "
              f"{block['target_bad_rate']:.0%}")
        print(f"  score distribution: min={distribution['min']} "
              f"p10={distribution['p10']} median={distribution['median']} "
              f"p90={distribution['p90']} max={distribution['max']}")
        print("  bad rate by score band:")
        for band in block["bad_rate_by_band"]:
            bar = "#" * int(round(band["bad_rate"] * 60))
            print(f"    {band['score_from']:>4}-{band['score_to']:<4} "
                  f"n={band['population']:<4} bad={band['bad_rate']:6.1%} {bar}")

        configured, recommended = block["configured"], block["recommended"]
        print(f"  configured : accept={configured['accept_cutoff']} "
              f"refer={configured['refer_floor']} -> approval "
              f"{configured['approval_rate']:.1%}, approved bad rate "
              f"{configured['bad_rate_approved']:.1%}")
        if recommended["accept_cutoff"] is None:
            print("  recommended: no cut-off satisfies the target bad rate - "
                  "revisit appetite, pricing or the model")
        else:
            print(f"  recommended: accept={recommended['accept_cutoff']} "
                  f"refer={recommended['refer_floor']} -> approval "
                  f"{recommended['approval_rate']:.1%}, approved bad rate "
                  f"{recommended['bad_rate_approved']:.1%}")
            swap = block["swap_set"]
            if swap["bad_rate_of_swap_set"] is None:
                print("  swap set   : none - recommended matches the "
                      "configured cut-off, no applicants move")
            else:
                print(f"  swap set   : {swap['direction']}, "
                      f"{swap['applicants_affected']} applicants "
                      f"({swap['share_of_population']:.1%}) move, "
                      f"their bad rate {swap['bad_rate_of_swap_set']:.1%}")
        thin = block["thin_file_note"]
        print(f"  thin file  : {thin['share_of_population']:.1%} of population, "
              f"bad rate {thin['bad_rate']:.1%}, segment Gini "
              f"{thin['segment_gini_out_of_time']:.3f} -> refer, do not "
              f"auto-decide\n")


if __name__ == "__main__":
    main()
