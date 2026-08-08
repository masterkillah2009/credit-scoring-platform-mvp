"""Synthetic development sample for the application scorecard.

The prototype has no access to real lending history (business-case ISS-09), so
this module generates a *labelled* synthetic sample whose structure mirrors a
Zambian payroll/microfinance book: salaried and informal applicants, a
meaningful share of thin-file customers, and default labels produced from a
latent risk process rather than from the features the model will later use
naively.

This is a stand-in for a real development sample and is labelled as such
everywhere it surfaces. It exists to prove the pipeline, not to assert
predictive performance on real borrowers.
"""
from __future__ import annotations

import numpy as np

# Latent-risk coefficients used to generate labels. The scorecard must
# *recover* something like these from data; they are not used at scoring time.
_TRUE = {
    "intercept": -3.95,          # targets a ~12-14% through-the-cycle bad rate
    "age": -0.022,               # older applicants slightly safer
    "employment_months": -0.0060,
    "existing_dsr": 2.40,
    "requested_to_income": 0.090,
    "worst_dpd": 0.0160,
    "enquiries_6m": 0.160,
    "history_months": -0.0090,
    "utilisation": 1.10,
    "prior_default": 1.30,
    "relationship_months": -0.0080,
    "thin_file": 0.50,
    "informal": 0.40,
}


def generate(n: int = 12_000, seed: int = 20260719) -> dict[str, np.ndarray]:
    """Return a dict of column arrays plus the binary ``default`` label."""
    rng = np.random.default_rng(seed)

    informal = rng.binomial(1, 0.32, n)                       # informal income
    age = np.clip(rng.normal(36, 10, n), 18, 72).round()
    employment_months = np.where(
        informal == 1,
        np.clip(rng.exponential(26, n), 0, 400).round(),
        np.clip(rng.gamma(2.4, 26, n), 0, 480).round(),
    )
    income = np.where(
        informal == 1,
        np.clip(rng.lognormal(7.9, 0.62, n), 900, 40_000),
        np.clip(rng.lognormal(8.5, 0.55, n), 1_800, 90_000),
    ).round(2)

    existing_debt_service = np.clip(
        income * rng.beta(1.6, 7.0, n), 0, income * 0.75).round(2)
    existing_dsr = existing_debt_service / income

    requested = np.clip(income * rng.gamma(2.2, 1.7, n), 500, 150_000).round(2)
    requested_to_income = requested / income
    tenor = rng.choice([6, 9, 12, 18, 24, 36, 48], n,
                       p=[0.08, 0.10, 0.26, 0.20, 0.18, 0.12, 0.06])

    # Thin file: no bureau record at all. Much commoner among informal earners.
    thin_file = rng.binomial(1, np.where(informal == 1, 0.42, 0.14))

    open_facilities = rng.poisson(1.7, n)
    history_months = np.clip(rng.gamma(2.0, 22, n), 0, 420).round()
    enquiries_6m = rng.poisson(1.25, n)
    worst_dpd = np.where(
        rng.binomial(1, 0.28, n) == 1,
        rng.choice([15, 30, 60, 90, 120, 180], n,
                   p=[0.34, 0.26, 0.17, 0.11, 0.07, 0.05]),
        0,
    )
    utilisation = np.clip(rng.beta(2.0, 3.4, n) * 1.25, 0, 1.6)
    prior_default = rng.binomial(
        1, np.clip(0.05 + 0.0011 * worst_dpd, 0, 0.6), n)
    relationship_months = np.clip(rng.gamma(1.6, 15, n), 0, 300).round()
    dependants = rng.poisson(2.1, n)

    # Bureau fields are unobserved for thin-file applicants. The underlying
    # ("true") values still drive the latent default process below - which is
    # exactly why thin files are riskier to underwrite: information is missing,
    # not absent from reality.
    nan = float("nan")
    open_facilities_obs = np.where(thin_file == 1, nan, open_facilities)
    history_months_obs = np.where(thin_file == 1, nan, history_months)
    enquiries_6m_obs = np.where(thin_file == 1, nan, enquiries_6m)
    worst_dpd_obs = np.where(thin_file == 1, nan, worst_dpd)
    utilisation_obs = np.where(thin_file == 1, nan, utilisation)
    prior_default_obs = np.where(thin_file == 1, nan, prior_default)

    # Latent default process uses the *true* (including unobserved) risk
    logit = (
        _TRUE["intercept"]
        + _TRUE["age"] * (age - 36)
        + _TRUE["employment_months"] * (employment_months - 60)
        + _TRUE["existing_dsr"] * existing_dsr
        + _TRUE["requested_to_income"] * requested_to_income
        + _TRUE["worst_dpd"] * worst_dpd
        + _TRUE["enquiries_6m"] * enquiries_6m
        + _TRUE["history_months"] * (history_months - 48)
        + _TRUE["utilisation"] * utilisation
        + _TRUE["prior_default"] * prior_default
        + _TRUE["relationship_months"] * (relationship_months - 24)
        + _TRUE["thin_file"] * thin_file
        + _TRUE["informal"] * informal
        + rng.normal(0, 0.35, n)          # unexplained heterogeneity
    )
    probability = 1.0 / (1.0 + np.exp(-logit))
    default = rng.binomial(1, probability)

    return {
        "age_years": age,
        "employment_months": employment_months,
        "existing_dsr": existing_dsr,
        "requested_to_income": requested_to_income,
        "bureau_worst_dpd": worst_dpd_obs,
        "bureau_open_facilities": open_facilities_obs,
        "bureau_enquiries_6m": enquiries_6m_obs,
        "credit_history_months": history_months_obs,
        "revolving_utilisation": utilisation_obs,
        "prior_default": prior_default_obs,
        "relationship_months": relationship_months,
        # carried for context / affordability demos, not scorecard inputs
        "monthly_income": income,
        "existing_monthly_debt_service": existing_debt_service,
        "requested_amount": requested,
        "tenor_months": tenor,
        "dependants": dependants,
        "informal": informal,
        "thin_file": thin_file,
        "default": default,
    }


def split(data: dict[str, np.ndarray], *, seed: int = 7,
          train: float = 0.6, valid: float = 0.2
          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Development / validation / out-of-time style index split."""
    n = len(next(iter(data.values())))
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    a = int(n * train)
    b = int(n * (train + valid))
    return order[:a], order[a:b], order[b:]
