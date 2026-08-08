# Model card — APPLICATION_LR_V1

> A model card states what a model is for, what it was built on, how well it
> works and where it must not be used. It exists so that a validator, a
> regulator or a future maintainer can judge the model without reading the code.

## Status

| | |
|---|---|
| Model | `APPLICATION_LR_V1` v2.0.0 |
| Type | Segmented weight-of-evidence logistic-regression scorecard |
| **Governance status** | **PROTOTYPE — NOT APPROVED FOR PRODUCTION** |
| Independent validation | Not performed |
| Approval reference | None |

No decision produced by this model may be applied to a real borrower. It exists
to prove the pipeline and to support demonstration.

## Purpose

Estimates the probability that a retail credit applicant will default — defined
as 90+ days past due within a 12-month performance window — at the point of
application. The output is a probability, which the platform scales to each
tenant's own score range and grade bands.

The model does **not** decide anything. It produces one input to the decision
engine, alongside an independent affordability assessment and fraud, identity
and AML results, which are combined only by tenant policy.

## Training data

| | |
|---|---|
| Source | **Synthetic** (`model/synth.py`, seed 20260719) |
| Records | 12,000 (7,200 development / 2,400 validation / 2,400 out-of-time) |
| Bad rate | 18.6% |
| Thin-file share | ~23% |

The sample is generated from a latent risk process designed to resemble a
Zambian payroll and microfinance book: salaried and informal earners, a
meaningful thin-file population, and bureau fields unobserved for applicants
with no credit record. **Nothing about real borrowers can be inferred from
performance on this data.** Obtaining a real development sample under a
data-sharing agreement is business-case item ISS-09.

## Segmentation

| Segment | Population | Characteristics | Rationale |
|---|---|---|---|
| `BUREAU` | Applicants with a retrievable bureau record | 8 | Full information set |
| `THIN` | No bureau record at all | 5 | No bureau characteristic can act as a missing-information proxy |

Segmentation replaced a single model in which every bureau characteristic
shared one MISSING bin. That design counted the same "information absent"
effect once per characteristic, producing collinearity severe enough to flip
two coefficients to the wrong sign.

## Performance (synthetic data)

| Sample | Segment | n | AUC | Gini | KS | Brier |
|---|---|---|---|---|---|---|
| Development | all | 7,200 | 0.708 | 0.416 | 0.317 | 0.138 |
| Validation | all | 2,400 | 0.712 | 0.425 | 0.320 | 0.133 |
| Out-of-time | all | 2,400 | 0.694 | **0.388** | 0.302 | 0.131 |
| Out-of-time | bureau | 1,868 | 0.707 | 0.413 | 0.327 | 0.119 |
| Out-of-time | thin | 532 | 0.595 | **0.189** | 0.216 | 0.176 |

**The thin-file segment discriminates weakly (Gini 0.19) while carrying higher
risk (22.7% bad rate).** Score alone cannot responsibly decide those cases,
which is why policy rule `R-THN-01` refers them to manual underwriting. Better
thin-file discrimination requires alternative data — mobile-money and payroll
cash flow — not a better-tuned bureau model.

## Inputs

Twelve characteristics from the versioned feature store, each carrying full
metadata, a documented missing-value treatment and an observation window.
Missing values are never imputed as zero.

**Excluded by design:** `gender` is collected for fairness testing and
regulatory reporting only. It is flagged sensitive in the feature store,
excluded from the scoring feature set, and a test asserts it never reaches a
model.

## Development controls

- Supervised monotonic binning: adjacent bins merged until bad rates are
  monotonic and every bin holds at least 5% of the segment.
- Information-value screening at 0.02.
- Variance-inflation diagnostics per segment (maximum observed: 1.02).
- Sign-convention enforcement: every coefficient in a model of P(default) must
  be negative; offenders are removed and the model refitted automatically.
- Development, validation and out-of-time evaluation.

## Known limitations

1. Synthetic training data; no inference about real borrowers.
2. No reject inference — the sample contains no rejected applicants.
3. No fairness testing; the synthetic sample carries no protected attributes.
4. Calibration is only as good as the synthetic generator's base rate.
5. No behavioural, early-warning, collections or recovery models exist.
6. Cut-offs are calibrated separately (`model/calibrate_cutoffs.py`) and are
   tenant policy, not model output.

## Before production

Independent model validation, a real development sample under a data-sharing
agreement, reject inference, fairness testing against lawfully held
demographics, and Model Governance Committee approval — none of which has been
performed.
