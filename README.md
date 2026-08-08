# Retail Lending Scoring Platform — Reference Prototype

A runnable reference implementation of the core decisioning path specified in
the platform documentation stack (HLD v0.1, BRD v0.1, IPSRS v0.1, Programme
Charter v0.1). Its purpose is to prove the specification is implementable, give
you something to demonstrate to anchor customers, and let a development firm
estimate the production build against working code rather than prose.

**This is a prototype, not a product.** The scorecard is trained on synthetic
data and is labelled `PROTOTYPE - NOT APPROVED FOR PRODUCTION` inside the
artefact itself. No production personal data, real partner API, or approved
model is involved.

---

## Requirements

Python 3.10+. **The runtime has no third-party dependencies** — it uses only the
standard library, so it runs anywhere without installation. `numpy` is needed
only to retrain the scorecard offline.

## Quick start

```bash
cd prototype

# 1. Train the segmented scorecard (writes artefacts/, needs numpy)
python3 -m model.train_scorecard

# 2. Derive cut-offs from the score distribution and risk appetite
python3 -m model.calibrate_cutoffs

# 3. Score three illustrative applicants across two tenants
python3 demo_phase1.py

# 4. Run the decision engine across six outcome scenarios
python3 demo_phase2.py
python3 demo_phase2.py --json     # full decision contract for one case

# 5. Exercise the API: partner outage, idempotency, audit, metering
python3 demo_phase3.py

# 6. Or run the API as a service
python3 -m api.server        # http://localhost:8080, GET /openapi.json

# 7. Batch scoring, monitoring pack and documentation
python3 demo_phase4.py
python3 demo_phase4.py --serve   # leaves the console running

# 8. Regenerate the traceability matrix from the source
python3 -m tools.traceability

# 9. Run the test suite (192 tests, one process per module)
python3 run_tests.py
python3 run_tests.py phase3      # or a single module
```

---

## Corrections applied after the first build

Three findings from reviewing the first working version were fixed at the root
rather than patched. All three now have tests that fail if the defect returns.

### 1. Collinearity from a shared missing bin → segmented scorecards

**Finding.** Every bureau characteristic shared one `MISSING` bin, so the same
"information is absent" effect was counted once per bureau characteristic. Two
coefficients came out positive — the wrong sign for a model of P(default) —
meaning the model was crediting risk factors as protective.

**Correction.** The population is split into two segments with genuinely
different information sets (`BUREAU`, `THIN`), each with its own fitted
scorecard. The `THIN` model contains **no bureau characteristic at all**, so no
characteristic can act as a missing-information proxy. Variance-inflation
diagnostics are now computed and reported per segment.

**Result.** Maximum VIF fell to **1.02** (from values that produced sign
reversals); no coefficient requires removal; overall out-of-time Gini rose from
0.373 to **0.388**. Tests: `test_no_characteristic_is_collinear`,
`test_all_coefficients_carry_the_expected_sign`,
`test_thin_segment_contains_no_bureau_characteristic`.

### 2. False reason codes for thin-file applicants → structural prevention

**Finding.** An applicant with no credit record received the reason
"Recent missed or late payments were found on your credit record" — false, and a
consumer-protection exposure.

**Correction.** Two layers. Structurally, a thin-file applicant is scored by a
model that has no bureau characteristic, so such a reason cannot be
constructed. Defensively, the engine also suppresses any reason drawn from a
`MISSING` bin, and suppresses immaterial point losses so a strong applicant is
not handed a list of trivial adverse factors. Thin-file scores never report
`HIGH` confidence, because that segment's discrimination is materially weaker.

**Result.** The thin-file applicant now receives only reasons that are true of
her — relationship tenure, age band, and the honest statement that no credit
record was found. Tests: `test_thin_file_gets_no_bureau_based_reasons`,
`test_thin_file_score_never_reports_high_confidence`.

### 3. Hand-set cut-offs → calibrated from the score distribution

**Finding.** Cut-offs were chosen by eye from a PD mapping. At the original
values a 17% PD applicant was auto-approved.

**Correction.** `model/calibrate_cutoffs.py` derives cut-offs from the
validation and out-of-time score distribution (the development sample is
excluded) against each tenant's stated risk appetite, and reports bad rate by
score band, the full approval/bad-rate trade-off curve, and a swap-set analysis
against the configured value. Configured cut-offs now cite the calibration, and
a test fails if they drift from it.

The first version of the recommendation rule was itself wrong: it proposed
approving **100% of MFI applicants**, because that population's overall bad rate
happened to sit just under the 18% appetite — a cut-off that satisfies the
constraint while using the model for nothing. The rule now also requires the
**marginal** business admitted at the cut-off to be within appetite, which is
the economically meaningful test.

**Result.**

| Tenant | Configured (was) | Calibrated | Approval | Approved bad rate |
|---|---|---|---|---|
| ZAM-PAY | 650 (by eye) | **649** | 22.5% | 6.5% |
| ZAM-MFI | 480 (by eye) | **447** | 64.0% | 11.1% |

The payroll cut-off was essentially confirmed; the MFI cut-off was materially
too conservative — 26% of applicants, with a 15.5% bad rate, could be approved
responsibly within an 18% appetite. That is a real inclusion finding, produced
by the calibration rather than by assertion. Tests:
`test_configured_cutoffs_match_the_calibration`,
`test_marginal_risk_is_within_appetite_at_the_cutoff`.

---

## Build phases

| Phase | Scope | Status |
|---|---|---|
| **1** | Configuration, feature store, segmented scorecard training, cut-off calibration, runtime scoring engine, reason codes, test suite | **Complete** (v2 — three review findings corrected, see above) |
| **2** | Affordability engine, declarative decision/policy engine, full decision contract with rule trace | **Complete** |
| **3** | REST API (stdlib HTTP), partner simulators with degradation, immutable audit + metering ledger (SQLite), OpenAPI spec | **Complete** |
| **4** | Underwriting console, batch scoring, monitoring metrics, documentation pack | **Complete** |
| **5** | Hosted demonstration: sign-in with roles, seeded scenarios, container, deployment guide | **Complete** |

---

## What Phase 1 delivers

### `core/config.py` — tenant and product configuration
Two tenants are configured deliberately to prove configuration-over-code
(BRD BR-TEN-03, BR-SCR-02):

| | ZAM-PAY (payroll lender) | ZAM-MFI (microfinance) |
|---|---|---|
| Score scale | 300–850, base 660 @ 15:1, PDO 20 | 0–1000, base 500 @ 10:1, PDO 40 |
| Accept cut-off | 649 (calibrated, 8% appetite) | 447 (calibrated, 18% appetite) |
| DSR ceiling | 40% | 50% |
| Isolation tier | shared schema | dedicated schema |
| Partner-outage policy | refer | score on partial data |

Both consume **the same PD model**. Nothing about a tenant's presentation,
policy or risk appetite is compiled into the engine.

### `core/features.py` — versioned feature store
- 13 feature definitions carrying the full IPSRS FR-FST-02 metadata contract;
  activation fails if any metadata is absent.
- **Missing values are `None`, never zero** (FR-FST-03). Each feature declares
  its treatment: `dedicated_bin`, `policy_referral` or `block`.
- Data-quality verdict (`OK` / `DEGRADED` / `BLOCK`) returned with every
  computation and carried into the score response.
- `gender` is defined but flagged `sensitive` and excluded from
  `SCORING_FEATURES` — held for fairness testing only (FR-FST-06).
- The feature-set version is a **content hash** of the definitions, so any
  change to a formula or window changes the version stamped on every score.

### `model/train_scorecard.py` — development pipeline
Weight-of-evidence binning → logistic regression → versioned JSON artefact.
Implements the governance checks the specification demands:

- **supervised monotonic binning** — adjacent bins are merged until bad rates
  are monotonic and every bin holds ≥ 5% of the sample;
- **information value** screening (characteristics below IV 0.02 are dropped);
- **sign-convention enforcement** — WoE is oriented so higher = safer, so every
  coefficient in a model of P(default) must be negative. Positive coefficients
  indicate collinearity; the offender is removed and the model refitted,
  automatically, with the action recorded in the artefact's `warnings`;
- **missing-bin neutralisation** — bureau characteristics have their MISSING bin
  set to neutral WoE so the thin-file effect is not counted five times;
- development / validation / out-of-time evaluation with AUC, Gini, KS, Brier
  and Cox calibration intercept and slope.

Current performance on the synthetic sample (`seed 20260719`, 12,000 records,
18.6% bad rate, 23% thin file):

| Sample | Segment | n | AUC | Gini | KS | Brier |
|---|---|---|---|---|---|---|
| Development | all | 7,200 | 0.708 | 0.416 | 0.317 | 0.138 |
| Validation | all | 2,400 | 0.712 | 0.425 | 0.320 | 0.133 |
| Out-of-time | all | 2,400 | 0.694 | **0.388** | 0.302 | 0.131 |
| Out-of-time | bureau | 1,868 | 0.707 | 0.413 | 0.327 | 0.119 |
| Out-of-time | thin | 532 | 0.595 | 0.189 | 0.216 | 0.176 |

Out-of-time Gini of 0.388 clears the Charter's KPI-05 target (≥ 0.35) — on
synthetic data, which proves the pipeline, not the market.

The segment split carries a finding worth acting on: **thin-file discrimination
is weak (Gini 0.19) while thin-file risk is high (22.7% bad rate)**. Scoring
alone cannot responsibly decide those cases, which is precisely why policy rule
`R-THN-01` refers them to manual underwriting, and why alternative data
(mobile-money and payroll cash flow) is the roadmap answer rather than a
better-tuned bureau model.

### `core/scorecard.py` — runtime scoring engine
- Loads an approved artefact and **never re-fits at runtime**.
- Returns the full FR-SCO-01 contract: PD, scaled score, grade, reason codes,
  model version, feature-set version, timestamp, DQ status, confidence.
- **Deterministic**: identical artefact plus identical values always produce
  identical output, protected by a golden-file regression test that refuses to
  pass silently if the artefact hash changes.
- Standard points arithmetic: `score = offset − factor × log_odds`, with
  `factor = PDO / ln 2`. A verified property test confirms that doubling the
  good:bad odds moves the score by exactly one PDO.
- Per-characteristic point attribution drives reason codes.

### `core/reason_codes.py` — governed explanation library
33 codes, each with a customer-facing sentence, an internal explanation and a
category. Two fairness guards are enforced in the engine, not left to callers:

1. a characteristic scored from its **MISSING bin never generates a substantive
   reason** — an applicant with no credit record is never told that arrears were
   found on it;
2. **immaterial point losses are suppressed**, so a strong applicant is not
   handed a list of trivial "adverse factors".

Sanctions and fraud declines deliberately return non-specific customer text
(no tipping off) while retaining the precise internal reason.

### `tests/test_phase1.py` — 38 tests, all passing
Grouped by the requirement they evidence: feature store (FR-FST), scoring
(FR-SCO), reason codes (FR-EXP), tenant isolation (BR-TEN) and model governance
(BR-GOV). Notable tests:

- missing values are `None`, never `0`;
- sensitive attributes never reach the scoring vector;
- scoring is deterministic across 25 repetitions and stable against a golden file;
- one PD renders as different scores per tenant but the PD is identical;
- score movement equals PDO when odds double;
- thin-file applicants receive no bureau-based reason codes;
- every emitted code exists in the governed library and reads as a sentence;
- grade bands tile the whole scale with no gaps or overlaps;
- the artefact's feature-set version matches the live feature store (guards
  against scoring on stale definitions).

---

## Demo output (abridged)

```
### Chanda - salaried teacher, clean file
  data quality : OK
  ZAM-PAY  PD= 6.05%  score=661  grade=B  cutoff=650 -> PASS
  ZAM-MFI  PD= 6.05%  score=525  grade=C  cutoff=480 -> PASS

### Mutinta - market trader, no bureau record
  data quality : DEGRADED  (thin file)
  ZAM-PAY  PD=17.37%  score=627  grade=C  cutoff=650 -> REFER
           reason: [NO_BUREAU_RECORD] We could not find a credit record for you...
  ZAM-MFI  PD=17.37%  score=457  grade=E  cutoff=480 -> REFER

### Joseph - salaried, recent arrears and high utilisation
  ZAM-PAY  PD=51.81%  score=580  grade=D  cutoff=650 -> FAIL
           reason: [RECENT_DELINQUENCY] Recent missed or late payments were found...
           reason: [HIGH_EXISTING_DEBT] Your existing loan repayments are high...
```

The three personas trace directly to IPSRS journeys JRN-01 to JRN-03.

---

## What Phase 2 delivers

### `core/money.py` — exact decimal arithmetic
IPSRS CST-06 forbids binary floating point for money, so every amount that
could reach a loan agreement passes through this module. Floats arriving from
JSON are converted via `str`, so `0.1` becomes exactly `0.10`. Two rounding
policies are applied deliberately:

- **instalments round up** — rounding a repayment down leaves a residual
  balance the schedule never collects;
- **principals round down** — the inverse annuity converts affordability
  capacity into a loan amount, and rounding up there would offer a facility the
  assessment has not approved.

`money(None)` raises rather than returning zero. A missing amount is a fact to
handle, never a zero to assume.

### `core/affordability.py` — repayment capacity
Answers "can this customer service this facility?" **without ever seeing the
credit score** (BR-AFF-05, verified by test). Capacity is the binding minimum
of two independent tests:

| Test | Question |
|---|---|
| Debt-service ratio | Would total debt service exceed the tenant's DSR ceiling? |
| Cash flow | What remains after verified income, modelled expenses, existing commitments and the required disposable-income buffer? |

The result reports which test bound, so an underwriter can see *why* capacity is
what it is. Income is haircut by evidential level — declared 50%, documented
75%, payroll or transaction verified 100% (tenant-configurable) — and the level
is derived from the partner data actually obtained, never self-asserted:
absent evidence means `DECLARED`. Expenses take the greater of declared costs
and a policy floor plus a per-dependant allowance, so an applicant understating
living costs cannot manufacture capacity.

### `core/decision.py` — declarative decision and policy engine
The only component that combines model output with policy. Rules are **data,
not code**: conditions are evaluated by a dispatch table over a read-only
context, with no `eval` anywhere (there is a test asserting this), so a tenant
can change policy without a release.

Three gates run in order, and **every rule is recorded whether it fired or
not**:

1. **Policy rules**, in precedence order: `hard_decline` → `insufficient` →
   `soft_decline` → `refer`. Verification failures deliberately outrank credit
   declines — if identity cannot be established there is no lawful basis to
   record an adverse credit decision about that person.
2. **Score band**: accept, referral band, or below floor.
3. **Affordability**, which can produce a **counteroffer** rather than a
   decline: a customer who can afford something, just not what they asked for,
   is offered the amount they can service, rounded down to a saleable
   increment.

Outcomes: approve, approve-with-counteroffer, decline (hard / soft / score /
affordability), refer, insufficient information. Every outcome carries reason
codes from the governed library, and every decline carries at least one
specific reason (tested).

### `core/pipeline.py` — orchestration
`validate → features → score → affordability → decide`. Scoring and
affordability run independently; only the decision engine sees both. A blocking
data-quality verdict stops the pipeline and returns insufficient information
rather than scoring on imputed values.

### Demonstration output

```
2. Counteroffer - creditworthy but asked for too much
  -> APPROVED (COUNTEROFFER)
     score 666 grade B PD 5.18% segment BUREAU confidence HIGH
     income  declared 9500.00 -> verified 9500.00 (PAYROLL_VERIFIED, haircut 1.00)
     capacity instalment 2300.00 (debt_service_ratio binding), requested 4939.97
     offer   ZMW 41900.00 over 24m at 28% -> instalment 2299.83,
             total repayable 55195.92, cost of credit 13924.42
     (requested 90000.00, offered 41900.00)
     reason  [COUNTEROFFER_REDUCED_AMOUNT] We can offer a smaller amount than you requested...
     trace   14 rules evaluated, 0 matched
               gate policy_rules: PASS - 14 rules evaluated, none decisive
               gate score_band: PASS - score 666 >= accept cut-off 649
               gate affordability: COUNTEROFFER - requested 90000.00 exceeds capacity
```

The decision contract returned by the API (Phase 3) carries the outcome, reason
codes, identifiers, assessment, **model / feature-set / policy / engine /
reason-code-library versions**, the priced offer, the full affordability object,
timestamps and expiry, and the complete rule and gate trace.

---

## What Phase 3 delivers

### `api/server.py` — REST API
Ten endpoints on the standard library, so the service starts with no
installation:

| Endpoint | Purpose |
|---|---|
| `POST /v1/applications/decision` | Submit an application, receive the decision contract |
| `POST /v1/prequalification` | Indicative eligibility with no bureau enquiry |
| `GET /v1/decisions/{id}` | Retrieve a stored decision |
| `GET /v1/applications/{id}/decisions` | Decision history |
| `GET /v1/audit/{correlation_id}` | Reconstruct the audit trail |
| `GET /v1/audit/verify` | Tamper-evidence check on the chain |
| `GET /v1/usage` | Metered usage and reconciliation |
| `GET /v1/partners/health` | Availability, latency percentiles, circuit state |
| `GET /openapi.json` | Machine-readable specification |
| `GET /healthz` | Liveness |

Cross-cutting behaviour is applied once, not per endpoint: API-key
authentication resolved to exactly one tenant, idempotency keys, correlation
ids echoed in the response header and stamped on every ledger row, per-tenant
rate limiting, and standardised errors carrying a machine-readable code and the
correlation id — never a stack trace or an internal path (there is a test for
that).

**Intake validation runs before any partner is called** — NRC format, ISO
dates, positive amounts, and explicit consent for bureau enquiry and automated
decisioning. A rejected application costs nothing: a test asserts zero partner
calls and zero metered events.

**Tenant isolation is enforced at the boundary.** A decision belonging to
another tenant returns exactly the same `404 DECISION_NOT_FOUND` as one that
does not exist — because existence is itself information.

### `partners/` — simulators and the connector framework
IPSRS CST-02 forbids inventing partner specifications, so the simulators make
no claim to mirror any real provider. They exist to exercise the platform's own
behaviour, and they do so deterministically: the same national ID always
returns the same bureau file, and failure injection is explicit rather than
random.

The connector framework applies one resilience contract to every partner:
per-partner timeout budget, bounded retries with exponential backoff, a circuit
breaker with half-open probing, health metrics, and **parallel retrieval** so
the budget is the slowest partner rather than the sum. A failed call yields
`ok=False` with a `None` payload — **a connector never invents data**; the
tenant's degradation policy decides what happens next.

Degradation in the demo, with the bureau down:

```
attempt 1: REFER   bureau ok=False attempts=2 circuit=CLOSED
attempt 2: REFER   bureau ok=False attempts=2 circuit=CLOSED
attempt 3: REFER   bureau ok=False attempts=2 circuit=OPEN
attempt 4: REFER   bureau ok=False attempts=0 circuit=OPEN   <- short-circuited
reasons: ['THIN_CREDIT_FILE', 'PARTNER_DATA_UNAVAILABLE']
gate:    unavailable: bureau; tenant policy 'refer' applied
```

The payroll lender refers; the microfinance tenant, configured for `partial`,
scores on what is available and says so. Same code, different configuration.

### `core/ledger.py` — immutable audit and metering
SQLite (schema portable to PostgreSQL), with two load-bearing properties:

- **Tamper evidence.** Audit rows form a SHA-256 hash chain *per tenant*.
  Altering any historical row breaks every subsequent link. A test modifies a
  row directly in the database and asserts that `verify_chain` reports the
  break at the exact sequence number.
- **Reconciliation.** Every metering row carries the correlation id of the work
  that produced it, so an invoice line traces back to a decision — and any
  decision without a meter, or meter without a decision, is detectable.
  Replayed idempotency keys are not billed twice (tested).

---

## What Phase 4 delivers

### `ui/index.html` — underwriting and decision console
A single self-contained page served at `/`. No build step, no framework, no
CDN — a test asserts there is no external reference, because the console must
run offline and on a low-bandwidth link, the same constraint the agent-capture
journey carries (NFR-10).

Four views: **Decision** (submit an application, see the outcome badge, reason
codes, priced offer, affordability breakdown and the full rule trace with fired
rules highlighted), **Recent decisions** (the referral queue), **Monitoring**
(KPI tiles, PSI band table, calibration status, top reason codes) and
**Partners** (health, latency percentiles, circuit state, metered usage).
Switching tenant in the header re-runs everything under that tenant's policy.

### `core/batch.py` — batch portfolio scoring
Per-row validation, itemised rejects, and a reconciliation assertion:
`submitted = processed + rejected`, or the batch failed however many decisions
it produced. Valid rows process despite invalid neighbours, so a tenant
resubmits only what failed.

```
submitted 120  processed 116  rejected 4  reconciled=True
outcomes: APPROVE=18  DECLINE=25  INSUFFICIENT_INFORMATION=45  REFER=28
rejected rows:
  R0025: national_id: required
  R0049: requested_amount: not a number ('not-a-number')
  R0073: consent.credit_bureau_enquiry: required before processing
  R0097: tenor_months: not an integer ('twelve')
```

A test asserts that a batch score and an API score for identical input are
identical — they run the same pipeline, so this holds by construction.

### `core/monitoring.py` — model and portfolio monitoring
PSI on the score distribution, CSI per characteristic, calibration where
outcomes exist, and approval/decline/referral/counteroffer rates by grade,
segment and data-quality status.

Every metric carries a **status against disclosed thresholds** (PSI: stable
below 0.10, warning to 0.25, breach above) so a dashboard can show red, amber
or green rather than expecting the reader to know what a PSI of 0.19 means. The
reference distribution is the scorecard's own development sample — stored as
PDs in the artefact, so one reference serves every tenant's score scale.

Two deliberate choices:

- **An absent metric reads `UNKNOWN`, never a healthy zero.** Calibration
  requires a matured performance window, so it says so rather than implying
  the model is well calibrated.
- **A breach is an instruction to investigate, not to switch the model off.**
  The runbook's first step is to rule out a data fault, because a partner
  returning empty responses shifts the distribution exactly like a marketing
  change would.

### `docs/` — documentation pack

| Document | Contents |
|---|---|
| `MODEL_CARD.md` | Purpose, training data, segmentation, performance by segment, excluded attributes, development controls, known limitations, what is required before production |
| `RUNBOOK.md` | Daily checks with healthy values, and procedures for partner outage, PSI breach, reconciliation break, audit-chain break and retraining |
| `API_GUIDE.md` | Authentication, headers, the decision contract, outcomes an integrator must handle, error codes, batch submission |
| `TRACEABILITY.md` | **Generated from the source** by `tools/traceability.py` |

The traceability matrix is generated rather than maintained, because a
hand-written matrix drifts from the code within a sprint. It scans every
requirement identifier cited in implementation and tests, and classifies each:

```
53 identifiers: 27 verified, 22 implemented but untested, 4 test-only
```

`python3 -m tools.traceability --check` exits non-zero when code claims a
requirement that no test names — suitable for a CI gate. A test also asserts
the committed matrix matches what the source says today.

---

## What Phase 5 delivers — the hosted demonstration

The prototype now runs as a service a prospect can be shown, rather than one
that has to be set up on their laptop. See **`DEPLOY.md`** for the full guide,
including a fifteen-minute demonstration script.

```bash
docker compose up -d --build     # http://localhost:8080
```

### `core/auth.py` — sign-in, roles and sessions
Demonstration-grade authentication on the standard library:

- passwords stored as **PBKDF2-HMAC-SHA256** with a per-user salt;
- **stateless HMAC-signed session tokens** with expiry and revocation;
- **constant-time comparison**, and a hash performed even for unknown users so
  response timing cannot enumerate valid usernames;
- **lockout** after five failed attempts;
- **six roles** mapped to permissions, checked before a handler runs rather
  than inside it, and every user bound to exactly one tenant;
- tokens carried in an `Authorization` header rather than a cookie, so the
  demonstration has **no cross-site request forgery surface at all**.

Failed sign-in returns one message for every cause — wrong password, unknown
user, locked account — because distinguishing them tells an attacker which
usernames exist. There is a test for that.

The console hides what a role may not do, and the API refuses it independently:
an underwriter sees only decisions, and `GET /v1/usage` returns `403` for them
whatever the interface shows.

### `demo/seed.py` — a repeatable walkthrough
Four named scenarios across two tenants, plus ~90 background decisions so the
monitoring view and queue open with content:

| Applicant | Tenant | Outcome |
|---|---|---|
| Chanda Mwale — salaried, clean file | ZAM-PAY | **Approved**, priced offer |
| Bwalya Phiri — asked for too much | ZAM-PAY | **Counteroffer** at K41,900 |
| Mutinta Banda — no bureau record | ZAM-MFI | **Referred**, thin-file model |
| Joseph Tembo — arrears on file | ZAM-PAY | **Declined**, soft |

The national registration numbers were chosen so the deterministic partner
simulators return the intended credit profile for each. Nothing is overridden
or faked — the scenarios are found, not forced.

### Container and deployment
Two-stage build: the first trains the scorecard (the only step needing numpy),
the second carries a runtime with **no third-party dependencies at all**. Runs
as an unprivileged user, declares a health check, keeps the ledger on a volume
that survives rebuilds, and seeds itself on first start. Total hosting cost for
a public demonstration is under US$15 a month.

### `run_tests.py` — one process per module
The suite runs each module in its own interpreter. That is deliberate: the
tests exercise module-level state — simulator failure injection, the rate
limiter, the artefact cache, the shared auth service — and sharing a process
lets one module's residue affect another's, producing results that depend on
the order things ran in.

Building it surfaced two defects worth recording. Capturing child output
through a pipe deadlocked once the kernel buffer filled — it looked exactly
like a hanging test suite. And password hashing at production strength (240,000
iterations) made the suite slow enough that people would stop running it, so
the work factor is now configurable, lowered only in tests, with a test
asserting the shipped default is unchanged.

---

## Deliberate limitations

| Prototype | Production requirement |
|---|---|
| Synthetic training data | Anchor-tenant history under a data-sharing agreement (business case ISS-09) |
| Configuration in Python dataclasses | Database-backed config with maker-checker approval (FR-ADM-04) |
| No persistence yet | PostgreSQL with row-level security; immutable audit trail (Phase 3) |
| No partner integrations | Connector framework built only against confirmed specifications (CST-02) |
| No independent validation | Independent model validation is a go-live gate (BR-GOV-05) |
| Demonstration sign-in (PBKDF2, HMAC sessions) | OAuth 2.0/OIDC with MFA via an external identity provider (FR-ADM-02) |
| Single scorecard | Behavioural, EWS, collections, recovery, fraud and limit models (Charter Phases 10–11) |

## Repository layout

```
prototype/
├── core/
│   ├── config.py          tenants, products, policy rules, score scales
│   ├── features.py        versioned feature store + data-quality verdict
│   ├── scorecard.py       runtime scoring engine (deterministic)
│   ├── reason_codes.py    governed explanation library
│   ├── money.py           exact decimal arithmetic + annuity maths
│   ├── affordability.py   DSR and cash-flow capacity, verification haircuts
│   ├── decision.py        declarative policy engine + decision contract
│   ├── pipeline.py        retrieve -> features -> score -> affordability -> decide
│   ├── ledger.py          hash-chained audit trail + metering ledger
│   ├── batch.py           batch scoring with rejects and reconciliation
│   ├── monitoring.py      PSI, CSI, calibration, approval metrics
│   └── auth.py            passwords, sessions, roles and permissions
├── api/
│   └── server.py          REST API, routing, validation, OpenAPI document
├── partners/
│   ├── simulators.py      deterministic partner stand-ins + failure injection
│   └── connector.py       timeout, retry, circuit breaker, parallel retrieval
├── model/
│   ├── synth.py             synthetic development sample generator
│   ├── train_scorecard.py   segmented WoE + logistic regression + artefact
│   └── calibrate_cutoffs.py cut-offs from score distribution + swap sets
├── artefacts/             scorecard, development report, feature dictionary
├── docs/                  model card, runbook, API guide, traceability matrix
├── tools/
│   └── traceability.py    generates the traceability matrix from source
├── ui/
│   └── index.html         underwriting and decision console (no dependencies)
├── demo/
│   └── seed.py            named scenarios + background portfolio
├── Dockerfile             two-stage build, non-root, healthcheck
├── docker-compose.yml     one-command deployment
├── DEPLOY.md              hosting, configuration, demonstration script
├── run_tests.py           per-module test runner
├── tests/                 192 stdlib tests + golden score file
├── demo_phase1.py         scoring demonstration
├── demo_phase2.py         decisioning demonstration (six outcome scenarios)
├── demo_phase3.py         API, partner outage, audit and metering over HTTP
└── demo_phase4.py         batch scoring, monitoring pack, documentation
```
