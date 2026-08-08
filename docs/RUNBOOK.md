# Operations runbook

Prototype-level procedures. Production runbooks are a Charter Phase 12
deliverable; this covers what an operator of *this* service needs.

## Starting and checking the service

```bash
python3 -m api.server            # listens on 127.0.0.1:8080
curl localhost:8080/healthz      # liveness, no credentials required
```

Open `http://localhost:8080/` for the underwriting console.

## Daily checks

| Check | Endpoint | Healthy | Action if not |
|---|---|---|---|
| Liveness | `GET /healthz` | `{"status":"ok"}` | Restart; check logs |
| Partner health | `GET /v1/partners/health` | Circuits `CLOSED`, availability > 99% | See "Partner outage" |
| Audit integrity | `GET /v1/audit/verify` | `intact: true` | **Escalate immediately** |
| Billing reconciliation | `GET /v1/usage` | `balanced: true` | See "Reconciliation break" |
| Model drift | `GET /v1/monitoring/summary` | `overall_status` `STABLE` | See "PSI breach" |

## Partner outage

**Symptom:** a partner's circuit reads `OPEN`; decisions carry
`PARTNER_DATA_UNAVAILABLE`.

The platform is already degrading per the tenant's configured policy — `refer`,
`partial` or `decline`. No emergency action is required to keep deciding.

1. Confirm which partner and for how long: `GET /v1/partners/health`.
2. Confirm the tenant's policy is the one they want during the outage. Changing
   it is a configuration change under maker-checker, not an operator action.
3. Notify affected tenants: referral volumes will rise, and underwriting
   queues with them.
4. The circuit half-opens automatically after the cooldown and closes on the
   first success. Do not restart the service to "clear" it — that discards the
   breaker's memory and re-hammers a partner that may still be down.

## PSI breach

**Symptom:** `population_stability.status` is `BREACH` (PSI ≥ 0.25).

A breach means the live applicant mix differs materially from the model's
development sample. It is an instruction to investigate, **not** to switch the
model off.

0. **Check the sample size first.** Compare `index` against `expected_noise`,
   which is what the index would read with no drift at all at this sample size
   and band count. If the two are close, the movement is sampling noise, not
   drift. Below 60 decisions the index is not reported at all and the status
   reads `UNKNOWN` — that is the metric declining to answer, and `UNKNOWN` is
   never to be read as healthy. A trustworthy ten-band reading needs roughly
   300 decisions in the window.
1. Read `largest_moves` in the monitoring pack — which score bands moved.
2. Rule out data faults first: a partner returning empty responses shifts the
   distribution exactly like a marketing change. Check partner health and the
   thin-file share (`decisions.by_segment`).
3. If the movement is genuine (campaign, seasonality, new channel), record it
   and monitor calibration as outcomes mature.
4. Model changes follow governance (BR-GOV-05). An operator never swaps a model.

## Reconciliation break

**Symptom:** `usage.reconciliation.balanced` is `false`.

- `unmetered` — decisions produced without a billing event. Revenue leakage.
- `orphan_meters` — billing events with no decision. A tenant would be
  overcharged; this is the more serious direction.

Do not correct the ledger. It is append-only by design, and adjustments
annotate rather than alter it. Raise an incident, identify the correlation ids
listed, and issue credits through the dispute path.

## Audit chain break

**Symptom:** `GET /v1/audit/verify` returns `intact: false` with a sequence
number.

Treat as a security incident. The chain only breaks if a historical row was
altered or deleted. Preserve the database file, do not restart the service, and
escalate to the security lead. The reported sequence number is where tampering
began.

## Retraining

```bash
python3 -m model.train_scorecard      # rebuilds the artefact
python3 -m model.calibrate_cutoffs    # re-derives cut-offs
python3 -m unittest discover -s tests
```

The golden-file test will **skip** with a message when the artefact hash
changes. That is deliberate: regenerating the golden file must be an act of
judgement accompanied by a model-version change, not a silent side effect.
