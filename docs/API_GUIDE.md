# API integration guide

Base URL `http://localhost:8080`. Machine-readable specification at
`GET /openapi.json`.

## Authentication

Every business endpoint requires an API key resolving to exactly one tenant:

```
X-API-Key: demo-key-payroll     # ZAM-PAY (payroll lender)
X-API-Key: demo-key-micro       # ZAM-MFI (microfinance)
```

Production uses OAuth 2.0/OIDC with mutual TLS for institutional partners
(IPSRS FR-API-02); the prototype demonstrates the contract, not the edge.

## Headers

| Header | Direction | Purpose |
|---|---|---|
| `X-API-Key` | request | Tenant authentication |
| `X-Correlation-Id` | both | Supplied or assigned; stamped on every audit and metering row |
| `Idempotency-Key` | request | Replaying a key returns the original decision and is not billed again |

## Submitting an application

```bash
curl -s localhost:8080/v1/applications/decision \
  -H 'X-API-Key: demo-key-payroll' \
  -H 'Idempotency-Key: app-0001' \
  -H 'Content-Type: application/json' \
  -d '{"application":{
        "national_id":"384756/61/1",
        "full_name":"Chanda Mwale",
        "date_of_birth":"1990-03-14",
        "application_date":"2026-07-19",
        "employer_code":"MOE-LSK-01",
        "requested_amount":20000,
        "tenor_months":18,
        "declared_monthly_income":12800,
        "declared_monthly_expenses":4200,
        "existing_monthly_debt_service":700,
        "dependants":1,
        "consent":{"credit_bureau_enquiry":true,"automated_decisioning":true}}}'
```

`consent.credit_bureau_enquiry` and `consent.automated_decisioning` are
mandatory. Without them the request is rejected at intake, no partner is
called, and nothing is billed.

## The decision contract

| Section | Contents |
|---|---|
| `outcome` | `APPROVE` · `DECLINE` · `REFER` · `INSUFFICIENT_INFORMATION` |
| `decline_type` | `hard` · `soft` · `score` · `affordability` · `partner_unavailable` |
| `reason_codes` | Governed codes with customer-appropriate text and category |
| `identifiers` | application, decision and correlation ids, tenant, product |
| `assessment` | score, PD, grade, data-quality status, confidence |
| `versions` | model, segment, feature set, policy, engine, reason-code library |
| `offer` | amount, tenor, rate, instalment, total repayable, cost of credit, counteroffer flag |
| `affordability` | verified income, expenses, DSR before/after, capacity, binding constraint |
| `trace` | every rule evaluated with its outcome, plus the decision gates |

Set `"audience": "internal"` in the request to receive internal reason text
instead of customer-facing wording.

## Outcomes to handle

- **APPROVE with `offer.is_counteroffer = true`** — the customer can afford
  something, but less than requested. Present the reduced amount; do not treat
  it as a decline.
- **REFER** — route to an underwriter. Common causes: thin credit file, score
  in the referral band, elevated fraud risk, or a partner outage under a
  `refer` degradation policy.
- **INSUFFICIENT_INFORMATION** — recoverable. `additional_information_required`
  states what is needed. This is not a credit decline and should not be
  presented as one.

## Errors

```json
{"error": {"code": "VALIDATION_FAILED",
           "message": "The application failed intake validation.",
           "details": {"errors": [{"field": "application.national_id",
                                   "error": "expected Zambian NRC format ######/##/#"}]}},
 "correlation_id": "COR-4A5F37335AC9"}
```

| Code | Status | Meaning |
|---|---|---|
| `VALIDATION_FAILED` | 400 | Field-level errors listed in `details.errors` |
| `MALFORMED_JSON` | 400 | Body is not a JSON object |
| `UNAUTHENTICATED` | 401 | Missing or unrecognised API key |
| `PRODUCT_NOT_FOUND` | 404 | Product not configured for this tenant |
| `DECISION_NOT_FOUND` | 404 | No such decision **for this tenant** |
| `RATE_LIMIT_EXCEEDED` | 429 | Per-tenant limit; retry next minute |
| `INTERNAL_ERROR` | 500 | Quote the correlation id when reporting |

Errors never contain stack traces or internal paths. The correlation id is the
thread an operator follows into the logs.

## Batch

```bash
curl -s localhost:8080/v1/batches -H 'X-API-Key: demo-key-payroll' \
  -H 'Content-Type: application/json' -d '{"use_sample": true, "rows": 40}'
```

Supply your own file as `{"csv": "row_id,national_id,...\n..."}`. Valid rows are
processed and invalid rows are itemised with the reason, so only failures need
resubmitting. `reconciled` must be `true`: submitted = processed + rejected.
