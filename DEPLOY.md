# Deploying the demonstration

A hosted instance of the console and API, suitable for showing to a prospective
customer or investor. It runs from one container, needs no database server, and
carries only synthetic data.

**Read §5 before putting this on the public internet.** It is demonstration
infrastructure, not production infrastructure, and the difference is stated
plainly there.

---

## 1. Run it locally

```bash
cd prototype
python3 -m model.train_scorecard      # one-off, needs numpy
python3 -m model.calibrate_cutoffs
python3 -m demo.seed                  # populates ~86 decisions
python3 -m api.server                 # http://localhost:8080
```

Sign in with any demonstration account (they are listed on the sign-in screen):

| Username | Role | Sees |
|---|---|---|
| `admin` | Platform administrator | Everything, both views, batch, usage |
| `risk` | Credit risk | Decisions, monitoring, partners, batch |
| `underwriter` | Underwriter | Decisions only |
| `compliance` | Compliance | Decisions, audit, monitoring |
| `mfi` | Credit risk (microfinance tenant) | The other tenant's data only |

Default passwords follow the pattern `demo-<role>-2026` — `demo-admin-2026`,
`demo-risk-2026`, `demo-uw-2026`, `demo-comp-2026`, `demo-mfi-2026`.

## 2. Run it in a container

```bash
docker compose up -d --build
open http://localhost:8080
```

The image is built in two stages: the first trains the scorecard (the only step
needing numpy), the second carries the runtime, which has **no third-party
dependencies at all**. The container runs as an unprivileged user, declares a
health check, and keeps the ledger on a named volume so it survives a rebuild.

First start seeds the demonstration data automatically. To reset:

```bash
docker compose down -v && docker compose up -d
```

## 3. Put it on the internet

### 3.1 Configure it first

Create a `.env` file next to `docker-compose.yml`:

```bash
# Session signing key - without this, sessions break on every restart
DEMO_SECRET_KEY=$(openssl rand -hex 32)

# Replace the default accounts. Passwords are hashed at startup, never stored.
DEMO_USERS='[
  {"username":"tw","password":"<a long random password>","role":"PLATFORM_ADMIN","tenant":"ZAM-PAY","name":"Twaambo Hamusute"},
  {"username":"guest","password":"<a different long password>","role":"VIEWER","tenant":"ZAM-PAY","name":"Guest reviewer"}
]'

DEMO_BANNER="DEMONSTRATION - synthetic data, prototype model, not for production use"
DEMO_SESSION_TTL=28800
```

Setting `DEMO_USERS` removes the account list from the sign-in screen, so
credentials are no longer discoverable by visiting the page.

### 3.2 A small virtual machine with TLS

Any provider will do; the service needs roughly 1 vCPU and 1 GB of memory.
Caddy is the shortest path to automatic certificates:

```bash
# /etc/caddy/Caddyfile
demo.your-domain.com {
    reverse_proxy localhost:8080
    encode gzip
    header {
        Strict-Transport-Security "max-age=31536000"
        X-Content-Type-Options "nosniff"
        X-Frame-Options "DENY"
        Referrer-Policy "no-referrer"
    }
}
```

```bash
docker compose up -d          # bind to 127.0.0.1 only
sudo systemctl reload caddy
```

In `docker-compose.yml`, publish to the loopback interface so nothing reaches
the container except through the proxy:

```yaml
ports:
  - "127.0.0.1:8080:8080"
```

### 3.3 Indicative cost

| Item | Monthly |
|---|---|
| Small VM (1 vCPU, 1–2 GB) | US$5–12 |
| Domain name | ~US$1 |
| TLS certificate (Let's Encrypt) | Free |
| **Total** | **Under US$15** |

## 4. Running a demonstration

Roughly fifteen minutes, in this order.

1. **Sign in as `risk`.** The monitoring view opens on a populated portfolio:
   approval mix, score distribution against the model's development sample, PSI
   with its status, and the leading reason codes.

   *On the PSI reading:* the seeded portfolio is drawn from the same
   distributional shapes as the model's development sample, so it normally
   reads STABLE or WARNING. Amber is worth explaining rather than hiding — it
   means the live population differs modestly from the one the model was built
   on, which is an instruction to investigate, not to act. To show a genuine
   breach, re-seed with `python3 -m demo.seed --force --drift` (or
   `docker compose exec platform python -m demo.seed --force --drift`), which
   creates a younger, thinner-file population borrowing more relative to
   income. That is the most persuasive ninety seconds in the demonstration:
   the monitoring notices a population change that no one told it about.
2. **Decide Chanda Mwale** (`632084/37/1`, K20,000 over 18 months, income
   K12,800). Approved in milliseconds; show the priced offer, then the rule
   trace — fourteen rules evaluated, none decisive, and the three gates in
   sequence.
3. **Decide Bwalya Phiri** (`749078/36/8`, ask K85,000 on income K9,500). A
   **counteroffer**: the DSR ceiling caps the instalment, the inverse annuity
   converts that to K41,900, and the offer comes back priced. *This is the
   revenue story — business that would otherwise walk out the door.*
4. **Switch to the `mfi` account and decide Mutinta Banda** (`414328/41/3`).
   No bureau record, so the thin-file model scores her, confidence drops to
   medium, and policy refers her to an underwriter rather than deciding. Show
   that she receives no bureau-based reason codes, because the model scoring
   her contains none.
5. **Decide Joseph Tembo** (`702326/48/9`). Declined on arrears — a soft
   decline, with the exact rule that fired named in the trace.
6. **Run a batch** of 120 rows. Four are rejected with itemised reasons, and
   the totals reconcile.
7. **Open the partners view.** Availability, latency percentiles and circuit
   state per partner; then the usage statement, reconciled one-to-one with
   decisions.
8. **Sign in as `underwriter`** to show that role limits what is visible, and
   that the API refuses what the interface hides.

Two lines worth saying aloud, because they are what a credit committee is
actually assessing: *every decision carries the model, feature-set and policy
versions that produced it*, and *the audit trail is hash-chained, so tampering
with history is detectable*.

## 5. What this is not

Stated plainly, because a prospect will ask and the answer should be the same
one the documentation gives.

| Demonstration | Production requires |
|---|---|
| Password sign-in built on the standard library | OAuth 2.0/OIDC with MFA via an external identity provider |
| SQLite on a single container volume | PostgreSQL with row-level security, replication and point-in-time recovery |
| Partner simulators | Connectors built from confirmed partner specifications |
| Scorecard trained on synthetic data | A model developed on real history and independently validated |
| One instance, no redundancy | Multi-zone deployment, 99.95% availability, tested disaster recovery |
| Illustrative billing rates | Contractual rating under maker-checker control |
| No penetration test | External test with critical findings closed before go-live |

The production path is the twelve-month plan in the delivery roadmap. This
deployment exists to show that the decisioning core works and to let a customer
touch it — not to carry anyone's credit application.

## 6. Operations

| Task | Command |
|---|---|
| Logs | `docker compose logs -f platform` |
| Health | `curl -s localhost:8080/healthz` |
| Back up the ledger | `docker compose cp platform:/data/ledger.db ./backup-$(date +%F).db` |
| Re-seed | `docker compose down -v && docker compose up -d` |
| Update | `git pull && docker compose up -d --build` (the ledger volume survives) |
| Verify audit integrity | Sign in as `compliance`, or `GET /v1/audit/verify` |

Rotating `DEMO_SECRET_KEY` invalidates every active session, which is the
fastest way to sign everyone out.
