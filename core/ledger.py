"""Immutable audit trail and usage-metering ledger.

Implements IPSRS FR-ADM-05 (immutable, tamper-evident audit of every access,
configuration change, score, decision and billing event) and FR-BIL-01..02
(a metering event written at source for every billable action, reconciling 1:1
with the decision record).

Two properties are load-bearing:

**Tamper evidence.** Audit rows form a hash chain *per tenant*: each row stores
the SHA-256 of its own content plus the previous row's hash. Altering or
deleting any historical row breaks every subsequent link, and ``verify_chain``
detects it. A per-tenant chain also means one tenant's activity cannot be
inferred from another's chain length.

**Reconciliation.** Metering rows carry the correlation identifier of the work
that produced them, so an invoice line can always be traced back to the exact
decision - and a decision that produced no metering event, or a metering event
with no decision, is detectable (``reconcile``). Billing disputes are resolved
against this ledger, and adjustments annotate it rather than altering it.

SQLite is used because it is in the standard library and needs no server. The
schema is deliberately portable to PostgreSQL, which is the production target.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sqlite3
import threading
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional

#: Ledger location. Overridable so a container can place it on a writable
#: volume that survives image replacement (DEMO_LEDGER).
DEFAULT_PATH = pathlib.Path(
    os.environ.get("DEMO_LEDGER")
    or (pathlib.Path(__file__).resolve().parents[1] / "ledger.db"))

#: Billable event types and their unit price. In production these live in the
#: tenant's contract under maker-checker control (IPSRS FR-BIL-02); here they
#: are illustrative and clearly labelled as such.
BILLABLE = {
    "PREQUALIFICATION": Decimal("0.05"),
    "APPLICATION_DECISION": Decimal("0.35"),
    "RESCORE": Decimal("0.20"),
    "BATCH_SCORE": Decimal("0.08"),
    "EXTERNAL_DATA_CALL": Decimal("0.00"),   # pass-through, priced per partner
    "MANUAL_REVIEW": Decimal("0.50"),
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_events (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_code    TEXT    NOT NULL,
    sequence       INTEGER NOT NULL,
    occurred_at    TEXT    NOT NULL,
    event_type     TEXT    NOT NULL,
    correlation_id TEXT,
    actor          TEXT    NOT NULL,
    payload        TEXT    NOT NULL,
    previous_hash  TEXT    NOT NULL,
    entry_hash     TEXT    NOT NULL,
    UNIQUE (tenant_code, sequence)
);
CREATE INDEX IF NOT EXISTS ix_audit_correlation
    ON audit_events (tenant_code, correlation_id);

CREATE TABLE IF NOT EXISTS metering_events (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_code    TEXT    NOT NULL,
    occurred_at    TEXT    NOT NULL,
    event_type     TEXT    NOT NULL,
    quantity       INTEGER NOT NULL DEFAULT 1,
    unit_price     TEXT    NOT NULL,
    billable       INTEGER NOT NULL DEFAULT 1,
    correlation_id TEXT,
    reference      TEXT,
    note           TEXT
);
CREATE INDEX IF NOT EXISTS ix_metering_tenant
    ON metering_events (tenant_code, occurred_at);

CREATE TABLE IF NOT EXISTS decisions (
    decision_id    TEXT PRIMARY KEY,
    tenant_code    TEXT NOT NULL,
    application_id TEXT NOT NULL,
    correlation_id TEXT NOT NULL,
    outcome        TEXT NOT NULL,
    decided_at     TEXT NOT NULL,
    expires_at     TEXT,
    payload        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_decisions_tenant_app
    ON decisions (tenant_code, application_id);

CREATE TABLE IF NOT EXISTS idempotency (
    tenant_code    TEXT NOT NULL,
    key            TEXT NOT NULL,
    decision_id    TEXT NOT NULL,
    stored_at      TEXT NOT NULL,
    PRIMARY KEY (tenant_code, key)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      default=str)


class Ledger:
    """Append-only audit and metering store with per-tenant hash chaining."""

    def __init__(self, path: Optional[pathlib.Path | str] = None):
        self.path = str(path or DEFAULT_PATH)
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.executescript(_SCHEMA)
        self._connection.commit()

    # -- audit ------------------------------------------------------------- #
    def record(self, *, tenant_code: str, event_type: str, payload: dict,
               actor: str = "system", correlation_id: Optional[str] = None
               ) -> dict[str, Any]:
        """Append one audit event, chained to the tenant's previous event."""
        with self._lock:
            cursor = self._connection.execute(
                "SELECT sequence, entry_hash FROM audit_events "
                "WHERE tenant_code = ? ORDER BY sequence DESC LIMIT 1",
                (tenant_code,))
            row = cursor.fetchone()
            sequence = (row["sequence"] + 1) if row else 1
            previous_hash = row["entry_hash"] if row else "GENESIS"

            occurred_at = _now()
            body = _canonical({
                "tenant_code": tenant_code, "sequence": sequence,
                "occurred_at": occurred_at, "event_type": event_type,
                "correlation_id": correlation_id, "actor": actor,
                "payload": payload,
            })
            entry_hash = hashlib.sha256(
                (previous_hash + body).encode()).hexdigest()

            self._connection.execute(
                "INSERT INTO audit_events (tenant_code, sequence, occurred_at, "
                "event_type, correlation_id, actor, payload, previous_hash, "
                "entry_hash) VALUES (?,?,?,?,?,?,?,?,?)",
                (tenant_code, sequence, occurred_at, event_type,
                 correlation_id, actor, _canonical(payload), previous_hash,
                 entry_hash))
            self._connection.commit()
            return {"sequence": sequence, "entry_hash": entry_hash,
                    "occurred_at": occurred_at}

    def audit_trail(self, *, tenant_code: str,
                    correlation_id: Optional[str] = None,
                    limit: int = 200) -> list[dict]:
        query = ("SELECT * FROM audit_events WHERE tenant_code = ?"
                 + (" AND correlation_id = ?" if correlation_id else "")
                 + " ORDER BY sequence ASC LIMIT ?")
        params: tuple = ((tenant_code, correlation_id, limit) if correlation_id
                         else (tenant_code, limit))
        rows = self._connection.execute(query, params).fetchall()
        return [{
            "sequence": row["sequence"],
            "occurred_at": row["occurred_at"],
            "event_type": row["event_type"],
            "correlation_id": row["correlation_id"],
            "actor": row["actor"],
            "payload": json.loads(row["payload"]),
            "entry_hash": row["entry_hash"],
            "previous_hash": row["previous_hash"],
        } for row in rows]

    def verify_chain(self, *, tenant_code: str) -> dict[str, Any]:
        """Recompute the chain and report the first break, if any."""
        rows = self._connection.execute(
            "SELECT * FROM audit_events WHERE tenant_code = ? "
            "ORDER BY sequence ASC", (tenant_code,)).fetchall()
        previous_hash = "GENESIS"
        for row in rows:
            body = _canonical({
                "tenant_code": row["tenant_code"], "sequence": row["sequence"],
                "occurred_at": row["occurred_at"],
                "event_type": row["event_type"],
                "correlation_id": row["correlation_id"], "actor": row["actor"],
                "payload": json.loads(row["payload"]),
            })
            expected = hashlib.sha256((previous_hash + body).encode()).hexdigest()
            if row["previous_hash"] != previous_hash or row["entry_hash"] != expected:
                return {"intact": False, "events": len(rows),
                        "broken_at_sequence": row["sequence"]}
            previous_hash = row["entry_hash"]
        return {"intact": True, "events": len(rows), "broken_at_sequence": None}

    # -- metering ---------------------------------------------------------- #
    def meter(self, *, tenant_code: str, event_type: str, quantity: int = 1,
              correlation_id: Optional[str] = None,
              reference: Optional[str] = None, billable: bool = True,
              note: Optional[str] = None) -> None:
        unit_price = BILLABLE.get(event_type, Decimal("0.00"))
        with self._lock:
            self._connection.execute(
                "INSERT INTO metering_events (tenant_code, occurred_at, "
                "event_type, quantity, unit_price, billable, correlation_id, "
                "reference, note) VALUES (?,?,?,?,?,?,?,?,?)",
                (tenant_code, _now(), event_type, quantity, str(unit_price),
                 1 if billable else 0, correlation_id, reference, note))
            self._connection.commit()

    def usage(self, *, tenant_code: str) -> dict[str, Any]:
        rows = self._connection.execute(
            "SELECT event_type, billable, SUM(quantity) AS quantity, "
            "unit_price FROM metering_events WHERE tenant_code = ? "
            "GROUP BY event_type, billable, unit_price", (tenant_code,)
        ).fetchall()
        lines, total = [], Decimal("0.00")
        for row in rows:
            amount = (Decimal(row["unit_price"]) * row["quantity"]
                      if row["billable"] else Decimal("0.00"))
            total += amount
            lines.append({
                "event_type": row["event_type"],
                "billable": bool(row["billable"]),
                "quantity": row["quantity"],
                "unit_price": row["unit_price"],
                "amount": str(amount.quantize(Decimal("0.01"))),
            })
        return {
            "tenant": tenant_code,
            "currency": "USD",
            "lines": sorted(lines, key=lambda line: line["event_type"]),
            "total": str(total.quantize(Decimal("0.01"))),
            "pricing_note": ("illustrative prototype rates; production rates "
                             "live in the tenant contract"),
        }

    def reconcile(self, *, tenant_code: str) -> dict[str, Any]:
        """Every billable decision has a meter, and every meter has a decision."""
        decisions = {row["correlation_id"] for row in self._connection.execute(
            "SELECT correlation_id FROM decisions WHERE tenant_code = ?",
            (tenant_code,))}
        metered = {row["correlation_id"] for row in self._connection.execute(
            "SELECT correlation_id FROM metering_events WHERE tenant_code = ? "
            "AND event_type = 'APPLICATION_DECISION'", (tenant_code,))}
        return {
            "decisions": len(decisions),
            "metered_decisions": len(metered),
            "unmetered": sorted(decisions - metered),
            "orphan_meters": sorted(metered - decisions),
            "balanced": decisions == metered,
        }

    # -- decisions and idempotency ----------------------------------------- #
    def store_decision(self, *, tenant_code: str, decision: Any) -> None:
        payload = decision.as_dict(audience="internal")
        with self._lock:
            self._connection.execute(
                "INSERT OR REPLACE INTO decisions (decision_id, tenant_code, "
                "application_id, correlation_id, outcome, decided_at, "
                "expires_at, payload) VALUES (?,?,?,?,?,?,?,?)",
                (decision.decision_id, tenant_code, decision.application_id,
                 decision.correlation_id, decision.outcome, decision.decided_at,
                 decision.expires_at, _canonical(payload)))
            self._connection.commit()

    def get_decision(self, *, tenant_code: str, decision_id: str
                     ) -> Optional[dict]:
        row = self._connection.execute(
            "SELECT payload FROM decisions WHERE tenant_code = ? "
            "AND decision_id = ?", (tenant_code, decision_id)).fetchone()
        return json.loads(row["payload"]) if row else None

    def decisions_for_application(self, *, tenant_code: str,
                                  application_id: str) -> list[dict]:
        rows = self._connection.execute(
            "SELECT payload FROM decisions WHERE tenant_code = ? "
            "AND application_id = ? ORDER BY decided_at ASC",
            (tenant_code, application_id)).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def recent_decisions(self, *, tenant_code: str, limit: int = 50) -> list[dict]:
        rows = self._connection.execute(
            "SELECT payload FROM decisions WHERE tenant_code = ? "
            "ORDER BY decided_at DESC, rowid DESC LIMIT ?",
            (tenant_code, limit)).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def decision_count(self, *, tenant_code: str) -> int:
        row = self._connection.execute(
            "SELECT COUNT(*) AS n FROM decisions WHERE tenant_code = ?",
            (tenant_code,)).fetchone()
        return int(row["n"])

    def remember_idempotency(self, *, tenant_code: str, key: str,
                             decision_id: str) -> None:
        with self._lock:
            self._connection.execute(
                "INSERT OR IGNORE INTO idempotency (tenant_code, key, "
                "decision_id, stored_at) VALUES (?,?,?,?)",
                (tenant_code, key, decision_id, _now()))
            self._connection.commit()

    def replay(self, *, tenant_code: str, key: str) -> Optional[str]:
        row = self._connection.execute(
            "SELECT decision_id FROM idempotency WHERE tenant_code = ? "
            "AND key = ?", (tenant_code, key)).fetchone()
        return row["decision_id"] if row else None

    def close(self) -> None:
        self._connection.close()
