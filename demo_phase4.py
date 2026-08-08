"""Phase 4 demonstration: batch scoring, monitoring and the console.

Run:  python3 demo_phase4.py            (runs the demonstration, then exits)
      python3 demo_phase4.py --serve    (leaves the console running)

Scores a synthetic portfolio, shows the reject handling and reconciliation,
then produces the monitoring pack a risk team would read: approval mix, score
distribution against the model's development sample, and the PSI verdict.
"""
from __future__ import annotations

import json
import pathlib
import socket
import sys
import tempfile
import threading
import urllib.request

from api.server import serve
from core.ledger import Ledger
from core.pipeline import Platform
from partners import simulators


def call(port: int, method: str, path: str, *, body=None,
         key: str = "demo-key-payroll"):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", method=method,
        data=json.dumps(body).encode() if body is not None else None)
    request.add_header("Content-Type", "application/json")
    if key:
        request.add_header("X-API-Key", key)
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())


def bar(share: float, width: int = 28) -> str:
    filled = int(round(share * width))
    return "#" * filled + "." * (width - filled)


def main() -> None:
    simulators.reset()
    ledger_path = pathlib.Path(tempfile.mkdtemp()) / "phase4.db"
    platform = Platform(ledger=Ledger(ledger_path))
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = serve(port, platform=platform)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    print("=" * 78)
    print(f"Underwriting console: http://127.0.0.1:{port}/")
    print("=" * 78)

    # 1 -- batch ------------------------------------------------------------
    print("\n1. Batch portfolio scoring (synthetic file with deliberate defects)")
    result = call(port, "POST", "/v1/batches",
                  body={"use_sample": True, "rows": 120})
    print(f"   batch {result['batch_id']}  checksum {result['checksum_sha256'][:16]}...")
    print(f"   submitted {result['submitted']}  processed {result['processed']}  "
          f"rejected {result['rejected']}  reconciled={result['reconciled']}")
    print("   outcomes: " + "  ".join(f"{k}={v}" for k, v in
                                      sorted(result["outcomes"].items())))
    print("   rejected rows:")
    for reject in result["rejects"]:
        print(f"     {reject['row_id']}: {'; '.join(reject['errors'])}")

    # 2 -- monitoring -------------------------------------------------------
    print("\n2. Monitoring pack")
    report = call(port, "GET", "/v1/monitoring/summary")
    decisions = report["decisions"]
    psi = report["population_stability"]

    print(f"   overall status: {report['overall_status']}   "
          f"model {report['model_version']}   n={decisions['n']}")
    print(f"   approval {decisions['approval_rate']:.1%}   "
          f"refer {decisions['referral_rate']:.1%}   "
          f"decline {decisions['decline_rate']:.1%}   "
          f"insufficient {decisions['insufficient_rate']:.1%}")
    print(f"   counteroffers {decisions['counteroffer_rate']:.1%} of decisions "
          f"({decisions['counteroffer_share_of_approvals'] or 0:.1%} of approvals)")
    print(f"   segments: " + "  ".join(f"{k}={v}" for k, v in
                                       decisions["by_segment"].items()))

    print(f"\n   PSI {psi['index']} -> {psi['status']} "
          f"(warning {psi['thresholds']['warning']}, "
          f"breach {psi['thresholds']['breach']})")
    print("   score distribution, development sample vs live batch:")
    print(f"     {'band':<14} {'development':>11} {'live':>7}   contribution")
    for band in psi["bands"]:
        low = "" if band["from"] is None else f"{band['from']:.0f}"
        high = "" if band["to"] is None else f"{band['to']:.0f}"
        label = f"{low or '<'}-{high or '+'}"
        print(f"     {label:<14} {band['expected_share']:>10.1%} "
              f"{band['actual_share']:>7.1%}   {band['contribution']:+.4f} "
              f"{bar(band['actual_share'])}")
    print("   largest moves: " + ", ".join(
        f"band {m['band']} ({m['contribution']:+.3f})"
        for m in psi["largest_moves"]))

    print(f"\n   calibration: {report['calibration']['status']} — "
          f"{report['calibration'].get('note', '')}")
    print("   top reason codes:")
    for code, count in decisions["top_reason_codes"][:6]:
        print(f"     {code:<32} {count}")

    # 3 -- usage -------------------------------------------------------------
    print("\n3. Usage and reconciliation")
    usage = call(port, "GET", "/v1/usage")
    for line in usage["lines"]:
        print(f"   {line['event_type']:<22} qty {line['quantity']:<5} "
              f"@ {line['unit_price']:<6} = {line['amount']}")
    print(f"   total {usage['currency']} {usage['total']}   "
          f"reconciled={usage['reconciliation']['balanced']}")

    # 4 -- documentation ------------------------------------------------------
    print("\n4. Documentation pack")
    for name in ("MODEL_CARD.md", "RUNBOOK.md", "API_GUIDE.md",
                 "TRACEABILITY.md"):
        path = pathlib.Path(__file__).resolve().parent / "docs" / name
        print(f"   docs/{name:<20} {len(path.read_text().splitlines()):>4} lines")

    print("\n" + "=" * 78)
    if "--serve" in sys.argv:
        print(f"Console left running: http://127.0.0.1:{port}/   (Ctrl-C to stop)")
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
    server.shutdown()
    print(f"ledger: {ledger_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
