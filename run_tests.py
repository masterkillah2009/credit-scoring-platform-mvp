#!/usr/bin/env python3
"""Run the test suite with one process per module.

Each module runs in its own interpreter. That is deliberate rather than
incidental: the suite exercises module-level state - partner-simulator failure
injection, the per-tenant rate limiter, the scorecard artefact cache, the
shared authentication service and the thread pools inside the connector
framework. Sharing one process lets one module's residue affect another's
timing, which produces the worst kind of test result: one that depends on the
order things ran in.

Process isolation is also what a CI pipeline does, so this mirrors the
environment the results will actually be judged in.

    python3 run_tests.py            run every module
    python3 run_tests.py phase3     run one module
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import tempfile
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent
MODULES = ["test_phase1", "test_phase2", "test_phase3", "test_phase4",
           "test_phase5"]
TIMEOUT_SECONDS = 180


def run(module: str) -> tuple[bool, int, float, str]:
    """Run one module, capturing output through a file rather than a pipe.

    A pipe has a fixed kernel buffer. A child that fills it blocks on write
    while the parent waits on exit - a deadlock that looks exactly like a
    hanging test suite, and one this runner hit before being fixed. A temporary
    file has no such limit.
    """
    started = time.monotonic()
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8",
                                errors="replace") as sink:
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "unittest", f"tests.{module}"],
                cwd=ROOT, stdout=sink, stderr=subprocess.STDOUT,
                timeout=TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            return False, 0, time.monotonic() - started, (
                f"timed out after {TIMEOUT_SECONDS}s")
        sink.seek(0)
        output = sink.read()
    elapsed = time.monotonic() - started
    del completed
    match = re.search(r"^Ran (\d+) test", output, re.MULTILINE)
    count = int(match.group(1)) if match else 0
    passed = bool(re.search(r"^OK", output, re.MULTILINE))
    return passed, count, elapsed, output


def main() -> None:
    modules = ([f"test_{sys.argv[1].removeprefix('test_')}"] if len(sys.argv) > 1
               else MODULES)
    total = failures = 0

    for module in modules:
        print(f"{module:<14}", end=" ", flush=True)
        passed, count, elapsed, output = run(module)
        total += count
        if passed:
            print(f"ok      {count:>3} tests  {elapsed:5.1f}s")
        else:
            failures += 1
            print(f"FAILED  {count:>3} tests  {elapsed:5.1f}s")
            for line in output.splitlines():
                if line.startswith(("FAIL:", "ERROR:", "AssertionError",
                                    "timed out")):
                    print(f"    {line}")

    print("-" * 46)
    if failures:
        print(f"FAIL   {failures} of {len(modules)} modules")
        raise SystemExit(1)
    print(f"PASS   {total} tests across {len(modules)} modules")


if __name__ == "__main__":
    main()
