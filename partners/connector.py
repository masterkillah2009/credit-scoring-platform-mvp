"""Connector framework: the resilience contract every partner call obeys.

Implements IPSRS FR-CNX-01..04 / BRD BR-DAT-01..06. Every connector, whatever
it talks to, gets the same behaviour without reimplementing it:

  * per-partner timeout budget, genuinely enforced (the call is abandoned, not
    merely measured after the fact)
  * bounded retries with exponential backoff, only for retryable failures
  * a circuit breaker so a failing partner is not hammered, with half-open
    probing to recover automatically
  * parallel execution across partners, so the latency budget is the slowest
    partner rather than the sum of all of them
  * health metrics per partner: calls, failures, timeouts, latency percentiles,
    circuit state
  * a structured result that records what happened, so a decision made on
    partial data can say exactly which source was missing and why

A connector never invents data. When a partner fails the result carries
``ok=False`` and the payload is ``None``; the tenant's degradation policy - not
the connector - decides what the platform does about it.
"""
from __future__ import annotations

import concurrent.futures
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from partners.simulators import PartnerError, PartnerTimeout

# Circuit states
CLOSED, OPEN, HALF_OPEN = "CLOSED", "OPEN", "HALF_OPEN"


@dataclass
class PartnerResult:
    """Outcome of one partner call, successful or not."""

    partner: str
    ok: bool
    payload: Optional[Any]
    latency_ms: int
    attempts: int
    error: Optional[str] = None
    circuit_state: str = CLOSED
    from_cache: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "partner": self.partner,
            "ok": self.ok,
            "latency_ms": self.latency_ms,
            "attempts": self.attempts,
            "error": self.error,
            "circuit_state": self.circuit_state,
            "has_payload": self.payload is not None,
        }


class CircuitBreaker:
    """Trips after consecutive failures; probes once the cooldown elapses."""

    def __init__(self, *, failure_threshold: int = 3, reset_after_s: float = 5.0):
        self.failure_threshold = failure_threshold
        self.reset_after_s = reset_after_s
        self._failures = 0
        self._opened_at = 0.0
        self._state = CLOSED
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            if (self._state == OPEN
                    and time.monotonic() - self._opened_at >= self.reset_after_s):
                self._state = HALF_OPEN
            return self._state

    def allows(self) -> bool:
        return self.state in (CLOSED, HALF_OPEN)

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._state = CLOSED

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._state = OPEN
                self._opened_at = time.monotonic()

    def reset(self) -> None:
        with self._lock:
            self._failures = 0
            self._state = CLOSED
            self._opened_at = 0.0


@dataclass
class PartnerHealth:
    calls: int = 0
    successes: int = 0
    failures: int = 0
    timeouts: int = 0
    short_circuited: int = 0
    latencies_ms: list[int] = field(default_factory=list)

    def observe(self, result: PartnerResult) -> None:
        self.calls += 1
        if result.ok:
            self.successes += 1
        else:
            self.failures += 1
            if result.error and "timeout" in result.error.lower():
                self.timeouts += 1
        self.latencies_ms.append(result.latency_ms)
        del self.latencies_ms[:-500]          # bounded window

    def snapshot(self, circuit_state: str) -> dict[str, Any]:
        ordered = sorted(self.latencies_ms)
        def percentile(p: float) -> Optional[int]:
            if not ordered:
                return None
            index = min(int(len(ordered) * p), len(ordered) - 1)
            return ordered[index]
        return {
            "calls": self.calls,
            "successes": self.successes,
            "failures": self.failures,
            "timeouts": self.timeouts,
            "short_circuited": self.short_circuited,
            "availability": (round(self.successes / self.calls, 4)
                             if self.calls else None),
            "latency_p50_ms": percentile(0.50),
            "latency_p95_ms": percentile(0.95),
            "circuit_state": circuit_state,
        }


class Connector:
    """One partner, wrapped in the resilience contract."""

    def __init__(self, name: str, call: Callable[..., Any], *,
                 timeout_ms: int, retries: int = 1,
                 backoff_ms: int = 50):
        self.name = name
        self._call = call
        self.timeout_ms = timeout_ms
        self.retries = retries
        self.backoff_ms = backoff_ms
        self.breaker = CircuitBreaker()
        self.health = PartnerHealth()

    def _call_with_timeout(self, **kwargs: Any) -> Any:
        """Call the partner, abandoning it if the budget expires.

        A daemon worker is used rather than ``ThreadPoolExecutor`` as a context
        manager: the executor's ``__exit__`` waits for its worker, so a slow
        partner would still consume the caller's latency budget even though the
        timeout had already fired - which defeats the purpose of having one.
        Here the caller returns the moment the budget expires; the abandoned
        worker finishes in the background and its result is discarded.

        A production implementation would set the timeout on the HTTP client
        itself (or use async I/O) so the socket is closed rather than orphaned.
        """
        outcome: "queue.Queue[tuple[bool, Any]]" = queue.Queue(maxsize=1)

        def worker() -> None:
            try:
                outcome.put((True, self._call(**kwargs)))
            except BaseException as exc:           # forwarded to the caller
                outcome.put((False, exc))

        threading.Thread(target=worker, daemon=True,
                         name=f"partner-{self.name}").start()
        try:
            ok, value = outcome.get(timeout=self.timeout_ms / 1000.0)
        except queue.Empty:
            raise concurrent.futures.TimeoutError(
                f"no response within {self.timeout_ms}ms")
        if ok:
            return value
        raise value

    def invoke(self, **kwargs: Any) -> PartnerResult:
        if not self.breaker.allows():
            self.health.short_circuited += 1
            result = PartnerResult(self.name, False, None, 0, 0,
                                   error="circuit open: call not attempted",
                                   circuit_state=self.breaker.state)
            self.health.observe(result)
            return result

        started = time.monotonic()
        attempts = 0
        last_error: Optional[str] = None

        for attempt in range(self.retries + 1):
            attempts = attempt + 1
            try:
                payload = self._call_with_timeout(**kwargs)
                elapsed = int((time.monotonic() - started) * 1000)
                self.breaker.record_success()
                result = PartnerResult(self.name, True, payload, elapsed,
                                       attempts, circuit_state=self.breaker.state)
                self.health.observe(result)
                return result
            except concurrent.futures.TimeoutError:
                last_error = f"timeout after {self.timeout_ms}ms"
            except PartnerTimeout as exc:
                last_error = f"timeout: {exc}"
            except PartnerError as exc:
                last_error = str(exc)
            except Exception as exc:                     # defensive
                last_error = f"unexpected error: {exc.__class__.__name__}: {exc}"

            if attempt < self.retries:
                time.sleep((self.backoff_ms * (2 ** attempt)) / 1000.0)

        elapsed = int((time.monotonic() - started) * 1000)
        self.breaker.record_failure()
        result = PartnerResult(self.name, False, None, elapsed, attempts,
                               error=last_error,
                               circuit_state=self.breaker.state)
        self.health.observe(result)
        return result


class ConnectorRegistry:
    """The set of connectors available to the platform."""

    def __init__(self, registry: dict[str, tuple[Callable[..., Any], int]]):
        self._connectors = {
            name: Connector(name, call, timeout_ms=timeout)
            for name, (call, timeout) in registry.items()
        }

    def __getitem__(self, name: str) -> Connector:
        return self._connectors[name]

    def fetch_all(self, requests: dict[str, dict]) -> dict[str, PartnerResult]:
        """Call several partners in parallel.

        The latency budget for retrieval is therefore the slowest partner, not
        the sum - which is what makes a sub-second decision achievable while
        consulting five external systems.
        """
        results: dict[str, PartnerResult] = {}
        if not requests:
            return results
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(requests)) as pool:
            futures = {
                pool.submit(self._connectors[name].invoke, **kwargs): name
                for name, kwargs in requests.items()
                if name in self._connectors
            }
            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                results[name] = future.result()
        return results

    def health(self) -> dict[str, dict]:
        return {name: connector.health.snapshot(connector.breaker.state)
                for name, connector in self._connectors.items()}

    def reset(self) -> None:
        for connector in self._connectors.values():
            connector.breaker.reset()
            connector.health = PartnerHealth()
