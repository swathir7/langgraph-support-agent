"""Retry, timeout, and circuit-breaker primitives for calls to external systems.

The order-tool agent (graph/order_agent.py) wraps every call to the mock order
API through these — not because the mock API is unreliable, but because a
real order/CRM/payment API is, and this is the reusable shape that survives
swapping the mock for a real HTTP client. Each primitive takes injectable
clock/sleep functions so tests run in milliseconds, not real wall-clock time.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar

T = TypeVar("T")


class RetryExhaustedError(Exception):
    def __init__(self, attempts: int, last_error: BaseException) -> None:
        super().__init__(f"Gave up after {attempts} attempts: {last_error}")
        self.attempts = attempts
        self.last_error = last_error


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_seconds: float = 0.2
    max_delay_seconds: float = 5.0

    def delay_for_attempt(self, attempt: int) -> float:
        """`attempt` is 0-indexed (the delay before retry #1, #2, ...)."""
        return min(self.base_delay_seconds * (2**attempt), self.max_delay_seconds)


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    policy: RetryPolicy,
    *,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    last_error: BaseException | None = None
    for attempt in range(policy.max_attempts):
        try:
            return await fn()
        except retry_on as error:
            last_error = error
            if attempt < policy.max_attempts - 1:
                await sleep(policy.delay_for_attempt(attempt))
    assert last_error is not None
    raise RetryExhaustedError(policy.max_attempts, last_error)


async def with_timeout(fn: Callable[[], Awaitable[T]], timeout_seconds: float) -> T:
    return await asyncio.wait_for(fn(), timeout=timeout_seconds)


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    pass


class CircuitBreaker:
    """Trips OPEN after `failure_threshold` consecutive failures, refuses calls
    for `reset_timeout_seconds`, then allows calls through again as HALF_OPEN
    probes. A successful probe closes the circuit; a failed one reopens it and
    resets the timer. This reference implementation does not serialize
    concurrent probes to exactly one in-flight call during HALF_OPEN — for a
    single-agent-per-conversation workload that's not the bottleneck; a
    high-concurrency deployment would add a lock around the state transition."""

    def __init__(
        self,
        *,
        failure_threshold: int = 3,
        reset_timeout_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.reset_timeout_seconds = reset_timeout_seconds
        self._clock = clock
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> CircuitState:
        if (
            self._state == CircuitState.OPEN
            and self._opened_at is not None
            and self._clock() - self._opened_at >= self.reset_timeout_seconds
        ):
            return CircuitState.HALF_OPEN
        return self._state

    async def call(
        self,
        fn: Callable[[], Awaitable[T]],
        *,
        count_failure_on: tuple[type[BaseException], ...] = (Exception,),
    ) -> T:
        """`count_failure_on` restricts which exceptions trip the breaker — by
        default any exception counts, but callers with a mix of infra failures
        and legitimate business-logic errors (e.g. "order not found" is not a
        sign the order API is unhealthy) should narrow this to just the
        infra-failure types, or a string of "not found" lookups against valid
        IDs would needlessly trip the circuit for everyone."""

        current = self.state
        if current == CircuitState.OPEN:
            raise CircuitOpenError(
                f"Circuit is open; retry after {self.reset_timeout_seconds}s from last failure."
            )
        try:
            result = await fn()
        except count_failure_on:
            self._record_failure()
            raise
        else:
            self._record_success()
            return result

    def _record_success(self) -> None:
        self._consecutive_failures = 0
        self._state = CircuitState.CLOSED
        self._opened_at = None

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.failure_threshold:
            self._state = CircuitState.OPEN
            self._opened_at = self._clock()


async def call_with_resilience(
    fn: Callable[[], Awaitable[T]],
    *,
    retry_policy: RetryPolicy,
    timeout_seconds: float,
    circuit_breaker: CircuitBreaker,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
) -> T:
    """The composed stack every order-API call in this repo goes through:
    circuit breaker (outermost — a tripped breaker should short-circuit before
    even attempting a retry loop) wrapping retry-with-backoff wrapping a
    per-attempt timeout. Both `graph/order_agent.py` and
    `graph/human_approval.py` call this rather than each composing the three
    primitives themselves."""

    async def _timed() -> T:
        return await with_timeout(fn, timeout_seconds)

    async def _retried() -> T:
        return await retry_async(_timed, retry_policy, retry_on=retry_on)

    # retry_async either succeeds, raises RetryExhaustedError once retries run out,
    # or (for an exception type not in `retry_on`) lets it straight through
    # unwrapped. Only RetryExhaustedError is a real "this backend is unhealthy"
    # signal — a single retried-and-recovered blip never reaches here, and a
    # non-retryable error (e.g. "not found") isn't wrapped, so it's excluded by
    # construction: the breaker only ever sees RetryExhaustedError as a failure.
    return await circuit_breaker.call(_retried, count_failure_on=(RetryExhaustedError,))
