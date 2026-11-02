import unittest

from support_agent.resilience import (
    CircuitBreaker,
    CircuitOpenError,
    RetryExhaustedError,
    RetryPolicy,
    call_with_resilience,
    retry_async,
    with_timeout,
)


class RetryPolicyTests(unittest.TestCase):
    def test_delay_grows_exponentially_up_to_the_cap(self) -> None:
        policy = RetryPolicy(base_delay_seconds=0.1, max_delay_seconds=1.0)
        self.assertAlmostEqual(policy.delay_for_attempt(0), 0.1)
        self.assertAlmostEqual(policy.delay_for_attempt(1), 0.2)
        self.assertAlmostEqual(policy.delay_for_attempt(2), 0.4)
        self.assertAlmostEqual(policy.delay_for_attempt(10), 1.0)  # capped


class RetryAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_succeeds_on_first_attempt_without_sleeping(self) -> None:
        calls = []

        async def fn():
            calls.append(1)
            return "ok"

        slept = []

        async def fake_sleep(seconds):
            slept.append(seconds)

        result = await retry_async(fn, RetryPolicy(max_attempts=3), sleep=fake_sleep)
        self.assertEqual(result, "ok")
        self.assertEqual(len(calls), 1)
        self.assertEqual(slept, [])

    async def test_recovers_after_transient_failures(self) -> None:
        attempts = {"count": 0}

        async def fn():
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise ConnectionError("flaky")
            return "recovered"

        slept = []

        async def fake_sleep(seconds):
            slept.append(seconds)

        result = await retry_async(
            fn, RetryPolicy(max_attempts=5, base_delay_seconds=0.01), retry_on=(ConnectionError,), sleep=fake_sleep
        )
        self.assertEqual(result, "recovered")
        self.assertEqual(attempts["count"], 3)
        self.assertEqual(len(slept), 2)  # one delay before each retry

    async def test_raises_retry_exhausted_after_max_attempts(self) -> None:
        async def fn():
            raise ConnectionError("always fails")

        async def fake_sleep(_seconds):
            return None

        with self.assertRaises(RetryExhaustedError) as context:
            await retry_async(
                fn, RetryPolicy(max_attempts=3, base_delay_seconds=0.01), retry_on=(ConnectionError,), sleep=fake_sleep
            )
        self.assertEqual(context.exception.attempts, 3)
        self.assertIsInstance(context.exception.last_error, ConnectionError)

    async def test_non_retryable_exception_propagates_immediately(self) -> None:
        calls = []

        async def fn():
            calls.append(1)
            raise ValueError("not retryable")

        with self.assertRaises(ValueError):
            await retry_async(fn, RetryPolicy(max_attempts=5), retry_on=(ConnectionError,))
        self.assertEqual(len(calls), 1)  # never retried


class TimeoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_fast_call_succeeds(self) -> None:
        async def fn():
            return "done"

        self.assertEqual(await with_timeout(fn, 1.0), "done")

    async def test_slow_call_times_out(self) -> None:
        import asyncio

        async def fn():
            await asyncio.sleep(1.0)
            return "too slow"

        with self.assertRaises(TimeoutError):
            await with_timeout(fn, 0.01)


class CircuitBreakerTests(unittest.IsolatedAsyncioTestCase):
    async def test_stays_closed_under_the_failure_threshold(self) -> None:
        breaker = CircuitBreaker(failure_threshold=3)

        async def failing():
            raise ConnectionError("boom")

        for _ in range(2):
            with self.assertRaises(ConnectionError):
                await breaker.call(failing)
        self.assertEqual(breaker.state.value, "closed")

    async def test_trips_open_after_the_failure_threshold(self) -> None:
        breaker = CircuitBreaker(failure_threshold=2)

        async def failing():
            raise ConnectionError("boom")

        for _ in range(2):
            with self.assertRaises(ConnectionError):
                await breaker.call(failing)
        self.assertEqual(breaker.state.value, "open")

        async def would_succeed():
            return "should not run"

        with self.assertRaises(CircuitOpenError):
            await breaker.call(would_succeed)

    async def test_half_opens_and_closes_after_reset_timeout_and_success(self) -> None:
        clock = {"now": 0.0}
        breaker = CircuitBreaker(failure_threshold=1, reset_timeout_seconds=10.0, clock=lambda: clock["now"])

        async def failing():
            raise ConnectionError("boom")

        with self.assertRaises(ConnectionError):
            await breaker.call(failing)
        self.assertEqual(breaker.state.value, "open")

        clock["now"] = 5.0  # before reset timeout
        self.assertEqual(breaker.state.value, "open")

        clock["now"] = 11.0  # past reset timeout
        self.assertEqual(breaker.state.value, "half_open")

        async def succeeding():
            return "recovered"

        result = await breaker.call(succeeding)
        self.assertEqual(result, "recovered")
        self.assertEqual(breaker.state.value, "closed")

    async def test_count_failure_on_excludes_business_errors(self) -> None:
        breaker = CircuitBreaker(failure_threshold=1)

        async def not_found():
            raise ValueError("not found")

        with self.assertRaises(ValueError):
            await breaker.call(not_found, count_failure_on=(ConnectionError,))
        # ValueError isn't in count_failure_on, so it never counted as a failure.
        self.assertEqual(breaker.state.value, "closed")


class CallWithResilienceTests(unittest.IsolatedAsyncioTestCase):
    async def test_success_passes_through(self) -> None:
        async def fn():
            return 42

        result = await call_with_resilience(
            fn,
            retry_policy=RetryPolicy(max_attempts=2, base_delay_seconds=0.001),
            timeout_seconds=1.0,
            circuit_breaker=CircuitBreaker(failure_threshold=5),
        )
        self.assertEqual(result, 42)

    async def test_business_error_does_not_trip_the_breaker(self) -> None:
        breaker = CircuitBreaker(failure_threshold=1)

        async def not_found():
            raise ValueError("no such record")

        for _ in range(3):
            with self.assertRaises(ValueError):
                await call_with_resilience(
                    not_found,
                    retry_policy=RetryPolicy(max_attempts=1),
                    timeout_seconds=1.0,
                    circuit_breaker=breaker,
                    retry_on=(ConnectionError,),
                )
        self.assertEqual(breaker.state.value, "closed")

    async def test_exhausted_retries_trip_the_breaker(self) -> None:
        breaker = CircuitBreaker(failure_threshold=1)

        async def flaky():
            raise ConnectionError("down")

        with self.assertRaises(RetryExhaustedError):
            await call_with_resilience(
                flaky,
                retry_policy=RetryPolicy(max_attempts=2, base_delay_seconds=0.001),
                timeout_seconds=1.0,
                circuit_breaker=breaker,
                retry_on=(ConnectionError,),
            )
        self.assertEqual(breaker.state.value, "open")


if __name__ == "__main__":
    unittest.main()
