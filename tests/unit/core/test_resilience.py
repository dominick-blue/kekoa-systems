"""Unit tests for resilience patterns."""

import time
from unittest.mock import MagicMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from kekoa.core.errors import CircuitOpenError, Err, Ok, Result
from kekoa.core.resilience import (
    Bulkhead,
    BulkheadConfig,
    BulkheadFullError,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    DegradationChain,
    RetryConfig,
    RetryPolicy,
    Timeout,
    TimeoutConfig,
    TimeoutError,
)


# ============================================================================
# CircuitBreaker Tests
# ============================================================================


class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig."""

    def test_defaults(self) -> None:
        """Test default configuration values."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.recovery_timeout == 60.0
        assert config.half_open_max_calls == 3
        assert config.success_threshold == 2
        assert config.excluded_exceptions == ()


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    def test_initial_state_closed(self) -> None:
        """Test circuit starts in closed state."""
        breaker = CircuitBreaker("test")
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    def test_success_keeps_closed(self) -> None:
        """Test successful calls keep circuit closed."""

        def success() -> Result[int, ValueError]:
            return Ok(42)

        breaker = CircuitBreaker("test")
        result = breaker.call(success)
        assert result.is_ok()
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    def test_failures_accumulate(self) -> None:
        """Test failures accumulate until threshold."""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = CircuitBreaker("test", config)

        def failure() -> Result[int, ValueError]:
            return Err(ValueError("error"))

        # First two failures don't open circuit
        breaker.call(failure)
        assert breaker.failure_count == 1
        assert breaker.state == CircuitState.CLOSED

        breaker.call(failure)
        assert breaker.failure_count == 2
        assert breaker.state == CircuitState.CLOSED

        # Third failure opens circuit
        breaker.call(failure)
        assert breaker.failure_count == 3
        assert breaker.state == CircuitState.OPEN

    def test_open_circuit_rejects_calls(self) -> None:
        """Test open circuit immediately rejects calls."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=60.0,  # Long timeout so we stay open
        )
        breaker = CircuitBreaker("test", config)

        def failure() -> Result[int, ValueError]:
            return Err(ValueError("error"))

        # Open the circuit
        breaker.call(failure)
        assert breaker.state == CircuitState.OPEN

        # Subsequent calls are rejected
        def success() -> Result[int, ValueError]:
            return Ok(42)

        result = breaker.call(success)
        assert result.is_err()
        assert isinstance(result, Err)
        assert isinstance(result.error, CircuitOpenError)

    def test_recovery_timeout_transitions_to_half_open(self) -> None:
        """Test circuit transitions to half-open after timeout."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=0.1,  # 100ms timeout
        )
        breaker = CircuitBreaker("test", config)

        def failure() -> Result[int, ValueError]:
            return Err(ValueError("error"))

        # Open the circuit
        breaker.call(failure)
        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(0.15)

        # Next call should transition to half-open
        def success() -> Result[int, ValueError]:
            return Ok(42)

        result = breaker.call(success)
        # After the successful call, circuit should close
        assert result.is_ok()

    def test_half_open_success_closes_circuit(self) -> None:
        """Test successful calls in half-open close the circuit."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=0.01,
            success_threshold=2,
        )
        breaker = CircuitBreaker("test", config)

        def failure() -> Result[int, ValueError]:
            return Err(ValueError("error"))

        def success() -> Result[int, ValueError]:
            return Ok(42)

        # Open the circuit
        breaker.call(failure)
        time.sleep(0.02)

        # First success in half-open
        breaker.call(success)
        # May still be half-open if success_threshold > 1

        # Second success should close
        breaker.call(success)
        assert breaker.state == CircuitState.CLOSED

    def test_half_open_failure_reopens_circuit(self) -> None:
        """Test failure in half-open reopens the circuit."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=0.01,
        )
        breaker = CircuitBreaker("test", config)

        def failure() -> Result[int, ValueError]:
            return Err(ValueError("error"))

        # Open the circuit
        breaker.call(failure)
        time.sleep(0.02)

        # Failure in half-open should reopen
        breaker.call(failure)
        assert breaker.state == CircuitState.OPEN

    def test_excluded_exceptions(self) -> None:
        """Test excluded exceptions don't count as failures."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            excluded_exceptions=(ValueError,),
        )
        breaker = CircuitBreaker("test", config)

        def excluded_failure() -> Result[int, ValueError]:
            return Err(ValueError("excluded"))

        def counted_failure() -> Result[int, TypeError]:
            return Err(TypeError("counted"))

        # Excluded exception doesn't count
        breaker.call(excluded_failure)
        assert breaker.failure_count == 0

        # Non-excluded exception counts
        breaker.call(counted_failure)
        assert breaker.failure_count == 1

    def test_success_resets_failure_count(self) -> None:
        """Test success resets failure count."""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = CircuitBreaker("test", config)

        def failure() -> Result[int, ValueError]:
            return Err(ValueError("error"))

        def success() -> Result[int, ValueError]:
            return Ok(42)

        # Accumulate some failures
        breaker.call(failure)
        breaker.call(failure)
        assert breaker.failure_count == 2

        # Success resets count
        breaker.call(success)
        assert breaker.failure_count == 0

    def test_reset(self) -> None:
        """Test manual reset."""
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = CircuitBreaker("test", config)

        def failure() -> Result[int, ValueError]:
            return Err(ValueError("error"))

        # Open the circuit
        breaker.call(failure)
        assert breaker.state == CircuitState.OPEN

        # Manual reset
        breaker.reset()
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    def test_force_open(self) -> None:
        """Test forcing circuit open."""
        breaker = CircuitBreaker("test")
        assert breaker.state == CircuitState.CLOSED

        breaker.force_open()
        assert breaker.state == CircuitState.OPEN


# ============================================================================
# RetryPolicy Tests
# ============================================================================


class TestRetryConfig:
    """Tests for RetryConfig."""

    def test_defaults(self) -> None:
        """Test default configuration values."""
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.base_delay == 0.1
        assert config.max_delay == 30.0
        assert config.exponential_base == 2.0
        assert config.jitter == 0.1


class TestRetryPolicy:
    """Tests for RetryPolicy."""

    def test_success_no_retry(self) -> None:
        """Test successful operation doesn't retry."""
        call_count = 0

        def success() -> Result[int, ValueError]:
            nonlocal call_count
            call_count += 1
            return Ok(42)

        policy = RetryPolicy(RetryConfig(max_attempts=3))
        result = policy.execute(success)

        assert result.is_ok()
        assert result.unwrap() == 42
        assert call_count == 1

    def test_failure_retries(self) -> None:
        """Test failed operation is retried."""
        call_count = 0

        def failure() -> Result[int, ValueError]:
            nonlocal call_count
            call_count += 1
            return Err(ValueError("error"))

        config = RetryConfig(max_attempts=3, base_delay=0.01)
        policy = RetryPolicy(config)
        result = policy.execute(failure)

        assert result.is_err()
        assert call_count == 3

    def test_succeeds_after_retries(self) -> None:
        """Test operation succeeds after initial failures."""
        call_count = 0

        def eventually_succeeds() -> Result[int, ValueError]:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return Err(ValueError("error"))
            return Ok(42)

        config = RetryConfig(max_attempts=5, base_delay=0.01)
        policy = RetryPolicy(config)
        result = policy.execute(eventually_succeeds)

        assert result.is_ok()
        assert result.unwrap() == 42
        assert call_count == 3

    def test_non_recoverable_not_retried(self) -> None:
        """Test non-recoverable errors are not retried."""
        call_count = 0

        def non_recoverable() -> Result[int, ValueError]:
            nonlocal call_count
            call_count += 1
            return Err(ValueError("fatal"), recoverable=False)

        config = RetryConfig(max_attempts=3, base_delay=0.01)
        policy = RetryPolicy(config)
        result = policy.execute(non_recoverable)

        assert result.is_err()
        assert call_count == 1

    def test_retryable_errors_filter(self) -> None:
        """Test only specified error types are retried."""
        call_count = 0

        def type_error() -> Result[int, TypeError]:
            nonlocal call_count
            call_count += 1
            return Err(TypeError("wrong type"))

        config = RetryConfig(
            max_attempts=3,
            base_delay=0.01,
            retryable_errors=(ValueError,),  # Only retry ValueError
        )
        policy = RetryPolicy(config)
        result = policy.execute(type_error)

        assert result.is_err()
        assert call_count == 1  # No retry for TypeError

    def test_delay_increases_exponentially(self) -> None:
        """Test delay increases with attempts."""
        config = RetryConfig(
            base_delay=0.1,
            max_delay=10.0,
            exponential_base=2.0,
            jitter=0.0,  # Disable jitter for predictable delays
        )
        policy = RetryPolicy(config)

        # Manually test delay calculation
        assert abs(policy._calculate_delay(0) - 0.1) < 0.01  # 0.1 * 2^0
        assert abs(policy._calculate_delay(1) - 0.2) < 0.01  # 0.1 * 2^1
        assert abs(policy._calculate_delay(2) - 0.4) < 0.01  # 0.1 * 2^2
        assert abs(policy._calculate_delay(3) - 0.8) < 0.01  # 0.1 * 2^3

    def test_delay_capped_at_max(self) -> None:
        """Test delay is capped at max_delay."""
        config = RetryConfig(
            base_delay=0.1,
            max_delay=0.5,
            exponential_base=2.0,
            jitter=0.0,
        )
        policy = RetryPolicy(config)

        # At attempt 10, delay would be 0.1 * 2^10 = 102.4
        # But should be capped at 0.5
        assert policy._calculate_delay(10) == 0.5


# ============================================================================
# Bulkhead Tests
# ============================================================================


class TestBulkheadConfig:
    """Tests for BulkheadConfig."""

    def test_defaults(self) -> None:
        """Test default configuration values."""
        config = BulkheadConfig()
        assert config.max_concurrent == 10
        assert config.max_wait == 5.0


class TestBulkhead:
    """Tests for Bulkhead."""

    def test_allows_concurrent_calls(self) -> None:
        """Test bulkhead allows calls within limit."""
        config = BulkheadConfig(max_concurrent=2)
        bulkhead = Bulkhead("test", config)

        def success() -> Result[int, ValueError]:
            return Ok(42)

        result = bulkhead.call(success)
        assert result.is_ok()
        assert bulkhead.current_count == 0  # Call completed

    def test_available_slots(self) -> None:
        """Test available slot tracking."""
        config = BulkheadConfig(max_concurrent=5)
        bulkhead = Bulkhead("test", config)
        assert bulkhead.available == 5

    def test_rejects_when_full(self) -> None:
        """Test bulkhead rejects when at capacity."""
        import threading

        config = BulkheadConfig(max_concurrent=1, max_wait=0.01)
        bulkhead = Bulkhead("test", config)

        started = threading.Event()
        finish = threading.Event()

        def blocking() -> Result[int, ValueError]:
            started.set()
            finish.wait()
            return Ok(42)

        def try_call() -> Result[int, BulkheadFullError]:
            return bulkhead.call(lambda: Ok(1))

        # Start blocking call in thread
        thread = threading.Thread(target=lambda: bulkhead.call(blocking))
        thread.start()
        started.wait()

        # Try to make another call - should be rejected
        result = try_call()
        assert result.is_err()
        assert isinstance(result, Err)
        assert isinstance(result.error, BulkheadFullError)

        # Clean up
        finish.set()
        thread.join()


# ============================================================================
# Timeout Tests
# ============================================================================


class TestTimeoutConfig:
    """Tests for TimeoutConfig."""

    def test_defaults(self) -> None:
        """Test default configuration values."""
        config = TimeoutConfig()
        assert config.timeout == 30.0
        assert config.operation_name == "operation"


class TestTimeout:
    """Tests for Timeout."""

    def test_within_timeout(self) -> None:
        """Test operation within timeout."""
        timeout = Timeout(TimeoutConfig(timeout=1.0))

        with timeout:
            time.sleep(0.1)

        assert not timeout.exceeded
        assert timeout.elapsed < 0.5

    def test_exceeds_timeout(self) -> None:
        """Test operation exceeding timeout."""
        timeout = Timeout(TimeoutConfig(timeout=0.1))

        with timeout:
            time.sleep(0.2)

        assert timeout.exceeded

    def test_remaining_time(self) -> None:
        """Test remaining time calculation."""
        timeout = Timeout(TimeoutConfig(timeout=1.0))

        with timeout:
            initial_remaining = timeout.remaining
            time.sleep(0.1)
            later_remaining = timeout.remaining

        assert later_remaining < initial_remaining

    def test_check_method(self) -> None:
        """Test check() returns Result."""
        timeout = Timeout(TimeoutConfig(timeout=0.1, operation_name="test_op"))

        with timeout:
            time.sleep(0.2)

        result = timeout.check()
        assert result.is_err()
        with pytest.raises(TimeoutError, match="test_op"):
            result.unwrap()


# ============================================================================
# DegradationChain Tests
# ============================================================================


class TestDegradationChain:
    """Tests for DegradationChain."""

    def test_first_step_succeeds(self) -> None:
        """Test chain returns first step if successful."""
        chain: DegradationChain[int] = DegradationChain()
        chain.add_step("full", lambda: Ok(100))
        chain.add_step("lite", lambda: Ok(80))
        chain.add_step("heuristic", lambda: Ok(50))

        result, level = chain.execute()
        assert result.is_ok()
        assert result.unwrap() == 100
        assert level == 0

    def test_falls_back_on_failure(self) -> None:
        """Test chain falls back when step fails."""
        chain: DegradationChain[int] = DegradationChain()
        chain.add_step("full", lambda: Err(ValueError("full failed")))
        chain.add_step("lite", lambda: Ok(80))
        chain.add_step("heuristic", lambda: Ok(50))

        result, level = chain.execute()
        assert result.is_ok()
        assert result.unwrap() == 80
        assert level == 1

    def test_falls_back_multiple_times(self) -> None:
        """Test chain can fall back multiple times."""
        chain: DegradationChain[int] = DegradationChain()
        chain.add_step("full", lambda: Err(ValueError("full failed")))
        chain.add_step("lite", lambda: Err(ValueError("lite failed")))
        chain.add_step("heuristic", lambda: Ok(50))

        result, level = chain.execute()
        assert result.is_ok()
        assert result.unwrap() == 50
        assert level == 2

    def test_all_steps_fail(self) -> None:
        """Test chain returns last error when all fail."""
        chain: DegradationChain[int] = DegradationChain()
        chain.add_step("full", lambda: Err(ValueError("full failed")))
        chain.add_step("lite", lambda: Err(ValueError("lite failed")))
        chain.add_step("heuristic", lambda: Err(ValueError("heuristic failed")))

        result, level = chain.execute()
        assert result.is_err()
        assert level == 2  # Last level

    def test_empty_chain(self) -> None:
        """Test empty chain returns error."""
        chain: DegradationChain[int] = DegradationChain()

        result, level = chain.execute()
        assert result.is_err()
        assert level == 0

    def test_step_names(self) -> None:
        """Test step names are tracked."""
        chain: DegradationChain[int] = DegradationChain()
        chain.add_step("full", lambda: Ok(100))
        chain.add_step("lite", lambda: Ok(80))
        chain.add_step("heuristic", lambda: Ok(50))

        assert chain.step_names() == ["full", "lite", "heuristic"]

    def test_chaining_add_step(self) -> None:
        """Test add_step returns self for chaining."""
        chain: DegradationChain[int] = DegradationChain()
        result = (
            chain.add_step("full", lambda: Ok(100))
            .add_step("lite", lambda: Ok(80))
            .add_step("heuristic", lambda: Ok(50))
        )

        assert result is chain
        assert len(chain.step_names()) == 3


# ============================================================================
# Property-Based Tests
# ============================================================================


class TestResilienceProperties:
    """Property-based tests for resilience patterns."""

    @given(st.integers(min_value=1, max_value=10))
    @settings(max_examples=50)
    def test_circuit_breaker_threshold_respected(self, threshold: int) -> None:
        """Property: circuit opens after exactly threshold failures."""
        config = CircuitBreakerConfig(failure_threshold=threshold)
        breaker = CircuitBreaker("test", config)

        def failure() -> Result[int, ValueError]:
            return Err(ValueError("error"))

        # Make threshold-1 failures, should still be closed
        for _ in range(threshold - 1):
            breaker.call(failure)
        assert breaker.state == CircuitState.CLOSED

        # One more failure should open
        breaker.call(failure)
        assert breaker.state == CircuitState.OPEN

    @given(st.integers(min_value=1, max_value=5))
    @settings(max_examples=50)
    def test_retry_makes_correct_attempts(self, attempts: int) -> None:
        """Property: retry makes exactly max_attempts calls."""
        call_count = 0

        def failure() -> Result[int, ValueError]:
            nonlocal call_count
            call_count += 1
            return Err(ValueError("error"))

        config = RetryConfig(max_attempts=attempts, base_delay=0.001)
        policy = RetryPolicy(config)
        policy.execute(failure)

        assert call_count == attempts
