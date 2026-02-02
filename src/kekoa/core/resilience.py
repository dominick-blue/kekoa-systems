"""Resilience patterns for KEKOA.

Implements reliability patterns for fault tolerance:
- CircuitBreaker: Prevent cascade failures by failing fast
- RetryPolicy: Automatic retry with exponential backoff
- Bulkhead: Isolate failures between components
- Timeout: Enforce time limits on operations

These patterns are critical for satellite systems where:
- Resources are constrained
- Communication is intermittent
- Recovery must be autonomous
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from threading import Lock
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

from kekoa.core.errors import CircuitOpenError, Err, Ok, Result

T = TypeVar("T")
E = TypeVar("E", bound=Exception)


# ============================================================================
# Circuit Breaker
# ============================================================================


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = auto()
    """Normal operation - requests pass through."""

    OPEN = auto()
    """Circuit is open - requests are rejected immediately."""

    HALF_OPEN = auto()
    """Testing recovery - limited requests allowed."""


@dataclass
class CircuitBreakerConfig:
    """Configuration for CircuitBreaker.

    Attributes:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds before attempting recovery
        half_open_max_calls: Max calls to test during half-open state
        success_threshold: Successes needed in half-open to close circuit
        excluded_exceptions: Exception types that don't count as failures
    """

    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_max_calls: int = 3
    success_threshold: int = 2
    excluded_exceptions: tuple[type[Exception], ...] = ()


class CircuitBreaker:
    """Circuit breaker to prevent cascade failures.

    The circuit breaker pattern prevents a service from repeatedly
    trying to execute an operation that's likely to fail. Instead,
    it fails fast and allows the system to recover.

    State transitions:
    - CLOSED -> OPEN: When failure_threshold is reached
    - OPEN -> HALF_OPEN: After recovery_timeout expires
    - HALF_OPEN -> CLOSED: When success_threshold is reached
    - HALF_OPEN -> OPEN: On any failure

    Example:
        breaker = CircuitBreaker(name="tle_fetch")

        result = breaker.call(fetch_tle, satellite_id=12345)
        if result.is_err():
            if isinstance(result.error, CircuitOpenError):
                # Circuit is open, fail fast
                use_cached_data()
            else:
                # Actual fetch error
                handle_error(result.error)
    """

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> None:
        """Initialize circuit breaker.

        Args:
            name: Name for identification and logging
            config: Circuit breaker configuration
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None
        self._half_open_calls = 0
        self._lock = Lock()

    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        with self._lock:
            return self._state

    @property
    def failure_count(self) -> int:
        """Current failure count."""
        with self._lock:
            return self._failure_count

    def call(
        self,
        func: Callable[..., Result[T, E]],
        *args: object,
        **kwargs: object,
    ) -> Result[T, E | CircuitOpenError]:
        """Execute function with circuit breaker protection.

        Args:
            func: Function returning Result[T, E]
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Function result or Err[CircuitOpenError] if circuit is open
        """
        with self._lock:
            # Check if we should transition from OPEN to HALF_OPEN
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    self._success_count = 0
                else:
                    return Err(CircuitOpenError(self.name))

            # In HALF_OPEN, limit calls
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.config.half_open_max_calls:
                    return Err(CircuitOpenError(self.name))
                self._half_open_calls += 1

        # Execute the function (outside lock)
        result = func(*args, **kwargs)

        # Update state based on result
        with self._lock:
            if result.is_ok():
                self._on_success()
            else:
                # Check if this error type should be excluded
                if isinstance(result, Err) and not self._is_excluded(result.error):
                    self._on_failure()

        return result

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self._last_failure_time is None:
            return True
        elapsed = time.monotonic() - self._last_failure_time
        return elapsed >= self.config.recovery_timeout

    def _on_success(self) -> None:
        """Handle successful call."""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.config.success_threshold:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._success_count = 0
        elif self._state == CircuitState.CLOSED:
            self._failure_count = 0

    def _on_failure(self) -> None:
        """Handle failed call."""
        self._failure_count += 1
        self._last_failure_time = time.monotonic()

        if self._state == CircuitState.HALF_OPEN:
            # Any failure in half-open immediately opens circuit
            self._state = CircuitState.OPEN
        elif self._state == CircuitState.CLOSED:
            if self._failure_count >= self.config.failure_threshold:
                self._state = CircuitState.OPEN

    def _is_excluded(self, error: Exception) -> bool:
        """Check if exception type should not count as failure."""
        return isinstance(error, self.config.excluded_exceptions)

    def reset(self) -> None:
        """Manually reset circuit to closed state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            self._half_open_calls = 0

    def force_open(self) -> None:
        """Manually force circuit to open state."""
        with self._lock:
            self._state = CircuitState.OPEN
            self._last_failure_time = time.monotonic()


# ============================================================================
# Retry Policy
# ============================================================================


@dataclass
class RetryConfig:
    """Configuration for RetryPolicy.

    Attributes:
        max_attempts: Maximum number of attempts (including first)
        base_delay: Initial delay between attempts (seconds)
        max_delay: Maximum delay between attempts (seconds)
        exponential_base: Base for exponential backoff
        jitter: Add random jitter to delays (0.0-1.0)
        retryable_errors: Exception types that can be retried
    """

    max_attempts: int = 3
    base_delay: float = 0.1
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: float = 0.1
    retryable_errors: tuple[type[Exception], ...] = ()


class RetryPolicy:
    """Retry policy with exponential backoff.

    Automatically retries failed operations with increasing delays.
    Includes jitter to prevent thundering herd problems.

    Delay calculation:
        delay = min(base_delay * (exponential_base ** attempt), max_delay)
        delay += random(-jitter, +jitter) * delay

    Example:
        policy = RetryPolicy(RetryConfig(max_attempts=3, base_delay=0.1))

        result = policy.execute(fetch_data, url=data_url)
        if result.is_err():
            # All retries exhausted
            handle_permanent_failure(result.error)
    """

    def __init__(self, config: RetryConfig | None = None) -> None:
        """Initialize retry policy.

        Args:
            config: Retry configuration
        """
        self.config = config or RetryConfig()

    def execute(
        self,
        func: Callable[..., Result[T, E]],
        *args: object,
        **kwargs: object,
    ) -> Result[T, E]:
        """Execute function with retry.

        Args:
            func: Function returning Result[T, E]
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result from successful attempt or last failed attempt
        """
        last_result: Result[T, E] | None = None

        for attempt in range(self.config.max_attempts):
            result = func(*args, **kwargs)

            if result.is_ok():
                return result

            last_result = result

            # Check if error is retryable
            if isinstance(result, Err):
                if not self._is_retryable(result.error):
                    return result

                # Check if error is marked non-recoverable
                if not result.recoverable:
                    return result

            # Don't sleep after last attempt
            if attempt < self.config.max_attempts - 1:
                delay = self._calculate_delay(attempt)
                time.sleep(delay)

        # Should never happen, but satisfy type checker
        assert last_result is not None
        return last_result

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        # Exponential backoff
        delay = self.config.base_delay * (self.config.exponential_base**attempt)

        # Cap at max delay
        delay = min(delay, self.config.max_delay)

        # Add jitter
        if self.config.jitter > 0:
            jitter_range = delay * self.config.jitter
            delay += random.uniform(-jitter_range, jitter_range)

        return max(0, delay)

    def _is_retryable(self, error: Exception) -> bool:
        """Check if error type is retryable."""
        if not self.config.retryable_errors:
            # If no specific types configured, retry all
            return True
        return isinstance(error, self.config.retryable_errors)


# ============================================================================
# Bulkhead
# ============================================================================


@dataclass
class BulkheadConfig:
    """Configuration for Bulkhead.

    Attributes:
        max_concurrent: Maximum concurrent calls
        max_wait: Maximum time to wait for slot (seconds)
    """

    max_concurrent: int = 10
    max_wait: float = 5.0


class BulkheadFullError(Exception):
    """Bulkhead is full, cannot accept more calls."""

    def __init__(self, name: str, max_concurrent: int) -> None:
        super().__init__(f"Bulkhead '{name}' is full (max {max_concurrent})")
        self.name = name
        self.max_concurrent = max_concurrent


class Bulkhead:
    """Bulkhead pattern for isolation.

    Limits concurrent access to a resource to prevent one
    component from consuming all resources.

    Example:
        bulkhead = Bulkhead("inference", BulkheadConfig(max_concurrent=5))

        result = bulkhead.call(run_inference, image=frame)
        if result.is_err():
            if isinstance(result.error, BulkheadFullError):
                # Too many concurrent calls
                queue_for_later(frame)
    """

    def __init__(
        self,
        name: str,
        config: BulkheadConfig | None = None,
    ) -> None:
        """Initialize bulkhead.

        Args:
            name: Name for identification
            config: Bulkhead configuration
        """
        self.name = name
        self.config = config or BulkheadConfig()
        self._current = 0
        self._lock = Lock()

    @property
    def current_count(self) -> int:
        """Current number of concurrent calls."""
        with self._lock:
            return self._current

    @property
    def available(self) -> int:
        """Number of available slots."""
        with self._lock:
            return self.config.max_concurrent - self._current

    def call(
        self,
        func: Callable[..., Result[T, E]],
        *args: object,
        **kwargs: object,
    ) -> Result[T, E | BulkheadFullError]:
        """Execute function within bulkhead.

        Args:
            func: Function returning Result[T, E]
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Function result or Err[BulkheadFullError] if full
        """
        # Try to acquire slot
        acquired = False
        start_time = time.monotonic()

        while not acquired:
            with self._lock:
                if self._current < self.config.max_concurrent:
                    self._current += 1
                    acquired = True

            if not acquired:
                # Check timeout
                elapsed = time.monotonic() - start_time
                if elapsed >= self.config.max_wait:
                    return Err(BulkheadFullError(self.name, self.config.max_concurrent))

                # Brief sleep before retry
                time.sleep(0.01)

        try:
            return func(*args, **kwargs)
        finally:
            with self._lock:
                self._current -= 1


# ============================================================================
# Timeout
# ============================================================================


class TimeoutError(Exception):
    """Operation exceeded timeout."""

    def __init__(self, operation: str, timeout_seconds: float) -> None:
        super().__init__(f"Operation '{operation}' timed out after {timeout_seconds}s")
        self.operation = operation
        self.timeout_seconds = timeout_seconds


@dataclass
class TimeoutConfig:
    """Configuration for Timeout.

    Attributes:
        timeout: Maximum execution time (seconds)
        operation_name: Name for error messages
    """

    timeout: float = 30.0
    operation_name: str = "operation"


# Note: True timeout enforcement requires threading or asyncio.
# This simple implementation just records elapsed time and checks after.
# For real timeout enforcement, use asyncio.timeout() or threading.


class Timeout:
    """Timeout wrapper for operations.

    Records execution time and can be used to check if timeout exceeded.
    Note: This does not interrupt running operations - for true
    cancellation, use asyncio with timeout context.

    Example:
        timeout = Timeout(TimeoutConfig(timeout=5.0, operation_name="inference"))

        with timeout:
            result = run_inference(frame)

        if timeout.exceeded:
            log_timeout_warning(timeout.elapsed)
    """

    def __init__(self, config: TimeoutConfig | None = None) -> None:
        """Initialize timeout.

        Args:
            config: Timeout configuration
        """
        self.config = config or TimeoutConfig()
        self._start_time: float | None = None
        self._end_time: float | None = None

    @property
    def elapsed(self) -> float:
        """Elapsed time in seconds."""
        if self._start_time is None:
            return 0.0
        end = self._end_time or time.monotonic()
        return end - self._start_time

    @property
    def exceeded(self) -> bool:
        """Check if timeout was exceeded."""
        return self.elapsed > self.config.timeout

    @property
    def remaining(self) -> float:
        """Remaining time before timeout."""
        return max(0.0, self.config.timeout - self.elapsed)

    def __enter__(self) -> Timeout:
        """Start timing."""
        self._start_time = time.monotonic()
        self._end_time = None
        return self

    def __exit__(self, *args: object) -> None:
        """Stop timing."""
        self._end_time = time.monotonic()

    def check(self) -> Result[None, TimeoutError]:
        """Check if timeout exceeded.

        Returns:
            Ok[None] if within timeout, Err[TimeoutError] if exceeded
        """
        if self.exceeded:
            return Err(TimeoutError(self.config.operation_name, self.config.timeout))
        return Ok(None)


# ============================================================================
# Degradation Chain
# ============================================================================


@dataclass
class DegradationStep(Generic[T]):
    """Single step in a degradation chain.

    Attributes:
        name: Step name for logging
        func: Function to execute
        fallback: Next step if this one fails (or None if last)
    """

    name: str
    func: Callable[[], Result[T, Exception]]
    fallback: DegradationStep[T] | None = None


class DegradationChain(Generic[T]):
    """Graceful degradation chain.

    Attempts operations in order of capability, falling back
    to simpler implementations on failure.

    Example:
        chain = DegradationChain[float]()
        chain.add_step("full_model", lambda: run_full_model(frame))
        chain.add_step("lite_model", lambda: run_lite_model(frame))
        chain.add_step("heuristic", lambda: apply_heuristic(frame))

        result, level = chain.execute()
        if result.is_ok():
            log_inference(result.value, degradation_level=level)
    """

    def __init__(self) -> None:
        """Initialize degradation chain."""
        self._steps: list[DegradationStep[T]] = []

    def add_step(
        self,
        name: str,
        func: Callable[[], Result[T, Exception]],
    ) -> DegradationChain[T]:
        """Add a step to the chain.

        Steps are attempted in the order they are added.

        Args:
            name: Step name
            func: Function returning Result

        Returns:
            Self for chaining
        """
        step: DegradationStep[T] = DegradationStep(name=name, func=func)

        if self._steps:
            self._steps[-1] = DegradationStep(
                name=self._steps[-1].name,
                func=self._steps[-1].func,
                fallback=step,
            )

        self._steps.append(step)
        return self

    def execute(self) -> tuple[Result[T, Exception], int]:
        """Execute the degradation chain.

        Returns:
            Tuple of (result, degradation_level) where level is 0-indexed
            step that succeeded (or last step if all failed)
        """
        if not self._steps:
            return Err(ValueError("Degradation chain is empty")), 0

        current = self._steps[0]
        level = 0

        while current is not None:
            result = current.func()

            if result.is_ok():
                return result, level

            # Try fallback
            if current.fallback is None:
                # No more fallbacks, return last error
                return result, level

            current = current.fallback
            level += 1

        # Should never reach here
        return Err(ValueError("Degradation chain exhausted")), level

    def step_names(self) -> list[str]:
        """Get names of all steps in order."""
        return [step.name for step in self._steps]
