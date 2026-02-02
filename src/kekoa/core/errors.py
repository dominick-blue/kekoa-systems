"""Error handling for KEKOA.

Implements a Result[T, E] type for explicit error handling without exceptions.
This follows the Rust/functional programming pattern for making errors explicit
in function signatures.

Usage:
    def divide(a: float, b: float) -> Result[float, DivisionError]:
        if b == 0:
            return Err(DivisionError("Division by zero"))
        return Ok(a / b)

    result = divide(10, 2)
    if result.is_ok():
        print(f"Result: {result.unwrap()}")
    else:
        print(f"Error: {result.error}")

    # Or use pattern matching (Python 3.10+):
    match result:
        case Ok(value):
            print(f"Result: {value}")
        case Err(error):
            print(f"Error: {error}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar, Union, cast, overload

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")
U = TypeVar("U")
E = TypeVar("E", bound=Exception)
F = TypeVar("F", bound=Exception)


# ============================================================================
# Result Type
# ============================================================================


@dataclass(frozen=True)
class Ok(Generic[T]):
    """Success variant of Result type.

    Contains the successful value of an operation.

    Attributes:
        value: The successful result value
    """

    value: T

    def is_ok(self) -> bool:
        """Check if this is a success result."""
        return True

    def is_err(self) -> bool:
        """Check if this is an error result."""
        return False

    def unwrap(self) -> T:
        """Extract the success value.

        Returns:
            The contained value

        Note:
            Safe to call on Ok - will always succeed.
        """
        return self.value

    def unwrap_or(self, default: T) -> T:
        """Extract the value or return a default.

        Args:
            default: Value to return if this were an Err (unused for Ok)

        Returns:
            The contained value
        """
        return self.value

    def unwrap_or_else(self, f: Callable[[Any], T]) -> T:
        """Extract the value or compute a default.

        Args:
            f: Function to compute default (unused for Ok)

        Returns:
            The contained value
        """
        return self.value

    def map(self, f: Callable[[T], U]) -> Ok[U]:
        """Transform the success value.

        Args:
            f: Function to apply to the value

        Returns:
            New Ok with transformed value
        """
        return Ok(f(self.value))

    def map_err(self, f: Callable[[Any], F]) -> Ok[T]:
        """Transform an error (no-op for Ok).

        Args:
            f: Function to apply to error (unused)

        Returns:
            Self unchanged
        """
        return self

    def and_then(self, f: Callable[[T], Result[U, E]]) -> Result[U, E]:
        """Chain operations that might fail.

        Args:
            f: Function returning a Result

        Returns:
            Result of applying f to the value
        """
        return f(self.value)

    def or_else(self, f: Callable[[Any], Result[T, F]]) -> Ok[T]:
        """Recover from an error (no-op for Ok).

        Args:
            f: Recovery function (unused)

        Returns:
            Self unchanged
        """
        return self


@dataclass(frozen=True)
class Err(Generic[E]):
    """Error variant of Result type.

    Contains the error from a failed operation.

    Attributes:
        error: The error that occurred
        recoverable: Whether this error can be recovered from
        context: Additional context about the error
    """

    error: E
    recoverable: bool = True
    context: str | None = None

    def is_ok(self) -> bool:
        """Check if this is a success result."""
        return False

    def is_err(self) -> bool:
        """Check if this is an error result."""
        return True

    def unwrap(self) -> Any:
        """Attempt to extract a success value.

        Raises:
            The contained error
        """
        raise self.error

    def unwrap_or(self, default: T) -> T:
        """Extract the value or return a default.

        Args:
            default: Value to return since this is an Err

        Returns:
            The default value
        """
        return default

    def unwrap_or_else(self, f: Callable[[E], T]) -> T:
        """Extract the value or compute a default from the error.

        Args:
            f: Function to compute default from error

        Returns:
            Result of f(error)
        """
        return f(self.error)

    def map(self, f: Callable[[Any], U]) -> Err[E]:
        """Transform the success value (no-op for Err).

        Args:
            f: Function to apply to value (unused)

        Returns:
            Self unchanged
        """
        return self

    def map_err(self, f: Callable[[E], F]) -> Err[F]:
        """Transform the error.

        Args:
            f: Function to apply to the error

        Returns:
            New Err with transformed error
        """
        return Err(f(self.error), self.recoverable, self.context)

    def and_then(self, f: Callable[[Any], Result[U, E]]) -> Err[E]:
        """Chain operations (no-op for Err).

        Args:
            f: Function returning a Result (unused)

        Returns:
            Self unchanged
        """
        return self

    def or_else(self, f: Callable[[E], Result[T, F]]) -> Result[T, F]:
        """Recover from an error.

        Args:
            f: Recovery function

        Returns:
            Result of applying f to the error
        """
        return f(self.error)

    def with_context(self, context: str) -> Err[E]:
        """Add context to this error.

        Args:
            context: Additional context string

        Returns:
            New Err with added context
        """
        existing = f"{self.context}: " if self.context else ""
        return Err(self.error, self.recoverable, f"{existing}{context}")


# Result type alias
Result = Union[Ok[T], Err[E]]


# ============================================================================
# Result Helper Functions
# ============================================================================


def try_wrap(f: Callable[[], T], error_type: type[E]) -> Result[T, E]:
    """Wrap a potentially-throwing function in a Result.

    Args:
        f: Function that might raise an exception
        error_type: Expected exception type to catch

    Returns:
        Ok with result or Err with caught exception

    Example:
        result = try_wrap(lambda: json.loads(data), json.JSONDecodeError)
    """
    try:
        return Ok(f())
    except error_type as e:
        return Err(e)


def collect_results(results: list[Result[T, E]]) -> Result[list[T], E]:
    """Collect a list of Results into a Result of list.

    Returns Ok with list of values if all succeed, or first Err encountered.

    Args:
        results: List of Result values

    Returns:
        Ok[list[T]] if all Ok, otherwise first Err[E]
    """
    values: list[T] = []
    for result in results:
        if result.is_err():
            return cast(Err[E], result)
        values.append(cast(Ok[T], result).value)
    return Ok(values)


def first_ok(results: list[Result[T, E]]) -> Result[T, list[E]]:
    """Return first Ok result, or all errors if none succeed.

    Args:
        results: List of Result values

    Returns:
        First Ok found, or Err with list of all errors
    """
    errors: list[E] = []
    for result in results:
        if result.is_ok():
            return cast(Ok[T], result)
        errors.append(cast(Err[E], result).error)
    return Err(AggregateError(errors))  # type: ignore[arg-type]


# ============================================================================
# Domain Error Hierarchy
# ============================================================================


class KekoaError(Exception):
    """Base exception for all KEKOA errors."""

    def __init__(self, message: str, recoverable: bool = True) -> None:
        super().__init__(message)
        self.recoverable = recoverable


class AggregateError(KekoaError):
    """Multiple errors aggregated together."""

    def __init__(self, errors: list[Exception]) -> None:
        messages = [str(e) for e in errors]
        super().__init__(f"Multiple errors: {messages}")
        self.errors = errors


# ============================================================================
# Orbital Domain Errors
# ============================================================================


class OrbitalError(KekoaError):
    """Base class for orbital mechanics errors."""


class TLEParseError(OrbitalError):
    """Error parsing TLE data."""

    def __init__(self, message: str, line_number: int | None = None) -> None:
        if line_number is not None:
            message = f"Line {line_number}: {message}"
        super().__init__(message)
        self.line_number = line_number


class TLENotFoundError(OrbitalError):
    """TLE not found for requested satellite."""

    def __init__(self, satellite_id: int) -> None:
        super().__init__(f"TLE not found for satellite {satellite_id}")
        self.satellite_id = satellite_id


class PropagationError(OrbitalError):
    """Error propagating orbital state."""

    def __init__(self, message: str, satellite_id: int | None = None) -> None:
        if satellite_id is not None:
            message = f"Satellite {satellite_id}: {message}"
        super().__init__(message)
        self.satellite_id = satellite_id


class EpochError(OrbitalError):
    """Error related to epoch handling."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class ContactCalculationError(OrbitalError):
    """Error calculating contact windows."""


# ============================================================================
# Federation Domain Errors
# ============================================================================


class FederationError(KekoaError):
    """Base class for federated learning errors."""


class AggregationError(FederationError):
    """Error aggregating gradients."""

    def __init__(self, message: str, round_id: int | None = None) -> None:
        if round_id is not None:
            message = f"Round {round_id}: {message}"
        super().__init__(message)
        self.round_id = round_id


class GradientValidationError(FederationError):
    """Invalid gradient received."""

    def __init__(self, message: str, node_id: str | None = None) -> None:
        if node_id is not None:
            message = f"Node {node_id}: {message}"
        super().__init__(message)
        self.node_id = node_id


class TopologyError(FederationError):
    """Error in topology management."""


class NodeNotFoundError(FederationError):
    """Node not found in topology."""

    def __init__(self, node_id: str) -> None:
        super().__init__(f"Node not found: {node_id}")
        self.node_id = node_id


# ============================================================================
# Inference Domain Errors
# ============================================================================


class InferenceError(KekoaError):
    """Base class for inference errors."""


class ModelNotFoundError(InferenceError):
    """Model not found for inference."""

    def __init__(self, model_id: str) -> None:
        super().__init__(f"Model not found: {model_id}")
        self.model_id = model_id


class ModelLoadError(InferenceError):
    """Error loading model."""

    def __init__(self, model_id: str, reason: str) -> None:
        super().__init__(f"Failed to load model {model_id}: {reason}")
        self.model_id = model_id


class InferenceTimeoutError(InferenceError):
    """Inference exceeded time budget."""

    def __init__(self, elapsed_ms: float, budget_ms: float) -> None:
        super().__init__(f"Inference timeout: {elapsed_ms:.1f}ms > {budget_ms:.1f}ms budget")
        self.elapsed_ms = elapsed_ms
        self.budget_ms = budget_ms


# ============================================================================
# Infrastructure Errors
# ============================================================================


class InfrastructureError(KekoaError):
    """Base class for infrastructure errors."""


class CircuitOpenError(InfrastructureError):
    """Circuit breaker is open, rejecting calls."""

    def __init__(self, component: str | None = None) -> None:
        message = "Circuit breaker is open"
        if component:
            message = f"{message} for {component}"
        super().__init__(message, recoverable=True)
        self.component = component


class PersistenceError(InfrastructureError):
    """Error with persistence layer."""


class CheckpointError(PersistenceError):
    """Error with checkpointing."""

    def __init__(self, message: str, key: str | None = None) -> None:
        if key:
            message = f"Checkpoint '{key}': {message}"
        super().__init__(message)
        self.key = key


class RepositoryError(InfrastructureError):
    """Error with data repository."""


class RefreshError(RepositoryError):
    """Error refreshing repository data."""


class CommunicationError(InfrastructureError):
    """Error in inter-satellite communication."""


class HealthCheckError(InfrastructureError):
    """Health check failed."""

    def __init__(self, component: str, reason: str) -> None:
        super().__init__(f"Health check failed for {component}: {reason}")
        self.component = component


# ============================================================================
# Configuration Errors
# ============================================================================


class ConfigurationError(KekoaError):
    """Error in configuration."""

    def __init__(self, message: str, key: str | None = None) -> None:
        if key:
            message = f"Config key '{key}': {message}"
        super().__init__(message, recoverable=False)
        self.key = key


class ValidationError(KekoaError):
    """Validation failed."""

    def __init__(self, message: str, field: str | None = None) -> None:
        if field:
            message = f"Field '{field}': {message}"
        super().__init__(message, recoverable=False)
        self.field = field
