"""Unit tests for error handling and Result type."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from kekoa.core.errors import (
    AggregateError,
    AggregationError,
    CheckpointError,
    CircuitOpenError,
    ConfigurationError,
    ContactCalculationError,
    Err,
    GradientValidationError,
    HealthCheckError,
    InferenceError,
    InferenceTimeoutError,
    KekoaError,
    ModelLoadError,
    ModelNotFoundError,
    NodeNotFoundError,
    Ok,
    OrbitalError,
    PersistenceError,
    PropagationError,
    RefreshError,
    RepositoryError,
    Result,
    TLENotFoundError,
    TLEParseError,
    TopologyError,
    ValidationError,
    collect_results,
    first_ok,
    try_wrap,
)


# ============================================================================
# Ok Tests
# ============================================================================


class TestOk:
    """Tests for Ok variant of Result."""

    def test_is_ok(self) -> None:
        """Test is_ok returns True."""
        result: Result[int, Exception] = Ok(42)
        assert result.is_ok()

    def test_is_err(self) -> None:
        """Test is_err returns False."""
        result: Result[int, Exception] = Ok(42)
        assert not result.is_err()

    def test_unwrap(self) -> None:
        """Test unwrap returns value."""
        result: Result[int, Exception] = Ok(42)
        assert result.unwrap() == 42

    def test_unwrap_or(self) -> None:
        """Test unwrap_or returns value, not default."""
        result: Result[int, Exception] = Ok(42)
        assert result.unwrap_or(0) == 42

    def test_unwrap_or_else(self) -> None:
        """Test unwrap_or_else returns value, not computed default."""
        result: Result[int, Exception] = Ok(42)
        assert result.unwrap_or_else(lambda e: 0) == 42

    def test_map(self) -> None:
        """Test map transforms value."""
        result: Result[int, Exception] = Ok(42)
        mapped = result.map(lambda x: x * 2)
        assert mapped.unwrap() == 84

    def test_map_err(self) -> None:
        """Test map_err is no-op for Ok."""
        result: Result[int, ValueError] = Ok(42)
        mapped = result.map_err(lambda e: TypeError(str(e)))
        assert mapped.unwrap() == 42

    def test_and_then_success(self) -> None:
        """Test and_then chains successful operations."""

        def double_if_even(x: int) -> Result[int, ValueError]:
            if x % 2 == 0:
                return Ok(x * 2)
            return Err(ValueError("not even"))

        result: Result[int, ValueError] = Ok(4)
        chained = result.and_then(double_if_even)
        assert chained.unwrap() == 8

    def test_and_then_failure(self) -> None:
        """Test and_then propagates failure."""

        def double_if_even(x: int) -> Result[int, ValueError]:
            if x % 2 == 0:
                return Ok(x * 2)
            return Err(ValueError("not even"))

        result: Result[int, ValueError] = Ok(3)
        chained = result.and_then(double_if_even)
        assert chained.is_err()

    def test_or_else(self) -> None:
        """Test or_else is no-op for Ok."""
        result: Result[int, ValueError] = Ok(42)
        recovered = result.or_else(lambda e: Ok(0))
        assert recovered.unwrap() == 42

    @given(st.integers())
    @settings(max_examples=100)
    def test_ok_identity(self, value: int) -> None:
        """Property: Ok wraps and unwraps any value."""
        result: Result[int, Exception] = Ok(value)
        assert result.unwrap() == value


# ============================================================================
# Err Tests
# ============================================================================


class TestErr:
    """Tests for Err variant of Result."""

    def test_is_ok(self) -> None:
        """Test is_ok returns False."""
        result: Result[int, ValueError] = Err(ValueError("error"))
        assert not result.is_ok()

    def test_is_err(self) -> None:
        """Test is_err returns True."""
        result: Result[int, ValueError] = Err(ValueError("error"))
        assert result.is_err()

    def test_unwrap_raises(self) -> None:
        """Test unwrap raises the error."""
        result: Result[int, ValueError] = Err(ValueError("test error"))
        with pytest.raises(ValueError, match="test error"):
            result.unwrap()

    def test_unwrap_or(self) -> None:
        """Test unwrap_or returns default."""
        result: Result[int, ValueError] = Err(ValueError("error"))
        assert result.unwrap_or(0) == 0

    def test_unwrap_or_else(self) -> None:
        """Test unwrap_or_else computes default from error."""
        result: Result[int, ValueError] = Err(ValueError("42"))
        assert result.unwrap_or_else(lambda e: int(str(e))) == 42

    def test_map(self) -> None:
        """Test map is no-op for Err."""
        result: Result[int, ValueError] = Err(ValueError("error"))
        mapped = result.map(lambda x: x * 2)
        assert mapped.is_err()

    def test_map_err(self) -> None:
        """Test map_err transforms error."""
        result: Result[int, ValueError] = Err(ValueError("original"))
        mapped = result.map_err(lambda e: TypeError(f"wrapped: {e}"))
        assert mapped.is_err()
        with pytest.raises(TypeError, match="wrapped: original"):
            mapped.unwrap()

    def test_and_then(self) -> None:
        """Test and_then is no-op for Err."""
        result: Result[int, ValueError] = Err(ValueError("error"))
        chained = result.and_then(lambda x: Ok(x * 2))
        assert chained.is_err()

    def test_or_else_success(self) -> None:
        """Test or_else can recover from error."""
        result: Result[int, ValueError] = Err(ValueError("error"))
        recovered = result.or_else(lambda e: Ok(0))
        assert recovered.unwrap() == 0

    def test_or_else_failure(self) -> None:
        """Test or_else can fail to different error."""
        result: Result[int, ValueError] = Err(ValueError("error"))
        recovered = result.or_else(lambda e: Err(TypeError("different")))
        assert recovered.is_err()

    def test_recoverable_default(self) -> None:
        """Test Err is recoverable by default."""
        result: Err[ValueError] = Err(ValueError("error"))
        assert result.recoverable

    def test_recoverable_false(self) -> None:
        """Test Err can be marked non-recoverable."""
        result: Err[ValueError] = Err(ValueError("error"), recoverable=False)
        assert not result.recoverable

    def test_context(self) -> None:
        """Test Err can have context."""
        result: Err[ValueError] = Err(ValueError("error"), context="in operation X")
        assert result.context == "in operation X"

    def test_with_context(self) -> None:
        """Test adding context to Err."""
        result: Err[ValueError] = Err(ValueError("error"))
        with_ctx = result.with_context("in operation X")
        assert with_ctx.context == "in operation X"

    def test_with_context_chains(self) -> None:
        """Test chaining context."""
        result: Err[ValueError] = Err(ValueError("error"), context="first")
        with_ctx = result.with_context("second")
        assert with_ctx.context == "first: second"


# ============================================================================
# Result Helper Function Tests
# ============================================================================


class TestTryWrap:
    """Tests for try_wrap helper."""

    def test_success(self) -> None:
        """Test wrapping successful operation."""
        result = try_wrap(lambda: 42, ValueError)
        assert result.is_ok()
        assert result.unwrap() == 42

    def test_failure(self) -> None:
        """Test wrapping failing operation."""

        def fails() -> int:
            raise ValueError("error")

        result = try_wrap(fails, ValueError)
        assert result.is_err()

    def test_wrong_exception_type(self) -> None:
        """Test that wrong exception type is not caught."""

        def fails() -> int:
            raise TypeError("wrong type")

        with pytest.raises(TypeError):
            try_wrap(fails, ValueError)


class TestCollectResults:
    """Tests for collect_results helper."""

    def test_all_ok(self) -> None:
        """Test collecting all Ok results."""
        results: list[Result[int, ValueError]] = [Ok(1), Ok(2), Ok(3)]
        collected = collect_results(results)
        assert collected.is_ok()
        assert collected.unwrap() == [1, 2, 3]

    def test_first_err(self) -> None:
        """Test that first error is returned."""
        results: list[Result[int, ValueError]] = [
            Ok(1),
            Err(ValueError("first")),
            Ok(3),
            Err(ValueError("second")),
        ]
        collected = collect_results(results)
        assert collected.is_err()
        with pytest.raises(ValueError, match="first"):
            collected.unwrap()

    def test_empty_list(self) -> None:
        """Test collecting empty list."""
        results: list[Result[int, ValueError]] = []
        collected = collect_results(results)
        assert collected.is_ok()
        assert collected.unwrap() == []


class TestFirstOk:
    """Tests for first_ok helper."""

    def test_first_ok_found(self) -> None:
        """Test finding first Ok result."""
        results: list[Result[int, ValueError]] = [
            Err(ValueError("first")),
            Ok(42),
            Err(ValueError("third")),
        ]
        result = first_ok(results)
        assert result.is_ok()
        assert result.unwrap() == 42

    def test_all_err(self) -> None:
        """Test when all results are Err."""
        results: list[Result[int, ValueError]] = [
            Err(ValueError("first")),
            Err(ValueError("second")),
        ]
        result = first_ok(results)
        assert result.is_err()

    def test_empty_list(self) -> None:
        """Test with empty list."""
        results: list[Result[int, ValueError]] = []
        result = first_ok(results)
        assert result.is_err()


# ============================================================================
# Domain Error Tests
# ============================================================================


class TestKekoaError:
    """Tests for base KekoaError."""

    def test_message(self) -> None:
        """Test error message."""
        error = KekoaError("test message")
        assert str(error) == "test message"

    def test_recoverable_default(self) -> None:
        """Test default recoverability."""
        error = KekoaError("test")
        assert error.recoverable

    def test_recoverable_false(self) -> None:
        """Test non-recoverable error."""
        error = KekoaError("test", recoverable=False)
        assert not error.recoverable


class TestOrbitalErrors:
    """Tests for orbital domain errors."""

    def test_tle_parse_error(self) -> None:
        """Test TLE parse error."""
        error = TLEParseError("invalid checksum", line_number=2)
        assert "Line 2" in str(error)
        assert "invalid checksum" in str(error)
        assert error.line_number == 2

    def test_tle_parse_error_no_line(self) -> None:
        """Test TLE parse error without line number."""
        error = TLEParseError("invalid format")
        assert str(error) == "invalid format"
        assert error.line_number is None

    def test_tle_not_found_error(self) -> None:
        """Test TLE not found error."""
        error = TLENotFoundError(25544)
        assert "25544" in str(error)
        assert error.satellite_id == 25544

    def test_propagation_error(self) -> None:
        """Test propagation error."""
        error = PropagationError("decayed", satellite_id=25544)
        assert "Satellite 25544" in str(error)
        assert "decayed" in str(error)
        assert error.satellite_id == 25544

    def test_contact_calculation_error(self) -> None:
        """Test contact calculation error."""
        error = ContactCalculationError("invalid geometry")
        assert str(error) == "invalid geometry"


class TestFederationErrors:
    """Tests for federation domain errors."""

    def test_aggregation_error(self) -> None:
        """Test aggregation error."""
        error = AggregationError("insufficient clients", round_id=5)
        assert "Round 5" in str(error)
        assert error.round_id == 5

    def test_gradient_validation_error(self) -> None:
        """Test gradient validation error."""
        error = GradientValidationError("NaN detected", node_id="SAT-001")
        assert "Node SAT-001" in str(error)
        assert error.node_id == "SAT-001"

    def test_topology_error(self) -> None:
        """Test topology error."""
        error = TopologyError("graph disconnected")
        assert str(error) == "graph disconnected"

    def test_node_not_found_error(self) -> None:
        """Test node not found error."""
        error = NodeNotFoundError("SAT-999")
        assert "SAT-999" in str(error)
        assert error.node_id == "SAT-999"


class TestInferenceErrors:
    """Tests for inference domain errors."""

    def test_model_not_found_error(self) -> None:
        """Test model not found error."""
        error = ModelNotFoundError("cloud_detect_v1")
        assert "cloud_detect_v1" in str(error)
        assert error.model_id == "cloud_detect_v1"

    def test_model_load_error(self) -> None:
        """Test model load error."""
        error = ModelLoadError("cloud_detect_v1", "corrupted weights")
        assert "cloud_detect_v1" in str(error)
        assert "corrupted weights" in str(error)
        assert error.model_id == "cloud_detect_v1"

    def test_inference_timeout_error(self) -> None:
        """Test inference timeout error."""
        error = InferenceTimeoutError(elapsed_ms=75.0, budget_ms=50.0)
        assert "75" in str(error)
        assert "50" in str(error)
        assert error.elapsed_ms == 75.0
        assert error.budget_ms == 50.0


class TestInfrastructureErrors:
    """Tests for infrastructure errors."""

    def test_circuit_open_error(self) -> None:
        """Test circuit open error."""
        error = CircuitOpenError("tle_fetch")
        assert "tle_fetch" in str(error)
        assert error.component == "tle_fetch"
        assert error.recoverable  # Circuit open is recoverable

    def test_circuit_open_error_no_component(self) -> None:
        """Test circuit open error without component."""
        error = CircuitOpenError()
        assert "Circuit breaker is open" in str(error)
        assert error.component is None

    def test_checkpoint_error(self) -> None:
        """Test checkpoint error."""
        error = CheckpointError("write failed", key="oak_state")
        assert "oak_state" in str(error)
        assert "write failed" in str(error)
        assert error.key == "oak_state"

    def test_health_check_error(self) -> None:
        """Test health check error."""
        error = HealthCheckError("OAK", "TLE data stale")
        assert "OAK" in str(error)
        assert "TLE data stale" in str(error)
        assert error.component == "OAK"


class TestConfigurationErrors:
    """Tests for configuration errors."""

    def test_configuration_error(self) -> None:
        """Test configuration error."""
        error = ConfigurationError("invalid value", key="propagation_horizon")
        assert "propagation_horizon" in str(error)
        assert "invalid value" in str(error)
        assert not error.recoverable  # Config errors are not recoverable

    def test_validation_error(self) -> None:
        """Test validation error."""
        error = ValidationError("must be positive", field="timeout")
        assert "timeout" in str(error)
        assert "must be positive" in str(error)
        assert not error.recoverable


class TestAggregateError:
    """Tests for aggregate error."""

    def test_aggregate_error(self) -> None:
        """Test aggregate error contains all errors."""
        errors = [
            ValueError("first"),
            TypeError("second"),
        ]
        error = AggregateError(errors)
        assert "first" in str(error)
        assert "second" in str(error)
        assert len(error.errors) == 2


# ============================================================================
# Error Hierarchy Tests
# ============================================================================


class TestErrorHierarchy:
    """Tests for error class hierarchy."""

    def test_orbital_errors_inherit_from_kekoa(self) -> None:
        """Test orbital errors inherit from KekoaError."""
        assert issubclass(OrbitalError, KekoaError)
        assert issubclass(TLEParseError, KekoaError)
        assert issubclass(PropagationError, KekoaError)

    def test_federation_errors_inherit_from_kekoa(self) -> None:
        """Test federation errors inherit from KekoaError."""
        assert issubclass(AggregationError, KekoaError)
        assert issubclass(TopologyError, KekoaError)

    def test_inference_errors_inherit_from_kekoa(self) -> None:
        """Test inference errors inherit from KekoaError."""
        assert issubclass(InferenceError, KekoaError)
        assert issubclass(ModelNotFoundError, KekoaError)

    def test_infrastructure_errors_inherit_from_kekoa(self) -> None:
        """Test infrastructure errors inherit from KekoaError."""
        assert issubclass(PersistenceError, KekoaError)
        assert issubclass(RepositoryError, KekoaError)
        assert issubclass(CircuitOpenError, KekoaError)
