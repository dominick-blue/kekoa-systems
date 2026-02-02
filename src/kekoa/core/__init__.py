"""Core module - Shared kernel for KEKOA.

Contains domain types, error handling, port interfaces, and resilience patterns.
"""

from kekoa.core.errors import (
    Err,
    KekoaError,
    Ok,
    PropagationError,
    Result,
    TLENotFoundError,
    TLEParseError,
)
from kekoa.core.ports import (
    CheckpointStore,
    ContactCalculator,
    HealthCheck,
    HealthStatus,
    Propagator,
    TLERepository,
    TopologyGenerator,
)
from kekoa.core.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    DegradationChain,
    RetryConfig,
    RetryPolicy,
)
from kekoa.core.types import (
    ContactWindow,
    Degrees,
    DegradationLevel,
    EclipseState,
    Kilometers,
    NodeID,
    PowerBudget,
    PowerState,
    StateVector,
    TLESet,
    TopologyGraph,
    Vector3D,
)

__all__ = [
    # Types
    "NodeID",
    "Kilometers",
    "Degrees",
    "Vector3D",
    "TLESet",
    "StateVector",
    "ContactWindow",
    "TopologyGraph",
    "EclipseState",
    "PowerState",
    "PowerBudget",
    "DegradationLevel",
    # Result type
    "Ok",
    "Err",
    "Result",
    # Errors
    "KekoaError",
    "TLEParseError",
    "TLENotFoundError",
    "PropagationError",
    # Ports
    "TLERepository",
    "Propagator",
    "ContactCalculator",
    "TopologyGenerator",
    "CheckpointStore",
    "HealthCheck",
    "HealthStatus",
    # Resilience
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitState",
    "RetryPolicy",
    "RetryConfig",
    "DegradationChain",
]
