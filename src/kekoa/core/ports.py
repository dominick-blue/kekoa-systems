"""Port interfaces (Protocols) for KEKOA.

Defines abstract interfaces that adapters must implement.
Following Hexagonal Architecture, these ports define the boundary
between the core domain and external systems.

Inbound Ports: Called by external systems to interact with the domain
Outbound Ports: Called by the domain to interact with external systems
"""

from __future__ import annotations

from abc import abstractmethod
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

from kekoa.core.errors import (
    CheckpointError,
    ContactCalculationError,
    HealthCheckError,
    PropagationError,
    RefreshError,
    RepositoryError,
    Result,
    TLENotFoundError,
)
from kekoa.core.types import (
    AggregatedModel,
    ContactWindow,
    DegradationLevel,
    ModelGradient,
    NodeID,
    PowerBudget,
    StateVector,
    TLESet,
    TopologyGraph,
)


# ============================================================================
# Health Status Types
# ============================================================================


class HealthStatus(Enum):
    """Health check status."""

    HEALTHY = auto()
    """Component is operating normally."""

    DEGRADED = auto()
    """Component is operating with reduced capability."""

    UNHEALTHY = auto()
    """Component is not operating correctly."""

    UNKNOWN = auto()
    """Health status could not be determined."""


# ============================================================================
# Outbound Ports - Data Repositories
# ============================================================================


@runtime_checkable
class TLERepository(Protocol):
    """Port for TLE data access.

    Implementations might read from:
    - Local files
    - Space-Track API
    - CelesTrak
    - Ground station uplink
    """

    @abstractmethod
    def get_tle(self, satellite_id: int) -> Result[TLESet, TLENotFoundError]:
        """Get TLE for a specific satellite.

        Args:
            satellite_id: NORAD catalog number

        Returns:
            Ok[TLESet] if found, Err[TLENotFoundError] if not
        """
        ...

    @abstractmethod
    def get_all(self) -> Result[list[TLESet], RepositoryError]:
        """Get all available TLEs.

        Returns:
            Ok[list[TLESet]] with all TLEs, or Err on failure
        """
        ...

    @abstractmethod
    def get_by_name(self, name: str) -> Result[TLESet, TLENotFoundError]:
        """Get TLE by satellite name.

        Args:
            name: Satellite name (e.g., "ISS (ZARYA)")

        Returns:
            Ok[TLESet] if found, Err[TLENotFoundError] if not
        """
        ...

    @abstractmethod
    def get_constellation(self, prefix: str) -> Result[list[TLESet], RepositoryError]:
        """Get all TLEs for a constellation by name prefix.

        Args:
            prefix: Name prefix (e.g., "STARLINK")

        Returns:
            Ok[list[TLESet]] with matching TLEs
        """
        ...

    @abstractmethod
    def refresh(self) -> Result[int, RefreshError]:
        """Refresh TLE data from source.

        Returns:
            Ok[int] with count of updated TLEs, or Err on failure
        """
        ...


@runtime_checkable
class ModelRepository(Protocol):
    """Port for ML model storage.

    Implementations might use:
    - Local filesystem
    - Object storage
    - Model registry
    """

    @abstractmethod
    def get_model(self, model_id: str, version: str | None = None) -> Result[bytes, RepositoryError]:
        """Get serialized model weights.

        Args:
            model_id: Model identifier
            version: Optional specific version (latest if None)

        Returns:
            Ok[bytes] with model weights, or Err on failure
        """
        ...

    @abstractmethod
    def save_model(
        self,
        model_id: str,
        weights: bytes,
        metadata: dict[str, str] | None = None,
    ) -> Result[str, RepositoryError]:
        """Save model weights.

        Args:
            model_id: Model identifier
            weights: Serialized model weights
            metadata: Optional metadata to store

        Returns:
            Ok[str] with version identifier, or Err on failure
        """
        ...

    @abstractmethod
    def list_versions(self, model_id: str) -> Result[list[str], RepositoryError]:
        """List available versions of a model.

        Args:
            model_id: Model identifier

        Returns:
            Ok[list[str]] with version identifiers, or Err on failure
        """
        ...

    @abstractmethod
    def delete_model(self, model_id: str, version: str | None = None) -> Result[None, RepositoryError]:
        """Delete a model or specific version.

        Args:
            model_id: Model identifier
            version: Specific version to delete (all if None)

        Returns:
            Ok[None] on success, or Err on failure
        """
        ...


# ============================================================================
# Outbound Ports - Orbital Calculations
# ============================================================================


@runtime_checkable
class Propagator(Protocol):
    """Port for orbital propagation.

    Implementations use SGP4 or similar propagators.
    """

    @abstractmethod
    def propagate(self, tle: TLESet, epoch: datetime) -> Result[StateVector, PropagationError]:
        """Propagate TLE to get state vector at epoch.

        Args:
            tle: Two-Line Element set
            epoch: Time to propagate to

        Returns:
            Ok[StateVector] with position and velocity, or Err on failure
        """
        ...

    @abstractmethod
    def propagate_range(
        self,
        tle: TLESet,
        start: datetime,
        end: datetime,
        step: timedelta,
    ) -> Result[list[StateVector], PropagationError]:
        """Propagate TLE over a time range.

        Args:
            tle: Two-Line Element set
            start: Start of range
            end: End of range
            step: Time step between points

        Returns:
            Ok[list[StateVector]] with ephemeris, or Err on failure
        """
        ...


@runtime_checkable
class ContactCalculator(Protocol):
    """Port for contact window calculation.

    Calculates when satellites can communicate based on
    line-of-sight and link budget constraints.
    """

    @abstractmethod
    def calculate(
        self,
        sv1: StateVector,
        sv2: StateVector,
        max_range: float = 5000.0,
    ) -> Result[ContactWindow | None, ContactCalculationError]:
        """Calculate if two satellites are in contact.

        Args:
            sv1: First satellite state vector
            sv2: Second satellite state vector
            max_range: Maximum communication range (km)

        Returns:
            Ok[ContactWindow] if in contact, Ok[None] if not, or Err on failure
        """
        ...

    @abstractmethod
    def find_contacts(
        self,
        tle1: TLESet,
        tle2: TLESet,
        start: datetime,
        end: datetime,
        step: timedelta,
    ) -> Result[list[ContactWindow], ContactCalculationError]:
        """Find all contact windows in time range.

        Args:
            tle1: First satellite TLE
            tle2: Second satellite TLE
            start: Start of search window
            end: End of search window
            step: Search resolution

        Returns:
            Ok[list[ContactWindow]] with all contacts, or Err on failure
        """
        ...


# ============================================================================
# Outbound Ports - Topology
# ============================================================================


@runtime_checkable
class TopologyGenerator(Protocol):
    """Port for constellation topology generation.

    Generates communication graphs for federated learning.
    """

    @abstractmethod
    def generate(
        self,
        windows: Sequence[ContactWindow],
        timestamp: datetime,
        horizon: timedelta,
    ) -> TopologyGraph:
        """Generate topology graph from contact windows.

        Args:
            windows: Available contact windows
            timestamp: Current time
            horizon: How far ahead topology is valid

        Returns:
            TopologyGraph representing constellation connectivity
        """
        ...

    @abstractmethod
    def update(
        self,
        current: TopologyGraph,
        new_windows: Sequence[ContactWindow],
        expired_windows: Sequence[ContactWindow],
    ) -> TopologyGraph:
        """Update topology with new/expired contacts.

        Args:
            current: Current topology graph
            new_windows: Newly available contacts
            expired_windows: Contacts that have ended

        Returns:
            Updated TopologyGraph
        """
        ...


# ============================================================================
# Outbound Ports - Persistence
# ============================================================================


@runtime_checkable
class CheckpointStore(Protocol):
    """Port for state checkpointing.

    Enables crash recovery by persisting critical state.
    """

    @abstractmethod
    def save(self, key: str, state: bytes) -> Result[None, CheckpointError]:
        """Save checkpoint state.

        Args:
            key: Checkpoint identifier
            state: Serialized state to save

        Returns:
            Ok[None] on success, or Err on failure
        """
        ...

    @abstractmethod
    def load(self, key: str) -> Result[bytes | None, CheckpointError]:
        """Load checkpoint state.

        Args:
            key: Checkpoint identifier

        Returns:
            Ok[bytes] if found, Ok[None] if not found, or Err on failure
        """
        ...

    @abstractmethod
    def delete(self, key: str) -> Result[None, CheckpointError]:
        """Delete a checkpoint.

        Args:
            key: Checkpoint identifier

        Returns:
            Ok[None] on success, or Err on failure
        """
        ...

    @abstractmethod
    def list_checkpoints(self, prefix: str = "") -> Result[list[str], CheckpointError]:
        """List available checkpoints.

        Args:
            prefix: Filter by key prefix

        Returns:
            Ok[list[str]] with checkpoint keys, or Err on failure
        """
        ...


@runtime_checkable
class StateStore(Protocol):
    """Port for persistent state storage.

    More general than CheckpointStore, for arbitrary key-value state.
    """

    @abstractmethod
    def get(self, key: str) -> Result[bytes | None, RepositoryError]:
        """Get value for key.

        Args:
            key: State key

        Returns:
            Ok[bytes] if found, Ok[None] if not, or Err on failure
        """
        ...

    @abstractmethod
    def set(self, key: str, value: bytes, ttl: timedelta | None = None) -> Result[None, RepositoryError]:
        """Set value for key.

        Args:
            key: State key
            value: Value to store
            ttl: Optional time-to-live

        Returns:
            Ok[None] on success, or Err on failure
        """
        ...

    @abstractmethod
    def delete(self, key: str) -> Result[None, RepositoryError]:
        """Delete key.

        Args:
            key: State key

        Returns:
            Ok[None] on success, or Err on failure
        """
        ...

    @abstractmethod
    def exists(self, key: str) -> Result[bool, RepositoryError]:
        """Check if key exists.

        Args:
            key: State key

        Returns:
            Ok[bool] indicating existence, or Err on failure
        """
        ...


# ============================================================================
# Outbound Ports - Communication
# ============================================================================


@runtime_checkable
class MeshClient(Protocol):
    """Port for Lattice Mesh communication.

    Interface for inter-satellite communication via Anduril Lattice.
    """

    @abstractmethod
    def send_gradient(
        self,
        target: NodeID,
        gradient: ModelGradient,
    ) -> Result[None, RepositoryError]:
        """Send gradient to target node.

        Args:
            target: Target node ID
            gradient: Gradient to send

        Returns:
            Ok[None] on success, or Err on failure
        """
        ...

    @abstractmethod
    def receive_gradients(
        self,
        timeout: timedelta,
    ) -> Result[list[ModelGradient], RepositoryError]:
        """Receive pending gradients.

        Args:
            timeout: How long to wait for gradients

        Returns:
            Ok[list[ModelGradient]] with received gradients, or Err on failure
        """
        ...

    @abstractmethod
    def broadcast_model(
        self,
        model: AggregatedModel,
        targets: Sequence[NodeID],
    ) -> Result[list[NodeID], RepositoryError]:
        """Broadcast aggregated model to nodes.

        Args:
            model: Model to broadcast
            targets: Target nodes

        Returns:
            Ok[list[NodeID]] with nodes that received model, or Err on failure
        """
        ...

    @abstractmethod
    def get_reachable_nodes(self) -> Result[list[NodeID], RepositoryError]:
        """Get currently reachable nodes.

        Returns:
            Ok[list[NodeID]] with reachable node IDs, or Err on failure
        """
        ...


# ============================================================================
# Outbound Ports - Power Management
# ============================================================================


@runtime_checkable
class PowerManager(Protocol):
    """Port for power state monitoring.

    Interface to the satellite's power subsystem.
    """

    @abstractmethod
    def get_power_budget(self) -> Result[PowerBudget, RepositoryError]:
        """Get current power budget.

        Returns:
            Ok[PowerBudget] with current state, or Err on failure
        """
        ...

    @abstractmethod
    def can_execute_workload(self, required_watts: float) -> Result[bool, RepositoryError]:
        """Check if workload can be executed.

        Args:
            required_watts: Power required for workload

        Returns:
            Ok[bool] indicating if workload can run, or Err on failure
        """
        ...


# ============================================================================
# Inbound Ports - Health Monitoring
# ============================================================================


@runtime_checkable
class HealthCheck(Protocol):
    """Port for component health checks.

    Used by infrastructure to assess system health.
    """

    @abstractmethod
    def check(self) -> HealthStatus:
        """Perform health check.

        Returns:
            Current health status
        """
        ...

    @abstractmethod
    def get_details(self) -> dict[str, str]:
        """Get detailed health information.

        Returns:
            Dictionary of health details
        """
        ...


@runtime_checkable
class HealthMonitor(Protocol):
    """Port for aggregating health across components."""

    @abstractmethod
    def register(self, name: str, check: HealthCheck) -> None:
        """Register a health check.

        Args:
            name: Component name
            check: Health check implementation
        """
        ...

    @abstractmethod
    def check_all(self) -> dict[str, HealthStatus]:
        """Check health of all registered components.

        Returns:
            Mapping of component name to health status
        """
        ...

    @abstractmethod
    def get_degradation_level(self) -> DegradationLevel:
        """Determine overall degradation level.

        Returns:
            System degradation level based on component health
        """
        ...


# ============================================================================
# Inbound Ports - Metrics
# ============================================================================


@runtime_checkable
class MetricsCollector(Protocol):
    """Port for metrics collection.

    Used to record operational metrics for monitoring.
    """

    @abstractmethod
    def counter(self, name: str, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
        """Increment a counter metric.

        Args:
            name: Metric name
            value: Value to add
            labels: Optional metric labels
        """
        ...

    @abstractmethod
    def gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Set a gauge metric.

        Args:
            name: Metric name
            value: Current value
            labels: Optional metric labels
        """
        ...

    @abstractmethod
    def histogram(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Record a histogram observation.

        Args:
            name: Metric name
            value: Observed value
            labels: Optional metric labels
        """
        ...


# ============================================================================
# Inbound Ports - Telemetry
# ============================================================================


@runtime_checkable
class TelemetryExporter(Protocol):
    """Port for exporting telemetry to ground.

    Used to send operational data during ground contacts.
    """

    @abstractmethod
    def queue_telemetry(self, topic: str, data: bytes, priority: int = 0) -> Result[None, RepositoryError]:
        """Queue telemetry for transmission.

        Args:
            topic: Telemetry topic/category
            data: Serialized telemetry data
            priority: Transmission priority (higher = more urgent)

        Returns:
            Ok[None] on success, or Err on failure
        """
        ...

    @abstractmethod
    def flush(self, timeout: timedelta) -> Result[int, RepositoryError]:
        """Attempt to transmit queued telemetry.

        Args:
            timeout: Maximum time to spend transmitting

        Returns:
            Ok[int] with count of transmitted items, or Err on failure
        """
        ...

    @abstractmethod
    def get_queue_depth(self) -> int:
        """Get number of queued telemetry items.

        Returns:
            Count of items waiting for transmission
        """
        ...
