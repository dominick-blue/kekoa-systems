"""Domain types for KEKOA.

All types are immutable (frozen dataclasses) to ensure determinism and thread safety.
Reference frames follow IERS/IAU conventions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Iterator

# ============================================================================
# Type Aliases
# ============================================================================

NodeID = str
"""Unique identifier for a satellite node in the constellation."""

Kilometers = float
"""Distance in kilometers."""

Degrees = float
"""Angle in degrees."""

Radians = float
"""Angle in radians."""

Seconds = float
"""Time duration in seconds."""

# ============================================================================
# Reference Frames
# ============================================================================

ReferenceFrame = Literal["ECI", "ECEF", "BODY", "LVLH"]
"""
Supported coordinate reference frames:
- ECI: Earth-Centered Inertial (J2000)
- ECEF: Earth-Centered Earth-Fixed
- BODY: Satellite body frame
- LVLH: Local Vertical Local Horizontal
"""


# ============================================================================
# Vector Types
# ============================================================================


@dataclass(frozen=True)
class Vector3D:
    """Three-dimensional vector with reference frame annotation.

    Attributes:
        x: X component
        y: Y component
        z: Z component
        frame: Coordinate reference frame (ECI, ECEF, BODY, LVLH)
    """

    x: float
    y: float
    z: float
    frame: ReferenceFrame

    def magnitude(self) -> float:
        """Compute the Euclidean norm of the vector."""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def dot(self, other: Vector3D) -> float:
        """Compute dot product with another vector.

        Args:
            other: Vector to dot with (must be in same frame)

        Returns:
            Scalar dot product

        Raises:
            ValueError: If vectors are in different reference frames
        """
        if self.frame != other.frame:
            raise ValueError(f"Cannot dot vectors in different frames: {self.frame} vs {other.frame}")
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Vector3D) -> Vector3D:
        """Compute cross product with another vector.

        Args:
            other: Vector to cross with (must be in same frame)

        Returns:
            Cross product vector in same frame

        Raises:
            ValueError: If vectors are in different reference frames
        """
        if self.frame != other.frame:
            raise ValueError(
                f"Cannot cross vectors in different frames: {self.frame} vs {other.frame}"
            )
        return Vector3D(
            x=self.y * other.z - self.z * other.y,
            y=self.z * other.x - self.x * other.z,
            z=self.x * other.y - self.y * other.x,
            frame=self.frame,
        )

    def scale(self, factor: float) -> Vector3D:
        """Scale the vector by a scalar factor."""
        return Vector3D(
            x=self.x * factor,
            y=self.y * factor,
            z=self.z * factor,
            frame=self.frame,
        )

    def add(self, other: Vector3D) -> Vector3D:
        """Add another vector.

        Args:
            other: Vector to add (must be in same frame)

        Returns:
            Sum vector in same frame

        Raises:
            ValueError: If vectors are in different reference frames
        """
        if self.frame != other.frame:
            raise ValueError(f"Cannot add vectors in different frames: {self.frame} vs {other.frame}")
        return Vector3D(
            x=self.x + other.x,
            y=self.y + other.y,
            z=self.z + other.z,
            frame=self.frame,
        )

    def subtract(self, other: Vector3D) -> Vector3D:
        """Subtract another vector.

        Args:
            other: Vector to subtract (must be in same frame)

        Returns:
            Difference vector in same frame

        Raises:
            ValueError: If vectors are in different reference frames
        """
        if self.frame != other.frame:
            raise ValueError(
                f"Cannot subtract vectors in different frames: {self.frame} vs {other.frame}"
            )
        return Vector3D(
            x=self.x - other.x,
            y=self.y - other.y,
            z=self.z - other.z,
            frame=self.frame,
        )

    def normalize(self) -> Vector3D:
        """Return unit vector in same direction.

        Returns:
            Normalized unit vector

        Raises:
            ValueError: If vector has zero magnitude
        """
        mag = self.magnitude()
        if mag == 0:
            raise ValueError("Cannot normalize zero vector")
        return self.scale(1.0 / mag)

    def to_tuple(self) -> tuple[float, float, float]:
        """Convert to tuple (x, y, z)."""
        return (self.x, self.y, self.z)


# ============================================================================
# Orbital Types
# ============================================================================


@dataclass(frozen=True)
class TLESet:
    """Two-Line Element set for orbital propagation.

    TLE format follows NORAD/Space-Track conventions.
    Lines are validated to be 69 characters per the TLE specification.

    Attributes:
        satellite_id: NORAD catalog number
        name: Satellite name (optional, from line 0)
        epoch: TLE epoch as datetime
        line1: First line of TLE (69 chars)
        line2: Second line of TLE (69 chars)
    """

    satellite_id: int
    name: str
    epoch: datetime
    line1: str
    line2: str

    def __post_init__(self) -> None:
        """Validate TLE line lengths."""
        if len(self.line1) != 69:
            raise ValueError(f"TLE line 1 must be 69 characters, got {len(self.line1)}")
        if len(self.line2) != 69:
            raise ValueError(f"TLE line 2 must be 69 characters, got {len(self.line2)}")
        if not self.line1.startswith("1 "):
            raise ValueError("TLE line 1 must start with '1 '")
        if not self.line2.startswith("2 "):
            raise ValueError("TLE line 2 must start with '2 '")


@dataclass(frozen=True)
class StateVector:
    """Satellite state vector (position and velocity at epoch).

    Attributes:
        position: Position vector in ECI frame (km)
        velocity: Velocity vector in ECI frame (km/s)
        epoch: Time at which state is valid
    """

    position: Vector3D
    velocity: Vector3D
    epoch: datetime

    def __post_init__(self) -> None:
        """Validate state vector reference frames."""
        if self.position.frame != "ECI":
            raise ValueError(f"Position must be in ECI frame, got {self.position.frame}")
        if self.velocity.frame != "ECI":
            raise ValueError(f"Velocity must be in ECI frame, got {self.velocity.frame}")

    def specific_energy(self) -> float:
        """Compute specific orbital energy (km²/s²).

        E = v²/2 - μ/r

        Returns:
            Specific orbital energy (negative for bound orbits)
        """
        mu = 398600.4418  # Earth's gravitational parameter (km³/s²)
        r = self.position.magnitude()
        v = self.velocity.magnitude()
        return (v**2) / 2 - mu / r

    def specific_angular_momentum(self) -> Vector3D:
        """Compute specific angular momentum vector (km²/s).

        h = r × v

        Returns:
            Angular momentum vector in ECI frame
        """
        return self.position.cross(self.velocity)

    def semi_major_axis(self) -> float:
        """Compute semi-major axis (km).

        a = -μ / (2E)

        Returns:
            Semi-major axis in kilometers

        Raises:
            ValueError: If orbit is parabolic (E = 0)
        """
        mu = 398600.4418
        energy = self.specific_energy()
        if abs(energy) < 1e-10:
            raise ValueError("Parabolic orbit has undefined semi-major axis")
        return -mu / (2 * energy)

    def orbital_period(self) -> timedelta:
        """Compute orbital period.

        T = 2π√(a³/μ)

        Returns:
            Orbital period as timedelta

        Raises:
            ValueError: If orbit is not elliptical (a <= 0)
        """
        mu = 398600.4418
        a = self.semi_major_axis()
        if a <= 0:
            raise ValueError("Hyperbolic orbit has no defined period")
        t_seconds = 2 * math.pi * math.sqrt(a**3 / mu)
        return timedelta(seconds=t_seconds)


@dataclass(frozen=True)
class OrbitalElements:
    """Classical Keplerian orbital elements.

    Attributes:
        semi_major_axis: Semi-major axis (km)
        eccentricity: Orbital eccentricity (dimensionless, 0 <= e < 1 for ellipse)
        inclination: Inclination (degrees, 0-180)
        raan: Right Ascension of Ascending Node (degrees, 0-360)
        arg_perigee: Argument of perigee (degrees, 0-360)
        true_anomaly: True anomaly (degrees, 0-360)
        epoch: Time at which elements are valid
    """

    semi_major_axis: Kilometers
    eccentricity: float
    inclination: Degrees
    raan: Degrees
    arg_perigee: Degrees
    true_anomaly: Degrees
    epoch: datetime


# ============================================================================
# Contact and Topology Types
# ============================================================================


@dataclass(frozen=True)
class ContactWindow:
    """Communication window between two satellites.

    Represents a period during which two satellites can communicate,
    determined by line-of-sight and link budget constraints.

    Attributes:
        satellite_a: Node ID of first satellite
        satellite_b: Node ID of second satellite
        aos: Acquisition of Signal time
        los: Loss of Signal time
        max_elevation: Maximum elevation angle (degrees)
        min_range: Minimum range during contact (km)
    """

    satellite_a: NodeID
    satellite_b: NodeID
    aos: datetime
    los: datetime
    max_elevation: Degrees
    min_range: Kilometers

    def __post_init__(self) -> None:
        """Validate contact window temporal ordering."""
        if self.los <= self.aos:
            raise ValueError(f"LOS ({self.los}) must be after AOS ({self.aos})")
        if self.max_elevation < 0 or self.max_elevation > 90:
            raise ValueError(f"Max elevation must be 0-90 degrees, got {self.max_elevation}")
        if self.min_range < 0:
            raise ValueError(f"Min range must be non-negative, got {self.min_range}")

    @property
    def duration(self) -> timedelta:
        """Duration of the contact window."""
        return self.los - self.aos

    def contains(self, timestamp: datetime) -> bool:
        """Check if timestamp falls within contact window."""
        return self.aos <= timestamp < self.los

    def overlaps(self, other: ContactWindow) -> bool:
        """Check if this window overlaps with another."""
        return self.aos < other.los and other.aos < self.los


@dataclass(frozen=True)
class TopologyEdge:
    """Edge in the constellation topology graph.

    Represents a communication link between two nodes with associated
    quality metrics.

    Attributes:
        source: Source node ID
        target: Target node ID
        contact: Associated contact window
        weight: Edge weight (typically inverse of link quality)
    """

    source: NodeID
    target: NodeID
    contact: ContactWindow
    weight: float = 1.0


@dataclass(frozen=True)
class TopologyGraph:
    """Snapshot of constellation topology at a point in time.

    Represents the communication graph for federated learning.
    Used by Flame's Topology Abstraction Graph (TAG) layer.

    Attributes:
        nodes: Set of node IDs in the topology
        edges: Tuple of topology edges
        timestamp: Time at which topology is valid
        horizon: How far ahead this topology is valid
    """

    nodes: frozenset[NodeID]
    edges: tuple[TopologyEdge, ...]
    timestamp: datetime
    horizon: timedelta

    def neighbors(self, node: NodeID) -> frozenset[NodeID]:
        """Get all neighbors of a node."""
        return frozenset(
            e.target if e.source == node else e.source
            for e in self.edges
            if e.source == node or e.target == node
        )

    def degree(self, node: NodeID) -> int:
        """Get the degree (number of neighbors) of a node."""
        return len(self.neighbors(node))

    def is_connected(self, source: NodeID, target: NodeID) -> bool:
        """Check if two nodes are directly connected."""
        return any(
            (e.source == source and e.target == target)
            or (e.source == target and e.target == source)
            for e in self.edges
        )


# ============================================================================
# Power and Eclipse Types
# ============================================================================


class EclipseState(Enum):
    """Satellite eclipse state relative to Sun.

    Determines power availability for computing workloads.
    """

    IN_SUN = auto()
    """Full sunlight - maximum solar power available."""

    IN_SHADOW = auto()
    """Full shadow (umbra) - battery power only."""

    PENUMBRA = auto()
    """Partial shadow - reduced solar power."""


class PowerState(Enum):
    """Satellite power budget state.

    Determines what workloads can be executed.
    """

    NOMINAL = auto()
    """Full power - all workloads available."""

    THROTTLED = auto()
    """Reduced power - defer non-critical workloads."""

    CRITICAL = auto()
    """Critical power - essential operations only."""


@dataclass(frozen=True)
class PowerBudget:
    """Power budget for satellite operations.

    Attributes:
        eclipse_state: Current eclipse state
        power_state: Current power budget state
        battery_soc: Battery state of charge (0.0-1.0)
        available_watts: Power available for compute (W)
        timestamp: Time of measurement
    """

    eclipse_state: EclipseState
    power_state: PowerState
    battery_soc: float
    available_watts: float
    timestamp: datetime

    def __post_init__(self) -> None:
        """Validate power budget values."""
        if not 0.0 <= self.battery_soc <= 1.0:
            raise ValueError(f"Battery SoC must be 0.0-1.0, got {self.battery_soc}")
        if self.available_watts < 0:
            raise ValueError(f"Available watts must be non-negative, got {self.available_watts}")


# ============================================================================
# Degradation Types
# ============================================================================


class DegradationLevel(Enum):
    """System degradation level for graceful degradation.

    Used by inference service to select appropriate model complexity.
    """

    FULL = auto()
    """Full capability - use primary model."""

    DEGRADED = auto()
    """Reduced capability - use simplified model."""

    MINIMAL = auto()
    """Minimal capability - use fallback/heuristic."""

    OFFLINE = auto()
    """Offline - no inference available."""


@dataclass(frozen=True)
class SystemHealth:
    """Overall system health status.

    Attributes:
        degradation_level: Current degradation level
        components: Health status of individual components
        timestamp: Time of assessment
    """

    degradation_level: DegradationLevel
    components: tuple[tuple[str, bool], ...]
    timestamp: datetime

    @property
    def healthy_components(self) -> tuple[str, ...]:
        """Get names of healthy components."""
        return tuple(name for name, healthy in self.components if healthy)

    @property
    def unhealthy_components(self) -> tuple[str, ...]:
        """Get names of unhealthy components."""
        return tuple(name for name, healthy in self.components if not healthy)


# ============================================================================
# Federated Learning Types
# ============================================================================


@dataclass(frozen=True)
class ModelGradient:
    """Gradient update from local training.

    Attributes:
        node_id: Node that produced this gradient
        round_id: Training round number
        layer_gradients: Mapping of layer name to gradient bytes
        sample_count: Number of samples used in training
        timestamp: When gradient was computed
    """

    node_id: NodeID
    round_id: int
    layer_gradients: tuple[tuple[str, bytes], ...]
    sample_count: int
    timestamp: datetime


@dataclass(frozen=True)
class AggregatedModel:
    """Aggregated model weights after federation round.

    Attributes:
        round_id: Federation round number
        model_weights: Serialized model weights
        participating_nodes: Nodes that contributed to aggregation
        total_samples: Total samples across all contributors
        timestamp: When aggregation completed
    """

    round_id: int
    model_weights: bytes
    participating_nodes: tuple[NodeID, ...]
    total_samples: int
    timestamp: datetime


# ============================================================================
# Sensor Types
# ============================================================================


@dataclass(frozen=True)
class SensorReading:
    """Generic sensor reading with metadata.

    Attributes:
        sensor_id: Sensor identifier
        value: Reading value
        unit: Unit of measurement
        timestamp: Time of reading
        quality: Quality indicator (0.0-1.0)
    """

    sensor_id: str
    value: float
    unit: str
    timestamp: datetime
    quality: float = 1.0

    def __post_init__(self) -> None:
        """Validate quality range."""
        if not 0.0 <= self.quality <= 1.0:
            raise ValueError(f"Quality must be 0.0-1.0, got {self.quality}")


@dataclass(frozen=True)
class ImageFrame:
    """Earth observation image frame metadata.

    Attributes:
        frame_id: Unique frame identifier
        timestamp: Capture timestamp
        dimensions: (width, height) in pixels
        cloud_cover: Estimated cloud cover (0.0-1.0)
        data_path: Path or reference to image data
    """

    frame_id: str
    timestamp: datetime
    dimensions: tuple[int, int]
    cloud_cover: float
    data_path: str

    def __post_init__(self) -> None:
        """Validate cloud cover range."""
        if not 0.0 <= self.cloud_cover <= 1.0:
            raise ValueError(f"Cloud cover must be 0.0-1.0, got {self.cloud_cover}")

    @property
    def should_discard(self) -> bool:
        """Check if frame should be discarded (>90% cloud cover)."""
        return self.cloud_cover > 0.9
