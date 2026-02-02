"""Pytest configuration and shared fixtures for KEKOA tests."""

from datetime import datetime, timezone

import pytest
from hypothesis import settings

from kekoa.core.types import (
    ContactWindow,
    DegradationLevel,
    EclipseState,
    PowerState,
    StateVector,
    TLESet,
    Vector3D,
)


# Configure Hypothesis for property-based testing
settings.register_profile("ci", max_examples=500, deadline=None)
settings.register_profile("dev", max_examples=100, deadline=5000)
settings.register_profile("quick", max_examples=10, deadline=1000)
settings.load_profile("dev")


# ============================================================================
# Time fixtures
# ============================================================================


@pytest.fixture
def epoch() -> datetime:
    """Standard test epoch (J2000)."""
    return datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def current_epoch() -> datetime:
    """Current time epoch for tests."""
    return datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)


# ============================================================================
# Vector fixtures
# ============================================================================


@pytest.fixture
def origin_vector() -> Vector3D:
    """Zero vector at origin in ECI frame."""
    return Vector3D(x=0.0, y=0.0, z=0.0, frame="ECI")


@pytest.fixture
def unit_x_vector() -> Vector3D:
    """Unit vector along X-axis in ECI frame."""
    return Vector3D(x=1.0, y=0.0, z=0.0, frame="ECI")


@pytest.fixture
def leo_position() -> Vector3D:
    """Typical LEO satellite position (400km altitude)."""
    return Vector3D(x=6778.0, y=0.0, z=0.0, frame="ECI")


@pytest.fixture
def leo_velocity() -> Vector3D:
    """Typical LEO satellite velocity (~7.7 km/s)."""
    return Vector3D(x=0.0, y=7.669, z=0.0, frame="ECI")


# ============================================================================
# TLE fixtures
# ============================================================================


@pytest.fixture
def iss_tle() -> TLESet:
    """International Space Station TLE (sample)."""
    return TLESet(
        satellite_id=25544,
        name="ISS (ZARYA)",
        epoch=datetime(2024, 1, 15, 0, 0, 0, tzinfo=timezone.utc),
        line1="1 25544U 98067A   24015.50000000  .00016717  00000-0  10270-3 0  9003",
        line2="2 25544  51.6400 247.4627 0006703 130.5360 325.0288 15.49815350 82563",
    )


@pytest.fixture
def starlink_tle() -> TLESet:
    """Starlink satellite TLE (sample)."""
    return TLESet(
        satellite_id=44713,
        name="STARLINK-1007",
        epoch=datetime(2024, 1, 15, 0, 0, 0, tzinfo=timezone.utc),
        line1="1 44713U 19074A   24015.50000000  .00001234  00000-0  12345-4 0  9999",
        line2="2 44713  53.0000 180.0000 0001234  90.0000 270.0000 15.05000000 12345",
    )


@pytest.fixture
def sample_tle_lines() -> tuple[str, str]:
    """Raw TLE lines for parsing tests."""
    line1 = "1 25544U 98067A   24015.50000000  .00016717  00000-0  10270-3 0  9003"
    line2 = "2 25544  51.6400 247.4627 0006703 130.5360 325.0288 15.49815350 82563"
    return line1, line2


# ============================================================================
# StateVector fixtures
# ============================================================================


@pytest.fixture
def leo_state_vector(leo_position: Vector3D, leo_velocity: Vector3D, epoch: datetime) -> StateVector:
    """LEO satellite state vector."""
    return StateVector(position=leo_position, velocity=leo_velocity, epoch=epoch)


# ============================================================================
# ContactWindow fixtures
# ============================================================================


@pytest.fixture
def sample_contact_window(current_epoch: datetime) -> ContactWindow:
    """Sample contact window between two satellites."""
    from datetime import timedelta

    return ContactWindow(
        satellite_a="SAT-001",
        satellite_b="SAT-002",
        aos=current_epoch,
        los=current_epoch + timedelta(minutes=10),
        max_elevation=45.0,
        min_range=500.0,
    )


# ============================================================================
# Enum fixtures
# ============================================================================


@pytest.fixture
def all_eclipse_states() -> list[EclipseState]:
    """All possible eclipse states."""
    return list(EclipseState)


@pytest.fixture
def all_power_states() -> list[PowerState]:
    """All possible power states."""
    return list(PowerState)


@pytest.fixture
def all_degradation_levels() -> list[DegradationLevel]:
    """All possible degradation levels."""
    return list(DegradationLevel)


# ============================================================================
# Determinism helpers
# ============================================================================


@pytest.fixture
def fixed_seed() -> int:
    """Fixed random seed for reproducible tests."""
    return 42
