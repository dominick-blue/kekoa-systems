"""Unit tests for core types."""

from datetime import datetime, timedelta, timezone

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from kekoa.core.types import (
    ContactWindow,
    DegradationLevel,
    EclipseState,
    ImageFrame,
    OrbitalElements,
    PowerBudget,
    PowerState,
    SensorReading,
    StateVector,
    TLESet,
    TopologyEdge,
    TopologyGraph,
    Vector3D,
)


# ============================================================================
# Vector3D Tests
# ============================================================================


class TestVector3D:
    """Tests for Vector3D type."""

    def test_creation(self) -> None:
        """Test basic vector creation."""
        v = Vector3D(x=1.0, y=2.0, z=3.0, frame="ECI")
        assert v.x == 1.0
        assert v.y == 2.0
        assert v.z == 3.0
        assert v.frame == "ECI"

    def test_immutability(self) -> None:
        """Test that vectors are immutable."""
        v = Vector3D(x=1.0, y=2.0, z=3.0, frame="ECI")
        with pytest.raises(AttributeError):
            v.x = 5.0  # type: ignore[misc]

    def test_magnitude(self) -> None:
        """Test magnitude calculation."""
        v = Vector3D(x=3.0, y=4.0, z=0.0, frame="ECI")
        assert v.magnitude() == 5.0

    def test_magnitude_zero_vector(self) -> None:
        """Test magnitude of zero vector."""
        v = Vector3D(x=0.0, y=0.0, z=0.0, frame="ECI")
        assert v.magnitude() == 0.0

    def test_dot_product(self) -> None:
        """Test dot product."""
        v1 = Vector3D(x=1.0, y=2.0, z=3.0, frame="ECI")
        v2 = Vector3D(x=4.0, y=5.0, z=6.0, frame="ECI")
        assert v1.dot(v2) == 32.0  # 1*4 + 2*5 + 3*6

    def test_dot_product_orthogonal(self) -> None:
        """Test dot product of orthogonal vectors."""
        v1 = Vector3D(x=1.0, y=0.0, z=0.0, frame="ECI")
        v2 = Vector3D(x=0.0, y=1.0, z=0.0, frame="ECI")
        assert v1.dot(v2) == 0.0

    def test_dot_product_frame_mismatch(self) -> None:
        """Test dot product with different frames raises error."""
        v1 = Vector3D(x=1.0, y=2.0, z=3.0, frame="ECI")
        v2 = Vector3D(x=4.0, y=5.0, z=6.0, frame="ECEF")
        with pytest.raises(ValueError, match="different frames"):
            v1.dot(v2)

    def test_cross_product(self) -> None:
        """Test cross product."""
        v1 = Vector3D(x=1.0, y=0.0, z=0.0, frame="ECI")
        v2 = Vector3D(x=0.0, y=1.0, z=0.0, frame="ECI")
        result = v1.cross(v2)
        assert result.x == 0.0
        assert result.y == 0.0
        assert result.z == 1.0
        assert result.frame == "ECI"

    def test_cross_product_frame_mismatch(self) -> None:
        """Test cross product with different frames raises error."""
        v1 = Vector3D(x=1.0, y=0.0, z=0.0, frame="ECI")
        v2 = Vector3D(x=0.0, y=1.0, z=0.0, frame="BODY")
        with pytest.raises(ValueError, match="different frames"):
            v1.cross(v2)

    def test_scale(self) -> None:
        """Test vector scaling."""
        v = Vector3D(x=1.0, y=2.0, z=3.0, frame="ECI")
        scaled = v.scale(2.0)
        assert scaled.x == 2.0
        assert scaled.y == 4.0
        assert scaled.z == 6.0
        assert scaled.frame == "ECI"

    def test_add(self) -> None:
        """Test vector addition."""
        v1 = Vector3D(x=1.0, y=2.0, z=3.0, frame="ECI")
        v2 = Vector3D(x=4.0, y=5.0, z=6.0, frame="ECI")
        result = v1.add(v2)
        assert result.x == 5.0
        assert result.y == 7.0
        assert result.z == 9.0

    def test_subtract(self) -> None:
        """Test vector subtraction."""
        v1 = Vector3D(x=4.0, y=5.0, z=6.0, frame="ECI")
        v2 = Vector3D(x=1.0, y=2.0, z=3.0, frame="ECI")
        result = v1.subtract(v2)
        assert result.x == 3.0
        assert result.y == 3.0
        assert result.z == 3.0

    def test_normalize(self) -> None:
        """Test vector normalization."""
        v = Vector3D(x=3.0, y=4.0, z=0.0, frame="ECI")
        unit = v.normalize()
        assert abs(unit.magnitude() - 1.0) < 1e-10
        assert abs(unit.x - 0.6) < 1e-10
        assert abs(unit.y - 0.8) < 1e-10

    def test_normalize_zero_vector(self) -> None:
        """Test normalizing zero vector raises error."""
        v = Vector3D(x=0.0, y=0.0, z=0.0, frame="ECI")
        with pytest.raises(ValueError, match="zero vector"):
            v.normalize()

    def test_to_tuple(self) -> None:
        """Test conversion to tuple."""
        v = Vector3D(x=1.0, y=2.0, z=3.0, frame="ECI")
        assert v.to_tuple() == (1.0, 2.0, 3.0)

    @given(
        x=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
        y=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
        z=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
    )
    @settings(max_examples=100)
    def test_magnitude_non_negative(self, x: float, y: float, z: float) -> None:
        """Property: magnitude is always non-negative."""
        v = Vector3D(x=x, y=y, z=z, frame="ECI")
        assert v.magnitude() >= 0.0

    @given(
        x=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
        y=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
        z=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
    )
    @settings(max_examples=100)
    def test_dot_product_commutative(self, x: float, y: float, z: float) -> None:
        """Property: dot product is commutative."""
        v1 = Vector3D(x=x, y=y, z=z, frame="ECI")
        v2 = Vector3D(x=z, y=x, z=y, frame="ECI")  # Different vector
        assert abs(v1.dot(v2) - v2.dot(v1)) < 1e-10


# ============================================================================
# TLESet Tests
# ============================================================================


class TestTLESet:
    """Tests for TLESet type."""

    def test_creation(self, iss_tle: TLESet) -> None:
        """Test TLE creation."""
        assert iss_tle.satellite_id == 25544
        assert iss_tle.name == "ISS (ZARYA)"
        assert len(iss_tle.line1) == 69
        assert len(iss_tle.line2) == 69

    def test_invalid_line1_length(self) -> None:
        """Test validation of line 1 length."""
        with pytest.raises(ValueError, match="line 1 must be 69"):
            TLESet(
                satellite_id=25544,
                name="TEST",
                epoch=datetime.now(timezone.utc),
                line1="1 25544U 98067A",  # Too short
                line2="2 25544  51.6400 247.4627 0006703 130.5360 325.0288 15.49815350 82563",
            )

    def test_invalid_line2_length(self) -> None:
        """Test validation of line 2 length."""
        with pytest.raises(ValueError, match="line 2 must be 69"):
            TLESet(
                satellite_id=25544,
                name="TEST",
                epoch=datetime.now(timezone.utc),
                line1="1 25544U 98067A   24015.50000000  .00016717  00000-0  10270-3 0  9003",
                line2="2 25544",  # Too short
            )

    def test_invalid_line1_prefix(self) -> None:
        """Test validation of line 1 prefix."""
        with pytest.raises(ValueError, match="line 1 must start with"):
            TLESet(
                satellite_id=25544,
                name="TEST",
                epoch=datetime.now(timezone.utc),
                line1="2 25544U 98067A   24015.50000000  .00016717  00000-0  10270-3 0  9003",
                line2="2 25544  51.6400 247.4627 0006703 130.5360 325.0288 15.49815350 82563",
            )

    def test_invalid_line2_prefix(self) -> None:
        """Test validation of line 2 prefix."""
        with pytest.raises(ValueError, match="line 2 must start with"):
            TLESet(
                satellite_id=25544,
                name="TEST",
                epoch=datetime.now(timezone.utc),
                line1="1 25544U 98067A   24015.50000000  .00016717  00000-0  10270-3 0  9003",
                line2="1 25544  51.6400 247.4627 0006703 130.5360 325.0288 15.49815350 82563",
            )


# ============================================================================
# StateVector Tests
# ============================================================================


class TestStateVector:
    """Tests for StateVector type."""

    def test_creation(self, leo_state_vector: StateVector) -> None:
        """Test state vector creation."""
        assert leo_state_vector.position.frame == "ECI"
        assert leo_state_vector.velocity.frame == "ECI"

    def test_invalid_position_frame(self) -> None:
        """Test that non-ECI position is rejected."""
        with pytest.raises(ValueError, match="Position must be in ECI"):
            StateVector(
                position=Vector3D(x=6778.0, y=0.0, z=0.0, frame="ECEF"),
                velocity=Vector3D(x=0.0, y=7.669, z=0.0, frame="ECI"),
                epoch=datetime.now(timezone.utc),
            )

    def test_invalid_velocity_frame(self) -> None:
        """Test that non-ECI velocity is rejected."""
        with pytest.raises(ValueError, match="Velocity must be in ECI"):
            StateVector(
                position=Vector3D(x=6778.0, y=0.0, z=0.0, frame="ECI"),
                velocity=Vector3D(x=0.0, y=7.669, z=0.0, frame="BODY"),
                epoch=datetime.now(timezone.utc),
            )

    def test_specific_energy(self, leo_state_vector: StateVector) -> None:
        """Test specific orbital energy calculation."""
        energy = leo_state_vector.specific_energy()
        # LEO should have negative energy (bound orbit)
        assert energy < 0

    def test_specific_angular_momentum(self, leo_state_vector: StateVector) -> None:
        """Test specific angular momentum calculation."""
        h = leo_state_vector.specific_angular_momentum()
        assert h.frame == "ECI"
        # For LEO with circular orbit, h ≈ r × v
        assert h.magnitude() > 0

    def test_semi_major_axis(self, leo_state_vector: StateVector) -> None:
        """Test semi-major axis calculation."""
        a = leo_state_vector.semi_major_axis()
        # LEO altitude ~400km, so a should be ~6778km
        assert 6500 < a < 7000

    def test_orbital_period(self, leo_state_vector: StateVector) -> None:
        """Test orbital period calculation."""
        period = leo_state_vector.orbital_period()
        # LEO period is ~90 minutes
        assert timedelta(minutes=85) < period < timedelta(minutes=95)


# ============================================================================
# ContactWindow Tests
# ============================================================================


class TestContactWindow:
    """Tests for ContactWindow type."""

    def test_creation(self, sample_contact_window: ContactWindow) -> None:
        """Test contact window creation."""
        assert sample_contact_window.satellite_a == "SAT-001"
        assert sample_contact_window.satellite_b == "SAT-002"

    def test_duration(self, sample_contact_window: ContactWindow) -> None:
        """Test duration property."""
        assert sample_contact_window.duration == timedelta(minutes=10)

    def test_contains_inside(self, sample_contact_window: ContactWindow, current_epoch: datetime) -> None:
        """Test contains for timestamp inside window."""
        inside = current_epoch + timedelta(minutes=5)
        assert sample_contact_window.contains(inside)

    def test_contains_outside(self, sample_contact_window: ContactWindow, current_epoch: datetime) -> None:
        """Test contains for timestamp outside window."""
        outside = current_epoch + timedelta(minutes=15)
        assert not sample_contact_window.contains(outside)

    def test_contains_at_aos(self, sample_contact_window: ContactWindow) -> None:
        """Test contains at AOS (should be included)."""
        assert sample_contact_window.contains(sample_contact_window.aos)

    def test_contains_at_los(self, sample_contact_window: ContactWindow) -> None:
        """Test contains at LOS (should not be included)."""
        assert not sample_contact_window.contains(sample_contact_window.los)

    def test_invalid_temporal_order(self, current_epoch: datetime) -> None:
        """Test that LOS before AOS is rejected."""
        with pytest.raises(ValueError, match="LOS .* must be after AOS"):
            ContactWindow(
                satellite_a="SAT-001",
                satellite_b="SAT-002",
                aos=current_epoch + timedelta(minutes=10),
                los=current_epoch,  # Before AOS
                max_elevation=45.0,
                min_range=500.0,
            )

    def test_invalid_elevation(self, current_epoch: datetime) -> None:
        """Test invalid elevation is rejected."""
        with pytest.raises(ValueError, match="Max elevation must be 0-90"):
            ContactWindow(
                satellite_a="SAT-001",
                satellite_b="SAT-002",
                aos=current_epoch,
                los=current_epoch + timedelta(minutes=10),
                max_elevation=100.0,  # Invalid
                min_range=500.0,
            )

    def test_invalid_range(self, current_epoch: datetime) -> None:
        """Test negative range is rejected."""
        with pytest.raises(ValueError, match="Min range must be non-negative"):
            ContactWindow(
                satellite_a="SAT-001",
                satellite_b="SAT-002",
                aos=current_epoch,
                los=current_epoch + timedelta(minutes=10),
                max_elevation=45.0,
                min_range=-100.0,  # Invalid
            )

    def test_overlaps_true(self, current_epoch: datetime) -> None:
        """Test overlapping windows."""
        w1 = ContactWindow(
            satellite_a="SAT-001",
            satellite_b="SAT-002",
            aos=current_epoch,
            los=current_epoch + timedelta(minutes=10),
            max_elevation=45.0,
            min_range=500.0,
        )
        w2 = ContactWindow(
            satellite_a="SAT-001",
            satellite_b="SAT-003",
            aos=current_epoch + timedelta(minutes=5),
            los=current_epoch + timedelta(minutes=15),
            max_elevation=45.0,
            min_range=500.0,
        )
        assert w1.overlaps(w2)
        assert w2.overlaps(w1)

    def test_overlaps_false(self, current_epoch: datetime) -> None:
        """Test non-overlapping windows."""
        w1 = ContactWindow(
            satellite_a="SAT-001",
            satellite_b="SAT-002",
            aos=current_epoch,
            los=current_epoch + timedelta(minutes=10),
            max_elevation=45.0,
            min_range=500.0,
        )
        w2 = ContactWindow(
            satellite_a="SAT-001",
            satellite_b="SAT-003",
            aos=current_epoch + timedelta(minutes=15),
            los=current_epoch + timedelta(minutes=25),
            max_elevation=45.0,
            min_range=500.0,
        )
        assert not w1.overlaps(w2)
        assert not w2.overlaps(w1)


# ============================================================================
# TopologyGraph Tests
# ============================================================================


class TestTopologyGraph:
    """Tests for TopologyGraph type."""

    @pytest.fixture
    def simple_topology(self, sample_contact_window: ContactWindow, current_epoch: datetime) -> TopologyGraph:
        """Create a simple topology for testing."""
        edge = TopologyEdge(
            source="SAT-001",
            target="SAT-002",
            contact=sample_contact_window,
            weight=1.0,
        )
        return TopologyGraph(
            nodes=frozenset({"SAT-001", "SAT-002", "SAT-003"}),
            edges=(edge,),
            timestamp=current_epoch,
            horizon=timedelta(minutes=90),
        )

    def test_neighbors(self, simple_topology: TopologyGraph) -> None:
        """Test neighbor lookup."""
        neighbors = simple_topology.neighbors("SAT-001")
        assert "SAT-002" in neighbors
        assert "SAT-003" not in neighbors

    def test_neighbors_isolated(self, simple_topology: TopologyGraph) -> None:
        """Test neighbors of isolated node."""
        neighbors = simple_topology.neighbors("SAT-003")
        assert len(neighbors) == 0

    def test_degree(self, simple_topology: TopologyGraph) -> None:
        """Test degree calculation."""
        assert simple_topology.degree("SAT-001") == 1
        assert simple_topology.degree("SAT-002") == 1
        assert simple_topology.degree("SAT-003") == 0

    def test_is_connected_true(self, simple_topology: TopologyGraph) -> None:
        """Test connected nodes."""
        assert simple_topology.is_connected("SAT-001", "SAT-002")
        assert simple_topology.is_connected("SAT-002", "SAT-001")

    def test_is_connected_false(self, simple_topology: TopologyGraph) -> None:
        """Test unconnected nodes."""
        assert not simple_topology.is_connected("SAT-001", "SAT-003")


# ============================================================================
# Enum Tests
# ============================================================================


class TestEnums:
    """Tests for enum types."""

    def test_eclipse_states(self, all_eclipse_states: list[EclipseState]) -> None:
        """Test all eclipse states exist."""
        assert EclipseState.IN_SUN in all_eclipse_states
        assert EclipseState.IN_SHADOW in all_eclipse_states
        assert EclipseState.PENUMBRA in all_eclipse_states
        assert len(all_eclipse_states) == 3

    def test_power_states(self, all_power_states: list[PowerState]) -> None:
        """Test all power states exist."""
        assert PowerState.NOMINAL in all_power_states
        assert PowerState.THROTTLED in all_power_states
        assert PowerState.CRITICAL in all_power_states
        assert len(all_power_states) == 3

    def test_degradation_levels(self, all_degradation_levels: list[DegradationLevel]) -> None:
        """Test all degradation levels exist."""
        assert DegradationLevel.FULL in all_degradation_levels
        assert DegradationLevel.DEGRADED in all_degradation_levels
        assert DegradationLevel.MINIMAL in all_degradation_levels
        assert DegradationLevel.OFFLINE in all_degradation_levels
        assert len(all_degradation_levels) == 4


# ============================================================================
# PowerBudget Tests
# ============================================================================


class TestPowerBudget:
    """Tests for PowerBudget type."""

    def test_creation(self, current_epoch: datetime) -> None:
        """Test power budget creation."""
        pb = PowerBudget(
            eclipse_state=EclipseState.IN_SUN,
            power_state=PowerState.NOMINAL,
            battery_soc=0.8,
            available_watts=50.0,
            timestamp=current_epoch,
        )
        assert pb.battery_soc == 0.8
        assert pb.available_watts == 50.0

    def test_invalid_soc_low(self, current_epoch: datetime) -> None:
        """Test invalid low SoC is rejected."""
        with pytest.raises(ValueError, match="Battery SoC must be 0.0-1.0"):
            PowerBudget(
                eclipse_state=EclipseState.IN_SUN,
                power_state=PowerState.NOMINAL,
                battery_soc=-0.1,
                available_watts=50.0,
                timestamp=current_epoch,
            )

    def test_invalid_soc_high(self, current_epoch: datetime) -> None:
        """Test invalid high SoC is rejected."""
        with pytest.raises(ValueError, match="Battery SoC must be 0.0-1.0"):
            PowerBudget(
                eclipse_state=EclipseState.IN_SUN,
                power_state=PowerState.NOMINAL,
                battery_soc=1.5,
                available_watts=50.0,
                timestamp=current_epoch,
            )

    def test_invalid_watts(self, current_epoch: datetime) -> None:
        """Test negative watts is rejected."""
        with pytest.raises(ValueError, match="Available watts must be non-negative"):
            PowerBudget(
                eclipse_state=EclipseState.IN_SUN,
                power_state=PowerState.NOMINAL,
                battery_soc=0.8,
                available_watts=-10.0,
                timestamp=current_epoch,
            )


# ============================================================================
# ImageFrame Tests
# ============================================================================


class TestImageFrame:
    """Tests for ImageFrame type."""

    def test_creation(self, current_epoch: datetime) -> None:
        """Test image frame creation."""
        frame = ImageFrame(
            frame_id="IMG-001",
            timestamp=current_epoch,
            dimensions=(1920, 1080),
            cloud_cover=0.3,
            data_path="/data/images/img001.tif",
        )
        assert frame.frame_id == "IMG-001"
        assert frame.dimensions == (1920, 1080)

    def test_should_discard_high_cloud(self, current_epoch: datetime) -> None:
        """Test discard threshold for high cloud cover."""
        frame = ImageFrame(
            frame_id="IMG-001",
            timestamp=current_epoch,
            dimensions=(1920, 1080),
            cloud_cover=0.95,
            data_path="/data/images/img001.tif",
        )
        assert frame.should_discard

    def test_should_not_discard_low_cloud(self, current_epoch: datetime) -> None:
        """Test discard threshold for low cloud cover."""
        frame = ImageFrame(
            frame_id="IMG-001",
            timestamp=current_epoch,
            dimensions=(1920, 1080),
            cloud_cover=0.3,
            data_path="/data/images/img001.tif",
        )
        assert not frame.should_discard

    def test_should_discard_at_threshold(self, current_epoch: datetime) -> None:
        """Test discard at exactly 90% cloud cover."""
        frame = ImageFrame(
            frame_id="IMG-001",
            timestamp=current_epoch,
            dimensions=(1920, 1080),
            cloud_cover=0.9,
            data_path="/data/images/img001.tif",
        )
        assert not frame.should_discard  # 0.9 is not > 0.9

    def test_invalid_cloud_cover(self, current_epoch: datetime) -> None:
        """Test invalid cloud cover is rejected."""
        with pytest.raises(ValueError, match="Cloud cover must be 0.0-1.0"):
            ImageFrame(
                frame_id="IMG-001",
                timestamp=current_epoch,
                dimensions=(1920, 1080),
                cloud_cover=1.5,
                data_path="/data/images/img001.tif",
            )


# ============================================================================
# SensorReading Tests
# ============================================================================


class TestSensorReading:
    """Tests for SensorReading type."""

    def test_creation(self, current_epoch: datetime) -> None:
        """Test sensor reading creation."""
        reading = SensorReading(
            sensor_id="TEMP-001",
            value=25.5,
            unit="celsius",
            timestamp=current_epoch,
            quality=0.95,
        )
        assert reading.sensor_id == "TEMP-001"
        assert reading.value == 25.5
        assert reading.quality == 0.95

    def test_default_quality(self, current_epoch: datetime) -> None:
        """Test default quality is 1.0."""
        reading = SensorReading(
            sensor_id="TEMP-001",
            value=25.5,
            unit="celsius",
            timestamp=current_epoch,
        )
        assert reading.quality == 1.0

    def test_invalid_quality(self, current_epoch: datetime) -> None:
        """Test invalid quality is rejected."""
        with pytest.raises(ValueError, match="Quality must be 0.0-1.0"):
            SensorReading(
                sensor_id="TEMP-001",
                value=25.5,
                unit="celsius",
                timestamp=current_epoch,
                quality=1.5,
            )
