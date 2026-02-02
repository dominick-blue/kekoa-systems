"""Configuration settings for KEKOA.

Uses Pydantic Settings for type-safe configuration with environment
variable support. Settings can be loaded from:
- Environment variables (KEKOA_* prefix)
- .env files
- Direct instantiation

Example:
    # From environment
    export KEKOA_LOG_LEVEL=DEBUG
    settings = KekoaSettings()

    # From .env file
    settings = KekoaSettings(_env_file=".env")

    # Direct
    settings = KekoaSettings(log_level="DEBUG")
"""

from __future__ import annotations

from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ============================================================================
# Enums
# ============================================================================


class LogLevel(str, Enum):
    """Logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AggregationStrategy(str, Enum):
    """Federated learning aggregation strategies."""

    FEDAVG = "fedavg"
    """Federated Averaging - weighted by sample count."""

    FEDPROX = "fedprox"
    """FedProx - adds proximal term for heterogeneous data."""

    SCAFFOLD = "scaffold"
    """SCAFFOLD - variance reduction for non-IID data."""


class CompressionMethod(str, Enum):
    """Gradient compression methods."""

    NONE = "none"
    """No compression."""

    TOPK = "topk"
    """Top-K sparsification."""

    QUANTIZATION = "quantization"
    """Gradient quantization."""

    RANDOM = "random"
    """Random sparsification."""


class ModelComplexity(str, Enum):
    """Model complexity levels for degradation."""

    FULL = "full"
    """Full model - highest accuracy."""

    LITE = "lite"
    """Lite model - reduced parameters."""

    MICRO = "micro"
    """Micro model - minimal footprint."""

    HEURISTIC = "heuristic"
    """Heuristic fallback - no ML."""


# ============================================================================
# Component Settings
# ============================================================================


class OAKSettings(BaseSettings):
    """Orbital Availability Kernel (OAK) settings.

    Controls orbital propagation and scheduling behavior.
    """

    model_config = SettingsConfigDict(
        env_prefix="KEKOA_OAK_",
        extra="ignore",
    )

    # TLE configuration
    tle_source_path: Path = Field(
        default=Path("data/tle"),
        description="Path to TLE data files",
    )
    tle_refresh_interval_hours: float = Field(
        default=24.0,
        ge=0.1,
        le=168.0,
        description="Hours between TLE refresh",
    )
    tle_max_age_days: float = Field(
        default=14.0,
        ge=1.0,
        le=30.0,
        description="Maximum TLE age before warning",
    )

    # Propagation settings
    propagation_horizon_minutes: int = Field(
        default=90,
        ge=10,
        le=1440,
        description="How far ahead to compute contacts",
    )
    propagation_step_seconds: float = Field(
        default=60.0,
        ge=1.0,
        le=300.0,
        description="Time step for propagation",
    )
    max_propagation_error_km: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Maximum acceptable propagation error",
    )

    # Contact calculation
    max_contact_range_km: float = Field(
        default=5000.0,
        ge=100.0,
        le=50000.0,
        description="Maximum ISL range for contact",
    )
    min_elevation_degrees: float = Field(
        default=5.0,
        ge=0.0,
        le=90.0,
        description="Minimum elevation for ground contact",
    )

    # Performance
    max_satellites: int = Field(
        default=1000,
        ge=1,
        le=10000,
        description="Maximum satellites to track",
    )
    contact_cache_ttl_seconds: int = Field(
        default=300,
        ge=60,
        le=3600,
        description="Contact window cache TTL",
    )

    @property
    def tle_refresh_interval(self) -> timedelta:
        """TLE refresh interval as timedelta."""
        return timedelta(hours=self.tle_refresh_interval_hours)

    @property
    def propagation_horizon(self) -> timedelta:
        """Propagation horizon as timedelta."""
        return timedelta(minutes=self.propagation_horizon_minutes)

    @property
    def propagation_step(self) -> timedelta:
        """Propagation step as timedelta."""
        return timedelta(seconds=self.propagation_step_seconds)


class FlameSettings(BaseSettings):
    """Flame federated learning settings.

    Controls federation behavior and communication.
    """

    model_config = SettingsConfigDict(
        env_prefix="KEKOA_FLAME_",
        extra="ignore",
    )

    # Aggregation
    aggregation_strategy: AggregationStrategy = Field(
        default=AggregationStrategy.FEDAVG,
        description="Gradient aggregation strategy",
    )
    min_clients_per_round: int = Field(
        default=3,
        ge=1,
        le=100,
        description="Minimum clients for valid round",
    )
    round_timeout_seconds: int = Field(
        default=300,
        ge=30,
        le=3600,
        description="Maximum time to wait for round",
    )

    # Compression
    compression_method: CompressionMethod = Field(
        default=CompressionMethod.TOPK,
        description="Gradient compression method",
    )
    compression_ratio: float = Field(
        default=0.1,
        ge=0.01,
        le=1.0,
        description="Compression ratio (1.0 = no compression)",
    )

    # FedProx specific
    fedprox_mu: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="FedProx proximal term coefficient",
    )

    # Topology
    topology_update_interval_seconds: int = Field(
        default=60,
        ge=10,
        le=600,
        description="Topology refresh interval",
    )
    max_hops: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum hops for gradient relay",
    )

    # Client selection
    client_selection_strategy: Literal["random", "round_robin", "availability"] = Field(
        default="availability",
        description="How to select clients for rounds",
    )

    @property
    def round_timeout(self) -> timedelta:
        """Round timeout as timedelta."""
        return timedelta(seconds=self.round_timeout_seconds)

    @property
    def topology_update_interval(self) -> timedelta:
        """Topology update interval as timedelta."""
        return timedelta(seconds=self.topology_update_interval_seconds)


class InferenceSettings(BaseSettings):
    """Inference service settings.

    Controls edge inference behavior and degradation.
    """

    model_config = SettingsConfigDict(
        env_prefix="KEKOA_INFERENCE_",
        extra="ignore",
    )

    # Model paths
    model_path: Path = Field(
        default=Path("models"),
        description="Path to model files",
    )
    full_model_name: str = Field(
        default="cloud_detect_full.pt",
        description="Full model filename",
    )
    lite_model_name: str = Field(
        default="cloud_detect_lite.pt",
        description="Lite model filename",
    )
    micro_model_name: str = Field(
        default="cloud_detect_micro.pt",
        description="Micro model filename",
    )

    # Performance
    inference_timeout_ms: int = Field(
        default=50,
        ge=10,
        le=1000,
        description="Maximum inference time",
    )
    batch_size: int = Field(
        default=1,
        ge=1,
        le=32,
        description="Inference batch size",
    )

    # Degradation
    default_complexity: ModelComplexity = Field(
        default=ModelComplexity.FULL,
        description="Default model complexity",
    )
    enable_degradation: bool = Field(
        default=True,
        description="Enable automatic degradation",
    )
    degradation_power_threshold: float = Field(
        default=0.3,
        ge=0.1,
        le=0.9,
        description="Battery SoC to trigger degradation",
    )

    # Thresholds
    cloud_discard_threshold: float = Field(
        default=0.9,
        ge=0.5,
        le=1.0,
        description="Cloud cover threshold for discard",
    )
    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for prediction",
    )

    @property
    def inference_timeout(self) -> timedelta:
        """Inference timeout as timedelta."""
        return timedelta(milliseconds=self.inference_timeout_ms)


class ResilienceSettings(BaseSettings):
    """Resilience pattern settings.

    Controls circuit breakers, retries, and bulkheads.
    """

    model_config = SettingsConfigDict(
        env_prefix="KEKOA_RESILIENCE_",
        extra="ignore",
    )

    # Circuit breaker
    circuit_failure_threshold: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Failures before circuit opens",
    )
    circuit_recovery_timeout_seconds: float = Field(
        default=60.0,
        ge=5.0,
        le=600.0,
        description="Seconds before recovery attempt",
    )
    circuit_half_open_calls: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Test calls in half-open state",
    )

    # Retry
    retry_max_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum retry attempts",
    )
    retry_base_delay_seconds: float = Field(
        default=0.1,
        ge=0.01,
        le=10.0,
        description="Initial retry delay",
    )
    retry_max_delay_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Maximum retry delay",
    )

    # Bulkhead
    bulkhead_max_concurrent: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum concurrent operations",
    )
    bulkhead_max_wait_seconds: float = Field(
        default=5.0,
        ge=0.1,
        le=60.0,
        description="Maximum wait for slot",
    )

    @property
    def circuit_recovery_timeout(self) -> timedelta:
        """Circuit recovery timeout as timedelta."""
        return timedelta(seconds=self.circuit_recovery_timeout_seconds)


class PersistenceSettings(BaseSettings):
    """Persistence settings.

    Controls checkpointing and state storage.
    """

    model_config = SettingsConfigDict(
        env_prefix="KEKOA_PERSISTENCE_",
        extra="ignore",
    )

    # Checkpointing
    checkpoint_path: Path = Field(
        default=Path("checkpoints"),
        description="Path for checkpoint files",
    )
    checkpoint_interval_seconds: int = Field(
        default=300,
        ge=30,
        le=3600,
        description="Checkpoint interval",
    )
    max_checkpoints: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Maximum checkpoints to keep",
    )

    # State store
    state_store_path: Path = Field(
        default=Path("state"),
        description="Path for state store",
    )

    @property
    def checkpoint_interval(self) -> timedelta:
        """Checkpoint interval as timedelta."""
        return timedelta(seconds=self.checkpoint_interval_seconds)


class TelemetrySettings(BaseSettings):
    """Telemetry and observability settings."""

    model_config = SettingsConfigDict(
        env_prefix="KEKOA_TELEMETRY_",
        extra="ignore",
    )

    # Logging
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Logging level",
    )
    log_format: Literal["json", "console"] = Field(
        default="json",
        description="Log output format",
    )
    log_path: Path | None = Field(
        default=None,
        description="Log file path (None for stdout)",
    )

    # Metrics
    enable_metrics: bool = Field(
        default=True,
        description="Enable Prometheus metrics",
    )
    metrics_port: int = Field(
        default=9090,
        ge=1024,
        le=65535,
        description="Metrics server port",
    )

    # Telemetry export
    telemetry_queue_max_size: int = Field(
        default=1000,
        ge=100,
        le=100000,
        description="Max queued telemetry items",
    )
    telemetry_batch_size: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Items per transmission batch",
    )


# ============================================================================
# Main Settings Class
# ============================================================================


class KekoaSettings(BaseSettings):
    """Main KEKOA configuration.

    Aggregates all component settings. Load with:
        settings = KekoaSettings()

    Or from environment:
        export KEKOA_NODE_ID=SAT-001
        settings = KekoaSettings()
    """

    model_config = SettingsConfigDict(
        env_prefix="KEKOA_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Node identity
    node_id: str = Field(
        default="SAT-001",
        pattern=r"^[A-Z]+-\d{3}$",
        description="This satellite's node ID",
    )
    constellation_id: str = Field(
        default="KEKOA-DEV",
        description="Constellation identifier",
    )

    # Component settings
    oak: OAKSettings = Field(default_factory=OAKSettings)
    flame: FlameSettings = Field(default_factory=FlameSettings)
    inference: InferenceSettings = Field(default_factory=InferenceSettings)
    resilience: ResilienceSettings = Field(default_factory=ResilienceSettings)
    persistence: PersistenceSettings = Field(default_factory=PersistenceSettings)
    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)

    # System settings
    debug_mode: bool = Field(
        default=False,
        description="Enable debug mode",
    )
    dry_run: bool = Field(
        default=False,
        description="Disable actual operations",
    )

    @field_validator("node_id")
    @classmethod
    def validate_node_id(cls, v: str) -> str:
        """Ensure node_id is uppercase."""
        return v.upper()

    def is_simulation(self) -> bool:
        """Check if running in simulation mode."""
        return self.dry_run or self.constellation_id.endswith("-SIM")
