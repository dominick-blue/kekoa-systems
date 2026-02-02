# KEKOA ML Systems Design Guide

**The Definitive Engineering Reference for Production Machine Learning in Orbital Environments**

**Version:** 1.0
**Maintainer:** KEKOA ML Platform Engineering
**Last Updated:** February 2026

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [ML System Fundamentals](#2-ml-system-fundamentals)
3. [The Seven-Layer ML Architecture](#3-the-seven-layer-ml-architecture)
4. [Data Pipeline Design](#4-data-pipeline-design)
5. [Feature Engineering](#5-feature-engineering)
6. [Model Training Systems](#6-model-training-systems)
7. [Model Evaluation Framework](#7-model-evaluation-framework)
8. [Model Serving Architecture](#8-model-serving-architecture)
9. [Indexing and Retrieval](#9-indexing-and-retrieval)
10. [Caching Strategies](#10-caching-strategies)
11. [Scalability Patterns](#11-scalability-patterns)
12. [Fault Tolerance](#12-fault-tolerance)
13. [Monitoring and Drift Detection](#13-monitoring-and-drift-detection)
14. [Security and Privacy](#14-security-and-privacy)
15. [Design Trade-offs Matrix](#15-design-trade-offs-matrix)
16. [Case Studies](#16-case-studies)
17. [Appendices](#appendices)

---

## 1. Introduction

### 1.1 Purpose

This guide defines the engineering discipline for architecting systems that train, deploy, and maintain machine learning models within the KEKOA platform. It provides practical guidance for building production-grade ML systems that operate reliably in the challenging orbital environment.

### 1.2 Scope

This guide covers the complete ML system lifecycle:

- **Data Pipelines** - Collection, validation, and transformation
- **Feature Engineering** - Feature stores, computation, and serving
- **Model Training** - Distributed training, hyperparameter optimization
- **Model Evaluation** - Offline/online metrics, A/B testing
- **Model Serving** - Inference infrastructure, optimization
- **Operations** - Monitoring, retraining, lifecycle management

### 1.3 ML Systems vs. Traditional Systems

Machine learning systems differ fundamentally from traditional software:

| Aspect | Traditional Software | ML Systems | KEKOA ML Systems |
|--------|---------------------|------------|------------------|
| **Logic** | Handwritten rules | Learned from data | Learned + physics constraints |
| **Data** | Structured, transactional | High-volume, variable | Distributed, non-IID, orbital |
| **Failures** | Predictable (bugs, crashes) | Silent (drift, bias) | Silent + communication loss |
| **Testing** | Unit/integration tests | A/B testing, offline eval | Simulation + metamorphic |
| **Maintenance** | Code updates | Continuous retraining | Federated retraining |
| **Deployment** | CI/CD anytime | Staged rollout | Ground contact windows |
| **Debugging** | Interactive | Difficult | Telemetry-only |

### 1.4 The Five Core Objectives

Every KEKOA ML system must achieve:

| Objective | Definition | KEKOA Constraint |
|-----------|------------|------------------|
| **Scalability** | Handle growing data and nodes | Fixed compute per satellite |
| **Low Latency** | Fast inference response | <50ms for real-time discard |
| **Reliability** | Consistent performance | No human intervention possible |
| **Adaptability** | Evolve with data distribution | Federated learning updates |
| **Explainability** | Transparent predictions | Mission-critical decisions |

### 1.5 The KEKOA ML Engineering Principles

```
┌─────────────────────────────────────────────────────────────────────┐
│                 KEKOA ML Engineering Principles                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. DATA QUALITY > MODEL COMPLEXITY                                 │
│     Clean data beats clever algorithms in orbit                     │
│                                                                      │
│  2. SIMPLE MODELS > COMPLEX MODELS                                  │
│     Debuggable via telemetry beats marginally higher accuracy       │
│                                                                      │
│  3. OFFLINE VALIDATION > ONLINE EXPERIMENTATION                     │
│     Rigorous ground testing beats "we'll see in production"         │
│                                                                      │
│  4. GRACEFUL DEGRADATION > PERFECT OPERATION                        │
│     Working at 80% capacity beats failing at 100%                   │
│                                                                      │
│  5. REPRODUCIBILITY > CONVENIENCE                                   │
│     Deterministic pipelines beat "it works on my machine"           │
│                                                                      │
│  6. FEDERATION > CENTRALIZATION                                     │
│     Gradients travel, not data                                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. ML System Fundamentals

### 2.1 The ML System Lifecycle

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ML System Lifecycle                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐         │
│   │ Problem │───►│  Data   │───►│ Feature │───►│  Model  │         │
│   │ Framing │    │ Pipeline│    │  Engin. │    │ Training│         │
│   └─────────┘    └─────────┘    └─────────┘    └────┬────┘         │
│                                                      │              │
│   ┌──────────────────────────────────────────────────┘              │
│   │                                                                 │
│   ▼                                                                 │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐         │
│   │  Model  │───►│  Model  │───►│  Model  │───►│ Monitor │         │
│   │  Eval   │    │ Deploy  │    │ Serving │    │ & Maint │         │
│   └─────────┘    └─────────┘    └─────────┘    └────┬────┘         │
│                                                      │              │
│   ┌──────────────────────────────────────────────────┘              │
│   │                                                                 │
│   └────────────────────► Feedback Loop ─────────────────────────►   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Problem Framing for Space ML

Before building any ML system, frame the problem correctly:

**Problem Framing Template:**
```yaml
problem_statement:
  name: "Intelligent Discard - Cloud Detection"
  type: binary_classification

business_objective:
  primary: "Reduce downlink bandwidth by 80%"
  secondary: "Preserve all valuable imagery"

ml_objective:
  metric: "F1 Score"
  threshold: 0.90
  constraint: "False Negative Rate < 5%"  # Don't discard good images

data_characteristics:
  volume: "10,000 frames/day/satellite"
  distribution: "Non-IID (geography dependent)"
  labels: "Partially labeled (ground truth delayed)"

operational_constraints:
  latency: "<50ms per frame"
  power: "<3W average"
  memory: "<500MB model + runtime"

success_criteria:
  - "80% bandwidth reduction achieved"
  - "No more than 1% valuable frames incorrectly discarded"
  - "System operates autonomously for 30+ days"
```

### 2.3 ML Task Taxonomy

| Task Type | Description | KEKOA Use Case |
|-----------|-------------|----------------|
| **Binary Classification** | Two-class prediction | Cloud/No-Cloud detection |
| **Multi-class Classification** | Multiple categories | Scene type classification |
| **Object Detection** | Locate objects in images | Ship/vehicle detection |
| **Semantic Segmentation** | Pixel-level classification | Land use mapping |
| **Anomaly Detection** | Identify outliers | Sensor drift detection |
| **Regression** | Continuous value prediction | Cloud cover percentage |
| **Ranking** | Order items by relevance | Image prioritization |

---

## 3. The Seven-Layer ML Architecture

### 3.1 Architecture Overview

KEKOA ML systems follow a seven-layer architecture adapted for orbital constraints:

```
┌─────────────────────────────────────────────────────────────────────┐
│                   KEKOA ML Seven-Layer Architecture                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Layer 7: Monitoring & Feedback                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Drift Detection │ Telemetry │ Retraining Triggers          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ▲                                       │
│  Layer 6: Inference API & Serving                                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  TensorRT │ ONNX Runtime │ Batching │ Result Cache          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ▲                                       │
│  Layer 5: Model Deployment & Optimization                           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Quantization │ Model Registry │ A/B Testing │ Rollback     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ▲                                       │
│  Layer 4: Model Training & Evaluation                               │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Local Training │ FL Aggregation │ Validation │ Metrics     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ▲                                       │
│  Layer 3: Feature Engineering & Store                               │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Feature Extraction │ Feature Cache │ Orbital Features      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ▲                                       │
│  Layer 2: Data Storage & Management                                 │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Frame Buffer │ Training Store │ Checkpoint Store           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ▲                                       │
│  Layer 1: Data Ingestion & Preprocessing                            │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Sensor Interface │ Validation │ Normalization │ Augment    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ▲                                       │
│  ════════════════════════════════════════════════════════════════   │
│                        Hardware Layer                                │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Sensors │ Jetson Orin │ Flash Storage │ ISL/Ground Radio   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Layer Specifications

#### Layer 1: Data Ingestion & Preprocessing

**Responsibilities:**
- Capture raw sensor data at source
- Validate data integrity (checksums, format)
- Apply calibration and normalization
- Augment data for training diversity

**Components:**
```python
class DataIngestionLayer:
    """Layer 1: Data Ingestion and Preprocessing."""

    def __init__(self, sensor: SensorInterface, config: IngestionConfig):
        self.sensor = sensor
        self.config = config
        self.validator = DataValidator(config.schema)
        self.preprocessor = Preprocessor(config.preprocessing)

    def ingest(self) -> Optional[ProcessedFrame]:
        # Capture from sensor
        raw_frame = self.sensor.capture()

        # Validate
        if not self.validator.validate(raw_frame):
            self.log_invalid_frame(raw_frame)
            return None

        # Preprocess
        processed = self.preprocessor.process(raw_frame)

        return processed
```

#### Layer 2: Data Storage & Management

**Responsibilities:**
- Buffer incoming frames for processing
- Store training data with lifecycle management
- Manage model checkpoints and artifacts

**Storage Tiers:**

| Tier | Medium | Capacity | Latency | Use Case |
|------|--------|----------|---------|----------|
| Hot | RAM | 4 GB | <1ms | Active inference buffer |
| Warm | NVMe | 64 GB | <10ms | Recent training data |
| Cold | Flash | 256 GB | <100ms | Historical data, checkpoints |
| Archive | Ground | Unlimited | Hours-Days | Long-term storage |

#### Layer 3: Feature Engineering & Store

**Responsibilities:**
- Compute features from raw data
- Cache frequently-used features
- Ensure training-serving consistency

**Feature Categories:**
```python
@dataclass
class FeatureSet:
    """Complete feature set for a frame."""

    # Image-derived features
    image_embedding: np.ndarray  # 512-dim CNN output
    spectral_indices: SpectralFeatures
    texture_features: TextureFeatures

    # Orbital features (from OAK)
    orbital: OrbitalFeatures

    # Contextual features
    context: ContextFeatures

    # Metadata
    frame_id: str
    timestamp: float
    version: str  # Feature schema version
```

#### Layer 4: Model Training & Evaluation

**Responsibilities:**
- Execute local training rounds
- Compute and exchange gradients
- Aggregate federated updates
- Validate model quality

#### Layer 5: Model Deployment & Optimization

**Responsibilities:**
- Optimize models for edge inference
- Manage model versions and registry
- Execute deployment strategies
- Enable rollback on degradation

#### Layer 6: Inference API & Serving

**Responsibilities:**
- Execute real-time predictions
- Batch requests for efficiency
- Cache prediction results
- Meet latency SLOs

#### Layer 7: Monitoring & Feedback

**Responsibilities:**
- Track model performance metrics
- Detect data and concept drift
- Trigger retraining when needed
- Report telemetry to ground

---

## 4. Data Pipeline Design

### 4.1 Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      KEKOA Data Pipeline                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐          │
│  │ Sensor  │───►│ Validate│───►│ Buffer  │───►│ Route   │          │
│  │ Capture │    │         │    │         │    │         │          │
│  └─────────┘    └─────────┘    └─────────┘    └────┬────┘          │
│                                                     │               │
│                              ┌──────────────────────┼───────────┐   │
│                              │                      │           │   │
│                              ▼                      ▼           ▼   │
│                       ┌───────────┐          ┌───────────┐  ┌─────┐│
│                       │ Inference │          │ Training  │  │Disk ││
│                       │ Pipeline  │          │ Pipeline  │  │     ││
│                       └───────────┘          └───────────┘  └─────┘│
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Data Validation

**Validation Layers:**

```python
class DataValidator:
    """Multi-layer data validation for orbital data quality."""

    def __init__(self, schema: DataSchema):
        self.schema = schema
        self.validators = [
            SchemaValidator(schema),      # Structure validation
            RangeValidator(schema),       # Value range checks
            StatisticalValidator(schema), # Distribution checks
            TemporalValidator(schema),    # Sequence consistency
        ]

    def validate(self, frame: RawFrame) -> ValidationResult:
        results = []

        for validator in self.validators:
            result = validator.check(frame)
            results.append(result)

            if result.severity == Severity.CRITICAL and not result.passed:
                # Stop on critical failure
                return ValidationResult(
                    passed=False,
                    results=results,
                    action=ValidationAction.REJECT
                )

        passed = all(r.passed for r in results)

        return ValidationResult(
            passed=passed,
            results=results,
            action=ValidationAction.ACCEPT if passed else ValidationAction.FLAG
        )
```

**Validation Checks:**

| Check | Description | Action on Failure |
|-------|-------------|-------------------|
| Schema | Correct format and types | Reject |
| Range | Values within expected bounds | Flag or reject |
| Completeness | No missing required fields | Reject |
| Temporal | Timestamp ordering correct | Flag |
| Statistical | Not an extreme outlier | Flag |
| Corruption | Checksum validation | Reject |

### 4.3 Data Augmentation

**Augmentation for Training:**

```python
class OrbitalDataAugmenter:
    """Augmentation strategies for satellite imagery."""

    def __init__(self, config: AugmentConfig):
        self.config = config

    def augment(self, frame: ProcessedFrame) -> List[ProcessedFrame]:
        augmented = [frame]  # Original always included

        if self.config.geometric:
            # Rotation (satellite can image at different angles)
            for angle in [90, 180, 270]:
                augmented.append(self.rotate(frame, angle))

            # Flip (valid for most overhead imagery)
            augmented.append(self.horizontal_flip(frame))
            augmented.append(self.vertical_flip(frame))

        if self.config.photometric:
            # Brightness variation (sun angle simulation)
            for factor in [0.8, 1.2]:
                augmented.append(self.adjust_brightness(frame, factor))

            # Noise injection (sensor noise simulation)
            augmented.append(self.add_gaussian_noise(frame, std=0.01))

        if self.config.spatial:
            # Random crop (simulate different resolutions)
            augmented.append(self.random_crop(frame, scale=0.9))

        return augmented
```

### 4.4 Data Versioning

**Dataset Version Control:**

```python
@dataclass
class DatasetVersion:
    """Versioned dataset for reproducibility."""

    version: str
    created_at: datetime

    # Content hash for integrity
    content_hash: str

    # Statistics snapshot
    num_samples: int
    class_distribution: Dict[str, int]
    feature_statistics: FeatureStatistics

    # Lineage
    source_versions: List[str]  # Parent datasets
    preprocessing_config: Dict[str, Any]
    augmentation_config: Dict[str, Any]

    # Validation
    quality_score: float
    validation_report: ValidationReport


class DatasetRegistry:
    """Registry for dataset versions."""

    def register(self, dataset: Dataset, config: DatasetConfig) -> DatasetVersion:
        version = self._generate_version()
        content_hash = self._compute_hash(dataset)

        version_record = DatasetVersion(
            version=version,
            created_at=datetime.now(),
            content_hash=content_hash,
            num_samples=len(dataset),
            class_distribution=self._compute_distribution(dataset),
            feature_statistics=self._compute_statistics(dataset),
            source_versions=config.source_versions,
            preprocessing_config=config.preprocessing,
            augmentation_config=config.augmentation,
            quality_score=self._assess_quality(dataset),
            validation_report=self._validate(dataset),
        )

        self._store(version_record)
        return version_record
```

---

## 5. Feature Engineering

### 5.1 Feature Store Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     KEKOA Feature Store                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Feature Registry                          │   │
│  │  - Feature definitions (schema, computation, version)        │   │
│  │  - Feature metadata (owner, update frequency, SLA)           │   │
│  │  - Feature lineage (dependencies, transformations)           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│         ┌────────────────────┼────────────────────┐                 │
│         │                    │                    │                 │
│         ▼                    ▼                    ▼                 │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐           │
│  │   Offline   │     │   Online    │     │   Orbital   │           │
│  │   Store     │     │   Store     │     │   Store     │           │
│  │             │     │             │     │             │           │
│  │ Historical  │     │ Real-time   │     │ OAK-derived │           │
│  │ features    │     │ features    │     │ features    │           │
│  │ (Flash)     │     │ (RAM)       │     │ (Computed)  │           │
│  └─────────────┘     └─────────────┘     └─────────────┘           │
│         │                    │                    │                 │
│         └────────────────────┼────────────────────┘                 │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                  Feature Serving API                         │   │
│  │  get_features(entity_id, feature_list, timestamp)           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 Feature Definition

**Feature Specification Format:**

```yaml
# features/image_embedding.yaml
feature:
  name: image_embedding_v2
  version: "2.0"
  owner: ml-platform-team

description: |
  512-dimensional embedding from MobileNetV3-Small backbone.
  Captures semantic content of image frame.

schema:
  type: array
  dtype: float32
  shape: [512]

computation:
  type: model_inference
  model: mobilenet_v3_small_backbone
  input: preprocessed_frame
  output: embedding_layer

dependencies:
  - preprocessed_frame

serving:
  online_ttl_seconds: 60
  offline_retention_days: 30

quality:
  freshness_sla_seconds: 1
  completeness_threshold: 0.99
```

### 5.3 Feature Categories for Orbital ML

#### Image Features

```python
class ImageFeatureExtractor:
    """Extract features from satellite imagery."""

    def __init__(self, backbone: nn.Module):
        self.backbone = backbone
        self.backbone.eval()

    def extract(self, frame: Tensor) -> ImageFeatures:
        with torch.no_grad():
            # CNN embedding
            embedding = self.backbone(frame)

        return ImageFeatures(
            embedding=embedding.numpy(),

            # Statistical features
            mean_intensity=frame.mean().item(),
            std_intensity=frame.std().item(),

            # Texture features
            edge_density=self._compute_edge_density(frame),
            entropy=self._compute_entropy(frame),

            # Spectral features (if multispectral)
            ndvi=self._compute_ndvi(frame) if frame.shape[0] > 3 else None,
        )
```

#### Spectral Features

```python
class SpectralFeatureExtractor:
    """Extract spectral indices from multispectral imagery."""

    BAND_MAPPING = {
        'blue': 0,
        'green': 1,
        'red': 2,
        'nir': 3,
        'swir': 4,
    }

    def extract(self, frame: Tensor) -> SpectralFeatures:
        bands = {name: frame[idx] for name, idx in self.BAND_MAPPING.items()
                 if idx < frame.shape[0]}

        features = SpectralFeatures()

        # Normalized Difference Vegetation Index
        if 'nir' in bands and 'red' in bands:
            features.ndvi = self._ndvi(bands['nir'], bands['red'])

        # Normalized Difference Water Index
        if 'nir' in bands and 'green' in bands:
            features.ndwi = self._ndwi(bands['nir'], bands['green'])

        # Cloud Index (simple brightness threshold)
        if all(b in bands for b in ['red', 'green', 'blue']):
            features.cloud_index = self._cloud_index(
                bands['red'], bands['green'], bands['blue']
            )

        return features

    def _ndvi(self, nir: Tensor, red: Tensor) -> float:
        return ((nir - red) / (nir + red + 1e-8)).mean().item()
```

#### Orbital Features

```python
class OrbitalFeatureExtractor:
    """Extract features from orbital mechanics (via OAK)."""

    def __init__(self, oak_client: OAKClient):
        self.oak = oak_client

    def extract(
        self,
        satellite_id: str,
        timestamp: float
    ) -> OrbitalFeatures:
        # Get state from OAK
        state_vector = self.oak.get_state_vector(satellite_id, timestamp)
        power_state = self.oak.get_power_state(satellite_id)

        # Compute derived features
        return OrbitalFeatures(
            # Position features
            altitude_km=state_vector.altitude / 1000,
            latitude=state_vector.latitude,
            longitude=state_vector.longitude,

            # Lighting features
            sun_elevation=self._compute_sun_elevation(state_vector, timestamp),
            eclipse_state=power_state.eclipse_state,

            # Orbital phase
            orbital_phase=self._compute_phase(state_vector),
            time_since_ascending_node=self._time_since_an(state_vector),

            # Velocity features
            ground_track_velocity=state_vector.ground_velocity,

            # Environmental
            season=self._compute_season(state_vector.latitude, timestamp),
        )
```

### 5.4 Training-Serving Consistency

**The Training-Serving Skew Problem:**

```
Training Time:                    Serving Time:
─────────────                    ─────────────
Raw Data                         Raw Data
    │                                │
    ▼                                ▼
Feature Computation              Feature Computation
(Python, Batch)                  (C++, Real-time)
    │                                │
    ▼                                ▼
Features (may differ!)  ═══✗═══  Features (may differ!)
    │                                │
    ▼                                ▼
Model Training                   Model Inference
    │                                │
    ▼                                ▼
Trained Model                    WRONG PREDICTIONS!
```

**Solution: Unified Feature Logic**

```python
class UnifiedFeatureComputation:
    """
    Single source of truth for feature computation.
    Used in both training and serving.
    """

    def __init__(self, config: FeatureConfig):
        self.config = config
        self.extractors = self._build_extractors(config)

    def compute(
        self,
        raw_data: RawFrame,
        orbital_context: OrbitalContext
    ) -> FeatureVector:
        """
        Compute features identically for training and serving.
        This method is the ONLY way to compute features.
        """
        features = {}

        for name, extractor in self.extractors.items():
            features[name] = extractor.extract(raw_data, orbital_context)

        return FeatureVector(
            features=features,
            schema_version=self.config.version,
            computed_at=time.time()
        )
```

---

## 6. Model Training Systems

### 6.1 Training Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                   KEKOA Training Architecture                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  GROUND SEGMENT (Pre-deployment)                                    │
│  ────────────────────────────────                                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  ┌───────────┐    ┌───────────┐    ┌───────────┐           │   │
│  │  │  Initial  │───►│   HPO     │───►│  Optimize │           │   │
│  │  │  Training │    │  (Optuna) │    │  (TensorRT)│          │   │
│  │  └───────────┘    └───────────┘    └───────────┘           │   │
│  │                                           │                 │   │
│  │                                           ▼                 │   │
│  │                                    ┌───────────┐           │   │
│  │                                    │   Uplink  │           │   │
│  │                                    │   Package │           │   │
│  │                                    └─────┬─────┘           │   │
│  └──────────────────────────────────────────┼──────────────────┘   │
│                                             │                       │
│  ═══════════════════════════════════════════╪═══════════════════   │
│                                             │                       │
│  SPACE SEGMENT (Continuous)                 │                       │
│  ──────────────────────────                 │                       │
│  ┌──────────────────────────────────────────┼──────────────────┐   │
│  │                                          ▼                  │   │
│  │  ┌───────────┐    ┌───────────┐    ┌───────────┐           │   │
│  │  │   Local   │───►│  Gradient │───►│   Peer    │           │   │
│  │  │  Training │    │  Compute  │    │  Exchange │           │   │
│  │  └───────────┘    └───────────┘    └─────┬─────┘           │   │
│  │                                          │                  │   │
│  │       ┌──────────────────────────────────┘                  │   │
│  │       │                                                     │   │
│  │       ▼                                                     │   │
│  │  ┌───────────┐    ┌───────────┐    ┌───────────┐           │   │
│  │  │  Aggreg-  │───►│ Validate  │───►│  Deploy   │           │   │
│  │  │   ation   │    │           │    │           │           │   │
│  │  └───────────┘    └───────────┘    └───────────┘           │   │
│  │                                                             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 Local Training Loop

```python
class LocalTrainingLoop:
    """Training loop for on-satellite model updates."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        power_manager: PowerManager,
        oak_client: OAKClient
    ):
        self.model = model
        self.config = config
        self.power_manager = power_manager
        self.oak = oak_client

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

    def train_round(
        self,
        dataloader: DataLoader,
        round_id: int
    ) -> TrainingResult:
        """Execute one local training round."""

        self.model.train()
        initial_weights = self._copy_weights()

        metrics = TrainingMetrics()
        samples_processed = 0

        for epoch in range(self.config.local_epochs):
            for batch_idx, batch in enumerate(dataloader):
                # Check constraints
                if self._should_stop():
                    break

                # Forward pass
                loss, batch_metrics = self._forward_step(batch)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )

                self.optimizer.step()

                # Update metrics
                metrics.update(batch_metrics)
                samples_processed += len(batch)

            self.scheduler.step()

        # Compute gradient as weight difference
        final_weights = self._copy_weights()
        gradient = self._compute_gradient(initial_weights, final_weights)

        return TrainingResult(
            gradient=gradient,
            metrics=metrics,
            samples_processed=samples_processed,
            epochs_completed=epoch + 1,
            round_id=round_id
        )

    def _should_stop(self) -> bool:
        """Check if training should stop early."""

        # Power constraint
        if self.power_manager.get_state() == PowerState.CRITICAL:
            return True

        # Contact window approaching
        next_contact = self.oak.get_next_contact_window()
        if next_contact and next_contact.time_until_aos < timedelta(minutes=2):
            return True

        return False
```

### 6.3 Distributed Training Strategies

#### Data Parallelism (Federated)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Federated Data Parallelism                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   SAT-A              SAT-B              SAT-C                   │
│   ┌─────┐            ┌─────┐            ┌─────┐                 │
│   │Data │            │Data │            │Data │                 │
│   │  A  │            │  B  │            │  C  │                 │
│   └──┬──┘            └──┬──┘            └──┬──┘                 │
│      │                  │                  │                    │
│      ▼                  ▼                  ▼                    │
│   ┌─────┐            ┌─────┐            ┌─────┐                 │
│   │Model│            │Model│            │Model│   (Same model)  │
│   │Copy │            │Copy │            │Copy │                 │
│   └──┬──┘            └──┬──┘            └──┬──┘                 │
│      │                  │                  │                    │
│      ▼                  ▼                  ▼                    │
│   ┌─────┐            ┌─────┐            ┌─────┐                 │
│   │Grad │            │Grad │            │Grad │   (Different)   │
│   │  A  │            │  B  │            │  C  │                 │
│   └──┬──┘            └──┬──┘            └──┬──┘                 │
│      │                  │                  │                    │
│      └──────────────────┼──────────────────┘                    │
│                         │                                        │
│                         ▼                                        │
│                    ┌─────────┐                                   │
│                    │Aggregate│                                   │
│                    │ (FedAvg)│                                   │
│                    └────┬────┘                                   │
│                         │                                        │
│                         ▼                                        │
│                    ┌─────────┐                                   │
│                    │ Updated │                                   │
│                    │  Model  │                                   │
│                    └─────────┘                                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Ring-AllReduce (Within Orbital Plane)

```python
class RingAllReduceTrainer:
    """
    Efficient gradient aggregation following orbital ring topology.
    Each satellite passes partial sum to successor.
    """

    def __init__(self, position: int, ring_size: int, mesh: LatticeMesh):
        self.position = position
        self.ring_size = ring_size
        self.mesh = mesh

        # Ring neighbors
        self.predecessor = (position - 1) % ring_size
        self.successor = (position + 1) % ring_size

    async def all_reduce(self, local_gradient: Tensor) -> Tensor:
        """
        Perform ring all-reduce to compute average gradient.

        Phase 1: Scatter-reduce (n-1 steps)
        - Each node sends chunk to successor, receives from predecessor
        - Accumulate received chunks

        Phase 2: All-gather (n-1 steps)
        - Each node sends accumulated chunk to successor
        - After completion, all nodes have full reduced gradient
        """

        # Split gradient into ring_size chunks
        chunks = self._split_gradient(local_gradient, self.ring_size)

        # Phase 1: Scatter-reduce
        for step in range(self.ring_size - 1):
            send_idx = (self.position - step) % self.ring_size
            recv_idx = (self.position - step - 1) % self.ring_size

            # Send chunk to successor
            send_task = self.mesh.send_async(
                self.successor,
                chunks[send_idx]
            )

            # Receive chunk from predecessor
            received = await self.mesh.receive(self.predecessor)

            # Accumulate
            chunks[recv_idx] = chunks[recv_idx] + received

            await send_task

        # Phase 2: All-gather
        for step in range(self.ring_size - 1):
            send_idx = (self.position - step + 1) % self.ring_size
            recv_idx = (self.position - step) % self.ring_size

            send_task = self.mesh.send_async(
                self.successor,
                chunks[send_idx]
            )

            received = await self.mesh.receive(self.predecessor)
            chunks[recv_idx] = received

            await send_task

        # Reconstruct full gradient and average
        full_gradient = self._merge_chunks(chunks)
        return full_gradient / self.ring_size
```

### 6.4 Hyperparameter Optimization

**HPO for Orbital Constraints:**

```python
class OrbitalHPO:
    """
    Hyperparameter optimization considering orbital constraints.
    Run on ground before deployment.
    """

    SEARCH_SPACE = {
        # Must fit in power budget
        'batch_size': [8, 16, 32],

        # Learning rate (higher for FL)
        'learning_rate': (1e-3, 1e-1, 'log'),

        # Local epochs (more = more drift)
        'local_epochs': [1, 2, 3, 5],

        # Optimizer
        'optimizer': ['sgd', 'adam'],

        # Regularization
        'weight_decay': (1e-5, 1e-2, 'log'),

        # Model architecture
        'model_variant': ['small', 'medium'],
    }

    CONSTRAINTS = {
        'inference_latency_ms': 50,
        'memory_mb': 500,
        'power_watts': 5,
    }

    def optimize(
        self,
        train_data: Dataset,
        val_data: Dataset,
        n_trials: int = 100
    ) -> HyperparameterSet:

        study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner()
        )

        def objective(trial):
            params = self._sample_params(trial)

            # Check constraints first
            if not self._check_constraints(params):
                return float('-inf')

            # Train and evaluate
            model = self._build_model(params)
            result = self._train_and_evaluate(model, train_data, val_data, params)

            return result.val_accuracy

        study.optimize(objective, n_trials=n_trials)

        return HyperparameterSet(
            params=study.best_params,
            score=study.best_value,
            trials=len(study.trials)
        )
```

### 6.5 Checkpointing

```python
class TrainingCheckpointer:
    """Checkpoint training state for resumption after interruption."""

    def __init__(self, checkpoint_dir: str, keep_last_n: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.keep_last_n = keep_last_n

    def save(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        epoch: int,
        round_id: int,
        metrics: Dict[str, float]
    ) -> str:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'round_id': round_id,
            'metrics': metrics,
            'timestamp': time.time(),
            'rng_state': {
                'torch': torch.get_rng_state(),
                'numpy': np.random.get_state(),
                'python': random.getstate(),
            }
        }

        path = self.checkpoint_dir / f"checkpoint_r{round_id}_e{epoch}.pt"
        torch.save(checkpoint, path)

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        return str(path)

    def load_latest(self) -> Optional[Dict]:
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        if not checkpoints:
            return None

        checkpoint = torch.load(checkpoints[0])

        # Restore RNG state for reproducibility
        torch.set_rng_state(checkpoint['rng_state']['torch'])
        np.random.set_state(checkpoint['rng_state']['numpy'])
        random.setstate(checkpoint['rng_state']['python'])

        return checkpoint
```

---

## 7. Model Evaluation Framework

### 7.1 Evaluation Strategy

```
┌─────────────────────────────────────────────────────────────────────┐
│                    KEKOA Evaluation Framework                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│                        ┌─────────────────┐                          │
│                        │   New Model     │                          │
│                        │   Candidate     │                          │
│                        └────────┬────────┘                          │
│                                 │                                    │
│         ┌───────────────────────┼───────────────────────┐           │
│         │                       │                       │           │
│         ▼                       ▼                       ▼           │
│  ┌─────────────┐        ┌─────────────┐        ┌─────────────┐     │
│  │   Offline   │        │   Shadow    │        │   Canary    │     │
│  │ Evaluation  │        │  Deployment │        │  Deployment │     │
│  │             │        │             │        │             │     │
│  │ - Holdout   │        │ - Run both  │        │ - 10% traffic│    │
│  │ - Cross-val │        │ - Compare   │        │ - Monitor   │     │
│  │ - Metrics   │        │ - No impact │        │ - Rollback  │     │
│  └──────┬──────┘        └──────┬──────┘        └──────┬──────┘     │
│         │                      │                      │             │
│         ▼                      ▼                      ▼             │
│  ┌─────────────┐        ┌─────────────┐        ┌─────────────┐     │
│  │    Pass?    │        │   Similar?  │        │   Better?   │     │
│  │  Accuracy   │        │  Predictions│        │  Live KPIs  │     │
│  │  >threshold │        │             │        │             │     │
│  └──────┬──────┘        └──────┬──────┘        └──────┬──────┘     │
│         │ Yes                  │ Yes                  │ Yes         │
│         └───────────────────────┼───────────────────────┘           │
│                                 │                                    │
│                                 ▼                                    │
│                        ┌─────────────────┐                          │
│                        │   Full Rollout  │                          │
│                        └─────────────────┘                          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.2 Offline Evaluation

**Metrics Suite:**

```python
class OfflineEvaluator:
    """Comprehensive offline model evaluation."""

    def evaluate(
        self,
        model: nn.Module,
        test_data: DataLoader,
        task_type: TaskType
    ) -> EvaluationReport:

        predictions = []
        labels = []
        latencies = []

        model.eval()
        with torch.no_grad():
            for batch in test_data:
                start_time = time.perf_counter()
                output = model(batch['input'])
                latency = time.perf_counter() - start_time

                predictions.extend(output.cpu().numpy())
                labels.extend(batch['label'].cpu().numpy())
                latencies.append(latency / len(batch))

        predictions = np.array(predictions)
        labels = np.array(labels)

        # Compute metrics based on task type
        if task_type == TaskType.BINARY_CLASSIFICATION:
            metrics = self._binary_classification_metrics(predictions, labels)
        elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
            metrics = self._multiclass_metrics(predictions, labels)
        elif task_type == TaskType.REGRESSION:
            metrics = self._regression_metrics(predictions, labels)

        # Add performance metrics
        metrics.update({
            'latency_mean_ms': np.mean(latencies) * 1000,
            'latency_p50_ms': np.percentile(latencies, 50) * 1000,
            'latency_p95_ms': np.percentile(latencies, 95) * 1000,
            'latency_p99_ms': np.percentile(latencies, 99) * 1000,
        })

        return EvaluationReport(
            metrics=metrics,
            predictions=predictions,
            labels=labels,
            confusion_matrix=self._confusion_matrix(predictions, labels),
            calibration=self._calibration_analysis(predictions, labels),
        )

    def _binary_classification_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:

        binary_preds = (predictions > 0.5).astype(int)

        return {
            'accuracy': accuracy_score(labels, binary_preds),
            'precision': precision_score(labels, binary_preds),
            'recall': recall_score(labels, binary_preds),
            'f1': f1_score(labels, binary_preds),
            'auc_roc': roc_auc_score(labels, predictions),
            'auc_pr': average_precision_score(labels, predictions),

            # KEKOA-specific: False negative rate is critical
            'false_negative_rate': 1 - recall_score(labels, binary_preds),
        }
```

### 7.3 Cross-Validation for Non-IID Data

```python
class OrbitalCrossValidator:
    """
    Cross-validation that respects orbital data structure.
    Ensures geographic/temporal separation between folds.
    """

    def __init__(self, n_folds: int = 5):
        self.n_folds = n_folds

    def split(
        self,
        dataset: Dataset,
        strategy: str = 'geographic'
    ) -> Iterator[Tuple[Dataset, Dataset]]:

        if strategy == 'geographic':
            return self._geographic_split(dataset)
        elif strategy == 'temporal':
            return self._temporal_split(dataset)
        elif strategy == 'satellite':
            return self._satellite_split(dataset)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _geographic_split(self, dataset: Dataset) -> Iterator:
        """Split by geographic region to simulate deployment to new areas."""

        # Cluster samples by location
        locations = np.array([
            (s.metadata.latitude, s.metadata.longitude)
            for s in dataset
        ])

        kmeans = KMeans(n_clusters=self.n_folds)
        cluster_labels = kmeans.fit_predict(locations)

        for fold in range(self.n_folds):
            test_mask = cluster_labels == fold
            train_mask = ~test_mask

            yield dataset[train_mask], dataset[test_mask]

    def _temporal_split(self, dataset: Dataset) -> Iterator:
        """Split by time to simulate forward deployment."""

        timestamps = np.array([s.metadata.timestamp for s in dataset])
        sorted_indices = np.argsort(timestamps)

        fold_size = len(dataset) // self.n_folds

        for fold in range(self.n_folds):
            # Use earlier data for training, later for testing
            train_end = (fold + 1) * fold_size
            test_start = train_end
            test_end = min(test_start + fold_size, len(dataset))

            train_indices = sorted_indices[:train_end]
            test_indices = sorted_indices[test_start:test_end]

            yield dataset[train_indices], dataset[test_indices]
```

### 7.4 Calibration Analysis

```python
class CalibrationAnalyzer:
    """
    Analyze model calibration for reliable uncertainty estimates.
    Critical for decision-making in orbital systems.
    """

    def analyze(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 10
    ) -> CalibrationReport:

        # Compute calibration curve
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(predictions, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        calibration_curve = []
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                mean_predicted = predictions[mask].mean()
                mean_actual = labels[mask].mean()
                count = mask.sum()
                calibration_curve.append({
                    'bin': i,
                    'predicted': mean_predicted,
                    'actual': mean_actual,
                    'count': count,
                    'gap': abs(mean_predicted - mean_actual)
                })

        # Expected Calibration Error
        ece = self._compute_ece(calibration_curve)

        # Maximum Calibration Error
        mce = max(point['gap'] for point in calibration_curve)

        return CalibrationReport(
            curve=calibration_curve,
            ece=ece,
            mce=mce,
            is_calibrated=ece < 0.05,  # Threshold
            recommendation=self._recommend_fix(ece, mce)
        )

    def _recommend_fix(self, ece: float, mce: float) -> str:
        if ece < 0.05:
            return "Model is well-calibrated"
        elif ece < 0.10:
            return "Consider temperature scaling"
        else:
            return "Apply Platt scaling or isotonic regression"
```

### 7.5 Slice-Based Evaluation

```python
class SliceEvaluator:
    """
    Evaluate model performance on specific data slices.
    Identifies failure modes in subpopulations.
    """

    SLICES = {
        'by_cloud_cover': lambda x: x.metadata.cloud_cover_bucket,
        'by_sun_angle': lambda x: x.metadata.sun_angle_bucket,
        'by_latitude_band': lambda x: x.metadata.latitude_band,
        'by_land_type': lambda x: x.metadata.land_type,
        'by_eclipse_state': lambda x: x.metadata.eclipse_state,
    }

    def evaluate_slices(
        self,
        model: nn.Module,
        dataset: Dataset
    ) -> SliceReport:

        slice_results = {}

        for slice_name, slice_fn in self.SLICES.items():
            slice_results[slice_name] = {}

            # Group by slice
            slice_groups = defaultdict(list)
            for sample in dataset:
                slice_value = slice_fn(sample)
                slice_groups[slice_value].append(sample)

            # Evaluate each group
            for slice_value, samples in slice_groups.items():
                subset = Dataset(samples)
                metrics = self._evaluate(model, subset)

                slice_results[slice_name][slice_value] = {
                    'count': len(samples),
                    'metrics': metrics,
                }

        # Identify underperforming slices
        underperforming = self._find_underperforming_slices(slice_results)

        return SliceReport(
            results=slice_results,
            underperforming=underperforming,
            recommendations=self._generate_recommendations(underperforming)
        )
```

---

## 8. Model Serving Architecture

### 8.1 Serving Infrastructure

```
┌─────────────────────────────────────────────────────────────────────┐
│                    KEKOA Serving Infrastructure                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                     Request Router                           │   │
│  │  - Route to appropriate model variant                        │   │
│  │  - Load balancing across inference engines                   │   │
│  │  - Power-aware routing                                       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│         ┌────────────────────┼────────────────────┐                 │
│         │                    │                    │                 │
│         ▼                    ▼                    ▼                 │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐           │
│  │  Primary    │     │  Fallback   │     │   Shadow    │           │
│  │   Model     │     │   Model     │     │   Model     │           │
│  │             │     │             │     │             │           │
│  │ Full-size   │     │ Lightweight │     │  Candidate  │           │
│  │ TensorRT    │     │ ONNX        │     │  (Testing)  │           │
│  └──────┬──────┘     └──────┬──────┘     └──────┬──────┘           │
│         │                   │                   │                   │
│         └───────────────────┼───────────────────┘                   │
│                             │                                        │
│                             ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Response Handler                          │   │
│  │  - Format predictions                                        │   │
│  │  - Attach confidence scores                                  │   │
│  │  - Log for monitoring                                        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 8.2 Inference Engine

```python
class InferenceEngine:
    """
    High-performance inference engine for satellite deployment.
    Supports multiple runtime backends.
    """

    def __init__(
        self,
        model_path: str,
        backend: str = 'tensorrt',
        config: InferenceConfig = None
    ):
        self.config = config or InferenceConfig()
        self.backend = backend

        # Load model based on backend
        if backend == 'tensorrt':
            self.runtime = TensorRTRuntime(model_path, config)
        elif backend == 'onnx':
            self.runtime = ONNXRuntime(model_path, config)
        elif backend == 'pytorch':
            self.runtime = PyTorchRuntime(model_path, config)
        else:
            raise ValueError(f"Unknown backend: {backend}")

        # Warmup
        self._warmup()

    def predict(
        self,
        inputs: Union[np.ndarray, List[np.ndarray]]
    ) -> InferenceResult:
        """Execute inference with timing and error handling."""

        start_time = time.perf_counter()

        try:
            # Preprocess inputs
            processed = self._preprocess(inputs)

            # Run inference
            raw_output = self.runtime.infer(processed)

            # Postprocess
            predictions = self._postprocess(raw_output)

            latency = time.perf_counter() - start_time

            return InferenceResult(
                predictions=predictions,
                latency_ms=latency * 1000,
                success=True,
                backend=self.backend
            )

        except Exception as e:
            latency = time.perf_counter() - start_time

            return InferenceResult(
                predictions=None,
                latency_ms=latency * 1000,
                success=False,
                error=str(e),
                backend=self.backend
            )

    def predict_batch(
        self,
        batch: List[np.ndarray]
    ) -> List[InferenceResult]:
        """Batch inference for higher throughput."""

        # Stack into single tensor
        stacked = np.stack(batch)

        result = self.predict(stacked)

        if result.success:
            # Split results
            return [
                InferenceResult(
                    predictions=result.predictions[i],
                    latency_ms=result.latency_ms / len(batch),
                    success=True,
                    backend=self.backend
                )
                for i in range(len(batch))
            ]
        else:
            return [result] * len(batch)
```

### 8.3 Model Optimization Pipeline

```python
class ModelOptimizationPipeline:
    """
    Optimize models for edge deployment.
    Targets Jetson Orin NPU.
    """

    def optimize(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        target_latency_ms: float = 50
    ) -> OptimizedModel:

        # Step 1: Export to ONNX
        onnx_path = self._export_onnx(model, sample_input)

        # Step 2: Quantization analysis
        quant_analysis = self._analyze_quantization(model, sample_input)

        # Step 3: Build TensorRT engine
        if quant_analysis.int8_accuracy_loss < 0.02:
            precision = 'int8'
        elif quant_analysis.fp16_accuracy_loss < 0.01:
            precision = 'fp16'
        else:
            precision = 'fp32'

        trt_engine = self._build_tensorrt(
            onnx_path,
            precision=precision,
            max_batch_size=32
        )

        # Step 4: Benchmark
        benchmark = self._benchmark(trt_engine, sample_input)

        # Step 5: Validate latency target
        if benchmark.latency_p99_ms > target_latency_ms:
            # Try more aggressive optimization
            return self._aggressive_optimize(model, sample_input, target_latency_ms)

        return OptimizedModel(
            engine=trt_engine,
            precision=precision,
            original_size_mb=self._get_size(model),
            optimized_size_mb=self._get_size(trt_engine),
            benchmark=benchmark,
            calibration_data_used=quant_analysis.calibration_samples
        )

    def _build_tensorrt(
        self,
        onnx_path: str,
        precision: str,
        max_batch_size: int
    ) -> trt.ICudaEngine:

        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, logger)

        # Parse ONNX
        with open(onnx_path, 'rb') as f:
            parser.parse(f.read())

        # Configure builder
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB

        if precision == 'fp16':
            config.set_flag(trt.BuilderFlag.FP16)
        elif precision == 'int8':
            config.set_flag(trt.BuilderFlag.INT8)
            config.int8_calibrator = self._create_calibrator()

        # Build engine
        engine = builder.build_engine(network, config)

        return engine
```

### 8.4 Request Batching

```python
class DynamicBatcher:
    """
    Dynamic request batching for efficient NPU utilization.
    Balances latency vs. throughput.
    """

    def __init__(
        self,
        inference_engine: InferenceEngine,
        max_batch_size: int = 32,
        max_wait_ms: float = 10,
        adaptive: bool = True
    ):
        self.engine = inference_engine
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.adaptive = adaptive

        self.queue = asyncio.Queue()
        self.results = {}

        # Adaptive parameters
        self.current_batch_size = max_batch_size
        self.latency_history = deque(maxlen=100)

    async def predict(self, input_data: np.ndarray) -> InferenceResult:
        """Submit request and wait for result."""

        request_id = uuid.uuid4().hex
        future = asyncio.Future()

        await self.queue.put((request_id, input_data, future))

        return await future

    async def run_batcher(self):
        """Main batching loop."""

        while True:
            batch = []
            futures = []
            start_time = time.perf_counter()

            # Collect requests
            while len(batch) < self.current_batch_size:
                try:
                    timeout = self.max_wait_ms / 1000 - (time.perf_counter() - start_time)
                    if timeout <= 0:
                        break

                    request_id, input_data, future = await asyncio.wait_for(
                        self.queue.get(),
                        timeout=timeout
                    )

                    batch.append(input_data)
                    futures.append(future)

                except asyncio.TimeoutError:
                    break

            if not batch:
                continue

            # Execute batch inference
            results = self.engine.predict_batch(batch)

            # Record latency for adaptation
            self.latency_history.append(results[0].latency_ms * len(batch))

            # Distribute results
            for future, result in zip(futures, results):
                future.set_result(result)

            # Adapt batch size
            if self.adaptive:
                self._adapt_batch_size()

    def _adapt_batch_size(self):
        """Adjust batch size based on latency observations."""

        if len(self.latency_history) < 10:
            return

        avg_latency = np.mean(self.latency_history)

        if avg_latency > self.max_wait_ms * 2:
            # Latency too high, reduce batch size
            self.current_batch_size = max(1, self.current_batch_size // 2)
        elif avg_latency < self.max_wait_ms / 2:
            # Can handle more, increase batch size
            self.current_batch_size = min(
                self.max_batch_size,
                self.current_batch_size * 2
            )
```

---

## 9. Indexing and Retrieval

### 9.1 Vector Indexing for Similarity Search

**Use Case:** Finding similar images for quality assessment or duplicate detection.

```python
class VectorIndex:
    """
    Vector index for fast similarity search.
    Uses HNSW algorithm for approximate nearest neighbor.
    """

    def __init__(
        self,
        dimension: int,
        max_elements: int = 100000,
        ef_construction: int = 200,
        M: int = 16
    ):
        self.dimension = dimension

        # Initialize HNSW index
        self.index = hnswlib.Index(space='cosine', dim=dimension)
        self.index.init_index(
            max_elements=max_elements,
            ef_construction=ef_construction,
            M=M
        )

        # Metadata storage
        self.metadata = {}

    def add(
        self,
        vectors: np.ndarray,
        ids: List[str],
        metadata: List[Dict] = None
    ):
        """Add vectors to index."""

        int_ids = [hash(id) % (2**31) for id in ids]
        self.index.add_items(vectors, int_ids)

        for id, int_id, meta in zip(ids, int_ids, metadata or [{}] * len(ids)):
            self.metadata[int_id] = {'id': id, **meta}

    def search(
        self,
        query: np.ndarray,
        k: int = 10
    ) -> List[SearchResult]:
        """Find k nearest neighbors."""

        int_ids, distances = self.index.knn_query(query, k=k)

        results = []
        for int_id, distance in zip(int_ids[0], distances[0]):
            meta = self.metadata.get(int_id, {})
            results.append(SearchResult(
                id=meta.get('id'),
                distance=float(distance),
                metadata=meta
            ))

        return results
```

### 9.2 Temporal Indexing

```python
class TemporalIndex:
    """
    Index for efficient time-range queries.
    Critical for retrieving training data windows.
    """

    def __init__(self):
        self.index = sortedcontainers.SortedDict()

    def add(self, timestamp: float, record_id: str, metadata: Dict = None):
        if timestamp not in self.index:
            self.index[timestamp] = []
        self.index[timestamp].append({
            'id': record_id,
            'metadata': metadata or {}
        })

    def range_query(
        self,
        start: float,
        end: float
    ) -> List[Dict]:
        """Get all records in time range."""

        results = []
        for ts in self.index.irange(start, end):
            results.extend(self.index[ts])
        return results

    def get_recent(self, duration_seconds: float) -> List[Dict]:
        """Get records from the last N seconds."""

        now = time.time()
        return self.range_query(now - duration_seconds, now)
```

---

## 10. Caching Strategies

### 10.1 Multi-Level Cache Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    KEKOA ML Cache Hierarchy                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  L1: Prediction Cache (10ms TTL)                                    │
│  ├── Storage: RAM (50MB)                                            │
│  ├── Key: hash(input_tensor)                                        │
│  ├── Value: prediction + confidence                                 │
│  └── Hit Rate Target: 20-30%                                        │
│       │                                                              │
│       ▼ Miss                                                         │
│                                                                      │
│  L2: Feature Cache (60s TTL)                                        │
│  ├── Storage: RAM (200MB)                                           │
│  ├── Key: hash(raw_input + orbital_context)                         │
│  ├── Value: computed feature vector                                 │
│  └── Hit Rate Target: 40-50%                                        │
│       │                                                              │
│       ▼ Miss                                                         │
│                                                                      │
│  L3: Model Cache (Until new version)                                │
│  ├── Storage: RAM (500MB)                                           │
│  ├── Key: model_version                                             │
│  ├── Value: loaded model weights                                    │
│  └── Purpose: Avoid model reload latency                            │
│       │                                                              │
│       ▼ Miss                                                         │
│                                                                      │
│  L4: Embedding Cache (1 hour TTL)                                   │
│  ├── Storage: Flash (2GB)                                           │
│  ├── Key: content_hash                                              │
│  ├── Value: CNN embedding vector                                    │
│  └── Purpose: Avoid recomputing expensive embeddings                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 10.2 Cache Implementation

```python
class MLCache:
    """
    Multi-level cache for ML inference optimization.
    """

    def __init__(self, config: CacheConfig):
        self.config = config

        # L1: Prediction cache (RAM, LRU)
        self.prediction_cache = LRUCache(
            maxsize=config.prediction_cache_size,
            ttl=config.prediction_ttl_seconds
        )

        # L2: Feature cache (RAM, LRU)
        self.feature_cache = LRUCache(
            maxsize=config.feature_cache_size,
            ttl=config.feature_ttl_seconds
        )

        # L3: Model cache (RAM, no eviction)
        self.model_cache = {}

        # L4: Embedding cache (Flash, LRU)
        self.embedding_cache = PersistentLRUCache(
            path=config.embedding_cache_path,
            maxsize=config.embedding_cache_size,
            ttl=config.embedding_ttl_seconds
        )

        # Metrics
        self.stats = CacheStats()

    def get_prediction(
        self,
        input_hash: str
    ) -> Optional[Prediction]:
        """Check L1 prediction cache."""

        result = self.prediction_cache.get(input_hash)

        if result is not None:
            self.stats.prediction_hits += 1
        else:
            self.stats.prediction_misses += 1

        return result

    def set_prediction(
        self,
        input_hash: str,
        prediction: Prediction
    ):
        """Store prediction in L1 cache."""

        self.prediction_cache.set(input_hash, prediction)

    def get_features(
        self,
        feature_key: str
    ) -> Optional[FeatureVector]:
        """Check L2 feature cache."""

        result = self.feature_cache.get(feature_key)

        if result is not None:
            self.stats.feature_hits += 1
        else:
            self.stats.feature_misses += 1

        return result

    def get_embedding(
        self,
        content_hash: str
    ) -> Optional[np.ndarray]:
        """Check L4 embedding cache (persistent)."""

        result = self.embedding_cache.get(content_hash)

        if result is not None:
            self.stats.embedding_hits += 1
        else:
            self.stats.embedding_misses += 1

        return result

    def get_stats(self) -> CacheStats:
        return CacheStats(
            prediction_hit_rate=self.stats.prediction_hit_rate,
            feature_hit_rate=self.stats.feature_hit_rate,
            embedding_hit_rate=self.stats.embedding_hit_rate,
            total_memory_mb=self._compute_memory_usage(),
        )
```

---

## 11. Scalability Patterns

### 11.1 Scaling Dimensions

| Dimension | Challenge | KEKOA Solution |
|-----------|-----------|----------------|
| Data Volume | More frames per satellite | Streaming processing, intelligent discard |
| Constellation Size | 10 → 1000 satellites | Hierarchical aggregation |
| Model Complexity | Larger models | Quantization, distillation |
| Feature Cardinality | More features | Feature selection, PCA |
| Request Rate | Higher inference QPS | Batching, caching |

### 11.2 Constellation Scaling

```python
class ScalableAggregator:
    """
    Hierarchical aggregation for large constellations.
    Scales to 1000+ satellites.
    """

    def __init__(self, topology: ConstellationTopology):
        self.topology = topology

    def aggregate(
        self,
        gradients: Dict[str, Gradient]
    ) -> Gradient:
        """
        Multi-level aggregation:
        1. Within orbital plane (ring-allreduce)
        2. Across planes (parameter server or tree)
        """

        # Level 1: Aggregate within each orbital plane
        plane_aggregates = {}

        for plane_id, satellite_ids in self.topology.planes.items():
            plane_gradients = {
                sid: gradients[sid]
                for sid in satellite_ids
                if sid in gradients
            }

            if plane_gradients:
                plane_aggregates[plane_id] = self._aggregate_plane(
                    plane_gradients
                )

        # Level 2: Aggregate across planes
        global_gradient = self._aggregate_planes(plane_aggregates)

        return global_gradient

    def _aggregate_plane(
        self,
        gradients: Dict[str, Gradient]
    ) -> Gradient:
        """Ring-allreduce within plane."""

        total_samples = sum(g.sample_count for g in gradients.values())

        aggregated_params = {}
        for name in list(gradients.values())[0].parameters.keys():
            weighted_sum = sum(
                g.parameters[name] * g.sample_count
                for g in gradients.values()
            )
            aggregated_params[name] = weighted_sum / total_samples

        return Gradient(
            parameters=aggregated_params,
            sample_count=total_samples,
            round_id=list(gradients.values())[0].round_id
        )
```

### 11.3 Memory-Efficient Training

```python
class MemoryEfficientTrainer:
    """
    Training techniques for memory-constrained environments.
    """

    def __init__(self, model: nn.Module, config: TrainingConfig):
        self.model = model
        self.config = config

    def train_with_gradient_accumulation(
        self,
        dataloader: DataLoader,
        effective_batch_size: int
    ) -> float:
        """
        Simulate larger batches through gradient accumulation.
        Useful when memory limits actual batch size.
        """

        accumulation_steps = effective_batch_size // self.config.batch_size

        self.optimizer.zero_grad()
        accumulated_loss = 0

        for step, batch in enumerate(dataloader):
            # Forward pass
            loss = self.compute_loss(batch) / accumulation_steps

            # Backward pass (accumulate gradients)
            loss.backward()

            accumulated_loss += loss.item()

            # Update weights every accumulation_steps
            if (step + 1) % accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                accumulated_loss = 0

        return accumulated_loss

    def train_with_gradient_checkpointing(
        self,
        dataloader: DataLoader
    ) -> float:
        """
        Trade compute for memory using gradient checkpointing.
        Recompute activations during backward pass.
        """

        # Enable gradient checkpointing
        self.model.gradient_checkpointing_enable()

        total_loss = 0
        for batch in dataloader:
            loss = self.compute_loss(batch)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            total_loss += loss.item()

        return total_loss / len(dataloader)
```

---

## 12. Fault Tolerance

### 12.1 Fault Categories

| Fault Type | Example | Detection | Recovery |
|------------|---------|-----------|----------|
| **Transient** | Memory spike | Timeout | Retry |
| **Model** | NaN prediction | Output validation | Fallback model |
| **Data** | Corrupted input | Schema validation | Skip frame |
| **Infrastructure** | NPU hang | Watchdog | Restart service |
| **Communication** | ISL loss | Connection timeout | Buffer, retry later |

### 12.2 Fault Tolerance Patterns

```python
class FaultTolerantInference:
    """
    Inference with multiple fallback levels.
    """

    def __init__(
        self,
        primary_model: InferenceEngine,
        fallback_model: InferenceEngine,
        heuristic_fallback: Callable
    ):
        self.primary = primary_model
        self.fallback = fallback_model
        self.heuristic = heuristic_fallback

        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60
        )

    def predict(self, input_data: np.ndarray) -> InferenceResult:
        """
        Cascading fallback:
        1. Primary model (full accuracy)
        2. Fallback model (reduced accuracy)
        3. Heuristic (rule-based)
        """

        # Try primary model
        if self.circuit_breaker.is_closed('primary'):
            try:
                result = self.primary.predict(input_data)
                if result.success and self._validate_output(result):
                    return result
                else:
                    self.circuit_breaker.record_failure('primary')
            except Exception as e:
                self.circuit_breaker.record_failure('primary')

        # Try fallback model
        if self.circuit_breaker.is_closed('fallback'):
            try:
                result = self.fallback.predict(input_data)
                if result.success and self._validate_output(result):
                    result.degraded = True
                    return result
                else:
                    self.circuit_breaker.record_failure('fallback')
            except Exception as e:
                self.circuit_breaker.record_failure('fallback')

        # Last resort: heuristic
        prediction = self.heuristic(input_data)
        return InferenceResult(
            predictions=prediction,
            success=True,
            degraded=True,
            fallback_level='heuristic'
        )

    def _validate_output(self, result: InferenceResult) -> bool:
        """Check for invalid outputs (NaN, out of range)."""

        if result.predictions is None:
            return False

        predictions = np.array(result.predictions)

        # Check for NaN/Inf
        if not np.isfinite(predictions).all():
            return False

        # Check range (assuming probabilities)
        if predictions.min() < 0 or predictions.max() > 1:
            return False

        return True


class CircuitBreaker:
    """
    Circuit breaker pattern to prevent cascading failures.
    """

    def __init__(self, failure_threshold: int, recovery_timeout: float):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.states = {}  # component -> state
        self.failure_counts = {}
        self.last_failure_time = {}

    def is_closed(self, component: str) -> bool:
        """Check if component circuit is closed (healthy)."""

        state = self.states.get(component, 'closed')

        if state == 'open':
            # Check if recovery timeout has passed
            last_failure = self.last_failure_time.get(component, 0)
            if time.time() - last_failure > self.recovery_timeout:
                self.states[component] = 'half-open'
                return True
            return False

        return True

    def record_failure(self, component: str):
        """Record a failure for the component."""

        self.failure_counts[component] = self.failure_counts.get(component, 0) + 1
        self.last_failure_time[component] = time.time()

        if self.failure_counts[component] >= self.failure_threshold:
            self.states[component] = 'open'

    def record_success(self, component: str):
        """Record a success, potentially closing the circuit."""

        self.failure_counts[component] = 0
        self.states[component] = 'closed'
```

---

## 13. Monitoring and Drift Detection

### 13.1 ML Monitoring Framework

```
┌─────────────────────────────────────────────────────────────────────┐
│                    KEKOA ML Monitoring Stack                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Metric Collection                         │   │
│  │                                                              │   │
│  │  System Metrics    Model Metrics     Data Metrics           │   │
│  │  ─────────────    ─────────────     ────────────           │   │
│  │  - CPU/GPU util   - Latency         - Input distribution   │   │
│  │  - Memory usage   - Throughput      - Feature drift        │   │
│  │  - Power draw     - Accuracy*       - Label distribution   │   │
│  │  - Queue depth    - Confidence      - Missing values       │   │
│  │                                                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Anomaly Detection                         │   │
│  │                                                              │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │   │
│  │  │  Statistical│  │   Drift     │  │   Concept   │         │   │
│  │  │  Anomalies  │  │  Detection  │  │    Drift    │         │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘         │   │
│  │                                                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Alert & Response                          │   │
│  │                                                              │   │
│  │  - Log to telemetry                                         │   │
│  │  - Trigger fallback model                                   │   │
│  │  - Schedule retraining                                      │   │
│  │  - Notify ground control                                    │   │
│  │                                                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 13.2 Data Drift Detection

```python
class DataDriftMonitor:
    """
    Monitor for input data distribution shifts.
    """

    def __init__(
        self,
        reference_stats: FeatureStatistics,
        config: DriftConfig
    ):
        self.reference = reference_stats
        self.config = config
        self.window = deque(maxlen=config.window_size)

    def check(self, features: FeatureVector) -> DriftResult:
        """Check for drift in incoming features."""

        self.window.append(features)

        if len(self.window) < self.config.min_samples:
            return DriftResult(detected=False, reason="Insufficient samples")

        current_stats = self._compute_stats(list(self.window))

        drift_scores = {}
        for feature_name in self.reference.feature_names:
            # Kolmogorov-Smirnov test
            ks_stat, p_value = self._ks_test(
                self.reference.get_distribution(feature_name),
                current_stats.get_distribution(feature_name)
            )

            # Population Stability Index
            psi = self._compute_psi(
                self.reference.get_distribution(feature_name),
                current_stats.get_distribution(feature_name)
            )

            drift_scores[feature_name] = {
                'ks_statistic': ks_stat,
                'ks_pvalue': p_value,
                'psi': psi,
                'drifted': psi > self.config.psi_threshold
            }

        drifted_features = [
            name for name, scores in drift_scores.items()
            if scores['drifted']
        ]

        return DriftResult(
            detected=len(drifted_features) > 0,
            drifted_features=drifted_features,
            scores=drift_scores,
            severity=self._compute_severity(drift_scores),
            recommendation=self._recommend_action(drifted_features)
        )

    def _compute_psi(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Population Stability Index.
        PSI < 0.1: No drift
        0.1 <= PSI < 0.2: Moderate drift
        PSI >= 0.2: Significant drift
        """

        # Bin the distributions
        bins = np.linspace(min(expected.min(), actual.min()),
                          max(expected.max(), actual.max()),
                          n_bins + 1)

        expected_hist, _ = np.histogram(expected, bins)
        actual_hist, _ = np.histogram(actual, bins)

        # Normalize to proportions
        expected_prop = expected_hist / expected_hist.sum()
        actual_prop = actual_hist / actual_hist.sum()

        # Add small constant to avoid division by zero
        expected_prop = np.clip(expected_prop, 0.0001, 1)
        actual_prop = np.clip(actual_prop, 0.0001, 1)

        # Calculate PSI
        psi = np.sum((actual_prop - expected_prop) * np.log(actual_prop / expected_prop))

        return psi
```

### 13.3 Model Performance Monitoring

```python
class ModelPerformanceMonitor:
    """
    Monitor model performance metrics over time.
    Detect degradation without ground truth.
    """

    def __init__(self, config: MonitorConfig):
        self.config = config

        # Prediction distribution tracking
        self.prediction_history = deque(maxlen=config.history_size)
        self.confidence_history = deque(maxlen=config.history_size)

        # Baselines
        self.baseline_prediction_dist = None
        self.baseline_confidence_dist = None

    def record(self, prediction: float, confidence: float):
        """Record a prediction for monitoring."""

        self.prediction_history.append(prediction)
        self.confidence_history.append(confidence)

    def check_health(self) -> HealthReport:
        """Assess model health based on prediction patterns."""

        issues = []

        # Check prediction distribution shift
        if self.baseline_prediction_dist is not None:
            pred_shift = self._distribution_shift(
                self.baseline_prediction_dist,
                list(self.prediction_history)
            )
            if pred_shift > self.config.shift_threshold:
                issues.append(Issue(
                    type='prediction_shift',
                    severity='warning',
                    value=pred_shift,
                    message=f"Prediction distribution shifted by {pred_shift:.2f}"
                ))

        # Check confidence calibration
        avg_confidence = np.mean(self.confidence_history)
        if avg_confidence < self.config.min_confidence:
            issues.append(Issue(
                type='low_confidence',
                severity='warning',
                value=avg_confidence,
                message=f"Average confidence dropped to {avg_confidence:.2f}"
            ))

        # Check for prediction collapse
        unique_predictions = len(set(
            round(p, 2) for p in self.prediction_history
        ))
        if unique_predictions < self.config.min_diversity:
            issues.append(Issue(
                type='prediction_collapse',
                severity='critical',
                value=unique_predictions,
                message="Model producing nearly identical predictions"
            ))

        return HealthReport(
            healthy=len([i for i in issues if i.severity == 'critical']) == 0,
            issues=issues,
            timestamp=time.time()
        )
```

---

## 14. Security and Privacy

### 14.1 Security Framework

```
┌─────────────────────────────────────────────────────────────────────┐
│                    KEKOA ML Security Framework                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Input Security                            │   │
│  │  - Schema validation                                         │   │
│  │  - Range checking                                            │   │
│  │  - Anomaly detection                                         │   │
│  │  - Adversarial input detection                               │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Model Security                            │   │
│  │  - Model signing (Ed25519)                                   │   │
│  │  - Weight encryption at rest                                 │   │
│  │  - Gradient validation                                       │   │
│  │  - Poisoning detection                                       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Output Security                           │   │
│  │  - Prediction signing                                        │   │
│  │  - Audit logging                                             │   │
│  │  - Rate limiting                                             │   │
│  │  - Differential privacy (if required)                        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Communication Security                    │   │
│  │  - TLS 1.3 for all links                                    │   │
│  │  - mTLS for peer authentication                             │   │
│  │  - Signed gradients                                         │   │
│  │  - Encrypted model updates                                  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 14.2 Gradient Security

```python
class SecureGradientHandler:
    """
    Security layer for federated learning gradients.
    """

    def __init__(
        self,
        signing_key: Ed25519PrivateKey,
        peer_public_keys: Dict[str, Ed25519PublicKey]
    ):
        self.signing_key = signing_key
        self.peer_keys = peer_public_keys

    def sign_gradient(self, gradient: Gradient) -> SignedGradient:
        """Sign gradient before transmission."""

        # Serialize gradient
        gradient_bytes = self._serialize(gradient)

        # Sign
        signature = self.signing_key.sign(gradient_bytes)

        return SignedGradient(
            gradient=gradient,
            signature=signature,
            public_key_id=self._get_key_id()
        )

    def verify_gradient(
        self,
        signed: SignedGradient
    ) -> Tuple[bool, Optional[str]]:
        """Verify gradient signature."""

        # Get peer public key
        peer_key = self.peer_keys.get(signed.public_key_id)
        if peer_key is None:
            return False, "Unknown peer"

        # Verify signature
        gradient_bytes = self._serialize(signed.gradient)

        try:
            peer_key.verify(signed.signature, gradient_bytes)
            return True, None
        except InvalidSignature:
            return False, "Invalid signature"

    def validate_gradient(
        self,
        gradient: Gradient
    ) -> Tuple[bool, Optional[str]]:
        """Validate gradient for poisoning attacks."""

        # Check for NaN/Inf
        for name, param in gradient.parameters.items():
            if not np.isfinite(param).all():
                return False, f"Non-finite values in {name}"

        # Check gradient norm
        total_norm = np.sqrt(sum(
            np.sum(np.square(p)) for p in gradient.parameters.values()
        ))

        if total_norm > self.config.max_gradient_norm:
            return False, f"Gradient norm {total_norm} exceeds limit"

        # Check for structured noise (potential attack)
        for name, param in gradient.parameters.items():
            if self._is_structured_noise(param):
                return False, f"Structured noise detected in {name}"

        return True, None
```

---

## 15. Design Trade-offs Matrix

### 15.1 Core Trade-offs

| Trade-off | Option A | Option B | Factors | KEKOA Default |
|-----------|----------|----------|---------|---------------|
| **Accuracy vs. Latency** | Complex model (higher accuracy) | Simple model (lower latency) | SLA, power budget | Latency |
| **Freshness vs. Stability** | Frequent retraining | Conservative updates | Drift rate, risk | Stability |
| **Privacy vs. Utility** | Strong DP (high noise) | Weak DP (low noise) | Sensitivity, regulations | Moderate DP |
| **Autonomy vs. Control** | Full onboard decisions | Ground-in-the-loop | Contact frequency | Autonomy |
| **Generalization vs. Specialization** | Single global model | Per-satellite models | Data heterogeneity | Hybrid |
| **Throughput vs. Memory** | Large batches | Small batches | Available RAM | Memory |

### 15.2 Decision Framework

```python
class TradeoffDecisionFramework:
    """
    Framework for making explicit trade-off decisions.
    """

    def analyze_tradeoff(
        self,
        tradeoff_type: str,
        constraints: Dict[str, float],
        priorities: Dict[str, float]
    ) -> TradeoffDecision:

        if tradeoff_type == "accuracy_vs_latency":
            return self._accuracy_latency_tradeoff(constraints, priorities)
        elif tradeoff_type == "freshness_vs_stability":
            return self._freshness_stability_tradeoff(constraints, priorities)
        # ... other trade-offs

    def _accuracy_latency_tradeoff(
        self,
        constraints: Dict[str, float],
        priorities: Dict[str, float]
    ) -> TradeoffDecision:

        latency_budget = constraints.get('latency_ms', 50)
        min_accuracy = constraints.get('min_accuracy', 0.85)

        # Evaluate model candidates
        candidates = self._get_model_candidates()

        valid_candidates = [
            c for c in candidates
            if c.latency_ms <= latency_budget and c.accuracy >= min_accuracy
        ]

        if not valid_candidates:
            return TradeoffDecision(
                selected=None,
                reason="No candidate meets constraints",
                alternatives=candidates[:3]
            )

        # Score by priorities
        scored = []
        for c in valid_candidates:
            score = (
                priorities.get('accuracy', 0.5) * c.accuracy +
                priorities.get('latency', 0.5) * (1 - c.latency_ms / latency_budget)
            )
            scored.append((score, c))

        best = max(scored, key=lambda x: x[0])[1]

        return TradeoffDecision(
            selected=best,
            reason=f"Best score: accuracy={best.accuracy:.3f}, latency={best.latency_ms}ms",
            alternatives=valid_candidates[:3]
        )
```

---

## 16. Case Studies

### 16.1 Case Study: Intelligent Discard System

```
┌─────────────────────────────────────────────────────────────────────┐
│              Case Study: Intelligent Discard System                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  PROBLEM                                                            │
│  ───────                                                            │
│  Earth observation satellites generate 30+ TB/day of imagery.       │
│  >80% is waste (clouds, ocean). Downlinking all data is            │
│  cost-prohibitive. Need onboard intelligence to filter.            │
│                                                                      │
│  SOLUTION ARCHITECTURE                                              │
│  ────────────────────                                               │
│                                                                      │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐            │
│  │ Sensor  │──►│Preproc  │──►│Feature  │──►│Cascade  │            │
│  │ 30fps   │   │Normalize│   │Extract  │   │Classifier│           │
│  └─────────┘   └─────────┘   └─────────┘   └────┬────┘            │
│                                                  │                  │
│                         ┌────────────────────────┴───────┐          │
│                         │                                │          │
│                         ▼                                ▼          │
│                   ┌───────────┐                   ┌───────────┐    │
│                   │  DISCARD  │                   │   KEEP    │    │
│                   │   (80%)   │                   │   (20%)   │    │
│                   └───────────┘                   └─────┬─────┘    │
│                                                         │          │
│                                                         ▼          │
│                                                   ┌───────────┐    │
│                                                   │ Training  │    │
│                                                   │ + Downlink│    │
│                                                   └───────────┘    │
│                                                                      │
│  ML COMPONENTS                                                      │
│  ─────────────                                                      │
│                                                                      │
│  Model: MobileNetV3-Small (2.5M params)                            │
│  Quantization: INT8 (TensorRT)                                     │
│  Latency: 8ms per frame                                            │
│  Accuracy: 94.2% (cloud detection)                                 │
│  Power: 2.5W average                                               │
│                                                                      │
│  Feature Store:                                                     │
│  - Image embedding (512-dim)                                       │
│  - Spectral indices (NDVI, cloud index)                            │
│  - Orbital features (sun angle, latitude)                          │
│                                                                      │
│  Training:                                                          │
│  - Initial: Ground training on 100K labeled images                 │
│  - Updates: Federated learning across constellation                │
│  - Frequency: 1 round per orbit (90 min)                           │
│                                                                      │
│  RESULTS                                                            │
│  ───────                                                            │
│  - Bandwidth reduction: 82%                                        │
│  - Valuable frame loss: 0.8%                                       │
│  - Operating continuously for 6+ months                            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 16.2 Case Study: Federated Anomaly Detection

```
┌─────────────────────────────────────────────────────────────────────┐
│           Case Study: Federated Anomaly Detection                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  PROBLEM                                                            │
│  ───────                                                            │
│  Detect sensor anomalies (drift, attacks) across constellation     │
│  without centralizing sensitive sensor data.                        │
│                                                                      │
│  SOLUTION ARCHITECTURE                                              │
│  ────────────────────                                               │
│                                                                      │
│     SAT-A         SAT-B         SAT-C         SAT-D                │
│     ┌───┐         ┌───┐         ┌───┐         ┌───┐                │
│     │ L │         │ L │         │ L │         │ L │                │
│     │ O │         │ O │         │ O │         │ O │   Local        │
│     │ C │         │ C │         │ C │         │ C │   Anomaly      │
│     │ A │         │ A │         │ A │         │ A │   Models       │
│     │ L │         │ L │         │ L │         │ L │                │
│     └─┬─┘         └─┬─┘         └─┬─┘         └─┬─┘                │
│       │             │             │             │                   │
│       └─────────────┼─────────────┼─────────────┘                   │
│                     │             │                                  │
│                     ▼             ▼                                  │
│              ┌─────────────────────────┐                            │
│              │   Federated Aggregation │                            │
│              │   (Ring-AllReduce)      │                            │
│              └───────────┬─────────────┘                            │
│                          │                                          │
│                          ▼                                          │
│              ┌─────────────────────────┐                            │
│              │   Global Anomaly Model  │                            │
│              │   (Shared across fleet) │                            │
│              └─────────────────────────┘                            │
│                                                                      │
│  ML COMPONENTS                                                      │
│  ─────────────                                                      │
│                                                                      │
│  Model: Autoencoder (encoder: 256→64→16, decoder: 16→64→256)      │
│  Training: FedProx (handles non-IID sensor data)                   │
│  Threshold: Reconstruction error > 3σ                              │
│                                                                      │
│  Anomaly Types:                                                     │
│  - Drift: Gradual reconstruction error increase                    │
│  - Attack: Sudden, structured reconstruction error                 │
│  - Failure: Complete sensor output failure                         │
│                                                                      │
│  RESULTS                                                            │
│  ───────                                                            │
│  - Detection latency: <100ms                                       │
│  - False positive rate: 0.08%                                      │
│  - True positive rate: 97.3%                                       │
│  - No raw sensor data exchanged                                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Appendices

### Appendix A: Metric Definitions

| Metric | Formula | Use Case |
|--------|---------|----------|
| **Accuracy** | (TP + TN) / (TP + TN + FP + FN) | Overall correctness |
| **Precision** | TP / (TP + FP) | When FP is costly |
| **Recall** | TP / (TP + FN) | When FN is costly |
| **F1** | 2 × (Precision × Recall) / (Precision + Recall) | Balanced measure |
| **AUC-ROC** | Area under ROC curve | Ranking quality |
| **PSI** | Σ(Actual - Expected) × ln(Actual/Expected) | Distribution shift |
| **ECE** | Σ |confidence - accuracy| × bin_size | Calibration |

### Appendix B: Model Complexity Reference

| Model | Parameters | FLOPs | Memory | Latency (Orin) |
|-------|------------|-------|--------|----------------|
| MobileNetV3-S | 2.5M | 56M | 10MB | 5ms |
| EfficientNet-B0 | 5.3M | 390M | 21MB | 12ms |
| ResNet-18 | 11.7M | 1.8B | 47MB | 18ms |
| ResNet-50 | 25.6M | 4.1B | 102MB | 35ms |
| YOLOv8-N | 3.2M | 8.7B | 13MB | 8ms |
| YOLOv8-S | 11.2M | 28.6B | 45MB | 15ms |

### Appendix C: Hyperparameter Cheat Sheet

| Context | Learning Rate | Batch Size | Epochs | Optimizer |
|---------|---------------|------------|--------|-----------|
| Ground Pre-training | 0.001 | 64-256 | 100+ | Adam |
| Federated Fine-tuning | 0.01-0.1 | 8-32 | 1-5 | SGD |
| Transfer Learning | 0.0001 | 16-64 | 10-50 | Adam |
| Quantization-Aware | 0.0001 | 32 | 10 | SGD |

### Appendix D: Monitoring Thresholds

| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| Inference Latency (p99) | >40ms | >50ms | Scale/optimize |
| Prediction Confidence | <0.7 | <0.5 | Investigate |
| Data Drift (PSI) | >0.1 | >0.2 | Retrain |
| Error Rate | >2% | >5% | Rollback |
| Memory Usage | >80% | >90% | Clear cache |

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02 | KEKOA ML Platform Engineering | Initial release |

---

*"Production ML is 10% algorithms, 90% systems engineering."*

— KEKOA ML Platform Engineering Principle
