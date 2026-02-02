# KEKOA AI Systems Design Guide

**The Definitive Engineering Reference for Space-Based Machine Learning Systems**

**Version:** 1.0
**Maintainer:** KEKOA ML Engineering
**Last Updated:** February 2026

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [AI System Architecture Layers](#2-ai-system-architecture-layers)
3. [The Orbital ML Pipeline](#3-the-orbital-ml-pipeline)
4. [Data Engineering for Space Systems](#4-data-engineering-for-space-systems)
5. [Model Design Patterns](#5-model-design-patterns)
6. [Federated Learning Architecture](#6-federated-learning-architecture)
7. [Edge Inference Design](#7-edge-inference-design)
8. [Model Serving Infrastructure](#8-model-serving-infrastructure)
9. [MLOps for Orbital Systems](#9-mlops-for-orbital-systems)
10. [Performance Optimization](#10-performance-optimization)
11. [Monitoring and Observability](#11-monitoring-and-observability)
12. [Security and Adversarial Robustness](#12-security-and-adversarial-robustness)
13. [Design Trade-offs](#13-design-trade-offs)
14. [Reference Architectures](#14-reference-architectures)
15. [Anti-Patterns](#15-anti-patterns)
16. [Appendices](#appendices)

---

## 1. Introduction

### 1.1 Purpose

This guide establishes the engineering standards and practices for designing AI/ML systems within the KEKOA platform. It addresses the unique challenges of deploying machine learning on satellite edge compute while orchestrating federated learning across LEO constellations.

### 1.2 Scope

This guide applies to all AI/ML systems developed under KEKOA, including:

- **Inference Pipelines** - Edge inference for Intelligent Discard
- **Federated Learning** - Distributed training via Flame framework
- **Anomaly Detection** - Guardian Agent drift and attack detection (Horizons 2-3)
- **Model Management** - Versioning, deployment, and lifecycle
- **Feature Engineering** - Orbital-aware feature computation

### 1.3 What Makes Space AI Different?

Traditional AI system design assumes:
- Abundant compute resources
- Reliable network connectivity
- Centralized data access
- Easy model updates

Space-based AI inverts all of these assumptions:

| Traditional AI | KEKOA Space AI |
|----------------|----------------|
| Cloud GPUs (A100, H100) | Edge NPU (Jetson Orin, 275 TOPS) |
| Always-connected | Intermittent (orbital windows) |
| Centralized data lake | Distributed, non-IID data |
| Deploy anytime | Deploy during ground contact |
| Unlimited power | Power-constrained (eclipse) |
| Debug interactively | Debug via telemetry |

### 1.4 The KEKOA AI Philosophy

> *"Move logic to data, not data to logic."*

Traditional satellite operations downlink raw data for ground processing. KEKOA inverts this paradigm—we push intelligence to the edge, transmitting only insights (model gradients, inference results) rather than raw sensor data.

**Core Tenets:**

1. **Edge-First Inference**: Process at the source, transmit results
2. **Federated-First Training**: Learn collaboratively without centralizing data
3. **Physics-Aware ML**: Incorporate orbital mechanics into ML pipelines
4. **Graceful Degradation**: Models function in isolation if network fails
5. **Deterministic Reproducibility**: Same inputs → same outputs, always

---

## 2. AI System Architecture Layers

### 2.1 The Three-Layer Model

KEKOA AI systems are organized into three fundamental layers:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         SERVING LAYER                                │
│   Real-time inference, prediction APIs, result aggregation          │
├─────────────────────────────────────────────────────────────────────┤
│                         MODEL LAYER                                  │
│   Training orchestration, aggregation, model registry, validation   │
├─────────────────────────────────────────────────────────────────────┤
│                         DATA LAYER                                   │
│   Data collection, preprocessing, feature engineering, storage      │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Layer Responsibilities

#### Data Layer

| Component | Responsibility | KEKOA Implementation |
|-----------|---------------|---------------------|
| Data Ingestion | Capture sensor data | Sensor interface → frame buffer |
| Preprocessing | Clean, normalize, transform | Onboard preprocessing pipeline |
| Feature Store | Compute and cache features | Local feature cache + orbital features from OAK |
| Data Validation | Ensure data quality | Schema validation, range checks |

#### Model Layer

| Component | Responsibility | KEKOA Implementation |
|-----------|---------------|---------------------|
| Local Training | Train on local data | PyTorch on Jetson NPU |
| Gradient Computation | Generate model updates | Flame LocalTrainer |
| Aggregation | Combine distributed updates | Flame FederatedAggregator |
| Model Registry | Version and track models | Local registry + ground sync |
| Validation | Verify model quality | Holdout validation, metamorphic tests |

#### Serving Layer

| Component | Responsibility | KEKOA Implementation |
|-----------|---------------|---------------------|
| Inference Engine | Execute predictions | ONNX Runtime / TensorRT |
| Early Exit | Fast rejection of invalid data | Cloud cover classifier |
| Result Caching | Cache frequent predictions | LRU cache for repeated patterns |
| Feedback Loop | Collect inference outcomes | Telemetry to ground for analysis |

### 2.3 Cross-Layer Integration

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Satellite Edge Node                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                     SERVING LAYER                             │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌───────────────────┐    │  │
│  │  │  Inference  │  │   Early     │  │   Result          │    │  │
│  │  │   Engine    │  │   Exit      │  │   Cache           │    │  │
│  │  └──────┬──────┘  └──────┬──────┘  └─────────┬─────────┘    │  │
│  │         │                │                   │               │  │
│  │         └────────────────┼───────────────────┘               │  │
│  │                          │ Predictions                       │  │
│  └──────────────────────────┼───────────────────────────────────┘  │
│                             │                                       │
│  ┌──────────────────────────┼───────────────────────────────────┐  │
│  │                     MODEL LAYER                               │  │
│  │                          │                                    │  │
│  │  ┌─────────────┐  ┌──────▼──────┐  ┌───────────────────┐    │  │
│  │  │   Local     │  │   Model     │  │   Gradient        │    │  │
│  │  │   Trainer   │  │   Store     │  │   Buffer          │    │  │
│  │  └──────┬──────┘  └─────────────┘  └─────────┬─────────┘    │  │
│  │         │                                    │               │  │
│  │         │ Training Data        Gradients ────┘               │  │
│  │         │                          │                         │  │
│  └─────────┼──────────────────────────┼─────────────────────────┘  │
│            │                          │                            │
│  ┌─────────┼──────────────────────────┼─────────────────────────┐  │
│  │         │      DATA LAYER          │                          │  │
│  │         │                          ▼                          │  │
│  │  ┌──────▼──────┐  ┌─────────────┐  ┌───────────────────┐    │  │
│  │  │   Feature   │  │   Sensor    │  │   Orbital         │    │  │
│  │  │   Store     │  │   Buffer    │  │   Features (OAK)  │    │  │
│  │  └─────────────┘  └─────────────┘  └───────────────────┘    │  │
│  │                                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ════════════════════════════════════════════════════════════════  │
│                         Lattice Mesh API                            │
│              (Gradient Exchange, Model Sync, Telemetry)             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. The Orbital ML Pipeline

### 3.1 Pipeline Overview

The KEKOA ML pipeline extends the traditional ML pipeline with orbital-aware components:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     KEKOA Orbital ML Pipeline                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐        │
│    │ Sensor  │───►│ Preproc │───►│ Feature │───►│ Early   │        │
│    │ Capture │    │         │    │ Extract │    │ Exit    │        │
│    └─────────┘    └─────────┘    └─────────┘    └────┬────┘        │
│                                                      │              │
│                              ┌───────────────────────┴─────┐        │
│                              │                             │        │
│                              ▼ Discard                     ▼ Keep   │
│                        ┌──────────┐               ┌──────────────┐  │
│                        │ Bandwidth│               │ Local        │  │
│                        │ Saved!   │               │ Training     │  │
│                        └──────────┘               └───────┬──────┘  │
│                                                           │         │
│    ┌─────────────────────────────────────────────────────┘         │
│    │                                                                │
│    ▼                                                                │
│    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐        │
│    │Gradient │───►│ OAK     │───►│ Peer    │───►│ Aggreg- │        │
│    │ Compute │    │ Schedule│    │ Exchange│    │ ation   │        │
│    └─────────┘    └─────────┘    └─────────┘    └────┬────┘        │
│                                                      │              │
│                                                      ▼              │
│                                                ┌──────────┐         │
│                                                │  Model   │         │
│                                                │  Update  │         │
│                                                └────┬─────┘         │
│                                                     │               │
│    ┌────────────────────────────────────────────────┘               │
│    │                                                                │
│    ▼                                                                │
│    ┌─────────┐    ┌─────────┐    ┌─────────┐                       │
│    │Validate │───►│ Deploy  │───►│ Monitor │──────┐                │
│    │         │    │         │    │         │      │                │
│    └─────────┘    └─────────┘    └─────────┘      │                │
│                                                    │                │
│    ┌───────────────────────────────────────────────┘                │
│    │ Feedback Loop (Drift Detection, Retraining Trigger)           │
│    └────────────────────────────────────────────────────────────────┤
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Pipeline Stages

#### Stage 1: Sensor Capture

**Input:** Raw sensor data (imagery, RF, telemetry)
**Output:** Timestamped data frames in standard format

**Design Considerations:**
- Frame rate: Configurable based on mission (1-30 fps typical)
- Resolution: Full resolution captured, downsampled for processing
- Metadata: Timestamp, satellite ID, pointing vector, sun angle

```python
@dataclass
class SensorFrame:
    frame_id: str
    timestamp: Epoch
    satellite_id: str
    sensor_type: SensorType
    data: np.ndarray  # Raw sensor data
    metadata: FrameMetadata

@dataclass
class FrameMetadata:
    pointing_vector: Vector3D  # ECI frame
    sun_angle: float  # Degrees from nadir
    ground_location: Optional[LatLon]
    cloud_cover_estimate: Optional[float]  # From onboard weather model
```

#### Stage 2: Preprocessing

**Input:** Raw sensor frames
**Output:** Normalized, calibrated data ready for feature extraction

**Preprocessing Steps:**
1. **Radiometric Calibration**: Apply sensor-specific gain/offset
2. **Geometric Correction**: Orthorectification if applicable
3. **Normalization**: Scale to [0, 1] or standardize (μ=0, σ=1)
4. **Augmentation** (training only): Rotation, flip, noise injection

**Power-Aware Preprocessing:**
```python
class PowerAwarePreprocessor:
    def __init__(self, power_manager: PowerManager):
        self.power_manager = power_manager

    def process(self, frame: SensorFrame) -> ProcessedFrame:
        power_state = self.power_manager.get_state()

        if power_state == PowerState.CRITICAL:
            # Minimal preprocessing only
            return self.minimal_preprocess(frame)
        elif power_state == PowerState.THROTTLED:
            # Skip expensive augmentation
            return self.standard_preprocess(frame, augment=False)
        else:
            # Full preprocessing pipeline
            return self.full_preprocess(frame)
```

#### Stage 3: Feature Extraction

**Input:** Preprocessed data frames
**Output:** Feature vectors for training and inference

**Feature Categories:**

| Category | Examples | Computation |
|----------|----------|-------------|
| **Image Features** | CNN embeddings, edge density | Neural network forward pass |
| **Spectral Features** | Band ratios, vegetation indices | Arithmetic on bands |
| **Temporal Features** | Change detection, motion vectors | Frame differencing |
| **Orbital Features** | Sun angle, eclipse state, altitude | From OAK |
| **Contextual Features** | Geographic region, season | Lookup from position |

**Orbital Feature Integration:**
```python
class OrbitalFeatureExtractor:
    def __init__(self, oak_client: OAKClient):
        self.oak = oak_client

    def extract(self, frame: SensorFrame) -> OrbitalFeatures:
        state_vector = self.oak.get_state_vector(
            frame.satellite_id,
            frame.timestamp
        )
        power_state = self.oak.get_power_state(frame.satellite_id)

        return OrbitalFeatures(
            altitude_km=state_vector.altitude,
            sun_angle_deg=self.compute_sun_angle(state_vector),
            eclipse_state=power_state.eclipse_state,
            ground_track_velocity=state_vector.ground_velocity,
            orbital_phase=self.compute_orbital_phase(state_vector),
        )
```

#### Stage 4: Early Exit (Intelligent Discard)

**Input:** Feature vectors
**Output:** Binary decision (keep/discard) + confidence

**Design Pattern: Cascading Classifiers**

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Stage 1   │────►│   Stage 2   │────►│   Stage 3   │
│  (Fastest)  │     │  (Medium)   │     │  (Accurate) │
│             │     │             │     │             │
│ Cloud mask  │     │ Scene type  │     │ Full model  │
│ threshold   │     │ classifier  │     │ inference   │
└──────┬──────┘     └──────┬──────┘     └─────────────┘
       │                   │
       ▼ DISCARD           ▼ DISCARD
   (Obvious             (Low value
    clouds)              scenes)
```

**Early Exit Implementation:**
```python
class CascadingEarlyExit:
    """Multi-stage early exit for bandwidth optimization."""

    def __init__(self, stages: List[ExitStage]):
        self.stages = stages  # Ordered fastest → slowest

    def should_discard(self, features: Features) -> Tuple[bool, float, str]:
        """Returns (discard, confidence, reason)."""

        for stage in self.stages:
            decision, confidence = stage.evaluate(features)

            if decision == Decision.DISCARD:
                return True, confidence, stage.name
            elif decision == Decision.KEEP:
                return False, confidence, stage.name
            # Decision.UNCERTAIN → continue to next stage

        # All stages uncertain → keep by default
        return False, 0.5, "uncertain"


class CloudCoverStage(ExitStage):
    """Fast cloud cover detection using spectral ratios."""

    CLOUD_THRESHOLD = 0.90  # 90% cloud cover → discard

    def evaluate(self, features: Features) -> Tuple[Decision, float]:
        cloud_ratio = features.spectral.cloud_index

        if cloud_ratio > self.CLOUD_THRESHOLD:
            return Decision.DISCARD, cloud_ratio
        elif cloud_ratio < 0.10:
            return Decision.KEEP, 1.0 - cloud_ratio
        else:
            return Decision.UNCERTAIN, 0.5
```

#### Stage 5: Local Training

**Input:** Kept frames with features
**Output:** Model gradients

**Design Considerations:**
- **Batch Size**: Constrained by memory (typical: 8-32)
- **Epochs**: Limited by power budget (typical: 1-5 per round)
- **Optimizer**: SGD preferred for FL (momentum optional)

**Federated-Ready Training Loop:**
```python
class LocalTrainer:
    def __init__(self, model: nn.Module, config: TrainingConfig):
        self.model = model
        self.config = config
        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum
        )

    def train_round(
        self,
        dataloader: DataLoader,
        global_model_state: Dict
    ) -> Gradient:
        # Load global model weights
        self.model.load_state_dict(global_model_state)
        initial_params = self.get_params()

        # Local training
        self.model.train()
        samples_processed = 0

        for epoch in range(self.config.local_epochs):
            for batch in dataloader:
                if self.should_stop_training():
                    break

                loss = self.compute_loss(batch)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                samples_processed += len(batch)

        # Compute gradient as weight difference
        final_params = self.get_params()
        gradient = self.compute_gradient(initial_params, final_params)

        return Gradient(
            parameters=gradient,
            sample_count=samples_processed,
            round_id=self.current_round,
            satellite_id=self.satellite_id,
            timestamp=time.time()
        )

    def should_stop_training(self) -> bool:
        """Check power and contact window constraints."""
        power_state = self.power_manager.get_state()
        if power_state == PowerState.CRITICAL:
            return True

        # Check if contact window is approaching
        next_window = self.oak.get_next_contact_window()
        if next_window.time_until_aos < timedelta(minutes=5):
            return True  # Save gradients for transmission

        return False
```

#### Stage 6: Gradient Exchange

**Input:** Local gradients
**Output:** Gradients exchanged with peers

**OAK-Scheduled Exchange:**
```python
class OrbitalGradientExchange:
    def __init__(self, oak: OAKClient, mesh: LatticeMeshClient):
        self.oak = oak
        self.mesh = mesh

    async def exchange_gradients(
        self,
        local_gradient: Gradient
    ) -> List[Gradient]:
        # Get current topology from OAK
        topology = self.oak.get_current_topology()
        neighbors = topology.get_neighbors(self.satellite_id)

        received_gradients = [local_gradient]

        # Exchange with available neighbors
        for neighbor_id in neighbors:
            contact = topology.get_contact(self.satellite_id, neighbor_id)

            if contact.is_active():
                try:
                    # Send our gradient
                    await self.mesh.send(
                        neighbor_id,
                        GradientMessage(local_gradient)
                    )

                    # Receive their gradient
                    response = await self.mesh.receive(
                        neighbor_id,
                        timeout=contact.remaining_duration
                    )
                    received_gradients.append(response.gradient)

                except ContactLostError:
                    # Contact window closed, continue with others
                    pass

        return received_gradients
```

#### Stage 7: Aggregation

**Input:** Collected gradients from peers
**Output:** Aggregated gradient or updated model

**Aggregation Strategies:**

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **FedAvg** | Weighted average by sample count | Standard FL |
| **FedProx** | FedAvg + regularization term | Heterogeneous data |
| **Ring-AllReduce** | Pass-and-add around ring | Single orbital plane |
| **Hierarchical** | Aggregate within plane, then across | Multi-plane constellation |

**Ring-AllReduce Implementation:**
```python
class RingAllReduceAggregator:
    """Efficient gradient aggregation for ring topology."""

    def __init__(self, position_in_ring: int, ring_size: int):
        self.position = position_in_ring
        self.ring_size = ring_size

    def aggregate(
        self,
        local_gradient: Gradient,
        predecessor_partial: Optional[Gradient]
    ) -> Tuple[Gradient, Gradient]:
        """
        Returns (updated_local, forward_to_successor).

        After ring_size - 1 iterations, all nodes have full sum.
        """
        if predecessor_partial is None:
            # First iteration: forward local gradient
            return local_gradient, local_gradient

        # Add predecessor's partial to local
        combined = self.add_gradients(local_gradient, predecessor_partial)

        # Forward combined gradient
        return combined, combined

    def finalize(self, summed_gradient: Gradient) -> Gradient:
        """Normalize by total sample count."""
        return Gradient(
            parameters=summed_gradient.parameters / summed_gradient.sample_count,
            sample_count=summed_gradient.sample_count,
            round_id=summed_gradient.round_id,
            satellite_id="aggregated",
            timestamp=time.time()
        )
```

#### Stage 8: Model Update & Validation

**Input:** Aggregated gradient
**Output:** Updated model (if validation passes)

**Validation Strategy:**
```python
class ModelValidator:
    def __init__(self, holdout_data: DataLoader, threshold: float = 0.95):
        self.holdout = holdout_data
        self.threshold = threshold
        self.baseline_accuracy = None

    def validate(self, new_model: nn.Module) -> ValidationResult:
        accuracy = self.evaluate(new_model)

        if self.baseline_accuracy is None:
            self.baseline_accuracy = accuracy
            return ValidationResult(
                passed=True,
                accuracy=accuracy,
                relative_change=0.0
            )

        relative_change = (accuracy - self.baseline_accuracy) / self.baseline_accuracy

        # Reject if accuracy drops more than 5%
        passed = relative_change > -0.05

        if passed:
            self.baseline_accuracy = accuracy

        return ValidationResult(
            passed=passed,
            accuracy=accuracy,
            relative_change=relative_change
        )
```

#### Stage 9: Deployment & Monitoring

**Input:** Validated model
**Output:** Model deployed to inference engine

**Hot-Swap Deployment:**
```python
class HotSwapDeployer:
    """Deploy new models without inference interruption."""

    def __init__(self, inference_engine: InferenceEngine):
        self.engine = inference_engine
        self.current_version = None
        self.rollback_version = None

    def deploy(self, model: nn.Module, version: str) -> DeployResult:
        # Keep current model as rollback
        self.rollback_version = self.current_version

        # Optimize for inference
        optimized = self.optimize_for_inference(model)

        # Atomic swap
        self.engine.swap_model(optimized)
        self.current_version = version

        # Monitor for regression
        self.start_canary_monitoring(version)

        return DeployResult(
            version=version,
            optimization_stats=optimized.stats,
            rollback_available=self.rollback_version is not None
        )

    def rollback(self):
        if self.rollback_version:
            self.engine.swap_model(self.rollback_version)
            self.current_version = self.rollback_version
            self.rollback_version = None
```

---

## 4. Data Engineering for Space Systems

### 4.1 Data Characteristics

Space-based ML systems deal with unique data challenges:

| Challenge | Description | Mitigation |
|-----------|-------------|------------|
| **Non-IID Distribution** | Each satellite sees different geography | Federated learning, data augmentation |
| **Class Imbalance** | 70%+ ocean, rare event detection | Weighted sampling, focal loss |
| **Temporal Correlation** | Sequential frames are highly similar | Skip frames, change detection |
| **Sensor Drift** | Degradation over mission lifetime | Drift detection (Guardian Agent) |
| **Labeling Scarcity** | Limited ground truth in orbit | Self-supervised learning, pseudo-labels |

### 4.2 Feature Store Design

**Dual-Mode Feature Store:**

```
┌─────────────────────────────────────────────────────────────────┐
│                     KEKOA Feature Store                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────┐    ┌────────────────────────────┐  │
│  │    OFFLINE STORE       │    │      ONLINE STORE          │  │
│  │                        │    │                            │  │
│  │  Purpose: Training     │    │  Purpose: Inference        │  │
│  │  Latency: Seconds OK   │    │  Latency: <10ms required   │  │
│  │  Storage: Flash/Ground │    │  Storage: RAM              │  │
│  │  Features: Historical  │    │  Features: Latest only     │  │
│  │                        │    │                            │  │
│  │  ┌─────────────────┐   │    │  ┌─────────────────────┐   │  │
│  │  │ Training Data   │   │    │  │   Feature Cache     │   │  │
│  │  │ (Parquet/HDF5)  │   │    │  │   (LRU, 1GB max)    │   │  │
│  │  └─────────────────┘   │    │  └─────────────────────┘   │  │
│  │                        │    │                            │  │
│  └────────────────────────┘    └────────────────────────────┘  │
│              │                              ▲                    │
│              │                              │                    │
│              ▼                              │                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Feature Computation Pipeline                │   │
│  │  Raw Data → Preprocessing → Feature Extraction → Store   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 Data Versioning

**Schema Evolution Strategy:**
```protobuf
// Feature schema with version tracking
message FeatureRecord {
  uint32 schema_version = 1;
  string feature_set_id = 2;
  google.protobuf.Timestamp timestamp = 3;

  oneof features {
    FeatureSetV1 v1 = 10;
    FeatureSetV2 v2 = 11;
    // Add new versions here
  }
}

message FeatureSetV1 {
  repeated float image_embedding = 1;  // 512-dim
  float cloud_cover = 2;
  float sun_angle = 3;
}

message FeatureSetV2 {
  repeated float image_embedding = 1;  // 768-dim (upgraded)
  float cloud_cover = 2;
  float sun_angle = 3;
  float vegetation_index = 4;  // New in V2
  EclipseState eclipse_state = 5;  // New in V2
}
```

### 4.4 Data Quality Gates

```python
class DataQualityGate:
    """Validate data before training or inference."""

    QUALITY_CHECKS = [
        ("completeness", lambda x: x.notna().mean() > 0.95),
        ("range_valid", lambda x: ((x >= 0) & (x <= 1)).all()),
        ("no_duplicates", lambda x: x.duplicated().sum() == 0),
        ("temporal_order", lambda x: x.index.is_monotonic_increasing),
    ]

    def validate(self, data: pd.DataFrame) -> QualityReport:
        results = {}
        for name, check in self.QUALITY_CHECKS:
            try:
                results[name] = check(data)
            except Exception as e:
                results[name] = False

        passed = all(results.values())
        return QualityReport(passed=passed, checks=results)
```

---

## 5. Model Design Patterns

### 5.1 Model Selection Criteria

| Criterion | Weight | Consideration |
|-----------|--------|---------------|
| **Inference Latency** | High | Must fit in sensor frame rate |
| **Memory Footprint** | High | Jetson has limited RAM |
| **Power Consumption** | High | NPU vs GPU vs CPU trade-offs |
| **Accuracy** | Medium | Good enough > perfect |
| **Trainability** | Medium | Must work with federated learning |
| **Interpretability** | Low-Medium | Useful for debugging |

### 5.2 Recommended Architectures

#### For Image Classification (Intelligent Discard)

| Model | Params | Latency (Orin) | Accuracy | Recommendation |
|-------|--------|----------------|----------|----------------|
| MobileNetV3-Small | 2.5M | 5ms | 67.4% | **Default choice** |
| EfficientNet-B0 | 5.3M | 12ms | 77.1% | High-accuracy mode |
| ResNet-18 | 11.7M | 18ms | 69.8% | Good FL convergence |
| ViT-Tiny | 5.7M | 25ms | 72.2% | Future consideration |

#### For Object Detection

| Model | Params | Latency (Orin) | mAP | Recommendation |
|-------|--------|----------------|-----|----------------|
| YOLO-NAS-S | 12M | 15ms | 47.5 | **Default choice** |
| YOLOv8-Nano | 3.2M | 8ms | 37.3 | Power-constrained |
| RT-DETR-L | 32M | 45ms | 53.0 | Ground processing |

### 5.3 Model Compression Techniques

**Quantization Pipeline:**
```python
class QuantizationPipeline:
    """Progressive quantization for edge deployment."""

    def quantize(
        self,
        model: nn.Module,
        calibration_data: DataLoader,
        target: str = "int8"
    ) -> QuantizedModel:

        if target == "fp16":
            # Simple half-precision
            return self.to_fp16(model)

        elif target == "int8":
            # Post-training quantization
            return self.ptq_int8(model, calibration_data)

        elif target == "int4":
            # Aggressive quantization (accuracy loss expected)
            return self.ptq_int4(model, calibration_data)

    def ptq_int8(
        self,
        model: nn.Module,
        calibration_data: DataLoader
    ) -> QuantizedModel:
        # Collect activation statistics
        model.eval()
        with torch.no_grad():
            for batch in calibration_data:
                model(batch)

        # Apply quantization
        quantized = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8
        )

        return QuantizedModel(
            model=quantized,
            original_size=self.get_size(model),
            quantized_size=self.get_size(quantized),
            compression_ratio=self.get_size(model) / self.get_size(quantized)
        )
```

### 5.4 Federated Learning-Friendly Design

**Design Principles for FL:**

1. **Avoid Batch Normalization**: Use Group Norm or Layer Norm instead
2. **Initialize Carefully**: Consistent initialization across nodes
3. **Keep It Simple**: Complex architectures diverge in FL

```python
class FLFriendlyConvBlock(nn.Module):
    """Convolution block designed for federated learning."""

    def __init__(self, in_channels: int, out_channels: int, groups: int = 8):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        # Group Norm instead of Batch Norm for FL stability
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))
```

---

## 6. Federated Learning Architecture

### 6.1 Federated Learning Fundamentals

**Why Federated Learning for KEKOA:**

| Benefit | Explanation |
|---------|-------------|
| **Bandwidth Savings** | Gradients (KB) << Raw data (GB) |
| **Data Privacy** | Sensor data never leaves satellite |
| **Resilience** | No single point of failure |
| **Latency** | Local inference, no round-trip |
| **Regulatory** | Data sovereignty compliance |

### 6.2 FL Algorithm Selection

| Algorithm | Synchrony | Best For | KEKOA Use Case |
|-----------|-----------|----------|----------------|
| **FedAvg** | Sync | Baseline, simple | Initial deployment |
| **FedProx** | Sync | Heterogeneous data | Mixed sensors |
| **FedBuff** | Async | Variable availability | **Recommended** |
| **SCAFFOLD** | Sync | Variance reduction | High accuracy needs |
| **FedGSM** | Async | Known staleness | Orbital scheduling |

**FedBuff for Orbital Systems:**
```python
class FedBuffAggregator:
    """
    Buffered asynchronous federated averaging.
    Aggregates when K gradients received, not waiting for all N.
    """

    def __init__(self, K: int, global_model: nn.Module):
        self.K = K  # Buffer size threshold
        self.buffer: List[Gradient] = []
        self.global_model = global_model
        self.round = 0

    def receive_gradient(self, gradient: Gradient) -> Optional[nn.Module]:
        """Returns updated model if aggregation triggered."""

        # Validate gradient freshness
        if gradient.round_id < self.round - 1:
            # Too stale, discard
            return None

        self.buffer.append(gradient)

        if len(self.buffer) >= self.K:
            return self.aggregate_and_update()

        return None

    def aggregate_and_update(self) -> nn.Module:
        # Weighted average by sample count
        total_samples = sum(g.sample_count for g in self.buffer)

        aggregated = {}
        for name, param in self.global_model.named_parameters():
            aggregated[name] = sum(
                g.parameters[name] * g.sample_count / total_samples
                for g in self.buffer
            )

        # Apply aggregated gradient
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                param -= self.learning_rate * aggregated[name]

        # Clear buffer, increment round
        self.buffer = []
        self.round += 1

        return self.global_model
```

### 6.3 Topology-Aware Aggregation

**Integration with OAK:**
```
┌─────────────────────────────────────────────────────────────────┐
│                 Topology-Aware FL Orchestration                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   OAK (Orbital Availability Kernel)                             │
│   │                                                             │
│   ├─► Contact_Window_Table                                      │
│   │   │                                                         │
│   │   └─► Which satellites can communicate when                 │
│   │                                                             │
│   └─► Topology_Graph                                            │
│       │                                                         │
│       └─► Current network structure                             │
│                                                                 │
│   Flame (Federated Learning Engine)                             │
│   │                                                             │
│   ├─► Orbital_Selector                                          │
│   │   │                                                         │
│   │   └─► Selects participants based on OAK topology            │
│   │                                                             │
│   ├─► Gradient_Router                                           │
│   │   │                                                         │
│   │   └─► Routes gradients along available ISL paths            │
│   │                                                             │
│   └─► Hierarchical_Aggregator                                   │
│       │                                                         │
│       └─► Aggregates within planes, then across planes          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.4 Handling Non-IID Data

**Problem:** Each satellite observes different geographic regions, creating non-IID (non-Independent and Identically Distributed) data.

**Solutions:**

| Strategy | Implementation | Trade-off |
|----------|---------------|-----------|
| **FedProx** | Add proximal term to loss | Slower convergence |
| **Data Sharing** | Exchange small data samples | Bandwidth cost |
| **Clustering** | Group similar satellites | Complexity |
| **Personalization** | Local adaptation layers | Larger models |

**Personalization Pattern:**
```python
class PersonalizedModel(nn.Module):
    """
    Model with shared backbone and personalized head.
    Backbone is federated; head is local.
    """

    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone  # Shared across federation
        self.head = head  # Local to each satellite

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

    def get_federated_params(self) -> Dict[str, Tensor]:
        """Only backbone params participate in FL."""
        return {f"backbone.{k}": v for k, v in self.backbone.state_dict().items()}

    def get_local_params(self) -> Dict[str, Tensor]:
        """Head params stay local."""
        return {f"head.{k}": v for k, v in self.head.state_dict().items()}
```

---

## 7. Edge Inference Design

### 7.1 Inference Engine Selection

| Engine | Strengths | Weaknesses | KEKOA Use |
|--------|-----------|------------|-----------|
| **TensorRT** | Fastest on Jetson | NVIDIA-only | Primary |
| **ONNX Runtime** | Cross-platform | Less optimization | Fallback |
| **PyTorch** | Flexibility | Slower | Development |
| **TFLite** | Small footprint | Limited ops | Microcontrollers |

### 7.2 Inference Pipeline Design

```
┌─────────────────────────────────────────────────────────────────┐
│                     Inference Pipeline                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐      │
│  │ Input   │───►│ Preproc │───►│ Infer   │───►│ Postproc│      │
│  │ Queue   │    │ (CPU)   │    │ (NPU)   │    │ (CPU)   │      │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘      │
│       │              │              │              │            │
│       │              │              │              │            │
│       ▼              ▼              ▼              ▼            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Pipeline Controller                   │   │
│  │  - Batching: Accumulate inputs for efficiency           │   │
│  │  - Scheduling: NPU/GPU/CPU task assignment              │   │
│  │  - Memory: Buffer management, zero-copy where possible  │   │
│  │  - Monitoring: Latency tracking, throughput metrics     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 7.3 Latency Optimization

**Batching Strategy:**
```python
class AdaptiveBatcher:
    """
    Dynamically adjust batch size based on queue depth and latency.
    """

    def __init__(
        self,
        min_batch: int = 1,
        max_batch: int = 32,
        max_wait_ms: float = 10.0
    ):
        self.min_batch = min_batch
        self.max_batch = max_batch
        self.max_wait_ms = max_wait_ms
        self.queue: List[InferenceRequest] = []
        self.queue_lock = threading.Lock()

    def add_request(self, request: InferenceRequest):
        with self.queue_lock:
            self.queue.append(request)

    def get_batch(self) -> List[InferenceRequest]:
        """Block until batch is ready."""
        start_time = time.time()

        while True:
            with self.queue_lock:
                queue_size = len(self.queue)
                elapsed_ms = (time.time() - start_time) * 1000

                # Return batch if max size reached or timeout
                if queue_size >= self.max_batch:
                    batch = self.queue[:self.max_batch]
                    self.queue = self.queue[self.max_batch:]
                    return batch

                if queue_size >= self.min_batch and elapsed_ms >= self.max_wait_ms:
                    batch = self.queue[:]
                    self.queue = []
                    return batch

            time.sleep(0.001)  # 1ms polling
```

### 7.4 Power-Aware Inference

```python
class PowerAwareInferenceEngine:
    """Adjust inference behavior based on power state."""

    def __init__(self, models: Dict[str, nn.Module], power_manager: PowerManager):
        self.models = {
            "full": models["full"],          # Full accuracy
            "efficient": models["efficient"], # Reduced accuracy, lower power
            "minimal": models["minimal"],     # Bare minimum
        }
        self.power_manager = power_manager
        self.current_model = "full"

    def infer(self, input: Tensor) -> Tensor:
        power_state = self.power_manager.get_state()

        # Select model based on power
        if power_state == PowerState.NOMINAL:
            model_key = "full"
        elif power_state == PowerState.THROTTLED:
            model_key = "efficient"
        else:  # CRITICAL
            model_key = "minimal"

        # Hot-swap if needed
        if model_key != self.current_model:
            self.current_model = model_key
            self.log_model_switch(model_key, power_state)

        return self.models[self.current_model](input)
```

---

## 8. Model Serving Infrastructure

### 8.1 Model Registry

**Local Model Registry Schema:**
```python
@dataclass
class ModelRecord:
    model_id: str
    version: str
    architecture: str
    parameters_path: str
    metrics: Dict[str, float]
    training_config: Dict[str, Any]
    fl_round: int
    created_at: datetime
    deployed_at: Optional[datetime]
    status: ModelStatus  # TRAINING, VALIDATED, DEPLOYED, DEPRECATED

    # Lineage tracking
    parent_version: Optional[str]
    aggregation_sources: List[str]  # Satellite IDs that contributed


class LocalModelRegistry:
    """SQLite-backed model registry for satellite."""

    def __init__(self, db_path: str):
        self.db = sqlite3.connect(db_path)
        self._init_schema()

    def register(self, model: nn.Module, record: ModelRecord) -> str:
        # Save model weights
        torch.save(model.state_dict(), record.parameters_path)

        # Save metadata
        self._insert_record(record)

        return record.model_id

    def get_deployed_model(self) -> Tuple[nn.Module, ModelRecord]:
        record = self._get_by_status(ModelStatus.DEPLOYED)
        model = self._load_model(record)
        return model, record

    def promote_to_deployed(self, model_id: str):
        # Demote current deployed model
        current = self._get_by_status(ModelStatus.DEPLOYED)
        if current:
            self._update_status(current.model_id, ModelStatus.DEPRECATED)

        # Promote new model
        self._update_status(model_id, ModelStatus.DEPLOYED)
        self._update_deployed_at(model_id, datetime.now())
```

### 8.2 A/B Testing in Orbit

**Shadow Deployment Pattern:**
```python
class ShadowDeployment:
    """
    Run new model in shadow mode alongside production.
    Compare predictions without affecting production.
    """

    def __init__(
        self,
        production_model: nn.Module,
        shadow_model: nn.Module,
        comparator: PredictionComparator
    ):
        self.production = production_model
        self.shadow = shadow_model
        self.comparator = comparator
        self.metrics: List[ComparisonMetric] = []

    def infer(self, input: Tensor) -> Tensor:
        # Production prediction (returned to user)
        prod_output = self.production(input)

        # Shadow prediction (logged only)
        with torch.no_grad():
            shadow_output = self.shadow(input)

        # Compare and log
        comparison = self.comparator.compare(prod_output, shadow_output)
        self.metrics.append(comparison)

        return prod_output

    def get_shadow_report(self) -> ShadowReport:
        return ShadowReport(
            agreement_rate=self._calc_agreement_rate(),
            divergence_distribution=self._calc_divergence_dist(),
            recommendation=self._should_promote_shadow()
        )
```

### 8.3 Model Rollback

```python
class RollbackManager:
    """Automatic rollback on model degradation."""

    def __init__(
        self,
        registry: LocalModelRegistry,
        monitor: InferenceMonitor,
        rollback_threshold: float = 0.10  # 10% accuracy drop
    ):
        self.registry = registry
        self.monitor = monitor
        self.threshold = rollback_threshold
        self.baseline_accuracy: Optional[float] = None

    def check_and_rollback(self) -> Optional[str]:
        current_accuracy = self.monitor.get_rolling_accuracy()

        if self.baseline_accuracy is None:
            self.baseline_accuracy = current_accuracy
            return None

        degradation = (self.baseline_accuracy - current_accuracy) / self.baseline_accuracy

        if degradation > self.threshold:
            # Trigger rollback
            previous_version = self.registry.get_previous_version()
            self.registry.promote_to_deployed(previous_version.model_id)

            # Reset baseline
            self.baseline_accuracy = None

            return previous_version.model_id

        return None
```

---

## 9. MLOps for Orbital Systems

### 9.1 The Orbital MLOps Lifecycle

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Orbital MLOps Lifecycle                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│     GROUND SEGMENT                      SPACE SEGMENT               │
│     ─────────────                      ─────────────                │
│                                                                      │
│   ┌─────────────┐                    ┌─────────────┐               │
│   │   Model     │                    │   Local     │               │
│   │   Research  │                    │   Training  │               │
│   └──────┬──────┘                    └──────┬──────┘               │
│          │                                  │                       │
│          ▼                                  ▼                       │
│   ┌─────────────┐                    ┌─────────────┐               │
│   │   Ground    │                    │   Gradient  │               │
│   │   Training  │                    │   Exchange  │               │
│   └──────┬──────┘                    └──────┬──────┘               │
│          │                                  │                       │
│          ▼                                  ▼                       │
│   ┌─────────────┐    ◄── Uplink ──   ┌─────────────┐               │
│   │   Initial   │────────────────────│ Aggregation │               │
│   │   Model     │                    └──────┬──────┘               │
│   └──────┬──────┘                           │                       │
│          │                                  ▼                       │
│          │                           ┌─────────────┐               │
│          │                           │ Validation  │               │
│          │                           └──────┬──────┘               │
│          │                                  │                       │
│          │                                  ▼                       │
│          │                           ┌─────────────┐               │
│          │                           │ Deployment  │               │
│          │                           └──────┬──────┘               │
│          │                                  │                       │
│          │                                  ▼                       │
│          │       ── Downlink ──►     ┌─────────────┐               │
│          │                           │ Monitoring  │               │
│          │                           └──────┬──────┘               │
│          │                                  │                       │
│          ▼                                  ▼                       │
│   ┌─────────────┐                    ┌─────────────┐               │
│   │  Analysis   │◄───────────────────│  Telemetry  │               │
│   │  & Retrain  │                    │             │               │
│   └─────────────┘                    └─────────────┘               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 9.2 Experiment Tracking

**Lightweight Experiment Tracker:**
```python
class OrbitalExperimentTracker:
    """
    Minimal experiment tracking for satellite environment.
    Syncs to ground during contact windows.
    """

    def __init__(self, storage_path: str):
        self.storage = storage_path
        self.current_experiment: Optional[Experiment] = None

    def start_experiment(self, config: ExperimentConfig) -> str:
        experiment_id = self._generate_id()

        self.current_experiment = Experiment(
            id=experiment_id,
            config=config,
            start_time=datetime.now(),
            metrics=[],
            artifacts=[],
        )

        return experiment_id

    def log_metric(self, name: str, value: float, step: int):
        self.current_experiment.metrics.append(
            Metric(name=name, value=value, step=step, timestamp=datetime.now())
        )

    def log_artifact(self, name: str, path: str):
        self.current_experiment.artifacts.append(
            Artifact(name=name, path=path)
        )

    def end_experiment(self, status: str = "completed"):
        self.current_experiment.end_time = datetime.now()
        self.current_experiment.status = status

        # Persist locally
        self._save_experiment(self.current_experiment)

        # Queue for ground sync
        self._queue_for_sync(self.current_experiment)
```

### 9.3 CI/CD for Space

**Ground-Based Pipeline:**
```yaml
# .github/workflows/model-release.yml
name: Model Release Pipeline

on:
  push:
    paths:
      - 'models/**'
      - 'training/**'

jobs:
  train-and-validate:
    runs-on: gpu-runner
    steps:
      - name: Train on ground data
        run: python train.py --config configs/release.yaml

      - name: Validate accuracy
        run: python validate.py --threshold 0.85

      - name: Run property-based tests
        run: pytest tests/property_tests.py -v

      - name: Quantize for edge
        run: python quantize.py --target int8

      - name: Test on Jetson emulator
        run: python emulator_test.py --platform orin

  package-for-upload:
    needs: train-and-validate
    runs-on: ubuntu-latest
    steps:
      - name: Create deployment package
        run: |
          python package.py \
            --model outputs/model.onnx \
            --config outputs/config.json \
            --version ${{ github.sha }}

      - name: Sign package
        run: python sign.py --key ${{ secrets.SIGNING_KEY }}

      - name: Upload to staging
        run: |
          aws s3 cp outputs/package.tar.gz.sig \
            s3://kekoa-model-staging/${{ github.sha }}/
```

### 9.4 Model Update Protocol

**Uplink Procedure:**
```python
class ModelUplinkManager:
    """Manage model updates during ground contacts."""

    def __init__(self, ground_link: GroundLink, registry: LocalModelRegistry):
        self.link = ground_link
        self.registry = registry

    async def receive_model_update(self) -> Optional[ModelUpdate]:
        """Called during ground contact window."""

        # Check for pending updates
        update_manifest = await self.link.check_for_updates()

        if not update_manifest:
            return None

        # Validate update signature
        if not self.verify_signature(update_manifest):
            self.log_security_event("Invalid model signature")
            return None

        # Download model chunks
        model_data = await self.download_with_resume(
            update_manifest.chunks,
            timeout=self.link.remaining_contact_time()
        )

        if not model_data:
            # Partial download, will resume next contact
            return None

        # Register but don't deploy (requires validation)
        record = ModelRecord(
            model_id=update_manifest.model_id,
            version=update_manifest.version,
            status=ModelStatus.PENDING_VALIDATION,
            ...
        )

        self.registry.register(model_data, record)

        return ModelUpdate(
            model_id=record.model_id,
            action="validate_and_deploy"
        )
```

---

## 10. Performance Optimization

### 10.1 Compute Optimization

**NPU Utilization:**
```python
class JetsonOptimizer:
    """Optimize model for Jetson NPU execution."""

    def optimize(self, model: nn.Module, sample_input: Tensor) -> OptimizedModel:
        # Export to ONNX
        onnx_path = self.export_onnx(model, sample_input)

        # Convert to TensorRT
        trt_engine = self.build_tensorrt_engine(
            onnx_path,
            fp16_mode=True,
            int8_mode=True,
            int8_calibrator=self.create_calibrator(),
            max_batch_size=32,
            max_workspace_size=1 << 30,  # 1GB
        )

        # Profile and validate
        profile = self.profile_engine(trt_engine)

        return OptimizedModel(
            engine=trt_engine,
            input_shape=sample_input.shape,
            latency_ms=profile.median_latency,
            throughput=profile.throughput,
            memory_mb=profile.memory_usage
        )
```

### 10.2 Memory Optimization

**Memory Budget Management:**
```python
class MemoryBudgetManager:
    """Enforce memory limits for ML operations."""

    def __init__(self, total_budget_mb: int):
        self.total = total_budget_mb
        self.allocations: Dict[str, int] = {}

    def allocate(self, component: str, requested_mb: int) -> bool:
        current_usage = sum(self.allocations.values())

        if current_usage + requested_mb > self.total:
            return False

        self.allocations[component] = requested_mb
        return True

    def get_budget_for_component(self, component: str) -> int:
        """Suggest memory budget based on priority."""
        priorities = {
            "inference_engine": 0.40,  # 40% for inference
            "model_cache": 0.20,       # 20% for model cache
            "feature_store": 0.15,     # 15% for features
            "training_buffer": 0.15,   # 15% for training
            "system_overhead": 0.10,   # 10% reserved
        }

        return int(self.total * priorities.get(component, 0.05))
```

### 10.3 Caching Strategy

**Multi-Level Cache:**
```
┌─────────────────────────────────────────────────────────────────┐
│                     KEKOA Cache Hierarchy                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Level 1: Prediction Cache (RAM, 100MB)                         │
│  ├── Key: Hash(input_features)                                  │
│  ├── Value: Prediction result                                   │
│  ├── TTL: 60 seconds                                            │
│  └── Use: Repeated inference on similar frames                  │
│                                                                  │
│  Level 2: Feature Cache (RAM, 200MB)                            │
│  ├── Key: Hash(raw_input)                                       │
│  ├── Value: Computed features                                   │
│  ├── TTL: 5 minutes                                             │
│  └── Use: Avoid recomputing expensive features                  │
│                                                                  │
│  Level 3: Model Cache (RAM, 500MB)                              │
│  ├── Key: Model version                                         │
│  ├── Value: Loaded model weights                                │
│  ├── TTL: Until new model deployed                              │
│  └── Use: Avoid model loading latency                           │
│                                                                  │
│  Level 4: Persistent Cache (Flash, 2GB)                         │
│  ├── Key: Various                                               │
│  ├── Value: Training data, checkpoints                          │
│  ├── TTL: Mission-dependent                                     │
│  └── Use: Survive power cycles                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 10.4 Bandwidth Optimization

**Gradient Compression:**
```python
class GradientCompressor:
    """Compress gradients for transmission over ISL."""

    def compress(
        self,
        gradient: Dict[str, Tensor],
        method: str = "topk"
    ) -> CompressedGradient:

        if method == "topk":
            return self.topk_compression(gradient, k=0.01)  # Keep top 1%
        elif method == "quantize":
            return self.quantize_compression(gradient, bits=8)
        elif method == "random":
            return self.random_sparsification(gradient, p=0.01)
        else:
            raise ValueError(f"Unknown compression method: {method}")

    def topk_compression(
        self,
        gradient: Dict[str, Tensor],
        k: float
    ) -> CompressedGradient:
        compressed = {}
        indices = {}

        for name, tensor in gradient.items():
            flat = tensor.flatten()
            num_elements = int(len(flat) * k)

            # Get top-k by magnitude
            topk_vals, topk_idx = torch.topk(flat.abs(), num_elements)

            compressed[name] = flat[topk_idx]
            indices[name] = topk_idx

        original_size = sum(t.numel() * 4 for t in gradient.values())
        compressed_size = sum(t.numel() * 4 for t in compressed.values())
        compressed_size += sum(t.numel() * 4 for t in indices.values())

        return CompressedGradient(
            values=compressed,
            indices=indices,
            compression_ratio=original_size / compressed_size
        )
```

---

## 11. Monitoring and Observability

### 11.1 ML-Specific Metrics

| Metric Category | Metrics | Collection Frequency |
|-----------------|---------|---------------------|
| **Inference** | Latency (p50, p95, p99), throughput, error rate | Per-request |
| **Model Quality** | Accuracy (rolling), prediction distribution | Per-batch |
| **Data Quality** | Feature drift, missing values, outliers | Per-batch |
| **Resource** | GPU/NPU utilization, memory, power | Per-second |
| **FL Health** | Gradient norm, convergence, participation | Per-round |

### 11.2 Drift Detection

**Data Drift Detection:**
```python
class DriftDetector:
    """Detect distribution shift in input features."""

    def __init__(self, reference_stats: FeatureStatistics):
        self.reference = reference_stats
        self.window: List[FeatureVector] = []
        self.window_size = 1000

    def check(self, features: FeatureVector) -> DriftReport:
        self.window.append(features)

        if len(self.window) < self.window_size:
            return DriftReport(detected=False, score=0.0)

        # Compute current statistics
        current_stats = self.compute_stats(self.window[-self.window_size:])

        # Compare to reference using Kolmogorov-Smirnov test
        drift_scores = {}
        for feature_name in self.reference.features:
            ks_stat, p_value = self.ks_test(
                self.reference.get_distribution(feature_name),
                current_stats.get_distribution(feature_name)
            )
            drift_scores[feature_name] = ks_stat

        max_drift = max(drift_scores.values())

        return DriftReport(
            detected=max_drift > 0.1,  # Threshold
            score=max_drift,
            per_feature_scores=drift_scores,
            recommendation="retrain" if max_drift > 0.2 else "monitor"
        )
```

**Concept Drift Detection:**
```python
class ConceptDriftDetector:
    """Detect when model predictions no longer match reality."""

    def __init__(self, window_size: int = 500):
        self.predictions: List[float] = []
        self.actuals: List[float] = []
        self.window_size = window_size
        self.baseline_error: Optional[float] = None

    def record(self, prediction: float, actual: float):
        self.predictions.append(prediction)
        self.actuals.append(actual)

        # Keep window size
        if len(self.predictions) > self.window_size * 2:
            self.predictions = self.predictions[-self.window_size:]
            self.actuals = self.actuals[-self.window_size:]

    def check(self) -> ConceptDriftReport:
        if len(self.predictions) < self.window_size:
            return ConceptDriftReport(detected=False)

        current_error = self.compute_error(
            self.predictions[-self.window_size:],
            self.actuals[-self.window_size:]
        )

        if self.baseline_error is None:
            self.baseline_error = current_error
            return ConceptDriftReport(detected=False, error=current_error)

        error_increase = (current_error - self.baseline_error) / self.baseline_error

        return ConceptDriftReport(
            detected=error_increase > 0.10,  # 10% degradation
            error=current_error,
            baseline_error=self.baseline_error,
            error_increase=error_increase
        )
```

### 11.3 Telemetry Design

**ML Telemetry Schema:**
```protobuf
message MLTelemetry {
  string satellite_id = 1;
  google.protobuf.Timestamp timestamp = 2;

  // Inference metrics
  InferenceMetrics inference = 3;

  // Model state
  ModelState model = 4;

  // Federated learning state
  FLState fl = 5;

  // Alerts
  repeated Alert alerts = 6;
}

message InferenceMetrics {
  uint64 total_inferences = 1;
  uint64 frames_discarded = 2;
  float discard_rate = 3;

  LatencyStats latency = 4;
  PredictionDistribution predictions = 5;
}

message FLState {
  uint32 current_round = 1;
  uint32 rounds_participated = 2;
  uint32 gradients_sent = 3;
  uint32 gradients_received = 4;
  float local_loss = 5;
  google.protobuf.Timestamp last_aggregation = 6;
}

message Alert {
  AlertSeverity severity = 1;
  string category = 2;  // "drift", "accuracy", "resource", "security"
  string message = 3;
  map<string, string> context = 4;
}
```

### 11.4 Alerting Rules

```yaml
# alerting_rules.yaml
rules:
  - name: InferenceLatencyHigh
    condition: inference.latency.p99 > 100ms
    severity: WARNING
    action: log_and_notify

  - name: AccuracyDrop
    condition: model.rolling_accuracy < 0.80
    severity: CRITICAL
    action: trigger_rollback

  - name: DataDriftDetected
    condition: drift.score > 0.15
    severity: WARNING
    action: log_and_schedule_retrain

  - name: FLConvergenceStall
    condition: fl.rounds_without_improvement > 5
    severity: WARNING
    action: adjust_learning_rate

  - name: GradientAnomaly
    condition: fl.gradient_norm > 10 * fl.historical_gradient_norm
    severity: CRITICAL
    action: reject_gradient_and_alert
```

---

## 12. Security and Adversarial Robustness

### 12.1 Threat Model for Orbital ML

| Threat | Attack Vector | Impact | Mitigation |
|--------|---------------|--------|------------|
| **Model Poisoning** | Malicious gradient injection | Degraded accuracy | Gradient validation, reputation |
| **Data Poisoning** | Corrupted training data | Backdoor in model | Data validation, anomaly detection |
| **Model Stealing** | Gradient reconstruction | IP theft | Differential privacy, encryption |
| **Evasion** | Adversarial inputs | Wrong predictions | Adversarial training, input validation |
| **Inference Attacks** | Model query patterns | Privacy leak | Rate limiting, query obfuscation |

### 12.2 Gradient Validation (Guardian Agent)

```python
class GradientValidator:
    """Validate incoming gradients before aggregation."""

    def __init__(self, config: ValidationConfig):
        self.config = config
        self.gradient_history: Dict[str, List[Tensor]] = defaultdict(list)

    def validate(self, gradient: Gradient) -> ValidationResult:
        checks = [
            self.check_norm(gradient),
            self.check_statistical_anomaly(gradient),
            self.check_structural_pattern(gradient),
            self.check_reputation(gradient.satellite_id),
        ]

        passed = all(c.passed for c in checks)

        return ValidationResult(
            passed=passed,
            checks=checks,
            action="accept" if passed else "reject"
        )

    def check_norm(self, gradient: Gradient) -> Check:
        """Reject gradients with abnormally large norms."""
        norm = self.compute_gradient_norm(gradient)
        threshold = self.config.max_gradient_norm

        return Check(
            name="norm",
            passed=norm < threshold,
            value=norm,
            threshold=threshold
        )

    def check_statistical_anomaly(self, gradient: Gradient) -> Check:
        """Detect statistically anomalous gradients (>3σ)."""
        history = self.gradient_history[gradient.satellite_id]

        if len(history) < 10:
            # Not enough history
            return Check(name="statistical", passed=True)

        mean = torch.stack(history).mean(dim=0)
        std = torch.stack(history).std(dim=0)

        z_score = (gradient.parameters - mean) / (std + 1e-8)
        max_z = z_score.abs().max().item()

        return Check(
            name="statistical",
            passed=max_z < 3.0,
            value=max_z,
            threshold=3.0
        )
```

### 12.3 Differential Privacy for FL

```python
class DifferentiallyPrivateFL:
    """Apply differential privacy to federated learning."""

    def __init__(self, epsilon: float, delta: float, clip_norm: float):
        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm

    def privatize_gradient(self, gradient: Gradient) -> Gradient:
        # Clip gradient norm
        clipped = self.clip_gradient(gradient, self.clip_norm)

        # Add calibrated noise
        noise_scale = self.compute_noise_scale()
        noised = self.add_gaussian_noise(clipped, noise_scale)

        return Gradient(
            parameters=noised,
            sample_count=gradient.sample_count,
            round_id=gradient.round_id,
            satellite_id=gradient.satellite_id,
            timestamp=gradient.timestamp,
            privacy_budget_used=self.epsilon
        )

    def compute_noise_scale(self) -> float:
        """Compute noise scale for Gaussian mechanism."""
        return self.clip_norm * math.sqrt(2 * math.log(1.25 / self.delta)) / self.epsilon
```

### 12.4 Adversarial Robustness

```python
class AdversariallyRobustModel(nn.Module):
    """Model with built-in adversarial defenses."""

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        self.input_preprocessor = InputPreprocessor()

    def forward(self, x: Tensor) -> Tensor:
        # Input preprocessing as defense
        x = self.input_preprocessor(x)

        # Standard forward pass
        return self.base_model(x)

    def adversarial_training_step(
        self,
        x: Tensor,
        y: Tensor,
        epsilon: float = 0.01
    ) -> Tensor:
        """Train with adversarial examples."""
        # Generate adversarial examples (PGD)
        x_adv = self.pgd_attack(x, y, epsilon)

        # Train on both clean and adversarial
        loss_clean = F.cross_entropy(self(x), y)
        loss_adv = F.cross_entropy(self(x_adv), y)

        return 0.5 * loss_clean + 0.5 * loss_adv
```

---

## 13. Design Trade-offs

### 13.1 Fundamental Tensions

| Trade-off | Option A | Option B | KEKOA Default |
|-----------|----------|----------|---------------|
| **Accuracy vs. Latency** | Complex model, high accuracy | Simple model, low latency | Latency (edge constraint) |
| **Privacy vs. Utility** | Minimal data sharing | Full data centralization | Privacy (federated) |
| **Freshness vs. Bandwidth** | Frequent updates | Batched updates | Batched (orbital windows) |
| **Autonomy vs. Control** | Full onboard decisions | Ground-in-the-loop | Autonomy (DDIL) |
| **Complexity vs. Reliability** | Feature-rich | Simple and proven | Reliability |
| **Generalization vs. Personalization** | Global model | Per-satellite model | Hybrid (TAG) |

### 13.2 Decision Framework

```
For each design decision:

1. Identify the trade-off dimension
2. Assess constraints:
   - Power budget?
   - Bandwidth limit?
   - Latency requirement?
   - Security classification?
3. Evaluate options against constraints
4. Document decision and rationale
5. Define reversal conditions
```

### 13.3 Trade-off Documentation Template

```markdown
## Trade-off: Model Complexity vs. Power Consumption

### Context
Selecting model architecture for Intelligent Discard inference.

### Options
| Option | Accuracy | Latency | Power | Memory |
|--------|----------|---------|-------|--------|
| MobileNetV3-Small | 67.4% | 5ms | 2W | 10MB |
| EfficientNet-B0 | 77.1% | 12ms | 5W | 20MB |
| ResNet-18 | 69.8% | 18ms | 8W | 45MB |

### Constraints
- Eclipse power budget: 10W total compute
- Inference latency target: <20ms
- Memory budget: 100MB for inference

### Decision
**MobileNetV3-Small** selected as default.

### Rationale
- Meets latency constraint with margin (5ms < 20ms)
- Lowest power consumption (2W) enables operation during eclipse
- Accuracy sufficient for cloud detection (67% acceptable)
- Memory efficient, leaving room for other components

### Reversal Conditions
- If accuracy proves insufficient (>10% false negatives on valuable scenes)
- If power budget increases (e.g., larger solar panels)
- If customer requires higher accuracy for premium tier

### Approved
- Technical Lead: [Name]
- Date: [Date]
```

---

## 14. Reference Architectures

### 14.1 Intelligent Discard Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Intelligent Discard Architecture                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐                                                    │
│  │   Sensor    │                                                    │
│  │   (Camera)  │                                                    │
│  └──────┬──────┘                                                    │
│         │ Raw frames (30 fps, 12MP)                                 │
│         ▼                                                           │
│  ┌─────────────┐                                                    │
│  │   Frame     │                                                    │
│  │   Buffer    │  Ring buffer, 100 frames                          │
│  └──────┬──────┘                                                    │
│         │                                                           │
│         ▼                                                           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Early Exit Cascade                        │   │
│  │                                                              │   │
│  │  Stage 1          Stage 2          Stage 3                  │   │
│  │  ┌───────┐        ┌───────┐        ┌───────┐               │   │
│  │  │ Cloud │───────►│ Scene │───────►│ Value │               │   │
│  │  │ Mask  │        │ Class │        │ Score │               │   │
│  │  └───┬───┘        └───┬───┘        └───┬───┘               │   │
│  │      │                │                │                    │   │
│  │      ▼                ▼                ▼                    │   │
│  │   Discard          Discard          Keep/Discard           │   │
│  │   (>90% cloud)     (ocean/ice)      (threshold)            │   │
│  │                                                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│         │ Keep                          │ Discard                   │
│         ▼                               ▼                           │
│  ┌─────────────┐                 ┌─────────────┐                   │
│  │  Training   │                 │  Telemetry  │                   │
│  │   Buffer    │                 │  (discard   │                   │
│  │             │                 │   stats)    │                   │
│  └──────┬──────┘                 └─────────────┘                   │
│         │                                                           │
│         ▼                                                           │
│  ┌─────────────┐                                                    │
│  │   Local     │  Update model weights                             │
│  │   Training  │  with kept frames                                 │
│  └─────────────┘                                                    │
│                                                                      │
│  Bandwidth Savings: 80%+ (typical)                                  │
│  Latency: <20ms per frame                                          │
│  Power: 3W average                                                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 14.2 Federated Learning Deployment

```
┌─────────────────────────────────────────────────────────────────────┐
│              Constellation-Wide Federated Learning                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ORBITAL PLANE 1                    ORBITAL PLANE 2                 │
│  ──────────────                    ──────────────                   │
│                                                                      │
│      SAT-1A ◄────► SAT-1B              SAT-2A ◄────► SAT-2B        │
│        ▲              │                  ▲              │           │
│        │              ▼                  │              ▼           │
│      SAT-1F        SAT-1C              SAT-2F        SAT-2C        │
│        ▲              │                  ▲              │           │
│        │              ▼                  │              ▼           │
│      SAT-1E ◄────► SAT-1D              SAT-2E ◄────► SAT-2D        │
│                                                                      │
│         │                                  │                         │
│         └──────────── ISL ─────────────────┘                        │
│                        │                                             │
│                        ▼                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                  Hierarchical Aggregation                    │   │
│  │                                                              │   │
│  │   Phase 1: Intra-plane (Ring-AllReduce)                     │   │
│  │   - SAT-1A..F aggregate within Plane 1                      │   │
│  │   - SAT-2A..F aggregate within Plane 2                      │   │
│  │                                                              │   │
│  │   Phase 2: Inter-plane (Cross-seam ISL)                     │   │
│  │   - Plane leaders exchange aggregates                       │   │
│  │   - Final global model distributed                          │   │
│  │                                                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  Round Duration: ~90 minutes (1 orbit)                              │
│  Participants per Round: 8-12 (availability dependent)             │
│  Gradient Size: 50KB compressed                                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 14.3 Ground-Space ML Sync

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Ground-Space ML Synchronization                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  GROUND SEGMENT                                                      │
│  ──────────────                                                      │
│                                                                      │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐               │
│  │   Model     │   │  Telemetry  │   │   Ground    │               │
│  │   Factory   │   │  Analyzer   │   │   Station   │               │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘               │
│         │                 │                 │                        │
│         │    Ground       │                 │                        │
│         │    Truth        │                 │                        │
│         ▼                 ▼                 ▼                        │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Mission Control Center                    │   │
│  │                                                              │   │
│  │  - Aggregate constellation telemetry                        │   │
│  │  - Train improved models on ground data                     │   │
│  │  - Package and sign model updates                           │   │
│  │  - Schedule uplinks during contact windows                  │   │
│  │                                                              │   │
│  └──────────────────────────┬──────────────────────────────────┘   │
│                             │                                        │
│  ═══════════════════════════╪════════════════════════════════════   │
│              RF UPLINK/DOWNLINK (Contact Windows)                   │
│  ═══════════════════════════╪════════════════════════════════════   │
│                             │                                        │
│  SPACE SEGMENT              │                                        │
│  ─────────────              │                                        │
│                             ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                      Satellite Node                          │   │
│  │                                                              │   │
│  │  ┌───────────────────────────────────────────────────────┐  │   │
│  │  │   Uplink Handler                                       │  │   │
│  │  │   - Receive model updates                              │  │   │
│  │  │   - Verify signatures                                  │  │   │
│  │  │   - Queue for validation                               │  │   │
│  │  └───────────────────────────────────────────────────────┘  │   │
│  │                                                              │   │
│  │  ┌───────────────────────────────────────────────────────┐  │   │
│  │  │   Downlink Handler                                     │  │   │
│  │  │   - Compress telemetry                                 │  │   │
│  │  │   - Prioritize by importance                           │  │   │
│  │  │   - Queue for transmission                             │  │   │
│  │  └───────────────────────────────────────────────────────┘  │   │
│  │                                                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  Uplink Budget: 10MB per contact (model updates)                    │
│  Downlink Budget: 100MB per contact (telemetry + results)           │
│  Contact Frequency: 4-8 per day (ground station dependent)         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 15. Anti-Patterns

### 15.1 Patterns to Avoid

#### Anti-Pattern: Centralized Training Assumption

**Symptom:** Designing training pipelines that require all data in one place.

**Why It's Wrong:** Data cannot be centralized from orbit. Bandwidth is precious.

**Correct Approach:** Design for federated learning from day one. Gradients move, not data.

---

#### Anti-Pattern: Cloud-Scale Resource Assumption

**Symptom:** "We'll just use more GPUs" or "We'll scale horizontally."

**Why It's Wrong:** A satellite has exactly one compute module. No scaling.

**Correct Approach:** Design for fixed resource envelope. Optimize within constraints.

---

#### Anti-Pattern: Interactive Debugging

**Symptom:** Debugging strategy relies on print statements, breakpoints, or SSH access.

**Why It's Wrong:** You cannot SSH into a satellite. Real-time debugging is impossible.

**Correct Approach:** Extensive logging, telemetry, and ground-based replay.

---

#### Anti-Pattern: Frequent Model Updates

**Symptom:** CI/CD pipeline deploys models multiple times per day.

**Why It's Wrong:** Ground contacts are limited. Updates must be carefully scheduled.

**Correct Approach:** Batch model updates. Extensive ground validation before uplink.

---

#### Anti-Pattern: Ignoring Non-IID Data

**Symptom:** Using FedAvg directly without addressing data heterogeneity.

**Why It's Wrong:** Each satellite sees different geography. Models diverge.

**Correct Approach:** Use FedProx, clustering, or personalization strategies.

---

#### Anti-Pattern: Single Model Deployment

**Symptom:** One model version, no fallback, no rollback capability.

**Why It's Wrong:** Bad model update can brick the inference system.

**Correct Approach:** Always maintain rollback version. Shadow deployment for validation.

---

#### Anti-Pattern: Unbounded Resource Consumption

**Symptom:** Training or inference can consume arbitrary memory/compute.

**Why It's Wrong:** Runaway process can destabilize entire satellite.

**Correct Approach:** Hard resource limits, watchdog timers, graceful degradation.

---

## Appendices

### Appendix A: ML Metrics Glossary

| Metric | Definition | Good Value |
|--------|------------|------------|
| **Accuracy** | Correct predictions / total predictions | >85% |
| **Precision** | True positives / predicted positives | >90% |
| **Recall** | True positives / actual positives | >80% |
| **F1 Score** | Harmonic mean of precision and recall | >85% |
| **Latency (p99)** | 99th percentile inference time | <50ms |
| **Throughput** | Inferences per second | >20/s |
| **Discard Rate** | Frames discarded / total frames | 70-90% |
| **Gradient Norm** | L2 norm of gradient vector | <10 |
| **Convergence Rate** | Loss reduction per round | >1% |

### Appendix B: Model Size Reference

| Model | Parameters | Size (FP32) | Size (INT8) |
|-------|------------|-------------|-------------|
| MobileNetV3-Small | 2.5M | 10MB | 2.5MB |
| EfficientNet-B0 | 5.3M | 21MB | 5.3MB |
| ResNet-18 | 11.7M | 47MB | 12MB |
| YOLOv8-Nano | 3.2M | 13MB | 3.2MB |
| ViT-Tiny | 5.7M | 23MB | 5.7MB |

### Appendix C: Jetson Orin Specifications

| Specification | Value |
|--------------|-------|
| AI Performance | 275 TOPS (INT8) |
| GPU | Ampere architecture, 2048 CUDA cores |
| CPU | 12-core Arm Cortex-A78AE |
| Memory | 32GB LPDDR5 |
| Power | 15-60W configurable |
| Storage | NVMe SSD support |

### Appendix D: Federated Learning Hyperparameters

| Parameter | Recommended Range | Notes |
|-----------|------------------|-------|
| Learning Rate | 0.01 - 0.1 | Higher for FL |
| Local Epochs | 1 - 5 | More epochs = more drift |
| Batch Size | 8 - 32 | Memory constrained |
| Clients per Round | 5 - 20 | Availability dependent |
| Aggregation Buffer (K) | 3 - 10 | For FedBuff |
| Gradient Clip Norm | 1.0 - 10.0 | For DP and stability |

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02 | KEKOA ML Engineering | Initial release |

---

*"Intelligence at the edge, insights from the constellation."*

— KEKOA ML Engineering Principle
