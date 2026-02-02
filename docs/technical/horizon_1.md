# Horizon 1: Deterministic Core Platform (OAK + Flame)

## System Design Specification

**Version:** 1.0
**Status:** DRAFT - PENDING APPROVAL
**Scope:** Core Platform Architecture, Formal Verification Strategy, and Simulation Framework

---

## 1. Executive Summary

Horizon 1 delivers the foundational KEKOA platform: the **Orbital Availability Kernel (OAK)** integrated with the **Flame Federated Learning Engine**. This phase focuses on deterministic physics-based scheduling and basic federated learning capabilities, culminating in the "Intelligent Discard" commercial MVP.

This document specifies:
1. System architecture and component design
2. Formal methods verification strategy
3. Simulation and testing framework
4. Acceptance criteria aligned with PRD requirements

---

## 2. System Architecture

### 2.1 Component Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    KEKOA Horizon 1 Container                    │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Application Layer                     │   │
│  │  ┌─────────────────┐    ┌─────────────────────────────┐ │   │
│  │  │  Flame Engine   │◄──►│    Inference Pipeline       │ │   │
│  │  │  (FL + TAG)     │    │  (Early Exit / Discard)     │ │   │
│  │  └────────┬────────┘    └─────────────────────────────┘ │   │
│  └───────────┼──────────────────────────────────────────────┘   │
│              │                                                   │
│  ┌───────────▼──────────────────────────────────────────────┐   │
│  │                    Orchestration Layer                    │   │
│  │  ┌─────────────────────────────────────────────────────┐ │   │
│  │  │              OAK (Orbital Availability Kernel)       │ │   │
│  │  │  ┌───────────┐  ┌───────────┐  ┌────────────────┐  │ │   │
│  │  │  │ TLE Parser│  │   SGP4    │  │ Contact Window │  │ │   │
│  │  │  │           │  │ Propagator│  │   Calculator   │  │ │   │
│  │  │  └───────────┘  └───────────┘  └────────────────┘  │ │   │
│  │  │  ┌───────────┐  ┌───────────┐  ┌────────────────┐  │ │   │
│  │  │  │  Eclipse  │  │   Power   │  │   Topology     │  │ │   │
│  │  │  │ Predictor │  │  Manager  │  │   Generator    │  │ │   │
│  │  │  └───────────┘  └───────────┘  └────────────────┘  │ │   │
│  │  └─────────────────────────────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Transport Layer                        │   │
│  │         Lattice Mesh API (gRPC/Protobuf)                 │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Specifications

#### 2.2.1 OAK (Orbital Availability Kernel)

**Purpose:** Transform orbital mechanics into deterministic scheduling decisions.

| Module | Responsibility | Interface |
|--------|---------------|-----------|
| `TLEParser` | Parse Two-Line Element sets from Space-Track format | `parse(raw: str) -> TLESet` |
| `SGP4Propagator` | Compute satellite position/velocity at any epoch | `propagate(tle: TLESet, t: Epoch) -> StateVector` |
| `ContactWindowCalculator` | Compute ISL visibility windows between satellite pairs | `calculate(s1: StateVector, s2: StateVector) -> ContactWindow` |
| `EclipsePredictor` | Determine Sun/Shadow state for power management | `predict(sv: StateVector, t: Epoch) -> EclipseState` |
| `PowerManager` | Expose power state flags based on eclipse and battery | `get_state() -> PowerStateFlag` |
| `TopologyGenerator` | Build dynamic network graph from contact windows | `generate(windows: List[ContactWindow]) -> TopologyGraph` |

**State Machine: OAK Lifecycle**

```
                    ┌─────────────┐
                    │    INIT     │
                    └──────┬──────┘
                           │ TLE_LOADED
                           ▼
                    ┌─────────────┐
         ┌─────────►│   READY     │◄─────────┐
         │          └──────┬──────┘          │
         │                 │ PROPAGATE       │
         │                 ▼                 │
         │          ┌─────────────┐          │
         │          │ COMPUTING   │          │
         │          └──────┬──────┘          │
         │                 │                 │
         │    ┌────────────┴────────────┐    │
         │    ▼                         ▼    │
    ┌────────────┐               ┌───────────┐
    │  IN_SHADOW │               │  IN_SUN   │
    │ (Throttled)│               │  (Full)   │
    └─────┬──────┘               └─────┬─────┘
          │                            │
          └────────────────────────────┘
                       │ TLE_UPDATE
                       ▼
                ┌─────────────┐
                │  UPDATING   │
                └──────┬──────┘
                       │ COMPLETE
                       └───────────► READY
```

#### 2.2.2 Flame Engine (Federated Learning)

**Purpose:** Execute distributed machine learning with topology-aware orchestration.

| Module | Responsibility | Interface |
|--------|---------------|-----------|
| `TopologyAbstractionGraph` | Abstract network topology from physical routing | `update(graph: TopologyGraph) -> None` |
| `FederatedAggregator` | Aggregate model gradients from peers | `aggregate(gradients: List[Gradient]) -> GlobalModel` |
| `LocalTrainer` | Execute local training rounds | `train(data: DataBatch, model: Model) -> Gradient` |
| `OrbitalSelector` | Select clients based on OAK availability | `select(available: Set[NodeID]) -> Set[NodeID]` |

**State Machine: Flame Training Round**

```
┌──────────────┐
│  IDLE        │
└──────┬───────┘
       │ START_ROUND
       ▼
┌──────────────┐    QUERY_OAK    ┌──────────────┐
│  SELECTING   │────────────────►│ AWAITING_OAK │
└──────────────┘                 └──────┬───────┘
       ▲                                │ TOPOLOGY_RECEIVED
       │ NO_CLIENTS                     ▼
       │                         ┌──────────────┐
       │                         │  TRAINING    │
       │                         └──────┬───────┘
       │                                │
       │         ┌──────────────────────┼──────────────────────┐
       │         ▼                      ▼                      ▼
       │  ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
       │  │ COLLECTING  │       │  TIMEOUT    │       │   ERROR     │
       │  │  GRADIENTS  │       │ (Straggler) │       │             │
       │  └──────┬──────┘       └──────┬──────┘       └──────┬──────┘
       │         │                     │                     │
       │         └─────────────────────┼─────────────────────┘
       │                               ▼
       │                        ┌─────────────┐
       │                        │ AGGREGATING │
       │                        └──────┬──────┘
       │                               │ ROUND_COMPLETE
       │                               ▼
       │                        ┌─────────────┐
       └────────────────────────│    IDLE     │
                                └─────────────┘
```

#### 2.2.3 Inference Pipeline (Intelligent Discard)

**Purpose:** Filter sensor data before training/transmission to save bandwidth.

| Module | Responsibility | Interface |
|--------|---------------|-----------|
| `CloudCoverDetector` | Classify cloud coverage percentage | `detect(frame: ImageFrame) -> float` |
| `EarlyExitClassifier` | Decision gate for frame discard | `should_discard(frame: ImageFrame) -> bool` |
| `BandwidthTracker` | Monitor savings from intelligent discard | `record(discarded: int, transmitted: int) -> Metrics` |

---

## 3. Data Structures & Protocols

### 3.1 Core Data Types

```
TLESet := {
    satellite_id: CatalogNumber,
    epoch: Epoch,
    line1: String[69],
    line2: String[69],
    classification: {U, C, S}  -- Unclassified, Confidential, Secret
}

StateVector := {
    position: Vector3D,      -- ECI frame, km
    velocity: Vector3D,      -- ECI frame, km/s
    epoch: Epoch,
    covariance: Matrix6x6    -- Optional uncertainty
}

ContactWindow := {
    satellite_a: NodeID,
    satellite_b: NodeID,
    aos: Epoch,              -- Acquisition of Signal
    los: Epoch,              -- Loss of Signal
    max_elevation: Degrees,
    min_range: Kilometers
}

PowerStateFlag := {
    state: {NOMINAL, THROTTLED, CRITICAL},
    eclipse_state: {IN_SUN, IN_SHADOW, PENUMBRA},
    battery_percent: Float[0..100],
    compute_budget: Float[0..1]   -- Fraction of full compute
}

TopologyGraph := {
    nodes: Set[NodeID],
    edges: Set[(NodeID, NodeID, ContactWindow)],
    timestamp: Epoch,
    horizon: Duration         -- Validity period (90 min default)
}

Gradient := {
    node_id: NodeID,
    round: RoundNumber,
    parameters: TensorDict,
    sample_count: Integer,
    timestamp: Epoch
}
```

### 3.2 Interface Contracts (gRPC/Protobuf)

```protobuf
service OrbitalAvailability {
    rpc GetContactWindows(TimeRange) returns (ContactWindowTable);
    rpc GetPowerState(NodeID) returns (PowerStateFlag);
    rpc SubscribeTopology(TopologyRequest) returns (stream TopologyGraph);
    rpc UpdateTLE(TLEUpdateRequest) returns (TLEUpdateResponse);
}

service FederatedLearning {
    rpc StartRound(RoundConfig) returns (RoundHandle);
    rpc SubmitGradient(Gradient) returns (SubmitResponse);
    rpc GetGlobalModel(ModelRequest) returns (Model);
    rpc UpdateTopology(TopologyGraph) returns (UpdateResponse);
}
```

---

## 4. Formal Methods Verification Strategy

### 4.1 Verification Approach Overview

We employ a **multi-layer formal verification strategy** combining:

1. **TLA+ Specifications** - Distributed protocol correctness
2. **Property-Based Testing (Hypothesis)** - Implementation verification
3. **Model Checking (SPIN/Promela)** - Concurrency verification
4. **Metamorphic Testing** - Physics oracle validation
5. **Refinement Proofs** - Spec-to-implementation traceability

### 4.2 TLA+ Specifications

#### 4.2.1 OAK Availability Invariants

```tla
---------------------------- MODULE OAK ----------------------------
EXTENDS Integers, Reals, Sequences, FiniteSets

CONSTANTS
    Satellites,         \* Set of satellite IDs
    MaxPropagationError \* Maximum allowed position error (km)

VARIABLES
    tle_cache,          \* Current TLE data per satellite
    state_vectors,      \* Computed positions
    contact_windows,    \* Current contact window table
    power_states,       \* Power state per satellite
    epoch               \* Current simulation time

TypeInvariant ==
    /\ tle_cache \in [Satellites -> TLESet \cup {NULL}]
    /\ state_vectors \in [Satellites -> StateVector \cup {NULL}]
    /\ contact_windows \in SUBSET ContactWindow
    /\ power_states \in [Satellites -> PowerStateFlag]
    /\ epoch \in Epoch

\* CRITICAL INVARIANT: Position accuracy
PositionAccuracy ==
    \A s \in Satellites :
        state_vectors[s] # NULL =>
            PropagationError(state_vectors[s], TruthPosition(s, epoch))
            < MaxPropagationError

\* CRITICAL INVARIANT: Contact window determinism
ContactDeterminism ==
    \A cw1, cw2 \in contact_windows :
        (cw1.satellite_a = cw2.satellite_a /\
         cw1.satellite_b = cw2.satellite_b /\
         cw1.aos = cw2.aos) => cw1 = cw2

\* CRITICAL INVARIANT: Power state consistency
PowerConsistency ==
    \A s \in Satellites :
        (IsInEclipse(state_vectors[s], epoch) =>
            power_states[s].eclipse_state \in {IN_SHADOW, PENUMBRA})

\* SAFETY: No contact window extends beyond horizon
HorizonBound ==
    \A cw \in contact_windows :
        cw.los <= epoch + HORIZON_DURATION

\* LIVENESS: Eventually produce contact table
ContactTableProgress ==
    <>(\A s1, s2 \in Satellites :
        s1 # s2 => ContactComputed(s1, s2))

=======================================================================
```

#### 4.2.2 Flame Federated Learning Protocol

```tla
------------------------- MODULE FlameFL ---------------------------
EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    Nodes,              \* Set of participating nodes
    MaxRounds,          \* Maximum training rounds
    MinQuorum           \* Minimum nodes for aggregation

VARIABLES
    round,              \* Current round number
    node_states,        \* State per node
    gradients,          \* Collected gradients
    global_model,       \* Current global model version
    topology            \* Current TAG

\* Node states
NodeState == {"idle", "training", "submitting", "waiting"}

TypeInvariant ==
    /\ round \in 0..MaxRounds
    /\ node_states \in [Nodes -> NodeState]
    /\ gradients \in [Nodes -> Gradient \cup {NULL}]
    /\ global_model \in Model
    /\ topology \in TopologyGraph

\* SAFETY: Only aggregate from connected nodes
TopologyRespect ==
    \A g \in {n \in Nodes : gradients[n] # NULL} :
        g \in topology.nodes

\* SAFETY: No stale gradients in aggregation
GradientFreshness ==
    \A n \in Nodes :
        gradients[n] # NULL => gradients[n].round = round

\* SAFETY: Quorum requirement
QuorumSafety ==
    (round' > round) =>
        Cardinality({n \in Nodes : gradients[n] # NULL}) >= MinQuorum

\* LIVENESS: Training makes progress
TrainingProgress ==
    [](round < MaxRounds => <>(round' > round))

\* FAIRNESS: All available nodes eventually participate
ParticipationFairness ==
    \A n \in Nodes :
        [](n \in topology.nodes => <>(gradients[n] # NULL))

\* Dynamic topology update (from Guardian Agent, future)
TopologyUpdate(new_topology) ==
    /\ topology' = new_topology
    /\ \* Exclude removed nodes from current round
       gradients' = [n \in Nodes |->
           IF n \in new_topology.nodes THEN gradients[n] ELSE NULL]
    /\ UNCHANGED <<round, node_states, global_model>>

=======================================================================
```

### 4.3 Model Checking with SPIN/Promela

For concurrency verification of the message-passing protocol:

```promela
/* OAK-Flame Coordination Protocol */

mtype = {TLE_UPDATE, TOPOLOGY_REQ, TOPOLOGY_RESP,
         START_ROUND, GRADIENT, AGGREGATE};

chan oak_to_flame = [10] of {mtype, int};
chan flame_to_oak = [10] of {mtype, int};
chan node_to_aggregator = [100] of {mtype, int, int};

int current_round = 0;
int gradients_received = 0;
int MIN_QUORUM = 3;
bool topology_valid = false;

active proctype OAK() {
    int epoch = 0;
    do
    :: flame_to_oak?TOPOLOGY_REQ, _ ->
        /* Compute contact windows */
        atomic {
            topology_valid = true;
            oak_to_flame!TOPOLOGY_RESP, epoch;
        }
    :: timeout ->
        epoch++;
        /* Periodic TLE update check */
    od
}

active proctype FlameAggregator() {
    do
    :: (current_round < 10) ->
        /* Request topology from OAK */
        flame_to_oak!TOPOLOGY_REQ, current_round;
        oak_to_flame?TOPOLOGY_RESP, _;

        /* Wait for gradients */
        do
        :: (gradients_received >= MIN_QUORUM) ->
            /* Aggregate and advance round */
            atomic {
                current_round++;
                gradients_received = 0;
            }
            break;
        :: node_to_aggregator?GRADIENT, _, _ ->
            gradients_received++;
        od
    od
}

/* Verify: No deadlock, eventual progress */
ltl progress { []<>(current_round > 0) }
ltl no_stale_topology { [](gradients_received > 0 -> topology_valid) }
```

### 4.4 Property-Based Testing (Hypothesis)

```python
# Property-based tests for OAK implementation

from hypothesis import given, strategies as st, assume, settings
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant
import numpy as np

class OAKPropertyTests:
    """Property-based tests for Orbital Availability Kernel"""

    @given(
        tle_line1=st.text(min_size=69, max_size=69, alphabet=st.sampled_from("0123456789 .+-")),
        tle_line2=st.text(min_size=69, max_size=69, alphabet=st.sampled_from("0123456789 .+-")),
        propagation_minutes=st.floats(min_value=0, max_value=90)
    )
    def test_propagation_determinism(self, tle_line1, tle_line2, propagation_minutes):
        """Property: Same TLE + epoch always produces same state vector"""
        assume(is_valid_tle(tle_line1, tle_line2))

        sv1 = propagate(tle_line1, tle_line2, propagation_minutes)
        sv2 = propagate(tle_line1, tle_line2, propagation_minutes)

        assert np.allclose(sv1.position, sv2.position, rtol=1e-10)
        assert np.allclose(sv1.velocity, sv2.velocity, rtol=1e-10)

    @given(
        epoch=st.floats(min_value=0, max_value=86400 * 365),
        sat_positions=st.lists(
            st.tuples(
                st.floats(min_value=-42164, max_value=42164),  # GEO radius
                st.floats(min_value=-42164, max_value=42164),
                st.floats(min_value=-42164, max_value=42164)
            ),
            min_size=2, max_size=50
        )
    )
    def test_contact_window_symmetry(self, epoch, sat_positions):
        """Property: Contact window A->B equals B->A"""
        for i, pos_a in enumerate(sat_positions):
            for j, pos_b in enumerate(sat_positions):
                if i >= j:
                    continue
                assume(np.linalg.norm(pos_a) > 6378)  # Above Earth surface
                assume(np.linalg.norm(pos_b) > 6378)

                cw_ab = compute_contact_window(pos_a, pos_b, epoch)
                cw_ba = compute_contact_window(pos_b, pos_a, epoch)

                assert cw_ab.aos == cw_ba.aos
                assert cw_ab.los == cw_ba.los

    @given(
        position=st.tuples(
            st.floats(min_value=-42164, max_value=42164),
            st.floats(min_value=-42164, max_value=42164),
            st.floats(min_value=-42164, max_value=42164)
        ),
        sun_position=st.tuples(
            st.floats(min_value=1.47e8, max_value=1.52e8),
            st.floats(min_value=-1e7, max_value=1e7),
            st.floats(min_value=-1e7, max_value=1e7)
        )
    )
    def test_eclipse_exclusivity(self, position, sun_position):
        """Property: Satellite is in exactly one eclipse state"""
        assume(np.linalg.norm(position) > 6378)

        eclipse_state = compute_eclipse_state(position, sun_position)

        states = [eclipse_state.in_sun, eclipse_state.in_shadow, eclipse_state.in_penumbra]
        assert sum(states) == 1  # Exactly one state is True


class FlameProtocolStateMachine(RuleBasedStateMachine):
    """Stateful property-based testing for Flame protocol"""

    def __init__(self):
        super().__init__()
        self.nodes = set()
        self.round = 0
        self.gradients = {}
        self.topology = set()

    @rule(node_id=st.integers(min_value=0, max_value=99))
    def add_node(self, node_id):
        self.nodes.add(node_id)
        self.topology.add(node_id)

    @rule(node_id=st.integers(min_value=0, max_value=99))
    def remove_node(self, node_id):
        self.topology.discard(node_id)
        # Gradient should be invalidated
        self.gradients.pop(node_id, None)

    @rule(node_id=st.integers(min_value=0, max_value=99))
    def submit_gradient(self, node_id):
        assume(node_id in self.topology)
        self.gradients[node_id] = {"round": self.round, "data": "gradient_data"}

    @rule()
    def aggregate(self):
        assume(len(self.gradients) >= 3)  # MIN_QUORUM
        self.round += 1
        self.gradients = {}

    @invariant()
    def gradients_from_topology_only(self):
        """All gradients must come from nodes in current topology"""
        for node_id in self.gradients:
            assert node_id in self.topology

    @invariant()
    def gradients_are_current(self):
        """No stale gradients from previous rounds"""
        for node_id, gradient in self.gradients.items():
            assert gradient["round"] == self.round
```

### 4.5 Metamorphic Testing for Physics Validation

Since orbital mechanics has no simple oracle, we use **metamorphic relations**:

| Metamorphic Relation | Description | Test Procedure |
|---------------------|-------------|----------------|
| **MR1: Time Symmetry** | Propagate forward T, then backward T = original state | `prop(prop(sv, +T), -T) ≈ sv` |
| **MR2: Keplerian Invariants** | Semi-major axis conserved in 2-body | `sma(sv_t0) ≈ sma(sv_t1)` for close epochs |
| **MR3: Eclipse Periodicity** | Eclipse pattern repeats with orbital period | `eclipse(t) = eclipse(t + T_orbit)` |
| **MR4: Contact Transitivity** | If A sees B and B sees C, valid path exists | Validated via graph connectivity |
| **MR5: Energy Conservation** | Specific orbital energy conserved | `ε(sv_t0) ≈ ε(sv_t1)` |

```python
class MetamorphicOracleTests:
    """Metamorphic testing for orbital mechanics validation"""

    @given(
        tle=valid_tle_strategy(),
        forward_time=st.floats(min_value=1, max_value=180),  # minutes
    )
    def test_time_reversal_symmetry(self, tle, forward_time):
        """MR1: Forward then backward propagation returns to start"""
        sv_initial = propagate(tle, 0)
        sv_forward = propagate(tle, forward_time)
        sv_reversed = propagate_from_state(sv_forward, -forward_time)

        # Allow for numerical precision loss
        assert np.allclose(sv_initial.position, sv_reversed.position, rtol=1e-6)

    @given(
        tle=valid_tle_strategy(),
        t1=st.floats(min_value=0, max_value=90),
        t2=st.floats(min_value=0, max_value=90),
    )
    def test_keplerian_invariants(self, tle, t1, t2):
        """MR2: Semi-major axis conserved in short propagations"""
        assume(abs(t2 - t1) < 10)  # Short time span

        sv1 = propagate(tle, t1)
        sv2 = propagate(tle, t2)

        sma1 = compute_semimajor_axis(sv1)
        sma2 = compute_semimajor_axis(sv2)

        # Allow 0.1% deviation due to perturbations
        assert abs(sma1 - sma2) / sma1 < 0.001

    @given(
        tle=valid_tle_strategy(),
        base_epoch=st.floats(min_value=0, max_value=1440),
    )
    def test_eclipse_periodicity(self, tle, base_epoch):
        """MR3: Eclipse state repeats with orbital period"""
        orbital_period = compute_orbital_period(tle)

        state_t0 = compute_eclipse_state(propagate(tle, base_epoch))
        state_t1 = compute_eclipse_state(propagate(tle, base_epoch + orbital_period))

        # Eclipse state should repeat (approximately, due to Earth rotation)
        assert state_t0.in_shadow == state_t1.in_shadow or \
               abs(base_epoch % orbital_period) < 5  # Near transition
```

---

## 5. Simulation Framework

### 5.1 Simulation Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Simulation Harness                           │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐ │
│  │  Constellation  │  │   Environment   │  │     Scenario        │ │
│  │    Generator    │  │    Injector     │  │     Executor        │ │
│  │                 │  │                 │  │                     │ │
│  │ - Walker-Delta  │  │ - Eclipse Model │  │ - Deterministic     │ │
│  │ - TLE Factory   │  │ - Comm Delays   │  │ - Monte Carlo       │ │
│  │ - Orbit Params  │  │ - Fault Inject  │  │ - Stress Testing    │ │
│  └────────┬────────┘  └────────┬────────┘  └──────────┬──────────┘ │
│           │                    │                      │            │
│           └────────────────────┼──────────────────────┘            │
│                                ▼                                    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    Discrete Event Simulator                   │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────────────┐  │  │
│  │  │   Event    │  │   Clock    │  │     State Manager      │  │  │
│  │  │   Queue    │  │  Manager   │  │   (Checkpointable)     │  │  │
│  │  └────────────┘  └────────────┘  └────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                │                                    │
│  ┌─────────────────────────────┴────────────────────────────────┐  │
│  │                    System Under Test (SUT)                    │  │
│  │         ┌─────────────────────────────────────────┐          │  │
│  │         │   OAK + Flame (Containerized Image)     │          │  │
│  │         └─────────────────────────────────────────┘          │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                │                                    │
│  ┌─────────────────────────────┴────────────────────────────────┐  │
│  │                      Verification Layer                       │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────────────┐  │  │
│  │  │  Invariant │  │   Trace    │  │     Coverage           │  │  │
│  │  │  Checker   │  │  Recorder  │  │     Analyzer           │  │  │
│  │  └────────────┘  └────────────┘  └────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 Constellation Scenarios

| Scenario ID | Description | Satellites | Configuration | Purpose |
|-------------|-------------|------------|---------------|---------|
| `SC-001` | Minimal Pair | 2 | Same orbital plane, 180° apart | Basic ISL validation |
| `SC-002` | Single Plane | 10 | LEO (550km), evenly spaced | Ring topology testing |
| `SC-003` | Walker-Delta Small | 24 | 4 planes × 6 sats, 55° inc | Realistic small constellation |
| `SC-004` | Walker-Delta Medium | 72 | 6 planes × 12 sats, 53° inc | Scaling verification |
| `SC-005` | Heterogeneous | 50 | Mixed altitudes (400-600km) | Cross-shell communication |
| `SC-006` | Polar + Equatorial | 30 | 15 polar + 15 equatorial | Coverage gap analysis |

### 5.3 Fault Injection Scenarios

| Fault ID | Type | Description | Expected Behavior |
|----------|------|-------------|-------------------|
| `FI-001` | TLE Staleness | TLE data 24+ hours old | Graceful degradation, warning |
| `FI-002` | Contact Loss | Simulated ISL failure mid-round | Round continues with remaining nodes |
| `FI-003` | Eclipse Transition | Power state change during training | Gradient submission paused |
| `FI-004` | Topology Partition | Network split into 2 components | Each partition continues independently |
| `FI-005` | Clock Skew | ±500ms epoch disagreement | Contact windows still functional |
| `FI-006` | High Straggler Rate | 40% nodes consistently slow | Async aggregation kicks in |

### 5.4 Deterministic Replay

All simulations support **deterministic replay** through:

1. **Seed Management**: All randomness derives from a single master seed
2. **Event Logging**: Complete event trace with timestamps
3. **State Snapshots**: Periodic checkpoints for fast-forward
4. **Input Recording**: External inputs (TLEs, sensor data) logged

```python
class DeterministicSimulator:
    """Deterministic discrete-event simulator with replay support"""

    def __init__(self, seed: int, scenario: Scenario):
        self.rng = np.random.default_rng(seed)
        self.event_queue = PriorityQueue()
        self.clock = SimulatedClock()
        self.trace = EventTrace()
        self.checkpoints = {}

    def run(self, until: SimTime) -> SimulationResult:
        while self.clock.now < until and not self.event_queue.empty():
            event = self.event_queue.get()
            self.clock.advance_to(event.time)

            # Log for replay
            self.trace.record(event)

            # Execute and collect new events
            new_events = event.execute(self.state)
            for e in new_events:
                self.event_queue.put(e)

            # Periodic checkpoint
            if self.should_checkpoint():
                self.checkpoints[self.clock.now] = self.state.snapshot()

        return SimulationResult(self.trace, self.state)

    def replay_from(self, checkpoint_time: SimTime, until: SimTime):
        """Replay from checkpoint with identical results"""
        self.state = self.checkpoints[checkpoint_time].restore()
        self.clock = SimulatedClock(start=checkpoint_time)
        self.rng = np.random.default_rng(self.seed)  # Reset RNG
        return self.run(until)
```

---

## 6. Acceptance Criteria & Test Plan

### 6.1 Functional Requirements Verification

| Requirement | Verification Method | Acceptance Criteria |
|-------------|--------------------|--------------------|
| **FR-OAK-01** (TLE Ingestion) | Unit Test + Property Test | Parse 100% of valid Space-Track TLEs; <1km propagation error vs. Skyfield reference |
| **FR-OAK-02** (Contact Windows) | Integration Test + Metamorphic | Contact table generated in <1s for 100-sat constellation; symmetric windows verified |
| **FR-OAK-03** (Power State) | Simulation + Property Test | Eclipse prediction matches NASA ephemeris within ±30 seconds |
| **FR-FL-01** (TAG) | TLA+ Model Check + Integration | Protocol proven deadlock-free; dynamic updates complete in <100ms |
| **FR-FL-02** (Early Exit) | Performance Test | >90% cloud frames discarded; <50ms inference latency |
| **FR-FL-03** (Dynamic Topology) | Stateful Property Test | Topology updates don't corrupt in-flight gradients; no restart required |

### 6.2 KPI Verification

| KPI | Test Method | Measurement |
|-----|-------------|-------------|
| Scheduler Overhead < 5% CPU | Load Test | Profile OAK under 1000-sat constellation for 1 hour |
| Propagation Error < 1km | Validation Test | Compare against JPL Horizons for 100 satellites over 24 hours |
| Contact Window Accuracy | Ground Truth | Validate against STK/GMAT for 10 representative passes |
| Topology Update < 100ms | Latency Test | Measure p99 latency over 10,000 topology changes |

### 6.3 Test Pyramid

```
                    ┌───────────────┐
                    │   End-to-End  │  ← 5 Scenario Tests
                    │   (E2E)       │    (SC-001 through SC-005)
                    └───────┬───────┘
                            │
                  ┌─────────┴─────────┐
                  │    Integration    │  ← 20 Integration Tests
                  │    Tests          │    (OAK↔Flame, Flame↔Transport)
                  └─────────┬─────────┘
                            │
            ┌───────────────┴───────────────┐
            │      Component Tests          │  ← 50 Component Tests
            │  (OAK, Flame, Inference)      │    (Per-module verification)
            └───────────────┬───────────────┘
                            │
    ┌───────────────────────┴───────────────────────┐
    │              Property-Based Tests              │  ← 1000+ Generated Tests
    │   (Hypothesis, Metamorphic, Fuzzing)          │    (Per-function invariants)
    └───────────────────────┬───────────────────────┘
                            │
    ┌───────────────────────┴───────────────────────┐
    │              Formal Verification               │
    │   (TLA+ Model Check, SPIN Verification)       │
    └───────────────────────────────────────────────┘
```

---

## 7. Implementation Phases

### Phase 1: Foundation (Weeks 1-3)
- [ ] TLE Parser with property-based tests
- [ ] SGP4 Propagator with metamorphic validation
- [ ] Basic Contact Window Calculator
- [ ] TLA+ specification for OAK state machine

### Phase 2: OAK Core (Weeks 4-6)
- [ ] Eclipse Predictor with NASA ephemeris validation
- [ ] Power Manager integration
- [ ] Topology Generator
- [ ] Model checking with SPIN

### Phase 3: Flame Integration (Weeks 7-9)
- [ ] Topology Abstraction Graph implementation
- [ ] Orbital Selector (OAK → Flame bridge)
- [ ] Federated Aggregator
- [ ] TLA+ verification of FL protocol

### Phase 4: Inference Pipeline (Weeks 10-11)
- [ ] Cloud Cover Detector (PyTorch model)
- [ ] Early Exit Classifier
- [ ] Bandwidth Tracker
- [ ] Integration with Flame training loop

### Phase 5: System Integration (Weeks 12-14)
- [ ] Docker containerization
- [ ] gRPC/Protobuf interface implementation
- [ ] Lattice Mesh API stub integration
- [ ] End-to-end scenario testing

### Phase 6: Validation (Weeks 15-16)
- [ ] Full simulation suite execution
- [ ] KPI verification
- [ ] Performance profiling
- [ ] Documentation and handoff

---

## 8. Dependencies & Technology Stack

### 8.1 Core Dependencies

| Component | Library | Version | Purpose |
|-----------|---------|---------|---------|
| Orbital Mechanics | `sgp4` | 2.22+ | TLE propagation |
| Orbital Mechanics | `skyfield` | 1.45+ | Reference ephemeris, validation |
| Federated Learning | `flame` | 0.3.0+ | FL framework (Georgia Tech) |
| ML Framework | `pytorch` | 2.0+ | Model training/inference |
| RPC | `grpcio` | 1.50+ | Service interfaces |
| Serialization | `protobuf` | 4.0+ | Message encoding |
| Property Testing | `hypothesis` | 6.0+ | Property-based testing |
| Formal Methods | `tla+` | 1.8+ | Specification language |
| Model Checking | `spin` | 6.5+ | Concurrency verification |

### 8.2 Development Tools

| Tool | Purpose |
|------|---------|
| Docker | Containerization |
| pytest | Test framework |
| mypy | Static type checking |
| ruff | Linting |
| coverage | Code coverage |
| hypothesis | Property-based testing |
| tlc | TLA+ model checker |

---

## 9. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| SGP4 accuracy insufficient for ISL timing | Low | High | Cross-validate with high-fidelity propagator (GMAT) |
| Flame framework incompatible with orbital topology | Medium | High | Early prototype of OrbitalSelector; fallback to FedAvg |
| TLA+ specs too complex for meaningful verification | Medium | Medium | Start with core invariants; iterate |
| Eclipse prediction edge cases | Low | Medium | Extensive property testing; comparison with NASA data |
| gRPC latency impacts real-time scheduling | Low | Medium | Local caching; async topology updates |

---

## 10. Appendices

### A. TLA+ Model Checking Commands

```bash
# Check OAK invariants
tlc -config OAK.cfg -workers auto OAK.tla

# Check Flame protocol
tlc -config FlameFL.cfg -workers auto FlameFL.tla

# Generate state space visualization
tlc -dump dot states.dot OAK.tla
```

### B. Hypothesis Test Configuration

```python
# conftest.py
from hypothesis import settings, Verbosity

settings.register_profile("ci", max_examples=1000, deadline=None)
settings.register_profile("dev", max_examples=100, deadline=5000)
settings.register_profile("debug", max_examples=10, verbosity=Verbosity.verbose)
```

### C. Simulation Scenario Definition Format

```yaml
# scenario_sc003.yaml
scenario:
  id: SC-003
  name: Walker-Delta Small
  description: 24-satellite Walker-Delta constellation

constellation:
  type: walker-delta
  planes: 4
  satellites_per_plane: 6
  altitude_km: 550
  inclination_deg: 55
  phasing_factor: 1

simulation:
  duration_hours: 24
  time_step_seconds: 10
  seed: 42

faults:
  - type: contact_loss
    probability: 0.01
    duration_seconds: 60

verification:
  invariants:
    - position_accuracy_km: 1.0
    - contact_window_symmetry: true
    - topology_connected: true
```

---

**Document Status:** Ready for Review
**Next Action:** Stakeholder approval before implementation begins
