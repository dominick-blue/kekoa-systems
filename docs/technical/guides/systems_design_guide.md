# KEKOA Systems Design Guide

**The Definitive Engineering Reference for Space-Based Distributed Systems**

**Version:** 1.0
**Maintainer:** KEKOA Engineering
**Last Updated:** February 2026

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [KEKOA Design Philosophy](#2-kekoa-design-philosophy)
3. [The Seven Quality Pillars](#3-the-seven-quality-pillars)
4. [The KEKOA Design Framework](#4-the-kekoa-design-framework)
5. [Architectural Patterns](#5-architectural-patterns)
6. [Data Modeling for Space Systems](#6-data-modeling-for-space-systems)
7. [Communication Design](#7-communication-design)
8. [Formal Methods Integration](#8-formal-methods-integration)
9. [Design Review Process](#9-design-review-process)
10. [Reference Architectures](#10-reference-architectures)
11. [Anti-Patterns](#11-anti-patterns)
12. [Appendices](#appendices)

---

## 1. Introduction

### 1.1 Purpose

This guide establishes the engineering standards and practices for designing systems within the KEKOA platform. It ensures consistency, quality, and alignment with our mission of enabling deterministic federated learning for LEO satellite constellations.

### 1.2 Scope

This guide applies to all software systems developed under the KEKOA umbrella, including:

- **OAK (Orbital Availability Kernel)** - Physics-based scheduling
- **Flame Integration Layer** - Federated learning orchestration
- **Guardian Agent** - Cyber-physical security (Horizons 2-3)
- **Ground Segment Software** - Mission control interfaces
- **Simulation & Testing Infrastructure** - Verification systems

### 1.3 What is Systems Design?

Systems Design is the disciplined process of defining how software components integrate to satisfy functional requirements while meeting non-functional constraints. For KEKOA, this means bridging:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│    Business     │     │    Systems      │     │   Technical     │
│   Objectives    │────►│    Design       │────►│ Implementation  │
│                 │     │                 │     │                 │
│ - Bandwidth     │     │ - Architecture  │     │ - Code          │
│   savings       │     │ - Interfaces    │     │ - Tests         │
│ - Latency       │     │ - Data models   │     │ - Deployment    │
│   reduction     │     │ - Protocols     │     │ - Operations    │
│ - Resilience    │     │ - Trade-offs    │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### 1.4 Why This Matters for Space Systems

Space systems operate under constraints that terrestrial systems rarely encounter:

| Constraint | Impact | Design Response |
|------------|--------|-----------------|
| **Deterministic Intermittency** | Nodes disconnect predictably (orbital mechanics) | Schedule around physics, not probability |
| **DDIL Environment** | Denied, Disrupted, Intermittent, Limited comms | Autonomous operation, graceful degradation |
| **Power Scarcity** | Eclipse periods, battery limits | Compute budgeting, power-aware scheduling |
| **Radiation Effects** | SEU/SEL events, sensor degradation | Fault tolerance, drift detection |
| **Update Difficulty** | Can't SSH into a satellite easily | Defensive coding, extensive simulation |
| **High Stakes** | Hardware is irreplaceable | Formal verification, exhaustive testing |

---

## 2. KEKOA Design Philosophy

### 2.1 Core Principles

#### Principle 1: Physics First

> *"The orbit dictates the architecture."*

Every design decision must respect orbital mechanics. Contact windows are not random—they are deterministic functions of Keplerian elements. Design systems that exploit this predictability rather than treating it as an obstacle.

**Good:** Pre-computing the next 90 minutes of topology from TLE data.
**Bad:** Implementing retry logic with exponential backoff for "dropped" connections.

#### Principle 2: Determinism Over Convenience

> *"Reproducibility is not optional."*

Space systems cannot be debugged interactively. Every system must be:
- **Reproducible**: Same inputs produce same outputs
- **Traceable**: Every state change is logged
- **Replayable**: Failures can be recreated on the ground

**Good:** Seeded random number generators with logged seeds.
**Bad:** Using system time as an implicit random seed.

#### Principle 3: Graceful Degradation

> *"Partial operation beats total failure."*

Systems must continue providing value even when components fail. Design for the degraded case first, then optimize for the happy path.

**Good:** FL round completes with 6/10 nodes if 4 are in eclipse.
**Bad:** FL round fails entirely if any node is unavailable.

#### Principle 4: Verification Over Testing

> *"Prove correctness; don't just check examples."*

Testing shows the presence of bugs; verification proves their absence. For safety-critical space systems, we employ formal methods alongside traditional testing.

**Good:** TLA+ specification proving deadlock freedom.
**Bad:** "I ran it 100 times and it didn't deadlock."

#### Principle 5: Simplicity is Survivability

> *"Every line of code is a liability in orbit."*

Complex systems fail in complex ways. Prefer simple, well-understood solutions over clever optimizations. The satellite can't call for help when your elegant solution breaks.

**Good:** Ring-all-reduce aggregation following orbital planes.
**Bad:** Adaptive topology optimization with ML-based routing.

### 2.2 The KEKOA Engineering Oath

Every engineer contributing to KEKOA systems commits to:

1. **I will understand the physics** before writing the code.
2. **I will specify before implementing** - formal specs precede implementation.
3. **I will design for failure** - assume every component can fail.
4. **I will verify, not just test** - proofs over examples where possible.
5. **I will document decisions** - rationale matters as much as results.
6. **I will consider operations** - deployability is a first-class requirement.

---

## 3. The Seven Quality Pillars

KEKOA systems must excel across seven quality dimensions. These pillars are not independent—they interact and sometimes conflict. Good design navigates these trade-offs explicitly.

### 3.1 Determinism

**Definition:** Given identical inputs and initial state, the system produces identical outputs.

**Why Critical for KEKOA:** Space systems operate autonomously. Ground operators must predict system behavior from telemetry. Non-deterministic systems are effectively undebuggable.

**Metrics:**
- Bit-exact reproducibility rate: **100%** required
- State divergence after 1M operations: **0 bits**

**Design Checklist:**
- [ ] All random sources use seeded PRNGs with logged seeds
- [ ] Floating-point operations use consistent rounding modes
- [ ] Hash maps iterate in deterministic order (or use ordered maps)
- [ ] Timestamps derive from logical clocks, not wall clocks
- [ ] Thread scheduling is controlled or avoided

### 3.2 Reliability

**Definition:** The system performs its intended function correctly over time.

**Why Critical for KEKOA:** Satellite hardware cannot be repaired. Software must compensate for degrading sensors, radiation-induced faults, and aging components.

**Metrics:**
| Metric | Target | Measurement |
|--------|--------|-------------|
| MTBF (Mean Time Between Failures) | > 10,000 hours | Simulation + flight heritage |
| Recovery Time | < 1 orbit (90 min) | Fault injection testing |
| Data Integrity | 99.9999% | Checksums + ECC verification |

**Design Checklist:**
- [ ] All external inputs are validated before processing
- [ ] Watchdog timers reset on healthy operation
- [ ] State can be reconstructed from persistent storage
- [ ] Fallback modes exist for every critical function
- [ ] Byzantine fault tolerance for multi-node consensus

### 3.3 Availability

**Definition:** The system is operational when needed.

**Why Critical for KEKOA:** Orbital windows are finite. If a system isn't ready when the contact window opens, the opportunity is lost for another 90 minutes.

**Metrics:**
- System availability during contact windows: **> 99.9%**
- Cold start time: **< 30 seconds**
- Hot failover time: **< 1 second**

**Design Checklist:**
- [ ] Pre-computation completes before contact windows open
- [ ] No blocking operations during critical communication periods
- [ ] State checkpoints enable rapid recovery
- [ ] Redundant paths exist for critical data flows

### 3.4 Performance

**Definition:** The system meets latency and throughput requirements.

**Why Critical for KEKOA:** Edge compute on satellites is resource-constrained. Wasted cycles mean wasted power. Slow inference means missed opportunities.

**Metrics:**
| Operation | Latency Target | Throughput Target |
|-----------|---------------|-------------------|
| Orbital propagation (single sat) | < 1 ms | 1000/sec |
| Contact window calculation (pair) | < 10 ms | 100/sec |
| Topology generation (100 sats) | < 1 sec | 1/min |
| Inference (single frame) | < 50 ms | 20/sec |
| Gradient aggregation | < 5 sec | 1/round |

**Design Checklist:**
- [ ] Hot paths are profiled and optimized
- [ ] Memory allocation is minimized in critical sections
- [ ] Batch operations where possible
- [ ] Async I/O for network operations
- [ ] CPU budget allocated and enforced (< 5% for OAK)

### 3.5 Scalability

**Definition:** The system handles growth in load, data, or nodes.

**Why Critical for KEKOA:** Constellations are growing from tens to thousands of satellites. Designs must scale without architectural rewrites.

**Scaling Dimensions:**
- **Constellation size:** 10 → 100 → 1000+ satellites
- **Data volume:** MB → GB → TB per orbit
- **Model complexity:** CNN → Transformer → Foundation models
- **Ground stations:** Single → Distributed → Global network

**Design Checklist:**
- [ ] Algorithms are O(n log n) or better for constellation size
- [ ] State is partitionable across nodes
- [ ] No single points of coordination bottleneck
- [ ] Tested at 10x current scale requirements

### 3.6 Maintainability

**Definition:** The system can be understood, modified, and extended by engineers.

**Why Critical for KEKOA:** Space missions last decades. The engineers who deploy the system may not be the ones maintaining it in year 10.

**Metrics:**
- Time for new engineer to make first meaningful contribution: **< 2 weeks**
- Code coverage: **> 80%**
- Documentation coverage: **100% of public APIs**

**Design Checklist:**
- [ ] Code follows KEKOA style guide
- [ ] Every module has a clear single responsibility
- [ ] Dependencies are explicit and minimal
- [ ] Configuration is externalized and documented
- [ ] Deprecation paths exist for all interfaces

### 3.7 Security

**Definition:** The system resists unauthorized access, modification, and denial of service.

**Why Critical for KEKOA:** Space systems are high-value targets. A compromised satellite can poison the entire federated learning network.

**Threat Model (SPARTA Framework):**
| Threat Category | Example | Mitigation |
|-----------------|---------|------------|
| Reconnaissance | TLE analysis for targeting | Operational security |
| Jamming | RF interference | Spread spectrum, ISL fallback |
| Spoofing | False gradient injection | Cryptographic signing |
| Hijacking | Command injection | Zero-trust authentication |
| Data Exfiltration | Model theft | Differential privacy |

**Design Checklist:**
- [ ] All inputs are sanitized (defense in depth)
- [ ] Authentication for all inter-node communication
- [ ] Encryption at rest and in transit
- [ ] Audit logs for security-relevant events
- [ ] Gradient validation before aggregation (Horizon 2+)

---

## 4. The KEKOA Design Framework

### 4.1 Overview

The KEKOA Design Framework is a structured approach to system design that ensures all critical aspects are addressed. It consists of eight phases:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    KEKOA Design Framework                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Phase 1          Phase 2          Phase 3          Phase 4        │
│  ┌─────────┐      ┌─────────┐      ┌─────────┐      ┌─────────┐   │
│  │ Mission │      │ Require-│      │ Arch.   │      │ Data    │   │
│  │ Context │─────►│ ments   │─────►│ Design  │─────►│ Design  │   │
│  └─────────┘      └─────────┘      └─────────┘      └─────────┘   │
│                                                                     │
│  Phase 5          Phase 6          Phase 7          Phase 8        │
│  ┌─────────┐      ┌─────────┐      ┌─────────┐      ┌─────────┐   │
│  │ Protocol│      │ Formal  │      │ Trade-  │      │ Review  │   │
│  │ Design  │─────►│ Verify  │─────►│ offs    │─────►│ & Doc   │   │
│  └─────────┘      └─────────┘      └─────────┘      └─────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Phase 1: Mission Context

**Goal:** Understand the operational environment and constraints before designing.

**Inputs:**
- Mission concept of operations (CONOPS)
- Constellation parameters (altitude, inclination, plane count)
- Hardware specifications (OBC model, power budget, storage)
- Regulatory requirements (ITAR, encryption export controls)

**Outputs:**
- Mission Context Document
- Constraint Matrix
- Stakeholder Map

**Key Questions:**
1. What orbital regime? (LEO, MEO, GEO, cislunar)
2. What is the communications architecture? (bent-pipe, ISL, optical)
3. What are the power constraints? (eclipse duration, battery capacity)
4. What is the compute budget? (MIPS, memory, storage)
5. What is the mission lifetime? (1 year, 5 years, 15 years)
6. What are the security requirements? (classification level, key management)

**KEKOA-Specific Considerations:**
```
Mission Context Template

Constellation:
  - Type: Walker-Delta / Sun-synchronous / Custom
  - Altitude: ___ km
  - Inclination: ___ degrees
  - Planes: ___
  - Satellites per plane: ___
  - Total satellites: ___

Hardware:
  - OBC: Nvidia Jetson ___ / ARM ___ / Custom
  - Memory: ___ GB RAM
  - Storage: ___ GB
  - Power budget: ___ W (nominal) / ___ W (eclipse)

Communications:
  - ISL: Yes / No
  - ISL data rate: ___ Mbps
  - Ground link: S-band / X-band / Ka-band / Optical
  - Ground link data rate: ___ Mbps

Latency Constraints:
  - Ground contact frequency: ___ per orbit
  - Maximum autonomous period: ___ hours
  - Real-time requirements: ___
```

### 4.3 Phase 2: Requirements Engineering

**Goal:** Capture complete, consistent, and verifiable requirements.

**Functional Requirements (FR):**
What the system must *do*.

**Non-Functional Requirements (NFR):**
How well the system must *perform*.

**Requirements Template:**
```
[REQ-ID]: FR-OAK-001
Category: Functional
Component: OAK
Priority: Must Have (MoSCoW)
Statement: The system SHALL ingest TLE data in Space-Track format
           and propagate satellite positions with < 1 km error.
Rationale: Accurate position data is required for contact window
           calculation and topology generation.
Verification: Test against JPL Horizons reference ephemeris.
Trace: PRD Section 2.1, Stakeholder Interview 2024-01-15
```

**Non-Functional Requirements Categories:**

| Category | Abbreviation | Example |
|----------|--------------|---------|
| Performance | NFR-PERF | Latency < 50ms p99 |
| Scalability | NFR-SCALE | Support 1000+ satellites |
| Reliability | NFR-REL | MTBF > 10,000 hours |
| Security | NFR-SEC | AES-256 encryption required |
| Maintainability | NFR-MAINT | 80% code coverage |
| Portability | NFR-PORT | Run on Jetson Orin/AGX |
| Determinism | NFR-DET | Bit-exact reproducibility |

### 4.4 Phase 3: Architecture Design

**Goal:** Define the high-level structure and major component interactions.

**Architecture Artifacts:**

1. **Context Diagram:** System boundary and external interfaces
2. **Container Diagram:** Major deployable units
3. **Component Diagram:** Internal structure of containers
4. **Deployment Diagram:** Physical/virtual infrastructure mapping

**KEKOA Standard Architecture Layers:**

```
┌─────────────────────────────────────────────────────────────────┐
│                      Application Layer                          │
│  (Flame Engine, Inference Pipeline, Mission-Specific Logic)    │
├─────────────────────────────────────────────────────────────────┤
│                    Orchestration Layer                          │
│  (OAK, Power Manager, Guardian Agent)                          │
├─────────────────────────────────────────────────────────────────┤
│                      Platform Layer                             │
│  (Lattice SDK, Container Runtime, OS Abstraction)              │
├─────────────────────────────────────────────────────────────────┤
│                      Transport Layer                            │
│  (gRPC, Protobuf, Lattice Mesh API)                            │
├─────────────────────────────────────────────────────────────────┤
│                      Hardware Layer                             │
│  (Jetson OBC, Sensors, ISL Radio, Ground Radio)                │
└─────────────────────────────────────────────────────────────────┘
```

**Interface Definition Requirements:**
- All inter-component interfaces defined in Protobuf
- Versioned schemas with backward compatibility rules
- Clear ownership for each interface

### 4.5 Phase 4: Data Design

**Goal:** Define data structures, storage, and lifecycle.

**Data Categories for Space Systems:**

| Category | Volatility | Storage | Example |
|----------|------------|---------|---------|
| Ephemeral | Seconds | Memory only | Sensor frame buffer |
| Session | Minutes-Hours | Memory + checkpoint | Training round state |
| Persistent | Days-Months | Flash storage | Model weights, TLE cache |
| Archival | Mission lifetime | Ground storage | Telemetry logs |

**Data Design Checklist:**
- [ ] Every data type has a defined schema (Protobuf/JSON Schema)
- [ ] Data lifecycle is documented (creation, update, deletion)
- [ ] Storage requirements are calculated for mission lifetime
- [ ] Corruption detection (checksums) is implemented
- [ ] Recovery procedures exist for each data category

**State Machine Documentation:**
Every stateful component requires:
1. State diagram with all states and transitions
2. Invariants that hold in each state
3. Guard conditions for each transition
4. Actions performed on transitions

### 4.6 Phase 5: Protocol Design

**Goal:** Define communication protocols and message flows.

**Protocol Design Template:**
```
Protocol: Gradient Exchange
Version: 1.0
Participants: Local Trainer (LT), Aggregator (AGG)
Preconditions: Training round active, LT in TAG topology

Message Flow:
1. AGG -> LT: ROUND_START {round_id, global_model_hash}
2. LT: Validate model hash, begin local training
3. LT -> AGG: GRADIENT_SUBMIT {round_id, gradient, sample_count, signature}
4. AGG: Validate signature, buffer gradient
5. AGG -> LT: GRADIENT_ACK {round_id, status}

Error Handling:
- Timeout (30s): LT retries up to 3 times
- Invalid signature: AGG rejects, logs security event
- Stale round_id: AGG responds with ROUND_EXPIRED

Idempotency: GRADIENT_SUBMIT is idempotent (same gradient can be resubmitted)
```

**Timing Diagrams:**
Use sequence diagrams for complex multi-party protocols.

### 4.7 Phase 6: Formal Verification

**Goal:** Prove critical properties hold for all possible executions.

**When to Apply Formal Methods:**

| Criticality | Verification Approach |
|-------------|----------------------|
| Safety-critical (loss of mission) | Full TLA+ specification + model checking |
| Mission-critical (degraded operation) | Property-based testing + key invariants |
| Operational (inconvenience) | Traditional unit/integration testing |

**Formal Verification Artifacts:**

1. **TLA+ Specification:** State machine and invariants
2. **Model Checking Results:** TLC output showing states explored
3. **Property Test Suite:** Hypothesis tests for implementation
4. **Metamorphic Relations:** For oracle-free domains (physics)

**Minimum Formal Verification for KEKOA:**
- [ ] State machine specs for all stateful components
- [ ] Deadlock freedom proof for distributed protocols
- [ ] Liveness proof (progress guarantees)
- [ ] Property-based tests covering all public APIs

### 4.8 Phase 7: Trade-off Analysis

**Goal:** Explicitly document design trade-offs and their rationale.

**Trade-off Documentation Template:**
```
Trade-off: Synchronous vs. Asynchronous FL Aggregation

Options:
A) Synchronous (FedAvg): Wait for all nodes before aggregating
B) Asynchronous (FedBuff): Aggregate when K of N gradients received

Analysis:
                    | Sync (A)        | Async (B)       |
--------------------|-----------------|-----------------|
Convergence quality | Higher          | Lower (staleness)|
Latency per round   | Bounded by slowest | Bounded by fastest K |
Complexity          | Lower           | Higher          |
Eclipse tolerance   | Poor            | Good            |

Decision: Option B (Asynchronous)

Rationale: Eclipse periods make synchronous aggregation impractical.
           Staleness can be mitigated with gradient weighting.

Reversibility: Medium - requires protocol change but not architecture change

Approved by: [Technical Lead]
Date: [Date]
```

**Common KEKOA Trade-offs:**

| Trade-off | Tension | KEKOA Default |
|-----------|---------|---------------|
| Latency vs. Accuracy | Faster propagation vs. higher fidelity | Accuracy (SGP4 sufficient) |
| Power vs. Frequency | More compute vs. battery life | Battery life (throttle in eclipse) |
| Bandwidth vs. Model Size | Larger models vs. link capacity | Gradient compression |
| Autonomy vs. Control | Onboard decisions vs. ground oversight | Autonomy with telemetry |

### 4.9 Phase 8: Review and Documentation

**Goal:** Validate design through peer review and create lasting documentation.

**Design Review Checklist:**
- [ ] All requirements traced to design elements
- [ ] All interfaces defined and versioned
- [ ] Trade-offs documented with rationale
- [ ] Formal verification complete for critical paths
- [ ] Failure modes analyzed (FMEA)
- [ ] Security threat model reviewed
- [ ] Scalability analysis performed
- [ ] Operations procedures drafted

**Required Documentation:**
1. **Design Document:** This deliverable
2. **Interface Control Document (ICD):** All external interfaces
3. **Verification Matrix:** Requirements to test mapping
4. **Operations Guide:** Deployment and monitoring procedures

---

## 5. Architectural Patterns

### 5.1 Recommended Patterns

#### Pattern: Deterministic Scheduler

**Problem:** Unpredictable availability in traditional distributed systems.
**Solution:** Use orbital mechanics to pre-compute availability windows.

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   TLE       │────►│   SGP4      │────►│  Contact    │
│   Ingestion │     │ Propagation │     │  Windows    │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                                               ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Task      │◄────│   Schedule  │◄────│  Topology   │
│   Dispatch  │     │   Builder   │     │   Graph     │
└─────────────┘     └─────────────┘     └─────────────┘
```

**When to use:** Any scheduling decision that depends on satellite availability.

---

#### Pattern: Graceful Degradation Chain

**Problem:** Single component failure cascades to system failure.
**Solution:** Define explicit fallback behaviors for each capability.

```
┌───────────────────────────────────────────────────────────────┐
│                    Capability: Model Training                  │
├───────────────────────────────────────────────────────────────┤
│ Level 0 (Nominal):     Full federation, all nodes             │
│ Level 1 (Degraded):    Partial federation (>50% nodes)        │
│ Level 2 (Minimal):     Local training only, no aggregation    │
│ Level 3 (Survival):    Inference only, frozen model           │
│ Level 4 (Safe Mode):   Essential telemetry only               │
└───────────────────────────────────────────────────────────────┘
```

**When to use:** All mission-critical capabilities.

---

#### Pattern: Checkpoint-and-Replay

**Problem:** Long-running operations interrupted by eclipse or contact loss.
**Solution:** Periodic state checkpoints enabling resumption.

```python
class CheckpointableOperation:
    def execute(self):
        checkpoint = self.load_checkpoint()
        if checkpoint:
            self.state = checkpoint.state
            self.progress = checkpoint.progress

        while not self.complete():
            self.step()
            if self.should_checkpoint():
                self.save_checkpoint()

        self.clear_checkpoint()
```

**When to use:** Operations exceeding 1 minute duration.

---

#### Pattern: Ring-All-Reduce Topology

**Problem:** Star topology creates aggregator bottleneck.
**Solution:** Ring topology following orbital planes.

```
Orbital Plane View:

    SAT-1 ──────► SAT-2
      ▲            │
      │            ▼
    SAT-6        SAT-3
      ▲            │
      │            ▼
    SAT-5 ◄────── SAT-4

Each satellite:
1. Receives partial sum from predecessor
2. Adds local gradient
3. Forwards to successor
4. After full ring: all have global sum
```

**When to use:** Gradient aggregation in single-plane constellations.

---

#### Pattern: Topology Abstraction Graph (TAG)

**Problem:** ML code coupled to network topology changes.
**Solution:** Abstract layer between logical task graph and physical network.

```
┌─────────────────────────────────────────────────────────────┐
│                    Logical Task Graph                        │
│     (Trainer) ──► (Aggregator) ──► (Validator)              │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼ TAG Mapping
┌─────────────────────────────────────────────────────────────┐
│                  Physical Network Graph                      │
│    SAT-1 ═══ SAT-2 ═══ SAT-3       SAT-4 ═══ SAT-5         │
│      │                               │                       │
│      └───────────── ISL ─────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

**When to use:** All federated learning deployments.

---

### 5.2 Pattern Selection Matrix

| Scenario | Recommended Pattern(s) |
|----------|----------------------|
| Scheduling FL rounds | Deterministic Scheduler |
| Handling eclipse periods | Graceful Degradation + Checkpoint |
| Multi-plane aggregation | Ring-All-Reduce per plane + Hierarchical |
| Dynamic constellation | TAG + Deterministic Scheduler |
| Long model training | Checkpoint-and-Replay |
| Heterogeneous hardware | TAG + Model Clustering |

---

## 6. Data Modeling for Space Systems

### 6.1 Temporal Data Considerations

Space systems are inherently temporal. All data models must account for:

1. **Epoch Sensitivity:** Position data is meaningless without timestamp
2. **Validity Windows:** Data expires (TLEs degrade, contacts close)
3. **Clock Synchronization:** Nodes may have drifting clocks

**Temporal Data Pattern:**
```protobuf
message TemporalData {
  google.protobuf.Timestamp epoch = 1;
  google.protobuf.Duration validity = 2;
  uint64 sequence_number = 3;  // Monotonic for ordering
  bytes data = 4;
}
```

### 6.2 Coordinate Frame Discipline

**Golden Rule:** Always document the coordinate frame.

| Frame | Abbreviation | Use Case |
|-------|--------------|----------|
| Earth-Centered Inertial | ECI | Orbital propagation |
| Earth-Centered Earth-Fixed | ECEF | Ground station positions |
| Topocentric Horizon | SEZ | Elevation/azimuth calculations |
| Satellite Body Frame | BODY | Attitude-dependent sensors |

**Code Convention:**
```python
# GOOD: Explicit frame annotation
position_eci: Vector3D_ECI = propagate(tle, epoch)
position_ecef: Vector3D_ECEF = eci_to_ecef(position_eci, epoch)

# BAD: Ambiguous frame
position = propagate(tle, epoch)  # Which frame?
```

### 6.3 Units Discipline

**Golden Rule:** Use SI units internally, convert at boundaries.

| Quantity | Internal Unit | Common External Units |
|----------|---------------|----------------------|
| Distance | meters (m) | km, AU, Earth radii |
| Time | seconds (s) | minutes, hours, Julian days |
| Angle | radians (rad) | degrees, arcminutes |
| Mass | kilograms (kg) | grams, pounds |
| Data | bytes (B) | KB, MB, GB |

**Code Convention:**
```python
# GOOD: Internal SI, convert at boundary
altitude_m: float = 550_000.0  # 550 km in meters
display_altitude_km: float = altitude_m / 1000.0

# BAD: Mixed units
altitude = 550  # km? m? Who knows?
```

---

## 7. Communication Design

### 7.1 Protocol Selection

| Protocol | Strengths | Weaknesses | KEKOA Use Case |
|----------|-----------|------------|----------------|
| gRPC | Efficient, typed, streaming | Complex setup | Inter-satellite FL |
| REST | Simple, cacheable | Verbose, no streaming | Ground management API |
| Protobuf | Compact, versioned | Requires compilation | All message encoding |
| MQTT | Lightweight pub/sub | No guaranteed order | Telemetry broadcast |

**KEKOA Standard:** gRPC with Protobuf for all satellite-to-satellite communication.

### 7.2 Message Design Principles

1. **Self-describing:** Include version and type information
2. **Compact:** Minimize bytes over constrained links
3. **Idempotent:** Retry-safe operations where possible
4. **Signed:** Cryptographic integrity for security

**Message Envelope:**
```protobuf
message KEKOAMessage {
  // Header
  uint32 version = 1;
  string message_type = 2;
  string sender_id = 3;
  uint64 sequence = 4;
  google.protobuf.Timestamp timestamp = 5;

  // Payload
  google.protobuf.Any payload = 6;

  // Integrity
  bytes signature = 7;  // Ed25519 signature
}
```

### 7.3 Error Handling

**Error Categories:**

| Category | Retry? | Example |
|----------|--------|---------|
| Transient | Yes, with backoff | Network timeout |
| Permanent | No | Invalid message format |
| Rate-limited | Yes, after delay | Quota exceeded |
| Degraded | Fallback | Peer in eclipse |

**Retry Policy:**
```python
RETRY_CONFIG = {
    "max_attempts": 3,
    "initial_delay_ms": 100,
    "max_delay_ms": 5000,
    "backoff_multiplier": 2.0,
    "retryable_codes": [UNAVAILABLE, DEADLINE_EXCEEDED],
}
```

---

## 8. Formal Methods Integration

### 8.1 The KEKOA Verification Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                    Verification Confidence                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ████████████████████████████████  TLA+ Model Checking          │
│  (Highest Confidence - Proves for all states)                   │
│                                                                  │
│  ██████████████████████████  Property-Based Testing              │
│  (High Confidence - Thousands of generated cases)               │
│                                                                  │
│  ████████████████████  Metamorphic Testing                       │
│  (Medium-High - Oracle-free validation)                         │
│                                                                  │
│  ██████████████  Integration Testing                             │
│  (Medium - Known scenarios)                                      │
│                                                                  │
│  ██████████  Unit Testing                                        │
│  (Lower - Single function, selected inputs)                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 When to Use Each Method

| Method | Cost | When to Use |
|--------|------|-------------|
| TLA+ | High | Distributed protocols, consensus, state machines |
| SPIN | Medium | Concurrency, deadlock detection |
| Hypothesis | Low | All pure functions, data validation |
| Metamorphic | Medium | Physics, ML (no oracle) |
| Traditional | Low | CRUD operations, UI logic |

### 8.3 TLA+ Quick Reference

**Specification Structure:**
```tla
---- MODULE ExampleModule ----
EXTENDS Integers, Sequences, FiniteSets

CONSTANTS Nodes, MaxValue

VARIABLES state, messages

TypeInvariant ==
    /\ state \in [Nodes -> {"idle", "active", "done"}]
    /\ messages \in SUBSET Message

SafetyProperty ==
    \* Property that must always hold
    \A n \in Nodes: state[n] = "done" => ResultValid(n)

LivenessProperty ==
    \* Property that must eventually hold
    <>(\A n \in Nodes: state[n] = "done")

Init ==
    /\ state = [n \in Nodes |-> "idle"]
    /\ messages = {}

Next ==
    \/ \E n \in Nodes: StartAction(n)
    \/ \E n \in Nodes: CompleteAction(n)
    \/ UNCHANGED <<state, messages>>

Spec == Init /\ [][Next]_<<state, messages>> /\ WF_<<state, messages>>(Next)

====
```

### 8.4 Property-Based Testing Patterns

**Pattern 1: Round-trip Property**
```python
@given(st.binary())
def test_serialize_roundtrip(data):
    """Serialize then deserialize equals original"""
    serialized = serialize(data)
    deserialized = deserialize(serialized)
    assert deserialized == data
```

**Pattern 2: Invariant Property**
```python
@given(valid_tle(), st.floats(0, 90))
def test_orbital_energy_conserved(tle, time_minutes):
    """Specific orbital energy is conserved"""
    sv1 = propagate(tle, 0)
    sv2 = propagate(tle, time_minutes)
    assert abs(specific_energy(sv1) - specific_energy(sv2)) < 0.001
```

**Pattern 3: Metamorphic Property**
```python
@given(position_strategy(), rotation_strategy())
def test_visibility_rotation_invariant(position, rotation):
    """Visibility is invariant to satellite body rotation"""
    vis1 = is_visible(position, target)
    rotated_position = rotate(position, rotation)
    vis2 = is_visible(rotated_position, target)
    assert vis1 == vis2  # Body rotation doesn't affect visibility
```

---

## 9. Design Review Process

### 9.1 Review Stages

| Stage | Timing | Focus | Participants |
|-------|--------|-------|--------------|
| **Concept Review** | Phase 1-2 | Mission fit, feasibility | Leads + Stakeholders |
| **Architecture Review** | Phase 3 | Structure, patterns | Engineering team |
| **Interface Review** | Phase 4-5 | APIs, protocols | Dependent teams |
| **Verification Review** | Phase 6 | Proofs, test coverage | QA + Security |
| **Final Review** | Phase 8 | Complete design | All + Management |

### 9.2 Review Checklist

**Architecture Review Checklist:**
- [ ] All quality pillars addressed
- [ ] Patterns justified and appropriate
- [ ] Failure modes identified
- [ ] Scalability path clear
- [ ] Security threat model complete
- [ ] No single points of failure

**Interface Review Checklist:**
- [ ] All interfaces defined in Protobuf
- [ ] Versioning strategy documented
- [ ] Error codes comprehensive
- [ ] Idempotency documented
- [ ] Rate limits specified

**Verification Review Checklist:**
- [ ] TLA+ specs for stateful components
- [ ] Property tests for pure functions
- [ ] Metamorphic tests for physics
- [ ] Coverage meets thresholds
- [ ] Model checking complete

### 9.3 Review Meeting Protocol

1. **Pre-read:** All participants review documents 48 hours before
2. **Presentation:** Designer walks through key decisions (30 min)
3. **Questions:** Open discussion (30 min)
4. **Action Items:** Capture concerns and required changes
5. **Decision:** Approve, Approve with conditions, or Revise and re-review

---

## 10. Reference Architectures

### 10.1 OAK Reference Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     OAK Service Container                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────┐    ┌────────────────┐    ┌───────────────┐ │
│  │   TLE Store    │    │   Ephemeris    │    │    Clock      │ │
│  │   (SQLite)     │    │   Cache        │    │   Service     │ │
│  └───────┬────────┘    └───────┬────────┘    └───────┬───────┘ │
│          │                     │                     │          │
│          └──────────┬──────────┴──────────┬──────────┘          │
│                     │                     │                      │
│               ┌─────▼─────┐         ┌─────▼─────┐               │
│               │   SGP4    │         │  Eclipse  │               │
│               │  Engine   │         │ Predictor │               │
│               └─────┬─────┘         └─────┬─────┘               │
│                     │                     │                      │
│                     └──────────┬──────────┘                      │
│                                │                                 │
│                     ┌──────────▼──────────┐                     │
│                     │   Contact Window    │                     │
│                     │    Calculator       │                     │
│                     └──────────┬──────────┘                     │
│                                │                                 │
│                     ┌──────────▼──────────┐                     │
│                     │   Topology Graph    │                     │
│                     │    Generator        │                     │
│                     └──────────┬──────────┘                     │
│                                │                                 │
├────────────────────────────────┼────────────────────────────────┤
│                     ┌──────────▼──────────┐                     │
│                     │    gRPC Server      │                     │
│                     │  OrbitalAvailability│                     │
│                     └─────────────────────┘                     │
└─────────────────────────────────────────────────────────────────┘
```

### 10.2 Flame Integration Reference Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   Flame + OAK Integration                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Flame Controller                      │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │   │
│  │  │   Round      │  │   Model      │  │   Gradient    │  │   │
│  │  │   Manager    │  │   Registry   │  │   Aggregator  │  │   │
│  │  └──────┬───────┘  └──────────────┘  └───────┬───────┘  │   │
│  │         │                                    │          │   │
│  │         └──────────────┬─────────────────────┘          │   │
│  │                        │                                 │   │
│  │              ┌─────────▼─────────┐                      │   │
│  │              │  Orbital Selector │◄───────────┐         │   │
│  │              │  (OAK Adapter)    │            │         │   │
│  │              └───────────────────┘            │         │   │
│  └───────────────────────────────────────────────┼─────────┘   │
│                                                  │              │
│  ┌───────────────────────────────────────────────┼─────────┐   │
│  │                      OAK                       │         │   │
│  │              ┌────────────────────┐           │         │   │
│  │              │   TopologyGraph    │───────────┘         │   │
│  │              │   (Contact Windows)│                     │   │
│  │              └────────────────────┘                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 10.3 Full Horizon 1 Deployment

```
┌─────────────────────────────────────────────────────────────────┐
│                    Satellite Edge Node                           │
│                    (Nvidia Jetson Orin)                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 Docker Container                         │   │
│  │  ┌─────────────────────────────────────────────────┐    │   │
│  │  │              KEKOA Application                   │    │   │
│  │  │                                                  │    │   │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │    │   │
│  │  │  │   OAK    │  │  Flame   │  │  Inference   │  │    │   │
│  │  │  │          │◄─┤  Engine  │◄─┤  Pipeline    │  │    │   │
│  │  │  └──────────┘  └──────────┘  └──────────────┘  │    │   │
│  │  │       │              │              ▲          │    │   │
│  │  │       │              │              │          │    │   │
│  │  │       ▼              ▼              │          │    │   │
│  │  │  ┌────────────────────────────────────────┐   │    │   │
│  │  │  │           Lattice SDK                   │   │    │   │
│  │  │  └────────────────────────────────────────┘   │    │   │
│  │  └───────────────────────┬──────────────────────┘    │   │
│  └──────────────────────────┼───────────────────────────┘   │
│                             │                                │
├─────────────────────────────┼────────────────────────────────┤
│  ┌──────────────────────────▼───────────────────────────┐   │
│  │                 Lattice Runtime                       │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  │   │
│  │  │ Mesh Network │  │   Entity     │  │  Telemetry │  │   │
│  │  │    Manager   │  │   Manager    │  │   Forwarder│  │   │
│  │  └──────────────┘  └──────────────┘  └────────────┘  │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐ │
│  │  ISL Radio   │  │ Ground Radio │  │  Sensor Interface  │ │
│  └──────────────┘  └──────────────┘  └────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## 11. Anti-Patterns

### 11.1 Patterns to Avoid

#### Anti-Pattern: Probabilistic Availability Assumption

**Symptom:** Using random dropout or timeout-based failure detection.

**Why It's Wrong:** Orbital mechanics are deterministic. Treating connectivity as random wastes predictable contact windows.

**Correct Approach:** Use OAK to pre-compute availability. Schedule around physics.

---

#### Anti-Pattern: Ground-in-the-Loop Dependency

**Symptom:** System halts or degrades significantly when ground contact is lost.

**Why It's Wrong:** LEO satellites have limited ground contact. Autonomy is essential.

**Correct Approach:** Design for full autonomous operation. Ground contact is for monitoring and updates, not control.

---

#### Anti-Pattern: Optimistic Resource Budgeting

**Symptom:** "It should fit in memory" or "We'll have enough power."

**Why It's Wrong:** Space is unforgiving. Resource exhaustion means mission degradation.

**Correct Approach:** Explicit resource budgets with 30% margin. Continuous monitoring.

---

#### Anti-Pattern: Non-deterministic Debugging

**Symptom:** "It works on my machine" or "Can't reproduce the failure."

**Why It's Wrong:** You cannot interactively debug a satellite. Every failure must be reproducible.

**Correct Approach:** Seeded randomness, complete logging, deterministic replay capability.

---

#### Anti-Pattern: Big Bang Deployment

**Symptom:** Deploy entire constellation at once with new software.

**Why It's Wrong:** No rollback in space. Bugs affect entire constellation.

**Correct Approach:** Canary deployments. Test on 1 satellite before rolling to all.

---

#### Anti-Pattern: Implicit State

**Symptom:** System behavior depends on unlogged internal state.

**Why It's Wrong:** State reconstruction impossible after anomaly.

**Correct Approach:** All state explicitly modeled, persisted, and logged.

---

## Appendices

### Appendix A: Glossary

| Term | Definition |
|------|------------|
| **AOS** | Acquisition of Signal - start of contact window |
| **DDIL** | Denied, Disrupted, Intermittent, Limited - communication constraints |
| **ECI** | Earth-Centered Inertial - coordinate frame fixed to stars |
| **ECEF** | Earth-Centered Earth-Fixed - coordinate frame rotating with Earth |
| **FL** | Federated Learning |
| **ISL** | Inter-Satellite Link - direct satellite-to-satellite communication |
| **LEO** | Low Earth Orbit - typically 400-2000 km altitude |
| **LOS** | Loss of Signal - end of contact window |
| **OAK** | Orbital Availability Kernel |
| **OBC** | Onboard Computer |
| **SEU** | Single Event Upset - radiation-induced bit flip |
| **SGP4** | Simplified General Perturbations 4 - orbital propagation model |
| **TAG** | Topology Abstraction Graph |
| **TLE** | Two-Line Element - compact orbital parameter format |

### Appendix B: Reference Documents

1. KEKOA Master PRD (docs/strategy/master_prd.md)
2. Horizon 1 Technical Specification (docs/technical/horizon_1.md)
3. LatticeOS Integration Research (docs/research/LatticeOS_for_Federated_Learning_Satellites.md)
4. Flame Framework Documentation (external)
5. SGP4 Algorithm Specification (AIAA/AAS Astrodynamics Specialist Conference)

### Appendix C: Tool Configuration

**TLA+ Toolbox Configuration:**
```
TLC Model Checker Settings:
- Workers: Auto (use all cores)
- Fingerprint seed: 0 (deterministic)
- Depth: 100 (initial), increase if needed
- Coverage: Enabled
```

**Hypothesis Settings:**
```python
# pytest.ini
[tool:pytest]
hypothesis_profile = ci

# conftest.py
from hypothesis import settings, Phase
settings.register_profile("ci",
    max_examples=1000,
    phases=[Phase.generate, Phase.shrink],
    deadline=None
)
```

### Appendix D: Template Library

Design document templates are available at:
- `docs/templates/design_document_template.md`
- `docs/templates/interface_control_document_template.md`
- `docs/templates/verification_matrix_template.md`

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02 | KEKOA Engineering | Initial release |

---

*"The orbit is not an obstacle—it is the architecture."*

— KEKOA Engineering Principle
