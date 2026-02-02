# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KEKOA (Knowledge Extraction Kernel for Operational Automation) is a deterministic federated learning platform for LEO satellite constellations. The system integrates orbital mechanics with the Flame federated learning framework to enable collaborative learning on satellite edge compute modules.

**Current Status:** Research and documentation phase - no source code deployed yet.

## Architecture

### Core Components (Three-Horizon Development)

**Horizon 1 - Core Platform (OAK + Flame):**
- **OAK (Orbital Availability Kernel):** Physics-based scheduler using SGP4 propagation and TLE data
- **Flame:** Federated learning engine with Topology Abstraction Graph (TAG) for dynamic topologies
- Focus: Deterministic orbital availability + basic FL
- Commercial MVP: "Intelligent Discard" for bandwidth savings

**Horizon 2 - Guardian Agent V1:**
- Security middleware for sensor drift detection
- Autonomous recalibration using peer validation
- Goal: Extend satellite operational life by 2+ years

**Horizon 3 - Guardian Agent V2:**
- Adversarial defense with zero-trust security
- Cryptographic verification and topology severing
- Target: SDA Tranche 2 Program of Record

### Data Flow
1. Sensor captures raw data
2. OAK checks Power_State (eclipse prediction)
3. Flame generates Model_Gradient
4. Guardian Agent validates gradient (sanity + reputation checks)
5. Data transmitted via Lattice Mesh to neighbor satellites

### Deployment
- Containerized (Docker) on satellite OBC (Nvidia Jetson)
- Managed through Anduril LatticeOS
- Transport via Lattice Mesh API (gRPC/Protobuf preferred over REST)

## Key Technologies

| Technology | Purpose |
|-----------|---------|
| Flame (Georgia Tech) | FL engine with TAG topology abstraction |
| SGP4 + Skyfield | Orbital propagation from TLE data |
| Anduril LatticeOS | C2 platform and deployment |
| PyTorch | ML training framework |
| gRPC/Protobuf | Inter-satellite communication |

## Key Technical Concepts

### Deterministic vs Stochastic Availability
Traditional FL (FedAvg) models random client dropout. KEKOA exploits predictable orbital mechanics - satellites don't randomly disconnect, they pass over the horizon. OAK transforms availability from a probability distribution to a deterministic boolean function.

### The Straggler Problem Solution
OAK uses TLE sets and SGP4 to predict when clients will be available, enabling preemptive exclusion of unavailable nodes from training rounds rather than waiting for timeouts.

### Bandwidth Bottleneck
Modern EO satellites generate 30+ TB/day but >80% is waste (clouds, empty ocean). KEKOA enables onboard inference to discard invalid frames, transmitting only model gradients (KB) instead of raw data (GB).

## System Requirements (from PRD)

### OAK (Physics Layer)
- FR-OAK-01: Ingest TLE data, propagate position with <1km error
- FR-OAK-02: Output deterministic Contact_Window_Table for 90-minute horizon
- FR-OAK-03: Expose Power_State_Flag based on eclipse prediction

### Flame (Application Layer)
- FR-FL-01: Utilize Topology Abstraction Graph (TAG)
- FR-FL-02: Support "Early Exit" inference (discard >90% cloud cover)
- FR-FL-03: Accept dynamic topology updates without restart

### Guardian Agent (Security Layer)
- FR-SEC-01: Detect drift (<5% deviation/week = natural degradation)
- FR-SEC-02: Detect attacks (>3 sigma sudden deviation)
- FR-SEC-03: Issue TAG_Update_Request to isolate malicious peers

## KPIs
| Metric | Threshold |
|--------|-----------|
| Scheduler Overhead | < 5% CPU |
| Attack Detection | < 100ms |
| False Positive Rate | < 0.1% |
| Recovery Time | < 1 Orbit (90 min) |

## Documentation Structure

- `docs/technical/` - Horizon phase specifications (horizon_1.md, horizon_2.md, horizon_3.md)
- `docs/technical/guides/` - Engineering design guides (see below)
- `docs/strategy/` - PRD, GTM strategy, business model
- `docs/research/` - Technical deep-dives on orbital mechanics, LatticeOS integration

## Engineering Design Guides

The following guides define KEKOA engineering standards. **Reference these guides when reviewing or writing code.**

| Guide | Path | Use When |
|-------|------|----------|
| **Systems Design Guide** | `docs/technical/guides/systems_design_guide.md` | Reviewing architecture, interfaces, formal verification, trade-offs |
| **AI Systems Design Guide** | `docs/technical/guides/ai_systems_design_guide.md` | Reviewing ML pipelines, federated learning, edge inference, model serving |
| **ML Systems Design Guide** | `docs/technical/guides/ml_systems_design_guide.md` | Reviewing training systems, evaluation, feature engineering, MLOps |

### Code Review Standards

When reviewing code, validate against these guide principles:

**From Systems Design Guide:**
- [ ] Physics First: Does the design respect orbital mechanics constraints?
- [ ] Determinism: Is the code reproducible with seeded randomness?
- [ ] Graceful Degradation: Does the system handle partial failures?
- [ ] Formal Verification: Are critical components specified in TLA+ or property-tested?

**From AI Systems Design Guide:**
- [ ] Edge-First: Does inference happen at the source?
- [ ] Federated-First: Do gradients move instead of data?
- [ ] Power-Aware: Does the system respect eclipse/power constraints?
- [ ] Non-IID Handling: Is data heterogeneity addressed?

**From ML Systems Design Guide:**
- [ ] Training-Serving Consistency: Are features computed identically?
- [ ] Model Validation: Is there offline evaluation before deployment?
- [ ] Drift Detection: Are monitoring and alerts in place?
- [ ] Fault Tolerance: Are fallback models available?

### Anti-Patterns to Flag

Reference the anti-patterns sections in each guide. Key violations to catch:

- **Probabilistic Availability Assumption**: Using random dropout instead of OAK scheduling
- **Ground-in-the-Loop Dependency**: Requiring ground contact for normal operation
- **Non-deterministic Debugging**: Code that can't be reproduced via telemetry
- **Centralized Training Assumption**: Designing for data centralization
- **Unbounded Resource Consumption**: Missing memory/power limits

## Target Markets

**Primary:** Space Development Agency (SDA) Tranche 2
**Commercial:** Earth Observation operators (Planet, Maxar, BlackSky)
**Emerging:** High-frequency trading via LEO optical links
