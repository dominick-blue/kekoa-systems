# KEKOA

**Knowledge Extraction Kernel for Operational Automation**

A deterministic federated learning platform for Low Earth Orbit (LEO) satellite constellations.

## Overview

KEKOA addresses the fundamental mismatch between terrestrial federated learning frameworks and the space domain. Traditional FL systems assume stochastic client availability (random dropouts). In LEO, satellite availability is deterministic—governed by orbital mechanics, not random events.

KEKOA integrates the **Orbital Availability Kernel (OAK)** with the **Flame** federated learning framework to enable collaborative learning across satellite constellations, with deployment via **Anduril LatticeOS**.

### The Problem

- **Data Gravity:** Modern EO satellites generate 30+ TB/day, but downlink capacity handles only ~20%
- **GITL Latency:** Ground-in-the-loop processing introduces 45-90+ minute delays
- **Straggler Problem:** Standard FL wastes cycles waiting for satellites that are predictably unavailable

### The Solution

KEKOA moves intelligence to the edge. By predicting satellite availability using SGP4 propagation and TLE data, the system:

- Schedules FL rounds only with available nodes
- Enables onboard inference to discard invalid data (clouds, empty ocean)
- Transmits model gradients (KB) instead of raw imagery (GB)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Satellite OBC (Jetson)                   │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────┐    ┌─────────┐    ┌──────────────────────┐    │
│  │   OAK   │───▶│  Flame  │───▶│    Guardian Agent    │    │
│  │ Physics │    │   ML    │    │      Security        │    │
│  └─────────┘    └─────────┘    └──────────────────────┘    │
│       │              │                    │                 │
│       └──────────────┴────────────────────┘                 │
│                          │                                  │
│                   Lattice Mesh API                          │
└─────────────────────────────────────────────────────────────┘
                           │
                    Inter-Satellite Link
```

### Components

| Component | Function |
|-----------|----------|
| **OAK** | Ingests TLE data, predicts contact windows, manages power state awareness |
| **Flame** | Topology-agnostic FL with dynamic TAG updates for changing constellation geometry |
| **Guardian Agent** | Distinguishes sensor drift from adversarial attacks, isolates compromised nodes |

## Roadmap

### Horizon 1: Deterministic Core
- OAK + Flame integration
- TLE ingestion and visibility calculation
- "Intelligent Discard" commercial MVP
- **Goal:** 50% downlink cost savings

### Horizon 2: Self-Healing Agent
- Sensor drift detection
- Autonomous recalibration via peer validation
- **Goal:** Extend satellite operational life by 2+ years

### Horizon 3: Warfighter Agent
- Adversarial defense (zero-trust)
- Cryptographic verification
- Topology severing for compromised nodes
- **Goal:** SDA Tranche 2 Program of Record

## Technology Stack

- **Orbital Propagation:** SGP4, Skyfield (Python)
- **Federated Learning:** Flame framework (Georgia Tech)
- **ML Framework:** PyTorch
- **Deployment:** Docker containers via Anduril LatticeOS
- **Communication:** gRPC/Protobuf (ISL), REST (telemetry)
- **Hardware Target:** NVIDIA Jetson (satellite OBC)

## Documentation

| Document | Description |
|----------|-------------|
| [Horizon 1 Spec](docs/technical/horizon_1.md) | Core platform requirements |
| [Master PRD](docs/strategy/master_prd.md) | Product requirements and use cases |
| [LatticeOS Integration](docs/research/LatticeOS_for_Federated_Learning_Satellites.md) | Technical feasibility and commercialization |
| [Orbital Availability Research](docs/research/deep_deep_kekoa_and_orbital_availability.md) | Deterministic FL foundations |

## Use Cases

### Commercial
- **Intelligent Discard:** Filter 80% waste data onboard, downlink only valid scenes
- **Autonomous Recalibration:** Detect and correct sensor drift without ground intervention

### Defense
- **Hypersonic Tracking:** Real-time target handoff between satellites using FL
- **Adversarial Resilience:** Detect and isolate nodes injecting poisoned data

### Financial
- **Flash Intelligence:** Compute insights on-orbit, deliver via low-latency optical links

## Performance Targets

| Metric | Threshold |
|--------|-----------|
| Scheduler CPU Overhead | < 5% |
| Attack Detection Latency | < 100ms |
| False Positive Rate | < 0.1% |
| Network Recovery Time | < 1 orbit (90 min) |

## Status

**Current Phase:** Research and documentation

Development begins with the Python prototype for TLE parsing and visibility window calculation.

## License

Proprietary - All rights reserved
