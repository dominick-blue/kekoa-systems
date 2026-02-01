This is a smart pivot. You are correct that the **Guardian Agent** (Cyber-Physical Security) is functionally distinct from the **OAK** (Orbital Mechanics). OAK is about *physics and scheduling*; the Agent is about *security and integrity*.

However, splitting them into entirely separate products risks creating "silos" where the scheduler doesn't know the node is compromised.

**Recommendation:** We should adopt a **"Core + Module"** strategy.

1. **Core Platform (KEKOA OS):** OAK + Flame (The "Brain" and "Clock").
2. **Module (Guardian):** The Security Middleware (The "Immune System").

Here is the summary of Use Cases, the Product Roadmap, and the finalized System PRD that unifies them.

---

### 1. Summary of Use Cases (The "Dual-Use" Spectrum)

We are solving three distinct classes of problems. The architecture must support all three.

#### **A. Commercial: "The Economic Efficiency Engine"**

* **Primary User:** Planet, Maxar, BlackSky.
* **Use Case 1: Intelligent Discard (Bandwidth Saver).**
* *Problem:* Paying to downlink clouds and empty ocean.
* *Solution:* Onboard inference filters out 80% of waste data. Downlink only valid scenes.


* **Use Case 2: Maintenance & Longevity (The "Drift" Agent).**
* *Problem:* Radiation degrades sensors over time, requiring expensive manual recalibration.
* *Solution:* The **Guardian Agent** detects "Natural Degradation" (drift) and auto-recalibrates the sensor weights using peer validation from healthy satellites.



#### **B. Defense: "The Sovereign Kill Chain"**

* **Primary User:** Space Force (SDA), NRO.
* **Use Case 3: Hypersonic Tracking (The "Hand-off").**
* *Problem:* Targets move too fast for ground coordination.
* *Solution:* OAK schedules deterministic hand-offs of target data between satellites in real-time.


* **Use Case 4: Adversarial Resilience (The "Poison" Defense).**
* *Problem:* Enemy jams a satellite or injects false data (spoofing) to break the model.
* *Solution:* The **Guardian Agent** distinguishes between "Noise" (drift) and "Attack" (poison). It cryptographically isolates the compromised node from the federation.



#### **C. Financial/Logistics: "Flash Intelligence"**

* **Primary User:** Hedge Funds, Commodity Traders.
* **Use Case 5: The Arbitrage Signal.**
* *Problem:* Fiber optic cables are too slow (glass vs. vacuum).
* *Solution:* Compute specific insights (e.g., "Oil tank volume in Cushing, OK") on-orbit and laser-link the result to a trading desk milliseconds faster than terrestrial fiber.



---

### 2. The Product Roadmap

We will develop this in **Three Horizons**, treating the Agent as a "Premium Module" on top of the Core.

#### **Horizon 1: The Deterministic Core (OAK + Flame)**

* **Focus:** Physics & Basic FL.
* **Deliverable:** A containerized application that uses TLE data to schedule training rounds.
* **Key Feature:** "Intelligent Discard" (Commercial MVP).
* **Goal:** Prove we can save customers 50% on downlink costs.

#### **Horizon 2: The "Self-Healing" Agent (Cyber-Physical V1)**

* **Focus:** Reliability & Natural Degradation.
* **Deliverable:** The **Guardian Agent** module trained to detect *sensor drift*.
* **Key Feature:** Autonomous Recalibration.
* **Goal:** Prove we can extend the useful life of a satellite by 2+ years (Commercial Upsell).

#### **Horizon 3: The "Warfighter" Agent (Cyber-Physical V2)**

* **Focus:** Adversarial Defense (Zero Trust).
* **Deliverable:** Hardened Agent with cryptographic verification and topology severing.
* **Key Feature:** "Fight Through" capability (maintaining model convergence while under attack).
* **Goal:** SDA Tranche 2 Program of Record (Defense Contract).

---

### 3. Master Systems PRD (v3.0)

This PRD defines the **System Architecture** that holds these modules together.

# System Requirements Document (SRD): KEKOA Platform

**Scope:** Core Architecture & Module Interfaces
**Status:** APPROVED FOR DEVELOPMENT

## 1. System Components

The KEKOA platform consists of three tightly coupled components running on the Satellite Edge Compute Module (e.g., Nvidia Jetson).

1. **OAK (Orbital Availability Kernel):** The Scheduler (Physics).
2. **Flame (Federated Learning Engine):** The Learner (ML).
3. **Guardian (Cyber-Physical Agent):** The Protector (Security).

## 2. Core Functional Requirements (The "Must Haves")

### 2.1 Component: OAK (Physics Layer)

* **FR-OAK-01:** System **must** ingest standard TLE (Two-Line Element) data to propagate satellite position with <1km error margin.
* **FR-OAK-02:** System **must** output a deterministic `Contact_Window_Table` predicting ISL (Inter-Satellite Link) availability for the next 90 minutes (1 orbit).
* **FR-OAK-03:** System **must** expose a `Power_State_Flag` based on eclipse prediction (Sun/Shadow) to throttle compute during battery-only operations.

### 2.2 Component: Flame (Application Layer)

* **FR-FL-01:** System **must** utilize a Topology Abstraction Graph (TAG) to separate ML tasks from network routing.
* **FR-FL-02:** System **must** support "Early Exit" inference, discarding data frames with >90% cloud cover before they enter the training pipeline.
* **FR-FL-03:** System **must** accept dynamic topology updates from the **Guardian Agent** (e.g., "Node B is banned") without restarting the training round.

### 2.3 Component: Guardian Agent (Security Layer)

* **FR-SEC-01 (Drift Detection):** Agent **must** analyze local parameter updates against a "Golden Reference" (historical average). If deviation is gradual (<5% per week), classify as **Drift** and re-weight.
* **FR-SEC-02 (Attack Detection):** If parameter update deviation is sudden (>3 sigma) and vectorized (structured noise), classify as **Adversarial Attack**.
* **FR-SEC-03 (Isolation):** Upon classifying an Attack, Agent **must** issue a `TAG_Update_Request` to the Flame Engine to logically sever connections to the malicious peer.

## 3. Data Flow & Interfaces

### 3.1 The "Learning Loop"

1. **Sensor** captures Raw Data (Image).
2. **OAK** checks `Power_State`. If `Safe`, passes data to **Flame**.
3. **Flame** generates `Model_Gradient` (Learning update).
4. **Guardian Agent** intercepts `Model_Gradient`.
* *Check 1:* Is this gradient mathematically sound? (Sanity Check).
* *Check 2:* Is the neighbor requesting this update trusted? (Reputation Check).


5. If **Pass**, data is sent via **Lattice Mesh** to neighbor.

## 4. Key Performance Indicators (KPIs)

| Metric | Threshold | Rationale |
| --- | --- | --- |
| **Scheduler Overhead** | < 5% CPU | OAK physics math cannot starve the AI model. |
| **Attack Detection** | < 100ms | Must block poisoned data before it merges into the global model. |
| **False Positive Rate** | < 0.1% | We cannot accidentally ban a healthy satellite (waste of asset). |
| **Recovery Time** | < 1 Orbit | If a node is attacked, the network must heal within 90 mins. |

## 5. Deployment Strategy (Lattice Integration)

* **Containerization:** The entire stack (OAK, Flame, Guardian) runs as a single Docker container deployed via Anduril Lattice.
* **Transport:** We strictly use the **Lattice Mesh API** for transport. We do *not* build our own radio protocol. We operate at Layer 7 (Application), relying on Lattice for Layer 3 (Network) resilience.

---

### Product Manager Note:

**Decision:** We will execute **Horizon 1 (OAK+Flame)** immediately to secure commercial pilots. The **Guardian Agent (Horizon 2/3)** runs in parallel as a research track, to be merged into the master branch once the "Anomaly Differentiation" algorithms are validated on the ground. This minimizes engineering risk while keeping the sales story (Dual-Use) intact.