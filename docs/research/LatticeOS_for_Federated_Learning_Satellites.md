# **Operationalizing Autonomous Space Architectures: Integrating the KEKOA Knowledge Extraction Kernel with Anduril LatticeOS and the Flame Federated Learning Framework**

The rapid proliferation of Low Earth Orbit (LEO) satellite constellations has fundamentally redefined the requirements for space-based command, control, and computational infrastructure. As constellations transition from static communication relays to dynamic, distributed mesh networks, the limitations of traditional ground-in-the-loop control systems have become increasingly apparent. The transition toward decentralized autonomy is exemplified by the development of the Knowledge Extraction Kernel for Operational Automation (KEKOA), a framework designed to embed intelligence directly into satellite on-board computers (OBCs).1 At the core of this evolution is the "Orbital Availability Kernel," a module integrated into the Flame Federated Learning framework that replaces stochastic terrestrial availability models with deterministic orbital mechanics.1 The convergence of KEKOA’s algorithmic resilience and Anduril’s LatticeOS platform provides a robust pathway for the commercialization and operationalization of autonomous satellite systems, particularly in contested or high-velocity orbital environments.2

## **Deterministic Intermittency and the Mathematical Modeling of Orbital Availability**

The fundamental distinction between terrestrial Federated Learning (FL) and satellite-based FL lies in the nature of client availability. In terrestrial applications, such as mobile device networks, client dropouts are modeled as stochastic processes influenced by battery life, user movement, and signal interference.1 In contrast, LEO satellite constellations operate under the regime of deterministic intermittency, where connectivity is governed by the predictable laws of orbital mechanics.1 The visibility window (![][image1]) between a satellite ![][image2] and a ground station or another satellite ![][image3] is not a random variable but a function of time ![][image4], altitude, inclination, and the minimum elevation angle ![][image5] required for a stable link.1

The "Orbital Availability Kernel" represents a significant departure from existing FL logic, such as that found in frameworks like FedScale or Flower, which rely on static availability traces like the MobiPerf dataset.1 By ingesting Two-Line Element (TLE) sets, the kernel propagates the satellite’s position and velocity vectors to determine real-time and future visibility.1 This capability is critical for "Orbital Selector" logic, which preemptively filters clients based on their predicted trajectory to ensure that a training round can be completed before the satellite drifts out of the visibility cone.1

### **Mathematical Foundations of the Orbital Availability Module**

To calculate the visibility window ![][image6], the kernel utilizes high-precision orbital propagators such as SGP4 or the Skyfield library in a Python environment.1 The process begins with the ingestion of TLE data, which provides the mean motion, eccentricity, inclination, and other Keplerian elements required to define the satellite's state vector in the Earth-Centered Inertial (ECI) frame. The transformation of these coordinates into the Topocentric-Horizon (Azimuth-Elevation-Range) frame relative to a specific ground station ![][image3] at coordinates ![][image7] is essential for determining the elevation angle ![][image8].

The elevation angle is derived from the slant range vector ![][image9], which is the difference between the satellite position vector ![][image10] and the ground station position vector ![][image11] in the Earth-Centered Earth-Fixed (ECEF) frame. The relationship is defined as:

![][image12]  
where ![][image13] is the local zenith unit vector. The visibility window is thus a binary state:

![][image14]  
This deterministic model allows the Federated Learning aggregator to optimize its selection strategy, prioritizing satellites with the longest remaining visibility time. This optimization is vital for convergence speed and model stability, as it mitigates the "straggler" effect often caused by mid-round connectivity failures in LEO environments.1

| Parameter | Description | Technical Implementation |
| :---- | :---- | :---- |
| **Input Data** | Two-Line Elements (TLEs) 1 | Parsed via sgp4 or skyfield Python libraries |
| **Coordinate Frame** | ECI to Topocentric-Horizon | Managed by Skyfield ephemeris files (e.g., DE421) |
| **Threshold** | Minimum Elevation (![][image5]) 1 | Configurable based on antenna gain and signal-to-noise ratio |
| **Output** | Visibility Window (![][image1]) 1 | Binary availability trace for Flame "Orbital Selector" |
| **Constellation Model** | Walker-Delta Configuration 1 | Simulated via shells of TLE-derived state vectors |

## **Architectural Overview of the Flame Framework**

The implementation of the Orbital Availability Kernel occurs within the Flame framework, a topology-agnostic Federated Learning system designed to provide fine-grained control over distributed machine learning workflows.1 Flame distinguishes itself from simplistic client-server FL architectures by utilizing a Topology Abstraction Graph (TAG), which decouples the machine learning logic from the underlying deployment details.1

### **Topology Abstraction and Role-Based Execution**

Within a TAG, every participant in the FL job is assigned a "Role," such as a trainer, an intermediate aggregator, or a global coordinator.1 These roles are interconnected via "Channels," which abstract the communication backend (e.g., gRPC, MQTT, or Kafka).1 For satellite constellations, this abstraction is particularly powerful, as it allows the same machine learning code to be deployed across diverse topologies, including Hierarchical FL (H-FL) for edge-cloud collaboration and Hybrid FL for peer-to-peer weight exchange.1

The Flame SDK decomposes these roles into "Tasklets," which are small, modular units of work such as load\_data, train, evaluate, and aggregate.1 The researcher’s task of swapping out the static trace reader with the Orbital Mechanics Module involves modifying the Selector role to ingest real-time TLE data instead of historical CSV files.1 This transition enables "Orbital Awareness," where the system understands that its clients are moving at 7.5 km/s and selects them accordingly to maximize training efficiency.

### **Comparison of Federated Learning Frameworks for Edge Deployment**

The evaluation of Flame against other frameworks like FedML and Flower reveals significant advantages in terms of extensibility and support for complex topologies required in the space domain.1 While FedML uses a coarse client-server abstraction that struggles with the multi-level hierarchies common in satellite shells, Flame’s TAG allows for the explicit customization of intermediate nodes.1

| Feature | Flame (TAG-based) | FedML (Client-Server) | Flower (Two-tier) |
| :---- | :---- | :---- | :---- |
| **Topology Support** | Hierarchical, Hybrid, P2P, CO-FL 1 | C-FL, 1-level H-FL | Classical C-FL |
| **Extensibility** | High (Role/Tasklet logic) 1 | Low (Requires core code changes) | Moderate (Low-level APIs) |
| **Backend** | Per-channel (gRPC, MQTT, etc.) 1 | Global (Single protocol) | Global (gRPC) |
| **Management** | Compute-agnostic (Deployers) 1 | Compute-centric | Manual / Third-party |
| **Space Suitability** | High (Orbital Awareness ready) | Low (Static assumptions) | Moderate |

## **KEKOA: A Decentralized Autonomy Framework for Contested Orbits**

The Knowledge Extraction Kernel for Operational Automation (KEKOA) is a proposed decentralized autonomy framework designed to operate directly on satellite OBCs.1 It aims to address the unique failure modes of LEO infrastructure, including radiation-induced Single-Event Upsets (SEUs), collision risks, and ground-link cyber interceptions.1 KEKOA’s architecture is bi-layered, combining the collaborative power of Federated Learning with the execution speed of Autonomous AI Agents.1

### **Federated Learning for Anomaly Detection**

The first layer of KEKOA utilizes FL to aggregate telemetry and payload data across the constellation.1 This allows for the training of global anomaly detection models that can identify subtle patterns of system degradation or coordinated cyber-attacks.1 The benefit of FL in this context is twofold: it preserves critical downlink bandwidth by only transmitting model weights rather than raw telemetry, and it keeps sensitive data decentralized, enhancing the security posture of the constellation.4

The accuracy of these models is paramount during "multi-vector cyber-physical attacks," such as those occurring during solar storm events where traditional sensors may provide noisy or misleading data.1 By leveraging the consensus of the constellation, KEKOA can distinguish between environmental noise and actual adversarial activity.1

### **Autonomous Overwatch Protocols**

The second layer of KEKOA involves the deployment of "overwatch" agents capable of modifying satellite attitude, managing power distribution, and executing collision avoidance maneuvers without ground intervention.1 These agents operate at the machine speed of the OBC, enabling response speeds that are up to 400% faster than traditional telemetry-command loops.1 This "overwatch" capability is essential in the current era of "Space as a Contested Warfighting Domain," where adversaries may employ kinetic or non-kinetic means to disrupt orbital assets.8

## **Anduril LatticeOS: A Multi-Domain Battle Management Platform**

Anduril’s LatticeOS is an AI-powered operating system designed to integrate disparate sensors and effectors into a unified command and control (C2) engine.3 While Lattice has achieved significant maturity in land, sea, and air domains—powering autonomous surveillance towers, underwater vehicles, and interceptor drones—its expansion into the space domain represents a strategic pivot toward "All-Domain" mission autonomy.3

### **The Lattice SDK and Open Data Models**

The Lattice SDK provides the technical foundation for third-party developers to integrate their applications and hardware into the Lattice ecosystem.14 The SDK is built upon open data models that define how "Entities" (e.g., satellites, threats) are characterized and how "Tasks" (e.g., move, sense, mitigate) are communicated to autonomous agents.14

1. **Entities API**: This API manages the lifecycle of objects within the Common Operational Picture (COP). For a satellite constellation, each satellite is modeled as an entity with attributes for position, velocity, and health.14  
2. **Tasks API**: This interface enables the routing and execution of sequential commands. KEKOA’s overwatch protocols would be implemented as tasks that Lattice routes to the appropriate satellite OBC.14  
3. **Objects API**: This service acts as a content-delivery network for the tactical edge, providing resilient data storage. It is the ideal mechanism for distributing the Federated Learning model weights and checkpoints across the constellation.14

### **Protocol and Architecture Insights**

Lattice supports both REST and gRPC protocols.18 While REST is suitable for asynchronous data transmission and familiar HTTP toolsets, gRPC is optimized for performance-critical hardware integrations where low bandwidth and high latency are concerns—conditions inherent to satellite communications.18 gRPC’s use of Protobuf (binary encoding) significantly reduces the on-the-wire footprint compared to JSON, making it the preferred choice for cross-link model updates in an FL round.18

| Protocol | Optimization | Primary Use Case in KEKOA |
| :---- | :---- | :---- |
| **gRPC** | Binary Encoding (Protobuf), Streaming | Model weight exchange, Real-time C2 18 |
| **REST** | JSON format, HTTP familiar | Asynchronous telemetry, Management API 18 |
| **Lattice Mesh** | Decentralized Networking | Peer-to-peer data transport in DDIL environments 3 |
| **Object API** | Edge Storage / CDN | Storing ML model checkpoints and data bins 14 |

## **Strategic Alignment: LatticeOS as a Commercial Pathway for KEKOA**

The request to identify a commercial solution for KEKOA using Anduril's LatticeOS is supported by the technical and strategic alignment between the two frameworks. LatticeOS provides the "Management Plane" and "Data Mesh" that the Flame-based KEKOA project currently lacks for physical deployment.1

### **Mapping Flame/KEKOA to the Lattice Ecosystem**

The transition from a research prototype to a commercial solution involves mapping the logical components of the Flame framework to the physical and software interfaces of LatticeOS. The researcher’s "Orbital Availability Kernel" would function as a "Lattice Data Service"—a software module that enriches entity data (satellites) with visibility metadata.18

1. **Deployment**: Flame’s "Controller" and "Deployer" logic, which currently targets Kubernetes or Docker, can be extended to integrate with Lattice’s Mission Autonomy layer.1 This allows the FL job to be orchestrated across a fleet of satellites managed by Lattice.  
2. **Communication**: The per-channel backend flexibility in Flame allows for the integration of "Lattice Mesh" as the primary communication protocol.1 Lattice Mesh is specifically designed for Disrupted, Disconnected, Intermittent, and Low-bandwidth (DDIL) environments, which perfectly matches the "deterministic intermittency" identified in the KEKOA proposal.1  
3. **Autonomous Response**: The "Overwatch" protocols in KEKOA align with Lattice’s "Intent to Task" feature, where high-level operator intent is broken down into discrete tasks distributed across unmanned systems.12

### **Commercialization via the Lattice Partner Program**

The Lattice Partner Program offers a structured pathway for innovative solutions like KEKOA to be integrated and sold alongside Anduril’s native capabilities.21 Partners receive access to developer sandboxes, representative datasets, and dedicated technical support to accelerate their integration.19

Already, industry leaders such as Apex Space and Impulse Space have onboarded their spacecraft into the Lattice ecosystem.21 Apex utilizes Lattice as a C2 option for its rapidly deployable platforms, while Impulse Space uses it to allow a single operator to task and maneuver multiple spacecraft simultaneously.21 For a commercial KEKOA solution, this means immediate interoperability with existing satellite buses and a proven "sensor-to-shooter" infrastructure.11

## **Technical Feasibility of the Orbital Availability Kernel Prototype**

The current task of writing a Python prototype that ingests TLEs to output visibility windows is highly compatible with the Lattice SDK's Python bindings.1 The researcher’s technical stack—including Python, Skyfield, and PyTorch—is standard for AI and orbital development within the Lattice environment.14

### **Implementing the Visibility Module as a Lattice Service**

To implement the "Visibility Window" (![][image1]) as a commercial-grade service, the Python code would be encapsulated as a Lattice Data Service.18 This service would periodically fetch the latest TLEs from a source like Space-Track.org or a private company repository, propagate the orbits, and update the "Available" status of satellite entities in the Lattice COP.14

Python

\# Conceptual Python implementation for Lattice Integration  
from anduril import Lattice  
from skyfield.api import load, wgs84  
from sgp4.api import Satrec

\# Initialize Lattice Client and Ephemeris  
client \= Lattice(token="YOUR\_AUTH\_TOKEN")  
ts \= load.timescale()  
planets \= load('de421.bsp')

def calculate\_visibility(tle\_line1, tle\_line2, gs\_coords):  
    satellite \= Satrec.twoline2rv(tle\_line1, tle\_line2)  
    \# Logic to propagate and calculate elevation angle relative to gs\_coords  
    \# If elevation \> theta\_min, status \= VISIBLE  
    \#...  
    return status

def publish\_to\_lattice(sat\_id, status):  
    \# Mapping KEKOA logic to Lattice Entities API  
    client.entities.update\_entity(  
        entity\_id=sat\_id,  
        attributes={"availability\_status": status}  
    )

This implementation transforms a research module into a live data feed that can trigger autonomous tasks. For instance, if the Orbital Availability Kernel predicts a loss of visibility in 120 seconds, the Lattice Task Manager can proactively migrate a model aggregation task to a different satellite shell, ensuring the FL round completes without data loss.14

## **Edge Computing and Bandwidth Optimization in Space**

One of the most critical challenges in satellite constellations is the "bottleneck" of ground-link bandwidth.4 Advanced sensors, such as hyperspectral imagers or synthetic-aperture radars, generate gigabytes of data per second.4 Transmitting this raw data to the ground is often impossible given that a commercial satellite may only be over a ground station for 5 to 15 minutes per 90-minute orbit.4

### **The Role of Edge AI in KEKOA**

KEKOA addresses this by shifting the "center of gravity" for data processing from the ground to the satellite OBC.1 By performing Federated Learning on the edge, the constellation effectively filters out the noise and only transmits the "Knowledge"—the trained model weights and anomaly alerts—back to Earth.1 This reduces the required bandwidth by orders of magnitude.4

Anduril’s "Menace" family of systems and its focus on edge processing provide the necessary hardware-software synergy to support these high-compute FL tasks.2 Lattice OS is designed to "instrument the tactical edge," backhauling only the most important tactical data for enterprise-level AI training and inferencing.26

### **Bandwidth and Latency Gains via KEKOA and Lattice**

| Mission Component | Traditional Ground-in-the-loop | KEKOA \+ LatticeOS |
| :---- | :---- | :---- |
| **Data Handling** | Raw telemetry downlink 4 | Local processing / Weight aggregation 1 |
| **Response Latency** | 15–90+ minutes (Orbit dependent) 1 | \< 1 second (Local OBC response) 1 |
| **C2 Scalability** | Low (Human-per-satellite) | High (Autonomous fleet management) 3 |
| **Connectivity** | Centralized / Hub-and-Spoke | Decentralized Mesh / P2P 3 |
| **Reliability** | Vulnerable to link jamming 1 | Resilient via autonomous overwatch 1 |

## **Convergence and Stability: Proving the "Orbital Awareness" Hypothesis**

The researcher’s objective is to demonstrate that orbital-aware client selection outperforms standard selection in terms of model convergence.1 In the context of a Walker-Delta constellation, satellites move in predictable relative patterns. A "Classical FL" approach, where all trainers talk to a single ground aggregator, suffers when satellites drift out of range mid-round, leading to incomplete gradients and model divergence.1

### **Experimental Baseline and the "Pivot"**

The "February Milestone" transition from static traces to orbital logic allows for the execution of a "Baseline Experiment" in March.1 By running the same FL job (e.g., training a CIFAR-10 classifier) under two selection strategies, the researcher can quantify the performance gain of orbital awareness.1

1. **Control Group**: Standard FL using random or trace-based selection. Satellites are treated as relatively stable clients.1  
2. **Experimental Group**: "Orbital FL" using the new module to filter clients based on their predicted visibility window.1

Flame’s ability to easily transform topologies—such as moving from Classical to Hierarchical—allows the researcher to test if "Intermediate Aggregators" (lead satellites in a plane) further improve stability by shortening the communication path and reducing the impact of intermittent ground links.1

## **Security and Resilience in Contested Space Environments**

The "Stretch Goal" of the research involves testing if the availability logic makes the system more resilient to "Jamming".1 In a contested environment, an adversary may target specific ground links or satellite cross-links to disrupt the Federated Learning process. Because the Orbital Availability Kernel is deterministic and physics-based, it can distinguish between a natural loss of signal (due to the earth’s horizon) and an artificial disruption (due to jamming).1

### **Cybersecurity and Authentication in SAGINs**

Satellite-Ground Integrated Networks (SAGINs) are increasingly vulnerable to cyber-attacks, including man-in-the-middle interceptions and replay attacks.1 Recent research suggests that lattice-based cryptography and attribute-based encryption (AAQ-PEKS) are essential for securing these high-velocity networks against future quantum threats.27 Anduril’s "First-class Security" approach in the Lattice SDK—utilizing industry-standard authentication and secured data distribution—provides the necessary protection layer for KEKOA’s decentralized data mesh.14

The integration of Palantir’s "Maven" system and Anduril’s "Lattice" further accelerates the ability to deploy new, secure AI applications to the battlefield.26 This partnership ensures that the "Knowledge" extracted by KEKOA is not only processed at machine speed but is also securely backhauled to government enclaves for further strategic analysis.26

## **Operationalization of the February Milestone**

As the researcher enters the "Build Phase," the focus is on the implementation of Python logic to parse TLEs and calculate ![][image1].1 This technical milestone is the prerequisite for all subsequent commercial and research outcomes.

### **Technical Implementation Steps for the Researcher**

The researcher should focus on the following core tasks to ensure the Python prototype meets the requirements of both the academic project and a potential LatticeOS integration:

1. **TLE Ingestion**: Develop a robust parser for the 69-character TLE format using the sgp4 library. Ensure the code can handle "Shell" updates for a large Walker-Delta constellation (e.g., 500+ satellites).1  
2. **Ground Station Modeling**: Implement the Topocentric transformation to account for the rotation of the Earth, which is critical for calculating ground-to-satellite visibility.1  
3. **Kernel Integration**: Follow the Flame SDK’s "Developer Programming Model" to create a new OrbitalSelector class.1 This class should override the default select\_clients function to use the output of the orbital propagation module.1  
4. **Lattice Preparation**: Structure the code modularly so that the visibility calculations are decoupled from the FL training loops. This allows the same logic to be exposed as a Lattice Data Service later in the project lifecycle.18

## **Analysis of Commercial Solution Potential**

The request asks whether there is an application for Anduril’s LatticeOS to support a commercial solution for KEKOA. The evidence strongly indicates a positive answer, predicated on the following commercial factors:

1. **Product-Market Fit**: There is a clear military and commercial demand for rapidly deployable, autonomous satellite platforms.8 KEKOA’s ability to provide 400% faster response times through on-board anomaly detection is a unique value proposition.1  
2. **Platform Synergy**: LatticeOS already manages the "Common Operational Picture" (Understand) and the "Task Manager" (Decide).3 KEKOA provides the "Domain Intelligence" (Learn) that enriches this picture.1  
3. **Revenue Model**: Anduril operates as a "tech-first" startup, favoring firm-fixed-price commercial models and rapid iteration.20 This is highly conducive to onboarding academic research (like the Orbital Availability Kernel) into a productized ecosystem via the Lattice Partner Program.20  
4. **Operational Heritage**: With Lattice already integrated into U.S. Space Force projects and central to multinational exercises like "Valiant Shield," a KEKOA-enabled Lattice instance would enter an ecosystem with established authority and user trust.2

| Commercial Aspect | LatticeOS Capability | KEKOA Contribution |
| :---- | :---- | :---- |
| **C2 Platform** | Understand and Decide (All-Domain) 3 | Decentralized anomaly detection 1 |
| **Autonomy** | Mission Autonomy / Tasking 12 | Overwatch agents / Local OBC response 1 |
| **Data Transport** | Lattice Mesh (Resilient networking) 3 | Deterministic intermittency modeling 1 |
| **Commercialization** | Lattice Partner Program / SDK 19 | Proprietary FL and Anomaly Kernels 1 |
| **Deployment** | Tactical Edge Compute (Menace) 2 | Embedded logic for satellite OBCs 1 |

## **Conclusion and Future Outlook**

The development of the "Orbital Availability Kernel" within the Flame framework represents a critical technical advancement for the next generation of LEO satellite constellations. By transitioning from terrestrial-based stochastic models to deterministic orbital mechanics, the researcher is solving the fundamental problem of client selection in high-velocity space environments.1 The Knowledge Extraction Kernel for Operational Automation (KEKOA) leverages this stability to provide real-time, decentralized anomaly detection and autonomous overwatch, significantly outperforming traditional ground-centric control systems.1

Anduril’s LatticeOS platform provides the ideal commercial and operational framework for this research. Through the Lattice SDK, open data models, and the resilient Lattice Mesh, the KEKOA framework can be productized and deployed across diverse satellite shells and ground station networks.3 The "February Milestone"—the creation of the Python-based visibility prototype—is the foundational step toward this integrated future. As constellations move toward proliferated LEO architectures, the marriage of orbital awareness, federated learning, and mission autonomy will be the "unfair advantage" that ensures resilience and superiority in the contested domain of space.8

#### **Works cited**

1. kekoa\_research\_question\_dominick\_blue (1).pdf  
2. Anduril Expands Capabilities into the Space Domain, accessed February 1, 2026, [https://www.anduril.com/news/anduril-expands-space](https://www.anduril.com/news/anduril-expands-space)  
3. Command & Control \- Anduril, accessed February 1, 2026, [https://www.anduril.com/lattice/command-and-control](https://www.anduril.com/lattice/command-and-control)  
4. That Computes\! Edge Computing Reshapes Space Possibilities \- Kratos Space, accessed February 1, 2026, [https://www.kratosspace.com/constellations/articles/edge-computing-reshapes-space-possibilities](https://www.kratosspace.com/constellations/articles/edge-computing-reshapes-space-possibilities)  
5. Federated Learning Models in Decentralized Critical Infrastructure \- ResearchGate, accessed February 1, 2026, [https://www.researchgate.net/publication/375616429\_Federated\_Learning\_Models\_in\_Decentralized\_Critical\_Infrastructure](https://www.researchgate.net/publication/375616429_Federated_Learning_Models_in_Decentralized_Critical_Infrastructure)  
6. AI SDK Core: Telemetry, accessed February 1, 2026, [https://ai-sdk.dev/docs/ai-sdk-core/telemetry](https://ai-sdk.dev/docs/ai-sdk-core/telemetry)  
7. Compute-Update Federated Learning: A Lattice Coding Approach \- IEEE Xplore, accessed February 1, 2026, [https://ieeexplore.ieee.org/iel8/78/10347386/10742892.pdf](https://ieeexplore.ieee.org/iel8/78/10347386/10742892.pdf)  
8. MSM\_Sep2024.pdf \- MilSat Magazine, accessed February 1, 2026, [http://milsatmagazine.com/2024/MSM\_Sep2024.pdf](http://milsatmagazine.com/2024/MSM_Sep2024.pdf)  
9. Advanced Techniques for Steerable Antennas \- IRIS, accessed February 1, 2026, [https://iris.unipv.it/retrieve/e1f104fc-28b7-8c6e-e053-1005fe0aa0dd/Thesis.pdf](https://iris.unipv.it/retrieve/e1f104fc-28b7-8c6e-e053-1005fe0aa0dd/Thesis.pdf)  
10. Space | Anduril, accessed February 1, 2026, [https://www.anduril.com/space/space](https://www.anduril.com/space/space)  
11. Anduril Industries Product Cheatsheet: Autonomous Defense Systems & AI, accessed February 1, 2026, [https://cheatsheets.davidveksler.com/anduril-products.html](https://cheatsheets.davidveksler.com/anduril-products.html)  
12. Mission Autonomy \- Anduril, accessed February 1, 2026, [https://www.anduril.com/lattice/mission-autonomy](https://www.anduril.com/lattice/mission-autonomy)  
13. Anduril's Lattice Showcased in U.S. Central Command's Desert Guardian 1.0, accessed February 1, 2026, [https://www.anduril.com/news/anduril-s-lattice-showcased-in-u-s-central-command-s-desert-guardian-1-0](https://www.anduril.com/news/anduril-s-lattice-showcased-in-u-s-central-command-s-desert-guardian-1-0)  
14. Lattice SDK | Anduril, accessed February 1, 2026, [https://www.anduril.com/lattice/lattice-sdk](https://www.anduril.com/lattice/lattice-sdk)  
15. Docs · anduril/lattice-sdk \- Buf, accessed February 1, 2026, [https://buf.build/anduril/lattice-sdk](https://buf.build/anduril/lattice-sdk)  
16. Anduril | Documentation: Build with Lattice, accessed February 1, 2026, [https://developer.anduril.com/](https://developer.anduril.com/)  
17. Sample apps \- Build with Lattice | Anduril | Documentation, accessed February 1, 2026, [https://developer.anduril.com/samples/overview](https://developer.anduril.com/samples/overview)  
18. Building with Lattice | Anduril | Documentation, accessed February 1, 2026, [https://developer.anduril.com/guides/concepts/overview](https://developer.anduril.com/guides/concepts/overview)  
19. The Contours of War are Changing \- We Must Prepare for Interoperability at the Edge, accessed February 1, 2026, [https://www.anduril.com/news/the-contours-of-war-are-changing-we-must-prepare-for-interoperability-at-the-edge](https://www.anduril.com/news/the-contours-of-war-are-changing-we-must-prepare-for-interoperability-at-the-edge)  
20. How Anduril Is Driving National Security Innovation \- GovCon Wire, accessed February 1, 2026, [https://www.govconwire.com/articles/anduril-uav-uuv-lattice-homeland-security](https://www.govconwire.com/articles/anduril-uav-uuv-lattice-homeland-security)  
21. Lattice Partner Program \- Anduril, accessed February 1, 2026, [https://www.anduril.com/lattice/lattice-partner-program](https://www.anduril.com/lattice/lattice-partner-program)  
22. Tactical Data Mesh and Anduril's Lattice System \- A New Era in Modern Warfare \- Scribd, accessed February 1, 2026, [https://www.scribd.com/document/814657804/Tactical-Data-Mesh-and-Anduril-s-Lattice-System-A-New-Era-in-Modern-Warfare](https://www.scribd.com/document/814657804/Tactical-Data-Mesh-and-Anduril-s-Lattice-System-A-New-Era-in-Modern-Warfare)  
23. anduril-lattice-sdk \- PyPI, accessed February 1, 2026, [https://pypi.org/project/anduril-lattice-sdk/1.6.0/](https://pypi.org/project/anduril-lattice-sdk/1.6.0/)  
24. Aalyria Spacetime 101 rev. 11/14/2024, accessed February 1, 2026, [https://cdn.prod.website-files.com/63e03a88bb9f184ceb4ab190/677eee4d88c38b62f82c91e2\_Aalyria%20Spacetime%20101%20Jan%202025.pdf](https://cdn.prod.website-files.com/63e03a88bb9f184ceb4ab190/677eee4d88c38b62f82c91e2_Aalyria%20Spacetime%20101%20Jan%202025.pdf)  
25. Anduril Showcases Advanced Edge Computing and Communications Capabilities at USMC Steel Knight Exercise, accessed February 1, 2026, [https://www.anduril.com/news/anduril-showcases-advanced-edge-computing-and-communications-capabilities-at-usmc-steel-knight](https://www.anduril.com/news/anduril-showcases-advanced-edge-computing-and-communications-capabilities-at-usmc-steel-knight)  
26. Anduril and Palantir to Accelerate AI Capabilities for National Security, accessed February 1, 2026, [https://investors.palantir.com/news-details/2024/Anduril-and-Palantir-to-Accelerate-AI-Capabilities-for-National-Security/](https://investors.palantir.com/news-details/2024/Anduril-and-Palantir-to-Accelerate-AI-Capabilities-for-National-Security/)  
27. Privacy-Preserving in Cloud Networks: An Efficient, Revocable and Authenticated Encrypted Search Scheme \- ResearchGate, accessed February 1, 2026, [https://www.researchgate.net/publication/397811425\_Privacy-Preserving\_in\_Cloud\_Networks\_An\_Efficient\_Revocable\_and\_Authenticated\_Encrypted\_Search\_Scheme](https://www.researchgate.net/publication/397811425_Privacy-Preserving_in_Cloud_Networks_An_Efficient_Revocable_and_Authenticated_Encrypted_Search_Scheme)  
28. Toward Secure and Lightweight Access Authentication in SAGINs \- ResearchGate, accessed February 1, 2026, [https://www.researchgate.net/publication/348323951\_Toward\_Secure\_and\_Lightweight\_Access\_Authentication\_in\_SAGINs](https://www.researchgate.net/publication/348323951_Toward_Secure_and_Lightweight_Access_Authentication_in_SAGINs)  
29. Shui Yu 0001 \- DBLP, accessed February 1, 2026, [https://dblp.org/pid/90/3575-1](https://dblp.org/pid/90/3575-1)  
30. Dual-use tech: the Anduril example \- Privacy International, accessed February 1, 2026, [https://privacyinternational.org/report/5704/dual-use-tech-anduril-example](https://privacyinternational.org/report/5704/dual-use-tech-anduril-example)  
31. anduril.pdf \- AWS, accessed February 1, 2026, [https://sacra-pdfs.s3.us-east-2.amazonaws.com/anduril.pdf](https://sacra-pdfs.s3.us-east-2.amazonaws.com/anduril.pdf)  
32. Examples of Abuse Timeline \- Privacy International, accessed February 1, 2026, [https://privacyinternational.org/abusetimeline](https://privacyinternational.org/abusetimeline)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACsAAAAXCAYAAACS5bYWAAACeUlEQVR4Xu2WSchOURjH/+Z5KJHMYzYkkbBRKEQRYWFhKjYyZEhICSXCwsawkcROdjJlipRpIRnTJ0qxMIUkw///Pc95771Pr4W895P6fvWrc57n3vc995xzn3uARhr5/xhKr9PTdErI5ZlO79AOMUHW0XO0U0zUmr50F/3kDiimK1ygP+ncmCC3YLmlMVEWK2B/eDgmyFj6GpY/GHJiCH1Ct8dEWbSgn+mVEG9Kb9Dx9B19UExXOESnxmCZXKIvQ2w1PertM7DZ7Zql62lGb9LmIV4qx+kP2tr7g+gr2t37O2CDneX9xBy6P8RK5wBsMG1oE3qZrsrlZ3p+Wy7Wit6nw3OxBkEzq4ogltN7KC5tL9hg07YQGvi+XH8BfQi7v1TO0jraj76lY/JJ5yNsxsUw+pi2y9L1PKLjQqzmqOjLi3RNyCWewh5IW0UzP7mQBXrAVqdliNecF/Q7PRUTOa7Rb/QYitshMZ+e9/ZsuhX2YJGJdBPtBvuY7IWVvj50M91Nu1SuDqgCaKDPaOeQy3MStm+1/B1DTqjeboENQFtBtXlw4Qob/EZYBbkLWw1tqQ+wT7dqu35nfbohMpJ+paNiIqAvmJZZZ4pq6CFUc/XnqijVrtOM9adX6QyPjaDPYfeIE7CvalVUlhbFYBW0RNNi0En7VXntZ03A79DsfkG2iiqRR7yt/f6eDvR+KcyDnb6E9vMyWAnTaUz1eCWyU9sEetvbQu/JQm9Pgr3oOlSlma85e+haby+B1e3F3u9N38DqsNgAe/kSdbSnt1Uy9ZKWWqvbw16ORDz7ahD6NIu2yPanSJ/4hH5LZ45/xk408GHnbxgdA3/CLxtJcjvenqETAAAAAElFTkSuQmCC>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAXCAYAAADduLXGAAAA1klEQVR4XmNgGLSAF4i50QXRQTsQfwfi/0BciiaHFaQwQBRboktgA7OB+CsQs6JLYAN3gHg3uiA2IMsAcUINkpg4EFsh8eEglgGi2AaImYC4E4iXAPFFIA5FUgcGc4D4GwMk2KYAsRkQZzFADIhDUgcGIPdeZYBo0oKKqQNxNhBzwBSBQDQDxARrBogb3wDxXGQFyGAmA2qQbQDiW1B2JBC7Q9lgcA2IdyHxVwHxASh7CwNS9LMD8V8gzoEJAIELEL9mgGjC8JwcEDOiiYE8JYomNgrgAACXXCOZ5tyyogAAAABJRU5ErkJggg==>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABEAAAAYCAYAAAAcYhYyAAABEElEQVR4Xu3SvyuFcRTH8ePHJFeRieiWQVJks7Ew3PIH3MFAVgvDrWtjUCaDgbJISRYGufJPWCXFaCElk4H3cc73cfpORsPzqVc9z+f7fE/PL5EyZf6eTsxh0M87MIV5dKeLyDhq6AldkVMc4E1s2AWa2McrhrGDXTTwgVndmDKDTYziC4/o9zW9C+1eMO2d5hYn4VxWMIZFsQ06NGXEu43QaZ5xnHU/ORR7nPbQLYkNmQzdhHfLoSvygMusOxJ7J3HwFj7l95GL6FfR6etZ/4TzrLvHtR/voS8t1MWG6GdNqXq3GjrdoN2a2Ps6C2uyjTu0hW4B72IXx1yhhRsMxIUKumLh6c0Lz5DYD1rm3+cbWwYurM1cW5kAAAAASUVORK5CYII=>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAcAAAAYCAYAAAA20uedAAAAkElEQVR4XmNgGOTABYgXoQvCwFEgPo4uCAI8QPwbiDuQBfmAWAWIY4D4PxCnA7EqELOCJLOAeCcQP2OA6ASxQVgaJAkDIPuOIQvAADcQ/wLidnQJEHBjgNjnji4BAm0MEPtALsYAILtOIvHnATELjPMCiOdA2WlAXAKTAIEKIH4PxDOBuBVZAgZEGXDYOTQAAF7KGBZyukz/AAAAAElFTkSuQmCC>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAB4AAAAVCAYAAABR915hAAAB9UlEQVR4Xu2Vy0uVQRjGX2+YROpKkwqEAjPRIry0kDYFqaQIUksX5m0TLVsUKKkoikZghWCbFARFRPIPyAsuRBD/gNCFGoiIIOpCTJ+ndz7P6yy8wHdw4w9+nJln5sycb76ZOSJXXDJ3YJEfRpNY2AV/wS9wDF470SNKdMAp0R9A1mB1pDk65MN9mGuyedhv6qETDxfhby//Cye8LFRewENYY7Ik+A+OmCx0volOnG6yYpc1myxUYuAK3IPLcMl9botOXBl0DJv7ohP88PJpuAOTvfy8pMBaP7Q8F5243mRp8AAOmOyiFMLvfmh5KTox32lAo+jRemiy0OG55cR5rs6j9Uf0Mgng7dUES0T7f4Q98LroBcPbzu6FUtgJb7r6XdgNH8By2MYwEa7CN67TBzgnepwC3sJs0VVgmYzCWdFBb8MtmABviR7LPtGVI83wNVyHT8WMXSW6mQbhkG1wFMBncMFk4xI591lwQ/SE5MAbopfPPdf+GH6Cn12dD3sMlzPDBh4tostLeJdvwkxXfw9/ujLhvpkxdcL/gApXLrMNZ8Ev8v2QR6JnPYCTcNB3ok89DOtcnavHh9qFqa7/qbvdwsH4foIz3QB7I83/l50bhpuHfIXt8JWrP4GTrkxaTflMuIMDuIniTJ3wwrDY/ux74n/9CCgKUv0nfRPWAAAAAElFTkSuQmCC>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEEAAAAYCAYAAACldpB6AAAD9UlEQVR4Xu2YechOeRTHj30fsm9j37IkkZDImDAhW5ZQthgN2beQJWtjSyhLYiYhf1hKzdjCTJayJWQb5hUllDX7+v06v9/z3Oe89z7eXu/zJr2f+tb9nXPvfe793fM75/wekRxy+F6oaA0RVLKG74X20C5rjOAPqJ81ehpCx6G/oE7GF6QLdBYqZh1gMnQAKm4dKaQGdBeqYx0RVIAeQK2tg1SFfoeeO/HmYRyCPkJ9rAOcFvWNsI4Uwt9cZo2OatAeqLSxT4X+h/IZe4wxoi+ywTpAS+i+qH+d8ZG60HVovnWkiHbQPYmOvAXQW6iIsecXfc6Rxh6Ds/MC+sfYc0MnobbQY+hyojvGeugXa0wRzAOrrTHACdFnDmMWdN4agxyB7hjbeOhPd7xPNBrKxN2fyQOdgvIaeyrgR3kNjTN2fsRaUAPoDbQRqi3po6U/9EHSv0OMraInFHRj3pTJp7wbM8w4CT3d2NMbWmVsqaKy6DMwUQdhwuNHuijqZ7Ln2EZnc1F/D2OPsVb0hEJQLuioJM54d+efF7AVgC5BjQO2VNJG9BnqWYcjKh94SopeP8o6PIwEVggyGrogiSHuv4JfHoQTsiIwHgRdEb0+FfD+70WTXBiMAOaEZDyEZlujZz+UJlpiHkEtgk7HM9EIIY2ga5J+1q9CrYwtq5gAPbVGB5+D+WCRdRhuSJJKxmaIOgxNND7Pf6ITxSXDSOmY4NU2ltEU9aW+liGi0Wh7ANJB1Jes6WMS50QNtw7PbdFQ220dAY6JrrktkrgsPAOgg+64FzRXdMIsbHlnQGVFm6zlokmsCjQTWgKVip0dp6voi4ZF6ULoncS72mmi1SIIm0Fe/7Oxf4YVgRNwEyphfEF2it6Ey+AH4yPsF1iL+WJcEuwtWKqCcFKmi1aUc6LRw6XFMGcLzjLI+0zxFwRgxeLvD7QOsF00UgkT9Y6Az+OjJbQzbipaf5tZh4EdI8Ode44wODnsGfhSrDBh5/ELV4f+hbo5WxPolug1ZJtoFxtGGjTH2Ehb0YZvk2ikhn2k36AnEtHTsPxxvX0Jhmpna3T4fEA/8wUnNgpGw0uJRx1L8WZ3zHzCB63pxhZ2i3ut0VEYKmeNAdhErbTGrKSv6G6SMF/8Kloq2bWxnxgr8fX6E3TGHRPmocHumOuVCZoh6yMlSFXolUTsCJNQXzTaM7rzzBTc1U1yx8NE+46hbvyj6FaWdZ4waTFpetIk/scHkx6Ta7JeY6noJDJ/ZBS2AGusMaspKokPZf974MuxxSYMW7/+iW/VPbwXy1kUXNN/i052RuBy5+YwchudXSyWiISUSTjJX2qMPCyfYb1FtsPNSw7fIp8AOuy7ZSHvuwQAAAAASUVORK5CYII=>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEIAAAAYCAYAAABOQSt5AAADp0lEQVR4Xu2YR4gUURCGy5wwB8xiPCgI5nRZFMGIAVHMARVFVAQxYGA96EkvJhQMizmAWTHiYkIRBEXFDIoeTCgmVEz/vzXNvK15PdM962VhP/jZ6b+6e+e9rlf1ekRKKCEKNaFK1izmcEyVrZmOhtANqLoNeKhmjRjUgMpaswiUhmpJ+D3bQlegKjbgowJ0HZpsAx5aQ1usGYNc6Au0zvjZcA36Df2FOpuYyxromOikpWU1dBMqZQMepkMTrBmTDaJfvpsNZMEe6LOEZwRhBr+GptmAC0/6BPW0gRD2Qk2tGZMOohPxP7LiIXTamh6mQC+g8jYQMBe6a800XLVGltyD3kr6J5mJBqITusgGPJSBfkLDbCDgHHTEmh7qQeOgS1AjE8uGxaKDGGQDMRgtySXG2jVAtBiH8Qhab82AJ6I1IoyW0D7RCTsJbRatJyeg2s55ceHy+iN672zZBP2CtokOcL7oMu/vnuRwCrptTRKkywwbSDAQ+gj1SRzzSzdJfD4DHU98zpbL0Deoqg1E5IHo989xPC7dsAxfK7ocU2gmmlrBQF3aic7ueMdjPw5g8eG1jR0vDr2hd6L3mFQ4FIn6oteuNP4z6JDxAuaIttuUNtpR9Gb8a9kqWmWZNaQVtD0ZLrgpr+VGLC79oOdQF+gHdL5wOBIjRf9/L8drnvBmO54LHyqXY0rnCC4cYQPgMXTQOZ4KTXSOWSeeOsdRGQJ9kOQegk+PTyluAd4IfYXKOd5C0YGGZely6JU1CddmWPu5BR12jndKsj70EL2OVTqAs8zKnY5Rohkw3PGGit6Lhc6lBVTReC5s+WeNx0LIukO4m2R7ddkhhZd3IV6Kf8vcF/ouySeVn/jLdseCMytxHJAn+jTGGj9gjGiFn2d8PlHWijuOx0zlvbhZ8sFuxThbcABfrOgtE/3OR51YALfknAwveZIcpIWD4hI4INpmuYNj5+DatnBdvhfNHAuzhd2HVdvHKtGsyEkcc7d7X/SdhO3bwuLObtPe+PtFs+SC+He/b6AF1gxgfeATTpeGK0RTN9MbHDdd3Pv7YCzsXYZ+J6iu8VkH2hiPsIDbcwNYH1KKoeiy5rJkt/HCVsLZX2IDDrskWR/SwfSP8gYbBU5OlHeIqOwWHUdaBou+wYVV7ovW8MDfMfIlc9ZEhbVkpjWzhC2WnSpTMS9gqfjXNwsT13Am+HtAV2sWAU5C2FKKAzOedS5s2+0lF6pjzWJOd9FdcAklROAf5+OsjJLvUZQAAAAASUVORK5CYII=>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAB4AAAAYCAYAAADtaU2/AAACEklEQVR4Xu2VP0iVYRTGj2lBZklSgiUIlSJtLRISQhi5JOQmIUhEFraICDWZg5qDDoENgQrW5iJogzooiJWCDkJbhU6mBEYOokV/ntP5Pjv3ufe7fdfG+sEP7vuc7736nffPFflPPE5xEMFpDv6GajjKYQTDsJ7D/XAGfoBlXIigCH6El7iQKYuwj8OAK/AZh+A+XIEHuRCXy3AD5nMh4CV8zSE4BN/Cu1yIi65rP4cBefAr7OFCQDtc5jAVumvPufEBuAtbXKYcE3uuAf6Ad2CpJLf1BvwOT1L+i8PwEXwDZ+G4/G5rsdgXXwvGIc1wEq6JvbF+VvkYVYjNr6NcCuE8nBHbiUyV2MRyLgTo+r7i0FEgNv8eF0bgNjwr1lbV0wi/iW0U5gj8ItatdGzChz44Ltb/z/Cds9I90wq33NhzVextarhAvIedPrggNrHbh8RNsWdOcEFsnq6v7uwossW6ctuH2qod+MCHRK3YH77IBbG1XXDjIZjjxoreeDpfL5kExuC0JK6t7uQQPTI6UY8Nsw4Hgs9NsM3VQsLl0H8ggVz4HM6JXQK6u/WoeFZhB2WKduoTfAq7qBai36V7iDuxh55bfbssLojdWi84DNCLId0aD8LHHMalRGwvZPpLc17s1ov7i5aSXrgkyec8HVPwCYeZoms0AW9xIYLrYtcv39374qj8+ZYK0TOe6uz/g/wEA/NcscdK8lQAAAAASUVORK5CYII=>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAYCAYAAADDLGwtAAAAv0lEQVR4Xu3QMQtBYRTG8VOUlMHCQFa+gOxsBptB2ZltssmgGOzMfAGDfABlM8ssVsXM/3buvR1vdzCweepXPOcZ3q7IP79MEg2U3YNNFwcMsMUc8bcF6eGMjP8/jydq4YJkcUfTdGnR4dR0MsEDCdNVRYcj08lR9E02Q9FhPSiCt/SDgqRwxd500hYdzkw3xgVF08kCNyyxwg5rlOzIywkb/3cBOXML45Xu+yLTEh1W3IObjuinibmHqHw0+m5eF+ghYle6+xAAAAAASUVORK5CYII=>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAB0AAAAYCAYAAAAGXva8AAABcklEQVR4Xu3USygFcRTH8eNRnimxsEB5v8LCggUbG0UpZSWllAXyKBZKLCxkx1pJ1rJg61WytyULd2FJCVko8T3+Z+7MnVLudXfmV59m5sy5nWb+/zsiUaJEifJDmjCBeSwhD8OYRHWgL60ZwyM+8Y4jXNj1Cyr91vSmXdwQdYgKfNh1V6AvrWkQf+i41XrQH+9IPZnhgpd68Yd2hO79JdmIoSBU/06d+ENbQ/f+kk5ch4tefjtU++awJm6Xe+nDCrrRa7V1HOMWG1aLpwQD4g8dQm1Ch0shTpGBTXF/K82quAFa38O+1XNxglHkWC2eBbzh2bziKqHDpVnc/S1xT6MbpFHcb72n1qEzdq7rqf1ldp1SdMgULnGPfCzjLNATQ5ud63re+LeSTwsekGXu7Kjrq69ao8OeUIURLGJb3KvVJUg65TjAtLg1043j1XcxK+4Tql8yff26noM4xw5qrD+llIYLlmI76mYKbpoiq/3TfAGahERiJUI0MwAAAABJRU5ErkJggg==>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABcAAAAYCAYAAAARfGZ1AAABQklEQVR4Xu3UvyuFURzH8S+JRAaKUgzyI4MMFondZVKyKkWZ/BooPwcWf4BFWUxGPwYrWZQyyJ/BgAxKvL/3+3XPcaf7PM/6fOpV5znf0/e5neecK5InT57M6cc8lrGBekxjEV3RulSZxQt+8IUL3PrzOzrD0nQZFGumztGBb38ejtalSp+E5nM+N4aJ0ooM6ZXQfKisljk9EpoPlNUyp9Lm+i22xLZuHI0+Xyt2wnZ8vpQWTEpoPoXueIGnDVdolXAAmr22K9ajHfc+V8waPvHmPvAYL/DcYMXHI3iOame4E7sb+oJEaRI7mn9nfhNHoVw8EMd4xVI0X1F0T/Wi1aAOD5jx2inWfay7oBJnAYc4ENtv3XvNNvawjxOxH5Io1ajy8SieopqmQew/KVWuxY6fbsklCv/L2aLN9COuSoIb/At3dDx+6wiSjgAAAABJRU5ErkJggg==>

[image12]: <data:image/png;base64,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>

[image13]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAXCAYAAAA7kX6CAAAA4UlEQVR4Xu3SPwtBYRQG8COZ/RsMinwIk01hkCIZmRT5DgaT72A2yGqw+AIGq5mUZBH5swjPe53Xe+7tymS7T/2W83huchF5+Zkc3Fxc2QXOsIcAb6xU4cnmUIMMpGEhupEe6DS46INP3Ot8V3YQFZ2VDtwhJm5xOJAZlkX3SRvGjtuUzGjo6L6mSWa0hbC9dk8CjmSGRXvtnhaZwQaCosvyPSVuVpJwIjMs2Gvq8T0ij+oVzLhQBrJEQrCEB/hloX5VPVL/khLkoQJdWHOnvpEtE3q/R/VE/QA3K/68l7/kBTqXR0xAkYnKAAAAAElFTkSuQmCC>

[image14]: <data:image/png;base64,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>