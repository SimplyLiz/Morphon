# MORPHON
## Adaptive Intelligence Engine
### Produktkonzept & Geschäftsplan — TasteHub GmbH, Wien

---

## Executive Summary

**MORPHON** ist eine Software-Plattform, die eine fundamental neue Art von AI ermöglicht: Systeme, die sich zur Laufzeit selbst organisieren, wachsen und anpassen — ohne Retraining, ohne fixe Architektur, ohne Cloud-Abhängigkeit.

Statt bestehende Transformer-Modelle zu verbessern, baut MORPHON auf einem neuen Compute-Paradigma auf: **Morphogenic Intelligence (MI)** — inspiriert von biologischer Neuroplastizität, morphogenetischer Selbstorganisation und Drei-Faktor-Lernregeln.

**Das Produkt in einem Satz:**
> MORPHON ist ein SDK + Runtime, mit dem Entwickler adaptive AI-Systeme bauen, die auf Geräten leben, kontinuierlich lernen und ihre eigene Architektur in Echtzeit optimieren.

---

## 1. Problem

### 1.1 Was heute kaputt ist

Heutige AI-Systeme haben ein fundamentales Design-Problem:

**Statisch nach Training.** Ein GPT-4 oder Llama-Modell lernt nichts Neues nach dem Deployment. Es kann sich nicht an den spezifischen User, die spezifische Umgebung, die spezifischen Daten anpassen. Jede Anpassung erfordert Fine-Tuning — teuer, langsam, zentralisiert.

**Cloud-abhängig.** Die meisten leistungsfähigen Modelle laufen in Rechenzentren. Das bedeutet Latenz, Datenschutzrisiken, laufende Kosten und Abhängigkeit von Internetverbindung. Für Echtzeit-Anwendungen (Robotik, Medizintechnik, industrielle Steuerung) ist das oft inakzeptabel.

**Energiefresser.** Transformer-basierte Modelle verarbeiten *jeden* Input mit dem vollen Netzwerk. Das menschliche Gehirn verbraucht ~20 Watt; das Training eines großen LLMs verbraucht so viel Energie wie hunderte Haushalte über Tage. Für Edge-Deployment ist das unhaltbar.

**Kein echtes kontinuierliches Lernen.** Versuche, Modelle online weiterzutrainieren, führen zu katastrophischem Vergessen. Die Architektur ist nicht dafür gebaut.

### 1.2 Marktvalidierung

Der Neuromorphic Computing Markt wächst mit einer prognostizierten CAGR von ~87% (2024–2030, MarketsAndMarkets) von $28.5M auf $1.33B. Das Software-Segment wächst dabei am schnellsten (CAGR 94%). Europa wird als schnellstwachsende Region prognostiziert.

Aber: Fast alle Player fokussieren auf **Hardware** (Chips, FPGAs). Die Software-Schicht — das, was Entwickler tatsächlich nutzen — ist unterentwickelt. Genau hier positioniert sich MORPHON.

---

## 2. Lösung: Das MORPHON-Produkt

### 2.1 Produktarchitektur

MORPHON besteht aus drei Schichten:

```
┌─────────────────────────────────────────────────┐
│                MORPHON Studio                     │
│  Visual IDE für MI-System-Design & Monitoring     │
│  (Web-App, Electron, oder VS Code Extension)      │
├─────────────────────────────────────────────────┤
│                MORPHON SDK                        │
│  Rust-Core + Bindings (Python, C++, WASM, Dart)   │
│  APIs für Morphon-Definition, Topologie,          │
│  Neuromodulation, Gedächtnissysteme               │
├─────────────────────────────────────────────────┤
│              MORPHON Runtime                      │
│  Plattformübergreifende Execution Engine          │
│  CPU / GPU / FPGA / Neuromorphic HW               │
│  Adaptive Ressourcenverteilung                    │
└─────────────────────────────────────────────────┘
```

### 2.2 MORPHON SDK — Das Kernprodukt

Das SDK ermöglicht Entwicklern, MI-Systeme zu definieren, zu starten und zu nutzen:

```python
import morphon

# Ein MI-System erstellen
system = morphon.System(
    seed_size=100,              # Initiale Morphon-Anzahl
    growth_program="cortical",  # Vordefiniertes Developmental Program
    modulation_channels=[
        morphon.Reward(),       # Dopamin-analog
        morphon.Novelty(),      # Acetylcholin-analog
        morphon.Arousal(),      # Noradrenalin-analog
        morphon.Homeostasis(),  # Serotonin-analog
    ],
    cell_types=[                # Verfügbare Differenzierungsziele
        morphon.Sensory(),      # Input-Verarbeitung
        morphon.Associative(),  # Muster-Erkennung
        morphon.Motor(),        # Output-Generierung
        morphon.Modulatory(),   # Interne Regulation
    ],
    memory=morphon.TripleMemory(
        working=morphon.AttractorMemory(capacity=7),
        episodic=morphon.ConsolidatingMemory(retention="24h"),
        procedural=morphon.TopologicalMemory(),
    ),
    lifecycle=morphon.FullCellCycle(
        division=True,          # Mitose erlaubt
        differentiation=True,   # Funktionswechsel erlaubt
        fusion=True,            # Cluster-Bildung erlaubt
        apoptosis=True,         # Programmierter Zelltod erlaubt
    ),
)

# Das System "wachsen" lassen (nicht "trainieren"!)
system.develop(
    environment=my_data_stream,
    duration=morphon.Hours(2),
    reward_fn=my_reward_function,
)

# Live-Inferenz mit kontinuierlichem Lernen
result = system.process(input_data)
# Das System hat dabei gelernt — keine separate Trainingsphase nötig

# Neuromodulation als API (ein Aufruf, kein Retraining)
system.inject_reward(0.8)      # Verstärkt kürzlich aktive Verbindungen
system.inject_novelty(0.6)     # Erhöht Plastizität systemweit
system.inject_arousal(0.9)     # Alarm: Erhöht Sensitivität

# Beobachten, wie sich das System organisiert
stats = system.inspect()
print(f"Morphons: {stats.total_morphons}")         # Wächst/schrumpft dynamisch
print(f"Clusters: {stats.fused_clusters}")          # Emergente Verbünde
print(f"Cell types: {stats.differentiation_map}")   # Wer ist was geworden?
print(f"Lineage depth: {stats.max_generation}")     # Wie viele Teilungen?
```

**Kern-Features des SDK:**

**Developmental Programs (vordefiniert):**
- `cortical` — für Klassifikation, Pattern Recognition
- `hippocampal` — für sequenzielles Lernen, Zeitreihen
- `cerebellar` — für Motorsteuerung, Robotik
- `custom` — eigene Wachstumsregeln definieren

**Topologie-Management:**
- Automatische Synaptogenese und Pruning
- Zellteilung mit Vererbung (Mitose — Kopie + Mutation)
- Migration im Informationsraum
- Export/Import von Topologien (Checkpoint = gesamte Struktur inkl. Lineage-Tree)

**Cell Lifecycle API (NEU):**
- Differenzierung: Morphons spezialisieren sich (Activation Function, Rezeptoren ändern sich)
- Transdifferenzierung: Funktionswechsel A→B ohne Rückkehr zum Stammzellzustand
- Dedifferenzierung: Unter Stress zurück zur Flexibilität
- Fusion: Morphon-Cluster verschmelzen zu funktionalen Einheiten (Cluster-as-Unit)
- Defusion: Cluster zerfallen unter divergierendem Prediction Error
- Apoptose: Programmierter Zelltod für nutzlose Morphons
- Lineage Tracking: Stammbaum des gesamten Netzwerks (wer stammt von wem ab?)

**Neuromodulation-API:**
- Reward-Signale aus beliebigen Quellen (RL, User-Feedback, Sensoren)
- Novelty-Detection (automatisch oder manuell)
- Arousal-Steuerung für Exploration vs. Exploitation
- Homöostatische Regulation (automatisch)

**Gedächtnis-Systeme:**
- Working Memory: Persistent Activity Patterns
- Episodic Memory: One-Shot-Learning mit Konsolidierung
- Procedural Memory: Topologie-als-Wissen

### 2.3 MORPHON Runtime

Die Runtime führt MI-Systeme effizient aus:

**Multi-Backend:**
- CPU (x86, ARM, RISC-V) via Rust-native Execution
- GPU (CUDA, Metal, Vulkan) für große Systeme
- WASM für Browser-Deployment
- Neuromorphe Hardware (Intel Loihi 2, SpiNNaker 2) als Stretch Goal

**Adaptive Compute:**
- Nur aktive Morphons verbrauchen Ressourcen (Sparse Activation)
- Dynamische Thread-Pool-Skalierung basierend auf Netzwerk-Aktivität
- Memory-Mapped Topologie für große Systeme (>1M Morphons)

**Edge-First:**
- Minimaler Footprint: ~5 MB für Runtime + kleines MI-System
- Keine Cloud-Abhängigkeit
- On-Device-Lernen ohne Datenexport

### 2.4 MORPHON Studio

Visual IDE für das Design, Monitoring und Debugging von MI-Systemen:

**Live-Topologie-Visualisierung:**
- 3D-Graph-Ansicht des Morphon-Netzwerks in Echtzeit
- Farbcodierung nach Aktivität, Alter, Cluster-Zugehörigkeit
- Zoom von Gesamt-Topologie bis zu einzelnen Synapsen

**Development Playground:**
- Drag-and-Drop-Umgebungen (Sensordaten, Spielwelten, Datenströme)
- Echtzeit-Metriken: Netzwerkgröße, Aktivitätsmuster, Modulations-Level
- Time-Lapse der Netzwerkentwicklung (das "Wachsen" beobachten)

**Debugging-Tools:**
- "Warum hat es das gelernt?" — Trace-Back durch Eligibility-Histories
- Topologie-Diff zwischen Zeitpunkten
- Neuromodulations-Replay

---

## 3. Zielgruppen & Use Cases

### 3.1 Primäre Zielgruppe: Edge AI Entwickler

**Wer:** Entwickler und Teams, die AI auf Geräten deployen (IoT, Robotik, Automotive, Medizintechnik, Wearables).

**Pain Point:** Bestehende Edge-AI-Lösungen (TensorFlow Lite, ONNX Runtime, Liquid AI LEAP) deployen statische Modelle. Wenn sich die Umgebung ändert, muss das Modell in der Cloud neu trainiert und re-deployed werden.

**MORPHON-Lösung:** Das MI-System *lebt* auf dem Gerät und passt sich kontinuierlich an — ohne Cloud-Roundtrip, ohne Retraining.

**Beispiel-Use-Cases:**

| Use Case | Beschreibung | Warum MORPHON |
|---|---|---|
| **Industrielle Anomalieerkennung** | Sensor-Monitoring in Fabriken | Maschinen verschleißen; Morphons **transdifferenzieren** ihre Sensitivität auf veränderte Signalmuster, ohne Retraining. Morphon-Cluster **fusionieren** um Multi-Sensor-Korrelationen als Einheit zu erkennen. |
| **Personalisierte Gesundheitsüberwachung** | Wearable-basierte Gesundheits-AI | Jeder Körper ist anders; Morphons **differenzieren sich** auf den individuellen Nutzer. Neue Symptommuster lösen **Zellteilung** aus — das System wächst dort, wo es neue Komplexität erkennt. |
| **Robotik-Steuerung** | Adaptive Motorsteuerung für Roboterarme | Verschleiß ändert die Kinematik; Motor-Morphons **dedifferenzieren** sich kurzfristig (zurück zur Flexibilität), lernen die neue Mechanik, und **redifferenzieren** sich dann auf die neue Realität. |
| **Smart Agriculture** | Sensor-basierte Pflanzenüberwachung | Saisonale Veränderungen; Morphon-Cluster **fusionieren und defusionieren** mit dem Jahreszyklus — im Winter schrumpft das System (Apoptose), im Frühling wächst es (Neurogenese). |
| **Autonome Navigation** | Drohnen/Rover in unbekanntem Terrain | Neues Terrain = hoher Prediction Error; Sensory-Morphons **teilen sich**, um mehr Kapazität für unbekannte Patterns zu schaffen. **Migration** reorganisiert das Netzwerk für den neuen Kontext. |

### 3.2 Sekundäre Zielgruppe: AI-Forscher

**Wer:** Forschungsgruppen an Universitäten und Forschungsinstituten (Computational Neuroscience, Neuromorphic Computing, Artificial Life).

**Pain Point:** Kein einheitliches Framework für Experimente mit struktureller Plastizität, Drei-Faktor-Lernen und morphogenetischer Selbstorganisation. Jedes Lab baut seine eigene Simulation.

**MORPHON-Lösung:** Open-Source-SDK mit standardisierten Primitiven, reproduzierbaren Experimenten und Benchmark-Suites.

### 3.3 Tertiäre Zielgruppe: Neuromorphic Hardware Hersteller

**Wer:** Intel (Loihi), SpiNNcloud, BrainChip, SynSense, Innatera.

**Pain Point:** Diese Firmen haben Hardware, aber unzureichende Software-Stacks. Ihre SDKs sind proprietär und limitiert.

**MORPHON-Lösung:** Hardware-agnostischer Software-Stack, der auf ihrer Hardware besser performt als auf CPUs — was den Absatz ihrer Chips treibt.

---

## 4. Geschäftsmodell

### 4.1 Open Core + Commercial

```
┌────────────────────────────────────────────────┐
│           MORPHON Enterprise                     │
│  Kommerziell: Studio Pro, Support, Consulting     │
│  Preis: Subscription-basiert                      │
├────────────────────────────────────────────────┤
│           MORPHON Professional                    │
│  Fair-Source: SDK + Runtime (volle Features)       │
│  Gratis unter $10M Revenue, danach Lizenz         │
├────────────────────────────────────────────────┤
│           MORPHON Core                            │
│  Open Source (Apache 2.0): Kern-Engine, Basis-SDK  │
│  Community, Forschung, Education                   │
└────────────────────────────────────────────────┘
```

### 4.2 Revenue Streams

| Stream | Beschreibung | Zielpreis |
|---|---|---|
| **MORPHON Professional License** | Voller SDK + Runtime für kommerzielle Nutzung (>$10M Revenue) | €499/Entwickler/Monat |
| **MORPHON Enterprise** | Studio Pro + Priority Support + Custom Development Programs | €2.000–10.000/Monat |
| **MORPHON Hardware Partnerships** | Co-Development mit Chip-Herstellern, Lizenzgebühren pro Chip | Revenue Share |
| **Consulting & Custom Development** | Maßgeschneiderte MI-Systeme für spezifische Industrieanwendungen | €150–250/Stunde |
| **Training & Certification** | "Morphogenic Intelligence Developer" Zertifizierungsprogramm | €1.500/Teilnehmer |

### 4.3 Preisphilosophie (Fair-Source, wie CKB)

- **Gratis** für: Open-Source-Projekte, Forschung, Education, Startups <$10M Revenue
- **Fair-Source** für: Unternehmen >$10M Revenue
- **Enterprise** für: Unternehmen, die Support, SLA und Custom Features brauchen
- Kein Vendor-Lock-In: Daten und Topologien gehören immer dem Kunden

---

## 5. Wettbewerbsanalyse

### 5.1 Direkter Wettbewerb

| Firma | Produkt | Fokus | Wo MORPHON anders ist |
|---|---|---|---|
| **Liquid AI** (MIT Spinoff, $37.5M Seed) | LFM2 / LEAP SDK | Effiziente *statische* Modelle für Edge | MORPHON lernt *nach* Deployment weiter; Liquid deployt fixe Modelle |
| **BrainChip** (ASX: BRN) | Akida Chip + SDK | Neuromorphe Hardware + SNN | Hardware-abhängig; MORPHON ist Software-first und hardware-agnostisch |
| **SpiNNcloud** (DE, TU Dresden Spinoff) | SpiNNaker2 | Neuromorphe Supercomputer | Fokus auf große Simulationen, nicht Edge-Deployment |
| **Intel** | Loihi 2 + Lava Framework | Neuromorphe Forschungs-Hardware | Proprietär, Hardware-gebunden, kein kommerzielles SDK |
| **SynSense** (CH) | Xylo / Speck Chips | Ultra-low-power SNN Chips | Hardware-Fokus; limitiertes Software-Ökosystem |
| **Cortical Labs** (AU, $10M+) | CL1 Biological Computer | Echte Neuronen auf Silizium | Biologische Zellkultur = limitierte Lebensdauer, Reproduzierbarkeit, Skalierung. MORPHON emuliert die Prinzipien in Software — unbegrenzt skalierbar, reproduzierbar, deployable |
| **Unconventional AI** (US, $475M Seed!) | Brain-inspired analog compute | Hardware-Paradigma (analog statt digital) | Noch im Prototyp-Stadium; Hardware-fokussiert; MORPHON ist Software-first und heute lauffähig |

### 5.2 Akademischer Wettbewerb

| Forschungsgruppe | Ansatz | Was fehlt vs. MORPHON |
|---|---|---|
| **NDP** (IT Uni Copenhagen, 2023) | Self-assembling Neural Networks via Neural Developmental Programs | Kein aktivitätsabhängiges Wachstum, kein Drei-Faktor-Lernen, kein Lifecycle (Differenzierung/Fusion/Apoptose) |
| **SMGrNN** (Jia, 2025) | Growing Neural Network mit Structural Plasticity Module | Nur Wachstum + Pruning; keine Differenzierung, keine Fusion, keine Neuromodulation, kein SDK |
| **SAPIN** (Hill, 2025/2026) | Structurally Adaptive Predictive Inference via Active Inference | Nur Migration + lokale Lernregel; keine Zellteilung, keine Differenzierung, nur CartPole-Benchmark |
| **LNDP** (2024) | Lifelong Neural Developmental Program mit Reward-modulierter Plastizität | Nächster Verwandter; aber kein vollständiger Zelllebenszyklus, kein SDK, kein Produkt |

**Kritische Einsicht:** Jedes dieser Projekte löst *ein Stück* des Puzzles. Keines vereint strukturelle Plastizität + Drei-Faktor-Lernen + Differenzierung + Fusion + Migration + Developmental Programs in einem kohärenten System. Das ist MORPHONs Position.

### 5.3 MORPHON's Unique Position

Kein bestehender Player bietet alle diese zusammen:
1. **Software-first** (Hardware-agnostisch, läuft auf CPUs)
2. **Vollständiger Zelllebenszyklus** (Teilung → Differenzierung → Fusion → Migration → Apoptose)
3. **Drei-Faktor-Neuromodulation** als Backprop-Ersatz (vier Kanäle als API-Endpunkte)
4. **Developer-freundlich** (SDK mit Python/Rust/WASM-Bindings, nicht akademischer Prototyp)
5. **Emergente Kritikalität** — System konvergiert spontan zum optimalen Informationsverarbeitungszustand

Liquid AI kommt am nächsten beim Edge-Deployment, aber ihre Modelle sind nach dem Training statisch. Cortical Labs kommt am nächsten bei biologischen Prinzipien, aber sie brauchen echte Neuronen. MORPHON ist die Software-Brücke: biologische Prinzipien, digital implementiert, auf jedem Gerät deploybar.

### 5.4 Marktvalidierung durch jüngste Investments

| Event | Betrag | Signal für MORPHON |
|---|---|---|
| Unconventional AI Seed (März 2026) | $475M | Massives Interesse an Post-Transformer-Compute |
| Liquid AI Seed (2024) | $37.5M | Edge-AI-Markt validiert |
| Cortical Labs CL1 Launch (März 2025) | — | Biologische Compute-Prinzipien kommerziell viable |
| SpiNNcloud Leipzig Deployment (Juli 2025) | — | Neuromorphe Supercomputer in Produktion |
| Neuromorphic Market CAGR | ~87% bis 2030 | Software-Segment wächst am schnellsten (94% CAGR) |

---

## 6. Technologie-Roadmap

### Phase 1: Foundation (Q2–Q4 2026)
**Ziel:** Funktionsfähiger Prototyp, der auf einer Benchmark-Aufgabe zeigt, dass MI funktioniert.

- Morphon-Engine in Rust implementieren
- Lokales Lernen (Drei-Faktor-Regel) + Synaptogenese/Pruning
- Python SDK (Basis-API)
- Benchmark: CartPole, MNIST-Klassifikation, einfache Anomalieerkennung
- Paper auf arXiv publizieren

**Deliverables:** Open-Source Rust-Crate `morphon-core`, Python-Bindings, Benchmark-Ergebnisse

### Phase 2: SDK & Growth (Q1–Q2 2027)
**Ziel:** Developer-taugliches SDK mit vordefinierte Developmental Programs.

- Vollständiges SDK (Rust + Python + C++ Bindings)
- 4 vordefinierte Developmental Programs (cortical, hippocampal, cerebellar, custom)
- Triple-Memory-System
- WASM-Runtime für Browser-Demos
- MORPHON Studio v0.1 (Web-basiert, Live-Topologie-Visualisierung)
- Developer Preview Programm starten

**Deliverables:** SDK v1.0, Studio Beta, Dokumentation, Tutorials

### Phase 3: Edge Deployment (Q3 2027 – Q1 2028)
**Ziel:** Production-ready für Edge-Deployment.

- GPU-beschleunigte Runtime (CUDA + Metal)
- ARM-optimierte Runtime (Raspberry Pi, NVIDIA Jetson, mobile Geräte)
- Dart/Flutter-Bindings (für Mobile-Integration)
- Erste Hardware-Partnership (Intel Loihi oder BrainChip Akida)
- Enterprise-Pilot mit 2–3 Industriekunden
- MORPHON Professional Launch

**Deliverables:** Runtime v1.0, erste Hardware-Integration, Pilotprojekte

### Phase 4: Scale (2028+)
**Ziel:** Ökosystem-Aufbau und Skalierung.

- Marketplace für Developmental Programs (Community-contributed)
- Hardware-Partnerships mit 3+ Chip-Herstellern
- MORPHON Enterprise Launch
- Zertifizierungsprogramm
- Erste "Foundation MI" — ein vortrainiertes MI-System als Basis für Domain-spezifische Anpassung

---

## 7. Go-to-Market Strategie

### 7.1 Community-First (Monate 1–12)

**Ziel:** Mindshare in der AI/Neuromorphic-Community aufbauen.

- Open-Source-Launch auf GitHub (Apache 2.0 Lizenz für Core)
- arXiv Paper: "Morphogenic Intelligence: A Post-Transformer Architecture with Structural Plasticity"
- Blog-Serie auf Substack/LinkedIn: "Building the Next AI Paradigm"
- Demo-Videos: "Watch a MORPHON System Grow in Real-Time"
- Talks auf relevanten Konferenzen:
  - ICONS (International Conference on Neuromorphic Systems) — Call for Papers April 2026
  - NeurIPS Workshop on Neuromorphic Computing
  - Vienna AI Meetups / Austrian AI Association
- Hacker News / Reddit r/MachineLearning Launch-Posts

### 7.2 Research Partnerships (Monate 6–18)

**Ziel:** Akademische Validierung und Forschungskooperationen.

- Kooperation mit TU Wien (Institut für Computertechnik / Neuromorphic Computing)
- Kooperation mit Michael Levins Lab (Tufts University) — MORPHON als Simulationsplattform für morphogenetische Forschung
- SpiNNcloud / TU Dresden — Hardware-Integration-Partnerschaft
- FFG-Forschungsprojekt beantragen (Basisprogramm oder Impact Innovation)

### 7.3 Developer Adoption (Monate 12–24)

**Ziel:** 1.000+ aktive Entwickler, 50+ GitHub-Stars.

- MORPHON Playground: Browser-basierte interaktive Demo (WASM)
- "30 Days of MORPHON" Tutorial-Serie
- Hackathon: "Build Something That Grows"
- Integration mit gängigen ML-Ökosystemen (PyTorch-Interop, HuggingFace Model-Export)

### 7.4 Enterprise Sales (Monate 18–36)

**Ziel:** 3–5 zahlende Enterprise-Kunden.

- Pilotprojekte mit Industriepartnern (Automotive, Medizintechnik, Industrieautomation)
- DACH-fokussierter Vertrieb (kurze Wege, EU-Datenschutz als Vorteil)
- Messen: Embedded World (Nürnberg), Hannover Messe, electronica (München)

---

## 8. Funding-Strategie

### 8.1 Phase 1: Bootstrapping + Grants (2026)

| Quelle | Betrag | Status |
|---|---|---|
| FFG Basisprogramm (Experimentelle Entwicklung) | €200.000–500.000 | Antragsfähig als F&E-Projekt |
| FFG Impact Innovation | €100.000–200.000 | Bereits bekannt durch TasteHub-Anwendung |
| AI Mission Austria (aws/FFG/FWF Joint Initiative) | €50.000–200.000 | Neues AI-Förderprogramm |
| Horizon Europe EIC Pathfinder | bis €4M | Für radikal neue Technologien — passt perfekt |
| GenAI4EU (Digital Europe Programme) | variabel | EU-Förderung für generative AI "made in Europe" |
| FWF Emerging Fields | anteilig (€35M Topf 2026) | Für neue Forschungsgebiete in Österreich |
| Eigenkapital / Revenue aus CKB-Consulting | €30.000–50.000 | Laufend |

**Realistisches Szenario Phase 1:** €200.000–400.000 aus Grants + Eigenkapital

### 8.2 Phase 2: Pre-Seed (2027)

- €500.000–1.500.000 Pre-Seed-Runde
- Target-Investoren:
  - EWOR (Already known)
  - Speedinvest (Wien, Deep-Tech-Fokus)
  - Plug and Play (München Standort, Mobility/IoT-Verticals)
  - Hardware-strategische Investoren (BrainChip Ventures, Intel Capital)
  - FFG Venture Capital Initiative

### 8.3 Phase 3: Seed (2028)

- €2.000.000–5.000.000 Seed-Runde
- Milestone: Nachgewiesene Edge-Deployment-Fähigkeit + erste zahlende Kunden
- Target: European Deep-Tech VCs (Earlybird, Cherry Ventures, Point Nine)

---

## 9. Team-Anforderungen

### 9.1 Kern-Team (Phase 1, 2–3 Personen)

| Rolle | Profil | Wer |
|---|---|---|
| **CEO / Product** | Technische Vision, Business Development, Funding | Lisa (TasteHub GmbH) — Jura + 20y Software + AI-Architektur-Erfahrung |
| **CTO / Engine Lead** | Rust-Experte, Systems Programming, GPU Computing | Zu rekrutieren — idealerweise Erfahrung mit Spiking Neural Networks oder Simulation |
| **Research Lead** | Computational Neuroscience, Plastizitätsregeln | Zu rekrutieren — idealerweise PhD mit Publikationen in SNN/Neuromorphic |

### 9.2 Erweitertes Team (Phase 2, 5–7 Personen)

- +1 Developer Experience (SDK-Design, Dokumentation, Tutorials)
- +1 Visual/Frontend (MORPHON Studio)
- +1 Hardware-Integration (FPGA/Loihi)
- +1 Community/Marketing

### 9.3 Advisory Board

Wunschliste:
- **Michael Levin** (Tufts) — Morphogenetische Intelligenz
- **Wulfram Gerstner** (EPFL) — Drei-Faktor-Lernregeln
- **Ramin Hasani** (Liquid AI) — Liquid Neural Networks, Edge AI
- **Christian Mayr** (TU Dresden / SpiNNcloud) — Neuromorphe Hardware

---

## 10. IP-Strategie

### 10.1 Open Source vs. Proprietär

- **Open Source (Apache 2.0):** Morphon-Core-Engine, Basis-SDK, Forschungs-Benchmarks
- **Fair-Source (BSL oder eigene Lizenz):** Advanced Features, Developmental Programs, Studio
- **Proprietär:** Enterprise-Features, Hardware-Optimierungen, Kundenspezifische Lösungen

### 10.2 Patente

Potentiell patentierbare Innovationen:
- Morphon-Migration-Algorithmus (Compute-Einheiten, die sich im Informationsraum selbst positionieren)
- Vier-Kanal-Neuromodulation als Backprop-Ersatz in Software-Systemen
- Developmental Programs als vordefinierte Wachstumsmuster für adaptive AI

**Strategie:** Zunächst defensiv (arXiv-Publikation als Prior Art), dann gezielt 1–2 Kernpatente in EU + US.

---

## 11. Risiken & Mitigationen

| Risiko | Wahrscheinlichkeit | Impact | Mitigation |
|---|---|---|---|
| **MI performt nicht auf Transformer-Niveau** | Hoch (kurzfristig) | Hoch | Nicht gegen LLMs antreten, sondern für Nischen positionieren wo Transformer versagen (Edge, kontinuierliches Lernen, Adaptation) |
| **Zu früh für den Markt** | Mittel | Hoch | Community-First-Approach; Open-Source-Validierung bevor kommerzialisiert wird |
| **Recruiting in Wien schwierig** | Mittel | Mittel | Remote-First; Anbindung an TU Wien / IST Austria; PhD-Studierende als erste Researcher |
| **Neuromorphe Hardware-Partner nicht interessiert** | Niedrig | Mittel | Software-first-Ansatz; MORPHON muss auch auf CPUs gut funktionieren |
| **Großer Player (Google, Intel) baut dasselbe** | Niedrig | Hoch | Speed-Vorteil durch Fokus; Community-Lock-In; Open-Source-Ökosystem schwer zu kopieren |
| **Fundingkrise Deep-Tech in EU** | Mittel | Hoch | Grant-Heavy starten (FFG, Horizon Europe); Revenue durch CKB-Consulting überbrücken |

---

## 12. Metriken & Meilensteine

### Jahr 1 (2026)
- [ ] Morphon-Core-Engine funktionsfähig (CartPole-Benchmark gelöst)
- [ ] arXiv Paper publiziert
- [ ] GitHub Open-Source-Launch mit ≥100 Stars
- [ ] FFG-Antrag eingereicht und bewilligt
- [ ] 1 Konferenz-Talk gehalten

### Jahr 2 (2027)
- [ ] SDK v1.0 released
- [ ] MORPHON Studio Beta live
- [ ] 500+ GitHub Stars, 100+ aktive Entwickler
- [ ] Pre-Seed abgeschlossen
- [ ] 1 Hardware-Partnership signed
- [ ] 2 Research-Paper publiziert

### Jahr 3 (2028)
- [ ] 3+ zahlende Enterprise-Kunden
- [ ] Runtime auf 3+ Plattformen (CPU, GPU, 1 neuromorphe HW)
- [ ] Seed-Runde abgeschlossen
- [ ] €500K+ ARR

---

## 13. Warum jetzt? Warum Wien? Warum wir?

### Warum jetzt?
- **Markt explodiert**: Der neuromorphe Computing-Markt wächst mit ~87% CAGR, wobei das Software-Segment am schnellsten wächst (94% CAGR). Aber die Software-Schicht fehlt.
- **Forschung konvergiert**: 2025/2026 ist das Jahr, in dem die Einzelkomponenten erstmals demonstriert werden — SMGrNN (Dez 2025), SAPIN (Feb 2026), NDP (2023), LNDP (2024). Niemand hat sie noch zusammengeführt.
- **Massive Investments validieren den Markt**: Unconventional AI ($475M Seed, März 2026!), Liquid AI ($37.5M), Cortical Labs CL1 Launch — der Markt für Post-Transformer-Compute ist real.
- **Biologisches Compute bewiesen**: Cortical Labs' CL1 zeigt, dass biologische Prinzipien kommerziell funktionieren. MORPHON bringt diese Prinzipien in Software — ohne die Limitierungen von Zellkulturen.
- **Organoide zeigen spontane Selbstorganisation**: bioRxiv 2025 beweist, dass echte Neuronen spontan Small-World-Topologien und Module bilden — die Biologie validiert unsere Architektur.
- **EU fördert massiv**: GenAI4EU, Horizon Europe EIC Pathfinder, AI Mission Austria, FWF Emerging Fields (€35M Topf 2026)
- **Continual Learning ist ein ungelöstes Problem**: Der umfassende Survey (arXiv 2025) stellt klar: CL auf Neuromorphic braucht einen "Paradigmenwechsel" — und genau den bietet MORPHON.

### Warum Wien?
- Zentrale EU-Lage mit Zugang zu DACH-Industriekunden
- Starkes Förderlandschaft (FFG, aws, FWF, Horizon Europe)
- Niedrigere Kosten als London/SF/Berlin für Deep-Tech-Startups
- TU Wien und IST Austria als Forschungspartner
- EU-Datenschutz als Wettbewerbsvorteil (Edge-First = Privacy-First)

### Warum dieses Team?
- Lisa: Seltene Kombination aus 20+ Jahren Software-Engineering + juristischem Hintergrund (IP/Lizenzierung) + AI-Architektur-Forschung (ANCS/AXION, Cognitive Vault). Bereits Erfahrung mit Fair-Source-Business-Modell (CKB). Deep-Domain-Knowledge in Bioinformatik (BioLab/CellForge → natürliche Brücke zu biologisch inspirierter AI).
- TasteHub GmbH: Bestehende Firmenstruktur, FFG-Erfahrung, Produktportfolio als Revenue-Bridge.
- Netzwerk: LinkedIn-Präsenz in AI/Dev-Community, Hacker News-Engagement, Open-Source-Track-Record (GitHub: SimplyLiz).

---

## 14. Der nächste Schritt

**Sofort (April 2026):**
1. Morphon-Core-Engine in Rust starten (Proof-of-Concept, 2–4 Wochen für Basis-Implementierung)
2. FFG Impact Innovation Call prüfen (nächster Stichtag recherchieren)
3. arXiv Paper schreiben (basierend auf diesem Konzept + erste Benchmark-Ergebnisse)

**Kurzfristig (Mai–Juni 2026):**
4. GitHub-Repo aufsetzen, README schreiben, erste Demos
5. LinkedIn/HN Thought-Leadership-Posts starten
6. Kontakt zu TU Wien (Prof. Grosu, Institut für Computertechnik) und IST Austria

**Mittelfristig (Q3–Q4 2026):**
7. ICONS 2025 Konferenz besuchen (falls noch möglich) oder für nächsten Call vorbereiten
8. Developer Preview Programm starten
9. Erste Pre-Seed-Gespräche

---

*MORPHON — Intelligence that doesn't just learn, but grows.*

*TasteHub GmbH, Wien, Österreich*
*März 2026*
