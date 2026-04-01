# FFG Förderantrag — MORPHON
## Impact Innovation / Basisprogramm (Experimentelle Entwicklung)
### Entwurf — Stand: März 2026

---

## 1. Projektübersicht

**Projekttitel:** MORPHON — Adaptive Intelligence Engine auf Basis Morphogener Intelligenz

**Antragstellerin:** TasteHub GmbH, Wien

**Geschäftsführerin:** Lisa — Software-Architektin, Juristin (IP/Lizenzrecht), AI-Forscherin

**Programmlinie:** FFG Basisprogramm — Experimentelle Entwicklung

**Geplante Laufzeit:** 18 Monate (Q3 2026 – Q4 2027)

**Gesamtkosten:** ca. €300.000

**Beantragte Förderung:** ca. €150.000 (50% Förderquote, Experimentelle Entwicklung, KMU-Zuschlag)

### Kurzfassung

Das Projekt MORPHON entwickelt eine neuartige Software-Plattform für Morphogene Intelligenz (MI) — eine radikal neue AI-Architektur, die nicht neuronale Netze simuliert, sondern die Organisationsprinzipien lebender Systeme als Compute-Primitive nutzt. Im Gegensatz zu heutigen Transformer-basierten Systemen, die nach dem Training statisch bleiben und auf Cloud-Infrastruktur angewiesen sind, wachsen MORPHON-Systeme zur Laufzeit, organisieren sich selbst und lernen kontinuierlich — ohne Retraining, ohne fixe Architektur, ohne Cloud-Abhängigkeit.

MORPHON vereint erstmals sechs biologische Prinzipien in einer kohärenten Software-Architektur: strukturelle Plastizität, Drei-Faktor-Lernen mit neuromodulatorischer Steuerung, morphogenetische Selbstorganisation, Zellteilung mit Vererbung, Differenzierung und Transdifferenzierung sowie Fusion und Apoptose. Die Implementierung erfolgt als hochperformante Rust-Engine mit Bindings für Python, C++ und WebAssembly.

Das Projektergebnis ist ein Open-Core-SDK, das Entwicklern ermöglicht, adaptive AI-Systeme zu bauen, die auf Edge-Geräten leben, kontinuierlich lernen und ihre eigene Architektur in Echtzeit optimieren. Zieldomänen sind Edge AI, IoT, Robotik und Medizintechnik.

---

## 2. Stand der Technik und Problemstellung

### 2.1 Fundamentale Limitierungen heutiger AI-Systeme

Die gegenwärtige Generation von AI-Systemen — dominiert von Transformer-Architekturen — leidet unter vier strukturellen Defiziten, die nicht durch inkrementelle Verbesserung, sondern nur durch einen Paradigmenwechsel lösbar sind:

**Statische Architektur nach Training.** Heutige Modelle wie GPT-4, Llama oder Gemini lernen ausschließlich während einer separaten Trainingsphase. Nach dem Deployment sind sie eingefroren. Jede Anpassung an neue Datenverteilungen, veränderte Umgebungsbedingungen oder spezifische Nutzerkontexte erfordert aufwändiges Fine-Tuning — ein zentralisierter, kostspieliger und langsamer Prozess. Versuche, Modelle online weiterzutrainieren, führen regelmäßig zu katastrophischem Vergessen (McCloskey & Cohen, 1989; Kirkpatrick et al., 2017), da die Architektur nicht für kontinuierliches Lernen konzipiert ist.

**Cloud-Abhängigkeit.** Leistungsfähige Modelle erfordern Rechenzentren mit spezialisierter Hardware (GPU/TPU-Cluster). Für Anwendungen mit strikten Latenz-, Datenschutz- oder Verfügbarkeitsanforderungen — Robotik, Medizintechnik, industrielle Steuerung, autonome Navigation — stellt diese Abhängigkeit ein fundamentales Hindernis dar. Edge-Deployment beschränkt sich derzeit auf das Ausführen vortrainierter, komprimierter Modelle ohne Lernfähigkeit.

**Energieineffizienz.** Transformer-Architekturen verarbeiten jeden Input mit dem vollen Netzwerk (dense computation). Das menschliche Gehirn verbraucht ca. 20 Watt; das Training eines großen Sprachmodells konsumiert Energie im Umfang von hunderten Haushalten über Tage (Patterson et al., 2021). Diese Energieintensität ist für Edge-Deployment und ressourcenbeschränkte Umgebungen nicht tragbar.

**Fehlende Adaptivität auf Strukturebene.** Bestehende Ansätze für neuronale Architektursuche (NAS) und Pruning operieren ausschließlich offline. Es existiert kein produktionsreifes System, das seine eigene Topologie — Anzahl, Typ und Vernetzung seiner Verarbeitungseinheiten — zur Laufzeit als Reaktion auf Erfahrungen anpasst.

### 2.2 Der Neuromorphic-Computing-Markt: Wachsende Hardware, fehlende Software

Der Markt für neuromorphes Computing wächst laut MarketsAndMarkets (2024) mit einer prognostizierten CAGR von ca. 87% (2024–2030) von $28,5 Mio. auf $1,33 Mrd. Das Software-Segment zeigt dabei das stärkste Wachstum (CAGR 94%). Europa wird als schnellstwachsende Region prognostiziert.

Der Großteil der Investitionen fließt jedoch in Hardware: Intel entwickelt den Loihi-2-Chip, BrainChip (ASX: BRN) produziert den Akida-Prozessor, SpiNNcloud (TU Dresden) baut den SpiNNaker-2-Supercomputer, und Unconventional AI hat kürzlich $475 Mio. Seed-Funding für hirninspierten analogen Compute eingeworben. Liquid AI (MIT Spinoff, $37,5 Mio. Seed) entwickelt effiziente Modelle für Edge, die jedoch nach Deployment statisch bleiben. Cortical Labs (Australien) experimentiert mit biologischen Neuronen auf Silizium — ein faszinierender, aber hinsichtlich Lebensdauer, Reproduzierbarkeit und Skalierung limitierter Ansatz.

Die kritische Lücke ist die **Software-Schicht**: Es fehlt ein hardware-agnostisches, produktionstaugliches Framework, das die Prinzipien neuromorpher und bioinspirierter Informationsverarbeitung als Software-Abstraktionen bereitstellt. Genau diese Lücke adressiert MORPHON.

### 2.3 Akademischer Forschungsstand

Die wissenschaftliche Grundlage für MORPHON liegt in der Konvergenz mehrerer aktueller Forschungslinien:

- **Strukturelle Plastizität:** Jia (2025) demonstriert mit SMGrNN ein Netzwerk, dessen Topologie sich zur Laufzeit durch ein Structural Plasticity Module verändert. Hill (2025/2026) zeigt mit SAPIN, dass Zellen lernen können, sich im Informationsraum zu positionieren — inspiriert von Michael Levins Arbeiten zur morphogenetischen Kollektivintelligenz.
- **Drei-Faktor-Lernregeln:** Frémaux & Gerstner (2016/2018) formalisieren das mathematische Fundament: Eligibility Traces markieren Hebb'sche Koinzidenzen, die erst durch ein neuromodulatorisches Signal (Dopamin-analog) in effektive Gewichtsänderungen konvertiert werden. Der umfassende Review in Patterns/Cell Press (2025) bestätigt die Überlegenheit gegenüber klassischem Hebb'schem Lernen.
- **Morphogenetische Kollektivintelligenz:** Levins Konzept der „Multiscale Competency Architecture" (2023) zeigt, dass biologische Systeme als geschachtelte Problemlösungsebenen organisiert sind, wobei jede Ebene die Energielandschaft der anderen deformiert.
- **Neural Developmental Programs:** Die IT University Copenhagen (2023) hat mit NDP neuronale Netze gebaut, die durch einen embryonalen Entwicklungsprozess wachsen. Die kritische Limitierung — kein aktivitätsabhängiges Wachstum — ist genau der Ansatzpunkt für MORPHON.

**Entscheidende Erkenntnis:** Jedes dieser Forschungsprojekte löst ein Teilproblem. Keines vereint strukturelle Plastizität, Drei-Faktor-Lernen, Differenzierung, Fusion, Migration und Developmental Programs in einem kohärenten, produktionstauglichen System. Diese Synthese ist der Kern des Innovationsgehalts von MORPHON.

---

## 3. Innovationsgehalt

### 3.1 Sechs biologische Prinzipien — erstmals als kohärente Software-Architektur

MORPHON ist das erste System, das die folgenden sechs Prinzipien gemeinsam implementiert und als programmierbare Software-Abstraktionen bereitstellt:

1. **Strukturelle Plastizität:** Das Netzwerk wächst, pruned und reorganisiert sich zur Laufzeit. Morphons (die grundlegenden Verarbeitungseinheiten) werden dynamisch erzeugt und entfernt, basierend auf Aktivitätsmustern und Prediction Error.

2. **Drei-Faktor-Lernen mit rezeptorgesteuerter Modulation:** Synaptische Plastizität ergibt sich aus drei Faktoren: präsynaptische Aktivität, postsynaptische Aktivität und ein neuromodulatorisches Signal. Die Innovation liegt in der rezeptorgesteuerten Modulation — jedes Morphon besitzt individuelle Rezeptorprofile für vier neuromodulatorische Kanäle (Reward/Dopamin-analog, Novelty/Acetylcholin-analog, Arousal/Noradrenalin-analog, Homeostasis/Serotonin-analog). Die Wirkung eines globalen Modulationssignals hängt damit vom lokalen Rezeptorzustand ab — ein Mechanismus, der biologisch validiert, aber in AI-Systemen noch nie implementiert wurde.

3. **Morphogenetische Selbstorganisation im hyperbolischen Informationsraum:** Morphons positionieren sich selbst in einem kontinuierlichen Informationsraum, gesteuert durch „Desire-Gradienten" — eine Analogie zu chemotaktischen Gradienten in der biologischen Morphogenese. Die Wahl eines hyperbolischen Raums (im Gegensatz zu euklidischen Embeddings) ist eine spezifische Innovation: Hyperbolische Geometrie bildet hierarchische Strukturen mit exponentiell wachsender Kapazität natürlich ab, was der emergenten Modularität biologischer Netzwerke entspricht.

4. **Zellteilung mit Vererbung (Mitose-Prinzip):** Morphons können sich replizieren, wobei die Tochterzelle die Gewichte, Rezeptorprofile und den internen Zustand der Elternzelle erbt — mit stochastischer Mutation. Dieser Mechanismus ermöglicht gerichtetes Wachstum: Das System expandiert dort, wo Kapazität benötigt wird.

5. **Differenzierung und Transdifferenzierung:** Morphons beginnen als undifferenzierte Stammzellen und spezialisieren sich basierend auf ihrer Aktivitätsgeschichte und ihrem lokalen Kontext. Bereits differenzierte Morphons können unter spezifischen Bedingungen (hoher Prediction Error, neuromodulatorischer Reset) direkt in einen anderen Funktionstyp übergehen (Transdifferenzierung), ohne den Umweg über einen Stammzellzustand.

6. **Fusion, Defusion und Apoptose — vollständiger Zelllebenszyklus:** Morphons können zu funktionalen Clustern verschmelzen (Synzytium-Prinzip), die als einzelne Verarbeitungseinheit operieren. Cluster zerfallen bei divergierendem Prediction Error (Defusion). Dauerhaft inaktive Morphons werden durch programmierten Zelltod (Apoptose) entfernt. Dieser vollständige Lebenszyklus ist in keinem bestehenden AI-System implementiert.

### 3.2 Tag-and-Capture-Mechanismus für verzögerte Belohnung

Ein spezifisches algorithmisches Novum ist der Tag-and-Capture-Mechanismus: Synapsen setzen bei Hebb'scher Koinzidenz einen Eligibility Trace (Tag), der exponentiell zerfällt. Wenn innerhalb des Zeitfensters ein neuromodulatorisches Signal eintrifft, wird die markierte Synapse „captured" — die Gewichtsänderung wird effektiv. Dieses Prinzip, formalisiert durch Frémaux & Gerstner, wird in MORPHON erstmals als performante Systemkomponente in einer Laufzeitumgebung implementiert, die Tausende bis Millionen solcher Traces parallel verwaltet.

### 3.3 Software-first, hardware-agnostisch

Im Gegensatz zu nahezu allen Wettbewerbern verfolgt MORPHON einen konsequent software-basierten Ansatz. Die Rust-Engine abstrahiert die biologisch inspirierten Prinzipien als programmierbare Primitiven und läuft auf Standard-Hardware (x86, ARM, RISC-V), GPUs (CUDA, Metal, Vulkan) und im Browser (via WebAssembly). Neuromorphe Hardware (Intel Loihi 2, SpiNNaker 2) wird als optionales Backend unterstützt, nicht vorausgesetzt.

### 3.4 Open-Core-Geschäftsmodell als Innovationstreiber

Das Lizenzmodell — Apache 2.0 für den Kern, Fair-Source für professionelle Features — ist selbst eine Innovation im europäischen Deep-Tech-Kontext. Es ermöglicht akademische Reproduzierbarkeit, Community-Building und gleichzeitig nachhaltige kommerzielle Verwertung. Die Antragstellerin verfügt über direkte Erfahrung mit diesem Modell durch ihr CKB-Produkt.

---

## 4. Projektziele und erwartete Ergebnisse

### 4.1 Gesamtziel

Entwicklung eines funktionsfähigen Prototypen der MORPHON-Plattform, der die Machbarkeit von Morphogener Intelligenz als Software-Paradigma demonstriert, wissenschaftlich validiert und als Open-Source-SDK für die Entwickler-Community bereitstellt.

### 4.2 Phase 1: Proof of Concept (Monate 1–10)

**Ziel:** Funktionsfähiger Prototyp der Rust-Engine mit Nachweis der Kernmechanismen.

**Erwartete Ergebnisse:**
- Lauffähige MORPHON-Engine in Rust, die alle sechs biologischen Prinzipien implementiert
- Morphon-Lebenszyklus vollständig operabel: Zellteilung, Differenzierung, Transdifferenzierung, Fusion, Defusion, Apoptose
- Drei-Faktor-Lernregel mit rezeptorgesteuerter Modulation und Tag-and-Capture-Mechanismus
- Hyperbolischer Informationsraum mit desire-getriebener Migration
- Benchmark-Resultate auf CartPole (Reinforcement Learning) und MNIST (Pattern Recognition) als Basisvalidierung
- Peer-reviewtes Preprint (arXiv) mit formaler Beschreibung der MI-Architektur und Benchmark-Ergebnissen

### 4.3 Phase 2: SDK und Developer Preview (Monate 11–18)

**Ziel:** Produktionstaugliches SDK mit Bindings und Community-Launch.

**Erwartete Ergebnisse:**
- MORPHON SDK v1.0 mit stabiler öffentlicher API
- Python-Bindings (via PyO3) für Data-Science- und ML-Community
- WebAssembly-Runtime für Browser-Deployment und interaktive Demos
- Developer Preview mit Dokumentation, Tutorials und Beispielprojekten
- Developmental Programs für drei Referenz-Szenarien (Klassifikation, Zeitreihen, Motorsteuerung)
- Community-Launch auf GitHub mit Apache-2.0-Lizenz für den Kern

### 4.4 Messbare Erfolgskriterien

| Kriterium | Zielwert |
|---|---|
| CartPole-Benchmark | Lösung durch MORPHON mit ≤500 Morphons |
| MNIST-Klassifikation | ≥90% Accuracy ohne fixe Architektur |
| Laufzeitadaption | Nachweisbare Verbesserung nach Verteilungsshift ohne Retraining |
| Energieeffizienz | ≤50% der FLOPs eines gleichwertigen statischen Netzwerks (durch Sparse Activation) |
| SDK-Adoption | ≥50 GitHub Stars und ≥10 externe Contributoren innerhalb von 3 Monaten nach Launch |
| Wissenschaftliche Publikation | ≥1 Preprint auf arXiv, Einreichung bei ICML/NeurIPS/ICLR |

---

## 5. Methodik und Arbeitsplan

### AP1: Core Engine Development (Monate 1–6)

**Verantwortlich:** Lisa (Architektur, Gesamtleitung), CTO (Implementierung)

**Inhalte:**
- Implementierung der Morphon-Datenstruktur und des Lebenszyklus-Managements in Rust
- Entwicklung des hyperbolischen Informationsraums (Poincaré-Disk oder Lorentz-Modell) mit effizienten Nearest-Neighbor-Operationen
- Implementierung der Drei-Faktor-Lernregel mit Eligibility Traces und rezeptorgesteuerter Modulation
- Synaptogenese und Pruning basierend auf Aktivitätsstatistiken
- Zellteilung mit Gewichts- und Rezeptorvererbung
- Cluster-Management für Fusion und Defusion
- Apoptose-Mechanismus mit homöostatischer Regulation
- Neuromodulatorisches System: vier Kanäle, globale und lokale Signalverbreitung
- Unit- und Integrationstests für alle Kernmechanismen

**Meilenstein M1 (Monat 6):** Lauffähige Engine, die ein Morphon-System initialisieren, wachsen lassen und durch Neuromodulation steuern kann. Alle sechs Prinzipien einzeln validiert.

### AP2: Benchmarks und wissenschaftliche Publikation (Monate 7–10)

**Verantwortlich:** Research Lead, Lisa

**Inhalte:**
- Design der Benchmark-Suite: CartPole (OpenAI Gym), MNIST (standardisiert), ein proprietärer Adaptionsbenchmark (Verteilungsshift-Szenario)
- Systematische Experimente: Parameterraum-Exploration, Ablationsstudien (welche Prinzipien tragen wie viel bei?)
- Vergleich mit Baselines: statisches MLP, SMGrNN, SAPIN, Standard-SNN
- Messung emergenter Eigenschaften: Topologieentwicklung, Clusterbildung, Differenzierungsverteilung, Kritikalitätsanalyse (Power-Law-Verteilungen)
- Verfassen des wissenschaftlichen Papers (arXiv-Preprint)
- Formale Beschreibung der MI-Architektur mit mathematischer Notation

**Meilenstein M2 (Monat 10):** Benchmark-Ergebnisse dokumentiert. Preprint auf arXiv veröffentlicht. Einreichung bei einer Top-Konferenz (ICML, NeurIPS oder ICLR) vorbereitet.

### AP3: SDK und Bindings (Monate 11–14)

**Verantwortlich:** CTO, Lisa

**Inhalte:**
- Design der öffentlichen API (Rust-native, Python via PyO3, WASM via wasm-bindgen)
- Entwicklung von drei Developmental Programs (cortical, hippocampal, cerebellar)
- Serialisierung und Deserialisierung von Morphon-Systemen (Checkpoint-Management inklusive Lineage-Trees)
- Performance-Optimierung: SIMD-Nutzung, Parallelisierung über rayon, GPU-Kernels für große Systeme
- Dokumentation: API-Referenz, Architektur-Guide, Getting-Started-Tutorial
- CI/CD-Pipeline mit automatisierten Tests und Benchmark-Regression

**Meilenstein M3 (Monat 14):** SDK v1.0-rc mit Python-Bindings und WASM-Runtime. API stabil. Dokumentation vollständig.

### AP4: Community Launch und Validierung (Monate 15–18)

**Verantwortlich:** Lisa, gesamtes Team

**Inhalte:**
- Open-Source-Launch auf GitHub (Apache 2.0 für Kern)
- Erstellung von Beispielprojekten und interaktiven Demos (WASM-basiert)
- Outreach: Konferenzbeiträge (EuroSciPy, RustConf, Neuromorphic Computing Workshop), Blog-Posts, Social Media
- Feedback-Integration aus Early Adopters
- Vorbereitung der Fair-Source Professional Edition
- Marktvalidierung durch Gespräche mit potenziellen Enterprise-Kunden und Hardware-Partnern

**Meilenstein M4 (Monat 18):** Öffentlicher Launch abgeschlossen. Community aktiv. Mindestens ein LOI (Letter of Intent) für Enterprise-Pilotprojekt oder Hardware-Partnerschaft.

---

## 6. Verwertung und Marktpotenzial

### 6.1 Geschäftsmodell: Open Core mit Fair-Source-Schicht

MORPHON verfolgt ein dreistufiges Lizenzmodell, das wissenschaftliche Offenheit, Community-Wachstum und kommerzielle Nachhaltigkeit verbindet:

- **MORPHON Core** (Apache 2.0): Kern-Engine, Basis-SDK, Referenz-Developmental-Programs. Frei für Forschung, Education, Open Source und kommerzielle Nutzung.
- **MORPHON Professional** (Fair-Source): Vollständiges SDK mit erweiterten Features (fortgeschrittene Developmental Programs, Performance-Optimierungen, Multi-GPU-Support). Gratis für Unternehmen unter $10M Jahresumsatz, lizenzpflichtig darüber.
- **MORPHON Enterprise**: Studio Pro (Visual IDE), Priority Support, Custom Development Programs, SLA. Subscription-basiert.

Dieses Modell ist erprobt: Die Antragstellerin betreibt bereits das CKB-Produkt unter einem vergleichbaren Fair-Source-Modell und verfügt über operative Erfahrung mit der Balance zwischen Open-Source-Community und kommerzieller Verwertung.

### 6.2 Zielmärkte

**Primärmarkt — Edge AI und IoT (TAM: $65 Mrd. bis 2030):**
Industrielle Anomalieerkennung, personalisierte Gesundheitsüberwachung (Wearables), Smart Agriculture, Gebäudeautomation. MORPHON-Systeme leben auf dem Gerät, lernen kontinuierlich und benötigen keinen Cloud-Roundtrip.

**Sekundärmarkt — Robotik (TAM: $12 Mrd. Softwareanteil bis 2030):**
Adaptive Motorsteuerung, autonome Navigation, kollaborative Robotik. Morphons adaptieren sich an mechanischen Verschleiß und neue Umgebungen ohne Re-Deployment.

**Tertiärmarkt — Medizintechnik:**
Personalisierte Diagnosesysteme, adaptive Prothesen- und Exoskelettsteuerung. Datenschutzkonformes On-Device-Learning ohne Patientendatenexport.

**Strategischer Markt — Neuromorphic Hardware Partnerships:**
Intel (Loihi), SpiNNcloud, BrainChip und andere Chiphersteller benötigen Software-Stacks für ihre Hardware. MORPHON als hardware-agnostischer Software-Layer kann als Referenz-Framework dienen und Lizenzeinnahmen (Revenue Share) generieren.

### 6.3 Revenue-Projektion

| Revenue Stream | Zielpreis | Markteintritt |
|---|---|---|
| Professional License | €499/Entwickler/Monat | Q1 2028 |
| Enterprise Subscription | €2.000–10.000/Monat | Q3 2028 |
| Hardware Partnerships (Revenue Share) | Verhandlungsbasis | Q1 2029 |
| Consulting & Custom Development | €150–250/Stunde | Q2 2028 |
| Training & Certification | €1.500/Teilnehmer | Q4 2028 |

**Konservative Prognose:** Bei 20 Professional-Lizenzen und 3 Enterprise-Kunden nach 12 Monaten post-Launch: ca. €190.000 ARR. Break-Even bei ca. 50 Professional-Lizenzen.

### 6.4 Standortvorteil Österreich und Europa

MORPHON positioniert Österreich in einem globalen Zukunftsmarkt, der derzeit von US-amerikanischen (Liquid AI, Unconventional AI) und australischen (Cortical Labs) Akteuren dominiert wird. Als europäischer Open-Source-Anbieter mit DSGVO-konformem On-Device-Learning bietet MORPHON einen souveränitätsrelevanten Vorteil, der in der aktuellen europäischen AI-Strategie (EU AI Act, Digital Sovereignty) politisch gewünscht ist.

---

## 7. Qualifikation des Teams

### 7.1 Lisa — Geschäftsführerin und Projektleiterin

Lisa bringt eine ungewöhnliche Kombination von Kompetenzen in dieses Projekt ein:

**Software-Engineering (20+ Jahre):** Umfangreiche Erfahrung in Systemarchitektur, von Embedded Systems über Enterprise-Anwendungen bis zu AI-Plattformen. Tiefe Kompetenz in Rust, Python und WebAssembly.

**Juristische Expertise (IP/Lizenzrecht):** Fundiertes Verständnis geistigen Eigentums, Open-Source-Lizenzierung und kommerzieller Verwertungsstrategien. Direkt relevant für die Open-Core-Strategie und die Navigation des komplexen IP-Umfelds im AI-Bereich.

**AI-Architekturforschung:** Entwicklerin des ANCS/AXION-Frameworks (Autonomous Neural Coupling System / Adaptive eXperience-driven Intelligence Orchestration Network) und des Cognitive Vault-Konzepts — Vorarbeiten, die direkt in das MORPHON-Architekturdesign einfließen. Diese Arbeiten demonstrieren Kompetenz im Design neuartiger AI-Architekturen, die über Transformer-Paradigmen hinausgehen.

**Fair-Source-Geschäftsmodell:** Operative Erfahrung mit der CKB-Produktlinie, die unter einem Fair-Source-Lizenzmodell betrieben wird. Nachgewiesene Fähigkeit, Open-Source-Community-Building mit kommerzieller Nachhaltigkeit zu verbinden.

### 7.2 Geplante Rekrutierung

**CTO — Rust/GPU-Spezialist (ab Projektmonat 1):**
Profil: Mindestens 5 Jahre Erfahrung in Systems Programming (Rust), GPU-Computing (CUDA oder Metal) und Performance-Optimierung. Idealerweise Erfahrung mit numerischem Computing oder Simulationssoftware. Verantwortlich für die Kernimplementierung der MORPHON-Engine.

Rekrutierungsstrategie: Aktive Ansprache über Rust-Community (Rust Vienna Meetup, RustConf), spezialisierte Jobplattformen (Rust Jobs, Hacker News Who's Hiring). Wettbewerbsfähiges Gehaltspaket plus Equity-Beteiligung.

**Research Lead — Computational Neuroscience (ab Projektmonat 3–4):**
Profil: PhD in Computational Neuroscience, Neuroinformatics oder verwandtem Gebiet. Tiefe Kenntnis in Spiking Neural Networks, Plastizitätsregeln und bioinspiriertem Computing. Verantwortlich für die wissenschaftliche Validierung, Benchmark-Design und Publikationsstrategie.

Rekrutierungsstrategie: Kooperation mit TU Wien, Universität Wien (Fakultät für Informatik), IST Austria und europäischen Partnern (Human Brain Project Alumni, TU Graz). Postdoc-Level-Position mit Möglichkeit zur Erstautorschaft bei Publikationen.

### 7.3 Externe Expertise

Für spezifische Fragestellungen (hyperbolische Geometrie, Kritikalitätstheorie, WASM-Optimierung) wird projektbezogen auf externe Berater zurückgegriffen. Budget für externe Dienstleistungen ist im Kostenplan vorgesehen.

---

## 8. Kostenplan

### 8.1 Gesamtübersicht

| Kostenart | Betrag (€) | Anteil |
|---|---|---|
| Personalkosten | 200.000 | 67% |
| Sachmittel / Equipment | 20.000 | 7% |
| Externe Dienstleistungen | 30.000 | 10% |
| Gemeinkosten (Overhead) | 50.000 | 16% |
| **Gesamt** | **300.000** | **100%** |

### 8.2 Personalkosten (€200.000)

Basis: 1,5 FTE über 18 Monate (Mischkalkulation)

| Position | FTE | Monate | Kosten (€) |
|---|---|---|---|
| Lisa (Projektleitung, Architektur) | 0,5 | 18 | 63.000 |
| CTO — Rust/GPU (Rekrutierung ab M1) | 0,7 | 16 | 95.000 |
| Research Lead (Rekrutierung ab M3–4) | 0,3 | 14 | 42.000 |
| **Gesamt Personal** | **1,5 (Ø)** | — | **200.000** |

### 8.3 Sachmittel und Equipment (€20.000)

- High-Performance-Workstation mit GPU (NVIDIA RTX 4090 oder A6000) für Engine-Entwicklung und Benchmarks: €8.000
- Apple Silicon Mac Studio für Cross-Platform-Testing und Metal-Backend: €4.000
- Cloud-GPU-Budget (Lambda Labs, RunPod) für großskalige Experimente: €5.000
- Neuromorphe Hardware-Evaluierungsboards (Intel Loihi 2 Community Access, BrainChip Akida Eval Kit): €3.000

### 8.4 Externe Dienstleistungen (€30.000)

- Fachberatung Hyperbolische Geometrie / Informationsgeometrie: €8.000
- Fachberatung Spiking Neural Networks / Kritikalitätstheorie: €8.000
- WASM-Optimierung und Browser-Runtime-Expertise: €6.000
- Grafik und UX-Design (Dokumentation, Website, Demos): €5.000
- Rechts- und Patentberatung (Schutzstrategie): €3.000

### 8.5 Gemeinkosten (€50.000)

- Büro- und Infrastrukturkosten (Co-Working, Internetanbindung): €18.000
- Software-Lizenzen und Tooling (CI/CD, Cloud-Dienste, Entwicklungstools): €8.000
- Konferenzbesuche und Reisen (ICML, NeurIPS, RustConf, EuroSciPy): €12.000
- Publikationskosten (Open Access Fees): €4.000
- Versicherungen, Buchhaltung, Administration: €8.000

---

## 9. Risiken und Mitigationen

### 9.1 Performance-Risiko

**Risiko:** Die MORPHON-Engine erreicht nicht die Benchmark-Ziele. Morphogene Intelligenz funktioniert als Konzept, aber die Performance reicht nicht für praxisrelevante Anwendungen.

**Wahrscheinlichkeit:** Mittel. Einzelne Prinzipien (strukturelle Plastizität, Drei-Faktor-Lernen) sind akademisch validiert. Die Kombination und die Leistung in einem integrierten System sind jedoch nicht vorab garantiert.

**Mitigation:** (a) Modularer Aufbau — jedes Prinzip wird einzeln validiert, bevor die Integration erfolgt. Ablationsstudien identifizieren frühzeitig, welche Komponenten den größten Beitrag leisten. (b) Iterative Benchmark-Strategie — begonnen wird mit einfachen Umgebungen (CartPole), bevor komplexere Szenarien (MNIST, Verteilungsshift) angegangen werden. (c) Fallback: Selbst wenn die volle MI-Architektur sub-optimal performt, haben einzelne Innovationen (Tag-and-Capture in Rust, hyperbolischer Informationsraum, Zellteilungsmechanismus) eigenständigen Wert als Open-Source-Bibliotheken.

### 9.2 Markttiming-Risiko

**Risiko:** Der Neuromorphic-Computing-Markt entwickelt sich langsamer als prognostiziert, oder ein dominanter Player (Intel, Liquid AI, Unconventional AI) etabliert einen de-facto Software-Standard, bevor MORPHON Marktreife erreicht.

**Wahrscheinlichkeit:** Gering bis mittel. Die Marktprognosen sind robust (multiple Quellen bestätigen hohes Wachstum). Die Etablierung eines offenen Standards durch einen der genannten Akteure ist aufgrund ihrer proprietären Hardware-Fokussierung unwahrscheinlich.

**Mitigation:** (a) Open-Source-First-Strategie — durch die Apache-2.0-Lizenzierung des Kerns wird MORPHON als neutraler Standard positioniert, der auch neben proprietären Lösungen koexistieren kann. (b) Hardware-Agnostizität — MORPHON ist nicht an den Erfolg eines einzelnen Chipherstellers gebunden. (c) Akademische Publikation — die wissenschaftliche Veröffentlichung sichert Priorität und Sichtbarkeit unabhängig vom kommerziellen Markttiming.

### 9.3 Rekrutierungsrisiko

**Risiko:** Die Rekrutierung des CTO (Rust/GPU) und des Research Lead (Computational Neuroscience PhD) gelingt nicht im geplanten Zeitrahmen. Spezialisierte Fachkräfte sind in diesem Segment stark nachgefragt.

**Wahrscheinlichkeit:** Mittel. Rust-Expertinnen und -Experten mit GPU-Erfahrung und Computational-Neuroscience-Postdocs mit Interesse an industrienaher Forschung sind knappe Profile.

**Mitigation:** (a) Frühzeitiger Rekrutierungsbeginn — die aktive Suche beginnt vor Projektstart, um Verzögerungen zu minimieren. (b) Remote-First-Policy — der Kandidatenpool wird europaweit erweitert (nicht auf Wien beschränkt). (c) Attraktivität des Projekts — die Kombination aus Cutting-Edge-Technologie, Open-Source-Ethos und Equity-Beteiligung ist für die Zielgruppe intrinsisch motivierend. (d) Fallback: Für die ersten Monate kann Lisa die Kernarchitektur allein vorantreiben; die Rust-Implementierung kann durch erfahrene Freelancer (Rust-Ökosystem) überbrückt werden. Externe Computational-Neuroscience-Beratung kann den Research Lead temporär substituieren.

### 9.4 IP- und Regulierungsrisiko

**Risiko:** Patentansprüche durch Dritte auf Teilaspekte der MI-Architektur (insbesondere im Bereich Spiking Neural Networks und neuromorphe Algorithmen). Regulatorische Anforderungen durch den EU AI Act.

**Wahrscheinlichkeit:** Gering. Die meisten relevanten Algorithmen sind akademisch publiziert und frei nutzbar. MORPHON betreibt keine Hochrisiko-AI im Sinne des EU AI Acts.

**Mitigation:** (a) Defensive Publikationsstrategie — frühzeitige arXiv-Veröffentlichung etabliert Prior Art. (b) IP-Monitoring durch die juristische Expertise der Geschäftsführerin. (c) Apache-2.0-Lizenzierung bietet Patentschutz durch die implizite Patentlizenz. (d) Budget für Rechtsberatung ist im Kostenplan vorgesehen.

---

## Anhang: Referenzen

- Frémaux, N. & Gerstner, W. (2016). Neuromodulated Spike-Timing-Dependent Plasticity, and Theory of Three-Factor Learning Rules. *Frontiers in Neural Circuits*, 9:85.
- Hill, A. (2025/2026). SAPIN — Structurally Adaptive Predictive Inference Network. arXiv:2511.02241.
- Jia, X. (2025). Self-Motivated Growing Neural Network (SMGrNN). arXiv:2512.12713.
- Levin, M. (2023). Bioelectric networks: the cognitive glue enabling evolutionary scaling from physiology to mind. *Animal Cognition*, 26: 1865–1891.
- IT University Copenhagen (2023). Towards Self-Assembling Artificial Neural Networks through Neural Developmental Programs. arXiv:2307.08197.
- Patterns/Cell Press (2025). Three-Factor Learning in Spiking Neural Networks — Overview. *Patterns*, S2666-3899(25)00262-4.
- MarketsAndMarkets (2024). Neuromorphic Computing Market Report.

---

*Dieses Dokument ist ein Entwurf und dient als Arbeitsgrundlage für die finale Antragstellung. Formatierung und Gliederung werden an die spezifischen FFG-Einreichformulare angepasst.*
