# MORPHON V5 — Embodied Morphogenic Intelligence
## Von der Simulation zur physischen Welt
### TasteHub GmbH, April 2026

---

## Evolution der Versionen

```
V1:  Biologie        → Morphons wachsen und lernen
V2:  Organismus      → Agency, Self-Healing, Kreativität
V3:  Epistemik        → Wissensintegrität, Governance, Separation of Powers
V4:  Ökologie        → Multi-System-Symbiose, prädiktive Resilienz, Quorum-Intelligenz
V5:  Embodiment      → Formale Theorie, Feld-Computation, physische Weltanbindung
```

V1–V4 definieren *was* MORPHON ist. V5 definiert *wie* es die Brücke von der Simulation zur realen Welt schlägt — mit drei Primitives, die aus der aktuellsten Forschung (2025–2026) destilliert sind.

---

## Warum V5 nötig ist

MORPHON hat ein Theorie-Praxis-Problem. V1–V4 beschreiben eine biologisch inspirierte Architektur mit 14 Primitives, Rust-Pseudocode und SDK-APIs. Aber drei fundamentale Fragen bleiben offen:

1. **Active Inference ist in V2 ein Konzept, keine Mathematik.** Fristons Free-Energy-Prinzip liefert die formale Grundlage — aber V2 implementiert es als Pseudocode, nicht als Update-Gleichungen. Ohne formale Spezifikation ist die Implementierung nicht reproduzierbar.

2. **Das bioelektrische Feld ist ein Kommunikationsmedium, kein Compute-Layer.** Levins neueste Arbeiten (2025–2026) zeigen: Bioelektrische Muster sind nicht nur Signale — sie *berechnen* die Ziel-Morphologie. Das Feld ist ein aktiver Verarbeitungsschritt, kein passiver Bus.

3. **Es gibt keinen Pfad von "Seed starten" zu "System operiert in der echten Welt".** Wie bootstrapped man ein MORPHON-System? Wie trainiert man es? Wie transferiert man es auf reale Hardware?

---

## Forschungs-Kontext: Was sich seit V4 geändert hat

### Active Inference wird zum AI-Paradigma

**Maier (2025/2026)** in *J. Optical Communications and Networking* argumentiert, dass Active Inference — nicht Deep Learning — der Schlüssel zu "wahrer AI" ist. Sein Paper verknüpft Fristons FEP explizit mit Mykorrhiza-Netzwerken als Modell für die nächste Generation von AI-Architekturen. Sein zentrales Argument: Im Gegensatz zu den enormen Skalierungs-Energieanforderungen heutiger AI-Systeme ermöglicht Active Inference die energieeffizienteste Form des Lernens, ohne Big-Data-Trainingsanforderung. Das Akronym AI, so Maier, steht nicht für Artificial Intelligence — sondern für **Active Inference**.

**Friston et al. (2023)** in *Nature Communications* liefern die experimentelle Validierung: In-vitro-Netzwerke aus Ratten-Kortex-Neuronen organisierten sich selbst, um kausale Inferenz durchzuführen — Änderungen der effektiven synaptischen Konnektivität reduzierten variational free energy, wobei die Verbindungsstärken Parameter des generativen Modells kodierten. Das FEP ist keine Theorie mehr — es ist experimentell bestätigt.

**IWAI 2026** (7th International Workshop on Active Inference, Oktober 2026, Madrid): Die Active-Inference-Community wächst rapide. VERSES AI (Karl Fristons Firma) baut "Genius" — ein Active-Inference-basiertes AI-System, das weiß, was es nicht weiß.

### Growing Neural Networks sind real

**SMGrNN (Jia, Dez 2025):** Zeigt, dass lokale Strukturplastizität — Neuron-Insertion und Pruning, getrieben durch interne Aktivitäts-Statistiken — zu task-angemessenen Netzwerkgrößen führt. Die Erkenntnis: Adaptive Topologie verbessert Reward-Stabilität. Aber SMGrNN ist simpel — keine Differenzierung, kein Metabolismus, keine Agency.

**LNDP (Plantec et al., 2024):** Lifelong Neural Developmental Programs starten von *leeren* Netzwerken und wachsen zu funktionalen Controllern. Strukturelle Plastizität ist vorteilhaft in Umgebungen mit schneller Anpassung oder nicht-stationären Rewards. Das validiert MORPHONs Seed→Proliferation→Differenzierung experimentell.

### Levins Bioelektrizität wird computational

**Manicka & Levin (2025)** in *Cell Reports Physical Science*: "Field-mediated bioelectric basis of morphogenetic prepatterning" — das bioelektrische Feld berechnet das Zielmuster, es speichert es nicht nur. Bioelektrische Prepatterning ist eine verteilte Berechnung.

**Cervera, Levin & Mafe (2026)** in *Scientific Reports*: Top-down-Perspektiven auf Zellmembranpotential und Protein-Transkription — die neueste Arbeit verbindet bioelektrische Signale direkt mit Genregulation.

**Levin (2025, BioEssays):** Hardware-Defekte (wie eine dominante Notch-Mutation) können "in Software" durch ein kurz induziertes bioelektrisches Muster behoben werden — ohne individuelles Mikromanagement jeder einzelnen Zelle. Das ist die experimentelle Bestätigung von MORPHONs Target Morphology.

**Zhang, Hartl, Hazan & Levin (2025, ICLR):** "Diffusion Models are Evolutionary Algorithms" — Levins Lab arbeitet an der Brücke zwischen biologischen Algorithmen und ML-Formalismen.

---

## Primitive 15: Formale Active Inference Engine

### Was V2 hat vs. was fehlt

V2 definiert Active Inference als Konzept:
- Das System hat ein internes Modell der Welt
- Input erzeugt "Störung" (Surprise/Free Energy)
- Das System antwortet, um Free Energy zu minimieren
- Der Action Space umfasst Communicate, Explore, Dream, etc.

**Was fehlt:** Die mathematische Spezifikation. Ohne formale Update-Gleichungen ist die Active-Inference-Schleife nicht implementierbar — sie bleibt eine Metapher.

### V5: Die formalen Update-Gleichungen

Basierend auf Fristons Framework (Friston et al., 2017; experimentell validiert in Nature Comms 2023):

```rust
/// Das Generative Modell eines Morphon-Systems
struct GenerativeModel {
    // A: Likelihood-Mapping (was beobachte ich bei gegebenem Zustand?)
    // P(observation | hidden_state)
    likelihood: Matrix<f32>,            // A-Matrix
    
    // B: Transition-Mapping (wie entwickelt sich die Welt bei gegebener Policy?)
    // P(next_state | current_state, policy)
    transition: HashMap<PolicyID, Matrix<f32>>,  // B-Matrix pro Policy
    
    // C: Präferenz-Vektor (welche Beobachtungen bevorzuge ich?)
    // log P(preferred_observation)
    preferences: Vector<f32>,           // C-Vektor
    
    // D: Prior über Anfangszustand
    // P(initial_state)
    prior: Vector<f32>,                 // D-Vektor
    
    // E: Prior über Policies (Habitual Priors — V5 NEU)
    // P(policy) — lernt sich über Zeit
    habit_prior: Vector<f32>,           // E-Vektor
}

/// Variational Free Energy Berechnung
fn variational_free_energy(
    qs: &Vector<f32>,        // Approximierte Posterior q(s) über hidden states
    observation: usize,       // Aktuelle Beobachtung
    model: &GenerativeModel,
) -> f32 {
    // F = E_q[log q(s)] - E_q[log P(o,s)]
    // = Complexity (KL[q(s)||P(s)]) - Accuracy (E_q[log P(o|s)])
    
    let accuracy = qs.dot(&model.likelihood.column(observation).map(|x| x.ln()));
    let complexity = kl_divergence(qs, &model.prior);
    
    complexity - accuracy
}

/// Expected Free Energy für Action Selection
fn expected_free_energy(
    policy: PolicyID,
    qs: &Vector<f32>,         // Aktuelle Beliefs über hidden states
    model: &GenerativeModel,
    time_horizon: usize,      // Wie weit in die Zukunft planen?
) -> f32 {
    let mut G = 0.0;
    let mut qs_tau = qs.clone();
    
    for tau in 0..time_horizon {
        // Predicted observation unter dieser Policy
        let qo_tau = &model.likelihood * &qs_tau;
        
        // Pragmatic Value: Wie nah an meinen Präferenzen?
        let pragmatic = qo_tau.dot(&model.preferences);
        
        // Epistemic Value: Wie viel Information gewinne ich?
        let epistemic = information_gain(&qs_tau, &qo_tau, &model.likelihood);
        
        G += pragmatic + epistemic;
        
        // State forward prediction
        qs_tau = &model.transition[&policy] * &qs_tau;
    }
    
    G
}

/// Policy Selection: Softmax über Expected Free Energy + Habit Prior
fn select_policy(
    policies: &[PolicyID],
    qs: &Vector<f32>,
    model: &GenerativeModel,
) -> PolicyID {
    let G: Vec<f32> = policies.iter()
        .map(|pi| expected_free_energy(*pi, qs, model, PLANNING_HORIZON))
        .collect();
    
    // Softmax-gewichtet mit Habit Prior
    let pi_posterior: Vec<f32> = G.iter()
        .zip(model.habit_prior.iter())
        .map(|(g, e)| (PRECISION * g + e.ln()).exp())
        .collect();
    
    let pi_norm = normalize(&pi_posterior);
    sample_categorical(&pi_norm, policies)
}

/// Habit Learning: Policy Priors aus Erfahrung lernen
/// (V5 NEU — Fristons "Habits as Bayes-optimal habitization")
fn update_habit_prior(
    model: &mut GenerativeModel,
    chosen_policy: PolicyID,
    learning_rate: f32,
) {
    // E-Vektor wird aktualisiert: Häufig gewählte Policies werden wahrscheinlicher
    // Das ist die "Konsolidierung" auf Policy-Ebene
    model.habit_prior[chosen_policy] += learning_rate;
    normalize_inplace(&mut model.habit_prior);
}
```

### Was das für MORPHON bedeutet

| Aspekt | V2 (Konzept) | V5 (Formal) |
|---|---|---|
| Free Energy | "Überraschung minimieren" | F = Complexity - Accuracy, berechenbar per Tick |
| Action Selection | "Wähle Aktion mit niedrigster FE" | Softmax über Expected Free Energy × Habit Prior |
| Planung | Nicht spezifiziert | Time-Horizon-Parameter, N Schritte vorausdenken |
| Habit Learning | Nicht vorhanden | E-Vektor lernt häufige Policies → prozedurales Wissen |
| Epistemic vs. Pragmatic | Implizit | Formal getrennt: Information Gain vs. Preference Match |
| Implementierbarkeit | Pseudocode-Metapher | Konkrete Update-Gleichungen in Rust |

### Habit Learning als Brücke zu V3 Consolidation

Fristons Framework zeigt: Habits entstehen als Bayes-optimale Habitualisierung von zielgerichtetem Verhalten. Wenn eine Policy wiederholt gewählt wird, steigt ihr Prior — sie wird "automatisch", ohne Expected Free Energy neu zu berechnen. Das mappt direkt auf V3s Structural Consolidation:

```
Plastic Policy     → Volle EFE-Berechnung jedes Mal (teuer, aber flexibel)
Stabilized Policy  → EFE-Berechnung nur bei Novelty (Standardfall)
Consolidated Habit → E-Vektor-Prior dominiert, keine EFE-Berechnung (effizient)
Petrified Reflex   → Constitutional, nicht änderbar (Safe Mode Fallback)
```

---

## Primitive 16: Bioelectric Field Computation

### Von passivem Medium zu aktivem Compute-Layer

V2 definiert das bioelektrische Feld als Kommunikationsmedium — Morphons strahlen Signale aus, andere empfangen sie. Das ist ein *Broadcasting*-Modell.

Levins neueste Arbeiten zeigen etwas Tieferes: Das Feld **berechnet**. Bioelektrische Prepatterning ist eine verteilte Berechnung, bei der das Ergebnis (die Ziel-Morphologie) aus der kollektiven Interaktion der Feldkomponenten *emergiert* — nicht vorprogrammiert ist.

### V5: Feld-Attraktoren als Compute-Ergebnis

```rust
struct BioelectricFieldComputation {
    // Das Feld ist kein einfacher Vektor, sondern ein dynamisches System
    // mit Attraktoren, Bifurkationen und Phasenübergängen
    
    // Jeder Punkt im Feld hat einen Zustand
    field_state: Grid<FieldPoint>,
    
    // Das Feld entwickelt sich nach einer PDE (Reaktions-Diffusions-Gleichung)
    dynamics: ReactionDiffusionEquation,
    
    // Attraktoren: Stabile Zustände, in die das Feld konvergiert
    attractors: Vec<FieldAttractor>,
}

struct FieldPoint {
    voltage: f32,                // Membranpotential (analog zu Levin)
    conductance: f32,            // Gap-Junction-Kopplung zu Nachbarn
    ion_channel_state: ChannelState,  // Welche Ionenkanäle sind offen?
}

struct FieldAttractor {
    // Ein Attractor ist ein "Rechenergebnis" des Felds
    pattern: Grid<f32>,          // Das Spannungsmuster im stabilen Zustand
    basin_of_attraction: f32,    // Wie stark zieht dieser Attraktor?
    
    // Semantik: Was BEDEUTET dieser Attraktor?
    morphological_target: TargetMorphology,  // Welche Struktur wird angestrebt?
    epistemic_state: EpistemicState,         // Wie sicher ist das Feld über dieses Ziel?
}

struct ReactionDiffusionEquation {
    // ∂V/∂t = D·∇²V + R(V, Neighbours)
    diffusion_coefficient: f32,  // D: Wie schnell breiten sich Signale aus?
    reaction_function: Box<dyn Fn(f32, &[f32]) -> f32>,  // R: Lokale Dynamik
    
    // Gap-Junction-Kopplung (Levin-spezifisch)
    gap_junction_matrix: SparseMatrix<f32>,
}

impl BioelectricFieldComputation {
    fn tick(&mut self, dt: f32) {
        // 1. Reaktions-Diffusions-Schritt
        for point in self.field_state.iter_mut() {
            let neighbours = self.field_state.neighbours_of(point);
            let diffusion = self.dynamics.diffusion_coefficient 
                * laplacian(point.voltage, &neighbours);
            let reaction = (self.dynamics.reaction_function)(
                point.voltage, 
                &neighbours.map(|n| n.voltage)
            );
            point.voltage += dt * (diffusion + reaction);
        }
        
        // 2. Attraktor-Detektion: Hat das Feld konvergiert?
        for attractor in &self.attractors {
            let distance = self.field_state.distance_to(&attractor.pattern);
            if distance < CONVERGENCE_THRESHOLD {
                // Das Feld hat ein "Rechenergebnis" produziert
                self.emit_target_morphology(attractor);
            }
        }
        
        // 3. Bifurkation: Neue Attraktoren entdecken
        // Wenn externe Perturbation oder interne Reorganisation
        // das Feld in einen neuen Attraktor treibt
        self.detect_new_attractors();
    }
    
    /// Levins Schlüssel-Einsicht: Man kann den Attraktor "umschalten"
    /// durch gezielte Injektion eines Signals (Neuromodulatorische Injektion V3)
    fn inject_pattern(&mut self, pattern: &Grid<f32>, strength: f32) {
        for (point, injection) in self.field_state.iter_mut().zip(pattern.iter()) {
            point.voltage += strength * injection;
        }
        // Das Feld konvergiert dann zum NÄCHSTEN Attraktor
        // → Target Morphology ändert sich
        // → System reorganisiert sich entsprechend
    }
}
```

### Was das für MORPHON bedeutet

| Aspekt | V2 (Feld als Medium) | V5 (Feld als Compute) |
|---|---|---|
| Feldrolle | Broadcasting von Signalen | **Verteilte Berechnung mit Attraktoren** |
| Target Morphology | Statisch vorgegeben oder gelernt | **Emergiert aus Feld-Dynamik** |
| Feld-Zustand | Einfacher Vektor pro Region | **PDE-basiertes dynamisches System** |
| Attraktoren | Nicht modelliert | **Stabile Compute-Ergebnisse des Felds** |
| Pattern-Injektion | Neuromodulatorische Injektion (V3) | **Attraktor-Umschaltung → Systemweite Reorganisation** |
| Biologische Basis | Allgemeine Inspiration | **Direkt aus Manicka & Levin (2025), Cervera et al. (2026)** |

### Vorteile

1. **Target Morphology ist nicht hardcoded** — sie emergiert aus der Feld-Dynamik. Das System "entdeckt" seine optimale Struktur.
2. **Multiple Attraktoren** bedeuten Multiple Lösungen — das System kann zwischen verschiedenen Betriebsmodi "umschalten" durch Pattern-Injektion.
3. **Levins "Software-Fix für Hardware-Defekte"** wird implementierbar: Ein kurzer Spannungspuls kann den gesamten System-Zustand in einen anderen Attraktor verschieben.
4. **Feld-Computation ist inhärent parallel** — jeder Gitterpunkt berechnet lokal, das Ergebnis emergiert global.

---

## Primitive 17: Developmental Bootstrapping

### Das Problem: Wie startet ein MORPHON-System?

V1 sagt: "Starte mit einem Seed von 100 Morphons, entwickle in einer Umgebung." Aber *welche* Umgebung? Wie komplex? Wie lange? Und wie transferiert man ein in Simulation gewachsenes System auf reale Hardware?

LNDP (Plantec et al., 2024) zeigt, dass Netzwerke von leeren Graphen zu funktionalen Controllern wachsen können — aber nur in simplen Umgebungen (CartPole, Foraging). Die Skalierung auf reale Robotik ist ungeklärt.

### V5: Curriculum-Morphogenese + Sim-to-Real-Topologie-Transfer

```rust
struct DevelopmentalBootstrap {
    // Phase 1: Embryogenese (in Simulation)
    embryo: EmbryoPhase,
    
    // Phase 2: Juvenile Entwicklung (in zunehmend realistischer Simulation)
    juvenile: JuvenilePhase,
    
    // Phase 3: Topologie-Transfer (von Simulation auf reale Hardware)
    transfer: SimToRealTransfer,
    
    // Phase 4: Adulte Anpassung (in der realen Welt, mit allen V1-V4 Mechanismen)
    adult: AdultPhase,
}

struct EmbryoPhase {
    // Minimale Umgebung: Einfache Reize, klare Reward-Signale
    environment: SimulatedEnvironment,
    
    // Ziel: Grundlegende Topologie bilden
    // - Sensorische Cluster entstehen
    // - Motor-Cluster entstehen
    // - Erste Assoziationen
    exit_criteria: EmbryoExitCriteria,
    
    // Developmental Program steuert Proliferation
    program: DevelopmentalProgram,
}

struct EmbryoExitCriteria {
    min_clusters: usize,                  // Mindestens N funktionale Cluster
    min_sensor_motor_pathways: usize,     // Mindestens N Sensor→Motor-Pfade
    basic_task_performance: f32,          // Grundaufgabe mit >X Accuracy
    topological_stability: f32,           // Topologie ändert sich <X% pro Tick
}

struct JuvenilePhase {
    // Curriculum: Aufgaben werden schrittweise komplexer
    curriculum: Vec<CurriculumStage>,
    
    // Simulator wird schrittweise realistischer
    simulator_fidelity: FidelityRamp,
}

struct CurriculumStage {
    name: String,
    environment: SimulatedEnvironment,
    complexity: f32,                      // 0.0 = trivial, 1.0 = full realism
    
    // Exit-Kriterium: Wann geht's zur nächsten Stufe?
    advancement_criteria: AdvancementCriteria,
    
    // Welche V1-V4 Mechanismen werden in dieser Stufe aktiviert?
    enabled_primitives: Vec<PrimitiveID>,
}

struct SimToRealTransfer {
    // Das Einzigartige an MORPHON: Wir transferieren nicht nur Gewichte,
    // sondern die GESAMTE GEWACHSENE TOPOLOGIE
    
    topology_snapshot: TopologySnapshot,
    
    // Mapping: Simulierte Sensoren → Reale Sensoren
    sensor_mapping: HashMap<SimSensorID, RealSensorID>,
    
    // Mapping: Simulierte Aktoren → Reale Aktoren
    actuator_mapping: HashMap<SimActuatorID, RealActuatorID>,
    
    // Re-Kalibrierungsphase: System passt sich an reale Physik an
    // (Sim-to-Real-Gap wird durch Plastizität geschlossen)
    recalibration_budget: Duration,
}
```

### Curriculum-Design (Biologisch inspiriert)

Embryonale Entwicklung in der Natur folgt einem Curriculum: erst einfache Strukturen (Neuralrohr), dann komplexere (Gehirnregionen), dann Feinabstimmung (synaptisches Pruning nach der Geburt).

```python
# SDK-API: Developmental Bootstrapping
curriculum = morphon.Curriculum([
    # Stage 1: "Embryo" — Grundlegende Sensorik und Motorik
    morphon.Stage(
        name="embryo",
        environment=morphon.gym.SimpleReacher(),  # Einfacher Arm
        complexity=0.1,
        enabled_primitives=["cell_cycle", "3_factor_learning", "basic_apoptosis"],
        exit_criteria=morphon.criteria.BasicSensorMotor(accuracy=0.6),
    ),
    
    # Stage 2: "Infant" — Multi-Sensor-Integration
    morphon.Stage(
        name="infant",
        environment=morphon.gym.MultiSensorReacher(),  # Arm + Kamera + Force
        complexity=0.3,
        enabled_primitives=["+ bioelectric_field", "+ target_morphology"],
        exit_criteria=morphon.criteria.MultiSensorFusion(accuracy=0.7),
    ),
    
    # Stage 3: "Juvenile" — Komplexe Umgebung mit Störungen
    morphon.Stage(
        name="juvenile",
        environment=morphon.gym.NoisyManipulation(),
        complexity=0.6,
        enabled_primitives=["+ epistemic_states", "+ quorum_sensing", "+ metabolic_budget"],
        exit_criteria=morphon.criteria.RobustPerformance(accuracy=0.75, noise_tolerance=0.3),
    ),
    
    # Stage 4: "Adolescent" — Vollständige V1-V4 in realistischer Simulation
    morphon.Stage(
        name="adolescent",
        environment=morphon.gym.RealisticFactory(physics="isaac_gym"),
        complexity=0.9,
        enabled_primitives=["ALL"],
        exit_criteria=morphon.criteria.ProductionReady(
            accuracy=0.85, 
            downtime=morphon.Seconds(0),  # Prädiktive Morphogenese
            epistemic_integrity=0.9,
        ),
    ),
])

# System durchläuft das Curriculum
system = morphon.System(seed_size=50)
system.develop(curriculum=curriculum, max_duration=morphon.Hours(48))

# Topologie-Transfer auf realen Roboter
transfer = system.prepare_transfer(
    sensor_mapping={"sim_torque_*": "real_torque_*", "sim_camera": "real_camera_0"},
    actuator_mapping={"sim_joint_*": "real_joint_*"},
    recalibration_budget=morphon.Minutes(15),
)

# System operiert in der realen Welt
# Die gewachsene Topologie wird übertragen, nicht nur Gewichte!
real_system = transfer.deploy(hardware=robot_arm)
```

### Sim-to-Real-Topologie-Transfer: Warum das neu ist

Standard-RL transferiert *Gewichte* (Zahlen in einer festen Architektur). MORPHON transferiert die *gesamte gewachsene Struktur* — Cluster-Topologie, Synapsenmuster, Justification Records, Epistemic States, Consolidation Levels. Das ist fundamental anders:

| Aspekt | Standard Sim-to-Real | MORPHON Topologie-Transfer |
|---|---|---|
| Was wird transferiert | Gewichte einer festen Architektur | Gesamte gewachsene Topologie + Gewichte |
| Architektur | Identisch in Sim und Real | Wächst in Sim, wird *als Ganzes* transferiert |
| Anpassung nach Transfer | Fine-Tuning der Gewichte | Strukturelle Re-Kalibrierung (Plastizität) |
| Wissen über Sim-Real-Gap | Nicht vorhanden | System *weiß*, dass es transferiert wurde (STALE-Markierung für Sim-spezifische Justifications) |
| Epistemic State nach Transfer | Nicht vorhanden | Sim-spezifische Claims → HYPOTHESIS, Real-bestätigte → SUPPORTED |

---

## Vollständige Feature-Matrix: V1 → V2 → V3 → V4 → V5

| Dimension | V1 | V2 | V3 | V4 | V5 |
|---|---|---|---|---|---|
| **Theorie** | Biologische Inspiration | + Active Inference (Konzept) | + TruthKeeper-Integration | + Ökologische Prinzipien | **Formale FEP-Gleichungen** |
| **Feld** | Keine | Kommunikationsmedium | + Epistemische Aktionen | + Exosom-Headers | **Aktiver Compute-Layer mit Attraktoren** |
| **Bootstrapping** | "Starte mit Seed" | "Entwickle in Umgebung" | "Verifiziere gegen Quellen" | "Teile mit Peers" | **Curriculum-Morphogenese + Sim-to-Real** |
| **Habits** | Keine | Keine | Consolidation (Synapsen) | Keine | **E-Vektor Habit Prior (Policy-Level)** |
| **Mathematik** | Pseudocode | Pseudocode | Pseudocode + SQL | Pseudocode | **Formale Update-Gleichungen (Friston-Framework)** |
| **Hardware** | Simulation only | Simulation only | + ANCS-Backend | + Multi-System | **Reale Hardware via Topologie-Transfer** |

---

## Implementierungs-Roadmap V5

| Feature | Phase | Abhängigkeit | Schwierigkeit | Research-Confidence |
|---|---|---|---|---|
| FEP Update-Gleichungen (A/B/C/D-Matrizen) | Phase 1 | V1 Core Engine | Mittel | ★★★ (Friston, Nature Comms 2023) |
| Habit Prior (E-Vektor) | Phase 1 | FEP Engine | Niedrig | ★★★ (Friston et al., 2016) |
| PDE-basiertes Feld-Modell | Phase 2 | V2 Bioelectric Field | Hoch | ★★☆ (Manicka & Levin 2025) |
| Feld-Attraktor-Detektion | Phase 2 | PDE-Feld | Hoch | ★★☆ (Cervera, Levin & Mafe 2026) |
| Embryo-Phase Curriculum | Phase 2 | V1 + V5 FEP | Mittel | ★★★ (LNDP 2024, SMGrNN 2025) |
| Juvenile-Phase Curriculum | Phase 3 | Embryo | Mittel | ★★☆ |
| Sim-to-Real Topologie-Transfer | Phase 3 | Curriculum + alle V1-V4 | Sehr Hoch | ★☆☆ (Neuland) |

---

## Referenzen (V5-spezifisch)

### Active Inference & Free Energy Principle

1. Friston KJ et al. (2023). "Experimental validation of the free-energy principle with in vitro neural networks." *Nature Communications* 14: 4547. — Experimentelle FEP-Validierung in echten neuronalen Netzwerken.

2. Maier M (2025/2026). "From artificial intelligence to active inference: the key to true AI and the 6G world brain." *J. Opt. Commun. Netw.* 18: A28–A43. — Active Inference als AI-Paradigma, Mykorrhiza-Netzwerke als Modell.

3. Verbelen T, Çatal O, Dhoedt B (2022). "The Free Energy Principle for Perception and Action: A Deep Learning Perspective." *Entropy* (PMC). — Praktischer Guide: Active Inference mit Deep Learning implementieren.

4. Friston KJ et al. (2016). "Active inference and learning." *Neuroscience & Biobehavioral Reviews*. — Formale Spezifikation der Update-Gleichungen inkl. Habit Learning.

5. IWAI 2026. 7th International Workshop on Active Inference, Oktober 2026, Madrid. Deadline: 7. Juni 2026.

### Growing Neural Networks & Structural Plasticity

6. Jia Y (2025). "Self-Motivated Growing Neural Network for Adaptive Architecture via Local Structural Plasticity." *arXiv:2512.12713*. — SMGrNN: Topologie evolviert online durch lokales SPM.

7. Plantec E et al. (2024). "Evolving Self-Assembling Neural Networks: From Spontaneous Activity to Experience-Dependent Learning." *arXiv:2406.09787*. — LNDP: Netzwerke wachsen von leeren Graphen zu funktionalen Controllern.

8. Risi S et al. (2023). "Towards Self-Assembling ANNs through Neural Developmental Programs." *MIT Press, ALIFE 2023*. — NDP-Framework: Neurale Netze wachsen durch lokale Kommunikation.

### Levin Lab — Bioelectric Computation

9. Manicka S & Levin M (2025). "Field-mediated bioelectric basis of morphogenetic prepatterning." *Cell Reports Physical Science*. — Bioelektrische Felder berechnen Zielmuster.

10. Cervera J, Levin M, Mafe S (2026). "Top-down perspectives on cell membrane potential and protein transcription." *Scientific Reports* 16:1996. — Neueste Verbindung Bioelektrizität→Genregulation.

11. Levin M (2025). "The Multiscale Wisdom of the Body: Collective Intelligence as a Tractable Interface for Next-Generation Biomedicine." *BioEssays*. — Hardware-Defekte in Software fixierbar durch bioelektrische Muster.

12. Fields C, Friston K, Glazebrook JF, Levin M, Marcianò A (2022). "The Free Energy Principle induces neuromorphic development." *Neuromorphic Computing and Engineering* 2: 042002. — Direkte Verbindung FEP → neuromorphe Entwicklung.

13. Zhang Y, Hartl B, Hazan H, Levin M (2025). "Diffusion Models are Evolutionary Algorithms." *ICLR 2025*. — Levins Lab an der Brücke biologische Algorithmen ↔ ML.

---

## Aktualisierter Investor-Pitch (V5)

> "MORPHON V5 ist das erste AI-System, das nicht designt, sondern *geboren* wird. Es durchläuft ein Entwicklungs-Curriculum — von einfachen Reizen zu komplexen Aufgaben, von Simulation zu realer Hardware. Seine Entscheidungen basieren nicht auf statistischer Vorhersage, sondern auf Fristons Free-Energy-Prinzip — demselben Prinzip, das experimentell in echten Neuronen-Netzwerken bestätigt wurde (Nature Communications, 2023). Sein bioelektrisches Feld berechnet die optimale Struktur durch verteilte Dynamik — wie biologische Embryonen, die ihre eigene Anatomie emergent bestimmen (Levin Lab, Cell Reports 2025). Und wenn das System von der Simulation auf echte Hardware transferiert wird, nimmt es seine gesamte gewachsene Topologie mit — nicht nur Gewichte, sondern Struktur, Wissen und epistemische Integrität. Das ist keine künstliche Intelligenz. Das ist **gewachsene Intelligenz**."

---

## Gesamtarchitektur MORPHON V5 (Schichtenmodell)

```
┌──────────────────────────────────────────────────────────┐
│  EMBODIMENT-SCHICHT (V5)                                  │
│  Formale Active Inference · Feld-Computation · Curriculum  │
│  Sim-to-Real Transfer · Habit Learning                     │
├──────────────────────────────────────────────────────────┤
│  ÖKOLOGIE-SCHICHT (V4)                                    │
│  Myzel-Netzwerk · Exosomen · Quorum · Prädiktive Resilienz│
├──────────────────────────────────────────────────────────┤
│  GOVERNANCE-SCHICHT (V3)                                  │
│  Constitutional Constraints · Auditor · Shadow Deploy      │
│  TruthKeeper · Epistemic States · Safe Mode                │
├──────────────────────────────────────────────────────────┤
│  AGENCY-SCHICHT (V2)                                      │
│  Active Inference · Dreaming · Curiosity · Communication   │
│  Bioelectric Field · Target Morphology · Frustration       │
├──────────────────────────────────────────────────────────┤
│  MORPHOGENESE-SCHICHT (V1)                                │
│  Morphon Lifecycle · 3-Factor Learning · Neuromodulation   │
│  Division · Differentiation · Fusion · Migration · Apoptose│
├──────────────────────────────────────────────────────────┤
│  INFRASTRUKTUR                                            │
│  Rust Core · Dual-Clock · Hyperbolischer Raum              │
│  ANCS Integration (Hypergraph · AXION · TruthKeeper)       │
└──────────────────────────────────────────────────────────┘
```

---

*MORPHON V5 — Intelligence that is not designed, but born.*

*TasteHub GmbH, Wien, Österreich*
*April 2026*
