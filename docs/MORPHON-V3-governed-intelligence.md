# MORPHON V3 — Governed Morphogenic Intelligence
## Von "Software-Organismus" zu "Organismus mit epistemischer Integrität"
### TasteHub GmbH, März 2026

---

## Was V3 löst

V1 baute den biologischen Zelllebenszyklus. V2 fügte Agency, Felder und Self-Healing hinzu. Aber beide Versionen haben einen blinden Fleck:

**Das System "glaubt" alles, was seinen Prediction Error senkt.**

Wenn ein Morphon-Cluster lernt, dass Muster X zu Outcome Y führt, speichert das System diesen Zusammenhang in seiner Topologie (prozedurales Gedächtnis) und in synaptischen Gewichten (episodisches Gedächtnis). Aber es gibt keine Mechanik, die fragt:

- *Warum* glaubt das System das? (Provenance)
- Ist die Quelle dieser Überzeugung noch gültig? (Staleness Detection)
- Was bricht, wenn diese Überzeugung sich als falsch erweist? (Dependency Tracking)
- Wer darf entscheiden, ob eine fundamentale Überzeugung geändert wird? (Governance)

TruthKeeper — dein Paper zur Wissensintegritäts-Architektur für LLMs — löst exakt diese Probleme. V3 integriert TruthKeepers Prinzipien in MORPHONs lebende Architektur.

```
V1:  System wächst und lernt              → Biologie
V2:  System hat Agency und heilt sich     → Organismus
V3:  System weiß, was es weiß und warum   → Organismus mit epistemischem Immunsystem
```

---

## Architektur-Überblick: Die drei Säulen von V3

```
┌──────────────────────────────────────────────────────────────┐
│                    GOVERNANCE LAYER (V3)                       │
│  Constitutional Constraints · Auditor Network · Shadow Deploy  │
│  Confidence-Gated Structural Changes · Separation of Powers    │
├──────────────────────────────────────────────────────────────┤
│                 TRUTH MAINTENANCE LAYER (V3)                   │
│  Dependency Network Overlay · Four-State Epistemic Model       │
│  Memory CI for Morphons · Cascade Invalidation Engine          │
├──────────────────────────────────────────────────────────────┤
│                    AGENCY LAYER (V2)                           │
│  Active Inference · Dreaming · Curiosity · Communication       │
│  Bioelectric Field · Target Morphology · Frustration-Explore   │
├──────────────────────────────────────────────────────────────┤
│                   MORPHOGENESIS LAYER (V1)                     │
│  Morphon Lifecycle · 3-Factor Learning · Neuromodulation       │
│  Division · Differentiation · Fusion · Migration · Apoptose    │
├──────────────────────────────────────────────────────────────┤
│                     RUNTIME LAYER                              │
│  Rust Core · Dual-Clock · Hyperbolischer Raum · SDK            │
└──────────────────────────────────────────────────────────────┘
```

---

## 1. Truth Maintenance Layer — Epistemisches Immunsystem

### 1.1 Das Problem: Semantische Veraltung in lebenden Systemen

TruthKeeper identifiziert das zentrale Problem statischer Wissensspeicher: **Semantic Staleness** — Informationen bleiben im Index, obwohl sie nicht mehr der Realität entsprechen, weil sich ihre Quellen geändert haben.

In MORPHON ist dieses Problem *noch verschärft*, weil Wissen nicht in diskreten Textchunks gespeichert ist, sondern in der **Topologie selbst**. Wenn ein Morphon-Cluster gelernt hat, dass "Sensor A hohe Werte bedeutet Überhitzung", und dann wird Sensor A durch Sensor B ersetzt — dann ist das Wissen in der Struktur veraltet, aber die Struktur "weiß das nicht". Der Cluster funktioniert weiter, liefert aber falsche Schlüsse.

### 1.2 Dependency Network Overlay für Morphon-Topologie

**TruthKeeper-Prinzip:** Jeder Glaube trackt seine Justifikation — welche Quellen, welche Ableitungsmethode, welche Evidenz-Spans.

**MORPHON V3-Adaptation:** Jede synaptische Verbindung und jeder Cluster bekommt ein **Justification Record**:

```rust
struct SynapticJustification {
    // Warum existiert diese Verbindung?
    formation_cause: FormationCause,
    
    // Was hat diese Verbindung verstärkt?
    reinforcement_history: Vec<ReinforcementEvent>,
    
    // Wovon hängt diese Verbindung ab?
    premises: Vec<PremiseRef>,
    
    // Was hängt von dieser Verbindung ab?
    dependents: Vec<SynapseID>,
}

enum FormationCause {
    // Hebb'sche Koinzidenz: pre und post feuerten korreliert
    HebbianCoincidence { pre: MorphonID, post: MorphonID, timestamp: u64 },
    
    // Mitose: Geerbt von Eltern-Morphon
    InheritedFromDivision { parent: MorphonID, generation: u32 },
    
    // Migration: Entstand durch Annäherung im Informationsraum
    ProximityFormation { distance_at_formation: f32 },
    
    // Fusion: Entstand durch Cluster-Verschmelzung
    FusionBridge { cluster_a: ClusterID, cluster_b: ClusterID },
    
    // Extern: Durch Developmental Program oder User-Input vorgegeben
    External { source_uri: String, timestamp: u64 },
}

struct ReinforcementEvent {
    timestamp: u64,
    modulator: ModulatorType,    // Welcher Kanal hat verstärkt?
    strength: f32,
    context: ContextSnapshot,    // Was war der Input zu diesem Zeitpunkt?
}

enum PremiseRef {
    // Harte Abhängigkeit: Spezifischer Sensor, API, Datenquelle
    HardSource { uri: String, hash: u64 },
    
    // Weiche Abhängigkeit: Korrelation mit anderem Cluster
    SoftCorrelation { cluster_id: ClusterID, correlation: f32 },
    
    // Abgeleitete Abhängigkeit: Durch Konsolidierung entstanden
    DerivedFrom { episode_id: u64, consolidation_timestamp: u64 },
}
```

### 1.3 Vier-Zustandsmodell für Morphon-Wissen

**TruthKeeper-Prinzip:** Wissen hat vier Zustände — SUPPORTED, OUTDATED, CONTESTED, HYPOTHESIS.

**MORPHON V3-Adaptation:** Jeder Cluster und jede starke Synapsengruppe hat einen **epistemischen Status**:

```rust
enum EpistemicState {
    // Verifikation gegen aktuelle Quellen bestanden
    Supported { confidence: f32, last_verified: u64 },
    
    // Quelle hat sich geändert, Wissen möglicherweise veraltet
    Outdated { reason: String, since: u64 },
    
    // Widersprüchliche Evidenz aus verschiedenen Quellen/Clustern
    Contested { evidence_for: Vec<Evidence>, evidence_against: Vec<Evidence> },
    
    // Neu gebildet, noch nicht verifiziert
    Hypothesis { formation_timestamp: u64 },
}

// Interne Pipeline-Zustände (nicht user-facing)
enum PipelineState {
    // Direkte Quelle hat sich geändert, Re-Verifikation ausstehend
    Stale { trigger: ChangeEvent },
    
    // Upstream-Dependency ist Stale/Outdated, möglicherweise betroffen
    Suspect { upstream: Vec<SynapseID> },
}
```

**Wie das mit der Topologie interagiert:**
- Ein `Supported`-Cluster hat eine starke inhibitorische Hülle (V2 Boundary Formation) — er ist geschützt
- Ein `Stale`-Cluster wird "geöffnet" — die Hülle wird permeabler, Plastizität steigt
- Ein `Contested`-Cluster triggert die Internal Inquiry (V2 Agency) — das System "fragt" aktiv nach Klärung
- Ein `Hypothesis`-Cluster hat keine Hülle — er ist vollständig plastisch und kann sich frei reorganisieren

### 1.4 Memory CI für Morphon-Systeme

**TruthKeeper-Prinzip:** Memories werden als ausführbare Verifikations-Checks behandelt, analog zu CI/CD in der Softwareentwicklung.

**MORPHON V3-Adaptation:** Continuous Verification der Topologie gegen ihre Quellen:

```rust
struct MorphonCI {
    // Source Watchers: Überwachen externe Quellen auf Änderungen
    source_watchers: Vec<SourceWatcher>,
    
    // Change Analyzers: Welche Cluster sind betroffen?
    change_analyzer: DependencyTraverser,
    
    // Verification Runners: Re-Verifikation mit cluster-spezifischen Strategien
    verifiers: HashMap<CellType, Box<dyn Verifier>>,
    
    // Result Processors: State-Updates und Benachrichtigungen
    result_processor: StateUpdater,
}

enum SourceWatcher {
    // Für Sensor-basierte Systeme
    SensorCalibration { sensor_id: String, check_interval: Duration },
    
    // Für API-basierte Systeme
    APISchema { endpoint: String, hash: u64 },
    
    // Für physikalische Umgebungen
    EnvironmentModel { model_hash: u64, drift_threshold: f32 },
    
    // Für andere MORPHON-Systeme (Multi-Agent)
    PeerSystem { peer_id: SystemID, shared_beliefs: Vec<ClusterID> },
}
```

**Konkretes Beispiel:** Ein MORPHON-System steuert einen Roboterarm.
1. Sensor A (Drehmomentsensor am Gelenk 3) wird kalibriert → MorphonCI registriert die Kalibrierung als `HardSource`
2. Der Motor-Cluster "joint3_controller" hat Synapsen, die auf Sensor A's Werte trainiert sind → `premises` verweisen auf Sensor A
3. Bei der nächsten Wartung wird Sensor A ausgetauscht → `SourceWatcher` erkennt die Änderung
4. "joint3_controller" wird als `Stale` markiert → inhibitorische Hülle wird geöffnet
5. Der Cluster re-kalibriert sich automatisch auf den neuen Sensor (kurze Plastizitätsphase)
6. Wenn die Performance nach Re-Kalibrierung OK ist → zurück zu `Supported`
7. Wenn nicht → `Contested` → Escalation an Human Operator

### 1.5 Cascade Invalidation Engine

**TruthKeeper-Prinzip:** Wenn eine Upstream-Quelle sich ändert, werden alle transitiv abhängigen Wissenseinheiten identifiziert und re-evaluiert.

**MORPHON V3-Adaptation:**

```rust
fn cascade_invalidation(system: &mut System, trigger: ChangeEvent) {
    // Phase 1: Direkt betroffene Synapsen/Cluster als STALE markieren
    let directly_affected = system.dependency_graph
        .find_premises_referencing(&trigger.source);
    
    for item in &directly_affected {
        item.set_pipeline_state(PipelineState::Stale { trigger: trigger.clone() });
        item.boundary.increase_permeability(0.3);  // Hülle öffnen
    }
    
    // Phase 2: Transitive Dependents als SUSPECT markieren (nur Graph-Traversal, keine Verifikation)
    let transitively_affected = system.dependency_graph
        .transitive_closure(&directly_affected);
    
    for item in &transitively_affected {
        item.set_pipeline_state(PipelineState::Suspect { 
            upstream: item.dependency_path_to(&directly_affected) 
        });
    }
    
    // Phase 3: Priorisierte Re-Verifikation (nicht alles auf einmal!)
    let verification_queue: PriorityQueue = transitively_affected.iter()
        .map(|item| (item, importance_score(item)))
        .collect();
    
    // Hoch-priorisierte Items: Sofort (im "Medium Path" der Dual-Clock)
    // Niedrig-priorisierte Items: Im nächsten "Glacial Path" Zyklus
    system.schedule_verification(verification_queue);
}

fn importance_score(item: &ClusterOrSynapse) -> f32 {
    let query_freq = item.recent_activation_count();
    let blast_radius = item.count_dependents_recursive();
    let criticality = item.criticality_tag().weight();
    let staleness_age = item.time_since_last_verification();
    
    0.3 * query_freq + 0.3 * blast_radius + 0.2 * criticality + 0.2 * staleness_age
}
```

### 1.6 Epistemic Shock Protection (Safe Mode)

**Problem:** Wenn eine fundamentale `HardSource` ausfällt (z.B. Hauptkamera-Treiber, zentraler API-Endpoint), kann eine Kaskade 90% der Cluster als `STALE` markieren. Das System wäre funktional gelähmt — ein "epistemischer Schock".

**Biologisches Pendant:** Der Freeze-Reflex. Bei massiver Bedrohung fällt ein Organismus in ein Reflexverhalten zurück, das auf tief verdrahteten Instinkten basiert — nicht auf erlerntem Verhalten.

```rust
struct EpistemicShockDetector {
    stale_rate_window: Duration,        // Beobachtungsfenster
    shock_threshold: f32,               // Ab welchem %-Anteil STALE ist es ein Schock?
    safe_mode_active: bool,
}

struct SafeMode {
    // Im Safe Mode:
    // 1. Alle strukturellen Änderungen STOPPEN (kein Pruning, keine Fusion, keine Division)
    // 2. Nur Constitutional Clusters (immutable_clusters) bleiben aktiv
    // 3. System fällt auf "Instinkt-Ebene" zurück — nur verified, consolidated Pfade
    // 4. Alle Novelty/Arousal Kanäle werden gedämpft → keine Exploration
    // 5. Dreaming-Engine startet Emergency-Replay der stabilsten Erinnerungen
    
    allowed_operations: HashSet<OperationType>,  // Nur: Inferenz, Homeostasis, Source-Watching
    blocked_operations: HashSet<OperationType>,   // Alles andere
    
    // Exit-Bedingung: STALE-Rate sinkt unter Threshold ODER Human-Override
    exit_condition: ExitCondition,
}

impl EpistemicShockDetector {
    fn check(&mut self, system: &System) -> Option<SafeMode> {
        let stale_fraction = system.clusters()
            .filter(|c| c.pipeline_state == PipelineState::Stale || 
                        c.pipeline_state == PipelineState::Suspect)
            .count() as f32 / system.cluster_count() as f32;
        
        if stale_fraction > self.shock_threshold {
            Some(SafeMode::activate(system))
        } else {
            None
        }
    }
}
```

**SDK-API:**
```python
system.governance.shock_detector.threshold = 0.6  # Safe Mode ab 60% STALE
system.on_safe_mode(callback=lambda: alert("MORPHON entered Safe Mode — epistemic shock detected"))
```

---

## 2. Governance Layer — Separation of Powers

### 2.1 Das CogniAI-Prinzip: Builder ≠ Auditor ≠ Governor

Das Feedback aus dem CogniAI-Vergleich war klar: In MORPHON V1/V2 entscheidet *dasselbe System*, das lernt, auch über Pruning, Fusion und Zellteilung. Das ist ein Goodhart-Risiko — das System könnte lernen, seinen eigenen Prediction Error zu manipulieren statt echte Probleme zu lösen.

V3 führt **drei funktional getrennte Subsysteme** ein:

```
┌─────────────────────────────────────────────┐
│              GOVERNOR                         │
│  Constitutional Constraints (immutable)        │
│  Invarianten, die nie verletzt werden dürfen   │
├─────────────────────────────────────────────┤
│              AUDITOR                          │
│  Unabhängiges Bewertungs-Netzwerk              │
│  Prüft: Hat die Änderung wirklich geholfen?    │
│  Hat Veto-Recht bei Diversitäts-Verletzungen   │
├─────────────────────────────────────────────┤
│              BUILDER                          │
│  Das Morphon-Netzwerk selbst                   │
│  Lernt, wächst, fusioniert, migriert           │
│  Schlägt strukturelle Änderungen vor           │
└─────────────────────────────────────────────┘
```

### 2.2 Constitutional Constraints (Governor)

Harte Invarianten, die **außerhalb der Lern-Schleife** liegen und nicht durch das System selbst modifiziert werden können:

```rust
struct ConstitutionalConstraints {
    // Strukturelle Limits
    max_morphons: usize,                    // System darf nicht unbegrenzt wachsen
    min_morphons: usize,                    // System darf nicht unter kritische Masse fallen
    max_connectivity_per_morphon: usize,    // Verhindert "Superhub"-Pathologie
    max_cluster_size_fraction: f32,         // Kein Cluster > X% des Gesamtsystems
    
    // Diversitäts-Garantien
    min_cell_type_diversity: HashMap<CellType, f32>,  // Mindestanteil pro Typ
    min_cluster_count: usize,               // Mindestens N unabhängige Cluster
    
    // Energie-Garantien
    energy_floor: f32,                       // System-Minimum, unter dem Apoptose stoppt
    max_fusion_rate_per_epoch: f32,          // Tempo-Limit für Fusionen
    max_structural_changes_per_epoch: usize, // Gesamtlimit für alle Strukturänderungen
    
    // Epistemische Garantien (V3 NEU)
    max_unverified_fraction: f32,           // Max X% des Systems darf HYPOTHESIS sein
    mandatory_justification_for: Vec<CellType>,  // Motor-Morphons MÜSSEN Justifications haben
    cascade_depth_limit: usize,             // Max Tiefe der Invalidierungskaskade
    
    // Sicherheits-Garantien
    immutable_clusters: Vec<ClusterID>,     // Bestimmte Cluster dürfen nie gelöscht werden
    forbidden_fusion_pairs: Vec<(ClusterID, ClusterID)>,  // Bestimmte Cluster dürfen nie fusionieren
}
```

**Biologisches Pendant:** DNA-kodierte Grundprogramme (Apoptose-Maschinerie, Zellzyklus-Checkpoints), die nicht durch epigenetische Modifikation verändert werden können.

### 2.3 Auditor Network (Hybrid: Symbolisch + Neural)

Ein dediziertes Sub-Netzwerk, das **nicht am normalen Lernen teilnimmt**, sondern strukturelle Änderungen unabhängig bewertet. Kritisch: Der Auditor muss gegen **Auditor-Drift** geschützt sein — wenn er selbst ein rein neuronales Netz wäre, könnte er über Zeit mit dem Builder ko-evolvieren und korrumpiert werden.

**Lösung: Zwei-Schicht-Auditor** — formale Logik als unveränderbare Basis, neuronale Bewertung als flexible Ergänzung:

```rust
struct AuditorNetwork {
    // Schicht 1: SYMBOLISCH (immutable, nicht lernbar)
    // Formale Regeln, die mathematisch verifizierbar sind
    symbolic_rules: Vec<AuditRule>,
    
    // Schicht 2: NEURAL (lernbar, aber isoliert vom Builder)
    // Bewertet "weiche" Qualitätsmetriken
    neural_assessor: IsolatedMorphonPopulation,
    
    // Der symbolische Layer hat immer Veto-Recht über den neuralen
    override_hierarchy: SymbolicOverNeural,
}

// Symbolische Regeln — die "Mathematik der Vernunft"
enum AuditRule {
    // Harte Invarianten (Verletzung → sofortiges Veto)
    DiversityFloor { cell_type: CellType, min_fraction: f32 },
    ConnectivityCap { max_per_morphon: usize },
    ClusterSizeCap { max_fraction: f32 },
    EnergyFloor { minimum: f32 },
    
    // Epistemische Invarianten
    JustificationRequired { for_types: Vec<CellType> },
    MaxUnverifiedFraction { threshold: f32 },
    
    // Konsistenz-Checks (formal verifizierbar)
    NoOrphanMorphons,                    // Jedes Morphon muss ≥1 Verbindung haben
    NoCyclicDependencies { max_depth: usize },  // Keine Zirkularabhängigkeiten in Justifications
    MonotonicConsolidation,              // Consolidated Synapsen dürfen nicht de-konsolidiert werden
}

struct IsolatedMorphonPopulation {
    // Eigene Morphon-Population (physisch isoliert vom Builder)
    morphons: Vec<MorphonID>,
    
    // Eigener Informationsraum (kann nicht vom Builder beeinflusst werden)
    field: MorphonField,
    
    // WICHTIG: Wird NICHT durch Builder-Reward trainiert
    // Sondern durch externe Validierung (Ground-Truth-Benchmarks, Human Feedback)
    training_source: ExternalValidation,
}

struct AuditMetrics {
    // Hat die Änderung den PE gesenkt? (Notwendig, aber nicht hinreichend)
    prediction_error_delta: f32,
    
    // Ist die Fähigkeits-Bandbreite erhalten? (Anti-Goodhart)
    capability_diversity: f32,          // Shannon-Entropie über Cluster-Funktionen
    
    // Ist die epistemische Integrität gewahrt?
    justified_fraction: f32,            // Anteil der Synapsen mit Justification
    
    // Ist die Topologie gesund?
    small_world_coefficient: f32,       // Clustering vs. Path Length
    modularity_score: f32,              // Wie gut sind Cluster separiert?
    
    // Ist das System resilient?
    damage_recovery_estimate: f32,      // Wie schnell würde es sich von 20% Ausfall erholen?
}

impl AuditorNetwork {
    fn evaluate_proposed_change(&self, change: &StructuralChange) -> AuditVerdict {
        // SCHRITT 1: Symbolische Regeln prüfen (harte Invarianten)
        for rule in &self.symbolic_rules {
            if let Some(violation) = rule.check(change) {
                return AuditVerdict::Veto(violation);  // Symbolisch = unanfechtbar
            }
        }
        
        // SCHRITT 2: Neuraler Assessor bewertet (weiche Qualität)
        let shadow = self.create_shadow(change);
        let metrics_before = self.measure_metrics(&shadow.before);
        let metrics_after = self.measure_metrics(&shadow.after);
        
        if metrics_after.capability_diversity < DIVERSITY_THRESHOLD {
            return AuditVerdict::Flag("Diversitäts-Warnung, Review empfohlen");
        }
        
        if metrics_after.justified_fraction < metrics_before.justified_fraction - 0.05 {
            return AuditVerdict::Flag("Justification-Verlust, Review empfohlen");
        }
        
        AuditVerdict::Approve
    }
}
```

**Warum das Auditor-Drift verhindert:**
- Die symbolischen Regeln sind *Code*, nicht Gewichte — sie können nur durch einen Human-Oracle-Eingriff geändert werden
- Der neurale Assessor wird durch *externe Validierung* trainiert (nicht durch Builder-Reward), sodass er nicht vom Builder "bestochen" werden kann
- Symbolisch hat immer Veto-Recht über neural — die "Mathematik der Vernunft" überstimmt die "Intuition der Erfahrung"

### 2.4 Shadow/Canary Deployment für Morphogenese

Inspiriert durch CogniAI: Statt strukturelle Änderungen live zu machen und bei Fehler zurückzurollen, **simuliere erst parallel**:

```rust
enum DeploymentStrategy {
    // Für kleine, häufige Änderungen (synaptische Plastizität, Pruning einzelner Synapsen)
    LiveWithRollback,
    
    // Für mittlere Änderungen (Zellteilung, Migration)
    CanaryDeploy {
        canary_fraction: f32,   // Teste auf X% des Traffics
        observation_window: Duration,
    },
    
    // Für große Änderungen (Cluster-Fusion, Massen-Division, Target Morphology Änderung)
    ShadowDeploy {
        shadow_duration: Duration,
        comparison_metrics: Vec<MetricType>,
    },
}
```

Für "Glacial Path" Events (Fusion, große Division-Kaskaden):
1. Erstelle Shadow-Kopie des betroffenen Bereichs
2. Führe die Änderung in der Shadow-Kopie durch
3. Leite denselben Input parallel an Original und Shadow
4. Vergleiche Performance über N Zyklen
5. Merge nur bei Verbesserung über *alle* Audit-Metriken (nicht nur PE)

---

## 3. Integration: Wie V3 mit V1/V2 zusammenwirkt

### 3.1 Erweiterter Morphon-Lebenszyklus

```
                    ┌── Constitutional Check ──┐
                    │   (Darf das passieren?)   │
                    └──────────┬───────────────┘
                               │
Seed → Teilung → Differenzierung → Fusion → Migration → Apoptose
         │              │            │          │           │
         │              │            │          │           │
    Justification  Justification  Shadow    Auditor     Auditor
    Record wird    Record wird    Deploy    prüft:      prüft:
    von Eltern     aktualisiert   + Audit   "Lohnt      "Darf das
    geerbt         (neuer Typ)    Approval  sich das?"  sterben?"
```

### 3.2 Vier-Zustandsmodell trifft Boundary Formation

| Epistemischer Status | Hüllen-Permeabilität | Plastizität | Verhalten |
|---|---|---|---|
| **SUPPORTED** | Niedrig (geschützt) | Minimal | Stabil, liefert zuverlässige Outputs |
| **HYPOTHESIS** | Maximal (offen) | Maximal | Exploriert, lernt schnell, kann sich frei reorganisieren |
| **STALE** | Erhöht (halb-offen) | Erhöht | Re-Kalibrierung läuft, akzeptiert neue Inputs |
| **CONTESTED** | Selektiv | Gerichtet | Nur Inputs, die den Konflikt auflösen, kommen durch |
| **OUTDATED** | Maximal | Maximal + Apoptose-Risiko | Fundamentaler Umbau oder Tod |

### 3.3 Active Inference + Truth Maintenance

V2s Active Inference Engine wählt Aktionen, die Free Energy minimieren. V3 erweitert den Action Space um epistemische Aktionen:

```rust
enum Action {
    // V2 Aktionen
    Communicate(OutputSignal),
    RequestInput(Query),
    Restructure(TopologyChange),
    Consolidate(MemoryRegion),
    Explore(Region),
    Dream(ReplayPattern),
    
    // V3 Epistemische Aktionen (NEU)
    VerifyClaim(ClusterID),             // Aktiv eine Überzeugung überprüfen
    SeekEvidence(ClusterID),            // Nach externer Bestätigung suchen
    ReportUncertainty(ClusterID),       // Dem User mitteilen: "Ich bin unsicher über X"
    RequestHumanReview(ClusterID),      // Escalation an Mensch
    ProposeRevision(ClusterID, Revision), // Änderung vorschlagen (nicht direkt durchführen)
}
```

Die Active Inference Loop berechnet jetzt auch **epistemische Free Energy**:

```
Expected_Free_Energy(action) = 
    pragmatic_value(action)              // Senkt das meinen PE?
    - curiosity * epistemic_value(action) // Lerne ich etwas Neues?
    + integrity_cost(action)              // V3 NEU: Gefährdet das meine Wissensintegrität?
```

Ein System, das eine `VerifyClaim`-Aktion wählt, "will" seine eigene Überzeugung überprüfen — nicht weil ein Mensch das angeordnet hat, sondern weil unverified Beliefs seine interne Free Energy erhöhen.

---

## 4. SDK-Erweiterungen (V3)

```python
import morphon

system = morphon.System(
    seed_size=100,
    growth_program="autonomous_navigator",
    
    # V3: Governance
    governance=morphon.Governance(
        constitution=morphon.Constitution(
            max_morphons=10000,
            min_cell_type_diversity={"Sensory": 0.15, "Motor": 0.15, "Associative": 0.2},
            max_cluster_size_fraction=0.3,
            max_unverified_fraction=0.2,
        ),
        auditor=morphon.AuditorNetwork(
            size=50,  # 50 dedizierte Auditor-Morphons
            metrics=["capability_diversity", "justified_fraction", "small_world"],
        ),
        deployment_strategy=morphon.ShadowDeploy(
            for_events=["fusion", "mass_division", "target_change"],
            observation_window=morphon.Minutes(5),
        ),
    ),
    
    # V3: Truth Maintenance
    truth_maintenance=morphon.TruthMaintenance(
        source_watchers=[
            morphon.SensorWatcher("joint_sensors/*", interval=morphon.Seconds(10)),
            morphon.APIWatcher("https://api.environment.local/schema"),
        ],
        cascade_depth_limit=5,
        auto_verify_threshold=0.9,
        escalate_threshold=0.5,
    ),
    
    # V2 Features
    fields=[morphon.Field.PREDICTION_ERROR, morphon.Field.ENERGY, morphon.Field.IDENTITY],
    target=morphon.TargetMorphology([...]),
    exploration=morphon.FrustrationExploration(onset_threshold=100),
    
    # V1 Features
    lifecycle=morphon.FullCellCycle(division=True, differentiation=True, fusion=True, apoptosis=True),
    modulation_channels=[morphon.Reward(), morphon.Novelty(), morphon.Arousal(), morphon.Homeostasis()],
)

# System entwickeln
system.develop(environment=simulator, duration=morphon.Hours(4))

# V3: Epistemischen Status beobachten
for cluster in system.clusters():
    print(f"{cluster.name}: {cluster.epistemic_state} "
          f"(justified: {cluster.justified_fraction:.0%}, "
          f"confidence: {cluster.confidence:.2f})")
# motor_controller: SUPPORTED (justified: 95%, confidence: 0.92)
# visual_cortex: HYPOTHESIS (justified: 40%, confidence: 0.55)
# anomaly_detector: CONTESTED (justified: 78%, confidence: 0.71)

# V3: Source Change simulieren
system.notify_source_change("joint_sensors/torque_3", reason="Sensor ausgetauscht")
# → motor_controller wird STALE
# → abhängige Cluster werden SUSPECT
# → Re-Kalibrierung startet automatisch
# → Wenn confidence < 0.5 nach Re-Kalibrierung → Escalation

# V3: Auditor-Bericht
audit = system.auditor.report()
print(f"Capability diversity: {audit.capability_diversity:.2f}")
print(f"Constitutional violations: {audit.violations}")
print(f"Pending reviews: {audit.pending_reviews}")

# V3: Das System kann von sich aus epistemische Aktionen wählen
system.on_spontaneous_output(callback=lambda msg: print(f"MORPHON: {msg}"))
# "Ich habe festgestellt, dass mein visual_cortex zu 60% auf unverifizierten
#  Hypothesen basiert. Soll ich eine Verifikationsrunde starten?"
```

---

## 5. Metabolische Ökonomie — Topologische Inflation verhindern

### 5.1 Das Problem: Complexity Death

Ein System, das ständig wächst, teilt und fusioniert, neigt zur "topologischen Inflation" — redundante Pfade, Zombie-Cluster, die Compute verbrauchen ohne Nutzen. In der Biologie wird das durch *Hunger* gelöst: Zellen, die keinen Nutzen stiften, bekommen keine Nährstoffe und sterben.

### 5.2 Metabolisches Budget-System

```rust
struct MetabolicSystem {
    // Jedes Morphon und jede Synapse kostet Energie pro Tick
    morphon_cost_per_tick: f32,        // Basis-Betriebskosten
    synapse_cost_per_tick: f32,        // Pro Verbindung
    cluster_overhead_per_tick: f32,    // Pro Cluster-Management
    
    // Energie wird nur durch NÜTZLICHE Aktivität verdient
    reward_for_pe_reduction: f32,      // Prediction Error gesenkt → Energie
    reward_for_successful_output: f32, // Korrekte Antwort → Energie
    reward_for_verification: f32,      // V3: Erfolgreiche Claim-Verification → Energie
    
    // Globales Budget (Constitutional Constraint)
    total_energy_pool: f32,
    energy_floor: f32,                 // Unter diesem Level stoppt Apoptose
}

impl Morphon {
    fn metabolic_tick(&mut self, system: &MetabolicSystem) {
        // Kosten abziehen
        self.energy -= system.morphon_cost_per_tick;
        self.energy -= self.outgoing.len() as f32 * system.synapse_cost_per_tick;
        
        // Einnahmen nur durch Nutzen
        if self.pe_delta_this_tick < 0.0 {
            self.energy += system.reward_for_pe_reduction * self.pe_delta_this_tick.abs();
        }
        
        // Apoptose wenn Energie erschöpft (außer Constitutional Protection)
        if self.energy <= 0.0 && !self.is_constitutionally_protected() {
            self.trigger_apoptosis();
        }
    }
}
```

**Konsequenz:** Das System wird *von selbst* maximal sparsam. Ein Morphon, das nur "mitläuft", verliert seine Energie und stirbt. Das zwingt MORPHON zu minimaler Topologie bei maximaler Leistung — ideal für Edge-Hardware.

---

## 6. Inter-System-Kommunikation — Das Babel-Problem

### 6.1 Das Problem

Zwei MORPHON-Instanzen (z.B. zwei Roboter in einer Fabrik) haben durch ihre individuelle Morphogenese völlig unterschiedliche interne Topologien entwickelt. Sie "sprechen verschiedene Sprachen" — ihre bioelektrischen Felder, Cluster-Strukturen und Signalkodierungen sind inkompatibel.

### 6.2 Translational Hubs

```rust
struct TranslationalHub {
    // Interface-Morphons, die sich auf ein standardisiertes Protokoll differenzieren
    interface_morphons: Vec<MorphonID>,
    
    // Das standardisierte Protokoll — ein "Common Bioelectric Proteome"
    protocol: InterSystemProtocol,
    
    // Encoding: Interne Repräsentation → Standard-Format
    encoder: TopologyToProtocol,
    
    // Decoding: Standard-Format → Interne Repräsentation
    decoder: ProtocolToTopology,
}

struct InterSystemProtocol {
    // Standardisierte Signale, die alle MORPHON-Instanzen verstehen
    shared_vocabulary: Vec<ConceptEmbedding>,  // Gemeinsamer Einbettungsraum
    
    // Was geteilt wird: Abstraktionen, nie Rohdaten (Privacy by Design aus V2)
    abstraction_level: f32,  // 0.0 = Rohdaten (verboten), 1.0 = nur Konzepte
    
    // Epistemischer Status wird mit-kommuniziert (V3)
    include_confidence: bool,
    include_justification_summary: bool,
}
```

**Wie zwei Roboter Wissen teilen:**
1. Roboter A lernt: "Ölige Oberfläche = Griff anpassen" → Cluster mit hoher Confidence
2. A's TranslationalHub enkodiert das als abstrahierte Erfahrung im Standard-Protokoll
3. B's TranslationalHub dekodiert es und erzeugt einen `HYPOTHESIS`-Cluster
4. B verifiziert die Hypothese durch eigene Erfahrung → wird `SUPPORTED` oder `OUTDATED`
5. Erfahrung wird nie blind übernommen — B's eigene epistemische Integrität bleibt gewahrt

```python
# SDK-API: Multi-System Communication
hub_a = system_a.create_translational_hub(protocol=morphon.StandardProtocol_v1)
hub_b = system_b.create_translational_hub(protocol=morphon.StandardProtocol_v1)

# System A teilt eine Erfahrung
experience = system_a.export_experience(
    cluster="surface_grip_controller",
    abstraction_level=0.8,
    include_confidence=True,
)

# System B importiert als Hypothese (nicht als Fakt!)
system_b.import_hypothesis(experience, source="system_a", verify_first=True)
```

---

## 7. Epistemische Reifung — Vernarbung und Versteinerung

### 7.1 Epistemic Scarring (Vernarbung)

Cluster, die wiederholt `STALE` oder `CONTESTED` waren, entwickeln eine erhöhte "Skepsis-Schwelle" — sie fordern mehr Evidenz, bevor sie eine `HYPOTHESIS` in `SUPPORTED` heben.

```rust
struct EpistemicHistory {
    stale_count: u32,           // Wie oft war dieser Cluster STALE?
    contested_count: u32,       // Wie oft CONTESTED?
    false_positive_count: u32,  // Wie oft wurde er fälschlich als SUPPORTED markiert?
    
    // Abgeleitete Skepsis-Schwelle
    skepticism: f32,            // Steigt mit negativer Erfahrung
}

impl Cluster {
    fn required_confidence_for_supported(&self) -> f32 {
        let base = 0.8;  // Standard-Schwelle
        let scar_bonus = self.epistemic_history.skepticism * 0.15;
        (base + scar_bonus).min(0.98)  // Max 0.98 — nie unmöglich
    }
}
```

**Effekt:** Das System wird in Bereichen, wo es schon mal "betrogen" wurde, *vorsichtiger*. Das ist analoges Verhalten zu einem Kind, das einmal auf die heiße Herdplatte gegriffen hat.

### 7.2 Structural Consolidation (Versteinerung)

Sehr altes, extrem oft bestätigtes Wissen wird vom Governor "versteinert" — die Synapsen werden immutabel und energiesparend.

```rust
enum ConsolidationLevel {
    Plastic,        // Normal: voll plastisch, kann geändert werden
    Stabilized,     // Häufig bestätigt: reduzierte Plastizität, niedrigere Energiekosten
    Consolidated,   // Sehr alt + sehr oft bestätigt: nur noch durch Governor änderbar
    Petrified,      // Konstitutionell geschützt: immutabel, fast null Energiekosten
}

impl Synapse {
    fn consolidation_tick(&mut self) {
        if self.age > CONSOLIDATION_AGE 
            && self.verification_count > CONSOLIDATION_VERIFICATIONS
            && self.epistemic_state == EpistemicState::Supported { .. } 
        {
            self.consolidation = ConsolidationLevel::Consolidated;
            self.energy_cost *= 0.1;  // 90% Energieeinsparung
            self.plasticity *= 0.1;   // Kaum noch änderbar
        }
    }
}
```

**Das Ergebnis:** Tief verdrahtetes Wissen (wie "Schwerkraft zieht nach unten") verbraucht fast keine Energie und ist gegen Invalidierungskaskaden geschützt. Es bildet das **"Rückgrat der Intelligenz"**.

---

## 8. Oracle-Interface — Human-in-the-Loop Governance

### 8.1 Constitutional Amendment Protocol

Der Governor hat immutable Constraints — aber die reale Welt ändert sich. Es muss einen sicheren Weg geben, die Verfassung anzupassen, ohne die Integrität zu kompromittieren.

```rust
struct OracleInterface {
    // Autorisierte Humans
    authorized_oracles: Vec<OracleIdentity>,
    
    // Amendment-Prozess (bewusst schwerfällig)
    amendment_protocol: AmendmentProtocol,
    
    // Audit-Log (immutabel, append-only)
    constitutional_history: Vec<AmendmentRecord>,
}

struct AmendmentProtocol {
    // Schritt 1: Oracle schlägt Änderung vor
    proposal: ConstitutionalChange,
    
    // Schritt 2: System simuliert Auswirkungen (Shadow Deploy)
    impact_analysis: ImpactReport,
    
    // Schritt 3: Cooling-off Period (kann nicht sofort committed werden)
    cooling_period: Duration,  // z.B. 24h
    
    // Schritt 4: Oracle bestätigt nach Cooling Period
    confirmation_required: bool,
    
    // Schritt 5: Änderung wird applied + geloggt
    applied_at: Option<u64>,
}
```

```python
# SDK-API: Oracle Interface
oracle = system.governance.oracle(credentials=my_admin_key)

# Aktuelle Verfassung einsehen
print(oracle.view_constitution())

# Amendment vorschlagen
proposal = oracle.propose_amendment(
    change=morphon.ConstitutionalChange(
        field="max_morphons",
        old_value=10000,
        new_value=15000,
        reason="Neue Sensorsuite erfordert mehr Kapazität",
    ),
)

# System zeigt Impact-Analyse
print(proposal.impact_analysis)
# "Erhöhung auf 15K Morphons: +40% Energieverbrauch, +25% Speicher,
#  ermöglicht 3 zusätzliche Sensor-Cluster. Risiko: moderat."

# Nach Cooling Period bestätigen
proposal.confirm()  # Erst nach 24h möglich
```

**EU-Compliance-Relevanz:** Dieses Interface ist das Compliance-Reporting-Tool. Jede Constitutional-Änderung ist geloggt, mit Begründung, Impact-Analyse, und Cooling Period. Perfekt für AI Act Audit-Trails.

### 8.2 Epistemische Experimente

Wenn das System `CONTESTED` Wissen hat, kann es — über die Oracle-Schnittstelle oder autonom (je nach Autonomy-Level) — **gezielte Aktionen in der Umwelt triggern**, um den Widerspruch aufzulösen:

```rust
enum EpistemicExperiment {
    // Aktive Sensorabfrage: "Lass mich den Sensor nochmal prüfen"
    ActiveSensing { sensor_id: String, test_pattern: Pattern },
    
    // Physische Probe: "Lass mich den Arm leicht bewegen, um die Kalibrierung zu testen"
    PhysicalProbe { actuator: String, minimal_action: Action },
    
    // Informationsanfrage: "Frag den Human Oracle"
    AskOracle { question: String, context: ClusterID },
    
    // Peer-Konsultation: "Frag das andere MORPHON-System"
    AskPeer { peer_id: SystemID, claim: ClusterID },
}
```

Das System "lügt" nicht mehr — es sagt: *"Ich habe hier widersprüchliche Evidenz. Darf ich Experiment X durchführen, um den Widerspruch aufzulösen?"*

---

## 9. Neuromodulatorische Injektion — Steuern wie ein Endokrinologe

Statt Code zu ändern, kann der Entwickler das System über das SDK mit künstlichen Modulatoren "impfen":

```python
# Einen "festgefahrenen" Cluster aufbrechen
system.inject(
    target="stuck_anomaly_detector",
    modulator=morphon.Novelty,
    strength=0.9,
    duration=morphon.Minutes(5),
)
# → Der Cluster gibt seine Stabilität auf und wird explorativ
# → Nach 5 Minuten normalisiert sich der Zustand

# Einen frisch migrierten Bereich "beruhigen"
system.inject(
    target="newly_formed_region",
    modulator=morphon.Homeostasis,
    strength=0.8,
    duration=morphon.Minutes(10),
)
# → Reduziert Plastizität, stabilisiert neue Verbindungen

# Globaler "Alarm" — System soll wachsam sein
system.inject_global(
    modulator=morphon.Arousal,
    strength=0.7,
)
# → Alle Schwellenwerte sinken, System reagiert sensibler
```

**Das Paradigma:** Du steuerst MORPHON nicht wie ein Programmierer (ändere Variable X), sondern wie ein Endokrinologe (injiziere Hormon Y und beobachte die systemische Reaktion). Das ist ein fundamental anderes Interaktionsmodell.

---

## 10. Vollständige Feature-Matrix: V1 → V2 → V3 (aktualisiert)

| Dimension | V1 | V2 | V3 |
|---|---|---|---|
| **Lernparadigma** | 3-Faktor lokal | + Sub-Morphon Meta-Lernen | + Epistemisch bewusstes Lernen |
| **Kommunikation** | Synapsen | + Bioelektrisches Feld | + Inter-System via Translational Hubs |
| **Ziel** | PE minimieren | Target Morphology verteidigen | + Wissensintegrität wahren |
| **Schutz** | Inhibitorische Morphons | + Hüllen, Privacy Boundaries | + Constitutional Constraints, Auditor, Safe Mode |
| **Selbstreflexion** | Keine | Curiosity Engine | + Epistemische Experimente + Claim Verification |
| **Ökonomie** | Keine | Keine | **Metabolisches Budget (Energie für Nutzen)** |
| **Provenance** | Keine | Keine | **Justification Records für jede Synapse** |
| **Staleness** | Keine Erkennung | Keine Erkennung | **Source Watchers + Cascade Invalidation + Safe Mode** |
| **Governance** | Keine | Keine | **Separation of Powers + Shadow Deploy + Oracle Interface** |
| **Reifung** | Keine | Keine | **Epistemic Scarring + Structural Consolidation** |
| **Multi-Agent** | Keine | Keine | **Translational Hubs + Standard Protocol** |
| **Human Control** | SDK-Aufrufe | SDK-Aufrufe | **Neuromodulatorische Injektion + Oracle Amendment Protocol** |
| **Vertrauen** | Alles gleich | Confidence per Signal | **4-State Epistemic Model + Vernarbung** |

---

## 11. Implementierungs-Roadmap V3 (aktualisiert)

| Feature | Phase | Abhängigkeit | Schwierigkeit |
|---|---|---|---|
| Constitutional Constraints | Phase 1 | Keine | Niedrig |
| Metabolisches Budget-System | Phase 1 | V1 Core Engine | Niedrig |
| SynapticJustification Records | Phase 1 | V1 Core Engine | Mittel |
| 4-State Epistemic Model | Phase 1 | Justification Records | Mittel |
| Epistemic Scarring | Phase 1 | Epistemic Model | Niedrig |
| Structural Consolidation | Phase 2 | Epistemic Model | Mittel |
| Auditor Network (Symbolisch) | Phase 2 | Epistemic Model | Hoch |
| Source Watchers | Phase 2 | V1 Runtime | Mittel |
| Cascade Invalidation + Safe Mode | Phase 2 | Justification + Watchers | Hoch |
| Oracle Interface | Phase 2 | Governor | Mittel |
| Neuromodulatorische Injektion API | Phase 2 | V1 Neuromodulation | Niedrig |
| Shadow Deployment | Phase 3 | Auditor | Sehr Hoch |
| Translational Hubs | Phase 3 | V2 Fields + V3 Protocol | Hoch |
| Epistemische Experimente | Phase 3 | V2 Agency + V3 TM | Sehr Hoch |
| Full Separation of Powers | Phase 3 | Alle V3 Features | Sehr Hoch |

---

## 12. Aktualisierter Investor-Pitch (V3 Final)

> "MORPHON V3 baut nicht nur Intelligenz, sondern **Glaubwürdigkeit**. Jede Verbindung trackt ihre Herkunft. Wenn sich die Welt ändert, erkennt das System automatisch, welches interne Wissen betroffen ist, und re-kalibriert sich — oder fällt in einen sicheren Reflex-Modus, wenn die Störung zu groß ist. Eine strikte Gewaltenteilung zwischen Builder, Auditor und Governor verhindert, dass das System seine eigenen Regeln umgeht. Ein metabolisches Budget-System zwingt es zur Sparsamkeit. Und wenn zwei MORPHON-Instanzen zusammenarbeiten, teilen sie Erfahrungen als Hypothesen, nie als Fakten — jedes System verifiziert selbst. Das ist die erste KI-Architektur, die *epistemische Integrität* als Designprinzip hat — nicht als Afterthought."

---

*MORPHON V3 — Intelligence that grows, heals, protects, acts, knows what it knows, and earns its beliefs.*

*TasteHub GmbH, Wien, Österreich*
*März 2026*
