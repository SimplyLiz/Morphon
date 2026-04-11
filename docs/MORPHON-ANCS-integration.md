# MORPHON × ANCS Integration
## Das Gehirn trifft sein Gedächtnis
### TasteHub GmbH, April 2026

---

## 1. Warum diese Integration Sinn macht

MORPHON ist eine **Compute-Architektur** — es verarbeitet, lernt, wächst. Aber seine interne Topologie allein reicht nicht: Ein Organismus braucht nicht nur Neuronen, sondern auch ein Langzeitgedächtnis, ein Immunsystem für Wissen, und eine effiziente Kodierung für das, was er weiß.

ANCS ist eine **Wissens-Architektur** — es speichert, komprimiert, verifiziert, invalidiert. Aber es hat kein eigenes "Denken" — es ist ein passiver Store, der von außen (einem LLM-Agenten) bedient wird.

Zusammen bilden sie etwas, das keines allein sein kann:

```
MORPHON allein:  Denkt, aber vergisst ineffizient und hat keine externe Wissensvalidierung
ANCS allein:     Erinnert sich, aber kann nicht denken oder wachsen
MORPHON + ANCS:  Ein System, das denkt, sich erinnert, sein Wissen validiert,
                 effizient komprimiert, und bei Quellenänderungen automatisch re-kalibriert
```

**Biologische Analogie:**
- MORPHON = Neokortex (Verarbeitung, Lernen, Plastizität)
- ANCS Hypergraph = Hippocampus (episodisches Gedächtnis, Konsolidierung)
- AXION = Neurotransmitter-Kodierung (effiziente Signalübertragung)
- TruthKeeper = Immunsystem (erkennt und eliminiert "falsche" Erinnerungen)

---

## 2. Architektur-Überblick

```
┌─────────────────────────────────────────────────────────────────┐
│                         MORPHON Runtime                          │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐    │
│  │ Morphon  │  │ Bioelektr│  │ Agency   │  │  Governance  │    │
│  │ Network  │  │ Feld     │  │ (Active  │  │  (Auditor +  │    │
│  │ (Builder)│  │          │  │ Inference│  │   Governor)  │    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────┬───────┘    │
│       │              │              │               │            │
│  ─────┴──────────────┴──────────────┴───────────────┴─────────  │
│                    ANCS Integration Layer                        │
│  ─────┬──────────────┬──────────────┬───────────────┬─────────  │
│       │              │              │               │            │
└───────┼──────────────┼──────────────┼───────────────┼────────────┘
        │              │              │               │
┌───────┴──────────────┴──────────────┴───────────────┴────────────┐
│                         ANCS Stack                                │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐    │
│  │ AXION        │  │ Hypergraph   │  │ TruthKeeper          │    │
│  │ Compression  │  │ Store        │  │ Epistemic Governance │    │
│  │ T0/T1/T2/T3  │  │ + Bi-Temporal│  │ + Source Watchers    │    │
│  └──────────────┘  └──────────────┘  └──────────────────────┘    │
│                                                                   │
│  PostgreSQL 16 + pgvector + Apache AGE                            │
└───────────────────────────────────────────────────────────────────┘
```

---

## 3. Integration Point 1: Justification Records → ANCS Hypergraph

### Das Problem

MORPHON V3 definiert `SynapticJustification` Records — jede Verbindung trackt, warum sie existiert, welche Quellen sie stützen, und was von ihr abhängt. Aber diese Records werden aktuell *in der Morphon-Runtime selbst* gespeichert — im RAM, an die Topologie gebunden.

Das ist problematisch:
- Bei Apoptose gehen die Justification Records verloren
- Kein System-übergreifendes Querying ("Welche Cluster hängen von Sensor X ab?")
- Keine historische Analyse ("Wie hat sich die Justification über Zeit verändert?")

### Die Lösung: ANCS als Justification Store

Jede `SynapticJustification` wird als **Hyperedge** im ANCS Hypergraph gespeichert:

```sql
-- ANCS Schema-Erweiterung für MORPHON Justifications
CREATE TABLE morphon_justifications (
    justification_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Was wird gerechtfertigt?
    morphon_id TEXT NOT NULL,           -- Morphon oder Synapse ID
    element_type TEXT NOT NULL,          -- 'synapse', 'cluster', 'morphon'
    
    -- Formation Cause (AXION T2 kodiert)
    formation_cause TEXT NOT NULL,       -- AXION T2: 'hebbian_coincidence', 'inherited', etc.
    formation_context TEXT,              -- AXION T3: Kontext der Entstehung
    
    -- Premises (Links zu ANCS Sources)
    -- Nutzt das bestehende ANCS dependency-System!
    -- Keine neue Tabelle nötig — wir verlinken in die dependencies-Tabelle
    
    -- Epistemic State (TruthKeeper-kompatibel)
    truth_state truth_state NOT NULL DEFAULT 'HYPOTHESIS',
    confidence REAL DEFAULT 0.5,
    
    -- Bi-temporal Tracking (ANCS-Standard)
    valid_from TIMESTAMPTZ NOT NULL DEFAULT now(),
    valid_to TIMESTAMPTZ,               -- NULL = noch gültig
    system_from TIMESTAMPTZ NOT NULL DEFAULT now(),
    system_to TIMESTAMPTZ,
    
    -- MORPHON-spezifische Metadaten
    reinforcement_count INTEGER DEFAULT 0,
    consolidation_level TEXT DEFAULT 'plastic',  -- plastic/stabilized/consolidated/petrified
    epistemic_scar_count INTEGER DEFAULT 0,
    
    -- Embedding für semantische Suche
    embedding vector(1536)
);

-- Nutze bestehende ANCS dependencies-Tabelle für Premise-Tracking
INSERT INTO dependencies (source_id, target_id, dep_type, confidence)
VALUES (
    $sensor_source_id,          -- ANCS Source (z.B. Sensor-Kalibrierung)
    $justification_id,          -- MORPHON Justification
    'HARD',                     -- Harte Abhängigkeit
    0.95
);
```

**Was das ermöglicht:**
- `compute_blast_radius()` (ANCS CTE-Funktion) funktioniert *automatisch* über MORPHON-Justifications — keine neue Implementierung nötig
- TruthKeeper Source Watchers triggern Cascade Invalidation die *direkt* auf MORPHON-Cluster wirkt
- Bi-temporale Queries: "Zeig mir alle Justifications, die am 15. März gültig waren"
- Cross-System-Queries: "Welche MORPHON-Instanzen hängen von API-Endpoint X ab?"

### MCP-Tool-Mapping

```
ANCS MCP Tool          → MORPHON Integration Layer Aufruf
──────────────────────────────────────────────────────
axion_store            → morphon.store_justification(synapse, cause, premises)
axion_query            → morphon.query_justifications(filter, temporal_scope)
axion_entity           → morphon.register_external_source(sensor, api, peer)
axion_forget           → morphon.metabolic_demotion(pressure_level)
```

---

## 4. Integration Point 2: AXION als Kommunikationsprotokoll

### Das Problem

MORPHON V3 definiert Translational Hubs für Inter-System-Kommunikation, aber das `InterSystemProtocol` ist bisher nur eine abstrakte Spezifikation — kein konkretes Format.

### Die Lösung: AXION *ist* das Protokoll

AXION wurde designt als "AI-native compressed language" — modell-agnostisch, maximal token-effizient, mit eingebauter Konfidenz-Markierung. Es ist das perfekte Protokoll für Morphon-zu-Morphon-Kommunikation.

**Drei Kommunikationsebenen, drei AXION-Tiers:**

```
┌──────────────────────────────────────────────────────┐
│  Ebene 1: Feld-Broadcast (T3 INT)                     │
│  Morphon strahlt Zustand ins bioelektrische Feld       │
│  Kodierung: [T3] $m_42 { stress:0.8; desire:high; ⏳now } │
│  → 5-8 Tokens pro Morphon pro Broadcast                │
├──────────────────────────────────────────────────────┤
│  Ebene 2: Cluster-zu-Cluster (T2 SEM)                 │
│  Cluster teilt Erfahrung mit anderem Cluster           │
│  Kodierung: [T2] $exp_grip { surface::oily →           │
│    grip_adjust<force:+20%,confidence:0.85!> }          │
│  → 10-20 Tokens pro Erfahrung                          │
├──────────────────────────────────────────────────────┤
│  Ebene 3: System-zu-System (T2 SEM + T0 VERB)        │
│  Vollständiger Wissensexport zwischen Instanzen        │
│  Kodierung: AXION-Block mit Justification-Referenzen   │
│  → Variabel, aber immer mit Epistemic State Tag        │
└──────────────────────────────────────────────────────┘
```

**Konkretes Beispiel: Zwei Roboter teilen Wissen**

Roboter A lernt etwas über ölige Oberflächen:

```axion
AX:1.0 @morphon_robot_a ⏳2026-04-01T14:23:00Z

[T2] $exp_oily_surface {
    surface_type::oily → grip_failure<prob:0.7>;
    countermeasure: grip_force_increase<+20%> → success<prob:0.9>!;
    µ truth_state:SUPPORTED; confidence:0.88;
    µ justification:#sensor_torque_3_cal_2026_03;
    µ reinforcement_count:47; consolidation:stabilized
}
```

Roboter B empfängt und importiert als HYPOTHESIS:

```axion
AX:1.0 @morphon_robot_b ⏳2026-04-01T14:23:05Z

[T2] $imported_exp {
    #exp_oily_surface<source:@morphon_robot_a>;
    µ truth_state:HYPOTHESIS;  -- IMMER Hypothesis bei Import!
    µ confidence:0.44;         -- Halbiert: 0.88 * 0.5 (Peer-Discount)
    µ requires_verification:TRUE
}
```

**Warum AXION perfekt passt:**
- Epistemischer Status ist in die Grammatik eingebaut (`!` = high confidence, `?` = uncertain)
- Justification-Referenzen sind Pointer (`#sensor_torque_3_cal`)
- Temporal Marker (`⏳`) sind First-Class
- Tiers bestimmen automatisch den Abstraktionsgrad
- Modell-agnostisch — funktioniert zwischen verschiedenen MORPHON-Versionen

---

## 5. Integration Point 3: TruthKeeper als externes epistemisches Immunsystem

### Das Problem

MORPHON V3 hat ein internes 4-State-Epistemic-Model und eine Cascade Invalidation Engine. Aber die Verifikation von Claims gegen *externe* Quellen (APIs, Sensordaten, Dokumente) erfordert Infrastruktur, die MORPHON nicht hat — HTTP-Watchers, Schema-Parser, MiniCheck-Integration.

### Die Lösung: TruthKeeper *ist* MORPHONs externes Immunsystem

```
MORPHON (intern)                    TruthKeeper/ANCS (extern)
──────────────────                  ─────────────────────────
Morphon-internes                    Source Watchers überwachen
Epistemic Model                ←──  externe Quellen (Git, APIs,
(4 States)                          Sensoren, Dokumente)
                                    
Cluster wird STALE             ←──  Source Change Event von
                                    TruthKeeper Watcher
                                    
Re-Kalibrierung läuft          ──→  MiniCheck verifiziert
                                    Claim gegen neue Quelle
                                    
Ergebnis: SUPPORTED            ←──  Verification Run Result
oder OUTDATED                       (confidence score)
                                    
Cascade Invalidation           ←──  ANCS compute_blast_radius()
(transitive Dependents)             traversiert Dependency Graph
```

**v4.3.0: Interne Reconsolidation als Brücke**

MORPHON v4.3.0 führt `reconsolidate()` ein — eine Nader-style labile Reconsolidation. Synapsen, die konsolidiert wurden, werden wieder labil (un-konsolidiert), wenn der Morphon-interne `desire`-Wert einen Schwellenwert `theta_reconsolidate` überschreitet. Das ist das interne Pendant zu TruthKeepers SUPPORTED→HYPOTHESIS-Übergang:

```
MORPHON intern                     TruthKeeper/ANCS extern
──────────────────────             ────────────────────────────
syn.consolidated = true            truth_state = SUPPORTED
  ↓ (desire > theta_reconsolidate)   ↓ (source change event)
syn.consolidated = false           truth_state = HYPOTHESIS
  ↓ (re-learning)                    ↓ (re-verification)
syn.consolidated = true            truth_state = SUPPORTED
```

Der Unterschied: MORPHON-seitige Reconsolidation reagiert auf interne metabolische Signale (Energiedruck, hohe Desire), ANCS-seitige auf externe Quell-Änderungen. Beide führen zum selben Zustandsübergang — labile Plastizität, Re-Lernen, Re-Konsolidierung. Im integrierten System werden beide Pfade in dieselbe `morphon_justifications`-Tabelle geschrieben: `consolidation_level` wechselt von `consolidated` → `plastic` → `consolidated`.

**Konkret — der Sensor-Austausch-Flow:**

```
1. Wartungstechniker tauscht Drehmomentsensor aus
   → TruthKeeper Source Watcher erkennt Kalibrier-Hash-Änderung

2. ANCS: SELECT * FROM dependencies 
         WHERE source_id = 'sensor_torque_3'
   → Findet: 3 MORPHON Justifications abhängig

3. ANCS → MORPHON Integration Layer:
   Event: { type: "source_change", 
            source: "sensor_torque_3",
            affected_justifications: [j_47, j_112, j_203] }

4. MORPHON Cascade Invalidation Engine:
   - j_47 → Cluster "joint3_controller" → STALE
   - j_112 → Cluster "safety_monitor" → SUSPECT (transitiv)
   - j_203 → Cluster "motion_planner" → SUSPECT (transitiv)

5. MORPHON öffnet inhibitorische Hülle von "joint3_controller"
   → Plastizität steigt, Re-Kalibrierung beginnt

6. Nach Re-Kalibrierung:
   MORPHON → ANCS: verification_run(j_47, new_evidence)
   
7. TruthKeeper: MiniCheck confidence = 0.91
   → j_47 wird SUPPORTED
   → Cascade: j_112, j_203 werden SUPPORTED (upstream resolved)

8. MORPHON schließt inhibitorische Hülle
   → System operiert wieder mit voller Confidence

Gesamtdauer: ~30 Sekunden (ANCS v1 Target: ≤ 30s Time-to-Freshness)
```

---

## 6. Integration Point 4: AXION Pressure System → Metabolisches Budget

### Das Problem

MORPHON V3 hat ein metabolisches Budget-System, aber der Druckmechanismus ist primitiv — linearer Energieverlust pro Tick. ANCS v2 hat ein ausgereiftes Dreistufen-Pressure-System (Normal → Fast-Path → Emergency).

### Die Lösung: F7-Pressure-Logik für MORPHON übernehmen

```rust
struct MorphonPressureSystem {
    // Direkt übernommen aus ANCS v2 F7
    pressure_low_threshold: f32,   // 0.70 — Pressure beginnt
    pressure_high_threshold: f32,  // 0.85 — Full Pressure
    emergency_threshold: f32,      // 0.95 — Emergency Hard-Demotion
    
    current_energy_usage: f32,     // 0.0 - 1.0 (Anteil des Budgets verbraucht)
}

enum PressureMode {
    Normal,       // energy_usage < 0.70 — Standard Forgetting Curves
    Pressure,     // 0.70 <= energy_usage < 0.85 — Fast-Path Demotion
    Emergency,    // energy_usage >= 0.85 — Emergency, nur essential Morphons
    Critical,     // energy_usage >= 0.95 — Safe Mode (V3 Sektion 1.6)
}

impl MorphonPressureSystem {
    fn compute_demotion_score(&self, morphon: &Morphon) -> f32 {
        match self.current_mode() {
            PressureMode::Normal => {
                // Volles Scoring: alle AXION-Faktoren (f1-f6)
                let f1_retrievability = morphon.fsrs_retrievability();
                let f2_graph_centrality = morphon.betweenness_centrality();
                let f3_encoding_surprise = morphon.information_content();
                let f4_recency = morphon.last_activation_recency();
                let f5_pinned = if morphon.is_constitutionally_protected() { 1.0 } else { 0.0 };
                let f6_task_relevance = morphon.current_task_relevance();
                
                0.15*f1 + 0.15*f2 + 0.15*f3 + 0.20*f4 + 0.15*f5 + 0.20*f6
            },
            PressureMode::Pressure => {
                // Fast-Path: nur f1 (Retrievability) + f5 (Pinned)
                // Inspiriert von ANCS v2 Focus Agent Insight:
                // "Under pressure, only recent access time matters"
                let f1 = morphon.fsrs_retrievability();
                let f5 = if morphon.is_constitutionally_protected() { 1.0 } else { 0.0 };
                0.7*f1 + 0.3*f5
            },
            PressureMode::Emergency => {
                // Nur Pinned-Flag — alles andere wird geprüft
                if morphon.is_constitutionally_protected() { 1.0 } else { 0.0 }
            },
            PressureMode::Critical => {
                // Safe Mode aktivieren — kein Demotion-Scoring mehr,
                // Governor übernimmt vollständig
                0.0
            },
        }
    }
}
```

**ANCS-Kompatibilität:** Die F1-F6-Faktoren sind *identisch* zu AXIONs Importance-Scoring. Wenn MORPHON ANCS als Backend nutzt, können dieselben `importance`-Scores sowohl für Morphon-Demotion als auch für Memory-Item-Demotion verwendet werden — ein Scoring-System für beides.

---

## 7. Integration Point 5: ANCS-Map → Parallel Developmental Program

### Das Problem

MORPHONs Developmental Program (Seed → Proliferation → Differenzierung → Pruning) läuft aktuell sequenziell. Bei großen Systemen (>10K Morphons) ist die Proliferationsphase ein Bottleneck.

### Die Lösung: ANCS-Map-Muster für parallele Morphogenese

ANCS-Map ist ein Schema-validierter paralleler Verarbeitungsoperator mit Retry-Logik und Fehlerisolation. Dasselbe Pattern funktioniert für Morphon-Proliferation:

```rust
struct ParallelProliferationOperator {
    // Inspiriert von ANCS-Map
    max_parallel_divisions: usize,     // Wie viele gleichzeitig?
    retry_policy: RetryPolicy,          // Was bei fehlgeschlagener Division?
    isolation: ErrorIsolation,          // Ein Fehler killt nicht den ganzen Batch
    
    // Schema-Validierung: Jede neue Division wird gegen Constitutional Constraints geprüft
    validator: ConstitutionalValidator,
}

impl ParallelProliferationOperator {
    fn proliferate_batch(&self, parents: Vec<MorphonID>, system: &mut System) -> BatchResult {
        // Parallel: Jeder Parent-Morphon teilt sich unabhängig
        let results: Vec<DivisionResult> = parents.par_iter()  // Rayon parallel
            .map(|parent| {
                // Schema-Check: Darf diese Division stattfinden?
                if !self.validator.check_division(parent, system) {
                    return DivisionResult::Blocked("Constitutional limit reached");
                }
                
                // Division durchführen (isoliert — Fehler betrifft nur diesen Morphon)
                match system.divide_morphon(parent) {
                    Ok(child) => DivisionResult::Success(child),
                    Err(e) => {
                        // Retry-Policy
                        match self.retry_policy.should_retry(&e) {
                            true => self.retry_division(parent, system),
                            false => DivisionResult::Failed(e),
                        }
                    }
                }
            })
            .collect();
        
        // Ergebnis-Analyse
        BatchResult {
            successful: results.iter().filter(|r| r.is_success()).count(),
            failed: results.iter().filter(|r| r.is_failed()).count(),
            blocked: results.iter().filter(|r| r.is_blocked()).count(),
        }
    }
}
```

---

## 8. Integration Point 6: CRDT-Namespaces → Multi-MORPHON Konsistenz

ANCS Phase 4 definiert CRDT-basierte Multi-Agent-Namespaces. Für MORPHON bedeutet das:

```
MORPHON Instance A          ANCS CRDT Layer          MORPHON Instance B
─────────────────           ───────────────          ─────────────────
Cluster "grip" lernt    →   CRDT Merge: Last-     ←  Cluster "grip" lernt
"ölig = +20% force"         Writer-Wins für          "ölig = +15% force"
                            einfache Facts,
                            Conflict-Flag für
                            widersprüchliche

Beide erhalten:
- Merged Fact: "ölig = Anpassung nötig" (CONTESTED, needs review)
- Conflict Details als ANCS Memory Item mit TruthKeeper-Tracking
- Automatische Verifikation: Welche Instanz hat bessere Evidenz?
```

**Tenant Isolation:** ANCS Phase 4 bringt SaaS-Tenant-Isolation. Jede MORPHON-Instanz bekommt einen eigenen ANCS-Namespace — keine Kreuzkontamination zwischen Kunden, aber opt-in Sharing über Translational Hubs möglich.

---

## 9. SDK: Unified API

```python
import morphon
from morphon.ancs import ANCSBackend

# MORPHON mit ANCS-Backend starten
backend = ANCSBackend(
    connection="postgresql://ancs:***@localhost:5432/ancs_morphon",
    axion_protocol_version="1.0",
    truthkeeper_enabled=True,
)

system = morphon.System(
    seed_size=100,
    growth_program="autonomous_navigator",
    
    # ANCS als externes Gedächtnis
    knowledge_backend=backend,
    
    # Justifications werden automatisch in ANCS Hypergraph gespeichert
    justification_store=backend.hypergraph,
    
    # TruthKeeper Source Watchers
    source_watchers=[
        backend.sensor_watcher("joint_sensors/*", interval=10),
        backend.api_watcher("https://api.environment.local/schema"),
    ],
    
    # AXION als Inter-System-Protokoll
    communication_protocol=backend.axion_protocol,
    
    # Pressure System aus ANCS v2
    pressure_system=backend.pressure_system(
        low=0.70, high=0.85, emergency=0.95
    ),
    
    # Alle V1/V2/V3 Features
    governance=morphon.Governance(...),
    fields=[morphon.Field.PREDICTION_ERROR, morphon.Field.ENERGY],
    lifecycle=morphon.FullCellCycle(division=True, differentiation=True, fusion=True),
    modulation_channels=[morphon.Reward(), morphon.Novelty(), morphon.Arousal(), morphon.Homeostasis()],
)

# System entwickeln — Justifications werden live in ANCS geschrieben
system.develop(environment=simulator, duration=morphon.Hours(4))

# ANCS-powered Queries über das Wissen des Systems
results = backend.query("""
    MATCH (j:MorphonJustification)-[:DEPENDS_ON]->(s:Source)
    WHERE s.uri CONTAINS 'torque_sensor'
    AND j.truth_state = 'SUPPORTED'
    RETURN j.morphon_id, j.confidence, j.consolidation_level
""")

# Multi-System: Robot B importiert von Robot A via AXION
experience = system.export_experience(
    cluster="surface_grip",
    format=morphon.AXION_T2,           # AXION-kodiert
    include_justifications=True,
    include_epistemic_state=True,
)

robot_b.import_hypothesis(
    experience,
    source="robot_a",
    peer_discount=0.5,                 # Halbiertes Confidence bei Peer-Import
    verify_via=backend.truthkeeper,    # TruthKeeper verifiziert
)

# Pressure-Situation: System unter Energiedruck
print(f"Pressure mode: {system.pressure.current_mode()}")
# PressureMode::Pressure — Fast-Path Demotion aktiv
print(f"Demotion candidates: {system.pressure.candidates_count()}")
# 47 Morphons unter Demotion-Threshold
```

---

## 10. Implementierungs-Roadmap

| Integration Point | MORPHON Phase | ANCS Phase | Schwierigkeit | Abhängigkeit |
|---|---|---|---|---|
| Justification Store im Hypergraph | MORPHON Phase 1 | ANCS Phase 0 (done) | Mittel | Schema-Erweiterung, 1 neue Tabelle |
| TruthKeeper Source Watchers → MORPHON Cascade | MORPHON Phase 2 | ANCS Phase 2 | Mittel | Event-Bus zwischen ANCS und MORPHON Runtime |
| AXION als Feld-Broadcast-Kodierung | MORPHON Phase 2 | ANCS Phase 1 | Niedrig | AXION T3 Encoder in Rust wrappen |
| AXION als Inter-System-Protokoll | MORPHON Phase 3 | ANCS Phase 1 | Mittel | Translational Hub + AXION Encoder/Decoder |
| F7 Pressure System → Metabolisches Budget | MORPHON Phase 1 | ANCS v2 | Niedrig | Port der F7-Logik nach Rust |
| ANCS-Map → Parallel Proliferation | MORPHON Phase 2 | ANCS Phase 1 | Mittel | Rayon-basierter Parallel-Operator |
| CRDT Multi-System Sync | MORPHON Phase 3 | ANCS Phase 4 | Hoch | CRDT-Library + Namespace-Isolation |
| Graph Queries (Apache AGE) | MORPHON Phase 3 | ANCS Phase 3 | Hoch | openCypher Queries über MORPHON Justifications |

**Quick Wins (sofort machbar):**
1. F7 Pressure-Logik nach Rust portieren (ANCS v2 ist spezifiziert, nur Portierung nötig)
2. Justification-Tabelle in ANCS Schema hinzufügen (eine Migration, ~50 Zeilen SQL) — `consolidation_level` (plastic/stabilized/consolidated/petrified) ist ab MORPHON v4.3.0 in der Runtime voll implementiert; das SQL-Schema ist direkt mappbar
3. AXION T3 Encoder als Rust-Crate wrappen (für Feld-Broadcast)
4. Reconsolidation-Events in ANCS schreiben: `reconsolidate()` (v4.3.0) generiert MORPHON-interne SUPPORTED→HYPOTHESIS-Übergänge — diese als ANCS-Events zu exportieren ist ein einzeiliger Hook in `system.rs`

---

## 11. Warum das besser ist als alles separat

| Szenario | Ohne Integration | Mit Integration |
|---|---|---|
| Sensor wird ausgetauscht | MORPHON bemerkt Drift erst durch steigenden PE (Minuten–Stunden) | TruthKeeper erkennt Source Change sofort, MORPHON re-kalibriert in <30s |
| System unter Energiedruck | Linearer Energieverlust, keine Priorisierung | ANCS F7 Dreistufen-Eskalation: Normal → Fast-Path → Emergency |
| Zwei Roboter teilen Wissen | Undefiniertes Protokoll | AXION T2 mit Epistemic State Tags, TruthKeeper-Verifikation, CRDT-Sync |
| Auditor fragt "Warum?" | Justification nur im RAM, geht bei Restart verloren | Justification im ANCS Hypergraph, bi-temporal, querybar, persistent |
| 100 MORPHON-Instanzen in Fabrik | Jede Instanz ist eine Insel | ANCS als zentraler Knowledge Layer mit Tenant-Isolation + opt-in Sharing |

---

*MORPHON × ANCS — The brain meets its memory.*

*TasteHub GmbH, Wien, Österreich*
*April 2026*
