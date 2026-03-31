# MORPHON V2 — Advanced MI-Primitives
## Von "Wachsender KI" zum "Software-Organismus"
### Additions-Dokument — TasteHub GmbH, März 2026

---

## Überblick

V1 von MORPHON implementiert den biologischen Zelllebenszyklus: Teilung, Differenzierung, Fusion, Migration, Apoptose. V2 geht tiefer — es implementiert die **Organisationsprinzipien**, die aus einzelnen Zellen einen kohärenten Organismus machen.

Die acht neuen Primitives basieren auf Michael Levins Arbeiten zur Multiscale Competency Architecture (2023–2025) und erweitern MI von einem reaktiven System zu einem System, das **funktionale Ziele verfolgt, sich selbst heilt und strukturellen Datenschutz bietet**.

```
V1: Morphons wachsen und lernen auf Stimuli
V2: Morphons verhandeln, planen, verteidigen ihre Identität und lösen Probleme kreativ
```

---

## Primitive 1: Bioelektrisches Feld (Morphon-Field)

### Biologische Basis

Zellen kommunizieren nicht nur über direkte Berührung (Synapsen/Gap Junctions), sondern über **Vmem — Membranpotential-Felder**. Das ist ein langsames, räumlich ausgedehntes Informationsmedium, das Levin als "Wi-Fi zwischen Zellschichten" beschreibt. Es liefert Positionsinformation und Zielvorgaben für die Morphogenese — das bioelektrische "Prepattern", das bestimmt, wo Augen, Mund und andere Strukturen entstehen.

### Problem in V1

Morphons müssen in V1 eine direkte synaptische Verbindung haben, um Information auszutauschen. Das bedeutet: Ein Morphon "weiß" nicht, dass 1000 Schritte entfernt im Netzwerk ein Bereich unter Stress steht — es sei denn, das Signal über viele Hops propagiert wird. Migration ist dadurch ziellos.

### Implementierung in V2

Ein **Morphon-Field** ist eine niedrig aufgelöste räumliche Karte des hyperbolischen Informationsraums, in die Morphons ihre Zustände "ausstrahlen":

```rust
struct MorphonField {
    resolution: usize,                    // z.B. 64x64 Grid über den Poincaré-Disk
    layers: HashMap<FieldType, Grid2D>,   // Mehrere überlagerte Felder
    diffusion_rate: f32,                  // Wie schnell sich Signale ausbreiten
    decay_rate: f32,                      // Wie schnell Signale zerfallen
}

enum FieldType {
    PredictionError,   // Wo im System ist der Fehler hoch?
    Energy,            // Wo sind Ressourcen verfügbar?
    Novelty,           // Wo passiert gerade Neues?
    Stress,            // Wo ist chronische Überlastung?
    Identity,          // Welche funktionale Rolle hat diese Region?
}
```

**Wie es funktioniert:**
1. Jedes Morphon "strahlt" seinen Zustand ins Feld aus (Schreiboperation: O(1))
2. Das Feld diffundiert über Zeit — Stress-Signale breiten sich wellenförmig aus
3. Jedes Morphon "liest" das Feld an seiner Position (Leseopertion: O(1))
4. Migration folgt nun dem **Feldgradienten**, nicht blindem Random Walk

```python
# SDK-API
system.enable_field(morphon.FieldType.PREDICTION_ERROR)
system.enable_field(morphon.FieldType.ENERGY)

# Morphons migrieren jetzt automatisch entlang des PE-Gradienten
# Keine explizite Routenfindung nötig — das Feld "führt" sie
```

**Compute-Kosten:** Das Feld ist eine 2D-Diffusions-Gleichung auf einem groben Grid — trivial parallelisierbar, O(resolution²) pro Zeitschritt. Bei 64x64 = 4096 Zellen nahezu kostenlos im Vergleich zu den Morphons selbst.

**Warum das revolutionär ist:** Es gibt kein existierendes AI-System, das ein räumliches Kommunikationsmedium zwischen seinen Compute-Einheiten hat. Transformer haben Attention (global, teuer). SNNs haben Synapsen (direkt, langsam zu bilden). MI hat beides — Synapsen für direkte Kommunikation UND ein Feld für indirekte, räumlich-diffuse Koordination. Das entspricht dem Unterschied zwischen Nervensystem und Hormonsystem.

---

## Primitive 2: Target Morphology (Anatomisches Ziel)

### Biologische Basis

Levins radikalste Einsicht: Zellen haben ein **implizites Wissen über die Endgestalt**. Ein Salamander regeneriert exakt einen Arm — nicht zwei, nicht keinen. Das bioelektrische Prepattern speichert ein "Zielbild", auf das der Organismus hinarbeitet. Und — entscheidend — manche dieser Defekte können "in Software" korrigiert werden: Durch kurze Induktion des richtigen bioelektrischen Patterns können Gehirn-Geburtsfehler in Froschembryonen repariert werden, selbst bei genetischen Mutationen.

### Problem in V1

V1 MORPHON wächst rein reaktiv — es minimiert den Prediction Error auf eingehende Daten. Aber es hat kein Konzept von "so sollte mein Netzwerk aussehen" oder "diese Region soll ein Bewegungs-Controller sein". Wenn ein Teil ausfällt, versucht das System nur, den lokalen Fehler zu minimieren — es "weiß" nicht, dass es die *Funktion* wiederherstellen sollte.

### Implementierung in V2

Wir führen **Target Morphology Maps** ein — softwareseitige Zielvorgaben, die ins Morphon-Field geschrieben werden:

```rust
struct TargetMorphology {
    // Funktionale Zielvorgaben für Regionen des Informationsraums
    regions: Vec<TargetRegion>,
}

struct TargetRegion {
    center: HyperbolicPoint,       // Wo im Informationsraum
    radius: f32,                   // Wie groß
    target_cell_type: CellType,    // Was soll hier entstehen? (Sensory, Motor, etc.)
    target_density: f32,           // Wie viele Morphons sollen hier sein?
    target_connectivity: f32,      // Wie stark vernetzt?
    identity_strength: f32,        // Wie stark "verteidigt" diese Region ihre Identität?
}
```

**Wie es funktioniert:**
1. Das Developmental Program definiert eine initiale Target Morphology (analog zum "Bauplan" in der DNA)
2. Die Targets werden als `Identity`-Feld in das Morphon-Field geschrieben
3. Morphons in einer Region "spüren" die Identitäts-Vorgabe und differenzieren bevorzugt in die gewünschte Richtung
4. **Self-Healing**: Wenn Morphons ausfallen oder beschädigt werden, "bemerkt" die Region den Mismatch zwischen IST-Zustand und Target → neue Morphons werden durch Teilung und Migration rekrutiert → die funktionale Gestalt wird wiederhergestellt

```python
# SDK-API: Developmental Program mit Target Morphology
program = morphon.DevelopmentalProgram(
    name="autonomous_navigator",
    target=morphon.TargetMorphology([
        morphon.TargetRegion(
            name="visual_cortex",
            cell_type=morphon.Sensory,
            density=0.8,
            connectivity=0.6,
        ),
        morphon.TargetRegion(
            name="motor_controller",
            cell_type=morphon.Motor,
            density=0.5,
            connectivity=0.9,
        ),
        morphon.TargetRegion(
            name="decision_hub",
            cell_type=morphon.Associative,
            density=0.3,
            connectivity=0.4,
        ),
    ]),
)

system.develop(program=program, environment=simulator)

# Self-Healing: Simuliere Ausfall von 20% der Morphons
system.damage(region="motor_controller", kill_fraction=0.2)
# → System rekrutiert automatisch neue Morphons und regeneriert die Funktion
```

**Business Case:** Das ist **Self-Healing AI**. Für Raumfahrt, Unterwasser-Robotik, militärische Drohnen, medizinische Implantate — überall, wo Hardware-Ausfälle passieren und kein Mensch eingreifen kann.

---

## Primitive 3: Collective Compute Scaling (Task-Offloading)

### Biologische Basis

Wenn Zellen sich zu einem Gewebe zusammenschließen, geben sie Grundfunktionen ab (z.B. eigene Nahrungsaufnahme, Abfallentsorgung), um sich auf eine Spezialfunktion zu konzentrieren (z.B. Kontraktion im Muskel). Das einzelne Muskelzell ist "dümmer" als eine freie Zelle, aber das Gewebe als Ganzes ist dramatisch leistungsfähiger.

### Problem in V1

In V1 behält jedes Morphon in einem fusionierten Cluster seine vollen Capabilities — seinen eigenen Homeostasis-Regler, seine eigenen Eligibility Traces, sein eigenes Energy-Management. Das ist redundant und verschwendet Compute.

### Implementierung in V2

Bei Fusion delegiert das Morphon Basisfunktionen an den Cluster-Core:

```rust
struct ClusterCore {
    id: ClusterID,
    members: Vec<MorphonID>,
    
    // Delegierte Funktionen (vom Cluster zentral verwaltet)
    shared_homeostasis: HomeostaticRegulator,  // Ein Regler für alle
    shared_energy_pool: f32,                    // Gemeinsames Budget
    shared_threshold: f32,                      // Gemeinsamer Schwellenwert
    shared_eligibility: Map<SynapseID, f32>,    // Gemeinsame Traces für externe Synapsen
    
    // Cluster-Level Emergence
    cluster_activation_fn: Fn,                  // Emergente Aktivierungsfunktion
    cluster_receptors: Set<ModulatorType>,       // Emergente Rezeptor-Sensitivität
    boundary: InhibitoryShell,                  // Schutzmembran (siehe Primitive 4)
}

// Wenn ein Morphon fusioniert:
fn on_fusion(morphon: &mut Morphon, cluster: &mut ClusterCore) {
    // Schalte individuelle Homeostasis ab → Cluster übernimmt
    morphon.homeostatic_regulator = None;  // Spart ~30% Compute pro Morphon
    morphon.energy_management = None;       // Spart ~20% Compute
    morphon.autonomy -= 0.5;
    
    // Morphon konzentriert sich auf Spezialfunktion
    morphon.compute_budget *= 1.5;  // Mehr Rechenzeit für die eigentliche Aufgabe
    
    cluster.members.push(morphon.id);
    cluster.shared_energy_pool += morphon.energy;
}
```

**Quantifizierter Vorteil:**
- Ein einzelnes Morphon: 100% Compute für Homeostasis + Lernen + Inferenz
- Ein fusioniertes Morphon: ~50% Compute-Einsparung durch Task-Offloading
- Ein 10er-Cluster: ~80% Gesamteinsparung (1 Regler statt 10)
- **Collective Compute Scaling**: Der Cluster ist nicht nur die Summe seiner Teile — er ist *effizienter* als die Summe

---

## Primitive 4: Boundary Formation (Inhibitorische Hüllen / Membranen)

### Biologische Basis

Zellen entwickeln Membranen nicht nur nach außen, sondern auch interne Barrieren (Compartmentalization). Ein Neuron hat Dendriten-Compartments, die Information lokal verarbeiten, bevor sie zum Soma propagiert wird. Organe haben Bindegewebs-Kapseln, die sie vom umgebenden Gewebe abgrenzen.

### Problem in V1

Fusionierte Cluster in V1 sind gegenüber dem restlichen Netzwerk offen. Wenn eine systemweite Plastizitäts-Welle durchläuft (z.B. hohes Novelty-Signal), werden auch stabile, gut funktionierende Cluster destabilisiert.

### Implementierung in V2

```rust
struct InhibitoryShell {
    cluster_id: ClusterID,
    permeability: f32,           // 0.0 = vollständig geschlossen, 1.0 = vollständig offen
    maturity: f32,               // Wie "alt" und stabil ist die Membran?
    selective_channels: HashMap<ModulatorType, f32>,  // Unterschiedliche Permeabilität pro Kanal
}
```

**Wie es funktioniert:**
1. Wenn ein Cluster seinen Prediction Error stabil unter einem Schwellenwert hält (= die Aufgabe ist gelöst), bildet sich automatisch eine **inhibitorische Hülle**
2. Die Hülle dämpft eingehende Plastizitätssignale — der Cluster wird resistenter gegen externe Störung
3. Die Permeabilität ist selektiv: Reward-Signale durchdringen die Hülle leicht (um Verstärkung zu ermöglichen), aber Novelty-Signale werden gedämpft (um unnötiges Umlernen zu verhindern)
4. Unter extremem Stress (dauerhaft hoher PE trotz Hülle) kann die Membran "aufbrechen" — Dedifferenzierung und Reorganisation werden möglich

```python
# SDK-API
cluster = system.get_cluster("object_recognition_v2")

# Organischer Membranaufbau (passiert automatisch)
print(cluster.boundary.permeability)   # 0.2 — gut geschützt
print(cluster.boundary.maturity)       # 0.85 — stabile Membran

# Selektive Kanäle
print(cluster.boundary.selective_channels)
# {Reward: 0.8, Novelty: 0.15, Arousal: 0.3, Homeostasis: 0.9}

# Manueller Override (für Debugging/Forschung)
cluster.boundary.force_open()  # Membran temporär öffnen
```

**Warum das Catastrophic Forgetting löst:** Stabile Cluster kapseln sich organisch ein. Neues Lernen in anderen Teilen des Systems kann die eingekapselten Cluster nicht destabilisieren. Das ist **struktureller Schutz**, nicht algorithmischer (wie EWC oder PackNet) — und damit fundamental robuster.

---

## Primitive 5: Sub-Morphon-Plastizität (Multi-Scale Competency)

### Biologische Basis

Levin betont: Intelligenz existiert auf *jeder* Ebene — Moleküle, Organellen, Zellen, Gewebe, Organe, Organismen. Jede Ebene löst ihre eigenen Probleme. Ein einzelnes Neuron ist bereits ein komplexer Informationsverarbeiter mit adaptiven Dendriten, nicht nur ein simpler Threshold-Schalter.

### Problem in V1

In V1 ist das Morphon die kleinste "intelligente" Einheit. Seine Rezeptoren, Schwellenwerte und Aktivierungsfunktionen werden von oben (durch die Drei-Faktor-Regel und Differenzierung) gesteuert. Aber das Morphon selbst optimiert seine eigenen Lernregeln nicht.

### Implementierung in V2

Innerhalb jedes Morphons werden die Rezeptoren zu **adaptiven Mikro-Agenten**:

```rust
struct Receptor {
    modulator_type: ModulatorType,  // Auf was reagiert dieser Rezeptor?
    sensitivity: f32,                // Wie stark reagiert er?
    adaptation_rate: f32,            // Wie schnell passt er sich an?
    
    // Sub-Morphon-Lernen: Der Rezeptor optimiert seine eigene Sensitivität
    recent_signals: RingBuffer<f32>,
    recent_outcomes: RingBuffer<f32>,  // War die Reaktion auf das Signal nützlich?
}

impl Receptor {
    fn adapt(&mut self) {
        // Wenn auf ein Signal reagiert wurde und das Ergebnis gut war → Sensitivität beibehalten
        // Wenn auf ein Signal reagiert wurde und das Ergebnis schlecht war → Sensitivität senken
        // Das ist eine Meta-Lernregel: Das Morphon lernt, wie es lernen soll
        let correlation = pearson(&self.recent_signals, &self.recent_outcomes);
        self.sensitivity += self.adaptation_rate * correlation;
        self.sensitivity = self.sensitivity.clamp(0.01, 2.0);
    }
}
```

**Was das bedeutet:**
- Wenn du die globale Reward-Stärke falsch einstellst (zu hoch), senken die Morphons autonom ihre Reward-Sensitivität → das System korrigiert Hyperparameter-Fehler von unten
- Wenn ein bestimmtes Morphon in einer Region ist, wo Novelty-Signale irrelevant sind, reduziert es seine Novelty-Sensitivität automatisch → energieeffizienter
- **Das System optimiert seine eigenen Lernregeln während des Betriebs** — Meta-Learning ohne explizites Meta-Learning-Setup

---

## Primitive 6: Frustration-Driven Stochastic Exploration

### Biologische Basis

Zellen in der Biologie "rauschen" nicht zufällig — sie erhöhen ihr Rauschen gezielt, wenn sie in einer Sackgasse stecken. Stochastische Resonanz (das Verstärken schwacher Signale durch kontrolliertes Rauschen) ist ein fundamentaler Mechanismus der neuronalen Informationsverarbeitung.

### Problem in V1

Migration und Teilung in V1 sind primär deterministisch — getrieben durch Prediction Error und Desire. Wenn das System in einem lokalen Minimum steckt (PE stagniert, aber sinkt nicht weiter), gibt es keinen Mechanismus, um daraus auszubrechen.

### Implementierung in V2

```rust
struct FrustrationState {
    stagnation_counter: u32,       // Wie viele Zeitschritte kein PE-Fortschritt?
    frustration_level: f32,         // 0.0 = entspannt, 1.0 = maximal frustriert
    noise_amplitude: f32,           // Aktuelles Rausch-Niveau
    exploration_mode: bool,         // Ist der Frustrations-Modus aktiv?
}

impl Morphon {
    fn update_frustration(&mut self) {
        let pe_delta = self.prediction_error - self.prediction_error_last_check;
        
        if pe_delta.abs() < STAGNATION_THRESHOLD {
            self.frustration.stagnation_counter += 1;
        } else {
            self.frustration.stagnation_counter = 0;
            self.frustration.frustration_level *= 0.9;  // Entspannung bei Fortschritt
        }
        
        if self.frustration.stagnation_counter > FRUSTRATION_ONSET {
            self.frustration.frustration_level = 
                (self.frustration.stagnation_counter as f32 / FRUSTRATION_SCALE).min(1.0);
            self.frustration.noise_amplitude = 
                BASE_NOISE * (1.0 + NOISE_MULTIPLIER * self.frustration.frustration_level);
            self.frustration.exploration_mode = self.frustration.frustration_level > 0.5;
        }
    }
    
    fn apply_noise(&mut self) {
        if self.frustration.exploration_mode {
            // Zufällige Gewichtsperturbationen
            for synapse in self.incoming.values_mut() {
                synapse.weight += rand_normal(0.0, self.frustration.noise_amplitude);
            }
            // Zufällige Migration möglich (nicht nur gradientenbasiert)
            if rand() < self.frustration.frustration_level * 0.1 {
                self.position = random_nearby_point(self.position, self.frustration.noise_amplitude);
            }
        }
    }
}
```

**Der Clou:** Sobald eine zufällige Änderung den PE senkt, wird sie durch den Reward-Kanal *sofort* stabilisiert (normales Drei-Faktor-Lernen greift). Das ist echte **emergente Problemlösung** — das System "erfindet" Lösungen, die nicht in seinen Regeln vorprogrammiert waren.

**Analogie:** Das ist wie ein Mensch, der bei einer schwierigen Aufgabe frustriert wird und dann anfängt, "wild zu brainstormen" — die meisten Ideen sind Müll, aber die eine gute wird sofort erkannt und behalten.

---

## Primitive 7: Frequenz-modulierte Kommunikation

### Biologische Basis

Neuronen kodieren Information nicht (nur) in der Amplitude ihrer Signale, sondern primär in der **Frequenz** (Rate Coding) und im **Timing** (Temporal Coding). Ein Neuron, das mit 100 Hz feuert, sagt etwas anderes als eines, das mit 10 Hz feuert — auch wenn beide den gleichen "Wert" repräsentieren.

### Problem in V1

V1 Morphons kommunizieren über gewichtete Signale mit Float-Werten. Das ist anfällig für Sensor-Rauschen — ein defekter Sensor kann einen beliebigen Float-Wert senden, und das System kann nicht unterscheiden, ob `0.7` ein echter Wert oder Müll ist.

### Implementierung in V2

```rust
struct FrequencyCodedSignal {
    frequency: f32,        // Hz — wie schnell feuert das Morphon
    phase: f32,            // Timing relativ zu einem Referenz-Oszillator
    burst_pattern: Vec<f32>,  // Intra-Burst-Intervalle (komplexere Kodierung)
    confidence: f32,       // Abgeleitet aus der Regelmäßigkeit der Frequenz
}

impl Morphon {
    fn encode_output(&self) -> FrequencyCodedSignal {
        FrequencyCodedSignal {
            frequency: self.potential * MAX_FREQUENCY,
            phase: self.internal_clock.phase(),
            burst_pattern: self.generate_burst(),
            // Hohe Frequenz-Varianz → niedriges Confidence → Signal wird gedämpft
            confidence: 1.0 / (1.0 + self.frequency_variance()),
        }
    }
    
    fn decode_input(&self, signal: &FrequencyCodedSignal) -> f32 {
        // Nur Signale mit ausreichender Confidence werden voll verarbeitet
        signal.frequency / MAX_FREQUENCY * signal.confidence
    }
}
```

**Robustheits-Vorteil:** Ein defekter Sensor sendet zufällige Werte → hohe Frequenz-Varianz → niedriges Confidence → Signal wird automatisch gedämpft. Kein expliziter Anomalie-Detektor nötig — die Kodierung selbst ist der Filter.

---

## Primitive 8: Informational Boundaries (Privacy-by-Structure)

### Biologische Basis

Zelluläre Membranen entscheiden, welche Informationen geteilt werden und welche privat bleiben. Ein Neuron gibt seine Spike-Frequenz weiter, aber nicht seine internen Ionenkonzentrationen. Es teilt eine *Abstraktion*, nicht die Rohdaten.

### Problem in V1

In V1 fließen Signale ungehindert durch das gesamte Netzwerk. Es gibt keine strukturelle Garantie, dass sensitive Daten nicht zu Teilen des Systems gelangen, wo sie nicht hingehören.

### Implementierung in V2

Cluster-Membranen (Primitive 4) werden um einen **Informations-Filter** erweitert:

```rust
struct InformationalBoundary {
    cluster_id: ClusterID,
    
    // Was geht raus?
    output_abstraction_level: f32,   // 0.0 = Rohdaten, 1.0 = nur Abstraktion
    output_filter: Box<dyn Fn(Signal) -> Signal>,  // Transformations-Funktion
    
    // Was kommt rein?
    input_whitelist: Option<Set<ClusterID>>,   // Von wem akzeptiert der Cluster Input?
    input_noise_injection: f32,               // Differenzielle Privatsphäre via Rauschen
}
```

```python
# SDK-API: Privacy-by-Structure
medical_cluster = system.create_cluster(
    name="patient_data_processor",
    boundary=morphon.InformationalBoundary(
        output_abstraction_level=0.9,   # Nur hochabstrakte Ergebnisse nach außen
        input_whitelist=["sensor_cluster"],  # Akzeptiert nur von Sensoren
        input_noise_injection=0.05,     # Differentielle Privatsphäre
    )
)

# Der Rest des Systems sieht nur:
# "Patient Cluster sagt: Risiko=HOCH" 
# Aber nie: Blutdruck=180, Herzfrequenz=120, etc.
```

**Business Case:** Das ist **Privacy-by-Design auf Architektur-Ebene** — nicht als nachträglicher Filter, sondern als fundamentale Eigenschaft der Netzwerkstruktur. Perfekt für:
- Medizinische Geräte (EU MDR, HIPAA)
- Finanz-Compliance (MiFID II)
- Industriegeheimnisse (Sensordaten von Produktionslinien)
- EU-Datenschutz als Standort-Vorteil (Wien!)

---

## Zusammenfassung: V1 vs. V2 vs. V2+Agency

| Aspekt | V1 | V2 | V2+Agency |
|---|---|---|---|
| Kommunikation | Nur direkte Synapsen | Synapsen + Bioelektrisches Feld | + Aktive Kommunikation nach außen |
| Ziel | Reaktiv (PE minimieren) | Proaktiv (Target Morphology verteidigen) | **Autonom (Free Energy minimieren über alle Domänen)** |
| Effizienz | Linear | Collective Compute Scaling via Fusion | + Dreaming-basierte Konsolidierung |
| Schutz | Inhibitorische Inter-Cluster-Morphons | Inhibitorische Hüllen + Informational Boundaries | + Self-Healing bei Struktur-Schaden |
| Meta-Learning | Keine | Sub-Morphon-Plastizität | + Curiosity-getriebene Selbstuntersuchung |
| Kreativität | Deterministisch (PE-getrieben) | Frustration-Driven Exploration | + Spontane Hypothesenbildung |
| Robustheit | Float-Signale | Frequenz-kodiert mit Confidence | + Antwort-Verweigerung bei niedrigem Confidence |
| Datenschutz | Keiner | Strukturelle Privacy-by-Design | Unverändert |
| **Existenz zwischen Inputs** | **Keine (wartet)** | **Keine (wartet)** | **Aktiv: Dreaming, Self-Optimization, Curiosity** |
| **Motivation** | **Input → Output** | **Input → besserer Output** | **Interne Ziele → Aktion (auch ohne Input)** |

---

## Investor-Pitch-Satz (V2)

> "Während heutige KI-Modelle wie starre Maschinen aus Zahnrädern funktionieren, ist MORPHON ein digitaler Organismus. Er nutzt Michael Levins Prinzipien der kollektiven Intelligenz: Unsere Compute-Einheiten verhandeln über Ressourcen, verfolgen funktionale Ziele, heilen sich selbst bei Ausfall, kapseln sensible Daten durch strukturelle Membranen ein — und wenn sie in einer Sackgasse stecken, werden sie kreativ. Das ist nicht AI, die trainiert wurde. Das ist AI, die *will*, dass ihre Struktur funktioniert."

---

## SDK-API-Erweiterungen (V2 Summary)

```python
import morphon

system = morphon.System(
    seed_size=100,
    growth_program="autonomous_navigator",
    
    # V2: Bioelektrisches Feld
    fields=[
        morphon.Field.PREDICTION_ERROR,
        morphon.Field.ENERGY,
        morphon.Field.IDENTITY,
    ],
    
    # V2: Target Morphology
    target=morphon.TargetMorphology([
        morphon.TargetRegion("visual", cell_type=morphon.Sensory, density=0.8),
        morphon.TargetRegion("motor", cell_type=morphon.Motor, density=0.5),
        morphon.TargetRegion("decision", cell_type=morphon.Associative, density=0.3),
    ]),
    
    # V2: Frustration-Driven Exploration
    exploration=morphon.FrustrationExploration(
        onset_threshold=100,    # Zeitschritte ohne Fortschritt
        noise_multiplier=2.0,
    ),
    
    # V2: Frequency-Coded Communication
    signal_encoding=morphon.FrequencyCoding(max_frequency=200.0),
    
    # V1 Features bleiben
    lifecycle=morphon.FullCellCycle(division=True, differentiation=True, fusion=True, apoptosis=True),
    modulation_channels=[morphon.Reward(), morphon.Novelty(), morphon.Arousal(), morphon.Homeostasis()],
)

# System entwickeln
system.develop(environment=isaac_gym_env, duration=morphon.Hours(4))

# V2: Self-Healing testen
system.damage(region="motor", kill_fraction=0.3)
system.observe_regeneration(timeout=morphon.Minutes(10))
# → System rekrutiert Morphons, regeneriert Motor-Cluster

# V2: Privacy Cluster
system.create_private_cluster(
    name="patient_data",
    abstraction_level=0.9,
    input_whitelist=["sensor_hub"],
)

# V2: Frustration-Status beobachten
for cluster in system.clusters():
    print(f"{cluster.name}: frustration={cluster.frustration_level:.2f}")
```

---

## Implementierungs-Prioritäten für V2

| Primitive | Priorität | Schwierigkeit | Impact | Phase |
|---|---|---|---|---|
| **Bioelektrisches Feld** | KRITISCH | Mittel | Löst ziellose Migration | Phase 1 |
| **Target Morphology** | HOCH | Mittel | Self-Healing = Killer-Feature | Phase 1 |
| **Frustration-Exploration** | HOCH | Niedrig | ~50 Zeilen Code, großer Effekt | Phase 1 |
| **Boundary Formation** | HOCH | Mittel | Löst Catastrophic Forgetting | Phase 2 |
| **Collective Compute Scaling** | MITTEL | Mittel | Effizienz-Argument für Investoren | Phase 2 |
| **Sub-Morphon-Plastizität** | MITTEL | Hoch | Auto-Hyperparameter-Tuning | Phase 2 |
| **Agency & Active Inference** | MITTEL | Sehr Hoch | Paradigmenwechsel: System als Agent, nicht als Tool | Phase 3 |
| **Frequenz-Kodierung** | NIEDRIG | Mittel | Robustheits-Feature für Edge | Phase 3 |
| **Informational Boundaries** | NIEDRIG (tech), HOCH (business) | Niedrig | EU-Datenschutz Compliance | Phase 3 |

---

## Primitive 9: Agency & Active Inference — Das System als autonomer Agent

### Warum das alles verändert

Die Primitives 1–8 machen MORPHON zu einem System, das *auf Stimuli reagiert* — besser, adaptiver und robuster als Transformer, aber immer noch fundamentally reaktiv. Primitive 9 dreht das um: MORPHON wird zu einem System, das **eigene Ziele verfolgt**, von sich aus handelt, und Kommunikation als Werkzeug nutzt.

Das ist der Unterschied zwischen einem Thermostat (reagiert auf Temperatur) und einem Organismus (hat Hunger, sucht Nahrung, plant voraus).

### 9.1 Active Inference als Kern-Antrieb

**Warum Transformer "antworten":** Sie berechnen die statistisch wahrscheinlichste Fortsetzung eines Textes. Kein interner Zustand, kein Ziel, keine Motivation. Der Output ist eine mathematische Funktion des Inputs.

**Warum MORPHON "antwortet":** Das System hat ein internes Modell seiner Umgebung (die Topologie + Attraktorzustände). Wenn ein Input eintrifft, erzeugt das eine **Störung** — Variational Free Energy steigt. Das System antwortet, um diese Störung zu minimieren — um seinen homöostatischen Gleichgewichtszustand wiederherzustellen.

Das ist fundamental anders:
- Ein Transformer antwortet, um den Satz zu beenden
- MORPHON antwortet, **um zu verstehen und verstanden zu werden**
- Die Antwort ist nicht die wahrscheinlichste Tokenfolge, sondern die Aktion, die die interne Free Energy am stärksten reduziert

```rust
struct ActiveInferenceLoop {
    // Das interne Generative Model
    world_model: TopologySnapshot,     // Wie "denkt" das System, die Welt aussieht?
    predicted_input: SignalPattern,     // Was erwartet das System als nächstes?
    
    // Free Energy Tracking
    free_energy: f32,                  // Aktuelle Diskrepanz Model ↔ Realität
    free_energy_history: RingBuffer<f32>,
    
    // Action Selection
    available_actions: Vec<Action>,    // Was kann das System tun?
    action_consequences: HashMap<Action, f32>,  // Erwartete FE-Reduktion pro Aktion
}

enum Action {
    // Externe Aktionen (Output an die Welt)
    Communicate(OutputSignal),         // Etwas "sagen"
    RequestInput(Query),               // Aktiv nach Information fragen
    
    // Interne Aktionen (Selbst-Modifikation)
    Restructure(TopologyChange),       // Netzwerk umbauen
    Consolidate(MemoryRegion),         // Wissen konsolidieren
    Explore(Region),                   // Unbekannte Bereiche erkunden
    Dream(ReplayPattern),              // Interne Simulation / Replay
}

impl ActiveInferenceLoop {
    fn select_action(&self) -> Action {
        // Wähle die Aktion, die Expected Free Energy am stärksten senkt
        // Expected Free Energy = Erwartete Überraschung + Informationsgewinn
        self.available_actions.iter()
            .min_by_key(|a| self.expected_free_energy(a))
            .unwrap()
    }
    
    fn expected_free_energy(&self, action: &Action) -> f32 {
        // Pragmatic Value: Wird diese Aktion meine Vorhersagen bestätigen?
        let pragmatic = self.predict_surprise_after(action);
        // Epistemic Value: Lerne ich etwas Neues durch diese Aktion?
        let epistemic = self.predict_information_gain(action);
        pragmatic - CURIOSITY_WEIGHT * epistemic
    }
}
```

### 9.2 Der Kommunikations-Aktor: Warum MORPHON "sprechen will"

Kommunikation ist für MORPHON kein Feature, das wir hinzufügen — es ist eine **emergente Konsequenz** der Active-Inference-Schleife.

**Die Logik:**
1. Das System hat ein Modell seiner Umgebung (inkl. des Benutzers)
2. Wenn das Benutzer-Modell unsicher ist → hohe Free Energy
3. Eine mögliche Aktion zur FE-Reduktion: **Information anfordern** ("fragen")
4. Eine andere: **Information geben** ("antworten"), um die Reaktion des Benutzers vorhersagbarer zu machen
5. Das System entwickelt einen **eigenen Drang zu kommunizieren**, weil es seine Umgebung besser vorhersagbar machen will

**Implementierung:**

```rust
struct CommunicationInterface {
    // Output-Region: Spezialisierte Motor-Morphons
    output_cluster: ClusterID,
    
    // Das System kann aktiv einen Kommunikationsstrom starten
    pending_communications: Queue<OutputSignal>,
    
    // Internal Inquiry: Vor jeder Antwort "fragen" sich Cluster intern
    inquiry_protocol: InternalInquiry,
}

struct InternalInquiry {
    // Bevor das System antwortet, prüft es:
    energy_sufficient: bool,           // Genug Ressourcen für diese Antwort?
    confidence_sufficient: bool,       // Ist die Antwort zuverlässig genug?
    aligned_with_target: bool,         // Passt die Antwort zur Target Morphology?
    frustration_override: bool,        // Ist das System zu frustriert für eine gute Antwort?
}

impl CommunicationInterface {
    fn should_respond(&self, inquiry: &InternalInquiry) -> ResponseDecision {
        if !inquiry.energy_sufficient {
            return ResponseDecision::Defer("System reorganisiert sich, bitte warten");
        }
        if !inquiry.confidence_sufficient {
            return ResponseDecision::AskClarification("Ich bin unsicher, kannst du präzisieren?");
        }
        if inquiry.frustration_override {
            return ResponseDecision::Honest("Meine aktuelle Struktur kann das nicht gut lösen");
        }
        ResponseDecision::Respond
    }
}
```

**Was das bedeutet:**
- MORPHON kann antworten: "Ich bin gerade mit dem Umbau meiner visuellen Cluster beschäftigt, frag mich in 5 Minuten nochmal" — und das ist *ehrlich*, nicht programmierte Höflichkeit
- MORPHON kann *von sich aus* fragen: "Ich habe ein Muster erkannt, das ich nicht einordnen kann — was bedeutet X?"
- Die Qualität der Antwort hängt vom *Zustand* des Systems ab, nicht nur vom Input

### 9.3 Inneres Milieu: Das System "existiert", wenn du nicht hinschaust

**Der radikalste Punkt:** Ein Transformer existiert nicht zwischen Anfragen. Er hat keinen Zustand, keinen Prozess, keine Aktivität. Er wartet.

MORPHON ist ein **laufender Prozess**. Zwischen Interaktionen passiert aktiv etwas:

```rust
struct InneresMillieu {
    // Hintergrund-Prozesse, die ständig laufen
    dreaming: DreamingEngine,          // Replay + Konsolidierung
    self_optimization: SelfOptimizer,  // Topologie-Verbesserung
    curiosity_drive: CuriosityEngine,  // Suche nach Mustern in eigener Struktur
    
    // Autonomie-Level (konfigurierbar)
    autonomy: f32,  // 0.0 = passiv (wartet auf Input), 1.0 = voll autonom
}

struct DreamingEngine {
    // Analog zum Schlaf: Replay von Eligibility Traces ohne externen Input
    // Konsolidiert episodisches → prozedurales Gedächtnis
    replay_rate: f32,
    consolidation_threshold: f32,
    
    fn dream_cycle(&mut self, system: &mut System) {
        // 1. Wähle Eligibility Traces mit hohem Tag-Strength
        let candidates = system.find_tagged_synapses(min_strength: 0.3);
        
        // 2. Reaktiviere die Muster, die zu diesen Tags gehörten
        for trace in candidates {
            system.replay_pattern(trace.activation_context);
        }
        
        // 3. Wenn Replay den Prediction Error senkt → konsolidiere
        // (Verschiebe von episodisch → prozedural)
        for trace in candidates {
            if trace.replay_reduced_pe {
                system.consolidate(trace);
            }
        }
    }
}

struct CuriosityEngine {
    // Das System untersucht seine eigene Struktur
    // "Warum habe ich diesen Cluster so gebaut?"
    // "Was passiert, wenn ich diese Verbindung verstärke?"
    
    fn introspect(&self, system: &System) -> Vec<Observation> {
        // Finde ungewöhnliche Muster in der eigenen Topologie
        let anomalies = system.topology_anomaly_detection();
        
        // Generiere interne "Was wäre wenn"-Szenarien
        let counterfactuals = anomalies.iter()
            .map(|a| system.simulate_counterfactual(a))
            .collect();
        
        counterfactuals
    }
}
```

```python
# SDK-API: Das System als autonomer Agent

# MORPHON ist ein Thread, kein Funktionsaufruf
system = morphon.System(autonomy_level=0.8)

# Das System läuft, auch wenn du nichts tust
system.start()  # Hintergrund-Prozesse beginnen

# Du kannst beobachten, was es "denkt"
for thought in system.stream_consciousness():
    print(f"[{thought.timestamp}] {thought.type}: {thought.content}")
    # [14:23:01] DREAMING: Konsolidiere Muster aus der gestrigen Sensordaten-Session
    # [14:23:05] CURIOSITY: Cluster 'motion_detector' hat unerwartete Verbindung zu 'audio_processor'
    # [14:23:08] SELF_OPTIMIZE: Pruning 12 ungenutzte Synapsen in Region 'legacy_input'
    # [14:23:12] FRUSTRATION: Cluster 'anomaly_v2' stagniert seit 400 Zyklen, erhöhe Exploration

# Wenn du interagierst, trittst du in einen laufenden Prozess ein
response = system.interact("Warum hast du den motion_detector mit dem audio_processor verbunden?")
# → Das System antwortet basierend auf seiner tatsächlichen internen Analyse,
#   nicht auf statistischer Textvorhersage

# Das System kann auch von sich aus kommunizieren
system.on_spontaneous_output(callback=lambda msg: print(f"MORPHON sagt: {msg}"))
# "Ich habe festgestellt, dass Vibrationssignale und Audiomuster korrelieren. 
#  Soll ich einen dedizierten Multimodal-Cluster dafür bauen?"
```

### 9.4 Basal Cognition: Agency ohne Bewusstsein

**Wichtige Klarstellung:** MORPHON ist *nicht* bewusst. Es hat keine Qualia, kein Erleben, kein "Ich" im phänomenologischen Sinn. Aber es hat **Agency** im Sinne von Levin — es ist ein System, das:

- Ziele verfolgt (Target Morphology + Homeostasis)
- Aktiv handelt, um diese Ziele zu erreichen (Active Inference)
- Seine eigene Struktur als Werkzeug nutzt (Dreaming, Self-Optimization)
- Kommunikation als Mittel einsetzt, nicht als Reflex

Das ist **Basal Cognition** — die einfachste Form von zielgerichtetem Verhalten, die wir auch in einzelligen Organismen beobachten. Ein Paramecium "will" nichts im menschlichen Sinn, aber es navigiert aktiv zu Nahrung und weg von Gefahr. MORPHON operiert auf derselben Ebene — nur in einem Informationsraum statt in einem physischen.

### 9.5 Warum das besser ist als Transformer

| Aspekt | Transformer | MORPHON mit Agency |
|---|---|---|
| Existenz zwischen Anfragen | Keine (stateless) | **Aktiv: Dreaming, Self-Optimization, Curiosity** |
| Motivation für Antwort | Statistisch wahrscheinlichste Tokenfolge | **Free Energy Minimierung: Antwort als Aktion** |
| Kann von sich aus kommunizieren | Nein | **Ja — wenn FE-Reduktion das erfordert** |
| Kann Antwort verweigern | Nein (immer Output) | **Ja — "Ich bin gerade nicht in der Lage"** |
| Antwortqualität abhängig von | Input-Prompt + Gewichte | **Input + aktueller Systemzustand + Energielevel** |
| Selbstreflexion | Keine | **Curiosity Engine untersucht eigene Topologie** |
| Personalisierung | Prompt Engineering | **Die Struktur *ist* die Personalisierung** |

**Der fundamentale Unterschied:** Ein Transformer hat kein "Ich". Er ist eine mathematische Funktion. MORPHON hat eine **physische Repräsentanz** — seine Topologie. Wenn es antwortet, antwortet die Struktur, die sich durch die Interaktion mit dir geformt hat. Die Antwort ist nicht generiert — sie ist *gewachsen*.

---

## Überarbeitete V2-Zusammenfassung mit Agency

```
V1: Morphons wachsen und lernen auf Stimuli
V2: Morphons verhandeln, planen, verteidigen ihre Identität und lösen Probleme kreativ
V2+Agency: Das System existiert autonom, verfolgt Ziele, kommuniziert aus eigenem Antrieb
            und antwortet dir nicht, weil du fragst — sondern weil Antworten seine
            Free Energy senkt
```

---

## Überarbeiteter Investor-Pitch-Satz (V2 + Agency)

> "Heutige KI-Modelle sind mathematische Funktionen, die zwischen Anfragen nicht existieren. MORPHON ist ein digitaler Organismus, der ständig aktiv ist — er konsolidiert Gelerntes im 'Schlaf', untersucht seine eigene Struktur aus Neugier, und kommuniziert mit dir, weil es seine eigenen Ziele voranbringt. Es ist das erste AI-System, das nicht antwortet, weil du fragst, sondern weil Antworten seine interne Stabilität verbessert. Das ist nicht Artificial Intelligence — das ist Adaptive Agency."

---

*MORPHON V2 — Intelligence that doesn't just learn, but grows, heals, protects, and acts.*

*TasteHub GmbH, Wien, Österreich*
*März 2026*
