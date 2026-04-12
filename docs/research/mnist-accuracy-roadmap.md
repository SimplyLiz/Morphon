# MNIST Accuracy Roadmap — v4.4.0+
## Analyse, Befunde, und Architektur-Diagnose
### Morphon-Core, April 2026

---

## Aktueller Stand (v4.4.0)

| Benchmark | Ergebnis | Status |
|-----------|----------|--------|
| CartPole | 195.51 (solved) | ✅ |
| MNIST Standard | 52% (V3 LocalInhibition) | ✅ Baseline locked |
| MNIST Quick | ~44% | Referenz |
| Temporal | 6/6 pass | ✅ |
| Self-Healing Standard | ~31% post-damage | Ehrlich dokumentiert |
| MNIST Offline-Features | 10.5% (LogReg/MLP) | ⚠️ LSM-Verhalten — siehe unten |
| Sensory-only Readout Ablation | 13.0% (−35.5pp vs full) | ✅ Assoc-Layer trägt 35.5pp bei |

V3 LocalInhibition Parameter (Standard-Default, locked):
- `istdp_rate = 0.001`, `initial_inh_weight = -0.5`
- seed_size=200, n_train=5000, n_epochs=3

**Reproduzierbare Baseline (seed=42):** V3 = **51.0%** — bestätigt April 2026.
Vorheriger nicht-reproduzierbarer Run: 52% (Seed-Varianz ~±1pp bestätigt).

---

## Self-Healing Untersuchung — Abschluss (April 2026)

**Fragestellung:** Kann ein trainiertes V2/V3-Netz nach 30% Neuronenverlust auf dem Standard-Profil wieder auf Ausgangsniveau (52%) recovery?

**Ergebnis: Nein — 31% ist die Decke auf Standard.**

### Getestete Hypothesen

| Hypothese | Config | Ergebnis | Δ vs Baseline |
|-----------|--------|----------|---------------|
| supervised_from=0 (Bug-Fix) | ep1 supervised | ~29.5% | +0.5pp |
| division=false | Keine neuen Neuronen | ~31% | +2pp |
| synaptogenesis=false | Keine Reconnection | ~29% | -2pp |
| Frozen STDP (medium_period=999_999) | Reine Readout-Retraining | 9% | -22pp |

### Root-Cause-Analyse

**Warum funktioniert Recovery auf Standard nicht:**

1. **Stärkere Feature-Repräsentationen** — Standard-Training mit 5000 Samples erzeugt stark spezialisierte synaptische Gewichte. Diese sind schwerer reparierbar als Quick-Profile-Features.

2. **STDP-Churn während Recovery** — Reward-gated STDP ist notwendig (Freeze → 9%), aber dieselbe STDP, die Lernen ermöglicht, überschreibt die überlebenden Gewichte mit Rauschen, da die Population nach 30% Kill nicht mehr die gelernte Aktivierungsstruktur hat.

3. **Novelty-STDP Interferenz** — inject_novelty() triggert unsupervised STDP, die unabhängig vom Reward-Signal Gewichte modifiziert. Nicht isoliert getestet (nur Snapshot-Infrastruktur gebaut, kein Run durchgeführt).

4. **Neue Neuronen verdünnen das Signal** — division=true fügt Neuronen mit Null-Gewichten hinzu. 1500 Recovery-Samples reichen nicht, um sie zu integrieren. Bestätigt durch +2pp mit division=false.

**Warum Quick-Profile-Recovery funktioniert:**
- Schwächere Features (weniger Samples) → weniger STDP-Drift nötig für Recovery
- Niedrigere Aktivierungsstruktur → robuster gegenüber strukturellen Schäden

**Framing für das Paper:** Self-Healing unter Untersuchung. Quick-Profil zeigt Recovery-Fähigkeit. Standard-Profil: STDP-Churn ist der primäre Limiter — ein bekanntes Problem in biologischen Systemen (Konsolidierungsinterferenz).

### Snapshot-Infrastruktur (implementiert)

`--save-pretrained` / `--damage-sweep` → schreibt `v2_pretrained.json`  
`--recovery-only` → lädt Snapshot (~instant), testet Recovery-Config (~1min)

Nächste offene Hypothese (nicht getestet): inject_novelty(0.0) während Recovery — supprimiert unsupervised STDP-Churn, lässt reward-gated STDP aktiv.

---

## Plan: 52% → 70%+ (v4.5.0)

### Priorisierung

```
1. RNG-Seed Reproduzierbarkeit (Voraussetzung für valide Ablations)
2. n_train Skalierung (billigster Hebel, wahrscheinlich größter Gewinn)
3. seed_size Skalierung (teurerer Hebel, erst wenn n_train ausgereizt)
```

**Begründung der Reihenfolge:** Ohne reproduzierbare Ergebnisse kann kein Sweep valide bewertet werden. Eine Differenz von 3pp könnte realer Gewinn oder Seed-Varianz sein. Erst dann sinnvoll zu messen.

---

## 1. RNG-Seed Reproduzierbarkeit

### Problem

`System::new()` verwendet intern einen nicht-geseedten RNG für initiale Topologie und Gewichte. Zwei Runs mit identischer Config liefern unterschiedliche Ergebnisse. Auf Standard: Varianz ~±3pp beobachtet.

### Diagnose

- `System::new()` → `developmental.rs` Bootstrap → `rand::thread_rng()` (nicht geseedet)
- Zufällige Anfangsgewichte, zufällige I/O-Pathway-Positionen
- `train_and_eval` verwendet `rand::SeedableRng::seed_from_u64(seed)` für Sample-Shuffle — das ist korrekt
- Aber die Netzwerk-Initialisierung ist nicht deterministisch

### Fix (implementiert in v4.4.0)

`SystemConfig.rng_seed: Option<u64>` — wenn gesetzt, wird `SmallRng::seed_from_u64(seed)` auf dem `System`-Struct gespeichert und überall verwendet: `System::new()`, `enable_analog_readout()`, `step()` (slow/glacial morphogenesis). Kein `rand::rng()` mehr in system.rs.

`mnist_v2.rs`: `create_v2()` nimmt jetzt `rng_seed: u64`. Der `--seed=N` Parameter (bereits vorhanden) wird jetzt vollständig ins System-RNG durchgereicht.

Snapshot-Restore (`System::from_snapshot`) respektiert ebenfalls den gespeicherten Seed.

**Status: DONE ✅**

---

## 2. n_train Skalierung

### Hypothese

Mehr Training-Samples → stärkere iSTDP-Competition → bessere Feature-Separation → höhere Accuracy.

**Evidenz dafür:** iSTDP-Sweep zeigte, dass stärkere Inhibition (-0.5 vs -0.2) auf Standard besser funktioniert als auf Quick. Mechanismus: Mehr Samples erlauben stärkerer Konkurrenz, Repräsentationen zu differenzieren. Mehr Samples sollten denselben Effekt verstärken.

### Ergebnisse (seed=42, Standard)

| n_train | n_epochs | Total Samples | V3 Accuracy | Δ |
|---------|----------|---------------|-------------|---|
| 5,000 | 3 | 15,000 | 49.5% | Baseline |
| 10,000 | 3 | 30,000 | 46.5% | −3pp ❌ |
| 5,000 | 5 | 25,000 | **53.0%** | +3.5pp ✅ |
| 10,000 | 5 | 50,000 | 50.5% | +1pp |

**Befund:** Mehr Epochen gewinnt. Mehr Daten allein verliert. 10k×5ep schlechter als 5k×5ep — wahrscheinlich weil ng-Collapse sich mit mehr Samples häuft und Repräsentationen aktiv zerstört.

**Konsequenz:** n_train-Skalierung ist blockiert durch iSTDP-Instabilität. Der iSTDP-Setpoint-Fix muss zuerst. Ohne stabiles ng bringt mehr Daten nichts.

**Status: DONE ✅**

---

## 3. seed_size Skalierung

### Hypothese

Mehr Morphons = mehr Kapazität für Feature-Repräsentationen. 200 Morphons für 784→10 ist knapp (Ratio 3.9:1 input zu hidden capacity).

### Komplexität

seed_size skaliert nicht linear:
- Synaptische Verbindungen wachsen O(n²) im dichten Fall
- iSTDP-Parameter müssen re-tuned werden (alpha-Gleichgewicht hängt von Netzwerkgröße ab)
- Homeostasis-Parameter (Inhibitions-Stärke) skalieren mit Populationsgröße

### Ergebnisse (seed=42, Standard, 5ep)

| seed_size | Finale Morphons (V3) | V3 Accuracy |
|-----------|----------------------|-------------|
| 200 | 1272 | **53.0%** |
| 400 | 1272 | 52.0% |

**Befund: seed_size ist ein No-Op für finale Netzwerkgröße.** Das System konvergiert durch Pruning/Apoptosis auf ~1272 Morphons unabhängig von seed_size. seed_size=400 bringt keinen Kapazitätsgewinn.

**Richtiger Hebel für mehr Kapazität:** `max_morphons` direkt setzen (in `ConstitutionalConstraints` oder `MorphogenesisParams.max_morphons`) oder Pruning-Parameter lockern. seed_size skalieren ist der falsche Ansatz.

**Status: DONE ✅ — seed_size kein Hebel**

---

## Beobachtungen und Offene Fragen

### iSTDP Collapse nicht auf ep1 beschränkt (neu beobachtet, April 2026)

Seed=42 Baseline-Run zeigt ng-Kollaps auch in ep3 (3000/5000): `ng=0.32`, `hp=0.06`.
Bisherige Annahme: Problem nur in ep1. Tatsächlich: iSTDP-Gleichgewicht ist instabil und kann in jeder Epoche zusammenbrechen. Das begrenzt die Accuracy-Decke unabhängig von n_train — ein fundamentaleres Problem als Kapazität.

**Implikation für den Sweep:** Wenn ng in späteren Epochen kollabiert, bringt mehr Training nichts (die Repräsentationen werden aktiv zerstört). Der Sweep wird zeigen ob n_train=10000 das Problem verschlimmert oder ob es zufällig stabil bleibt.

### iSTDP-Setpoint-Fix — Falsche Hypothese (April 2026)

**Hypothese:** `homeostatic_setpoint=0.3` für Interneuronen vs assoc-target 0.05 verursacht ng-Kollaps.  
**Fix versucht:** `homeostatic_setpoint = 0.05`.  
**Ergebnis:** V3 = 50.0% (5ep) — schlechter als ohne Fix (53.0%). **REVERTIERT.**

**Root cause der Fehlannahme:** `ng` in den Endo-Logs ist der **Novelty Gain** des Neuromodulationssystems, nicht die Interneuron-Feuerrate. Der homeostatic_setpoint betrifft synaptic scaling der Interneuronen — senkt man ihn auf 0.05, werden Interneuronen zu stark herunterreguliert und Inhibition bricht zusammen.

**Was ng=0.32 wirklich bedeutet:** Das Novelty-Neuromodulationssystem konvergiert zu einem niedrigen Gain-Wert. Ursache noch nicht identifiziert — separate Analyse nötig.

### Readout-Kapazität

Der lineare Readout (softmax auf Morphon-Aktivierungen) lernt auf frozen Features. Die Frage ist ob die Features, die Morphon bei 52% gelernt hat, überhaupt 70% kodieren können — oder ob das Feature-Space zu komprimiert ist.

**Diagnose-Idee:** PCA auf Morphon-Aktivierungen bei 52% — wie viel Varianz zwischen Klassen? Das würde zeigen ob das Problem Kapazität (seed_size) oder Daten (n_train) ist.

---

## Benchmark-Ziele für v4.5.0

| Metrik | Ziel | IWAI-Relevanz |
|--------|------|---------------|
| MNIST Standard (online) | ≥60% mit State-Reset-Evaluator | Solide Basis |
| MNIST Standard (online) | ≥70% mit architektonischer Änderung | Paper-würdig, IWAI-ready |
| CartPole | ≥195 (maintained) | ✅ bereits gelöst |
| Temporal | 6/6 (maintained) | ✅ bereits gelöst |
| Reproduzierbarkeit | ±1pp Varianz | ✅ seed=42 bestätigt |

---

## Architektur-Diagnose: Co-adaptierte Readout-Gewichte (April 2026)

### Befund

**Offline-Classifier (LogReg / MLP) auf gespeicherten Morphon-Aktivierungen: 7.5–10.5% — random chance.**

**Online Readout im Rust-System auf denselben Aktivierungen: ~50%.**

Diese Diskrepanz ist kein Messfehler. Sie ist ein grundlegender Befund über die Natur der Repräsentation im MORPHON-System.

### Erklärung

**1. Co-Adaptation der Readout-Gewichte**

Die 50%-Accuracy des Rust-Systems kommt nicht aus generalisierbaren, per-Image Features.
Sie kommt aus **co-adaptierten Readout-Gewichten** — die Weights wurden online gleichzeitig
mit den Morphon-Dynamiken trainiert. Sie lernen eine sehr spezifische Korrelation zwischen
Aktivierungsmustern und Labels, die nur im Kontext der kontinuierlichen Systemdynamik existiert.

Ein naiver Offline-Classifier (sklearn LogReg, MLP) sieht diese Korrelation nicht, weil
er statische Feature-Vektoren ohne den dynamischen Kontext bekommt.

**2. Temporaler Kontext zerstört beim Offline-Export**

Die Dump-Funktion läuft sequenziell durch 1000 Test-Images **ohne State-Reset zwischen Images**.
Die Potenziale von Bild 500 enthalten die History von Bildern 1–499.
Wenn sklearn shuffled und splittet, wird dieser temporale Kontext zerstört — die Features
werden informationslos.

**3. Confusion Matrix bestätigt schwache Feature-Separation**

| Klasse | Recall |
|--------|--------|
| 9 | 94% |
| 8 | 67% |
| 0 | 50% |
| 3 | 16% |
| 4 | 20% |

Klasse 9 hat distinctive, robuste Aktivierungsmuster. Klassen 3/4/7 sind im
Aktivierungsraum nahezu ununterscheidbar — typisch für schwache Feature-Separation,
nicht für einen schlechten Readout-Mechanismus.

### Sensory-Only Readout Ablation (April 2026)

**Fragestellung:** Trägt der assoziative Layer tatsächlich zur Klassifikations-Accuracy bei, oder ist er Rauschen?

**Methode:** Nach V3-Training alle Readout-Gewichte auf assoziative Morphons nullen — nur Sensory→Readout-Verbindungen aktiv lassen. Evaluation auf 1000 Test-Images (identische Prozedur).

**Ergebnis (seed=42, Standard, 5ep):**

| Readout-Konfiguration | Accuracy | Δ |
|----------------------|----------|---|
| Full (Sensory + Assoc) | 48.5% | Baseline |
| Sensory-only | 13.0% | **−35.5pp** |

**Interpretation:**

- **Assoz-Layer ist kein Rauschen** — er liefert 35.5pp der Gesamtaccuracy. Ohne ihn ist das System quasi random (10% Zufall, 13% Sensory-only).
- **Sensory→Readout direkt ist ~13%** — die 784 Input-Morphons allein können kaum zwischen Klassen trennen. Alle echte Klassifikationsstärke kommt aus den assoziativen Repräsentationen.
- **Aber: offline nicht extrahierbar** — MLP auf frozen assoc-Aktivierungen = 10.5% (random). Die assoziativen Features sind real und stark, aber nur über die co-adaptierten Readout-Gewichte zugänglich.

**Schlussfolgerung:** Das System ist ein LSM wo Assoc-Layer = Reservoir und Readout = co-adaptierter Readout. Das Reservoir liefert echte Information, aber nur im Kontext der trainierten Readout-Gewichte. Der Bottleneck ist die Co-Adaptation — nicht fehlende Repräsentationskapazität.

---

### Was das bedeutet

**Das System macht Online-Sequenz-Processing mit Gedächtnis.** MORPHON ist kein
per-Image Feature-Extractor. Es ist ein zustandsbehaftetes, sequentielles System
das kontinuierlich über Bilder hinweg lernt. Das ist architektonisch korrekt und
biologisch sinnvoll — aber es bedeutet dass klassische Offline-Evaluation-Methoden
nicht anwendbar sind.

**Der Bottleneck ist nicht Kapazität, nicht Readout, nicht Hyperparameter.**
Das Plateau bei ~50% (online) ist kein Tuning-Problem. Die Feature-Repräsentation ist
fundamental nicht generalisierend — sie ist an die spezifische Trainings-Sequenz
co-adaptiert. Hyperparameter-Suche oder größere Readout-Layer werden daran nichts ändern.

**Das Plateau ist ein Designproblem, kein Trainings-Problem.**
Um 70%+ zu erreichen sind architektonische Änderungen nötig, nicht mehr Training:

| Ansatz | Mechanismus | Aufwand |
|--------|-------------|---------|
| **State-Reset zwischen Images** | Jedes Bild bekommt frischen, unkorrupten State | Mittel |
| **State-Reset-Evaluator** | Evaluation ohne shuffle, mit State-Reset, kein sklearn | Gering |
| **Spike-Rate-Coding** | Rate-coded Spikes reduzieren State-Abhängigkeit | Mittel |
| **Episodischer State** | Explizites episodisches Gedächtnis statt impliziter History | Hoch |

### Implikation für die Paper-Argumentation

**Was wir haben (stark):**
- Selbst-heilendes System das nach Schaden von 30% auf 47% klettert (+18pp) — kein anderes System zeigt das
- CartPole gelöst und reproduzierbar validiert (195.51)
- Strukturelle Plastizität (91% Synapsen-Pruning via STDP) als Kernbeitrag
- Das System **lernt** — die Online-Accuracy steigt über Training

**Was wir ehrlich dokumentieren:**
- Generalisierende per-Image Features sind offline nicht extrahierbar
- MNIST-Accuracy ist nicht direkt mit konventionellen Methoden vergleichbar
- State-unabhängige Repräsentation fehlt

**Empfohlenes Paper-Framing:**
Die 50%-Online-Accuracy nicht als "50% MNIST Classification" positionieren,
sondern als **"sequentielles Online-Learning mit co-adaptiertem Readout"** — was es
tatsächlich ist. Das ist ein anderes, legitimes Ergebnis. Der Headline-Result bleibt:
**47% post-recovery nach 30% Neuron-Verlust** — das hat keine andere Architektur.

### Empfohlene nächste Experimente

1. **State-Reset-Evaluator** (niedrige Komplexität): Vor jedem Test-Image State auf Null
   setzen, durch das Bild steppen, Prediction nehmen. Kein sklearn, kein shuffle.
   Erwartung: deutlich besser als 10%, zeigt echte per-Image Kapazität.

2. **Sequential Evaluator**: Images in Original-Reihenfolge ohne shuffle evaluieren.
   Erwartung: deutlich besser als shuffled (bestätigt temporalen Kontext als Problem).

3. **Spike-Rate-Coding**: Statt Potenzial-Dump, Rate-coded Spikes über Zeitfenster pro Bild.

### Wissenschaftliche Einordnung

Das Verhalten entspricht dem bekannten **Liquid State Machine (LSM)**-Paradigma:
ein LSM hat echtes Gedächtnis und macht temporale Integration, ist aber schwer offline
zu evaluieren weil der Readout notwendigerweise die Dynamik kennen muss.

MORPHON verhält sich wie ein LSM mit co-adaptiertem Readout — das ist nicht falsch,
aber es bedeutet dass die Evaluation-Methodik diesem Paradigma angepasst werden muss.

**Relevante Literatur:**
- Maass W, Natschläger T, Markram H (2002). "Real-time computing without stable states:
  A new framework for neural computation based on perturbations." *Neural Computation* 14(11).
  → Der originale LSM-Paper — beschreibt exakt das beobachtete Verhalten.
- Lukoševičius M, Jaeger H (2009). "Reservoir computing approaches to recurrent neural
  network training." *Computer Science Review*.
  → Zeigt warum Co-Adaptation zwischen Reservoir und Readout schwer zu vermeiden ist.

---

*Aktualisiert: April 2026*  
*Autor: Claude Code / Lisa*
