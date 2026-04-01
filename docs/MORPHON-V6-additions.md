# MORPHON V6 — Cross-Pollination: Ingestible & CCE Patterns
## Konzeptuelle Transfers aus Document Intelligence und Context Compression
### TasteHub GmbH, April 2026

---

## Kontext

Zwei bestehende TasteHub-Projekte — **Ingestible** (Document Ingestion Pipeline) und **Context Compression Engine** (CCE, lossless LLM-Kontext-Kompression) — lösen Probleme in der LLM-Welt, deren zugrundeliegende Muster auf MORPHONs neuromorphe Architektur übertragbar sind.

Dieses Dokument extrahiert fünf Konzepte, die MORPHONs Lernfähigkeit, Gedächtnisabfrage und Ressourceneffizienz direkt verbessern.

```
Ingestible:  Dokument → hierarchische Chunks → Multi-Index-Suche → RRF-Fusion
CCE:         Nachrichten → Klassifikation → Deduplizierung → Importance Scoring → Kompression

MORPHON:     Stimulus → Morphon-Netzwerk → Lernen → Morphogenese → Verhalten
```

---

## 1. Importance-Aware Pruning via Forward-Reference Density

**Quelle:** CCE — `src/importance.ts`

**Problem in MORPHON:** Synapse-Pruning nutzt derzeit `usage_count` und `age` — beides rückwärtsgerichtete Metriken. Eine Synapse, die selten feuert aber downstream kritische Outputs ermöglicht, wird fälschlich als unwichtig eingestuft.

**CCE-Konzept:** Messages werden nach *Forward-Reference Density* bewertet — wie viele spätere Messages dieselben Entitäten referenzieren. Items mit hoher Forward-Reference widerstehen der Kompression.

**Transfer auf MORPHON:** Synapsen nicht nur danach bewerten, wie oft sie gefeuert haben, sondern danach, wie oft ihre post-synaptischen Targets nützlichen Output produziert haben. Das ist im Kern ein Credit-Assignment-Signal für Pruning:

```
importance(synapse) = α * usage_count / age           // bisherig: backward-looking
                    + β * downstream_reward_correlation // neu: forward-looking
```

`downstream_reward_correlation` misst, wie stark das Feuern dieser Synapse mit positivem Reward am Output-Layer korreliert. Existierende Eligibility Traces können als Approximation dienen — Synapsen mit hoher durchschnittlicher `eligibility * reward` über die letzten N Schritte haben hohe Forward-Reference.

**Aufwand:** Gering. Die Daten (`eligibility`, `reward`) existieren bereits. Erfordert ein gleitendes Fenster pro Synapse (~1 f64) und eine Anpassung der Pruning-Heuristik in `morphogenesis.rs`.

---

## 2. Contradiction-Driven Reconsolidation

**Quelle:** CCE — `src/contradiction.ts`

**Problem in MORPHON:** Synaptic Tagging hat ein Zwei-Phasen-Commit (Tag → Consolidation), aber es gibt keinen Mechanismus für *Un-Konsolidierung*. Wenn ein konsolidiertes Gewicht beginnt, konsistent falsche Vorhersagen zu erzeugen, bleibt es bestehen. Frühes Lernen (z.B. schlechte CartPole-Policies aus den ersten Episoden) überlebt, obwohl es durch späteres Lernen widerlegt wird.

**CCE-Konzept:** Contradiction Detection identifiziert, wenn eine spätere Nachricht eine frühere überschreibt ("actually, don't use X, use Y"). Die überschriebene Nachricht wird als *superseded* markiert und ihr Einfluss reduziert.

**Transfer auf MORPHON:** Wenn eine konsolidierte Synapse konsistente Prediction Errors produziert, wird Reconsolidation ausgelöst:

```
Für jede konsolidierte Synapse:
  if avg_prediction_error(post_target, window=100) > θ_reconsolidate:
    consolidated = false           // Konsolidierung aufheben
    tag = current_eligibility      // Neues Tag setzen
    tag_strength *= decay_factor   // Altes Tag abschwächen
    weight *= reconsolidate_factor // Gewicht teilweise zurücksetzen (nicht löschen)
```

Biologisches Vorbild: Gedächtnis-Rekonsolidierung in der Amygdala — abgerufene Erinnerungen werden kurzzeitig labil und können modifiziert werden, bevor sie re-konsolidiert werden (Nader et al. 2000).

**Aufwand:** Mittel. Erfordert ein `prediction_error`-Tracking pro Synapse (oder pro Post-Target-Morphon als Proxy) und einen neuen Reconsolidation-Pfad in `learning.rs`.

**Phase 1 Relevanz:** Direkt nützlich für CartPole — verhindert, dass schlechte Policies aus frühen Episoden in konsolidierten Gewichten eingefroren werden.

---

## 3. Graceful Degradation unter Budget-Druck

**Quelle:** CCE — `createEscalatingSummarizer()` (Three-Level Fallback)

**Problem in MORPHON:** Wenn ein Morphon seine Energie erschöpft, stirbt es (Apoptose). Es gibt keinen Zwischenzustand zwischen "volle Funktionalität" und "Tod".

**CCE-Konzept:** Drei-Level-Fallback bei Kompression: LLM-Summarization → Deterministische Kompression → Hard Truncation. Jede Stufe opfert Qualität, aber das System bleibt funktional.

**Transfer auf MORPHON:** Drei Energie-Betriebsmodi pro Morphon:

```
energy > 0.5:  Full Mode
  → Drei-Faktor-Lernen (Hebbian × Eligibility × Modulation)
  → Volle Morphogenese-Teilnahme
  → Normale Feuerrate

0.2 < energy ≤ 0.5:  Degraded Mode
  → Vereinfachtes Hebbian-Lernen (kein Modulations-Term)
  → Keine Morphogenese (kein Division, kein Migration)
  → Reduzierte Feuerrate (erhöhter Threshold)

energy ≤ 0.2:  Survival Mode
  → Kein Lernen
  → Kein Feuern (silent, aber alive)
  → Energie-Regeneration priorisiert
  → Apoptose nur bei energy = 0
```

Vorteil: Morphons, die temporär unter Druck stehen, überleben und können sich erholen, statt unwiderruflich zu sterben. Das Netzwerk verliert weniger strukturelles Wissen.

**Aufwand:** Gering bis mittel. Der `step()`-Pfad in `system.rs` muss die Energielevel abfragen und den Lern-/Feuer-Pfad entsprechend verzweigen.

---

## 4. Multi-Index Memory Retrieval mit Rank Fusion

**Quelle:** Ingestible — `search/hybrid.py` (Reciprocal Rank Fusion)

**Problem in MORPHON:** `TripleMemory` hat drei Speicher (Working, Episodic, Semantic/Triple Store), die unabhängig abgefragt werden. Es gibt keine Fusion der Ergebnisse — das System muss sich für *einen* Speicher entscheiden oder alle sequenziell abfragen.

**Ingestible-Konzept:** Drei unabhängige Suchindizes (Vector, BM25, Concept) liefern Rankings, die via Reciprocal Rank Fusion (RRF) kombiniert werden:

```
RRF_score(item) = Σ  1 / (k + rank_in_index_i)
                  i
```

Items, die in mehreren Indizes auftauchen, erhalten höhere Scores — ohne Score-Normalisierung.

**Transfer auf MORPHON:** Bei Memory Retrieval (z.B. für kontextuelle Modulation oder Pattern Completion):

```
fn retrieve_fused(&self, query: &[f64], top_k: usize) -> Vec<(MemoryItem, f64)> {
    let working  = self.working_memory.rank(query);   // Rang nach Recency
    let episodic = self.episodic_memory.rank(query);   // Rang nach Similarity
    let triples  = self.triple_store.rank(query);      // Rang nach Relational Match

    reciprocal_rank_fusion(vec![working, episodic, triples], k=60, top_k)
}
```

Wenn dasselbe Pattern in Working Memory (kürzlich gesehen), Episodic Memory (ähnlich zu etwas Gelerntem) *und* Triple Store (relational verbunden) auftaucht, ist es mit hoher Wahrscheinlichkeit relevant.

**Aufwand:** Gering. RRF ist ~20 Zeilen Rust. Erfordert, dass jeder Memory-Store eine `rank()`-Methode exponiert, die eine sortierte Liste zurückgibt.

**Phase 1 Relevanz:** Direkt nützlich — verbessert die Qualität der assoziativen Erinnerungsabfrage, was die Lerneffizienz bei CartPole und MNIST steigern sollte.

---

## 5. Auto-Merge als Cluster-Fusion-Signal

**Quelle:** Ingestible — `search/auto_merge.py`

**Problem in MORPHON:** Cluster Fusion basiert auf Co-Activation-Korrelation über Zeit. Das funktioniert, hat aber eine lange Anlaufzeit — Morphons müssen viele Male gemeinsam feuern, bevor die Korrelation den Fusion-Threshold erreicht.

**Ingestible-Konzept:** Auto-Merge: Wenn 3+ Kind-Chunks aus demselben Eltern-Abschnitt alle auf dieselbe Query matchen, wird statt der Kinder der Eltern-Chunk zurückgegeben. Das ist ein instantaner Bottom-Up-Abstraktions-Trigger.

**Transfer auf MORPHON:** Wenn N Morphons in räumlicher Nähe *alle* auf dasselbe Input-Pattern reagieren, ist das ein sofortiges Fusion-Signal — sie kodieren redundante Information:

```
Für jedes Input-Pattern:
  responding = morphons die gefeuert haben UND spatial_distance < θ_proximity
  if |responding| ≥ auto_merge_threshold (z.B. 3):
    → Fusion-Kandidat registrieren
    → Wenn innerhalb von M Schritten erneut getriggert: Fusion ausführen
```

Der Zwei-Schritt-Trigger (erst registrieren, dann bestätigen) verhindert voreilige Fusion bei zufälligen Co-Aktivierungen.

**Aufwand:** Mittel. Erfordert ein Spatial-Proximity-Tracking pro Schritt und einen Fusion-Kandidaten-Buffer in `morphogenesis.rs`.

---

## Priorisierung für Phase 1

| # | Konzept | Aufwand | Phase 1 Impact | Priorität |
|---|---------|---------|----------------|-----------|
| 2 | Contradiction-Driven Reconsolidation | Mittel | Hoch (CartPole) | **P0** |
| 4 | Multi-Index Memory Retrieval (RRF) | Gering | Hoch (beide Benchmarks) | **P0** |
| 1 | Importance-Aware Pruning | Gering | Mittel | P1 |
| 3 | Graceful Degradation | Gering-Mittel | Mittel | P1 |
| 5 | Auto-Merge Fusion Signal | Mittel | Niedrig | P2 |

---

## Quellen

- **Ingestible** — TasteHub GmbH, `github.com/SimplyLiz/Ingestible` (PolyForm Small Business License)
- **Context Compression Engine** — TasteHub GmbH, `github.com/SimplyLiz/ContextCompressionEngine` (AGPL-3.0 / Commercial)
- Nader, K., Schafe, G. E., & LeDoux, J. E. (2000). Fear memories require protein synthesis in the amygdala for reconsolidation after retrieval. *Nature*, 406(6797), 722-726.
- Frey, U., & Morris, R. G. (1997). Synaptic tagging and long-term potentiation. *Nature*, 385(6616), 533-536.
