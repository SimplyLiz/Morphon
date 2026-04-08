# Knowledge Hypergraph — Persistent Memory Substrate

## Status: Deferred (post ANCS-Core Phases 0-4)

This is the optional persistent, queryable knowledge layer for Morphon. Extracted from the ANCS integration plan because it's a separate concern from the core memory backend work (trait abstraction, importance scoring, reconsolidation, fused retrieval).

Build this when:
- Phases 0-4 of ANCS-Core are validated and shipping
- There's a concrete need for persistence across restarts
- Multi-instance (multi-robot) knowledge sharing is on the roadmap
- Auditability / bi-temporal queries are required (e.g. "what did the system know at step 5000?")

---

## The Problem

Morphon's topology (petgraph) tracks morphon connectivity. Justification records on synapses track *why* connections exist. But there's no queryable graph of *knowledge entities and their relations* — no way to ask "which clusters depend on sensor X?" or "show me all contested knowledge about grip force."

`snapshot.rs` does full-state JSON dumps but has no incremental persistence, no bi-temporal history, and no ability to query across time.

---

## Architecture

A second graph structure alongside petgraph, purpose-built for knowledge entities rather than morphon connectivity.

```
petgraph (topology.rs)          KnowledgeGraph (knowledge.rs)
─────────────────────           ────────────────────────────
Morphon → Morphon edges         Entity → Entity hyperedges
Synapse weights + learning      Confidence + epistemic state
Structural connectivity         Semantic relations
In-memory only                  Optional persistence (redb)
Single-writer (System::step)    Concurrent reads (DashMap)
```

### Why not just extend petgraph?

petgraph edges are binary (pre → post). Knowledge relations are n-ary (hyperedges with role-labeled participants: agent, patient, instrument, temporal, causal). Forcing hyperedges into a binary graph requires intermediate "relation nodes" — awkward and slow to traverse.

---

## Data Model

### Entities

```rust
pub struct Entity {
    pub id: EntityId,              // ULID
    pub canonical_label: String,
    pub aliases: Vec<String>,
    pub entity_type: EntityType,   // Sensor, Actuator, Concept, Cluster, External
    pub embedding: Option<Vec<f32>>,
    pub memory_items: Vec<MemoryItemId>,  // back-refs to ANCS-Core memory items
    pub morphon_refs: Vec<MorphonId>,     // which morphons represent this entity
    pub created_at: u64,           // step
    pub last_accessed: u64,
}

pub enum EntityType {
    Sensor,       // maps to Sensory morphons
    Actuator,     // maps to Motor morphons
    Concept,      // learned abstraction (Associative clusters)
    Cluster,      // direct ref to a Morphon cluster
    External,     // external source (API, sensor hardware, document)
}
```

### Hyperedges

```rust
pub struct Hyperedge {
    pub id: HyperedgeId,
    pub participants: Vec<(EntityId, Role)>,
    pub edge_type: EdgeType,
    pub confidence: f64,
    pub epistemic: MemoryEpistemicState,
    pub valid_from: u64,           // step when relation became true
    pub valid_to: Option<u64>,     // None = still valid
    pub system_from: u64,          // step when recorded
    pub system_to: Option<u64>,    // None = current record
    pub justification: Option<JustificationId>,
}

pub enum Role {
    Agent,
    Patient,
    Instrument,
    Source,
    Target,
    Temporal,
    Causal,
    Manner,
}

pub enum EdgeType {
    Causal,        // X causes Y
    Temporal,      // X before Y
    Dependency,    // X depends on Y (for TruthKeeper cascade)
    Similarity,    // X is like Y
    Composition,   // X is part of Y
    Justifies,     // X justifies synapse Y
}
```

### Entity Resolution

Cascading resolution: exact label → alias match → fuzzy (trigram or edit distance):

```rust
impl KnowledgeGraph {
    pub fn resolve(&self, surface_form: &str) -> Option<EntityId> {
        // 1. Exact canonical label
        if let Some(ids) = self.label_index.get(surface_form) {
            return Some(ids[0]);
        }
        // 2. Alias match
        for entry in self.entities.iter() {
            if entry.value().aliases.contains(&surface_form.to_string()) {
                return Some(*entry.key());
            }
        }
        // 3. Fuzzy (edit distance < 2)
        // ...
        None
    }
}
```

---

## Storage

### In-Memory (default)

```rust
pub struct KnowledgeGraph {
    entities: DashMap<EntityId, Entity>,
    hyperedges: DashMap<HyperedgeId, Hyperedge>,
    label_index: DashMap<String, Vec<EntityId>>,
    // Bi-temporal: all versions kept, filtered on read
    edge_history: DashMap<HyperedgeId, Vec<Hyperedge>>,
}
```

DashMap for concurrent reads during rayon-parallel morphon updates. Single writer (the memory step) holds no locks during reads.

### Persistent (feature `ancs-persist`)

redb as the storage backend:
- Pure Rust, no C dependencies (unlike RocksDB)
- ACID transactions
- mmap-based reads (zero-copy when possible)
- Fits the "one binary, no external services" philosophy

```rust
// Feature-gated persistence
#[cfg(feature = "ancs-persist")]
pub struct PersistentKnowledgeGraph {
    memory: KnowledgeGraph,          // hot data in DashMap
    db: redb::Database,              // cold data on disk
    sync_interval: u64,              // flush to disk every N steps
}
```

Write-behind: DashMap is always authoritative. redb is flushed periodically (every `sync_interval` steps, default 1000). On startup, redb hydrates the DashMap.

### Why not rkyv / zero-copy for everything?

rkyv is great for serialization but adds complexity to the data model (archived types, alignment constraints). For the hypergraph, the access pattern is random lookups by ID, not sequential scans. DashMap + redb covers this well. rkyv would matter if we were bulk-scanning millions of entities — unlikely at Morphon's current scale (hundreds of morphons, not millions).

---

## Integration with Morphon

### Automatic Entity Creation

When developmental programs create morphons:
- Each Sensory morphon → Sensor entity
- Each Motor morphon → Actuator entity
- Each Cluster → Cluster entity
- External inputs (if configured) → External entity

### Justification → Hyperedge Binding

When a synapse consolidates with a justification:
```rust
// In learning.rs, on capture:
if let Some(ref justification) = synapse.justification {
    knowledge_graph.add_edge(Hyperedge {
        participants: vec![
            (pre_entity, Role::Source),
            (post_entity, Role::Target),
        ],
        edge_type: EdgeType::Justifies,
        confidence: synapse.consolidation_level,
        epistemic: MemoryEpistemicState::Hypothesis,
        justification: Some(justification.id),
        // ...
    });
}
```

### TruthKeeper Cascade via Hyperedge Traversal

When an External entity's source changes:
1. Find all `EdgeType::Dependency` edges from that entity
2. Mark targets Stale
3. Recursively walk `Dependency` edges → mark transitive targets Suspect
4. Feed back into ANCS-Core reconsolidation (Phase 2 of main plan)

This is `compute_blast_radius()` from ANCS, but as a graph traversal in DashMap instead of a recursive CTE in PostgreSQL.

### Bi-Temporal Queries

```rust
impl KnowledgeGraph {
    /// What did we know at step N?
    pub fn at_step(&self, step: u64) -> TemporalView<'_> {
        TemporalView { graph: self, point_in_time: step }
    }
}

impl<'a> TemporalView<'a> {
    pub fn edges_for(&self, entity: EntityId) -> Vec<&Hyperedge> {
        self.graph.edge_history.get(&entity)
            .map(|versions| versions.iter()
                .filter(|e| e.valid_from <= self.point_in_time
                    && e.valid_to.map_or(true, |t| t > self.point_in_time))
                .collect())
            .unwrap_or_default()
    }
}
```

This enables counterfactual analysis: "what if we rolled back to step 5000 and replayed?"

---

## Dependencies

```toml
[dependencies]
dashmap = { version = "6", optional = true }
redb = { version = "2", optional = true }

[features]
knowledge-graph = ["dashmap"]
knowledge-persist = ["knowledge-graph", "redb"]
```

Zero impact on default build. `cargo build` produces the same binary as today.

---

## Relationship to ANCS-Core Plan

| ANCS-Core (main plan) | Knowledge Hypergraph (this doc) |
|---|---|
| `trait MemoryBackend` | Consumes memory items, creates entity back-refs |
| AXION importance scoring | Hyperedge confidence feeds into importance factor f3 (centrality) |
| TruthKeeper reconsolidation | Cascade traversal provides blast radius computation |
| RRF fused retrieval | Entity resolution provides an additional ranking signal |
| VBC-lite tier classification | Entity type influences tier routing (External sources → T0) |

The hypergraph is a **consumer** of ANCS-Core's memory items, not a replacement. It adds queryability and persistence on top of the in-process memory backend.

---

## What This Enables (Future)

- **Multi-instance sync**: two Morphon instances share a knowledge graph via CRDT merge on the DashMap layer. Each instance has its own namespace. Conflict → Contested epistemic state → TruthKeeper arbitration.
- **AXION wire protocol**: export a subgraph as AXION-encoded T2 payload for inter-system transfer. Peer-discount on import (confidence halved).
- **Audit queries**: "Show all justified synapses that depend on sensor_torque_3" — traverses Justifies + Dependency edges.
- **Counterfactual replay**: clone system state at step N using bi-temporal view, run alternative scenario, compare outcomes.

None of these are needed for Phase 1 benchmarks. They're the path from "research prototype" to "deployable system."
