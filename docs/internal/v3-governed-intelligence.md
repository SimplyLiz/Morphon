# V3 Phase 1: Governed Intelligence — Implementation Notes

## Overview

V3 Phase 1 adds epistemic integrity and self-governance to MORPHON. Three new modules provide constitutional constraints, synaptic provenance tracking, and a four-state epistemic model with scarring.

## New Modules

### `src/governance.rs` — Constitutional Constraints

Hard invariants enforced outside the learning loop. Cannot be modified by the system itself.

**`ConstitutionalConstraints`** (added to `SystemConfig`):
- `max_connectivity_per_morphon: usize` (50) — prevents superhub pathology
- `max_cluster_size_fraction: f64` (0.3) — no cluster > 30% of system
- `max_unverified_fraction: f64` (0.5) — Phase 1 soft limit
- `mandatory_justification_for: Vec<CellType>` — Motor synapses must have provenance
- `cascade_depth_limit: usize` (10) — max invalidation traversal depth (Phase 2)
- `max_fusion_rate_per_epoch: f64` (0.1) — tempo limit
- `max_structural_changes_per_epoch: usize` (50) — total budget
- `energy_floor: f64` (0.0) — minimum energy, prevents total starvation

**Enforcement points:**
- `check_connectivity()` called in `synaptogenesis()` before creating connections
- `check_cluster_size()` called in `fusion()`
- `enforce_energy_floor()` called on fast tick after all morphon steps

### `src/justification.rs` — Synaptic Provenance

Every synapse can track why it was formed and what reinforced it.

**`FormationCause`** enum:
- `HebbianCoincidence { step }` — from synaptogenesis
- `InheritedFromDivision { parent }` — from mitosis
- `ProximityFormation { distance }` — spatial proximity
- `FusionBridge { cluster }` — cluster fusion bridge
- `External { source }` — developmental program / user injection

**`SynapticJustification`**:
- `formation_cause` + `formation_step`
- `reinforcement_history: VecDeque<ReinforcementEvent>` — bounded ring buffer, capacity 16
- Helpers: `last_reinforcement_step()`, `has_reinforcement()`, `record_reinforcement()`

**Integration**: `Synapse.justification: Option<SynapticJustification>` with `#[serde(default)]` for backward compat. `Synapse::new_justified()` constructor available.

### `src/epistemic.rs` — Four-State Epistemic Model + Scarring

**`EpistemicState`** enum (per cluster):
| State | Meaning | Plasticity Effect |
|-------|---------|-------------------|
| `Hypothesis` | Newly formed, unverified | plasticity_rate x1.5 |
| `Supported` | Verified, consolidated | Normal (protected) |
| `Outdated` | Stale evidence (>5000 steps since reinforcement) | Unconsolidate stale synapses |
| `Contested` | Conflicting evidence (>25% minority) | Boost plasticity x1.2, block fusion |

**State transitions** (evaluated on glacial tick via `update_all_clusters()`):
- `Hypothesis -> Supported`: majority of member synapses consolidated with justification, confidence >= required threshold
- `Supported -> Outdated`: no reinforcement within staleness threshold (5000 steps)
- `Any -> Contested`: significant evidence on both sides (minority > 25%)

**`EpistemicHistory`** (scarring):
- Tracks `stale_count`, `contested_count`, `false_positive_count`
- Derives `skepticism: f64` [0, 1]
- Scarred clusters require higher confidence for `Hypothesis -> Supported` (base 0.8 + skepticism * 0.15, max 0.98)
- Counts decay slowly to prevent permanent scarring from early noise

## Modified Modules

### `morphon.rs`
- `Synapse`: added `justification: Option<SynapticJustification>` field
- `MetabolicConfig`: added `cluster_overhead_per_tick` (0.0005), `reward_for_successful_output` (0.05), `reward_for_verification` (0.0, Phase 2 placeholder)

### `morphogenesis.rs`
- `Cluster`: added `epistemic_state`, `epistemic_history` fields
- `synaptogenesis()`: takes `max_connectivity` param, checks governance cap before creating connections
- `step_slow()`: passes connectivity cap through

### `system.rs`
- `SystemConfig`: added `governance: ConstitutionalConstraints`
- Fast tick: cluster overhead deduction + energy floor enforcement
- Glacial tick: `update_all_clusters()` + `apply_epistemic_effects()` after structural changes
- `inject_reward_at()`: credits motor morphon energy via `reward_for_successful_output`

### `topology.rs`
- Added immutable `incoming_synapses()` method (returns `Vec<(MorphonId, &Synapse)>`)

### `diagnostics.rs`
- Added: `epistemic_supported/hypothesis/outdated/contested`, `justified_fraction`, `avg_skepticism`

## Backward Compatibility

All new fields use `#[serde(default)]`. Old snapshots deserialize without changes. Default `ConstitutionalConstraints` are permissive (energy_floor=0.0, connectivity=50) so existing behavior is unchanged.

## Phase 2 Preview

Phase 2 will activate:
- Source Watchers + Cascade Invalidation
- Auditor Network (symbolic rules layer)
- Oracle Interface for constitutional amendments
- `reward_for_verification` energy flow
- `max_unverified_fraction` enforcement
