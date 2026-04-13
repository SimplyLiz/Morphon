//! # MORPHON — Morphogenic Intelligence Engine
//!
//! A fundamentally new AI architecture inspired by biological neuroplasticity,
//! morphogenetic self-organization, and three-factor learning rules.
//!
//! Instead of static neural networks trained with backpropagation, MORPHON
//! creates systems that grow, self-organize, and adapt their own architecture
//! at runtime — without retraining, without fixed architecture, without cloud
//! dependency.
//!
//! ## Core Concepts
//!
//! - **Morphon**: Autonomous compute unit (not a classical neuron) with identity,
//!   internal state, and lifecycle (division, differentiation, fusion, death)
//! - **Resonance**: Local topology-based signal propagation (O(k·N) vs O(N²))
//! - **Three-Factor Learning**: Eligibility traces modulated by neuromodulatory
//!   broadcast signals (replaces backpropagation)
//! - **Morphogenesis**: The network grows, prunes, and reorganizes at runtime
//! - **Triple Memory**: Working (attractors), Episodic (one-shot), Procedural (topology)
//!
//! ## Quick Start
//!
//! ```rust
//! use morphon_core::system::{System, SystemConfig};
//!
//! let config = SystemConfig::default();
//! let mut system = System::new(config);
//!
//! // Feed input and get output (with continuous learning)
//! let output = system.process(&[1.0, 0.5, 0.3]);
//!
//! // Inject neuromodulatory signals
//! system.inject_reward(0.8);
//! system.inject_novelty(0.6);
//!
//! // Inspect the system's state
//! let stats = system.inspect();
//! println!("Morphons: {}", stats.total_morphons);
//! println!("Clusters: {}", stats.fused_clusters);
//! ```

pub mod types;
pub mod ancs;
pub mod hot_arrays;
pub mod morphon;
pub mod neuromodulation;
pub mod topology;
pub mod learning;
pub mod resonance;
pub mod morphogenesis;
pub mod memory;
pub mod developmental;
pub mod homeostasis;
pub mod scheduler;
pub mod system;
pub mod snapshot;
pub mod lineage;
pub mod diagnostics;
pub mod field;
pub mod justification;
pub mod governance;
pub mod epistemic;
pub mod endoquilibrium;
pub mod limbic;
#[cfg(feature = "python")]
pub mod python;
#[cfg(feature = "wasm")]
pub mod wasm;

// Re-export key types for convenience
pub use system::{System, SystemConfig, SystemStats};
pub use types::{
    CellType, DevelopmentalProgram, LifecycleConfig, ModulatorType, MorphonId,
};
pub use developmental::{DevelopmentalConfig, RecurrentConfig, TargetMorphology, TargetRegion};
pub use lineage::LineageTree;
pub use diagnostics::Diagnostics;
pub use field::{FieldConfig, FieldType, MorphonField};
pub use neuromodulation::Neuromodulation;
pub use governance::ConstitutionalConstraints;
pub use epistemic::{EpistemicState, EpistemicHistory};
pub use justification::{FormationCause, SynapticJustification};
pub use endoquilibrium::{Endoquilibrium, EndoConfig, ChannelState, DevelopmentalStage};
pub use ancs::{
    AncsConfig, InMemoryBackend, MemoryBackend, MemoryEpistemicState, MemoryItem,
    MemoryTier, PressureMode, SystemHeartbeat, classify_tier, compute_importance,
};
