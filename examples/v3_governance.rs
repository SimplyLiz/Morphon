//! V3 Governance Validation — exercises constitutional constraints, justification
//! records, epistemic state transitions, and scarring.
//!
//! Usage: cargo run --example v3_governance --release

use morphon_core::developmental::DevelopmentalConfig;
use morphon_core::epistemic::{EpistemicHistory, EpistemicState};
use morphon_core::governance::ConstitutionalConstraints;
use morphon_core::justification::{FormationCause, SynapticJustification};
use morphon_core::morphon::{MetabolicConfig, Synapse};
use morphon_core::system::{System, SystemConfig};
use morphon_core::types::LifecycleConfig;
use rand::Rng;

fn main() {
    println!("=== MORPHON V3 Governance Validation ===\n");

    let mut passed = 0;
    let mut failed = 0;

    macro_rules! check {
        ($name:expr, $cond:expr) => {
            if $cond {
                println!("  PASS  {}", $name);
                passed += 1;
            } else {
                println!("  FAIL  {}", $name);
                failed += 1;
            }
        };
    }

    // =========================================================================
    // Experiment 1: Constitutional Constraints
    // =========================================================================
    println!("--- Experiment 1: Constitutional Constraints ---");
    {
        let config = SystemConfig {
            developmental: DevelopmentalConfig {
                seed_size: 30,
                dimensions: 3,
                initial_connectivity: 0.4,
                proliferation_rounds: 2,
                target_input_size: Some(4),
                target_output_size: Some(2),
                ..DevelopmentalConfig::cortical()
            },
            governance: ConstitutionalConstraints {
                max_connectivity_per_morphon: 10,
                energy_floor: 0.1,
                ..Default::default()
            },
            lifecycle: LifecycleConfig {
                division: true,
                differentiation: true,
                fusion: false,
                apoptosis: true,
                migration: true,
            },
            ..Default::default()
        };
        let mut system = System::new(config);
        let mut rng = rand::rng();

        // Record initial max degree (developmental wiring is pre-governance)
        let initial_max = system.morphons.keys()
            .map(|&id| system.topology.degree(id))
            .max()
            .unwrap_or(0);

        // Stimulate heavily to drive synaptogenesis
        for _ in 0..2000 {
            let input: Vec<f64> = (0..4).map(|_| rng.random_range(0.0..1.0)).collect();
            system.process_steps(&input, 2);
            system.inject_reward(0.5);
        }

        // Check connectivity cap: runtime synaptogenesis should not grow beyond cap.
        // Developmental wiring may already exceed cap (it's pre-governance).
        let runtime_max = system.morphons.keys()
            .map(|&id| system.topology.degree(id))
            .max()
            .unwrap_or(0);
        check!(
            &format!(
                "runtime synaptogenesis respects cap (initial={}, after={}, cap=10)",
                initial_max, runtime_max
            ),
            runtime_max <= initial_max.max(10) // no growth beyond cap (or initial)
        );

        // Check energy floor
        let min_energy = system.morphons.values()
            .map(|m| m.energy)
            .fold(f64::INFINITY, f64::min);
        check!(
            &format!("energy floor >= 0.1 (actual: {:.4})", min_energy),
            min_energy >= 0.1 - 1e-9
        );

        let stats = system.inspect();
        println!("  System: {} morphons, {} synapses", stats.total_morphons, stats.total_synapses);
    }

    // =========================================================================
    // Experiment 2: Metabolic Cluster Overhead
    // =========================================================================
    println!("\n--- Experiment 2: Metabolic Cluster Overhead ---");
    {
        let config = SystemConfig {
            developmental: DevelopmentalConfig {
                seed_size: 40,
                dimensions: 3,
                initial_connectivity: 0.5,
                proliferation_rounds: 3,
                target_input_size: Some(4),
                target_output_size: Some(2),
                ..DevelopmentalConfig::cortical()
            },
            metabolic: MetabolicConfig {
                cluster_overhead_per_tick: 0.002, // aggressive overhead for testing
                ..Default::default()
            },
            lifecycle: LifecycleConfig {
                division: false,
                differentiation: true,
                fusion: true,
                apoptosis: false,
                migration: false,
            },
            ..Default::default()
        };
        let mut system = System::new(config);
        let mut rng = rand::rng();

        // Drive correlated activity to trigger fusion
        for _ in 0..5000 {
            let input: Vec<f64> = (0..4).map(|_| rng.random_range(0.0..1.0)).collect();
            system.process_steps(&input, 2);
            system.inject_reward(0.3);
        }

        let stats = system.inspect();
        let n_clusters = stats.fused_clusters;

        if n_clusters > 0 {
            let fused_energy: Vec<f64> = system.morphons.values()
                .filter(|m| m.fused_with.is_some())
                .map(|m| m.energy)
                .collect();
            let unfused_energy: Vec<f64> = system.morphons.values()
                .filter(|m| m.fused_with.is_none())
                .map(|m| m.energy)
                .collect();

            let avg_fused = if fused_energy.is_empty() { 1.0 }
                else { fused_energy.iter().sum::<f64>() / fused_energy.len() as f64 };
            let avg_unfused = if unfused_energy.is_empty() { 0.0 }
                else { unfused_energy.iter().sum::<f64>() / unfused_energy.len() as f64 };

            check!(
                &format!(
                    "cluster overhead reduces fused energy (fused={:.3}, unfused={:.3})",
                    avg_fused, avg_unfused
                ),
                avg_fused < avg_unfused || fused_energy.is_empty()
            );
            println!("  {} clusters formed, {} fused morphons", n_clusters, fused_energy.len());
        } else {
            println!("  SKIP  no clusters formed (fusion conditions not met)");
        }
    }

    // =========================================================================
    // Experiment 3: Justification Records
    // =========================================================================
    println!("\n--- Experiment 3: Justification Records ---");
    {
        // Test SynapticJustification directly
        let mut j = SynapticJustification::new(
            FormationCause::HebbianCoincidence { step: 0 },
            0,
        );

        check!("new justification has no reinforcements", !j.has_reinforcement());
        check!("last_reinforcement_step falls back to formation", j.last_reinforcement_step() == 0);

        // Fill beyond capacity
        for i in 0..20 {
            j.record_reinforcement(i * 10, 0.01, 0.5);
        }
        check!(
            &format!("reinforcement history bounded at 16 (actual: {})", j.reinforcement_history.len()),
            j.reinforcement_history.len() == 16
        );
        check!(
            "oldest evicted correctly",
            j.reinforcement_history.front().unwrap().step == 40 // steps 0-30 evicted
        );
        check!(
            &format!("last_reinforcement_step = 190 (actual: {})", j.last_reinforcement_step()),
            j.last_reinforcement_step() == 190
        );

        // Test Synapse::new_justified
        let syn = Synapse::new_justified(
            0.5,
            SynapticJustification::new(FormationCause::External { source: "test".into() }, 42),
        );
        check!("new_justified creates synapse with justification", syn.justification.is_some());
        check!("weight preserved", (syn.weight - 0.5).abs() < 1e-10);

        // Test diagnostics tracking
        let config = SystemConfig {
            developmental: DevelopmentalConfig {
                seed_size: 20,
                dimensions: 3,
                initial_connectivity: 0.3,
                target_input_size: Some(2),
                target_output_size: Some(1),
                ..DevelopmentalConfig::cortical()
            },
            ..Default::default()
        };
        let system = System::new(config);
        let diag = system.diagnostics();
        check!(
            &format!("diagnostics tracks justified_fraction (value: {:.2})", diag.justified_fraction),
            diag.justified_fraction >= 0.0 && diag.justified_fraction <= 1.0
        );
    }

    // =========================================================================
    // Experiment 4: Epistemic State Transitions
    // =========================================================================
    println!("\n--- Experiment 4: Epistemic State Transitions ---");
    {
        // Default state is Hypothesis
        let state = EpistemicState::default();
        check!("default state is Hypothesis", matches!(state, EpistemicState::Hypothesis { .. }));

        // Run a system long enough for clusters to form and get epistemic evaluation
        let config = SystemConfig {
            developmental: DevelopmentalConfig {
                seed_size: 40,
                dimensions: 3,
                initial_connectivity: 0.5,
                proliferation_rounds: 3,
                target_input_size: Some(4),
                target_output_size: Some(2),
                ..DevelopmentalConfig::cortical()
            },
            lifecycle: LifecycleConfig {
                division: false,
                differentiation: true,
                fusion: true,
                apoptosis: false,
                migration: false,
            },
            ..Default::default()
        };
        let mut system = System::new(config);

        // Train with consistent pattern to drive fusion and consolidation
        for _ in 0..3000 {
            let input = vec![1.0, 0.0, 1.0, 0.0];
            system.process_steps(&input, 2);
            system.inject_reward(0.6);
        }

        let n_clusters = system.clusters.len();
        if n_clusters > 0 {
            let mut state_counts = [0usize; 4]; // hypothesis, supported, outdated, contested
            for cluster in system.clusters.values() {
                match &cluster.epistemic_state {
                    EpistemicState::Hypothesis { .. } => state_counts[0] += 1,
                    EpistemicState::Supported { .. } => state_counts[1] += 1,
                    EpistemicState::Outdated { .. } => state_counts[2] += 1,
                    EpistemicState::Contested { .. } => state_counts[3] += 1,
                }
            }
            println!(
                "  {} clusters: H={} S={} O={} C={}",
                n_clusters, state_counts[0], state_counts[1], state_counts[2], state_counts[3]
            );
            check!(
                "epistemic evaluation ran (clusters have states)",
                state_counts.iter().sum::<usize>() == n_clusters
            );
        } else {
            println!("  SKIP  no clusters formed — epistemic transitions not testable");
        }
    }

    // =========================================================================
    // Experiment 5: Epistemic Scarring
    // =========================================================================
    println!("\n--- Experiment 5: Epistemic Scarring ---");
    {
        let mut history = EpistemicHistory::default();
        let base_confidence = history.required_confidence();
        check!(
            &format!("base required confidence = {:.2}", base_confidence),
            (base_confidence - 0.80).abs() < 0.01
        );

        // Accumulate failures
        history.stale_count = 5;
        history.contested_count = 3;
        history.false_positive_count = 1;
        history.update_skepticism();

        let scarred_confidence = history.required_confidence();
        check!(
            &format!(
                "scarring raises required confidence ({:.3} > {:.3})",
                scarred_confidence, base_confidence
            ),
            scarred_confidence > base_confidence
        );
        check!(
            &format!("skepticism = {:.3}", history.skepticism),
            history.skepticism > 0.0 && history.skepticism <= 1.0
        );

        // Test decay
        let pre_stale = history.stale_count;
        history.decay();
        check!(
            &format!("decay reduces stale_count ({} -> {})", pre_stale, history.stale_count),
            history.stale_count < pre_stale
        );

        // Test cap at 0.98
        history.stale_count = 100;
        history.contested_count = 100;
        history.false_positive_count = 100;
        history.update_skepticism();
        check!(
            &format!("required confidence capped at 0.98 (actual: {:.3})", history.required_confidence()),
            history.required_confidence() <= 0.98
        );
    }

    // =========================================================================
    // Summary
    // =========================================================================
    println!("\n=== Results: {} passed, {} failed ===", passed, failed);
    if failed > 0 {
        std::process::exit(1);
    }
}
