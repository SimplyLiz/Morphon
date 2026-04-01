//! MNIST V2 Comparison — proves V2 primitives improve classification.
//!
//! Runs two systems side-by-side on the same shuffled data:
//! - Baseline: fixed topology (identical to mnist.rs)
//! - V2: frustration + bioelectric field + target morphology + lifecycle ON
//!
//! After Phase 2 labeling, the V2 system undergoes a damage-recovery test:
//! kill 30% of associative morphons, measure accuracy drop, then let
//! self-healing recover and re-evaluate.
//!
//! Setup: Download MNIST files to ./data/ (unzipped)
//! Run: cargo run --example mnist_v2 --release
//! Run: cargo run --example mnist_v2 --release -- --standard
//! Run: cargo run --example mnist_v2 --release -- --extended

use mnist::MnistBuilder;
use morphon_core::developmental::DevelopmentalConfig;
use morphon_core::field::FieldConfig;
use morphon_core::homeostasis::HomeostasisParams;
use morphon_core::learning::LearningParams;
use morphon_core::morphogenesis::MorphogenesisParams;
use morphon_core::morphon::MetabolicConfig;
use morphon_core::scheduler::SchedulerConfig;
use morphon_core::system::{System, SystemConfig};
use morphon_core::types::*;
use morphon_core::MorphonId;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use serde_json::json;
use std::collections::HashMap;
use std::fs;

const IMG_PIXELS: usize = 28 * 28;
const NUM_CLASSES: usize = 10;
const STEPS_PER_IMAGE: usize = 30;

fn pixel_rates(raw: &[u8]) -> Vec<f64> {
    raw.iter().map(|&p| p as f64 / 255.0).collect()
}

fn poisson_frame(rates: &[f64], rng: &mut impl Rng) -> Vec<f64> {
    rates
        .iter()
        .map(|&r| {
            if rng.random_range(0.0..1.0) < r {
                3.0
            } else {
                0.0
            }
        })
        .collect()
}

fn present_image(system: &mut System, rates: &[f64], steps: usize, rng: &mut impl Rng) {
    for m in system.morphons.values_mut() {
        if m.cell_type != CellType::Sensory {
            m.potential = 0.0;
            m.prev_potential = 0.0;
            m.input_accumulator = 0.0;
            m.fired = false;
            m.refractory_timer = 0.0;
        }
    }
    system.resonance.clear();
    for _ in 0..steps {
        let frame = poisson_frame(rates, rng);
        system.feed_input(&frame);
        system.step();
    }
}

#[derive(Debug, Clone)]
struct PhaseStats {
    images_seen: usize,
    morphons: usize,
    synapses: usize,
    firing_rate: f64,
    avg_pe: f64,
    avg_frustration: f64,
    exploring: usize,
    field_pe: f64,
}

fn base_learning_params() -> LearningParams {
    LearningParams {
        tau_eligibility: 5.0,
        tau_trace: 8.0,
        a_plus: 1.0,
        a_minus: -0.8,
        tau_tag: 200.0,
        tag_threshold: 0.5,
        capture_threshold: 10.0,
        capture_rate: 0.5,
        weight_max: 1.5,
        weight_min: 0.01,
        alpha_reward: 0.3,
        alpha_novelty: 1.0,
        alpha_arousal: 0.0,
        alpha_homeostasis: 0.1,
        transmitter_potentiation: 0.0005,
        heterosynaptic_depression: 0.001,
    }
}

fn create_baseline_system() -> System {
    let config = SystemConfig {
        developmental: DevelopmentalConfig {
            seed_size: 500,
            dimensions: 6,
            initial_connectivity: 0.02,
            proliferation_rounds: 1,
            target_input_size: Some(IMG_PIXELS),
            target_output_size: Some(NUM_CLASSES),
            ..DevelopmentalConfig::cortical()
        },
        scheduler: SchedulerConfig {
            medium_period: 1,
            slow_period: 1000,
            glacial_period: 5000,
            homeostasis_period: 30,
            memory_period: 100,
        },
        learning: base_learning_params(),
        morphogenesis: MorphogenesisParams {
            migration_rate: 0.05,
            max_morphons: 2000,
            ..Default::default()
        },
        homeostasis: HomeostasisParams {
            kwta_fraction: 0.01,
            ..Default::default()
        },
        lifecycle: LifecycleConfig {
            division: false,
            differentiation: true,
            fusion: false,
            apoptosis: false,
            migration: false,
        },
        metabolic: MetabolicConfig::default(),
        dt: 1.0,
        working_memory_capacity: 7,
        episodic_memory_capacity: 500,
        ..Default::default()
    };
    System::new(config)
}

fn create_v2_system() -> System {
    let mut tm = morphon_core::TargetMorphology::cortical(6);
    tm.regions[0].target_density = 100; // Sensory
    tm.regions[1].target_density = 300; // Associative
    tm.regions[2].target_density = 20; // Motor
    tm.healing_threshold = 0.4;

    let config = SystemConfig {
        developmental: DevelopmentalConfig {
            seed_size: 500,
            dimensions: 6,
            initial_connectivity: 0.02,
            proliferation_rounds: 1,
            target_input_size: Some(IMG_PIXELS),
            target_output_size: Some(NUM_CLASSES),
            ..DevelopmentalConfig::cortical()
        },
        scheduler: SchedulerConfig {
            medium_period: 1,
            slow_period: 500,    // V2: balanced — field + migration without over-pruning
            glacial_period: 1500, // V2: division + healing every ~50 images
            homeostasis_period: 30,
            memory_period: 500,
        },
        learning: base_learning_params(),
        morphogenesis: MorphogenesisParams {
            migration_rate: 0.05,
            max_morphons: 2000,
            pruning_min_age: 500, // protect developmental connections longer — STDP needs time on 784 inputs
            frustration: FrustrationConfig {
                enabled: true,
                max_noise_multiplier: 3.0,
                weight_perturbation_scale: 0.005,
                ..Default::default()
            },
            ..Default::default()
        },
        homeostasis: HomeostasisParams {
            kwta_fraction: 0.01,
            ..Default::default()
        },
        lifecycle: LifecycleConfig {
            division: true,
            differentiation: true,
            fusion: false,
            apoptosis: false, // off during training — don't kill morphons still building connections
            migration: true,
        },
        field: FieldConfig {
            enabled: true,
            resolution: 32,
            diffusion_rate: 0.12,
            decay_rate: 0.04,
            active_layers: vec![
                morphon_core::field::FieldType::PredictionError,
                morphon_core::field::FieldType::Energy,
                morphon_core::field::FieldType::Stress,
                morphon_core::field::FieldType::Identity,
            ],
            migration_field_weight: 0.35,
        },
        target_morphology: Some(tm),
        metabolic: MetabolicConfig::default(),
        dt: 1.0,
        working_memory_capacity: 7,
        episodic_memory_capacity: 500,
        ..Default::default()
    };
    System::new(config)
}

fn run_phase1(
    system: &mut System,
    train_images: &[Vec<f64>],
    epochs: usize,
    samples: usize,
    label: &str,
    rng: &mut impl Rng,
) -> Vec<PhaseStats> {
    let mut stats = Vec::new();

    for epoch in 0..epochs {
        let mut indices: Vec<usize> = (0..train_images.len()).collect();
        indices.shuffle(rng);

        for (bi, &idx) in indices.iter().take(samples).enumerate() {
            present_image(system, &train_images[idx], STEPS_PER_IMAGE, rng);
            system.inject_novelty(0.2);

            if (bi + 1) % 500 == 0 {
                let s = system.inspect();
                let diag = system.diagnostics();
                let stat = PhaseStats {
                    images_seen: epoch * samples + bi + 1,
                    morphons: s.total_morphons,
                    synapses: s.total_synapses,
                    firing_rate: s.firing_rate,
                    avg_pe: s.avg_prediction_error,
                    avg_frustration: diag.avg_frustration,
                    exploring: diag.exploration_mode_count,
                    field_pe: diag.field_pe_mean,
                };
                println!(
                    "  [{}] Ep{} [{:>4}/{}] m={} s={} fr={:.3} pe={:.3} frust={:.3}({}) field={:.4}",
                    label,
                    epoch + 1,
                    bi + 1,
                    samples,
                    stat.morphons,
                    stat.synapses,
                    stat.firing_rate,
                    stat.avg_pe,
                    stat.avg_frustration,
                    stat.exploring,
                    stat.field_pe,
                );
                stats.push(stat);
            }
        }
    }
    stats
}

fn run_phase2_labeling(
    system: &mut System,
    train_images: &[Vec<f64>],
    train_labels: &[usize],
    samples: usize,
    rng: &mut impl Rng,
) -> HashMap<MorphonId, usize> {
    let assoc_ids: Vec<MorphonId> = system
        .morphons
        .values()
        .filter(|m| m.cell_type == CellType::Associative || m.cell_type == CellType::Stem)
        .map(|m| m.id)
        .collect();

    let mut neuron_class_counts: HashMap<MorphonId, Vec<usize>> = assoc_ids
        .iter()
        .map(|&id| (id, vec![0usize; NUM_CLASSES]))
        .collect();

    let mut indices: Vec<usize> = (0..train_images.len()).collect();
    indices.shuffle(rng);

    for (bi, &idx) in indices.iter().take(samples).enumerate() {
        present_image(system, &train_images[idx], STEPS_PER_IMAGE, rng);
        let label = train_labels[idx];

        for &aid in &assoc_ids {
            if let Some(m) = system.morphons.get(&aid) {
                if m.fired || m.activity_history.mean() > 0.05 {
                    if let Some(counts) = neuron_class_counts.get_mut(&aid) {
                        counts[label] += 1;
                    }
                }
            }
        }

        if (bi + 1) % 500 == 0 {
            println!("    labeling [{}/{}]", bi + 1, samples);
        }
    }

    neuron_class_counts
        .iter()
        .filter_map(|(&id, counts)| {
            let total: usize = counts.iter().sum();
            if total == 0 {
                return None;
            }
            let best = counts
                .iter()
                .enumerate()
                .max_by_key(|&(_, c)| *c)
                .map(|(i, _)| i)
                .unwrap_or(0);
            Some((id, best))
        })
        .collect()
}

fn evaluate_with_labels(
    system: &mut System,
    test_images: &[Vec<f64>],
    test_labels: &[usize],
    neuron_labels: &HashMap<MorphonId, usize>,
    n: usize,
    rng: &mut impl Rng,
) -> f64 {
    let mut correct = 0;
    let tested = test_images.len().min(n);

    for i in 0..tested {
        present_image(system, &test_images[i], STEPS_PER_IMAGE, rng);

        let mut class_votes = vec![0.0f64; NUM_CLASSES];
        for (&aid, &assigned_class) in neuron_labels {
            if let Some(m) = system.morphons.get(&aid) {
                let activity = if m.fired {
                    1.0
                } else {
                    m.activity_history.mean()
                };
                class_votes[assigned_class] += activity;
            }
            // Dead morphons (removed by damage) return None — zero votes, correct behavior
        }

        let pred = class_votes
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        if pred == test_labels[i] {
            correct += 1;
        }
    }

    correct as f64 / tested as f64 * 100.0
}

fn parse_profile() -> &'static str {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--extended") {
        "extended"
    } else if args.iter().any(|a| a == "--standard") {
        "standard"
    } else {
        "quick"
    }
}

fn main() {
    let profile = parse_profile();
    let (p1_epochs, p1_samples, p2_samples, test_n, recovery_samples) = match profile {
        "extended" => (3, 5000, 5000, 1000, 3000),
        "standard" => (2, 3000, 3000, 500, 1500),
        _ => (1, 1000, 1000, 200, 500),
    };

    println!(
        "=== MORPHON MNIST V2 Comparison [{}] ===\n",
        profile
    );
    println!("Loading MNIST from ./data/ ...");

    let mnist = MnistBuilder::new()
        .label_format_digit()
        .base_path("data")
        .training_set_length(10_000)
        .test_set_length(1_000)
        .finalize();

    let train_images: Vec<Vec<f64>> = (0..10_000)
        .map(|i| pixel_rates(&mnist.trn_img[i * IMG_PIXELS..(i + 1) * IMG_PIXELS]))
        .collect();
    let train_labels: Vec<usize> = mnist.trn_lbl.iter().map(|&l| l as usize).collect();
    let test_images: Vec<Vec<f64>> = (0..1_000)
        .map(|i| pixel_rates(&mnist.tst_img[i * IMG_PIXELS..(i + 1) * IMG_PIXELS]))
        .collect();
    let test_labels: Vec<usize> = mnist.tst_lbl.iter().map(|&l| l as usize).collect();

    println!("Train: {}, Test: {}\n", train_images.len(), test_images.len());

    // Fixed seed for reproducible comparison
    let seed = 42u64;

    // =========================================================================
    // BASELINE
    // =========================================================================
    println!("━━━ BASELINE (fixed topology) ━━━\n");
    let mut baseline = create_baseline_system();
    {
        let s = baseline.inspect();
        println!(
            "  Initial: {} morphons, {} synapses",
            s.total_morphons, s.total_synapses
        );
    }

    println!("\n  Phase 1: Unsupervised STDP ({} epochs × {} images)", p1_epochs, p1_samples);
    let mut rng_b = rand::rngs::StdRng::seed_from_u64(seed);
    let _baseline_stats = run_phase1(
        &mut baseline,
        &train_images,
        p1_epochs,
        p1_samples,
        "BASE",
        &mut rng_b,
    );

    println!("\n  Phase 2: Post-hoc labeling ({} images)", p2_samples);
    baseline.config.scheduler.medium_period = 999999; // freeze STDP
    let baseline_labels =
        run_phase2_labeling(&mut baseline, &train_images, &train_labels, p2_samples, &mut rng_b);
    println!("    {} neurons labeled", baseline_labels.len());

    println!("\n  Testing ({} images)...", test_n);
    let baseline_acc = evaluate_with_labels(
        &mut baseline,
        &test_images,
        &test_labels,
        &baseline_labels,
        test_n,
        &mut rng_b,
    );
    println!("  Baseline accuracy: {:.1}%", baseline_acc);

    // =========================================================================
    // V2
    // =========================================================================
    println!("\n━━━ V2 (frustration + field + target morphology) ━━━\n");
    let mut v2 = create_v2_system();
    {
        let s = v2.inspect();
        println!(
            "  Initial: {} morphons, {} synapses",
            s.total_morphons, s.total_synapses
        );
    }

    println!("\n  Phase 1: Unsupervised STDP + V2 primitives ({} epochs × {} images)", p1_epochs, p1_samples);
    let mut rng_v = rand::rngs::StdRng::seed_from_u64(seed); // same seed
    let _v2_stats = run_phase1(
        &mut v2,
        &train_images,
        p1_epochs,
        p1_samples,
        "V2  ",
        &mut rng_v,
    );

    // Freeze for Phase 2
    println!("\n  Phase 2: Post-hoc labeling ({} images)", p2_samples);
    v2.config.scheduler.medium_period = 999999;
    let saved_lifecycle = v2.config.lifecycle.clone();
    v2.config.lifecycle = LifecycleConfig {
        division: false,
        differentiation: false,
        fusion: false,
        apoptosis: false,
        migration: false,
    };
    let v2_labels =
        run_phase2_labeling(&mut v2, &train_images, &train_labels, p2_samples, &mut rng_v);
    println!("    {} neurons labeled", v2_labels.len());

    println!("\n  Testing ({} images)...", test_n);
    let v2_acc = evaluate_with_labels(
        &mut v2,
        &test_images,
        &test_labels,
        &v2_labels,
        test_n,
        &mut rng_v,
    );
    // Clone stats to release borrows before damage phase
    let v2_morphons_count = v2.inspect().total_morphons;
    let v2_synapses_count = { let s = v2.inspect(); s.total_synapses };
    let v2_fr = v2.inspect().firing_rate;
    let v2_avg_frust = v2.diagnostics().avg_frustration;
    let v2_exploring = v2.diagnostics().exploration_mode_count;
    let v2_field_pe = v2.diagnostics().field_pe_mean;
    println!("  V2 accuracy: {:.1}%", v2_acc);
    println!(
        "  Morphons: {} | Synapses: {} | FR: {:.3}",
        v2_morphons_count, v2_synapses_count, v2_fr
    );
    println!(
        "  Frustration: avg={:.3} exploring={}",
        v2_avg_frust, v2_exploring
    );
    println!("  Field PE: {:.4}", v2_field_pe);

    // =========================================================================
    // DAMAGE-RECOVERY (V2 only)
    // =========================================================================
    println!("\n━━━ DAMAGE-RECOVERY TEST (V2 system) ━━━\n");

    let pre_damage_acc = v2_acc;
    let morphons_before = v2.morphons.len();

    // Kill 30% of associative morphons
    let assoc_ids: Vec<MorphonId> = v2
        .morphons
        .values()
        .filter(|m| m.cell_type == CellType::Associative)
        .map(|m| m.id)
        .collect();
    let n_assoc = assoc_ids.len();
    let kill_count = (n_assoc as f64 * 0.3).ceil() as usize;
    let mut kill_rng = rand::rngs::StdRng::seed_from_u64(seed + 999);
    let mut kill_ids = assoc_ids;
    kill_ids.shuffle(&mut kill_rng);
    let kill_targets: Vec<MorphonId> = kill_ids.into_iter().take(kill_count).collect();
    for &id in &kill_targets {
        v2.morphons.remove(&id);
        v2.topology.remove_morphon(id);
    }
    let morphons_after_damage = v2.morphons.len();
    println!(
        "  Killed {} / {} associative morphons ({} → {} total)",
        kill_count,
        n_assoc,
        morphons_before,
        morphons_after_damage
    );

    // Post-damage accuracy (immediate, no recovery)
    let post_damage_acc = evaluate_with_labels(
        &mut v2,
        &test_images,
        &test_labels,
        &v2_labels, // still uses old labels — dead neurons vote zero
        test_n,
        &mut rng_v,
    );
    println!("  Post-damage accuracy: {:.1}%", post_damage_acc);

    // Recovery: re-enable lifecycle, run more training
    println!(
        "\n  Recovery: re-enable lifecycle, train {} images...",
        recovery_samples
    );
    v2.config.lifecycle = saved_lifecycle;
    v2.config.scheduler.medium_period = 1;
    v2.config.scheduler.slow_period = 300;
    v2.config.scheduler.glacial_period = 1500;

    let _recovery_stats = run_phase1(
        &mut v2,
        &train_images,
        1,
        recovery_samples,
        "RECV",
        &mut rng_v,
    );

    // Re-label (new morphons need labels)
    println!("\n  Re-labeling after recovery...");
    v2.config.scheduler.medium_period = 999999;
    v2.config.lifecycle = LifecycleConfig {
        division: false,
        differentiation: false,
        fusion: false,
        apoptosis: false,
        migration: false,
    };
    let recovery_labels =
        run_phase2_labeling(&mut v2, &train_images, &train_labels, p2_samples, &mut rng_v);
    println!("    {} neurons labeled", recovery_labels.len());

    let post_recovery_acc = evaluate_with_labels(
        &mut v2,
        &test_images,
        &test_labels,
        &recovery_labels,
        test_n,
        &mut rng_v,
    );
    let morphons_after_recovery = v2.morphons.len();

    // =========================================================================
    // COMPARISON REPORT
    // =========================================================================
    let delta_acc = v2_acc - baseline_acc;
    let damage_drop = post_damage_acc - pre_damage_acc;
    let recovery_gain = post_recovery_acc - post_damage_acc;

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!(
        "║  MORPHON MNIST V2 Comparison [{}]{}║",
        profile,
        " ".repeat(35 - profile.len())
    );
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!(
        "║  Baseline accuracy:     {:>5.1}%                               ║",
        baseline_acc
    );
    println!(
        "║  V2 accuracy:           {:>5.1}%  (delta {:+.1}pp)                 ║",
        v2_acc, delta_acc
    );
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Damage-Recovery (V2):                                      ║");
    println!(
        "║    Pre-damage:    {:>5.1}%                                     ║",
        pre_damage_acc
    );
    println!(
        "║    Post-damage:   {:>5.1}%  ({:+.1}pp)                            ║",
        post_damage_acc, damage_drop
    );
    println!(
        "║    Post-recovery: {:>5.1}%  ({:+.1}pp from damaged)               ║",
        post_recovery_acc, recovery_gain
    );
    println!(
        "║    Morphons: {} → {} → {}{}║",
        morphons_before,
        morphons_after_damage,
        morphons_after_recovery,
        " ".repeat(
            38 - format!(
                "{} → {} → {}",
                morphons_before, morphons_after_damage, morphons_after_recovery
            )
            .len()
        )
    );
    println!("╚══════════════════════════════════════════════════════════════╝");

    // Save results
    let version = env!("CARGO_PKG_VERSION");
    let results = json!({
        "benchmark": "mnist_v2",
        "profile": profile,
        "version": version,
        "timestamp": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
        "baseline": { "accuracy": baseline_acc },
        "v2": {
            "accuracy": v2_acc,
            "morphons": v2_morphons_count,
            "synapses": v2_synapses_count,
            "frustration": v2_avg_frust,
            "field_pe": v2_field_pe,
        },
        "damage_recovery": {
            "pre_damage": pre_damage_acc,
            "post_damage": post_damage_acc,
            "post_recovery": post_recovery_acc,
            "killed": kill_count,
            "morphons_before": morphons_before,
            "morphons_after_damage": morphons_after_damage,
            "morphons_after_recovery": morphons_after_recovery,
        },
    });

    let dir = format!("docs/benchmark_results/v{}", version);
    fs::create_dir_all(&dir).ok();
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let path = format!("{}/mnist_v2_{}.json", dir, ts);
    let json_str = serde_json::to_string_pretty(&results).unwrap();
    fs::write(&path, &json_str).unwrap();
    println!("\nResults saved to {}", path);
}
