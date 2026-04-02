//! MNIST V2 — Supervised analog readout on morphon features.
//!
//! Baseline vs V2 comparison. Both use supervised readout (softmax cross-entropy).
//! V2 features (field, frustration, target morphology) active during development.
//! Topology frozen during supervised training — readout trains on stable features.
//!
//! Setup: Download MNIST files to ./data/ (unzipped)
//! Run: cargo run --example mnist_v2 --release

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
use rand::SeedableRng;
use serde_json::json;
use std::fs;
use std::time::Instant;

const IMG_PIXELS: usize = 28 * 28;
const NUM_CLASSES: usize = 10;
const STEPS_PER_IMAGE: usize = 10;
const INPUT_AMP: f64 = 4.0;
const LR_START: f64 = 0.02;
const LR_END: f64 = 0.005;

fn pixel_inputs(raw: &[u8]) -> Vec<f64> {
    raw.iter().map(|&p| p as f64 / 255.0 * INPUT_AMP).collect()
}

fn present_image(system: &mut System, inputs: &[f64]) {
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
    for _ in 0..STEPS_PER_IMAGE {
        system.feed_input(inputs);
        system.step();
    }
}

fn base_learning_params() -> LearningParams {
    LearningParams {
        tau_eligibility: 5.0,
        tau_trace: 5.0,
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
        heterosynaptic_depression: 0.001, tag_accumulation_rate: 0.3,
    }
}

/// Scheduler for supervised training — no expensive structural ops.
fn training_scheduler() -> SchedulerConfig {
    SchedulerConfig {
        medium_period: 10,    // STDP once per image, not every sub-step
        slow_period: 5000,    // synaptogenesis/pruning very rare during training
        glacial_period: 50000, // effectively never during training
        homeostasis_period: 10, // once per image
        memory_period: 100,
    }
}

/// Frozen lifecycle — no structural changes during supervised training.
fn frozen_lifecycle() -> LifecycleConfig {
    LifecycleConfig {
        division: false,
        differentiation: false,
        fusion: false,
        apoptosis: false,
        migration: false,
    }
}

fn create_baseline(kwta_fraction: f64, local: bool) -> System {
    let (radius, local_k) = if local { (5.0, 5) } else { (0.0, 3) };
    let config = SystemConfig {
        developmental: DevelopmentalConfig {
            seed_size: 200,
            dimensions: 6,
            initial_connectivity: 0.02,
            proliferation_rounds: 1,
            target_input_size: Some(IMG_PIXELS),
            target_output_size: Some(NUM_CLASSES),
            ..DevelopmentalConfig::cortical()
        },
        scheduler: training_scheduler(),
        learning: base_learning_params(),
        morphogenesis: MorphogenesisParams::default(),
        homeostasis: HomeostasisParams {
            competition_mode: morphon_core::homeostasis::CompetitionMode::GlobalKWTA {
                fraction: kwta_fraction,
                local_radius: radius,
                local_k,
            },
            ..Default::default()
        },
        lifecycle: frozen_lifecycle(),
        metabolic: MetabolicConfig::default(),
        endoquilibrium: morphon_core::endoquilibrium::EndoConfig { enabled: true, ..Default::default() },
        dt: 1.0,
        working_memory_capacity: 7,
        episodic_memory_capacity: 500,
        ..Default::default()
    };
    let mut sys = System::new(config);
    sys.enable_analog_readout();
    sys
}

fn create_v2(kwta_fraction: f64, local: bool) -> System {
    let mut tm = morphon_core::TargetMorphology::cortical(6);
    tm.regions[0].target_density = 50;
    tm.regions[1].target_density = 150;
    tm.regions[2].target_density = 15;
    tm.healing_threshold = 0.4;

    let config = SystemConfig {
        developmental: DevelopmentalConfig {
            seed_size: 200,
            dimensions: 6,
            initial_connectivity: 0.02,
            proliferation_rounds: 1,
            target_input_size: Some(IMG_PIXELS),
            target_output_size: Some(NUM_CLASSES),
            ..DevelopmentalConfig::cortical()
        },
        // During training, use the same conservative scheduler.
        // V2 structural features ran during System::new() development.
        scheduler: training_scheduler(),
        learning: base_learning_params(),
        morphogenesis: MorphogenesisParams {
            pruning_min_age: 500,
            frustration: FrustrationConfig {
                enabled: true,
                max_noise_multiplier: 3.0,
                weight_perturbation_scale: 0.005,
                ..Default::default()
            },
            ..Default::default()
        },
        homeostasis: HomeostasisParams {
            competition_mode: morphon_core::homeostasis::CompetitionMode::GlobalKWTA {
                fraction: kwta_fraction,
                local_radius: if local { 5.0 } else { 0.0 },
                local_k: if local { 5 } else { 3 },
            },
            ..Default::default()
        },
        // Frozen during training — V2 features active during development only
        lifecycle: frozen_lifecycle(),
        field: FieldConfig {
            enabled: true,
            resolution: 16,
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
        endoquilibrium: morphon_core::endoquilibrium::EndoConfig { enabled: true, ..Default::default() },
        dt: 1.0,
        working_memory_capacity: 7,
        episodic_memory_capacity: 500,
        ..Default::default()
    };
    let mut sys = System::new(config);
    sys.enable_analog_readout();
    sys
}

fn classify(system: &mut System, inputs: &[f64]) -> usize {
    present_image(system, inputs);
    let outputs = system.read_output();
    outputs.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i).unwrap_or(0)
}

fn evaluate(system: &mut System, images: &[Vec<f64>], labels: &[usize], n: usize) -> f64 {
    let mut correct = 0;
    let tested = images.len().min(n);
    for i in 0..tested {
        if classify(system, &images[i]) == labels[i] { correct += 1; }
    }
    correct as f64 / tested as f64 * 100.0
}

fn train_and_eval(
    system: &mut System,
    train_images: &[Vec<f64>], train_labels: &[usize],
    test_images: &[Vec<f64>], test_labels: &[usize],
    n_train: usize, epochs: usize, label: &str,
    rng: &mut rand::rngs::StdRng,
) -> f64 {
    let start = Instant::now();
    for epoch in 0..epochs {
        let mut indices: Vec<usize> = (0..train_images.len()).collect();
        indices.shuffle(rng);

        // LR schedule: 0.05 → 0.005 over epochs
        let lr = LR_START * (LR_END / LR_START).powf(epoch as f64 / epochs.max(1) as f64);

        let mut running_correct = 0_u64;
        let mut running_total = 0_u64;

        for (bi, &idx) in indices.iter().take(n_train).enumerate() {
            present_image(system, &train_images[idx]);
            system.inject_novelty(0.05);

            // Supervised readout + contrastive reward at output ports
            let correct = train_labels[idx];
            system.train_readout(correct, lr);
            system.reward_contrastive(correct, 0.2, 0.1);

            // Track running accuracy and report to endo every image
            let predicted = system.read_output().iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i).unwrap_or(0);
            running_total += 1;
            if predicted == correct { running_correct += 1; }
            let running_acc = running_correct as f64 / running_total as f64;
            system.report_performance(running_acc);

            // Episode-end consolidation per image
            let reward = if predicted == correct { 1.0 } else { 0.0 };
            system.report_episode_end(reward);

            let report_interval = if n_train <= 500 { 250 } else { 1000 };
            if (bi + 1) % report_interval == 0 || (bi + 1 == n_train && epoch + 1 == epochs) {
                let acc = evaluate(system, test_images, test_labels, 100);
                eprintln!("  [{}] ep{} {:>5}/{} acc={:.1}% lr={:.4} ({:.0}s) {}",
                    label, epoch + 1, bi + 1, n_train, acc, lr, start.elapsed().as_secs(),
                    system.endo.summary());
            }
        }
    }
    evaluate(system, test_images, test_labels, 200)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let profile = if args.iter().any(|a| a == "--extended") { "extended" }
        else if args.iter().any(|a| a == "--standard") { "standard" }
        else if args.iter().any(|a| a == "--fast") { "fast" }
        else { "quick" };

    let fast = profile == "fast";
    let damage_sweep = args.iter().any(|a| a == "--damage-sweep");
    let seed: u64 = args.iter()
        .find(|a| a.starts_with("--seed="))
        .and_then(|a| a[7..].parse().ok())
        .unwrap_or(42);
    let (n_train, n_epochs, n_test, recovery_n) = match profile {
        "extended" => (10000, 3, 500, 3000),
        "standard" => (5000, 3, 300, 1500),
        "fast" => (500, 2, 100, if damage_sweep { 300 } else { 0 }),
        _ => (3000, 3, 200, 500),
    };

    eprintln!("=== MORPHON MNIST V2 — Supervised Readout [{}] seed={} ===\n", profile, seed);
    eprintln!("Loading MNIST ...");

    let mnist = MnistBuilder::new()
        .label_format_digit()
        .base_path("data")
        .training_set_length(10_000)
        .test_set_length(1_000)
        .finalize();

    let train_images: Vec<Vec<f64>> = (0..10_000)
        .map(|i| pixel_inputs(&mnist.trn_img[i * IMG_PIXELS..(i + 1) * IMG_PIXELS]))
        .collect();
    let train_labels: Vec<usize> = mnist.trn_lbl.iter().map(|&l| l as usize).collect();
    let test_images: Vec<Vec<f64>> = (0..1_000)
        .map(|i| pixel_inputs(&mnist.tst_img[i * IMG_PIXELS..(i + 1) * IMG_PIXELS]))
        .collect();
    let test_labels: Vec<usize> = mnist.tst_lbl.iter().map(|&l| l as usize).collect();
    eprintln!("Train: {}, Test: {}, using: {}\n", train_images.len(), test_images.len(), n_train);

    let no_kwta = args.iter().any(|a| a == "--no-kwta");
    let debug = args.iter().any(|a| a == "--debug");
    // --kwta=0.15 to override the default fraction
    let kwta_override: Option<f64> = args.iter()
        .find(|a| a.starts_with("--kwta="))
        .and_then(|a| a[7..].parse().ok());
    if no_kwta { eprintln!("*** k-WTA DISABLED ***\n"); }

    // === DIAGNOSTIC: is the hidden layer discriminative? (skipped in fast unless --debug) ===
    let kwta = if no_kwta { 1.0 } else { kwta_override.unwrap_or(0.05) };
    let use_local_kwta = kwta_override.is_none() && !no_kwta;
    if kwta_override.is_some() { eprintln!("k-WTA fraction override: {:.2} (global)\n", kwta); }
    if !fast || debug {
    eprintln!("━━━ DIAGNOSTIC ━━━");
    let mut diag_sys = create_baseline(kwta, use_local_kwta);

    // Collect assoc IDs in stable order
    let mut assoc_ids: Vec<MorphonId> = diag_sys.morphons.values()
        .filter(|m| m.cell_type == CellType::Associative || m.cell_type == CellType::Stem)
        .map(|m| m.id).collect();
    assoc_ids.sort();

    // Neighborhood size diagnostic for local k-WTA
    if debug {
        let radius = match &diag_sys.config.homeostasis.competition_mode {
            morphon_core::homeostasis::CompetitionMode::GlobalKWTA { local_radius, .. } => *local_radius,
            _ => 0.0,
        };
        if radius > 0.0 {
            let positions: Vec<_> = assoc_ids.iter()
                .filter_map(|&id| diag_sys.morphons.get(&id).map(|m| m.position.clone()))
                .collect();
            let n = positions.len();
            let mut sizes: Vec<usize> = (0..n).map(|i| {
                (0..n).filter(|&j| j != i && positions[i].distance(&positions[j]) < radius).count()
            }).collect();
            sizes.sort();
            eprintln!("  local k-WTA radius={:.1}: neighborhood sizes min={} p25={} median={} p75={} max={} (of {} assoc)",
                radius, sizes[0], sizes[n/4], sizes[n/2], sizes[3*n/4], sizes[n-1], n);

            // Pairwise distance distribution
            let mut dists: Vec<f64> = Vec::new();
            for i in 0..n.min(100) {
                for j in (i+1)..n.min(100) {
                    dists.push(positions[i].distance(&positions[j]));
                }
            }
            dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let dl = dists.len();
            if dl > 0 {
                eprintln!("  pairwise distances (first 100): min={:.2} p25={:.2} median={:.2} p75={:.2} max={:.2}",
                    dists[0], dists[dl/4], dists[dl/2], dists[3*dl/4], dists[dl-1]);
            }
        } else {
            eprintln!("  local k-WTA: disabled (global competition)");
        }
    }

    // Present one image per digit, record which assoc morphons fired
    let mut digit_firing: Vec<Vec<bool>> = Vec::new(); // [digit][assoc_idx] = fired
    for digit in 0..10 {
        let idx = train_labels.iter().position(|&l| l == digit).unwrap();
        present_image(&mut diag_sys, &train_images[idx]);
        let fired: Vec<bool> = assoc_ids.iter().map(|&id| {
            diag_sys.morphons.get(&id).map_or(false, |m| m.fired || m.potential > 0.3)
        }).collect();
        let n_fired = fired.iter().filter(|&&f| f).count();
        eprintln!("  digit {}: {}/{} assoc active", digit, n_fired, assoc_ids.len());
        digit_firing.push(fired);
    }

    // Jaccard similarity between digit pairs (on assoc firing patterns)
    eprintln!("\n  Jaccard similarity (assoc firing):");
    for i in 0..10 {
        let mut sims = String::new();
        for j in 0..10 {
            let both: usize = digit_firing[i].iter().zip(&digit_firing[j])
                .filter(|(a, b)| **a && **b).count();
            let either: usize = digit_firing[i].iter().zip(&digit_firing[j])
                .filter(|(a, b)| **a || **b).count();
            let jac = if either > 0 { both as f64 / either as f64 } else { 0.0 };
            sims.push_str(&format!("{:.2} ", jac));
        }
        eprintln!("  {}: {}", i, sims);
    }
    eprintln!();
    drop(diag_sys);
    } // end diagnostic

    // === BASELINE (skipped in fast mode) ===
    let base_acc = if !fast {
        eprintln!("━━━ BASELINE ━━━");
        let t0 = Instant::now();
        let mut base = create_baseline(kwta, use_local_kwta);
        eprintln!("  {} morphons, {} synapses (built in {:.1}s)",
            base.inspect().total_morphons, base.inspect().total_synapses, t0.elapsed().as_secs_f64());
        let mut rng_b = rand::rngs::StdRng::seed_from_u64(seed);
        let acc = train_and_eval(&mut base, &train_images, &train_labels,
            &test_images, &test_labels, n_train, n_epochs, "BASE", &mut rng_b);
        eprintln!("  Baseline: {:.1}% ({:.0}s)\n", acc, t0.elapsed().as_secs_f64());
        acc
    } else { 0.0 };

    // === V2 ===
    eprintln!("━━━ V2 ━━━");
    let t1 = Instant::now();
    let mut v2 = create_v2(kwta, use_local_kwta);
    eprintln!("  {} morphons, {} synapses (built in {:.1}s)",
        v2.inspect().total_morphons, v2.inspect().total_synapses, t1.elapsed().as_secs_f64());
    let mut rng_v = rand::rngs::StdRng::seed_from_u64(seed);
    let v2_acc = train_and_eval(&mut v2, &train_images, &train_labels,
        &test_images, &test_labels, n_train, n_epochs, "V2  ", &mut rng_v);
    let v2_m = v2.inspect().total_morphons;
    let v2_s = v2.inspect().total_synapses;
    eprintln!("  V2: {:.1}% | m={} s={} ({:.0}s)\n", v2_acc, v2_m, v2_s, t1.elapsed().as_secs_f64());

    // === DAMAGE (skipped in fast mode unless --damage-sweep) ===
    let (damaged_acc, recovery_acc, before, after, n_kill) = if !fast || damage_sweep {
        eprintln!("━━━ DAMAGE ━━━");
        let before = v2.morphons.len();
        let assoc: Vec<MorphonId> = v2.morphons.values()
            .filter(|m| m.cell_type == CellType::Associative).map(|m| m.id).collect();
        let n_kill = (assoc.len() as f64 * 0.3).ceil() as usize;
        let mut kr = rand::rngs::StdRng::seed_from_u64(seed + 999);
        let mut ids = assoc;
        ids.shuffle(&mut kr);
        for &id in ids.iter().take(n_kill) {
            v2.morphons.remove(&id);
            v2.topology.remove_morphon(id);
        }
        let damaged_acc = evaluate(&mut v2, &test_images, &test_labels, n_test);
        eprintln!("  Killed {}, {} → {} morphons, acc={:.1}%", n_kill, before, v2.morphons.len(), damaged_acc);

        // Enable lifecycle for recovery
        v2.config.lifecycle = LifecycleConfig {
            division: true, differentiation: true, fusion: false,
            apoptosis: false, migration: true,
        };
        v2.config.scheduler.slow_period = 200;
        v2.config.scheduler.glacial_period = 500;
        let recovery_acc = train_and_eval(&mut v2, &train_images, &train_labels,
            &test_images, &test_labels, recovery_n, 1, "RECV", &mut rng_v);
        let after = v2.morphons.len();
        eprintln!("  Recovery: {:.1}% | morphons: {}\n", recovery_acc, after);
        (damaged_acc, recovery_acc, before, after, n_kill)
    } else { (0.0, 0.0, v2_m, v2_m, 0) };

    // === SUMMARY ===
    if fast {
        eprintln!("╔═══════════════════════════════════════════╗");
        eprintln!("║  V2 (fast):     {:>5.1}%                    ║", v2_acc);
        eprintln!("║  Morphons: {}  Synapses: {}{}║", v2_m, v2_s,
            " ".repeat(22 - format!("{}  Synapses: {}", v2_m, v2_s).len().min(22)));
        eprintln!("╚═══════════════════════════════════════════╝");
    } else {
        eprintln!("╔═══════════════════════════════════════════╗");
        eprintln!("║  Baseline:      {:>5.1}%                    ║", base_acc);
        eprintln!("║  V2:            {:>5.1}%  ({:+.1}pp)            ║", v2_acc, v2_acc - base_acc);
        eprintln!("║  Post-damage:   {:>5.1}%                    ║", damaged_acc);
        eprintln!("║  Post-recovery: {:>5.1}%  ({:+.1}pp)            ║", recovery_acc, recovery_acc - damaged_acc);
        eprintln!("║  Morphons: {} → {} → {}{}║", before, before - n_kill, after,
            " ".repeat(27 - format!("{} → {} → {}", before, before - n_kill, after).len()));
        eprintln!("╚═══════════════════════════════════════════╝");
    }

    // Save
    let version = env!("CARGO_PKG_VERSION");
    let results = json!({
        "benchmark": "mnist_v2_supervised", "version": version,
        "baseline_acc": base_acc, "v2_acc": v2_acc,
        "damaged_acc": damaged_acc, "recovery_acc": recovery_acc,
        "v2_morphons": v2_m, "v2_synapses": v2_s,
    });
    let dir = format!("docs/benchmark_results/v{}", version);
    fs::create_dir_all(&dir).ok();
    let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    let path = format!("{}/mnist_v2_{}.json", dir, ts);
    fs::write(&path, serde_json::to_string_pretty(&results).unwrap()).unwrap();
    eprintln!("Saved to {}", path);
}
