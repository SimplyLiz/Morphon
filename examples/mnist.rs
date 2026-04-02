//! MNIST Benchmark — Two-phase learning (Diehl & Cook 2015 approach).
//!
//! Phase 1: Unsupervised STDP + k-WTA in the hidden layer. No labels used.
//!          Hidden neurons self-organize into digit-selective feature detectors.
//! Phase 2: Supervised analog readout trained on frozen hidden representations.
//!          Delta rule learns to map hidden activity patterns → digit classes.
//!
//! Setup: Download MNIST files to ./data/ (unzipped)
//! Run: cargo run --example mnist --release
//! Run: cargo run --example mnist --release -- --standard
//! Run: cargo run --example mnist --release -- --extended

use mnist::MnistBuilder;
use morphon_core::developmental::DevelopmentalConfig;
use morphon_core::homeostasis::HomeostasisParams;
use morphon_core::learning::LearningParams;
use morphon_core::morphogenesis::MorphogenesisParams;
use morphon_core::morphon::MetabolicConfig;
use morphon_core::scheduler::SchedulerConfig;
use morphon_core::system::{System, SystemConfig};
use morphon_core::types::LifecycleConfig;
use rand::Rng;
use rand::seq::SliceRandom;
use serde_json::json;
use std::fs;

const IMG_PIXELS: usize = 28 * 28; // 784
const NUM_CLASSES: usize = 10;
/// Simulation steps per image — Poisson spike trains need time for k-WTA
/// to select winners. Diehl & Cook use 350ms; we use 50 steps as a start.
const STEPS_PER_IMAGE: usize = 30; // reduced — fewer steps = sharper k-WTA selection

/// Convert raw pixels to firing rate probabilities [0, 1].
/// Each pixel's value becomes the probability of spiking on any given step.
fn pixel_rates(raw: &[u8]) -> Vec<f64> {
    raw.iter().map(|&p| p as f64 / 255.0).collect()
}

/// Generate a single Poisson spike-train frame from firing rates.
/// Each input neuron fires (value=3.0) or is silent (value=0.0)
/// with probability proportional to the pixel intensity.
fn poisson_frame(rates: &[f64], rng: &mut impl Rng) -> Vec<f64> {
    rates.iter().map(|&r| {
        if rng.random_range(0.0..1.0) < r { 3.0 } else { 0.0 }
    }).collect()
}

/// Present an image as Poisson spike trains over N steps.
/// Resets associative+motor potentials first (Diehl & Cook: 50ms rest between images).
/// This ensures each image competes fresh — otherwise the same neurons
/// win every image due to accumulated potential from previous inputs.
fn present_image(system: &mut System, rates: &[f64], steps: usize, rng: &mut impl Rng) {
    // Inter-image reset — clear the slate so k-WTA selects based on THIS image
    for m in system.morphons.values_mut() {
        if m.cell_type != morphon_core::CellType::Sensory {
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

fn create_system() -> System {
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
            medium_period: 1,   // STDP every step — Poisson needs frequent updates
            slow_period: 1000,  // structural changes very infrequent
            glacial_period: 5000,
            homeostasis_period: 30, // normalization once per image
            memory_period: 100,
        },
        learning: LearningParams {
            tau_eligibility: 5.0,   // slightly longer for Poisson integration
            tau_trace: 8.0,         // wider STDP window for spike train patterns
            a_plus: 1.0,            // moderate STDP — weight-dependent scaling in learning.rs
            a_minus: -0.8,          // strong LTD for competitive learning (Diehl & Cook)
            tau_tag: 200.0,
            tag_threshold: 0.5,
            capture_threshold: 10.0, // disable consolidation during unsupervised phase
            capture_rate: 0.5,
            weight_max: 1.5,        // tight bounds — prevents weight explosion
            weight_min: 0.01,
            alpha_reward: 0.3,      // mild — not the primary driver for MNIST
            alpha_novelty: 1.0,     // novelty drives plasticity (Diehl & Cook)
            alpha_arousal: 0.0,
            alpha_homeostasis: 0.1,
            transmitter_potentiation: 0.0005, // gentler for large network
            heterosynaptic_depression: 0.001, tag_accumulation_rate: 0.3,
        },
        morphogenesis: MorphogenesisParams {
            migration_rate: 0.05,
            max_morphons: Some(2000),
            ..Default::default()
        },
        homeostasis: HomeostasisParams {
            kwta_fraction: 0.01, // ~3-5 winners — forces class-selective specialization
            ..Default::default()
        },
        lifecycle: LifecycleConfig {
            division: false,     // fixed topology for STDP self-organization
            differentiation: true,
            fusion: false,
            apoptosis: false,    // keep all neurons — we need diversity
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

/// Classify using the analog readout (weighted sum of hidden potentials).
fn classify(system: &mut System, rates: &[f64], rng: &mut impl Rng) -> usize {
    present_image(system, rates, STEPS_PER_IMAGE, rng);
    let outputs = system.read_output();
    if outputs.len() < NUM_CLASSES { return 0; }
    outputs.iter()
        .take(NUM_CLASSES)
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Evaluate test accuracy and per-class breakdown.
fn evaluate(system: &mut System, images: &[Vec<f64>], labels: &[usize],
            n: usize, rng: &mut impl Rng) -> (f64, Vec<serde_json::Value>) {
    let mut cc = vec![0usize; 10];
    let mut ct = vec![0usize; 10];
    for i in 0..images.len().min(n) {
        let p = classify(system, &images[i], rng);
        ct[labels[i]] += 1;
        if p == labels[i] { cc[labels[i]] += 1; }
    }
    let total_correct: usize = cc.iter().sum();
    let total_n = images.len().min(n);
    let acc = total_correct as f64 / total_n as f64 * 100.0;

    let mut per_class = Vec::new();
    for c in 0..10 {
        if ct[c] > 0 {
            let class_acc = cc[c] as f64 / ct[c] as f64 * 100.0;
            println!("  {}: {:.1}% ({}/{})", c, class_acc, cc[c], ct[c]);
            per_class.push(json!({"digit": c, "accuracy": class_acc, "correct": cc[c], "total": ct[c]}));
        }
    }
    (acc, per_class)
}

fn parse_profile() -> &'static str {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--extended") { "extended" }
    else if args.iter().any(|a| a == "--standard") { "standard" }
    else { "quick" }
}

fn main() {
    let profile = parse_profile();
    let (phase1_epochs, phase1_samples, phase2_epochs, phase2_samples, test_n) = match profile {
        "extended" => (3, 5000, 3, 5000, 1000),
        "standard" => (2, 3000, 2, 3000, 500),
        _          => (1, 1000, 1, 1000, 200),
    };

    println!("=== MORPHON MNIST Benchmark — Two-Phase Learning [{}] ===\n", profile);
    println!("Loading MNIST from ./data/ ...");

    let mnist = MnistBuilder::new()
        .label_format_digit()
        .base_path("data")
        .training_set_length(10_000)
        .test_set_length(1_000)
        .finalize();

    // Store firing rate probabilities [0, 1] per pixel (Poisson encoding)
    let train_images: Vec<Vec<f64>> = (0..10_000)
        .map(|i| pixel_rates(&mnist.trn_img[i * IMG_PIXELS..(i + 1) * IMG_PIXELS]))
        .collect();
    let train_labels: Vec<usize> = mnist.trn_lbl.iter().map(|&l| l as usize).collect();
    let test_images: Vec<Vec<f64>> = (0..1_000)
        .map(|i| pixel_rates(&mnist.tst_img[i * IMG_PIXELS..(i + 1) * IMG_PIXELS]))
        .collect();
    let test_labels: Vec<usize> = mnist.tst_lbl.iter().map(|&l| l as usize).collect();

    println!("Train: {}, Test: {}\n", train_images.len(), test_images.len());

    let mut system = create_system();
    let s = system.inspect();
    println!("Initial: {} morphons, {} synapses, {} in, {} out",
        s.total_morphons, s.total_synapses, system.input_size(), system.output_size());
    println!("Types: {:?}\n", s.differentiation_map);

    let mut rng = rand::rng();

    // =========================================================================
    // PHASE 1: Unsupervised STDP + k-WTA with Poisson spike trains.
    // Each image is presented as a Poisson process over STEPS_PER_IMAGE steps.
    // No labels, no readout. Hidden neurons self-organize via STDP + k-WTA.
    // =========================================================================
    println!("--- PHASE 1: Unsupervised feature learning (Poisson + STDP + k-WTA, {} steps/image) ---\n",
        STEPS_PER_IMAGE);

    for epoch in 0..phase1_epochs {
        let mut indices: Vec<usize> = (0..train_images.len()).collect();
        indices.shuffle(&mut rng);

        for (bi, &idx) in indices.iter().take(phase1_samples).enumerate() {
            // Present image as Poisson spike trains — k-WTA selects winners
            present_image(&mut system, &train_images[idx], STEPS_PER_IMAGE, &mut rng);

            // Inject novelty to drive plasticity
            system.inject_novelty(0.2);

            if (bi + 1) % 500 == 0 {
                let s = system.inspect();
                let diag = system.diagnostics();
                println!("  Phase1 Ep{} [{:>4}/{}] m={} s={} fr={:.3} pe={:.3} | {}",
                    epoch + 1, bi + 1, phase1_samples,
                    s.total_morphons, s.total_synapses, s.firing_rate, s.avg_prediction_error,
                    diag.summary());
            }
        }

        let s = system.inspect();
        println!("Phase1 Epoch {} complete | m={} s={} fr={:.3}\n",
            epoch + 1, s.total_morphons, s.total_synapses, s.firing_rate);
    }

    // =========================================================================
    // PHASE 2: Post-hoc labeling (Diehl & Cook 2015)
    // Present labeled images, record which hidden neurons fire for each class.
    // Assign each neuron to the class it responds to most frequently.
    // At test time, classify by which class's assigned neurons are most active.
    // No gradient, no delta rule, no mode collapse — pure statistics.
    // =========================================================================
    println!("--- PHASE 2: Post-hoc neuron labeling (Diehl & Cook style) ---\n");

    // Freeze hidden layer
    system.config.scheduler.medium_period = 999999;

    // Collect associative morphon IDs
    let assoc_ids: Vec<morphon_core::MorphonId> = system.morphons.values()
        .filter(|m| m.cell_type == morphon_core::CellType::Associative
            || m.cell_type == morphon_core::CellType::Stem)
        .map(|m| m.id)
        .collect();

    // For each hidden neuron, count how many times it fired per class
    let mut neuron_class_counts: std::collections::HashMap<morphon_core::MorphonId, Vec<usize>> =
        assoc_ids.iter().map(|&id| (id, vec![0usize; NUM_CLASSES])).collect();

    let label_samples = phase2_samples.min(train_images.len());
    let mut indices: Vec<usize> = (0..train_images.len()).collect();
    indices.shuffle(&mut rng);

    println!("  Labeling {} samples...", label_samples);
    for (bi, &idx) in indices.iter().take(label_samples).enumerate() {
        present_image(&mut system, &train_images[idx], STEPS_PER_IMAGE, &mut rng);
        let label = train_labels[idx];

        // Record which hidden neurons fired during this image
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
            println!("  [{}/{}] labeled", bi + 1, label_samples);
        }
    }

    // Assign each neuron to its most-responded class
    let neuron_labels: std::collections::HashMap<morphon_core::MorphonId, usize> =
        neuron_class_counts.iter()
            .filter_map(|(&id, counts)| {
                let total: usize = counts.iter().sum();
                if total == 0 { return None; }
                let best_class = counts.iter().enumerate()
                    .max_by_key(|&(_, c)| *c)
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                Some((id, best_class))
            })
            .collect();

    let labeled_count = neuron_labels.len();
    let mut class_neuron_counts = vec![0usize; NUM_CLASSES];
    for &c in neuron_labels.values() { class_neuron_counts[c] += 1; }
    println!("  {} neurons labeled: {:?}\n", labeled_count, class_neuron_counts);

    // Test: classify by which class's neurons are most active
    println!("--- Testing with post-hoc labels ---\n");
    let mut cc = vec![0usize; NUM_CLASSES];
    let mut ct = vec![0usize; NUM_CLASSES];
    for i in 0..test_images.len().min(test_n) {
        present_image(&mut system, &test_images[i], STEPS_PER_IMAGE, &mut rng);

        // Vote: sum activity of neurons assigned to each class
        let mut class_votes = vec![0.0f64; NUM_CLASSES];
        for (&aid, &assigned_class) in &neuron_labels {
            if let Some(m) = system.morphons.get(&aid) {
                let activity = if m.fired { 1.0 } else { m.activity_history.mean() };
                class_votes[assigned_class] += activity;
            }
        }

        let pred = class_votes.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        ct[test_labels[i]] += 1;
        if pred == test_labels[i] { cc[test_labels[i]] += 1; }
    }

    // Per-class results from the post-hoc classification above
    println!("=== Final Per-Class Test Accuracy ===");
    let mut per_class = Vec::new();
    let total_correct: usize = cc.iter().sum();
    let total_tested: usize = ct.iter().sum();
    let test_acc = if total_tested > 0 { total_correct as f64 / total_tested as f64 * 100.0 } else { 0.0 };
    for c in 0..NUM_CLASSES {
        if ct[c] > 0 {
            let class_acc = cc[c] as f64 / ct[c] as f64 * 100.0;
            println!("  {}: {:.1}% ({}/{})", c, class_acc, cc[c], ct[c]);
            per_class.push(json!({"digit": c, "accuracy": class_acc, "correct": cc[c], "total": ct[c]}));
        }
    }

    println!("\n=== Final ===");
    let s = system.inspect();
    let diag = system.diagnostics();
    println!("Test accuracy: {:.1}%", test_acc);
    println!("Morphons: {} | Synapses: {} | Clusters: {} | Gen: {} | FR: {:.3}",
        s.total_morphons, s.total_synapses, s.fused_clusters, s.max_generation, s.firing_rate);
    println!("Learning: {}", diag.summary());

    // Save results
    let version = env!("CARGO_PKG_VERSION");
    let results = json!({
        "benchmark": "mnist",
        "profile": profile,
        "method": "two-phase (unsupervised STDP + supervised readout)",
        "version": version,
        "timestamp": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
        "phase1": { "epochs": phase1_epochs, "samples_per_epoch": phase1_samples },
        "phase2": { "epochs": phase2_epochs, "samples_per_epoch": phase2_samples },
        "results": { "test_accuracy": test_acc, "per_class": per_class },
        "system": {
            "morphons": s.total_morphons, "synapses": s.total_synapses,
            "clusters": s.fused_clusters, "generation": s.max_generation,
            "firing_rate": s.firing_rate, "prediction_error": s.avg_prediction_error,
        },
        "diagnostics": {
            "weight_mean": diag.weight_mean, "weight_std": diag.weight_std,
            "active_tags": diag.active_tags, "total_captures": diag.total_captures,
            "consolidated": diag.consolidated_count,
        },
    });

    let dir = format!("docs/benchmark_results/v{}", version);
    fs::create_dir_all(&dir).ok();
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    let run_path = format!("{}/mnist_{}.json", dir, ts);
    let latest_path = format!("{}/mnist_latest.json", dir);
    let json_str = serde_json::to_string_pretty(&results).unwrap();
    fs::write(&run_path, &json_str).unwrap();
    fs::write(&latest_path, &json_str).unwrap();
    println!("\nResults saved to {}", run_path);
}
