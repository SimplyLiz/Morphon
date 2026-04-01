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
use rand::seq::SliceRandom;
use serde_json::json;
use std::fs;

const IMG_PIXELS: usize = 28 * 28; // 784
const NUM_CLASSES: usize = 10;

/// Zero-bias encoding: pixel intensity maps directly to [0, 3].
fn encode_pixels(raw: &[u8]) -> Vec<f64> {
    raw.iter().map(|&p| (p as f64 / 255.0) * 3.0).collect()
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
            medium_period: 1,
            slow_period: 20,
            glacial_period: 500,
            homeostasis_period: 10,
            memory_period: 50,
        },
        learning: LearningParams {
            tau_eligibility: 2.0,
            tau_trace: 5.0,
            a_plus: 2.0,
            a_minus: -2.0,
            tau_tag: 200.0,
            tag_threshold: 0.5,
            capture_threshold: 0.2,
            capture_rate: 0.5,
            weight_max: 2.0,
            weight_min: 0.01,
            alpha_reward: 0.5,
            alpha_novelty: 3.0,
            alpha_arousal: 0.0,
            alpha_homeostasis: 0.1,
        },
        morphogenesis: MorphogenesisParams {
            migration_rate: 0.05,
            max_morphons: 2000,
            ..Default::default()
        },
        homeostasis: HomeostasisParams::default(),
        lifecycle: LifecycleConfig::default(),
        metabolic: MetabolicConfig::default(),
        dt: 1.0,
        working_memory_capacity: 7,
        episodic_memory_capacity: 500,
    };
    System::new(config)
}

/// Classify using the analog readout (weighted sum of hidden potentials).
fn classify(system: &mut System, pixels: &[f64], steps: usize) -> usize {
    let outputs = system.process_steps(pixels, steps);
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
            n: usize, steps: usize) -> (f64, Vec<serde_json::Value>) {
    let mut cc = vec![0usize; 10];
    let mut ct = vec![0usize; 10];
    for i in 0..images.len().min(n) {
        let p = classify(system, &images[i], steps);
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
        "standard" => (2, 2000, 2, 2000, 500),
        _          => (1, 500,  1, 500,  200),
    };

    println!("=== MORPHON MNIST Benchmark — Two-Phase Learning [{}] ===\n", profile);
    println!("Loading MNIST from ./data/ ...");

    let mnist = MnistBuilder::new()
        .label_format_digit()
        .base_path("data")
        .training_set_length(10_000)
        .test_set_length(1_000)
        .finalize();

    let train_images: Vec<Vec<f64>> = (0..10_000)
        .map(|i| encode_pixels(&mnist.trn_img[i * IMG_PIXELS..(i + 1) * IMG_PIXELS]))
        .collect();
    let train_labels: Vec<usize> = mnist.trn_lbl.iter().map(|&l| l as usize).collect();
    let test_images: Vec<Vec<f64>> = (0..1_000)
        .map(|i| encode_pixels(&mnist.tst_img[i * IMG_PIXELS..(i + 1) * IMG_PIXELS]))
        .collect();
    let test_labels: Vec<usize> = mnist.tst_lbl.iter().map(|&l| l as usize).collect();

    println!("Train: {}, Test: {}\n", train_images.len(), test_images.len());

    let mut system = create_system();
    let s = system.inspect();
    println!("Initial: {} morphons, {} synapses, {} in, {} out",
        s.total_morphons, s.total_synapses, system.input_size(), system.output_size());
    println!("Types: {:?}\n", s.differentiation_map);

    let mut rng = rand::rng();
    let steps = 5;

    // =========================================================================
    // PHASE 1: Unsupervised STDP + k-WTA (no labels, no readout)
    // The hidden layer self-organizes into feature detectors via STDP.
    // k-WTA ensures different neurons specialize on different patterns.
    // No analog readout training — hidden representations must stabilize first.
    // =========================================================================
    println!("--- PHASE 1: Unsupervised feature learning (STDP + k-WTA) ---\n");

    for epoch in 0..phase1_epochs {
        let mut indices: Vec<usize> = (0..train_images.len()).collect();
        indices.shuffle(&mut rng);

        for (bi, &idx) in indices.iter().take(phase1_samples).enumerate() {
            // Present image — STDP + k-WTA runs in the step() pipeline.
            // No labels, no readout training, no reward injection.
            // Just let the hidden layer see digits and self-organize.
            system.process_steps(&train_images[idx], steps);

            // Inject novelty to drive plasticity (CMA-ES finding: novelty is primary driver)
            system.inject_novelty(0.3);

            // One more step to let STDP propagate
            system.feed_input(&train_images[idx]);
            system.step();

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
    // PHASE 2: Supervised readout on frozen hidden representations
    // Enable analog readout and train it with the delta rule.
    // Hidden layer weights are NOT updated during this phase — only readout weights.
    // =========================================================================
    println!("--- PHASE 2: Supervised readout training (delta rule) ---\n");

    // Enable analog readout (creates random readout weights)
    system.enable_analog_readout();

    // Freeze hidden layer: disable STDP by setting medium_period very high
    // (no eligibility updates = no weight changes in the spiking network).
    // The k-WTA and spike propagation still run — only plasticity is frozen.
    system.config.scheduler.medium_period = 999999;

    for epoch in 0..phase2_epochs {
        let mut indices: Vec<usize> = (0..train_images.len()).collect();
        indices.shuffle(&mut rng);

        let mut correct = 0;
        let mut total = 0;

        for (bi, &idx) in indices.iter().take(phase2_samples).enumerate() {
            let pred = classify(&mut system, &train_images[idx], steps);
            let label = train_labels[idx];
            if pred == label { correct += 1; }
            total += 1;

            // Train readout with low learning rate — prevents early mode collapse.
            // With lr=0.1, the first class to get an advantage locks in within
            // ~100 samples. With lr=0.005, convergence is slow enough for all
            // 10 classes to compete for representation.
            system.train_readout(label, 0.005);

            if (bi + 1) % 500 == 0 {
                println!("  Phase2 Ep{} [{:>4}/{}] acc={:.1}%",
                    epoch + 1, bi + 1, phase2_samples,
                    correct as f64 / total as f64 * 100.0);
            }
        }

        // Test
        println!("\n  Test after Phase2 Epoch {}:", epoch + 1);
        let (test_acc, _) = evaluate(&mut system, &test_images, &test_labels, test_n, steps);
        println!("  => test={:.1}%\n", test_acc);
    }

    // Final evaluation with full per-class breakdown
    println!("=== Final Per-Class Test Accuracy ===");
    let (test_acc, per_class) = evaluate(&mut system, &test_images, &test_labels, test_n, steps);

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
