//! MNIST Benchmark — MI system learns digit classification through self-organization.
//!
//! Full 28x28=784 pixel input, 10 output classes.
//! Uses target_input_size/target_output_size for exact I/O matching.
//!
//! Setup: Download MNIST files to ./data/ (unzipped)
//! Run: cargo run --example mnist --release

use mnist::MnistBuilder;
use morphon_core::developmental::DevelopmentalConfig;
use morphon_core::homeostasis::HomeostasisParams;
use morphon_core::learning::LearningParams;
use morphon_core::morphogenesis::MorphogenesisParams;
use morphon_core::scheduler::SchedulerConfig;
use morphon_core::system::{System, SystemConfig};
use morphon_core::types::LifecycleConfig;
use rand::seq::SliceRandom;
use serde_json::json;
use std::fs;

const IMG_PIXELS: usize = 28 * 28; // 784
const NUM_CLASSES: usize = 10;

/// Zero-bias encoding: pixel intensity maps directly to [0, 3].
/// Dark pixels (background) stay at 0 — only active pixels carry signal.
/// This preserves digit-specific spatial patterns without bias masking.
fn encode_pixels(raw: &[u8]) -> Vec<f64> {
    raw.iter().map(|&p| (p as f64 / 255.0) * 3.0).collect()
}

fn create_system() -> System {
    let config = SystemConfig {
        developmental: DevelopmentalConfig {
            // Large enough for 784 inputs + 10 outputs + interior
            seed_size: 500,
            dimensions: 6,
            initial_connectivity: 0.02, // sparse — 500^2 * 0.02 = ~5000 connections
            proliferation_rounds: 1,
            target_input_size: Some(IMG_PIXELS),  // exactly 784 sensory morphons
            target_output_size: Some(NUM_CLASSES), // exactly 10 motor morphons
            ..DevelopmentalConfig::cortical()
        },
        scheduler: SchedulerConfig {
            medium_period: 1,
            slow_period: 20,
            glacial_period: 500,
            homeostasis_period: 10,
            memory_period: 50,
        },
        // Adapted from CMA-ES classify_tiny findings, scaled for 10-class MNIST.
        // Lower STDP magnitudes to prevent early mode collapse on one class.
        learning: LearningParams {
            tau_eligibility: 2.0,
            tau_trace: 5.0,
            a_plus: 2.0,           // gentler than CMA-ES ±5.0 (10 classes need more exploration)
            a_minus: -2.0,
            tau_tag: 200.0,
            tag_threshold: 0.5,
            capture_threshold: 0.2,
            capture_rate: 0.5,
            weight_max: 2.0,
            weight_min: 0.01,
            alpha_reward: 0.5,      // novelty-driven (CMA-ES finding)
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
        dt: 1.0,
        working_memory_capacity: 7,
        episodic_memory_capacity: 500,
    };
    System::new(config)
}

/// Classify an image. Returns predicted digit (0-9).
fn classify(system: &mut System, pixels: &[f64], steps: usize) -> usize {
    let outputs = system.process_steps(pixels, steps);
    if outputs.len() < NUM_CLASSES {
        return 0;
    }
    // Each of the 10 motor morphons represents a digit class
    outputs.iter()
        .take(NUM_CLASSES)
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn train_one(system: &mut System, pixels: &[f64], label: usize, steps: usize) -> bool {
    let pred = classify(system, pixels, steps);
    let correct = pred == label;

    // CMA-ES optimal recipe: teach_hidden is the PRIMARY mechanism.
    // Direct teaching signal to hidden layer (strength ~2.0) provides
    // credit assignment that reward broadcast cannot achieve.
    system.teach_hidden(label, 1.9);

    // Minimal contrastive reward (CMA-ES: reward=0.1, inhibit=0.0)
    system.reward_contrastive(label, 0.1, 0.0);

    // Novelty injection drives plasticity (CMA-ES: alpha_novelty=3.0)
    system.inject_novelty(0.3);

    // Let teaching signal propagate through weight updates
    for _ in 0..2 {
        system.feed_input(pixels);
        system.step();
    }

    correct
}

fn parse_profile() -> &'static str {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--extended") { "extended" }
    else if args.iter().any(|a| a == "--standard") { "standard" }
    else { "quick" }
}

fn main() {
    let profile = parse_profile();
    let (num_epochs, samples_per_epoch, test_eval_size) = match profile {
        "extended" => (5, 5000, 1000),
        "standard" => (3, 2000, 500),
        _          => (1, 500, 200),  // quick (default)
    };

    println!("=== MORPHON MNIST Benchmark (Full 784px) [{}] ===\n", profile);
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

    println!("Train: {}, Test: {}, Input: {}px, Output: {} classes\n",
        train_images.len(), test_images.len(), IMG_PIXELS, NUM_CLASSES);

    let mut system = create_system();
    let s = system.inspect();
    println!("Initial: {} morphons, {} synapses, {} in, {} out",
        s.total_morphons, s.total_synapses, system.input_size(), system.output_size());
    println!("Types: {:?}\n", s.differentiation_map);

    // No external warmup — System::new() handles developmental activity.
    // Input bias removed: zero-bias encoding preserves dynamic range.

    let mut rng = rand::rng();
    let steps_per_sample = 5;  // 784→371→10 needs more propagation time

    for epoch in 0..num_epochs {
        let mut indices: Vec<usize> = (0..train_images.len()).collect();
        indices.shuffle(&mut rng);

        let mut correct = 0;
        let mut total = 0;

        for (bi, &idx) in indices.iter().take(samples_per_epoch).enumerate() {
            let hit = train_one(&mut system, &train_images[idx], train_labels[idx], steps_per_sample);
            if hit { correct += 1; }
            total += 1;

            if (bi + 1) % 500 == 0 {
                let s = system.inspect();
                println!("  Ep {} [{:>4}/{}] acc={:.1}% m={} s={} fr={:.3} pe={:.3}",
                    epoch + 1, bi + 1, samples_per_epoch,
                    correct as f64 / total as f64 * 100.0,
                    s.total_morphons, s.total_synapses, s.firing_rate, s.avg_prediction_error);
            }
        }

        // Test
        let mut test_correct = 0;
        for i in 0..test_images.len().min(test_eval_size) {
            if classify(&mut system, &test_images[i], steps_per_sample) == test_labels[i] {
                test_correct += 1;
            }
        }
        let test_n = test_images.len().min(test_eval_size);
        let s = system.inspect();
        println!("Epoch {} | train={:.1}% | test={:.1}% | m={} s={} gen={}\n",
            epoch + 1,
            correct as f64 / total as f64 * 100.0,
            test_correct as f64 / test_n as f64 * 100.0,
            s.total_morphons, s.total_synapses, s.max_generation);
    }

    // Per-class accuracy
    println!("Per-class test accuracy:");
    let mut cc = vec![0usize; 10];
    let mut ct = vec![0usize; 10];
    for i in 0..test_images.len().min(test_eval_size) {
        let p = classify(&mut system, &test_images[i], steps_per_sample);
        ct[test_labels[i]] += 1;
        if p == test_labels[i] { cc[test_labels[i]] += 1; }
    }
    let mut per_class = Vec::new();
    for c in 0..10 {
        if ct[c] > 0 {
            let acc = cc[c] as f64 / ct[c] as f64 * 100.0;
            println!("  {}: {:.1}% ({}/{})", c, acc, cc[c], ct[c]);
            per_class.push(json!({"digit": c, "accuracy": acc, "correct": cc[c], "total": ct[c]}));
        }
    }

    // Save benchmark results
    let version = env!("CARGO_PKG_VERSION");
    let s = system.inspect();
    let diag = system.diagnostics();
    let test_n = test_images.len().min(test_eval_size);
    let test_correct: usize = cc.iter().sum();
    let test_acc = test_correct as f64 / test_n as f64 * 100.0;
    let results = json!({
        "benchmark": "mnist",
        "profile": profile,
        "version": version,
        "timestamp": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
        "train_samples": samples_per_epoch,
        "test_samples": test_n,
        "epochs": num_epochs,
        "results": {
            "test_accuracy": test_acc,
            "per_class": per_class,
        },
        "system": {
            "morphons": s.total_morphons,
            "synapses": s.total_synapses,
            "clusters": s.fused_clusters,
            "generation": s.max_generation,
            "firing_rate": s.firing_rate,
            "prediction_error": s.avg_prediction_error,
        },
        "diagnostics": {
            "weight_mean": diag.weight_mean,
            "weight_std": diag.weight_std,
            "active_tags": diag.active_tags,
            "total_captures": diag.total_captures,
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
