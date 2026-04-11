//! Anomaly Detection Benchmark — MI system detects outliers in synthetic time series.
//!
//! The system learns normal patterns in a periodic time series, then flags
//! anomalies via elevated prediction error. No explicit training/test split —
//! the system continuously adapts and learns what "normal" looks like.
//!
//! Run: cargo run --example anomaly --release

use morphon_core::developmental::DevelopmentalConfig;
use morphon_core::learning::LearningParams;
use morphon_core::morphogenesis::MorphogenesisParams;
use morphon_core::morphon::MetabolicConfig;
use morphon_core::scheduler::SchedulerConfig;
use morphon_core::system::{System, SystemConfig};
use morphon_core::types::LifecycleConfig;
use rand::Rng;
use serde_json::json;
use std::fs;

const WINDOW_SIZE: usize = 8;
const NUM_OUTPUTS: usize = 1;

/// Generate a synthetic time series: periodic signal + noise, with injected anomalies.
/// Returns (values, ground_truth_is_anomaly).
fn generate_series(length: usize, anomaly_rate: f64, rng: &mut impl Rng) -> (Vec<f64>, Vec<bool>) {
    let mut values = Vec::with_capacity(length);
    let mut is_anomaly = Vec::with_capacity(length);

    for t in 0..length {
        let base = (t as f64 * 0.1).sin() * 0.5 + 0.5; // periodic in [0, 1]
        let noise = rng.random_range(-0.05..0.05);

        let anomalous = rng.random_range(0.0..1.0) < anomaly_rate;
        let value = if anomalous {
            if rng.random_bool(0.5) {
                base + rng.random_range(0.8..1.5) // spike
            } else {
                base - rng.random_range(0.8..1.5) // drop
            }
        } else {
            base + noise
        };

        values.push(value);
        is_anomaly.push(anomalous);
    }

    (values, is_anomaly)
}

fn create_system() -> System {
    let config = SystemConfig {
        developmental: DevelopmentalConfig {
            seed_size: 40,
            dimensions: 4,
            initial_connectivity: 0.2,
            proliferation_rounds: 2,
            target_input_size: Some(WINDOW_SIZE),
            target_output_size: Some(NUM_OUTPUTS),
            ..DevelopmentalConfig::hippocampal()
        },
        scheduler: SchedulerConfig {
            medium_period: 1,
            slow_period: 10,
            glacial_period: 100,
            homeostasis_period: 10,
            memory_period: 25,
        },
        learning: LearningParams {
            tau_eligibility: 10.0,
            tau_trace: 10.0,
            a_plus: 1.0,
            a_minus: -1.0,
            tau_tag: 300.0,
            tag_threshold: 0.3,
            capture_threshold: 0.3,
            capture_rate: 0.15,
            weight_max: 3.0,
            weight_min: 0.01,
            alpha_reward: 1.5,
            alpha_novelty: 1.0,
            alpha_arousal: 0.5,
            alpha_homeostasis: 0.2,
            transmitter_potentiation: 0.001,
            heterosynaptic_depression: 0.002, tag_accumulation_rate: 0.3,
            ..Default::default()
        },
        morphogenesis: MorphogenesisParams {
            max_morphons: Some(200),
            ..Default::default()
        },
        homeostasis: Default::default(),
        lifecycle: LifecycleConfig::default(),
        metabolic: MetabolicConfig::default(),
        dt: 1.0,
        working_memory_capacity: 7,
        episodic_memory_capacity: 200,
        ..Default::default()
    };
    System::new(config)
}

fn parse_profile() -> &'static str {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--extended") { "extended" }
    else if args.iter().any(|a| a == "--standard") { "standard" }
    else { "quick" }
}

fn main() {
    let profile = parse_profile();
    let (series_len, learning_phase) = match profile {
        "extended" => (5000, 500),
        "standard" => (2000, 500),
        _          => (800, 300),  // quick (default)
    };

    println!("=== MORPHON Anomaly Detection Benchmark [{}] ===\n", profile);

    let mut rng = rand::rng();
    let anomaly_rate = 0.05;

    let (series, ground_truth) = generate_series(series_len, anomaly_rate, &mut rng);
    let total_anomalies = ground_truth.iter().filter(|&&a| a).count();
    println!(
        "Series: {} points, {} anomalies ({:.1}%)\n",
        series_len,
        total_anomalies,
        total_anomalies as f64 / series_len as f64 * 100.0
    );

    let mut system = create_system();
    let s = system.inspect();
    println!(
        "Initial: {} morphons, {} synapses, {} in, {} out",
        s.total_morphons,
        s.total_synapses,
        system.input_size(),
        system.output_size()
    );

    // Warm up with first window
    let warmup: Vec<f64> = series[..WINDOW_SIZE]
        .iter()
        .map(|&v| v.clamp(0.0, 2.0))
        .collect();
    for _ in 0..20 {
        system.process_steps(&warmup, 3);
    }

    let mut pe_history: Vec<f64> = Vec::new();
    let mut detections: Vec<(usize, f64, bool)> = Vec::new(); // (index, PE, is_true_anomaly)

    println!("\n--- Learning phase ({} steps) ---", learning_phase);

    for t in WINDOW_SIZE..series_len {
        let window: Vec<f64> = series[t - WINDOW_SIZE..t]
            .iter()
            .map(|&v| v.clamp(0.0, 2.0))
            .collect();

        let _output = system.process_steps(&window, 3);
        let stats = system.inspect();
        let pe = stats.avg_prediction_error;
        pe_history.push(pe);

        // Reward/novelty relative to recent PE baseline (not absolute thresholds)
        let recent_start = pe_history.len().saturating_sub(50);
        let baseline_pe = if pe_history.len() > 10 {
            pe_history[recent_start..].iter().sum::<f64>() / (pe_history.len() - recent_start) as f64
        } else {
            pe
        };

        if pe < baseline_pe * 0.8 {
            // PE is below recent average — system is predicting well
            system.inject_reward(0.5);
        } else if pe > baseline_pe * 1.5 {
            // PE spike — something unexpected
            system.inject_novelty(0.5);
            system.inject_arousal(0.3);
        }

        if t >= learning_phase {
            // Detection phase: adaptive threshold from recent PE distribution
            let recent_start = pe_history.len().saturating_sub(100);
            let recent_pe: &[f64] = &pe_history[recent_start..];
            let mean_pe = recent_pe.iter().sum::<f64>() / recent_pe.len() as f64;
            let std_pe = (recent_pe
                .iter()
                .map(|p| (p - mean_pe).powi(2))
                .sum::<f64>()
                / recent_pe.len() as f64)
                .sqrt();
            let threshold = mean_pe + 2.0 * std_pe;

            let detected = pe > threshold;
            let is_anomaly = ground_truth[t];
            detections.push((t, pe, is_anomaly));

            if detected && is_anomaly {
                println!(
                    "  t={:>4} PE={:.4} thr={:.4} [TP] val={:.2}",
                    t,
                    pe,
                    threshold,
                    series[t - 1]
                );
            }
        }

        if (t + 1) % 500 == 0 {
            let s = system.inspect();
            let diag = system.diagnostics();
            println!(
                "  Step {:>4} | m={} s={} fr={:.3} pe={:.4} | {}",
                t + 1,
                s.total_morphons,
                s.total_synapses,
                s.firing_rate,
                s.avg_prediction_error,
                diag.summary()
            );
        }
    }

    // === Calculate detection metrics ===
    println!("\n--- Detection Results ---\n");

    let detection_start = learning_phase - WINDOW_SIZE;
    let det_pe: &[f64] = &pe_history[detection_start..];
    let mean_pe = det_pe.iter().sum::<f64>() / det_pe.len() as f64;
    let std_pe = (det_pe
        .iter()
        .map(|p| (p - mean_pe).powi(2))
        .sum::<f64>()
        / det_pe.len() as f64)
        .sqrt();

    let mut sigma_results = Vec::new();
    for sigma in [1.5, 2.0, 2.5, 3.0] {
        let threshold = mean_pe + sigma * std_pe;

        let mut tp = 0_u64;
        let mut fp = 0_u64;
        let mut tn = 0_u64;
        let mut fn_ = 0_u64;

        for &(_t, pe, is_anomaly) in &detections {
            let detected = pe > threshold;
            match (detected, is_anomaly) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, true) => fn_ += 1,
                (false, false) => tn += 1,
            }
        }

        let precision = if tp + fp > 0 {
            tp as f64 / (tp + fp) as f64
        } else {
            0.0
        };
        let recall = if tp + fn_ > 0 {
            tp as f64 / (tp + fn_) as f64
        } else {
            0.0
        };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        println!(
            "  {:.1}\u{03c3} (thr={:.4}): P={:.1}% R={:.1}% F1={:.3} (TP={} FP={} FN={} TN={})",
            sigma,
            threshold,
            precision * 100.0,
            recall * 100.0,
            f1,
            tp,
            fp,
            fn_,
            tn
        );

        sigma_results.push(json!({
            "sigma": sigma,
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp, "fp": fp, "fn": fn_, "tn": tn,
        }));
    }

    println!("\n=== Final ===");
    let s = system.inspect();
    let diag = system.diagnostics();
    println!(
        "Morphons: {} | Synapses: {} | Clusters: {} | FR: {:.3}",
        s.total_morphons, s.total_synapses, s.fused_clusters, s.firing_rate
    );
    println!("Learning: {}", diag.summary());
    println!("Firing:   {}", diag.firing_summary());

    // Save benchmark results
    let version = env!("CARGO_PKG_VERSION");
    let results = json!({
        "benchmark": "anomaly",
        "profile": profile,
        "version": version,
        "timestamp": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
        "series_length": series_len,
        "anomaly_rate": anomaly_rate,
        "total_anomalies": total_anomalies,
        "learning_phase": learning_phase,
        "results": sigma_results,
        "system": {
            "morphons": s.total_morphons,
            "synapses": s.total_synapses,
            "clusters": s.fused_clusters,
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
    let run_path = format!("{}/anomaly_{}.json", dir, ts);
    let latest_path = format!("{}/anomaly_latest.json", dir);
    let json_str = serde_json::to_string_pretty(&results).unwrap();
    fs::write(&run_path, &json_str).unwrap();
    fs::write(&latest_path, &json_str).unwrap();
    println!("\nResults saved to {}", run_path);
}
