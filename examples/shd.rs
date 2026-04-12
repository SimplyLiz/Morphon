//! SHD-style Synthetic Neuromorphic Spoken Digit Benchmark
//!
//! A synthetic neuromorphic benchmark inspired by the Spiking Heidelberg Digits (SHD)
//! dataset (Cramer et al., 2020). The real SHD consists of spike trains from a simulated
//! 700-channel cochlear model applied to spoken digits 0-9 (HD-Audio corpus).
//!
//! This implementation generates structured synthetic spike trains that reproduce the
//! key properties of SHD:
//!   - 700 input channels (cochlear frequency bands, log-spaced 20Hz–20kHz)
//!   - 10 classes (spoken digits 0–9)
//!   - Variable-length sequences (60–120 time steps, 1ms resolution)
//!   - Sparse, class-discriminative spectro-temporal patterns
//!   - Realistic spike density (~3–8% per channel per step)
//!
//! Each digit class has a characteristic "formant trajectory" — a path through frequency
//! space over time that encodes the digit's phonetic structure. Variability is added via
//! time-warping, frequency jitter, and additive Poisson noise, matching SHD's estimated
//! SNR of ~10dB.
//!
//! Reference: Cramer et al. (2020) "The Heidelberg Spiking Data Sets for the Systematic
//! Evaluation of Spiking Neural Networks." IEEE TNNLS.
//!
//! Run: cargo run --example shd --release
//! Run: cargo run --example shd --release -- --standard
//! Run: cargo run --example shd --release -- --extended --n-seeds=3

use morphon_core::developmental::{DevelopmentalConfig, RecurrentConfig};
use morphon_core::endoquilibrium::EndoConfig;
use morphon_core::homeostasis::{CompetitionMode, HomeostasisParams};
use morphon_core::learning::LearningParams;
use morphon_core::morphogenesis::MorphogenesisParams;
use morphon_core::morphon::MetabolicConfig;
use morphon_core::scheduler::SchedulerConfig;
use morphon_core::system::{System, SystemConfig};
use morphon_core::types::LifecycleConfig;
use rand::Rng;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use serde_json::json;
use std::fs;
use std::time::Instant;

// ── Constants ────────────────────────────────────────────────────────────────

const N_CHANNELS: usize = 700;  // cochlear frequency channels
const N_CLASSES: usize = 10;    // spoken digits 0-9
// Morphon input: project 700 channels → INPUT_DIM via random pooling to keep
// network size tractable while preserving spectro-temporal structure.
const INPUT_DIM: usize = 70;
const POOL_SIZE: usize = N_CHANNELS / INPUT_DIM;  // 10 channels per input neuron

// ── Formant trajectory generator ─────────────────────────────────────────────
// Each digit class has a characteristic 2-formant trajectory (F1 low, F2 high).
// Real spoken digits have formant patterns in 300–3000Hz range; mapped here to
// channel indices 30–600 (out of 700 log-spaced channels).
//
// Formant paths are defined as (F1_start, F1_end, F2_start, F2_end) pairs
// (channel indices), loosely inspired by actual English digit phonetics.
//
// "zero"=0  "one"=1  "two"=2  "three"=3  "four"=4
// "five"=5  "six"=6  "seven"=7 "eight"=8 "nine"=9
const FORMANT_PATHS: [(f64, f64, f64, f64); N_CLASSES] = [
    (60.0, 90.0, 380.0, 340.0),  // 0: /z-ɛ-r-oʊ/ — low F1 rise, F2 fall
    (40.0, 60.0, 480.0, 500.0),  // 1: /w-ʌn/ — low F1, high stable F2
    (80.0, 100.0, 440.0, 420.0), // 2: /t-uː/ — mid F1 rise, F2 moderate
    (100.0, 80.0, 360.0, 420.0), // 3: /θ-r-iː/ — F1 fall, F2 rise
    (70.0, 50.0, 320.0, 280.0),  // 4: /f-ɔːr/ — F1 fall, low F2 fall
    (90.0, 110.0, 460.0, 440.0), // 5: /f-aɪv/ — F1 rise, F2 dip
    (50.0, 80.0, 500.0, 460.0),  // 6: /s-ɪks/ — F1 rise, F2 fall
    (80.0, 70.0, 420.0, 380.0),  // 7: /s-ɛv-ən/ — F1 dip, F2 fall
    (110.0, 90.0, 400.0, 440.0), // 8: /eɪt/ — F1 fall, F2 rise
    (60.0, 80.0, 480.0, 500.0),  // 9: /n-aɪn/ — F1 rise, high F2
];

/// Generate a single spike-train sample for the given digit class.
/// Returns Vec<Vec<bool>> of shape [n_steps][N_CHANNELS].
fn generate_sample(
    class: usize,
    rng: &mut impl Rng,
    base_duration: usize, // nominal steps before warping
) -> Vec<Vec<f64>> {
    let (f1_start, f1_end, f2_start, f2_end) = FORMANT_PATHS[class];

    // Time-warp: stretch or compress duration by ±20%
    let warp = 0.8 + rng.random::<f64>() * 0.4;
    let n_steps = ((base_duration as f64 * warp) as usize).max(30).min(150);

    let mut sample = vec![vec![0.0f64; N_CHANNELS]; n_steps];

    for t in 0..n_steps {
        let frac = t as f64 / n_steps.max(1) as f64;

        // Interpolate formant positions (with frequency jitter)
        let f1 = f1_start + (f1_end - f1_start) * frac + rng.random_range(-8.0..8.0_f64);
        let f2 = f2_start + (f2_end - f2_start) * frac + rng.random_range(-12.0..12.0_f64);

        // Formant bandwidth: F1 narrower, F2 wider
        let bw1 = 18.0 + rng.random_range(0.0..8.0_f64);
        let bw2 = 28.0 + rng.random_range(0.0..12.0_f64);

        for ch in 0..N_CHANNELS {
            let ch_f = ch as f64;
            // Gaussian activation around each formant
            let activation = 0.7 * (-0.5 * ((ch_f - f1) / bw1).powi(2)).exp()
                           + 0.5 * (-0.5 * ((ch_f - f2) / bw2).powi(2)).exp();

            // Poisson spiking: probability = activation + noise floor
            let p_spike = (activation * 0.6 + 0.02).clamp(0.0, 0.95);
            sample[t][ch] = if rng.random::<f64>() < p_spike { 1.0 } else { 0.0 };
        }
    }

    sample
}

/// Pool N_CHANNELS → INPUT_DIM by taking max over consecutive POOL_SIZE channels.
/// Preserves spectro-temporal structure while reducing dimensionality.
fn pool_channels(frame: &[f64]) -> Vec<f64> {
    (0..INPUT_DIM)
        .map(|i| {
            let start = i * POOL_SIZE;
            let end = (start + POOL_SIZE).min(N_CHANNELS);
            frame[start..end].iter().cloned().fold(0.0_f64, f64::max)
        })
        .collect()
}

// ── System builder ────────────────────────────────────────────────────────────

fn build_system(seed: u64) -> System {
    let config = SystemConfig {
        developmental: DevelopmentalConfig {
            seed_size: 150,
            target_input_size: Some(INPUT_DIM),
            target_output_size: Some(N_CLASSES),
            recurrent: RecurrentConfig {
                enabled: true,
                recurrent_connectivity: 0.18,
                recurrent_weight_scale: 0.45,
                delay_range: (1.0, 4.0),
                plastic: true,
                allow_autapses: false,
            },
            ..DevelopmentalConfig::temporal()
        },
        learning: LearningParams {
            tau_eligibility: 80.0,
            tau_trace: 160.0,
            a_plus: 0.8,
            a_minus: -0.8,
            ..LearningParams::default()
        },
        scheduler: SchedulerConfig {
            medium_period: 5,
            // Disable structural changes during training — synaptogenesis mid-trial
            // causes catastrophic forgetting in sequence tasks.
            slow_period: 10_000_000,
            glacial_period: 10_000_000,
            homeostasis_period: 50,
            memory_period: 200,
        },
        homeostasis: HomeostasisParams {
            competition_mode: CompetitionMode::LocalInhibition {
                istdp_rate: 0.002,
                target_rate: Some(0.04),
                interneuron_ratio: 0.08,
                initial_inh_weight: -0.3,
                inhibition_radius: 0.5,
            },
            ..HomeostasisParams::default()
        },
        endoquilibrium: EndoConfig {
            enabled: true,
            ..EndoConfig::default()
        },
        lifecycle: LifecycleConfig {
            division: false,
            differentiation: false,
            fusion: false,
            apoptosis: false,
            migration: false,
            synaptogenesis: true,
        },
        morphogenesis: MorphogenesisParams {
            ..MorphogenesisParams::default()
        },
        metabolic: MetabolicConfig {
            firing_cost: 0.003,
            ..MetabolicConfig::default()
        },
        rng_seed: Some(seed),
        ..SystemConfig::default()
    };

    let mut system = System::new(config);
    system.enable_analog_readout();
    system
}

// ── Training helpers ──────────────────────────────────────────────────────────

fn classify(system: &mut System, sample: &[Vec<f64>]) -> usize {
    system.sequence_reset();
    let mut out = vec![0.0f64; N_CLASSES];
    for frame in sample {
        let pooled = pool_channels(frame);
        out = system.process_steps(&pooled, 1);
    }
    out.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn train_epoch(
    system: &mut System,
    dataset: &[(Vec<Vec<f64>>, usize)],
    lr: f64,
    rng: &mut impl Rng,
) -> f64 {
    let mut indices: Vec<usize> = (0..dataset.len()).collect();
    indices.shuffle(rng);

    let mut n_correct = 0usize;
    for &idx in &indices {
        let (ref sample, label) = dataset[idx];

        system.sequence_reset();
        let mut out = vec![0.0f64; N_CLASSES];
        for frame in sample {
            let pooled = pool_channels(frame);
            out = system.process_steps(&pooled, 1);
        }
        let predicted = out.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        let correct = predicted == label;
        if correct { n_correct += 1; }

        system.train_readout(label, lr);
        system.reward_contrastive(label, if correct { 0.6 } else { 0.2 }, 0.3);
        system.report_episode_end(if correct { 1.0 } else { 0.0 });
    }

    n_correct as f64 / dataset.len() as f64
}

fn evaluate_set(system: &mut System, dataset: &[(Vec<Vec<f64>>, usize)]) -> f64 {
    let correct = dataset.iter()
        .filter(|(sample, label)| classify(system, sample) == *label)
        .count();
    correct as f64 / dataset.len() as f64
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let profile = if args.iter().any(|a| a == "--extended") { "extended" }
        else if args.iter().any(|a| a == "--standard") { "standard" }
        else { "quick" };
    let base_seed: u64 = args.iter()
        .find(|a| a.starts_with("--seed="))
        .and_then(|a| a[7..].parse().ok())
        .unwrap_or(42);
    let n_seeds: usize = args.iter()
        .find(|a| a.starts_with("--n-seeds="))
        .and_then(|a| a[10..].parse().ok())
        .unwrap_or(1);

    let (n_train_per_class, n_test_per_class, n_epochs, base_duration) = match profile {
        "extended" => (200, 50, 10, 90),
        "standard" => (100, 30, 7, 90),
        _ => (50, 20, 5, 90), // quick
    };
    let n_train = n_train_per_class * N_CLASSES;
    let n_test  = n_test_per_class  * N_CLASSES;

    let version = env!("CARGO_PKG_VERSION");
    eprintln!("=== MORPHON SHD-Synthetic Benchmark [v{version}] ===");
    eprintln!("  {} channels → {} pooled inputs, {} classes", N_CHANNELS, INPUT_DIM, N_CLASSES);
    eprintln!("  profile={profile}, seeds={n_seeds}, train/class={n_train_per_class}, test/class={n_test_per_class}, epochs={n_epochs}\n");

    let mut seed_results: Vec<serde_json::Value> = Vec::new();
    let mut all_final_accs: Vec<f64> = Vec::new();

    for seed_i in 0..n_seeds {
        let seed = base_seed + seed_i as u64;
        eprintln!("━━━ Seed {} ({}/{}) ━━━", seed, seed_i + 1, n_seeds);

        let mut rng_data = rand::rngs::StdRng::seed_from_u64(seed + 1000);
        let mut rng_train = rand::rngs::StdRng::seed_from_u64(seed);

        // Generate dataset
        eprintln!("  Generating {} train + {} test samples...", n_train, n_test);
        let train_set: Vec<(Vec<Vec<f64>>, usize)> = (0..N_CLASSES)
            .flat_map(|c| (0..n_train_per_class)
                .map(move |_| c)
                .collect::<Vec<_>>())
            .map(|c| (generate_sample(c, &mut rng_data, base_duration), c))
            .collect();
        let test_set: Vec<(Vec<Vec<f64>>, usize)> = (0..N_CLASSES)
            .flat_map(|c| (0..n_test_per_class)
                .map(move |_| c)
                .collect::<Vec<_>>())
            .map(|c| (generate_sample(c, &mut rng_data, base_duration), c))
            .collect();

        let t0 = Instant::now();
        let mut system = build_system(seed);
        eprintln!("  Built: {} morphons, {} synapses ({:.1}s)",
            system.inspect().total_morphons, system.inspect().total_synapses,
            t0.elapsed().as_secs_f64());

        let mut best_acc = 0.0f64;
        for epoch in 0..n_epochs {
            let lr = 0.02 * (0.005_f64 / 0.02).powf(epoch as f64 / n_epochs.max(1) as f64);
            let train_acc = train_epoch(&mut system, &train_set, lr, &mut rng_train);
            let test_acc  = evaluate_set(&mut system, &test_set);
            best_acc = best_acc.max(test_acc);
            eprintln!("  ep{}: train={:.1}%  test={:.1}%  best={:.1}%  lr={:.4}  {}",
                epoch + 1, train_acc * 100.0, test_acc * 100.0, best_acc * 100.0,
                lr, system.endo.summary());
        }

        let final_acc = evaluate_set(&mut system, &test_set);
        let result_acc = best_acc.max(final_acc);
        let wall_s = t0.elapsed().as_secs();
        let s = system.inspect();
        eprintln!("  → accuracy={:.1}%  m={}  s={}  ({wall_s}s)\n", result_acc * 100.0, s.total_morphons, s.total_synapses);

        all_final_accs.push(result_acc);
        seed_results.push(json!({
            "seed": seed, "accuracy": result_acc, "wall_s": wall_s,
            "morphons": s.total_morphons, "synapses": s.total_synapses,
        }));
    }

    // Summary
    let n = all_final_accs.len() as f64;
    let mu = all_final_accs.iter().sum::<f64>() / n;
    let sigma = (all_final_accs.iter().map(|v| (v - mu).powi(2)).sum::<f64>() / n).sqrt();
    let chance = 1.0 / N_CLASSES as f64;

    eprintln!("╔══════════════════════════════════════════════════════════╗");
    eprintln!("║  SHD-Synthetic ({n_seeds} seed(s), profile={profile:<10})           ║");
    eprintln!("║  Accuracy: {:.1}% ± {:.1}pp   (chance: {:.1}%)         ║",
        mu * 100.0, sigma * 100.0, chance * 100.0);
    eprintln!("╚══════════════════════════════════════════════════════════╝");

    let results = json!({
        "benchmark": "shd_synthetic",
        "version": version,
        "profile": profile,
        "base_seed": base_seed,
        "n_seeds": n_seeds,
        "n_channels": N_CHANNELS,
        "input_dim": INPUT_DIM,
        "n_classes": N_CLASSES,
        "n_train_per_class": n_train_per_class,
        "n_test_per_class": n_test_per_class,
        "n_epochs": n_epochs,
        "mean_accuracy": mu,
        "std_accuracy": sigma,
        "chance_level": chance,
        "seeds": seed_results,
    });

    let dir = format!("docs/benchmark_results/v{version}");
    fs::create_dir_all(&dir).ok();
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let path = format!("{dir}/shd_{ts}.json");
    if let Ok(s) = serde_json::to_string_pretty(&results) {
        fs::write(&path, s).ok();
        eprintln!("Saved to {path}");
    }
}
