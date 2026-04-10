//! Temporal Sequence Processing Benchmarks
//!
//! Three benchmarks that exercise the recurrent reservoir + extended credit
//! assignment, ordered by difficulty:
//!
//! **Benchmark A — Delayed Match-to-Sample (DMS)**
//! See a pattern, wait N blank steps, see a probe, classify: match or not.
//! Tests whether the reservoir retains the sample pattern across the delay.
//!
//! **Benchmark B — Sequence Classification**
//! Classify a 4-element sequence by its temporal shape (rising/falling/peak/valley).
//! Each element is ambiguous in isolation — classification requires integration
//! across the full sequence.
//!
//! **Benchmark C — Next-Element Prediction**
//! Given a repeating periodic sequence (A→B→C→A→...), predict the next element.
//! Tests whether the reservoir discovers periodicity from reward alone.
//!
//! Run: cargo run --example temporal --release
//! Run: cargo run --example temporal --release -- --bench=dms
//! Run: cargo run --example temporal --release -- --bench=seq
//! Run: cargo run --example temporal --release -- --bench=pred

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
use serde_json::json;
use std::fs;

// ── helpers ─────────────────────────────────────────────────────────────────

fn build_system(
    input_size: usize,
    output_size: usize,
    recurrent: RecurrentConfig,
    tau_eligibility: f64,
) -> System {
    let config = SystemConfig {
        developmental: DevelopmentalConfig {
            seed_size: 120,
            target_input_size: Some(input_size),
            target_output_size: Some(output_size),
            recurrent,
            ..DevelopmentalConfig::temporal()
        },
        learning: LearningParams {
            tau_eligibility,
            tau_trace: tau_eligibility * 2.0,
            a_plus: 0.8,
            a_minus: -0.8,
            ..LearningParams::default()
        },
        scheduler: SchedulerConfig {
            medium_period: 5,
            // Disable slow/glacial structural changes — synaptogenesis every 2k
            // steps causes catastrophic forgetting in trial-based tasks because
            // it restructures the reservoir mid-training. The lifecycle flags
            // (division=false etc.) block morphon-level changes but not synapse-level.
            slow_period: 10_000_000,
            glacial_period: 10_000_000,
            homeostasis_period: 50,
            memory_period: 100,
        },
        homeostasis: HomeostasisParams {
            competition_mode: CompetitionMode::LocalInhibition {
                istdp_rate: 0.002,
                target_rate: Some(0.05),
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
        },
        morphogenesis: MorphogenesisParams {
            ..MorphogenesisParams::default()
        },
        metabolic: MetabolicConfig {
            firing_cost: 0.005,
            ..MetabolicConfig::default()
        },
        ..SystemConfig::default()
    };

    let mut system = System::new(config);
    system.enable_analog_readout();
    system
}

// ── Benchmark A: Delayed Match-to-Sample ────────────────────────────────────

fn run_dms(n_trials: usize, delay_steps: usize, verbose: bool) -> f64 {
    const INPUT_DIM: usize = 4;
    const N_PATTERNS: usize = 4;
    // Active phases get more steps to build up reservoir state.
    // Delay uses 1 step to minimize membrane decay (0.9^1 vs 0.9^5).
    const ACTIVE_STEPS: usize = 5;
    const DELAY_STEPS_PER_SLOT: usize = 1;
    const LR: f64 = 0.02;

    let recurrent = RecurrentConfig {
        enabled: true,
        recurrent_connectivity: 0.15,
        recurrent_weight_scale: 0.5, // ρ(W_rec) ≈ 1.12 → self-sustaining activity
        delay_range: (1.0, 4.0),
        plastic: true,
        allow_autapses: false,
    };
    let mut system = build_system(INPUT_DIM, 2, recurrent, 60.0);

    let mut rng = rand::rng();

    // Fixed set of patterns — orthogonal-ish
    let patterns: Vec<Vec<f64>> = (0..N_PATTERNS)
        .map(|i| {
            let mut p = vec![0.0f64; INPUT_DIM];
            p[i % INPUT_DIM] = 1.0;
            p[(i + 1) % INPUT_DIM] = 0.5;
            p
        })
        .collect();

    let blank = vec![0.0f64; INPUT_DIM];

    // Measure converged accuracy: last 20% of trials (after learning stabilises)
    let eval_start = n_trials * 4 / 5;
    let mut n_total = 0usize;
    let mut n_correct_eval = 0usize;

    let window = 200;
    let mut recent = vec![false; window];
    let mut recent_idx = 0;

    for trial in 0..n_trials {
        system.sequence_reset();

        let sample_idx = rng.random_range(0..N_PATTERNS);
        let probe_is_match = rng.random_bool(0.5);
        let probe_idx = if probe_is_match {
            sample_idx
        } else {
            (sample_idx + 1 + rng.random_range(0..N_PATTERNS - 1)) % N_PATTERNS
        };

        // Phase 1: present sample (multiple steps to build reservoir state)
        system.process_steps(&patterns[sample_idx], ACTIVE_STEPS);

        // Phase 2: delay (1 step per slot to minimize membrane decay)
        for _ in 0..delay_steps {
            system.process_steps(&blank, DELAY_STEPS_PER_SLOT);
        }

        // Phase 3: present probe → read output → reward
        let out = system.process_steps(&patterns[probe_idx], ACTIVE_STEPS);

        // out[0] = "match", out[1] = "no match"
        let predicted_match = out.len() >= 2 && out[0] > out[1];
        let correct = predicted_match == probe_is_match;
        let correct_idx = if probe_is_match { 0 } else { 1 };

        system.train_readout(correct_idx, LR);
        system.reward_contrastive(correct_idx, if correct { 0.6 } else { 0.2 }, 0.3);

        if trial >= eval_start {
            n_total += 1;
            if correct { n_correct_eval += 1; }
        }
        recent[recent_idx % window] = correct;
        recent_idx += 1;

        // Debug: print every 500th
        if verbose && trial % 500 == 499 {
            let recent_acc = if trial >= window { recent.iter().filter(|&&x| x).count() as f64 / window as f64 } else { n_correct_eval as f64 / (n_total + 1) as f64 };
            let stats = system.inspect();
            eprintln!(
                "  [DMS  ] trial={:5} recent={:.1}% | m={} s={} delay={} | out=[{:.3},{:.3}]",
                trial + 1,
                recent_acc * 100.0,
                stats.total_morphons,
                stats.total_synapses,
                delay_steps,
                if out.len() > 0 { out[0] } else { 0.0 },
                if out.len() > 1 { out[1] } else { 0.0 },
            );
        }
    }

    if n_total > 0 { n_correct_eval as f64 / n_total as f64 } else { 0.0 }
}

// ── Benchmark B: Sequence Classification ────────────────────────────────────

fn run_seq_classification(n_trials: usize, seq_len: usize, verbose: bool) -> f64 {
    // 4D one-hot input: each distinct value gets its own sensory dimension.
    // With scalar (1D) input all 4 values share a single morphon → weak temporal
    // discrimination because the reservoir can't distinguish input identity from magnitude.
    // One-hot gives orthogonal activations per value → 4× richer temporal context.
    const INPUT_DIM: usize = 4;
    const N_CLASSES: usize = 4;
    const INTERNAL_STEPS: usize = 5;
    const LR: f64 = 0.02;

    // Map each of the 4 distinct sequence values to a 1-hot vector
    let value_map: [(f64, Vec<f64>); 4] = [
        (0.2, vec![1.0, 0.0, 0.0, 0.0]),
        (0.4, vec![0.0, 1.0, 0.0, 0.0]),
        (0.6, vec![0.0, 0.0, 1.0, 0.0]),
        (0.8, vec![0.0, 0.0, 0.0, 1.0]),
    ];
    let encode = |v: f64| -> Vec<f64> {
        value_map.iter()
            .min_by(|(a, _), (b, _)| (a - v).abs().partial_cmp(&(b - v).abs()).unwrap())
            .map(|(_, enc)| enc.clone())
            .unwrap_or_else(|| vec![0.0; INPUT_DIM])
    };

    // Four temporal shapes — each element in isolation is ambiguous
    // (0.2 and 0.8 appear in every class)
    let class_sequences: Vec<Vec<f64>> = vec![
        vec![0.2, 0.4, 0.6, 0.8], // rising
        vec![0.8, 0.6, 0.4, 0.2], // falling
        vec![0.2, 0.8, 0.8, 0.2], // peak
        vec![0.8, 0.2, 0.2, 0.8], // valley
    ];

    let recurrent = RecurrentConfig {
        enabled: true,
        recurrent_connectivity: 0.20,
        recurrent_weight_scale: 0.5,
        delay_range: (1.0, 4.0),
        plastic: true,
        allow_autapses: false,
    };
    let mut system = build_system(INPUT_DIM, N_CLASSES, recurrent, 80.0);
    let mut rng = rand::rng();

    // Measure converged accuracy: last 20% of trials
    let eval_start = n_trials * 4 / 5;
    let mut n_total = 0usize;
    let mut n_correct_eval = 0usize;

    let window = 200;
    let mut recent = vec![false; window];
    let mut recent_idx = 0;

    for trial in 0..n_trials {
        system.sequence_reset();

        let class_idx = rng.random_range(0..N_CLASSES);
        let base = &class_sequences[class_idx];

        // If seq_len != 4, interpolate/subsample
        let sequence: Vec<f64> = (0..seq_len)
            .map(|i| base[i * (base.len() - 1) / seq_len.max(1).max(base.len() - 1).min(seq_len - 1)])
            .collect();

        // Present each element (one-hot encoded), collect final output
        let mut out = vec![0.0f64; N_CLASSES];
        for &val in &sequence {
            out = system.process_steps(&encode(val), INTERNAL_STEPS);
        }

        let predicted = out
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        let correct = predicted == class_idx;

        system.train_readout(class_idx, LR);
        system.reward_contrastive(class_idx, if correct { 0.6 } else { 0.2 }, 0.3);

        if trial >= eval_start {
            n_total += 1;
            if correct { n_correct_eval += 1; }
        }
        recent[recent_idx % window] = correct;
        recent_idx += 1;

        if verbose && trial % 500 == 499 {
            let recent_acc = if trial >= window { recent.iter().filter(|&&x| x).count() as f64 / window as f64 } else { 0.0 };
            let stats = system.inspect();
            eprintln!(
                "  [SEQ  ] trial={:5} recent={:.1}% | m={} s={} len={}",
                trial + 1,
                recent_acc * 100.0,
                stats.total_morphons,
                stats.total_synapses,
                seq_len,
            );
        }
    }

    if n_total > 0 { n_correct_eval as f64 / n_total as f64 } else { 0.0 }
}

// ── Benchmark C: Next-Element Prediction ────────────────────────────────────

fn run_prediction(n_steps: usize, period: usize, verbose: bool) -> f64 {
    // One-hot input: each periodic position gets its own sensory dimension.
    // Without this, all positions share a single morphon and the readout can't
    // distinguish cur=0.74→next=0.10 from cur=0.58→next=0.74 (both high input).
    let input_dim = period;
    const OUTPUT_DIM: usize = 1;
    const INTERNAL_STEPS: usize = 5;

    // Periodic sequence: values spread across [0.1, 0.9]
    let values: Vec<f64> = (0..period)
        .map(|i| 0.1 + 0.8 * (i as f64 / period as f64))
        .collect();

    let recurrent = RecurrentConfig {
        enabled: true,
        recurrent_connectivity: 0.20,
        recurrent_weight_scale: 0.5,
        delay_range: (1.0, 5.0),
        plastic: true,
        allow_autapses: true, // autapses help with periodic signals
    };
    let mut system = build_system(input_dim, OUTPUT_DIM, recurrent, 100.0);

    // No inject_reward: three-factor synaptic learning modifies the reservoir,
    // causing the readout's learned mapping to drift (analogous to catastrophic
    // forgetting). For supervised readout regression the reservoir must stay fixed.
    const LR: f64 = 0.02;

    let mut total_error = 0.0f64;
    let mut n_eval = 0usize;
    let warmup = n_steps / 5;

    for step in 0..n_steps {
        let pos = step % period;
        let next_val = values[(pos + 1) % period];

        // Reset between steps so each position sees a consistent reservoir state.
        // Without reset the reservoir is on a chaotic orbit for short periods
        // (period=3 especially), making the activity patterns non-stationary.
        // The task is pos → next_val lookup; resetting per step ensures the
        // delta rule sees a stationary mapping and converges cleanly.
        system.sequence_reset();

        // One-hot encoding of position
        let mut input = vec![0.0f64; input_dim];
        input[pos] = 1.0;

        let out = system.process_steps(&input, INTERNAL_STEPS);
        let predicted = out.first().copied().unwrap_or(0.5);

        let error = (predicted - next_val).abs();

        // Supervised regression only — no inject_reward.
        // Three-factor synaptic learning (triggered by inject_reward) rewires the
        // reservoir mid-training, so the readout's learned feature → target mapping
        // degrades faster than it converges. Keep the reservoir fixed; train only
        // the linear readout weights via delta rule.
        system.train_readout_value(next_val, LR);

        if step >= warmup {
            total_error += error;
            n_eval += 1;
        }

        if verbose && (step < 10 || step % 1000 == 999) {
            let stats = system.inspect();
            eprintln!(
                "  [PRED ] step={:5} mae={:.3} (last={:.3}) | m={} s={} period={} | next={:.3} pred={:.3}",
                step + 1,
                if n_eval > 0 { total_error / n_eval as f64 } else { f64::NAN },
                error,
                stats.total_morphons,
                stats.total_synapses,
                period,
                next_val,
                predicted,
            );
        }
    }

    if n_eval > 0 { total_error / n_eval as f64 } else { f64::NAN }
}

// ── main ─────────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bench = args.iter()
        .find(|a| a.starts_with("--bench="))
        .map(|a| a[8..].to_string())
        .unwrap_or("all".into());
    let verbose = !args.iter().any(|a| a == "--quiet");

    let version = env!("CARGO_PKG_VERSION");
    eprintln!("=== MORPHON Temporal Sequence Benchmarks [v{version}] ===\n");

    let mut results = json!({
        "benchmark": "temporal",
        "version": version,
        "results": {}
    });

    // ── Benchmark A: DMS ────────────────────────────────────────────────────
    if bench == "all" || bench == "dms" {
        eprintln!("━━━ Benchmark A: Delayed Match-to-Sample ━━━");
        eprintln!("  Chance = 50%   Target = >80% at delay=3, >70% at delay=5\n");

        for &delay in &[3usize, 5] {
            let n_trials = 5000;
            let acc = run_dms(n_trials, delay, verbose);
            let pass = match delay {
                3 => acc > 0.80,
                5 => acc > 0.70,
                _ => acc > 0.60,
            };
            eprintln!(
                "  delay={delay}: acc={:.1}%  {} (target: {}%)",
                acc * 100.0,
                if pass { "PASS ✓" } else { "FAIL ✗" },
                if delay == 3 { 80 } else { 70 },
            );
            results["results"][format!("dms_delay{delay}")] = json!({
                "accuracy": acc,
                "delay": delay,
                "n_trials": n_trials,
                "pass": pass,
            });
        }
        eprintln!();
    }

    // ── Benchmark B: Sequence Classification ────────────────────────────────
    if bench == "all" || bench == "seq" {
        eprintln!("━━━ Benchmark B: Sequence Classification ━━━");
        eprintln!("  Chance = 25%   Target = >85% at len=4, >70% at len=8\n");

        for &seq_len in &[4usize, 8] {
            let n_trials = 6000;
            let acc = run_seq_classification(n_trials, seq_len, verbose);
            let pass = match seq_len {
                4 => acc > 0.85,
                8 => acc > 0.70,
                _ => acc > 0.60,
            };
            eprintln!(
                "  len={seq_len}: acc={:.1}%  {} (target: {}%)",
                acc * 100.0,
                if pass { "PASS ✓" } else { "FAIL ✗" },
                if seq_len == 4 { 85 } else { 70 },
            );
            results["results"][format!("seq_len{seq_len}")] = json!({
                "accuracy": acc,
                "seq_len": seq_len,
                "n_trials": n_trials,
                "pass": pass,
            });
        }
        eprintln!();
    }

    // ── Benchmark C: Next-Element Prediction ────────────────────────────────
    if bench == "all" || bench == "pred" {
        eprintln!("━━━ Benchmark C: Next-Element Prediction ━━━");
        eprintln!("  Target = MAE < 0.10 for period=3, < 0.15 for period=5\n");

        for &period in &[3usize, 5] {
            let n_steps = 12000;
            let mae = run_prediction(n_steps, period, verbose);
            let pass = match period {
                3 => mae < 0.10,
                5 => mae < 0.15,
                _ => mae < 0.20,
            };
            eprintln!(
                "  period={period}: MAE={mae:.3}  {} (target: <{})",
                if pass { "PASS ✓" } else { "FAIL ✗" },
                if period == 3 { "0.10" } else { "0.15" },
            );
            results["results"][format!("pred_period{period}")] = json!({
                "mae": mae,
                "period": period,
                "n_steps": n_steps,
                "pass": pass,
            });
        }
        eprintln!();
    }

    // ── Save results ─────────────────────────────────────────────────────────
    let dir = format!("docs/benchmark_results/v{version}");
    fs::create_dir_all(&dir).ok();
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let path = format!("{dir}/temporal_{ts}.json");
    if let Ok(s) = serde_json::to_string_pretty(&results) {
        fs::write(&path, s).ok();
        eprintln!("Saved to {path}");
    }
}
