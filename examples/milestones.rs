//! Piagetian Milestones — Developmental Evaluation Benchmark
//!
//! Tests M0–M2 from the BSB/Piagetian milestone ladder. These require only
//! a trained Morphon system and measure properties that task benchmarks miss.
//!
//! Milestones tested:
//!   M0 — Sensorimotor Response: does the system discriminate different inputs?
//!   M1 — Habituation/Dishabituation: does PE decrease on repetition and spike on novelty?
//!   M2 — Object Permanence: does representation survive input occlusion?
//!
//! Profiles:
//!   quick    — M0 + M1 only, ~60s
//!   standard — M0 + M1 + M2, ~3min (default)
//!
//! Run: cargo run --example milestones --release
//! Run: cargo run --example milestones --release -- --quick
//!
//! Results saved to docs/benchmark_results/v{version}/milestones_latest.json

use morphon_core::developmental::DevelopmentalConfig;
use morphon_core::endoquilibrium::EndoConfig;
use morphon_core::homeostasis::{CompetitionMode, HomeostasisParams};
use morphon_core::learning::LearningParams;
use morphon_core::morphogenesis::MorphogenesisParams;
use morphon_core::scheduler::SchedulerConfig;
use morphon_core::system::{System, SystemConfig};
use rand::SeedableRng;
use rand::Rng;
use serde_json::json;
use std::fs;
use std::time::Instant;

const VERSION: &str = env!("CARGO_PKG_VERSION");
const INPUT_DIM: usize = 16;   // small enough for fast runs
const OUTPUT_DIM: usize = 8;
const SEED_SIZE: usize = 80;

fn create_system(rng_seed: u64) -> System {
    let dev = DevelopmentalConfig {
        seed_size: SEED_SIZE,
        dimensions: 4,
        target_input_size: Some(INPUT_DIM),
        target_output_size: Some(OUTPUT_DIM),
        ..DevelopmentalConfig::cortical()
    };
    let config = SystemConfig {
        developmental: dev,
        homeostasis: HomeostasisParams {
            competition_mode: CompetitionMode::LocalInhibition {
                interneuron_ratio: 0.1,
                istdp_rate: 0.005,
                initial_inh_weight: -0.3,
                inhibition_radius: 0.5,
                target_rate: None,
            },
            ..Default::default()
        },
        learning: LearningParams { ..Default::default() },
        morphogenesis: MorphogenesisParams { ..Default::default() },
        scheduler: SchedulerConfig { ..Default::default() },
        endoquilibrium: EndoConfig { enabled: true, ..Default::default() },
        rng_seed: Some(rng_seed),
        ..Default::default()
    };
    System::new(config)
}

/// Generate a normalized pattern for class `class_id` out of `n_classes`.
/// Patterns are orthogonal: each class activates a distinct segment of inputs.
fn class_pattern(class_id: usize, n_classes: usize) -> Vec<f64> {
    let mut p = vec![0.05_f64; INPUT_DIM]; // small background noise
    let seg = INPUT_DIM / n_classes;
    let start = (class_id * seg).min(INPUT_DIM);
    let end = ((class_id + 1) * seg).min(INPUT_DIM);
    for i in start..end {
        p[i] = 1.0;
    }
    p
}

fn zero_input() -> Vec<f64> {
    vec![0.0; INPUT_DIM]
}

fn read_pe(system: &System) -> f64 {
    system.inspect().avg_prediction_error
}

fn read_output_class(system: &mut System, input: &[f64]) -> usize {
    let out = system.process(input);
    out.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn output_vector(system: &mut System, input: &[f64]) -> Vec<f64> {
    system.process(input)
}

/// Warm up the system with random input so iSTDP settles and Endo leaves Proliferating.
fn warmup(system: &mut System, rng: &mut impl Rng, steps: usize) {
    for _ in 0..steps {
        let input: Vec<f64> = (0..INPUT_DIM).map(|_| rng.random::<f64>()).collect();
        system.process(&input);
    }
}

// ─── M0: Sensorimotor Response ────────────────────────────────────────────────

fn run_m0(system: &mut System, n_classes: usize, steps_per_pattern: usize) -> serde_json::Value {
    eprintln!("\n[M0] Sensorimotor Response — discriminability of {} patterns", n_classes);
    let t0 = Instant::now();

    let mut outputs: Vec<Vec<f64>> = Vec::new();
    for class in 0..n_classes {
        let p = class_pattern(class, n_classes);
        let mut acc = vec![0.0_f64; OUTPUT_DIM];
        for _ in 0..steps_per_pattern {
            let out = output_vector(system, &p);
            for (a, o) in acc.iter_mut().zip(out.iter()) {
                *a += o;
            }
        }
        // Normalize
        let s: f64 = acc.iter().sum::<f64>().max(1e-8);
        let normed: Vec<f64> = acc.iter().map(|x| x / s).collect();
        outputs.push(normed);
    }

    // Pairwise cosine distances
    let mut distances = Vec::new();
    for i in 0..n_classes {
        for j in (i + 1)..n_classes {
            let dot: f64 = outputs[i].iter().zip(outputs[j].iter()).map(|(a, b)| a * b).sum();
            let na: f64 = outputs[i].iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-8);
            let nb: f64 = outputs[j].iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-8);
            let cos_sim = (dot / (na * nb)).clamp(-1.0, 1.0);
            distances.push(1.0 - cos_sim); // cosine distance
        }
    }
    let mean_dist = distances.iter().sum::<f64>() / distances.len().max(1) as f64;
    let min_dist = distances.iter().cloned().fold(f64::INFINITY, f64::min);

    let passed = mean_dist > 0.2 && min_dist > 0.05;
    eprintln!(
        "[M0] mean_cosine_dist={:.3} min={:.3} → {}",
        mean_dist,
        min_dist,
        if passed { "PASS ✓" } else { "FAIL ✗" }
    );

    json!({
        "milestone": "M0_sensorimotor",
        "n_classes": n_classes,
        "mean_cosine_distance": mean_dist,
        "min_pairwise_distance": min_dist,
        "passed": passed,
        "criterion": "mean_dist > 0.20 && min_dist > 0.05",
        "duration_s": t0.elapsed().as_secs_f64(),
    })
}

// ─── M1: Habituation / Dishabituation ────────────────────────────────────────

fn run_m1(system: &mut System, habituation_steps: usize, test_steps: usize) -> serde_json::Value {
    eprintln!("\n[M1] Habituation / Dishabituation");
    let t0 = Instant::now();

    let familiar = class_pattern(0, 8);
    let novel = class_pattern(7, 8); // maximally different segment

    // Record initial PE
    let pe_initial: f64 = {
        let mut sum = 0.0;
        for _ in 0..10 {
            system.process(&familiar);
            sum += read_pe(system);
        }
        sum / 10.0
    };

    // Habituation phase
    for _ in 0..habituation_steps {
        system.process(&familiar);
    }

    // PE after habituation
    let pe_habituated: f64 = {
        let mut sum = 0.0;
        for _ in 0..test_steps {
            system.process(&familiar);
            sum += read_pe(system);
        }
        sum / test_steps as f64
    };

    // Novel stimulus — dishabituation
    let pe_novel: f64 = {
        let mut sum = 0.0;
        for _ in 0..test_steps {
            system.process(&novel);
            sum += read_pe(system);
        }
        sum / test_steps as f64
    };

    let habituation_drop = if pe_initial > 1e-8 {
        1.0 - pe_habituated / pe_initial
    } else {
        0.0
    };
    let dishabituation_ratio = if pe_habituated > 1e-8 {
        pe_novel / pe_habituated
    } else {
        1.0
    };

    let passed_habituation = habituation_drop >= 0.30;
    let passed_dishabituation = dishabituation_ratio >= 1.5;
    let passed = passed_habituation && passed_dishabituation;

    eprintln!(
        "[M1] pe_initial={:.4} pe_habituated={:.4} pe_novel={:.4}",
        pe_initial, pe_habituated, pe_novel
    );
    eprintln!(
        "[M1] habituation_drop={:.1}% (need ≥30%) dishabituation_ratio={:.2}× (need ≥1.5×) → {}",
        habituation_drop * 100.0,
        dishabituation_ratio,
        if passed { "PASS ✓" } else { "FAIL ✗" }
    );

    json!({
        "milestone": "M1_habituation",
        "pe_initial": pe_initial,
        "pe_habituated": pe_habituated,
        "pe_novel": pe_novel,
        "habituation_drop_pct": habituation_drop * 100.0,
        "dishabituation_ratio": dishabituation_ratio,
        "passed_habituation": passed_habituation,
        "passed_dishabituation": passed_dishabituation,
        "passed": passed,
        "criterion": "habituation_drop ≥ 30% AND dishabituation_ratio ≥ 1.5×",
        "duration_s": t0.elapsed().as_secs_f64(),
    })
}

// ─── M2: Object Permanence ────────────────────────────────────────────────────

fn run_m2(system: &mut System, encoding_steps: usize, max_occlusion: usize, n_classes: usize) -> serde_json::Value {
    eprintln!("\n[M2] Object Permanence — working memory across occlusion");
    eprintln!("[M2] metric: consecutive correct steps from step 0 (spec: 'stays classified')");
    let t0 = Instant::now();

    let mut results_per_class = Vec::new();

    for class in 0..n_classes {
        let pattern = class_pattern(class, n_classes);
        let zero = zero_input();

        // Reset transient state before each trial
        system.reset_voltages();

        // Encoding phase
        for _ in 0..encoding_steps {
            system.process(&pattern);
        }

        // Occlusion phase.
        // persist_steps = consecutive correct steps starting from step 0.
        // This matches the spec: "longest occlusion across which the output stays classified".
        // last_correct_step tracks the latest step where classification was ever correct
        // (useful to detect late recoveries — different from sustained persistence).
        let mut persist_steps = 0usize;
        let mut consecutive_broken = false;
        let mut last_correct_step = 0usize;
        for t in 0..max_occlusion {
            let predicted = read_output_class(system, &zero);
            if predicted == class {
                if !consecutive_broken {
                    persist_steps = t + 1;
                }
                last_correct_step = t + 1;
            } else {
                consecutive_broken = true;
            }
        }

        // Probe recovery: inject half-strength pattern at the end of occlusion.
        // Tests whether the trace, even if faded, can be reactivated.
        let probe: Vec<f64> = pattern.iter().map(|x| x * 0.5).collect();
        let recovered = read_output_class(system, &probe) == class;

        let tier = if persist_steps >= 100 && recovered { "Gold" }
            else if persist_steps >= 30 { "Silver" }
            else if persist_steps >= 10 { "Bronze" }
            else { "—" };

        results_per_class.push(json!({
            "class": class,
            "persist_steps": persist_steps,
            "last_correct_step": last_correct_step,
            "probe_recovery": recovered,
            "tier": tier,
        }));

        eprintln!(
            "[M2]   class={} consec={}  last={}  probe={}  {}",
            class,
            persist_steps,
            last_correct_step,
            if recovered { "✓" } else { "✗" },
            tier,
        );
    }

    // Medal: majority vote across classes at each tier.
    // Gold requires both ≥100 consecutive steps AND probe recovery.
    let bronze_threshold = 10usize;
    let silver_threshold = 30usize;
    let gold_threshold   = 100usize;
    let n = results_per_class.len() as f64;

    let bronze = results_per_class.iter()
        .filter(|r| r["persist_steps"].as_u64().unwrap_or(0) >= bronze_threshold as u64)
        .count() as f64 / n;
    let silver = results_per_class.iter()
        .filter(|r| r["persist_steps"].as_u64().unwrap_or(0) >= silver_threshold as u64)
        .count() as f64 / n;
    let gold = results_per_class.iter()
        .filter(|r| {
            r["persist_steps"].as_u64().unwrap_or(0) >= gold_threshold as u64
                && r["probe_recovery"].as_bool().unwrap_or(false)
        })
        .count() as f64 / n;
    let probe_recovery_rate = results_per_class.iter()
        .filter(|r| r["probe_recovery"].as_bool().unwrap_or(false))
        .count() as f64 / n;

    // Medal: majority (≥50%) of classes must achieve the tier.
    let medal = if gold >= 0.5 { "Gold" }
        else if silver >= 0.5 { "Silver" }
        else if bronze >= 0.5 { "Bronze" }
        else { "None" };

    // Individual best (for reporting even when majority threshold not met)
    let best_class = results_per_class.iter()
        .max_by_key(|r| r["persist_steps"].as_u64().unwrap_or(0));
    let best_persist = best_class
        .and_then(|r| r["persist_steps"].as_u64())
        .unwrap_or(0);
    let best_tier = best_class
        .and_then(|r| r["tier"].as_str())
        .unwrap_or("—");

    eprintln!(
        "[M2] bronze={:.0}% silver={:.0}% gold={:.0}% probe_recovery={:.0}%  medal={} (best: {}steps, {})",
        bronze * 100.0, silver * 100.0, gold * 100.0, probe_recovery_rate * 100.0,
        medal, best_persist, best_tier,
    );

    json!({
        "milestone": "M2_object_permanence",
        "medal": medal,
        "bronze_pct": bronze * 100.0,
        "silver_pct": silver * 100.0,
        "gold_pct": gold * 100.0,
        "probe_recovery_pct": probe_recovery_rate * 100.0,
        "best_single_class_persist_steps": best_persist,
        "best_single_class_tier": best_tier,
        "thresholds": {
            "bronze_steps": bronze_threshold,
            "silver_steps": silver_threshold,
            "gold_steps": gold_threshold,
            "gold_requires_probe_recovery": true,
        },
        "per_class": results_per_class,
        "criterion": "medal = Bronze if ≥50% classes persist ≥10 steps consecutive; Silver ≥30; Gold ≥100 + probe_recovery",
        "duration_s": t0.elapsed().as_secs_f64(),
    })
}

// ─── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let quick = args.iter().any(|a| a == "--quick");
    let profile = if quick { "quick" } else { "standard" };
    let seed: u64 = args.iter()
        .find(|a| a.starts_with("--seed="))
        .and_then(|a| a[7..].parse().ok())
        .unwrap_or(42);

    eprintln!("=== MORPHON MILESTONES v{VERSION} [{profile}] seed={seed} ===\n");
    eprintln!("Piagetian developmental evaluation: M0 (sensorimotor) + M1 (habituation) {}",
        if quick { "" } else { "+ M2 (object permanence)" });

    let mut system = create_system(seed);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    // Warm up: let iSTDP settle, Endo leave Proliferating
    let warmup_steps = if quick { 300 } else { 500 };
    eprintln!("[WARMUP] {} steps...", warmup_steps);
    warmup(&mut system, &mut rng, warmup_steps);
    eprintln!("[WARMUP] stage={:?}", system.endo.stage());

    let n_classes = 8;
    let t_total = Instant::now();

    // M0 — Note: without a trained readout, output vectors are not yet discriminable.
    // M0 failure pre-training is expected and informative. For a meaningful M0 result,
    // run after supervised training (e.g., after MNIST epoch 1 with --post-train flag).
    let m0 = run_m0(&mut system, n_classes, if quick { 5 } else { 10 });

    // M1
    let m1 = run_m1(
        &mut system,
        if quick { 100 } else { 200 }, // habituation_steps
        if quick { 10 } else { 20 },   // test_steps
    );

    // M2 (standard only)
    let m2 = if !quick {
        Some(run_m2(
            &mut system,
            30,   // encoding_steps
            100,  // max_occlusion
            n_classes,
        ))
    } else {
        None
    };

    let total_s = t_total.elapsed().as_secs_f64();

    // Summary
    eprintln!("\n╔══════════════════════════════════════════════════╗");
    eprintln!("║  MILESTONE RESULTS  [v{VERSION}] [{profile}]");
    eprintln!("║  M0 Sensorimotor: {}", if m0["passed"].as_bool().unwrap_or(false) { "PASS ✓" } else { "FAIL ✗" });
    eprintln!("║  M1 Habituation:  {}", if m1["passed"].as_bool().unwrap_or(false) { "PASS ✓" } else { "FAIL ✗" });
    if let Some(ref m2v) = m2 {
        let medal = m2v["medal"].as_str().unwrap_or("None");
        let best_steps = m2v["best_single_class_persist_steps"].as_u64().unwrap_or(0);
        let best_tier = m2v["best_single_class_tier"].as_str().unwrap_or("—");
        eprintln!("║  M2 Object Perm:  {} (best class: {}steps / {})", medal, best_steps, best_tier);
    }
    eprintln!("║  Total: {:.1}s", total_s);
    eprintln!("╚══════════════════════════════════════════════════╝");

    // Save results
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let out_dir = format!("docs/benchmark_results/v{VERSION}");
    fs::create_dir_all(&out_dir).ok();

    let mut result = json!({
        "benchmark": "milestones_piagetian",
        "version": VERSION,
        "profile": profile,
        "seed": seed,
        "total_duration_s": total_s,
        "m0": m0,
        "m1": m1,
    });
    if let Some(m2v) = m2 {
        result["m2"] = m2v;
    }

    let path = format!("{out_dir}/milestones_{ts}.json");
    fs::write(&path, serde_json::to_string_pretty(&result).unwrap()).ok();
    let latest = format!("{out_dir}/milestones_latest.json");
    fs::write(&latest, serde_json::to_string_pretty(&result).unwrap()).ok();
    eprintln!("[SAVED] {}", path);
}
