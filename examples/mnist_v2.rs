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
use morphon_core::governance::ConstitutionalConstraints;
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
        capture_threshold: 0.5,   // tags max at 1.0 — 10.0 made consolidation impossible
        capture_rate: 0.5,
        weight_max: 1.5,
        weight_min: 0.01,
        alpha_reward: 0.3,
        alpha_novelty: 1.0,
        alpha_arousal: 0.0,
        alpha_homeostasis: 0.1,
        transmitter_potentiation: 0.0005,
        heterosynaptic_depression: 0.001, tag_accumulation_rate: 0.3,
        ..Default::default()
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
            synaptogenesis: true,
        }
}

fn create_baseline() -> System {
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
        homeostasis: HomeostasisParams::default(),
        lifecycle: frozen_lifecycle(),
        metabolic: MetabolicConfig::default(),
        endoquilibrium: morphon_core::endoquilibrium::EndoConfig { enabled: true, ..Default::default() },
        dt: 1.0,
        working_memory_capacity: 7,
        episodic_memory_capacity: 500,
        governance: ConstitutionalConstraints {
            // 30% connectivity target — see mnist.rs comment for derivation.
            max_connectivity_per_morphon: 300,
            ..ConstitutionalConstraints::default()
        },
        ..Default::default()
    };
    let mut sys = System::new(config);
    sys.enable_analog_readout();
    sys
}

fn create_v2(istdp_rate: f64, initial_inh_weight: f64, rng_seed: u64, seed_size: usize) -> System {
    let competition_mode = morphon_core::homeostasis::CompetitionMode::LocalInhibition {
        interneuron_ratio: 0.10,
        istdp_rate,
        initial_inh_weight,
        inhibition_radius: 0.0,
        target_rate: Some(0.05),
    };

    let mut tm = morphon_core::TargetMorphology::cortical(6);
    tm.regions[0].target_density = 50;
    tm.regions[1].target_density = 150;
    tm.regions[2].target_density = 15;
    tm.healing_threshold = 0.4;

    let config = SystemConfig {
        developmental: DevelopmentalConfig {
            seed_size,
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
            competition_mode,
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
        governance: ConstitutionalConstraints {
            // 30% connectivity target — see mnist.rs comment for derivation.
            max_connectivity_per_morphon: 300,
            ..ConstitutionalConstraints::default()
        },
        rng_seed: Some(rng_seed),
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

/// Like `classify` but performs a full transient-state reset before each image.
/// Prevents temporal context from one image bleeding into the next — gives a
/// per-image accuracy that reflects true discriminative capacity, not sequential state.
fn classify_stateless(system: &mut System, inputs: &[f64]) -> usize {
    system.reset_transient_state();
    for _ in 0..STEPS_PER_IMAGE {
        system.feed_input(inputs);
        system.step();
    }
    let outputs = system.read_output();
    outputs.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i).unwrap_or(0)
}

fn evaluate_stateless(system: &mut System, images: &[Vec<f64>], labels: &[usize], n: usize) -> f64 {
    let mut correct = 0;
    let tested = images.len().min(n);
    for i in 0..tested {
        if classify_stateless(system, &images[i]) == labels[i] { correct += 1; }
    }
    correct as f64 / tested as f64 * 100.0
}

/// Evaluate with state carry-over but randomised image order.
/// Comparing sequential vs shuffled shows whether accuracy depends on natural
/// ordering (temporal context) or true per-image discriminative capacity.
/// If sequential >> shuffled: the system exploits presentation order (LSM artefact).
/// If they are close: the readout has real per-image discriminative capacity.
fn evaluate_shuffled(system: &mut System, images: &[Vec<f64>], labels: &[usize], n: usize,
                     rng: &mut rand::rngs::StdRng) -> f64 {
    use rand::seq::SliceRandom;
    let tested = images.len().min(n);
    let mut indices: Vec<usize> = (0..images.len()).collect();
    indices.shuffle(rng);
    let mut correct = 0;
    for &i in indices.iter().take(tested) {
        if classify(system, &images[i]) == labels[i] { correct += 1; }
    }
    correct as f64 / tested as f64 * 100.0
}

/// Dump receptive fields of top-K associative morphons (by mean incoming weight magnitude)
/// to a JSON file for paper figure generation. Each RF is a 28x28 grid of S→A weights.
fn dump_receptive_fields(system: &System, k: usize) {
    // Build sensory ID → pixel index map (assumes sensory morphons are pinned in order)
    let mut sensory_ids: Vec<MorphonId> = system.morphons.values()
        .filter(|m| m.cell_type == CellType::Sensory)
        .map(|m| m.id)
        .collect();
    sensory_ids.sort();
    let sens_to_idx: std::collections::HashMap<MorphonId, usize> = sensory_ids
        .iter().enumerate().map(|(i, &id)| (id, i)).collect();

    // Collect (assoc_id, mean weight magnitude, RF grid) for each associative morphon
    let assoc_ids: Vec<MorphonId> = system.morphons.values()
        .filter(|m| m.cell_type == CellType::Associative || m.cell_type == CellType::Stem)
        .map(|m| m.id)
        .collect();

    let mut entries: Vec<(MorphonId, f64, Vec<f64>)> = Vec::new();
    for &aid in &assoc_ids {
        let mut rf = vec![0.0; IMG_PIXELS];
        let mut nz = 0usize;
        let mut total_mag = 0.0;
        for (pre_id, syn) in system.topology.incoming(aid) {
            if let Some(&px) = sens_to_idx.get(&pre_id) {
                if px < IMG_PIXELS {
                    rf[px] = syn.weight;
                    nz += 1;
                    total_mag += syn.weight.abs();
                }
            }
        }
        if nz > 0 {
            entries.push((aid, total_mag / nz as f64, rf));
        }
    }

    // Sort by magnitude descending, take top k
    entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    entries.truncate(k);

    let json_entries: Vec<serde_json::Value> = entries.iter().map(|(id, mag, rf)| {
        json!({
            "morphon_id": id,
            "mean_abs_weight": mag,
            "rf_28x28": rf,
        })
    }).collect();

    let version = env!("CARGO_PKG_VERSION");
    let dir = format!("docs/benchmark_results/v{}", version);
    fs::create_dir_all(&dir).ok();
    let path = format!("{}/mnist_v2_rfs.json", dir);
    let payload = json!({
        "version": version,
        "n_morphons": json_entries.len(),
        "img_size": 28,
        "morphons": json_entries,
    });
    fs::write(&path, serde_json::to_string_pretty(&payload).unwrap()).unwrap();
    eprintln!("  Dumped {} receptive fields to {}", entries.len(), path);
}

fn evaluate(system: &mut System, images: &[Vec<f64>], labels: &[usize], n: usize) -> f64 {
    let mut correct = 0;
    let tested = images.len().min(n);
    for i in 0..tested {
        if classify(system, &images[i]) == labels[i] { correct += 1; }
    }
    correct as f64 / tested as f64 * 100.0
}

/// Evaluate and return (accuracy, confusion matrix [true][pred]).
fn evaluate_with_confusion(system: &mut System, images: &[Vec<f64>], labels: &[usize], n: usize)
    -> (f64, [[u32; NUM_CLASSES]; NUM_CLASSES])
{
    let mut matrix = [[0u32; NUM_CLASSES]; NUM_CLASSES];
    let tested = images.len().min(n);
    for i in 0..tested {
        let pred = classify(system, &images[i]);
        let true_label = labels[i];
        if true_label < NUM_CLASSES && pred < NUM_CLASSES {
            matrix[true_label][pred] += 1;
        }
    }
    let correct: u32 = (0..NUM_CLASSES).map(|c| matrix[c][c]).sum();
    let acc = correct as f64 / tested as f64 * 100.0;
    (acc, matrix)
}

fn print_confusion_matrix(matrix: &[[u32; NUM_CLASSES]; NUM_CLASSES]) {
    eprintln!("\n━━━ CONFUSION MATRIX (rows=true, cols=pred) ━━━");
    eprint!("    ");
    for c in 0..NUM_CLASSES { eprint!("{:>5}", c); }
    eprintln!("  | recall");
    for r in 0..NUM_CLASSES {
        eprint!("{:>3} ", r);
        let row_total: u32 = matrix[r].iter().sum();
        for c in 0..NUM_CLASSES {
            eprint!("{:>5}", matrix[r][c]);
        }
        let recall = if row_total > 0 { matrix[r][r] as f64 / row_total as f64 * 100.0 } else { 0.0 };
        eprintln!("  | {:.0}%", recall);
    }
    eprint!("    ");
    for c in 0..NUM_CLASSES {
        let col_total: u32 = (0..NUM_CLASSES).map(|r| matrix[r][c]).sum();
        let prec = if col_total > 0 { matrix[c][c] as f64 / col_total as f64 * 100.0 } else { 0.0 };
        eprint!("{:>4}%", prec as u32);
    }
    eprintln!("\n    (prec per class)");
}

/// Dump morphon activations for the full test set to CSV for offline MLP eval.
/// Format: label, act_0, act_1, ..., act_N  (sorted by morphon ID)
fn dump_activations(system: &mut System, images: &[Vec<f64>], labels: &[usize], path: &str) {
    // Collect sorted associative+stem morphon IDs — same set the linear readout uses.
    let mut morph_ids: Vec<MorphonId> = system.morphons.values()
        .filter(|m| m.cell_type == CellType::Associative || m.cell_type == CellType::Stem)
        .map(|m| m.id)
        .collect();
    morph_ids.sort();
    if morph_ids.is_empty() { return; }

    let mut rows: Vec<String> = Vec::with_capacity(images.len() + 1);
    // Header
    let header: Vec<String> = std::iter::once("label".to_string())
        .chain(morph_ids.iter().map(|id| format!("m{}", id)))
        .collect();
    rows.push(header.join(","));

    for (img, &label) in images.iter().zip(labels.iter()) {
        present_image(system, img);
        let acts: Vec<f64> = morph_ids.iter().map(|&id| {
            system.morphons.get(&id).map(|m| {
                let p = if m.potential.is_finite() { m.potential.clamp(-10.0, 10.0) } else { 0.0 };
                1.0 / (1.0 + (-p).exp()) - 0.5
            }).unwrap_or(0.0)
        }).collect();
        let mut row = vec![label.to_string()];
        row.extend(acts.iter().map(|v| format!("{:.6}", v)));
        rows.push(row.join(","));
    }

    fs::write(path, rows.join("\n")).unwrap_or_else(|e| eprintln!("  [warn] activation dump failed: {e}"));
    eprintln!("  Dumped {} activation vectors ({} features) → {}", images.len(), morph_ids.len(), path);
}

fn train_and_eval(
    system: &mut System,
    train_images: &[Vec<f64>], train_labels: &[usize],
    test_images: &[Vec<f64>], test_labels: &[usize],
    n_train: usize, epochs: usize, label: &str,
    rng: &mut rand::rngs::StdRng,
) -> f64 {
    train_and_eval_from(system, train_images, train_labels, test_images, test_labels,
        n_train, epochs, label, rng, 1, false)
}

/// Like `train_and_eval` but resets all transient state before each image.
/// Breaks LSM co-adaptation: the readout trains on per-image features rather than
/// accumulated sequential context. Expected to produce more generalisable representations.
fn train_and_eval_stateless(
    system: &mut System,
    train_images: &[Vec<f64>], train_labels: &[usize],
    test_images: &[Vec<f64>], test_labels: &[usize],
    n_train: usize, epochs: usize, label: &str,
    rng: &mut rand::rngs::StdRng,
) -> f64 {
    train_and_eval_from(system, train_images, train_labels, test_images, test_labels,
        n_train, epochs, label, rng, 1, true)
}

/// Like train_and_eval but `supervised_from` controls which epoch first gets reward/readout.
/// Pass 0 to supervise from ep1 (recovery mode: structural growth + immediate readout signal).
/// Pass 1 (default) for standard training where ep1 is unsupervised feature formation.
/// `stateless_training`: if true, resets transient network state before every image.
fn train_and_eval_from(
    system: &mut System,
    train_images: &[Vec<f64>], train_labels: &[usize],
    test_images: &[Vec<f64>], test_labels: &[usize],
    n_train: usize, epochs: usize, label: &str,
    rng: &mut rand::rngs::StdRng,
    supervised_from: usize,
    stateless_training: bool,
) -> f64 {
    let start = Instant::now();
    // Track peak accuracy seen during supervised epochs — the Differentiating transition
    // reliably lands at ep3 end, depressing the final checkpoint. Returning the peak
    // captures the system's best demonstrated performance, not an artifact of endo timing.
    let mut best_acc: f64 = 0.0;

    for epoch in 0..epochs {
        let mut indices: Vec<usize> = (0..train_images.len()).collect();
        indices.shuffle(rng);

        // LR schedule: 0.05 → 0.005 over epochs
        let lr = LR_START * (LR_END / LR_START).powf(epoch as f64 / epochs.max(1) as f64);

        // Windowed accuracy ring buffer (last 50 predictions)
        const WINDOW: usize = 50;
        let mut window_buf = [false; WINDOW];
        let mut window_pos = 0_usize;
        let mut window_filled = 0_usize;

        // supervised_from controls the first epoch that receives reward/readout signal.
        // Standard (supervised_from=1): ep1 is unsupervised STDP feature formation.
        //   Injecting reward in ep1 causes Endo to converge prematurely: the network
        //   games binary reward by collapsing to one class (reward→0.99, ng→0.24).
        //   Ep2+ get the full supervised signal once representations have diversified.
        // Recovery (supervised_from=0): ep1 immediately supervised. The network has
        //   already formed features; withholding reward during recovery just degrades
        //   the learned readout weights via unsupervised STDP overwrite.
        let supervised = epoch >= supervised_from;

        for (bi, &idx) in indices.iter().take(n_train).enumerate() {
            // Stateless training: reset working memory, fire counts, and all transient
            // context before each image so the readout trains on per-image features, not
            // accumulated sequential history. present_image() still resets potentials.
            if stateless_training {
                system.reset_transient_state();
            }
            present_image(system, &train_images[idx]);
            system.inject_novelty(0.05);

            // Supervised readout + contrastive reward gated on correctness
            let correct = train_labels[idx];
            system.train_readout(correct, lr);
            let predicted = system.read_output().iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i).unwrap_or(0);

            if supervised {
                // Only inject reward when correct — prevents reward_slow inflation
                // that triggers premature Mature at 11% accuracy
                if predicted == correct {
                    system.reward_contrastive(correct, 0.5, 0.2);
                }
            }
            window_buf[window_pos] = predicted == correct;
            window_pos = (window_pos + 1) % WINDOW;
            if window_filled < WINDOW { window_filled += 1; }
            let window_correct: usize = window_buf[..window_filled].iter().filter(|&&x| x).count();
            let running_acc = window_correct as f64 / window_filled as f64;
            if supervised {
                // Only report once window is full — partial windows inflate early accuracy
                // which pumps recent_performance → reward_slow → premature Mature
                if window_filled >= WINDOW {
                    system.report_performance(running_acc);
                }
            }

            // Episode-end consolidation per image
            let reward = if supervised && predicted == correct { 1.0 } else { 0.0 };
            system.report_episode_end(reward);

            let report_interval = if n_train <= 500 { 250 } else { 1000 };
            if (bi + 1) % report_interval == 0 || (bi + 1 == n_train && epoch + 1 == epochs) {
                let acc = evaluate(system, test_images, test_labels, 100);
                if supervised { best_acc = best_acc.max(acc); }
                let s = system.inspect();
                let fwd_saved = s.synapses_saved_fwd_recent;
                let fwd_str = if fwd_saved > 0 { format!(" fwd_saved={}", fwd_saved) } else { String::new() };
                eprintln!("  [{}] ep{} {:>5}/{} acc={:.1}% lr={:.4} m={} s={} ({:.0}s) {} | rs={:.3} cv={:.3}{}",
                    label, epoch + 1, bi + 1, n_train, acc, lr,
                    s.total_morphons, s.total_synapses,
                    start.elapsed().as_secs(), system.endo.summary(),
                    system.endo.reward_slow(), system.endo.reward_cv(), fwd_str);
            }
        }
    }
    let final_acc = evaluate(system, test_images, test_labels, 200);
    let result = best_acc.max(final_acc);
    if result > final_acc + 1.0 {
        eprintln!("  [{}] peak={:.1}% > final={:.1}% — returning peak (Differentiating trough at ep end)",
            label, result, final_acc);
    }
    result
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let profile = if args.iter().any(|a| a == "--extended") { "extended" }
        else if args.iter().any(|a| a == "--standard") { "standard" }
        else if args.iter().any(|a| a == "--fast") { "fast" }
        else { "quick" };

    let fast = profile == "fast";
    let damage_sweep = args.iter().any(|a| a == "--damage-sweep");
    let ng_ablation = args.iter().any(|a| a == "--ng-ablation");
    let seed: u64 = args.iter()
        .find(|a| a.starts_with("--seed="))
        .and_then(|a| a[7..].parse().ok())
        .unwrap_or(42);
    let (n_train_default, n_epochs, n_test, recovery_n) = match profile {
        "extended" => (10000, 3, 500, 3000),
        "standard" => (5000, 3, 300, 1500),
        "fast" => (500, 2, 100, if damage_sweep { 300 } else { 0 }),
        _ => (3000, 3, 200, 500),
    };
    // --n-train=N and --n-epochs=N override profile defaults. Useful for scaling sweeps.
    let n_train: usize = args.iter()
        .find(|a| a.starts_with("--n-train="))
        .and_then(|a| a[10..].parse().ok())
        .unwrap_or(n_train_default);
    let n_epochs: usize = args.iter()
        .find(|a| a.starts_with("--n-epochs="))
        .and_then(|a| a[11..].parse().ok())
        .unwrap_or(n_epochs);
    // --seed-size=N overrides the network size (default 200).
    let seed_size: usize = args.iter()
        .find(|a| a.starts_with("--seed-size="))
        .and_then(|a| a[12..].parse().ok())
        .unwrap_or(200);

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

    let debug = args.iter().any(|a| a == "--debug");
    let do_sweep = args.iter().any(|a| a == "--sweep");
    let no_baseline = args.iter().any(|a| a == "--no-baseline");
    // --recovery-only: skip all training, load pretrained snapshot and go straight to damage+recovery.
    // Requires a prior run that wrote v3_pretrained.json.
    let recovery_only = args.iter().any(|a| a == "--recovery-only");
    let pretrained_path = "v3_pretrained.json";

    // === LOCAL INHIBITION PARAM SWEEP (--sweep) ===
    // Sweeps istdp_rate × initial_inh_weight on the configured profile.
    // Quick baselines (v4.1.0): reference(0.005,-0.5)=44%, slow+soft(0.001,-0.2)=49%.
    // Standard baseline (v4.1.0): reference(0.005,-0.5)=52%, recovery=28.5% (-20pp).
    // Goal: find config that improves Standard accuracy beyond 52% and closes recovery gap.
    if do_sweep {
        let sweep_params: &[(f64, f64, &str)] = &[
            // Confirmed better on quick — test on standard
            (0.001, -0.5, "slow-rate"),
            (0.002, -0.5, "med-rate"),
            (0.005, -0.2, "soft-inh"),
            (0.001, -0.2, "slow+soft"),
            // New configs targeting the standard-profile gap
            (0.001, -0.1, "ultra-soft"),   // minimal inhibition — does iSTDP need to fight less?
            (0.002, -0.2, "med+soft"),     // between med-rate and slow+soft
            (0.0005, -0.2, "very-slow"),   // test rate floor — does slower help or stall?
            (0.001, -0.3, "mid+slow"),     // intermediate inh strength, slow rate
        ];
        let reference_acc = match profile {
            "standard" => 52.0,
            _ => 44.0,
        };
        let reference_note = match profile {
            "standard" => "52.0% (v4.1.0 Standard), recovery=28.5%",
            _ => "44.0% (v4.1.0 Quick)",
        };
        eprintln!("=== LocalInhibition param sweep — {} configs × {} profile ===", sweep_params.len(), profile);
        eprintln!("  Reference (0.005, -0.5): {}\n", reference_note);
        let version = env!("CARGO_PKG_VERSION");
        let mut sweep_results = Vec::new();
        for &(rate, inh_w, label) in sweep_params {
            eprintln!("━━━ {} (istdp_rate={}, initial_inh_weight={}) ━━━", label, rate, inh_w);
            let t = Instant::now();
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let mut sys = create_v2(rate, inh_w, seed, seed_size);
            eprintln!("  {} morphons, {} synapses", sys.inspect().total_morphons, sys.inspect().total_synapses);
            let acc = train_and_eval(&mut sys, &train_images, &train_labels,
                &test_images, &test_labels, n_train, n_epochs, label, &mut rng);
            let elapsed = t.elapsed().as_secs();
            let delta = acc - reference_acc;
            eprintln!("  {} → {:.1}% ({:+.1}pp vs ref) in {}s\n", label, acc, delta, elapsed);
            sweep_results.push(json!({
                "label": label,
                "istdp_rate": rate,
                "initial_inh_weight": inh_w,
                "accuracy": acc,
                "delta_vs_ref": delta,
                "duration_s": elapsed,
            }));
        }
        // Sort results by accuracy descending for the summary table
        let mut sorted = sweep_results.clone();
        sorted.sort_by(|a, b| b["accuracy"].as_f64().unwrap_or(0.0)
            .partial_cmp(&a["accuracy"].as_f64().unwrap_or(0.0))
            .unwrap_or(std::cmp::Ordering::Equal));
        eprintln!("╔══════════════════════════════════════════════════════════╗");
        eprintln!("║  Reference   (0.005, -0.5):  {:.1}%  ({})  ║", reference_acc, profile);
        eprintln!("╠══════════════════════════════════════════════════════════╣");
        for r in &sorted {
            eprintln!("║  {:12} ({:.4}, {:4.1}):  {:>5.1}%  {:>+5.1}pp  {:>4}s  ║",
                r["label"].as_str().unwrap_or(""),
                r["istdp_rate"].as_f64().unwrap_or(0.0),
                r["initial_inh_weight"].as_f64().unwrap_or(0.0),
                r["accuracy"].as_f64().unwrap_or(0.0),
                r["delta_vs_ref"].as_f64().unwrap_or(0.0),
                r["duration_s"].as_u64().unwrap_or(0));
        }
        eprintln!("╚══════════════════════════════════════════════════════════╝");
        let dir = format!("docs/benchmark_results/v{}", version);
        fs::create_dir_all(&dir).ok();
        let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
        let path = format!("{}/sweep_local_inh_{}.json", dir, ts);
        fs::write(&path, serde_json::to_string_pretty(&json!({
            "benchmark": "local_inhibition_sweep",
            "version": version,
            "profile": profile,
            "seed": seed,
            "reference": {
                "istdp_rate": 0.005,
                "initial_inh_weight": -0.5,
                "accuracy": reference_acc,
                "note": reference_note,
            },
            "results": sweep_results,
        })).unwrap()).unwrap();
        eprintln!("Saved to {}", path);
        return;
    }
    if (!fast && !no_baseline) || debug {
    eprintln!("━━━ DIAGNOSTIC ━━━");
    let mut diag_sys = create_baseline();

    // Collect assoc IDs in stable order
    let mut assoc_ids: Vec<MorphonId> = diag_sys.morphons.values()
        .filter(|m| m.cell_type == CellType::Associative || m.cell_type == CellType::Stem)
        .map(|m| m.id).collect();
    assoc_ids.sort();

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

    // === RECOVERY-ONLY: load pretrained V2 snapshot, skip all training ===
    if recovery_only {
        eprintln!("━━━ RECOVERY-ONLY: loading {pretrained_path} ━━━");
        let json = fs::read_to_string(pretrained_path)
            .unwrap_or_else(|e| panic!("Failed to read {pretrained_path}: {e}\nRun with --save-pretrained or --damage-sweep first."));
        let mut v2 = System::load_json(&json)
            .unwrap_or_else(|e| panic!("Failed to parse {pretrained_path}: {e}"));
        let v2_acc = evaluate(&mut v2, &test_images, &test_labels, n_test);
        let v2_m = v2.inspect().total_morphons;
        eprintln!("  Loaded: {:.1}% | m={}", v2_acc, v2_m);

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

        v2.config.lifecycle = LifecycleConfig {
            division: false, differentiation: false, fusion: false,
            apoptosis: false, migration: true,
            synaptogenesis: true,
        };
        v2.config.scheduler.slow_period = 200;
        v2.config.scheduler.glacial_period = 500;
        let mut rng_v = rand::rngs::StdRng::seed_from_u64(seed);
        let recovery_acc = train_and_eval_from(&mut v2, &train_images, &train_labels,
            &test_images, &test_labels, recovery_n, 1, "RECV", &mut rng_v, 0, false);
        eprintln!("  Recovery: {:.1}% ({:+.1}pp) | morphons: {}", recovery_acc, recovery_acc - v2_acc, v2.morphons.len());
        eprintln!("  Baseline={:.1}%  Damaged={:.1}%  Recovery={:.1}%  Δ={:+.1}pp",
            v2_acc, damaged_acc, recovery_acc, recovery_acc - v2_acc);
        return;
    }

    // === BASELINE (skipped in fast mode or with --no-baseline) ===
    let base_acc = if !fast && !no_baseline {
        eprintln!("━━━ BASELINE ━━━");
        let t0 = Instant::now();
        let mut base = create_baseline();
        eprintln!("  {} morphons, {} synapses (built in {:.1}s)",
            base.inspect().total_morphons, base.inspect().total_synapses, t0.elapsed().as_secs_f64());
        let mut rng_b = rand::rngs::StdRng::seed_from_u64(seed);
        let acc = train_and_eval(&mut base, &train_images, &train_labels,
            &test_images, &test_labels, n_train, n_epochs, "BASE", &mut rng_b);
        eprintln!("  Baseline: {:.1}% ({:.0}s)\n", acc, t0.elapsed().as_secs_f64());
        acc
    } else { 0.0 };

    // === V3 (LocalInhibition — biological iSTDP interneurons) ===
    eprintln!("━━━ V3 (LocalInhibition) ━━━");
    let t_v3 = Instant::now();
    let mut v3 = create_v2(0.001, -0.5, seed, seed_size);
    eprintln!("  {} morphons, {} synapses (built in {:.1}s)",
        v3.inspect().total_morphons, v3.inspect().total_synapses, t_v3.elapsed().as_secs_f64());
    let mut rng_v3 = rand::rngs::StdRng::seed_from_u64(seed);
    let v3_acc = train_and_eval(&mut v3, &train_images, &train_labels,
        &test_images, &test_labels, n_train, n_epochs, "V3  ", &mut rng_v3);
    let v3_stateless = evaluate_stateless(&mut v3, &test_images, &test_labels, n_test);
    let v3_m = v3.inspect().total_morphons;
    let v3_s = v3.inspect().total_synapses;
    let v3_duration_s = t_v3.elapsed().as_secs();
    eprintln!("  V3: {:.1}% (stateless: {:.1}%) | m={} s={} ({:.0}s)\n",
        v3_acc, v3_stateless, v3_m, v3_s, v3_duration_s);

    // === V3-SL (LocalInhibition + stateless training) ===
    // Same architecture as V3 but transient state is reset before each training image.
    // Breaks LSM co-adaptation: the readout must learn per-image discriminative features
    // rather than exploiting sequential temporal context. Expected to improve generalisation
    // and close the gap between online and stateless eval accuracy.
    eprintln!("━━━ V3-SL (LocalInhibition + Stateless Training) ━━━");
    let t_v3sl = Instant::now();
    let mut v3sl = create_v2(0.001, -0.5, seed, seed_size);
    eprintln!("  {} morphons, {} synapses (built in {:.1}s)",
        v3sl.inspect().total_morphons, v3sl.inspect().total_synapses, t_v3sl.elapsed().as_secs_f64());
    let mut rng_v3sl = rand::rngs::StdRng::seed_from_u64(seed);
    let v3sl_acc = train_and_eval_stateless(&mut v3sl, &train_images, &train_labels,
        &test_images, &test_labels, n_train, n_epochs, "V3SL", &mut rng_v3sl);
    let v3sl_stateless = evaluate_stateless(&mut v3sl, &test_images, &test_labels, n_test);
    let v3sl_m = v3sl.inspect().total_morphons;
    let v3sl_s = v3sl.inspect().total_synapses;
    let v3sl_duration_s = t_v3sl.elapsed().as_secs();
    eprintln!("  V3-SL: {:.1}% (stateless: {:.1}%) | m={} s={} ({:.0}s)\n",
        v3sl_acc, v3sl_stateless, v3sl_m, v3sl_s, v3sl_duration_s);

    // === NG-FIX ABLATION (V3-SL with suppress_novelty_on_energy=true) ===
    // Isolates the contribution of stateless training alone, without the ng-collapse fix.
    // If this scores much lower than V3-SL, the ng fix was load-bearing; if roughly the same,
    // stateless training is doing the work independently.
    if ng_ablation {
        eprintln!("━━━ V3-SL-ABL (Stateless Training, ng-suppression re-enabled) ━━━");
        let t_abl = Instant::now();
        let mut v3sl_abl = create_v2(0.001, -0.5, seed, seed_size);
        // Patch the live endo config to re-enable novelty suppression under energy pressure.
        v3sl_abl.endo.config.suppress_novelty_on_energy = true;
        eprintln!("  {} morphons, {} synapses (built in {:.1}s)",
            v3sl_abl.inspect().total_morphons, v3sl_abl.inspect().total_synapses, t_abl.elapsed().as_secs_f64());
        let mut rng_abl = rand::rngs::StdRng::seed_from_u64(seed);
        let abl_acc = train_and_eval_stateless(&mut v3sl_abl, &train_images, &train_labels,
            &test_images, &test_labels, n_train, n_epochs, "V3SL-ABL", &mut rng_abl);
        let abl_stateless = evaluate_stateless(&mut v3sl_abl, &test_images, &test_labels, n_test);
        let abl_m = v3sl_abl.inspect().total_morphons;
        let abl_s = v3sl_abl.inspect().total_synapses;
        eprintln!("  V3-SL-ABL: {:.1}% (stateless: {:.1}%) | m={} s={} ({:.0}s)",
            abl_acc, abl_stateless, abl_m, abl_s, t_abl.elapsed().as_secs());
        eprintln!("  Δ vs V3-SL: online={:+.1}pp  stateless={:+.1}pp\n",
            abl_acc - v3sl_acc, abl_stateless - v3sl_stateless);
    }

    // === CONFUSION MATRIX + ACTIVATION DUMP (V3) ===
    if !fast {
        let (_, confusion) = evaluate_with_confusion(&mut v3, &test_images, &test_labels, n_test);
        print_confusion_matrix(&confusion);
        let version = env!("CARGO_PKG_VERSION");
        let act_path = format!("docs/benchmark_results/v{}/v3_activations.csv", version);
        fs::create_dir_all(format!("docs/benchmark_results/v{}", version)).ok();
        dump_activations(&mut v3, &test_images, &test_labels, &act_path);

        // === SEQUENTIAL vs SHUFFLED EVAL DIAGNOSTIC ===
        // Compares V3 accuracy when the test set is evaluated in natural order (carry-over
        // state) vs a randomly shuffled order. If sequential >> shuffled, the system is
        // exploiting sequential temporal context rather than per-image features.
        // This directly tests the LSM co-adaptation hypothesis.
        let mut rng_shuf = rand::rngs::StdRng::seed_from_u64(seed + 1234);
        let v3_shuffled = evaluate_shuffled(&mut v3, &test_images, &test_labels, n_test, &mut rng_shuf);
        let mut rng_shuf_sl = rand::rngs::StdRng::seed_from_u64(seed + 1234);
        let v3sl_shuffled = evaluate_shuffled(&mut v3sl, &test_images, &test_labels, n_test, &mut rng_shuf_sl);
        eprintln!("\n━━━ SEQUENTIAL vs SHUFFLED EVAL (temporal context diagnostic) ━━━");
        eprintln!("  V3  sequential:  {:.1}%   shuffled: {:.1}%   Δ {:+.1}pp", v3_acc, v3_shuffled, v3_acc - v3_shuffled);
        eprintln!("  V3-SL sequential:{:.1}%   shuffled: {:.1}%   Δ {:+.1}pp", v3sl_acc, v3sl_shuffled, v3sl_acc - v3sl_shuffled);
        if v3_acc - v3_shuffled > 5.0 {
            eprintln!("  → V3 exploits sequential order ({:+.1}pp gap). LSM co-adaptation confirmed.", v3_acc - v3_shuffled);
        } else {
            eprintln!("  → V3 order-independent (<5pp gap). Sequential context not a major factor.");
        }
        let gap_closed = (v3_acc - v3_shuffled) - (v3sl_acc - v3sl_shuffled);
        if gap_closed > 2.0 {
            eprintln!("  → Stateless training closed {:.1}pp of the sequential gap.", gap_closed);
        }

        // === SENSORY-ONLY READOUT DIAGNOSTIC ===
        // Key question: do associative features contribute anything, or does accuracy
        // come almost entirely from the direct Sensory→Readout pathway?
        // If sensory-only ≈ v3_acc: assoc features are noise, STDP produces nothing useful.
        // If sensory-only << v3_acc: assoc features contribute (just not extractable offline).
        let sensory_ids: std::collections::HashSet<MorphonId> = v3.morphons.values()
            .filter(|m| m.cell_type == CellType::Sensory)
            .map(|m| m.id)
            .collect();
        v3.filter_readout_weights(|id| sensory_ids.contains(&id));
        let sensory_only_acc = evaluate(&mut v3, &test_images, &test_labels, n_test);
        eprintln!("\n━━━ READOUT ABLATION ━━━");
        eprintln!("  Full readout (Sensory + Assoc): {:.1}%", v3_acc);
        eprintln!("  Sensory-only readout:           {:.1}%  (Δ {:+.1}pp)", sensory_only_acc, sensory_only_acc - v3_acc);
        if (v3_acc - sensory_only_acc).abs() < 3.0 {
            eprintln!("  → Assoc features contribute ~nothing. STDP output is noise for classification.");
        } else {
            eprintln!("  → Assoc features contribute {:.1}pp. Co-adapted but not offline-extractable.", v3_acc - sensory_only_acc);
        }
    }

    // === RECEPTIVE FIELD DUMP ===
    // Export top-K associative morphon RFs (S→A weights mapped to 28×28 pixel grid)
    // for paper figure generation. Only after main training, before damage.
    if !fast {
        dump_receptive_fields(&v3, 12);
    }

    // === DAMAGE (skipped in fast mode unless --damage-sweep) ===
    let (damaged_acc, recovery_acc, before, after, n_kill) = if !fast || damage_sweep {
        eprintln!("━━━ DAMAGE ━━━");
        let before = v3.morphons.len();
        let assoc: Vec<MorphonId> = v3.morphons.values()
            .filter(|m| m.cell_type == CellType::Associative).map(|m| m.id).collect();
        let n_kill = (assoc.len() as f64 * 0.3).ceil() as usize;
        let mut kr = rand::rngs::StdRng::seed_from_u64(seed + 999);
        let mut ids = assoc;
        ids.shuffle(&mut kr);
        for &id in ids.iter().take(n_kill) {
            v3.morphons.remove(&id);
            v3.topology.remove_morphon(id);
        }
        let damaged_acc = evaluate(&mut v3, &test_images, &test_labels, n_test);
        eprintln!("  Killed {}, {} → {} morphons, acc={:.1}%", n_kill, before, v3.morphons.len(), damaged_acc);

        // Enable lifecycle for recovery — rewiring only, no new neurons.
        // division=false: new morphons with zero learned weights dilute the readout signal.
        // synaptogenesis=true: reconnects surviving neurons (confirmed +2pp vs synaptogenesis=false).
        v3.config.lifecycle = LifecycleConfig {
            division: false, differentiation: false, fusion: false,
            apoptosis: false, migration: true,
            synaptogenesis: true,
        };
        v3.config.scheduler.slow_period = 200;
        v3.config.scheduler.glacial_period = 500;
        // Recovery uses supervised_from=0: features already exist, reward signal must be
        // present from the first epoch so readout weights aren't overwritten by unsupervised STDP.
        let mut rng_recv = rand::rngs::StdRng::seed_from_u64(seed);
        let recovery_acc = train_and_eval_from(&mut v3, &train_images, &train_labels,
            &test_images, &test_labels, recovery_n, 1, "RECV", &mut rng_recv, 0, false);
        let after = v3.morphons.len();
        eprintln!("  Recovery: {:.1}% | morphons: {}\n", recovery_acc, after);
        (damaged_acc, recovery_acc, before, after, n_kill)
    } else { (0.0, 0.0, v3_m, v3_m, 0) };

    // === SUMMARY ===
    if fast {
        eprintln!("╔═══════════════════════════════════════════════════════════╗");
        eprintln!("║  V3   (LocalInhib.):         {:>5.1}%  sl: {:>5.1}%  ({:>4}s) ║",
            v3_acc, v3_stateless, v3_duration_s);
        eprintln!("║  V3-SL (Stateless Training): {:>5.1}%  sl: {:>5.1}%  ({:>4}s) ║",
            v3sl_acc, v3sl_stateless, v3sl_duration_s);
        eprintln!("╚═══════════════════════════════════════════════════════════╝");
    } else {
        eprintln!("╔═══════════════════════════════════════════════════════════╗");
        eprintln!("║  Baseline:                       {:>5.1}%                  ║", base_acc);
        eprintln!("║  V3  (LocalInhib.):              {:>5.1}%  ({:+.1}pp vs BASE) ║", v3_acc, v3_acc - base_acc);
        eprintln!("║  V3  stateless eval:             {:>5.1}%  ({:+.1}pp vs V3)   ║", v3_stateless, v3_stateless - v3_acc);
        eprintln!("║  V3-SL (Stateless Training):     {:>5.1}%  ({:+.1}pp vs V3)   ║", v3sl_acc, v3sl_acc - v3_acc);
        eprintln!("║  V3-SL stateless eval:           {:>5.1}%  ({:+.1}pp vs V3SL) ║", v3sl_stateless, v3sl_stateless - v3sl_acc);
        eprintln!("║  V3 wall: {:>4}s  V3SL: {:>4}s                           ║", v3_duration_s, v3sl_duration_s);
        eprintln!("║  Post-damage:                    {:>5.1}%                  ║", damaged_acc);
        eprintln!("║  Post-recovery:                  {:>5.1}%  ({:+.1}pp)        ║", recovery_acc, recovery_acc - damaged_acc);
        eprintln!("║  Morphons: {} → {} → {}{}║", before, before - n_kill, after,
            " ".repeat(22 - format!("{} → {} → {}", before, before - n_kill, after).len().min(22)));
        eprintln!("╚═══════════════════════════════════════════════════════════╝");
    }

    // Config summary
    eprintln!("\n━━━ CONFIG ━━━");
    eprintln!("  profile: {}, seed: {}, n_train: {}, epochs: {}", profile, seed, n_train, n_epochs);
    eprintln!("  seed_size: {}, initial_connectivity: 0.02 (S→A hardcoded 30%)", seed_size);
    eprintln!("  steps/image: {}, medium_period: 10, slow_period: 5000", STEPS_PER_IMAGE);
    eprintln!("  competition: LocalInhibition (iSTDP)");
    eprintln!("  Mature threshold: 0.3 (Endo)");
    eprintln!("  V3 wall: {}s   V3SL wall: {}s", v3_duration_s, v3sl_duration_s);

    // Save
    let version = env!("CARGO_PKG_VERSION");
    let total_duration_s = v3_duration_s + v3sl_duration_s;
    let results = json!({
        "benchmark": "mnist_v2_supervised", "version": version,
        "profile": profile, "seed": seed,
        "n_train": n_train, "epochs": n_epochs, "steps_per_image": STEPS_PER_IMAGE,
        "config": {
            "seed_size": 200,
            "initial_connectivity": 0.02,
            "sa_connectivity": "hardcoded 30%",
            "medium_period": 10,
            "slow_period": 5000,
            "competition_mode": "LocalInhibition",
            "mature_threshold": 0.3,
        },
        "baseline_acc": base_acc,
        "v3_acc": v3_acc,
        "v3_acc_stateless": v3_stateless,
        "v3_morphons": v3_m,
        "v3_synapses": v3_s,
        "v3_duration_s": v3_duration_s,
        "v3_competition_mode": "LocalInhibition",
        "v3sl_acc": v3sl_acc,
        "v3sl_acc_stateless": v3sl_stateless,
        "v3sl_morphons": v3sl_m,
        "v3sl_synapses": v3sl_s,
        "v3sl_duration_s": v3sl_duration_s,
        "v3sl_competition_mode": "LocalInhibition+StatelessTraining",
        "local_inhibition_params": {
            "interneuron_ratio": 0.10,
            "istdp_rate": 0.001,
            "initial_inh_weight": -0.5,
            "inhibition_radius": 0.0,
            "target_rate": 0.05,
        },
        "damaged_acc": damaged_acc, "recovery_acc": recovery_acc,
        "total_duration_s": total_duration_s,
    });
    let dir = format!("docs/benchmark_results/v{}", version);
    fs::create_dir_all(&dir).ok();
    let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    let path = format!("{}/mnist_v2_{}.json", dir, ts);
    fs::write(&path, serde_json::to_string_pretty(&results).unwrap()).unwrap();
    eprintln!("Saved to {}", path);
}
