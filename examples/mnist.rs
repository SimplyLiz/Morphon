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
use morphon_core::governance::ConstitutionalConstraints;
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
/// to select winners. Diehl & Cook use 350ms; we use 50 steps which gives
/// ~25 spikes per 50%-intensity pixel — enough for STDP coincidence detection.
const STEPS_PER_IMAGE: usize = 50;

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
/// Returns spike counts per Associative morphon (MorphonId → fire count).
fn present_image(system: &mut System, rates: &[f64], steps: usize, rng: &mut impl Rng)
    -> std::collections::HashMap<morphon_core::MorphonId, usize>
{
    // Inter-image reset — clear the slate so k-WTA selects based on THIS image.
    // Thresholds are NOT reset — adaptive thresholds persist across images
    // (Diehl & Cook: θ accumulates with firing, decays exponentially).
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

    // Reset synapse traces — prevents eligibility/STDP traces from the previous
    // image bleeding into this one. With tau_eligibility=5.0, traces from 5 steps
    // ago are still at 37% — enough to contaminate cross-image credit assignment.
    system.topology.update_all_synapses(|syn| {
        syn.eligibility = 0.0;
        syn.pre_trace = 0.0;
        syn.post_trace = 0.0;
    });

    // Track spike counts per associative morphon during this image presentation.
    // Counts (not binary) give graded labeling — a neuron that fires 40/50 steps
    // contributes more than one that fires 1/50 from Poisson noise.
    let mut spike_counts: std::collections::HashMap<morphon_core::MorphonId, usize> =
        std::collections::HashMap::new();

    for _ in 0..steps {
        let frame = poisson_frame(rates, rng);
        system.feed_input(&frame);
        // Continuous novelty injection — keeps modulation alive throughout presentation.
        // With novelty_decay=0.90/step, steady-state ≈ 0.03/0.10 = 0.3.
        // Without this, novelty (injected after the previous image) decays to ~0.002
        // by step 5, making three-factor weight updates near-zero.
        system.inject_novelty(0.03);
        system.step();

        // Record firings
        for m in system.morphons.values() {
            if m.fired && (m.cell_type == morphon_core::CellType::Associative
                || m.cell_type == morphon_core::CellType::Stem)
            {
                *spike_counts.entry(m.id).or_insert(0) += 1;
            }
        }
    }

    // Adaptive threshold decay (Diehl & Cook: τ_θ = 10^7ms, here simplified
    // to a per-image decay toward base threshold). Without this, thresholds
    // ratchet up forever and neurons that win early get permanently silenced.
    let theta_decay = 0.99; // ~1% decay per image → τ ≈ 100 images
    for m in system.morphons.values_mut() {
        if m.cell_type == morphon_core::CellType::Associative
            || m.cell_type == morphon_core::CellType::Stem
        {
            // Decay toward base threshold (0.5 is the default initial threshold)
            let base = 0.5;
            m.threshold = base + (m.threshold - base) * theta_decay;
        }
    }

    spike_counts
}

fn create_system(local_inh: bool) -> System {
    let competition_mode = if local_inh {
        // LocalInhibition: real inhibitory interneurons provide biological lateral
        // inhibition. Winners fire first, activate interneurons, which suppress
        // the rest via negative-weight synapses. This naturally generates the
        // anti-Hebbian LTD signal that non-winners need to specialize.
        morphon_core::homeostasis::CompetitionMode::LocalInhibition {
            interneuron_ratio: 0.10,     // 10% of associative budget → ~36 interneurons
            istdp_rate: 0.005,           // Vogels et al. — slow adaptation
            initial_inh_weight: -0.5,    // strong initial inhibition for MNIST competition
            inhibition_radius: 0.0,      // 0.0 = global groups (all-to-all inhibition)
            target_rate: Some(0.05),     // match k-WTA fraction — ~5% firing rate target
        }
    } else {
        morphon_core::homeostasis::CompetitionMode::GlobalKWTA {
            fraction: 0.05,
            local_radius: 0.0,
            local_k: 3,
        }
    };

    let config = SystemConfig {
        developmental: DevelopmentalConfig {
            seed_size: 500,
            dimensions: 6,
            initial_connectivity: 0.3,  // 30% S→A — sparse random receptive fields for STDP differentiation
            proliferation_rounds: 1,
            target_input_size: Some(IMG_PIXELS),
            target_output_size: Some(NUM_CLASSES),
            ..DevelopmentalConfig::cortical()
        },
        scheduler: SchedulerConfig {
            medium_period: 10,  // STDP every 10 steps — matches v2 baseline; period=1 compounds HSD 10× per fired morphon
            slow_period: 1000,  // structural changes very infrequent
            glacial_period: 5000,
            homeostasis_period: 50, // normalization once per image
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
            competition_mode,
            winner_boost: 0.04, // stronger adaptive threshold — faster winner rotation
            ..Default::default()
        },
        lifecycle: LifecycleConfig {
            division: false,     // fixed topology for STDP self-organization
            differentiation: true,
            fusion: false,
            apoptosis: false,    // OFF during Phase 1 — enabled at Phase 2 transition
            migration: false,
        },
        metabolic: MetabolicConfig {
            reward_energy_coefficient: 0.0,    // OFF during Phase 1 (no reward signal)
            superlinear_firing_factor: 0.0,    // OFF during Phase 1 (would kill indiscriminately)
            ..MetabolicConfig::default()
        },
        dt: 1.0,
        working_memory_capacity: 7,
        episodic_memory_capacity: 500,
        governance: ConstitutionalConstraints {
            // 30% connectivity target: sens→assoc dev wiring is 235 in-edges per assoc
            // (30% × 784). Cap at 300 gives synaptogenesis ~25% headroom for growth
            // while preventing rich-get-richer hub pathology.
            max_connectivity_per_morphon: 300,
            ..ConstitutionalConstraints::default()
        },
        ..Default::default()
    };
    System::new(config)
}

/// Classify using the analog readout (weighted sum of hidden potentials).
fn classify(system: &mut System, rates: &[f64], rng: &mut impl Rng) -> usize {
    let _fired = present_image(system, rates, STEPS_PER_IMAGE, rng);
    let outputs = system.read_output();
    if outputs.len() < NUM_CLASSES { return 0; }
    outputs.iter()
        .take(NUM_CLASSES)
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn parse_profile() -> &'static str {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--extended") { "extended" }
    else if args.iter().any(|a| a == "--standard") { "standard" }
    else { "quick" }
}

fn use_local_inhibition() -> bool {
    std::env::args().any(|a| a == "--local-inh")
}

fn main() {
    let profile = parse_profile();
    // phase3_samples: how many training images per Phase 3 epoch.
    // Standard/extended use full 10k set — more samples = better readout.
    // Quick uses 2k so the total run stays under ~5 minutes.
    let (phase1_epochs, phase1_samples, phase2_epochs, phase2_samples, phase3_samples, phase3_epochs, test_n) = match profile {
        "extended" => (3, 5000, 3, 5000, 10_000, 5, 1000),
        "standard" => (2, 3000, 2, 3000, 10_000, 3, 500),
        _          => (3, 1000, 2, 1000,  2_000, 5, 200),
    };

    let local_inh = use_local_inhibition();
    let competition = if local_inh { "LocalInhibition" } else { "GlobalKWTA" };
    println!("=== MORPHON MNIST Benchmark — Two-Phase Learning [{}, {}] ===\n", profile, competition);
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

    let mut system = create_system(local_inh);

    // Align homeostatic setpoint with k-WTA fraction (0.05 = 5% winners).
    // Default setpoint is 0.15 (15%), which causes synaptic_scaling to inflate
    // all weights 2× every homeostasis tick (scaling_factor = 0.15/0.05, clamped to 2.0).
    // It also makes the internal threshold adaptation in Morphon::step() fight
    // the Diehl & Cook adaptive threshold mechanism.
    for m in system.morphons.values_mut() {
        if m.cell_type == morphon_core::CellType::Associative
            || m.cell_type == morphon_core::CellType::Stem
        {
            m.homeostatic_setpoint = 0.05;
        }
    }

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
            // Present image as Poisson spike trains — k-WTA selects winners.
            // Novelty is injected continuously inside present_image() to keep
            // three-factor modulation alive throughout the 50-step presentation.
            let _fired = present_image(&mut system, &train_images[idx], STEPS_PER_IMAGE, &mut rng);

            // No label signal in Phase 1 — this phase is purely unsupervised.
            // reward_contrastive() belongs in Phase 1.5+ where the readout is enabled.

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
        let diag = system.diagnostics();
        println!("Phase1 Epoch {} complete | m={} s={} fr={:.3} w_mean={:.4} w_std={:.4}\n",
            epoch + 1, s.total_morphons, s.total_synapses, s.firing_rate,
            diag.weight_mean, diag.weight_std);
    }

    // =========================================================================
    // PHASE 1.5: Metabolic pruning — supervised reward drives hub death.
    // Enable metabolic pressure NOW that features exist from Phase 1.
    // Hubs fire for all classes → ±symmetric reward → zero energy income.
    // Specialists fire for one class → positive reward → energy income.
    // Superlinear firing cost starves hubs. Apoptosis kills them.
    // STDP stays ON so surviving morphons can refine features.
    // =========================================================================
    println!("--- PHASE 1.5: Metabolic pruning (reward-gated hub death) ---\n");

    system.config.metabolic.reward_energy_coefficient = 0.01;
    system.config.metabolic.superlinear_firing_factor = 2.0;
    system.config.lifecycle.apoptosis = true;

    // Disable HSD during pruning — we want metabolic/apoptotic pressure to select
    // neurons, not weight erosion. HSD during Phase 1 is already limited by
    // medium_period=10 (fires only on medium ticks where the morphon also fires).
    system.config.learning.heterosynaptic_depression = 0.0;

    // Enable analog readout for reward signal
    system.enable_analog_readout();

    let prune_samples = phase1_samples / 2; // half of Phase 1 budget
    let mut indices: Vec<usize> = (0..train_images.len()).collect();
    indices.shuffle(&mut rng);

    let s_before = system.inspect();
    for (bi, &idx) in indices.iter().take(prune_samples).enumerate() {
        let _fired = present_image(&mut system, &train_images[idx], STEPS_PER_IMAGE, &mut rng);
        let label = train_labels[idx];

        // Contrastive reward: injects into modulation.reward via inject_reward()
        // This is what makes reward_energy_coefficient work — specialists that
        // fire for the correct class earn energy, hubs don't.
        system.reward_contrastive(label, 0.5, 0.2);
        system.train_readout(label, 0.005);
        system.inject_novelty(0.3);

        if (bi + 1) % 500 == 0 || bi == 0 {
            let s = system.inspect();
            let diag = system.diagnostics();
            let deaths = s_before.total_morphons - s.total_morphons;
            println!("  Phase1.5 [{:>4}/{}] m={} s={} fr={:.3} deaths={} | {}",
                bi + 1, prune_samples,
                s.total_morphons, s.total_synapses, s.firing_rate, deaths,
                diag.summary());
        }
    }

    let s_after = system.inspect();
    let diag_after = system.diagnostics();
    let total_deaths = s_before.total_morphons - s_after.total_morphons;
    println!("Phase1.5 complete | m={} s={} fr={:.3} | {} morphons pruned | w_mean={:.4} w_std={:.4}\n",
        s_after.total_morphons, s_after.total_synapses, s_after.firing_rate, total_deaths,
        diag_after.weight_mean, diag_after.weight_std);

    // =========================================================================
    // PHASE 2: Post-hoc labeling (Diehl & Cook 2015)
    // Present labeled images, record which hidden neurons fire for each class.
    // Assign each neuron to the class it responds to most frequently.
    // At test time, classify by which class's assigned neurons are most active.
    // No gradient, no delta rule, no mode collapse — pure statistics.
    // =========================================================================
    println!("--- PHASE 2: Post-hoc neuron labeling (Diehl & Cook style) ---\n");

    // Freeze STDP weights — but keep k-WTA (homeostasis) and slow/glacial active.
    // medium_period freeze stops STDP weight updates so Phase 1 features are preserved.
    // homeostasis must keep running so k-WTA still selects sparse winners per image —
    // without it the hidden layer produces dense/no responses and labeling fails.
    // Synaptogenesis (slow path, period=1000) intentionally left on: it builds
    // class-specific connections between co-active neurons during labeling, giving
    // well-separated digit representations and higher posthoc accuracy.
    system.config.scheduler.medium_period = 999_999;
    // slow_period and glacial_period stay at their configured values (1000 / 5000)
    // homeostasis_period stays at 50 — k-WTA must keep running during labeling
    system.config.homeostasis.winner_boost = 0.0;
    system.config.lifecycle.apoptosis = false; // stop killing during eval

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
        let fired = present_image(&mut system, &train_images[idx], STEPS_PER_IMAGE, &mut rng);
        let label = train_labels[idx];

        // Record spike-count-weighted class responses. A neuron that fires 40/50 steps
        // contributes 40× more to its class count than one that fires 1/50 (noise).
        for (&aid, &spikes) in &fired {
            if let Some(counts) = neuron_class_counts.get_mut(&aid) {
                counts[label] += spikes;
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

    let max_class_share = class_neuron_counts.iter().max().copied().unwrap_or(0);
    let collapse_pct = max_class_share as f64 / labeled_count.max(1) as f64 * 100.0;
    if collapse_pct > 60.0 {
        println!("  *** WARNING: MODE COLLAPSE — {:.0}% of neurons assigned to one class ***\n", collapse_pct);
    }

    // Test: classify by which class's neurons are most active
    println!("--- Testing with post-hoc labels ---\n");
    let mut cc = vec![0usize; NUM_CLASSES];
    let mut ct = vec![0usize; NUM_CLASSES];
    for i in 0..test_images.len().min(test_n) {
        let fired = present_image(&mut system, &test_images[i], STEPS_PER_IMAGE, &mut rng);

        // Spike-count-weighted voting: each fired neuron's vote is proportional
        // to how many steps it fired, matching the graded labeling scheme.
        let mut class_votes = vec![0usize; NUM_CLASSES];
        for (&aid, &assigned_class) in &neuron_labels {
            if let Some(&spikes) = fired.get(&aid) {
                class_votes[assigned_class] += spikes;
            }
        }

        // Guard against silent network: if no neurons fired, skip this image
        // rather than letting max_by_key tie-break to the last index (digit 9).
        let max_votes = *class_votes.iter().max().unwrap_or(&0);
        if max_votes == 0 { ct[test_labels[i]] += 1; continue; }

        let pred = class_votes.iter().enumerate()
            .max_by_key(|&(_, count)| count)
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

    // =========================================================================
    // PHASE 3: Aggressive supervised readout training — frozen hidden layer.
    //
    // Post-hoc labeling (Phase 2) is a crude WTA vote: each neuron is pinned
    // to one class and contributes equally. It discards all graded signal from
    // neurons that respond weakly to a class.
    //
    // The delta-rule readout can use ALL 500 hidden neurons with learned weights,
    // extracting far more discriminative signal from the same STDP features.
    // Reservoir-computing literature shows this bumps ~52% post-hoc → 85%+.
    //
    // NOW freeze slow/glacial — Phase 2 synaptogenesis built class-specific topology;
    // Phase 3 needs the network to be structurally stable so the readout maps to
    // consistent hidden representations. Without this, ongoing synaptogenesis
    // rewires the reservoir mid-training → FR→0, 8% readout accuracy.
    system.config.scheduler.slow_period = 999_999;
    system.config.scheduler.glacial_period = 999_999;
    // =========================================================================
    println!("--- PHASE 3: Supervised readout fine-tuning ({phase3_epochs} epochs, LR=0.05) ---\n");

    const PHASE3_LR: f64 = 0.05;

    for epoch in 0..phase3_epochs {
        let mut indices: Vec<usize> = (0..train_images.len()).collect();
        indices.shuffle(&mut rng);

        for (bi, &idx) in indices.iter().take(phase3_samples).enumerate() {
            let _fired = present_image(&mut system, &train_images[idx], STEPS_PER_IMAGE, &mut rng);
            system.train_readout(train_labels[idx], PHASE3_LR);

            if (bi + 1) % 2000 == 0 {
                println!("  Phase3 Ep{} [{:>5}/{phase3_samples}]", epoch + 1, bi + 1);
            }
        }

        // Mid-training accuracy check (analog readout)
        let mut e_correct = 0usize;
        let e_total = test_images.len().min(test_n);
        for i in 0..e_total {
            let p = classify(&mut system, &test_images[i], &mut rng);
            if p == test_labels[i] { e_correct += 1; }
        }
        let e_acc = e_correct as f64 / e_total as f64 * 100.0;
        println!("Phase3 Epoch {}/{phase3_epochs} → readout acc: {e_acc:.1}%\n", epoch + 1);
    }

    // === Analog Readout Evaluation ===
    // Full per-class breakdown after Phase 3 readout training.
    println!("\n=== Analog Readout Test ===");
    let mut ar_cc = vec![0usize; NUM_CLASSES];
    let mut ar_ct = vec![0usize; NUM_CLASSES];
    for i in 0..test_images.len().min(test_n) {
        let _fired = present_image(&mut system, &test_images[i], STEPS_PER_IMAGE, &mut rng);
        let outputs = system.read_output();
        let pred = if outputs.len() >= NUM_CLASSES {
            outputs.iter().take(NUM_CLASSES).enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i).unwrap_or(0)
        } else { 0 };
        ar_ct[test_labels[i]] += 1;
        if pred == test_labels[i] { ar_cc[test_labels[i]] += 1; }
    }
    let ar_total_correct: usize = ar_cc.iter().sum();
    let ar_total_tested: usize = ar_ct.iter().sum();
    let ar_acc = if ar_total_tested > 0 { ar_total_correct as f64 / ar_total_tested as f64 * 100.0 } else { 0.0 };
    for c in 0..NUM_CLASSES {
        if ar_ct[c] > 0 {
            let class_acc = ar_cc[c] as f64 / ar_ct[c] as f64 * 100.0;
            println!("  {}: {:.1}% ({}/{})", c, class_acc, ar_cc[c], ar_ct[c]);
        }
    }
    println!("Analog readout accuracy: {:.1}%", ar_acc);

    println!("\n=== Final ===");
    let s = system.inspect();
    let diag = system.diagnostics();
    println!("Post-hoc accuracy: {:.1}%", test_acc);
    println!("Readout accuracy:  {:.1}%", ar_acc);
    println!("Morphons: {} | Synapses: {} | Clusters: {} | Gen: {} | FR: {:.3}",
        s.total_morphons, s.total_synapses, s.fused_clusters, s.max_generation, s.firing_rate);
    println!("Learning: {}", diag.summary());

    // Save results
    let version = env!("CARGO_PKG_VERSION");
    let results = json!({
        "benchmark": "mnist",
        "profile": profile,
        "method": format!("two-phase (unsupervised STDP + {})", competition),
        "version": version,
        "timestamp": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
        "phase1": { "epochs": phase1_epochs, "samples_per_epoch": phase1_samples },
        "phase2": { "epochs": phase2_epochs, "samples_per_epoch": phase2_samples },
        "phase3": { "epochs": phase3_epochs, "samples_per_epoch": phase3_samples, "lr": PHASE3_LR },
        "results": {
            "posthoc_accuracy": test_acc,
            "readout_accuracy": ar_acc,
            "per_class_posthoc": per_class,
        },
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
