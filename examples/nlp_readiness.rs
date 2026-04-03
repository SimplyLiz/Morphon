//! NLP Readiness Benchmark — measures how close Morphon is to handling language.
//!
//! Four tiers test capabilities that scale toward NLP:
//!   Tier 0 "Bag-of-Chars":  Can the system discriminate character distributions? (27-dim)
//!   Tier 1 "One-Hot Scale":  Can it handle full one-hot text encoding? (135-dim)
//!   Tier 2 "Memory":         Can it remember across sequential inputs? (27-dim x 3 steps)
//!   Tier 3 "Composition":    Can it combine token meanings? (54-dim, XOR)
//!
//! Outputs a readiness level 0-3 and saves JSON results.
//! All data is synthetic — no external downloads needed.
//!
//! Run: cargo run --example nlp_readiness --release
//! Run: cargo run --example nlp_readiness --release -- --standard
//! Run: cargo run --example nlp_readiness --release -- --extended

use morphon_core::developmental::DevelopmentalConfig;
use morphon_core::endoquilibrium::EndoConfig;
use morphon_core::homeostasis::HomeostasisParams;
use morphon_core::learning::LearningParams;
use morphon_core::morphogenesis::MorphogenesisParams;
use morphon_core::morphon::MetabolicConfig;
use morphon_core::scheduler::SchedulerConfig;
use morphon_core::system::{System, SystemConfig};
use morphon_core::types::LifecycleConfig;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde_json::json;
use std::fs;
use std::time::Instant;

// ── Constants ────────────────────────────────────────────────────────────────

const ALPHABET_SIZE: usize = 27; // a-z + space
const WORD_LEN: usize = 5;
const ONEHOT_DIM: usize = WORD_LEN * ALPHABET_SIZE; // 135
const TIER2_SEQ_LEN: usize = 3;
const TIER3_INPUT_DIM: usize = 2 * ALPHABET_SIZE; // 54
const STEPS_PER_PROCESS: usize = 5;
const AMPLITUDE: f64 = 3.0; // one-hot activation strength (matches classify_tiny)

const TIER0_PASS: f64 = 65.0;
const TIER1_PASS: f64 = 60.0;
const TIER2_PASS: f64 = 55.0;
const TIER3_PASS: f64 = 60.0;

const VOWELS: [char; 5] = ['a', 'e', 'i', 'o', 'u'];

// ── Encoding helpers ─────────────────────────────────────────────────────────

fn char_to_onehot(c: char) -> Vec<f64> {
    let mut v = vec![0.0; ALPHABET_SIZE];
    let idx = match c {
        'a'..='z' => (c as usize) - ('a' as usize),
        _ => 26,
    };
    v[idx] = AMPLITUDE;
    v
}

fn encode_flat_onehot(chars: &[char]) -> Vec<f64> {
    let mut out = Vec::with_capacity(chars.len() * ALPHABET_SIZE);
    for &c in chars {
        out.extend_from_slice(&char_to_onehot(c));
    }
    out
}

/// Bag-of-characters encoding: 27-dim frequency vector.
/// Each occurrence adds AMPLITUDE to that character's channel.
/// Produces strong signals comparable to classify_tiny's encoding.
fn encode_bag_of_chars(chars: &[char]) -> Vec<f64> {
    let mut freq = vec![0.0; ALPHABET_SIZE];
    for &c in chars {
        let idx = match c {
            'a'..='z' => (c as usize) - ('a' as usize),
            _ => 26,
        };
        freq[idx] += AMPLITUDE;
    }
    freq
}

fn is_vowel(c: char) -> bool {
    VOWELS.contains(&c)
}

fn random_vowel(rng: &mut impl Rng) -> char {
    VOWELS[rng.random_range(0..VOWELS.len())]
}

fn random_consonant(rng: &mut impl Rng) -> char {
    let consonants: Vec<char> = ('a'..='z').filter(|c| !is_vowel(*c)).collect();
    consonants[rng.random_range(0..consonants.len())]
}

fn random_letter(rng: &mut impl Rng) -> char {
    (b'a' + rng.random_range(0..26u8)) as char
}

// ── Data generators ──────────────────────────────────────────────────────────

/// Tier 0: Bag-of-characters — vowel-heavy (class 0) vs consonant-heavy (class 1).
/// 27-dim frequency encoding. Same task as before but at manageable input scale.
fn gen_tier0_sample(rng: &mut impl Rng) -> (Vec<f64>, usize) {
    let class = rng.random_range(0..2usize);
    let mut chars = vec![' '; WORD_LEN];
    match class {
        0 => {
            // 3 vowels, 2 consonants
            for i in 0..3 { chars[i] = random_vowel(rng); }
            for i in 3..5 { chars[i] = random_consonant(rng); }
        }
        _ => {
            // 4 consonants, 1 vowel
            for i in 0..4 { chars[i] = random_consonant(rng); }
            chars[4] = random_vowel(rng);
        }
    }
    // Shuffle to avoid ordering bias
    for i in (1..WORD_LEN).rev() {
        let j = rng.random_range(0..=i);
        chars.swap(i, j);
    }
    (encode_bag_of_chars(&chars), class)
}

/// Tier 1: Full one-hot encoding — vowel-heavy vs consonant-heavy at 135-dim scale.
/// Same underlying task as Tier 0 but tests whether the system can handle
/// the input dimensionality that real text requires.
fn gen_tier1_sample(rng: &mut impl Rng) -> (Vec<f64>, usize) {
    let class = rng.random_range(0..2usize);
    let mut chars = vec![' '; WORD_LEN];
    match class {
        0 => {
            for i in 0..3 { chars[i] = random_vowel(rng); }
            for i in 3..5 { chars[i] = random_consonant(rng); }
        }
        _ => {
            for i in 0..4 { chars[i] = random_consonant(rng); }
            chars[4] = random_vowel(rng);
        }
    }
    for i in (1..WORD_LEN).rev() {
        let j = rng.random_range(0..=i);
        chars.swap(i, j);
    }
    (encode_flat_onehot(&chars), class)
}

/// Tier 2: Sequential memory. First char determines class, fed one-at-a-time.
/// Class 0: first char is vowel. Class 1: first char is consonant.
fn gen_tier2_sample(rng: &mut impl Rng) -> (Vec<char>, usize) {
    let class = rng.random_range(0..2usize);
    let mut chars = Vec::with_capacity(TIER2_SEQ_LEN);
    chars.push(match class {
        0 => random_vowel(rng),
        _ => random_consonant(rng),
    });
    for _ in 1..TIER2_SEQ_LEN {
        chars.push(random_letter(rng));
    }
    (chars, class)
}

/// Tier 3: Compositional XOR. Two single-char "tokens", class = same-group vs cross-group.
/// Group A: vowels. Group B: consonants.
/// Class 0: both same group (VV or CC). Class 1: different groups (VC or CV).
fn gen_tier3_sample(rng: &mut impl Rng) -> (Vec<f64>, usize) {
    let a_is_vowel = rng.random_bool(0.5);
    let b_is_vowel = rng.random_bool(0.5);
    let class = if a_is_vowel == b_is_vowel { 0 } else { 1 };

    let a = if a_is_vowel { random_vowel(rng) } else { random_consonant(rng) };
    let b = if b_is_vowel { random_vowel(rng) } else { random_consonant(rng) };

    let mut input = char_to_onehot(a);
    input.extend_from_slice(&char_to_onehot(b));
    (input, class)
}

// ── System factory ───────────────────────────────────────────────────────────

fn default_learning() -> LearningParams {
    LearningParams {
        tau_eligibility: 10.0,
        tau_trace: 10.0,
        a_plus: 1.0,
        a_minus: -1.0,
        tau_tag: 200.0,
        tag_threshold: 0.3,
        capture_threshold: 0.2,
        capture_rate: 0.2,
        weight_max: 3.0,
        weight_min: 0.01,
        alpha_reward: 0.5,
        alpha_novelty: 3.0,
        alpha_arousal: 0.0,
        alpha_homeostasis: 0.1,
        transmitter_potentiation: 0.001,
        heterosynaptic_depression: 0.002,
        tag_accumulation_rate: 0.3,
    }
}

fn create_system(input_size: usize, output_size: usize, tier: usize) -> System {
    let (developmental, scheduler, learning) = match tier {
        0 => (
            // Tier 0: 27-dim bag-of-chars — small, classify_tiny-scale
            DevelopmentalConfig {
                seed_size: 40,
                dimensions: 4,
                initial_connectivity: 0.15,
                proliferation_rounds: 1,
                target_input_size: Some(input_size),
                target_output_size: Some(output_size),
                ..DevelopmentalConfig::cortical()
            },
            SchedulerConfig {
                medium_period: 99999, // three-factor OFF — pure delta rule
                slow_period: 99999,
                glacial_period: 99999,
                homeostasis_period: 10,
                memory_period: 99999,
            },
            default_learning(),
        ),
        1 => (
            // Tier 1: 135-dim one-hot — needs larger network + full pipeline
            DevelopmentalConfig {
                seed_size: 300,
                dimensions: 5,
                initial_connectivity: 0.15,
                proliferation_rounds: 1,
                target_input_size: Some(input_size),
                target_output_size: Some(output_size),
                ..DevelopmentalConfig::cortical()
            },
            SchedulerConfig {
                medium_period: 1,     // three-factor ON
                slow_period: 100,
                glacial_period: 99999,
                homeostasis_period: 10,
                memory_period: 99999,
            },
            default_learning(),
        ),
        2 => (
            // Tier 2: Sequential — hippocampal for temporal
            DevelopmentalConfig {
                seed_size: 50,
                dimensions: 4,
                initial_connectivity: 0.15,
                proliferation_rounds: 1,
                target_input_size: Some(input_size),
                target_output_size: Some(output_size),
                ..DevelopmentalConfig::hippocampal()
            },
            SchedulerConfig {
                medium_period: 1,
                slow_period: 100,
                glacial_period: 99999,
                homeostasis_period: 10,
                memory_period: 99999,
            },
            LearningParams {
                tau_eligibility: 15.0,
                tau_trace: 20.0,
                alpha_novelty: 2.0,
                ..default_learning()
            },
        ),
        _ => (
            // Tier 3: XOR — needs hidden layer learning
            DevelopmentalConfig {
                seed_size: 60,
                dimensions: 4,
                initial_connectivity: 0.15,
                proliferation_rounds: 1,
                target_input_size: Some(input_size),
                target_output_size: Some(output_size),
                ..DevelopmentalConfig::cortical()
            },
            SchedulerConfig {
                medium_period: 1,
                slow_period: 100,
                glacial_period: 99999,
                homeostasis_period: 10,
                memory_period: 99999,
            },
            default_learning(),
        ),
    };

    let config = SystemConfig {
        developmental,
        scheduler,
        learning,
        morphogenesis: MorphogenesisParams {
            max_morphons: Some(800),
            ..Default::default()
        },
        homeostasis: HomeostasisParams::default(),
        lifecycle: LifecycleConfig {
            division: false,
            differentiation: false,
            fusion: false,
            apoptosis: false,
            migration: true,
        },
        metabolic: MetabolicConfig::default(),
        endoquilibrium: EndoConfig::default(),
        dt: 1.0,
        working_memory_capacity: 7,
        episodic_memory_capacity: 100,
        ..Default::default()
    };
    System::new(config)
}

// ── Profile ──────────────────────────────────────────────────────────────────

struct ProfileParams {
    epochs_flat: usize,
    epochs_seq: usize,
    epochs_comp: usize,
    samples_per_epoch: usize,
    test_samples: usize,
}

fn parse_profile() -> (&'static str, ProfileParams) {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--extended") {
        ("extended", ProfileParams {
            epochs_flat: 300, epochs_seq: 300, epochs_comp: 500,
            samples_per_epoch: 1000, test_samples: 500,
        })
    } else if args.iter().any(|a| a == "--standard") {
        ("standard", ProfileParams {
            epochs_flat: 100, epochs_seq: 100, epochs_comp: 150,
            samples_per_epoch: 500, test_samples: 200,
        })
    } else {
        ("quick", ProfileParams {
            epochs_flat: 30, epochs_seq: 30, epochs_comp: 50,
            samples_per_epoch: 200, test_samples: 100,
        })
    }
}

// ── Shared helpers ───────────────────────────────────────────────────────────

fn classify(system: &mut System, input: &[f64], n_classes: usize) -> usize {
    let outputs = system.process_steps(input, STEPS_PER_PROCESS);
    if outputs.len() < n_classes { return 0; }
    outputs.iter()
        .take(n_classes)
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn evaluate(
    system: &mut System,
    gen_fn: &mut dyn FnMut() -> (Vec<f64>, usize),
    n_classes: usize,
    n_test: usize,
) -> f64 {
    let mut correct = 0;
    for _ in 0..n_test {
        let (input, label) = gen_fn();
        system.reset_voltages();
        let pred = classify(system, &input, n_classes);
        if pred == label { correct += 1; }
    }
    correct as f64 / n_test as f64 * 100.0
}

// ── Tier runners ─────────────────────────────────────────────────────────────

/// Tier 0: Bag-of-characters — vowel-heavy vs consonant-heavy (27-dim).
/// Same approach as classify_tiny: process_steps + teach_supervised + step.
fn run_tier0(params: &ProfileParams) -> (f64, serde_json::Value, u64) {
    let t0 = Instant::now();
    let mut system = create_system(ALPHABET_SIZE, 2, 0);
    let mut data_rng = StdRng::seed_from_u64(42);

    let s = system.inspect();
    println!("  System: {} morphons, {} synapses, {} in, {} out",
        s.total_morphons, s.total_synapses, system.input_size(), system.output_size());

    for epoch in 0..params.epochs_flat {
        for _ in 0..params.samples_per_epoch {
            let (input, label) = gen_tier0_sample(&mut data_rng);
            system.reset_voltages();
            let _pred = classify(&mut system, &input, 2);
            system.teach_supervised(label, 0.01);
            system.step();
        }
        if (epoch + 1) % 10 == 0 || epoch == 0 {
            let mut test_rng = StdRng::seed_from_u64(1000);
            let acc = evaluate(&mut system, &mut || gen_tier0_sample(&mut test_rng), 2, params.test_samples);
            println!("  Epoch {:>3} | test {:.1}%", epoch + 1, acc);
        }
    }

    let mut test_rng = StdRng::seed_from_u64(9999);
    let acc = evaluate(&mut system, &mut || gen_tier0_sample(&mut test_rng), 2, params.test_samples);
    let s = system.inspect();
    let stats = json!({
        "morphons": s.total_morphons, "synapses": s.total_synapses,
        "firing_rate": s.firing_rate,
    });
    (acc, stats, t0.elapsed().as_millis() as u64)
}

/// Tier 1: Full one-hot encoding at 135-dim — same task, tests dimensional scaling.
fn run_tier1(params: &ProfileParams) -> (f64, serde_json::Value, u64) {
    let t0 = Instant::now();
    let mut system = create_system(ONEHOT_DIM, 2, 1);
    let mut data_rng = StdRng::seed_from_u64(43);

    let s = system.inspect();
    println!("  System: {} morphons, {} synapses, {} in, {} out",
        s.total_morphons, s.total_synapses, system.input_size(), system.output_size());

    for epoch in 0..params.epochs_flat {
        for _ in 0..params.samples_per_epoch {
            let (input, label) = gen_tier1_sample(&mut data_rng);
            system.reset_voltages();
            let _pred = classify(&mut system, &input, 2);
            // Full pipeline: delta rule with raw input values + contrastive + novelty
            system.teach_supervised_with_input(&input, label, 0.01);
            system.reward_contrastive(label, 0.3, 0.15);
            system.inject_novelty(0.3);
            system.step();
        }
        if (epoch + 1) % 10 == 0 || epoch == 0 {
            let mut test_rng = StdRng::seed_from_u64(1001);
            let acc = evaluate(&mut system, &mut || gen_tier1_sample(&mut test_rng), 2, params.test_samples);
            println!("  Epoch {:>3} | test {:.1}%", epoch + 1, acc);
        }
    }

    let mut test_rng = StdRng::seed_from_u64(9998);
    let acc = evaluate(&mut system, &mut || gen_tier1_sample(&mut test_rng), 2, params.test_samples);
    let s = system.inspect();
    let stats = json!({
        "morphons": s.total_morphons, "synapses": s.total_synapses,
        "firing_rate": s.firing_rate,
    });
    (acc, stats, t0.elapsed().as_millis() as u64)
}

/// Tier 2: Sequential memory — chars fed one at a time, classify by first char.
fn run_tier2(params: &ProfileParams) -> (f64, serde_json::Value, u64) {
    let t0 = Instant::now();
    let mut system = create_system(ALPHABET_SIZE, 2, 2);
    let mut data_rng = StdRng::seed_from_u64(44);

    let s = system.inspect();
    println!("  System: {} morphons, {} synapses, {} in, {} out",
        s.total_morphons, s.total_synapses, system.input_size(), system.output_size());

    for epoch in 0..params.epochs_seq {
        for _ in 0..params.samples_per_epoch {
            let (chars, label) = gen_tier2_sample(&mut data_rng);
            system.reset_voltages();

            // Feed each character sequentially — no reset between chars
            let mut last_input = vec![0.0; ALPHABET_SIZE];
            for &c in &chars {
                last_input = char_to_onehot(c);
                system.process_steps(&last_input, STEPS_PER_PROCESS);
            }

            system.teach_supervised_with_input(&last_input, label, 0.01);
            system.reward_contrastive(label, 0.3, 0.15);
            system.inject_novelty(0.3);
            system.step();
        }
        if (epoch + 1) % 10 == 0 || epoch == 0 {
            let mut test_rng = StdRng::seed_from_u64(1002);
            let mut correct = 0;
            for _ in 0..params.test_samples {
                let (chars, label) = gen_tier2_sample(&mut test_rng);
                system.reset_voltages();
                for &c in &chars {
                    let input = char_to_onehot(c);
                    system.process_steps(&input, STEPS_PER_PROCESS);
                }
                let outputs = system.read_output();
                let pred = if outputs.len() >= 2 {
                    if outputs[0] >= outputs[1] { 0 } else { 1 }
                } else { 0 };
                if pred == label { correct += 1; }
            }
            let acc = correct as f64 / params.test_samples as f64 * 100.0;
            println!("  Epoch {:>3} | test {:.1}%", epoch + 1, acc);
        }
    }

    // Final evaluation
    let mut test_rng = StdRng::seed_from_u64(9997);
    let mut correct = 0;
    for _ in 0..params.test_samples {
        let (chars, label) = gen_tier2_sample(&mut test_rng);
        system.reset_voltages();
        for &c in &chars {
            let input = char_to_onehot(c);
            system.process_steps(&input, STEPS_PER_PROCESS);
        }
        let outputs = system.read_output();
        let pred = if outputs.len() >= 2 {
            if outputs[0] >= outputs[1] { 0 } else { 1 }
        } else { 0 };
        if pred == label { correct += 1; }
    }
    let acc = correct as f64 / params.test_samples as f64 * 100.0;

    let s = system.inspect();
    let stats = json!({
        "morphons": s.total_morphons, "synapses": s.total_synapses,
        "firing_rate": s.firing_rate,
    });
    (acc, stats, t0.elapsed().as_millis() as u64)
}

/// Tier 3: Compositional XOR — same-group vs cross-group token pairs.
fn run_tier3(params: &ProfileParams) -> (f64, serde_json::Value, u64) {
    let t0 = Instant::now();
    let mut system = create_system(TIER3_INPUT_DIM, 2, 3);
    let mut data_rng = StdRng::seed_from_u64(45);

    let s = system.inspect();
    println!("  System: {} morphons, {} synapses, {} in, {} out",
        s.total_morphons, s.total_synapses, system.input_size(), system.output_size());

    for epoch in 0..params.epochs_comp {
        for _ in 0..params.samples_per_epoch {
            let (input, label) = gen_tier3_sample(&mut data_rng);
            system.reset_voltages();
            let _pred = classify(&mut system, &input, 2);

            system.teach_supervised_with_input(&input, label, 0.02);
            system.reward_contrastive(label, 0.3, 0.15);
            system.inject_novelty(0.3);
            system.step();
        }
        if (epoch + 1) % 10 == 0 || epoch == 0 {
            let mut test_rng = StdRng::seed_from_u64(1003);
            let acc = evaluate(&mut system, &mut || gen_tier3_sample(&mut test_rng), 2, params.test_samples);
            println!("  Epoch {:>3} | test {:.1}%", epoch + 1, acc);
        }
    }

    let mut test_rng = StdRng::seed_from_u64(9996);
    let acc = evaluate(&mut system, &mut || gen_tier3_sample(&mut test_rng), 2, params.test_samples);
    let s = system.inspect();
    let stats = json!({
        "morphons": s.total_morphons, "synapses": s.total_synapses,
        "firing_rate": s.firing_rate,
    });
    (acc, stats, t0.elapsed().as_millis() as u64)
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    let (profile, params) = parse_profile();

    println!("=== MORPHON NLP Readiness Benchmark [{}] ===\n", profile);

    // ── Tier 0 ──
    println!("--- Tier 0: Bag-of-Characters (vowel vs consonant heavy, {} dim) ---", ALPHABET_SIZE);
    let (t0_acc, t0_stats, t0_ms) = run_tier0(&params);
    let t0_pass = t0_acc >= TIER0_PASS;
    println!("  => {:.1}% (pass={}, threshold={:.0}%, time={}ms)\n", t0_acc, t0_pass, TIER0_PASS, t0_ms);

    // ── Tier 1 ──
    println!("--- Tier 1: One-Hot Scale (same task, {} dim) ---", ONEHOT_DIM);
    let (t1_acc, t1_stats, t1_ms) = run_tier1(&params);
    let t1_pass = t1_acc >= TIER1_PASS;
    println!("  => {:.1}% (pass={}, threshold={:.0}%, time={}ms)\n", t1_acc, t1_pass, TIER1_PASS, t1_ms);

    // ── Tier 2 ──
    println!("--- Tier 2: Memory (sequential {}-char, classify by first, {} dim/step) ---",
        TIER2_SEQ_LEN, ALPHABET_SIZE);
    let (t2_acc, t2_stats, t2_ms) = run_tier2(&params);
    let t2_pass = t2_acc >= TIER2_PASS;
    println!("  => {:.1}% (pass={}, threshold={:.0}%, time={}ms)\n", t2_acc, t2_pass, TIER2_PASS, t2_ms);

    // ── Tier 3 ──
    println!("--- Tier 3: Composition (token-pair XOR, {} dim) ---", TIER3_INPUT_DIM);
    let (t3_acc, t3_stats, t3_ms) = run_tier3(&params);
    let t3_pass = t3_acc >= TIER3_PASS;
    println!("  => {:.1}% (pass={}, threshold={:.0}%, time={}ms)\n", t3_acc, t3_pass, TIER3_PASS, t3_ms);

    // ── Readiness level ──
    let readiness_level = if !t0_pass { 0 }
        else if !t1_pass { 1 }
        else if !t2_pass { 2 }
        else { 3 };

    let level_desc = match readiness_level {
        0 => "Not ready -- cannot discriminate text-like patterns",
        1 => "Can encode -- bag-of-chars works, one-hot scale fails",
        2 => "Scales -- handles one-hot encoding, no temporal memory",
        3 => "Temporal -- encodes + scales + memory across frames",
        _ => unreachable!(),
    };

    println!("========================================================");
    println!("  NLP READINESS LEVEL: {}/3", readiness_level);
    println!("  {}", level_desc);
    println!("  Composition (XOR): {}", if t3_pass { "PASS" } else { "FAIL" });
    println!("--------------------------------------------------------");
    println!("  Tier 0 Bag-of-Chars:  {:>5.1}% {} (>={:.0}%)", t0_acc,
        if t0_pass { "PASS" } else { "FAIL" }, TIER0_PASS);
    println!("  Tier 1 One-Hot Scale: {:>5.1}% {} (>={:.0}%)", t1_acc,
        if t1_pass { "PASS" } else { "FAIL" }, TIER1_PASS);
    println!("  Tier 2 Memory:        {:>5.1}% {} (>={:.0}%)", t2_acc,
        if t2_pass { "PASS" } else { "FAIL" }, TIER2_PASS);
    println!("  Tier 3 Composition:   {:>5.1}% {} (>={:.0}%)", t3_acc,
        if t3_pass { "PASS" } else { "FAIL" }, TIER3_PASS);
    println!("========================================================");
    println!();
    println!("Note: This benchmark tests NLP-relevant capabilities, not transformer internals.");
    println!("Morphon's path to language is brain-like circuits (competitive selection,");
    println!("working memory, recurrent dynamics) rather than QKV attention.");

    // ── Save JSON ──
    let version = env!("CARGO_PKG_VERSION");
    let summary = format!("NLP Readiness Level {}/3: {}", readiness_level, level_desc);

    let results = json!({
        "benchmark": "nlp_readiness",
        "version": version,
        "profile": profile,
        "timestamp": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
        "readiness_level": readiness_level,
        "composition_capable": t3_pass,
        "tiers": {
            "tier0_bag_of_chars": {
                "accuracy": t0_acc,
                "chance_level": 50.0,
                "pass_threshold": TIER0_PASS,
                "passed": t0_pass,
                "input_dim": ALPHABET_SIZE,
                "output_dim": 2,
                "encoding": "bag-of-characters (27-dim frequency)",
                "train_epochs": params.epochs_flat,
                "samples_per_epoch": params.samples_per_epoch,
                "test_samples": params.test_samples,
                "training_time_ms": t0_ms,
                "system": t0_stats,
            },
            "tier1_onehot_scale": {
                "accuracy": t1_acc,
                "chance_level": 50.0,
                "pass_threshold": TIER1_PASS,
                "passed": t1_pass,
                "input_dim": ONEHOT_DIM,
                "output_dim": 2,
                "encoding": "one-hot (5 chars x 27 = 135-dim)",
                "train_epochs": params.epochs_flat,
                "samples_per_epoch": params.samples_per_epoch,
                "test_samples": params.test_samples,
                "training_time_ms": t1_ms,
                "system": t1_stats,
            },
            "tier2_memory": {
                "accuracy": t2_acc,
                "chance_level": 50.0,
                "pass_threshold": TIER2_PASS,
                "passed": t2_pass,
                "input_dim": ALPHABET_SIZE,
                "output_dim": 2,
                "seq_length": TIER2_SEQ_LEN,
                "encoding": "one-hot sequential (27-dim per step)",
                "train_epochs": params.epochs_seq,
                "samples_per_epoch": params.samples_per_epoch,
                "test_samples": params.test_samples,
                "training_time_ms": t2_ms,
                "system": t2_stats,
            },
            "tier3_composition": {
                "accuracy": t3_acc,
                "chance_level": 50.0,
                "pass_threshold": TIER3_PASS,
                "passed": t3_pass,
                "input_dim": TIER3_INPUT_DIM,
                "output_dim": 2,
                "encoding": "one-hot pair (2 x 27 = 54-dim)",
                "train_epochs": params.epochs_comp,
                "samples_per_epoch": params.samples_per_epoch,
                "test_samples": params.test_samples,
                "training_time_ms": t3_ms,
                "system": t3_stats,
            },
        },
        "summary": summary,
    });

    let dir = format!("docs/benchmark_results/v{}", version);
    fs::create_dir_all(&dir).ok();
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    let run_path = format!("{}/nlp_{}.json", dir, ts);
    let latest_path = format!("{}/nlp_latest.json", dir);
    let json_str = serde_json::to_string_pretty(&results).unwrap();
    fs::write(&run_path, &json_str).unwrap();
    fs::write(&latest_path, &json_str).unwrap();
    println!("\nResults saved to {}", run_path);
}
