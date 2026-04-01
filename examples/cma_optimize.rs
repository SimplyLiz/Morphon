//! CMA-ES meta-learning — optimize learning parameters for classification.
//!
//! Uses the classify_tiny task as fitness function.
//! Searches over 15 key parameters to find a configuration where
//! the three-factor learning actually converges.
//!
//! Run: cargo run --example cma_optimize --release

use cmaes::{CMAESOptions, DVector};
use morphon_core::developmental::DevelopmentalConfig;
use morphon_core::learning::LearningParams;
use morphon_core::morphogenesis::MorphogenesisParams;
use morphon_core::morphon::MetabolicConfig;
use morphon_core::scheduler::SchedulerConfig;
use morphon_core::system::{System, SystemConfig};
use morphon_core::types::LifecycleConfig;
use rand::Rng;
use rand::RngCore;

const N_INPUTS: usize = 8;
const N_CLASSES: usize = 3;

/// Parameter vector → SystemConfig + training hyperparams.
/// Each element is in [0, 1] and gets mapped to the actual range.
struct ParamMapping {
    tau_eligibility: f64,
    tau_trace: f64,
    a_plus: f64,
    a_minus: f64,
    alpha_reward: f64,
    alpha_arousal: f64,
    tag_threshold: f64,
    capture_rate: f64,
    weight_max: f64,
    teach_strength: f64,
    reward_strength: f64,
    inhibit_strength: f64,
    input_bias: f64,
    input_scale: f64,
    alpha_novelty: f64,
}

fn decode_params(x: &DVector<f64>) -> ParamMapping {
    // Map each [unconstrained] → [lo, hi] via sigmoid
    let sig = |v: f64, lo: f64, hi: f64| -> f64 {
        let s = 1.0 / (1.0 + (-v).exp());
        lo + s * (hi - lo)
    };
    ParamMapping {
        tau_eligibility: sig(x[0], 1.0, 50.0),
        tau_trace: sig(x[1], 2.0, 30.0),
        a_plus: sig(x[2], 0.1, 5.0),
        a_minus: sig(x[3], -5.0, -0.01),
        alpha_reward: sig(x[4], 0.5, 10.0),
        alpha_arousal: sig(x[5], 0.0, 3.0),
        tag_threshold: sig(x[6], 0.05, 0.8),
        capture_rate: sig(x[7], 0.01, 1.0),
        weight_max: sig(x[8], 1.0, 10.0),
        teach_strength: sig(x[9], 0.0, 2.0),
        reward_strength: sig(x[10], 0.1, 2.0),
        inhibit_strength: sig(x[11], 0.0, 1.0),
        input_bias: sig(x[12], 0.0, 2.0),
        input_scale: sig(x[13], 0.5, 5.0),
        alpha_novelty: sig(x[14], 0.0, 3.0),
    }
}

fn make_sample(class: usize, bias: f64, scale: f64, rng: &mut impl Rng) -> Vec<f64> {
    let mut input = vec![bias; N_INPUTS];
    let noise = |rng: &mut dyn rand::RngCore| -> f64 {
        // Simple uniform noise without using the Rng trait method directly
        (rng.next_u32() as f64 / u32::MAX as f64 - 0.5) * 0.2
    };
    match class {
        0 => {
            input[0] = bias + scale + noise(rng);
            input[1] = bias + scale * 0.9 + noise(rng);
            input[2] = bias + scale * 0.7 + noise(rng);
        }
        1 => {
            input[3] = bias + scale + noise(rng);
            input[4] = bias + scale * 0.9 + noise(rng);
            input[5] = bias + scale * 0.7 + noise(rng);
        }
        2 => {
            input[6] = bias + scale * 1.2 + noise(rng);
            input[7] = bias + scale + noise(rng);
        }
        _ => {}
    }
    input
}

/// Evaluate a parameter vector. Returns negative accuracy (CMA-ES minimizes).
fn evaluate(x: &DVector<f64>) -> f64 {
    let p = decode_params(x);

    let config = SystemConfig {
        developmental: DevelopmentalConfig {
            seed_size: 30,
            dimensions: 4,
            initial_connectivity: 0.15,
            proliferation_rounds: 1,
            target_input_size: Some(N_INPUTS),
            target_output_size: Some(N_CLASSES),
            ..DevelopmentalConfig::cortical()
        },
        scheduler: SchedulerConfig {
            medium_period: 1,
            slow_period: 10,
            glacial_period: 100,
            homeostasis_period: 5,
            memory_period: 50,
        },
        learning: LearningParams {
            tau_eligibility: p.tau_eligibility,
            tau_trace: p.tau_trace,
            a_plus: p.a_plus,
            a_minus: p.a_minus,
            tau_tag: 200.0,
            tag_threshold: p.tag_threshold,
            capture_threshold: 0.2,
            capture_rate: p.capture_rate,
            weight_max: p.weight_max,
            weight_min: 0.01,
            alpha_reward: p.alpha_reward,
            alpha_novelty: p.alpha_novelty,
            alpha_arousal: p.alpha_arousal,
            alpha_homeostasis: 0.1,
            transmitter_potentiation: 0.001,
            heterosynaptic_depression: 0.002,
        },
        morphogenesis: MorphogenesisParams {
            max_morphons: 100,
            ..Default::default()
        },
        homeostasis: Default::default(),
        lifecycle: LifecycleConfig {
            division: false,
            fusion: false,
            apoptosis: false,
            differentiation: false,
            migration: true,
        },
        metabolic: MetabolicConfig::default(),
        dt: 1.0,
        working_memory_capacity: 7,
        episodic_memory_capacity: 100,
    };

    let mut system = System::new(config);
    let mut rng = rand::rng();

    // Train: 300 samples
    for _ in 0..300 {
        let label = (rng.next_u32() % N_CLASSES as u32) as usize;
        let input = make_sample(label, p.input_bias, p.input_scale, &mut rng);
        let _outputs = system.process_steps(&input, 3);

        system.teach_hidden(label, p.teach_strength);
        system.reward_contrastive(label, p.reward_strength, p.inhibit_strength);
        system.step();
    }

    // Test: 150 samples
    let mut correct = 0;
    let mut total = 0;
    for _ in 0..150 {
        let label = (rng.next_u32() % N_CLASSES as u32) as usize;
        let input = make_sample(label, p.input_bias, p.input_scale, &mut rng);
        let outputs = system.process_steps(&input, 3);

        if outputs.len() >= N_CLASSES {
            let pred = outputs.iter()
                .take(N_CLASSES)
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            if pred == label { correct += 1; }
        }
        total += 1;
    }

    let accuracy = correct as f64 / total as f64;
    // CMA-ES minimizes, so return negative accuracy
    -accuracy
}

fn main() {
    println!("=== CMA-ES Meta-Learning for MORPHON ===");
    println!("Optimizing 15 parameters on 3-class classification");
    println!("Population evaluates classify_tiny (300 train, 150 test)\n");

    let dim = 15;
    let initial_mean = vec![0.0; dim]; // sigmoid(0) = 0.5 = midpoint of each range
    let initial_sigma = 1.5; // explore broadly

    let mut cmaes_state = CMAESOptions::new(initial_mean, initial_sigma)
        .fun_target(-0.60) // stop if accuracy reaches 60% (well above 33% random)
        .max_generations(200)
        .population_size(20) // 20 candidates per generation
        .enable_printing(10) // print every 10 generations
        .build(evaluate)
        .unwrap();

    println!("Starting optimization (this may take a few minutes)...\n");

    let results = cmaes_state.run();

    println!("\n=== Results ===");
    println!("Termination: {:?}", results.reasons);

    if let Some(best) = results.overall_best {
        let accuracy = -best.value;
        println!("Best accuracy: {:.1}%", accuracy * 100.0);

        let p = decode_params(&best.point);
        println!("\nOptimal parameters:");
        println!("  tau_eligibility: {:.2}", p.tau_eligibility);
        println!("  tau_trace:       {:.2}", p.tau_trace);
        println!("  a_plus:          {:.3}", p.a_plus);
        println!("  a_minus:         {:.3}", p.a_minus);
        println!("  alpha_reward:    {:.2}", p.alpha_reward);
        println!("  alpha_novelty:   {:.2}", p.alpha_novelty);
        println!("  alpha_arousal:   {:.2}", p.alpha_arousal);
        println!("  tag_threshold:   {:.3}", p.tag_threshold);
        println!("  capture_rate:    {:.3}", p.capture_rate);
        println!("  weight_max:      {:.2}", p.weight_max);
        println!("  teach_strength:  {:.3}", p.teach_strength);
        println!("  reward_strength: {:.3}", p.reward_strength);
        println!("  inhibit_strength:{:.3}", p.inhibit_strength);
        println!("  input_bias:      {:.3}", p.input_bias);
        println!("  input_scale:     {:.3}", p.input_scale);

        // Save to JSON
        let json = serde_json::json!({
            "accuracy": accuracy,
            "params": {
                "tau_eligibility": p.tau_eligibility,
                "tau_trace": p.tau_trace,
                "a_plus": p.a_plus,
                "a_minus": p.a_minus,
                "alpha_reward": p.alpha_reward,
                "alpha_novelty": p.alpha_novelty,
                "alpha_arousal": p.alpha_arousal,
                "tag_threshold": p.tag_threshold,
                "capture_rate": p.capture_rate,
                "weight_max": p.weight_max,
                "teach_strength": p.teach_strength,
                "reward_strength": p.reward_strength,
                "inhibit_strength": p.inhibit_strength,
                "input_bias": p.input_bias,
                "input_scale": p.input_scale,
            }
        });
        std::fs::create_dir_all("docs/benchmark_results").ok();
        let path = "docs/benchmark_results/cma_best_params.json";
        std::fs::write(path, serde_json::to_string_pretty(&json).unwrap()).unwrap();
        println!("\nSaved to {}", path);
    }
}
