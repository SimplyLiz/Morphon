//! CMA-ES optimization for CartPole with Endoquilibrium.
//!
//! Searches 10 parameters: 5 Endo regulation gains + 5 learning/readout params.
//! Uses the same CartPole setup as cartpole.rs (population-coded 32 inputs, analog readout).
//! Fitness = average steps over 200 episodes (higher is better).
//!
//! Run: cargo run --example cma_endo --release

use cmaes::{CMAESOptions, DVector};
use morphon_core::developmental::DevelopmentalConfig;
use morphon_core::endoquilibrium::EndoConfig;
use morphon_core::homeostasis::HomeostasisParams;
use morphon_core::learning::LearningParams;
use morphon_core::morphogenesis::MorphogenesisParams;
use morphon_core::morphon::MetabolicConfig;
use morphon_core::scheduler::SchedulerConfig;
use morphon_core::system::{System, SystemConfig};
use morphon_core::types::LifecycleConfig;
use rand::Rng;

const GRAVITY: f64 = 9.8;
const CART_MASS: f64 = 1.0;
const POLE_MASS: f64 = 0.1;
const TOTAL_MASS: f64 = CART_MASS + POLE_MASS;
const POLE_HALF_LENGTH: f64 = 0.5;
const FORCE_MAG: f64 = 10.0;
const CP_DT: f64 = 0.02;
const X_THRESHOLD: f64 = 2.4;
const THETA_THRESHOLD: f64 = 12.0 * std::f64::consts::PI / 180.0;
const GAMMA: f64 = 0.9;
const INTERNAL_STEPS: usize = 4;

struct CartPole { x: f64, x_dot: f64, theta: f64, theta_dot: f64 }
impl CartPole {
    fn reset(&mut self, rng: &mut impl Rng) {
        self.x = rng.random_range(-0.05..0.05);
        self.x_dot = rng.random_range(-0.05..0.05);
        self.theta = rng.random_range(-0.05..0.05);
        self.theta_dot = rng.random_range(-0.05..0.05);
    }
    fn observe(&self) -> Vec<f64> {
        let raw = [self.x/X_THRESHOLD, self.x_dot.clamp(-3.0,3.0)/3.0,
                    self.theta/THETA_THRESHOLD, self.theta_dot.clamp(-3.0,3.0)/3.0];
        let centers: [f64; 8] = [-0.85, -0.60, -0.35, -0.10, 0.10, 0.35, 0.60, 0.85];
        let width = 0.3;
        let amp = 4.0;
        let mut out = Vec::with_capacity(32);
        for &val in &raw {
            for &center in &centers {
                let activation = (-(val - center).powi(2) / (2.0 * width * width)).exp() * amp;
                out.push(activation);
            }
        }
        out
    }
    fn step(&mut self, action: f64) -> bool {
        let f = action * FORCE_MAG;
        let ct = self.theta.cos(); let st = self.theta.sin();
        let tmp = (f + POLE_MASS*POLE_HALF_LENGTH*self.theta_dot.powi(2)*st)/TOTAL_MASS;
        let ta = (GRAVITY*st - ct*tmp)/(POLE_HALF_LENGTH*(4.0/3.0 - POLE_MASS*ct.powi(2)/TOTAL_MASS));
        let xa = tmp - POLE_MASS*POLE_HALF_LENGTH*ta*ct/TOTAL_MASS;
        self.x += CP_DT*self.x_dot; self.x_dot += CP_DT*xa;
        self.theta += CP_DT*self.theta_dot; self.theta_dot += CP_DT*ta;
        self.x.abs() < X_THRESHOLD && self.theta.abs() < THETA_THRESHOLD
    }
}

struct Critic { weights: [f64; 8], bias: f64, lr: f64 }
impl Critic {
    fn new() -> Self { Critic { weights: [0.0; 8], bias: 0.0, lr: 0.1 } }
    fn features(e: &CartPole) -> [f64; 8] {
        let s = [e.x/X_THRESHOLD, e.x_dot/3.0, e.theta/THETA_THRESHOLD, e.theta_dot/3.0];
        [s[0], s[1], s[2], s[3], s[0]*s[0], s[1]*s[1], s[2]*s[2], s[3]*s[3]]
    }
    fn predict(&self, e: &CartPole) -> f64 {
        Self::features(e).iter().zip(&self.weights).map(|(f,w)| f*w).sum::<f64>() + self.bias
    }
    fn update(&mut self, e: &CartPole, r: f64, ne: &CartPole, done: bool) -> f64 {
        let v = self.predict(e);
        let vn = if done { 0.0 } else { self.predict(ne) };
        let td = r + GAMMA * vn - v;
        let f = Self::features(e);
        for (w, fi) in self.weights.iter_mut().zip(&f) { *w = (*w + self.lr * td * fi).clamp(-10.0, 10.0); }
        self.bias = (self.bias + self.lr * td).clamp(-10.0, 10.0);
        td
    }
}

fn sig(v: f64, lo: f64, hi: f64) -> f64 {
    let s = 1.0 / (1.0 + (-v).exp());
    lo + s * (hi - lo)
}

/// 10-dimensional parameter vector:
///  [0] fr_deficit_threshold_k   [0.1, 1.5]   — Rule 1: how aggressively to lower thresholds
///  [1] fr_deficit_arousal_k     [0.05, 1.0]   — Rule 1: arousal response to low FR
///  [2] fr_deficit_novelty_k     [0.05, 0.8]   — Rule 1: novelty response to low FR
///  [3] smoothing_alpha          [0.02, 0.3]   — Channel smoothing (lower = slower response)
///  [4] fast_tau                 [20, 200]      — Fast EMA time constant
///  [5] readout_lr               [0.02, 0.4]   — Analog readout learning rate
///  [6] alpha_reward             [0.5, 5.0]    — Reward channel weight in three-factor rule
///  [7] tau_eligibility          [1.0, 15.0]   — Eligibility trace time constant
///  [8] a_minus                  [-2.0, -0.1]  — LTD magnitude
///  [9] capture_threshold        [0.3, 0.95]   — Tag-capture reward threshold
fn evaluate(x: &DVector<f64>) -> f64 {
    // Endo params
    let fr_deficit_threshold_k = sig(x[0], 0.1, 1.5) as f32;
    let fr_deficit_arousal_k = sig(x[1], 0.05, 1.0) as f32;
    let fr_deficit_novelty_k = sig(x[2], 0.05, 0.8) as f32;
    let smoothing_alpha = sig(x[3], 0.02, 0.3) as f32;
    let fast_tau = sig(x[4], 20.0, 200.0) as f32;
    // Learning params
    let readout_lr = sig(x[5], 0.02, 0.4);
    let alpha_reward = sig(x[6], 0.5, 5.0);
    let tau_elig = sig(x[7], 1.0, 15.0);
    let a_minus = sig(x[8], -2.0, -0.1);
    let capture_threshold = sig(x[9], 0.3, 0.95);

    let config = SystemConfig {
        developmental: DevelopmentalConfig {
            seed_size: 60, dimensions: 4, initial_connectivity: 0.25,
            proliferation_rounds: 2,
            target_input_size: Some(32), target_output_size: Some(2),
            ..DevelopmentalConfig::cerebellar()
        },
        scheduler: SchedulerConfig {
            medium_period: 1, slow_period: 10, glacial_period: 100,
            homeostasis_period: 10, memory_period: 25,
        },
        learning: LearningParams {
            tau_eligibility: tau_elig, tau_trace: 5.0,
            a_plus: 1.0, a_minus,
            tau_tag: 500.0, tag_threshold: 0.3,
            capture_threshold, capture_rate: 0.2,
            weight_max: 3.0, weight_min: 0.01,
            alpha_reward, alpha_novelty: 0.5, alpha_arousal: 0.5, alpha_homeostasis: 0.1,
            transmitter_potentiation: 0.002, heterosynaptic_depression: 0.003, tag_accumulation_rate: 0.3,
            ..Default::default()
        },
        morphogenesis: MorphogenesisParams {
            max_morphons: Some(300), division_threshold: 1.0, fusion_min_size: 2,
            apoptosis_min_age: 500, migration_rate: 0.08, ..Default::default()
        },
        homeostasis: HomeostasisParams { migration_cooldown_duration: 5.0, ..Default::default() },
        lifecycle: LifecycleConfig {
            division: false, differentiation: true, fusion: false,
            apoptosis: false, migration: false,
            synaptogenesis: true,
        },
        metabolic: MetabolicConfig::default(),
        endoquilibrium: EndoConfig {
            enabled: true,
            fast_tau,
            smoothing_alpha,
            fr_deficit_threshold_k,
            fr_deficit_arousal_k,
            fr_deficit_novelty_k,
            ..Default::default()
        },
        dt: 1.0, working_memory_capacity: 7, episodic_memory_capacity: 200,
        ..Default::default()
    };

    let mut system = System::new(config);
    system.enable_analog_readout();
    system.set_consolidation_gate(20.0);

    let mut env = CartPole { x: 0.0, x_dot: 0.0, theta: 0.01, theta_dot: 0.0 };
    let mut critic = Critic::new();
    let mut rng = rand::rng();
    let mut total_steps = 0usize;
    let n_episodes = 200;

    for _ in 0..20 { system.process_steps(&[2.0; 32], INTERNAL_STEPS); }

    for ep in 0..n_episodes {
        env.reset(&mut rng);
        let epsilon = (0.5 * (1.0 - ep as f64 / n_episodes as f64)).max(0.05);
        let mut steps = 0;

        for _ in 0..300 {
            let obs = env.observe();
            let pre = CartPole { x: env.x, x_dot: env.x_dot, theta: env.theta, theta_dot: env.theta_dot };
            let outputs = system.process_steps(&obs, INTERNAL_STEPS);
            let action = if rng.random_range(0.0..1.0) < epsilon {
                if rng.random_bool(0.5) { 1.0 } else { -1.0 }
            } else if outputs.len() >= 2 {
                if outputs[1] > outputs[0] { 1.0 } else { -1.0 }
            } else { 1.0 };

            let alive = env.step(action);
            steps += 1;
            let reward = if alive {
                1.0 + 0.5 * (1.0 - (env.theta / THETA_THRESHOLD).abs())
            } else { -1.0 };

            let _int_td = system.inject_td_error(reward, GAMMA);
            let td_error = critic.update(&pre, reward, &env, !alive);
            let chosen = if action > 0.0 { 1 } else { 0 };

            let base_lr = readout_lr;
            if td_error > 0.0 {
                system.train_readout(chosen, td_error.min(1.0) * base_lr);
                system.reward_contrastive(chosen, td_error.min(1.0) * 0.3, 0.1);
            } else {
                system.train_readout(1 - chosen, td_error.abs().min(1.0) * base_lr * 0.5);
            }

            let danger = (env.theta.abs() / THETA_THRESHOLD).min(1.0);
            if danger > 0.5 {
                system.inject_novelty((danger - 0.5) * 2.0);
            }

            if !alive {
                system.inject_arousal(0.8);
                system.inject_novelty(0.4);
                break;
            }
        }
        total_steps += steps;
        system.report_performance(steps as f64);
    }

    let avg = total_steps as f64 / n_episodes as f64;
    -avg // CMA-ES minimizes
}

fn main() {
    println!("=== CMA-ES Endoquilibrium + CartPole Optimization ===");
    println!("10 params: 5 Endo (fr_threshold, fr_arousal, fr_novelty, smoothing, fast_tau)");
    println!("         + 5 Learning (readout_lr, alpha_reward, tau_elig, a_minus, capture_threshold)");
    println!("Fitness: avg steps over 200 episodes\n");

    let dim = 10;
    let mut state = CMAESOptions::new(vec![0.0; dim], 1.0)
        .fun_target(-50.0) // stop if avg reaches 50
        .max_generations(200)
        .population_size(20) // larger pop for 10D
        .enable_printing(10)
        .build(evaluate)
        .unwrap();

    let results = state.run();

    println!("\n=== Results ===");
    println!("Termination: {:?}", results.reasons);

    if let Some(best) = results.overall_best {
        let avg = -best.value;
        println!("Best avg: {:.1} steps", avg);
        println!("\nOptimal Endo params:");
        println!("  fr_deficit_threshold_k: {:.4}", sig(best.point[0], 0.1, 1.5));
        println!("  fr_deficit_arousal_k:   {:.4}", sig(best.point[1], 0.05, 1.0));
        println!("  fr_deficit_novelty_k:   {:.4}", sig(best.point[2], 0.05, 0.8));
        println!("  smoothing_alpha:        {:.4}", sig(best.point[3], 0.02, 0.3));
        println!("  fast_tau:               {:.1}", sig(best.point[4], 20.0, 200.0));
        println!("\nOptimal Learning params:");
        println!("  readout_lr:             {:.4}", sig(best.point[5], 0.02, 0.4));
        println!("  alpha_reward:           {:.4}", sig(best.point[6], 0.5, 5.0));
        println!("  tau_eligibility:        {:.2}", sig(best.point[7], 1.0, 15.0));
        println!("  a_minus:                {:.4}", sig(best.point[8], -2.0, -0.1));
        println!("  capture_threshold:      {:.4}", sig(best.point[9], 0.3, 0.95));

        let json = serde_json::json!({
            "avg_steps": avg,
            "endo_params": {
                "fr_deficit_threshold_k": sig(best.point[0], 0.1, 1.5),
                "fr_deficit_arousal_k": sig(best.point[1], 0.05, 1.0),
                "fr_deficit_novelty_k": sig(best.point[2], 0.05, 0.8),
                "smoothing_alpha": sig(best.point[3], 0.02, 0.3),
                "fast_tau": sig(best.point[4], 20.0, 200.0),
            },
            "learning_params": {
                "readout_lr": sig(best.point[5], 0.02, 0.4),
                "alpha_reward": sig(best.point[6], 0.5, 5.0),
                "tau_eligibility": sig(best.point[7], 1.0, 15.0),
                "a_minus": sig(best.point[8], -2.0, -0.1),
                "capture_threshold": sig(best.point[9], 0.3, 0.95),
            }
        });
        let dir = format!("docs/benchmark_results/v{}", env!("CARGO_PKG_VERSION"));
        std::fs::create_dir_all(&dir).ok();
        let path = format!("{}/cma_endo_best.json", dir);
        std::fs::write(&path, serde_json::to_string_pretty(&json).unwrap()).unwrap();
        println!("\nSaved to {}", path);
    }
}
