//! CMA-ES optimization for CartPole with analog readout.
//!
//! Searches 5 parameters: readout_lr, dfa_lr, eligibility_tau, assoc_threshold_scale, td_gamma.
//! Fitness = average steps over 100 episodes (higher is better).
//!
//! Run: cargo run --example cma_cartpole --release

use cmaes::{CMAESOptions, DVector};
use morphon_core::developmental::DevelopmentalConfig;
use morphon_core::homeostasis::HomeostasisParams;
use morphon_core::learning::LearningParams;
use morphon_core::morphogenesis::MorphogenesisParams;
use morphon_core::morphon::MetabolicConfig;
use morphon_core::scheduler::SchedulerConfig;
use morphon_core::system::{System, SystemConfig};
use morphon_core::types::LifecycleConfig;
use rand::Rng;

// CartPole physics (same as cartpole.rs)
const GRAVITY: f64 = 9.8;
const CART_MASS: f64 = 1.0;
const POLE_MASS: f64 = 0.1;
const TOTAL_MASS: f64 = CART_MASS + POLE_MASS;
const POLE_HALF_LENGTH: f64 = 0.5;
const FORCE_MAG: f64 = 10.0;
const CP_DT: f64 = 0.02;
const X_THRESHOLD: f64 = 2.4;
const THETA_THRESHOLD: f64 = 12.0 * std::f64::consts::PI / 180.0;

struct CartPole { x: f64, x_dot: f64, theta: f64, theta_dot: f64 }
impl CartPole {
    fn reset(&mut self, rng: &mut impl Rng) {
        self.x = rng.random_range(-0.05..0.05);
        self.x_dot = rng.random_range(-0.05..0.05);
        self.theta = rng.random_range(-0.05..0.05);
        self.theta_dot = rng.random_range(-0.05..0.05);
    }
    fn observe(&self) -> [f64; 8] {
        let amp = 5.0;
        let r = [self.x/X_THRESHOLD, self.x_dot.clamp(-3.0,3.0)/3.0,
                  self.theta/THETA_THRESHOLD, self.theta_dot.clamp(-3.0,3.0)/3.0];
        [r[0].max(0.0)*amp, (-r[0]).max(0.0)*amp, r[1].max(0.0)*amp, (-r[1]).max(0.0)*amp,
         r[2].max(0.0)*amp, (-r[2]).max(0.0)*amp, r[3].max(0.0)*amp, (-r[3]).max(0.0)*amp]
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

// Simple TD critic (same as cartpole.rs)
struct Critic { weights: Vec<f64>, bias: f64, lr: f64, gamma: f64 }
impl Critic {
    fn new(gamma: f64) -> Self { Critic { weights: vec![0.0; 6], bias: 0.0, lr: 0.01, gamma } }
    fn features(e: &CartPole) -> Vec<f64> {
        vec![e.x/X_THRESHOLD, e.x_dot/3.0, e.theta/THETA_THRESHOLD,
             e.theta_dot/3.0, (e.theta/THETA_THRESHOLD).powi(2), (e.x/X_THRESHOLD).powi(2)]
    }
    fn predict(&self, e: &CartPole) -> f64 {
        Self::features(e).iter().zip(&self.weights).map(|(f,w)| f*w).sum::<f64>() + self.bias
    }
    fn update(&mut self, e: &CartPole, r: f64, ne: &CartPole, done: bool) -> f64 {
        let v = self.predict(e);
        let vn = if done { 0.0 } else { self.predict(ne) };
        let td = r + self.gamma * vn - v;
        let f = Self::features(e);
        for (w, fi) in self.weights.iter_mut().zip(&f) { *w += self.lr * td * fi; }
        self.bias += self.lr * td;
        td
    }
}

fn sig(v: f64, lo: f64, hi: f64) -> f64 {
    let s = 1.0 / (1.0 + (-v).exp());
    lo + s * (hi - lo)
}

/// Evaluate a parameter vector. Returns negative avg steps (CMA-ES minimizes).
fn evaluate(x: &DVector<f64>) -> f64 {
    let readout_lr = sig(x[0], 0.01, 0.5);
    let dfa_lr = sig(x[1], 0.001, 0.1);
    let tau_elig = sig(x[2], 5.0, 30.0);
    let threshold_scale = sig(x[3], 0.3, 0.8); // post-warmup threshold multiplier
    let td_gamma = sig(x[4], 0.9, 0.999);

    let config = SystemConfig {
        developmental: DevelopmentalConfig {
            seed_size: 60, dimensions: 4, initial_connectivity: 0.25,
            proliferation_rounds: 2,
            target_input_size: Some(8), target_output_size: Some(2),
            ..DevelopmentalConfig::cerebellar()
        },
        scheduler: SchedulerConfig {
            medium_period: 1, slow_period: 10, glacial_period: 100,
            homeostasis_period: 10, memory_period: 25,
        },
        learning: LearningParams {
            tau_eligibility: tau_elig, tau_trace: 12.0,
            a_plus: 1.0, a_minus: -0.8, tau_tag: 500.0,
            tag_threshold: 0.3, capture_threshold: 0.3, capture_rate: 0.2,
            weight_max: 5.0, weight_min: 0.01,
            alpha_reward: 3.0, alpha_novelty: 0.5, alpha_arousal: 1.0, alpha_homeostasis: 0.1,
        },
        morphogenesis: MorphogenesisParams {
            max_morphons: 300, division_threshold: 1.0, fusion_min_size: 2,
            apoptosis_min_age: 500, migration_rate: 0.08, ..Default::default()
        },
        homeostasis: HomeostasisParams { migration_cooldown_duration: 5.0, ..Default::default() },
        lifecycle: LifecycleConfig::default(),
        metabolic: MetabolicConfig::default(),
        dt: 1.0, working_memory_capacity: 7, episodic_memory_capacity: 200,
    };

    let mut system = System::new(config);
    system.enable_analog_readout();

    // Apply threshold scale to Associative morphons
    for m in system.morphons.values_mut() {
        if m.cell_type == morphon_core::CellType::Associative
            || m.cell_type == morphon_core::CellType::Stem {
            m.threshold *= threshold_scale;
        }
    }

    let mut env = CartPole { x: 0.0, x_dot: 0.0, theta: 0.01, theta_dot: 0.0 };
    let mut critic = Critic::new(td_gamma);
    let mut rng = rand::rng();
    let mut total_steps = 0usize;
    let n_episodes = 100;

    for ep in 0..n_episodes {
        env.reset(&mut rng);
        let epsilon = (0.5 * (1.0 - ep as f64 / n_episodes as f64)).max(0.05);
        let mut steps = 0;

        for _ in 0..200 {
            let obs = env.observe();
            let pre = CartPole { x: env.x, x_dot: env.x_dot, theta: env.theta, theta_dot: env.theta_dot };
            let outputs = system.process_steps(&obs, 5);
            let action = if rng.random_range(0.0..1.0) < epsilon {
                if rng.random_bool(0.5) { 1.0 } else { -1.0 }
            } else if outputs.len() >= 2 {
                if outputs[1] > outputs[0] { 1.0 } else { -1.0 }
            } else { 1.0 };

            let alive = env.step(action);
            steps += 1;
            let reward = if alive { 1.0 + 0.5 * (1.0 - (env.theta/THETA_THRESHOLD).abs()) } else { 0.0 };
            let td = critic.update(&pre, reward, &env, !alive);
            let chosen = if action > 0.0 { 1 } else { 0 };

            if td > 0.0 {
                system.train_readout(chosen, td.min(1.0) * readout_lr);
            } else {
                system.train_readout(1 - chosen, td.abs().min(1.0) * readout_lr * 0.5);
            }
            let scaled_td = (td * 0.3 + 0.5).clamp(0.0, 1.0);
            system.inject_reward(scaled_td);

            if !alive { break; }
        }
        total_steps += steps;
        system.report_performance(steps as f64);
    }

    let avg = total_steps as f64 / n_episodes as f64;
    -avg // CMA-ES minimizes
}

fn main() {
    println!("=== CMA-ES CartPole Optimization (Analog Readout) ===");
    println!("5 params: readout_lr, dfa_lr, tau_elig, threshold_scale, td_gamma");
    println!("Fitness: avg steps over 100 episodes\n");

    let dim = 5;
    let mut state = CMAESOptions::new(vec![0.0; dim], 1.5)
        .fun_target(-30.0) // stop if avg reaches 30 (3× random)
        .max_generations(150)
        .population_size(15)
        .enable_printing(10)
        .build(evaluate)
        .unwrap();

    let results = state.run();

    println!("\n=== Results ===");
    println!("Termination: {:?}", results.reasons);

    if let Some(best) = results.overall_best {
        let avg = -best.value;
        println!("Best avg: {:.1} steps", avg);
        println!("\nOptimal params:");
        println!("  readout_lr:       {:.4}", sig(best.point[0], 0.01, 0.5));
        println!("  dfa_lr:           {:.4}", sig(best.point[1], 0.001, 0.1));
        println!("  tau_eligibility:  {:.2}", sig(best.point[2], 5.0, 30.0));
        println!("  threshold_scale:  {:.3}", sig(best.point[3], 0.3, 0.8));
        println!("  td_gamma:         {:.4}", sig(best.point[4], 0.9, 0.999));

        let json = serde_json::json!({
            "avg_steps": avg,
            "params": {
                "readout_lr": sig(best.point[0], 0.01, 0.5),
                "dfa_lr": sig(best.point[1], 0.001, 0.1),
                "tau_eligibility": sig(best.point[2], 5.0, 30.0),
                "threshold_scale": sig(best.point[3], 0.3, 0.8),
                "td_gamma": sig(best.point[4], 0.9, 0.999),
            }
        });
        std::fs::create_dir_all("docs/benchmark_results").ok();
        std::fs::write("docs/benchmark_results/cma_cartpole_best.json",
            serde_json::to_string_pretty(&json).unwrap()).unwrap();
        println!("\nSaved to docs/benchmark_results/cma_cartpole_best.json");
    }
}
