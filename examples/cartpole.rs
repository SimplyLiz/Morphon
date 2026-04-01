//! CartPole Benchmark — MI system learns to balance a pole through self-organization.
//!
//! Uses a linear TD-error critic (Frémaux et al. 2013) to provide a state-dependent
//! learning signal. The critic is external to the MI system — biologically, this maps
//! to dopamine neurons in VTA/SNc computing TD error from striatal input. The TD error
//! δ = R + γV(s') - V(s) modulates the three-factor learning rule in the morphon network.
//!
//! Run: cargo run --example cartpole --release
//! Run: cargo run --example cartpole --release -- --standard
//! Run: cargo run --example cartpole --release -- --extended

use morphon_core::developmental::DevelopmentalConfig;
use morphon_core::homeostasis::HomeostasisParams;
use morphon_core::learning::LearningParams;
use morphon_core::morphogenesis::MorphogenesisParams;
use morphon_core::morphon::MetabolicConfig;
use morphon_core::scheduler::SchedulerConfig;
use morphon_core::system::{System, SystemConfig};
use morphon_core::types::LifecycleConfig;
use rand::Rng;
use serde_json::json;
use std::fs;

const GRAVITY: f64 = 9.8;
const CART_MASS: f64 = 1.0;
const POLE_MASS: f64 = 0.1;
const TOTAL_MASS: f64 = CART_MASS + POLE_MASS;
const POLE_HALF_LENGTH: f64 = 0.5;
const FORCE_MAG: f64 = 10.0;
const DT: f64 = 0.02;
const X_THRESHOLD: f64 = 2.4;
const THETA_THRESHOLD: f64 = 12.0 * std::f64::consts::PI / 180.0;
const INTERNAL_STEPS: usize = 6;
const GAMMA: f64 = 0.99;

struct CartPole {
    x: f64, x_dot: f64, theta: f64, theta_dot: f64,
}

impl CartPole {
    fn reset(&mut self, rng: &mut impl Rng) {
        self.x = rng.random_range(-0.05..0.05);
        self.x_dot = rng.random_range(-0.05..0.05);
        self.theta = rng.random_range(-0.05..0.05);
        self.theta_dot = rng.random_range(-0.05..0.05);
    }

    /// Sparse encoding: split each observation into positive/negative channels.
    /// Zero bias — inactive channels stay at 0, preserving class discrimination.
    /// Population-coded observation: 4 values × 8 threshold tiles = 32 channels.
    /// Each tile fires (value > 0) only when the observation falls in its receptive field.
    /// This gives the hidden layer fundamentally different input patterns for different
    /// pole angles — solving the "uniformity problem" where all states look identical.
    fn observe(&self) -> Vec<f64> {
        let raw = [
            self.x / X_THRESHOLD,
            self.x_dot.clamp(-3.0, 3.0) / 3.0,
            self.theta / THETA_THRESHOLD,
            self.theta_dot.clamp(-3.0, 3.0) / 3.0,
        ];
        // 8 tiles per observation, spanning [-1, 1] with Gaussian-like receptive fields
        let centers: [f64; 8] = [-0.85, -0.60, -0.35, -0.10, 0.10, 0.35, 0.60, 0.85];
        let width = 0.3; // receptive field width (sigma)
        let amp = 4.0;
        let mut out = Vec::with_capacity(32);
        for &val in &raw {
            for &center in &centers {
                // Gaussian tuning curve: peaks when val ≈ center
                let activation = (-(val - center).powi(2) / (2.0 * width * width)).exp() * amp;
                out.push(activation);
            }
        }
        out
    }

    fn step(&mut self, action: f64) -> bool {
        let force = action * FORCE_MAG;
        let cos_t = self.theta.cos();
        let sin_t = self.theta.sin();
        let temp = (force + POLE_MASS * POLE_HALF_LENGTH * self.theta_dot.powi(2) * sin_t) / TOTAL_MASS;
        let theta_acc = (GRAVITY * sin_t - cos_t * temp)
            / (POLE_HALF_LENGTH * (4.0 / 3.0 - POLE_MASS * cos_t.powi(2) / TOTAL_MASS));
        let x_acc = temp - POLE_MASS * POLE_HALF_LENGTH * theta_acc * cos_t / TOTAL_MASS;
        self.x += DT * self.x_dot;
        self.x_dot += DT * x_acc;
        self.theta += DT * self.theta_dot;
        self.theta_dot += DT * theta_acc;
        self.x.abs() < X_THRESHOLD && self.theta.abs() < THETA_THRESHOLD
    }
}

/// Linear TD-error critic — external to the MI system.
/// Biologically: dopamine neurons in VTA/SNc computing δ from striatal input.
struct Critic {
    weights: [f64; 8],
    bias: f64,
    lr: f64,
}

impl Critic {
    fn new() -> Self {
        Self { weights: [0.0; 8], bias: 0.0, lr: 0.02 }
    }

    fn features(env: &CartPole) -> [f64; 8] {
        let s = [env.x / X_THRESHOLD, env.x_dot / 3.0, env.theta / THETA_THRESHOLD, env.theta_dot / 3.0];
        [s[0], s[1], s[2], s[3], s[0]*s[0], s[1]*s[1], s[2]*s[2], s[3]*s[3]]
    }

    fn predict(&self, env: &CartPole) -> f64 {
        let f = Self::features(env);
        f.iter().zip(self.weights.iter()).map(|(fi, wi)| fi * wi).sum::<f64>() + self.bias
    }

    fn update(&mut self, env: &CartPole, reward: f64, next_env: &CartPole, done: bool) -> f64 {
        let v = self.predict(env);
        let v_next = if done { 0.0 } else { self.predict(next_env) };
        let td_error = reward + GAMMA * v_next - v;
        let f = Self::features(env);
        for i in 0..8 {
            self.weights[i] = (self.weights[i] + self.lr * td_error * f[i]).clamp(-10.0, 10.0);
        }
        self.bias = (self.bias + self.lr * td_error).clamp(-10.0, 10.0);
        td_error
    }
}

fn create_system() -> System {
    let config = SystemConfig {
        developmental: DevelopmentalConfig {
            seed_size: 60,
            dimensions: 4,
            initial_connectivity: 0.25,
            proliferation_rounds: 2,
            target_input_size: Some(32), // 4 obs × 8 Gaussian tiles (population coding)
            target_output_size: Some(2),
            ..DevelopmentalConfig::cerebellar()
        },
        scheduler: SchedulerConfig {
            medium_period: 1,
            slow_period: 10,
            glacial_period: 100,
            homeostasis_period: 10,
            memory_period: 25,
        },
        learning: LearningParams {
            tau_eligibility: 3.0,   // short — CartPole is reactive
            tau_trace: 5.0,
            a_plus: 1.0,
            a_minus: -0.5,          // weaker LTD to prevent inhibitory drift
            tau_tag: 500.0,
            tag_threshold: 0.3,
            capture_threshold: 10.0, // disable consolidation — stay plastic
            capture_rate: 0.2,
            weight_max: 3.0,        // tighter bounds prevent weight divergence
            weight_min: 0.01,
            alpha_reward: 2.0,
            alpha_novelty: 0.5,
            alpha_arousal: 0.5,
            alpha_homeostasis: 0.1,
            transmitter_potentiation: 0.002, // anti-silence: floor on dw when pre fires but post quiet
            heterosynaptic_depression: 0.003, // anti-runaway: depress all inputs when post fires
        },
        morphogenesis: MorphogenesisParams {
            migration_rate: 0.08,
            max_morphons: 300,
            division_threshold: 1.0,
            fusion_min_size: 2,
            apoptosis_min_age: 500,
            ..Default::default()
        },
        homeostasis: HomeostasisParams {
            migration_cooldown_duration: 5.0,
            ..Default::default()
        },
        lifecycle: LifecycleConfig::default(),
        metabolic: MetabolicConfig::default(),
        dt: 1.0,
        working_memory_capacity: 7,
        episodic_memory_capacity: 200,
    };
    let mut sys = System::new(config);
    sys.enable_analog_readout(); // Purkinje-style analog output bypass
    sys
}

fn select_action(outputs: &[f64], epsilon: f64, rng: &mut impl Rng) -> f64 {
    if rng.random_range(0.0..1.0) < epsilon {
        return if rng.random_bool(0.5) { 1.0 } else { -1.0 };
    }
    if outputs.len() < 2 {
        return if rng.random_bool(0.5) { 1.0 } else { -1.0 };
    }
    if outputs[1] > outputs[0] { 1.0 } else { -1.0 }
}

fn run_episode(
    system: &mut System, env: &mut CartPole, critic: &mut Critic,
    max_steps: usize, epsilon: f64, rng: &mut impl Rng,
) -> usize {
    env.reset(rng);
    let mut steps = 0;

    for _ in 0..max_steps {
        let obs = env.observe();
        let pre_state = CartPole { x: env.x, x_dot: env.x_dot, theta: env.theta, theta_dot: env.theta_dot };
        let outputs = system.process_steps(&obs, INTERNAL_STEPS);
        let action = select_action(&outputs, epsilon, rng);
        let alive = env.step(action);
        steps += 1;

        let reward = if alive {
            1.0 + 0.5 * (1.0 - (env.theta / THETA_THRESHOLD).abs())
        } else { -1.0 }; // negative terminal signal for stronger TD error on failure

        // Internal TD critic: sets last_td_error (DFA) + injects reward
        let _int_td = system.inject_td_error(reward, GAMMA);

        // External critic: better value estimates for readout training
        let td_error = critic.update(&pre_state, reward, env, !alive);
        let chosen = if action > 0.0 { 1 } else { 0 };

        // Train analog readout — both positive and negative TD
        let base_lr = 0.15;
        if td_error > 0.0 {
            system.train_readout(chosen, td_error.min(1.0) * base_lr);
        } else {
            let other = 1 - chosen;
            system.train_readout(other, td_error.abs().min(1.0) * base_lr * 0.5);
        }

        // TD(λ)-like trace stretching: when pole is in danger (>50% of threshold),
        // inject novelty to boost plasticity. This extends the effective eligibility
        // window during critical moments — the system learns more from near-failures.
        let danger = (env.theta.abs() / THETA_THRESHOLD).min(1.0);
        if danger > 0.5 {
            system.inject_novelty((danger - 0.5) * 2.0); // 0→1 as danger goes 0.5→1.0
        }

        if !alive {
            system.inject_arousal(0.8);
            system.inject_novelty(0.4);
            break;
        }
    }
    steps
}

fn parse_profile() -> &'static str {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--extended") { "extended" }
    else if args.iter().any(|a| a == "--standard") { "standard" }
    else { "quick" }
}

fn main() {
    let profile = parse_profile();
    let (num_episodes, max_steps) = match profile {
        "extended" => (3000, 500),
        "standard" => (1000, 500),
        _          => (200, 300),
    };

    println!("=== MORPHON CartPole Benchmark [{}] ===\n", profile);

    let mut system = create_system();
    let mut env = CartPole { x: 0.0, x_dot: 0.0, theta: 0.01, theta_dot: 0.0 };
    let mut critic = Critic::new();
    let mut rng = rand::rng();

    let stats = system.inspect();
    println!("Initial: {} morphons, {} synapses, {} in, {} out, {} critic",
        stats.total_morphons, stats.total_synapses, system.input_size(), system.output_size(),
        system.critic_size());
    println!("Types: {:?}\n", stats.differentiation_map);

    for _ in 0..20 { system.process_steps(&[2.0, 2.0, 2.0, 2.0], INTERNAL_STEPS); }
    let mut best = 0usize;
    let mut recent: Vec<usize> = Vec::new();

    for ep in 0..num_episodes {
        let epsilon = (0.5 * (1.0 - ep as f64 / num_episodes as f64)).max(0.05);
        let steps = run_episode(&mut system, &mut env, &mut critic, max_steps, epsilon, &mut rng);
        recent.push(steps);
        if recent.len() > 100 { recent.remove(0); }
        best = best.max(steps);
        system.report_performance(steps as f64);
        let avg = recent.iter().sum::<usize>() as f64 / recent.len() as f64;

        if (ep + 1) % 100 == 0 || steps >= 200 {
            let s = system.inspect();
            let diag = system.diagnostics();
            // Probe: run a test observation to see output differentiation
            let test_out = system.read_output();
            let out_diff = if test_out.len() >= 2 { (test_out[0] - test_out[1]).abs() } else { 0.0 };
            println!("Ep {:>4} | steps {:>3} | avg(100) {:>6.1} | best {:>3} | m {} s {} fr {:.3} pe {:.3} Δout {:.3} | {}",
                ep + 1, steps, avg, best, s.total_morphons, s.total_synapses, s.firing_rate, s.avg_prediction_error,
                out_diff, diag.summary());
        }

        if recent.len() >= 100 && avg >= 195.0 {
            println!("\n*** SOLVED at ep {}! avg = {:.1} ***", ep + 1, avg);
            break;
        }
    }

    println!("\n=== Final ===");
    let s = system.inspect();
    let diag = system.diagnostics();
    println!("Morphons: {} | Synapses: {} | Clusters: {} | Gen: {} | FR: {:.3}",
        s.total_morphons, s.total_synapses, s.fused_clusters, s.max_generation, s.firing_rate);
    println!("Types: {:?}", s.differentiation_map);
    println!("Learning: {}", diag.summary());

    let avg_100 = recent.iter().sum::<usize>() as f64 / recent.len().max(1) as f64;
    let solved = recent.len() >= 100 && avg_100 >= 195.0;
    let version = env!("CARGO_PKG_VERSION");
    let results = json!({
        "benchmark": "cartpole", "profile": profile, "version": version,
        "timestamp": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
        "episodes": num_episodes, "max_steps_per_episode": max_steps,
        "results": { "best_steps": best, "avg_last_100": avg_100, "solved": solved },
        "system": {
            "morphons": s.total_morphons, "synapses": s.total_synapses,
            "clusters": s.fused_clusters, "generation": s.max_generation,
            "firing_rate": s.firing_rate, "prediction_error": s.avg_prediction_error,
        },
        "diagnostics": {
            "weight_mean": diag.weight_mean, "weight_std": diag.weight_std,
            "active_tags": diag.active_tags, "total_captures": diag.total_captures,
        },
    });

    let dir = format!("docs/benchmark_results/v{}", version);
    fs::create_dir_all(&dir).ok();
    let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    let run_path = format!("{}/cartpole_{}.json", dir, ts);
    let latest_path = format!("{}/cartpole_latest.json", dir);
    let json_str = serde_json::to_string_pretty(&results).unwrap();
    fs::write(&run_path, &json_str).unwrap();
    fs::write(&latest_path, &json_str).unwrap();
    println!("\nResults saved to {}", run_path);
}
