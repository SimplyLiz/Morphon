//! CartPole Benchmark — MI system learns to balance a pole through self-organization.
//!
//! Run: cargo run --example cartpole --release

use morphon_core::developmental::DevelopmentalConfig;
use morphon_core::homeostasis::HomeostasisParams;
use morphon_core::learning::LearningParams;
use morphon_core::morphogenesis::MorphogenesisParams;
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

    /// 4 observations with bias to maintain network activity.
    fn observe(&self) -> [f64; 4] {
        let amp = 3.0;
        let bias = 1.0;
        [
            bias + self.x / X_THRESHOLD * amp,
            bias + self.x_dot.clamp(-3.0, 3.0) / 3.0 * amp,
            bias + self.theta / THETA_THRESHOLD * amp,
            bias + self.theta_dot.clamp(-3.0, 3.0) / 3.0 * amp,
        ]
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

fn create_system() -> System {
    let config = SystemConfig {
        developmental: DevelopmentalConfig {
            seed_size: 30,
            dimensions: 4,
            initial_connectivity: 0.25,
            proliferation_rounds: 2,
            // Exactly 4 inputs and 2 outputs for CartPole
            target_input_size: Some(4),
            target_output_size: Some(2),
            ..DevelopmentalConfig::cerebellar()
        },
        scheduler: SchedulerConfig {
            medium_period: 1,
            slow_period: 5,
            glacial_period: 50,
            homeostasis_period: 10,
            memory_period: 25,
        },
        learning: LearningParams {
            tau_eligibility: 8.0,
            tau_tag: 500.0,
            tag_threshold: 0.5,
            capture_threshold: 0.3,
            capture_rate: 0.2,
            weight_max: 3.0,
            weight_min: 0.01,
            alpha_reward: 2.0,
            alpha_novelty: 0.5,
            alpha_arousal: 1.5,
            alpha_homeostasis: 0.1,
        },
        morphogenesis: MorphogenesisParams {
            migration_rate: 0.08,
            max_morphons: 300,
            division_threshold: 0.8,
            fusion_min_size: 2,
            apoptosis_min_age: 500,
            ..Default::default()
        },
        homeostasis: HomeostasisParams {
            migration_cooldown_duration: 5.0,
            ..Default::default()
        },
        lifecycle: LifecycleConfig::default(),
        dt: 1.0,
        working_memory_capacity: 7,
        episodic_memory_capacity: 200,
    };
    System::new(config)
}

/// Binary action from 2 motor outputs.
fn select_action(outputs: &[f64], epsilon: f64, rng: &mut impl Rng) -> f64 {
    if rng.random_range(0.0..1.0) < epsilon {
        return if rng.random_bool(0.5) { 1.0 } else { -1.0 };
    }
    if outputs.len() < 2 {
        return if rng.random_bool(0.5) { 1.0 } else { -1.0 };
    }
    // output[0] = left, output[1] = right
    if outputs[1] > outputs[0] { 1.0 } else { -1.0 }
}

fn run_episode(system: &mut System, env: &mut CartPole, max_steps: usize, epsilon: f64, rng: &mut impl Rng) -> usize {
    env.reset(rng);
    let mut steps = 0;

    for _ in 0..max_steps {
        let obs = env.observe();
        // 3 internal steps per action to let signals propagate through the network
        let outputs = system.process_steps(&obs, 3);
        let action = select_action(&outputs, epsilon, rng);

        let alive = env.step(action);
        steps += 1;

        if alive {
            let angle_q = 1.0 - (env.theta / THETA_THRESHOLD).abs();
            let pos_q = 1.0 - (env.x / X_THRESHOLD).abs();
            let reward = 0.2 + 0.3 * angle_q + 0.1 * pos_q;
            // Contrastive: reward the action that was taken, mildly inhibit the other
            // output[0] = left, output[1] = right; action 1.0 = right (index 1), -1.0 = left (index 0)
            let chosen = if action > 0.0 { 1 } else { 0 };
            system.reward_contrastive(chosen, reward, reward * 0.3);
        } else {
            // Failure: arousal + penalize the action that caused the fall
            system.inject_arousal(0.9);
            let wrong = if action > 0.0 { 1 } else { 0 };
            system.inject_inhibition_at(wrong, 0.5);
            system.inject_novelty(0.3);
            break;
        }
    }

    let survival = steps as f64 / max_steps as f64;
    system.reward_contrastive(
        if steps > 50 { 0 } else { 1 }, // arbitrary, just injects global reward
        survival * 0.5,
        0.0, // no inhibition on episode-end bonus
    );
    steps
}

fn main() {
    println!("=== MORPHON CartPole Benchmark ===\n");

    let mut system = create_system();
    let mut env = CartPole { x: 0.0, x_dot: 0.0, theta: 0.01, theta_dot: 0.0 };
    let mut rng = rand::rng();

    let stats = system.inspect();
    println!("Initial: {} morphons, {} synapses, {} in, {} out",
        stats.total_morphons, stats.total_synapses, system.input_size(), system.output_size());
    println!("Types: {:?}\n", stats.differentiation_map);

    // Warm up
    for _ in 0..20 { system.process_steps(&[1.0, 1.0, 1.0, 1.0], 3); }

    let num_episodes = 1000;
    let max_steps = 500;
    let mut best = 0usize;
    let mut recent: Vec<usize> = Vec::new();

    for ep in 0..num_episodes {
        let epsilon = 0.5 * (1.0 - ep as f64 / num_episodes as f64).max(0.1);
        let steps = run_episode(&mut system, &mut env, max_steps, epsilon, &mut rng);
        recent.push(steps);
        if recent.len() > 100 { recent.remove(0); }
        best = best.max(steps);
        let avg = recent.iter().sum::<usize>() as f64 / recent.len() as f64;

        if (ep + 1) % 100 == 0 || steps >= 200 {
            let s = system.inspect();
            println!("Ep {:>4} | steps {:>3} | avg(100) {:>6.1} | best {:>3} | m {} s {} fr {:.3} pe {:.3}",
                ep + 1, steps, avg, best, s.total_morphons, s.total_synapses, s.firing_rate, s.avg_prediction_error);
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

    // Save benchmark results
    let avg_100 = recent.iter().sum::<usize>() as f64 / recent.len().max(1) as f64;
    let solved = recent.len() >= 100 && avg_100 >= 195.0;
    let version = env!("CARGO_PKG_VERSION");
    let results = json!({
        "benchmark": "cartpole",
        "version": version,
        "timestamp": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
        "episodes": num_episodes,
        "max_steps_per_episode": max_steps,
        "results": {
            "best_steps": best,
            "avg_last_100": avg_100,
            "solved": solved,
        },
        "system": {
            "morphons": s.total_morphons,
            "synapses": s.total_synapses,
            "clusters": s.fused_clusters,
            "generation": s.max_generation,
            "firing_rate": s.firing_rate,
            "prediction_error": s.avg_prediction_error,
        },
        "diagnostics": {
            "weight_mean": diag.weight_mean,
            "weight_std": diag.weight_std,
            "active_tags": diag.active_tags,
            "total_captures": diag.total_captures,
        },
    });

    let dir = format!("docs/benchmark_results/v{}", version);
    fs::create_dir_all(&dir).ok();
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    let run_path = format!("{}/cartpole_{}.json", dir, ts);
    let latest_path = format!("{}/cartpole_latest.json", dir);
    let json = serde_json::to_string_pretty(&results).unwrap();
    fs::write(&run_path, &json).unwrap();
    fs::write(&latest_path, &json).unwrap();
    println!("\nResults saved to {}", run_path);
}
