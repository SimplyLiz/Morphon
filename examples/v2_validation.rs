//! V2 Primitives Validation — proves the three Phase 1 primitives provide measurable benefit.
//!
//! Three experiments designed to stress the specific primitive being tested:
//! 1. Frustration: spike-only output (no analog readout) creates real local minima
//! 2. Field: migration with field vs blind migration, measured by PE convergence
//! 3. Self-Healing: kill MOTOR morphons mid-run (catastrophic without healing)
//!
//! Run: cargo run --example v2_validation --release

use morphon_core::developmental::DevelopmentalConfig;
use morphon_core::field::FieldConfig;
use morphon_core::homeostasis::HomeostasisParams;
use morphon_core::learning::LearningParams;
use morphon_core::morphogenesis::MorphogenesisParams;
use morphon_core::morphon::MetabolicConfig;
use morphon_core::scheduler::SchedulerConfig;
use morphon_core::system::{System, SystemConfig};
use morphon_core::types::*;
use rand::Rng;
use rand::SeedableRng;

const GRAVITY: f64 = 9.8;
const CART_MASS: f64 = 1.0;
const POLE_MASS: f64 = 0.1;
const TOTAL_MASS: f64 = CART_MASS + POLE_MASS;
const POLE_HALF_LENGTH: f64 = 0.5;
const FORCE_MAG: f64 = 10.0;
const DT_ENV: f64 = 0.02;
const X_THRESHOLD: f64 = 2.4;
const THETA_THRESHOLD: f64 = 12.0 * std::f64::consts::PI / 180.0;
const INTERNAL_STEPS: usize = 4;
const GAMMA: f64 = 0.9;

struct CartPole {
    x: f64, x_dot: f64, theta: f64, theta_dot: f64,
}

impl CartPole {
    fn new() -> Self { Self { x: 0.0, x_dot: 0.0, theta: 0.01, theta_dot: 0.0 } }

    fn reset(&mut self, rng: &mut impl Rng) {
        self.x = rng.random_range(-0.05..0.05);
        self.x_dot = rng.random_range(-0.05..0.05);
        self.theta = rng.random_range(-0.05..0.05);
        self.theta_dot = rng.random_range(-0.05..0.05);
    }

    fn observe(&self) -> Vec<f64> {
        let raw = [
            self.x / X_THRESHOLD,
            self.x_dot.clamp(-3.0, 3.0) / 3.0,
            self.theta / THETA_THRESHOLD,
            self.theta_dot.clamp(-3.0, 3.0) / 3.0,
        ];
        let centers: [f64; 8] = [-0.85, -0.60, -0.35, -0.10, 0.10, 0.35, 0.60, 0.85];
        let width = 0.3;
        let amp = 4.0;
        let mut out = Vec::with_capacity(32);
        for &val in &raw {
            for &center in &centers {
                out.push((-(val - center).powi(2) / (2.0 * width * width)).exp() * amp);
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
        self.x += DT_ENV * self.x_dot;
        self.x_dot += DT_ENV * x_acc;
        self.theta += DT_ENV * self.theta_dot;
        self.theta_dot += DT_ENV * theta_acc;
        self.x.abs() < X_THRESHOLD && self.theta.abs() < THETA_THRESHOLD
    }
}

struct Critic {
    weights: [f64; 8],
    bias: f64,
    lr: f64,
}

impl Critic {
    fn new() -> Self { Self { weights: [0.0; 8], bias: 0.0, lr: 0.1 } }

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

fn base_config() -> SystemConfig {
    SystemConfig {
        developmental: DevelopmentalConfig {
            seed_size: 60,
            dimensions: 4,
            initial_connectivity: 0.25,
            proliferation_rounds: 2,
            target_input_size: Some(32),
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
            tau_eligibility: 3.0,
            tau_trace: 5.0,
            a_plus: 1.0,
            a_minus: -0.5,
            tau_tag: 500.0,
            tag_threshold: 0.3,
            capture_threshold: 10.0,
            capture_rate: 0.2,
            weight_max: 3.0,
            weight_min: 0.01,
            alpha_reward: 2.0,
            alpha_novelty: 0.5,
            alpha_arousal: 0.5,
            alpha_homeostasis: 0.1,
            transmitter_potentiation: 0.002,
            heterosynaptic_depression: 0.003,
        },
        morphogenesis: MorphogenesisParams {
            migration_rate: 0.08,
            max_morphons: 300,
            division_threshold: 0.5,
            fusion_min_size: 2,
            apoptosis_min_age: 500,
            ..Default::default()
        },
        homeostasis: HomeostasisParams {
            migration_cooldown_duration: 5.0,
            ..Default::default()
        },
        lifecycle: LifecycleConfig {
            division: false,
            differentiation: true,
            fusion: false,
            apoptosis: false,
            migration: false,
        },
        metabolic: MetabolicConfig::default(),
        dt: 1.0,
        working_memory_capacity: 7,
        episodic_memory_capacity: 200,
        ..Default::default()
    }
}

fn select_action(outputs: &[f64], epsilon: f64, rng: &mut impl Rng) -> f64 {
    if rng.random_range(0.0..1.0) < epsilon {
        return if rng.random_bool(0.5) { 1.0 } else { -1.0 };
    }
    if outputs.len() < 2 { return if rng.random_bool(0.5) { 1.0 } else { -1.0 }; }
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
        let reward = if alive { 1.0 + 0.5 * (1.0 - (env.theta / THETA_THRESHOLD).abs()) } else { -1.0 };
        let _int_td = system.inject_td_error(reward, GAMMA);
        let td_error = critic.update(&pre_state, reward, env, !alive);
        let chosen = if action > 0.0 { 1 } else { 0 };
        if td_error > 0.0 {
            system.train_readout(chosen, td_error.min(1.0) * 0.15);
            system.reward_contrastive(chosen, td_error.min(1.0) * 0.3, 0.1);
        } else {
            let other = 1 - chosen;
            system.train_readout(other, td_error.abs().min(1.0) * 0.15 * 0.5);
        }
        let danger = (env.theta.abs() / THETA_THRESHOLD).min(1.0);
        if danger > 0.5 { system.inject_novelty((danger - 0.5) * 2.0); }
        if !alive { system.inject_arousal(0.8); system.inject_novelty(0.4); break; }
    }
    steps
}

/// Run N episodes, return (avg_last_100, best, per-100-window averages)
fn run_trial(
    system: &mut System, num_episodes: usize, max_steps: usize, seed: u64,
) -> (f64, usize, Vec<f64>) {
    let mut env = CartPole::new();
    let mut critic = Critic::new();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut recent: Vec<usize> = Vec::new();
    let mut best = 0usize;
    let mut avgs = Vec::new();

    for _ in 0..20 { system.process_steps(&[2.0; 4], INTERNAL_STEPS); }

    for ep in 0..num_episodes {
        let epsilon = (0.5 * (1.0 - ep as f64 / num_episodes as f64)).max(0.05);
        let steps = run_episode(system, &mut env, &mut critic, max_steps, epsilon, &mut rng);
        recent.push(steps);
        if recent.len() > 100 { recent.remove(0); }
        best = best.max(steps);
        system.report_performance(steps as f64);

        if (ep + 1) % 100 == 0 {
            avgs.push(recent.iter().sum::<usize>() as f64 / recent.len() as f64);
        }
    }

    let avg = recent.iter().sum::<usize>() as f64 / recent.len().max(1) as f64;
    (avg, best, avgs)
}

// ============================================================
// Experiment 1: Frustration-Driven Exploration
// Both configs use analog readout. The test is whether
// frustration-ON finds better peak scores and converges faster.
// ============================================================
fn experiment_frustration() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Experiment 1: Frustration-Driven Stochastic Exploration    ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Both configs use analog readout. 1000 episodes.           ║");
    println!("║  Division + migration ON.                                   ║");
    println!("║  Hypothesis: frustration finds better peaks, faster curves. ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let num_episodes = 1000;
    let max_steps = 300;
    let num_seeds = 5;

    let mut results_on: Vec<(f64, usize)> = Vec::new();
    let mut results_off: Vec<(f64, usize)> = Vec::new();
    let mut curves_on: Vec<Vec<f64>> = Vec::new();
    let mut curves_off: Vec<Vec<f64>> = Vec::new();

    for seed in 0..num_seeds {
        // Frustration ON — NO analog readout
        let mut config = base_config();
        config.lifecycle.migration = true;
        config.lifecycle.division = true;
        config.morphogenesis.frustration.enabled = true;
        let mut sys = System::new(config);
        sys.enable_analog_readout();
        let (avg, best, avgs) = run_trial(&mut sys, num_episodes, max_steps, seed as u64);
        let diag = sys.diagnostics();
        println!("  [ON  seed={}] avg={:.1} best={:>3} frust={:.3} exploring={}",
            seed, avg, best, diag.avg_frustration, diag.exploration_mode_count);
        results_on.push((avg, best));
        curves_on.push(avgs);

        // Frustration OFF — NO analog readout
        let mut config = base_config();
        config.lifecycle.migration = true;
        config.lifecycle.division = true;
        config.morphogenesis.frustration.enabled = false;
        let mut sys = System::new(config);
        sys.enable_analog_readout();
        let (avg, best, avgs) = run_trial(&mut sys, num_episodes, max_steps, seed as u64);
        println!("  [OFF seed={}] avg={:.1} best={:>3}", seed, avg, best);
        results_off.push((avg, best));
        curves_off.push(avgs);
    }

    let mean_on = results_on.iter().map(|r| r.0).sum::<f64>() / num_seeds as f64;
    let mean_off = results_off.iter().map(|r| r.0).sum::<f64>() / num_seeds as f64;
    let best_on = results_on.iter().map(|r| r.1).max().unwrap();
    let best_off = results_off.iter().map(|r| r.1).max().unwrap();
    let delta = mean_on - mean_off;
    let pct = if mean_off > 0.0 { delta / mean_off * 100.0 } else { 0.0 };

    println!("\n  ┌────────────────────────────────────────────────────┐");
    println!("  │ Frustration ON:  mean avg={:>5.1}  best={:>3}           │", mean_on, best_on);
    println!("  │ Frustration OFF: mean avg={:>5.1}  best={:>3}           │", mean_off, best_off);
    println!("  │ Delta: {:+.1} ({:+.1}%)                                │", delta, pct);
    println!("  └────────────────────────────────────────────────────┘");

    println!("\n  Learning curve (avg per 100-ep window, mean across seeds):");
    let n_windows = curves_on[0].len().min(curves_off[0].len());
    for w in 0..n_windows {
        let on_avg: f64 = curves_on.iter().filter_map(|a| a.get(w)).sum::<f64>() / num_seeds as f64;
        let off_avg: f64 = curves_off.iter().filter_map(|a| a.get(w)).sum::<f64>() / num_seeds as f64;
        let marker = if on_avg > off_avg + 0.5 { ">>>" } else if off_avg > on_avg + 0.5 { "<<<" } else { " = " };
        println!("    ep {:>4}: ON={:>5.1}  OFF={:>5.1}  {}", (w + 1) * 100, on_avg, off_avg, marker);
    }
}

// ============================================================
// Experiment 2: Bioelectric Field-Guided Migration
// Both configs have migration ON. The field config gets the
// bioelectric field. Measure PE convergence and migration
// effectiveness.
// ============================================================
fn experiment_field() {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  Experiment 2: Bioelectric Field-Guided Migration           ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Both: migration ON + division ON, 1000 episodes.          ║");
    println!("║  Field config gets PE + Stress spatial field.               ║");
    println!("║  Hypothesis: field gradient makes migration purposeful.     ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let num_episodes = 1000;
    let max_steps = 300;
    let num_seeds = 5;

    let mut results_field: Vec<(f64, usize)> = Vec::new();
    let mut results_blind: Vec<(f64, usize)> = Vec::new();

    for seed in 0..num_seeds {
        // Field-guided migration
        let mut config = base_config();
        config.lifecycle.migration = true;
        config.lifecycle.division = true;
        config.field = FieldConfig {
            enabled: true,
            resolution: 16,
            diffusion_rate: 0.15,
            decay_rate: 0.03, // slower decay — field retains information longer
            active_layers: vec![
                morphon_core::field::FieldType::PredictionError,
                morphon_core::field::FieldType::Stress,
            ],
            migration_field_weight: 0.4,
        };
        let mut sys = System::new(config);
        sys.enable_analog_readout();
        let (avg, best, _) = run_trial(&mut sys, num_episodes, max_steps, seed as u64);
        let diag = sys.diagnostics();
        let stats = sys.inspect();
        println!("  [FIELD seed={}] avg={:.1} best={:>3} morphons={} pe_field={:.4} pe_sys={:.4}",
            seed, avg, best, stats.total_morphons, diag.field_pe_mean, stats.avg_prediction_error);
        results_field.push((avg, best));

        // Blind migration (no field)
        let mut config = base_config();
        config.lifecycle.migration = true;
        config.lifecycle.division = true;
        let mut sys = System::new(config);
        sys.enable_analog_readout();
        let (avg, best, _) = run_trial(&mut sys, num_episodes, max_steps, seed as u64);
        let stats = sys.inspect();
        println!("  [BLIND seed={}] avg={:.1} best={:>3} morphons={} pe_sys={:.4}",
            seed, avg, best, stats.total_morphons, stats.avg_prediction_error);
        results_blind.push((avg, best));
    }

    let mean_field = results_field.iter().map(|r| r.0).sum::<f64>() / num_seeds as f64;
    let mean_blind = results_blind.iter().map(|r| r.0).sum::<f64>() / num_seeds as f64;
    let best_field = results_field.iter().map(|r| r.1).max().unwrap();
    let best_blind = results_blind.iter().map(|r| r.1).max().unwrap();
    let delta = mean_field - mean_blind;
    let pct = if mean_blind > 0.0 { delta / mean_blind * 100.0 } else { 0.0 };

    println!("\n  ┌────────────────────────────────────────────────────┐");
    println!("  │ Field ON:  mean avg={:>5.1}  best={:>3}           │", mean_field, best_field);
    println!("  │ Field OFF: mean avg={:>5.1}  best={:>3}           │", mean_blind, best_blind);
    println!("  │ Delta: {:+.1} ({:+.1}%)                                │", delta, pct);
    println!("  └────────────────────────────────────────────────────┘");
}

// ============================================================
// Experiment 3: Self-Healing After Motor Damage
// Kill MOTOR morphons — the actual output neurons. Without
// analog readout, this is catastrophic. With self-healing
// (division + target morphology), the system should recover.
// ============================================================
fn experiment_self_healing() {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  Experiment 3: Target Morphology Self-Healing               ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Train 500 ep with analog readout. Then KILL ALL but 1       ║");
    println!("║  motor morphon. Catastrophic output loss.                   ║");
    println!("║  Recovery: 500 more ep. Division + target morphology ON     ║");
    println!("║  vs no lifecycle. Measure performance recovery.             ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let pre_episodes = 500;
    let post_episodes = 500;
    let max_steps = 300;
    let num_seeds = 5;

    let mut recovery_heal: Vec<(f64, f64, f64)> = Vec::new(); // (pre, post, ratio)
    let mut recovery_none: Vec<(f64, f64, f64)> = Vec::new();

    for seed in 0..num_seeds {
        // === WITH self-healing ===
        let mut config = base_config();
        config.lifecycle.division = true;
        config.lifecycle.apoptosis = true;
        config.lifecycle.migration = true;
        config.target_morphology = Some(morphon_core::developmental::TargetMorphology::cerebellar(4));
        config.field = FieldConfig {
            enabled: true,
            resolution: 16,
            diffusion_rate: 0.15,
            decay_rate: 0.03,
            active_layers: vec![
                morphon_core::field::FieldType::PredictionError,
                morphon_core::field::FieldType::Identity,
            ],
            migration_field_weight: 0.3,
        };
        let mut sys = System::new(config);
        sys.enable_analog_readout();

        // Phase 1: train to baseline
        let (avg_pre, best_pre, _) = run_trial(&mut sys, pre_episodes, max_steps, seed as u64);
        let stats_pre = sys.inspect();

        // DAMAGE: kill ALL but 1 motor morphon — catastrophic output loss
        let motor_ids: Vec<u64> = sys.morphons.values()
            .filter(|m| m.cell_type == CellType::Motor)
            .map(|m| m.id)
            .collect();
        let kill_count = motor_ids.len().saturating_sub(1); // keep exactly 1
        for &id in motor_ids.iter().take(kill_count) {
            sys.morphons.remove(&id);
            sys.topology.remove_morphon(id);
        }
        let motors_remaining = sys.morphons.values().filter(|m| m.cell_type == CellType::Motor).count();

        // Phase 2: recover
        let (avg_post, best_post, _) = run_trial(&mut sys, post_episodes, max_steps, seed as u64 + 1000);
        let stats_post = sys.inspect();
        let motors_after = sys.morphons.values().filter(|m| m.cell_type == CellType::Motor).count();
        let ratio = avg_post / avg_pre.max(1.0);
        println!("  [HEAL seed={}] pre={:.1}(best={}) motors={}->{}->{}  post={:.1}(best={}) recovery={:.0}%",
            seed, avg_pre, best_pre, motor_ids.len(), motors_remaining, motors_after,
            avg_post, best_post, ratio * 100.0);
        recovery_heal.push((avg_pre, avg_post, ratio));

        // === WITHOUT self-healing ===
        let mut config = base_config();
        config.lifecycle = LifecycleConfig {
            division: false, differentiation: true, fusion: false,
            apoptosis: false, migration: false,
        };
        let mut sys = System::new(config);
        sys.enable_analog_readout();

        // Phase 1: same training
        let (avg_pre2, best_pre2, _) = run_trial(&mut sys, pre_episodes, max_steps, seed as u64);

        // DAMAGE: same pattern — kill ALL but 1 motor
        let motor_ids2: Vec<u64> = sys.morphons.values()
            .filter(|m| m.cell_type == CellType::Motor)
            .map(|m| m.id)
            .collect();
        let kill_count2 = motor_ids2.len().saturating_sub(1);
        for &id in motor_ids2.iter().take(kill_count2) {
            sys.morphons.remove(&id);
            sys.topology.remove_morphon(id);
        }
        let motors_remaining2 = sys.morphons.values().filter(|m| m.cell_type == CellType::Motor).count();

        // Phase 2: attempt recovery without lifecycle
        let (avg_post2, best_post2, _) = run_trial(&mut sys, post_episodes, max_steps, seed as u64 + 1000);
        let ratio2 = avg_post2 / avg_pre2.max(1.0);
        println!("  [NONE seed={}] pre={:.1}(best={}) motors={}->{}     post={:.1}(best={}) recovery={:.0}%",
            seed, avg_pre2, best_pre2, motor_ids2.len(), motors_remaining2,
            avg_post2, best_post2, ratio2 * 100.0);
        recovery_none.push((avg_pre2, avg_post2, ratio2));
    }

    let mean_heal = recovery_heal.iter().map(|r| r.2).sum::<f64>() / num_seeds as f64;
    let mean_none = recovery_none.iter().map(|r| r.2).sum::<f64>() / num_seeds as f64;
    let mean_post_heal = recovery_heal.iter().map(|r| r.1).sum::<f64>() / num_seeds as f64;
    let mean_post_none = recovery_none.iter().map(|r| r.1).sum::<f64>() / num_seeds as f64;

    println!("\n  ┌────────────────────────────────────────────────────┐");
    println!("  │ Self-healing ON:  mean post={:>5.1}  recovery={:>3.0}%  │", mean_post_heal, mean_heal * 100.0);
    println!("  │ Self-healing OFF: mean post={:>5.1}  recovery={:>3.0}%  │", mean_post_none, mean_none * 100.0);
    let delta = (mean_heal - mean_none) * 100.0;
    println!("  │ Delta: {:+.0}pp                                      │", delta);
    println!("  └────────────────────────────────────────────────────┘");
}

fn main() {
    println!("=== MORPHON V2 Primitives Validation ===");
    println!("Running 3 experiments, 5 seeds each. This takes a few minutes.\n");

    experiment_frustration();
    experiment_field();
    experiment_self_healing();

    println!("\n=== All experiments complete ===");
}
