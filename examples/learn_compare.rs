//! Head-to-head comparison of 3 learning approaches on the same task.
//!
//! Option A: Three-factor (teach_hidden + contrastive reward) — current MI approach
//! Option B: Direct supervised (teach_supervised) — delta rule, no three-factor
//! Option C: Minimal no-hidden (2 sensory → 2 motor, direct delta rule)
//!
//! All use the same 2-class synthetic data:
//!   Class 0: inputs [0,1,2,3] are high
//!   Class 1: inputs [4,5,6,7] are high
//!
//! Run: cargo run --example learn_compare --release

use morphon_core::developmental::DevelopmentalConfig;
use morphon_core::learning::LearningParams;
use morphon_core::morphogenesis::MorphogenesisParams;
use morphon_core::morphon::MetabolicConfig;
use morphon_core::scheduler::SchedulerConfig;
use morphon_core::system::{System, SystemConfig};
use morphon_core::types::LifecycleConfig;
use rand::Rng;

const N_INPUTS: usize = 8;
const N_CLASSES: usize = 2;

fn make_sample(class: usize, rng: &mut impl Rng) -> Vec<f64> {
    let bias = 0.0; // NO bias — let class signal dominate
    let scale = 3.0;
    let mut input = vec![bias; N_INPUTS];
    let mut noise = || rng.random_range(-0.2..0.2);
    match class {
        0 => { input[0] += scale + noise(); input[1] += scale * 0.8 + noise(); input[2] += scale * 0.6 + noise(); input[3] += scale * 0.4 + noise(); }
        1 => { input[4] += scale + noise(); input[5] += scale * 0.8 + noise(); input[6] += scale * 0.6 + noise(); input[7] += scale * 0.4 + noise(); }
        _ => {}
    }
    input
}

fn make_system(target_inputs: usize, target_outputs: usize) -> SystemConfig {
    SystemConfig {
        developmental: DevelopmentalConfig {
            seed_size: 25,
            dimensions: 4,
            initial_connectivity: 0.0, // no random — only I/O pathways
            proliferation_rounds: 1,
            target_input_size: Some(target_inputs),
            target_output_size: Some(target_outputs),
            ..DevelopmentalConfig::cortical()
        },
        scheduler: SchedulerConfig {
            medium_period: 1,
            slow_period: 50,
            glacial_period: 500,
            homeostasis_period: 10,
            memory_period: 100,
        },
        learning: LearningParams {
            tau_eligibility: 1.2,
            tau_trace: 3.07,
            a_plus: 4.99,
            a_minus: -4.94,
            tau_tag: 200.0,
            tag_threshold: 0.8,
            capture_threshold: 0.2,
            capture_rate: 1.0,
            weight_max: 2.07,
            weight_min: 0.01,
            alpha_reward: 0.5,
            alpha_novelty: 3.0,
            alpha_arousal: 0.0,
            alpha_homeostasis: 0.1,
            transmitter_potentiation: 0.001,
            heterosynaptic_depression: 0.002, tag_accumulation_rate: 0.3,
            ..Default::default()
        },
        morphogenesis: MorphogenesisParams {
            max_morphons: Some(60),
            ..Default::default()
        },
        homeostasis: Default::default(),
        lifecycle: LifecycleConfig {
            division: false,
            fusion: false,
            apoptosis: false,
            differentiation: false,
            migration: false,
            synaptogenesis: true,
        },
        metabolic: MetabolicConfig::default(),
        dt: 1.0,
        working_memory_capacity: 7,
        episodic_memory_capacity: 50,
        ..Default::default()
    }
}

fn classify(system: &mut System, input: &[f64]) -> usize {
    let outputs = system.process_steps(input, 3);
    if outputs.len() < N_CLASSES { return 0; }
    if outputs[1] > outputs[0] { 1 } else { 0 }
}

fn evaluate(system: &mut System, rng: &mut impl Rng, n: usize) -> f64 {
    let mut correct = 0;
    for _ in 0..n {
        let label = if rng.random_bool(0.5) { 1 } else { 0 };
        let input = make_sample(label, rng);
        let pred = classify(system, &input);
        if pred == label { correct += 1; }
    }
    correct as f64 / n as f64 * 100.0
}

fn run_option_a(rng: &mut impl Rng) -> Vec<f64> {
    println!("--- Option A: Three-Factor (teach_hidden + contrastive) ---");
    let mut system = System::new(make_system(N_INPUTS, N_CLASSES));
    let s = system.inspect();
    println!("  {} morphons, {} synapses, {} in, {} out", s.total_morphons, s.total_synapses, system.input_size(), system.output_size());

    let mut accuracies = Vec::new();
    for epoch in 0..50 {
        for _ in 0..200 {
            let label = if rng.random_bool(0.5) { 1 } else { 0 };
            let input = make_sample(label, rng);
            let _pred = classify(&mut system, &input);
            system.teach_hidden(label, 1.92);
            system.reward_contrastive(label, 0.1, 0.0);
            system.inject_novelty(0.3);
            system.step();
        }
        let acc = evaluate(&mut system, rng, 100);
        accuracies.push(acc);
        if (epoch + 1) % 10 == 0 {
            println!("  Epoch {:>2}: {:.1}%", epoch + 1, acc);
        }
    }
    accuracies
}

fn run_option_b(rng: &mut impl Rng) -> Vec<f64> {
    println!("--- Option B: Direct Supervised (delta rule, 3-factor OFF) ---");
    let mut cfg = make_system(N_INPUTS, N_CLASSES);
    cfg.scheduler.medium_period = 99999; // disable three-factor weight updates
    let mut system = System::new(cfg);
    let s = system.inspect();
    println!("  {} morphons, {} synapses, {} in, {} out", s.total_morphons, s.total_synapses, system.input_size(), system.output_size());

    let mut accuracies = Vec::new();
    for epoch in 0..50 {
        for _ in 0..200 {
            let label = if rng.random_bool(0.5) { 1 } else { 0 };
            let input = make_sample(label, rng);
            let _outputs = system.process_steps(&input, 5);
            system.teach_supervised(label, 0.05);
        }
        let acc = evaluate(&mut system, rng, 100);
        accuracies.push(acc);
        if (epoch + 1) % 10 == 0 {
            println!("  Epoch {:>2}: {:.1}%", epoch + 1, acc);
        }
    }
    accuracies
}

fn run_option_c(rng: &mut impl Rng) -> Vec<f64> {
    println!("--- Option C: Minimal (no hidden layer, direct delta) ---");
    // Tiny system: just sensory → motor directly
    let config = SystemConfig {
        developmental: DevelopmentalConfig {
            seed_size: 4, // just enough for I/O
            dimensions: 2,
            initial_connectivity: 0.0,
            proliferation_rounds: 0, // no growth
            target_input_size: Some(N_INPUTS),
            target_output_size: Some(N_CLASSES),
            ..DevelopmentalConfig::cortical()
        },
        scheduler: SchedulerConfig {
            medium_period: 1,
            slow_period: 1000,
            glacial_period: 10000,
            homeostasis_period: 1000,
            memory_period: 10000,
        },
        learning: LearningParams::default(),
        morphogenesis: MorphogenesisParams {
            max_morphons: Some(20),
            ..Default::default()
        },
        homeostasis: Default::default(),
        lifecycle: LifecycleConfig {
            division: false,
            fusion: false,
            apoptosis: false,
            differentiation: false,
            migration: false,
            synaptogenesis: true,
        },
        metabolic: MetabolicConfig::default(),
        dt: 1.0,
        working_memory_capacity: 3,
        episodic_memory_capacity: 10,
        ..Default::default()
    };
    let mut system = System::new(config);
    let s = system.inspect();
    println!("  {} morphons, {} synapses, {} in, {} out", s.total_morphons, s.total_synapses, system.input_size(), system.output_size());

    let mut accuracies = Vec::new();
    for epoch in 0..50 {
        for _ in 0..200 {
            let label = if rng.random_bool(0.5) { 1 } else { 0 };
            let input = make_sample(label, rng);
            let _outputs = system.process_steps(&input, 5);
            system.teach_supervised(label, 0.05);
        }
        let acc = evaluate(&mut system, rng, 100);
        accuracies.push(acc);
        if (epoch + 1) % 10 == 0 {
            println!("  Epoch {:>2}: {:.1}%", epoch + 1, acc);
        }
    }
    accuracies
}

fn main() {
    println!("=== Learning Approach Comparison ===");
    println!("2 classes, 8 inputs, 200 train/epoch, 100 test, 20 epochs\n");

    let mut rng = rand::rng();

    let a = run_option_a(&mut rng);
    println!();
    let b = run_option_b(&mut rng);
    println!();
    let c = run_option_c(&mut rng);

    println!("\n=== Summary ===");
    println!("Random baseline: 50.0%\n");
    println!("{:<35} {:>8} {:>8} {:>8}", "", "Ep 10", "Ep 30", "Ep 50");
    println!("{:<35} {:>7.1}% {:>7.1}% {:>7.1}%", "A: Three-Factor (MI native)", a[9], a[29], a[49]);
    println!("{:<35} {:>7.1}% {:>7.1}% {:>7.1}%", "B: Supervised Delta (SADP-like)", b[9], b[29], b[49]);
    println!("{:<35} {:>7.1}% {:>7.1}% {:>7.1}%", "C: Minimal No-Hidden (delta)", c[9], c[29], c[49]);

    // Debug Option C: check if motor outputs are class-differentiated
    println!("\nOption C motor outputs after training:");
    let mut system_c = System::new(SystemConfig {
        developmental: DevelopmentalConfig {
            seed_size: 4,
            dimensions: 2,
            initial_connectivity: 0.0,
            proliferation_rounds: 0,
            target_input_size: Some(N_INPUTS),
            target_output_size: Some(N_CLASSES),
            ..DevelopmentalConfig::cortical()
        },
        scheduler: SchedulerConfig { medium_period: 1, slow_period: 1000, glacial_period: 10000, homeostasis_period: 1000, memory_period: 10000 },
        learning: LearningParams::default(),
        morphogenesis: MorphogenesisParams { max_morphons: Some(20), ..Default::default() },
        homeostasis: Default::default(),
        lifecycle: LifecycleConfig { division: false, fusion: false, apoptosis: false, differentiation: false, migration: false,
            synaptogenesis: true,
        },
        metabolic: MetabolicConfig::default(),
        dt: 1.0, working_memory_capacity: 3, episodic_memory_capacity: 10,
        ..Default::default()
    });
    // Train
    for _ in 0..2000 {
        let label = if rng.random_bool(0.5) { 1 } else { 0 };
        let input = make_sample(label, &mut rng);
        let _ = system_c.process_steps(&input, 5);
        system_c.teach_supervised(label, 0.05);
    }
    // Test each class
    for class in 0..2 {
        let input = make_sample(class, &mut rng);
        let out = system_c.process_steps(&input, 5);
        println!("  Class {} → outputs {:?}", class, &out[..N_CLASSES.min(out.len())]);
    }
    // Check weights
    let s = system_c.inspect();
    let d = system_c.diagnostics();
    println!("  morphons={} synapses={} w_mean={:.4} w_std={:.4}", s.total_morphons, s.total_synapses, d.weight_mean, d.weight_std);

    let best_a = a.iter().cloned().fold(0.0f64, f64::max);
    let best_b = b.iter().cloned().fold(0.0f64, f64::max);
    let best_c = c.iter().cloned().fold(0.0f64, f64::max);
    println!("\nBest accuracy: A={:.1}% B={:.1}% C={:.1}%", best_a, best_b, best_c);
}
