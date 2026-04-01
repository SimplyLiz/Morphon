//! Tiny classification test — fast iteration on learning dynamics.
//!
//! 3 classes, 8 inputs, synthetic data. Runs in seconds.
//! Each class has a distinct input pattern:
//!   Class 0: high values in inputs 0-2, low elsewhere
//!   Class 1: high values in inputs 3-5, low elsewhere
//!   Class 2: high values in inputs 6-7, low elsewhere
//!
//! Run: cargo run --example classify_tiny --release

use morphon_core::developmental::DevelopmentalConfig;
use morphon_core::learning::LearningParams;
use morphon_core::morphogenesis::MorphogenesisParams;
use morphon_core::scheduler::SchedulerConfig;
use morphon_core::system::{System, SystemConfig};
use morphon_core::types::LifecycleConfig;
use rand::Rng;

const N_INPUTS: usize = 8;
const N_CLASSES: usize = 3;
const N_TRAIN: usize = 500;
const N_TEST: usize = 100;

/// Generate a sample for the given class with noise.
fn make_sample(class: usize, rng: &mut impl Rng) -> Vec<f64> {
    let mut input = vec![0.3; N_INPUTS]; // baseline
    let mut noise = || rng.random_range(-0.1..0.1);
    match class {
        0 => { input[0] = 2.0 + noise(); input[1] = 1.8 + noise(); input[2] = 1.5 + noise(); }
        1 => { input[3] = 2.0 + noise(); input[4] = 1.8 + noise(); input[5] = 1.5 + noise(); }
        2 => { input[6] = 2.5 + noise(); input[7] = 2.0 + noise(); }
        _ => {}
    }
    input
}

fn create_system() -> System {
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
            tau_eligibility: 5.0,
            tau_trace: 10.0,
            a_plus: 1.0,
            a_minus: -1.0,
            tau_tag: 200.0,
            tag_threshold: 0.3,
            capture_threshold: 0.2,
            capture_rate: 0.2,
            weight_max: 3.0,
            weight_min: 0.01,
            alpha_reward: 5.0,
            alpha_novelty: 0.5,
            alpha_arousal: 1.0,
            alpha_homeostasis: 0.1,
        },
        morphogenesis: MorphogenesisParams {
            max_morphons: 100,
            ..Default::default()
        },
        homeostasis: Default::default(),
        lifecycle: LifecycleConfig {
            division: false,   // keep it simple — no growth
            fusion: false,
            apoptosis: false,
            differentiation: false,
            migration: true,
        },
        dt: 1.0,
        working_memory_capacity: 7,
        episodic_memory_capacity: 100,
    };
    System::new(config)
}

fn classify(system: &mut System, input: &[f64]) -> usize {
    let outputs = system.process_steps(input, 3);
    if outputs.len() < N_CLASSES { return 0; }
    outputs.iter()
        .take(N_CLASSES)
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn main() {
    println!("=== Tiny Classification Test ===");
    println!("{} inputs, {} classes, {} train, {} test\n", N_INPUTS, N_CLASSES, N_TRAIN, N_TEST);

    let mut system = create_system();
    let mut rng = rand::rng();

    let s = system.inspect();
    println!("System: {} morphons, {} synapses, {} in, {} out",
        s.total_morphons, s.total_synapses, system.input_size(), system.output_size());
    println!("Types: {:?}", s.differentiation_map);

    // Check motor firing before training
    let test_input = make_sample(0, &mut rng);
    let outputs = system.process_steps(&test_input, 3);
    println!("Pre-train motor outputs: {:?}", &outputs[..N_CLASSES.min(outputs.len())]);

    // Diagnostic: check firing by type
    let diag = system.diagnostics();
    println!("Pre-train firing: {}", diag.firing_summary());
    println!();

    // Trace first sample
    {
        let input = make_sample(0, &mut rng);
        println!("\nTrace class 0 input: {:?}", &input[..4]);
        system.feed_input(&input);
        system.step();
        // Check a few morphons
        let mut sens_pots = Vec::new();
        let mut assoc_pots = Vec::new();
        let mut motor_pots = Vec::new();
        for m in system.morphons.values() {
            match m.cell_type {
                morphon_core::CellType::Sensory => sens_pots.push((m.potential, m.threshold, m.fired)),
                morphon_core::CellType::Associative => assoc_pots.push((m.potential, m.threshold, m.fired)),
                morphon_core::CellType::Motor => motor_pots.push((m.potential, m.threshold, m.fired)),
                _ => {}
            }
        }
        println!("Sensory (pot,thr,fire): {:?}", &sens_pots[..3.min(sens_pots.len())]);
        println!("Assoc   (pot,thr,fire): {:?}", &assoc_pots[..3.min(assoc_pots.len())]);
        println!("Motor   (pot,thr,fire): {:?}", &motor_pots);
        println!("Spikes pending: {}", system.diagnostics().spikes_pending);
        // Run 2 more steps to let spikes propagate
        system.step();
        system.step();
        motor_pots.clear();
        for m in system.morphons.values() {
            if m.cell_type == morphon_core::CellType::Motor {
                motor_pots.push((m.potential, m.threshold, m.fired));
            }
        }
        println!("Motor after 3 steps: {:?}", &motor_pots);
    }
    println!();

    // Train
    for epoch in 0..30 {
        let mut correct = 0;
        for _ in 0..N_TRAIN {
            let label = rng.random_range(0..N_CLASSES as u32) as usize;
            let input = make_sample(label, &mut rng);
            let pred = classify(&mut system, &input);

            // SADP-inspired hidden teaching: tell the associative layer which
            // output is correct BEFORE applying reward. This gives hidden neurons
            // class-specific credit.
            system.teach_hidden(label, 0.5);

            // Contrastive reward at the motor layer
            system.reward_contrastive(
                label,
                if pred == label { 1.0 } else { 0.8 },
                0.5,
            );
            if pred != label {
                system.inject_arousal(0.3);
            }

            // Let teaching + reward propagate through the weight update path
            system.step();

            if pred == label { correct += 1; }
        }

        // Test
        let mut test_correct = 0;
        let mut per_class = vec![(0usize, 0usize); N_CLASSES];
        for _ in 0..N_TEST {
            let label = rng.random_range(0..N_CLASSES as u32) as usize;
            let input = make_sample(label, &mut rng);
            let pred = classify(&mut system, &input);
            per_class[label].1 += 1;
            if pred == label {
                test_correct += 1;
                per_class[label].0 += 1;
            }
        }

        let s = system.inspect();
        let diag = system.diagnostics();
        println!(
            "Epoch {} | train {:.1}% | test {:.1}% | m={} s={} fr={:.3} pe={:.3} | {}",
            epoch + 1,
            correct as f64 / N_TRAIN as f64 * 100.0,
            test_correct as f64 / N_TEST as f64 * 100.0,
            s.total_morphons, s.total_synapses, s.firing_rate, s.avg_prediction_error,
            diag.firing_summary(),
        );
        for (c, (hit, total)) in per_class.iter().enumerate() {
            if *total > 0 {
                print!("  c{}={:.0}%", c, *hit as f64 / *total as f64 * 100.0);
            }
        }
        println!();
    }

    // Final motor output for each class
    println!("\nFinal motor outputs per class:");
    for c in 0..N_CLASSES {
        let input = make_sample(c, &mut rng);
        let outputs = system.process_steps(&input, 3);
        let motor: Vec<f64> = outputs.iter().take(N_CLASSES).map(|v| (v * 1000.0).round() / 1000.0).collect();
        println!("  Class {} input → motor {:?}", c, motor);
    }
}
