//! Focused 3-class classification test.
//!
//! Uses the configuration that achieved 62% on 2-class (learn_compare Option B):
//! - Zero bias encoding
//! - teach_supervised with binary pre_act gate
//! - Three-factor DISABLED (medium_period = 99999)
//! - lr = 0.01, 100 epochs × 500 samples
//!
//! Run: cargo run --example classify_3class --release

use morphon_core::developmental::DevelopmentalConfig;
use morphon_core::learning::LearningParams;
use morphon_core::morphogenesis::MorphogenesisParams;
use morphon_core::morphon::MetabolicConfig;
use morphon_core::scheduler::SchedulerConfig;
use morphon_core::system::{System, SystemConfig};
use morphon_core::types::LifecycleConfig;
use rand::{Rng, RngCore};

const N_INPUTS: usize = 9; // 3 per class, no overlap
const N_CLASSES: usize = 3;

/// Zero-bias, non-overlapping class patterns.
fn make_sample(class: usize, rng: &mut impl Rng) -> Vec<f64> {
    let scale = 3.0;
    let mut input = vec![0.0; N_INPUTS];
    let base = class * 3; // class 0: [0,1,2], class 1: [3,4,5], class 2: [6,7,8]
    input[base] = scale + rng.random_range(-0.2..0.2);
    input[base + 1] = scale * 0.7 + rng.random_range(-0.2..0.2);
    input[base + 2] = scale * 0.4 + rng.random_range(-0.2..0.2);
    input
}

fn main() {
    println!("=== 3-Class Classification (supervised delta, zero bias) ===\n");

    let config = SystemConfig {
        developmental: DevelopmentalConfig {
            seed_size: 25,
            dimensions: 4,
            initial_connectivity: 0.0,
            proliferation_rounds: 1,
            target_input_size: Some(N_INPUTS),
            target_output_size: Some(N_CLASSES),
            ..DevelopmentalConfig::cortical()
        },
        scheduler: SchedulerConfig {
            medium_period: 99999, // three-factor OFF
            slow_period: 99999,
            glacial_period: 99999,
            homeostasis_period: 10,
            memory_period: 99999,
        },
        learning: LearningParams::default(),
        morphogenesis: MorphogenesisParams { max_morphons: 60, ..Default::default() },
        homeostasis: Default::default(),
        lifecycle: LifecycleConfig {
            division: false, fusion: false, apoptosis: false,
            differentiation: false, migration: false,
        },
        metabolic: MetabolicConfig::default(),
        dt: 1.0,
        working_memory_capacity: 3,
        episodic_memory_capacity: 10,
        ..Default::default()
    };

    let mut system = System::new(config);
    system.enable_analog_readout(); // Purkinje-style analog output bypass
    let mut rng = rand::rng();

    let s = system.inspect();
    println!("{} morphons, {} synapses, {} in, {} out (analog readout ON)",
        s.total_morphons, s.total_synapses, system.input_size(), system.output_size());
    println!("Types: {:?}\n", s.differentiation_map);

    // === EXTERNAL CONTROL: simple logistic regression alongside MI ===
    // This proves the TASK is solvable. If this learns and MI doesn't,
    // the problem is in the MI propagation pipeline.
    let mut ext_weights = vec![vec![0.0f64; N_INPUTS]; N_CLASSES]; // [class][input]

    for epoch in 0..300 {
        let mut correct = 0;
        let mut ext_correct = 0;
        let mut total = 0;
        for _ in 0..500 {
            let label = (rng.next_u64() % N_CLASSES as u64) as usize;
            let input = make_sample(label, &mut rng);

            // MI propagation + analog readout training (Purkinje-style)
            let _out = system.process_steps(&input, 5);
            system.train_readout(label, 0.05);
            let outputs = system.read_output();
            if outputs.len() >= N_CLASSES {
                let pred = outputs.iter().take(N_CLASSES).enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i).unwrap_or(0);
                if pred == label { correct += 1; }
            }

            // External logistic regression (same data, same lr)
            let ext_out: Vec<f64> = (0..N_CLASSES).map(|c|
                input.iter().zip(ext_weights[c].iter()).map(|(x, w)| x * w).sum::<f64>()
            ).collect();
            let ext_pred = ext_out.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i).unwrap_or(0);
            if ext_pred == label { ext_correct += 1; }
            // Delta rule on external weights
            for c in 0..N_CLASSES {
                let target = if c == label { 1.0 } else { 0.0 };
                let sigmoid = 1.0 / (1.0 + (-ext_out[c]).exp());
                let error = target - sigmoid;
                for (i, &x) in input.iter().enumerate() {
                    ext_weights[c][i] += 0.02 * x * error;
                }
            }

            total += 1;
        }

        // Test
        let mut tc = 0;
        let mut tt = 0;
        let mut per_class = vec![(0usize, 0usize); N_CLASSES];
        for _ in 0..200 {
            let label = (rng.next_u64() % N_CLASSES as u64) as usize;
            let input = make_sample(label, &mut rng);
            let outputs = system.process_steps(&input, 5);
            if outputs.len() >= N_CLASSES {
                let pred = outputs.iter().take(N_CLASSES).enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i).unwrap_or(0);
                per_class[label].1 += 1;
                if pred == label { tc += 1; per_class[label].0 += 1; }
            }
            tt += 1;
        }

        let train_acc = correct as f64 / total as f64 * 100.0;
        let ext_train_acc = ext_correct as f64 / total as f64 * 100.0;
        let test_acc = tc as f64 / tt as f64 * 100.0;

        if (epoch + 1) % 10 == 0 || test_acc > 45.0 {
            println!("Epoch {:>3} | MI={:.1}% ext={:.1}% | test {:.1}% | {}",
                epoch + 1, train_acc, ext_train_acc, test_acc,
                per_class.iter().enumerate()
                    .map(|(c, (h, t))| format!("c{}={:.0}%", c, if *t > 0 { *h as f64 / *t as f64 * 100.0 } else { 0.0 }))
                    .collect::<Vec<_>>().join(" "));
        }
    }

    println!("\nFinal motor outputs:");
    for c in 0..N_CLASSES {
        let input = make_sample(c, &mut rng);
        let out = system.process_steps(&input, 5);
        let motor: Vec<f64> = out.iter().take(N_CLASSES).map(|v| (v * 100.0).round() / 100.0).collect();
        println!("  Class {} → {:?}", c, motor);
    }
}
