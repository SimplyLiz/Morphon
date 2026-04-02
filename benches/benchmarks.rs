use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use morphon_core::system::{System, SystemConfig};
use morphon_core::developmental::DevelopmentalConfig;
use morphon_core::morphon::{Morphon, Synapse};
use morphon_core::topology::Topology;
use morphon_core::resonance::ResonanceEngine;
use morphon_core::learning::{self, LearningParams};
use morphon_core::morphogenesis::{self, MorphogenesisParams};
use morphon_core::homeostasis;
use morphon_core::types::*;
use std::collections::HashMap;

fn bench_system_step(c: &mut Criterion) {
    let config = SystemConfig::default();
    let mut system = System::new(config);

    c.bench_function("system_step_100_morphons", |b| {
        b.iter(|| {
            system.step();
        });
    });
}

fn bench_process_input(c: &mut Criterion) {
    let config = SystemConfig::default();
    let mut system = System::new(config);
    let input = vec![1.0, 0.5, 0.3, 0.8, 0.1];

    c.bench_function("process_input", |b| {
        b.iter(|| {
            system.process(&input);
        });
    });
}

// === Resonance benchmarks ===

fn bench_resonance_propagate(c: &mut Criterion) {
    let mut group = c.benchmark_group("resonance_propagate");

    for &size in &[50, 200, 500] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &n| {
            let mut rng = rand::rng();
            let mut morphons = HashMap::new();
            let mut topo = Topology::new();

            for i in 0..n {
                let mut m = Morphon::new(i, HyperbolicPoint::random(3, &mut rng));
                m.fired = i % 3 == 0; // ~33% firing
                morphons.insert(i, m);
                topo.add_morphon(i);
            }

            // Sparse connectivity: ~5 outgoing per morphon
            for i in 0..n {
                for j in 1..=5 {
                    let target = (i + j * 7) % n;
                    if target != i {
                        topo.add_synapse(i, target, Synapse::new(0.3));
                    }
                }
            }

            let mut engine = ResonanceEngine::new();

            b.iter(|| {
                engine.propagate(&morphons, &topo);
                let mut m_clone = morphons.clone();
                engine.deliver(&mut m_clone, 1.0);
            });
        });
    }
    group.finish();
}

// === Learning benchmarks ===

fn bench_learning_update(c: &mut Criterion) {
    let params = LearningParams::default();

    c.bench_function("eligibility_update_1000_synapses", |b| {
        let mut synapses: Vec<Synapse> = (0..1000).map(|_| Synapse::new(0.5)).collect();

        b.iter(|| {
            for syn in &mut synapses {
                learning::update_eligibility(syn, true, 0.8, &params, 1.0);
            }
        });
    });
}

fn bench_weight_update(c: &mut Criterion) {
    let params = LearningParams::default();
    let mut modulation = morphon_core::Neuromodulation::default();
    modulation.inject_reward(0.5);
    let receptors = default_receptors(CellType::Associative);

    c.bench_function("weight_update_1000_synapses", |b| {
        let mut synapses: Vec<Synapse> = (0..1000)
            .map(|_| {
                let mut s = Synapse::new(0.5);
                s.eligibility = 0.3;
                s
            })
            .collect();

        b.iter(|| {
            for syn in &mut synapses {
                learning::apply_weight_update(syn, &modulation, &params, 0.01, &receptors, [1.0; 4], &Default::default());
            }
        });
    });
}

// === Morphogenesis benchmarks ===

fn bench_pruning(c: &mut Criterion) {
    let lp = LearningParams::default();

    c.bench_function("pruning_500_edges", |b| {
        b.iter_batched(
            || {
                let mut topo = Topology::new();
                for i in 0..100 {
                    topo.add_morphon(i);
                }
                for i in 0..100 {
                    for j in 1..=5 {
                        let target = (i + j * 13) % 100;
                        if target != i {
                            let mut syn = Synapse::new(0.0005);
                            syn.age = 200;
                            syn.usage_count = 1;
                            topo.add_synapse(i, target, syn);
                        }
                    }
                }
                topo
            },
            |mut topo| {
                let morphons = std::collections::HashMap::new();
                morphogenesis::pruning(&mut topo, &lp, &morphons);
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

fn bench_synaptogenesis(c: &mut Criterion) {
    let params = MorphogenesisParams::default();

    c.bench_function("synaptogenesis_100_morphons", |b| {
        let mut rng = rand::rng();
        let mut morphons = HashMap::new();
        let mut topo = Topology::new();

        for i in 0..100 {
            let pos = HyperbolicPoint {
                coords: vec![
                    (i as f64 / 100.0) * 0.5,
                    ((i * 7) as f64 % 100.0 / 100.0) * 0.5,
                    0.0,
                ],
                curvature: 1.0,
            };
            let mut m = Morphon::new(i, pos);
            m.cell_type = CellType::Associative;
            for _ in 0..100 {
                m.activity_history.push(0.4);
            }
            morphons.insert(i, m);
            topo.add_morphon(i);
        }

        b.iter(|| {
            morphogenesis::synaptogenesis(&morphons, &mut topo, &params, &mut rng);
        });
    });
}

// === Scaling benchmarks ===

fn bench_system_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("system_step_scaling");
    group.sample_size(20);

    for &morphon_target in &[50, 200, 500] {
        group.bench_with_input(
            BenchmarkId::from_parameter(morphon_target),
            &morphon_target,
            |b, &target| {
                let config = SystemConfig {
                    developmental: DevelopmentalConfig {
                        seed_size: target,
                        proliferation_rounds: 0, // no proliferation, exact size
                        ..DevelopmentalConfig::cortical()
                    },
                    ..Default::default()
                };
                let mut system = System::new(config);

                // Warm up
                for _ in 0..50 {
                    system.step();
                }

                b.iter(|| {
                    system.feed_input(&[0.5, 0.3, 0.8]);
                    system.step();
                });
            },
        );
    }
    group.finish();
}

// === Synaptic scaling benchmark ===

fn bench_synaptic_scaling(c: &mut Criterion) {
    c.bench_function("synaptic_scaling_100_morphons", |b| {
        b.iter_batched(
            || {
                let mut rng = rand::rng();
                let mut morphons = HashMap::new();
                let mut topo = Topology::new();

                for i in 0..100u64 {
                    let mut m = Morphon::new(i, HyperbolicPoint::random(3, &mut rng));
                    m.homeostatic_setpoint = 0.1;
                    for _ in 0..100 {
                        m.activity_history.push(0.2);
                    }
                    morphons.insert(i, m);
                    topo.add_morphon(i);
                }

                for i in 0..100u64 {
                    for j in 1..=5 {
                        let target = (i + j * 7) % 100;
                        if target != i {
                            topo.add_synapse(i, target, Synapse::new(0.5));
                        }
                    }
                }

                (morphons, topo)
            },
            |(morphons, mut topo)| {
                homeostasis::synaptic_scaling(&morphons, &mut topo);
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

criterion_group!(
    benches,
    bench_system_step,
    bench_process_input,
    bench_resonance_propagate,
    bench_learning_update,
    bench_weight_update,
    bench_pruning,
    bench_synaptogenesis,
    bench_system_scaling,
    bench_synaptic_scaling,
);
criterion_main!(benches);
