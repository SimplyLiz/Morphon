use criterion::{criterion_group, criterion_main, Criterion};
use morphon_core::system::{System, SystemConfig};

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

criterion_group!(benches, bench_system_step, bench_process_input);
criterion_main!(benches);
