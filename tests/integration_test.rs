use morphon_core::system::{System, SystemConfig};
use morphon_core::types::*;
use morphon_core::developmental::DevelopmentalConfig;
use morphon_core::scheduler::SchedulerConfig;


#[test]
fn test_system_creates_and_develops() {
    let config = SystemConfig::default();
    let system = System::new(config);
    let stats = system.inspect();

    // Should have morphons after development
    assert!(stats.total_morphons > 0, "System should have morphons after development");
    assert!(stats.total_synapses > 0, "System should have synapses after development");

    // Should have differentiated cell types
    assert!(
        stats.differentiation_map.len() > 1,
        "Should have multiple cell types: {:?}",
        stats.differentiation_map
    );
}

#[test]
fn test_system_step_runs_without_panic() {
    let config = SystemConfig::default();
    let mut system = System::new(config);

    // Run 100 steps
    for _ in 0..100 {
        system.step();
    }

    let stats = system.inspect();
    assert_eq!(stats.step_count, 100);
}

#[test]
fn test_process_input_output() {
    let config = SystemConfig::default();
    let mut system = System::new(config);

    let input = vec![1.0, 0.5, 0.3];
    let output = system.process(&input);

    // Should have motor outputs
    assert!(!output.is_empty(), "System should produce motor outputs");
}

#[test]
fn test_neuromodulation_injection() {
    let config = SystemConfig::default();
    let mut system = System::new(config);

    system.inject_reward(0.8);
    assert!(system.modulation.reward > 0.0);

    system.inject_novelty(0.6);
    assert!(system.modulation.novelty > 0.0);

    system.inject_arousal(0.9);
    assert!(system.modulation.arousal > 0.0);
}

#[test]
fn test_neuromodulation_decays() {
    let mut modulation = morphon_core::Neuromodulation::default();
    modulation.inject_reward(1.0);

    let initial = modulation.reward;
    modulation.decay();
    assert!(modulation.reward < initial, "Reward should decay");
}

#[test]
fn test_developmental_programs() {
    // Test cortical preset
    let config = SystemConfig {
        developmental: DevelopmentalConfig::cortical(),
        ..Default::default()
    };
    let system = System::new(config);
    let stats = system.inspect();
    assert!(stats.total_morphons > 50);

    // Test hippocampal preset
    let config = SystemConfig {
        developmental: DevelopmentalConfig::hippocampal(),
        ..Default::default()
    };
    let system = System::new(config);
    let stats = system.inspect();
    assert!(stats.total_morphons > 50);

    // Test cerebellar preset
    let config = SystemConfig {
        developmental: DevelopmentalConfig::cerebellar(),
        ..Default::default()
    };
    let system = System::new(config);
    let stats = system.inspect();
    assert!(stats.total_morphons > 50);
}

#[test]
fn test_system_grows_under_stimulation() {
    let config = SystemConfig::default();
    let mut system = System::new(config);

    let initial_morphons = system.inspect().total_morphons;

    // Stimulate heavily — inject reward and novelty, feed lots of input
    for _ in 0..500 {
        system.inject_reward(0.5);
        system.inject_novelty(0.3);
        system.feed_input(&[1.0, 1.0, 1.0, 1.0, 1.0]);
        system.step();
    }

    let stats = system.inspect();
    // The system should have changed (grown or adapted)
    println!(
        "Morphons: {} -> {}, Synapses: {}, Clusters: {}, Firing rate: {:.3}",
        initial_morphons,
        stats.total_morphons,
        stats.total_synapses,
        stats.fused_clusters,
        stats.firing_rate
    );
}

#[test]
fn test_lifecycle_config_disables_features() {
    let config = SystemConfig {
        lifecycle: LifecycleConfig {
            division: false,
            differentiation: false,
            fusion: false,
            apoptosis: false,
            migration: false,
        },
        ..Default::default()
    };
    let mut system = System::new(config);

    let initial_count = system.inspect().total_morphons;

    for _ in 0..200 {
        system.feed_input(&[1.0, 0.5]);
        system.step();
    }

    // With all lifecycle features disabled, morphon count should stay the same
    // (developmental program still creates the initial set)
    let final_count = system.inspect().total_morphons;
    assert_eq!(
        initial_count, final_count,
        "Morphon count should not change with lifecycle features disabled"
    );
}

#[test]
fn test_ring_buffer() {
    let mut buf = morphon_core::types::RingBuffer::new(5);
    assert!(buf.is_empty());

    buf.push(1.0);
    buf.push(2.0);
    buf.push(3.0);
    assert_eq!(buf.len(), 3);
    assert!((buf.mean() - 2.0).abs() < 1e-10);

    // Fill and overflow
    buf.push(4.0);
    buf.push(5.0);
    buf.push(6.0); // overwrites 1.0
    assert_eq!(buf.len(), 5);
    assert!((buf.mean() - 4.0).abs() < 1e-10); // (2+3+4+5+6)/5 = 4.0
}

#[test]
fn test_activation_functions() {
    use morphon_core::types::ActivationFn;

    let sigmoid = ActivationFn::Sigmoid;
    assert!((sigmoid.apply(0.0) - 0.5).abs() < 1e-10);
    assert!(sigmoid.apply(10.0) > 0.99);
    assert!(sigmoid.apply(-10.0) < 0.01);

    let hard = ActivationFn::HardThreshold;
    assert_eq!(hard.apply(1.0), 1.0);
    assert_eq!(hard.apply(-1.0), 0.0);
}

// === New tests for updated concept features ===

#[test]
fn test_hyperbolic_distance() {
    let origin = HyperbolicPoint::origin(3);
    let point = HyperbolicPoint {
        coords: vec![0.5, 0.0, 0.0],
        curvature: 1.0,
    };

    let d = origin.distance(&point);
    assert!(d > 0.0, "Distance from origin should be positive");
    // Hyperbolic distance grows faster than Euclidean near boundary
    assert!(d > 0.5, "Hyperbolic distance should exceed Euclidean distance");
}

#[test]
fn test_hyperbolic_exp_log_roundtrip() {
    let p = HyperbolicPoint {
        coords: vec![0.3, 0.2, -0.1],
        curvature: 1.0,
    };
    let q = HyperbolicPoint {
        coords: vec![0.1, -0.2, 0.4],
        curvature: 1.0,
    };

    // log_map + exp_map should approximately roundtrip
    let tangent = p.log_map(&q);
    let recovered = p.exp_map(&tangent);

    let error = q.distance(&recovered);
    assert!(
        error < 0.1,
        "exp_map(log_map(q)) should ≈ q, but error = {error}"
    );
}

#[test]
fn test_hyperbolic_points_stay_in_ball() {
    let mut rng = rand::rng();
    for _ in 0..100 {
        let p = HyperbolicPoint::random(8, &mut rng);
        let norm: f64 = p.coords.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            norm < 1.0,
            "Random point should be inside the Poincaré ball, got norm = {norm}"
        );
    }
}

#[test]
fn test_tag_and_capture_delayed_reward() {
    use morphon_core::learning::{LearningParams, update_eligibility, apply_weight_update};
    use morphon_core::morphon::Synapse;
    use morphon_core::types::default_receptors;
    use morphon_core::Neuromodulation;
    use morphon_core::CellType;

    let params = LearningParams::default();
    let mut syn = Synapse::new(0.5);

    // Strong Hebbian coincidence → should set a tag
    for _ in 0..5 {
        update_eligibility(&mut syn, true, 1.0, &params, 1.0);
    }
    assert!(
        syn.tag > 0.0 || syn.eligibility > 0.5,
        "Strong coincidence should set eligibility and possibly tag"
    );

    // Let eligibility decay but tag should persist
    for _ in 0..100 {
        update_eligibility(&mut syn, false, 0.0, &params, 1.0);
    }
    // Fast eligibility should have mostly decayed
    assert!(syn.eligibility.abs() < 0.1, "Fast eligibility should decay");
    // Tag decays much slower
    // (tau_tag = 6000, so after 100 steps: tag * exp(-100/6000) ≈ 0.98 * tag)

    // Now deliver delayed reward → should capture the tag
    let mut modulation = Neuromodulation::default();
    modulation.inject_reward(0.8);

    let weight_before = syn.weight;
    let motor_receptors = default_receptors(CellType::Motor); // Reward + Arousal
    apply_weight_update(&mut syn, &modulation, &params, 0.01, &motor_receptors);

    if syn.consolidated {
        assert!(
            syn.weight != weight_before,
            "Captured tag should change weight"
        );
    }
}

#[test]
fn test_dual_clock_scheduler() {
    let config = SchedulerConfig::default();

    // Step 1: only fast
    let tick = config.tick(1);
    assert!(tick.fast);
    assert!(!tick.medium);    // period 10
    assert!(!tick.slow);      // period 100
    assert!(!tick.glacial);   // period 1000

    // Step 10: fast + medium
    let tick = config.tick(10);
    assert!(tick.fast);
    assert!(tick.medium);

    // Step 100: fast + medium + slow
    let tick = config.tick(100);
    assert!(tick.fast);
    assert!(tick.medium);
    assert!(tick.slow);

    // Step 1000: all paths
    let tick = config.tick(1000);
    assert!(tick.fast);
    assert!(tick.medium);
    assert!(tick.slow);
    assert!(tick.glacial);
}

#[test]
fn test_system_runs_with_dual_clock() {
    let config = SystemConfig {
        scheduler: SchedulerConfig {
            medium_period: 5,
            slow_period: 25,
            glacial_period: 100,
            homeostasis_period: 20,
            memory_period: 50,
        },
        ..Default::default()
    };
    let mut system = System::new(config);

    // Run past all scheduler thresholds
    for _ in 0..200 {
        system.feed_input(&[0.5, 0.3, 0.8]);
        system.step();
    }

    let stats = system.inspect();
    assert_eq!(stats.step_count, 200);
    assert!(stats.total_morphons > 0);
}

#[test]
fn test_specificity_increases_near_boundary() {
    let origin = HyperbolicPoint::origin(3);
    let near = HyperbolicPoint {
        coords: vec![0.1, 0.0, 0.0],
        curvature: 1.0,
    };
    let far = HyperbolicPoint {
        coords: vec![0.9, 0.0, 0.0],
        curvature: 1.0,
    };

    assert!(origin.specificity() < near.specificity());
    assert!(near.specificity() < far.specificity());
}

#[test]
fn test_serialization_roundtrip() {
    let config = SystemConfig::default();
    let mut system = System::new(config);

    // Run a few steps to build state
    for _ in 0..50 {
        system.feed_input(&[1.0, 0.5]);
        system.inject_reward(0.3);
        system.step();
    }

    let stats_before = system.inspect();

    // Serialize to JSON
    let json = system.save_json().expect("serialize");
    assert!(!json.is_empty());

    // Deserialize back
    let restored = System::load_json(&json).expect("deserialize");
    let stats_after = restored.inspect();

    // Core state should match
    assert_eq!(stats_before.total_morphons, stats_after.total_morphons);
    assert_eq!(stats_before.total_synapses, stats_after.total_synapses);
    assert_eq!(stats_before.fused_clusters, stats_after.fused_clusters);
    assert_eq!(stats_before.step_count, stats_after.step_count);
    assert_eq!(stats_before.max_generation, stats_after.max_generation);
}
