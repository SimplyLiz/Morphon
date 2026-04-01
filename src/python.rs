//! Python bindings via PyO3.
//!
//! Exposes the core MI System API to Python:
//!
//! ```python
//! import morphon
//!
//! system = morphon.System()
//! output = system.process([1.0, 0.5, 0.3])
//! system.inject_reward(0.8)
//! stats = system.inspect()
//! print(stats.total_morphons)
//! ```

#[cfg(feature = "python")]
mod bindings {
    use pyo3::prelude::*;
    use std::collections::HashMap;

    use crate::developmental::DevelopmentalConfig as RustDevConfig;
    use crate::system::{System as RustSystem, SystemConfig as RustSystemConfig};

    /// Python-facing system statistics.
    #[pyclass(name = "SystemStats")]
    #[derive(Clone)]
    pub struct PySystemStats {
        #[pyo3(get)]
        pub total_morphons: usize,
        #[pyo3(get)]
        pub total_synapses: usize,
        #[pyo3(get)]
        pub fused_clusters: usize,
        #[pyo3(get)]
        pub max_generation: u32,
        #[pyo3(get)]
        pub avg_energy: f64,
        #[pyo3(get)]
        pub avg_prediction_error: f64,
        #[pyo3(get)]
        pub firing_rate: f64,
        #[pyo3(get)]
        pub working_memory_items: usize,
        #[pyo3(get)]
        pub episodic_memory_items: usize,
        #[pyo3(get)]
        pub step_count: u64,
        #[pyo3(get)]
        pub total_born: usize,
        #[pyo3(get)]
        pub total_died: usize,
        #[pyo3(get)]
        pub total_transdifferentiations: usize,
        #[pyo3(get)]
        pub differentiation_map: HashMap<String, usize>,
    }

    #[pymethods]
    impl PySystemStats {
        fn __repr__(&self) -> String {
            format!(
                "SystemStats(morphons={}, synapses={}, clusters={}, steps={})",
                self.total_morphons, self.total_synapses, self.fused_clusters, self.step_count
            )
        }
    }

    /// The Morphogenic Intelligence System.
    #[pyclass(name = "System")]
    pub struct PySystem {
        inner: RustSystem,
    }

    #[pymethods]
    impl PySystem {
        /// Create a new MI System.
        ///
        /// Args:
        ///     seed_size: Number of initial Morphons (default: 100)
        ///     growth_program: "cortical", "hippocampal", or "cerebellar" (default: "cortical")
        ///     dimensions: Dimensionality of information space (default: 8)
        #[new]
        #[pyo3(signature = (seed_size=100, growth_program="cortical", dimensions=8))]
        fn new(seed_size: usize, growth_program: &str, dimensions: usize) -> PyResult<Self> {
            let dev_config = match growth_program {
                "cortical" => RustDevConfig::cortical(),
                "hippocampal" => RustDevConfig::hippocampal(),
                "cerebellar" => RustDevConfig::cerebellar(),
                other => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Unknown growth program: '{}'. Use 'cortical', 'hippocampal', or 'cerebellar'.",
                        other
                    )));
                }
            };

            let config = RustSystemConfig {
                developmental: RustDevConfig {
                    seed_size,
                    dimensions,
                    ..dev_config
                },
                ..Default::default()
            };

            Ok(PySystem {
                inner: RustSystem::new(config),
            })
        }

        /// Feed input and get output (single inference step with learning).
        fn process(&mut self, input: Vec<f64>) -> Vec<f64> {
            self.inner.process(&input)
        }

        /// Feed external input to sensory Morphons.
        fn feed_input(&mut self, input: Vec<f64>) {
            self.inner.feed_input(&input);
        }

        /// Run one simulation step.
        fn step(&mut self) {
            self.inner.step();
        }

        /// Read output from motor Morphons.
        fn read_output(&self) -> Vec<f64> {
            self.inner.read_output()
        }

        /// Inject a reward signal (dopamine analog, 0.0-1.0).
        fn inject_reward(&mut self, strength: f64) {
            self.inner.inject_reward(strength);
        }

        /// Inject a novelty signal (acetylcholine analog, 0.0-1.0).
        fn inject_novelty(&mut self, strength: f64) {
            self.inner.inject_novelty(strength);
        }

        /// Inject an arousal signal (noradrenaline analog, 0.0-1.0).
        fn inject_arousal(&mut self, strength: f64) {
            self.inner.inject_arousal(strength);
        }

        /// Targeted reward at a specific output port (0-indexed).
        fn inject_reward_at(&mut self, output_index: usize, strength: f64) {
            self.inner.inject_reward_at(output_index, strength);
        }

        /// Targeted inhibition at a specific output port (0-indexed).
        fn inject_inhibition_at(&mut self, output_index: usize, strength: f64) {
            self.inner.inject_inhibition_at(output_index, strength);
        }

        /// Contrastive reward: reward correct output, inhibit incorrect ones.
        fn reward_contrastive(&mut self, correct_index: usize, reward_strength: f64, inhibit_strength: f64) {
            self.inner.reward_contrastive(correct_index, reward_strength, inhibit_strength);
        }

        /// SADP-inspired hidden layer teaching signal for classification.
        fn teach_hidden(&mut self, correct_index: usize, strength: f64) {
            self.inner.teach_hidden(correct_index, strength);
        }

        /// Enable analog readout (Purkinje-style output bypass).
        fn enable_analog_readout(&mut self) {
            self.inner.enable_analog_readout();
        }

        /// Train analog readout weights using delta rule.
        fn train_readout(&mut self, correct_index: usize, learning_rate: f64) {
            self.inner.train_readout(correct_index, learning_rate);
        }

        /// Get system inspection statistics.
        fn inspect(&self) -> PySystemStats {
            let stats = self.inner.inspect();
            let differentiation_map: HashMap<String, usize> = stats
                .differentiation_map
                .into_iter()
                .map(|(k, v)| (format!("{:?}", k), v))
                .collect();

            PySystemStats {
                total_morphons: stats.total_morphons,
                total_synapses: stats.total_synapses,
                fused_clusters: stats.fused_clusters,
                max_generation: stats.max_generation,
                avg_energy: stats.avg_energy,
                avg_prediction_error: stats.avg_prediction_error,
                firing_rate: stats.firing_rate,
                working_memory_items: stats.working_memory_items,
                episodic_memory_items: stats.episodic_memory_items,
                step_count: stats.step_count,
                total_born: stats.total_born,
                total_died: stats.total_died,
                total_transdifferentiations: stats.total_transdifferentiations,
                differentiation_map,
            }
        }

        /// Save system state to JSON string.
        fn save_json(&self) -> PyResult<String> {
            self.inner.save_json().map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Serialization error: {}", e))
            })
        }

        /// Load system from JSON string.
        #[staticmethod]
        fn load_json(json: &str) -> PyResult<Self> {
            let system = RustSystem::load_json(json).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Deserialization error: {}", e))
            })?;
            Ok(PySystem { inner: system })
        }

        /// Get the current simulation step number.
        #[getter]
        fn step_count(&self) -> u64 {
            self.inner.step_count()
        }

        fn __repr__(&self) -> String {
            let stats = self.inner.inspect();
            format!(
                "morphon.System(morphons={}, synapses={}, step={})",
                stats.total_morphons, stats.total_synapses, stats.step_count
            )
        }
    }

    /// Python module definition.
    #[pymodule]
    pub fn morphon(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<PySystem>()?;
        m.add_class::<PySystemStats>()?;
        Ok(())
    }
}
