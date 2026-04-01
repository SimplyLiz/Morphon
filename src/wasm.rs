//! WebAssembly bindings via wasm-bindgen.
//!
//! Exposes the MI System to JavaScript for browser-based demos.
//!
//! Build: wasm-pack build --target web --features wasm --no-default-features
//!
//! ```js
//! import init, { WasmSystem } from './morphon_core.js';
//! await init();
//!
//! const system = new WasmSystem(50, "cortical", 4);
//! const output = system.process(new Float64Array([1.0, 0.5, 0.3]));
//! system.inject_reward(0.8);
//! const stats = system.inspect();
//! console.log(JSON.parse(stats));
//! ```

#[cfg(feature = "wasm")]
mod bindings {
    use wasm_bindgen::prelude::*;

    use crate::developmental::DevelopmentalConfig;
    use crate::field::FieldConfig;
    use crate::system::{System, SystemConfig};

    /// MI System for WebAssembly.
    #[wasm_bindgen]
    pub struct WasmSystem {
        inner: System,
    }

    #[wasm_bindgen]
    impl WasmSystem {
        /// Create a new MI System.
        ///
        /// @param seed_size - Number of initial Morphons
        /// @param growth_program - "cortical", "hippocampal", or "cerebellar"
        /// @param dimensions - Dimensionality of information space
        #[wasm_bindgen(constructor)]
        pub fn new(seed_size: usize, growth_program: &str, dimensions: usize) -> Result<WasmSystem, JsError> {
            let dev_config = match growth_program {
                "cortical" => DevelopmentalConfig::cortical(),
                "hippocampal" => DevelopmentalConfig::hippocampal(),
                "cerebellar" => DevelopmentalConfig::cerebellar(),
                other => {
                    return Err(JsError::new(&format!(
                        "Unknown growth program: '{}'. Use 'cortical', 'hippocampal', or 'cerebellar'.",
                        other
                    )));
                }
            };

            let config = SystemConfig {
                developmental: DevelopmentalConfig {
                    seed_size,
                    dimensions,
                    ..dev_config
                },
                field: FieldConfig { enabled: true, ..Default::default() },
                ..Default::default()
            };

            Ok(WasmSystem {
                inner: System::new(config),
            })
        }

        /// Feed input and get output (single inference step with learning).
        /// Returns a Float64Array of motor outputs.
        pub fn process(&mut self, input: &[f64]) -> Vec<f64> {
            self.inner.process(input)
        }

        /// Run one simulation step.
        pub fn step(&mut self) {
            self.inner.step();
        }

        /// Feed external input to sensory Morphons.
        pub fn feed_input(&mut self, input: &[f64]) {
            self.inner.feed_input(input);
        }

        /// Read output from motor Morphons.
        pub fn read_output(&self) -> Vec<f64> {
            self.inner.read_output()
        }

        /// Get IDs of morphons that fired this step (cheap — no serialization).
        pub fn fired_ids(&self) -> Vec<u32> {
            self.inner.morphons.values()
                .filter(|m| m.fired)
                .map(|m| m.id as u32)
                .collect()
        }

        /// Get pending spike data as a flat array for visualization.
        /// Layout: [source_id, target_id, initial_delay, remaining_delay, strength] × N
        pub fn pending_spikes_flat(&self) -> Vec<f64> {
            let pending = self.inner.resonance.pending_spikes();
            let mut buf = Vec::with_capacity(pending.len() * 5);
            for s in pending.iter() {
                buf.push(s.source as f64);
                buf.push(s.target as f64);
                buf.push(s.initial_delay);
                buf.push(s.delay);
                buf.push(s.strength);
            }
            buf
        }

        /// Get IDs of morphons that received a spike delivery this step.
        /// Used for arrival flash effect in the visualizer.
        pub fn delivered_target_ids(&self) -> Vec<u32> {
            self.inner.resonance.last_delivered()
                .iter()
                .map(|s| s.target as u32)
                .collect()
        }

        /// Inject a reward signal (dopamine analog, 0.0-1.0).
        pub fn inject_reward(&mut self, strength: f64) {
            self.inner.inject_reward(strength);
        }

        /// Inject a novelty signal (acetylcholine analog, 0.0-1.0).
        pub fn inject_novelty(&mut self, strength: f64) {
            self.inner.inject_novelty(strength);
        }

        /// Inject an arousal signal (noradrenaline analog, 0.0-1.0).
        pub fn inject_arousal(&mut self, strength: f64) {
            self.inner.inject_arousal(strength);
        }

        /// Targeted reward at a specific output port.
        pub fn inject_reward_at(&mut self, output_index: usize, strength: f64) {
            self.inner.inject_reward_at(output_index, strength);
        }

        /// Targeted inhibition at a specific output port.
        pub fn inject_inhibition_at(&mut self, output_index: usize, strength: f64) {
            self.inner.inject_inhibition_at(output_index, strength);
        }

        /// Contrastive reward: reward correct output, inhibit incorrect ones.
        pub fn reward_contrastive(&mut self, correct_index: usize, reward_strength: f64, inhibit_strength: f64) {
            self.inner.reward_contrastive(correct_index, reward_strength, inhibit_strength);
        }

        /// SADP-inspired hidden layer teaching signal for classification.
        pub fn teach_hidden(&mut self, correct_index: usize, strength: f64) {
            self.inner.teach_hidden(correct_index, strength);
        }

        /// Enable analog readout (Purkinje-style output bypass).
        pub fn enable_analog_readout(&mut self) {
            self.inner.enable_analog_readout();
        }

        /// Train analog readout weights using delta rule.
        pub fn train_readout(&mut self, correct_index: usize, learning_rate: f64) {
            self.inner.train_readout(correct_index, learning_rate);
        }

        /// Get system statistics as a JSON string.
        pub fn inspect(&self) -> String {
            let stats = self.inner.inspect();
            serde_json::to_string(&stats).unwrap_or_default()
        }

        /// Get the lineage tree as a JSON string.
        pub fn lineage_json(&self) -> String {
            self.inner.lineage_tree().to_json()
        }

        /// Save system state to JSON.
        pub fn save_json(&self) -> Result<String, JsError> {
            self.inner
                .save_json()
                .map_err(|e| JsError::new(&format!("Serialization error: {}", e)))
        }

        /// Load system from JSON.
        #[wasm_bindgen(js_name = loadJson)]
        pub fn load_json(json: &str) -> Result<WasmSystem, JsError> {
            let system = System::load_json(json)
                .map_err(|e| JsError::new(&format!("Deserialization error: {}", e)))?;
            Ok(WasmSystem { inner: system })
        }

        /// Get the topology as a JSON object with nodes and edges for visualization.
        pub fn topology_json(&self) -> String {
            let nodes: Vec<serde_json::Value> = self
                .inner
                .morphons
                .values()
                .map(|m| {
                    // V3: look up cluster epistemic state if morphon is fused
                    let (cluster_id, epistemic_state, skepticism) = m.fused_with
                        .and_then(|cid| self.inner.clusters.get(&cid).map(|c| (cid, c)))
                        .map(|(cid, c)| {
                            let state_label = match &c.epistemic_state {
                                crate::epistemic::EpistemicState::Supported { .. } => "Supported",
                                crate::epistemic::EpistemicState::Hypothesis { .. } => "Hypothesis",
                                crate::epistemic::EpistemicState::Outdated { .. } => "Outdated",
                                crate::epistemic::EpistemicState::Contested { .. } => "Contested",
                            };
                            (Some(cid), state_label, c.epistemic_history.skepticism)
                        })
                        .unwrap_or((None, "none", 0.0));

                    serde_json::json!({
                        "id": m.id,
                        "cell_type": format!("{:?}", m.cell_type),
                        "x": m.position.coords.first().copied().unwrap_or(0.0),
                        "y": m.position.coords.get(1).copied().unwrap_or(0.0),
                        "z": m.position.coords.get(2).copied().unwrap_or(0.0),
                        "potential": m.potential,
                        "fired": m.fired,
                        "energy": m.energy,
                        "specificity": m.position.specificity(),
                        "generation": m.generation,
                        "differentiation": m.differentiation_level,
                        "prediction_error": m.prediction_error,
                        "desire": m.desire,
                        "threshold": m.threshold,
                        "age": m.age,
                        "autonomy": m.autonomy,
                        "division_pressure": m.division_pressure,
                        "firing_rate": m.activity_history.mean(),
                        "fused": m.fused_with.is_some(),
                        // V3: epistemic data
                        "cluster_id": cluster_id,
                        "epistemic_state": epistemic_state,
                        "skepticism": skepticism,
                    })
                })
                .collect();

            let edges: Vec<serde_json::Value> = self
                .inner
                .topology
                .all_edges()
                .into_iter()
                .filter_map(|(from, to, ei)| {
                    self.inner.topology.graph.edge_weight(ei).map(|syn| {
                        let (justified, formation_cause, reinforcement_count) = syn.justification
                            .as_ref()
                            .map(|j| {
                                let cause = match &j.formation_cause {
                                    crate::justification::FormationCause::HebbianCoincidence { .. } => "Hebbian",
                                    crate::justification::FormationCause::InheritedFromDivision { .. } => "Division",
                                    crate::justification::FormationCause::ProximityFormation { .. } => "Proximity",
                                    crate::justification::FormationCause::FusionBridge { .. } => "Fusion",
                                    crate::justification::FormationCause::External { .. } => "External",
                                };
                                (true, cause, j.reinforcement_history.len())
                            })
                            .unwrap_or((false, "none", 0));

                        serde_json::json!({
                            "from": from,
                            "to": to,
                            "weight": syn.weight,
                            "eligibility": syn.eligibility,
                            "consolidated": syn.consolidated,
                            // V3: justification data
                            "justified": justified,
                            "formation_cause": formation_cause,
                            "reinforcement_count": reinforcement_count,
                        })
                    })
                })
                .collect();

            serde_json::json!({
                "nodes": nodes,
                "edges": edges,
            })
            .to_string()
        }

        /// Get V3 governance metrics as a JSON string.
        pub fn governance_json(&self) -> String {
            let diag = &self.inner.diag;
            let mut supported = 0usize;
            let mut hypothesis = 0usize;
            let mut outdated = 0usize;
            let mut contested = 0usize;
            let mut skepticism_sum = 0.0f64;

            for c in self.inner.clusters.values() {
                match &c.epistemic_state {
                    crate::epistemic::EpistemicState::Supported { .. } => supported += 1,
                    crate::epistemic::EpistemicState::Hypothesis { .. } => hypothesis += 1,
                    crate::epistemic::EpistemicState::Outdated { .. } => outdated += 1,
                    crate::epistemic::EpistemicState::Contested { .. } => contested += 1,
                }
                skepticism_sum += c.epistemic_history.skepticism;
            }
            let n_clusters = self.inner.clusters.len().max(1) as f64;

            serde_json::json!({
                "justified_fraction": diag.justified_fraction,
                "consolidated_fraction": diag.consolidated_fraction,
                "cluster_states": {
                    "supported": supported,
                    "hypothesis": hypothesis,
                    "outdated": outdated,
                    "contested": contested,
                },
                "avg_skepticism": skepticism_sum / n_clusters,
                "total_clusters": self.inner.clusters.len(),
            })
            .to_string()
        }

        /// Get the neuromodulation state as a JSON string.
        pub fn modulation_json(&self) -> String {
            serde_json::json!({
                "reward": self.inner.modulation.reward,
                "novelty": self.inner.modulation.novelty,
                "arousal": self.inner.modulation.arousal,
                "homeostasis": self.inner.modulation.homeostasis,
            })
            .to_string()
        }

        /// Number of input ports.
        pub fn input_size(&self) -> usize {
            self.inner.input_size()
        }

        /// Number of output ports.
        pub fn output_size(&self) -> usize {
            self.inner.output_size()
        }

        /// Current simulation step.
        pub fn step_count(&self) -> u64 {
            self.inner.step_count()
        }
    }
}
