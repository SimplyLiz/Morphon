//! Bioelektrisches Feld — spatial field for indirect morphon communication.
//!
//! A 2D scalar field over the Poincare disk that morphons write to and read from.
//! Enables indirect, spatially-diffuse coordination analogous to the biological
//! difference between the nervous system (synapses) and the hormonal system (fields).
//!
//! The field operates on a coarse grid projected from the first two coordinates
//! of the hyperbolic information space. Multiple named layers (PredictionError,
//! Energy, Stress, Identity, etc.) diffuse independently.

use crate::morphon::Morphon;
use crate::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Types of information carried in the field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FieldType {
    /// Where in the system is prediction error high?
    PredictionError,
    /// Where are metabolic resources available?
    Energy,
    /// Where is novelty occurring?
    Novelty,
    /// Where is chronic frustration/stagnation? (from V2 frustration state)
    Stress,
    /// Functional role identity for target morphology regions.
    Identity,
}

/// Configuration for the morphon field system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldConfig {
    /// Whether the field system is enabled.
    pub enabled: bool,
    /// Grid resolution per axis (square grid over the 2D projection).
    pub resolution: usize,
    /// Diffusion rate per slow tick (fraction of Laplacian applied).
    pub diffusion_rate: f64,
    /// Exponential decay rate per slow tick.
    pub decay_rate: f64,
    /// Which field layers are active.
    pub active_layers: Vec<FieldType>,
    /// Weight of field gradient in migration (0.0 = ignore, 1.0 = field only).
    pub migration_field_weight: f64,
}

impl Default for FieldConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            resolution: 32,
            diffusion_rate: 0.1,
            decay_rate: 0.05,
            active_layers: vec![FieldType::PredictionError, FieldType::Energy, FieldType::Stress],
            migration_field_weight: 0.3,
        }
    }
}

/// A single 2D scalar field layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldLayer {
    /// Flat row-major grid: data[y * resolution + x].
    pub data: Vec<f64>,
    pub resolution: usize,
}

impl FieldLayer {
    pub fn new(resolution: usize) -> Self {
        Self {
            data: vec![0.0; resolution * resolution],
            resolution,
        }
    }

    /// Write a value at grid position (additive).
    #[inline]
    pub fn write(&mut self, x: usize, y: usize, value: f64) {
        self.data[y * self.resolution + x] += value;
    }

    /// Read the value at grid position.
    #[inline]
    pub fn read(&self, x: usize, y: usize) -> f64 {
        self.data[y * self.resolution + x]
    }

    /// Compute gradient at (x, y) via central differences.
    /// Returns (dx, dy) in grid space.
    pub fn gradient(&self, x: usize, y: usize) -> (f64, f64) {
        let r = self.resolution;
        let left = if x > 0 { self.data[y * r + (x - 1)] } else { self.data[y * r + x] };
        let right = if x < r - 1 { self.data[y * r + (x + 1)] } else { self.data[y * r + x] };
        let up = if y > 0 { self.data[(y - 1) * r + x] } else { self.data[y * r + x] };
        let down = if y < r - 1 { self.data[(y + 1) * r + x] } else { self.data[y * r + x] };
        ((right - left) * 0.5, (down - up) * 0.5)
    }

    /// Run one diffusion + decay step (discrete 2D heat equation).
    /// O(resolution^2). At 32x32 = 1024 cells this is negligible.
    pub fn diffuse_and_decay(&mut self, diffusion_rate: f64, decay_rate: f64) {
        let r = self.resolution;
        let mut next = vec![0.0; r * r];
        for y in 0..r {
            for x in 0..r {
                let idx = y * r + x;
                let center = self.data[idx];
                let left = if x > 0 { self.data[idx - 1] } else { center };
                let right = if x < r - 1 { self.data[idx + 1] } else { center };
                let up = if y > 0 { self.data[idx - r] } else { center };
                let down = if y < r - 1 { self.data[idx + r] } else { center };
                let laplacian = left + right + up + down - 4.0 * center;
                next[idx] = (center + diffusion_rate * laplacian) * (1.0 - decay_rate);
            }
        }
        self.data = next;
    }

    /// Peak value in the layer.
    pub fn max(&self) -> f64 {
        self.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }

    /// Mean value in the layer.
    pub fn mean(&self) -> f64 {
        if self.data.is_empty() {
            return 0.0;
        }
        self.data.iter().sum::<f64>() / self.data.len() as f64
    }
}

/// The full morphon field system: multiple named layers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MorphonField {
    pub layers: HashMap<FieldType, FieldLayer>,
    pub config: FieldConfig,
}

impl MorphonField {
    /// Create a new field with layers for each active type.
    pub fn new(config: FieldConfig) -> Self {
        let mut layers = HashMap::new();
        for &ft in &config.active_layers {
            layers.insert(ft, FieldLayer::new(config.resolution));
        }
        Self { layers, config }
    }

    /// Project a HyperbolicPoint to (grid_x, grid_y) indices.
    /// Uses the first two coordinates of the Poincare ball, which lie in (-1, 1).
    pub fn project(&self, pos: &HyperbolicPoint) -> (usize, usize) {
        let r = self.config.resolution;
        let x = pos.coords.first().copied().unwrap_or(0.0);
        let y = pos.coords.get(1).copied().unwrap_or(0.0);
        let gx = ((x + 1.0) * 0.5 * (r - 1) as f64)
            .round()
            .clamp(0.0, (r - 1) as f64) as usize;
        let gy = ((y + 1.0) * 0.5 * (r - 1) as f64)
            .round()
            .clamp(0.0, (r - 1) as f64) as usize;
        (gx, gy)
    }

    /// Write phase: all morphons broadcast their state to the field.
    /// Additive on top of the existing (decayed) field — builds a diffused
    /// moving average across slow ticks. Decay in `diffuse_and_decay` handles cleanup.
    pub fn write_from_morphons(&mut self, morphons: &HashMap<MorphonId, Morphon>) {
        for m in morphons.values() {
            let (gx, gy) = self.project(&m.position);
            if let Some(layer) = self.layers.get_mut(&FieldType::PredictionError) {
                layer.write(gx, gy, m.prediction_error);
            }
            if let Some(layer) = self.layers.get_mut(&FieldType::Energy) {
                layer.write(gx, gy, m.energy);
            }
            if let Some(layer) = self.layers.get_mut(&FieldType::Stress) {
                layer.write(gx, gy, m.frustration.frustration_level);
            }
            if let Some(layer) = self.layers.get_mut(&FieldType::Novelty) {
                layer.write(gx, gy, m.prediction_error * 0.5); // novelty approximation
            }
        }
    }

    /// Read the field gradient at a morphon's position for a given layer.
    /// Returns (dx, dy) in grid space.
    pub fn gradient_at(&self, pos: &HyperbolicPoint, layer: FieldType) -> Option<(f64, f64)> {
        self.layers.get(&layer).map(|l| {
            let (gx, gy) = self.project(pos);
            l.gradient(gx, gy)
        })
    }

    /// Diffuse all layers.
    pub fn diffuse(&mut self) {
        let dr = self.config.diffusion_rate;
        let dc = self.config.decay_rate;
        for layer in self.layers.values_mut() {
            layer.diffuse_and_decay(dr, dc);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_field() -> MorphonField {
        MorphonField::new(FieldConfig {
            enabled: true,
            resolution: 8,
            diffusion_rate: 0.1,
            decay_rate: 0.05,
            active_layers: vec![FieldType::PredictionError, FieldType::Energy],
            migration_field_weight: 0.3,
        })
    }

    #[test]
    fn field_layer_write_read() {
        let mut layer = FieldLayer::new(8);
        layer.write(3, 4, 1.5);
        assert!((layer.read(3, 4) - 1.5).abs() < f64::EPSILON);
        // Additive
        layer.write(3, 4, 0.5);
        assert!((layer.read(3, 4) - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn field_diffusion_spreads_value() {
        let mut layer = FieldLayer::new(8);
        layer.write(4, 4, 10.0);
        let center_before = layer.read(4, 4);
        layer.diffuse_and_decay(0.1, 0.0); // no decay, just diffusion
        let center_after = layer.read(4, 4);
        let neighbor = layer.read(5, 4);
        assert!(center_after < center_before, "center should lose value");
        assert!(neighbor > 0.0, "neighbor should gain value from diffusion");
    }

    #[test]
    fn field_decay_reduces_values() {
        let mut layer = FieldLayer::new(8);
        layer.write(4, 4, 10.0);
        layer.diffuse_and_decay(0.0, 0.1); // no diffusion, just decay
        let after = layer.read(4, 4);
        assert!(after < 10.0, "decay should reduce value");
        assert!((after - 9.0).abs() < 0.01, "10 * (1 - 0.1) = 9.0");
    }

    #[test]
    fn field_gradient_points_toward_high() {
        let mut layer = FieldLayer::new(8);
        layer.write(6, 4, 10.0); // high value at right
        let (dx, _dy) = layer.gradient(5, 4); // gradient at (5,4) should point right (+dx)
        assert!(dx > 0.0, "gradient dx should be positive toward high value");
    }

    #[test]
    fn field_projection_clamps_to_grid() {
        let field = make_field();
        // Point at boundary of Poincare ball
        let pos = HyperbolicPoint {
            coords: vec![0.99, -0.99, 0.0, 0.0],
            curvature: 1.0,
        };
        let (gx, gy) = field.project(&pos);
        assert!(gx < field.config.resolution);
        assert!(gy < field.config.resolution);

        // Origin
        let origin = HyperbolicPoint::origin(4);
        let (gx, gy) = field.project(&origin);
        // Origin (0,0) maps to center of grid
        assert_eq!(gx, 4); // (0+1)*0.5*7 = 3.5 rounds to 4
        assert_eq!(gy, 4);
    }

    #[test]
    fn field_write_from_morphons_populates_layers() {
        let mut field = make_field();
        let mut morphons = HashMap::new();
        let mut m = crate::morphon::Morphon::new(1, HyperbolicPoint::origin(4));
        m.prediction_error = 0.5;
        m.energy = 0.8;
        morphons.insert(1, m);

        field.write_from_morphons(&morphons);

        let pe_layer = field.layers.get(&FieldType::PredictionError).unwrap();
        assert!(pe_layer.max() > 0.0, "PE layer should have values after write");
        let energy_layer = field.layers.get(&FieldType::Energy).unwrap();
        assert!(energy_layer.max() > 0.0, "Energy layer should have values after write");
    }
}
