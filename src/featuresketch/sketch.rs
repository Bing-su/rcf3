#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use rand::rngs::Xoshiro256PlusPlus;

use crate::math;

use super::chain::{ChainLayout, ChainLevel};
use super::config::FeatureSketchConfig;
use super::projection::{Seed4, random_state};

const EPSILON: f64 = 1e-12;
const EPSILON_MASS: f64 = 1e-12;

#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub(crate) struct DecayedValue {
    value: f64,
    epoch: u64,
}

impl DecayedValue {
    fn decayed(&self, target_epoch: u64, half_life: u64) -> f64 {
        self.value * decay_factor(target_epoch.saturating_sub(self.epoch), half_life)
    }

    fn decay_to(&mut self, target_epoch: u64, half_life: u64) {
        self.value = self.decayed(target_epoch, half_life);
        self.epoch = target_epoch;
    }

    fn increment_after_decay(&mut self, target_epoch: u64, half_life: u64) {
        self.decay_to(target_epoch, half_life);
        self.value += 1.0;
    }
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub(crate) struct RowHashSeeds {
    seeds: Vec<Seed4>,
}

impl RowHashSeeds {
    fn new(rows: usize, rng: &mut Xoshiro256PlusPlus) -> Self {
        let seeds = (0..rows).map(|_| Seed4::from_rng(rng)).collect();
        Self { seeds }
    }
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub(crate) struct LevelSketch {
    rows: usize,
    buckets: usize,
    row_hash_seeds: RowHashSeeds,
    cells: Vec<DecayedValue>,
}

impl LevelSketch {
    fn new(rows: usize, buckets: usize, rng: &mut Xoshiro256PlusPlus) -> Self {
        Self {
            rows,
            buckets,
            row_hash_seeds: RowHashSeeds::new(rows, rng),
            cells: vec![DecayedValue::default(); rows * buckets],
        }
    }

    fn min_count(&self, bin: i64, epoch: u64, half_life: u64) -> f64 {
        let mut min_count = f64::INFINITY;
        for row in 0..self.rows {
            let index = self.cell_index(row, bin);
            min_count = min_count.min(self.cells[index].decayed(epoch, half_life));
        }
        min_count
    }

    fn increment(&mut self, bin: i64, epoch: u64, half_life: u64) {
        for row in 0..self.rows {
            let index = self.cell_index(row, bin);
            self.cells[index].increment_after_decay(epoch, half_life);
        }
    }

    fn cell_index(&self, row: usize, bin: i64) -> usize {
        let seed = self.row_hash_seeds.seeds[row];
        let bucket =
            (random_state(seed).hash_one((row as u64, bin)) % self.buckets as u64) as usize;
        row * self.buckets + bucket
    }
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub(crate) struct EnsembleSketch {
    levels: Vec<LevelSketch>,
    reference_masses: Vec<DecayedValue>,
}

impl EnsembleSketch {
    pub(crate) fn new(
        config: &FeatureSketchConfig,
        levels: usize,
        rng: &mut Xoshiro256PlusPlus,
    ) -> Self {
        Self {
            levels: (0..levels)
                .map(|_| LevelSketch::new(config.sketch_rows(), config.sketch_buckets(), rng))
                .collect(),
            reference_masses: vec![DecayedValue::default(); levels],
        }
    }

    pub(crate) fn score(
        &self,
        layout: &ChainLayout,
        vector: &[f64],
        epoch: u64,
        config: &FeatureSketchConfig,
    ) -> f64 {
        let half_life = config.decay_half_life();
        let chain_count = layout.len() / layout.chain_depth();
        let mut chain_sum = 0.0;
        for (chain_index, chain_levels) in layout.levels().chunks(layout.chain_depth()).enumerate()
        {
            chain_sum += self.chain_surprise(
                chain_index,
                layout.chain_depth(),
                chain_levels,
                vector,
                epoch,
                half_life,
            );
        }
        chain_sum / chain_count as f64
    }

    pub(crate) fn update(
        &mut self,
        layout: &ChainLayout,
        vector: &[f64],
        epoch: u64,
        config: &FeatureSketchConfig,
    ) {
        let half_life = config.decay_half_life();
        for (level_index, level) in layout.levels().iter().enumerate() {
            self.update_level(level_index, level, vector, epoch, half_life);
        }
    }

    fn chain_surprise(
        &self,
        chain_index: usize,
        chain_depth: usize,
        chain_levels: &[ChainLevel],
        vector: &[f64],
        epoch: u64,
        half_life: u64,
    ) -> f64 {
        let mut surprise: f64 = 0.0;
        for (level_offset, level) in chain_levels.iter().enumerate() {
            let level_index = chain_index * chain_depth + level_offset;
            surprise =
                surprise.max(self.level_surprise(level_index, level, vector, epoch, half_life));
        }
        surprise
    }

    fn level_surprise(
        &self,
        level_index: usize,
        level: &ChainLevel,
        vector: &[f64],
        epoch: u64,
        half_life: u64,
    ) -> f64 {
        let bin = level.bin(vector);
        let count = self.levels[level_index].min_count(bin, epoch, half_life);
        let mass = self.reference_masses[level_index].decayed(epoch, half_life);
        surprise_from_density(count, mass, level.bin_volume_ratio)
    }

    fn update_level(
        &mut self,
        level_index: usize,
        level: &ChainLevel,
        vector: &[f64],
        epoch: u64,
        half_life: u64,
    ) {
        let bin = level.bin(vector);
        self.levels[level_index].increment(bin, epoch, half_life);
        self.reference_masses[level_index].increment_after_decay(epoch, half_life);
    }
}

fn surprise_from_density(count: f64, reference_mass: f64, bin_volume_ratio: f64) -> f64 {
    let observed_mass_ratio = count / reference_mass.max(EPSILON_MASS);
    let density_ratio = (observed_mass_ratio / bin_volume_ratio.max(EPSILON)).clamp(EPSILON, 1.0);
    -math::ln(density_ratio)
}

fn decay_factor(delta: u64, half_life: u64) -> f64 {
    if delta == 0 {
        1.0
    } else {
        math::powf(0.5, delta as f64 / half_life as f64)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use rand::prelude::*;

    use super::*;

    #[test]
    fn increment_after_decay_applies_decay_before_incrementing() {
        let mut value = DecayedValue {
            value: 8.0,
            epoch: 0,
        };

        value.increment_after_decay(2, 2);

        assert_abs_diff_eq!(value.value, 5.0, epsilon = 1.0e-12);
        assert_eq!(value.epoch, 2);
    }

    #[test]
    fn level_sketch_increment_preserves_decay_order() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
        let mut sketch = LevelSketch::new(2, 64, &mut rng);

        sketch.increment(3, 1, 2);
        assert_abs_diff_eq!(sketch.min_count(3, 1, 2), 1.0, epsilon = 1.0e-12);

        sketch.increment(3, 3, 2);
        assert_abs_diff_eq!(sketch.min_count(3, 3, 2), 1.5, epsilon = 1.0e-12);
    }

    #[test]
    fn surprise_from_density_normalizes_by_mass_and_bin_volume() {
        assert_abs_diff_eq!(
            surprise_from_density(25.0, 100.0, 0.25),
            0.0,
            epsilon = 1.0e-12
        );
        assert_abs_diff_eq!(
            surprise_from_density(0.0, 100.0, 0.25),
            -math::ln(EPSILON),
            epsilon = 1.0e-12
        );
    }
}
