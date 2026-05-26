#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use rand::rngs::Xoshiro256PlusPlus;

use crate::math;

use super::chain::ChainLayout;
use super::config::FeatureSketchConfig;
use super::projection::{Seed4, random_state};

const EPSILON: f64 = 1e-12;
const EPSILON_MASS: f64 = 1e-12;

#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub(crate) struct DecayedValue {
    value: f64,
    epoch: u64,
}

impl DecayedValue {
    fn zero() -> Self {
        Self {
            value: 0.0,
            epoch: 0,
        }
    }

    fn decayed(&self, target_epoch: u64, half_life: u64) -> f64 {
        self.value * decay_factor(target_epoch.saturating_sub(self.epoch), half_life)
    }

    fn decay_to(&mut self, target_epoch: u64, half_life: u64) {
        self.value = self.decayed(target_epoch, half_life);
        self.epoch = target_epoch;
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
            cells: vec![DecayedValue::zero(); rows * buckets],
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
            self.cells[index].decay_to(epoch, half_life);
            self.cells[index].value += 1.0;
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
            reference_masses: vec![DecayedValue::zero(); levels],
        }
    }

    pub(crate) fn score(
        &self,
        layout: &ChainLayout,
        vector: &[f64],
        epoch: u64,
        config: &FeatureSketchConfig,
    ) -> f64 {
        let mut chain_sum = 0.0;
        for (chain_index, chain_levels) in layout.levels().chunks(layout.chain_depth()).enumerate()
        {
            let mut chain_surprise: f64 = 0.0;
            for (level_offset, level) in chain_levels.iter().enumerate() {
                let level_index = chain_index * layout.chain_depth() + level_offset;
                let bin = level.bin(vector);
                let count =
                    self.levels[level_index].min_count(bin, epoch, config.decay_half_life());
                let mass = self.reference_masses[level_index]
                    .decayed(epoch, config.decay_half_life())
                    .max(EPSILON_MASS);
                let observed_mass_ratio = count / mass;
                let density_ratio =
                    (observed_mass_ratio / level.bin_volume_ratio.max(EPSILON)).clamp(EPSILON, 1.0);
                chain_surprise = chain_surprise.max(-math::ln(density_ratio));
            }
            chain_sum += chain_surprise;
        }
        chain_sum / (layout.len() / layout.chain_depth()) as f64
    }

    pub(crate) fn update(
        &mut self,
        layout: &ChainLayout,
        vector: &[f64],
        epoch: u64,
        config: &FeatureSketchConfig,
    ) {
        for (level_index, level) in layout.levels().iter().enumerate() {
            let bin = level.bin(vector);
            self.levels[level_index].increment(bin, epoch, config.decay_half_life());
            self.reference_masses[level_index].decay_to(epoch, config.decay_half_life());
            self.reference_masses[level_index].value += 1.0;
        }
    }
}

fn decay_factor(delta: u64, half_life: u64) -> f64 {
    if delta == 0 {
        1.0
    } else {
        math::powf(0.5, delta as f64 / half_life as f64)
    }
}
