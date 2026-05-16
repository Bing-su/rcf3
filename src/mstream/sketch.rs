use ndarray::{Array1, Array2, Array3, ArrayBase, DataMut, Dimension};
use rand::prelude::*;
use rand::rngs::Xoshiro256PlusPlus;

use crate::error::Result;

use super::math::{ceil_log2, floor_f64, uniform_symmetric};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

fn decay_counts<S, D>(count: &mut ArrayBase<S, D>, factor: f64)
where
    S: DataMut<Elem = f64>,
    D: Dimension,
{
    count.mapv_inplace(|v| v * factor);
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub(crate) struct NumericSketch {
    num_buckets: usize,
    count: Array1<f64>,
}

impl NumericSketch {
    pub(crate) fn new(num_buckets: usize) -> Self {
        Self {
            num_buckets,
            count: Array1::zeros(num_buckets),
        }
    }

    fn hash(&self, value: f64) -> usize {
        let scaled = value * (self.num_buckets.saturating_sub(1) as f64);
        let bucket = floor_f64(scaled) as isize;
        bucket.rem_euclid(self.num_buckets as isize) as usize
    }

    pub(crate) fn insert(&mut self, value: f64, weight: f64) {
        let bucket = self.hash(value);
        self.count[bucket] += weight;
    }

    pub(crate) fn get_count(&self, value: f64) -> f64 {
        let bucket = self.hash(value);
        self.count[bucket]
    }

    pub(crate) fn lower(&mut self, factor: f64) {
        decay_counts(&mut self.count, factor);
    }
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub(crate) struct CategoricalSketch {
    num_rows: usize,
    num_buckets: usize,
    hash_a: Array1<i64>,
    hash_b: Array1<i64>,
    count: Array2<f64>,
}

impl CategoricalSketch {
    pub(crate) fn new(num_rows: usize, num_buckets: usize, rng: &mut Xoshiro256PlusPlus) -> Self {
        let hash_a = Array1::from_shape_simple_fn(num_rows, || {
            (rng.next_u64() % (num_buckets as u64 - 1) + 1) as i64
        });
        let hash_b =
            Array1::from_shape_simple_fn(num_rows, || (rng.next_u64() % num_buckets as u64) as i64);

        Self {
            num_rows,
            num_buckets,
            hash_a,
            hash_b,
            count: Array2::zeros((num_rows, num_buckets)),
        }
    }

    fn hash(&self, value: i64, row: usize) -> usize {
        let state = ahash::RandomState::with_seeds(
            self.hash_a[row] as u64,
            self.hash_b[row] as u64,
            row as u64,
            self.num_buckets as u64,
        );
        (state.hash_one(value) % self.num_buckets as u64) as usize
    }

    pub(crate) fn insert(&mut self, value: i64, weight: f64) {
        for row in 0..self.num_rows {
            let bucket = self.hash(value, row);
            self.count[[row, bucket]] += weight;
        }
    }

    pub(crate) fn get_count(&self, value: i64) -> f64 {
        let mut min_count = f64::INFINITY;
        for row in 0..self.num_rows {
            let bucket = self.hash(value, row);
            min_count = min_count.min(self.count[[row, bucket]]);
        }
        min_count
    }

    pub(crate) fn lower(&mut self, factor: f64) {
        decay_counts(&mut self.count, factor);
    }
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub(crate) struct RecordSketch {
    num_rows: usize,
    num_buckets: usize,
    numeric_dim: usize,
    categorical_dim: usize,
    log_buckets: usize,
    numeric_planes: Array3<f64>,
    categorical_coeffs: Array2<i64>,
    count: Array2<f64>,
}

impl RecordSketch {
    pub(crate) fn new(
        num_rows: usize,
        num_buckets: usize,
        numeric_dim: usize,
        categorical_dim: usize,
        rng: &mut Xoshiro256PlusPlus,
    ) -> Result<Self> {
        let log_buckets = ceil_log2(num_buckets)?;

        let numeric_planes =
            Array3::from_shape_simple_fn((num_rows, log_buckets, numeric_dim), || {
                uniform_symmetric(rng.next_u64())
            });

        let categorical_coeffs =
            Array2::from_shape_fn((num_rows, categorical_dim), |(row, col)| {
                if col + 1 == categorical_dim {
                    (rng.next_u64() % num_buckets as u64) as i64
                } else {
                    let _ = row;
                    (rng.next_u64() % (num_buckets as u64 - 1) + 1) as i64
                }
            });

        Ok(Self {
            num_rows,
            num_buckets,
            numeric_dim,
            categorical_dim,
            log_buckets,
            numeric_planes,
            categorical_coeffs,
            count: Array2::zeros((num_rows, num_buckets)),
        })
    }

    fn numeric_hash(&self, numeric: &[f64], row: usize) -> usize {
        if self.numeric_dim == 0 {
            return 0;
        }

        let mut bits = 0usize;
        for iter in 0..self.log_buckets {
            let mut sum = 0.0;
            for (k, value) in numeric.iter().enumerate().take(self.numeric_dim) {
                sum += self.numeric_planes[[row, iter, k]] * value;
            }
            if sum >= 0.0 && iter < usize::BITS as usize {
                bits |= 1usize << iter;
            }
        }
        bits
    }

    fn categorical_hash(&self, categorical: &[i64], row: usize) -> usize {
        if self.categorical_dim == 0 {
            return 0;
        }

        let mut state = ahash::RandomState::with_seeds(
            self.categorical_coeffs[[row, 0]] as u64,
            row as u64,
            self.num_buckets as u64,
            self.categorical_dim as u64,
        );
        for value in categorical.iter().take(self.categorical_dim) {
            state = ahash::RandomState::with_seeds(
                state.hash_one(*value),
                row as u64,
                self.num_buckets as u64,
                self.categorical_dim as u64,
            );
        }
        (state.hash_one(row as u64) % self.num_buckets as u64) as usize
    }

    pub(crate) fn insert(&mut self, numeric: &[f64], categorical: &[i64], weight: f64) {
        for row in 0..self.num_rows {
            let bucket1 = self.numeric_hash(numeric, row);
            let bucket2 = self.categorical_hash(categorical, row);
            let bucket = (bucket1 + bucket2) % self.num_buckets;
            self.count[[row, bucket]] += weight;
        }
    }

    pub(crate) fn get_count(&self, numeric: &[f64], categorical: &[i64]) -> f64 {
        let mut min_count = f64::INFINITY;
        for row in 0..self.num_rows {
            let bucket1 = self.numeric_hash(numeric, row);
            let bucket2 = self.categorical_hash(categorical, row);
            let bucket = (bucket1 + bucket2) % self.num_buckets;
            min_count = min_count.min(self.count[[row, bucket]]);
        }
        min_count
    }

    pub(crate) fn lower(&mut self, factor: f64) {
        decay_counts(&mut self.count, factor);
    }
}
