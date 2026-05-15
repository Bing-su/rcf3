#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use rand::prelude::*;
use rand::rngs::Xoshiro256PlusPlus;

use crate::error::Result;

use super::math::{ceil_log2, floor_f64, uniform_symmetric};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub(crate) struct NumericSketch {
    num_buckets: usize,
    count: Vec<f64>,
}

impl NumericSketch {
    pub(crate) fn new(num_buckets: usize) -> Self {
        Self {
            num_buckets,
            count: vec![0.0; num_buckets],
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
        for v in &mut self.count {
            *v *= factor;
        }
    }
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub(crate) struct CategoricalSketch {
    num_rows: usize,
    num_buckets: usize,
    hash_a: Vec<i64>,
    hash_b: Vec<i64>,
    count: Vec<Vec<f64>>,
}

impl CategoricalSketch {
    pub(crate) fn new(num_rows: usize, num_buckets: usize, rng: &mut Xoshiro256PlusPlus) -> Self {
        let mut hash_a = vec![0_i64; num_rows];
        let mut hash_b = vec![0_i64; num_rows];
        for i in 0..num_rows {
            hash_a[i] = (rng.next_u64() % (num_buckets as u64 - 1) + 1) as i64;
            hash_b[i] = (rng.next_u64() % num_buckets as u64) as i64;
        }

        Self {
            num_rows,
            num_buckets,
            hash_a,
            hash_b,
            count: vec![vec![0.0; num_buckets]; num_rows],
        }
    }

    fn hash(&self, value: i64, row: usize) -> usize {
        let m = self.num_buckets as i128;
        let v = value as i128;
        let a = self.hash_a[row] as i128;
        let b = self.hash_b[row] as i128;
        ((v * a + b).rem_euclid(m)) as usize
    }

    pub(crate) fn insert(&mut self, value: i64, weight: f64) {
        for row in 0..self.num_rows {
            let bucket = self.hash(value, row);
            self.count[row][bucket] += weight;
        }
    }

    pub(crate) fn get_count(&self, value: i64) -> f64 {
        let mut min_count = f64::INFINITY;
        for row in 0..self.num_rows {
            let bucket = self.hash(value, row);
            min_count = min_count.min(self.count[row][bucket]);
        }
        min_count
    }

    pub(crate) fn lower(&mut self, factor: f64) {
        for row in &mut self.count {
            for v in row {
                *v *= factor;
            }
        }
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
    numeric_planes: Vec<Vec<Vec<f64>>>,
    categorical_coeffs: Vec<Vec<i64>>,
    count: Vec<Vec<f64>>,
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

        let mut numeric_planes = vec![vec![vec![0.0; numeric_dim]; log_buckets]; num_rows];
        for row_planes in &mut numeric_planes {
            for plane in row_planes {
                for w in plane {
                    *w = uniform_symmetric(rng);
                }
            }
        }

        let mut categorical_coeffs = vec![vec![0_i64; categorical_dim]; num_rows];
        for row_coeffs in &mut categorical_coeffs {
            if categorical_dim > 0 {
                for coeff in row_coeffs
                    .iter_mut()
                    .take(categorical_dim.saturating_sub(1))
                {
                    *coeff = (rng.next_u64() % (num_buckets as u64 - 1) + 1) as i64;
                }
                row_coeffs[categorical_dim - 1] = (rng.next_u64() % num_buckets as u64) as i64;
            }
        }

        Ok(Self {
            num_rows,
            num_buckets,
            numeric_dim,
            categorical_dim,
            log_buckets,
            numeric_planes,
            categorical_coeffs,
            count: vec![vec![0.0; num_buckets]; num_rows],
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
                sum += self.numeric_planes[row][iter][k] * value;
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

        let mut resid = 0_i128;
        let m = self.num_buckets as i128;
        for (k, value) in categorical.iter().enumerate().take(self.categorical_dim) {
            resid =
                (resid + self.categorical_coeffs[row][k] as i128 * (*value as i128)).rem_euclid(m);
        }
        resid as usize
    }

    pub(crate) fn insert(&mut self, numeric: &[f64], categorical: &[i64], weight: f64) {
        for row in 0..self.num_rows {
            let bucket1 = self.numeric_hash(numeric, row);
            let bucket2 = self.categorical_hash(categorical, row);
            let bucket = (bucket1 + bucket2) % self.num_buckets;
            self.count[row][bucket] += weight;
        }
    }

    pub(crate) fn get_count(&self, numeric: &[f64], categorical: &[i64]) -> f64 {
        let mut min_count = f64::INFINITY;
        for row in 0..self.num_rows {
            let bucket1 = self.numeric_hash(numeric, row);
            let bucket2 = self.categorical_hash(categorical, row);
            let bucket = (bucket1 + bucket2) % self.num_buckets;
            min_count = min_count.min(self.count[row][bucket]);
        }
        min_count
    }

    pub(crate) fn lower(&mut self, factor: f64) {
        for row in &mut self.count {
            for v in row {
                *v *= factor;
            }
        }
    }
}
