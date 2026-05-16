use ndarray::{
    Array1, Array2, Array3, ArrayBase, ArrayView1, ArrayView2, DataMut, Dimension, aview1, s,
};
use rand::prelude::*;
use rand::rngs::Xoshiro256PlusPlus;
use rand_distr::StandardNormal;

use crate::error::{RcfError, Result};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

fn decay_counts<S, D>(count: &mut ArrayBase<S, D>, factor: f64)
where
    S: DataMut<Elem = f64>,
    D: Dimension,
{
    count.mapv_inplace(|v| v * factor);
}

fn ceil_log2(value: usize) -> Result<usize> {
    if value < 2 {
        return Err(RcfError::InvalidArgument("num_buckets must be >= 2".into()));
    }

    let bits = ((value - 1).ilog2() + 1) as usize;
    Ok(bits)
}

fn numeric_hash_bits(planes: ArrayView2<'_, f64>, numeric: ArrayView1<'_, f64>) -> usize {
    debug_assert_eq!(
        planes.ncols(),
        numeric.len(),
        "input numeric dimension does not match sketch configuration"
    );

    planes
        .rows()
        .into_iter()
        .enumerate()
        .fold(0usize, |mut bits, (bit, plane_row)| {
            if plane_row.dot(&numeric) > 0.0 {
                bits |= 1usize << bit;
            }
            bits
        })
}

fn categorical_hash_resid(
    coeffs_row: ArrayView1<'_, i64>,
    categorical: &[i64],
    row: usize,
    num_buckets: usize,
) -> usize {
    let mut resid = 0u128;
    for (col, value) in categorical.iter().enumerate().take(coeffs_row.len()) {
        let state = ahash::RandomState::with_seeds(
            coeffs_row[col] as u64,
            row as u64,
            col as u64,
            num_buckets as u64,
        );
        resid = (resid + state.hash_one(*value) as u128) % num_buckets as u128;
    }
    resid as usize
}

fn modular_bucket_sum(lhs: usize, rhs: usize, num_buckets: usize) -> usize {
    ((lhs as u128 + rhs as u128) % num_buckets as u128) as usize
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
        let upper = (self.num_buckets - 1) as f64;
        let scaled = libm::floor(value * self.num_buckets as f64);
        scaled.clamp(0.0, upper) as usize
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

    pub(crate) fn zeroed_like(&self) -> Self {
        Self {
            num_rows: self.num_rows,
            num_buckets: self.num_buckets,
            hash_a: self.hash_a.clone(),
            hash_b: self.hash_b.clone(),
            count: Array2::zeros((self.num_rows, self.num_buckets)),
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
                rng.sample(StandardNormal)
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
            numeric_planes,
            categorical_coeffs,
            count: Array2::zeros((num_rows, num_buckets)),
        })
    }

    fn numeric_hash(&self, numeric: &[f64], row: usize) -> usize {
        if self.numeric_dim == 0 {
            return 0;
        }

        let numeric_view = aview1(numeric);
        numeric_hash_bits(self.numeric_planes.slice(s![row, .., ..]), numeric_view)
    }

    fn categorical_hash(&self, categorical: &[i64], row: usize) -> usize {
        if self.categorical_dim == 0 {
            return 0;
        }

        categorical_hash_resid(
            self.categorical_coeffs.slice(s![row, ..]),
            categorical,
            row,
            self.num_buckets,
        )
    }

    fn hash(&self, numeric: &[f64], categorical: &[i64], row: usize) -> usize {
        modular_bucket_sum(
            self.numeric_hash(numeric, row),
            self.categorical_hash(categorical, row),
            self.num_buckets,
        )
    }

    pub(crate) fn insert(&mut self, numeric: &[f64], categorical: &[i64], weight: f64) {
        for row in 0..self.num_rows {
            let bucket = self.hash(numeric, categorical, row);
            self.count[[row, bucket]] += weight;
        }
    }

    pub(crate) fn get_count(&self, numeric: &[f64], categorical: &[i64]) -> f64 {
        let mut min_count = f64::INFINITY;
        for row in 0..self.num_rows {
            let bucket = self.hash(numeric, categorical, row);
            min_count = min_count.min(self.count[[row, bucket]]);
        }
        min_count
    }

    pub(crate) fn lower(&mut self, factor: f64) {
        decay_counts(&mut self.count, factor);
    }

    pub(crate) fn zeroed_like(&self) -> Self {
        Self {
            num_rows: self.num_rows,
            num_buckets: self.num_buckets,
            numeric_dim: self.numeric_dim,
            categorical_dim: self.categorical_dim,
            numeric_planes: self.numeric_planes.clone(),
            categorical_coeffs: self.categorical_coeffs.clone(),
            count: Array2::zeros((self.num_rows, self.num_buckets)),
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::{vec, vec::Vec};

    use ndarray::array;
    use proptest::prelude::*;
    use rstest::rstest;

    use super::*;

    fn legacy_numeric_hash_bits(
        planes: ArrayView2<'_, f64>,
        numeric: ArrayView1<'_, f64>,
    ) -> usize {
        let mut bits = 0usize;
        for iter in 0..planes.nrows() {
            let mut sum = 0.0;
            for col in 0..numeric.len() {
                sum += planes[[iter, col]] * numeric[col];
            }
            if sum > 0.0 && iter < usize::BITS as usize {
                bits |= 1usize << iter;
            }
        }
        bits
    }

    #[test]
    fn numeric_hash_maps_upper_bound_to_last_bucket() {
        let sketch = NumericSketch::new(8);

        assert_eq!(sketch.hash(0.0), 0);
        assert_eq!(sketch.hash(1.0), 7);
    }

    #[test]
    fn numeric_hash_bits_is_consistent_for_same_inputs() {
        let planes = array![[1.0, -1.0], [-1.0, 1.0]];
        let numeric = array![0.5, 0.25];

        let first = numeric_hash_bits(planes.view(), numeric.view());
        let second = numeric_hash_bits(planes.view(), numeric.view());

        assert_eq!(first, second);
        assert_eq!(first, 0b01);
    }

    #[test]
    fn numeric_hash_bits_treats_zero_projection_as_non_positive() {
        let planes = array![[1.0, -1.0], [-1.0, 1.0]];
        let numeric = array![0.0, 0.0];

        assert_eq!(numeric_hash_bits(planes.view(), numeric.view()), 0b00);
    }

    #[rstest]
    #[case(array![[1.0, -2.0], [0.5, 0.25]], array![0.5, 1.0])]
    #[case(array![[-1.0, -1.0], [2.0, -0.5]], array![0.75, -0.25])]
    #[case(array![[3.0, 1.0], [-3.0, -1.0]], array![0.0, 0.0])]
    fn numeric_hash_bits_matches_legacy_loop(
        #[case] planes: ndarray::Array2<f64>,
        #[case] numeric: ndarray::Array1<f64>,
    ) {
        let current = numeric_hash_bits(planes.view(), numeric.view());
        let legacy = legacy_numeric_hash_bits(planes.view(), numeric.view());

        assert_eq!(current, legacy);
    }

    #[rstest]
    #[should_panic(expected = "input numeric dimension does not match sketch configuration")]
    fn numeric_hash_bits_rejects_dimension_mismatch() {
        let planes = array![[1.0, 0.0], [0.0, 1.0]];
        let numeric = array![0.5];

        let _ = numeric_hash_bits(planes.view(), numeric.view());
    }

    proptest! {
        #[test]
        fn numeric_hash_bits_matches_legacy_for_random_inputs(
            rows in 1usize..=5,
            dim in 1usize..=5,
            plane_values in prop::collection::vec(-5.0f64..5.0, 1..=25),
            numeric_values in prop::collection::vec(-5.0f64..5.0, 1..=5),
        ) {
            let plane_len = rows * dim;
            let planes = plane_values
                .into_iter()
                .cycle()
                .take(plane_len)
                .collect::<Vec<_>>();
            let numeric = numeric_values
                .into_iter()
                .cycle()
                .take(dim)
                .collect::<Vec<_>>();

            let planes = ndarray::Array2::from_shape_vec((rows, dim), planes).unwrap();
            let numeric = ndarray::Array1::from(numeric);

            let current = numeric_hash_bits(planes.view(), numeric.view());
            let legacy = legacy_numeric_hash_bits(planes.view(), numeric.view());

            prop_assert_eq!(current, legacy);
        }
    }

    #[rstest]
    #[case(array![3_i64, 5_i64, 7_i64], vec![11_i64, 13_i64, 17_i64], 2, 16)]
    #[case(array![1_i64, 2_i64], vec![9_i64, 8_i64], 0, 8)]
    fn categorical_hash_resid_is_deterministic(
        #[case] coeffs_row: ndarray::Array1<i64>,
        #[case] categorical: Vec<i64>,
        #[case] row: usize,
        #[case] num_buckets: usize,
    ) {
        let first = categorical_hash_resid(coeffs_row.view(), &categorical, row, num_buckets);
        let second = categorical_hash_resid(coeffs_row.view(), &categorical, row, num_buckets);

        assert_eq!(first, second);
        assert!(first < num_buckets);
    }

    #[test]
    fn modular_bucket_sum_handles_usize_overflow_boundary() {
        let lhs = usize::MAX;
        let rhs = usize::MAX - 1;
        let num_buckets = 1024;

        let expected = 1021; // (lhs + rhs) % num_buckets = (2 * usize::MAX - 1) % 1024

        assert_eq!(modular_bucket_sum(lhs, rhs, num_buckets), expected);
    }

    #[test]
    fn zeroed_record_sketch_preserves_hash_layout() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
        let sketch = RecordSketch::new(2, 16, 2, 2, &mut rng).unwrap();
        let clone = sketch.zeroed_like();

        assert_eq!(
            sketch.numeric_hash(&[1.0, 2.0], 0),
            clone.numeric_hash(&[1.0, 2.0], 0)
        );
        assert_eq!(
            sketch.categorical_hash(&[3, 4], 1),
            clone.categorical_hash(&[3, 4], 1)
        );
        assert_eq!(clone.get_count(&[1.0, 2.0], &[3, 4]), 0.0);
    }

    #[test]
    fn zeroed_categorical_sketch_preserves_hash_layout() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
        let sketch = CategoricalSketch::new(2, 16, &mut rng);
        let clone = sketch.zeroed_like();

        for row in 0..2 {
            assert_eq!(sketch.hash(42, row), clone.hash(42, row));
        }
        assert_eq!(clone.get_count(42), 0.0);
    }
}
