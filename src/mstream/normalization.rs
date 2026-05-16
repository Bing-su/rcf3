#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use crate::error::{RcfError, Result};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Raw and normalized views of one numerical record.
#[derive(Debug)]
pub(crate) struct NormalizedRecord {
    pub(crate) raw: Vec<f64>,
    pub(crate) normalized: Vec<f64>,
}

/// Normalizes numerical features against their observed streaming ranges.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub(crate) struct NumericRangeNormalizer {
    pub(crate) min_numeric: Vec<f64>,
    pub(crate) max_numeric: Vec<f64>,
}

impl NumericRangeNormalizer {
    pub(crate) fn new(dim: usize) -> Self {
        Self {
            min_numeric: vec![f64::INFINITY; dim],
            max_numeric: vec![f64::NEG_INFINITY; dim],
        }
    }

    pub(crate) fn normalize(
        &mut self,
        numeric: &[f32],
        entries_seen: u64,
    ) -> Result<NormalizedRecord> {
        debug_assert_eq!(numeric.len(), self.min_numeric.len());
        debug_assert_eq!(numeric.len(), self.max_numeric.len());

        let mut raw_values = Vec::with_capacity(numeric.len());
        let mut normalized = Vec::with_capacity(numeric.len());

        for (index, value) in numeric.iter().enumerate() {
            let raw = f64::from(*value);
            validate_raw(raw)?;
            raw_values.push(raw);

            let transformed = libm::log(1.0 + raw);
            normalized.push(self.normalize_transformed(index, transformed, entries_seen));
        }

        Ok(NormalizedRecord {
            raw: raw_values,
            normalized,
        })
    }

    pub(crate) fn preview(&self, numeric: &[f32], entries_seen: u64) -> Result<NormalizedRecord> {
        debug_assert_eq!(numeric.len(), self.min_numeric.len());
        debug_assert_eq!(numeric.len(), self.max_numeric.len());

        let mut raw_values = Vec::with_capacity(numeric.len());
        let mut normalized = Vec::with_capacity(numeric.len());

        for (index, value) in numeric.iter().enumerate() {
            let raw = f64::from(*value);
            validate_raw(raw)?;
            raw_values.push(raw);

            let transformed = libm::log(1.0 + raw);
            normalized.push(self.preview_transformed(index, transformed, entries_seen));
        }

        Ok(NormalizedRecord {
            raw: raw_values,
            normalized,
        })
    }

    fn normalize_transformed(&mut self, index: usize, transformed: f64, entries_seen: u64) -> f64 {
        if entries_seen == 0 {
            self.min_numeric[index] = transformed;
            self.max_numeric[index] = transformed;
            return 0.0;
        }

        self.min_numeric[index] = self.min_numeric[index].min(transformed);
        self.max_numeric[index] = self.max_numeric[index].max(transformed);

        let span = self.max_numeric[index] - self.min_numeric[index];
        if span <= f64::EPSILON {
            0.0
        } else {
            (transformed - self.min_numeric[index]) / span
        }
    }

    fn preview_transformed(&self, index: usize, transformed: f64, entries_seen: u64) -> f64 {
        if entries_seen == 0 {
            return 0.0;
        }

        let min = self.min_numeric[index].min(transformed);
        let max = self.max_numeric[index].max(transformed);
        let span = max - min;
        if span <= f64::EPSILON {
            0.0
        } else {
            (transformed - min) / span
        }
    }
}

fn validate_raw(raw: f64) -> Result<()> {
    if !raw.is_finite() {
        return Err(RcfError::InvalidArgument(
            "numeric values must be finite".into(),
        ));
    }
    if raw <= -1.0 {
        return Err(RcfError::InvalidArgument(
            "numeric value must be > -1.0 for ln(1+x) transform".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case(vec![0.0], vec![0.0])]
    #[case(vec![1.0], vec![0.0])]
    fn first_record_starts_at_zero(#[case] input: Vec<f32>, #[case] expected: Vec<f64>) {
        let mut normalizer = NumericRangeNormalizer::new(input.len());
        let output = normalizer.normalize(&input, 0).unwrap();

        assert_eq!(output.normalized, expected);
    }

    #[test]
    fn repeated_value_remains_zero() {
        let mut normalizer = NumericRangeNormalizer::new(1);
        normalizer.normalize(&[1.0], 0).unwrap();

        let output = normalizer.normalize(&[1.0], 1).unwrap();

        assert_eq!(output.normalized, vec![0.0]);
    }

    #[test]
    fn preview_does_not_mutate_observed_range() {
        let mut normalizer = NumericRangeNormalizer::new(1);
        normalizer.normalize(&[1.0], 0).unwrap();
        let before_min = normalizer.min_numeric.clone();
        let before_max = normalizer.max_numeric.clone();

        let preview = normalizer.preview(&[9.0], 1).unwrap();

        assert_eq!(preview.normalized, vec![1.0]);
        assert_eq!(normalizer.min_numeric, before_min);
        assert_eq!(normalizer.max_numeric, before_max);
    }

    #[rstest]
    #[case(f32::NAN)]
    #[case(f32::INFINITY)]
    #[case(-1.0)]
    #[case(-2.0)]
    fn rejects_invalid_values(#[case] value: f32) {
        let mut normalizer = NumericRangeNormalizer::new(1);
        let err = normalizer.normalize(&[value], 0).unwrap_err();

        assert!(matches!(err, RcfError::InvalidArgument(_)));
    }

    proptest! {
        #[test]
        fn normalized_values_stay_finite_and_bounded(
            records in prop::collection::vec(
                prop::collection::vec(-0.99f32..100.0, 3),
                1..=32,
            ),
        ) {
            let mut normalizer = NumericRangeNormalizer::new(3);

            for (entries_seen, record) in records.iter().enumerate() {
                let output = normalizer.normalize(record, entries_seen as u64).unwrap();
                for value in output.normalized {
                    prop_assert!(value.is_finite());
                    prop_assert!((0.0..=1.0).contains(&value));
                }
            }
        }
    }
}
