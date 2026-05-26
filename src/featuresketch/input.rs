#[cfg(not(feature = "std"))]
use alloc::{collections::BTreeMap, string::String, vec::Vec};
#[cfg(feature = "std")]
use std::collections::BTreeMap;

use crate::error::{RcfError, Result};
use crate::math;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct NormalizedFeature {
    pub(crate) name: String,
    pub(crate) value: f64,
}

pub(crate) fn normalize<I, N>(features: I) -> Result<Vec<NormalizedFeature>>
where
    I: IntoIterator<Item = (N, f64)>,
    N: AsRef<str>,
{
    let mut combined = BTreeMap::<String, f64>::new();
    for (name, value) in features {
        if !value.is_finite() {
            return Err(RcfError::InvalidArgument(
                "feature values must be finite".into(),
            ));
        }
        let entry = combined.entry(String::from(name.as_ref())).or_insert(0.0);
        *entry += value;
        if !entry.is_finite() {
            return Err(RcfError::InvalidArgument(
                "combined feature values must be finite".into(),
            ));
        }
    }

    Ok(combined
        .into_iter()
        .map(|(name, value)| NormalizedFeature {
            name,
            value: math::asinh(value),
        })
        .collect())
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::{collections::BTreeMap, format, vec::Vec};
    #[cfg(feature = "std")]
    use std::collections::BTreeMap;

    use approx::{abs_diff_eq, assert_abs_diff_eq};
    use itertools::izip;
    use proptest::prelude::*;
    use rstest::rstest;

    use super::*;

    #[test]
    fn combines_duplicate_names_and_preserves_zero() {
        let features = normalize([("a", 1.0), ("b", 0.0), ("a", -1.0)]).unwrap();
        assert_eq!(features.len(), 2);
        assert_eq!(features[0].name, "a");
        assert_abs_diff_eq!(features[0].value, 0.0, epsilon = 1.0e-12);
        assert_eq!(features[1].name, "b");
        assert_abs_diff_eq!(features[1].value, 0.0, epsilon = 1.0e-12);
    }

    #[rstest]
    #[case(f64::NAN)]
    #[case(f64::INFINITY)]
    #[case(f64::NEG_INFINITY)]
    fn rejects_non_finite_values(#[case] value: f64) {
        assert!(matches!(
            normalize([("a", value)]),
            Err(RcfError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rejects_non_finite_duplicate_sum() {
        assert!(matches!(
            normalize([("a", f64::MAX), ("a", f64::MAX)]),
            Err(RcfError::InvalidArgument(_))
        ));
    }

    proptest! {
        #[test]
        fn normalize_matches_grouped_asinh_sum(
            entries in prop::collection::vec((0usize..6, -1.0e6f64..1.0e6f64), 0..40)
        ) {
            let input: Vec<(String, f64)> = entries
                .iter()
                .map(|(key, value)| (format!("feature:{key}"), *value))
                .collect();
            let normalized = normalize(input).unwrap();

            let mut expected = BTreeMap::<String, f64>::new();
            for (key, value) in entries {
                *expected.entry(format!("feature:{key}")).or_insert(0.0) += value;
            }
            let expected: Vec<_> = expected
                .into_iter()
                .map(|(name, value)| NormalizedFeature {
                    name,
                    value: math::asinh(value),
                })
                .collect();

            prop_assert_eq!(normalized.len(), expected.len());
            for (actual, expected) in izip!(normalized, expected) {
                prop_assert_eq!(&actual.name, &expected.name);
                prop_assert!(abs_diff_eq!(actual.value, expected.value, epsilon = 1.0e-12));
            }
        }
    }
}
