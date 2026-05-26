#[cfg(not(feature = "std"))]
use alloc::{collections::BTreeMap, string::String, vec::Vec};
#[cfg(feature = "std")]
use std::{collections::BTreeMap, string::String};

use crate::error::{RcfError, Result};
use crate::math;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct NormalizedFeature {
    pub(crate) name: String,
    pub(crate) value: f64,
}

pub(crate) fn normalize<I, N>(features: I) -> Result<Vec<NormalizedFeature>>
where
    I: IntoIterator<Item = (N, f32)>,
    N: AsRef<str>,
{
    let mut combined = BTreeMap::<String, f32>::new();
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
            value: math::asinh(f64::from(value)),
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combines_duplicate_names_and_preserves_zero() {
        let features = normalize([("a", 1.0), ("b", 0.0), ("a", -1.0)]).unwrap();
        assert_eq!(
            features,
            vec![
                NormalizedFeature {
                    name: "a".into(),
                    value: 0.0
                },
                NormalizedFeature {
                    name: "b".into(),
                    value: 0.0
                }
            ]
        );
    }

    #[test]
    fn rejects_non_finite_values() {
        assert!(matches!(
            normalize([("a", f32::NAN)]),
            Err(RcfError::InvalidArgument(_))
        ));
    }
}
