/// Property-based integration tests using proptest.
///
/// These tests verify `Forest` behavior through the public `rcf3` API:
///   - `Forest::entries_seen` monotonicity
///   - Inlier vs outlier score ordering after training
///   - Dimension-mismatch error handling
///   - `near_neighbors` returns distance-consistent nearest points
///   - `impute` preserves observed values and returns in-range estimates
use approx::abs_diff_eq;
use proptest::prelude::*;
use rcf3::{Attribution, Forest, RcfError};

// ---------------------------------------------------------------------------
// Forest::entries_seen monotonicity
// ---------------------------------------------------------------------------
mod forest_entries_seen {
    use super::*;

    proptest! {
        #[test]
        fn entries_seen_equals_update_count(n in 1usize..=30, seed in any::<u64>()) {
            let mut f = Forest::builder(2)
                .num_trees(5)
                .capacity(50)
                .seed(seed)
                .build()
                .unwrap();
            for i in 0..n {
                f.update(&[i as f32 * 0.1, i as f32 * 0.2]).unwrap();
            }
            prop_assert_eq!(f.entries_seen(), n as u64);
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]
        #[test]
        fn wrong_full_dim_update_does_not_increment_entries_seen(seed in any::<u64>()) {
        let mut f = Forest::builder(2)
            .shingle_size(3)
            .internal_shingling(false)
            .num_trees(5)
            .capacity(20)
            .seed(seed)
            .build()
            .unwrap();

        let err = f.update(&[0.0, 0.0]).unwrap_err();

        prop_assert!(
            matches!(
                err,
                RcfError::DimensionMismatch {
                    expected: 6,
                    got: 2
                }
            ),
            "unexpected error variant: {err:?}"
        );
        prop_assert_eq!(f.entries_seen(), 0);
        }
    }
}

// ---------------------------------------------------------------------------
// Outlier score and attribution properties.
// ---------------------------------------------------------------------------
mod forest_outlier_properties {
    use super::*;

    /// Small forest trained on a fixed cluster, reused across cases.
    fn trained_cluster_forest(seed: u64) -> Forest {
        let mut f = Forest::builder(2)
            .num_trees(10)
            .capacity(50)
            .output_after(10)
            .seed(seed)
            .build()
            .unwrap();
        for i in 0..80u32 {
            let x = (i % 5) as f32 * 0.1;
            f.update(&[x, x]).unwrap();
        }
        assert!(f.is_ready());
        f
    }

    proptest! {
        #[test]
        fn far_outlier_scores_above_cluster(
            sign_x in prop_oneof![Just(-1.0f32), Just(1.0f32)],
            sign_y in prop_oneof![Just(-1.0f32), Just(1.0f32)],
            magnitude in 5.0f32..25.0f32,
            seed in any::<u64>(),
        ) {
            let f = trained_cluster_forest(seed);
            let inlier = f.score(&[0.2, 0.2]).unwrap();
            let outlier = f.score(&[sign_x * magnitude, sign_y * magnitude]).unwrap();
            prop_assert!(
                outlier > inlier,
                "outlier score {outlier} should exceed inlier score {inlier}"
            );
        }

        #[test]
        fn far_outlier_displacement_above_cluster(
            sign_x in prop_oneof![Just(-1.0f32), Just(1.0f32)],
            sign_y in prop_oneof![Just(-1.0f32), Just(1.0f32)],
            magnitude in 5.0f32..25.0f32,
            seed in any::<u64>(),
        ) {
            let f = trained_cluster_forest(seed);
            let inlier = f.displacement_score(&[0.2, 0.2]).unwrap();
            let outlier = f
                .displacement_score(&[sign_x * magnitude, sign_y * magnitude])
                .unwrap();
            prop_assert!(
                outlier > inlier,
                "outlier displacement {outlier} should exceed inlier displacement {inlier}"
            );
        }

        #[test]
        fn axis_spike_attribution_marks_spiked_dimension(
            magnitude in 5.0f32..25.0f32,
            seed in any::<u64>(),
        ) {
            let f = trained_cluster_forest(seed);
            let query = [magnitude, 0.2];
            let score = f.score(&query).unwrap();
            let attr = f.attribution(&query).unwrap();
            let attr_total: f64 = attr.iter().copied().map(Attribution::total).sum();
            let attr_ratio = attr_total / score;
            prop_assert_eq!(attr.len(), 2);
            prop_assert!(
                attr[0].total() > 0.0,
                "x-axis spike should receive positive attribution: attr={attr:?}, seed={seed}"
            );
            prop_assert!(
                attr_ratio.is_finite() && (0.05..=1.01).contains(&attr_ratio),
                "attribution total should be a meaningful bounded fraction of score: attr_total={attr_total}, score={score}, ratio={attr_ratio}, seed={seed}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------
mod forest_error_handling {
    use super::*;

    proptest! {
        #[test]
        fn wrong_dim_update_gives_dimension_mismatch(
            wrong_dim in (1usize..=10).prop_filter("not 2", |&d| d != 2),
            seed in any::<u64>(),
        ) {
            let mut f = Forest::builder(2)
                .num_trees(5)
                .capacity(20)
                .seed(seed)
                .build()
                .unwrap();
            let vec: Vec<f32> = vec![0.0f32; wrong_dim];
            let err = f.update(&vec).unwrap_err();
            prop_assert!(
                matches!(
                    err,
                    RcfError::DimensionMismatch { expected: 2, got }
                    if got == wrong_dim
                ),
                "unexpected error variant: {err:?}"
            );
        }
    }

    #[test]
    fn zero_input_dim_is_invalid_argument() {
        let err = Forest::builder(0).build().unwrap_err();
        assert!(matches!(err, RcfError::InvalidArgument(_)));
    }
}

// ---------------------------------------------------------------------------
// near_neighbors output invariants
// ---------------------------------------------------------------------------
mod neighbor_result_properties {
    use super::*;

    fn l1_distance(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(left, right)| f64::from((left - right).abs()))
            .sum()
    }

    fn trained_forest_for_nn(seed: u64) -> Forest {
        let mut f = Forest::builder(2)
            .num_trees(20)
            .capacity(128)
            .output_after(20)
            .seed(seed)
            .build()
            .unwrap();
        for i in 0..100u32 {
            let dx = (i % 10) as f32 * 0.02 - 0.09;
            let dy = (i / 10) as f32 * 0.02 - 0.09;
            f.update(&[0.5 + dx, 0.5 + dy]).unwrap();
        }
        f
    }

    proptest! {
        #[test]
        fn far_outlier_returns_distance_consistent_neighbors(
            sign_x in prop_oneof![Just(-1.0f32), Just(1.0f32)],
            sign_y in prop_oneof![Just(-1.0f32), Just(1.0f32)],
            magnitude in 5.0f32..25.0f32,
            top_k in 1usize..=10,
            seed in any::<u64>(),
        ) {
            let f = trained_forest_for_nn(seed);
            let query = [0.5 + sign_x * magnitude, 0.5 + sign_y * magnitude];
            let results = f.near_neighbors(&query, top_k, 0).unwrap();

            prop_assert!(!results.is_empty());
            prop_assert!(results.len() <= top_k);
            for w in results.windows(2) {
                prop_assert!(
                    w[0].distance <= w[1].distance,
                    "neighbors are not sorted ascending by distance: {} > {}",
                    w[0].distance,
                    w[1].distance
                );
            }
            for r in &results {
                prop_assert!(r.distance >= 0.0, "distance={}", r.distance);
                prop_assert!(r.score >= 0.0, "score={}", r.score);
                let expected = l1_distance(&query, &r.point);
                prop_assert!(
                    abs_diff_eq!(r.distance, expected, epsilon = 1e-9),
                    "reported distance {} does not match point distance {}",
                    r.distance,
                    expected
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// impute output invariants
// ---------------------------------------------------------------------------
mod imputation_properties {
    use super::*;

    fn impute_training_rows() -> Vec<[f32; 3]> {
        (0..64u32)
            .map(|i| {
                let x = i as f32 * 0.25 - 4.0;
                [x, 2.0 * x + 1.0, -x + 0.5]
            })
            .collect()
    }

    fn trained_forest_for_impute(seed: u64) -> Forest {
        let mut f = Forest::builder(3)
            .shingle_size(1)
            .internal_shingling(false)
            .num_trees(10)
            .capacity(128)
            .output_after(10)
            .seed(seed)
            .build()
            .unwrap();
        for row in impute_training_rows() {
            f.update(&row).unwrap();
        }
        f
    }

    proptest! {
        #[test]
        fn preserves_observed_values_and_imputes_training_range(
            row_idx in 0usize..64,
            missing_idx in 0usize..3,
            seed in any::<u64>(),
        ) {
            let f = trained_forest_for_impute(seed);
            let original = impute_training_rows()[row_idx];
            let mut query = original;
            query[missing_idx] = 0.0;
            let out = f.impute(&query, &[missing_idx], 1.0).unwrap();

            prop_assert_eq!(out.len(), 3);
            for i in 0..3 {
                if i != missing_idx {
                    prop_assert_eq!(out[i], original[i]);
                }
            }
            let rows = impute_training_rows();
            let min_seen = rows
                .iter()
                .map(|row| row[missing_idx])
                .fold(f32::INFINITY, f32::min);
            let max_seen = rows
                .iter()
                .map(|row| row[missing_idx])
                .fold(f32::NEG_INFINITY, f32::max);
            prop_assert!(
                out[missing_idx].is_finite()
                    && out[missing_idx] >= min_seen
                    && out[missing_idx] <= max_seen,
                "expected finite imputed value in the training range [{}, {}] at dim {}, got {}",
                min_seen,
                max_seen,
                missing_idx,
                out[missing_idx]
            );
        }
    }
}
