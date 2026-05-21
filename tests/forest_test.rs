/// Property-based integration tests using proptest.
///
/// These tests verify invariants of the public `rcf3` API:
///   - `Forest::entries_seen` monotonicity
///   - Inlier vs outlier score ordering after training
///   - Dimension-mismatch error handling
///   - `near_neighbors` returns distance-consistent nearest points
///   - `impute` reconstructs held-out values from seen points
use approx::abs_diff_eq;
use proptest::prelude::*;
use rcf3::{Forest, RcfError};

// ---------------------------------------------------------------------------
// Public RCF facade shape
// ---------------------------------------------------------------------------
mod public_api_surface {
    use super::*;
    use rcf3::{Attribution, ForestBuilder, NeighborResult, RcfConfig, rcf};

    #[test]
    fn top_level_facade_exports_expected_user_facing_types() {
        let config = RcfConfig::new(2)
            .with_shingle_size(3)
            .with_capacity(64)
            .with_num_trees(7)
            .with_time_decay(0.01)
            .with_output_after(5)
            .with_internal_shingling(false)
            .with_initial_accept_fraction(0.25);

        assert_eq!(config.input_dim(), 2);
        assert_eq!(config.shingle_size(), 3);
        assert_eq!(config.capacity(), 64);
        assert_eq!(config.num_trees(), 7);
        assert_eq!(config.time_decay(), 0.01);
        assert_eq!(config.output_after(), 5);
        assert!(!config.internal_shingling());
        assert_eq!(config.initial_accept_fraction(), 0.25);

        let builder: ForestBuilder = Forest::builder(2);
        let forest = builder.seed(7).build().unwrap();
        assert_eq!(forest.config().input_dim(), 2);

        let attr = Attribution {
            below: 1.25,
            above: 0.75,
        };
        assert_eq!(attr.total(), 2.0);

        let neighbor = NeighborResult {
            score: 0.5,
            point: vec![1.0, 2.0],
            distance: 3.0,
        };
        assert_eq!(neighbor.point, vec![1.0, 2.0]);
    }

    #[test]
    fn rcf_module_facade_exports_same_user_facing_types() {
        let config = rcf::RcfConfig::new(1).with_capacity(16);
        let mut forest = rcf::Forest::from_config_seeded(&config, 11).unwrap();
        forest.update(&[1.0]).unwrap();

        let attr = rcf::Attribution {
            below: 0.0,
            above: 0.0,
        };
        let neighbor = rcf::NeighborResult {
            score: 0.0,
            point: vec![1.0],
            distance: 0.0,
        };

        assert_eq!(forest.entries_seen(), 1);
        assert_eq!(attr.total(), 0.0);
        assert_eq!(neighbor.distance, 0.0);
    }
}

// ---------------------------------------------------------------------------
// Forest::entries_seen monotonicity
// ---------------------------------------------------------------------------
mod forest_entries_seen {
    use super::*;

    proptest! {
        #[test]
        fn entries_seen_equals_update_count(n in 1usize..=30) {
            let mut f = Forest::builder(2)
                .num_trees(5)
                .capacity(50)
                .seed(42)
                .build()
                .unwrap();
            for i in 0..n {
                f.update(&[i as f32 * 0.1, i as f32 * 0.2]).unwrap();
            }
            prop_assert_eq!(f.entries_seen(), n as u64);
        }
    }

    #[test]
    fn wrong_full_dim_update_does_not_increment_entries_seen() {
        let mut f = Forest::builder(2)
            .shingle_size(3)
            .internal_shingling(false)
            .num_trees(5)
            .capacity(20)
            .seed(0)
            .build()
            .unwrap();

        let err = f.update(&[0.0, 0.0]).unwrap_err();

        assert!(
            matches!(
                err,
                RcfError::DimensionMismatch {
                    expected: 6,
                    got: 2
                }
            ),
            "unexpected error variant: {err:?}"
        );
        assert_eq!(f.entries_seen(), 0);
    }
}

// ---------------------------------------------------------------------------
// Outlier score and attribution properties.
// ---------------------------------------------------------------------------
mod forest_outlier_properties {
    use super::*;

    /// Small forest trained on a fixed cluster, reused across cases.
    fn trained_cluster_forest() -> Forest {
        let mut f = Forest::builder(2)
            .num_trees(10)
            .capacity(50)
            .output_after(10)
            .seed(42)
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
        ) {
            let f = trained_cluster_forest();
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
        ) {
            let f = trained_cluster_forest();
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
        fn axis_spike_dominates_matching_attribution_axis(magnitude in 5.0f32..25.0f32) {
            let f = trained_cluster_forest();
            let attr = f.attribution(&[magnitude, 0.2]).unwrap();
            let dim0 = attr[0].total();
            let dim1 = attr[1].total();
            prop_assert!(
                dim0 > dim1,
                "x-axis anomaly should dominate attribution: dim0={dim0}, dim1={dim1}"
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
        ) {
            let mut f = Forest::builder(2)
                .num_trees(5)
                .capacity(20)
                .seed(0)
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

    fn trained_forest_for_nn() -> Forest {
        let mut f = Forest::builder(2)
            .num_trees(20)
            .capacity(128)
            .output_after(20)
            .seed(99)
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
        ) {
            let f = trained_forest_for_nn();
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

        #[test]
        fn results_sorted_by_distance(
            sign_x in prop_oneof![Just(-1.0f32), Just(1.0f32)],
            sign_y in prop_oneof![Just(-1.0f32), Just(1.0f32)],
            magnitude in 5.0f32..25.0f32,
            top_k in 2usize..=10,
        ) {
            let f = trained_forest_for_nn();
            let query = [0.5 + sign_x * magnitude, 0.5 + sign_y * magnitude];
            let results = f.near_neighbors(&query, top_k, 0).unwrap();

            prop_assert!(!results.is_empty());
            for w in results.windows(2) {
                prop_assert!(
                    w[0].distance <= w[1].distance,
                    "neighbors are not sorted ascending by distance: {} > {}",
                    w[0].distance,
                    w[1].distance
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

    fn trained_forest_for_impute() -> Forest {
        let mut f = Forest::builder(3)
            .shingle_size(1)
            .internal_shingling(false)
            .num_trees(10)
            .capacity(128)
            .output_after(10)
            .seed(77)
            .build()
            .unwrap();
        for row in impute_training_rows() {
            f.update(&row).unwrap();
        }
        f
    }

    proptest! {
        #[test]
        fn recovers_seen_point_value(row_idx in 0usize..64, missing_idx in 0usize..3) {
            let f = trained_forest_for_impute();
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
            prop_assert!(
                abs_diff_eq!(out[missing_idx], original[missing_idx], epsilon = 1e-5),
                "expected imputed value {} at dim {}, got {}",
                original[missing_idx],
                missing_idx,
                out[missing_idx]
            );
        }
    }
}
