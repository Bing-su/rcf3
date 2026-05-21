#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use rand::prelude::*;
use rand::rngs::Xoshiro256PlusPlus;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::{
    bounding_box::BoundingBox,
    forest::NeighborCandidate,
    node_arena::{NULL, Node, NodeArena},
    point_store::PointStore,
};

mod mutation;
mod traversal;

// ---------------------------------------------------------------------------
// RcfTree
// ---------------------------------------------------------------------------

/// A single Random Cut Tree.  All point-data is owned by the shared
/// [`PointStore`]; the tree only stores indices and bounding-box metadata.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub(super) struct RcfTree {
    root: usize,
    tree_mass: usize,
    arena: NodeArena,
    rng: Xoshiro256PlusPlus,
    dims: usize,
    #[cfg_attr(feature = "serde", serde(skip, default))]
    path_scratch: Vec<(usize, usize)>,
}

fn split_children(query_value: f32, cut_val: f32, left: usize, right: usize) -> (usize, usize) {
    if query_value <= cut_val {
        (left, right)
    } else {
        (right, left)
    }
}

fn split_child_only(query_value: f32, cut_val: f32, left: usize, right: usize) -> usize {
    if query_value <= cut_val { left } else { right }
}

fn child_for_query(
    query: &[f32],
    cut_dim: usize,
    cut_val: f32,
    left: usize,
    right: usize,
) -> usize {
    split_child_only(query[cut_dim], cut_val, left, right)
}

fn blend_with_cut_probability(prob: f64, child_score: f64, fallback_score: f64) -> f64 {
    if prob == 0.0 {
        child_score
    } else {
        (1.0 - prob) * child_score + prob * fallback_score
    }
}

fn nn_threshold(percentile: usize) -> f64 {
    percentile as f64 / 100.0
}

fn should_descend_primary(probability_of_cut: f64, depth: usize, threshold: f64) -> bool {
    probability_of_cut > threshold || depth == 0
}

fn should_descend_secondary(probability_of_cut: f64) -> bool {
    probability_of_cut > 0.5
}

fn consider_impute_candidate(
    best: Option<NeighborCandidate>,
    candidate: NeighborCandidate,
    centrality: f64,
    rng: &mut Xoshiro256PlusPlus,
) -> Option<NeighborCandidate> {
    match best {
        None => Some(candidate),
        Some(best_candidate) => {
            let best_dist = best_candidate.distance;
            let dist = candidate.distance;
            let accept = if centrality >= 1.0 {
                dist < best_dist
            } else {
                let r: f64 = rng.random::<f64>();
                dist < best_dist || r < 1.0 - centrality
            };
            if accept {
                Some(candidate)
            } else {
                Some(best_candidate)
            }
        }
    }
}

impl RcfTree {
    /// Treat a tree with no root or zero total mass as empty for traversal entry points.
    fn is_effectively_empty(&self) -> bool {
        self.root == NULL || self.tree_mass == 0
    }

    /// Create a new random-cut tree with the given dimensionality, point
    /// capacity hint, and RNG seed.
    pub(super) fn new(dims: usize, capacity: usize, seed: u64) -> Self {
        RcfTree {
            root: NULL,
            tree_mass: 0,
            arena: NodeArena::new(2 * capacity + 4),
            rng: Xoshiro256PlusPlus::seed_from_u64(seed),
            dims,
            path_scratch: Vec::new(),
        }
    }
    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Whether the tree currently has no root node.
    #[cfg(test)]
    fn is_empty(&self) -> bool {
        self.root == NULL
    }
}

// ---------------------------------------------------------------------------
// Utility: compute bounding box of a sub-tree
// ---------------------------------------------------------------------------

/// Build an owned bounding box for the entire sub-tree rooted at `node_id`.
fn subtree_bbox_owned(arena: &NodeArena, node_id: usize, point_store: &PointStore) -> BoundingBox {
    match arena.get(node_id) {
        Node::Leaf { point_idx, .. } => BoundingBox::from_point(point_store.get(*point_idx)),
        Node::Internal { bbox, .. } => bbox.clone(),
    }
}

/// Merge the bounding box for `node_id` into `target` without cloning leaf boxes.
fn merge_subtree_bbox_into(
    target: &mut BoundingBox,
    arena: &NodeArena,
    node_id: usize,
    point_store: &PointStore,
) {
    match arena.get(node_id) {
        Node::Leaf { point_idx, .. } => target.merge_point(point_store.get(*point_idx)),
        Node::Internal { bbox, .. } => target.merge(bbox),
    }
}
// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::{vec, vec::Vec};

    use approx::assert_abs_diff_eq;
    use rstest::*;

    use super::*;
    use crate::rcf::{point_store::PointStore, score::ScoreMode};

    fn make_store_and_tree(points: &[Vec<f32>]) -> (PointStore, RcfTree) {
        let dim = points[0].len();
        let mut store = PointStore::new(dim, 1, 64, false);
        let mut tree = RcfTree::new(dim, 32, 42);

        for p in points {
            let idx = store.add(p).unwrap();
            tree.insert(idx, &store).unwrap();
        }
        (store, tree)
    }

    #[test]
    fn insert_and_tree_mass() {
        let pts: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32, 0.0]).collect();
        let (_, tree) = make_store_and_tree(&pts);
        assert_eq!(tree.tree_mass, 10);
    }

    #[test]
    fn insert_delete_restores_mass() {
        let pts: Vec<Vec<f32>> = (0..5).map(|i| vec![i as f32]).collect();
        let dim = 1;
        let mut store = PointStore::new(dim, 1, 16, false);
        let mut tree = RcfTree::new(dim, 16, 99);
        let mut idxs = Vec::new();
        for p in &pts {
            let idx = store.add(p).unwrap();
            idxs.push(idx);
            tree.insert(idx, &store).unwrap();
        }
        assert_eq!(tree.tree_mass, 5);
        tree.delete(idxs[2], &store).unwrap();
        assert_eq!(tree.tree_mass, 4);
    }

    #[test]
    fn duplicate_insert_delete_preserves_remaining_mass() {
        let mut store = PointStore::new(1, 1, 8, false);
        let first = store.add(&[1.0]).unwrap();
        let duplicate = store.add(&[1.0]).unwrap();
        let mut tree = RcfTree::new(1, 8, 7);

        tree.insert(first, &store).unwrap();
        tree.insert(duplicate, &store).unwrap();
        assert_eq!(tree.tree_mass, 2);

        tree.delete(duplicate, &store).unwrap();
        assert_eq!(tree.tree_mass, 1);
        assert!(!tree.is_empty());
    }

    #[rstest]
    #[case(vec![100.0f32, 0.0])]
    #[case(vec![-50.0f32, -50.0])]
    #[case(vec![0.0f32, 200.0])]
    fn score_outlier_higher_than_inlier(#[case] outlier: Vec<f32>) {
        // Build a dense cluster, then score a far outlier vs a center point.
        let mut pts: Vec<Vec<f32>> = (0..50).map(|_| vec![0.5f32, 0.5]).collect();
        pts.push(vec![0.4, 0.6]); // slight inlier
        let (store, tree) = make_store_and_tree(&pts);

        let mode = ScoreMode::standard();
        let inlier_score = tree.raw_score(&[0.5, 0.5], &store, &mode);
        let outlier_score = tree.raw_score(&outlier, &store, &mode);
        assert!(
            outlier_score > inlier_score,
            "outlier={outlier_score} inlier={inlier_score}"
        );
    }

    #[test]
    fn empty_tree_returns_zero_score() {
        let store = PointStore::new(2, 1, 4, false);
        let tree = RcfTree::new(2, 4, 0);
        assert_abs_diff_eq!(
            tree.raw_score(&[1.0, 1.0], &store, &ScoreMode::standard()),
            0.0,
            epsilon = 1e-12
        );
    }

    #[rstest]
    #[case::val_below_cut(0.2f32, 0.3f32, 10usize, 20usize, 10usize, 20usize)]
    #[case::val_above_cut(0.4f32, 0.3f32, 10usize, 20usize, 20usize, 10usize)]
    #[case::val_equal_cut(0.3f32, 0.3f32, 10usize, 20usize, 10usize, 20usize)]
    fn split_children_respects_cut_direction(
        #[case] val: f32,
        #[case] cut: f32,
        #[case] a: usize,
        #[case] b: usize,
        #[case] expected_primary: usize,
        #[case] expected_secondary: usize,
    ) {
        let (primary, secondary) = split_children(val, cut, a, b);
        assert_eq!(primary, expected_primary);
        assert_eq!(secondary, expected_secondary);
        assert_eq!(split_child_only(val, cut, a, b), expected_primary);
    }

    #[rstest]
    #[case::mass_10(10, 0.1)]
    #[case::mass_40(40, 0.4)]
    #[case::mass_100(100, 1.0)]
    fn nn_threshold_scales_with_tree_mass(#[case] mass: usize, #[case] expected: f64) {
        assert_abs_diff_eq!(nn_threshold(mass), expected, epsilon = 1e-12);
    }

    #[test]
    fn near_neighbor_threshold_helpers_behave_as_expected() {
        let th = nn_threshold(40);

        assert!(should_descend_primary(0.41, 3, th));
        assert!(should_descend_primary(0.0, 0, th));
        assert!(!should_descend_primary(0.39, 2, th));

        assert!(should_descend_secondary(0.51));
        assert!(!should_descend_secondary(0.5));
    }

    #[test]
    fn consider_impute_candidate_prefers_nearer_at_full_centrality() {
        let best = NeighborCandidate {
            score: 1.0,
            point_idx: 10,
            distance: 2.0,
        };
        let nearer = NeighborCandidate {
            score: 1.0,
            point_idx: 11,
            distance: 1.0,
        };
        let farther = NeighborCandidate {
            score: 1.0,
            point_idx: 12,
            distance: 3.0,
        };
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(123);

        let chosen = consider_impute_candidate(Some(best.clone()), nearer, 1.0, &mut rng).unwrap();
        assert_eq!(chosen.point_idx, 11);

        let chosen = consider_impute_candidate(Some(best), farther, 1.0, &mut rng).unwrap();
        assert_eq!(chosen.point_idx, 10);
    }

    #[cfg(feature = "std")]
    mod proptest_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn raw_score_non_negative(
                x in -10f32..10f32,
                y in -10f32..10f32,
            ) {
                let pts: Vec<Vec<f32>> = vec![
                    vec![0.0, 0.0],
                    vec![1.0, 0.0],
                    vec![0.0, 1.0],
                    vec![1.0, 1.0],
                    vec![0.5, 0.5],
                ];
                let (store, tree) = make_store_and_tree(&pts);
                let score = tree.raw_score(&[x, y], &store, &ScoreMode::standard());
                prop_assert!(score >= 0.0, "score={score}");
            }

            #[test]
            fn insert_delete_restores_mass(n in 2usize..=16) {
                let pts: Vec<Vec<f32>> = (0..n).map(|i| vec![i as f32, 0.0]).collect();
                let dim = 2;
                let mut store = PointStore::new(dim, 1, 64, false);
                let mut tree = RcfTree::new(dim, 32, 42);
                let mut idxs = Vec::new();
                for p in &pts {
                    let idx = store.add(p).unwrap();
                    idxs.push(idx);
                    tree.insert(idx, &store).unwrap();
                }
                let mass_before = tree.tree_mass;
                let last_idx = idxs[n - 1];
                tree.delete(last_idx, &store).unwrap();
                prop_assert_eq!(tree.tree_mass, mass_before - 1);
            }
        }
    }
}
