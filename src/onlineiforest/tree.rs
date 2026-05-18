#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, vec::Vec};

use rand::prelude::*;
use rand::rngs::Xoshiro256PlusPlus;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::node::{Node, Split, Support};

fn split_threshold(max_leaf_samples: usize, depth: usize) -> usize {
    max_leaf_samples.saturating_mul(1usize.checked_shl(depth as u32).unwrap_or(usize::MAX))
}

fn should_split(height: usize, max_leaf_samples: usize, depth: usize, depth_limit: f64) -> bool {
    height >= split_threshold(max_leaf_samples, depth) && (depth as f64) < depth_limit
}

fn residual_path_length(height: usize, max_leaf_samples: usize) -> f64 {
    if height < max_leaf_samples {
        0.0
    } else {
        libm::log2(height as f64 / max_leaf_samples as f64)
    }
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub(crate) struct OnlineITree {
    root: Option<Node>,
    rng: Xoshiro256PlusPlus,
}

impl OnlineITree {
    pub(crate) fn new(seed: u64) -> Self {
        Self {
            root: None,
            rng: Xoshiro256PlusPlus::seed_from_u64(seed),
        }
    }

    pub(crate) fn learn(&mut self, point: &[f32], max_leaf_samples: usize, depth_limit: f64) {
        match self.root.as_mut() {
            Some(root) => {
                Self::learn_node(root, point, max_leaf_samples, depth_limit, 0, &mut self.rng)
            }
            None => self.root = Some(Node::new(1, Support::from_point(point))),
        }
    }

    pub(crate) fn forget(&mut self, point: &[f32], max_leaf_samples: usize) {
        let Some(root) = self.root.as_mut() else {
            return;
        };
        Self::forget_node(root, point, max_leaf_samples, 0);
        if root.height == 0 {
            self.root = None;
        }
    }

    pub(crate) fn point_depth(&self, point: &[f32], max_leaf_samples: usize) -> f64 {
        self.root
            .as_ref()
            .map(|root| Self::point_depth_node(root, point, max_leaf_samples, 0))
            .unwrap_or(0.0)
    }

    fn learn_node(
        node: &mut Node,
        point: &[f32],
        max_leaf_samples: usize,
        depth_limit: f64,
        depth: usize,
        rng: &mut Xoshiro256PlusPlus,
    ) {
        node.height = node.height.saturating_add(1);
        node.support.expand(point);

        if node.is_leaf() {
            if should_split(node.height, max_leaf_samples, depth, depth_limit) {
                Self::split_leaf(node, depth, rng);
            }
            return;
        }

        let split = node.split.as_mut().expect("checked non-leaf");
        let child = if point[split.dimension] < split.value {
            &mut split.left
        } else {
            &mut split.right
        };
        Self::learn_node(child, point, max_leaf_samples, depth_limit, depth + 1, rng);
    }

    fn split_leaf(node: &mut Node, depth: usize, rng: &mut Xoshiro256PlusPlus) {
        let Some((dimension, value)) = node.support.sample_split(rng) else {
            return;
        };

        let mut left_points = Vec::new();
        let mut right_points = Vec::new();
        // New child bins are initialized from synthetic samples drawn from the
        // parent support, exactly as the paper's piecewise-uniform approximation
        // prescribes; they are not reconstructed from historical observations.
        for _ in 0..node.height {
            let point = node.support.sample_point(rng);
            if point[dimension] < value {
                left_points.push(point);
            } else {
                right_points.push(point);
            }
        }

        let (left_region, right_region) = node.support.split_regions(dimension, value);
        // The paper leaves the empty-partition edge case implicit. Preserve a
        // geometric half-region when one synthetic side gets no samples so the
        // newborn child still has a valid support rectangle.
        let left_support = Support::from_points(&left_points).unwrap_or(left_region);
        let right_support = Support::from_points(&right_points).unwrap_or(right_region);

        debug_assert!(depth < usize::MAX);
        node.split = Some(Split {
            dimension,
            value,
            left: Box::new(Node::new(left_points.len(), left_support)),
            right: Box::new(Node::new(right_points.len(), right_support)),
        });
    }

    fn forget_node(node: &mut Node, point: &[f32], max_leaf_samples: usize, depth: usize) {
        node.height = node.height.saturating_sub(1);

        let Some(split) = node.split.as_mut() else {
            return;
        };

        if node.height < split_threshold(max_leaf_samples, depth) {
            node.support = Support::merged(&split.left.support, &split.right.support);
            node.split = None;
            return;
        }

        let child = if point[split.dimension] < split.value {
            &mut split.left
        } else {
            &mut split.right
        };
        Self::forget_node(child, point, max_leaf_samples, depth + 1);
    }

    fn point_depth_node(node: &Node, point: &[f32], max_leaf_samples: usize, depth: usize) -> f64 {
        let Some(split) = node.split.as_ref() else {
            return depth as f64 + residual_path_length(node.height, max_leaf_samples);
        };

        let child = if point[split.dimension] < split.value {
            &split.left
        } else {
            &split.right
        };
        Self::point_depth_node(child, point, max_leaf_samples, depth + 1)
    }

    #[cfg(test)]
    pub(crate) fn root(&self) -> Option<&Node> {
        self.root.as_ref()
    }

    #[cfg(test)]
    pub(crate) fn root_height(&self) -> usize {
        self.root.as_ref().map_or(0, |node| node.height)
    }

    #[cfg(test)]
    pub(crate) fn supports_are_nested(&self) -> bool {
        fn check(node: &Node) -> bool {
            match &node.split {
                Some(split) => {
                    node.support.contains_support(&split.left.support)
                        && node.support.contains_support(&split.right.support)
                        && check(&split.left)
                        && check(&split.right)
                }
                None => true,
            }
        }
        self.root.as_ref().map(check).unwrap_or(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn residual_depth_matches_expected() {
        assert_eq!(residual_path_length(3, 4), 0.0);
        assert_eq!(residual_path_length(4, 4), 0.0);
        assert_eq!(residual_path_length(8, 4), 1.0);
    }

    #[test]
    fn fractional_depth_limit_allows_all_integer_depths_below_it() {
        assert!(should_split(4, 2, 1, 1.5));
        assert!(!should_split(8, 2, 2, 1.5));
    }

    #[test]
    fn seeded_splits_are_reproducible() {
        let mut left = OnlineITree::new(7);
        let mut right = OnlineITree::new(7);
        let points = [[0.0], [1.0], [2.0], [3.0]];
        for point in points {
            left.learn(&point, 2, 4.0);
            right.learn(&point, 2, 4.0);
        }
        let left_root = left.root().unwrap();
        let right_root = right.root().unwrap();
        assert_eq!(
            left_root.split.as_ref().unwrap().dimension,
            right_root.split.as_ref().unwrap().dimension
        );
        assert_eq!(
            left_root.split.as_ref().unwrap().value,
            right_root.split.as_ref().unwrap().value
        );
    }

    #[test]
    fn forgetting_collapses_split_below_threshold() {
        let mut tree = OnlineITree::new(11);
        tree.learn(&[0.0], 2, 4.0);
        tree.learn(&[1.0], 2, 4.0);
        assert!(tree.root().unwrap().split.is_some());
        tree.forget(&[0.0], 2);
        assert!(tree.root().unwrap().split.is_none());
    }

    #[test]
    fn repeated_identical_points_do_not_attempt_degenerate_split() {
        let mut tree = OnlineITree::new(17);
        for _ in 0..8 {
            tree.learn(&[1.0, 1.0], 2, 4.0);
        }
        assert!(tree.root().unwrap().split.is_none());
    }
}
