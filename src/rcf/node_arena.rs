#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::bounding_box::BoundingBox;

pub(super) const NULL: usize = usize::MAX;

// ---------------------------------------------------------------------------
// Node representation
// ---------------------------------------------------------------------------

/// A node in the random-cut tree.
///
/// Leaves track the point they contain and the number of duplicate arrivals.
/// Internal nodes track the split criterion and the tight bounding-box of
/// their entire sub-tree (kept up-to-date on every insert / delete).
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub(super) enum Node {
    Leaf {
        /// Index into the forest-wide [`crate::rcf::point_store::PointStore`].
        point_idx: usize,
        /// Number of identical (duplicate) points at this leaf.
        mass: usize,
    },
    Internal {
        left: usize,
        right: usize,
        cut_dim: usize,
        cut_val: f32,
        /// Total number of point-observations in this sub-tree (counting duplicates).
        mass: usize,
        bbox: BoundingBox,
    },
}

impl Node {
    pub(super) fn mass(&self) -> usize {
        match self {
            Node::Leaf { mass, .. } => *mass,
            Node::Internal { mass, .. } => *mass,
        }
    }
}

// ---------------------------------------------------------------------------
// Arena allocator
// ---------------------------------------------------------------------------

/// Arena of [`Node`]s with O(1) alloc / free.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub(super) struct NodeArena {
    nodes: Vec<Option<Node>>,
    free: Vec<usize>,
}

impl NodeArena {
    pub(super) fn new(initial_capacity: usize) -> Self {
        NodeArena {
            nodes: Vec::with_capacity(initial_capacity),
            free: Vec::new(),
        }
    }

    /// Allocate a new node and return its index.
    pub(super) fn alloc(&mut self, node: Node) -> usize {
        if let Some(id) = self.free.pop() {
            self.nodes[id] = Some(node);
            id
        } else {
            let id = self.nodes.len();
            self.nodes.push(Some(node));
            id
        }
    }

    /// Return the node at `id`.  Panics if `id` is NULL or freed.
    pub(super) fn get(&self, id: usize) -> &Node {
        self.nodes[id]
            .as_ref()
            .expect("accessed freed or uninitialized node")
    }

    /// Mutably return the node at `id`.
    pub(super) fn get_mut(&mut self, id: usize) -> &mut Node {
        self.nodes[id]
            .as_mut()
            .expect("accessed freed or uninitialized node")
    }

    /// Free the node at `id`.
    pub(super) fn free(&mut self, id: usize) {
        debug_assert!(id < self.nodes.len());
        self.nodes[id] = None;
        self.free.push(id);
    }

    /// Number of allocated slots (including freed ones).
    #[cfg(all(test, feature = "std"))]
    fn slot_count(&self) -> usize {
        self.nodes.len()
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn leaf_node() -> Node {
        Node::Leaf {
            point_idx: 0,
            mass: 1,
        }
    }

    proptest! {
        #[test]
        fn slot_count_increases_with_alloc(n in 1usize..=32) {
            let mut arena = NodeArena::new(4);
            let before = arena.slot_count();
            for _ in 0..n {
                arena.alloc(leaf_node());
            }
            prop_assert!(arena.slot_count() > before);
        }

        #[test]
        fn alloc_then_free_restores_slot(n in 1usize..=16) {
            let mut arena = NodeArena::new(n);
            let ids: Vec<usize> = (0..n).map(|_| arena.alloc(leaf_node())).collect();
            let count_after_alloc = arena.slot_count();
            for id in ids {
                arena.free(id);
            }
            // Re-alloc should reuse freed slots, not grow beyond count_after_alloc
            for _ in 0..n {
                arena.alloc(leaf_node());
            }
            prop_assert!(arena.slot_count() <= count_after_alloc + 1);
        }
    }
}
