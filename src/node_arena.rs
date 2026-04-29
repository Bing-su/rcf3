use crate::bounding_box::BoundingBox;
use serde::{Deserialize, Serialize};

pub const NULL: usize = usize::MAX;

// ---------------------------------------------------------------------------
// Node representation
// ---------------------------------------------------------------------------

/// A node in the random-cut tree.
///
/// Leaves track the point they contain and the number of duplicate arrivals.
/// Internal nodes track the split criterion and the tight bounding-box of
/// their entire sub-tree (kept up-to-date on every insert / delete).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Node {
    Leaf {
        /// Index into the forest-wide [`crate::point_store::PointStore`].
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
    pub fn mass(&self) -> usize {
        match self {
            Node::Leaf { mass, .. } => *mass,
            Node::Internal { mass, .. } => *mass,
        }
    }

    pub fn is_leaf(&self) -> bool {
        matches!(self, Node::Leaf { .. })
    }
}

// ---------------------------------------------------------------------------
// Arena allocator
// ---------------------------------------------------------------------------

/// Arena of [`Node`]s with O(1) alloc / free.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeArena {
    nodes: Vec<Option<Node>>,
    free: Vec<usize>,
}

impl NodeArena {
    pub fn new(initial_capacity: usize) -> Self {
        NodeArena {
            nodes: Vec::with_capacity(initial_capacity),
            free: Vec::new(),
        }
    }

    /// Allocate a new node and return its index.
    pub fn alloc(&mut self, node: Node) -> usize {
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
    pub fn get(&self, id: usize) -> &Node {
        self.nodes[id]
            .as_ref()
            .expect("accessed freed or uninitialized node")
    }

    /// Mutably return the node at `id`.
    pub fn get_mut(&mut self, id: usize) -> &mut Node {
        self.nodes[id]
            .as_mut()
            .expect("accessed freed or uninitialized node")
    }

    /// Free the node at `id`.
    pub fn free(&mut self, id: usize) {
        debug_assert!(id < self.nodes.len());
        self.nodes[id] = None;
        self.free.push(id);
    }

    /// Number of allocated slots (including freed ones).
    pub fn slot_count(&self) -> usize {
        self.nodes.len()
    }
}
