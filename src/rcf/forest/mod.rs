#[cfg(all(not(feature = "std"), feature = "serde"))]
use alloc::format;
#[cfg(all(not(feature = "std"), feature = "serde"))]
use alloc::string::String;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::fmt;

use rand::prelude::*;
use rand::rngs::Xoshiro256PlusPlus;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::{config::RcfConfig, point_store::PointStore, sampler::Sampler, tree::RcfTree};
#[cfg(feature = "serde")]
use crate::error::RcfError;
use crate::error::Result;

mod impute;
mod query;
mod update;

/// Intermediate candidate collected from a single tree during near-neighbour search.
///
/// Gathered per-tree inside [`Forest::near_neighbors`] and then aggregated
/// and converted into [`NeighborResult`].
#[derive(Clone, Debug, PartialEq)]
pub(in crate::rcf) struct NeighborCandidate {
    /// Anomaly score of this candidate point.
    pub(in crate::rcf) score: f64,
    /// Index of the point in the [`PointStore`].
    pub(in crate::rcf) point_idx: usize,
    /// L1 distance to the query point.
    pub(in crate::rcf) distance: f64,
}

/// Per-tree update decision kept from sampler acceptance through sampler finalization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct AcceptedUpdate {
    pub(super) tree_index: usize,
    pub(super) evicted_point: Option<usize>,
    pub(super) weight: f64,
}

impl From<(f64, usize, f64)> for NeighborCandidate {
    fn from(value: (f64, usize, f64)) -> Self {
        Self {
            score: value.0,
            point_idx: value.1,
            distance: value.2,
        }
    }
}

impl From<NeighborCandidate> for (f64, usize, f64) {
    fn from(value: NeighborCandidate) -> Self {
        (value.score, value.point_idx, value.distance)
    }
}

/// A near-neighbour result returned by [`Forest::near_neighbors`].
///
/// Candidates collected across all trees are deduplicated and
/// aggregated by point index, then returned sorted by distance (ascending).
#[derive(Clone, Debug, PartialEq)]
pub struct NeighborResult {
    /// Anomaly score of this point.
    pub score: f64,
    /// Coordinate vector of the point (length = `input_dim * shingle_size`).
    pub point: Vec<f32>,
    /// L1 distance to the query point.
    pub distance: f64,
}

impl From<(f64, Vec<f32>, f64)> for NeighborResult {
    fn from(value: (f64, Vec<f32>, f64)) -> Self {
        Self {
            score: value.0,
            point: value.1,
            distance: value.2,
        }
    }
}

impl From<NeighborResult> for (f64, Vec<f32>, f64) {
    fn from(value: NeighborResult) -> Self {
        (value.score, value.point, value.distance)
    }
}

// ---------------------------------------------------------------------------
// Forest
// ---------------------------------------------------------------------------

/// A Random Cut Forest: an ensemble of random-cut trees sharing point storage.
///
/// # Typical usage
/// ```ignore
/// let mut forest = Forest::builder(2)
///     .shingle_size(1)
///     .num_trees(50)
///     .capacity(256)
///     .build()?;
/// for point in stream {
///     forest.update(&point)?;
///     if forest.is_ready() {
///         let score = forest.score(&point)?;
///     }
/// }
/// ```
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Forest {
    config: RcfConfig,
    trees: Vec<RcfTree>,
    samplers: Vec<Sampler>,
    point_store: PointStore,
    entries_seen: u64,
    rng: Xoshiro256PlusPlus,
    #[cfg_attr(feature = "serde", serde(skip, default))]
    accepted_updates: Vec<AcceptedUpdate>,
    #[cfg_attr(feature = "serde", serde(skip, default))]
    staged_accepted_updates: Vec<AcceptedUpdate>,
}

impl fmt::Debug for Forest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Forest")
            .field("config", &self.config)
            .field("trees", &self.trees)
            .field("samplers", &self.samplers)
            .field("point_store", &self.point_store)
            .field("entries_seen", &self.entries_seen)
            .field("rng", &self.rng)
            .finish()
    }
}

impl Forest {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    fn new_internal(config: RcfConfig, seed: u64) -> Result<Self> {
        config.validate()?;

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let dim = config.dim();
        let capacity = config.capacity();
        let num_trees = config.num_trees();

        let store_capacity = config.point_store_capacity();

        let trees: Vec<RcfTree> = (0..num_trees)
            .map(|_| RcfTree::new(dim, capacity, rng.next_u64()))
            .collect();

        let samplers: Vec<Sampler> = (0..num_trees).map(|_| Sampler::new(capacity)).collect();

        let point_store = PointStore::new(
            config.input_dim(),
            config.shingle_size(),
            store_capacity,
            config.internal_shingling(),
        );

        Ok(Forest {
            config,
            trees,
            samplers,
            point_store,
            entries_seen: 0,
            rng: Xoshiro256PlusPlus::seed_from_u64(rng.next_u64()),
            accepted_updates: Vec::with_capacity(num_trees),
            staged_accepted_updates: Vec::with_capacity(num_trees),
        })
    }

    /// Create a forest from a [`RcfConfig`] with a random seed.
    pub fn from_config(config: &RcfConfig) -> Result<Self> {
        let mut seed_rng: Xoshiro256PlusPlus = rand::make_rng();
        Self::new_internal(config.clone(), seed_rng.next_u64())
    }

    /// Create a forest from a [`RcfConfig`] with a deterministic seed.
    pub fn from_config_seeded(config: &RcfConfig, seed: u64) -> Result<Self> {
        Self::new_internal(config.clone(), seed)
    }

    // -----------------------------------------------------------------------
    // Builder
    // -----------------------------------------------------------------------

    /// Convenience entry point.
    pub fn builder(input_dim: usize) -> ForestBuilder {
        ForestBuilder::new(input_dim)
    }
    // -----------------------------------------------------------------------
    // Readiness
    // -----------------------------------------------------------------------

    /// Returns `true` once enough observations have been processed that the
    /// scoring functions return meaningful values.
    pub fn is_ready(&self) -> bool {
        let needed = self.config.effective_output_after()
            + if self.config.internal_shingling() {
                self.config.shingle_size().saturating_sub(1)
            } else {
                0
            };
        self.entries_seen as usize > needed
    }
    // -----------------------------------------------------------------------
    // Save / Load
    // -----------------------------------------------------------------------

    /// Serialize the entire forest state to a JSON string.
    #[cfg(feature = "serde")]
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self)
            .map_err(|e| RcfError::Runtime(format!("failed to serialize forest: {e}")))
    }

    /// Deserialize a forest from a JSON string previously written by
    /// [`Self::to_json`].
    #[cfg(feature = "serde")]
    pub fn from_json(json: impl AsRef<[u8]>) -> Result<Self> {
        serde_json::from_slice(json.as_ref())
            .map_err(|e| RcfError::InvalidArgument(format!("invalid forest JSON: {e}")))
    }

    /// Serialize the entire forest state to a JSON file.
    #[cfg(all(feature = "serde", feature = "std"))]
    pub fn save_json(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        let json = self.to_json()?;
        std::fs::write(path.as_ref(), json).map_err(|e| RcfError::Io(e.to_string()))
    }

    /// Deserialize a forest from a JSON file previously written by
    /// [`Self::save_json`].
    #[cfg(all(feature = "serde", feature = "std"))]
    pub fn load_json(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let data = std::fs::read(path.as_ref()).map_err(|e| RcfError::Io(e.to_string()))?;
        Self::from_json(&data)
    }

    // -----------------------------------------------------------------------
    // Diagnostics
    // -----------------------------------------------------------------------

    /// Number of observations processed so far.
    pub fn entries_seen(&self) -> u64 {
        self.entries_seen
    }

    /// Number of trees in the ensemble.
    pub fn num_trees(&self) -> usize {
        self.trees.len()
    }

    pub fn config(&self) -> &RcfConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// ForestBuilder
// ---------------------------------------------------------------------------

/// Fluent builder for constructing a [`Forest`] with custom hyperparameters.
pub struct ForestBuilder {
    config: RcfConfig,
    seed: Option<u64>,
}

impl ForestBuilder {
    /// Create a builder for the given base dimension.
    pub fn new(input_dim: usize) -> Self {
        let config = RcfConfig::new(input_dim);
        ForestBuilder { config, seed: None }
    }

    /// Set the shingle size.
    pub fn shingle_size(mut self, n: usize) -> Self {
        self.config = self.config.with_shingle_size(n);
        self
    }

    /// Set the number of trees in the ensemble.
    pub fn num_trees(mut self, n: usize) -> Self {
        self.config = self.config.with_num_trees(n);
        self
    }

    /// Set the maximum number of points retained per tree.
    pub fn capacity(mut self, c: usize) -> Self {
        self.config = self.config.with_capacity(c);
        self
    }

    /// Set the finite non-negative exponential time-decay rate for sampling
    /// weights.
    ///
    /// Use `0.0` to keep the default behavior (`0.1 / capacity`).
    pub fn time_decay(mut self, d: f64) -> Self {
        self.config = self.config.with_time_decay(d);
        self
    }

    /// Set the minimum updates before non-trivial scores are returned.
    pub fn output_after(mut self, n: usize) -> Self {
        self.config = self.config.with_output_after(n);
        self
    }

    /// Enable or disable internal shingle buffer management.
    pub fn internal_shingling(mut self, v: bool) -> Self {
        self.config = self.config.with_internal_shingling(v);
        self
    }

    /// Set the finite warm-up acceptance fraction for the sampler.
    ///
    /// Must be in `[0.0, 1.0]`; lower values throttle acceptance more while
    /// each tree sampler is below capacity.
    pub fn initial_accept_fraction(mut self, f: f64) -> Self {
        self.config = self.config.with_initial_accept_fraction(f);
        self
    }

    /// Set a deterministic seed for all trees in the forest.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = Some(s);
        self
    }

    /// Build a forest from the accumulated configuration.
    pub fn build(self) -> Result<Forest> {
        match self.seed {
            Some(s) => Forest::from_config_seeded(&self.config, s),
            None => Forest::from_config(&self.config),
        }
    }
}
#[cfg(test)]
mod tests;
