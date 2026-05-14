#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Hyperparameters for a Random Cut Forest.
///
/// Use [`RcfConfig::new`] then chain the builder methods, or deserialise from JSON.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RcfConfig {
    /// Number of base feature dimensions per observation (before shingling).
    pub input_dim: usize,

    /// Temporal window size. When `internal_shingling` is true the forest
    /// maintains a rolling buffer and the effective model dimension is
    /// `input_dim * shingle_size`.
    #[cfg_attr(feature = "serde", serde(default = "default_shingle_size"))]
    pub shingle_size: usize,

    /// Maximum number of points stored per tree.
    #[cfg_attr(feature = "serde", serde(default = "default_capacity"))]
    pub capacity: usize,

    /// Number of trees in the forest.
    #[cfg_attr(feature = "serde", serde(default = "default_num_trees"))]
    pub num_trees: usize,

    /// Exponential time-decay rate applied to sampling weights.
    /// `0.0` means "use the default `0.1 / capacity`".
    #[cfg_attr(feature = "serde", serde(default))]
    pub time_decay: f64,

    /// Minimum number of updates before `score` / `attribution` / etc. return
    /// non-trivial results.  `0` means "use `1 + capacity / 4`".
    #[cfg_attr(feature = "serde", serde(default))]
    pub output_after: usize,

    /// When true the forest manages the shingle buffer automatically so callers
    /// pass one base observation at a time.
    #[cfg_attr(feature = "serde", serde(default = "default_internal_shingling"))]
    pub internal_shingling: bool,

    /// Controls how quickly the sampler fills to capacity during warm-up.
    #[cfg_attr(feature = "serde", serde(default = "default_initial_accept_fraction"))]
    pub initial_accept_fraction: f64,
}

fn default_shingle_size() -> usize {
    1
}
fn default_capacity() -> usize {
    256
}
fn default_num_trees() -> usize {
    50
}
fn default_internal_shingling() -> bool {
    true
}
fn default_initial_accept_fraction() -> f64 {
    0.125
}

impl RcfConfig {
    /// Create a config with defaults for all optional parameters.
    pub fn new(input_dim: usize) -> Self {
        Self {
            input_dim,
            shingle_size: default_shingle_size(),
            capacity: default_capacity(),
            num_trees: default_num_trees(),
            time_decay: 0.0,
            output_after: 0,
            internal_shingling: default_internal_shingling(),
            initial_accept_fraction: default_initial_accept_fraction(),
        }
    }

    /// Set the temporal window size. `1` disables shingling.
    pub fn with_shingle_size(mut self, v: usize) -> Self {
        self.shingle_size = v;
        self
    }

    /// Set the maximum number of points retained per tree.
    pub fn with_capacity(mut self, v: usize) -> Self {
        self.capacity = v;
        self
    }

    /// Set the number of trees in the forest ensemble.
    pub fn with_num_trees(mut self, v: usize) -> Self {
        self.num_trees = v;
        self
    }

    /// Set the exponential time-decay rate for sampling weights.
    ///
    /// Use `0.0` to keep the default behavior (`0.1 / capacity`).
    pub fn with_time_decay(mut self, v: f64) -> Self {
        self.time_decay = v;
        self
    }

    /// Set the minimum updates before non-trivial scores are returned.
    ///
    /// Use `0` to keep the default behavior (`1 + capacity / 4`).
    pub fn with_output_after(mut self, v: usize) -> Self {
        self.output_after = v;
        self
    }

    /// Enable or disable internal shingle buffer management.
    pub fn with_internal_shingling(mut self, v: bool) -> Self {
        self.internal_shingling = v;
        self
    }

    /// Set the warm-up acceptance fraction for the sampler.
    pub fn with_initial_accept_fraction(mut self, v: f64) -> Self {
        self.initial_accept_fraction = v;
        self
    }

    /// Effective time-decay (resolves the `0.0 → default` convention).
    pub fn effective_time_decay(&self) -> f64 {
        if self.time_decay == 0.0 {
            0.1 / self.capacity as f64
        } else {
            self.time_decay
        }
    }

    /// Effective output threshold (resolves the `0 → default` convention).
    pub fn effective_output_after(&self) -> usize {
        if self.output_after == 0 {
            1 + self.capacity / 4
        } else {
            self.output_after
        }
    }

    /// Full dimensionality seen by each tree.
    pub fn dim(&self) -> usize {
        self.input_dim * self.shingle_size
    }
}
