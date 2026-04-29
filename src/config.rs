use burn::prelude::*;

/// Hyperparameters for a Random Cut Forest.
///
/// Use [`RcfConfig::new`] then chain the builder methods, or deserialise from JSON.
#[derive(Config, Debug)]
pub struct RcfConfig {
    /// Number of base feature dimensions per observation (before shingling).
    pub input_dim: usize,

    /// Temporal window size. When `internal_shingling` is true the forest
    /// maintains a rolling buffer and the effective model dimension is
    /// `input_dim * shingle_size`.
    #[config(default = 1)]
    pub shingle_size: usize,

    /// Maximum number of points stored per tree.
    #[config(default = 256)]
    pub capacity: usize,

    /// Number of trees in the forest.
    #[config(default = 50)]
    pub num_trees: usize,

    /// Exponential time-decay rate applied to sampling weights.
    /// `0.0` means "use the default `0.1 / capacity`".
    #[config(default = 0.0)]
    pub time_decay: f64,

    /// Minimum number of updates before `score` / `attribution` / etc. return
    /// non-trivial results.  `0` means "use `1 + capacity / 4`".
    #[config(default = 0)]
    pub output_after: usize,

    /// When true the forest manages the shingle buffer automatically so callers
    /// pass one base observation at a time.
    #[config(default = true)]
    pub internal_shingling: bool,

    /// Controls how quickly the sampler fills to capacity during warm-up.
    #[config(default = 0.125)]
    pub initial_accept_fraction: f64,
}

impl RcfConfig {
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
