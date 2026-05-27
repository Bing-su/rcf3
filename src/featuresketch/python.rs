#![cfg(feature = "python")]

use std::path::PathBuf;

use pyo3::prelude::*;

use super::FeatureSketch;
use crate::pyutil::{KeyValueLike, StrOrBytes, to_py_err};

/// FeatureSketch detector for sparse, schema-evolving feature streams.
///
/// Feature events can be passed as either a mapping from feature name to value
/// or as a sequence of `(name, value)` pairs. Duplicate names are combined
/// before scoring, and values must be finite.
///
/// Parameters
/// ----------
/// value_projection_dims : int, optional
///     Number of random projection dimensions for feature values (default 32).
/// presence_projection_dims : int, optional
///     Number of random projection dimensions for feature presence (default 32).
/// chains_per_ensemble : int, optional
///     Number of chains in each sketch ensemble (default 16).
/// chain_depth : int, optional
///     Number of bins traversed by each chain (default 8).
/// sketch_rows : int, optional
///     Number of hash rows in each count-min sketch (default 2).
/// sketch_buckets : int, optional
///     Number of buckets per sketch row (default 2048).
/// decay_half_life : int, optional
///     Event-count half-life used for temporal decay (default 2048).
/// seed : int, optional
///     Random seed for deterministic projections, chains, and sketches.
#[pyclass(name = "FeatureSketch", module = "rcf3.rcf3", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyFeatureSketch {
    inner: FeatureSketch,
}

#[pymethods]
impl PyFeatureSketch {
    #[new]
    #[pyo3(signature = (
        value_projection_dims = 32,
        presence_projection_dims = 32,
        chains_per_ensemble = 16,
        chain_depth = 8,
        sketch_rows = 2,
        sketch_buckets = 2048,
        decay_half_life = 2048,
        seed = None
    ))]
    fn py_new(
        value_projection_dims: usize,
        presence_projection_dims: usize,
        chains_per_ensemble: usize,
        chain_depth: usize,
        sketch_rows: usize,
        sketch_buckets: usize,
        decay_half_life: u64,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let mut builder = FeatureSketch::builder()
            .value_projection_dims(value_projection_dims)
            .presence_projection_dims(presence_projection_dims)
            .chains_per_ensemble(chains_per_ensemble)
            .chain_depth(chain_depth)
            .sketch_rows(sketch_rows)
            .sketch_buckets(sketch_buckets)
            .decay_half_life(decay_half_life);
        if let Some(seed) = seed {
            builder = builder.seed(seed);
        }
        let inner = builder.build().map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// Ingest a feature event without returning its score.
    fn update(&mut self, feature: KeyValueLike) -> PyResult<()> {
        self.inner.update(feature).map_err(to_py_err)
    }

    /// Preview the current anomaly score for a feature event without mutating state.
    ///
    /// This is the same pre-ingest score that `update_and_score()` would return
    /// if called next. It is computed against the current sketches and does not
    /// advance the decay epoch or `entries_seen()`.
    fn score(&self, feature: KeyValueLike) -> PyResult<f64> {
        self.inner.score(feature).map_err(to_py_err)
    }

    /// Return the current anomaly score for a feature event, then ingest it.
    fn update_and_score(&mut self, feature: KeyValueLike) -> PyResult<f64> {
        self.inner.update_and_score(feature).map_err(to_py_err)
    }

    /// Return True once the detector has processed at least one event.
    fn is_ready(&self) -> bool {
        self.inner.is_ready()
    }

    /// Return the number of processed events.
    fn entries_seen(&self) -> u64 {
        self.inner.entries_seen()
    }

    /// Serialize the detector state to JSON.
    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(to_py_err)
    }

    /// Deserialize detector state from a JSON string previously written by `to_json()`.
    #[staticmethod]
    fn from_json(json: StrOrBytes) -> PyResult<Self> {
        let inner = FeatureSketch::from_json(json).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// Serialize the detector state to a JSON file.
    fn save_json(&self, path: PathBuf) -> PyResult<()> {
        self.inner.save_json(path).map_err(to_py_err)
    }

    /// Deserialize detector state from a JSON file previously written by `save_json()`.
    #[staticmethod]
    fn load_json(path: PathBuf) -> PyResult<Self> {
        let inner = FeatureSketch::load_json(path).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    // ---------------------------------------------------------------------------
    // Python Magic Methods
    // ---------------------------------------------------------------------------

    fn __repr__(&self) -> String {
        let c = self.inner.config();
        format!(
            "FeatureSketch(value_projection_dims={}, presence_projection_dims={}, chains_per_ensemble={}, chain_depth={}, sketch_rows={}, sketch_buckets={}, decay_half_life={}, entries_seen={})",
            c.value_projection_dims(),
            c.presence_projection_dims(),
            c.chains_per_ensemble(),
            c.chain_depth(),
            c.sketch_rows(),
            c.sketch_buckets(),
            c.decay_half_life(),
            self.inner.entries_seen(),
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    fn __copy__(&self) -> Self {
        self.clone()
    }

    #[allow(unused_variables)]
    fn __deepcopy__<'py>(&self, memo: Bound<'py, PyAny>) -> Self {
        self.clone()
    }

    fn __getstate__(&self) -> PyResult<String> {
        self.to_json()
    }

    fn __setstate__(&mut self, state: String) -> PyResult<()> {
        let new = Self::from_json(state.into())?;
        *self = new;
        Ok(())
    }
}
