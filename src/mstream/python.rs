#![cfg(feature = "python")]

use std::path::PathBuf;

use pyo3::prelude::*;

use super::{MStream, MStreamScore};
use crate::pyutil::{StrOrBytes, to_py_err};

#[derive(IntoPyObject)]
struct PyMStreamScore {
    total: f64,
    record: f64,
    numeric_features: Vec<f64>,
    categorical_features: Vec<f64>,
}

impl From<MStreamScore> for PyMStreamScore {
    fn from(value: MStreamScore) -> Self {
        Self {
            total: value.total,
            record: value.record,
            numeric_features: value.numeric_features,
            categorical_features: value.categorical_features,
        }
    }
}

/// mStream detector for mixed numerical/categorical records.
///
/// `timestamp` is interpreted as a logical time tick, not as wall-clock time.
/// Scores are invariant to adding a constant offset to all timestamps, while a
/// gap of `k` ticks applies the temporal decay factor `alpha` exactly `k`
/// times.
///
/// Parameters
/// ----------
/// numeric_dim : int
///     Number of numerical features in each record.
/// categorical_dim : int
///     Number of categorical features in each record.
/// num_rows : int, optional
///     Number of hash rows (default 2).
/// num_buckets : int, optional
///     Number of buckets per hash row (default 1024).
/// alpha : float, optional
///     Temporal decay factor in `(0, 1)` (default 0.8).
/// seed : int, optional
///     Random seed for deterministic hashing.
#[pyclass(name = "MStream", module = "rcf3.rcf3", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyMStream {
    inner: MStream,
}

#[pymethods]
impl PyMStream {
    #[new]
    #[pyo3(signature = (
        numeric_dim,
        categorical_dim,
        num_rows = 2,
        num_buckets = 1024,
        alpha = 0.8,
        seed = None
    ))]
    fn py_new(
        numeric_dim: usize,
        categorical_dim: usize,
        num_rows: usize,
        num_buckets: usize,
        alpha: f64,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let mut builder = MStream::builder(numeric_dim, categorical_dim)
            .num_rows(num_rows)
            .num_buckets(num_buckets)
            .alpha(alpha);
        if let Some(seed) = seed {
            builder = builder.seed(seed);
        }
        let inner = builder.build().map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// Ingest a record without returning its score.
    fn update(&mut self, numeric: Vec<f32>, categorical: Vec<i64>, timestamp: u64) -> PyResult<()> {
        self.inner
            .update(&numeric, &categorical, timestamp)
            .map_err(to_py_err)
    }

    /// Preview the anomaly score for a record without mutating detector state.
    ///
    /// The preview answers what this record would score if it were ingested
    /// next, using the same timestamp semantics as `update_and_score()`.
    fn score(&self, numeric: Vec<f32>, categorical: Vec<i64>, timestamp: u64) -> PyResult<f64> {
        self.inner
            .score(&numeric, &categorical, timestamp)
            .map_err(to_py_err)
    }

    /// Ingest a record and return its anomaly score.
    ///
    /// `timestamp` must be a monotonically non-decreasing tick index. Only tick
    /// differences matter: shifting all timestamps by the same constant does
    /// not change the scores.
    fn update_and_score(
        &mut self,
        numeric: Vec<f32>,
        categorical: Vec<i64>,
        timestamp: u64,
    ) -> PyResult<f64> {
        self.inner
            .update_and_score(&numeric, &categorical, timestamp)
            .map_err(to_py_err)
    }

    /// Preview the decomposed anomaly score without mutating detector state.
    fn score_detailed(
        &self,
        numeric: Vec<f32>,
        categorical: Vec<i64>,
        timestamp: u64,
    ) -> PyResult<PyMStreamScore> {
        self.inner
            .score_detailed(&numeric, &categorical, timestamp)
            .map(PyMStreamScore::from)
            .map_err(to_py_err)
    }

    /// Ingest a record and return the decomposed score used to form the final anomaly score.
    fn update_and_score_detailed(
        &mut self,
        numeric: Vec<f32>,
        categorical: Vec<i64>,
        timestamp: u64,
    ) -> PyResult<PyMStreamScore> {
        self.inner
            .update_and_score_detailed(&numeric, &categorical, timestamp)
            .map(PyMStreamScore::from)
            .map_err(to_py_err)
    }

    /// Return True once the detector has processed at least one record.
    fn is_ready(&self) -> bool {
        self.inner.is_ready()
    }

    /// Return the number of processed records.
    fn entries_seen(&self) -> u64 {
        self.inner.entries_seen()
    }

    /// Return the last timestamp observed by the detector.
    fn current_time(&self) -> Option<u64> {
        self.inner.current_time()
    }

    /// Serialize the detector state to JSON.
    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(to_py_err)
    }

    /// Deserialize detector state from a JSON string previously written by `to_json()`.
    #[staticmethod]
    fn from_json(json: StrOrBytes) -> PyResult<Self> {
        let inner = MStream::from_json(json).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// Serialize the detector state to a JSON file.
    fn save_json(&self, path: PathBuf) -> PyResult<()> {
        self.inner.save_json(path).map_err(to_py_err)
    }

    /// Deserialize detector state from a JSON file previously written by `save_json()`.
    #[staticmethod]
    fn load_json(path: PathBuf) -> PyResult<Self> {
        let inner = MStream::load_json(path).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    // ---------------------------------------------------------------------------
    // Python Magic Methods
    // ---------------------------------------------------------------------------

    fn __repr__(&self) -> String {
        let c = self.inner.config();
        format!(
            "MStream(numeric_dim={}, categorical_dim={}, num_rows={}, num_buckets={}, alpha={}, entries_seen={})",
            c.numeric_dim(),
            c.categorical_dim(),
            c.num_rows(),
            c.num_buckets(),
            c.alpha(),
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

    fn __getnewargs__(&self) -> (usize, usize) {
        let c = self.inner.config();
        (c.numeric_dim(), c.categorical_dim())
    }
}
