from collections.abc import Mapping, Sequence
from os import PathLike
from typing import Any, Final, Self, SupportsFloat, SupportsInt, TypedDict, final

__version__: Final[str]

_KeyValueLike = Mapping[str, SupportsFloat] | Sequence[tuple[str, SupportsFloat]]

@final
class _NeighborResult(TypedDict):
    score: float
    point: list[float]
    distance: float

@final
class _Attribution(TypedDict):
    below: float
    above: float

@final
class _MStreamScore(TypedDict):
    total: float
    record: float
    numeric_features: list[float]
    categorical_features: list[float]

@final
class FeatureSketch:
    """
    FeatureSketch detector for sparse, schema-evolving feature streams.

    Feature events can be passed as either a mapping from feature name to value
    or as a sequence of `(name, value)` pairs. Duplicate names are combined
    before scoring, and values must be finite.

    Parameters
    ----------
    value_projection_dims : int, optional
        Number of random projection dimensions for feature values (default 32).
    presence_projection_dims : int, optional
        Number of random projection dimensions for feature presence (default 32).
    chains_per_ensemble : int, optional
        Number of chains in each sketch ensemble (default 16).
    chain_depth : int, optional
        Number of bins traversed by each chain (default 8).
    sketch_rows : int, optional
        Number of hash rows in each count-min sketch (default 2).
    sketch_buckets : int, optional
        Number of buckets per sketch row (default 2048).
    decay_half_life : int, optional
        Event-count half-life used for temporal decay (default 2048).
    seed : int, optional
        Random seed for deterministic projections, chains, and sketches.
    """
    def __new__(
        cls,
        /,
        value_projection_dims: SupportsInt = 32,
        presence_projection_dims: SupportsInt = 32,
        chains_per_ensemble: SupportsInt = 16,
        chain_depth: SupportsInt = 8,
        sketch_rows: SupportsInt = 2,
        sketch_buckets: SupportsInt = 2048,
        decay_half_life: SupportsInt = 2048,
        seed: SupportsInt | None = None,
    ) -> Self: ...
    def update(self, /, feature: _KeyValueLike) -> None:
        "Ingest a feature event without returning its score."
    def update_and_score(self, /, feature: _KeyValueLike) -> float:
        "Return the current anomaly score for a feature event, then ingest it."
    def score(self, /, feature: _KeyValueLike) -> float:
        """
        Preview the current anomaly score for a feature event without mutating state.

        This is the same pre-ingest score that `update_and_score()` would return
        if called next. It is computed against the current sketches and does not
        advance the decay epoch or `entries_seen()`.
        """
    def is_ready(self, /) -> bool:
        "Return True once the detector has processed at least one event."
    def entries_seen(self, /) -> int:
        "Return the number of processed events."
    def to_json(self, /) -> str:
        "Serialize the detector state to JSON."
    @staticmethod
    def from_json(json: str | bytes | bytearray | memoryview) -> FeatureSketch:
        "Deserialize detector state from a JSON string previously written by `to_json()`."
    def save_json(self, /, path: str | PathLike[str]) -> None:
        "Serialize the detector state to a JSON file."
    @staticmethod
    def load_json(path: str | PathLike[str]) -> FeatureSketch:
        "Deserialize detector state from a JSON file previously written by `save_json()`."
    def __repr__(self, /) -> str: ...
    def __str__(self, /) -> str: ...
    def __copy__(self, /) -> Self: ...
    def __deepcopy__(self, /, memo: Any) -> Self: ...
    def __getstate__(self, /) -> str: ...
    def __setstate__(self, /, state: str) -> None: ...

@final
class Forest:
    """
    A Random Cut Forest: an ensemble of Random Cut Trees sharing point storage.

    Parameters
    ----------
    input_dim : int
        Number of features per observation (before shingling).
    shingle_size : int, optional
        Temporal window size (default 1, no shingling).
    num_trees : int, optional
        Number of trees in the ensemble (default 50).
    capacity : int, optional
        Maximum samples per tree (default 256).
    time_decay : float, optional
        Finite non-negative exponential decay for sample weights
        (default 0 = auto, computed as 0.1 / capacity).
    output_after : int, optional
        Minimum observations before scoring starts
        (default 0 = auto, computed as 1 + capacity // 4).
    internal_shingling : bool, optional
        When True, pass one base observation at a time and the forest
        maintains the rolling shingle buffer (default True).
    initial_accept_fraction : float, optional
        Finite value in [0.0, 1.0] controlling warm-up sampler acceptance
        before capacity (default 0.125).
    seed : int, optional
        Random seed for deterministic forests.
    """
    def __new__(
        cls,
        /,
        input_dim: SupportsInt,
        shingle_size: SupportsInt = 1,
        num_trees: SupportsInt = 50,
        capacity: SupportsInt = 256,
        time_decay: SupportsFloat = 0.0,
        output_after: SupportsInt = 0,
        internal_shingling: bool = True,
        initial_accept_fraction: SupportsFloat = 0.125,
        seed: SupportsInt | None = None,
    ) -> Self: ...
    def update(self, /, point: Sequence[SupportsFloat]) -> None:
        """
        Incorporate a new observation into the forest.

        When `internal_shingling` is True, pass one base observation of length
        `input_dim`. Otherwise pass the full shingled vector of length
        `input_dim * shingle_size`.
        """
    def score(self, /, point: Sequence[SupportsFloat]) -> float:
        "Anomaly score for `point`. Higher means more anomalous."
    def displacement_score(self, /, point: Sequence[SupportsFloat]) -> float:
        "Displacement-based anomaly score."
    def attribution(self, /, point: Sequence[SupportsFloat]) -> list[_Attribution]:
        """
        Per-dimension attribution of the anomaly score.

        Returns a list of length `input_dim * shingle_size`.
        """
    def density(self, /, point: Sequence[SupportsFloat]) -> float:
        "Density estimate at `point`. Higher means a denser neighbourhood."
    def near_neighbors(
        self,
        /,
        point: Sequence[SupportsFloat],
        top_k: SupportsInt = 10,
        percentile: SupportsInt = 50,
    ) -> list[_NeighborResult]:
        """
        Find approximate near-neighbours of `point`.

        `percentile` controls per-tree traversal aggressiveness in `[0, 100]`;
        lower values visit more branches and usually return more candidates.
        Returns a list sorted by distance, with duplicate points across trees
        merged by point index. At most `top_k` results are returned.
        """
    def impute(
        self,
        /,
        point: Sequence[SupportsFloat],
        missing: Sequence[SupportsInt],
        centrality: SupportsFloat = 1.0,
    ) -> list[float]:
        """
        Impute the `missing` positions of `point`.

        `point` must have the full dimensionality (`input_dim * shingle_size`).
        Values at `missing` indices are ignored; the returned list fills them
        with the median of the nearest-neighbour estimates across all trees.
        When `centrality = 1.0`, the nearest neighbour in each tree is selected
        deterministically; lower values introduce randomness.
        """
    def extrapolate(self, /, look_ahead: SupportsInt) -> list[float]:
        """
        Predict the next `look_ahead` base observations beyond the current shingle buffer.

        Requires `internal_shingling = True`, `shingle_size > 1`, and
        `look_ahead <= shingle_size`. Returns a list of length
        `look_ahead * input_dim`.
        """
    def is_ready(self, /) -> bool:
        "Return True once scoring functions return meaningful values."
    def entries_seen(self, /) -> int:
        "Number of observations processed so far."
    def num_trees(self, /) -> int:
        "Number of trees in the ensemble."
    def to_json(self, /) -> str:
        "Serialize the entire forest state to a JSON string."
    @staticmethod
    def from_json(json: str | bytes | bytearray | memoryview) -> Forest:
        "Deserialize a forest from a JSON string previously written by `to_json()`."
    def save_json(self, /, path: str | PathLike[str]) -> None:
        "Serialize the entire forest state to a JSON file."
    @staticmethod
    def load_json(path: str | PathLike[str]) -> Forest:
        "Deserialize a forest from a JSON file previously written by `save_json()`."
    def __repr__(self, /) -> str: ...
    def __str__(self, /) -> str: ...
    def __copy__(self, /) -> Self: ...
    def __deepcopy__(self, /, memo: Any) -> Self: ...
    def __getstate__(self, /) -> str: ...
    def __setstate__(self, /, state: str) -> None: ...

@final
class MStream:
    """
    mStream detector for mixed numerical/categorical records.

    `timestamp` is interpreted as a logical time tick, not as wall-clock time.
    Scores are invariant to adding a constant offset to all timestamps, while a
    gap of `k` ticks applies the temporal decay factor `alpha` exactly `k`
    times.

    Parameters
    ----------
    numeric_dim : int
        Number of numerical features in each record.
    categorical_dim : int
        Number of categorical features in each record.
    num_rows : int, optional
        Number of hash rows (default 2).
    num_buckets : int, optional
        Number of buckets per hash row (default 1024).
    alpha : float, optional
        Temporal decay factor in `(0, 1)` (default 0.8).
    seed : int, optional
        Random seed for deterministic hashing.
    """
    def __new__(
        cls,
        /,
        numeric_dim: SupportsInt,
        categorical_dim: SupportsInt,
        num_rows: SupportsInt = 2,
        num_buckets: SupportsInt = 1024,
        alpha: SupportsFloat = 0.8,
        seed: SupportsInt | None = None,
    ) -> Self: ...
    def update(
        self,
        /,
        numeric: Sequence[SupportsFloat],
        categorical: Sequence[SupportsInt],
        timestamp: SupportsInt,
    ) -> None:
        "Ingest a record without returning its score."
    def update_and_score(
        self,
        /,
        numeric: Sequence[SupportsFloat],
        categorical: Sequence[SupportsInt],
        timestamp: SupportsInt,
    ) -> float:
        """
        Ingest a record and return its anomaly score.

        `timestamp` must be a monotonically non-decreasing tick index. Only tick
        differences matter: shifting all timestamps by the same constant does
        not change the scores.
        """
    def update_and_score_detailed(
        self,
        /,
        numeric: Sequence[SupportsFloat],
        categorical: Sequence[SupportsInt],
        timestamp: SupportsInt,
    ) -> _MStreamScore:
        "Ingest a record and return the decomposed score used to form the final anomaly score."
    def score(
        self,
        /,
        numeric: Sequence[SupportsFloat],
        categorical: Sequence[SupportsInt],
        timestamp: SupportsInt,
    ) -> float:
        """
        Preview the anomaly score for a record without mutating detector state.

        The preview answers what this record would score if it were ingested
        next, using the same timestamp semantics as `update_and_score()`.
        """
    def score_detailed(
        self,
        /,
        numeric: Sequence[SupportsFloat],
        categorical: Sequence[SupportsInt],
        timestamp: SupportsInt,
    ) -> _MStreamScore:
        "Preview the decomposed anomaly score without mutating detector state."
    def is_ready(self, /) -> bool:
        "Return True once the detector has processed at least one record."
    def entries_seen(self, /) -> int:
        "Return the number of processed records."
    def current_time(self, /) -> int | None:
        "Return the last timestamp observed by the detector."
    def to_json(self, /) -> str:
        "Serialize the detector state to JSON."
    @staticmethod
    def from_json(json: str | bytes | bytearray | memoryview) -> MStream:
        "Deserialize detector state from a JSON string previously written by `to_json()`."
    def save_json(self, /, path: str | PathLike[str]) -> None:
        "Serialize the detector state to a JSON file."
    @staticmethod
    def load_json(path: str | PathLike[str]) -> MStream:
        "Deserialize detector state from a JSON file previously written by `save_json()`."
    def __repr__(self, /) -> str: ...
    def __str__(self, /) -> str: ...
    def __copy__(self, /) -> Self: ...
    def __deepcopy__(self, /, memo: Any) -> Self: ...
    def __getstate__(self, /) -> str: ...
    def __setstate__(self, /, state: str) -> None: ...

@final
class OnlineIForest:
    """
    Online Isolation Forest detector for numerical streams.

    Use `update()` or `update_and_score()` to ingest observations. Use
    `score()` to preview the current anomaly score for a point without mutating
    detector state.

    Parameters
    ----------
    input_dim : int
        Number of numerical features in each point.
    num_trees : int, optional
        Number of trees in the ensemble (default 32).
    window_size : int, optional
        Number of recent points retained by the sliding window (default 2048).
    max_leaf_samples : int, optional
        Base leaf-splitting threshold (default 32).
    seed : int, optional
        Random seed for deterministic trees.
    """
    def __new__(
        cls,
        /,
        input_dim: SupportsInt,
        num_trees: SupportsInt = 32,
        window_size: SupportsInt = 2048,
        max_leaf_samples: SupportsInt = 32,
        seed: SupportsInt | None = None,
    ) -> Self: ...
    def update(self, /, point: Sequence[SupportsFloat]) -> None:
        "Ingest a point without returning its score."
    def update_and_score(self, /, point: Sequence[SupportsFloat]) -> float:
        "Ingest a point and return its anomaly score under the updated forest."
    def score(self, /, point: Sequence[SupportsFloat]) -> float:
        """
        Preview the current anomaly score for `point` without mutating state.

        This can differ from `update_and_score()` because the preview is
        computed before `point` is learned by the forest. By contrast,
        `update_and_score(point)` returns the same value as calling
        `update(point)` and then `score(point)`.

        Calling this before `is_ready()` is allowed, but the value is not a
        stable anomaly estimate yet.
        """
    def is_ready(self, /) -> bool:
        "Return True once at least one point has been processed."
    def entries_seen(self, /) -> int:
        "Number of points processed so far."
    def num_trees(self, /) -> int:
        "Number of trees in the ensemble."
    def to_json(self, /) -> str:
        "Serialize detector state to JSON."
    @staticmethod
    def from_json(json: str | bytes | bytearray | memoryview) -> OnlineIForest:
        "Deserialize detector state from JSON previously written by `to_json()`."
    def save_json(self, /, path: str | PathLike[str]) -> None:
        "Serialize detector state to a JSON file."
    @staticmethod
    def load_json(path: str | PathLike[str]) -> OnlineIForest:
        "Deserialize detector state from a JSON file previously written by `save_json()`."
    def __repr__(self, /) -> str: ...
    def __str__(self, /) -> str: ...
    def __copy__(self, /) -> Self: ...
    def __deepcopy__(self, /, memo: Any) -> Self: ...
    def __getstate__(self, /) -> str: ...
    def __setstate__(self, /, state: str) -> None: ...
