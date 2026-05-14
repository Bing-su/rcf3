from collections.abc import Sequence
from os import PathLike
from typing import Any, Final, Self, SupportsFloat, SupportsInt, TypedDict, final

__version__: Final[str]

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
class Forest:
    """
    A Random Cut Forest anomaly detector.

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
        Exponential decay for sample weights (default 0 = auto).
    output_after : int, optional
        Minimum observations before scoring starts (default 0 = auto).
    internal_shingling : bool, optional
        When True, pass one base observation at a time and the forest
        maintains the rolling shingle buffer (default True).
    seed : int, optional
        Random seed for deterministic forests.
    """
    def __copy__(self, /) -> Self: ...
    def __deepcopy__(self, /, memo: Any) -> Self: ...
    def __getstate__(self, /) -> str: ...
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
        seed: SupportsInt | None = None,
    ) -> Self: ...
    def __repr__(self, /) -> str: ...
    def __setstate__(self, /, state: str) -> None: ...
    def __str__(self, /) -> str: ...
    def attribution(self, /, point: Sequence[SupportsFloat]) -> list[_Attribution]:
        """
        Per-dimension attribution of the anomaly score.

        Returns a list of dict objects with keys: `below`, `above`.
        `above` captures contribution from cuts above the query value;
        `below` captures contribution from cuts below the query value.
        """
    def density(self, /, point: Sequence[SupportsFloat]) -> float:
        "Density estimate at `point` (higher means more typical)."
    def displacement_score(self, /, point: Sequence[SupportsFloat]) -> float:
        "Displacement-based anomaly score for `point` (higher means more anomalous)."
    def entries_seen(self, /) -> int:
        "Number of observations processed so far."
    def extrapolate(self, /, look_ahead: SupportsInt) -> list[float]:
        """
        Predict the next `look_ahead` base observations.

        Requires `internal_shingling = True` and `shingle_size > 1`.
        Returns a flat list of length `look_ahead * input_dim`.
        """
    @staticmethod
    def from_json(json: str) -> Forest:
        "Load a forest from a JSON string."
    def impute(
        self,
        /,
        point: Sequence[SupportsFloat],
        missing: Sequence[SupportsInt],
        centrality: SupportsFloat = 1.0,
    ) -> list[float]:
        """
        Impute the missing dimensions of `point`.

        Parameters
        ----------
        point : list[float]
            Full-dimensional query (missing values will be ignored).
        missing : list[int]
            Indices of dimensions to impute.
        centrality : float, optional
            1.0 = always pick the nearest candidate (deterministic).
        """
    def is_ready(self, /) -> bool:
        "Whether the forest has seen enough observations to return scores."
    @staticmethod
    def load_json(path: str | PathLike[str]) -> Forest:
        "Load a forest from a JSON file path."
    def near_neighbors(
        self,
        /,
        point: Sequence[SupportsFloat],
        top_k: SupportsInt = 10,
        percentile: SupportsInt = 50,
    ) -> list[_NeighborResult]:
        """
        Find approximate near-neighbours of `point`.

        Returns a list of dict objects with keys: `score`, `point`, `distance`.
        """
    def num_trees(self, /) -> int:
        "Number of trees."
    def save_json(self, /, path: str | PathLike[str]) -> None:
        "Serialise the forest state to a JSON file path."
    def score(self, /, point: Sequence[SupportsFloat]) -> float:
        "Anomaly score for `point` (higher means more anomalous). Returns 0.0 before ready."
    def to_json(self, /) -> str:
        "Serialise the forest state to a JSON string."
    def update(self, /, point: Sequence[SupportsFloat]) -> None:
        "Update the forest with a new observation."
