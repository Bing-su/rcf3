import pickle
import tempfile
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rcf3 import OnlineIForest

FINITE_F32 = st.floats(
    min_value=-1_000.0,
    max_value=1_000.0,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)


def vector_strategy(dim: int) -> st.SearchStrategy[list[float]]:
    return st.lists(FINITE_F32, min_size=dim, max_size=dim)


def make_detector(
    *,
    input_dim: int = 2,
    seed: int | None = None,
    num_trees: int = 8,
    window_size: int = 32,
    max_leaf_samples: int = 4,
) -> OnlineIForest:
    return OnlineIForest(
        input_dim=input_dim,
        num_trees=num_trees,
        window_size=window_size,
        max_leaf_samples=max_leaf_samples,
        seed=seed,
    )


@settings(max_examples=25, deadline=None)
@given(
    input_dim=st.integers(min_value=1, max_value=5),
    stream_len=st.integers(min_value=1, max_value=32),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    data=st.data(),
)
def test_seeded_stream_is_deterministic(
    input_dim: int,
    stream_len: int,
    seed: int,
    data: st.DataObject,
) -> None:
    stream = [
        data.draw(vector_strategy(input_dim), label=f"point_{idx}")
        for idx in range(stream_len)
    ]

    left = make_detector(input_dim=input_dim, seed=seed)
    right = make_detector(input_dim=input_dim, seed=seed)

    left_scores = [left.update_and_score(point) for point in stream]
    right_scores = [right.update_and_score(point) for point in stream]

    assert left_scores == pytest.approx(right_scores, abs=1e-12)
    assert left.entries_seen() == stream_len
    assert right.entries_seen() == stream_len


@settings(max_examples=25, deadline=None)
@given(
    input_dim=st.integers(min_value=1, max_value=5),
    warm_len=st.integers(min_value=1, max_value=24),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    data=st.data(),
)
def test_update_and_score_matches_update_then_score(
    input_dim: int,
    warm_len: int,
    seed: int,
    data: st.DataObject,
) -> None:
    commit = make_detector(input_dim=input_dim, seed=seed)
    split = make_detector(input_dim=input_dim, seed=seed)

    for idx in range(warm_len):
        point = data.draw(vector_strategy(input_dim), label=f"warm_{idx}")
        commit.update(point)
        split.update(point)

    query = data.draw(vector_strategy(input_dim), label="query")
    committed = commit.update_and_score(query)
    split.update(query)
    preview_after_commit = split.score(query)

    assert committed == pytest.approx(preview_after_commit, abs=1e-12)


class TestRoundTrip:
    def _inner(
        self,
        input_dim: int,
        seed: int,
        data: st.DataObject,
        roundtrip: Callable[[OnlineIForest], OnlineIForest],
    ) -> None:
        detector = make_detector(input_dim=input_dim, seed=seed)
        for idx in range(32):
            detector.update(data.draw(vector_strategy(input_dim), label=f"point_{idx}"))
        query = data.draw(vector_strategy(input_dim), label="query")

        restored = roundtrip(detector)

        assert restored.entries_seen() == detector.entries_seen()
        assert restored.num_trees() == detector.num_trees()
        assert restored.score(query) == pytest.approx(detector.score(query), abs=1e-12)

    @settings(max_examples=15, deadline=None)
    @given(
        input_dim=st.integers(min_value=1, max_value=5),
        seed=st.integers(min_value=0, max_value=2**32 - 1),
        data=st.data(),
    )
    def test_json_roundtrip_preserves_state_and_scores(
        self,
        input_dim: int,
        seed: int,
        data: st.DataObject,
    ) -> None:
        self._inner(
            input_dim=input_dim,
            seed=seed,
            data=data,
            roundtrip=lambda detector: OnlineIForest.from_json(detector.to_json()),
        )

    @settings(max_examples=15, deadline=None)
    @given(
        input_dim=st.integers(min_value=1, max_value=5),
        seed=st.integers(min_value=0, max_value=2**32 - 1),
        protocol=st.integers(
            min_value=pickle.DEFAULT_PROTOCOL,
            max_value=pickle.HIGHEST_PROTOCOL,
        ),
        data=st.data(),
    )
    def test_pickle_roundtrip_preserves_state_and_scores(
        self,
        input_dim: int,
        seed: int,
        protocol: int,
        data: st.DataObject,
    ) -> None:
        self._inner(
            input_dim=input_dim,
            seed=seed,
            data=data,
            roundtrip=lambda detector: pickle.loads(  # noqa: S301
                pickle.dumps(detector, protocol=protocol)
            ),
        )

    @settings(max_examples=15, deadline=None)
    @given(
        input_dim=st.integers(min_value=1, max_value=5),
        seed=st.integers(min_value=0, max_value=2**32 - 1),
        data=st.data(),
    )
    def test_deepcopy_preserves_state_and_scores(
        self,
        input_dim: int,
        seed: int,
        data: st.DataObject,
    ) -> None:
        self._inner(
            input_dim=input_dim,
            seed=seed,
            data=data,
            roundtrip=deepcopy,
        )


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_status_accessors_repr_and_score_before_ready(seed: int) -> None:
    detector = make_detector(input_dim=2, seed=seed)

    assert not detector.is_ready()
    assert detector.entries_seen() == 0
    assert detector.num_trees() == 8
    assert detector.score([0.0, 0.0]) == pytest.approx(1.0, abs=1e-12)

    detector.update([0.0, 0.0])

    assert detector.is_ready()
    assert detector.entries_seen() == 1
    assert (
        repr(detector)
        == "OnlineIForest(input_dim=2, num_trees=8, window_size=32, max_leaf_samples=4, entries_seen=1)"
    )


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_score_preview_does_not_mutate_state(seed: int) -> None:
    detector = make_detector(input_dim=2, seed=seed)
    detector.update([0.0, 0.0])

    entries_before = detector.entries_seen()
    first = detector.score([1.0, 1.0])
    second = detector.score([1.0, 1.0])

    assert detector.entries_seen() == entries_before
    assert first == pytest.approx(second, abs=1e-12)


@settings(max_examples=15, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_save_load_json_preserves_state_and_scores(seed: int) -> None:
    detector = make_detector(input_dim=2, seed=seed)
    for idx in range(12):
        detector.update([idx * 0.1, idx * 0.2])

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "onlineiforest.json"
        detector.save_json(path)
        restored = OnlineIForest.load_json(path)

    assert restored.entries_seen() == detector.entries_seen()
    assert restored.score([2.0, 4.0]) == pytest.approx(
        detector.score([2.0, 4.0]), abs=1e-12
    )


@pytest.mark.parametrize(
    "point",
    [
        [0.0],
        [0.0, 1.0, 2.0],
    ],
)
def test_dimension_mismatch_raises_value_error(point: list[float]) -> None:
    detector = make_detector(input_dim=2)

    with pytest.raises(ValueError, match="dimension mismatch"):
        detector.update(point)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_values_raise_value_error(value: float) -> None:
    detector = make_detector(input_dim=2)

    with pytest.raises(ValueError, match="must be finite"):
        detector.score([value, 0.0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_dim": 0},
        {"input_dim": 1, "num_trees": 0},
        {"input_dim": 1, "window_size": 0},
        {"input_dim": 1, "max_leaf_samples": 0},
        {"input_dim": 1, "window_size": 4, "max_leaf_samples": 4},
    ],
)
def test_invalid_constructor_arguments_raise_value_error(
    kwargs: dict[str, int],
) -> None:
    with pytest.raises(ValueError, match=r"must be|greater than"):
        OnlineIForest(**kwargs)
