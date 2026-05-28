import pickle
from collections.abc import Callable
from copy import deepcopy

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rcf3 import MStream

VALID_NUMERIC_F64 = st.floats(
    min_value=-1_000.0,
    max_value=1_000.0,
    allow_nan=False,
    allow_infinity=False,
    width=64,
)
CATEGORICAL_I64 = st.integers(min_value=-(2**31), max_value=2**31 - 1)


def numeric_strategy(dim: int) -> st.SearchStrategy[list[float]]:
    return st.lists(VALID_NUMERIC_F64, min_size=dim, max_size=dim)


def categorical_strategy(dim: int) -> st.SearchStrategy[list[int]]:
    return st.lists(CATEGORICAL_I64, min_size=dim, max_size=dim)


def make_detector(
    *,
    numeric_dim: int = 2,
    categorical_dim: int = 1,
    seed: int = 7,
) -> MStream:
    return MStream(
        numeric_dim=numeric_dim,
        categorical_dim=categorical_dim,
        seed=seed,
    )


def warm_detector(detector: MStream) -> None:
    detector.update([0.1, 0.2], [1], 1)
    detector.update([0.2, 0.3], [1], 2)


def timestamps_from_gaps(gaps: list[int]) -> list[int]:
    current = 1
    timestamps = []
    for gap in gaps:
        current += gap
        timestamps.append(current)
    return timestamps


@settings(max_examples=30, deadline=None)
@given(
    numeric_dim=st.integers(min_value=0, max_value=4),
    categorical_dim=st.integers(min_value=0, max_value=4),
    stream_len=st.integers(min_value=1, max_value=24),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    data=st.data(),
)
def test_seeded_stream_is_deterministic(
    numeric_dim: int,
    categorical_dim: int,
    stream_len: int,
    seed: int,
    data: st.DataObject,
) -> None:
    if numeric_dim == 0 and categorical_dim == 0:
        numeric_dim = 1

    gaps = data.draw(
        st.lists(
            st.integers(min_value=0, max_value=5),
            min_size=stream_len,
            max_size=stream_len,
        ),
        label="gaps",
    )
    timestamps = timestamps_from_gaps(gaps)
    stream = [
        (
            data.draw(numeric_strategy(numeric_dim), label=f"numeric_{index}"),
            data.draw(
                categorical_strategy(categorical_dim), label=f"categorical_{index}"
            ),
            timestamp,
        )
        for index, timestamp in enumerate(timestamps)
    ]

    left = make_detector(
        numeric_dim=numeric_dim, categorical_dim=categorical_dim, seed=seed
    )
    right = make_detector(
        numeric_dim=numeric_dim, categorical_dim=categorical_dim, seed=seed
    )

    left_scores = [left.update_and_score(*record) for record in stream]
    right_scores = [right.update_and_score(*record) for record in stream]

    assert left_scores == pytest.approx(right_scores, abs=1e-12)


@settings(max_examples=30, deadline=None)
@given(
    numeric_dim=st.integers(min_value=0, max_value=4),
    categorical_dim=st.integers(min_value=0, max_value=4),
    stream_len=st.integers(min_value=1, max_value=20),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    data=st.data(),
)
def test_preview_matches_committed_score(
    numeric_dim: int,
    categorical_dim: int,
    stream_len: int,
    seed: int,
    data: st.DataObject,
) -> None:
    if numeric_dim == 0 and categorical_dim == 0:
        categorical_dim = 1

    detector = make_detector(
        numeric_dim=numeric_dim,
        categorical_dim=categorical_dim,
        seed=seed,
    )
    gaps = data.draw(
        st.lists(
            st.integers(min_value=0, max_value=4),
            min_size=stream_len,
            max_size=stream_len,
        ),
        label="warmup_gaps",
    )
    for index, timestamp in enumerate(timestamps_from_gaps(gaps)):
        detector.update(
            data.draw(numeric_strategy(numeric_dim), label=f"warm_numeric_{index}"),
            data.draw(
                categorical_strategy(categorical_dim), label=f"warm_categorical_{index}"
            ),
            timestamp,
        )

    current_time = detector.current_time()
    assert current_time is not None
    next_timestamp = current_time + data.draw(
        st.integers(min_value=0, max_value=4),
        label="next_gap",
    )
    query_numeric = data.draw(numeric_strategy(numeric_dim), label="query_numeric")
    query_categorical = data.draw(
        categorical_strategy(categorical_dim),
        label="query_categorical",
    )

    preview = detector.score(query_numeric, query_categorical, next_timestamp)
    committed = detector.update_and_score(
        query_numeric, query_categorical, next_timestamp
    )

    assert committed == pytest.approx(preview, abs=1e-12)


@settings(max_examples=30, deadline=None)
@given(
    numeric_dim=st.integers(min_value=0, max_value=4),
    categorical_dim=st.integers(min_value=0, max_value=4),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    data=st.data(),
)
def test_detailed_score_shape_and_feature_lengths(
    numeric_dim: int,
    categorical_dim: int,
    seed: int,
    data: st.DataObject,
) -> None:
    if numeric_dim == 0 and categorical_dim == 0:
        numeric_dim = 1

    detector = make_detector(
        numeric_dim=numeric_dim,
        categorical_dim=categorical_dim,
        seed=seed,
    )
    detector.update(
        data.draw(numeric_strategy(numeric_dim), label="warm_numeric"),
        data.draw(categorical_strategy(categorical_dim), label="warm_categorical"),
        1,
    )

    detailed = detector.score_detailed(
        data.draw(numeric_strategy(numeric_dim), label="query_numeric"),
        data.draw(categorical_strategy(categorical_dim), label="query_categorical"),
        2,
    )

    assert set(detailed) == {
        "total",
        "record",
        "numeric_features",
        "categorical_features",
    }
    assert len(detailed["numeric_features"]) == numeric_dim
    assert len(detailed["categorical_features"]) == categorical_dim
    assert detailed["total"] == pytest.approx(
        detailed["record"]
        + sum(detailed["numeric_features"])
        + sum(detailed["categorical_features"]),
        abs=1e-12,
    )


def test_status_accessors_and_repr() -> None:
    detector = make_detector()

    assert not detector.is_ready()
    assert detector.entries_seen() == 0
    assert detector.current_time() is None

    detector.update([0.1, 0.2], [1], 5)

    assert detector.is_ready()
    assert detector.entries_seen() == 1
    assert detector.current_time() == 5
    assert (
        repr(detector)
        == "MStream(numeric_dim=2, categorical_dim=1, num_rows=2, num_buckets=1024, alpha=0.8, entries_seen=1)"
    )


@pytest.mark.parametrize(
    ("numeric", "categorical", "timestamp"),
    [
        ([0.1], [1], 1),
        ([0.1, 0.2], [1, 2], 1),
    ],
)
def test_dimension_mismatches_raise_value_error(
    numeric: list[float],
    categorical: list[int],
    timestamp: int,
) -> None:
    detector = make_detector()

    with pytest.raises(ValueError, match="dimension mismatch"):
        detector.update(numeric, categorical, timestamp)


def test_invalid_timestamp_and_non_finite_numeric_values_raise_value_error() -> None:
    detector = make_detector()
    detector.update([0.1, 0.2], [1], 2)

    with pytest.raises(ValueError, match="non-decreasing"):
        detector.update([0.2, 0.3], [1], 1)

    with pytest.raises(ValueError, match="must be finite"):
        detector.score([float("nan"), 0.3], [1], 2)


def test_negative_numeric_values_below_minus_one_are_supported() -> None:
    detector = make_detector()

    detector.update([-2.0, -10.0], [1], 1)
    preview = detector.score([-1_000.0, 0.3], [1], 2)
    committed = detector.update_and_score([-1_000.0, 0.3], [1], 2)

    assert preview == pytest.approx(committed, abs=1e-12)


class TestRoundTrip:
    def _inner(
        self,
        numeric_dim: int,
        categorical_dim: int,
        seed: int,
        data: st.DataObject,
        roundtrip: Callable[[MStream], MStream],
    ) -> None:
        if numeric_dim == 0 and categorical_dim == 0:
            categorical_dim = 1

        detector = make_detector(
            numeric_dim=numeric_dim,
            categorical_dim=categorical_dim,
            seed=seed,
        )
        gaps = data.draw(
            st.lists(st.integers(min_value=0, max_value=4), min_size=1, max_size=24),
            label="gaps",
        )
        for index, timestamp in enumerate(timestamps_from_gaps(gaps)):
            detector.update(
                data.draw(numeric_strategy(numeric_dim), label=f"numeric_{index}"),
                data.draw(
                    categorical_strategy(categorical_dim),
                    label=f"categorical_{index}",
                ),
                timestamp,
            )

        current_time = detector.current_time()
        assert current_time is not None
        query = (
            data.draw(numeric_strategy(numeric_dim), label="query_numeric"),
            data.draw(categorical_strategy(categorical_dim), label="query_categorical"),
            current_time,
        )
        restored = roundtrip(detector)

        assert restored.entries_seen() == detector.entries_seen()
        assert restored.current_time() == detector.current_time()
        assert restored.score(*query) == pytest.approx(
            detector.score(*query), abs=1e-12
        )

    @settings(max_examples=20, deadline=None)
    @given(
        numeric_dim=st.integers(min_value=0, max_value=4),
        categorical_dim=st.integers(min_value=0, max_value=4),
        seed=st.integers(min_value=0, max_value=2**32 - 1),
        data=st.data(),
    )
    def test_json_roundtrip_preserves_state_and_scores(
        self,
        numeric_dim: int,
        categorical_dim: int,
        seed: int,
        data: st.DataObject,
    ) -> None:
        self._inner(
            numeric_dim=numeric_dim,
            categorical_dim=categorical_dim,
            seed=seed,
            data=data,
            roundtrip=lambda detector: MStream.from_json(detector.to_json()),
        )

    @settings(max_examples=20, deadline=None)
    @given(
        numeric_dim=st.integers(min_value=0, max_value=4),
        categorical_dim=st.integers(min_value=0, max_value=4),
        seed=st.integers(min_value=0, max_value=2**32 - 1),
        protocol=st.integers(
            min_value=pickle.DEFAULT_PROTOCOL,
            max_value=pickle.HIGHEST_PROTOCOL,
        ),
        data=st.data(),
    )
    def test_pickle_roundtrip_preserves_state_and_scores(
        self,
        numeric_dim: int,
        categorical_dim: int,
        seed: int,
        protocol: int,
        data: st.DataObject,
    ) -> None:
        self._inner(
            numeric_dim=numeric_dim,
            categorical_dim=categorical_dim,
            seed=seed,
            data=data,
            roundtrip=lambda detector: pickle.loads(  # noqa: S301
                pickle.dumps(detector, protocol=protocol)
            ),
        )

    @settings(max_examples=20, deadline=None)
    @given(
        numeric_dim=st.integers(min_value=0, max_value=4),
        categorical_dim=st.integers(min_value=0, max_value=4),
        seed=st.integers(min_value=0, max_value=2**32 - 1),
        data=st.data(),
    )
    def test_deepcopy_preserves_state_and_scores(
        self,
        numeric_dim: int,
        categorical_dim: int,
        seed: int,
        data: st.DataObject,
    ) -> None:
        self._inner(
            numeric_dim=numeric_dim,
            categorical_dim=categorical_dim,
            seed=seed,
            data=data,
            roundtrip=deepcopy,
        )
