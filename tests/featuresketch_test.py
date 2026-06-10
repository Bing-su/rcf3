import pickle
import tempfile
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rcf3 import FeatureSketch

FINITE_F64 = st.floats(
    min_value=-1_000.0,
    max_value=1_000.0,
    allow_nan=False,
    allow_infinity=False,
    width=64,
)


def feature_event_strategy() -> st.SearchStrategy[list[tuple[str, float]]]:
    return st.lists(
        st.tuples(
            st.sampled_from(["country:kr", "country:us", "device:ios", "device:web"]),
            FINITE_F64,
        ),
        min_size=0,
        max_size=12,
    )


def make_detector(*, seed: int | None = None) -> FeatureSketch:
    return FeatureSketch(
        value_projection_dims=8,
        presence_projection_dims=8,
        chains_per_ensemble=8,
        chain_depth=4,
        sketch_rows=2,
        sketch_buckets=128,
        decay_half_life=128,
        seed=seed,
    )


@settings(max_examples=25, deadline=None)
@given(
    stream_len=st.integers(min_value=1, max_value=24),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    data=st.data(),
)
def test_seeded_stream_is_deterministic(
    stream_len: int,
    seed: int,
    data: st.DataObject,
) -> None:
    stream = [
        data.draw(feature_event_strategy(), label=f"event_{idx}")
        for idx in range(stream_len)
    ]

    left = make_detector(seed=seed)
    right = make_detector(seed=seed)

    left_scores = [left.update_and_score(event) for event in stream]
    right_scores = [right.update_and_score(event) for event in stream]

    assert left_scores == pytest.approx(right_scores, abs=1e-12)
    assert left.entries_seen() == stream_len
    assert right.entries_seen() == stream_len


@settings(max_examples=25, deadline=None)
@given(
    warm_len=st.integers(min_value=1, max_value=24),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    data=st.data(),
)
def test_score_preview_does_not_mutate_state(
    warm_len: int,
    seed: int,
    data: st.DataObject,
) -> None:
    detector = make_detector(seed=seed)
    for idx in range(warm_len):
        detector.update(data.draw(feature_event_strategy(), label=f"warm_{idx}"))

    query = data.draw(feature_event_strategy(), label="query")
    entries_before = detector.entries_seen()
    first = detector.score(query)
    second = detector.score(query)

    assert detector.entries_seen() == entries_before
    assert first == pytest.approx(second, abs=1e-12)


@settings(max_examples=25, deadline=None)
@given(
    warm_len=st.integers(min_value=1, max_value=24),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    data=st.data(),
)
def test_update_and_score_matches_score_then_update(
    warm_len: int,
    seed: int,
    data: st.DataObject,
) -> None:
    commit = make_detector(seed=seed)
    split = make_detector(seed=seed)

    for idx in range(warm_len):
        event = data.draw(feature_event_strategy(), label=f"warm_{idx}")
        commit.update(event)
        split.update(event)

    query = data.draw(feature_event_strategy(), label="query")
    preview = split.score(query)
    committed = commit.update_and_score(query)

    assert committed == pytest.approx(preview, abs=1e-12)


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_mapping_and_sequence_inputs_are_equivalent(seed: int) -> None:
    detector = make_detector(seed=seed)

    sequence = [("device:web", 2.0), ("country:kr", 1.0), ("device:web", 2.0)]
    mapping = {"device:web": 4.0, "country:kr": 1.0}

    assert detector.score(sequence) == pytest.approx(detector.score(mapping), abs=1e-12)


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_status_accessors_and_repr(seed: int) -> None:
    detector = make_detector(seed=seed)

    assert not detector.is_ready()
    assert detector.entries_seen() == 0

    detector.update({"device:web": 1.0})

    assert detector.is_ready()
    assert detector.entries_seen() == 1
    assert (
        repr(detector)
        == "FeatureSketch(value_projection_dims=8, presence_projection_dims=8, chains_per_ensemble=8, chain_depth=4, sketch_rows=2, sketch_buckets=128, decay_half_life=128, entries_seen=1)"
    )


class TestRoundTrip:
    def _inner(
        self,
        seed: int,
        data: st.DataObject,
        roundtrip: Callable[[FeatureSketch], FeatureSketch],
    ) -> None:
        detector = make_detector(seed=seed)
        for idx in range(24):
            detector.update(data.draw(feature_event_strategy(), label=f"event_{idx}"))
        query = data.draw(feature_event_strategy(), label="query")

        restored = roundtrip(detector)

        assert restored.entries_seen() == detector.entries_seen()
        assert restored.score(query) == pytest.approx(detector.score(query), abs=1e-12)

    @settings(max_examples=15, deadline=None)
    @given(
        seed=st.integers(min_value=0, max_value=2**32 - 1),
        data=st.data(),
    )
    def test_json_roundtrip_preserves_state_and_scores(
        self,
        seed: int,
        data: st.DataObject,
    ) -> None:
        self._inner(
            seed=seed,
            data=data,
            roundtrip=lambda detector: FeatureSketch.from_json(detector.to_json()),
        )

    @settings(max_examples=15, deadline=None)
    @given(
        seed=st.integers(min_value=0, max_value=2**32 - 1),
        protocol=st.integers(
            min_value=pickle.DEFAULT_PROTOCOL,
            max_value=pickle.HIGHEST_PROTOCOL,
        ),
        data=st.data(),
    )
    def test_pickle_roundtrip_preserves_state_and_scores(
        self,
        seed: int,
        protocol: int,
        data: st.DataObject,
    ) -> None:
        self._inner(
            seed=seed,
            data=data,
            roundtrip=lambda detector: pickle.loads(  # noqa: S301
                pickle.dumps(detector, protocol=protocol)
            ),
        )

    @settings(max_examples=15, deadline=None)
    @given(
        seed=st.integers(min_value=0, max_value=2**32 - 1),
        data=st.data(),
    )
    def test_deepcopy_preserves_state_and_scores(
        self,
        seed: int,
        data: st.DataObject,
    ) -> None:
        self._inner(seed=seed, data=data, roundtrip=deepcopy)


@settings(max_examples=15, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_save_load_json_preserves_state_and_scores(seed: int) -> None:
    detector = make_detector(seed=seed)
    for idx in range(12):
        detector.update({"event": float(idx), "device:web": 1.0})

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "featuresketch.json"
        detector.save_json(path)
        restored = FeatureSketch.load_json(path)

    assert restored.entries_seen() == detector.entries_seen()
    assert restored.score({"event": 2.0}) == pytest.approx(
        detector.score({"event": 2.0}), abs=1e-12
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_values_raise_value_error(value: float) -> None:
    detector = make_detector()

    with pytest.raises(ValueError, match="feature values must be finite"):
        detector.score({"bad": value})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"value_projection_dims": 0},
        {"presence_projection_dims": 0},
        {"chains_per_ensemble": 0},
        {"chain_depth": 0},
        {"sketch_rows": 0},
        {"sketch_buckets": 1},
        {"decay_half_life": 0},
    ],
)
def test_invalid_constructor_arguments_raise_value_error(
    kwargs: dict[str, int],
) -> None:
    with pytest.raises(ValueError, match="must be"):
        FeatureSketch(**kwargs)
