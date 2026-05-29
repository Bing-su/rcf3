import math
import pickle
from collections.abc import Callable
from copy import deepcopy

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rcf3 import Forest

DEFAULT_NUM_TREES = 40
DEFAULT_CAPACITY = 64
DEFAULT_OUTPUT_AFTER = 1


FINITE_F32 = st.floats(
    min_value=-1_000.0,
    max_value=1_000.0,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)


def vector_strategy(dim: int) -> st.SearchStrategy[list[float]]:
    return st.lists(FINITE_F32, min_size=dim, max_size=dim)


def make_forest(  # noqa: PLR0913
    *,
    input_dim: int,
    seed: int,
    shingle_size: int = 1,
    internal_shingling: bool = True,
    num_trees: int = DEFAULT_NUM_TREES,
    capacity: int = DEFAULT_CAPACITY,
    output_after: int = DEFAULT_OUTPUT_AFTER,
    initial_accept_fraction: float = 0.125,
) -> Forest:
    """Create a Forest test fixture with deterministic defaults."""
    return Forest(
        input_dim=input_dim,
        shingle_size=shingle_size,
        num_trees=num_trees,
        capacity=capacity,
        output_after=output_after,
        internal_shingling=internal_shingling,
        initial_accept_fraction=initial_accept_fraction,
        seed=seed,
    )


def warm_forest(
    *,
    input_dim: int,
    shingle_size: int = 1,
    internal_shingling: bool = True,
    seed: int = 7,
    updates: int = 64,
) -> Forest:
    """Create a forest and warm it with synthetic observations."""
    forest = make_forest(
        input_dim=input_dim,
        seed=seed,
        shingle_size=shingle_size,
        internal_shingling=internal_shingling,
    )
    obs_dim = input_dim if internal_shingling else input_dim * shingle_size
    base = [0.125] * obs_dim
    for i in range(updates):
        point = [x + (i % 5) * 0.01 for x in base]
        forest.update(point)
    return forest


@settings(max_examples=30, deadline=None)
@given(
    input_dim=st.integers(min_value=1, max_value=6),
    stream_len=st.integers(min_value=24, max_value=72),
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
        data.draw(vector_strategy(input_dim), label=f"p{i}") for i in range(stream_len)
    ]
    queries = [data.draw(vector_strategy(input_dim), label=f"q{i}") for i in range(6)]

    f1 = make_forest(input_dim=input_dim, seed=seed)
    f2 = make_forest(input_dim=input_dim, seed=seed)

    for point in stream:
        f1.update(point)
        f2.update(point)

    assert f1.entries_seen() == f2.entries_seen()
    for q in queries:
        assert f1.score(q) == pytest.approx(f2.score(q), abs=1e-12)


@settings(max_examples=25, deadline=None)
@given(
    input_dim=st.integers(min_value=1, max_value=5),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    data=st.data(),
)
def test_update_and_score_matches_score_then_update(
    input_dim: int,
    seed: int,
    data: st.DataObject,
) -> None:
    warmup = [
        data.draw(vector_strategy(input_dim), label=f"warmup{i}") for i in range(32)
    ]
    point = data.draw(vector_strategy(input_dim), label="point")
    probe = data.draw(vector_strategy(input_dim), label="probe")

    manual = make_forest(input_dim=input_dim, seed=seed)
    for item in warmup:
        manual.update(item)
    fused = deepcopy(manual)

    expected = manual.score(point)
    manual.update(point)
    actual = fused.update_and_score(point)

    assert actual == pytest.approx(expected, abs=1e-12)
    assert fused.entries_seen() == manual.entries_seen()
    assert fused.score(probe) == pytest.approx(manual.score(probe), abs=1e-12)


@settings(max_examples=25, deadline=None)
@given(
    input_dim=st.integers(min_value=1, max_value=5),
    shingle_size=st.integers(min_value=1, max_value=4),
    capacity=st.integers(min_value=8, max_value=128),
    internal_shingling=st.booleans(),
    output_after=st.integers(min_value=0, max_value=20),
)
def test_is_ready_threshold_contract(
    input_dim: int,
    shingle_size: int,
    capacity: int,
    internal_shingling: bool,
    output_after: int,
) -> None:
    forest = make_forest(
        input_dim=input_dim,
        seed=19,
        shingle_size=shingle_size,
        num_trees=30,
        capacity=capacity,
        output_after=output_after,
        internal_shingling=internal_shingling,
    )

    effective_output_after = (1 + capacity // 4) if output_after == 0 else output_after
    needed = effective_output_after + (shingle_size - 1 if internal_shingling else 0)
    obs_dim = input_dim if internal_shingling else input_dim * shingle_size
    obs = [0.0] * obs_dim

    for _ in range(needed):
        forest.update(obs)
        assert not forest.is_ready()

    forest.update(obs)
    assert forest.is_ready()


class TestRoundTrip:
    """Round-trip serialization tests preserving state and scoring."""

    def _inner(
        self,
        input_dim: int,
        seed: int,
        data: st.DataObject,
        roundtrip: Callable[[Forest], Forest],
    ) -> None:
        """Assert that a round-trip transform preserves entries and score."""
        forest = make_forest(input_dim=input_dim, seed=seed)
        points = [
            data.draw(vector_strategy(input_dim), label=f"u{i}") for i in range(48)
        ]
        query = data.draw(vector_strategy(input_dim), label="query")

        for point in points:
            forest.update(point)

        before = forest.score(query)
        restored = roundtrip(forest)

        assert restored.entries_seen() == forest.entries_seen()
        assert restored.score(query) == pytest.approx(before, abs=1e-12)

    @settings(max_examples=20, deadline=None)
    @given(
        input_dim=st.integers(min_value=1, max_value=6),
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
            roundtrip=lambda f: Forest.from_json(f.to_json()),
        )

    @settings(max_examples=20, deadline=None)
    @given(
        input_dim=st.integers(min_value=1, max_value=6),
        seed=st.integers(min_value=0, max_value=2**32 - 1),
        protocol=st.integers(
            min_value=pickle.DEFAULT_PROTOCOL, max_value=pickle.HIGHEST_PROTOCOL
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
            roundtrip=lambda f: pickle.loads(pickle.dumps(f, protocol=protocol)),  # noqa: S301
        )

    @settings(max_examples=20, deadline=None)
    @given(
        input_dim=st.integers(min_value=1, max_value=6),
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
@given(
    input_dim=st.integers(min_value=2, max_value=6),
    top_k=st.integers(min_value=1, max_value=12),
    percentile=st.integers(min_value=0, max_value=100),
    data=st.data(),
)
def test_near_neighbors_are_bounded_and_sorted(
    input_dim: int,
    top_k: int,
    percentile: int,
    data: st.DataObject,
) -> None:
    forest = make_forest(input_dim=input_dim, seed=23, num_trees=50)
    updates = [data.draw(vector_strategy(input_dim), label=f"u{i}") for i in range(56)]
    query = data.draw(vector_strategy(input_dim), label="query")

    for point in updates:
        forest.update(point)

    results = forest.near_neighbors(query, top_k=top_k, percentile=percentile)
    assert len(results) <= top_k
    assert all(len(item["point"]) == input_dim for item in results)

    distances = [item["distance"] for item in results]
    assert distances == sorted(distances)


@settings(max_examples=25, deadline=None)
@given(
    input_dim=st.integers(min_value=2, max_value=8),
    data=st.data(),
)
def test_impute_deterministic_with_centrality_one_and_preserves_observed(
    input_dim: int,
    data: st.DataObject,
) -> None:
    forest = warm_forest(input_dim=input_dim, seed=29)
    query = data.draw(vector_strategy(input_dim), label="query")
    missing = sorted(
        data.draw(
            st.sets(
                st.integers(min_value=0, max_value=input_dim - 1),
                min_size=1,
                max_size=input_dim - 1,
            ),
            label="missing",
        )
    )

    out1 = forest.impute(query, missing, centrality=1.0)
    out2 = forest.impute(query, missing, centrality=1.0)
    out3 = forest.impute(query, missing, centrality=1.0)

    assert len(out1) == input_dim
    assert out1 == pytest.approx(out2, abs=1e-7)
    assert out2 == pytest.approx(out3, abs=1e-7)

    missing_set = set(missing)
    for i, original in enumerate(query):
        if i not in missing_set:
            assert out1[i] == pytest.approx(original, abs=1e-6)
        assert math.isfinite(out1[i])


@settings(max_examples=15, deadline=None)
@given(
    input_dim=st.integers(min_value=1, max_value=4),
    shingle_size=st.integers(min_value=2, max_value=4),
    data=st.data(),
)
def test_extrapolate_shape_and_finite_values(
    input_dim: int,
    shingle_size: int,
    data: st.DataObject,
) -> None:
    look_ahead = data.draw(
        st.integers(min_value=0, max_value=shingle_size), label="look_ahead"
    )
    forest = warm_forest(
        input_dim=input_dim,
        shingle_size=shingle_size,
        internal_shingling=True,
        seed=31,
        updates=96,
    )
    out = forest.extrapolate(look_ahead)

    assert len(out) == look_ahead * input_dim
    assert all(math.isfinite(x) for x in out)


@settings(max_examples=10, deadline=None)
@given(
    input_dim=st.integers(min_value=1, max_value=5),
    look_ahead=st.integers(min_value=1, max_value=5),
)
def test_extrapolate_requires_internal_shingling(
    input_dim: int, look_ahead: int
) -> None:
    forest = warm_forest(
        input_dim=input_dim,
        shingle_size=2,
        internal_shingling=False,
        seed=37,
        updates=96,
    )
    with pytest.raises(ValueError, match="internal_shingling"):
        forest.extrapolate(look_ahead)


@settings(max_examples=10, deadline=None)
@given(
    input_dim=st.integers(min_value=1, max_value=5),
    look_ahead=st.integers(min_value=1, max_value=5),
)
def test_extrapolate_requires_shingle_size_gt_one(
    input_dim: int, look_ahead: int
) -> None:
    forest = warm_forest(
        input_dim=input_dim,
        shingle_size=1,
        internal_shingling=True,
        seed=41,
        updates=96,
    )
    with pytest.raises(ValueError, match="shingle_size"):
        forest.extrapolate(look_ahead)


@settings(max_examples=12, deadline=None)
@given(
    input_dim=st.integers(min_value=1, max_value=5),
    shingle_size=st.integers(min_value=2, max_value=5),
    extra=st.integers(min_value=1, max_value=5),
)
def test_extrapolate_rejects_look_ahead_beyond_shingle_window(
    input_dim: int,
    shingle_size: int,
    extra: int,
) -> None:
    forest = warm_forest(
        input_dim=input_dim,
        shingle_size=shingle_size,
        internal_shingling=True,
        seed=43,
        updates=96,
    )
    with pytest.raises(ValueError, match="look_ahead"):
        forest.extrapolate(shingle_size + extra)


@settings(max_examples=15, deadline=None)
@given(input_dim=st.integers(min_value=2, max_value=8), data=st.data())
def test_dimension_mismatch_raises_value_error(
    input_dim: int, data: st.DataObject
) -> None:
    forest = make_forest(input_dim=input_dim, seed=11)
    bad_point = data.draw(vector_strategy(input_dim + 1), label="bad")

    with pytest.raises(ValueError, match="dimension mismatch"):
        forest.update(bad_point)


def test_initial_accept_fraction_constructor_argument_is_accepted() -> None:
    forest = make_forest(input_dim=2, seed=17, initial_accept_fraction=1.0)

    forest.update([0.1, 0.2])

    assert forest.entries_seen() == 1


@pytest.mark.parametrize(
    "value",
    [-0.1, 1.1, math.nan, math.inf, -math.inf],
)
def test_invalid_initial_accept_fraction_raises_value_error(value: float) -> None:
    with pytest.raises(ValueError, match="initial_accept_fraction"):
        make_forest(input_dim=2, seed=17, initial_accept_fraction=value)


def test_impute_not_ready_raises_runtime_error() -> None:
    forest = make_forest(input_dim=4, seed=3)

    assert not forest.is_ready()
    with pytest.raises(RuntimeError):
        forest.impute([0.1, 0.2, 0.3, 0.4], [1], centrality=1.0)
