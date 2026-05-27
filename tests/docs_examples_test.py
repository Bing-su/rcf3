import math
import tempfile

from rcf3 import FeatureSketch, Forest, MStream, OnlineIForest


def test_creating_forest_basic():
    """Verify basic Forest construction with explicit 2D defaults."""
    forest = Forest(
        input_dim=2,
        shingle_size=1,
        num_trees=50,
        capacity=256,
    )
    assert forest.num_trees() == 50
    print("✓ test_creating_forest_basic passed")


def test_creating_forest_with_time_series():
    """Test creating a forest with time series."""
    forest = Forest(
        input_dim=4,
        shingle_size=8,
        num_trees=100,
        capacity=512,
        time_decay=0.01,
        internal_shingling=True,
    )
    assert forest.num_trees() == 100
    print("✓ test_creating_forest_with_time_series passed")


def test_basic_operations():
    """Test update/is_ready/score/entries_seen usage sequence."""
    forest = Forest(input_dim=2, capacity=256, num_trees=50)

    point = [1.5, 2.3]

    if forest.is_ready():
        score = forest.score(point)
        print(f"Anomaly score: {score}")
        assert score >= 0.0

    forest.update(point)
    print(f"Entries seen: {forest.entries_seen()}")
    print("✓ test_basic_operations passed")


def test_scoring_methods():
    """Test anomaly score, displacement score, and density on one query."""
    forest = Forest(input_dim=3, capacity=256, num_trees=50)

    # Feed some data to warm up
    for _ in range(100):
        forest.update([1.5, 2.3, -0.5])

    point = [1.5, 2.3, -0.5]

    # Anomaly score
    score = forest.score(point)
    assert score >= 0.0
    print(f"RCF Score: {score}")

    # Displacement score
    displacement = forest.displacement_score(point)
    assert displacement >= 0.0
    print(f"Displacement Score: {displacement}")

    # Density estimate
    density = forest.density(point)
    assert density >= 0.0
    print(f"Density: {density}")

    print("✓ test_scoring_methods passed")


def test_feature_attribution():
    """Test feature attribution."""
    forest = Forest(input_dim=3, capacity=256, num_trees=50)

    # Feed normal data first
    for _ in range(100):
        forest.update([1.0, 2.0, 3.0])

    point = [1.5, 2.3, 100.0]
    attribution = forest.attribution(point)

    for i, attr in enumerate(attribution):
        print(f"Dimension {i}: below={attr['below']}, above={attr['above']}")
        assert attr["below"] >= 0.0
        assert attr["above"] >= 0.0

    print("✓ test_feature_attribution passed")


def test_neighborhood_search():
    """Test neighborhood search."""
    forest = Forest(input_dim=2, capacity=256, num_trees=50)

    # Feed some data points
    data = [
        [1.0, 2.0],
        [1.1, 2.1],
        [1.2, 2.2],
        [1.3, 2.3],
        [1.4, 2.4],
        [5.0, 6.0],
        [5.1, 6.1],
        [5.2, 6.2],
    ]

    for point in data:
        forest.update(point)

    neighbors = forest.near_neighbors([1.5, 2.3], top_k=10, percentile=50)

    print(f"Found {len(neighbors)} neighbors: {neighbors}")
    print("✓ test_neighborhood_search passed")


def test_missing_value_imputation():
    """Test missing value imputation."""
    forest = Forest(input_dim=3)

    # Feed some complete data to train
    for i in range(100):
        forest.update([1.0 + i * 0.01, 2.0, 3.0])

    # Use centrality=1.0 for deterministic nearest-candidate imputation.
    point = [1.5, float("nan"), 3.0]
    missing = [1]
    imputed = forest.impute(point, missing, centrality=1.0)

    assert not math.isnan(imputed[1])
    assert len(imputed) == 3

    print("✓ test_missing_value_imputation passed")


def test_time_series_forecasting():
    """Test forecasting with internal shingling enabled."""
    forest = Forest(
        input_dim=4,
        shingle_size=8,
        internal_shingling=True,
    )

    # Feed observations one at a time
    stream = [
        [1.0, 2.0, 3.0, 4.0],
        [1.1, 2.1, 3.1, 4.1],
        [1.2, 2.2, 3.2, 4.2],
        [1.3, 2.3, 3.3, 4.3],
        [1.4, 2.4, 3.4, 4.4],
        [1.5, 2.5, 3.5, 4.5],
        [1.6, 2.6, 3.6, 4.6],
        [1.7, 2.7, 3.7, 4.7],
        [1.8, 2.8, 3.8, 4.8],
        [1.9, 2.9, 3.9, 4.9],
    ]

    for point in stream:
        forest.update(point)

    predictions = forest.extrapolate(5)
    assert len(predictions) == 20

    print("✓ test_time_series_forecasting passed")


def test_serialization():
    """Test JSON serialization and deserialization."""
    forest = Forest(input_dim=2)

    # Feed some data
    for _ in range(50):
        forest.update([1.5, 2.3])

    json_str = forest.to_json()
    assert len(json_str) > 0
    loaded = Forest.from_json(json_str)
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = f"{tmp_dir}/forest.json"
        forest.save_json(path)
        loaded_from_file = Forest.load_json(path)

    assert loaded.num_trees() == forest.num_trees()
    assert loaded_from_file.num_trees() == forest.num_trees()

    print("✓ test_serialization passed")


def test_pickle_serialization():
    """Test pickle serialization."""
    import pickle

    forest = Forest(input_dim=2, capacity=256, num_trees=50)

    # Feed some data
    for _ in range(50):
        forest.update([1.5, 2.3])

    with tempfile.TemporaryDirectory() as tmp_dir:
        pickle_path = f"{tmp_dir}/forest.pkl"

        # Save
        with open(pickle_path, "wb") as f:
            pickle.dump(forest, f)

        # Load
        with open(pickle_path, "rb") as f:
            loaded = pickle.load(f)

    assert loaded.num_trees() == forest.num_trees()
    print("✓ test_pickle_serialization passed")


def test_anomaly_detection_example():
    """Test the anomaly detection example from the Forest documentation."""
    forest = Forest(input_dim=3, capacity=256, num_trees=50)

    # Warm up the forest with many normal data points
    for i in range(200):
        val = i * 0.01
        forest.update([1.0 + val, 2.0 + val, 3.0 + val])

    for point in ([1.0, 2.0, 3.0], [1.1, 2.1, 3.1], [100.0, 200.0, 300.0]):
        if forest.is_ready():
            print(f"Point: {point}, score={forest.score(point)}")
        forest.update(point)

    print("✓ test_anomaly_detection_example passed")


def test_mstream_basic_usage():
    """Test the basic MStream documentation flow."""
    detector = MStream(
        numeric_dim=2,
        categorical_dim=1,
        alpha=0.8,
        num_rows=2,
        num_buckets=1024,
        seed=7,
    )

    score = detector.update_and_score([1.5, 2.0], [7], 1)
    assert score >= 0.0


def test_mstream_preview_and_detailed_scores():
    """Test preview parity and detailed MStream documentation examples."""
    detector = MStream(numeric_dim=2, categorical_dim=1, seed=7)
    detector.update([1.5, 2.0], [7], 1)

    preview = detector.score([1.5, 2.0], [7], 2)
    committed = detector.update_and_score([1.5, 2.0], [7], 2)
    assert preview == committed

    detailed = detector.score_detailed([1.5, 2.0], [7], 3)
    assert len(detailed["numeric_features"]) == 2
    assert len(detailed["categorical_features"]) == 1
    assert detector.is_ready()
    assert detector.entries_seen() == 2
    assert detector.current_time() == 2


def test_mstream_serialization():
    """Test the MStream JSON documentation example."""
    detector = MStream(numeric_dim=2, categorical_dim=1, seed=7)
    detector.update([1.5, 2.0], [7], 1)

    json_str = detector.to_json()
    restored = MStream.from_json(json_str)
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = f"{tmp_dir}/mstream.json"
        detector.save_json(path)
        restored_from_file = MStream.load_json(path)

    assert restored.entries_seen() == detector.entries_seen()
    assert restored.current_time() == detector.current_time()
    assert restored_from_file.entries_seen() == detector.entries_seen()
    assert restored_from_file.current_time() == detector.current_time()


def test_mstream_practical_example():
    """Test the practical MStream documentation example."""
    detector = MStream(numeric_dim=2, categorical_dim=2, seed=2026, num_buckets=512)

    normal = detector.update_and_score([0.0, 3.2], [1, 10], 1)
    suspicious = detector.score_detailed([12.0, 0.3], [99, 10], 2)

    print(f"normal={normal}, suspicious={suspicious['total']}")
    print(f"failed-attempt contribution={suspicious['numeric_features'][0]}")
    print(f"country contribution={suspicious['categorical_features'][0]}")


def test_featuresketch_basic_usage():
    """Test the basic FeatureSketch documentation flow."""
    detector = FeatureSketch(
        value_projection_dims=32,
        presence_projection_dims=32,
        chains_per_ensemble=16,
        chain_depth=8,
        sketch_rows=2,
        sketch_buckets=2048,
        decay_half_life=2048,
        seed=42,
    )

    score = detector.update_and_score(
        {
            "endpoint:/login": 1.0,
            "status:200": 1.0,
            "bytes": 812.0,
        }
    )
    assert score >= 0.0

    event = {
        "endpoint:/admin": 1.0,
        "status:401": 1.0,
        "bytes": 12000.0,
    }
    preview = detector.score(event)
    committed = detector.update_and_score(event)
    assert preview == committed
    assert detector.is_ready()
    assert detector.entries_seen() == 2


def test_featuresketch_serialization():
    """Test the FeatureSketch JSON documentation example."""
    detector = FeatureSketch(seed=7)
    detector.update(
        {
            "endpoint:/login": 1.0,
            "status:200": 1.0,
            "bytes": 812.0,
        }
    )

    json_str = detector.to_json()
    restored = FeatureSketch.from_json(json_str)
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = f"{tmp_dir}/featuresketch.json"
        detector.save_json(path)
        restored_from_file = FeatureSketch.load_json(path)

    assert restored.entries_seen() == detector.entries_seen()
    assert restored_from_file.entries_seen() == detector.entries_seen()


def test_featuresketch_practical_example():
    """Test the practical FeatureSketch documentation example."""
    detector = FeatureSketch(seed=2026, sketch_buckets=512)

    for _ in range(64):
        detector.update(
            {
                "endpoint:/login": 1.0,
                "status:200": 1.0,
                "bytes": 750.0,
            }
        )

    normal = detector.score(
        {
            "endpoint:/login": 1.0,
            "status:200": 1.0,
            "bytes": 790.0,
        }
    )
    suspicious = detector.score(
        {
            "endpoint:/admin": 1.0,
            "status:401": 1.0,
            "bytes": 12000.0,
        }
    )

    print(f"normal={normal}, suspicious={suspicious}")


def test_onlineiforest_basic_usage():
    """Test the basic OnlineIForest documentation flow."""
    detector = OnlineIForest(
        input_dim=2,
        num_trees=32,
        window_size=128,
        max_leaf_samples=8,
        seed=7,
    )

    score = detector.update_and_score([1.5, 2.3])
    preview = detector.score([1.6, 2.4])

    assert score >= 0.0
    assert preview >= 0.0
    assert detector.is_ready()
    assert detector.entries_seen() == 1
    assert detector.num_trees() == 32


def test_onlineiforest_serialization():
    """Test the OnlineIForest JSON documentation example."""
    detector = OnlineIForest(input_dim=2, window_size=128, max_leaf_samples=8, seed=7)
    detector.update([1.5, 2.3])

    json_str = detector.to_json()
    restored = OnlineIForest.from_json(json_str)
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = f"{tmp_dir}/onlineiforest.json"
        detector.save_json(path)
        restored_from_file = OnlineIForest.load_json(path)

    assert restored.entries_seen() == detector.entries_seen()
    assert restored.num_trees() == detector.num_trees()
    assert restored_from_file.entries_seen() == detector.entries_seen()
    assert restored_from_file.num_trees() == detector.num_trees()


def test_onlineiforest_practical_example():
    """Test the practical OnlineIForest documentation example."""
    detector = OnlineIForest(
        input_dim=2,
        window_size=128,
        max_leaf_samples=8,
        seed=2026,
    )

    for i in range(64):
        value = i * 0.01
        detector.update([value, value + 1.0])

    normal_score = detector.score([0.5, 1.5])
    anomaly_score = detector.score([10.0, -10.0])

    print(f"normal={normal_score}, anomaly={anomaly_score}")


if __name__ == "__main__":
    import sys

    print("Running Python documentation examples tests...\n")

    try:
        test_creating_forest_basic()
        test_creating_forest_with_time_series()
        test_basic_operations()
        test_scoring_methods()
        test_feature_attribution()
        test_neighborhood_search()
        test_missing_value_imputation()
        test_time_series_forecasting()
        test_serialization()
        test_pickle_serialization()
        test_anomaly_detection_example()
        test_mstream_basic_usage()
        test_mstream_preview_and_detailed_scores()
        test_mstream_serialization()
        test_mstream_practical_example()
        test_featuresketch_basic_usage()
        test_featuresketch_serialization()
        test_featuresketch_practical_example()
        test_onlineiforest_basic_usage()
        test_onlineiforest_serialization()
        test_onlineiforest_practical_example()

        print("\n✅ All tests passed!")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
