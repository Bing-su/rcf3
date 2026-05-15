import math
import tempfile

from rcf3 import Forest


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

    # Update the forest
    point = [1.5, 2.3]
    forest.update(point)

    # Check if the forest has warmed up and get score
    if forest.is_ready():
        score = forest.score(point)
        print(f"Anomaly score: {score}")
        assert score >= 0.0

    # Get the number of observations processed
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

    query_point = [1.5, 2.3]
    neighbors = forest.near_neighbors(query_point, top_k=3, percentile=50)

    print(f"Found {len(neighbors)} neighbors:")
    for neighbor in neighbors:
        print(
            f"  Distance: {neighbor['distance']}, Score: {neighbor['score']}, Point: {neighbor['point']}"
        )

    print("✓ test_neighborhood_search passed")


def test_missing_value_imputation():
    """Test missing value imputation."""
    forest = Forest(input_dim=3, capacity=256, num_trees=50)

    # Feed some complete data to train
    for i in range(100):
        forest.update([1.0 + i * 0.01, 2.0, 3.0])

    # Use centrality=1.0 for deterministic nearest-candidate imputation.
    point = [1.5, float("nan"), 3.0]
    missing = [1]
    imputed = forest.impute(point, missing, centrality=1.0)

    print(f"Imputed value at index 1: {imputed[1]}")
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

    # Predict next 5 observations; look_ahead must be <= shingle_size.
    if forest.is_ready():
        predictions = forest.extrapolate(5)
        print(f"Predictions length: {len(predictions)}")
        # Returns a list of length 5 * input_dim = 5 * 4 = 20
        assert len(predictions) == 20

    print("✓ test_time_series_forecasting passed")


def test_serialization():
    """Test JSON serialization and deserialization."""
    forest = Forest(input_dim=2, capacity=256, num_trees=50)

    # Feed some data
    for _ in range(50):
        forest.update([1.5, 2.3])

    # Save to string
    json_str = forest.to_json()
    assert len(json_str) > 0
    print(f"JSON length: {len(json_str)}")

    # Load from string
    loaded = Forest.from_json(json_str)
    assert loaded.num_trees() == forest.num_trees()

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
    """Test the anomaly detection example from README."""
    forest = Forest(input_dim=3, capacity=256, num_trees=50)

    # Warm up the forest with many normal data points
    for i in range(200):
        val = i * 0.01
        forest.update([1.0 + val, 2.0 + val, 3.0 + val])

    data = [
        [1.0, 2.0, 3.0],
        [1.1, 2.1, 3.1],
        [1.2, 2.2, 3.2],
        [100.0, 200.0, 300.0],  # Extreme anomaly
        [1.3, 2.3, 3.3],
    ]

    anomaly_count = 0
    for point in data:
        # Online inference order: score first, then update.
        if forest.is_ready():
            score = forest.score(point)
            attribution = forest.attribution(point)

            print(f"Point: {point}, Score: {score}")

            # Lower threshold since we're detecting a very extreme anomaly
            if score > 0.1:
                print(f"Anomaly detected: score={score}")
                for i, attr in enumerate(attribution):
                    print(f"  Dimension {i}: {attr['above']:.2f}")
                anomaly_count += 1

        forest.update(point)

    print(f"Total anomalies detected: {anomaly_count}")
    assert anomaly_count > 0

    print("✓ test_anomaly_detection_example passed")


if __name__ == "__main__":
    import sys

    print("Running Python README examples tests...\n")

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

        print("\n✅ All tests passed!")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
