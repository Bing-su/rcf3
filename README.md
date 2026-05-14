# rcf3

A Rust implementation of the Random Cut Forest (RCF) algorithm for anomaly detection in streaming data.

## Overview

Random Cut Forest is an ensemble-based anomaly detection algorithm that uses randomized decision trees to identify anomalies in both univariate and multivariate time series data. It's particularly effective for:

- **Anomaly Detection**: Identifying unusual patterns in streaming data
- **Time Series Analysis**: Detecting changes in temporal patterns and seasonality
- **Interpretability**: Provides feature attribution scores to understand which dimensions contribute to anomalies

This implementation provides both Rust and Python APIs with support for advanced features like missing value imputation, neighborhood search, and time series forecasting.

## Features

The crate supports several compile-time features:

### `std` (enabled by default)

Enables use of the Rust standard library. Disable this for `no_std` environments:

```toml
[dependencies]
rcf3 = { version = "0.1", default-features = false }
```

### `serde` (enabled by default)

Provides JSON serialization and deserialization support for `Forest` objects. Allows saving and loading trained models:

```toml
[dependencies]
rcf3 = { version = "0.1", features = ["serde", "std"] }
```

### `python` (optional)

Builds Python bindings using PyO3, enabling use from Python. Automatically enables `serde` and `std`:

```toml
[dependencies]
rcf3 = { version = "0.1", features = ["python"] }
```

To use just the core algorithm without serialization:

```toml
[dependencies]
rcf3 = { version = "0.1", default-features = false, features = ["std"] }
```

## Configuration Options

All forests are configured via `RcfConfig` with the following parameters:

| Parameter                 | Type    | Default      | Description                                                                                                                  |
| ------------------------- | ------- | ------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| `input_dim`               | `usize` | **Required** | Number of base feature dimensions per observation (before shingling)                                                         |
| `shingle_size`            | `usize` | `1`          | Temporal window size. When `internal_shingling` is true, the effective model dimension becomes `input_dim * shingle_size`    |
| `capacity`                | `usize` | `256`        | Maximum number of points stored per tree                                                                                     |
| `num_trees`               | `usize` | `50`         | Number of trees in the ensemble                                                                                              |
| `time_decay`              | `f64`   | `0.0`        | Exponential time-decay rate applied to sampling weights. `0.0` uses the default: `0.1 / capacity`                            |
| `output_after`            | `usize` | `0`          | Minimum number of updates before score/attribution/etc. return non-trivial results. `0` uses the default: `1 + capacity / 4` |
| `internal_shingling`      | `bool`  | `true`       | When true, the forest automatically manages the shingle buffer so callers pass one base observation at a time                |
| `initial_accept_fraction` | `f64`   | `0.125`      | Controls how quickly the sampler fills to capacity during warm-up                                                            |

## Rust API

### Creating a Forest

Use the builder pattern to create a configured forest:

```rust
use rcf3::Forest;

let forest = Forest::builder(2, 1)  // 2D input, shingle size 1 (default)
    .num_trees(50)
    .capacity(256)
    .build()?;
```

With time series (shingling):

```rust
let forest = Forest::builder(4, 8)  // 4D input, window size 8
    .num_trees(100)
    .capacity(512)
    .time_decay(0.01)
    .build()?;
```

`internal_shingling` is `true` by default, so you only need to set it explicitly when turning it off.

From a config object:

```rust
use rcf3::{RcfConfig, Forest};

let config = RcfConfig::new(3)
    .with_num_trees(75)
    .with_capacity(512)
    .with_shingle_size(4);

let forest = Forest::from_config(&config)?;
```

### Basic Operations

For online anomaly detection, the recommended order is to score first and then update.
The snippets below are minimal API examples showing each operation separately.

**Update the forest with a new observation:**

```rust
let point = vec![1.5, 2.3];
forest.update(&point)?;
```

**Check if the forest has warmed up:**

```rust
if forest.is_ready() {
    let score = forest.score(&point)?;
    println!("Anomaly score: {}", score);
}
```

**Get the number of observations processed:**

```rust
println!("Entries seen: {}", forest.entries_seen());
```

### Scoring Methods

**Anomaly Score (RCF Score):**

The primary anomaly metric. Lower scores indicate normal behavior; higher scores indicate anomalies.

```rust
let point = vec![1.5, 2.3, -0.5];
let score = forest.score(&point)?;
if score > threshold {
    println!("Anomaly detected!");
}
```

**Displacement Score:**

A displacement-based anomaly metric that measures how far a point is from the expected region:

```rust
let displacement = forest.displacement_score(&point)?;
```

**Density Estimate:**

Returns an estimate of the probability density at the given point. Higher density = normal behavior:

```rust
let density = forest.density(&point)?;
```

### Feature Attribution

Understand which dimensions contribute to the anomaly score:

```rust
let point = vec![1.5, 2.3, 100.0];  // Third dimension is anomalous
let attribution = forest.attribution(&point)?;

for (i, attr) in attribution.iter().enumerate() {
    println!("Dimension {}: below={}, above={}", i, attr.below, attr.above);
}
```

Each dimension returns `below` and `above` scores indicating how much that dimension contributes to the overall anomaly:

- `above`: contribution from cuts above the query value (query is unusually small)
- `below`: contribution from cuts below the query value (query is unusually large)

### Neighborhood Search

Find approximate near-neighbors of a query point:

```rust
let point = vec![1.5, 2.3];
let neighbors = forest.near_neighbors(&point, 10, 50)?;

for neighbor in neighbors {
    println!("Distance: {}, Score: {}", neighbor.distance, neighbor.score);
    println!("Point: {:?}", neighbor.point);
}
```

Parameters:

- `top_k`: Maximum number of neighbors to return (default 10)
- `percentile`: Percentile threshold for filtering candidates (default 50)

### Missing Value Imputation

Impute missing dimensions using learned data distribution:

```rust
let point = vec![1.5, f32::NAN, 3.0];  // Missing value at index 1
let missing = vec![1];  // Indices of missing dimensions
let imputed = forest.impute(&point, &missing, 1.0)?;

println!("Imputed value at index 1: {}", imputed[1]);
```

Parameters:

- `point`: Full-dimensional query (missing values will be ignored)
- `missing`: Indices of dimensions to impute
- `centrality`: Controls how deterministic the imputation is (1.0 = always pick nearest candidate)

### Time Series Forecasting

Predict future observations (requires `internal_shingling = true` and `shingle_size > 1`):

```rust
let forest = Forest::builder(4, 8)
    .build()?;

// Feed observations one at a time
for point in stream {
    forest.update(&point)?;
}

// Predict the next 5 observations (look_ahead must be <= shingle_size)
let predictions = forest.extrapolate(5)?;
// Returns a flat list of length 5 * input_dim
```

### Serialization

Save and load trained models using JSON:

```rust
// Save to string
let json_str = forest.to_json()?;

// Save to file
forest.save_json("forest.json")?;

// Load from string
let loaded = Forest::from_json(&json_str)?;

// Load from file
let loaded = Forest::load_json("forest.json")?;
```

## Python API

The Python API mirrors the Rust interface. Create forests, update them, and compute scores exactly like in Rust:

### Creating a Forest

```python
from rcf3 import Forest

forest = Forest(
    input_dim=2,
    shingle_size=1,
    num_trees=50,
    capacity=256,
)
```

With time series:

```python
forest = Forest(
    input_dim=4,
    shingle_size=8,
    num_trees=100,
    capacity=512,
    time_decay=0.01,
    internal_shingling=True,
)
```

### Basic Operations

```python
# Update the forest
point = [1.5, 2.3]
forest.update(point)

# Check if ready
if forest.is_ready():
    score = forest.score(point)
    print(f"Anomaly score: {score}")

# Get the number of observations processed
print(f"Entries seen: {forest.entries_seen()}")
```

### Scoring Methods

```python
point = [1.5, 2.3, -0.5]

# Anomaly score
score = forest.score(point)

# Displacement score
displacement = forest.displacement_score(point)

# Density estimate
density = forest.density(point)
```

### Feature Attribution

```python
point = [1.5, 2.3, 100.0]
attribution = forest.attribution(point)

for i, attr in enumerate(attribution):
    print(f"Dimension {i}: below={attr['below']}, above={attr['above']}")
```

### Neighborhood Search

```python
point = [1.5, 2.3]
neighbors = forest.near_neighbors(point, top_k=10, percentile=50)

for neighbor in neighbors:
    print(f"Distance: {neighbor['distance']}")
    print(f"Score: {neighbor['score']}")
    print(f"Point: {neighbor['point']}")
```

### Missing Value Imputation

```python
point = [1.5, float('nan'), 3.0]
missing = [1]  # Index to impute
imputed = forest.impute(point, missing, centrality=1.0)

print(f"Imputed value: {imputed[1]}")
```

### Time Series Forecasting

```python
forest = Forest(
    input_dim=4,
    shingle_size=8,
    internal_shingling=True,
)

# Feed observations one at a time
for point in stream:
    forest.update(point)

# Predict next 5 observations
predictions = forest.extrapolate(5)
# Returns a list of length 5 * input_dim
```

### Serialization

```python
# Save to string
json_str = forest.to_json()

# Save to file
forest.save_json("forest.json")

# Load from string
loaded = Forest.from_json(json_str)

# Load from file
loaded = Forest.load_json("forest.json")
```

You can also use pickle for Python serialization:

```python
import pickle

# Save
with open("forest.pkl", "wb") as f:
    pickle.dump(forest, f)

# Load
with open("forest.pkl", "rb") as f:
    forest = pickle.load(f)
```

## Example: Detecting Anomalies in a Data Stream

### Rust

```rust
use rcf3::Forest;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut forest = Forest::builder(3, 1)
        .capacity(256)
        .num_trees(50)
        .build()?;

    // Warm up the forest with many normal data points
    for i in 0..200 {
        let val = (i as f32) * 0.01;
        forest.update(&vec![1.0 + val, 2.0 + val, 3.0 + val])?;
    }

    let data = vec![
        vec![1.0, 2.0, 3.0],
        vec![1.1, 2.1, 3.1],
        vec![1.2, 2.2, 3.2],
        vec![100.0, 200.0, 300.0], // Extreme anomaly
        vec![1.3, 2.3, 3.3],
    ];

    let mut anomaly_count = 0;
    for point in data {
        // Online inference order: score first, then update.
        if forest.is_ready() {
            let score = forest.score(&point)?;
            let attribution = forest.attribution(&point)?;

            println!("Point: {:?}, Score: {}", point, score);

            // Lower threshold since we're detecting a very extreme anomaly
            if score > 0.1 {
                println!("Anomaly detected: score={}", score);
                for (i, attr) in attribution.iter().enumerate() {
                    println!("  Dimension {}: {:.2}", i, attr.above);
                }
                anomaly_count += 1;
            }
        }

        forest.update(&point)?;
    }

    println!("Total anomalies detected: {}", anomaly_count);

    Ok(())
}
```

### Python

```python
from rcf3 import Forest

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
```

## License

Licensed under the Apache License 2.0.
