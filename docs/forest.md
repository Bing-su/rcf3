# Forest API

`Forest` is the Random Cut Forest detector in `rcf3`. It operates on numerical observations and supports anomaly detection, feature attribution, neighborhood search, missing-value imputation, and time-series forecasting.

## Configuration

| Parameter                 | Type            | Default      | Description                                                                          |
| ------------------------- | --------------- | ------------ | ------------------------------------------------------------------------------------ |
| `input_dim`               | `usize` / `int` | **Required** | Number of base feature dimensions per observation                                    |
| `shingle_size`            | `usize` / `int` | `1`          | Temporal window size                                                                 |
| `capacity`                | `usize` / `int` | `256`        | Maximum number of points stored per tree                                             |
| `num_trees`               | `usize` / `int` | `50`         | Number of trees in the ensemble                                                      |
| `time_decay`              | `f64` / `float` | `0.0`        | Exponential time-decay rate; `0.0` uses the automatic default                        |
| `output_after`            | `usize` / `int` | `0`          | Minimum number of updates before non-trivial outputs; `0` uses the automatic default |
| `internal_shingling`      | `bool`          | `true`       | Whether the forest manages its rolling shingle buffer internally                     |
| `initial_accept_fraction` | `f64`           | `0.125`      | Warm-up sampling behavior                                                            |

## Rust API

### Creating a forest

```rust
use rcf3::Forest;

let forest = Forest::builder(2)
    .shingle_size(1)
    .num_trees(50)
    .capacity(256)
    .build()?;
```

With time-series shingling:

```rust
let forest = Forest::builder(4)
    .shingle_size(8)
    .num_trees(100)
    .capacity(512)
    .time_decay(0.01)
    .build()?;
```

From a config object:

```rust
use rcf3::{Forest, RcfConfig};

let config = RcfConfig::new(3)
    .with_num_trees(75)
    .with_capacity(512)
    .with_shingle_size(4);

let forest = Forest::from_config(&config)?;
```

### Basic operations

For online anomaly detection, score first and then update:

```rust
let point = vec![1.5, 2.3];

if forest.is_ready() {
    let score = forest.score(&point)?;
    println!("Anomaly score: {score}");
}

forest.update(&point)?;
println!("Entries seen: {}", forest.entries_seen());
```

### Scoring and interpretation

```rust
let point = vec![1.5, 2.3, -0.5];
let score = forest.score(&point)?;
let displacement = forest.displacement_score(&point)?;
let density = forest.density(&point)?;
```

Feature attribution:

```rust
let point = vec![1.5, 2.3, 100.0];
let attribution = forest.attribution(&point)?;

for (i, attr) in attribution.iter().enumerate() {
    println!("Dimension {i}: below={}, above={}", attr.below, attr.above);
}
```

- `above`: contribution from cuts above the query value
- `below`: contribution from cuts below the query value

### Neighborhood search

```rust
let neighbors = forest.near_neighbors(&[1.5, 2.3], 10, 50)?;

for neighbor in neighbors {
    println!("distance={}, score={}", neighbor.distance, neighbor.score);
}
```

### Missing-value imputation

```rust
let point = vec![1.5, f32::NAN, 3.0];
let missing = vec![1];
let imputed = forest.impute(&point, &missing, 1.0)?;
```

### Time-series forecasting

Forecasting requires `internal_shingling = true` and `shingle_size > 1`:

```rust
let mut forest = Forest::builder(4)
    .shingle_size(8)
    .build()?;

for point in stream {
    forest.update(&point)?;
}

let predictions = forest.extrapolate(5)?;
```

`extrapolate(look_ahead)` returns a flat vector of length `look_ahead * input_dim`.

### Serialization

```rust
let json_str = forest.to_json()?;
forest.save_json("forest.json")?;

let loaded = Forest::from_json(&json_str)?;
let loaded_from_file = Forest::load_json("forest.json")?;
```

## Python API

### Creating a forest

```python
from rcf3 import Forest

forest = Forest(
    input_dim=2,
    shingle_size=1,
    num_trees=50,
    capacity=256,
)
```

With time-series shingling:

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

### Basic operations

```python
point = [1.5, 2.3]

if forest.is_ready():
    score = forest.score(point)
    print(f"Anomaly score: {score}")

forest.update(point)
print(f"Entries seen: {forest.entries_seen()}")
```

### Scoring and interpretation

```python
point = [1.5, 2.3, -0.5]

score = forest.score(point)
displacement = forest.displacement_score(point)
density = forest.density(point)
```

Feature attribution:

```python
point = [1.5, 2.3, 100.0]
attribution = forest.attribution(point)

for i, attr in enumerate(attribution):
    print(f"Dimension {i}: below={attr['below']}, above={attr['above']}")
```

### Neighborhood search

```python
neighbors = forest.near_neighbors([1.5, 2.3], top_k=10, percentile=50)
```

### Missing-value imputation

```python
point = [1.5, float("nan"), 3.0]
missing = [1]
imputed = forest.impute(point, missing, centrality=1.0)
```

### Time-series forecasting

```python
forest = Forest(input_dim=4, shingle_size=8, internal_shingling=True)

for point in stream:
    forest.update(point)

predictions = forest.extrapolate(5)
```

### Serialization

```python
json_str = forest.to_json()
forest.save_json("forest.json")

loaded = Forest.from_json(json_str)
loaded_from_file = Forest.load_json("forest.json")
```

Python objects also support pickle round-trips.

## End-to-end anomaly example

### Rust

```rust
use rcf3::Forest;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut forest = Forest::builder(3)
        .capacity(256)
        .num_trees(50)
        .build()?;

    for i in 0..200 {
        let val = (i as f32) * 0.01;
        forest.update(&[1.0 + val, 2.0 + val, 3.0 + val])?;
    }

    let data = vec![
        vec![1.0, 2.0, 3.0],
        vec![1.1, 2.1, 3.1],
        vec![100.0, 200.0, 300.0],
    ];

    for point in data {
        if forest.is_ready() {
            let score = forest.score(&point)?;
            println!("Point: {point:?}, score={score}");
        }
        forest.update(&point)?;
    }

    Ok(())
}
```

### Python

```python
from rcf3 import Forest

forest = Forest(input_dim=3, capacity=256, num_trees=50)

for i in range(200):
    val = i * 0.01
    forest.update([1.0 + val, 2.0 + val, 3.0 + val])

for point in ([1.0, 2.0, 3.0], [1.1, 2.1, 3.1], [100.0, 200.0, 300.0]):
    if forest.is_ready():
        print(f"Point: {point}, score={forest.score(point)}")
    forest.update(point)
```
