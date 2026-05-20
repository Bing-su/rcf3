# OnlineIForest API

`OnlineIForest` is an Online Isolation Forest detector for numerical streams.
It keeps a sliding window of recent points and updates each tree incrementally
as new points arrive.

Use `update()` or `update_and_score()` to ingest observations. Use `score()` to
preview the current anomaly score for a point without mutating detector state.

## Configuration

| Parameter          | Type            | Default      | Description                                      |
| ------------------ | --------------- | ------------ | ------------------------------------------------ |
| `input_dim`        | `usize` / `int` | **Required** | Number of numerical features in each point       |
| `num_trees`        | `usize` / `int` | `32`         | Number of trees in the ensemble                  |
| `window_size`      | `usize` / `int` | `2048`       | Number of recent points retained by the detector |
| `max_leaf_samples` | `usize` / `int` | `32`         | Base leaf-splitting threshold                    |
| `seed`             | `u64` / `int`   | random       | Optional seed for deterministic trees            |

`window_size` must be greater than `max_leaf_samples`.

## Implementation choices

`OnlineIForest` keeps the Online Isolation Forest update structure, while
making a few practical choices around edge cases and API shape:

- Split sampling skips zero-width dimensions. The paper samples from every
  feature dimension, but dimensions whose minimum and maximum are equal cannot
  produce a useful split; a fully degenerate support region remains an unsplit
  leaf.
- When a split's synthetic samples all fall on one side, the paper leaves the
  empty-partition edge case implicit. This implementation preserves the
  geometric half-region for the empty child so both children keep valid support
  rectangles.
- The paper's streamed score is computed after the incoming point is learned.
  This library exposes both operations explicitly: `score(point)` previews the
  current forest without mutation, while `update_and_score(point)` learns the
  point first and then scores it.

## Rust API

### Creating a detector

```rust
use rcf3::OnlineIForest;

let mut detector = OnlineIForest::builder(2)
    .num_trees(32)
    .window_size(128)
    .max_leaf_samples(8)
    .seed(7)
    .build()?;
```

### Update and preview scoring

Use `update_and_score` when you want to ingest the point and receive its anomaly
score under the updated forest:

```rust
let score = detector.update_and_score(&[1.5, 2.3])?;
```

Use `score` to preview the current anomaly score without mutating detector
state:

```rust
let preview = detector.score(&[1.6, 2.4])?;
```

For `OnlineIForest`, preview scoring and committed scoring are intentionally
different operations: `score(point)` evaluates the current forest before the
point is learned, while `update_and_score(point)` learns the point first and
then scores it. Calling `update(point)` followed by `score(point)` returns the
same value as `update_and_score(point)` from the same starting state.

### Status accessors

```rust
assert!(detector.is_ready());
assert_eq!(detector.entries_seen(), 1);
assert_eq!(detector.num_trees(), 32);
```

`is_ready()` becomes `true` after the first processed point.

### Serialization

With the `serde` feature enabled:

```rust
let json = detector.to_json()?;
let restored = OnlineIForest::from_json(json)?;
```

With both `serde` and `std` enabled, file helpers are also available:

```rust
detector.save_json("onlineiforest.json")?;
let restored = OnlineIForest::load_json("onlineiforest.json")?;
```

### Practical example

```rust
use rcf3::OnlineIForest;

let mut detector = OnlineIForest::builder(2)
    .window_size(128)
    .max_leaf_samples(8)
    .seed(2026)
    .build()?;

for i in 0..64 {
    let value = (i as f32) * 0.01;
    detector.update(&[value, value + 1.0])?;
}

let normal_score = detector.score(&[0.5, 1.5])?;
let anomaly_score = detector.score(&[10.0, -10.0])?;

println!("normal={normal_score}, anomaly={anomaly_score}");
```

## Python API

### Creating a detector

```python
from rcf3 import OnlineIForest

detector = OnlineIForest(
    input_dim=2,
    num_trees=32,
    window_size=128,
    max_leaf_samples=8,
    seed=7,
)
```

### Update and preview scoring

```python
score = detector.update_and_score([1.5, 2.3])
preview = detector.score([1.6, 2.4])
```

For `OnlineIForest`, `score(point)` previews the current forest before the point
is learned. `update_and_score(point)` learns the point first and then scores it.

### Status accessors

```python
assert detector.is_ready()
assert detector.entries_seen() == 1
assert detector.num_trees() == 32
```

### Serialization

```python
json_str = detector.to_json()
restored = OnlineIForest.from_json(json_str)

detector.save_json("onlineiforest.json")
restored_from_file = OnlineIForest.load_json("onlineiforest.json")
```

`OnlineIForest` also supports normal Python pickle round-trips through its
JSON-backed state hooks.

### Practical example

```python
from rcf3 import OnlineIForest

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
```
