# MStream API

`MStream` implements the multi-aspect streaming anomaly detector described in the mStream paper. It is designed for events that contain separate numerical and categorical features, such as login traffic with byte counts plus country or endpoint IDs.

## Configuration

| Parameter         | Type            | Default      | Description                                   |
| ----------------- | --------------- | ------------ | --------------------------------------------- |
| `numeric_dim`     | `usize` / `int` | **Required** | Number of numerical features in each record   |
| `categorical_dim` | `usize` / `int` | **Required** | Number of categorical features in each record |
| `num_rows`        | `usize` / `int` | `2`          | Number of hash rows                           |
| `num_buckets`     | `usize` / `int` | `1024`       | Number of buckets per hash row                |
| `alpha`           | `f64` / `float` | `0.8`        | Temporal decay factor in `(0, 1)`             |
| `seed`            | `u64` / `int`   | random       | Optional seed for deterministic hashing       |

At least one of `numeric_dim` or `categorical_dim` must be greater than zero.

## Timestamp semantics

`timestamp` is a logical tick index, not wall-clock time. Only differences between timestamps matter.

- shifting all timestamps by a constant leaves scores unchanged
- a gap of `k` ticks applies the temporal decay factor `alpha` exactly `k` times
- timestamps must be positive and monotonically non-decreasing

For paper-style usage, pass the same timestamp for all records in the same tick, then increment it when the stream advances.

## Rust API

### Creating a detector

```rust
use rcf3::MStream;

let mut detector = MStream::builder(2, 1)
    .alpha(0.8)
    .num_rows(2)
    .num_buckets(1024)
    .seed(7)
    .build()?;
```

### Update and preview scoring

Use `update_and_score` when you want to ingest the event and receive its anomaly score:

```rust
let score = detector.update_and_score(&[1.5, 2.0], &[7], 1)?;
```

Use `score` to preview what the next inserted record would score without mutating detector state:

```rust
let preview = detector.score(&[1.5, 2.0], &[7], 2)?;
let committed = detector.update_and_score(&[1.5, 2.0], &[7], 2)?;
assert_eq!(preview, committed);
```

Use `update` when you want to ingest without returning the score.

### Detailed scores

The final score is the sum of one record-level contribution plus one contribution per numerical and categorical feature:

```rust
let detailed = detector.score_detailed(&[1.5, 2.0], &[7], 3)?;
assert_eq!(detailed.numeric_features.len(), 2);
assert_eq!(detailed.categorical_features.len(), 1);
```

`MStreamScore` contains:

- `total`
- `record`
- `numeric_features`
- `categorical_features`

Use `update_and_score_detailed` when you want that decomposition for a committed insert.

### Status accessors

```rust
assert!(detector.is_ready());
assert_eq!(detector.entries_seen(), 2);
assert_eq!(detector.current_time(), Some(2));
```

`is_ready()` becomes `true` after the first processed record.

### Serialization

With the `serde` feature enabled:

```rust
let json = detector.to_json()?;
let restored = MStream::from_json(json)?;
```

With both `serde` and `std` enabled, file helpers are also available:

```rust
detector.save_json("mstream.json")?;
let restored = MStream::load_json("mstream.json")?;
```

### Practical example

```rust
use rcf3::MStream;

let mut detector = MStream::builder(2, 2)
    .seed(2026)
    .num_buckets(512)
    .build()?;

let normal = detector.update_and_score(&[0.0, 3.2], &[1, 10], 1)?;
let suspicious = detector.score_detailed(&[12.0, 0.3], &[99, 10], 2)?;

println!("normal={normal}, suspicious={}", suspicious.total);
println!("failed-attempt contribution={}", suspicious.numeric_features[0]);
println!("country contribution={}", suspicious.categorical_features[0]);
```

## Python API

### Creating a detector

```python
from rcf3 import MStream

detector = MStream(
    numeric_dim=2,
    categorical_dim=1,
    alpha=0.8,
    num_rows=2,
    num_buckets=1024,
    seed=7,
)
```

### Update and preview scoring

```python
score = detector.update_and_score([1.5, 2.0], [7], 1)

preview = detector.score([1.5, 2.0], [7], 2)
committed = detector.update_and_score([1.5, 2.0], [7], 2)
assert preview == committed
```

### Detailed scores

Python returns the decomposition as a dict-like object:

```python
detailed = detector.score_detailed([1.5, 2.0], [7], 3)
assert len(detailed["numeric_features"]) == 2
assert len(detailed["categorical_features"]) == 1
```

The available keys are:

- `total`
- `record`
- `numeric_features`
- `categorical_features`

### Status accessors

```python
assert detector.is_ready()
assert detector.entries_seen() == 2
assert detector.current_time() == 2
```

### Serialization

```python
json_str = detector.to_json()
restored = MStream.from_json(json_str)

detector.save_json("mstream.json")
restored_from_file = MStream.load_json("mstream.json")
```

`MStream` also supports normal Python pickle round-trips through its JSON-backed state hooks.

### Practical example

```python
from rcf3 import MStream

detector = MStream(numeric_dim=2, categorical_dim=2, seed=2026, num_buckets=512)

normal = detector.update_and_score([0.0, 3.2], [1, 10], 1)
suspicious = detector.score_detailed([12.0, 0.3], [99, 10], 2)

print(f"normal={normal}, suspicious={suspicious['total']}")
print(f"failed-attempt contribution={suspicious['numeric_features'][0]}")
print(f"country contribution={suspicious['categorical_features'][0]}")
```
