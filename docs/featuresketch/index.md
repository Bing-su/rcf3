# FeatureSketch API

`FeatureSketch` is a sparse streaming anomaly detector for events whose feature
names can change over time. It is useful for logs, counters, request metadata,
or one-hot encoded categories where new keys may appear and old keys may stop
appearing.

This detector is an independent `rcf3` implementation. It is inspired by sparse
projection and sketch-based stream detectors, but it is not a paper-faithful
implementation of xStream or any other specific algorithm. The design notes and
literature background live in [research.md](research.md).

## Event Format

Each event is a sparse collection of `(feature_name, value)` pairs:

- feature names are strings
- values must be finite numbers
- duplicate feature names are summed before scoring or update
- an empty event is valid
- missing and zero-valued features are different

A feature with value `0.0` is still present. It contributes to the feature-set
signal even though its numeric magnitude is zero. For categorical values, encode
only active categories:

```text
status:401 -> 1.0
endpoint:/login -> 1.0
```

Do not emit inactive categories as `0.0` unless you intentionally want them to
count as observed features.

## Configuration

| Parameter                | Rust builder method        | Python parameter           | Default | Description                                   | Constraints    |
| ------------------------ | -------------------------- | -------------------------- | ------: | --------------------------------------------- | -------------- |
| Value projection dims    | `value_projection_dims`    | `value_projection_dims`    |      32 | Random projection dimensions for values       | Must be `> 0`  |
| Presence projection dims | `presence_projection_dims` | `presence_projection_dims` |      32 | Random projection dimensions for feature keys | Must be `> 0`  |
| Chains per ensemble      | `chains_per_ensemble`      | `chains_per_ensemble`      |      16 | Number of sketch chains per signal            | Must be `> 0`  |
| Chain depth              | `chain_depth`              | `chain_depth`              |       8 | Multi-scale binning depth                     | Must be `> 0`  |
| Sketch rows              | `sketch_rows`              | `sketch_rows`              |       2 | Count-min rows per chain level                | Must be `> 0`  |
| Sketch buckets           | `sketch_buckets`           | `sketch_buckets`           |    2048 | Buckets per count-min row                     | Must be `>= 2` |
| Decay half-life          | `decay_half_life`          | `decay_half_life`          |    2048 | Event-count half-life for adaptation          | Must be `> 0`  |
| Seed                     | `seed`                     | `seed`                     |  random | Deterministic projection and sketch layout    | Optional       |

Larger projection dimensions, more chains, deeper chains, more rows, and more
buckets can improve stability or reduce sketch collisions, but they increase
CPU and memory cost. A shorter half-life adapts faster to changing streams; a
longer half-life keeps older behavior influential for longer.

## Implementation Choices

`FeatureSketch` keeps no feature-name registry. Feature names are hashed into a
fixed projection and sketch layout, so long-running memory does not grow with
the number of names seen historically.

The detector maintains two internal signals:

- value projections for unusual numeric magnitudes
- presence projections for unusual observed feature sets

Presence tracking is why feature shrink is visible: if a previously common key
disappears from an event, the feature set changes even when remaining numeric
values look ordinary.

Values are normalized with `asinh(x)` before projection, so all finite signed
values are accepted, including large positive and negative values. The anomaly
score is a sketch-density surprise score; higher values mean more anomalous
events. It is useful for ranking or thresholding within one configured detector,
not as a calibrated probability.

## Rust API

### Creating a detector

```rust
use rcf3::FeatureSketch;

let mut detector = FeatureSketch::builder()
    .value_projection_dims(32)
    .presence_projection_dims(32)
    .chains_per_ensemble(16)
    .chain_depth(8)
    .sketch_rows(2)
    .sketch_buckets(2048)
    .decay_half_life(2048)
    .seed(42)
    .build()?;
```

### Update and preview scoring

Use `update_and_score` when you want the pre-ingest anomaly score and want to
commit the same event:

```rust
let score = detector.update_and_score([
    ("endpoint:/login", 1.0),
    ("status:200", 1.0),
    ("bytes", 812.0),
])?;
assert!(score >= 0.0);
```

Use `score` to preview an event without mutating detector state:

```rust
let event = [
    ("endpoint:/admin", 1.0),
    ("status:401", 1.0),
    ("bytes", 12000.0),
];

let preview = detector.score(event)?;
let committed = detector.update_and_score(event)?;
assert_eq!(preview, committed);
```

Use `update` when you want to ingest without returning the score.

### Status accessors

```rust
assert!(detector.is_ready());
assert_eq!(detector.entries_seen(), 2);
```

`is_ready()` becomes `true` after the first processed event. Early scores are
still warmup scores; production pipelines commonly ignore an initial warmup
period before thresholding.

### Serialization

With the `serde` feature enabled:

```rust
let json = detector.to_json()?;
let restored = FeatureSketch::from_json(json)?;
```

With both `serde` and `std` enabled, file helpers are also available:

```rust
detector.save_json("featuresketch.json")?;
let restored_from_file = FeatureSketch::load_json("featuresketch.json")?;
```

### Practical example

```rust
use rcf3::FeatureSketch;

let mut detector = FeatureSketch::builder()
    .seed(2026)
    .sketch_buckets(512)
    .build()?;

for _ in 0..64 {
    detector.update([
        ("endpoint:/login", 1.0),
        ("status:200", 1.0),
        ("bytes", 750.0),
    ])?;
}

let normal = detector.score([
    ("endpoint:/login", 1.0),
    ("status:200", 1.0),
    ("bytes", 790.0),
])?;
let suspicious = detector.score([
    ("endpoint:/admin", 1.0),
    ("status:401", 1.0),
    ("bytes", 12000.0),
])?;

println!("normal={normal}, suspicious={suspicious}");
```

## Python API

### Creating a detector

```python
from rcf3 import FeatureSketch

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
```

### Update and preview scoring

Python accepts either a mapping or a sequence of `(name, value)` pairs:

```python
score = detector.update_and_score({
    "endpoint:/login": 1.0,
    "status:200": 1.0,
    "bytes": 812.0,
})
assert score >= 0.0

event = {
    "endpoint:/admin": 1.0,
    "status:401": 1.0,
    "bytes": 12000.0,
}
preview = detector.score(event)
committed = detector.update_and_score(event)
assert preview == committed
```

Use `update(event)` when you want to ingest without returning the score.

### Status accessors

```python
assert detector.is_ready()
assert detector.entries_seen() == 2
```

### Serialization

```python
json_str = detector.to_json()
restored = FeatureSketch.from_json(json_str)

detector.save_json("featuresketch.json")
restored_from_file = FeatureSketch.load_json("featuresketch.json")
```

`FeatureSketch` also supports normal Python pickle round-trips through its
JSON-backed state hooks.

### Practical example

```python
from rcf3 import FeatureSketch

detector = FeatureSketch(seed=2026, sketch_buckets=512)

for _ in range(64):
    detector.update({
        "endpoint:/login": 1.0,
        "status:200": 1.0,
        "bytes": 750.0,
    })

normal = detector.score({
    "endpoint:/login": 1.0,
    "status:200": 1.0,
    "bytes": 790.0,
})
suspicious = detector.score({
    "endpoint:/admin": 1.0,
    "status:401": 1.0,
    "bytes": 12000.0,
})

print(f"normal={normal}, suspicious={suspicious}")
```
