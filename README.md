# RCF3

[![Crates.io](https://img.shields.io/crates/v/rcf3)](https://crates.io/crates/rcf3)
[![PyPI](https://img.shields.io/pypi/v/rcf3)](https://pypi.org/project/rcf3/)
[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://bing-su.github.io/rcf3/)
[![License](https://img.shields.io/crates/l/rcf3)](https://github.com/Bing-su/rcf3/blob/main/LICENSE-Apache)

`rcf3` provides streaming anomaly detectors in Rust with Python bindings.

## What it provides

- **Random Cut Forest** for online anomaly detection, feature attribution, neighborhood search, missing-value imputation, and time-series forecasting
- **Online Isolation Forest** for numerical streams with sliding-window updates
- **mStream** for mixed numerical/categorical streaming anomaly detection with feature-level score decomposition
- **FeatureSketch** for sparse feature-name streams whose schema can grow or shrink over time
- **Rust and Python APIs** over the same core implementation

## Documentation

- [Documentation index](https://bing-su.github.io/rcf3/)
- [Forest API guide](https://bing-su.github.io/rcf3/forest/)
- [OnlineIForest API guide](https://bing-su.github.io/rcf3/onlineiforest/)
- [MStream API guide](https://bing-su.github.io/rcf3/mstream/)
- [FeatureSketch API guide](https://bing-su.github.io/rcf3/featuresketch/)

## Quick orientation

Use `Forest` when your stream is represented as numerical observations and you want the broader RCF feature set:

```rust
use rcf3::Forest;

let mut forest = Forest::builder(2).build()?;
forest.update(&[1.5, 2.3])?;
```

Use `OnlineIForest` when you want a compact numerical detector with sliding-window updates:

```rust
use rcf3::OnlineIForest;

let mut detector = OnlineIForest::builder(2).build()?;
let score = detector.update_and_score(&[1.5, 2.3])?;
```

Use `MStream` when each event has separate numerical and categorical aspects:

```rust
use rcf3::MStream;

let mut detector = MStream::builder(2, 1).build()?;
let score = detector.update_and_score(&[1.5, 2.0], &[7], 1)?;
```

Use `FeatureSketch` when each event is a sparse set of named features and the feature universe is not fixed:

```rust
use rcf3::FeatureSketch;

let mut detector = FeatureSketch::builder().build()?;
let score = detector.update_and_score([
    ("endpoint:/login", 1.0),
    ("status:401", 1.0),
    ("bytes", 812.0),
])?;
```

## Features

The crate supports several compile-time features:

### `std` (enabled by default)

Enables use of the Rust standard library. Disable this for `no_std` environments:

```toml
[dependencies]
rcf3 = { version = "0.4", default-features = false }
```

### `serde` (enabled by default)

Provides JSON serialization and deserialization support for persisted detector state:

```toml
[dependencies]
rcf3 = { version = "0.4", features = ["serde", "std"] }
```

### `python` (optional)

Builds Python bindings using PyO3, enabling use from Python. Automatically enables `serde` and `std`:

```toml
[dependencies]
rcf3 = { version = "0.4", features = ["python"] }
```

To use just the core algorithm without serialization:

```toml
[dependencies]
rcf3 = { version = "0.4", default-features = false, features = ["std"] }
```

## License

Licensed under either of the Apache License 2.0 or MIT license, at your option.
