# rcf3

`rcf3` is a Rust implementation of Random Cut Forest (RCF) for anomaly detection in streaming data, with Python bindings and an mStream detector for mixed numerical and categorical streams.

## What it provides

- **Random Cut Forest** for online anomaly detection, feature attribution, neighborhood search, missing-value imputation, and time-series forecasting
- **mStream** for mixed numerical/categorical streaming anomaly detection with feature-level score decomposition
- **Rust and Python APIs** over the same core implementation

## Documentation

- [Documentation index](docs/README.md)
- [Forest API guide](docs/forest.md)
- [MStream API guide](docs/mstream.md)

## Quick orientation

Use `Forest` when your stream is represented as numerical observations and you want the broader RCF feature set:

```rust
use rcf3::Forest;

let mut forest = Forest::builder(2).build()?;
forest.update(&[1.5, 2.3])?;
```

Use `MStream` when each event has separate numerical and categorical aspects:

```rust
use rcf3::MStream;

let mut detector = MStream::builder(2, 1).build()?;
let score = detector.update_and_score(&[1.5, 2.0], &[7], 1)?;
```

## Features

The crate supports several compile-time features:

### `std` (enabled by default)

Enables use of the Rust standard library. Disable this for `no_std` environments:

```toml
[dependencies]
rcf3 = { version = "0.1", default-features = false }
```

### `serde` (enabled by default)

Provides JSON serialization and deserialization support for persisted detector state:

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

## License

Licensed under the Apache License 2.0.
