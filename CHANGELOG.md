## [0.4.0] - 2026-05-24

### 🚀 Features

- Adds Online Isolation Forest detector (#8)
- _(rcf)_ [**breaking**] Tighten RCF public API and lazy update storage (#11)
- _(rcf)_ [**breaking**] Preflight forest updates and split tests (#15)
- _(error)_ [**breaking**] Classify serialization and tree-state failures as runtime
- _(api)_ [**breaking**] Narrow public facade and clarify input errors
- [**breaking**] Make error module private

### 🐛 Bug Fixes

- Reduce memory allocation for RCF, etc (#10)

### 💼 Other

- _(deps)_ Bump the default group with 2 updates (#9)

### 🚜 Refactor

- _(bench)_ Clarify benchmark setup and cases (#12)
- Centralize std and no_std math wrappers (#14)

### 📚 Documentation

- Add badges
- Standardize configuration constraints in documentation

### ⚙️ Miscellaneous Tasks

- Add oxfmt pre-commit hook
- Add continuous benchmarking workflow with Bencher
- Update trigger paths for continuous benchmarking workflow
- Fix bencher workflow
- Enable manual triggering for the Bencher workflow
- Add git-cliff config
- Release version 0.4.0

## [0.3.0] - 2026-05-18

### 🚀 Features

- Add mStream (#4)
- Add documentation workflow and initial docs structure (#5)

### 🐛 Bug Fixes

- Generalize JSON serialization and deserialization APIs
- Reorganize and document near-neighbor types in forest.rs
- Enhance MStream to support negative numeric values (#6)

### 🚜 Refactor

- Consolidate RCF modules and extract Python utilities (#7)

### 📚 Documentation

- Update zensical config

### 🧪 Testing

- Add comprehensive property-based tests with proptest

### ⚙️ Miscellaneous Tasks

- Change license to MIT OR Apache 2.0
- Move python test files
- Remove pprof profiler
- Update project urls
- Bump version to 0.3.0, update license, and adjust CI paths

## [0.2.0] - 2026-05-14

### 🚀 Features

- [**breaking**] Fix ForestBuilder API for shingle_size

### 🐛 Bug Fixes

- Gate serialization test on `serde` feature
- Enhance api documentation and clarity
- Remove ordered_float crate

### 💼 Other

- _(deps)_ Bump https://github.com/tombi-toml/tombi-pre-commit from v0.11.1 to 0.11.3 in the default group (#3)

### ⚡ Performance

- Optimize core algorithm hot paths (#2)

### 🧪 Testing

- Fix no-default-features test on windows

### ⚙️ Miscellaneous Tasks

- Fix crate keywords
- Generate pypi publish attestations
- Publish Rust crate to crates.io
- Consolidate PyPI attestation generation
- Enhance release workflow verbosity and reliability
- Add pre-commit linting workflow and dependabot config
- V0.2.0
- Fix pypi attestation

## [0.1.0] - 2026-05-13

### 🚀 Features

- Initial autopilot
- Decouple from burn and parallelize RCF operations
- String-based serialization for Forest
- Optimize impute method and add benchmark
- Improve PyForest Python API with pickling, copying, and better defaults
- Make Python bindings an optional Cargo feature
- Make serde optional
- Use structs for NeighborCandidate, NeighborResult, and Attribution
- Add type stub for python rcf3 module
- Change internel rng
- Enhance Attribution struct ergonomics and safety
- Remove rayon dependency and parallel processing
- Make crate no-std compatible (#1)
- Add comprehensive README and verify examples

### 🐛 Bug Fixes

- Remove unused variable
- Improve file path handling
- Rename python module
- Remove unnecessary #[inline] attributes

### 💼 Other

- Optimize release build profile
- Add Criterion benchmarks and PProf profiler
- Centralize serde feature definitions and add README
- Configure minimal dependency features and optimize RNG

### 🚜 Refactor

- Optimize point access and clarify float types
- Improve core modules with helper functions and robust float comparisons
- Switch PointStore internal storage to ndarray::Array2
- Improve numerical precision and streamline RNG management
- Simplify point slice access in PointStore
- Extract RcfTree empty state check

### 🎨 Styling

- Add visual separators for Python magic methods

### 🧪 Testing

- Consolidate and improve tests with rstest
- Add anomaly detection simulation tests
- Python tests and fixes

### ⚙️ Miscellaneous Tasks

- Rename arcf -> rcf3
- Update python metadata
- Add pre-commit hooks
- Add workflows, nox
- Fix failed tests
- Add release workflow
- Configure Rust targets for wheel builds and optimize sdist runner
- Fix python build profile, readme
- Fix manylinux version, publish dry-run
- Switch to uv for PyPI publication
