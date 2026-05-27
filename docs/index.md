# RCF3 documentation

[![Crates.io](https://img.shields.io/crates/v/rcf3)](https://crates.io/crates/rcf3)
[![PyPI](https://img.shields.io/pypi/v/rcf3)](https://pypi.org/project/rcf3/)
[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://bing-su.github.io/rcf3/)
![License](https://img.shields.io/crates/l/rcf3)

`rcf3` exposes the following public detector families:

- [Forest API](forest.md): Random Cut Forest for numerical streaming data, including anomaly scores, attribution, neighborhood search, imputation, forecasting, and serialization
- [OnlineIForest API](onlineiforest.md): Online Isolation Forest for numerical streams with sliding-window updates and preview scoring
- [MStream API](mstream.md): mixed numerical/categorical streaming anomaly detection with logical timestamps and decomposed scores
- [FeatureSketch API](featuresketch/index.md): sparse feature-name anomaly detection for streams whose schema can grow or shrink over time

Start with the guide that matches the shape of your data:

- choose **Forest** for numerical observations and the full RCF feature set
- choose **OnlineIForest** for a compact numerical detector with update-after-learning scores
- choose **MStream** when events combine numerical and categorical aspects
- choose **FeatureSketch** when each event is a sparse set of named features and the feature universe is not fixed
