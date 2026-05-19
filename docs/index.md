# RCF3 documentation

`rcf3` exposes three public detector families:

- [Forest API](forest.md): Random Cut Forest for numerical streaming data, including anomaly scores, attribution, neighborhood search, imputation, forecasting, and serialization
- [OnlineIForest API](onlineiforest.md): Online Isolation Forest for numerical streams with sliding-window updates and preview scoring
- [MStream API](mstream.md): mixed numerical/categorical streaming anomaly detection with logical timestamps and decomposed scores

Start with the guide that matches the shape of your data:

- choose **Forest** for numerical observations and the full RCF feature set
- choose **OnlineIForest** for a compact numerical detector with update-after-learning scores
- choose **MStream** when events combine numerical and categorical aspects
