# RCF3 documentation

`rcf3` exposes two public detector families:

- [Forest API](forest.md): Random Cut Forest for numerical streaming data, including anomaly scores, attribution, neighborhood search, imputation, forecasting, and serialization
- [MStream API](mstream.md): mixed numerical/categorical streaming anomaly detection with logical timestamps and decomposed scores

Start with the guide that matches the shape of your data:

- choose **Forest** for numerical observations and the full RCF feature set
- choose **MStream** when events combine numerical and categorical aspects
