# Rationale and peer references

The 2026-10-11 OpenI review found two complete same-topic projects:
[djk/SSME_lite](https://openi.pcl.ac.cn/djk/SSME_lite) and
[hearin/SSME-Lite](https://openi.pcl.ac.cn/hearin/SSME-Lite).
[yaoyc/SSME-Lite](https://openi.pcl.ac.cn/yaoyc/SSME-Lite) is the engineering
contribution snapshot of the first team, not an independent complete package.

The user selected four improvements after that review: measured scalability,
configuration of fitted model files, label-efficiency evaluation, and multiclass/
redundancy stress. The first project's configuration, caching and measurement
work and the second project's scaling records, label-efficiency questions and
negative results motivated the study questions. This implementation follows
our existing interfaces and contains no copied peer source.

The current package already had ranking, interactive reports, controlled
leave-one-model-out influence and label-budget experiments. Those were retained;
new studies add fixed-target comparisons, explicit performance evidence and
stress/failure records. Bootstrap coverage, adaptive estimators and crossfit
were not selected for this implementation.

Peer performance/error numbers are not treated as directly comparable to our
measurements because the algorithms, hardware, input distributions and targets
differ. Our full protocol and results are saved alongside this file.
