# ensembled-stream-classification
Review of ensembled learning methods for data stream classification

- `AccuracyWeightedEnsembleClassifier`
- `BatchIncrementalClassifier`
- `DynamicWeightedMajorityClassifier`
- Custom implementation of ACE/AEC, utilizing classes of scikit-multiflow
- Some ensemble with no detection w/ and w/out DDM (Dedicated Drift Method)


# Plan
- Batch classifier always removes oldest expert, AWE and DWM classifiers remove weakest experts. We can simulate this and compare.
- They're supposed to handle gradua drifts well, as they store data and gradually upgrade their new experts (check how much weaker for sudden drifts)
- Drift detection: EDDM, ADWIN (ADWIN przetrzymuje dane)

## Draft
- Three phases:
    - [-] Analyse stream data
        - [-] Plot how distribution changes on SEA Generator with sudden drifts
        - [-] Plot other streams to show how it changes, compare with no drifts where possible
    - [-] Prepare baseline models (straightforward ensemble)
    - [-] Compare it to a single classifier
    - [-] Show how concept drifts affect it
    - [-] Improve the baseline
    - [-] Detect concept drift
    - [-] Detect drift dynamic (sudden/graduate)
    - [-] Use proper drift alignment strategy based on detected dynamic (replace experts, replace an expert, re-learn on new batch, do nothing, ...)
    - [-] Results: median, average, minmax, variance, g-mean, maybe kappa (?)