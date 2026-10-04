"""XGBoost binary classifier for kinematic head impact detection.

Mirrors the SVM baseline in src/baseline.py: both receive the same tabular
features produced by src/features.extract_features() and are evaluated on the
same 70/30 stratified train/eval split (seed=0) used throughout the project.
"""

from xgboost import XGBClassifier

# Project-wide random seed (matches preprocess.train_eval_split default)
SEED = 0


def build_xgboost(seed: int = SEED) -> XGBClassifier:
    """Return a reproducible XGBClassifier with sensible defaults for this
    small dataset (527 samples, ~200 engineered features).

    Hyperparameter rationale
    ------------------------
    n_estimators=200    – enough rounds for stable convergence on 370 training
                          samples without over-fitting; early stopping is not
                          used here to keep evaluation simple and comparable.
    max_depth=4         – shallow trees reduce variance on small data.
    learning_rate=0.05  – conservative rate pairs well with 200 rounds.
    subsample=0.8       – row sub-sampling adds regularisation.
    colsample_bytree=0.8 – column sub-sampling, mirrors scikit-learn's
                           `max_features` convention.
    eval_metric="logloss" – log-loss is well-suited to binary:logistic.
    use_label_encoder=False – avoids deprecation warnings in xgboost >= 1.6.
    """
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        use_label_encoder=False,
    )
