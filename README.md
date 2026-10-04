# Kinematic Head Impact Detection

Machine learning project for detecting real head impacts from kinematic data.

## Models

- HIKNet
- RecursiveNet
- SVM
- XGBoost (New Model)

## Dataset

The project uses a synthetic dataset with the same structure as the reference project:

- 527 samples
- 6 kinematic channels
- 199 time points per channel
- 264 real impacts
- 263 false positives

## Objective

To compare HIKNet, RecursiveNet, SVM, and XGBoost on the same reconstructed dataset. The synthetic data is not the original Stanford dataset used in the paper.

## Evaluation Metrics

- Accuracy
- Precision
- Specificity
- Sensitivity
- ROC-AUC
- PR-AUC

## Run the shared model comparison

Run the final experiment from the repository root:

```bash
python -m scripts.run_final_comparison
```

All four models use the same stratified 70/30 holdout split (seed 0 by default).
HIKNet and RecursiveNet use standardized raw signals and an additional
training-only validation split for early stopping. SVM and XGBoost use the same
176 extracted features. The script saves the metrics and split indices to
`results/final_comparison.json` and a table to `results/final_comparison.csv`.

The separate `scripts/train_svm.py` experiment uses the Wu 2017 feature
workbook (387 training samples, 411 features, and a separate test set); its
results must not be presented as an SVM evaluation on the synthetic 527-sample
dataset.
