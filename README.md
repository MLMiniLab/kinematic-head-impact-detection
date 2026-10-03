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

To reproduce the models from the reference work and evaluate a new XGBoost model using the same dataset and evaluation metrics.

## Evaluation Metrics

- Accuracy
- Precision
- Specificity
- Sensitivity
- ROC-AUC
- PR-AUC
