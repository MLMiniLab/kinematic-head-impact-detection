# Kinematic Head Impact Detection

This project compares machine-learning models for classifying kinematic
recordings as real head impacts or false positives. It includes HIKNet,
RecursiveNet, SVM, and XGBoost.

## Project goal

The project aims to understand and reproduce the experiments in the HIKNet
reference work as closely as the available data permits, then evaluate
XGBoost as an additional model.

The original paper describes 527 recordings collected from Stanford football
athletes using an instrumented mouthguard. Those original recordings are not
available in this repository. Our main comparison therefore uses synthetic
data created by the code in this project. The synthetic data matches the
stated sample count, signal dimensions, and class balance, but it is not the
original Stanford data and should not be described as a direct reproduction
on that data.

## Datasets

### Synthetic kinematic dataset

The current files are `data/data.mat` and `data/labels.mat`.

- 527 samples
- 6 kinematic channels per sample
- 199 time points per channel
- 264 impact labels (`1`)
- 263 false-positive labels (`0`)
- Nominal sampling frequency: 1000 Hz

The data generator is `src/mock.py`. Each channel is formed from simulated
damped sinusoidal pulses, a higher-frequency component, and random noise.
Different assumed frequency, decay, and amplitude ranges are used for
impact-like and false-positive-like signals. A small number of samples are
given a signal from the opposite class to create overlap. These are
simulation assumptions, not measurements from Stanford athletes.

The current `.mat` files match the output of this generator with seed 0.
To regenerate them, run:

```bash
python -m scripts.make_mock_data --seed 0
```

### Separate Wu 2017 feature dataset

`data/wu2017_features.xlsx` is used by a separate SVM workflow. It contains
387 training examples with 411 features and a separate 32-example test set.
This is not the dataset used for the final four-model comparison. Do not
combine its SVM result with the synthetic-data results as if all models used
the same examples.

## Data preparation

The six channels are linear acceleration in the X, Y, and Z directions,
followed by angular velocity in X, Y, and Z. The MATLAB files store signals
as `(samples, 199, 6)`; `src/data.py` loads and transposes them to
`(samples, 6, 199)`.

HIKNet and RecursiveNet use the raw time-series after standardizing each
sample and channel over its 199 time points.

SVM and XGBoost use the same 176 engineered features from
`src/features.py`:

| Feature group | Number |
|---|---:|
| Peak values for linear velocity, linear acceleration, angular velocity, and angular acceleration, including X, Y, Z, and magnitude | 16 |
| Half-maximum duration for linear and angular acceleration, including X, Y, Z, and magnitude | 8 |
| Linear-acceleration power spectral density at 10 to 200 Hz in 10 Hz steps, for four axes/magnitudes | 80 |
| Angular-acceleration power spectral density at 10 to 180 Hz in 10 Hz steps, for four axes/magnitudes | 72 |
| Total | 176 |

Linear velocity is estimated by integrating linear acceleration. Angular
acceleration is estimated by differentiating angular velocity. The SVM
pipeline standardizes features using `StandardScaler`, fitted on the training
data. XGBoost uses the extracted features without this scaler.

## Final evaluation protocol

Run `scripts/run_final_comparison.py` to perform the final comparison.

- Outer split: stratified 70% training and 30% evaluation
- Default random seed: 0
- Training examples: 369
- Held-out evaluation examples: 158, with 79 examples from each class
- The same outer training and evaluation examples are used for all four models
- HIKNet and RecursiveNet reserve 15% of the outer training data for
  early-stopping validation; this validation data is not the final evaluation
  set
- The JSON output records the sample indices for both sets

The neural networks train on the remaining 313 training examples after the
validation split. SVM and XGBoost train on all 369 outer training examples.
SVM and XGBoost both use the same 176 features.

This reported comparison uses one random split. Results can vary with a
different seed and should be treated as a single-run comparison, not as a
multi-run estimate of expected performance.

## Models and settings

| Model | Input | Main settings |
|---|---|---|
| HIKNet | Standardized raw signal, 6 channels by 199 time points | 150 filters; kernel size 15; dropout 0.4; global average pooling head; Adam optimizer; learning rate 0.001; batch size 32; binary cross-entropy with logits; maximum 50 epochs; early-stopping patience 5 |
| RecursiveNet | Standardized raw signal, 6 channels by 199 time points | Convolution blocks with 32 and 64 filters and 3 by 3 kernels; dropout 0.25 in specified blocks; dense layer size 256; Adam optimizer; learning rate 0.001; batch size 32; binary cross-entropy with logits; maximum 50 epochs; early-stopping patience 5 |
| SVM | 176 engineered features | RBF kernel; `StandardScaler` plus scikit-learn `SVC` defaults |
| XGBoost | Same 176 engineered features as SVM | Binary logistic objective; 200 trees; maximum depth 4; learning rate 0.05; row subsampling 0.8; column subsampling 0.8; log-loss evaluation metric; seed 0 |

Neural-network training and early stopping are implemented in `src/train.py`.
Model architecture definitions are in `src/hiknet.py` and
`src/recursivenet.py`.

## Evaluation metrics

Metrics are computed in `src/metrics.py`.

| Metric | Meaning |
|---|---|
| Accuracy | Fraction of all predictions that are correct |
| Precision | Fraction of predicted impacts that are actual impacts |
| Specificity | Fraction of false positives correctly identified |
| Sensitivity | Fraction of impacts correctly detected |
| ROC-AUC | Area under the ROC curve, using model scores |
| PR-AUC | Trapezoidal area under the precision-recall curve, using model scores |

## Final comparison results

Results below are from the synthetic dataset, seed 0, and the shared
stratified 70/30 holdout split.

| Model | Dataset | Split | Accuracy | Precision | Specificity | Sensitivity | ROC-AUC | PR-AUC |
|---|---|---|---:|---:|---:|---:|---:|---:|
| HIKNet | Synthetic, 527 samples | Stratified 70/30 | 0.911 | 0.901 | 0.899 | 0.924 | 0.938 | 0.929 |
| RecursiveNet | Synthetic, 527 samples | Stratified 70/30 | 0.785 | 0.808 | 0.823 | 0.747 | 0.877 | 0.867 |
| SVM | Synthetic, 527 samples | Stratified 70/30 | 0.911 | 0.922 | 0.924 | 0.899 | 0.940 | 0.923 |
| XGBoost | Synthetic, 527 samples | Stratified 70/30 | 0.899 | 0.889 | 0.886 | 0.911 | 0.951 | 0.935 |

Confusion counts on the 158-example evaluation set:

| Model | True impacts detected (TP) | False positives predicted as impacts (FP) | False positives correctly identified (TN) | Impacts missed (FN) |
|---|---:|---:|---:|---:|
| HIKNet | 73 | 8 | 71 | 6 |
| RecursiveNet | 59 | 14 | 65 | 20 |
| SVM | 71 | 6 | 73 | 8 |
| XGBoost | 72 | 9 | 70 | 7 |

For this split, HIKNet and SVM have the highest accuracy at about 91.1%. SVM
has the highest precision and specificity. XGBoost has the highest ROC-AUC
and PR-AUC. RecursiveNet has lower scores in this run.

These are results on synthetic data, not the original Stanford recordings.
The HIKNet accuracy reported in the paper (98.2%, as noted during this
project) is not directly comparable to our result because the dataset and
evaluation setup are different. We cannot conclude from this experiment that
XGBoost outperforms the paper's models on the original data.

## Separate Wu 2017 SVM workflow

The Wu SVM workflow is separate from the final comparison:

1. `scripts/select_features.py` performs forward feature selection on the Wu
   training data, using 10-fold cross-validation by default and selecting up
   to 10 features.
2. `scripts/train_svm.py` trains and evaluates an SVM using the selected Wu
   features. It reports cross-validation results on the workbook's training
   data and results on the separate 32-example test set.

The final SVM row in the results table above does not use Wu data. It uses
the synthetic 527-sample dataset and all 176 features, matching the XGBoost
input.

## Run the final comparison

From the repository root:

```bash
python -m scripts.run_final_comparison
```

Optional arguments:

```bash
python -m scripts.run_final_comparison --seed 0 --epochs 50 --validation-size 0.15
```

The script writes:

- `results/final_comparison.csv`: a readable metrics table
- `results/final_comparison.json`: metrics, model inputs, class counts, feature
  names, and exact train/evaluation sample indices

## Important files

### Data and source code

| Path | Purpose |
|---|---|
| `data/data.mat` | Synthetic kinematic recordings used by the final comparison |
| `data/labels.mat` | Labels for the synthetic recordings |
| `data/wu2017_features.xlsx` | Separate Wu 2017 feature workbook |
| `src/data.py` | Dataset loading, channel definitions, shape checks, and class counts |
| `src/mock.py` | Synthetic signal and label generator |
| `src/preprocess.py` | Signal standardization and stratified splitting |
| `src/features.py` | Engineered feature extraction and Wu workbook loader |
| `src/hiknet.py` | HIKNet architecture |
| `src/recursivenet.py` | RecursiveNet architecture |
| `src/baseline.py` | SVM setup and feature-selection helper |
| `src/xgboost_model.py` | XGBoost configuration |
| `src/train.py` | Neural-network training, early stopping, and evaluation |
| `src/metrics.py` | Evaluation metric calculations |
| `src/spectrum.py` | FFT helper for frequency-domain plots |
| `src/tuning.py` | Helpers for repeated HIKNet tuning experiments |

### Experiment and plotting scripts

| Path | Purpose |
|---|---|
| `scripts/run_final_comparison.py` | Final shared four-model holdout comparison |
| `scripts/make_mock_data.py` | Generate synthetic MATLAB data files |
| `scripts/check_data.py` | Check dataset shape, class balance, and signal ranges |
| `scripts/train_hiknet.py` | Standalone HIKNet training and evaluation |
| `scripts/compare_models.py` | Earlier HIKNet and RecursiveNet cross-validation workflow |
| `scripts/evaluate.py` | Earlier HIKNet and synthetic-data SVM workflow |
| `scripts/train_svm.py` | Separate Wu-workbook SVM experiment |
| `scripts/select_features.py` | Feature selection for the Wu SVM experiment |
| `scripts/train_xgboost.py` | Standalone XGBoost experiment and plots |
| `scripts/tune_hparams.py` | HIKNet filter, kernel, and dropout sweeps |
| `scripts/tune_head.py` | HIKNet classification-head comparison |
| `scripts/plot_eda.py` | Time-domain and frequency-domain exploratory plots |
| `scripts/plot_curves.py` | ROC and precision-recall plots |
| `scripts/demo.py` | Example HIKNet predictions using a saved checkpoint |
| `scripts/make_writeup.py` | Generate the project report PDF |
| `scripts/make_slides.py` | Generate the presentation PDF |

### Results and supporting material

| Path | Purpose |
|---|---|
| `results/final_comparison.csv` | Final four-model results table |
| `results/final_comparison.json` | Detailed results and split metadata |
| `results/xgboost_metrics.json` | Earlier standalone XGBoost metrics |
| `figures/` | Exploratory, tuning, ROC/PR, and XGBoost figures |
| `docs/writeup.pdf` | Existing project report; verify it includes the latest comparison |
| `docs/slides.pdf` | Existing project slides; verify they include the latest comparison |
| `resources/Guidelines and Instructions_Mini Project Assignment.pdf` | Assignment instructions |
| `resources/wu2017/wu2017.pdf` | Wu 2017 paper |
| `resources/wu2017/MOESM1.doc` | Wu supplementary material |
| `resources/Project_Code_zip/Project_Code_zip/` | Supplied reference Python and MATLAB scripts |
| `requirements.txt` | Python dependencies |
| `LICENSE` | Repository license |

## Limitations

- The original Stanford recordings are not available; the main comparison
  uses synthetic signals.
- The synthetic generator's signal patterns are assumptions and may not
  represent real mouthguard recordings.
- The reported comparison uses one train/evaluation split and one seed.
- HIKNet and RecursiveNet use raw signals, while SVM and XGBoost use
  engineered features, so their input representations differ.
- The existing report and slides should be checked to make sure they reflect
  the final results in `results/final_comparison.csv`.
