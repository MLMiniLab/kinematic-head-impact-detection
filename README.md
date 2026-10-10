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

## Setup and run

Use Python 3.10 or newer, then run these commands from the repository root
(PowerShell on Windows):

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m scripts.run_final_comparison --synthetic-seed 42 --seed 0
python -m scripts.make_writeup
```

The comparison command generates its 527-sample dataset in memory; it does not
require the original Stanford recordings or the local `.mat` files. It writes
the shared-split results to `results/final_comparison.csv` and
`results/final_comparison.json`. The report command generates the two-page
`docs/writeup.pdf` from those saved results.

To prepare the standalone HIKNet live demo, generate local synthetic `.mat`
files, train a checkpoint, then run predictions:

```powershell
python -m scripts.make_mock_data --seed 0
python -m scripts.train_hiknet --seed 0
python -m scripts.demo --checkpoint checkpoints/hiknet.pt
```

The checkpoint is local and ignored by Git. The demo illustrates inference on
the mock recordings; it is not an additional final-comparison evaluation.

## Datasets

### Synthetic kinematic dataset

Optional local MATLAB files are `data/data.mat` and `data/labels.mat`.
They are ignored by Git and are not required for the documented final
comparison command.

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

The local `.mat` files used by standalone scripts match the output of this
generator with seed 0. To regenerate them, run:

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
- Final comparison synthetic-data seed: 42
- Outer split random seed: 0
- Training examples: 369
- Held-out evaluation examples: 158, with 79 examples from each class
- The same outer training and evaluation examples are used for all four models
- HIKNet and RecursiveNet reserve 15% of the outer training data for
  early-stopping validation; this validation data is not the final evaluation
  set
- The JSON output records the sample indices for both sets

For the final comparison, the synthetic generator is run in memory with seed
42 instead of reusing the seed-0 `.mat` files used by earlier tuning work. The
sample identities are therefore new, while the generator and its assumptions
remain the same. This avoids direct sample reuse; it does not turn synthetic
data into independent real-world validation. The neural networks fit on 313
examples and use 56 training-only examples for early stopping. SVM and XGBoost
train on all 369 outer training examples. SVM and XGBoost both use the same
176 features.

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

Results below are from a fresh synthetic generator run (seed 42) and the
shared stratified 70/30 holdout split (split seed 0).

| Model | Dataset | Split | Accuracy | Precision | Specificity | Sensitivity | ROC-AUC | PR-AUC |
|---|---|---|---:|---:|---:|---:|---:|---:|
| HIKNet | Synthetic, 527 samples | Stratified 70/30 | 0.911 | 0.901 | 0.899 | 0.924 | 0.946 | 0.933 |
| RecursiveNet | Synthetic, 527 samples | Stratified 70/30 | 0.848 | 0.857 | 0.861 | 0.835 | 0.917 | 0.910 |
| SVM | Synthetic, 527 samples | Stratified 70/30 | 0.791 | 0.871 | 0.899 | 0.684 | 0.905 | 0.890 |
| XGBoost | Synthetic, 527 samples | Stratified 70/30 | 0.892 | 0.897 | 0.899 | 0.886 | 0.933 | 0.892 |

Confusion counts on the 158-example evaluation set:

| Model | True impacts detected (TP) | False positives predicted as impacts (FP) | False positives correctly identified (TN) | Impacts missed (FN) |
|---|---:|---:|---:|---:|
| HIKNet | 73 | 8 | 71 | 6 |
| RecursiveNet | 66 | 11 | 68 | 13 |
| SVM | 54 | 8 | 71 | 25 |
| XGBoost | 70 | 8 | 71 | 9 |

For this split, HIKNet has the highest accuracy, sensitivity, ROC-AUC, and
PR-AUC. XGBoost is close in accuracy (89.2%) and has ROC-AUC 0.933, but it
does not exceed HIKNet overall. All three of HIKNet, SVM, and XGBoost have
specificity 0.899. These findings describe only this one synthetic holdout.

These are results on synthetic data, not the original Stanford recordings.
The HIKNet accuracy reported in the paper (98.2%, as noted during this
project) is not directly comparable to our result because the dataset and
evaluation setup are different. In this run, XGBoost was lower than HIKNet
in accuracy and area-under-curve metrics. We cannot conclude from this
experiment that XGBoost or any other model performs similarly on original
Stanford recordings.

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
python -m scripts.run_final_comparison --synthetic-seed 42 --seed 0
```

Optional arguments:

```bash
python -m scripts.run_final_comparison --seed 0 --epochs 50 --validation-size 0.15
```

Use `--synthetic-seed` to generate a fresh simulated dataset in memory. If it
is omitted, the script loads `data/data.mat` and `data/labels.mat` instead.
To load the saved seed-42 MATLAB dataset from `data/synthetic_seed42/` instead,
pass `--saved-seed42`. This option cannot be combined with `--synthetic-seed`.
Use `--output-suffix seed42_saved` to preserve the default comparison files and
write `results/final_comparison_seed42_saved.csv` and
`results/final_comparison_seed42_saved.json`. The saved JSON records the data
source, seeds, and exact shared split indices.

To generate the saved seed-42 dataset without replacing the seed-0 MATLAB
files in `data/`, run:

```powershell
python -m scripts.make_mock_data --seed 42 --data-dir data/synthetic_seed42
```

The generated `.mat` files in that subdirectory are ignored by Git, like the
top-level generated MATLAB files.

The script writes:

- `results/final_comparison.csv`: a readable metrics table
- `results/final_comparison.json`: metrics, model inputs, class counts, feature
  names, and exact train/evaluation sample indices

## Important files

### Data and source code

| Path | Purpose |
|---|---|
| `data/data.mat` | Seed-0 synthetic kinematic recordings for scripts that load local MATLAB data |
| `data/labels.mat` | Labels for the seed-0 synthetic recordings |
| `data/synthetic_seed42/data.mat` | Optional saved seed-42 synthetic recordings for the final comparison |
| `data/synthetic_seed42/labels.mat` | Labels for the saved seed-42 recordings |
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
| `results/final_comparison_seed42_saved.csv` | Comparison rerun using the saved seed-42 MATLAB dataset |
| `results/final_comparison_seed42_saved.json` | Saved-dataset comparison metadata and exact split indices |
| `results/hiknet_metrics.json` | Separate standalone HIKNet evaluation on the seed-0 local `.mat` dataset; not the final four-model run |
| `results/hiknet_scores.npz` | Labels and scores from that standalone HIKNet evaluation |
| `results/xgboost_metrics.json` | Earlier standalone XGBoost metrics |
| `figures/` | Exploratory, tuning, ROC/PR, and XGBoost figures |
| `docs/writeup.pdf` | Two-page project report generated from the final comparison artifacts |
| `docs/slides.pdf` | Presentation file to be prepared and updated by the project team |
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
- The final comparison uses a new synthetic generator seed (42), but model
  choices were developed using the same simulator; this is not independent
  validation on real-world data.
- HIKNet and RecursiveNet use raw signals, while SVM and XGBoost use
  engineered features, so their input representations differ.
