"""Train and evaluate XGBoost on the kinematic head impact dataset.

Uses the SAME pipeline as the SVM baseline in scripts/evaluate.py:
  - load_dataset()          → raw signals (N, 6, 199)
  - extract_features()      → tabular hand-crafted features
  - train_eval_split()      → 70/30 stratified split, seed=0
  - compute_metrics()       → accuracy, precision, specificity, sensitivity,
                               roc_auc, pr_auc

Figures are written to figures/ and metrics to results/xgboost_metrics.json.

Usage
-----
    python -m scripts.train_xgboost
    python -m scripts.train_xgboost --seed 0 --data-dir data
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from src.data import DATA_DIR, load_dataset
from src.features import extract_features
from src.metrics import compute_metrics, format_metrics
from src.preprocess import train_eval_split
from src.xgboost_model import SEED, build_xgboost

ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, out_path: Path) -> None:
    """Save a confusion-matrix figure to *out_path*."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["False positive", "Impact"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("XGBoost – Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def plot_roc(y_true, y_score, out_path: Path) -> None:
    """Save an ROC-curve figure to *out_path*."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(fpr, tpr, color="#2e7d32", lw=2, label=f"XGBoost (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--", label="Chance")
    ax.set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="ROC curve – XGBoost",
        xlim=(-0.01, 1.01),
        ylim=(-0.01, 1.01),
        aspect="equal",
    )
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def plot_pr(y_true, y_score, out_path: Path) -> None:
    """Save a Precision-Recall curve figure to *out_path*."""
    p, r, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(r, p)
    prevalence = float(np.sum(y_true) / len(y_true))
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(r, p, color="#2e7d32", lw=2, drawstyle="steps-post",
            label=f"XGBoost (AUC = {pr_auc:.3f})")
    ax.axhline(prevalence, color="0.6", lw=1, ls="--",
               label=f"Chance ({prevalence:.2f})")
    ax.set(
        xlabel="Recall",
        ylabel="Precision",
        title="Precision-Recall curve – XGBoost",
        xlim=(-0.01, 1.01),
        ylim=(-0.01, 1.01),
        aspect="equal",
    )
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate XGBoost on the head-impact dataset."
    )
    parser.add_argument("--data-dir", default=DATA_DIR,
                        help="Directory containing data.mat and labels.mat")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed (default: 0, matching the rest of the project)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load data and extract tabular features  (same as evaluate.py)
    # ------------------------------------------------------------------
    print("Loading dataset …")
    X, y = load_dataset(args.data_dir)
    print(f"  dataset: {X.shape}  labels: {y.shape}")
    print(f"  impacts: {int(np.sum(y == 1))}  false positives: {int(np.sum(y == 0))}")

    print("Extracting hand-crafted features …")
    F, feat_names = extract_features(X)
    print(f"  feature matrix: {F.shape}  ({len(feat_names)} features)")

    # ------------------------------------------------------------------
    # 2. Train/eval split — identical seed to the rest of the project
    #    train_eval_split() is stratified (class-balanced) by design.
    # ------------------------------------------------------------------
    # We call train_eval_split on both F and X so that indices align;
    # only F is used for XGBoost.
    F_train, y_train, F_eval, y_eval = train_eval_split(F, y, seed=args.seed)
    print(f"\ntrain {len(y_train)}  eval {len(y_eval)}")

    # Sanity check: feature matrix must not contain look-ahead information.
    # extract_features() operates per-sample so there is no leakage.

    # ------------------------------------------------------------------
    # 3. Fit XGBoost on training split (no fitting on eval data)
    # ------------------------------------------------------------------
    print("\nFitting XGBoost …")
    model = build_xgboost(seed=args.seed)
    model.fit(F_train, y_train)

    # ------------------------------------------------------------------
    # 4. Predict on eval split
    # ------------------------------------------------------------------
    y_pred = model.predict(F_eval)               # hard labels (0 / 1)
    y_score = model.predict_proba(F_eval)[:, 1]  # P(impact) for AUC metrics

    # ------------------------------------------------------------------
    # 5. Compute metrics using the project's shared function
    # ------------------------------------------------------------------
    metrics = compute_metrics(y_eval, y_pred, y_score)

    print(f"\nXGBoost eval set (n={len(y_eval)})")
    print(format_metrics(metrics))

    # ------------------------------------------------------------------
    # 6. Save metrics JSON
    # ------------------------------------------------------------------
    RESULTS_DIR.mkdir(exist_ok=True)
    out_json = RESULTS_DIR / "xgboost_metrics.json"
    out_json.write_text(json.dumps({"seed": args.seed, "eval": metrics}, indent=2))
    print(f"\nsaved {out_json}")

    # ------------------------------------------------------------------
    # 7. Save figures
    # ------------------------------------------------------------------
    FIGURES_DIR.mkdir(exist_ok=True)
    plot_confusion_matrix(y_eval, y_pred,  FIGURES_DIR / "xgboost_confusion_matrix.png")
    plot_roc(y_eval, y_score,             FIGURES_DIR / "xgboost_roc_curve.png")
    plot_pr(y_eval, y_score,              FIGURES_DIR / "xgboost_pr_curve.png")


if __name__ == "__main__":
    main()
