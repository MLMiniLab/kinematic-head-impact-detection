import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.baseline import build_svm
from src.data import DATA_DIR, load_dataset
from src.features import extract_features
from src.hiknet import HIKNet
from src.metrics import compute_metrics, format_metrics
from src.preprocess import standardize, train_eval_split
from src.recursivenet import RecursiveNet
from src.train import evaluate, fit, set_seed
from src.xgboost_model import build_xgboost

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
METRIC_KEYS = (
    "accuracy",
    "precision",
    "specificity",
    "sensitivity",
    "roc_auc",
    "pr_auc",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare all four models on one stratified 70/30 dataset split."
    )
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--validation-size", type=float, default=0.15)
    args = parser.parse_args()

    X, y = load_dataset(args.data_dir)
    indices = np.arange(len(y))
    train_idx, y_train, eval_idx, y_eval = train_eval_split(
        indices, y, eval_size=0.3, seed=args.seed
    )
    train_idx = train_idx.astype(int)
    eval_idx = eval_idx.astype(int)

    features, feature_names = extract_features(X)
    X_standardized = standardize(X)
    X_train, X_eval = X_standardized[train_idx], X_standardized[eval_idx]
    F_train, F_eval = features[train_idx], features[eval_idx]

    X_fit, y_fit, X_val, y_val = train_eval_split(
        X_train,
        y_train,
        eval_size=args.validation_size,
        seed=args.seed + 1,
    )

    results = {}
    for model_name, model_builder in (
        ("HIKNet", HIKNet),
        ("RecursiveNet", RecursiveNet),
    ):
        set_seed(args.seed)
        model, info = fit(
            model_builder(),
            X_fit,
            y_fit,
            X_val,
            y_val,
            epochs=args.epochs,
            seed=args.seed,
            verbose=True,
        )
        metrics, _ = evaluate(model, X_eval, y_eval)
        results[model_name] = {
            "dataset": "synthetic 527-sample kinematic dataset",
            "split": "stratified 70/30 holdout",
            "train_samples": int(len(train_idx)),
            "evaluation_samples": int(len(eval_idx)),
            "input": "standardized raw 6-channel, 199-timestep signals",
            "early_stopping": (
                f"training-only validation split ({args.validation_size:.0%}); "
                f"best epoch {info['best_epoch']}"
            ),
            "metrics": metrics,
        }

    svm = build_svm()
    svm.fit(F_train, y_train)
    svm_scores = svm.decision_function(F_eval)
    results["SVM"] = {
        "dataset": "synthetic 527-sample kinematic dataset",
        "split": "stratified 70/30 holdout",
        "train_samples": int(len(train_idx)),
        "evaluation_samples": int(len(eval_idx)),
        "input": f"all {len(feature_names)} extracted features",
        "metrics": compute_metrics(y_eval, svm.predict(F_eval), svm_scores),
    }

    xgboost = build_xgboost(seed=args.seed)
    xgboost.fit(F_train, y_train)
    xgboost_scores = xgboost.predict_proba(F_eval)[:, 1]
    results["XGBoost"] = {
        "dataset": "synthetic 527-sample kinematic dataset",
        "split": "stratified 70/30 holdout",
        "train_samples": int(len(train_idx)),
        "evaluation_samples": int(len(eval_idx)),
        "input": f"all {len(feature_names)} extracted features (same as SVM)",
        "metrics": compute_metrics(
            y_eval, xgboost.predict(F_eval), xgboost_scores
        ),
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    output = {
        "seed": args.seed,
        "dataset_samples": int(len(y)),
        "dataset_shape": list(X.shape[1:]),
        "class_counts": {
            "impact": int(np.sum(y == 1)),
            "false_positive": int(np.sum(y == 0)),
        },
        "train_indices": train_idx.tolist(),
        "evaluation_indices": eval_idx.tolist(),
        "shared_tabular_features": feature_names,
        "models": results,
    }
    json_path = RESULTS_DIR / "final_comparison.json"
    json_path.write_text(json.dumps(output, indent=2))

    csv_path = RESULTS_DIR / "final_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=(
                "model",
                "dataset",
                "split",
                "train_samples",
                "evaluation_samples",
                *METRIC_KEYS,
            ),
        )
        writer.writeheader()
        for model_name, result in results.items():
            writer.writerow(
                {
                    "model": model_name,
                    "dataset": result["dataset"],
                    "split": result["split"],
                    "train_samples": result["train_samples"],
                    "evaluation_samples": result["evaluation_samples"],
                    **{
                        key: result["metrics"][key]
                        for key in METRIC_KEYS
                    },
                }
            )

    print("\nFinal comparison (shared 70/30 holdout)")
    for model_name, result in results.items():
        print(f"\n{model_name}")
        print(format_metrics(result["metrics"]))
    print(f"\nsaved {json_path}")
    print(f"saved {csv_path}")


if __name__ == "__main__":
    main()
