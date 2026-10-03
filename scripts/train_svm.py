import argparse
import json
from pathlib import Path

import numpy as np

from src.baseline import build_svm
from src.features import load_wu_features
from src.metrics import compute_metrics, format_metrics
from src.preprocess import kfold_splits

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    selected = json.loads((RESULTS_DIR / "selected_features.json").read_text())
    idx = selected["indices"]
    print("features:", ", ".join(selected["names"]))

    splits, _ = load_wu_features()
    F_train, y_train = splits["train"]
    F_test, y_test = splits["test"]
    F_train, F_test = F_train[:, idx], F_test[:, idx]

    cv_pred = np.empty(len(y_train), dtype=int)
    cv_score = np.empty(len(y_train))
    for train, test in kfold_splits(y_train, k=args.folds, seed=args.seed):
        model = build_svm().fit(F_train[train], y_train[train])
        cv_pred[test] = model.predict(F_train[test])
        cv_score[test] = model.decision_function(F_train[test])
    cv = compute_metrics(y_train, cv_pred, cv_score)

    model = build_svm().fit(F_train, y_train)
    test = compute_metrics(y_test, model.predict(F_test), model.decision_function(F_test))

    print(f"\ncollegiate, {args.folds}-fold cv (n={len(y_train)})")
    print(format_metrics(cv))
    print(f"\nyouth test set (n={len(y_test)})")
    print(format_metrics(test))

    out = RESULTS_DIR / "svm_metrics.json"
    out.write_text(json.dumps({"features": selected["names"], "cv": cv, "test": test}, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
