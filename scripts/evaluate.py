import argparse
import json
from pathlib import Path

from src.baseline import build_svm, forward_select
from src.data import DATA_DIR, load_dataset
from src.features import extract_features
from src.hiknet import HIKNet
from src.metrics import compute_metrics
from src.preprocess import standardize, train_eval_split
from src.train import evaluate, fit

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
KEYS = ["accuracy", "precision", "specificity", "sensitivity", "roc_auc", "pr_auc"]
HEADER = ["Accuracy", "Precision", "Specificity", "Sensitivity", "ROC AUC", "PR AUC"]


def table(results):
    lines = ["| Model | " + " | ".join(HEADER) + " |", "|---" * (len(HEADER) + 1) + "|"]
    for name, r in results.items():
        lines.append(f"| {name} | " + " | ".join(f"{r['metrics'][k]:.3f}" for k in KEYS) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--max-features", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    X, y = load_dataset(args.data_dir)
    F, names = extract_features(X)

    # same seed -> same split for raw signals, standardized signals and features
    X_train, y_train, X_eval, y_eval = train_eval_split(standardize(X), y, seed=args.seed)
    F_train, _, F_eval, _ = train_eval_split(F, y, seed=args.seed)
    print(f"train {len(y_train)}  eval {len(y_eval)}")

    print("\nHIKNet")
    model, info = fit(HIKNet(), X_train, y_train, X_eval, y_eval, epochs=args.epochs, seed=args.seed)
    m_hik, s_hik = evaluate(model, X_eval, y_eval)
    print(f"  best epoch {info['best_epoch']}")

    print("\nSVM forward selection (train part only)")
    idx, _ = forward_select(F_train, y_train, max_features=args.max_features, seed=args.seed, names=names)
    svm = build_svm().fit(F_train[:, idx], y_train)
    s_svm = svm.decision_function(F_eval[:, idx])
    m_svm = compute_metrics(y_eval, svm.predict(F_eval[:, idx]), s_svm)

    results = {
        "HIKNet": {"metrics": m_hik, "best_epoch": info["best_epoch"], "scores": s_hik.tolist()},
        "SVM": {"metrics": m_svm, "features": [names[i] for i in idx], "scores": s_svm.tolist()},
    }
    print("\n" + table(results))

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "final_comparison.json"
    out.write_text(json.dumps({"seed": args.seed, "y": y_eval.tolist(), **results}, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
