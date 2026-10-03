import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import DATA_DIR, load_dataset
from src.preprocess import standardize
from src.tuning import format_row, run_configs

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"

DEFAULT = {"filters": 150, "kernel": 15, "dropout": 0.4}
GRID = {
    "filters": [15, 30, 60, 100, 150, 200],
    "kernel": [5, 9, 15, 21, 31],
    "dropout": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
}


def plot(sweeps, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, (param, rows) in zip(axes, sweeps.items()):
        xs = [r["value"] for r in rows]
        for metric, label in [("roc_auc", "ROC AUC"), ("accuracy", "Accuracy")]:
            ax.errorbar(xs, [r[metric]["mean"] for r in rows], yerr=[r[metric]["std"] for r in rows],
                        marker="o", capsize=3, label=label)
        ax.axvline(DEFAULT[param], color="grey", ls=":", lw=1)
        ax.set_xlabel(param)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("mean ± std (eval set)")
    axes[0].legend(loc="lower right")
    fig.suptitle("HIKNet one-at-a-time hyperparameter sweeps (dotted = default)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    X, y = load_dataset(args.data_dir)
    X = standardize(X)
    keys = [(p, v) for p, vals in GRID.items() for v in vals]
    configs = [{**DEFAULT, p: v} for p, v in keys]
    t = time.time()
    res = run_configs(X, y, configs, repeats=args.repeats, seed=args.seed, n_jobs=args.n_jobs,
                      epochs=args.epochs)
    print(f"one-at-a-time sweeps, {args.repeats} repeats, 70/30 split ({time.time() - t:.0f}s)")

    sweeps = {p: [] for p in GRID}
    for (p, v), r in zip(keys, res):
        sweeps[p].append({"value": v, **r})
    best = {}
    for p, rows in sweeps.items():
        print(p)
        for r in rows:
            print(format_row(str(r["value"]), r))
        best[p] = max(rows, key=lambda r: r["roc_auc"]["mean"])["value"]
    print(f"best per param by roc_auc: {best}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "tune_hparams.json"
    out.write_text(json.dumps({"epochs": args.epochs, "repeats": args.repeats, "default": DEFAULT,
                               "best": best, "sweeps": sweeps}, indent=2))
    FIG_DIR.mkdir(exist_ok=True)
    plot(sweeps, FIG_DIR / "tuning.png")
    print(f"saved {out}, {FIG_DIR / 'tuning.png'}")


if __name__ == "__main__":
    main()
