import argparse
import json
import time
from pathlib import Path

from src.data import DATA_DIR, load_dataset
from src.hiknet import HEADS
from src.preprocess import standardize
from src.tuning import format_row, run_configs

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"


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
    t = time.time()
    res = run_configs(X, y, [{"head": h} for h in HEADS], repeats=args.repeats, seed=args.seed,
                      n_jobs=args.n_jobs, epochs=args.epochs)
    print(f"head comparison, {args.repeats} repeats, 70/30 split ({time.time() - t:.0f}s)")
    for h, r in zip(HEADS, res):
        print(format_row(h, r))
    best = max(zip(HEADS, res), key=lambda p: p[1]["roc_auc"]["mean"])[0]
    print(f"best head by roc_auc: {best}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "tune_head.json"
    out.write_text(json.dumps({"epochs": args.epochs, "repeats": args.repeats, "best": best,
                               "results": dict(zip(HEADS, res))}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
