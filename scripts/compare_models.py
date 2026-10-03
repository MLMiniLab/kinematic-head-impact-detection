import argparse
import json
import os
import time
from pathlib import Path

import torch

from src.data import DATA_DIR, load_dataset
from src.hiknet import HIKNet
from src.recursivenet import RecursiveNet
from src.train import cross_validate, mean_metrics

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
MODELS = {"hiknet": HIKNet, "recursivenet": RecursiveNet}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.set_num_threads(os.cpu_count() or 1)
    X, y = load_dataset(args.data_dir)
    results = {}
    for name, build in MODELS.items():
        t0 = time.time()
        folds, _ = cross_validate(build, X, y, k=args.folds, seed=args.seed, epochs=args.epochs)
        mean = mean_metrics(folds)
        print(f"\n{name}: {args.folds}-fold cv, {time.time() - t0:.0f}s")
        print("\n".join(f"  {k:<12} {v:.3f}" for k, v in mean.items()))
        results[name] = {"mean": mean, "folds": folds}

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "model_comparison.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
