import argparse
import json
from pathlib import Path

import numpy as np

from src.data import DATA_DIR, load_dataset
from src.hiknet import HIKNet
from src.metrics import format_metrics
from src.preprocess import standardize, train_eval_split
from src.train import evaluate, fit

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
CKPT_DIR = ROOT / "checkpoints"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation-size", type=float, default=0.15)
    args = parser.parse_args()

    X, y = load_dataset(args.data_dir)
    X_train, y_train, X_eval, y_eval = train_eval_split(
        standardize(X), y, seed=args.seed
    )
    X_fit, y_fit, X_val, y_val = train_eval_split(
        X_train, y_train, eval_size=args.validation_size, seed=args.seed + 1
    )
    print(
        f"fit {len(y_fit)}  validation {len(y_val)}  "
        f"test {len(y_eval)}"
    )

    CKPT_DIR.mkdir(exist_ok=True)
    ckpt = CKPT_DIR / "hiknet.pt"
    model, info = fit(HIKNet(), X_fit, y_fit, X_val, y_val, epochs=args.epochs,
                      seed=args.seed, checkpoint=ckpt, verbose=True)
    m, scores = evaluate(model, X_eval, y_eval)
    print(f"\neval set (n={len(y_eval)}), best epoch {info['best_epoch']}")
    print(format_metrics(m))

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "hiknet_metrics.json"
    out.write_text(json.dumps({"best_epoch": info["best_epoch"], "eval": m}, indent=2))
    np.savez(RESULTS_DIR / "hiknet_scores.npz", y=y_eval, scores=scores)
    print(f"saved {out}, weights in {ckpt}")


if __name__ == "__main__":
    main()
