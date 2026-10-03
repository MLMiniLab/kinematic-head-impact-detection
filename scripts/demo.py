import argparse
from pathlib import Path

import numpy as np
import torch

from src.data import DATA_DIR, load_dataset
from src.hiknet import HIKNet
from src.preprocess import standardize
from src.train import predict_scores

CKPT = Path(__file__).resolve().parent.parent / "checkpoints" / "hiknet.pt"


def main():
    parser = argparse.ArgumentParser(description="classify a few recordings with the trained HIKNet")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--checkpoint", default=CKPT)
    parser.add_argument("-n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"missing {args.checkpoint}, run python -m scripts.train_hiknet first")

    X, y = load_dataset(args.data_dir)
    model = HIKNet()
    model.load_state_dict(torch.load(args.checkpoint))

    idx = np.random.default_rng(args.seed).choice(len(y), size=args.n, replace=False)
    scores = predict_scores(model, standardize(X[idx]))

    names = {1: "impact", 0: "false positive"}
    print(f"{'sample':>6}  {'p(impact)':>9}  {'predicted':<14} {'label':<14}")
    for i, s in zip(idx, scores):
        pred = int(s >= 0.5)
        mark = "" if pred == y[i] else "  <- wrong"
        print(f"{i:>6}  {s:9.3f}  {names[pred]:<14} {names[y[i]]:<14}{mark}")


if __name__ == "__main__":
    main()
