import argparse
import json
from pathlib import Path

from src.baseline import forward_select
from src.features import load_wu_features

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-features", type=int, default=10)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    splits, names = load_wu_features()
    F, y = splits["train"]
    print(f"forward selection over {F.shape[1]} features, {len(y)} samples")

    selected, history = forward_select(
        F, y, max_features=args.max_features, k=args.folds, seed=args.seed, names=names
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "selected_features.json"
    out.write_text(json.dumps({
        "indices": selected,
        "names": [names[i] for i in selected],
        "cv_auc": history,
    }, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
