import argparse

import numpy as np

from src.data import CHANNELS, DATA_DIR, class_balance, load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DATA_DIR)
    args = parser.parse_args()

    X, y = load_dataset(args.data_dir)
    counts = class_balance(y)

    print(f"samples: {counts['total']}  shape: {X.shape[1:]}")
    print(f"real impacts: {counts['impact']}  false impacts: {counts['false']}")

    if counts["impact"] != 264 or counts["false"] != 263:
        print("warning: counts differ from the paper (264 / 263)")
    if np.isnan(X).any():
        print(f"warning: {int(np.isnan(X).any(axis=(1, 2)).sum())} samples contain NaN")

    print("\nper channel range:")
    for i, name in enumerate(CHANNELS):
        print(f"  {name:<10} min {X[:, i].min():9.2f}  max {X[:, i].max():9.2f}")


if __name__ == "__main__":
    main()
