import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio

from src.data import DATA_DIR
from src.mock import make_mock_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    X, y = make_mock_dataset(seed=args.seed)
    out = Path(args.data_dir)
    out.mkdir(exist_ok=True)
    # same layout as the lab files: data (n, 199, 6), labels 1 = impact
    sio.savemat(out / "data.mat", {"data": np.transpose(X, (0, 2, 1))})
    sio.savemat(out / "labels.mat", {"label_impact_noimpact": y.reshape(-1, 1)})
    print(f"wrote {len(y)} mock samples to {out}")


if __name__ == "__main__":
    main()
