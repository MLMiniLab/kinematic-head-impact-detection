from pathlib import Path

import numpy as np
import scipy.io as sio

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CHANNELS = ["lin_acc_x", "lin_acc_y", "lin_acc_z", "ang_vel_x", "ang_vel_y", "ang_vel_z"]
N_TIMESTEPS = 199
FS = 1000


def load_dataset(data_dir=DATA_DIR):
    data_dir = Path(data_dir)
    data_path = data_dir / "data.mat"
    labels_path = data_dir / "labels.mat"

    for p in (data_path, labels_path):
        if not p.exists():
            raise FileNotFoundError(f"missing {p}, see README for the expected data files")

    data = sio.loadmat(data_path)["data"]
    labels = sio.loadmat(labels_path)["label_impact_noimpact"]

    X = np.transpose(data, (0, 2, 1)).astype(np.float64)
    y = np.asarray(labels).ravel().astype(int)

    if X.shape[1:] != (len(CHANNELS), N_TIMESTEPS):
        raise ValueError(f"expected samples of shape (6, 199), got {X.shape[1:]}")
    if len(X) != len(y):
        raise ValueError(f"{len(X)} samples but {len(y)} labels")

    return X, y


def class_balance(y):
    n_impact = int(np.sum(y == 1))
    n_false = int(np.sum(y == 0))
    return {"impact": n_impact, "false": n_false, "total": len(y)}
