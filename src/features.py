from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.signal import periodogram

from src.data import DATA_DIR, FS

G = 9.81
AXES = ["x", "y", "z", "mag"]
LIN_PSD_FREQS = range(10, 201, 10)
ANG_PSD_FREQS = range(10, 181, 10)

WU_FILE = DATA_DIR / "wu2017_features.xlsx"


def with_magnitude(xyz):
    return np.concatenate([xyz, np.linalg.norm(xyz, axis=1, keepdims=True)], axis=1)


def half_max_width(s):
    s = np.abs(s)
    above = s >= s.max(axis=-1, keepdims=True) / 2
    first = above.argmax(axis=-1)
    last = s.shape[-1] - 1 - above[..., ::-1].argmax(axis=-1)
    return (last - first + 1) / FS * 1000


def psd_at(s, freqs):
    f, p = periodogram(s, fs=FS, nfft=200, axis=-1)
    idx = [int(np.argmin(np.abs(f - fr))) for fr in freqs]
    return p[..., idx]


def extract_features(X):
    lin_acc = X[:, :3]
    ang_vel = X[:, 3:]
    lin_vel = cumulative_trapezoid(lin_acc * G, dx=1 / FS, axis=-1, initial=0)
    ang_acc = np.gradient(ang_vel, 1 / FS, axis=-1)

    signals = {
        "lin vel": with_magnitude(lin_vel),
        "lin acc": with_magnitude(lin_acc),
        "ang vel": with_magnitude(ang_vel - ang_vel[..., :1]),
        "ang acc": with_magnitude(ang_acc),
    }

    columns, names = [], []
    for name, s in signals.items():
        for i, ax in enumerate(AXES):
            columns.append(np.abs(s[:, i]).max(axis=-1))
            names.append(f"{name} peak {ax}")
    for name in ["lin acc", "ang acc"]:
        for i, ax in enumerate(AXES):
            columns.append(half_max_width(signals[name][:, i]))
            names.append(f"{name} dur {ax}")
    for name, freqs in [("lin acc", LIN_PSD_FREQS), ("ang acc", ANG_PSD_FREQS)]:
        for i, ax in enumerate(AXES):
            p = psd_at(signals[name][:, i], freqs)
            for j, fr in enumerate(freqs):
                columns.append(p[:, j])
                names.append(f"{name} {ax} PSD {fr}Hz")

    return np.column_stack(columns), names


def load_wu_features(path=WU_FILE):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"missing {path}, see README")
    sheets = pd.read_excel(path, sheet_name=None)
    out = {}
    for split, tag in [("train", "Training"), ("test", "Testing")]:
        F = sheets[f"Feature Matrix {tag}"]
        y = sheets[f"Label Vector {tag}"]["label"].to_numpy().astype(int)
        names = [str(c).strip("'") for c in F.columns]
        out[split] = (F.to_numpy(dtype=float), y)
    return out, names
