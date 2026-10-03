import numpy as np


def standardize(X, eps=1e-8):
    mean = X.mean(axis=2, keepdims=True)
    std = X.std(axis=2, keepdims=True)
    return (X - mean) / (std + eps)


def train_eval_split(X, y, eval_size=0.3, seed=0):
    rng = np.random.default_rng(seed)
    train_idx, eval_idx = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        n_eval = int(round(len(idx) * eval_size))
        eval_idx.extend(idx[:n_eval])
        train_idx.extend(idx[n_eval:])
    train_idx = rng.permutation(train_idx)
    eval_idx = rng.permutation(eval_idx)
    return X[train_idx], y[train_idx], X[eval_idx], y[eval_idx]


def kfold_splits(y, k=10, seed=0):
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))
