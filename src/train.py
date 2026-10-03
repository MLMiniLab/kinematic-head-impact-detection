import copy

import numpy as np
import torch
from torch import nn

from src.metrics import compute_metrics
from src.preprocess import kfold_splits, standardize


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_tensor(X):
    return torch.as_tensor(X, dtype=torch.float32)


@torch.no_grad()
def predict_scores(model, X, batch_size=256):
    model.eval()
    X = to_tensor(X)
    out = [torch.sigmoid(model(X[i:i + batch_size])) for i in range(0, len(X), batch_size)]
    return torch.cat(out).numpy()


@torch.no_grad()
def val_loss(model, X, y, loss_fn):
    model.eval()
    return loss_fn(model(to_tensor(X)), to_tensor(y)).item()


def fit(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, lr=1e-3,
        patience=5, seed=0, checkpoint=None, verbose=False):
    """Adam + binary cross entropy with early stopping on validation loss.
    Restores the weights from the best epoch, and saves them to `checkpoint` if given."""
    set_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    Xt, yt = to_tensor(X_train), to_tensor(y_train)
    gen = torch.Generator().manual_seed(seed)

    best_loss, best_state, best_epoch, wait = np.inf, None, 0, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xt), generator=gen)
        for i in range(0, len(Xt), batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()

        vl = val_loss(model, X_val, y_val, loss_fn)
        history.append(vl)
        if verbose:
            print(f"  epoch {epoch:3d}  train loss {loss.item():.4f}  val loss {vl:.4f}")
        if vl < best_loss:
            best_loss, best_epoch, wait = vl, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
            if checkpoint is not None:
                torch.save(best_state, checkpoint)
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"  early stop, best epoch {best_epoch}")
                break

    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "best_val_loss": best_loss, "val_loss": history}


def evaluate(model, X, y):
    scores = predict_scores(model, X)
    return compute_metrics(y, (scores >= 0.5).astype(int), scores), scores


def cross_validate(build_model, X, y, k=10, seed=0, **fit_kwargs):
    """k-fold cv. Each fold's held out part is used for early stopping and scoring, like the
    original HIKNet code. Returns per fold metrics and out of fold scores."""
    X = standardize(X)
    folds, scores = [], np.empty(len(y))
    for i, (train, test) in enumerate(kfold_splits(y, k=k, seed=seed)):
        model = build_model()
        fit(model, X[train], y[train], X[test], y[test], seed=seed + i, **fit_kwargs)
        m, s = evaluate(model, X[test], y[test])
        folds.append(m)
        scores[test] = s
    return folds, scores


def mean_metrics(folds):
    keys = [k for k in folds[0] if isinstance(folds[0][k], float)]
    return {k: float(np.mean([f[k] for f in folds])) for k in keys}
