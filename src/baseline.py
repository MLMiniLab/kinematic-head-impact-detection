import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.preprocess import kfold_splits


def build_svm():
    return make_pipeline(StandardScaler(), SVC(kernel="rbf"))


def cv_auc(F, y, folds):
    scores = np.empty(len(y))
    for train, test in folds:
        model = build_svm().fit(F[train], y[train])
        scores[test] = model.decision_function(F[test])
    return roc_auc_score(y, scores)


def forward_select(F, y, max_features=10, k=10, seed=0, tol=1e-4, names=None):
    folds = kfold_splits(y, k=k, seed=seed)
    selected, history = [], []
    best = 0.0

    while len(selected) < max_features:
        candidates = [j for j in range(F.shape[1]) if j not in selected]
        aucs = Parallel(n_jobs=-1)(
            delayed(cv_auc)(F[:, selected + [j]], y, folds) for j in candidates
        )
        i = int(np.argmax(aucs))
        if aucs[i] - best < tol:
            break
        best = aucs[i]
        selected.append(candidates[i])
        history.append(best)
        label = names[candidates[i]] if names else candidates[i]
        print(f"  {len(selected):2d}  {label:<32} cv auc {best:.4f}")

    return selected, history
