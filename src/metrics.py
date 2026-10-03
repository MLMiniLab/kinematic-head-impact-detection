import numpy as np
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_auc_score


def compute_metrics(y_true, y_pred, y_score=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    m = {
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "sensitivity": tp / (tp + fn) if tp + fn else 0.0,
    }
    if y_score is not None:
        m["roc_auc"] = roc_auc_score(y_true, y_score)
        p, r, _ = precision_recall_curve(y_true, y_score)
        m["pr_auc"] = auc(r, p)
    m = {k: float(v) for k, v in m.items()}
    m.update(tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))
    return m


def format_metrics(m):
    keys = ["accuracy", "precision", "specificity", "sensitivity", "roc_auc", "pr_auc"]
    lines = [f"  {k:<12} {m[k]:.3f}" for k in keys if k in m]
    lines.append(f"  tp {m['tp']}  fp {m['fp']}  tn {m['tn']}  fn {m['fn']}")
    return "\n".join(lines)
