import numpy as np
import torch
from joblib import Parallel, delayed

from src.hiknet import HIKNet
from src.preprocess import train_eval_split
from src.train import evaluate, fit

METRICS = ["accuracy", "precision", "specificity", "sensitivity", "roc_auc", "pr_auc"]


def run_once(X, y, config, seed, eval_size=0.3, **fit_kwargs):
    """One 70/30 stratified split + fresh init, both driven by `seed`. X must be standardized."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    X_train, y_train, X_eval, y_eval = train_eval_split(X, y, eval_size=eval_size, seed=seed)
    model = HIKNet(n_channels=X.shape[1], n_timesteps=X.shape[2], **config)
    fit(model, X_train, y_train, X_eval, y_eval, seed=seed, **fit_kwargs)
    m, _ = evaluate(model, X_eval, y_eval)
    return {k: m[k] for k in METRICS}


def summarize(runs):
    return {k: {"mean": float(np.mean([r[k] for r in runs])), "std": float(np.std([r[k] for r in runs]))}
            for k in METRICS}


def run_configs(X, y, configs, repeats=10, seed=0, n_jobs=-1, **fit_kwargs):
    """Runs every config `repeats` times (seeds seed..seed+repeats-1), all in one parallel pool.
    Returns one summary per config with mean/std per metric and the raw runs."""
    jobs = [(i, r) for i in range(len(configs)) for r in range(repeats)]
    out = Parallel(n_jobs=n_jobs)(delayed(run_once)(X, y, configs[i], seed + r, **fit_kwargs) for i, r in jobs)
    results = []
    for i, config in enumerate(configs):
        runs = [o for (j, _), o in zip(jobs, out) if j == i]
        results.append({"config": config, **summarize(runs), "runs": runs})
    return results


def run_config(X, y, config, repeats=10, seed=0, n_jobs=-1, **fit_kwargs):
    return run_configs(X, y, [config], repeats, seed, n_jobs, **fit_kwargs)[0]


def format_row(name, res):
    return (f"  {name:<16} roc_auc {res['roc_auc']['mean']:.3f} +- {res['roc_auc']['std']:.3f}"
            f"   accuracy {res['accuracy']['mean']:.3f} +- {res['accuracy']['std']:.3f}")
