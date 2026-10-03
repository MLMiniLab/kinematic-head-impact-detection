import numpy as np

from src.data import FS, N_TIMESTEPS

PRE_TRIGGER = 50
N_IMPACT = 264
N_FALSE = 263


def damped_pulse(t, freq, decay, phase):
    env = np.where(t >= 0, np.exp(-decay * t), 0.0)
    return env * np.sin(2 * np.pi * freq * t + phase)


def make_sample(rng, impact):
    t = (np.arange(N_TIMESTEPS) - PRE_TRIGGER) / FS
    # real impacts ring at 20-30 Hz, false positives (chewing, biting, drops) carry more high frequency energy
    if impact:
        freq = rng.uniform(18, 32)
        decay = rng.uniform(25, 60)
        hf_weight = rng.uniform(0.0, 0.35)
    else:
        freq = rng.uniform(25, 60)
        decay = rng.uniform(40, 120)
        hf_weight = rng.uniform(0.2, 0.9)

    lin_peak = rng.lognormal(np.log(25), 0.4)
    ang_peak = rng.lognormal(np.log(12), 0.5) * (1.0 if impact else 0.6)
    direction = rng.normal(size=6)
    direction[:3] /= np.linalg.norm(direction[:3])
    direction[3:] /= np.linalg.norm(direction[3:])

    X = np.empty((6, N_TIMESTEPS))
    for c in range(6):
        base = damped_pulse(t, freq * rng.uniform(0.9, 1.1), decay, rng.uniform(0, 2 * np.pi))
        hf = damped_pulse(t, rng.uniform(80, 200), decay * 1.5, rng.uniform(0, 2 * np.pi))
        peak = lin_peak if c < 3 else ang_peak
        X[c] = peak * direction[c] * (base + hf_weight * hf)
        X[c] += rng.normal(scale=0.03 * peak, size=N_TIMESTEPS)
    return X


def make_mock_dataset(seed=0):
    rng = np.random.default_rng(seed)
    y = np.array([1] * N_IMPACT + [0] * N_FALSE)
    # a few samples drawn from the other class, so the task is not perfectly separable
    swap = np.zeros(len(y), dtype=bool)
    swap[rng.choice(N_IMPACT, size=8, replace=False)] = True
    swap[N_IMPACT + rng.choice(N_FALSE, size=8, replace=False)] = True
    X = np.stack([make_sample(rng, bool(label) != s) for label, s in zip(y, swap)])
    order = rng.permutation(len(y))
    return X[order], y[order]
