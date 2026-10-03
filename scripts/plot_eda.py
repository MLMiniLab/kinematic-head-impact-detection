import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data import DATA_DIR, FS, load_dataset
from src.spectrum import fft_amplitude

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
PRE_TRIGGER = 50
AXES = ["x", "y", "z"]
COLORS = ["tab:blue", "tab:orange", "tab:olive"]


def time_axis(n):
    return (np.arange(n) - PRE_TRIGGER) / FS


def plot_traces(ax, t, xyz, ylabel):
    for trace, name, color in zip(xyz, AXES, COLORS):
        ax.plot(t, trace, color=color, label=name)
    ax.plot(t, np.linalg.norm(xyz, axis=0), color="k", label="mag")
    ax.set_xlim(-0.01, 0.1)
    ax.set_ylabel(ylabel)


def plot_time_domain(X, y, impact_i, false_i, out):
    impacts = np.flatnonzero(y == 1)
    falses = np.flatnonzero(y == 0)
    picks = [("True impact", impacts[impact_i]), ("False positive", falses[false_i])]
    t = time_axis(X.shape[2])

    fig, axs = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    for col, (title, idx) in enumerate(picks):
        plot_traces(axs[0, col], t, X[idx, 3:], "rad/s")
        plot_traces(axs[1, col], t, X[idx, :3], "g")
        axs[0, col].set_title(f"{title} (sample {idx})")
        axs[1, col].set_xlabel("Time (s)")
    axs[0, 0].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def plot_spectra(ax, xyz, ylabel):
    f, amp = fft_amplitude(xyz)
    _, amp_mag = fft_amplitude(np.linalg.norm(xyz, axis=0))
    for a, name, color in zip(amp, AXES, COLORS):
        ax.plot(f, a, color=color, label=name)
    ax.plot(f, amp_mag, color="k", label="mag")
    ax.set_ylabel(ylabel)


def plot_freq_domain(X, y, impact_i, false_i, out):
    impacts = np.flatnonzero(y == 1)
    falses = np.flatnonzero(y == 0)
    picks = [("True impact", impacts[impact_i]), ("False positive", falses[false_i])]

    fig, axs = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    for col, (title, idx) in enumerate(picks):
        plot_spectra(axs[0, col], X[idx, 3:], "Amplitude (ang vel)")
        plot_spectra(axs[1, col], X[idx, :3], "Amplitude (lin acc)")
        axs[0, col].set_title(f"{title} (sample {idx})")
        axs[1, col].set_xlabel("Frequency (Hz)")
    axs[0, 0].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def plot_mean_spectrum(X, y, out):
    mags = np.stack([np.linalg.norm(X[:, :3], axis=1), np.linalg.norm(X[:, 3:], axis=1)], axis=1)
    f, amp = fft_amplitude(mags - mags.mean(axis=2, keepdims=True))
    amp = amp / amp.sum(axis=2, keepdims=True)

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))
    for i, name in enumerate(["lin acc mag", "ang vel mag"]):
        for label, color, text in [(1, "tab:green", "true impacts"), (0, "tab:red", "false positives")]:
            axs[i].plot(f, amp[y == label, i].mean(axis=0), color=color, label=text)
        axs[i].set_title(f"Mean normalized spectrum, {name}")
        axs[i].set_xlabel("Frequency (Hz)")
    axs[0].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--impact", type=int, default=4)
    parser.add_argument("--false", type=int, default=15)
    args = parser.parse_args()

    X, y = load_dataset(args.data_dir)
    FIG_DIR.mkdir(exist_ok=True)
    plot_time_domain(X, y, args.impact, args.false, FIG_DIR / "time_domain.png")
    plot_freq_domain(X, y, args.impact, args.false, FIG_DIR / "freq_domain.png")
    plot_mean_spectrum(X, y, FIG_DIR / "mean_spectrum.png")


if __name__ == "__main__":
    main()
