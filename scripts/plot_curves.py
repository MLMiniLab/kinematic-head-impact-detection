import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import auc, precision_recall_curve, roc_curve

ROOT = Path(__file__).resolve().parent.parent
COLORS = {"HIKNet": "#1f5fa8", "SVM": "#d0612a"}


def plot(res, y, kind, out):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    for name in ["HIKNet", "SVM"]:
        s = res[name]["scores"]
        if kind == "roc":
            fpr, tpr, _ = roc_curve(y, s)
            ax.plot(fpr, tpr, color=COLORS[name], lw=2, label=f"{name} (AUC = {auc(fpr, tpr):.3f})")
        else:
            p, r, _ = precision_recall_curve(y, s)
            ax.plot(r, p, color=COLORS[name], lw=2, drawstyle="steps-post", label=f"{name} (AUC = {auc(r, p):.3f})")
    if kind == "roc":
        ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--", label="Chance")
        ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    else:
        prev = sum(y) / len(y)
        ax.axhline(prev, color="0.6", lw=1, ls="--", label=f"Chance ({prev:.2f})")
        ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
    ax.set(xlim=(-0.01, 1.01), ylim=(-0.01, 1.01), aspect="equal")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right" if kind == "roc" else "lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=ROOT / "results" / "final_comparison.json")
    parser.add_argument("--out-dir", default=ROOT / "figures")
    args = parser.parse_args()

    res = json.loads(Path(args.results).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    plot(res, res["y"], "roc", out_dir / "roc_curves.png")
    plot(res, res["y"], "pr", out_dir / "pr_curves.png")


if __name__ == "__main__":
    main()
