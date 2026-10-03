import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = ROOT / "figures"
OUT = ROOT / "docs" / "writeup.pdf"

W, H = 8.27, 11.69  # A4 in inches
LEFT, RIGHT = 0.08, 0.92
SIZE = 9.5
WRAP = 100
LINE = 0.0149  # line height as a fraction of page height at SIZE


def load(name):
    return json.loads((RES / name).read_text())


class Page:
    def __init__(self, pdf):
        self.pdf = pdf
        self.fig = plt.figure(figsize=(W, H))
        self.y = 0.95

    def heading(self, text):
        self.y -= 0.006
        self.fig.text(LEFT, self.y, text, size=11.5, weight="bold", va="top")
        self.y -= 0.022

    def para(self, text, size=SIZE):
        lines = textwrap.wrap(text, WRAP)
        self.fig.text(LEFT, self.y, "\n".join(lines), size=size, va="top", linespacing=1.3)
        self.y -= LINE * len(lines) + 0.007

    def image(self, paths):
        n = len(paths)
        width = (RIGHT - LEFT) / n
        img = mpimg.imread(paths[0])
        height = width * img.shape[0] / img.shape[1] * W / H
        for i, p in enumerate(paths):
            ax = self.fig.add_axes([LEFT + i * width, self.y - height, width, height])
            ax.imshow(mpimg.imread(p))
            ax.axis("off")
        self.y -= height + 0.008

    def table(self, header, rows, col0=0.2):
        ax = self.fig.add_axes([LEFT, self.y - 0.022 * (len(rows) + 1), RIGHT - LEFT, 0.022 * (len(rows) + 1)])
        ax.axis("off")
        widths = [col0] + [(1 - col0) / (len(header) - 1)] * (len(header) - 1)
        t = ax.table(cellText=rows, colLabels=header, colWidths=widths, loc="center", cellLoc="center")
        t.auto_set_font_size(False)
        t.set_fontsize(SIZE)
        t.scale(1, 1.25)
        for (r, c), cell in t.get_celld().items():
            cell.set_edgecolor("0.75")
            if r == 0:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#e8eef6")
        self.y -= 0.022 * (len(rows) + 1) + 0.012

    def close(self, n):
        self.fig.text(0.5, 0.025, f"{n} / 2", size=8.5, ha="center", color="0.4")
        self.pdf.savefig(self.fig)
        plt.close(self.fig)


def pct(x):
    return f"{100 * x:.1f}"


def main():
    svm = load("svm_metrics.json")
    head = load("tune_head.json")
    hp = load("tune_hparams.json")
    final = load("final_comparison.json")
    comp_path = RES / "model_comparison.json"
    comp = json.loads(comp_path.read_text()) if comp_path.exists() else None

    hr = head["results"]
    sw = hp["sweeps"]
    acc = lambda s, v: next(r for r in sw[s] if r["value"] == v)["accuracy"]["mean"]
    hik, sv = final["HIKNet"]["metrics"], final["SVM"]["metrics"]
    n_eval = len(final["y"])
    k = len(comp["hiknet"]["folds"]) if comp else 10

    OUT.parent.mkdir(exist_ok=True)
    with PdfPages(OUT) as pdf:
        p = Page(pdf)
        p.fig.text(LEFT, p.y, "HIKNet: Head Impact Detection from Kinematic Data", size=16, weight="bold", va="top")
        p.y -= 0.03
        p.fig.text(LEFT, p.y, "Guru R Bharadwaj (PES1UG24CS177)  -  UE24CS352A Machine Learning mini project",
                   size=SIZE, va="top", color="0.3")
        p.y -= 0.03

        p.heading("1. Problem and data")
        p.para("Instrumented mouthguards record linear acceleration and angular velocity whenever a 10g threshold "
               "is crossed, but many recordings are not head impacts (chewing, biting, dropping the mouthguard). "
               "The task is binary classification of each 6 x 199 recording (lin acc x/y/z, ang vel x/y/z, "
               "1000 Hz, ~0.2 s) into real impact vs false positive. This is a PyTorch port and extension of "
               "HIKNet (Fanton, Gaudio, Ling, CS229, Stanford).")
        p.para("Important: the original lab data was never released, so all neural network results here come from "
               "a synthetic mock dataset with the same shape and class balance: 527 samples, 264 impacts and 263 "
               "false positives. Impacts are simulated as damped 18-32 Hz pulses, false positives ring faster with "
               "more 80-200 Hz energy, and 16 samples are drawn from the opposite class so the task is not trivially "
               "separable. The numbers below therefore say nothing about real-world performance; they only "
               "check that the pipeline and models behave sensibly.")

        p.heading("2. Method")
        p.para(f"SVM baseline. An RBF SVM (standardized inputs, default C and gamma) with greedy forward feature "
               f"selection by 10-fold CV ROC AUC, up to 10 features. On the public Wu et al. (2017) feature set "
               f"(411 features, 387 collegiate samples, 32 youth test samples) it reaches {pct(svm['cv']['accuracy'])}% "
               f"accuracy, {pct(svm['cv']['sensitivity'])}% sensitivity and ROC AUC {svm['cv']['roc_auc']:.3f} in "
               f"10-fold CV, and {pct(svm['test']['accuracy'])}% accuracy on the youth test set. For the mock data the "
               f"same procedure runs on time-domain and PSD features computed from the raw traces.")
        p.para("HIKNet. Two 1D conv blocks (Conv1d with 150 filters and kernel 15, ReLU, max-pool 2, dropout) over "
               "the 6 channels, then the (time x filter) map is treated as a one-channel image and passed through "
               "a 2D conv (150 filters, 3 x 15 kernel, stride 3 along filters), ReLU, a pooling head and a single "
               "logit. Trained with Adam (lr 1e-3, batch 32) and binary cross-entropy, with early stopping "
               "(patience 5) on validation loss, restoring the best epoch. Inputs are standardized per channel.")
        p.para("RecursiveNet. The second architecture from the original code: a fully 2D CNN on the 6 x 199 "
               "signal image with 3 x 3 convolutions (32 and 64 filters), dropout 0.25, two U-Net-style skip "
               "concatenations, and a 256-unit dense layer before the output. It has far more parameters "
               "than HIKNet because of the large flattened dense layer.")

        p.heading("3. Tuning")
        p.para(f"Each configuration was trained {head['repeats']} times with different seeds ({head['epochs']} "
               f"epochs max) on the 70/30 split and averaged. Output head: global average pooling "
               f"{pct(hr['gap']['accuracy']['mean'])}% accuracy (ROC AUC {hr['gap']['roc_auc']['mean']:.3f}), global "
               f"max pooling {pct(hr['gmp']['accuracy']['mean'])}%, flatten + dense only "
               f"{pct(hr['dense']['accuracy']['mean'])}%, so GAP was kept. Filters (15-200) barely matter "
               f"({pct(acc('filters', 15))}-{pct(acc('filters', 150))}%). Kernel width matters most: 15 gives "
               f"{pct(acc('kernel', 15))}% vs {pct(acc('kernel', 5))}% for 5 and {pct(acc('kernel', 31))}% for 31. "
               f"Dropout is flat from 0 to 0.5 and drops to {pct(acc('dropout', 0.6))}% at 0.6. The best setting was "
               f"filters {hp['best']['filters']}, kernel {hp['best']['kernel']}, dropout {hp['best']['dropout']}, "
               f"within noise of the default (dropout 0.4), which was kept.")
        p.image([FIG / "tuning.png"])
        p.close(1)

        p = Page(pdf)
        p.heading("4. Results")
        p.para(f"Final comparison on the held-out 30% of the mock data (n = {n_eval}, seed {final['seed']}). "
               f"The SVM uses features selected on the training part only. HIKNet stopped at epoch "
               f"{final['HIKNet']['best_epoch']}. RecursiveNet vs HIKNet is reported separately as {k}-fold CV means "
               f"(each fold's held-out part used for early stopping and scoring, as in the original code).")
        header = ["Model", "Accuracy", "Precision", "Specificity", "Sensitivity", "ROC AUC", "PR AUC"]
        keys = ["accuracy", "precision", "specificity", "sensitivity", "roc_auc", "pr_auc"]
        row = lambda name, m: [name] + [f"{m[k]:.3f}" for k in keys]
        rows = [row("HIKNet (split)", hik), row("SVM (split)", sv)]
        if comp:
            rows += [row(f"HIKNet ({k}-fold)", comp["hiknet"]["mean"]),
                     row(f"RecursiveNet ({k}-fold)", comp["recursivenet"]["mean"])]
        else:
            rows += [[f"RecursiveNet ({k}-fold)"] + ["pending"] * 6]
        p.table(header, rows, col0=0.26)
        p.image([FIG / "roc_curves.png", FIG / "pr_curves.png"])
        p.para(f"On the split HIKNet beats the SVM in accuracy ({pct(hik['accuracy'])}% vs {pct(sv['accuracy'])}%) "
               f"and especially sensitivity ({pct(hik['sensitivity'])}% vs {pct(sv['sensitivity'])}%); the SVM "
               f"is slightly more specific. ROC AUC is {hik['roc_auc']:.3f} vs {sv['roc_auc']:.3f}. The SVM's "
               f"selected features are mostly 20-30 Hz PSD bins, the band where the mock impacts were placed."
               + (f" In {k}-fold CV, RecursiveNet reaches {pct(comp['recursivenet']['mean']['accuracy'])}% accuracy "
                  f"vs {pct(comp['hiknet']['mean']['accuracy'])}% for HIKNet, with ROC AUC "
                  f"{comp['recursivenet']['mean']['roc_auc']:.3f} vs {comp['hiknet']['mean']['roc_auc']:.3f}; "
                  f"HIKNet gets there with far fewer parameters." if comp else ""))

        p.heading("5. Limitations")
        p.para("(1) Mock data: everything except the Wu et al. SVM numbers comes from synthetic signals whose "
               "class difference is a designed frequency shift, so absolute numbers are not comparable to the "
               "original report or to real deployment. (2) The evaluation split is also used for early stopping "
               "(as in the original code), so neural network scores are optimistically biased; a separate "
               "validation set would remove this. (3) The final table is a single 70/30 split with one seed "
               f"(n = {n_eval}); the 10-repeat tuning spread (std about 3 points of accuracy) shows single-split "
               "differences of a few points are within noise. (4) The SVM runs with default hyperparameters.")

        p.heading("6. Conclusion")
        p.para("The PyTorch port of HIKNet trains reliably end to end: global average pooling is clearly the "
               "right head, kernel width is the most sensitive hyperparameter, and filter count and moderate "
               "dropout matter little. On the mock data HIKNet outperforms a feature-selected SVM, mainly by "
               "catching more true impacts. Real conclusions need the lab recordings; the code is ready to "
               "rerun unchanged once data/data.mat and data/labels.mat are available.")
        p.close(2)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
