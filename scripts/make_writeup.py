import csv
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUTPUT = ROOT / "docs" / "writeup.pdf"
PAGE_SIZE = (8.27, 11.69)


def add_heading(fig, y, text):
    fig.text(0.08, y, text, fontsize=10.5, weight="bold", va="top")
    return y - 0.027


def add_paragraph(fig, y, text, width=104, fontsize=8.4, line_height=0.017):
    lines = textwrap.wrap(text, width=width)
    fig.text(
        0.08,
        y,
        "\n".join(lines),
        fontsize=fontsize,
        va="top",
        linespacing=1.2,
    )
    return y - line_height * len(lines) - 0.012


def add_table(fig, y, headers, rows, widths=None, fontsize=7.2, height=None):
    width = 0.84
    left = 0.08
    height = height or 0.027 * (len(rows) + 1)
    ax = fig.add_axes([left, y - height, width, height])
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        colWidths=widths,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 1.22)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("0.75")
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e8eef6")
    return y - height - 0.016


def page(title, number):
    fig = plt.figure(figsize=PAGE_SIZE)
    fig.text(0.08, 0.96, title, fontsize=15, weight="bold", va="top")
    fig.text(0.5, 0.025, f"{number} / 2", fontsize=8, ha="center", color="0.4")
    return fig, 0.91


def main():
    comparison = json.loads((RESULTS / "final_comparison.json").read_text())
    with (RESULTS / "final_comparison.csv").open(
        newline="", encoding="utf-8"
    ) as result_file:
        rows = list(csv.DictReader(result_file))
    rows_by_model = {row["model"]: row for row in rows}

    def leaders(metric):
        best = max(float(row[metric]) for row in rows)
        return ", ".join(
            row["model"] for row in rows if float(row[metric]) == best
        )

    OUTPUT.parent.mkdir(exist_ok=True)
    with PdfPages(OUTPUT) as pdf:
        fig, y = page("Kinematic Head Impact Detection", 1)
        y = add_paragraph(
            fig,
            y,
            "Machine-learning mini-project comparing HIKNet, RecursiveNet, "
            "support vector machine (SVM), and XGBoost for classifying "
            "kinematic recordings as real impacts or false positives.",
            fontsize=9,
        )
        y = add_heading(fig, y, "Problem and dataset")
        y = add_paragraph(
            fig,
            y,
            "The reference paper reports 527 mouthguard recordings collected "
            "from Stanford football athletes. Those original recordings are "
            "not available in this repository. Our experiment therefore uses "
            "synthetic data: 527 samples, six channels, 199 time points per "
            "channel, 264 impact labels, and 263 false-positive labels. "
            "Signals are simulated from damped pulses, higher-frequency "
            "components, and noise; they are not measured athlete data.",
        )
        y = add_paragraph(
            fig,
            y,
            "The generator seed for this comparison is "
            f"{comparison.get('synthetic_seed')}; the holdout split seed is "
            f"{comparison['seed']}. The synthetic signals reproduce the "
            "stated format and class counts, not the real dataset "
            "distribution. These results therefore do not establish "
            "real-world performance.",
        )
        y = add_heading(fig, y, "Method and implementation")
        y = add_paragraph(
            fig,
            y,
            "A stratified 70/30 split gives 369 training and 158 held-out "
            "evaluation samples (79 per class). HIKNet and RecursiveNet "
            "receive per-sample, per-channel standardized raw signals. "
            "A separate 15% of the outer training data is used for neural "
            "early stopping; the held-out set is scored only after training. "
            "SVM and XGBoost use the same 176 per-recording engineered "
            "features: peaks, durations, and spectral power features. "
            "SVM standardizes features in a training-fitted pipeline.",
        )
        y = add_table(
            fig,
            y,
            ["Model", "Input", "Main configuration"],
            [
                [
                    "HIKNet",
                    "Standardized raw signals",
                    "150 filters; kernel 15; dropout 0.4; GAP head",
                ],
                [
                    "RecursiveNet",
                    "Standardized raw signals",
                    "32/64 filters; 3x3 conv; dropout 0.25; dense 256",
                ],
                [
                    "SVM",
                    "176 engineered features",
                    "RBF kernel; StandardScaler; default SVC parameters",
                ],
                [
                    "XGBoost",
                    "Same 176 features",
                    "200 trees; depth 4; learning rate 0.05; subsampling 0.8",
                ],
            ],
            widths=[0.15, 0.24, 0.61],
            fontsize=7,
            height=0.17,
        )
        y = add_paragraph(
            fig,
            y,
            "Neural networks use Adam (learning rate 0.001), batch size 32, "
            "binary cross-entropy with logits, at most 50 epochs, and "
            "early-stopping patience 5. XGBoost uses a binary logistic "
            "objective and log-loss fitting metric. XGBoost was selected as "
            "the additional model because boosted trees can learn nonlinear "
            "interactions in engineered tabular features.",
            fontsize=8,
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig, y = page("Results, comparison, and limitations", 2)
        metric_names = [
            ("accuracy", "Accuracy"),
            ("precision", "Precision"),
            ("specificity", "Specificity"),
            ("sensitivity", "Sensitivity"),
            ("roc_auc", "ROC-AUC"),
            ("pr_auc", "PR-AUC"),
        ]
        y = add_heading(fig, y, "Four-model results")
        result_table = [
            [row["model"]]
            + [f"{float(row[key]):.3f}" for key, _ in metric_names]
            for row in rows
        ]
        y = add_table(
            fig,
            y,
            ["Model"] + [label for _, label in metric_names],
            result_table,
            widths=[0.19] + [0.135] * len(metric_names),
            fontsize=7.2,
            height=0.12,
        )
        y = add_paragraph(
            fig,
            y,
            f"On this single synthetic-data split, {leaders('accuracy')} "
            f"had the highest accuracy; {leaders('precision')} had the "
            f"highest precision; {leaders('specificity')} had the highest "
            f"specificity; {leaders('roc_auc')} had the highest ROC-AUC; "
            f"and {leaders('pr_auc')} had the highest PR-AUC. These findings "
            "describe this split only and are not evidence that XGBoost "
            "outperforms the paper's models on Stanford recordings.",
        )
        y = add_heading(fig, y, "Paper result versus this experiment")
        y = add_paragraph(
            fig,
            y,
            "The paper abstract reports HIKNet accuracy 98.2%, precision "
            "97.6%, specificity 96.7%, and sensitivity 99.3%. Our HIKNet "
            f"values are {float(rows_by_model['HIKNet']['accuracy']):.1%}, "
            f"{float(rows_by_model['HIKNet']['precision']):.1%}, "
            f"{float(rows_by_model['HIKNet']['specificity']):.1%}, and "
            f"{float(rows_by_model['HIKNet']['sensitivity']):.1%}, "
            "respectively. This is a contextual comparison, not a "
            "like-for-like reproduction: the paper used real recordings, "
            "whereas this run used simulated signals and a different "
            "evaluation procedure.",
        )
        y = add_heading(fig, y, "Conclusions and limitations")
        y = add_paragraph(
            fig,
            y,
            "The four-model pipeline runs on a common dataset split and "
            "reports accuracy, precision, specificity, sensitivity, "
            "ROC-AUC, and PR-AUC. The final comparison used synthetic "
            f"generator seed {comparison.get('synthetic_seed')} and outer "
            "split seed "
            f"{comparison['seed']}. The generator is artificial and encodes "
            "assumed class-related signal patterns. The result is one "
            "holdout run, not a multi-seed estimate and not validation on "
            "real mouthguard data.",
        )
        y = add_paragraph(
            fig,
            y,
            "The separate Wu 2017 workbook (387 training samples, 411 "
            "features, and a 32-sample test set) belongs to a different SVM "
            "workflow. It is not used for the SVM row above. The main "
            "experiment is implemented in scripts/run_final_comparison.py; "
            "metrics are saved to results/final_comparison.csv and "
            "results/final_comparison.json.",
            fontsize=8,
        )
        pdf.savefig(fig)
        plt.close(fig)

    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
