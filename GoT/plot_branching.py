"""Plots for the offline GoT analysis.

Produces three figures from the CSVs in results/:
  fig1_collapse_fingerprint.{png,pdf}  - pass@1 vs pass@3 vs maj@3 per model
  fig2_killed_rescue.{png,pdf}         - cross-model rescue on the killed set
  fig3_full_ensemble.{png,pdf}         - same on full Math500
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

MODEL_ORDER = ["base", "sft", "drpo", "openai_nano"]
MODEL_LABEL = {
    "base": "Base",
    "sft": "SFT",
    "drpo": "DRPO",
    "openai_nano": "nano",
}
MODEL_COLOR = {
    "base": "#4C78A8",
    "sft": "#59A14F",
    "drpo": "#E15759",
    "openai_nano": "#9C755F",
}

STRATEGY_ORDER = [
    "drpo_self_maj3",
    "drpo+base",
    "drpo+sft",
    "drpo+openai_nano",
    "drpo+all_donors",
]
STRATEGY_LABEL = {
    "drpo_self_maj3":  "DRPO self maj@3",
    "drpo+base":       "DRPO + base",
    "drpo+sft":        "DRPO + sft",
    "drpo+openai_nano":"DRPO + nano",
    "drpo+all_donors": "DRPO + all donors",
}


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "-",
        "savefig.bbox": "tight",
        "savefig.dpi": 200,
    })


def save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(path)
        print(f"  wrote {path}")


def fig_collapse(summary_csv, out_dir):
    df = pd.read_csv(summary_csv).set_index("model_tag").loc[MODEL_ORDER]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    metrics = ["pass_at_1", "maj_at_3", "pass_at_3"]
    metric_label = {"pass_at_1": "pass@1", "maj_at_3": "maj@3", "pass_at_3": "pass@3"}
    metric_color = {"pass_at_1": "#BBBBBB", "maj_at_3": "#4477AA", "pass_at_3": "#222222"}

    x = np.arange(len(MODEL_ORDER))
    w = 0.26
    for i, m in enumerate(metrics):
        ax.bar(x + (i - 1) * w, df[m].values, w,
               label=metric_label[m], color=metric_color[m],
               edgecolor="white", linewidth=0.5)

    # Annotate maj@3 - pass@1 gap above each model.
    gap = (df["maj_at_3"] - df["pass_at_1"]).values
    top = df[metrics].max(axis=1).values
    for xi, g, t in zip(x, gap, top):
        ax.annotate(f"+{g*100:.1f}", xy=(xi, t + 0.012),
                    ha="center", va="bottom", fontsize=10,
                    color="#333333", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABEL[m] for m in MODEL_ORDER])
    ax.set_ylabel("Accuracy on Math500")
    ax.set_ylim(0, 0.95)
    ax.set_title("Self-consistency gap: collapsed models gain little from voting\n"
                 "(annotation = maj@3 − pass@1, in points)")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    save(fig, out_dir, "fig1_collapse_fingerprint")
    plt.close(fig)


def fig_rescue(killed_csv, out_dir, n_killed):
    df = pd.read_csv(killed_csv).set_index("strategy").loc[STRATEGY_ORDER].reset_index()
    fig, ax = plt.subplots(figsize=(7.0, 3.6))

    y = np.arange(len(df))[::-1]  # top strategy first
    pct = df["recovered_pct"].values
    n_rec = df["voted_correct"].values

    bars = ax.barh(y, pct, color=["#BBBBBB"] + ["#88CCEE"] * 3 + ["#117733"],
                   edgecolor="white", linewidth=0.5)
    for yi, p, n in zip(y, pct, n_rec):
        ax.annotate(f"{n}/{n_killed}  ({p*100:.0f}%)",
                    xy=(p + 0.015, yi), va="center", fontsize=10)

    ax.set_yticks(y)
    ax.set_yticklabels([STRATEGY_LABEL[s] for s in df["strategy"]])
    ax.set_xlabel("Fraction of killed problems recovered")
    ax.set_xlim(0, max(pct.max() * 1.25, 0.2))
    ax.set_title(f"Cross-model GoT intervention recovers DRPO collapse\n"
                 f"({n_killed} problems where Base solved at least once but DRPO 0/3)")
    fig.tight_layout()
    save(fig, out_dir, "fig2_killed_rescue")
    plt.close(fig)


def fig_full_ensemble(full_csv, summary_csv, out_dir):
    full = pd.read_csv(full_csv).set_index("strategy").loc[STRATEGY_ORDER].reset_index()
    summary = pd.read_csv(summary_csv).set_index("model_tag")

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    x = np.arange(len(full))
    pct = full["recovered_pct"].values
    bars = ax.bar(x, pct, color=["#E15759", "#4C78A8", "#59A14F", "#9C755F", "#117733"],
                  edgecolor="white", linewidth=0.5)
    for xi, p in zip(x, pct):
        ax.annotate(f"{p*100:.1f}", xy=(xi, p + 0.005), ha="center", fontsize=10)

    # Reference lines: best single model pass@1 and pass@3.
    best_p1 = summary["pass_at_1"].max()
    best_p3 = summary["pass_at_3"].max()
    ax.axhline(best_p1, color="#666666", linestyle="--", linewidth=1)
    ax.text(len(x) - 0.4, best_p1 + 0.005, f"best single pass@1 = {best_p1:.3f}",
            ha="right", fontsize=9, color="#666666")
    ax.axhline(best_p3, color="#666666", linestyle=":", linewidth=1)
    ax.text(len(x) - 0.4, best_p3 + 0.005, f"best single pass@3 = {best_p3:.3f}",
            ha="right", fontsize=9, color="#666666")

    ax.set_xticks(x)
    ax.set_xticklabels([STRATEGY_LABEL[s] for s in full["strategy"]],
                       rotation=15, ha="right")
    ax.set_ylabel("maj-vote accuracy on full Math500")
    ax.set_ylim(0.65, 0.88)
    ax.set_title("Pooling branches across models beats every single model")
    fig.tight_layout()
    save(fig, out_dir, "fig3_full_ensemble")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--out_dir", default="results/figures")
    parser.add_argument("--n_killed", type=int, default=25)
    args = parser.parse_args()

    style()
    results_dir = os.path.join(HERE, args.results_dir)
    out_dir = os.path.join(HERE, args.out_dir)

    fig_collapse(os.path.join(results_dir, "branching_summary.csv"), out_dir)
    fig_rescue(os.path.join(results_dir, "intervention_drpo_killed.csv"),
               out_dir, args.n_killed)
    fig_full_ensemble(os.path.join(results_dir, "intervention_drpo_full.csv"),
                      os.path.join(results_dir, "branching_summary.csv"),
                      out_dir)


if __name__ == "__main__":
    main()
