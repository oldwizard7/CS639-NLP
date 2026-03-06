import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
import random

SAVE_DIR = "./mmluFigures"
os.makedirs(SAVE_DIR, exist_ok=True)

plt.rcParams.update({
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

mmlu_raw = load_dataset("cais/mmlu", "all", split="test")
mmlu_df = pd.DataFrame(mmlu_raw)
print(mmlu_df.head)
print(mmlu_df.info())
print(mmlu_df["subject"].unique())

#separate mathematical questions and non-mathematical questions
MATH_SUBJECTS = [
    "abstract_algebra",
    "college_mathematics",
    "high_school_mathematics",
    "high_school_statistics",
    "college_physics",
    "high_school_physics",
    "conceptual_physics",
    "electrical_engineering",
    "astronomy"
]

mmlu_df["category"] = mmlu_df["subject"].apply(
    lambda s: "Mathematical" if s in MATH_SUBJECTS else "Non-Mathematical"
)

mmlu_df["question_len"] = mmlu_df["question"].apply(lambda x: len(x.split()))

fig, ax = plt.subplots(figsize=(6, 4))
fig.suptitle("Mathematical vs. Non-Mathematical")

cat_counts = mmlu_df["category"].value_counts()
bars = ax.bar(cat_counts.index, cat_counts.values, edgecolor="white", width=0.5)

for bar, v in zip(bars, cat_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 50, str(v), ha="center")

ax.set_ylabel("Number of Problems")
ax.set_ylim(0, max(cat_counts.values) * 1.15)

plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/mmlu_fig1_math_vs_non-math.pdf", bbox_inches="tight")
plt.savefig(f"{SAVE_DIR}/mmlu_fig1_math_vs_non-math.png", bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(figsize=(7, 6))
fig.suptitle("Top 10 Subjects by Problem Count")

top_subjects = mmlu_df["subject"].value_counts().head(10)
colors = sns.color_palette("muted", 10)

ax.barh(
    top_subjects.index[::-1],
    top_subjects.values[::-1],
    color=colors[::-1],
    edgecolor="white"
)
ax.set_xlabel("Number of Problems")

plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/mmlu_fig2_top10.pdf", bbox_inches="tight")
plt.savefig(f"{SAVE_DIR}/mmlu_fig2_top10.png", bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(figsize=(7, 4))
fig.suptitle("Question Length Distribution")

_, bin_edges = np.histogram(mmlu_df["question_len"], bins="scott")

ax.hist(mmlu_df["question_len"], bins=bin_edges,
        alpha=0.7, edgecolor="white", density=False, label="All")
ax.hist(mmlu_df[mmlu_df["category"] == "Mathematical"]["question_len"], bins=bin_edges,
        alpha=0.5, edgecolor="white", density=False, label="Math")
ax.hist(mmlu_df[mmlu_df["category"] == "Non-Mathematical"]["question_len"], bins=bin_edges,
        alpha=0.5, edgecolor="white", density=False, label="Non-Math")

median_all = mmlu_df["question_len"].median()
median_math = mmlu_df[mmlu_df["category"] == "Mathematical"]["question_len"].median()
median_nonmath = mmlu_df[mmlu_df["category"] == "Non-Mathematical"]["question_len"].median()

ax.axvline(median_all, color="black", linestyle="--", linewidth=1.0, label=f"Overall median: {median_all:.0f}")
ax.axvline(median_math, color="steelblue", linestyle="--", linewidth=1.0, label=f"Math median: {median_math:.0f}")
ax.axvline(median_nonmath, color="orange", linestyle="--", linewidth=1.0, label=f"Non-Math median: {median_nonmath:.0f}")

ax.set_xlabel("Number of Words")
ax.set_ylabel("Count")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/mmlu_fig3_question_length.pdf", bbox_inches="tight")
plt.savefig(f"{SAVE_DIR}/mmlu_fig3_question_length.png", bbox_inches="tight")
plt.show()

def get_representative(df, category, seed=42):
    random.seed(seed)
    subset = df[df["category"] == category].copy()

    q25 = subset["question_len"].quantile(0.25)
    q75 = subset["question_len"].quantile(0.75)
    subset = subset[
        (subset["question_len"] >= q25) &
        (subset["question_len"] <= q75)
    ]
    
    sample = subset.sample(1, random_state=seed).iloc[0]
    return sample

math_case = get_representative(mmlu_df, "Mathematical")
nonmath_case = get_representative(mmlu_df, "Non-Mathematical")

choices_labels = ["A", "B", "C", "D"]

def print_case(label, row):
    print(f"{'='*60}")
    print(f"[{label}]")
    print(f"Subject  : {row['subject']}")
    print(f"Category : {row['category']}")
    print(f"Length   : {row['question_len']} words")
    print(f"\nQuestion :\n{row['question']}")
    print(f"\nChoices  :")
    for i, choice in enumerate(row["choices"]):
        print(f"  ({choices_labels[i]}) {choice}")
    correct = choices_labels[row["answer"]]
    print(f"\nAnswer   : ({correct}) {row['choices'][row['answer']]}")
    print()

print_case("Case A: Mathematical", math_case)
print_case("Case B: Non-Mathematical", nonmath_case)