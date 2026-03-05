import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

SAVE_DIR = "./mmluFigures"
os.makedirs(SAVE_DIR, exist_ok=True)

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
print("Saved: mmlu_fig1_math_vs_non-math")