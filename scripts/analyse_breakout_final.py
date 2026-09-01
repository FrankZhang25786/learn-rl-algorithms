from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path("results/final_evaluation/breakout")
OUT = ROOT / "analysis"
OUT.mkdir(exist_ok=True)


FILES = {
    "teacher": ROOT / "teacher" / "returns_breakout.npy",
    "rnn_h32_seq20": ROOT / "rnn_h32_seq20" / "returns_breakout.npy",
    "rnn_h32_seq100": ROOT / "rnn_h32_seq100" / "returns_breakout.npy",
    "mlp_h32_seq20": ROOT / "mlp_h32_seq20" / "returns_breakout.npy",
    "mlp_h32_seq100": ROOT / "mlp_h32_seq100" / "returns_breakout.npy",
}


# ---------------------------------------------------------
# Load final-update score for each matched seed
# ---------------------------------------------------------
scores = {}

for name, path in FILES.items():

    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")

    data = np.load(path)

    if data.shape[0] != 16:
        raise ValueError(
            f"{name}: expected 16 seeds, got shape {data.shape}"
        )

    scores[name] = np.asarray(data[:, -1], dtype=float)


# ---------------------------------------------------------
# Bootstrap CI for mean
# ---------------------------------------------------------
def bootstrap_mean_ci(x, n_boot=20000, seed=42):

    x = np.asarray(x)

    rng = np.random.default_rng(seed)

    idx = rng.integers(
        0,
        len(x),
        size=(n_boot, len(x)),
    )

    boot_means = x[idx].mean(axis=1)

    return np.quantile(
        boot_means,
        [0.025, 0.975],
    )


# ---------------------------------------------------------
# Individual model summaries
# ---------------------------------------------------------
teacher_mean = scores["teacher"].mean()

summary_rows = []

for name, x in scores.items():

    ci_low, ci_high = bootstrap_mean_ci(x)

    summary_rows.append(
        {
            "model": name,
            "n": len(x),
            "mean": x.mean(),
            "median": np.median(x),
            "std": x.std(ddof=1),
            "min": x.min(),
            "max": x.max(),
            "bootstrap_ci95_low": ci_low,
            "bootstrap_ci95_high": ci_high,
            "teacher_retention_pct":
                100.0 * x.mean() / teacher_mean,
        }
    )


summary = pd.DataFrame(summary_rows)


# ---------------------------------------------------------
# Paired comparisons
# ---------------------------------------------------------
comparisons = [
    (
        "mlp_h32_seq20",
        "rnn_h32_seq20",
        "MLP vs RNN, seq20",
    ),
    (
        "mlp_h32_seq100",
        "rnn_h32_seq100",
        "MLP vs RNN, seq100",
    ),
    (
        "rnn_h32_seq100",
        "rnn_h32_seq20",
        "RNN seq100 vs seq20",
    ),
    (
        "mlp_h32_seq100",
        "mlp_h32_seq20",
        "MLP seq100 vs seq20",
    ),
]


paired_rows = []

for a, b, label in comparisons:

    xa = scores[a]
    xb = scores[b]

    diff = xa - xb

    ci_low, ci_high = bootstrap_mean_ci(diff)

    if np.allclose(diff, 0):
        statistic = 0.0
        p_value = 1.0
    else:
        result = wilcoxon(
            diff,
            alternative="two-sided",
        )
        statistic = result.statistic
        p_value = result.pvalue

    paired_rows.append(
        {
            "comparison": label,
            "n_pairs": len(diff),
            "mean_A": xa.mean(),
            "mean_B": xb.mean(),
            "mean_difference_A_minus_B": diff.mean(),
            "median_difference": np.median(diff),
            "paired_ci95_low": ci_low,
            "paired_ci95_high": ci_high,
            "A_wins": int((diff > 0).sum()),
            "B_wins": int((diff < 0).sum()),
            "ties": int((diff == 0).sum()),
            "wilcoxon_statistic": statistic,
            "wilcoxon_p": p_value,
        }
    )


paired = pd.DataFrame(paired_rows)


# ---------------------------------------------------------
# Save raw matched-seed values
# ---------------------------------------------------------
seed_scores = pd.DataFrame(
    {
        "seed_index": np.arange(16),
        **scores,
    }
)


seed_scores.to_csv(
    OUT / "breakout_seed_scores.csv",
    index=False,
)

summary.to_csv(
    OUT / "breakout_summary.csv",
    index=False,
)

paired.to_csv(
    OUT / "breakout_paired_comparisons.csv",
    index=False,
)


# ---------------------------------------------------------
# Print results
# ---------------------------------------------------------
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print("\n=== BREAKOUT MODEL SUMMARY ===\n")
print(summary.to_string(index=False))

print("\n=== BREAKOUT PAIRED COMPARISONS ===\n")
print(paired.to_string(index=False))

print("\nSaved:")
print(OUT / "breakout_seed_scores.csv")
print(OUT / "breakout_summary.csv")
print(OUT / "breakout_paired_comparisons.csv")