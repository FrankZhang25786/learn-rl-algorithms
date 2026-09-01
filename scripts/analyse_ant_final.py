"""Reproduce the formal matched-seed Ant n=16 statistical analysis."""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE_ROOT = REPO_ROOT / "results" / "final_evaluation" / "ant"
DEFAULT_OUTPUT_DIR = DEFAULT_ARCHIVE_ROOT
LOG_ROOT = REPO_ROOT / "results" / "log_archive" / "ant_formal_n16_20260901"

N_SEEDS = 16
N_UPDATES = 1464
N_BOOTSTRAP = 20_000
BOOTSTRAP_SEED = 42

CONDITIONS = {
    "teacher": {
        "label": "Teacher",
        "archive": "formal_n16_teacher",
        "checkpoint": "save_files/pretrained/multi_OPEN.npy",
        "parameters": 3571,
        "log": "teacher_retry.log",
    },
    "corrected_rnn_seq20": {
        "label": "Corrected RNN seq20",
        "provenance_label": "Corrected RNN seq20 @900",
        "archive": "formal_n16_corrected_rnn_h32_seq20_prngfix",
        "checkpoint": (
            "save_files/open_recurrent_distil/"
            "2026-09-01_00:48:04.331554/curr_param_8.npy"
        ),
        "parameters": 3571,
        "log": "formal_n16_corrected_rnn_h32_seq20_prngfix.log",
    },
    "corrected_rnn_seq100": {
        "label": "Corrected RNN seq100",
        "provenance_label": "Corrected RNN seq100 @900",
        "archive": "formal_n16_corrected_rnn_h32_seq100_prngfix",
        "checkpoint": (
            "save_files/open_recurrent_distil/"
            "2026-09-01_01:27:41.643380/curr_param_8.npy"
        ),
        "parameters": 3571,
        "log": "formal_n16_corrected_rnn_h32_seq100_prngfix.log",
    },
    "mlp_seq20": {
        "label": "MLP seq20",
        "provenance_label": "MLP seq20 @900",
        "archive": "formal_n16_mlp_h32_seq20",
        "checkpoint": (
            "save_files/open_recurrent_to_ff_distil/"
            "2026-08-24_22-53-51.517510/curr_param_8.npy"
        ),
        "parameters": 1923,
        "log": "formal_n16_mlp_h32_seq20.log",
    },
    "mlp_seq100": {
        "label": "MLP seq100",
        "provenance_label": "MLP seq100 @900",
        "archive": "formal_n16_mlp_h32_seq100",
        "checkpoint": (
            "save_files/open_recurrent_to_ff_distil/"
            "2026-08-25_00-11-42.707317/curr_param_8.npy"
        ),
        "parameters": 1923,
        "log": "formal_n16_mlp_h32_seq100.log",
    },
}

COMPARISONS = [
    ("A", "corrected_rnn_seq20", "mlp_seq20"),
    ("B", "corrected_rnn_seq100", "mlp_seq100"),
    ("C", "corrected_rnn_seq100", "corrected_rnn_seq20"),
    ("D", "mlp_seq100", "mlp_seq20"),
    ("E", "corrected_rnn_seq20", "teacher"),
    ("F", "corrected_rnn_seq100", "teacher"),
    ("G", "mlp_seq20", "teacher"),
    ("H", "mlp_seq100", "teacher"),
]

CSV_FIELDS = [
    "row_type",
    "id",
    "condition_A",
    "condition_B",
    "n",
    "mean_A",
    "mean_B",
    "sd",
    "median",
    "min",
    "max",
    "mean_difference",
    "median_difference",
    "ci95_low",
    "ci95_high",
    "A_wins",
    "ties",
    "B_wins",
    "wilcoxon_p",
    "scores",
]


def bootstrap_mean_ci(values):
    """Return the fixed-protocol percentile CI for a sample mean."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(
        0,
        len(values),
        size=(N_BOOTSTRAP, len(values)),
    )
    means = values[indices].mean(axis=1)
    return np.quantile(means, [0.025, 0.975])


def parse_runtime(log_path):
    text = log_path.read_text(encoding="utf-8")
    match = re.search(r"^runtime = ([0-9.]+)$", text, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"Runtime not found in {log_path}")
    return float(match.group(1))


def load_inputs(archive_root):
    scores = {}
    provenance = {}

    for key, condition in CONDITIONS.items():
        condition_dir = archive_root / condition["archive"]
        returns_path = condition_dir / "returns_ant.npy"
        metadata_path = condition_dir / "metadata.json"

        returns = np.load(returns_path)
        if returns.shape != (N_SEEDS, N_UPDATES):
            raise ValueError(
                f"{key}: expected {(N_SEEDS, N_UPDATES)}, "
                f"found {returns.shape}"
            )
        if not np.isfinite(returns).all():
            raise ValueError(f"{key}: returns contain non-finite values")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("num_runs") != N_SEEDS:
            raise ValueError(f"{key}: metadata num_runs is not {N_SEEDS}")
        if metadata.get("rng_base_key") != BOOTSTRAP_SEED:
            raise ValueError(f"{key}: metadata RNG base key is not 42")
        if metadata.get("parameter_count") != condition["parameters"]:
            raise ValueError(f"{key}: metadata parameter count mismatch")
        if metadata.get("returns_shape") != [N_SEEDS, N_UPDATES]:
            raise ValueError(f"{key}: metadata returns shape mismatch")
        if not str(metadata.get("checkpoint", "")).endswith(
            condition["checkpoint"]
        ):
            raise ValueError(f"{key}: metadata checkpoint mismatch")

        # The final column is the prespecified per-seed endpoint. Row order is
        # retained unchanged so every subsequent comparison remains paired.
        scores[key] = np.asarray(returns[:, -1], dtype=float)
        provenance[key] = {
            "sha256": metadata["checkpoint_sha256"],
            "runtime": parse_runtime(LOG_ROOT / condition["log"]),
            "shape": returns.shape,
        }

    return scores, provenance


def calculate_statistics(scores):
    summaries = {}
    for key, values in scores.items():
        ci_low, ci_high = bootstrap_mean_ci(values)
        summaries[key] = {
            "n": len(values),
            "mean": values.mean(),
            "sd": values.std(ddof=1),
            "median": np.median(values),
            "min": values.min(),
            "max": values.max(),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "scores": values,
        }

    paired = []
    for comparison_id, key_a, key_b in COMPARISONS:
        values_a = scores[key_a]
        values_b = scores[key_b]
        differences = values_a - values_b
        ci_low, ci_high = bootstrap_mean_ci(differences)
        p_value = wilcoxon(
            differences,
            alternative="two-sided",
        ).pvalue
        paired.append(
            {
                "id": comparison_id,
                "key_a": key_a,
                "key_b": key_b,
                "mean_a": values_a.mean(),
                "mean_b": values_b.mean(),
                "mean_difference": differences.mean(),
                "median_difference": np.median(differences),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "a_wins": int((differences > 0).sum()),
                "ties": int((differences == 0).sum()),
                "b_wins": int((differences < 0).sum()),
                "wilcoxon_p": p_value,
            }
        )

    return summaries, paired


def write_csv(path, summaries, paired):
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for key, summary in summaries.items():
            writer.writerow(
                {
                    "row_type": "condition",
                    "id": key,
                    "condition_A": key,
                    "condition_B": "",
                    "n": summary["n"],
                    "mean_A": summary["mean"],
                    "mean_B": "",
                    "sd": summary["sd"],
                    "median": summary["median"],
                    "min": summary["min"],
                    "max": summary["max"],
                    "mean_difference": "",
                    "median_difference": "",
                    "ci95_low": summary["ci_low"],
                    "ci95_high": summary["ci_high"],
                    "A_wins": "",
                    "ties": "",
                    "B_wins": "",
                    "wilcoxon_p": "",
                    "scores": "|".join(
                        str(float(value)) for value in summary["scores"]
                    ),
                }
            )

        for result in paired:
            writer.writerow(
                {
                    "row_type": "comparison",
                    "id": result["id"],
                    "condition_A": result["key_a"],
                    "condition_B": result["key_b"],
                    "n": N_SEEDS,
                    "mean_A": result["mean_a"],
                    "mean_B": result["mean_b"],
                    "sd": "",
                    "median": "",
                    "min": "",
                    "max": "",
                    "mean_difference": result["mean_difference"],
                    "median_difference": result["median_difference"],
                    "ci95_low": result["ci_low"],
                    "ci95_high": result["ci_high"],
                    "A_wins": result["a_wins"],
                    "ties": result["ties"],
                    "B_wins": result["b_wins"],
                    "wilcoxon_p": result["wilcoxon_p"],
                    "scores": "",
                }
            )


def format_p_value(p_value):
    if p_value < 0.0001:
        return f"{p_value:.7f}"
    return f"{p_value:.6f}"


def render_markdown(summaries, paired, provenance):
    lines = [
        "# Formal Ant n=16 analysis",
        "",
        "## Protocol and integrity",
        "",
        "This report analyses the final column of each archived `returns_ant.npy`, preserving the common seed order generated from RNG base key 42. All five arrays have shape `(16, 1464)` and contain only finite values. Statistics follow the previously fixed protocol: sample SD (`ddof=1`); 20,000 percentile bootstrap resamples with NumPy `default_rng(42)`; and two-sided paired SciPy Wilcoxon tests without result-dependent method changes. The Ant configuration is shared across conditions: Brax Ant, 2,048 environments, 10 steps, 30,000,000 timesteps, 4 update epochs, 32 minibatches, normalized observations/rewards, clipped actions, and 1,464 PPO updates.",
        "",
        "## Provenance",
        "",
        "| Condition | Checkpoint | SHA256 | Parameters | Evaluator runtime (s) | Shape |",
        "|---|---|---|---:|---:|---:|",
    ]

    for key, condition in CONDITIONS.items():
        item = provenance[key]
        label = condition.get("provenance_label", condition["label"])
        lines.append(
            f"| {label} | `{condition['checkpoint']}` | "
            f"`{item['sha256']}` | {condition['parameters']:,} | "
            f"{item['runtime']:.3f} | `{item['shape']}` |"
        )

    lines.extend(
        [
            "",
            "The formal queue records source commit `529aa2257839dd8ba359600a0114d8232f2106e9`. This commit is not embedded directly in each condition's `metadata.json`; it is preserved in `results/log_archive/ant_formal_n16_20260901/after_teacher.log` and `queue.log`.",
            "",
            "## Condition summaries",
            "",
            "| Condition | Mean | SD | Median | Min | Max | Bootstrap 95% CI |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for key, condition in CONDITIONS.items():
        summary = summaries[key]
        lines.append(
            f"| {condition['label']} | {summary['mean']:.3f} | "
            f"{summary['sd']:.3f} | {summary['median']:.3f} | "
            f"{summary['min']:.3f} | {summary['max']:.3f} | "
            f"[{summary['ci_low']:.3f}, {summary['ci_high']:.3f}] |"
        )

    lines.extend(["", "### Exact final scores, seed indices 0–15", ""])
    for key, condition in CONDITIONS.items():
        scores = ", ".join(
            f"{value:.6f}" for value in summaries[key]["scores"]
        )
        lines.append(f"- **{condition['label']}:** {scores}.")

    lines.extend(
        [
            "",
            "## Paired comparisons",
            "",
            "Differences are A minus B; confidence intervals bootstrap the 16 paired differences.",
            "",
            "| ID / comparison | Mean A | Mean B | Mean difference | Median difference | Bootstrap 95% CI | A wins | Ties | B wins | Wilcoxon p |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for result in paired:
        label_a = CONDITIONS[result["key_a"]]["label"]
        label_b = CONDITIONS[result["key_b"]]["label"]
        if result["id"] == "C":
            label_b = "corrected RNN seq20"
        elif result["key_b"] == "teacher":
            label_b = "teacher"
        lines.append(
            f"| {result['id']}. {label_a} vs {label_b} | "
            f"{result['mean_a']:.3f} | {result['mean_b']:.3f} | "
            f"{result['mean_difference']:.3f} | "
            f"{result['median_difference']:.3f} | "
            f"[{result['ci_low']:.3f}, {result['ci_high']:.3f}] | "
            f"{result['a_wins']} | {result['ties']} | {result['b_wins']} | "
            f"{format_p_value(result['wilcoxon_p'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "1. **Architecture:** MLP decisively outperforms corrected RNN at both sequence lengths on Ant. Every matched seed favours MLP, with large paired effects and p=3.05e-05 in both comparisons.",
            "2. **Cross-environment consistency:** Ant reinforces the Breakout architecture result and the strong CartPole seq100 result. CartPole seq20 was less decisive, so the defensible claim is that the MLP advantage is robust for seq100 and for the harder/longer-horizon Breakout and Ant settings, not necessarily uniformly significant in every condition.",
            "3. **Sequence length:** Seq100 greatly improves both Ant students: +908.95 for RNN and +3243.44 for MLP, each winning all 16 pairs. This is much stronger than CartPole's ambiguous RNN sequence effect and agrees with Breakout's corrected seq100 RNN advantage.",
            "4. **Teacher proximity:** Corrected RNN seq20 fails badly and is negative on every seed. Corrected RNN seq100 reaches only about 20% of the teacher mean. MLP seq20 reaches about 25%; MLP seq100 exceeds the teacher mean by about 51% and wins 15/16 matched seeds.",
            "5. **Carry-horizon hypothesis:** Ant strengthens, but does not prove, the hypothesis. The recurrent student remains far below its recurrent teacher and stateless MLP counterpart over the longest optimizer horizon, while longer distillation sequences substantially help. This pattern is mechanistically compatible with hidden-state horizon mismatch; it is not a causal isolation experiment.",
            "6. **Defensible cross-environment conclusion:** After correcting PRNG propagation, RNN weakness persists. MLP is decisively better on Ant and Breakout and on CartPole seq100. Longer sequences help RNN on Breakout and Ant, but not reliably on CartPole; they help MLP strongly on CartPole and Ant. Effects therefore depend on both architecture and environment/horizon.",
            "7. **Primary versus limitation:** Primary dissertation results should be the matched-seed condition summaries and prespecified paired tests from corrected @900 checkpoints. The PRNG issue, synthetic-to-deployment mismatch, recurrent carry horizon, parameter-count difference, and absence of a targeted carry-ablation experiment should be presented as limitations or mechanistic hypotheses—not as established causal explanations.",
            "",
            "## Source paths",
            "",
            "- `results/final_evaluation/ant/formal_n16_teacher/`",
            "- `results/final_evaluation/ant/formal_n16_corrected_rnn_h32_seq20_prngfix/`",
            "- `results/final_evaluation/ant/formal_n16_corrected_rnn_h32_seq100_prngfix/`",
            "- `results/final_evaluation/ant/formal_n16_mlp_h32_seq20/`",
            "- `results/final_evaluation/ant/formal_n16_mlp_h32_seq100/`",
            "- `results/log_archive/ant_formal_n16_20260901/teacher_retry.log`",
            "- `results/log_archive/ant_formal_n16_20260901/after_teacher.log`",
            "- `results/log_archive/ant_formal_n16_20260901/formal_n16_corrected_rnn_h32_seq20_prngfix.log`",
            "- `results/log_archive/ant_formal_n16_20260901/formal_n16_corrected_rnn_h32_seq100_prngfix.log`",
            "- `results/log_archive/ant_formal_n16_20260901/formal_n16_mlp_h32_seq20.log`",
            "- `results/log_archive/ant_formal_n16_20260901/formal_n16_mlp_h32_seq100.log`",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce the formal matched-seed Ant n=16 analysis."
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=DEFAULT_ARCHIVE_ROOT,
        help="Directory containing the five formal Ant condition archives.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for the Markdown report and summary CSV.",
    )
    args = parser.parse_args()

    archive_root = args.archive_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scores, provenance = load_inputs(archive_root)
    summaries, paired = calculate_statistics(scores)

    csv_path = output_dir / "ant_formal_n16_summary.csv"
    report_path = output_dir / "ant_formal_n16_analysis.md"
    write_csv(csv_path, summaries, paired)
    report_path.write_text(
        render_markdown(summaries, paired, provenance),
        encoding="utf-8",
    )

    print(f"Wrote {report_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
