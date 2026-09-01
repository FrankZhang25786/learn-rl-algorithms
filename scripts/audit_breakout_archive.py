from pathlib import Path
import hashlib
import re

import numpy as np
import pandas as pd

from algorithms.utils.configs import all_configs


ROOT = Path("results/final_evaluation/breakout")
OUT = ROOT / "analysis"
OUT.mkdir(exist_ok=True)

EXPECTED = {
    "teacher": {
        "checkpoint": "save_files/pretrained/multi_OPEN.npy",
        "evaluator": "algorithms/evaluation/eval_open_recurrent_bb.py",
        "params": 3571,
        "seq": "NA",
    },
    "rnn_h32_seq20": {
        "checkpoint":
            "save_files/open_recurrent_distil/"
            "2026-08-10_17:04:41.684883/curr_param_8.npy",
        "evaluator": "algorithms/evaluation/eval_open_recurrent_bb.py",
        "params": 3571,
        "seq": "20",
    },
    "rnn_h32_seq100": {
        "checkpoint":
            "save_files/open_recurrent_distil/"
            "2026-08-10_19:51:22.871913/curr_param_8.npy",
        "evaluator": "algorithms/evaluation/eval_open_recurrent_bb.py",
        "params": 3571,
        "seq": "100",
    },
    "mlp_h32_seq20": {
        "checkpoint":
            "save_files/open_recurrent_to_ff_distil/"
            "2026-08-24_22-53-51.517510/curr_param_8.npy",
        "evaluator": "algorithms/evaluation/eval_open_ff_bb.py",
        "params": 1923,
        "seq": "20",
    },
    "mlp_h32_seq100": {
        "checkpoint":
            "save_files/open_recurrent_to_ff_distil/"
            "2026-08-25_00-11-42.707317/curr_param_8.npy",
        "evaluator": "algorithms/evaluation/eval_open_ff_bb.py",
        "params": 1923,
        "seq": "100",
    },
}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def metadata_value(text, key):
    match = re.search(
        rf"^{re.escape(key)}:\s*(.+)$",
        text,
        flags=re.MULTILINE,
    )
    return match.group(1).strip() if match else None


manifest = pd.read_csv(ROOT / "manifest.csv")

rows = []
all_pass = True

checkpoint_hashes = {}
returns_hashes = {}

for condition, expected in EXPECTED.items():

    errors = []

    condition_dir = ROOT / condition
    returns_path = condition_dir / "returns_breakout.npy"
    log_path = condition_dir / "evaluation.log"
    metadata_path = condition_dir / "metadata.txt"
    checkpoint_path = Path(expected["checkpoint"])
    evaluator_path = Path(expected["evaluator"])

    # --------------------------------------------------
    # Required files
    # --------------------------------------------------
    for p in [
        condition_dir,
        returns_path,
        log_path,
        metadata_path,
        checkpoint_path,
        evaluator_path,
    ]:
        if not p.exists():
            errors.append(f"missing:{p}")

    if errors:
        rows.append({
            "condition": condition,
            "status": "FAIL",
            "errors": "; ".join(errors),
        })
        all_pass = False
        continue

    # --------------------------------------------------
    # Raw returns
    # --------------------------------------------------
    returns = np.load(returns_path)

    if returns.shape != (16, 1220):
        errors.append(
            f"returns_shape={returns.shape}, expected=(16,1220)"
        )

    if not np.isfinite(returns).all():
        errors.append("non_finite_returns")

    final_mean = float(returns[:, -1].mean())

    # --------------------------------------------------
    # Checkpoint
    # --------------------------------------------------
    params = np.load(checkpoint_path, allow_pickle=True)
    parameter_count = int(np.asarray(params).size)

    if parameter_count != expected["params"]:
        errors.append(
            f"params={parameter_count}, expected={expected['params']}"
        )

    checkpoint_hash = sha256(checkpoint_path)
    returns_hash = sha256(returns_path)

    checkpoint_hashes[condition] = checkpoint_hash
    returns_hashes[condition] = returns_hash

    # --------------------------------------------------
    # Metadata
    # --------------------------------------------------
    metadata = metadata_path.read_text()

    if metadata_value(metadata, "condition") != condition:
        errors.append("metadata_condition_mismatch")

    if metadata_value(metadata, "checkpoint") != expected["checkpoint"]:
        errors.append("metadata_checkpoint_mismatch")

    evaluator_meta = (
        metadata_value(metadata, "evaluation_script")
        or metadata_value(metadata, "evaluator")
    )

    if evaluator_meta != expected["evaluator"]:
        errors.append("metadata_evaluator_mismatch")

    seq_meta = (
        metadata_value(metadata, "distillation_seq_length")
        or metadata_value(metadata, "seq_length")
    )

    if str(seq_meta) != expected["seq"]:
        errors.append("metadata_seq_mismatch")

    if metadata_value(metadata, "num_runs") != "16":
        errors.append("metadata_num_runs_mismatch")

    # --------------------------------------------------
    # Log
    # --------------------------------------------------
    log_text = log_path.read_text()

    runtime_match = re.search(
        r"runtime = ([0-9.]+)",
        log_text,
    )
    fitness_match = re.search(
        r"learned fitness: ([0-9.eE+-]+)",
        log_text,
    )

    if runtime_match is None:
        errors.append("runtime_missing_from_log")
        runtime = np.nan
    else:
        runtime = float(runtime_match.group(1))

    if fitness_match is None:
        errors.append("fitness_missing_from_log")
        printed_fitness = np.nan
    else:
        printed_fitness = float(fitness_match.group(1))

    # --------------------------------------------------
    # Manifest
    # --------------------------------------------------
    manifest_row = manifest[
        manifest["condition"] == condition
    ]

    if len(manifest_row) != 1:
        errors.append(
            f"manifest_rows={len(manifest_row)}, expected=1"
        )
    else:
        manifest_row = manifest_row.iloc[0]

        if not np.isclose(
            float(manifest_row["runtime_seconds"]),
            runtime,
        ):
            errors.append("manifest_runtime_mismatch")

        if not np.isclose(
            float(manifest_row["learned_fitness"]),
            printed_fitness,
        ):
            errors.append("manifest_fitness_mismatch")

    # --------------------------------------------------
    # Seed protocol in evaluator
    # --------------------------------------------------
    evaluator_text = evaluator_path.read_text()

    if "PRNGKey(42)" not in evaluator_text:
        errors.append("PRNGKey42_not_found")

    if not re.search(
        r"jax\.random\.split\(rng,\s*num_runs\)",
        evaluator_text,
    ):
        errors.append("matched_seed_split_not_found")

    status = "PASS" if not errors else "FAIL"

    if errors:
        all_pass = False

    rows.append({
        "condition": condition,
        "status": status,
        "parameter_count": parameter_count,
        "returns_shape": str(returns.shape),
        "final_column_mean": final_mean,
        "printed_fitness": printed_fitness,
        "runtime_seconds": runtime,
        "checkpoint_sha256": checkpoint_hash,
        "returns_sha256": returns_hash,
        "errors": "; ".join(errors),
    })


# ------------------------------------------------------
# Global checks
# ------------------------------------------------------
global_errors = []

if len(manifest) != 5:
    global_errors.append(
        f"manifest_has_{len(manifest)}_rows_not_5"
    )

if manifest["condition"].nunique() != 5:
    global_errors.append("manifest_conditions_not_unique")

if (
    len(checkpoint_hashes) == 5
    and len(set(checkpoint_hashes.values())) != 5
):
    global_errors.append(
        "duplicate_checkpoint_content_detected"
    )

if (
    len(returns_hashes) == 5
    and len(set(returns_hashes.values())) != 5
):
    global_errors.append(
        "duplicate_returns_file_detected"
    )


# ------------------------------------------------------
# Formal Breakout config check
# ------------------------------------------------------
cfg = all_configs["breakout"]

expected_config = {
    "NUM_ENVS": 64,
    "NUM_STEPS": 128,
    "TOTAL_TIMESTEPS": 1e7,
    "UPDATE_EPOCHS": 4,
    "NUM_MINIBATCHES": 8,
    "HSIZE": 64,
}

for key, value in expected_config.items():
    if cfg[key] != value:
        global_errors.append(
            f"breakout_config_{key}={cfg[key]}_expected={value}"
        )

num_updates = int(
    cfg["TOTAL_TIMESTEPS"]
    // cfg["NUM_STEPS"]
    // cfg["NUM_ENVS"]
)

if num_updates != 1220:
    global_errors.append(
        f"breakout_NUM_UPDATES={num_updates}_expected=1220"
    )


if global_errors:
    all_pass = False


# ------------------------------------------------------
# Save report
# ------------------------------------------------------
report = pd.DataFrame(rows)

report.to_csv(
    OUT / "breakout_archive_audit.csv",
    index=False,
)

print("\n=== BREAKOUT ARCHIVE AUDIT ===\n")

cols = [
    "condition",
    "status",
    "parameter_count",
    "returns_shape",
    "final_column_mean",
    "printed_fitness",
    "runtime_seconds",
    "errors",
]

print(report[cols].to_string(index=False))

print("\n=== GLOBAL CHECKS ===\n")

if global_errors:
    for error in global_errors:
        print("FAIL:", error)
else:
    print("PASS: manifest has five unique conditions")
    print("PASS: five checkpoint files are distinct")
    print("PASS: five raw returns files are distinct")
    print("PASS: formal Breakout configuration verified")
    print("PASS: NUM_UPDATES = 1220")

print("\n=== FINAL AUDIT STATUS ===")
print("PASS" if all_pass else "FAIL")

print("\nSaved:")
print(OUT / "breakout_archive_audit.csv")