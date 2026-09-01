from pathlib import Path
import csv
import re

ROOT = Path("results/final_evaluation/breakout")

conditions = {
    "teacher": {
        "architecture": "recurrent_OPEN_teacher",
        "seq_length": "NA",
        "checkpoint_update": "pretrained",
        "checkpoint": "save_files/pretrained/multi_OPEN.npy",
        "evaluator": "algorithms/evaluation/eval_open_recurrent_bb.py",
    },
    "rnn_h32_seq20": {
        "architecture": "RNN",
        "seq_length": "20",
        "checkpoint_update": "900",
        "checkpoint": "save_files/open_recurrent_distil/2026-08-10_17:04:41.684883/curr_param_8.npy",
        "evaluator": "algorithms/evaluation/eval_open_recurrent_bb.py",
    },
    "rnn_h32_seq100": {
        "architecture": "RNN",
        "seq_length": "100",
        "checkpoint_update": "900",
        "checkpoint": "save_files/open_recurrent_distil/2026-08-10_19:51:22.871913/curr_param_8.npy",
        "evaluator": "algorithms/evaluation/eval_open_recurrent_bb.py",
    },
    "mlp_h32_seq20": {
        "architecture": "MLP",
        "seq_length": "20",
        "checkpoint_update": "900",
        "checkpoint": "save_files/open_recurrent_to_ff_distil/2026-08-24_22-53-51.517510/curr_param_8.npy",
        "evaluator": "algorithms/evaluation/eval_open_ff_bb.py",
    },
    "mlp_h32_seq100": {
        "architecture": "MLP",
        "seq_length": "100",
        "checkpoint_update": "900",
        "checkpoint": "save_files/open_recurrent_to_ff_distil/2026-08-25_00-11-42.707317/curr_param_8.npy",
        "evaluator": "algorithms/evaluation/eval_open_ff_bb.py",
    },
}


def parse_log(log_path):
    text = log_path.read_text()

    runtime_match = re.search(r"runtime = ([0-9.]+)", text)
    fitness_match = re.search(r"learned fitness: ([0-9.eE+-]+)", text)

    runtime = float(runtime_match.group(1)) if runtime_match else None
    fitness = float(fitness_match.group(1)) if fitness_match else None

    return runtime, fitness


manifest_rows = []

for name, meta in conditions.items():
    condition_dir = ROOT / name
    log_path = condition_dir / "evaluation.log"
    returns_path = condition_dir / "returns_breakout.npy"

    if not log_path.exists():
        raise FileNotFoundError(f"Missing log: {log_path}")

    if not returns_path.exists():
        raise FileNotFoundError(f"Missing returns file: {returns_path}")

    runtime, fitness = parse_log(log_path)

    metadata_text = f"""environment: breakout
condition: {name}
architecture: {meta["architecture"]}
student_hsize: 32
distillation_seq_length: {meta["seq_length"]}
checkpoint_update: {meta["checkpoint_update"]}
checkpoint: {meta["checkpoint"]}

num_runs: 16
seed_protocol: PRNGKey(42), matched across conditions

evaluation_script: {meta["evaluator"]}

breakout_config:
NUM_ENVS=64
NUM_STEPS=128
TOTAL_TIMESTEPS=10000000
UPDATE_EPOCHS=4
NUM_MINIBATCHES=8
NUM_UPDATES=1220
actor_critic_hsize=64

runtime_seconds: {runtime}
printed_learned_fitness: {fitness}
"""

    (condition_dir / "metadata.txt").write_text(metadata_text)

    manifest_rows.append(
        {
            "condition": name,
            "architecture": meta["architecture"],
            "hsize": 32,
            "seq_length": meta["seq_length"],
            "checkpoint_update": meta["checkpoint_update"],
            "num_runs": 16,
            "runtime_seconds": runtime,
            "learned_fitness": fitness,
        }
    )

manifest_path = ROOT / "manifest.csv"

with manifest_path.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "condition",
            "architecture",
            "hsize",
            "seq_length",
            "checkpoint_update",
            "num_runs",
            "runtime_seconds",
            "learned_fitness",
        ],
    )
    writer.writeheader()
    writer.writerows(manifest_rows)

print("Breakout metadata archive complete.")
print(f"Manifest: {manifest_path}")