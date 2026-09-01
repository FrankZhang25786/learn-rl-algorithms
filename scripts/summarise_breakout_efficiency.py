from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("results/final_evaluation/breakout")

CHECKPOINTS = {
    "teacher": Path("save_files/pretrained/multi_OPEN.npy"),
    "rnn_h32_seq20": Path(
        "save_files/open_recurrent_distil/"
        "2026-08-10_17:04:41.684883/curr_param_8.npy"
    ),
    "rnn_h32_seq100": Path(
        "save_files/open_recurrent_distil/"
        "2026-08-10_19:51:22.871913/curr_param_8.npy"
    ),
    "mlp_h32_seq20": Path(
        "save_files/open_recurrent_to_ff_distil/"
        "2026-08-24_22-53-51.517510/curr_param_8.npy"
    ),
    "mlp_h32_seq100": Path(
        "save_files/open_recurrent_to_ff_distil/"
        "2026-08-25_00-11-42.707317/curr_param_8.npy"
    ),
}

manifest = pd.read_csv(ROOT / "manifest.csv")

rows = []

for condition, path in CHECKPOINTS.items():
    params = np.load(path, allow_pickle=True)

    parameter_count = int(np.asarray(params).size)

    runtime = float(
        manifest.loc[
            manifest["condition"] == condition,
            "runtime_seconds"
        ].iloc[0]
    )

    rows.append({
        "condition": condition,
        "parameter_count": parameter_count,
        "evaluation_runtime_seconds": runtime,
        "runtime_minutes": runtime / 60.0,
    })

df = pd.DataFrame(rows)

out_dir = ROOT / "analysis"
out_dir.mkdir(exist_ok=True)

df.to_csv(
    out_dir / "breakout_efficiency.csv",
    index=False
)

print("\n=== BREAKOUT EFFICIENCY ===\n")
print(df.to_string(index=False))

print("\nSaved:")
print(out_dir / "breakout_efficiency.csv")