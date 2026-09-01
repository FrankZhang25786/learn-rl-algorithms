from pathlib import Path
import pandas as pd

OUT = Path("results/final_evaluation/breakout/analysis")

rows = [
    ["rnn_h32_seq20", 20, 4.306406e-07, 4.252174e-07],
    ["rnn_h32_seq100", 100, 3.049373e-06, 3.035080e-06],
    ["mlp_h32_seq20", 20, 8.755405e-07, 8.721397e-07],
    ["mlp_h32_seq100", 100, 4.064702e-06, 3.901092e-06],
]

df = pd.DataFrame(
    rows,
    columns=[
        "condition",
        "seq_length",
        "train_loss_at_900",
        "test_loss_at_900",
    ],
)

df.to_csv(OUT / "distillation_fidelity_at_900.csv", index=False)

print(df.to_string(index=False))
print("\nSaved:", OUT / "distillation_fidelity_at_900.csv")