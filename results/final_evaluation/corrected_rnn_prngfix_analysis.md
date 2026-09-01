# Corrected recurrent OPEN distillation analysis

## Scope and protocol

This analysis uses only archived matched-seed n=16 results from the corrected 900-update (`curr_param_8.npy`) RNN checkpoints. No evaluation was rerun. Each seed score is the final column of its archived return curve. Summary SD is the sample SD (`ddof=1`). Confidence intervals are percentile 95% CIs from 20,000 bootstrap resamples using NumPy `default_rng(42)`, matching `scripts/analyse_breakout_final.py`. Paired comparisons use seed-aligned differences and SciPy's two-sided `wilcoxon` with its default exact/approximation selection, also matching the prior formal analysis.

## Corrected condition summaries

| Environment / condition | Mean | SD | Median | Min | Max | Bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| CartPole RNN seq20 | 80.5117 | 107.9328 | 41.1836 | 9.3164 | 430.0215 | [37.9737, 138.7520] |
| CartPole RNN seq100 | 51.7627 | 13.4278 | 51.1768 | 34.1543 | 77.7832 | [45.7203, 58.2976] |
| Breakout RNN seq20 | 0.095032 | 0.204321 | 0 | 0 | 0.511108 | [0, 0.190865] |
| Breakout RNN seq100 | 0.423187 | 0.038667 | 0.412903 | 0.364380 | 0.539429 | [0.406837, 0.443398] |

### Exact final scores by seed index 0–15

- **CartPole RNN seq20:** 68.837891, 50.035156, 21.732422, 134.324219, 31.750000, 430.021484, 20.705078, 31.958984, 14.423828, 32.332031, 86.777344, 15.369141, 9.316406, 62.107422, 221.644531, 56.851562.
- **CartPole RNN seq100:** 77.783203, 72.345703, 40.486328, 58.251953, 52.625000, 49.728516, 55.130859, 55.066406, 70.916016, 34.154297, 46.392578, 58.396484, 36.642578, 38.052734, 38.433594, 43.796875.
- **Breakout RNN seq20:** 0, 0, 0.511108, 0, 0, 0, 0.508179, 0, 0, 0, 0.501221, 0, 0, 0, 0, 0.
- **Breakout RNN seq100:** 0.400391, 0.405640, 0.410767, 0.364380, 0.435669, 0.441895, 0.399414, 0.415039, 0.539429, 0.459473, 0.439209, 0.404419, 0.395020, 0.400635, 0.426514, 0.433105.

## Paired comparisons

Differences are A minus B. Wins and losses use exact per-seed differences.

| ID / comparison (A vs B) | Mean A | Mean B | Mean diff. | Median diff. | Bootstrap 95% CI, mean diff. | A wins | Ties | B wins | Wilcoxon p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A. CartPole corrected seq20 vs pre-fix seq20 | 80.5117 | 113.2610 | -32.7493 | -31.2197 | [-96.7485, 42.7170] | 5 | 0 | 11 | 0.116669 |
| B. CartPole corrected seq100 vs pre-fix seq100 | 51.7627 | 63.2152 | -11.4525 | -6.0928 | [-21.5324, -2.9073] | 3 | 0 | 13 | 0.024963 |
| C. CartPole corrected seq20 vs MLP seq20 | 80.5117 | 131.0045 | -50.4928 | -85.9854 | [-96.8664, 11.4261] | 2 | 0 | 14 | 0.057678 |
| D. CartPole corrected seq100 vs MLP seq100 | 51.7627 | 449.1340 | -397.3713 | -446.1543 | [-436.6929, -352.8657] | 0 | 0 | 16 | 0.0000305 |
| E. CartPole corrected seq100 vs corrected seq20 | 51.7627 | 80.5117 | -28.7490 | 13.8496 | [-87.8320, 15.0022] | 10 | 0 | 6 | 0.979950 |
| F. Breakout corrected seq20 vs pre-fix seq20 | 0.095032 | 0.094276 | 0.000755 | 0 | [-0.093567, 0.095833] | 1 | 14 | 1 | 0.654721 |
| G. Breakout corrected seq100 vs pre-fix seq100 | 0.423187 | 0.369072 | 0.054115 | 0.000854 | [-0.005272, 0.127464] | 9 | 1 | 6 | 0.280531 |
| H. Breakout corrected seq20 vs MLP seq20 | 0.095032 | 6.995674 | -6.900642 | -6.071838 | [-8.897509, -5.020744] | 0 | 0 | 16 | 0.0000305 |
| I. Breakout corrected seq100 vs MLP seq100 | 0.423187 | 7.718597 | -7.295410 | -1.590576 | [-12.583550, -2.752631] | 0 | 0 | 16 | 0.0000305 |
| J. Breakout corrected seq100 vs corrected seq20 | 0.423187 | 0.095032 | 0.328156 | 0.405029 | [0.222303, 0.418733] | 13 | 0 | 3 | 0.000427 |

## Distillation losses

The exact @900 loss is recoverable: it is the 90th logged inner epoch, immediately before the outer loop saves `curr_param_8.npy`. The 100th record is the @1000 endpoint associated with `curr_param_9.npy` and is not the dissertation checkpoint.

| Condition | Pre-fix @900 train | Pre-fix @900 test | Corrected @900 train | Corrected @900 test | Corrected @1000 train | Corrected @1000 test |
|---|---:|---:|---:|---:|---:|---:|
| RNN seq20 | 4.306406e-07 | 4.252174e-07 | 4.288152e-07 | 4.240539e-07 | 3.980629e-07 | 3.850805e-07 |
| RNN seq100 | 3.049373e-06 | 3.035080e-06 | 3.097842e-06 | 3.020755e-06 | 2.832955e-06 | 2.835529e-06 |

At @900, corrected test loss changed by only about -0.27% for seq20 and -0.47% for seq100. The PRNG correction therefore increased training-data diversity without materially changing held-out synthetic fidelity at the selected checkpoint.

## Interpretation

1. **Did the fix materially improve downstream RNN performance?** No consistent improvement occurred. CartPole means decreased (seq100 significantly by paired Wilcoxon); Breakout seq20 was effectively unchanged and seq100 improved modestly but non-significantly relative to pre-fix. The defect mattered methodologically, but it does not explain the poor downstream RNN results.
2. **Does the RNN-versus-MLP result survive?** Yes. MLP remains much stronger for Breakout at both sequence lengths (16/16 wins, p=3.05e-05) and for CartPole seq100 (16/16 wins, p=3.05e-05). CartPole seq20 also favours MLP in mean and 14/16 pairs, narrowly missing 0.05 (p=0.0577).
3. **Does the sequence-length conclusion survive?** It remains environment-dependent. Breakout seq100 decisively outperforms seq20 after correction (p=0.000427). CartPole has no reliable mean advantage: seq100 wins 10/16 pairs but seq20 has a much larger, high-variance mean driven partly by one 430-point seed; Wilcoxon p=0.980.
4. **Is carry-horizon mismatch now a stronger candidate?** Yes. Once PRNG diversity is corrected without recovering downstream performance, the train/deployment hidden-state-horizon mismatch becomes a stronger remaining mechanistic explanation, though this analysis does not prove causality.
5. **Is formal Ant n=16 defensible now?** Yes, provided it is framed as evaluation of the corrected, currently specified recurrent pipeline. The PRNG confound is removed and checkpoints/protocol are validated; the unresolved carry-horizon limitation should be disclosed rather than silently treated as solved.
6. **Which dissertation values should be replaced?** Replace all pre-fix h32 RNN seq20/seq100 CartPole and Breakout values with the corrected @900 summaries and paired comparisons above. Retain the pre-fix values only as explicitly labelled audit/sensitivity results. Do not substitute the corrected @1000 losses or `curr_param_9.npy` checkpoints.

## Sources

- Corrected CartPole arrays and provenance:
  - `results/final_evaluation/cartpole/corrected_rnn_h32_seq20_prngfix/`
  - `results/final_evaluation/cartpole/corrected_rnn_h32_seq100_prngfix/`
- Corrected Breakout arrays and provenance:
  - `results/final_evaluation/breakout/corrected_rnn_h32_seq20_prngfix/`
  - `results/final_evaluation/breakout/corrected_rnn_h32_seq100_prngfix/`
- Pre-fix CartPole RNN arrays:
  - `results/final_evaluation/raw/cartpole/rnn_h32_seq20/returns_900_n16.npy`
  - `results/final_evaluation/raw/cartpole/rnn_h32_seq100/returns_900_n16.npy`
- CartPole MLP arrays:
  - `results/final_evaluation/raw/cartpole/mlp_h32_seq20/returns_900_n16.npy`
  - `results/final_evaluation/raw/cartpole/mlp_h32_seq100/returns_900_n16.npy`
- Pre-fix and MLP Breakout arrays:
  - `results/final_evaluation/breakout/rnn_h32_seq20/returns_breakout.npy`
  - `results/final_evaluation/breakout/rnn_h32_seq100/returns_breakout.npy`
  - `results/final_evaluation/breakout/mlp_h32_seq20/returns_breakout.npy`
  - `results/final_evaluation/breakout/mlp_h32_seq100/returns_breakout.npy`
- Corrected training logs:
  - `results/log_archive/distillation_training/rnn_h32_seq20_prngfix.log`
  - `results/log_archive/distillation_training/rnn_h32_seq100_prngfix.log`
- Pre-fix training logs:
  - `results/log_archive/distillation_training/distil_same32_seq20_stage1.log`
  - `results/log_archive/distillation_training/distil_same32_seq100_stage1.log`
- Statistical reference implementation: `scripts/analyse_breakout_final.py`.
- Source commit: `529aa2257839dd8ba359600a0114d8232f2106e9`.
