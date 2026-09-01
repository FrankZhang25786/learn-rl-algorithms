# Formal Ant n=16 analysis

## Protocol and integrity

This report analyses the final column of each archived `returns_ant.npy`, preserving the common seed order generated from RNG base key 42. All five arrays have shape `(16, 1464)` and contain only finite values. Statistics follow the previously fixed protocol: sample SD (`ddof=1`); 20,000 percentile bootstrap resamples with NumPy `default_rng(42)`; and two-sided paired SciPy Wilcoxon tests without result-dependent method changes. The Ant configuration is shared across conditions: Brax Ant, 2,048 environments, 10 steps, 30,000,000 timesteps, 4 update epochs, 32 minibatches, normalized observations/rewards, clipped actions, and 1,464 PPO updates.

## Provenance

| Condition | Checkpoint | SHA256 | Parameters | Evaluator runtime (s) | Shape |
|---|---|---|---:|---:|---:|
| Teacher | `save_files/pretrained/multi_OPEN.npy` | `9ba2dea70b0b308a365516dab092808b7b9211c70072bca934d83affaee5f60c` | 3,571 | 4214.730 | `(16, 1464)` |
| Corrected RNN seq20 @900 | `save_files/open_recurrent_distil/2026-09-01_00:48:04.331554/curr_param_8.npy` | `4b6651c4a1ec5b133915e6a7230ff0507929c3b98ffe02ee6031c39de357416c` | 3,571 | 4212.800 | `(16, 1464)` |
| Corrected RNN seq100 @900 | `save_files/open_recurrent_distil/2026-09-01_01:27:41.643380/curr_param_8.npy` | `d76343a26fb3f0643819b581dfdb30b5ef9b66e201830219a50a8bbd108f9e2b` | 3,571 | 4210.670 | `(16, 1464)` |
| MLP seq20 @900 | `save_files/open_recurrent_to_ff_distil/2026-08-24_22-53-51.517510/curr_param_8.npy` | `ce3b522399914d6c526e4f68dff35d46ab6f4ba3cc5cac94e67674b5791f8103` | 1,923 | 4013.758 | `(16, 1464)` |
| MLP seq100 @900 | `save_files/open_recurrent_to_ff_distil/2026-08-25_00-11-42.707317/curr_param_8.npy` | `27f1afe74fd55b3377eec57d6e73360786773da1e639840f4aef2c08b394ec1d` | 1,923 | 4014.382 | `(16, 1464)` |

The formal queue records source commit `529aa2257839dd8ba359600a0114d8232f2106e9`. This commit is not embedded directly in each condition's `metadata.json`; it is preserved in `results/log_archive/ant_formal_n16_20260901/after_teacher.log` and `queue.log`.

## Condition summaries

| Condition | Mean | SD | Median | Min | Max | Bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| Teacher | 2579.193 | 553.018 | 2500.995 | 1454.581 | 3684.251 | [2316.189, 2844.126] |
| Corrected RNN seq20 | -383.936 | 223.321 | -282.917 | -942.307 | -253.780 | [-498.942, -283.937] |
| Corrected RNN seq100 | 525.012 | 50.059 | 526.227 | 432.012 | 617.010 | [501.251, 548.959] |
| MLP seq20 | 644.589 | 28.098 | 650.209 | 597.785 | 689.678 | [631.120, 657.895] |
| MLP seq100 | 3888.027 | 467.472 | 3989.703 | 2666.703 | 4444.855 | [3645.334, 4084.537] |

### Exact final scores, seed indices 0–15

- **Teacher:** 3269.770020, 2160.078125, 1454.581055, 2279.746094, 3083.223145, 2320.108154, 2552.830566, 2445.831787, 2362.484131, 3307.486328, 2612.844482, 2776.236328, 3684.250732, 2525.999756, 1955.619019, 2475.990967.
- **Corrected RNN seq20:** -253.779785, -282.854919, -269.970825, -295.447693, -789.347839, -942.307068, -289.776733, -310.515015, -274.684296, -274.830872, -302.973724, -282.979065, -281.439545, -260.721497, -282.515198, -748.839355.
- **Corrected RNN seq100:** 522.033508, 566.020081, 551.943481, 553.774109, 506.116455, 517.796997, 466.472412, 552.992249, 617.010437, 495.237854, 530.419739, 573.776306, 451.182831, 432.012421, 488.224609, 575.174255.
- **MLP seq20:** 664.980286, 670.428711, 619.603699, 689.677734, 629.556641, 648.207275, 629.273621, 684.987366, 597.784851, 652.211243, 658.930298, 612.673889, 602.323425, 630.656799, 661.105774, 661.022461.
- **MLP seq100:** 2890.703369, 4228.708984, 4101.621094, 3880.619629, 3924.158691, 3995.542236, 4249.222168, 3946.635254, 4106.026855, 3716.282471, 3900.211426, 3983.864014, 4023.949219, 4444.854980, 4149.321289, 2666.702637.

## Paired comparisons

Differences are A minus B; confidence intervals bootstrap the 16 paired differences.

| ID / comparison | Mean A | Mean B | Mean difference | Median difference | Bootstrap 95% CI | A wins | Ties | B wins | Wilcoxon p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A. Corrected RNN seq20 vs MLP seq20 | -383.936 | 644.589 | -1028.525 | -935.332 | [-1145.412, -931.868] | 0 | 0 | 16 | 0.0000305 |
| B. Corrected RNN seq100 vs MLP seq100 | 525.012 | 3888.027 | -3363.015 | -3447.894 | [-3567.123, -3112.846] | 0 | 0 | 16 | 0.0000305 |
| C. Corrected RNN seq100 vs corrected RNN seq20 | 525.012 | -383.936 | 908.948 | 841.134 | [809.509, 1027.208] | 16 | 0 | 0 | 0.0000305 |
| D. MLP seq100 vs MLP seq20 | 3888.027 | 644.589 | 3243.438 | 3359.263 | [2996.578, 3444.407] | 16 | 0 | 0 | 0.0000305 |
| E. Corrected RNN seq20 vs teacher | -383.936 | 2579.193 | -2963.129 | -2879.213 | [-3244.628, -2675.338] | 0 | 0 | 16 | 0.0000305 |
| F. Corrected RNN seq100 vs teacher | 525.012 | 2579.193 | -2054.181 | -1991.621 | [-2329.443, -1781.292] | 0 | 0 | 16 | 0.0000305 |
| G. MLP seq20 vs teacher | 644.589 | 2579.193 | -1934.604 | -1855.156 | [-2203.393, -1669.668] | 0 | 0 | 16 | 0.0000305 |
| H. MLP seq100 vs teacher | 3888.027 | 2579.193 | 1308.834 | 1550.839 | [907.186, 1689.829] | 15 | 0 | 1 | 0.000153 |

## Interpretation

1. **Architecture:** MLP decisively outperforms corrected RNN at both sequence lengths on Ant. Every matched seed favours MLP, with large paired effects and p=3.05e-05 in both comparisons.
2. **Cross-environment consistency:** Ant reinforces the Breakout architecture result and the strong CartPole seq100 result. CartPole seq20 was less decisive, so the defensible claim is that the MLP advantage is robust for seq100 and for the harder/longer-horizon Breakout and Ant settings, not necessarily uniformly significant in every condition.
3. **Sequence length:** Seq100 greatly improves both Ant students: +908.95 for RNN and +3243.44 for MLP, each winning all 16 pairs. This is much stronger than CartPole's ambiguous RNN sequence effect and agrees with Breakout's corrected seq100 RNN advantage.
4. **Teacher proximity:** Corrected RNN seq20 fails badly and is negative on every seed. Corrected RNN seq100 reaches only about 20% of the teacher mean. MLP seq20 reaches about 25%; MLP seq100 exceeds the teacher mean by about 51% and wins 15/16 matched seeds.
5. **Carry-horizon hypothesis:** Ant strengthens, but does not prove, the hypothesis. The recurrent student remains far below its recurrent teacher and stateless MLP counterpart over the longest optimizer horizon, while longer distillation sequences substantially help. This pattern is mechanistically compatible with hidden-state horizon mismatch; it is not a causal isolation experiment.
6. **Defensible cross-environment conclusion:** After correcting PRNG propagation, RNN weakness persists. MLP is decisively better on Ant and Breakout and on CartPole seq100. Longer sequences help RNN on Breakout and Ant, but not reliably on CartPole; they help MLP strongly on CartPole and Ant. Effects therefore depend on both architecture and environment/horizon.
7. **Primary versus limitation:** Primary dissertation results should be the matched-seed condition summaries and prespecified paired tests from corrected @900 checkpoints. The PRNG issue, synthetic-to-deployment mismatch, recurrent carry horizon, parameter-count difference, and absence of a targeted carry-ablation experiment should be presented as limitations or mechanistic hypotheses—not as established causal explanations.

## Source paths

- `results/final_evaluation/ant/formal_n16_teacher/`
- `results/final_evaluation/ant/formal_n16_corrected_rnn_h32_seq20_prngfix/`
- `results/final_evaluation/ant/formal_n16_corrected_rnn_h32_seq100_prngfix/`
- `results/final_evaluation/ant/formal_n16_mlp_h32_seq20/`
- `results/final_evaluation/ant/formal_n16_mlp_h32_seq100/`
- `results/log_archive/ant_formal_n16_20260901/teacher_retry.log`
- `results/log_archive/ant_formal_n16_20260901/after_teacher.log`
- `results/log_archive/ant_formal_n16_20260901/formal_n16_corrected_rnn_h32_seq20_prngfix.log`
- `results/log_archive/ant_formal_n16_20260901/formal_n16_corrected_rnn_h32_seq100_prngfix.log`
- `results/log_archive/ant_formal_n16_20260901/formal_n16_mlp_h32_seq20.log`
- `results/log_archive/ant_formal_n16_20260901/formal_n16_mlp_h32_seq100.log`
