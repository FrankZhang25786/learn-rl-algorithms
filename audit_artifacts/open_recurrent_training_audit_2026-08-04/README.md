# Recurrent OPEN Breakout Training-Signal Audit

Date preserved: 2026-08-04

## Scope and conclusion

This directory preserves the completed recurrent OPEN Breakout run, the training-path audit, corrected evaluation code, raw diagnostics, and environment/repository state.

The completed 800-generation run is an **infrastructure/pipeline run**, not valid evidence of successful meta-learning. It did not optimize a checkpoint-dependent post-update objective:

1. Breakout used one inner update.
2. Candidate fitness came from the trajectory collected before that update.
3. `training_prop` and `batch_prop` both evaluated to `0 / 0` and became NaN.
4. The sole learned-optimizer update made all actor parameters, critic parameters, and recurrent carries NaN.
5. All 64 candidates had identical raw fitness in every one of the 800 generations.
6. Centered-rank shaping assigned distinct ranks to tied values, allowing OpenES to drift without a candidate-dependent performance signal.

No training issue described here has been fixed in training code.

## Completed run configuration

Run directory: `save_files/open_recurrent_bb/2026-08-03_16-06-12`

Run log: `baseline_800gen.log` (preserved at `logs/root/baseline_800gen.log`)

Training entry point: `algorithms/learning_algorithms/black_box_learning/open_recurrent_bb.py`

The log begins at 2026-08-03 16:06:09 and covers generations 0 through 799. The entry-point defaults and artifacts establish:

| Setting | Value | Source |
|---|---:|---|
| generations | 800 | `open_recurrent_bb.py:642` |
| environment | `breakout` | `open_recurrent_bb.py:643` |
| population | 64 | `open_recurrent_bb.py:645`; 64 logged entries/generation |
| rollouts/candidate | 1 | `open_recurrent_bb.py:646` |
| ES learning rate | 0.03 | `open_recurrent_bb.py:644` |
| ES noise level | 0.03 | `open_recurrent_bb.py:648` |
| checkpoint interval | 8 | `open_recurrent_bb.py:647` |
| optimizer hidden size | 16 | `open_recurrent_bb.py:652`; 1,211 optimizer parameters |
| initial ES RNG | 42 | `open_recurrent_bb.py:722` |

The exact shell launch command was not preserved locally. The completed log and artifacts match these defaults.

## Exact Breakout inner-loop configuration

Defined in `algorithms/utils/configs.py:55-79`:

| Setting | Value |
|---|---:|
| `NUM_ENVS` | 2 |
| `NUM_STEPS` | 16 |
| `TOTAL_TIMESTEPS` | 32 |
| `UPDATE_EPOCHS` | 1 |
| `NUM_MINIBATCHES` | 1 |
| environment | `Breakout-MinAtar` |
| actor hidden size | 8 |
| activation | ReLU |
| gamma | 0.99 |
| GAE lambda | 0.95 |
| PPO clip epsilon | 0.2 |
| entropy coefficient | 0.01 |
| value coefficient | 0.5 |
| max gradient norm | 0.5 |

`open_recurrent_bb.py:161-170` computes:

```text
NUM_UPDATES = TOTAL_TIMESTEPS // NUM_STEPS // NUM_ENVS
            = 32 // 16 // 2
            = 1
MINIBATCH_SIZE = 2 * 16 // 1 = 32
TOTAL_UPDATES = 1 * 1 * 1 = 1
```

## Actual training execution order

For every optimizer candidate:

1. Actor and critic are freshly initialized (`open_recurrent_bb.py:202-217`).
2. The environment is reset (`open_recurrent_bb.py:249-252`).
3. A 16-step trajectory is collected with those initial parameters (`open_recurrent_bb.py:255-295`).
4. GAE and PPO gradients are computed (`open_recurrent_bb.py:297-407`).
5. The learned optimizer is called once (`open_recurrent_bb.py:460-476`).
6. The returned metric is read from the already-collected `traj_batch.info` (`open_recurrent_bb.py:553-565`).
7. There is no second rollout because `NUM_UPDATES == 1` (`open_recurrent_bb.py:631-635`).

Although metric extraction is textually after `opt.update`, its data are from the pre-update trajectory. `make_rollout` uses that sole metric as candidate fitness (`open_recurrent_bb.py:697-705`). The learned update cannot influence measured behavior.

## NaN schedule proof and propagation

Training uses the unguarded expressions at `open_recurrent_bb.py:388-395`:

```python
training_prop = (...) / (config["NUM_UPDATES"] - 1)
batch_prop = (...) / (config["UPDATE_EPOCHS"] - 1)
```

At the only update, actor iteration is zero:

```text
training_prop = 0 / (1 - 1) = 0 / 0 = NaN
batch_prop    = 0 / (1 - 1) = 0 / 0 = NaN
```

The diagnostic through the unmodified training `make_train` printed:

```text
computed_NUM_UPDATES 1
training_prop nan
batch_prop nan
```

These values are passed directly to `opt.update` (`open_recurrent_bb.py:466-475`). In `algorithms/utils/architectures/open_recurrent.py` they become optimizer features (`:149-150`), are tiled and concatenated (`:206-244`), then enter `gru.apply` and the learned MLP (`:246-264`). There is no finiteness guard.

Matched-RNG training-path diagnostic:

```text
candidate       actor NaNs  critic NaNs  actor-carry NaNs
generation 792  3307/3307   3289/3289    26456/26456
zero            3307/3307   3289/3289    26456/26456
fixed random    3307/3307   3289/3289    26456/26456
```

The rollout metric remained finite because it had already been collected:

```text
fitness: generation_792=0.0, zero=0.0, fixed_random=0.0
all pairwise fitness equal: True
metrics NaNs: 0
```

The only `nan_to_num` is applied later to scalar fitness at `open_recurrent_bb.py:794`; it does not repair optimizer inputs or actor/critic state.

## Population RNG and zero fitness spread

At `open_recurrent_bb.py:768-770`, the same rollout RNG is tiled across all population members. All candidates therefore have identical initial actor/critic parameters, environment randomness, and pre-update actions. Candidate-specific learned updates happen only after measured data have been collected.

Parsing the completed log produced:

```text
logged generations: 800 (0 through 799)
nonzero raw-fitness spreads: 0
unique raw-fitness spreads: [0.0]
```

Every logged population vector is a single scalar repeated 64 times. Variation between generations reflects common rollout randomness, not candidate quality.

## Centered-rank tie handling

Training creates a centered-rank `FitnessShaper` at `open_recurrent_bb.py:738-740`, applies it at `:846`, and passes the result to OpenES at `:848`.

Installed evosax source:

- `/home/ubuntu/projects/meta-rl-venv/lib/python3.11/site-packages/evosax/utils/reshape_fitness.py:59-70`
- `/home/ubuntu/projects/meta-rl-venv/lib/python3.11/site-packages/evosax/strategies/open_es.py:119-136`

The rank implementation assigns sequential ranks after `argsort` and does not preserve ties. For 64 identical raw fitness values, it produced 64 distinct shaped values spanning `-0.5` to `0.5`. OpenES then computes a noise-weighted gradient from those artificial ranks. Consequently, saved checkpoints can drift even though raw fitness contains no optimizer-dependent information.

## Corrected evaluation findings

Only evaluation code was changed. The evaluator now supports a configurable adaptation horizon and a distinct post-adaptation evaluation rollout. Training behavior remains unchanged.

Relevant preserved code:

- `evaluation_code/eval_open_recurrent_bb.py`
- `evaluation_code/compare_open_recurrent_adaptation.py`
- evaluator pre-change backups in `evaluation_code/`
- `evaluator_diff.patch`

### Initial generation-792/zero/random comparison

With matched seed 0 and ten runs:

- one and five adaptation updates produced identical return arrays;
- at ten and twenty updates, fixed random diverged behaviorally;
- generation 792 and zero still had identical sampled return arrays;
- generation 792 nevertheless produced nonzero actor-parameter displacement.

### Larger matched comparison

Experiment: adaptation updates 20, 50, and 100; seeds 0-4; 50 runs/seed; 250 matched runs per horizon.

Raw and aggregate results are preserved under `evaluation_results/comparison_20_50_100/`.

Final return mean ± standard deviation:

| Updates | Generation 792 | Zero | Fixed random |
|---:|---:|---:|---:|
| 20 | 0.392 ± 0.456 | 0.380 ± 0.447 | 0.404 ± 0.453 |
| 50 | 0.386 ± 0.484 | 0.382 ± 0.468 | 0.384 ± 0.478 |
| 100 | 0.370 ± 0.456 | 0.356 ± 0.449 | 0.374 ± 0.453 |

Paired generation-792-minus-zero differences:

| Updates | Mean difference | 95% interval | wins/ties/losses |
|---:|---:|---:|---:|
| 20 | +0.012 | [-0.0026, +0.0266] | 6/242/2 |
| 50 | +0.004 | [-0.0084, +0.0164] | 6/240/4 |
| 100 | +0.014 | [-0.0140, +0.0420] | 13/229/8 |

Every interval includes zero. Generation 792 changes actor parameters, but policy distributions remain extremely close to zero: matched stochastic action agreement is 99.9-100%, mean symmetric KL is approximately `2.0e-6` to `1.53e-5`, and mean absolute probability difference is approximately `5.26e-4` to `1.42e-3`. All stored numeric arrays are finite and contain zero NaNs.

There is no reliable behavioral or return advantage over the zero optimizer in these corrected evaluations.

## Interpretation

The completed run demonstrates that the software stack could initialize models, execute environments, invoke OpenES, log metrics, save checkpoints, and run periodic evaluation. It does not demonstrate successful meta-learning because its outer objective was independent of candidate learned updates and its only inner update was non-finite.

Treat the completed 800-generation run as an infrastructure/pipeline artifact only.

## Recommended next step

1. Create a separate training-fix branch.
2. Repair the inner objective so fitness is measured after multiple finite learned-optimizer updates.
3. Guard single-point schedule denominators semantically.
4. handle tied fitness without artificial ordering.
5. Add finiteness assertions for schedule features, optimizer inputs, updates, and actor/critic state.
6. Validate candidate-dependent post-update fitness with a tiny diagnostic run.
7. Retrain only after that diagnostic passes.

Do not interpret any issue as fixed in training code yet.

## Artifact layout

- `system_and_repo_state.txt`: repository, environment, device, and result inventory.
- `pip_freeze.txt`: requested PATH-based command result plus the completed-run virtual environment snapshot.
- `evaluator_diff.patch`: current evaluator diff against Git HEAD.
- `logs/root/`: completed-run and diagnostic logs.
- `evaluation_code/`: evaluator, comparison harness, backups, and evaluation scripts.
- `evaluation_results/checkpoint_evaluation_2026-08-03/`: six selected checkpoint bundles.
- `evaluation_results/multiseed_evaluation_2026-08-04/`: generation 136/792 multi-seed bundles.
- `evaluation_results/save_files_eval/`: preserved evaluator outputs.
- `evaluation_results/comparison_20_50_100/`: raw NPZ files and JSON summaries.
- `evaluation_results/experiment_backups/`: existing completed-run tar backup.
- `summaries/`: compact CSV/JSON summaries.
- `checksums/SHA256SUMS`: per-file checksums for this audit directory.
