# OPEN Recurrent Black-Box Distillation Baseline

## Overview

This directory contains the curated summary results for the recurrent OPEN black-box distillation baseline evaluated on CartPole.

**Teacher**
- Pretrained recurrent OPEN
- Checkpoint: `multi_OPEN.npy`

**Student configurations**
- h32, sequence length 20
- h32, sequence length 100
- h16, sequence length 20
- h16, sequence length 100

## Experimental Records

### 01_distillation_training
Training logs for the four student configurations are retained in the local experiment archive.

### 02_checkpoint_evaluation_n8
CartPole evaluations were performed across distillation checkpoints using **8 evaluation runs per checkpoint**.

The h32-seq20 experiment required corrected checkpoint labels. Only the corrected mapping is used in the summary results. The original logs remain preserved in the raw local archive.

`eval_student_same32_seq20_cartpole_n8.log` corresponds to the final **900-update** h32-seq20 evaluation.

Missing values at 100 and 500 updates for h32-seq20 indicate that no traceable corrected evaluation log is available. They should **not** be interpreted as zero return.

### 03_reliability_n16
A separate reliability evaluation was performed using **16 runs** for the teacher and all four 900-update student configurations.

### 04_smoke_tests
Smoke tests were used to verify the distillation and evaluation pipelines before the formal experiments.

## Summary Files

### `results_summary.csv`
Checkpoint-wise CartPole performance for the four student configurations.

- Evaluation runs: n = 8
- Checkpoints: 100–1000 distillation updates where available

### `reliability_summary.csv`
Final teacher–student comparison using the reliability evaluation.

- Evaluation runs: n = 16
- Student checkpoint: 900 updates

## Key Observation

Checkpoint performance is strongly non-monotonic. More distillation updates and lower imitation loss do not necessarily translate into higher downstream RL return.

Therefore, imitation quality and downstream reinforcement-learning performance should be evaluated separately.

## Reproducibility Notes

The n=8 checkpoint sweep and n=16 reliability test are separate evaluation protocols and should not be combined as observations from the same experiment.

Complete raw training, evaluation, smoke-test, and diagnostic logs are preserved separately in the local experiment archive.