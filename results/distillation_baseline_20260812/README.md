\# OPEN Recurrent Black-Box Distillation Baseline



\## Overview



This directory contains the curated logs and summary results for the recurrent OPEN

black-box distillation baseline evaluated on CartPole.



Teacher:

\- Pretrained recurrent OPEN

\- Checkpoint: `multi\_OPEN.npy`



Student configurations:

\- h32, sequence length 20

\- h32, sequence length 100

\- h16, sequence length 20

\- h16, sequence length 100



\## Directory Structure



\### 01\_distillation\_training

Training logs for the four student configurations.



\### 02\_checkpoint\_evaluation\_n8

CartPole evaluations across distillation checkpoints.

Each checkpoint evaluation uses 8 runs.



The h32-seq20 experiment required corrected checkpoint labels.

Only the corrected evaluation logs are included in the curated directory.

The original logs remain available in the raw archive.



`eval\_student\_same32\_seq20\_cartpole\_n8.log` corresponds to the final

900-update h32-seq20 evaluation.



Missing values at 100 and 500 updates for h32-seq20 indicate that no

traceable corrected evaluation log is available. They should not be

interpreted as zero return.



\### 03\_reliability\_n16

Final reliability evaluation using 16 runs for the teacher and the four

900-update student configurations.



\### 04\_smoke\_tests

Smoke-test logs used to verify the distillation and evaluation pipelines.



\### 05\_previous\_blackbox\_diagnostics

Reserved for earlier black-box learning / diagnostic logs if required.



\## Summary Files



`results\_summary.csv`

\- Checkpoint-wise CartPole return

\- Evaluation protocol: n = 8



`reliability\_summary.csv`

\- Teacher/student comparison at the selected final checkpoint

\- Evaluation protocol: n = 16



\## Important Notes



Checkpoint performance is non-monotonic: lower distillation loss or more

distillation updates do not necessarily lead to higher downstream RL return.



The n=8 checkpoint sweep and n=16 reliability evaluation are separate

evaluation protocols and should not be combined as if they were from the

same experiment.

