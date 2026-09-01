#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=disabled
export MPLBACKEND=Agg

PY="${PYTHON:-python}"
EVAL="$SCRIPT_DIR/eval_ant_formal.py"

TEACHER="save_files/pretrained/multi_OPEN.npy"

RNN20="save_files/open_recurrent_distil/2026-09-01_00:48:04.331554/curr_param_8.npy"
RNN100="save_files/open_recurrent_distil/2026-09-01_01:27:41.643380/curr_param_8.npy"

MLP20="save_files/open_recurrent_to_ff_distil/2026-08-24_22-53-51.517510/curr_param_8.npy"
MLP100="save_files/open_recurrent_to_ff_distil/2026-08-25_00-11-42.707317/curr_param_8.npy"

LOGDIR="results/log_archive/ant_formal_n16_20260901"
mkdir -p "$LOGDIR"

# Let the queue script manage its own master log.
exec > >(tee -a "$LOGDIR/queue.log") 2>&1

echo "[QUEUE] Script started: $(date)"
echo "[QUEUE] Working directory: $(pwd)"
echo "[QUEUE] Git commit: $(git rev-parse HEAD)"

conditions=(
  "formal_n16_teacher"
  "formal_n16_corrected_rnn_h32_seq20_prngfix"
  "formal_n16_corrected_rnn_h32_seq100_prngfix"
  "formal_n16_mlp_h32_seq20"
  "formal_n16_mlp_h32_seq100"
)

check_file () {
    if [[ ! -f "$1" ]]; then
        echo "[FATAL] Missing file: $1"
        exit 1
    fi
}

check_file "$EVAL"
check_file "$TEACHER"
check_file "$RNN20"
check_file "$RNN100"
check_file "$MLP20"
check_file "$MLP100"

# Never overwrite a previous formal run.
for cond in "${conditions[@]}"; do
    if [[ -e "results/final_evaluation/ant/$cond" ]]; then
        echo "[FATAL] Archive already exists:"
        echo "results/final_evaluation/ant/$cond"
        echo "Refusing to overwrite."
        exit 1
    fi
done

echo "============================================================"
echo "ANT FORMAL N=16 QUEUE"
echo "Start: $(date)"
echo "Git commit: $(git rev-parse HEAD)"
echo "============================================================"
nvidia-smi || true

run_one () {
    arch="$1"
    ckpt="$2"
    cond="$3"

    echo
    echo "============================================================"
    echo "[START] $cond"
    echo "Time: $(date)"
    echo "Architecture: $arch"
    echo "Checkpoint: $ckpt"
    echo "============================================================"

    "$PY" -u "$EVAL" \
      --arch "$arch" \
      --file-name "$ckpt" \
      --condition "$cond" \
      --hsize 32 \
      --num-runs 16 \
      2>&1 | tee "$LOGDIR/${cond}.log"

    echo
    echo "[DONE] $cond"
    echo "Time: $(date)"
}

# 1. Teacher
run_one \
  recurrent \
  "$TEACHER" \
  "formal_n16_teacher"

# 2. Corrected RNN seq20 @900
run_one \
  recurrent \
  "$RNN20" \
  "formal_n16_corrected_rnn_h32_seq20_prngfix"

# 3. Corrected RNN seq100 @900
run_one \
  recurrent \
  "$RNN100" \
  "formal_n16_corrected_rnn_h32_seq100_prngfix"

# 4. MLP seq20 @900
run_one \
  ff \
  "$MLP20" \
  "formal_n16_mlp_h32_seq20"

# 5. MLP seq100 @900
run_one \
  ff \
  "$MLP100" \
  "formal_n16_mlp_h32_seq100"

touch "$LOGDIR/QUEUE_COMPLETE"

echo
echo "============================================================"
echo "ALL FIVE ANT FORMAL N=16 CONDITIONS COMPLETED"
echo "Finish: $(date)"
echo "============================================================"
