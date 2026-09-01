import argparse
import glob
import hashlib
import json
import os
import shutil
from datetime import datetime

import jax.numpy as jnp
import wandb

from algorithms.utils.configs import all_configs
from algorithms.evaluation.eval_open_recurrent_bb import eval_func as recurrent_eval
from algorithms.evaluation.eval_open_ff_bb import eval_func as ff_eval


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arch",
        choices=["recurrent", "ff"],
        required=True,
    )
    parser.add_argument(
        "--file-name",
        required=True,
    )
    parser.add_argument(
        "--condition",
        required=True,
        help="Archive name, e.g. teacher or rnn_h32_seq20",
    )
    parser.add_argument(
        "--hsize",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
    )
    args = parser.parse_args()

    checkpoint = os.path.abspath(args.file_name)

    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(checkpoint)

    # IMPORTANT:
    # No sanity overrides here.
    # This uses the repository's formal Ant configuration unchanged.
    cfg = all_configs["ant"]

    num_updates = (
        cfg["TOTAL_TIMESTEPS"]
        // cfg["NUM_STEPS"]
        // cfg["NUM_ENVS"]
    )

    params = jnp.load(checkpoint, allow_pickle=True)
    param_count = int(params.size)

    expected_params = 3571 if args.arch == "recurrent" else 1923
    if param_count != expected_params:
        raise ValueError(
            f"Unexpected parameter count for {args.arch}: "
            f"{param_count} != {expected_params}"
        )

    archive_dir = os.path.join(
        "results",
        "final_evaluation",
        "ant",
        args.condition,
    )
    os.makedirs(archive_dir, exist_ok=True)

    print("=== FORMAL ANT EVALUATION ===")
    print(f"condition: {args.condition}")
    print(f"arch: {args.arch}")
    print(f"checkpoint: {checkpoint}")
    print(f"checkpoint parameters: {param_count}")
    print(f"num_runs: {args.num_runs}")

    for key in [
        "ENV_NAME",
        "NUM_ENVS",
        "NUM_STEPS",
        "TOTAL_TIMESTEPS",
        "UPDATE_EPOCHS",
        "NUM_MINIBATCHES",
        "GAMMA",
        "CONTINUOUS",
        "NORMALIZE",
        "CLIP_ACTION",
    ]:
        print(f"{key}: {cfg[key]}")

    print(f"NUM_UPDATES: {num_updates}")

    metadata = {
        "condition": args.condition,
        "architecture": args.arch,
        "checkpoint": checkpoint,
        "checkpoint_sha256": sha256_file(checkpoint),
        "parameter_count": param_count,
        "optimizer_hsize": args.hsize,
        "num_runs": args.num_runs,
        "rng_base_key": 42,
        "environment": "ant",
        "config": {
            key: cfg[key]
            for key in [
                "ENV_NAME",
                "NUM_ENVS",
                "NUM_STEPS",
                "TOTAL_TIMESTEPS",
                "UPDATE_EPOCHS",
                "NUM_MINIBATCHES",
                "GAMMA",
                "GAE_LAMBDA",
                "CLIP_EPS",
                "ENT_COEF",
                "VF_COEF",
                "MAX_GRAD_NORM",
                "ACTIVATION",
                "HSIZE",
                "CONTINUOUS",
                "NORMALIZE",
                "CLIP_ACTION",
            ]
        },
        "num_updates": int(num_updates),
        "started_at": datetime.now().isoformat(),
    }

    with open(
        os.path.join(archive_dir, "metadata.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2)

    before = set(
        glob.glob("save_files/eval/*/returns_ant.npy")
    )

    wandb.init(
        project="meta-analysis",
        name=f"ant_{args.condition}",
        mode="disabled",
    )

    if args.arch == "recurrent":
        eval_func = recurrent_eval
    else:
        eval_func = ff_eval

    eval_func(
        envs=["ant"],
        meta_params=params,
        num_runs=args.num_runs,
        iteration=0,
        title=args.condition,
        hsize=args.hsize,
    )

    wandb.finish()

    after = set(
        glob.glob("save_files/eval/*/returns_ant.npy")
    )

    created = list(after - before)

    if len(created) == 1:
        source_returns = created[0]
    elif len(created) > 1:
        source_returns = max(created, key=os.path.getmtime)
    else:
        candidates = list(after)
        if not candidates:
            raise RuntimeError(
                "Evaluation completed but returns_ant.npy was not found."
            )
        source_returns = max(candidates, key=os.path.getmtime)

    destination = os.path.join(
        archive_dir,
        "returns_ant.npy",
    )
    shutil.copy2(source_returns, destination)

    returns = jnp.load(destination)

    metadata["completed_at"] = datetime.now().isoformat()
    metadata["source_returns_file"] = source_returns
    metadata["archived_returns_file"] = destination
    metadata["returns_shape"] = list(returns.shape)

    with open(
        os.path.join(archive_dir, "metadata.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2)

    print("\n=== ARCHIVE COMPLETE ===")
    print(f"returns source: {source_returns}")
    print(f"returns archive: {destination}")
    print(f"returns shape: {returns.shape}")
    print(
        f"metadata: "
        f"{os.path.join(archive_dir, 'metadata.json')}"
    )


if __name__ == "__main__":
    main()