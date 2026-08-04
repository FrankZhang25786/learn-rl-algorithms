import argparse
import json
import os
import os.path as osp
import time
from datetime import datetime

import gymnax
import jax
import jax.numpy as jnp
import numpy as np
from evosax import OpenES

from algorithms.evaluation.eval_open_recurrent_bb import Actor, make_train
from algorithms.utils.architectures.open_recurrent import recurrent_open_gru
from algorithms.utils.configs import all_configs
from algorithms.utils.wrappers import FlatWrapper, GymnaxGymWrapper


LABELS = ("generation_792", "zero", "fixed_random_seed12345")


def _flat_run(tree, candidate, run):
    return jax.flatten_util.ravel_pytree(
        jax.tree_util.tree_map(lambda x: x[candidate, run], tree)
    )[0]


def _initial_actor_params(actor, init_x, rngs):
    def init_one(rng):
        _, actor_rng = jax.random.split(rng)
        return actor.init(actor_rng, init_x)

    return jax.vmap(init_one)(rngs)


def _policy_comparison(actor, params_792, params_zero, obs_792, obs_zero, keys):
    def compare_one(p792, pzero, o792, ozero, key):
        observations = jnp.concatenate([o792, ozero], axis=0)
        pi_792, _ = actor.apply(p792, observations)
        pi_zero, _ = actor.apply(pzero, observations)
        probs_792 = pi_792.probs
        probs_zero = pi_zero.probs
        eps = 1e-8
        kl_792_zero = jnp.sum(
            probs_792 * (jnp.log(probs_792 + eps) - jnp.log(probs_zero + eps)),
            axis=-1,
        )
        kl_zero_792 = jnp.sum(
            probs_zero * (jnp.log(probs_zero + eps) - jnp.log(probs_792 + eps)),
            axis=-1,
        )
        symmetric_kl = 0.5 * (kl_792_zero + kl_zero_792)
        probability_mad = jnp.mean(jnp.abs(probs_792 - probs_zero), axis=-1)
        sampled_792 = pi_792.sample(seed=key)
        sampled_zero = pi_zero.sample(seed=key)
        greedy_792 = jnp.argmax(probs_792, axis=-1)
        greedy_zero = jnp.argmax(probs_zero, axis=-1)
        return (
            observations,
            probs_792,
            probs_zero,
            sampled_792,
            sampled_zero,
            greedy_792,
            greedy_zero,
            symmetric_kl,
            probability_mad,
        )

    return jax.vmap(compare_one)(params_792, params_zero, obs_792, obs_zero, keys)


def _summary_stats(x):
    x = np.asarray(x)
    return {"mean": float(x.mean()), "std": float(x.std())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-792",
        default="save_files/open_recurrent_bb/2026-08-03_16-06-12/curr_param_792.npy",
    )
    parser.add_argument("--zero-checkpoint", default="/tmp/open_zero.npy")
    parser.add_argument(
        "--random-checkpoint", default="/tmp/open_random_seed12345.npy"
    )
    parser.add_argument("--adaptation-updates", nargs="+", type=int, default=[20, 50, 100])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--num-runs", type=int, default=50)
    parser.add_argument("--hsize", type=int, default=16)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if min(args.adaptation_updates) < 1:
        raise ValueError("adaptation_updates must be at least 1")

    output_dir = args.output_dir or osp.join(
        "evaluation_results",
        "open_recurrent_gen792_vs_zero_random_"
        + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
    os.makedirs(output_dir, exist_ok=False)

    checkpoint_paths = (
        args.checkpoint_792,
        args.zero_checkpoint,
        args.random_checkpoint,
    )
    flat_checkpoints = jnp.stack(
        [jnp.asarray(np.load(path)) for path in checkpoint_paths]
    )
    meta_opt = recurrent_open_gru(
        hidden_size=args.hsize, gru_features=args.hsize // 2
    )
    placeholder = meta_opt.init(jax.random.PRNGKey(0))
    strategy = OpenES(
        popsize=2,
        pholder_params=placeholder,
        opt_name="adam",
        centered_rank=True,
        maximize=True,
    )
    meta_trees = strategy.param_reshaper.reshape(flat_checkpoints)

    config_base = dict(all_configs["breakout"])
    env, env_params = gymnax.make(config_base["ENV_NAME"])
    env = FlatWrapper(GymnaxGymWrapper(env, env_params, config_base))
    actor = Actor(env.action_space, config=config_base)
    init_x = jnp.zeros(env.observation_space)

    metadata = {
        "labels": LABELS,
        "checkpoint_paths": [osp.abspath(p) for p in checkpoint_paths],
        "adaptation_updates": args.adaptation_updates,
        "seeds": args.seeds,
        "num_runs": args.num_runs,
        "hsize": args.hsize,
        "environment": "breakout",
        "policy_comparison_observations": (
            "Pooled final observations from generation-792 and zero evaluation "
            "states; both final policies are applied to every pooled observation."
        ),
        "action_agreement": (
            "Categorical samples from both policies using the identical PRNG key "
            "on each matched observation."
        ),
    }
    with open(osp.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    summaries = {}
    for updates in args.adaptation_updates:
        config = dict(config_base)
        config["VISUALISE"] = True
        config["OPTIM_HSIZE"] = args.hsize
        config["ADAPTATION_UPDATES"] = updates
        inner = jax.jit(jax.vmap(make_train(config), in_axes=(0, None)))
        evaluate = jax.jit(
            jax.vmap(inner, in_axes=(None, strategy.param_reshaper.vmap_dict))
        )
        horizon_summary = {}

        for seed in args.seeds:
            rngs = jax.random.split(jax.random.PRNGKey(seed), args.num_runs)
            start = time.time()
            runner_state, metrics = evaluate(rngs, meta_trees)
            jax.block_until_ready(metrics["returned_episode_returns"])
            runtime = time.time() - start

            actor_tree = runner_state[0].params
            final_observations = runner_state[3]
            initial_tree = _initial_actor_params(actor, init_x, rngs)
            initial_flat = jnp.stack(
                [
                    jax.flatten_util.ravel_pytree(
                        jax.tree_util.tree_map(lambda x: x[run], initial_tree)
                    )[0]
                    for run in range(args.num_runs)
                ]
            )
            final_flat = jnp.stack(
                [
                    jnp.stack(
                        [
                            _flat_run(actor_tree, candidate, run)
                            for run in range(args.num_runs)
                        ]
                    )
                    for candidate in range(len(LABELS))
                ]
            )
            update_l2 = jnp.linalg.norm(
                final_flat - initial_flat[None, ...], axis=-1
            )
            pairwise_l2 = jnp.stack(
                [
                    jnp.linalg.norm(final_flat[0] - final_flat[1], axis=-1),
                    jnp.linalg.norm(final_flat[0] - final_flat[2], axis=-1),
                    jnp.linalg.norm(final_flat[1] - final_flat[2], axis=-1),
                ]
            )

            params_792 = jax.tree_util.tree_map(lambda x: x[0], actor_tree)
            params_zero = jax.tree_util.tree_map(lambda x: x[1], actor_tree)
            comparison_keys = jax.random.split(
                jax.random.fold_in(jax.random.PRNGKey(seed), 0xA6710),
                args.num_runs,
            )
            (
                matched_observations,
                probs_792,
                probs_zero,
                sampled_792,
                sampled_zero,
                greedy_792,
                greedy_zero,
                symmetric_kl,
                probability_mad,
            ) = _policy_comparison(
                actor,
                params_792,
                params_zero,
                final_observations[0],
                final_observations[1],
                comparison_keys,
            )

            raw_returns = metrics["returned_episode_returns"]
            final_returns = raw_returns[:, :, -1, :].mean(axis=-1)
            saved_return_curves = raw_returns.mean(axis=-1).mean(axis=-1)
            sampled_agreement = sampled_792 == sampled_zero
            greedy_agreement = greedy_792 == greedy_zero

            arrays = {
                "raw_evaluation_returns": raw_returns,
                "final_returns": final_returns,
                "saved_return_curves": saved_return_curves,
                "initial_actor_params": initial_flat,
                "final_actor_params": final_flat,
                "actor_update_l2": update_l2,
                "pairwise_actor_l2": pairwise_l2,
                "matched_observations": matched_observations,
                "probs_792": probs_792,
                "probs_zero": probs_zero,
                "sampled_actions_792": sampled_792,
                "sampled_actions_zero": sampled_zero,
                "sampled_action_agreement": sampled_agreement,
                "greedy_actions_792": greedy_792,
                "greedy_actions_zero": greedy_zero,
                "greedy_action_agreement": greedy_agreement,
                "symmetric_kl": symmetric_kl,
                "probability_mad": probability_mad,
            }
            arrays_np = {key: np.asarray(value) for key, value in arrays.items()}
            finite = {
                key: bool(np.isfinite(value).all())
                for key, value in arrays_np.items()
                if not np.issubdtype(value.dtype, np.bool_)
            }
            nan_counts = {
                key: int(np.isnan(value).sum())
                for key, value in arrays_np.items()
                if np.issubdtype(value.dtype, np.floating)
            }

            raw_path = osp.join(
                output_dir, f"raw_updates_{updates}_seed_{seed}.npz"
            )
            np.savez_compressed(
                raw_path,
                **arrays_np,
                runtime_seconds=np.asarray(runtime),
                labels=np.asarray(LABELS),
                pairwise_labels=np.asarray(
                    ["792_vs_zero", "792_vs_random", "zero_vs_random"]
                ),
            )

            seed_summary = {
                "runtime_seconds": runtime,
                "raw_path": osp.abspath(raw_path),
                "final_return": {
                    LABELS[i]: _summary_stats(arrays_np["final_returns"][i])
                    for i in range(len(LABELS))
                },
                "actor_update_l2": {
                    LABELS[i]: _summary_stats(arrays_np["actor_update_l2"][i])
                    for i in range(len(LABELS))
                },
                "pairwise_actor_l2": {
                    label: _summary_stats(arrays_np["pairwise_actor_l2"][i])
                    for i, label in enumerate(
                        ("792_vs_zero", "792_vs_random", "zero_vs_random")
                    )
                },
                "sampled_action_agreement_rate": float(
                    arrays_np["sampled_action_agreement"].mean()
                ),
                "greedy_action_agreement_rate": float(
                    arrays_np["greedy_action_agreement"].mean()
                ),
                "symmetric_kl": _summary_stats(arrays_np["symmetric_kl"]),
                "probability_mad": _summary_stats(arrays_np["probability_mad"]),
                "finite": finite,
                "nan_counts": nan_counts,
            }
            summary_path = osp.join(
                output_dir, f"summary_updates_{updates}_seed_{seed}.json"
            )
            with open(summary_path, "w") as f:
                json.dump(seed_summary, f, indent=2)
            horizon_summary[str(seed)] = seed_summary
            print(
                f"updates={updates} seed={seed} runtime={runtime:.3f}s "
                f"return_792={seed_summary['final_return']['generation_792']['mean']:.6f} "
                f"return_zero={seed_summary['final_return']['zero']['mean']:.6f} "
                f"agreement={seed_summary['sampled_action_agreement_rate']:.6f} "
                f"sym_kl={seed_summary['symmetric_kl']['mean']:.9f}",
                flush=True,
            )

        summaries[str(updates)] = horizon_summary

    with open(osp.join(output_dir, "all_seed_summaries.json"), "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"output_dir={osp.abspath(output_dir)}", flush=True)


if __name__ == "__main__":
    main()
