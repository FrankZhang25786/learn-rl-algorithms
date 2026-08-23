import argparse
import os
import os.path as osp
import warnings
from datetime import datetime
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.flatten_util
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from evosax import ParameterReshaper
from jax import lax

import wandb
from algorithms.evaluation.eval_open_ff_bb import eval_func
from algorithms.utils.learned_optimization.learned_optimization.learned_optimizers import common

api = wandb.Api()
warnings.simplefilter(action="ignore", category=FutureWarning)


def _second_moment_normalizer(x, axis, eps=1e-5):
    return x * lax.rsqrt(eps + jnp.mean(jnp.square(x), axis=axis, keepdims=True))


def transform_data_OPEN(p, g, mom, dorm, layer_prop, train_prop, batch_prop):
    """Build the same 19 OPEN features used by the recurrent teacher and FF OPEN."""
    eps1 = 1e-13
    p = jnp.expand_dims(p, -1)
    g = jnp.expand_dims(g, -1)

    gsign = jnp.sign(g)
    glog = jnp.log(jnp.abs(g) + eps1)

    momsign = jnp.sign(mom)
    momlog = jnp.log(jnp.abs(mom) + eps1)

    inp_stack = jnp.concatenate([p, momsign, momlog], axis=-1)
    axis = list(range(len(p.shape)))
    inp_stack_g = jnp.concatenate([inp_stack, gsign, glog], axis=-1)
    inp_stack_g = _second_moment_normalizer(inp_stack_g, axis=axis)

    train_prop = jnp.expand_dims(train_prop, -1)
    batch_prop = jnp.expand_dims(batch_prop, -1)
    inp = jnp.concatenate(
        [train_prop, layer_prop, batch_prop, dorm], axis=-1
    )
    return jnp.concatenate([inp_stack_g, inp], axis=-1)


@flax.struct.dataclass
class TeacherState:
    params: Any
    rolling_features: common.MomAccumulator
    iteration: jnp.ndarray
    carry: Any
    rng: jnp.ndarray


class RecurrentOPENTeacher:
    def __init__(self, hidden_size=32, gru_features=16):
        self.gru = nn.GRUCell(features=gru_features)
        self.gru_features = gru_features
        self.mod = nn.Sequential(
            [
                nn.Dense(hidden_size),
                nn.LayerNorm(),
                nn.relu,
                nn.Dense(hidden_size),
                nn.LayerNorm(),
                nn.relu,
                nn.Dense(3),
            ]
        )

    def init(self, key, pholder):
        keys = jax.random.split(key, 5)
        proxy_carry = self.gru.initialize_carry(keys[4], (1,))
        return {
            "params": self.mod.init(keys[0], jnp.zeros([self.gru_features])),
            "gru_params": self.gru.init(keys[2], proxy_carry, pholder),
        }


class FeedForwardStudent(nn.Module):
    hsize: int = 32

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hsize)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(self.hsize)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        return nn.Dense(3)(x)


def get_teacher_sequence(key, points, seq_length):
    """Generate one synthetic sequence using the fixed recurrent teacher.

    Returns exogenous per-step features plus the teacher's final parameter vector.
    The FF student is later unrolled on the same sequence, but without recurrent state.
    """
    train_len = 2000
    rng = jax.random.split(key, 5)
    training_step = jax.random.randint(
        rng[0], [points], minval=0, maxval=train_len - seq_length - 1
    )
    decays = jnp.asarray([0.1, 0.5, 0.9, 0.99, 0.999, 0.9999])
    p_init = jax.random.normal(rng[1], shape=(points,)) * config["p_std"]
    layer_props = jax.random.choice(
        rng[2], jnp.array([0.0, 0.5, 1.0]), replace=True, shape=(points, 1)
    )
    carry = teacher.gru.initialize_carry(rng[3], (points, 1))
    moms = common.vec_rolling_mom(decays).init(p_init)

    state = TeacherState(
        params=p_init,
        rolling_features=moms,
        iteration=training_step,
        carry=carry,
        rng=rng[4],
    )

    def teacher_step(state, _):
        curr_params = state.params
        iterate = state.iteration
        rng = state.rng

        train_prop = iterate / (train_len - 1)
        batch_prop = (iterate % 8) / 7
        iterate += 1

        rng, subkey = jax.random.split(rng)
        dorm = jnp.clip(
            jax.random.normal(subkey, shape=(points, 1)) * config["dorm_std"] + 1,
            0,
            10,
        )
        rng, subkey = jax.random.split(rng)
        g = jax.random.normal(subkey, shape=(points,)) * config["g_std"]
        new_mom = common.vec_rolling_mom(decays).update(state.rolling_features, g)
        rng, subkey = jax.random.split(rng)
        rand = jax.random.normal(subkey, (points,))

        inp = transform_data_OPEN(
            curr_params,
            g,
            new_mom.m,
            dorm,
            layer_props,
            train_prop,
            batch_prop,
        )
        new_carry, gru_out = teacher.gru.apply(
            teacher_params["gru_params"], state.carry, inp
        )
        out = teacher.mod.apply(teacher_params["params"], gru_out)
        update = (
            out[..., 0] * 1e-3 * jnp.exp(out[..., 1] * 1e-3)
            + out[..., 2] * 1e-3 * rand
        )
        update = update - update.mean()
        new_params = curr_params - update

        new_state = TeacherState(
            params=new_params,
            rolling_features=new_mom,
            iteration=iterate,
            carry=new_carry,
            rng=rng,
        )
        # These are the exogenous quantities needed to construct the student's
        # current 19-D input at the same synthetic optimisation step.
        xs = (g, new_mom.m, dorm, layer_props, train_prop, batch_prop, rand)
        return new_state, xs

    state, xs = jax.lax.scan(teacher_step, state, xs=None, length=seq_length)
    return (p_init, xs), state.params


def unroll_student(student_params, sequence):
    p_init, xs = sequence

    def student_step(params, inp):
        g, mom, dorm, layer_props, train_prop, batch_prop, rand = inp
        features = transform_data_OPEN(
            params, g, mom, dorm, layer_props, train_prop, batch_prop
        )
        out = student.apply(student_params, features)
        update = (
            out[..., 0] * 1e-3 * jnp.exp(out[..., 1] * 1e-3)
            + out[..., 2] * 1e-3 * rand
        )
        update = update - update.mean()
        return params - update, None

    final_params, _ = jax.lax.scan(student_step, p_init, xs)
    return final_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", nargs="+", default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-stop", type=float, default=1e-5)
    parser.add_argument("--file-name", type=str, default=None)
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--exp-num", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--num-batches", type=int, default=100)
    parser.add_argument("--seq-length", type=int, default=20)
    parser.add_argument("--hsize", type=int, default=32)
    parser.add_argument("--base-hsize", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--test-size", type=int, default=10000)
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    envs = args.envs or [
        "breakout",
        "cartpole",
        "asterix",
        "ant",
        "spaceinvaders",
        "freeway",
    ]
    config = {
        "envs": envs,
        "lr": args.lr,
        "iters": args.iters,
        "hsize": args.hsize,
        "dorm_std": 1,
        "g_std": 0.5,
        "p_std": 1,
    }

    assert (args.exp_name and args.exp_num is not None) or args.file_name

    wandb.init(
        project="meta-analysis",
        config=vars(args),
        name="OPEN_recurrent_to_ff_distil",
    )

    if args.exp_name:
        run_path = f"meta-analysis/{args.exp_name}"
        api.run(run_path)
        restored = wandb.restore(
            f"curr_param_{args.exp_num}.npy",
            run_path=run_path,
            root="save_files/params/",
            replace=True,
        )
        flat_teacher_params = jnp.array(jnp.load(restored.name, allow_pickle=True))
    else:
        flat_teacher_params = jnp.array(jnp.load(args.file_name, allow_pickle=True))

    teacher = RecurrentOPENTeacher(
        hidden_size=args.base_hsize,
        gru_features=int(args.base_hsize / 2),
    )
    teacher_placeholder = teacher.init(
        jax.random.PRNGKey(0), jnp.zeros((2, 19))
    )
    teacher_reshaper = ParameterReshaper(teacher_placeholder)
    teacher_params = teacher_reshaper.reshape_single(flat_teacher_params)

    student = FeedForwardStudent(hsize=args.hsize)
    rng = jax.random.PRNGKey(100)
    rng, init_key = jax.random.split(rng)
    student_params = student.init(init_key, jnp.zeros((2, 19)))

    total_steps = args.epochs * args.iters * args.num_batches
    schedule = optax.linear_schedule(args.lr, args.lr_stop, total_steps)
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(student_params)

    @jax.jit
    def loss_fn(params, sequence, targets):
        preds = unroll_student(params, sequence)
        return jnp.mean((preds - targets) ** 2)

    @jax.jit
    def train_step(params, opt_state, sequence, targets):
        loss, grads = jax.value_and_grad(loss_fn)(params, sequence, targets)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    test_sequence, test_targets = get_teacher_sequence(
        jax.random.PRNGKey(50), args.test_size, args.seq_length
    )

    save_root = "save_files/open_recurrent_to_ff_distil"
    os.makedirs(save_root, exist_ok=True)
    run_dir = osp.join(save_root, str(datetime.now()).replace(" ", "_").replace(":", "-"))
    os.makedirs(run_dir, exist_ok=False)

    key = jax.random.PRNGKey(42)
    for outer in range(args.epochs):
        for inner in range(args.iters):
            epoch_loss = 0.0
            for _ in range(args.num_batches):
                key, subkey = jax.random.split(key)
                train_sequence, train_targets = get_teacher_sequence(
                    subkey, args.batch_size, args.seq_length
                )
                student_params, opt_state, batch_loss = train_step(
                    student_params, opt_state, train_sequence, train_targets
                )
                epoch_loss += batch_loss

            epoch_loss /= args.num_batches
            test_loss = loss_fn(student_params, test_sequence, test_targets)
            print(
                f"Outer {outer}, Epoch {inner}, Train Loss: {epoch_loss:.6e}, "
                f"Test Loss: {test_loss:.6e}"
            )

        wandb.log(
            {"train_loss": epoch_loss, "test_loss": test_loss},
            step=outer,
        )

        checkpoint = osp.join(run_dir, f"curr_param_{outer}.npy")
        jnp.save(checkpoint, jax.flatten_util.ravel_pytree(student_params)[0])
        wandb.save(checkpoint, base_path=run_dir)

        if not args.skip_eval:
            eval_func(
                envs=config["envs"],
                meta_params=student_params,
                iteration=outer,
                title="",
                hsize=args.hsize,
            )
        plt.close("all")

    print(f"Saved checkpoints to: {run_dir}")
