"""
train.py
========
PPO training script for a single Baku ATSC scenario.

Usage
-----
  python train.py --scenario main
  python train.py --scenario bottleneck --steps 500000
  python train.py --scenario main --resume models/main/ppo_model.zip

Thesis §3.5 – §3.6:
  • Shared PPO policy, MLP-128-128
  • SubprocVecEnv(4) + VecNormalize (clip=10)
  • n_steps=1024, batch=64, epochs=10, γ=0.99, λ=0.95, ε=0.2, α=3e-4
  • ≈500 000 cumulative decision steps per scenario
  • B2 baseline computed from 200-step passive FT rollout before training
"""

from __future__ import annotations
import os
import sys
import argparse
import warnings
import logging

import numpy as np

os.environ.setdefault("SUMO_HOME", "/usr/share/sumo")
warnings.filterwarnings("ignore", category=UserWarning)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

from envs.baku_sumo_env   import BakuSUMOEnv
from envs.trimmed_env     import TrimmedTrafficEnv
from envs.scenario_configs import get_scenario_config
from baselines.fixed_time  import FixedTimeRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Hyper-parameters (thesis Table §3.5) ────────────────────────────────────

PPO_KWARGS = dict(
    learning_rate = 3e-4,
    n_steps       = 1024,
    batch_size    = 64,
    n_epochs      = 10,
    gamma         = 0.99,
    gae_lambda    = 0.95,
    ent_coef      = 0.01,
    clip_range    = 0.2,
    policy_kwargs = {"net_arch": [128, 128]},
    verbose       = 1,
)

N_ENVS       = 4      # parallel SUMO processes
TRAIN_SEED   = 42     # training seed (thesis §3.5)
WARMUP_STEPS = 200    # FT rollout length for B2 baseline


# ── Environment factory ──────────────────────────────────────────────────────

def _make_env(scenario_name: str, cfg: dict, rank: int, b0: float = 0.0):
    """Return a callable that creates a TrimmedTrafficEnv (for SubprocVecEnv)."""
    sumocfg   = cfg["sumocfg_path"]
    tl_ids    = cfg["tl_ids"]
    n_phases  = cfg["n_phases"]
    seed      = TRAIN_SEED + rank

    def _init():
        base = BakuSUMOEnv(
            tl_ids          = tl_ids,
            sumocfg_path    = sumocfg,
            n_phases_per_tl = n_phases,
            seed            = seed,
            b0              = b0,
            label           = f"train_{rank}",
        )
        return TrimmedTrafficEnv(base)

    return _init


# ── B2 baseline computation ──────────────────────────────────────────────────

def compute_b2_baseline(cfg: dict, n_steps: int = WARMUP_STEPS) -> float:
    """
    Run the fixed-time baseline for `n_steps` and compute the negative
    mean raw reward per step.  This offset is stored as b_0.
    """
    log.info("Computing B2 baseline (%d steps, seed %d)…", n_steps, TRAIN_SEED)

    runner = FixedTimeRunner(
        sumocfg_path = cfg["sumocfg_path"],
        tl_ids       = cfg["tl_ids"],
        seed         = TRAIN_SEED,
        label        = "b2_baseline",
    )
    runner.start()

    # Build a temporary env to get lane capacities and collect raw rewards
    tmp_env = BakuSUMOEnv(
        tl_ids          = cfg["tl_ids"],
        sumocfg_path    = cfg["sumocfg_path"],
        n_phases_per_tl = cfg["n_phases"],
        seed            = TRAIN_SEED,
        b0              = 0.0,
        label           = "b2_tmp",
    )
    tmp_env.reset()

    rewards = []
    for t in range(n_steps):
        # No-op: keep current green phase
        action = np.zeros(BakuSUMOEnv.MAX_TLS, dtype=np.int64)
        _, reward, done, _, _ = tmp_env.step(action)
        rewards.append(reward)
        if done:
            break

    tmp_env.close()
    runner.stop()

    if not rewards:
        return 0.0

    b0 = -float(np.mean(rewards))
    log.info("B2 baseline  b0 = %.6f  (mean raw reward = %.6f)", b0, -b0)
    return b0


# ── Main training loop ───────────────────────────────────────────────────────

def train(
    scenario_name: str,
    total_steps:   int   = 500_000,
    resume_path:   str | None = None,
):
    cfg      = get_scenario_config(scenario_name)
    out_dir  = os.path.join("models", scenario_name)
    os.makedirs(out_dir, exist_ok=True)
    log_dir  = os.path.join("logs", scenario_name)
    os.makedirs(log_dir, exist_ok=True)

    # ── B2 baseline ──────────────────────────────────────────────────────
    b0_path = os.path.join(out_dir, "b0.npy")
    if os.path.exists(b0_path):
        b0 = float(np.load(b0_path))
        log.info("Loaded B2 baseline: b0 = %.6f", b0)
    else:
        b0 = compute_b2_baseline(cfg)
        np.save(b0_path, np.array(b0))

    # ── Vectorised training environments ─────────────────────────────────
    env_fns  = [_make_env(scenario_name, cfg, rank, b0) for rank in range(N_ENVS)]
    vec_env  = SubprocVecEnv(env_fns, start_method="spawn")
    vec_env  = VecNormalize(
        vec_env,
        norm_obs     = True,
        norm_reward  = True,
        clip_obs     = 10.0,
        clip_reward  = 10.0,
        gamma        = PPO_KWARGS["gamma"],
    )

    # ── Model ─────────────────────────────────────────────────────────────
    if resume_path and os.path.exists(resume_path):
        log.info("Resuming from %s", resume_path)
        model = PPO.load(resume_path, env=vec_env, **{
            k: v for k, v in PPO_KWARGS.items()
            if k not in ("policy_kwargs",)
        })
        # Load VecNormalize stats if available
        vn_path = os.path.join(os.path.dirname(resume_path), "vec_normalize.pkl")
        if os.path.exists(vn_path):
            vec_env = VecNormalize.load(vn_path, vec_env)
            log.info("Loaded VecNormalize stats from %s", vn_path)
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            tensorboard_log = log_dir,
            **PPO_KWARGS,
        )

    # ── Checkpoint callback ───────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq   = max(1, 50_000 // N_ENVS),
        save_path   = out_dir,
        name_prefix = "ppo_ckpt",
        verbose     = 0,
    )

    log.info(
        "Training  scenario=%s  steps=%d  envs=%d  b0=%.4f",
        scenario_name, total_steps, N_ENVS, b0,
    )

    model.learn(
        total_timesteps        = total_steps,
        callback               = checkpoint_cb,
        reset_num_timesteps    = resume_path is None,
    )

    # ── Save ──────────────────────────────────────────────────────────────
    model_path = os.path.join(out_dir, "ppo_model")
    model.save(model_path)
    vec_env.save(os.path.join(out_dir, "vec_normalize.pkl"))
    log.info("Saved model → %s.zip", model_path)

    vec_env.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train PPO on a Baku SUMO scenario")
    p.add_argument(
        "--scenario", required=True,
        choices=["bottleneck", "main", "pedestrian", "hexagon"],
        help="Scenario to train on",
    )
    p.add_argument("--steps",  type=int, default=500_000,
                   help="Total training timesteps (default 500 000)")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to .zip checkpoint to resume from")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    train(args.scenario, total_steps=args.steps, resume_path=args.resume)
