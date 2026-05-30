"""
train.py
========
PPO training script for a single Baku ATSC scenario.

Usage
-----
  python train.py --scenario main
  python train.py --scenario main --steps 500000 --seed 42
  python train.py --scenario main --resume models/main/ppo_model.zip
  # Strict reproducibility (bit-closer TB curves; slower):
  python train.py --scenario main --cpu --n-envs 1 --seed 42

Thesis §3.5 – §3.6:
  • Shared PPO policy, MLP-128-128
  • SubprocVecEnv(4) + VecNormalize (clip=10)
  • n_steps=1024, batch=64, epochs=10, γ=0.99, λ=0.95, ε=0.2, α=3e-4
  • ≈500 000 cumulative decision steps per scenario
  • B2 baseline: 200-step passive fixed-time rollout → b0, optional trace
    `models/<scenario>/b2_rewards_200.npy` + sha256 in `.meta` (delete `b0.npy`
    to recompute).

Reproducibility
---------------
  `repro.apply_reproducibility` seeds Python, NumPy, and PyTorch; sets CuDNN
  deterministic on GPU. PPO is constructed with an explicit `seed=`.
  `repro_run_manifest.json` is written with versions and flags.  GPU training
  can still differ slightly run-to-run; use `--cpu` for stricter matching.
  TensorBoard episode return is **VecNormalize**-scaled, not raw R (see thesis).
"""

from __future__ import annotations
import os
import csv
import sys
import tempfile
import argparse
import warnings
import logging
from typing import Any, Callable, Dict, List

import numpy as np

os.environ.setdefault("SUMO_HOME", "/usr/share/sumo")
warnings.filterwarnings("ignore", category=UserWarning)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from envs.baku_sumo_env   import BakuSUMOEnv
from envs.trimmed_env     import TrimmedTrafficEnv
from envs.scenario_configs import get_scenario_config
from baselines.fixed_time  import FixedTimeRunner
from repro import apply_reproducibility, run_manifest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _pct_improvement(ft_val: float, rl_val: float, maximise: bool) -> float:
    if ft_val == 0.0:
        return 0.0
    if maximise:
        return (rl_val - ft_val) / ft_val * 100.0
    return (ft_val - rl_val) / ft_val * 100.0


# ── Live evaluation callback ──────────────────────────────────────────────────

class TrafficEvalCallback(BaseCallback):
    """
    Every `eval_freq` training timesteps, runs a short evaluation episode
    and prints a live row: PPO vs FT % improvement for all 4 metrics plus
    the mean normalised reward collected since the last print.

    Results are also saved to `<out_dir>/eval_progress.csv` at training end.
    """

    EVAL_STEPS   = 300   # decision steps per quick-eval episode
    WARMUP_STEPS =  30   # warm-up steps discarded from metric collection
    METRICS      = ("stopped_ratio", "waiting_time", "queue_length", "throughput")
    MAXIMISE     = {"throughput"}

    def __init__(
        self,
        cfg:       dict,
        b0:        float,
        eval_freq: int = 10_000,
        seed:      int = 42,
        out_dir:   str = ".",
    ):
        super().__init__(verbose=0)
        self.cfg       = cfg
        self.b0        = b0
        self.eval_freq = eval_freq
        self.seed      = seed
        self.out_dir   = out_dir

        self._next_eval    = eval_freq   # adjusted on training start for resumes
        self._ft: Dict[str, float] = {}
        self._history: List[dict]  = []
        self._step_rewards: List[float] = []
        self._best_wait: float = -np.inf   # best waiting-time improvement seen

    # ── SB3 callback hooks ────────────────────────────────────────────────

    def _on_training_start(self) -> None:
        # Align first eval to next boundary (handles resume from checkpoint)
        self._next_eval = (
            (self.num_timesteps // self.eval_freq) + 1
        ) * self.eval_freq

        log.info("EvalCallback: computing FT baseline (seed=%d, %d steps)…",
                 self.seed, self.EVAL_STEPS)
        runner = FixedTimeRunner(
            sumocfg_path = self.cfg["sumocfg_path"],
            tl_ids       = self.cfg["tl_ids"],
            seed         = self.seed,
            label        = "eval_cb_ft",
        )
        runner.start(seed=self.seed)
        self._ft = runner.run_and_aggregate(
            max_steps    = self.EVAL_STEPS,
            warmup_steps = self.WARMUP_STEPS,
        )
        runner.stop()

        log.info(
            "FT baseline  stop=%.4f  wait=%.2f  queue=%.4f  thr=%.4f",
            self._ft["stopped_ratio"], self._ft["waiting_time"],
            self._ft["queue_length"],  self._ft["throughput"],
        )

        print(f"\n{'─'*82}")
        print(
            f"  {'Timestep':>9}  {'Stop.Ratio%':>12}  {'WaitTime%':>10}  "
            f"{'Queue%':>8}  {'Throughput%':>12}  {'MeanRew':>9}"
        )
        print(f"{'─'*82}", flush=True)

    def _on_step(self) -> bool:
        rew = self.locals.get("rewards")
        if rew is not None:
            self._step_rewards.extend(np.asarray(rew).tolist())

        if self.num_timesteps >= self._next_eval:
            self._next_eval += self.eval_freq
            self._do_eval()
        return True

    def _on_training_end(self) -> None:
        print(f"{'─'*82}\n", flush=True)
        self._save_csv()

    # ── Evaluation episode ────────────────────────────────────────────────

    def _do_eval(self) -> None:
        base    = BakuSUMOEnv(
            tl_ids          = self.cfg["tl_ids"],
            sumocfg_path    = self.cfg["sumocfg_path"],
            n_phases_per_tl = self.cfg["n_phases"],
            seed            = self.seed,
            b0              = self.b0,
            meter_tls       = self.cfg.get("meter_tls"),
            label           = "eval_cb_rl",
        )
        trimmed = TrimmedTrafficEnv(base)
        vec     = DummyVecEnv([lambda: trimmed])   # type: ignore[arg-type]

        # Copy running normalisation stats from the live training VecNormalize
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            tmp_path = f.name
        try:
            self.training_env.save(tmp_path)
            eval_vn = VecNormalize.load(tmp_path, vec)
        finally:
            os.unlink(tmp_path)

        eval_vn.training    = False
        eval_vn.norm_reward = False

        obs  = eval_vn.reset()
        step_metrics: Dict[str, List[float]] = {m: [] for m in self.METRICS}

        for step in range(self.EVAL_STEPS):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, dones, _ = eval_vn.step(action)
            if step >= self.WARMUP_STEPS:
                m = trimmed.get_metrics()
                step_metrics["stopped_ratio"].append(m["stopped_ratio"])
                step_metrics["waiting_time"].append(m["waiting_time"])
                step_metrics["queue_length"].append(m["queue_length"])
                step_metrics["throughput"].append(m["arrived"])
            if bool(dones[0]):
                break

        eval_vn.close()

        rl  = {k: float(np.mean(v)) if v else 0.0 for k, v in step_metrics.items()}
        imp = {
            m: _pct_improvement(self._ft[m], rl[m], m in self.MAXIMISE)
            for m in self.METRICS
        }

        mean_rew = float(np.mean(self._step_rewards)) if self._step_rewards else 0.0
        self._step_rewards.clear()

        self._history.append({
            "timestep":    self.num_timesteps,
            "mean_reward": mean_rew,
            **{f"imp_{m}": imp[m] for m in self.METRICS},
        })

        # ── Best-model selection (anti-collapse) ──────────────────────────
        # Keep the checkpoint with the best waiting-time improvement (the
        # thesis primary metric) so a late-training collapse cannot destroy
        # the best policy. Recover with `--resume models/<sc>/ppo_best.zip`.
        best_marker = ""
        if imp["waiting_time"] > self._best_wait:
            self._best_wait = imp["waiting_time"]
            self.model.save(os.path.join(self.out_dir, "ppo_best"))
            try:
                self.training_env.save(
                    os.path.join(self.out_dir, "vec_normalize_best.pkl")
                )
            except Exception:
                pass
            best_marker = "  ← BEST"

        print(
            f"  {self.num_timesteps:>9,}"
            f"  {imp['stopped_ratio']:>+12.1f}"
            f"  {imp['waiting_time']:>+10.1f}"
            f"  {imp['queue_length']:>+8.1f}"
            f"  {imp['throughput']:>+12.1f}"
            f"  {mean_rew:>+9.4f}{best_marker}",
            flush=True,
        )

    # ── CSV export ────────────────────────────────────────────────────────

    def _save_csv(self) -> None:
        if not self._history:
            return
        csv_path = os.path.join(self.out_dir, "eval_progress.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._history[0].keys())
            writer.writeheader()
            writer.writerows(self._history)
        log.info("Eval progress saved → %s", csv_path)


# ── Hyper-parameters (thesis Table §3.5) ────────────────────────────────────

def linear_schedule(initial: float, final: float = 0.0):
    """SB3 learning-rate schedule: anneal `initial`→`final` over training.
    SB3 passes progress_remaining ∈ [1.0 (start) … 0.0 (end)]."""
    def f(progress_remaining: float) -> float:
        return final + progress_remaining * (initial - final)
    return f


PPO_KWARGS = dict(
    # Anneal LR 3e-4 → 3e-5. A constant LR let the converged policy take a few
    # large steps and fall off a cliff (the 450k→500k collapse on `main`).
    learning_rate = linear_schedule(3e-4, 3e-5),
    n_steps       = 1024,
    batch_size    = 64,
    n_epochs      = 10,
    gamma         = 0.99,
    gae_lambda    = 0.95,
    ent_coef      = 0.05,
    clip_range    = 0.2,
    # Early-stop the epoch loop if the policy moves too far in one update — the
    # primary guard against catastrophic policy collapse late in training.
    target_kl     = 0.03,
    policy_kwargs = {"net_arch": [128, 128]},
    verbose       = 1,
)

N_ENVS_DEFAULT = 4    # parallel SUMO processes (thesis SubprocVecEnv(4))
TRAIN_SEED_DEFAULT = 42
WARMUP_STEPS = 200    # FT rollout length for B2 baseline — match thesis / Progress Report


# ── Environment factory ──────────────────────────────────────────────────────

def _make_env(
    cfg: dict,
    rank: int,
    b0: float,
    training_seed: int,
) -> Callable[[], Any]:
    """Return a callable that creates a TrimmedTrafficEnv (for SubprocVecEnv)."""
    sumocfg   = cfg["sumocfg_path"]
    tl_ids    = cfg["tl_ids"]
    n_phases  = cfg["n_phases"]
    seed      = int(training_seed) + int(rank)

    def _init():
        base = BakuSUMOEnv(
            tl_ids          = tl_ids,
            sumocfg_path    = sumocfg,
            n_phases_per_tl = n_phases,
            seed            = seed,
            b0              = b0,
            meter_tls       = cfg.get("meter_tls"),
            label           = f"train_{rank}",
        )
        return TrimmedTrafficEnv(base)

    return _init


# ── B2 baseline computation ──────────────────────────────────────────────────

def compute_b2_baseline(
    cfg: dict,
    n_steps: int = WARMUP_STEPS,
    training_seed: int = TRAIN_SEED_DEFAULT,
    out_dir: str | None = None,
) -> float:
    """
    Run the fixed-time baseline for `n_steps` and compute the negative
    mean raw reward per step.  This offset is stored as b_0.

    If out_dir is set, writes b2_rewards_200.npy (per-step r_t) and a SHA256
    hash of that array for cross-run checks (reproducibility).
    """
    log.info("Computing B2 baseline (%d steps, seed %d)…", n_steps, training_seed)

    runner = FixedTimeRunner(
        sumocfg_path = cfg["sumocfg_path"],
        tl_ids       = cfg["tl_ids"],
        seed         = training_seed,
        label        = "b2_baseline",
    )
    runner.start()

    # Build a temporary env to get lane capacities and collect raw rewards
    tmp_env = BakuSUMOEnv(
        tl_ids          = cfg["tl_ids"],
        sumocfg_path    = cfg["sumocfg_path"],
        n_phases_per_tl = cfg["n_phases"],
        seed            = training_seed,
        b0              = 0.0,
        meter_tls       = cfg.get("meter_tls"),
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

    if out_dir is not None and rewards:
        import hashlib

        arr = np.array(rewards, dtype=np.float64)
        p = os.path.join(out_dir, "b2_rewards_200.npy")
        np.save(p, arr)
        h = hashlib.sha256(arr.tobytes()).hexdigest()[:16]
        log.info("B2 per-step reward trace → %s  (sha256[0:16]=%s)", p, h)
        with open(os.path.join(out_dir, "b2_rewards_200.meta.txt"), "w", encoding="utf-8") as f:
            f.write(f"seed={training_seed} n={len(rewards)} sha256_16={h}\n")

    return b0


# ── Main training loop ───────────────────────────────────────────────────────

def train(
    scenario_name: str,
    total_steps:   int   = 500_000,
    resume_path:   str | None = None,
    training_seed: int   = TRAIN_SEED_DEFAULT,
    n_envs:        int   = N_ENVS_DEFAULT,
    device:        str   = "auto",
):
    import torch

    if device == "cpu":
        use_cuda = False
    elif device == "auto":
        use_cuda = bool(torch.cuda.is_available())
    else:
        use_cuda = "cuda" in str(device).lower()

    apply_reproducibility(training_seed, use_cuda=use_cuda)

    cfg     = get_scenario_config(scenario_name)
    out_dir = os.path.join("models", scenario_name)
    os.makedirs(out_dir, exist_ok=True)
    log_dir = os.path.join("logs", scenario_name)
    os.makedirs(log_dir, exist_ok=True)

    # b0=0: let VecNormalize (norm_reward=True) handle reward centering.
    # The cached b0.npy was 87.15 — far too high — which drowned the gradient signal.
    b0 = 0.0
    log.info("b0 = 0.0 (reward centering delegated to VecNormalize)")

    env_fns = [
        _make_env(cfg, rank, b0, training_seed) for rank in range(n_envs)
    ]
    vec_env = SubprocVecEnv(env_fns, start_method="spawn")
    vec_env = VecNormalize(
        vec_env,
        norm_obs     = True,
        norm_reward  = True,
        clip_obs     = 10.0,
        clip_reward  = 10.0,
        gamma        = PPO_KWARGS["gamma"],
    )

    try:
        import tensorboard  # noqa: F401
        tb_log = log_dir
    except ImportError:
        tb_log = None

    ppo_extra = {"seed": training_seed, "device": device}

    if resume_path and os.path.exists(resume_path):
        log.info("Resuming from %s", resume_path)
        model = PPO.load(
            resume_path,
            env=vec_env,
            **{
                k: v
                for k, v in {**PPO_KWARGS, **ppo_extra}.items()
                if k not in ("policy_kwargs", "tensorboard_log", "verbose")
            },
            tensorboard_log=tb_log,
        )
        vn_path = os.path.join(os.path.dirname(resume_path), "vec_normalize.pkl")
        if os.path.exists(vn_path):
            vec_env = VecNormalize.load(vn_path, vec_env)
            log.info("Loaded VecNormalize stats from %s", vn_path)
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            tensorboard_log=tb_log,
            **PPO_KWARGS,
            **ppo_extra,
        )

    checkpoint_cb = CheckpointCallback(
        save_freq   = max(1, 50_000 // n_envs),
        save_path   = out_dir,
        name_prefix = "ppo_ckpt",
        verbose     = 0,
    )
    eval_cb = TrafficEvalCallback(
        cfg       = cfg,
        b0        = b0,
        eval_freq = 10_000,
        seed      = training_seed,
        out_dir   = out_dir,
    )

    run_manifest(
        out_dir,
        training_seed=training_seed,
        n_envs=n_envs,
        total_steps=total_steps,
        resume_path=resume_path,
        device=str(getattr(model, "device", device)),
        extra={"scenario": scenario_name, "b0": float(b0), "b2_steps": WARMUP_STEPS},
    )

    log.info(
        "Training  scenario=%s  steps=%d  envs=%d  seed=%d  device=%s  b0=%.4f",
        scenario_name, total_steps, n_envs, training_seed, device, b0,
    )

    model.learn(
        total_timesteps        = total_steps,
        callback               = [checkpoint_cb, eval_cb],
        reset_num_timesteps    = resume_path is None,
    )

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
    p.add_argument(
        "--seed", type=int, default=TRAIN_SEED_DEFAULT,
        help="Global RL/SUMO training seed (default 42, thesis protocol)",
    )
    p.add_argument(
        "--n-envs", type=int, default=N_ENVS_DEFAULT,
        help="Number of parallel SubprocVecEnv workers (default 4)",
    )
    p.add_argument(
        "--cpu", action="store_true",
        help="Force torch on CPU (most reproducible; slower)",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    train(
        args.scenario,
        total_steps  = args.steps,
        resume_path  = args.resume,
        training_seed= args.seed,
        n_envs       = args.n_envs,
        device       = "cpu" if args.cpu else "auto",
    )
