"""
evaluate.py
===========
Matched-snapshot evaluation of the trained PPO policy against the
fixed-time baseline on a set of independent demand seeds.

Thesis §3.7 (evaluation protocol):
  • T_eval  = 1 000 decision steps per episode
  • T_warm  = 100 steps discarded as warm-up
  • Metrics : stopped-vehicle ratio, average waiting time,
              average queue length, throughput
  • Δm      = (FT − RL) / FT × 100 %  for minimisation metrics
              (RL − FT) / FT × 100 %  for throughput

Usage
-----
  python evaluate.py --scenario main
  python evaluate.py --scenario bottleneck --seeds 42 123 456 789 1000
"""

from __future__ import annotations
import os
import sys
import argparse
import warnings
import logging
from typing import Dict, List

import numpy as np

os.environ.setdefault("SUMO_HOME", "/usr/share/sumo")
warnings.filterwarnings("ignore", category=UserWarning)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.baku_sumo_env    import BakuSUMOEnv
from envs.trimmed_env      import TrimmedTrafficEnv
from envs.scenario_configs  import get_scenario_config
from baselines.fixed_time   import FixedTimeRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Evaluation constants ─────────────────────────────────────────────────────
T_EVAL  = 1000    # total decision steps per episode
T_WARM  = 100     # warm-up steps (discarded from metric collection)
METRICS = ("stopped_ratio", "waiting_time", "queue_length", "throughput")


# ── Fixed-time baseline episode ──────────────────────────────────────────────

def run_fixed_time(cfg: dict, seed: int) -> Dict[str, float]:
    runner = FixedTimeRunner(
        sumocfg_path = cfg["sumocfg_path"],
        tl_ids       = cfg["tl_ids"],
        seed         = seed,
        label        = f"ft_{seed}",
    )
    runner.start(seed=seed)
    result = runner.run_and_aggregate(max_steps=T_EVAL, warmup_steps=T_WARM)
    runner.stop()
    return result


# ── PPO episode ──────────────────────────────────────────────────────────────

def run_ppo(cfg: dict, seed: int, model_dir: str) -> Dict[str, float]:
    model_zip = os.path.join(model_dir, "ppo_model.zip")
    if not os.path.exists(model_zip):
        raise FileNotFoundError(f"Model not found: {model_zip}")

    # Build env
    b0_path = os.path.join(model_dir, "b0.npy")
    b0      = float(np.load(b0_path)) if os.path.exists(b0_path) else 0.0

    base = BakuSUMOEnv(
        tl_ids          = cfg["tl_ids"],
        sumocfg_path    = cfg["sumocfg_path"],
        n_phases_per_tl = cfg["n_phases"],
        seed            = seed,
        b0              = b0,
        meter_tls       = cfg.get("meter_tls"),   # match training env (enables bottleneck metering action)
        label           = f"ppo_{seed}",
    )
    trimmed = TrimmedTrafficEnv(base)
    vec     = DummyVecEnv([lambda: trimmed])

    # Load VecNormalize stats (evaluation mode: no normalisation update)
    vn_path = os.path.join(model_dir, "vec_normalize.pkl")
    if os.path.exists(vn_path):
        vec = VecNormalize.load(vn_path, vec)
        vec.training     = False
        vec.norm_reward  = False
    else:
        log.warning("VecNormalize stats not found; running without normalisation")

    model = PPO.load(model_zip, env=vec)
    obs   = vec.reset()

    step_metrics: Dict[str, List[float]] = {m: [] for m in METRICS}
    step = 0

    while step < T_EVAL:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, info = vec.step(action)
        done = bool(dones[0])  # dones is np.ndarray shape (1,) from VecEnv

        if step >= T_WARM:
            m = trimmed.get_metrics()
            step_metrics["stopped_ratio"].append(m["stopped_ratio"])
            step_metrics["waiting_time"].append(m["waiting_time"])
            step_metrics["queue_length"].append(m["queue_length"])
            step_metrics["throughput"].append(m["arrived"])

        step += 1
        if done:
            break

    vec.close()
    return {k: float(np.mean(v)) if v else 0.0 for k, v in step_metrics.items()}


# ── Improvement calculation ───────────────────────────────────────────────────

def pct_improvement(ft_val: float, rl_val: float, maximise: bool) -> float:
    """
    Δm = (FT − RL) / FT × 100  for minimisation (lower = better).
    Δm = (RL − FT) / FT × 100  for throughput  (higher = better).
    """
    if ft_val == 0.0:
        return 0.0
    if maximise:
        return (rl_val - ft_val) / ft_val * 100.0
    return (ft_val - rl_val) / ft_val * 100.0


MAXIMISE = {"throughput": True}   # all other metrics: minimise


# ── Main evaluation loop ─────────────────────────────────────────────────────

def evaluate(
    scenario_name: str,
    seeds:         List[int] | None = None,
    model_dir:     str | None       = None,
    results_dir:   str              = "results",
):
    cfg       = get_scenario_config(scenario_name)
    seeds     = seeds or cfg["eval_seeds"]
    model_dir = model_dir or os.path.join("models", scenario_name)
    os.makedirs(results_dir, exist_ok=True)

    log.info("=== Evaluating  scenario=%s  seeds=%s ===", scenario_name, seeds)

    per_seed: Dict[int, Dict[str, float]] = {}

    for seed in seeds:
        log.info("Seed %d  —  running fixed-time baseline…", seed)
        ft = run_fixed_time(cfg, seed)

        log.info("Seed %d  —  running PPO policy…", seed)
        rl = run_ppo(cfg, seed, model_dir)

        improvements = {}
        for m in METRICS:
            improvements[m] = pct_improvement(
                ft[m], rl[m], maximise=MAXIMISE.get(m, False)
            )
        per_seed[seed] = improvements

        log.info(
            "Seed %4d │ stop_ratio %+.1f%%  wait_time %+.1f%%  "
            "queue %+.1f%%  throughput %+.1f%%",
            seed,
            improvements["stopped_ratio"],
            improvements["waiting_time"],
            improvements["queue_length"],
            improvements["throughput"],
        )

    # ── Aggregate ────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  {scenario_name.upper()} — PPO vs Fixed-Time: Mean % Improvement")
    print(f"{'─'*60}")
    print(f"  {'Seed':>6}  {'Stop.Ratio':>10}  {'Wait.Time':>10}  {'Queue':>8}  {'Throughput':>10}")
    print(f"{'─'*60}")

    for seed, imp in per_seed.items():
        print(
            f"  {seed:>6}  {imp['stopped_ratio']:+10.1f}  "
            f"{imp['waiting_time']:+10.1f}  {imp['queue_length']:+8.1f}  "
            f"{imp['throughput']:+10.1f}"
        )

    means = {
        m: np.mean([per_seed[s][m] for s in seeds])
        for m in METRICS
    }
    wins = {
        m: sum(1 for s in seeds if per_seed[s][m] >= 0.0)
        for m in METRICS
    }

    print(f"{'─'*60}")
    print(
        f"  {'Mean':>6}  {means['stopped_ratio']:+10.1f}  "
        f"{means['waiting_time']:+10.1f}  {means['queue_length']:+8.1f}  "
        f"{means['throughput']:+10.1f}"
    )
    print(
        f"  {'Wins':>6}  {wins['stopped_ratio']:>10}  "
        f"{wins['waiting_time']:>10}  {wins['queue_length']:>8}  "
        f"{wins['throughput']:>10}"
    )
    print(f"{'─'*60}\n")

    # ── Save ─────────────────────────────────────────────────────────────
    out_path = os.path.join(results_dir, f"{scenario_name}_eval.npz")
    np.savez(
        out_path,
        seeds    = np.array(seeds),
        **{m: np.array([per_seed[s][m] for s in seeds]) for m in METRICS},
    )
    log.info("Results saved → %s", out_path)

    return per_seed, means


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate PPO vs fixed-time on a Baku scenario")
    p.add_argument(
        "--scenario", required=True,
        choices=["bottleneck", "main", "pedestrian", "hexagon"],
    )
    p.add_argument("--seeds",   nargs="+", type=int, default=None,
                   help="Override evaluation seeds (space-separated integers)")
    p.add_argument("--model-dir", type=str, default=None,
                   help="Directory containing ppo_model.zip and vec_normalize.pkl")
    p.add_argument("--results-dir", type=str, default="results",
                   help="Output directory for .npz results file")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    evaluate(
        args.scenario,
        seeds       = args.seeds,
        model_dir   = args.model_dir,
        results_dir = args.results_dir,
    )
