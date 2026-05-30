"""
evaluate_stepwise.py
====================
Step-by-step evaluation of a trained PPO policy against the fixed-time
baseline, on a single demand seed, with a live per-decision-step readout
and a final performance-metric comparison.

Why a second eval script (vs. evaluate.py)?
  • evaluate.py aggregates over many seeds into one .npz row. This script
    instead replays ONE run and prints what happens at each decision step
    (FT vs RL side by side), which is what you watch when demoing a model.
  • It rebuilds the RL environment EXACTLY as train.py did — in particular
    it passes `meter_tls`, which evaluate.py omitted (that silently disabled
    the bottleneck agent's metering action at eval time).

Fixed-time config == training config
  The fixed-time baseline is produced by the SAME `FixedTimeRunner` over the
  SAME `.sumocfg` / `.net.xml` / `.rou.xml` files and the SAME seed (default
  42, the training seed). Those scenario files on disk are the ones training
  read (verified: each scenario's baku.rou.xml mtime precedes its trained
  ppo_model.zip mtime). Nothing about the FT programme is overridden here.

Protocol (matches evaluate.py / thesis §3.7)
  • T_eval  = 1000 decision steps (10 sim-seconds each)   [--steps]
  • T_warm  = 100  steps discarded from the aggregate      [--warmup]
  • Metrics : waiting time, queue length, stopped ratio, throughput
  • Δm      = (FT − RL)/FT × 100 %   for minimisation metrics
              (RL − FT)/FT × 100 %   for throughput (higher = better)

Usage
-----
  python evaluate_stepwise.py --scenario main
  python evaluate_stepwise.py --scenario bottleneck --seed 123 --print-every 10
  python evaluate_stepwise.py --all                       # all 4 scenarios
  python evaluate_stepwise.py --scenario hexagon --best   # use ppo_best.zip
  python evaluate_stepwise.py --scenario pedestrian --gui # watch RL in sumo-gui
"""

from __future__ import annotations
import os
import argparse
import warnings
import logging
from typing import Dict, List, Optional

import numpy as np

os.environ.setdefault("SUMO_HOME", "/usr/share/sumo")
warnings.filterwarnings("ignore", category=UserWarning)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.baku_sumo_env    import BakuSUMOEnv
from envs.trimmed_env      import TrimmedTrafficEnv
from envs.scenario_configs import get_scenario_config
from baselines.fixed_time  import FixedTimeRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("evaluate_stepwise")

# ── Evaluation constants (thesis §3.7) ───────────────────────────────────────
T_EVAL = 1000
T_WARM = 100
TRAIN_SEED = 42                       # the seed train.py used for its FT baseline

# metric key → (header label, unit, higher_is_better)
METRICS = {
    "waiting_time":  ("Wait(s)",  "s",     False),
    "queue_length":  ("Queue",    "veh",   False),
    "stopped_ratio": ("Stop.R",   "",      False),
    "throughput":    ("Thrpt",    "veh/st", True),
}
ALL_SCENARIOS = ["bottleneck", "main", "pedestrian", "hexagon"]


def pct_improvement(ft: float, rl: float, maximise: bool) -> float:
    """Δm per thesis: positive = RL better than fixed-time."""
    if ft == 0.0:
        return 0.0
    return (rl - ft) / ft * 100.0 if maximise else (ft - rl) / ft * 100.0


# ── Fixed-time episode (identical setup to training) ─────────────────────────

def run_fixed_time_stepwise(cfg: dict, seed: int, max_steps: int) -> Dict[str, List[float]]:
    """Replay the native fixed-time programme, returning per-step metric lists.

    Uses FixedTimeRunner with warmup_steps=0 so we keep every step for the
    live table; the warm-up window is dropped later, at aggregation time.
    """
    runner = FixedTimeRunner(
        sumocfg_path = cfg["sumocfg_path"],
        tl_ids       = cfg["tl_ids"],
        seed         = seed,
        label        = f"ft_step_{seed}",
    )
    runner.start(seed=seed)
    raw = runner.run_episode(max_steps=max_steps, warmup_steps=0)
    runner.stop()
    return raw   # keys: stopped_ratio, waiting_time, queue_length, throughput


# ── PPO episode (env rebuilt exactly as in train.py) ─────────────────────────

def run_ppo_stepwise(
    cfg:        dict,
    seed:       int,
    model_dir:  str,
    max_steps:  int,
    warmup:     int,
    ft:         Dict[str, List[float]],
    use_best:   bool = False,
    gui:        bool = False,
    print_every: int = 25,
) -> Dict[str, List[float]]:
    """Run the trained policy deterministically, printing a live FT-vs-RL row
    every `print_every` steps, and return per-step metric lists."""
    model_zip = os.path.join(model_dir, "ppo_best.zip" if use_best else "ppo_model.zip")
    vn_name   = "vec_normalize_best.pkl" if use_best else "vec_normalize.pkl"
    if not os.path.exists(model_zip):
        raise FileNotFoundError(f"Model not found: {model_zip}")

    # b0 only enters the reward, which evaluation never reads; train.py used
    # b0=0.0 (VecNormalize handled centering), so we mirror that exactly.
    base = BakuSUMOEnv(
        tl_ids          = cfg["tl_ids"],
        sumocfg_path    = cfg["sumocfg_path"],
        n_phases_per_tl = cfg["n_phases"],
        seed            = seed,
        b0              = 0.0,
        meter_tls       = cfg.get("meter_tls"),   # ← matches training (evaluate.py omitted this)
        use_gui         = gui,
        label           = f"ppo_step_{seed}",
    )
    trimmed = TrimmedTrafficEnv(base)
    vec     = DummyVecEnv([lambda: trimmed])

    vn_path = os.path.join(model_dir, vn_name)
    if os.path.exists(vn_path):
        vec = VecNormalize.load(vn_path, vec)
        vec.training    = False     # freeze running stats (evaluation)
        vec.norm_reward = False
    else:
        log.warning("VecNormalize stats (%s) not found; running un-normalised", vn_name)

    model = PPO.load(model_zip, env=vec)
    obs   = vec.reset()

    per_step: Dict[str, List[float]] = {m: [] for m in METRICS}

    _print_table_header(warmup)
    step = 0
    while step < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = vec.step(action)
        m = trimmed.get_metrics()
        row = {
            "waiting_time":  m["waiting_time"],
            "queue_length":  m["queue_length"],
            "stopped_ratio": m["stopped_ratio"],
            "throughput":    m["arrived"],
        }
        for k in METRICS:
            per_step[k].append(row[k])

        if step % print_every == 0 or step == warmup:
            _print_step_row(step, ft, row, is_warmup=step < warmup)

        step += 1
        if bool(dones[0]):
            log.info("RL episode ended early at step %d (network drained / gridlock guard)", step)
            break

    vec.close()
    return per_step


# ── Live table rendering ─────────────────────────────────────────────────────

def _print_table_header(warmup: int) -> None:
    print()
    print(f"  Live per-decision-step comparison  (warm-up = first {warmup} steps, dropped from the aggregate)")
    print(f"  {'Step':>5} │ {'FT wait':>8} {'RL wait':>8} │ {'FT queue':>8} {'RL queue':>8} "
          f"│ {'FT stop':>7} {'RL stop':>7} │ {'FT thr':>6} {'RL thr':>6}")
    print(f"  {'-'*5}┼{'-'*19}┼{'-'*19}┼{'-'*17}┼{'-'*15}", flush=True)


def _fmt(x: Optional[float], width: int, prec: int) -> str:
    return ("{:>%d}" % width).format("—") if x is None else ("{:>%d.%df}" % (width, prec)).format(x)


def _ft_at(ft: Dict[str, List[float]], key: str, idx: int) -> Optional[float]:
    seq = ft.get(key, [])
    return seq[idx] if idx < len(seq) else None


def _print_step_row(step: int, ft: Dict[str, List[float]], rl: dict, is_warmup: bool) -> None:
    tag = " (warm-up)" if is_warmup else ""
    print(
        f"  {step:>5} │ "
        f"{_fmt(_ft_at(ft,'waiting_time',step),8,2)} {rl['waiting_time']:>8.2f} │ "
        f"{_fmt(_ft_at(ft,'queue_length',step),8,2)} {rl['queue_length']:>8.2f} │ "
        f"{_fmt(_ft_at(ft,'stopped_ratio',step),7,3)} {rl['stopped_ratio']:>7.3f} │ "
        f"{_fmt(_ft_at(ft,'throughput',step),6,1)} {rl['throughput']:>6.1f}{tag}",
        flush=True,
    )


# ── Aggregate & report ───────────────────────────────────────────────────────

def _mean_after_warmup(seq: List[float], warmup: int) -> float:
    vals = seq[warmup:]
    return float(np.mean(vals)) if vals else 0.0


def report_scenario(
    scenario: str,
    ft:       Dict[str, List[float]],
    rl:       Dict[str, List[float]],
    seed:     int,
    warmup:   int,
) -> Dict[str, float]:
    n_ft, n_rl = len(ft["waiting_time"]), len(rl["waiting_time"])
    n_eval = max(0, min(n_ft, n_rl) - warmup)

    print()
    print("=" * 70)
    print(f"  {scenario.upper()} — PPO vs Fixed-Time   "
          f"seed={seed}   eval window = steps {warmup}–{min(n_ft, n_rl)}  ({n_eval} steps)")
    if n_ft != n_rl:
        print(f"  (episode lengths differ: FT={n_ft} steps, RL={n_rl} steps — aggregated over the shared window)")
    print("-" * 70)
    print(f"  {'Metric':<18}{'Fixed-Time':>13}{'PPO':>13}{'Improvement':>16}")
    print("-" * 70)

    improvements: Dict[str, float] = {}
    for key, (label, unit, maximise) in METRICS.items():
        fa = _mean_after_warmup(ft[key], warmup)
        ra = _mean_after_warmup(rl[key], warmup)
        imp = pct_improvement(fa, ra, maximise)
        improvements[key] = imp
        mark = "✓" if imp >= 0 else "✗"
        name = f"{label} ({unit})" if unit else label
        print(f"  {name:<18}{fa:>13.3f}{ra:>13.3f}{imp:>+14.1f}%  {mark}")

    print("-" * 70)
    wins = sum(1 for v in improvements.values() if v >= 0)
    print(f"  PPO wins on {wins}/{len(METRICS)} metrics   "
          f"(primary metric = waiting time: {improvements['waiting_time']:+.1f}%)")
    print("=" * 70)
    return improvements


# ── Per-scenario driver ──────────────────────────────────────────────────────

def evaluate_scenario(
    scenario:    str,
    seed:        int,
    max_steps:   int,
    warmup:      int,
    use_best:    bool,
    gui:         bool,
    print_every: int,
) -> Dict[str, float]:
    cfg       = get_scenario_config(scenario)
    model_dir = os.path.join("models", scenario)

    print("\n" + "#" * 70)
    print(f"#  SCENARIO: {scenario}   ({len(cfg['tl_ids'])} TL(s): {', '.join(cfg['tl_ids'])})")
    print(f"#  sumocfg : {cfg['sumocfg_path']}")
    print(f"#  model   : {os.path.join(model_dir, 'ppo_best.zip' if use_best else 'ppo_model.zip')}")
    print("#" * 70)

    log.info("Fixed-time baseline — same FixedTimeRunner / sumocfg / seed=%d as training…", seed)
    ft = run_fixed_time_stepwise(cfg, seed, max_steps)
    log.info("Fixed-time done (%d decision steps). Now replaying the PPO policy…", len(ft["waiting_time"]))

    rl = run_ppo_stepwise(
        cfg, seed, model_dir, max_steps, warmup, ft,
        use_best=use_best, gui=gui, print_every=print_every,
    )

    return report_scenario(scenario, ft, rl, seed, warmup)


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Step-by-step evaluation of a trained PPO policy vs the fixed-time baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--scenario", choices=ALL_SCENARIOS, help="Single scenario to evaluate")
    g.add_argument("--all", action="store_true", help="Evaluate all four scenarios in turn")
    p.add_argument("--seed", type=int, default=TRAIN_SEED,
                   help="Demand seed (FT and RL share it). Default = training seed")
    p.add_argument("--steps", type=int, default=T_EVAL, help="Decision steps per episode")
    p.add_argument("--warmup", type=int, default=T_WARM, help="Warm-up steps dropped from the aggregate")
    p.add_argument("--print-every", type=int, default=25, help="Print a live row every N steps")
    p.add_argument("--best", action="store_true",
                   help="Use ppo_best.zip + vec_normalize_best.pkl instead of the final model")
    p.add_argument("--gui", action="store_true",
                   help="Launch sumo-gui for the RL run (needs a display; FT still runs headless)")
    return p.parse_args(argv)


def main(argv=None):
    args      = _parse_args(argv)
    scenarios = ALL_SCENARIOS if args.all else [args.scenario]

    summary: Dict[str, Dict[str, float]] = {}
    for sc in scenarios:
        summary[sc] = evaluate_scenario(
            sc, args.seed, args.steps, args.warmup,
            use_best=args.best, gui=args.gui, print_every=args.print_every,
        )

    if len(scenarios) > 1:
        print("\n" + "=" * 70)
        print(f"  SUMMARY — mean % improvement of PPO over fixed-time   (seed={args.seed})")
        print("-" * 70)
        print(f"  {'Scenario':<14}{'Wait%':>11}{'Queue%':>11}{'Stop%':>11}{'Thrpt%':>11}")
        print("-" * 70)
        for sc in scenarios:
            imp = summary[sc]
            print(f"  {sc:<14}{imp['waiting_time']:>+10.1f} {imp['queue_length']:>+10.1f} "
                  f"{imp['stopped_ratio']:>+10.1f} {imp['throughput']:>+10.1f}")
        print("=" * 70)


if __name__ == "__main__":
    main()
