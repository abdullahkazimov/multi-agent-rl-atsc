"""
analyze_reward.py
=================
Per-σ gradient analysis of the composite reward function (thesis §3.4.1).

For each reward term f_j, computes:
  g_j = β_j · σ(f_j)

where σ(f_j) is the empirical standard deviation of the term under fixed-time
operation.  These gradient magnitudes represent the expected reward change per
one-standard-deviation change in the corresponding traffic metric.

The analysis was used to detect the original throughput-gradient imbalance
(g_tp was 13× smaller than g_wt) and to re-calibrate the weights to their
final values in the thesis.

Usage
-----
  python analyze_reward.py --scenario main --steps 300
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

import traci  # noqa: E402

from envs.baku_sumo_env    import BakuSUMOEnv
from envs.scenario_configs  import get_scenario_config
from rewards.reward_fn      import RewardFunction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def collect_fixed_time_samples(
    cfg:    dict,
    seed:   int = 42,
    n_steps: int = 300,
) -> Dict[str, List[float]]:
    """
    Run the fixed-time simulation and record the raw value of each
    reward-function input term at every decision step.
    """
    label = "analysis_env"
    cmd = [
        "sumo",
        "-c",                    cfg["sumocfg_path"],
        "--no-step-log",         "true",
        "--waiting-time-memory", "10000",
        "--no-warnings",         "true",
        "--seed",                str(seed),
        "--time-to-teleport",    "-1",
    ]
    traci.start(cmd, label=label)
    traci.simulationStep()   # warm up detector caches

    # Build lane index
    tl_ids   = cfg["tl_ids"]
    tl_lanes: Dict[str, List[str]] = {}
    lane_cap:  Dict[str, float]    = {}
    is_veh:    Dict[str, bool]     = {}
    VEH_SLOT  = BakuSUMOEnv.VEH_SLOT
    MIN_SPEED = BakuSUMOEnv.MIN_VEH_SPEED

    for tl_id in tl_ids:
        raw = traci.trafficlight.getControlledLanes(tl_id)
        seen, lanes = set(), []
        for ln in raw:
            if ln not in seen:
                seen.add(ln)
                lanes.append(ln)
        tl_lanes[tl_id] = lanes
        for ln in lanes:
            lane_cap[ln] = max(1.0, traci.lane.getLength(ln) / VEH_SLOT)
            is_veh[ln]   = traci.lane.getMaxSpeed(ln) >= MIN_SPEED

    samples: Dict[str, List[float]] = {
        "throughput":    [],
        "stopped_ratio": [],
        "waiting_time":  [],
        "queue_length":  [],
        "hotspot":       [],
        "coord_balance": [],
    }

    DELTA_TIME = BakuSUMOEnv.DELTA_TIME
    for _ in range(n_steps):
        arrived = 0
        for __ in range(DELTA_TIME):
            traci.simulationStep()
            arrived += int(traci.simulation.getArrivedNumber())
        sr_all, wt_all, ql_all, tl_sr = [], [], [], {tl: [] for tl in tl_ids}

        for tl_id in tl_ids:
            for ln in tl_lanes[tl_id]:
                if not is_veh[ln]:
                    continue
                halting = traci.lane.getLastStepHaltingNumber(ln)
                ratio   = min(1.0, halting / lane_cap[ln])
                sr_all.append(ratio)
                tl_sr[tl_id].append(ratio)
                wt_all.append(traci.lane.getWaitingTime(ln))
                ql_all.append(float(halting))

        if not sr_all:
            continue

        mean_sr = float(np.mean(sr_all))
        max_sr  = max(
            (float(np.mean(v)) if v else 0.0) for v in tl_sr.values()
        )

        samples["throughput"].append(float(arrived))
        samples["stopped_ratio"].append(mean_sr)
        samples["waiting_time"].append(float(np.mean(wt_all)) / BakuSUMOEnv.D_W)
        samples["queue_length"].append(float(np.mean(ql_all)) / BakuSUMOEnv.D_Q)
        samples["hotspot"].append(max_sr)
        samples["coord_balance"].append((max_sr - mean_sr) ** 2)

        if traci.simulation.getMinExpectedNumber() <= 0:
            break

    traci.close(wait=False)
    return samples


def analyze_reward(
    scenario_name: str,
    seed:          int = 42,
    n_steps:       int = 300,
):
    cfg = get_scenario_config(scenario_name)

    log.info("Collecting %d fixed-time samples for scenario '%s'…", n_steps, scenario_name)
    samples = collect_fixed_time_samples(cfg, seed=seed, n_steps=n_steps)

    sigmas = {k: float(np.std(v)) if v else 1e-9 for k, v in samples.items()}
    means  = {k: float(np.mean(v)) if v else 0.0 for k, v in samples.items()}

    rf = RewardFunction()
    gradients = rf.per_sigma_gradients(sigmas)

    print(f"\n{'═'*62}")
    print(f"  Per-σ Gradient Analysis  —  scenario: {scenario_name}")
    print(f"{'═'*62}")
    print(f"  {'Term':<18}  {'β':>8}  {'σ(f)':>10}  {'mean(f)':>10}  {'|g|=β·σ':>10}")
    print(f"{'─'*62}")

    betas = {
        "throughput":    rf.beta_tp,
        "stopped_ratio": rf.beta_sr,
        "waiting_time":  rf.beta_wt,
        "queue_length":  rf.beta_ql,
        "hotspot":       rf.beta_lc,
        "coord_balance": rf.beta_cc,
    }
    for term in ("throughput","stopped_ratio","waiting_time","queue_length","hotspot","coord_balance"):
        beta  = betas[term]
        sigma = sigmas[term]
        mean  = means[term]
        g     = abs(gradients[term])
        print(f"  {term:<18}  {beta:8.3f}  {sigma:10.5f}  {mean:10.5f}  {g:10.5f}")

    print(f"{'─'*62}")
    total_g = sum(abs(v) for v in gradients.values())
    print(f"  Total |g|: {total_g:.5f}")
    print(f"{'═'*62}\n")

    # Imbalance detection
    g_vals = {k: abs(v) for k, v in gradients.items()}
    max_term = max(g_vals, key=g_vals.get)
    min_term = min(g_vals, key=g_vals.get)
    ratio = g_vals[max_term] / max(g_vals[min_term], 1e-9)
    if ratio > 5:
        log.warning(
            "Gradient imbalance detected: '%s' (|g|=%.4f) is %.1f× larger "
            "than '%s' (|g|=%.4f). Consider rebalancing weights.",
            max_term, g_vals[max_term], ratio, min_term, g_vals[min_term],
        )
    else:
        log.info("Gradient balance OK (max/min ratio = %.2f)", ratio)

    return sigmas, gradients


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Reward gradient analysis")
    p.add_argument("--scenario", required=True,
                   choices=["bottleneck","main","pedestrian","hexagon"])
    p.add_argument("--seed",  type=int, default=42)
    p.add_argument("--steps", type=int, default=300)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    analyze_reward(args.scenario, seed=args.seed, n_steps=args.steps)
