"""
generate_anim_data.py
=====================
Generates per-step animation data: actual SUMO vehicle positions + speeds,
per-TL congestion metrics, and junction coordinates.

Each output file website/data/anim_{scenario}_s{seed}.json contains:
  • network: real junction positions (SUMO metres) + bounding box
  • vehicles_ft / vehicles_rl: per-step flat arrays [id,x,y,spd_kmh, ...]
    where id = adler32 hash of vehicle string (0–65535) for cross-step matching
  • fixed_time / ppo_best: aggregate metrics + per_tl per-step arrays
  • highlights: pre-computed significant improvement moments

Usage:
  python generate_anim_data.py              # all scenarios, all seeds
  python generate_anim_data.py --scenario main --seeds 42
"""
from __future__ import annotations
import os, json, argparse, warnings, logging, zlib
from datetime import datetime
import numpy as np

os.environ.setdefault("SUMO_HOME", "/usr/share/sumo")
warnings.filterwarnings("ignore", category=UserWarning)

import traci

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.baku_sumo_env    import BakuSUMOEnv
from envs.trimmed_env      import TrimmedTrafficEnv
from envs.scenario_configs import get_scenario_config
from baselines.fixed_time  import FixedTimeRunner

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("gen_anim")

ALL_SCENARIOS  = ["bottleneck", "main", "pedestrian", "hexagon"]
METRICS        = ["waiting_time", "queue_length", "stopped_ratio", "throughput"]
MAXIMIZE       = {"throughput"}
STEPS          = 300
WARMUP         = 30
DELTA_TIME     = 10      # sim-seconds per decision step
BBOX_PADDING   = 350     # metres of padding around junction bbox


# ── Helpers ────────────────────────────────────────────────────────────────

def veh_hash(veh_id: str) -> int:
    """Deterministic 16-bit hash of a vehicle ID string."""
    return zlib.adler32(veh_id.encode()) & 0xFFFF


def pct_imp(ft: float, rl: float, maximize: bool) -> float:
    if ft == 0.0: return 0.0
    return (rl - ft) / ft * 100 if maximize else (ft - rl) / ft * 100


def capture_network(tl_ids: list) -> dict:
    """Read junction positions from active TraCI connection.

    First tries traci.junction.getPosition(tl_id) directly — works when the
    TL ID matches the junction ID (main, hexagon scenarios).
    Falls back to averaging the endpoints of controlled lanes when the TL ID
    differs from the junction ID (bottleneck, pedestrian scenarios).
    """
    junctions = {}
    for tl in tl_ids:
        pos = None
        try:
            raw = traci.junction.getPosition(tl)
            pos = (raw[0], raw[1])
        except Exception:
            pass

        if pos is None:
            # Fallback: centroid of the stop-line ends of all controlled lanes
            try:
                xs, ys = [], []
                for ln in set(traci.trafficlight.getControlledLanes(tl)):
                    shape = traci.lane.getShape(ln)
                    if shape:                  # last point = junction end
                        xs.append(shape[-1][0])
                        ys.append(shape[-1][1])
                if xs:
                    pos = (sum(xs) / len(xs), sum(ys) / len(ys))
            except Exception:
                pass

        if pos:
            junctions[tl] = {"x": round(pos[0], 1), "y": round(pos[1], 1)}
        else:
            log.warning("Could not get position for TL %s — skipped", tl)

    xs = [j["x"] for j in junctions.values()] or [0.0]
    ys = [j["y"] for j in junctions.values()] or [0.0]
    bbox = {
        "min_x": min(xs) - BBOX_PADDING,
        "max_x": max(xs) + BBOX_PADDING,
        "min_y": min(ys) - BBOX_PADDING,
        "max_y": max(ys) + BBOX_PADDING,
    }
    return {"junctions": junctions, "bbox": bbox}


def capture_vehicles() -> list:
    """Snapshot all vehicle positions + speeds at the current simulation state.

    Returns flat list [id, x, y, spd_kmh, …] where:
      id      = 16-bit hash of SUMO vehicle ID (for cross-step interpolation)
      x, y    = integer SUMO metres (sufficient precision for visualisation)
      spd_kmh = integer km/h
    All values are compact integers to minimise JSON file size.
    """
    result = []
    for veh_id in traci.vehicle.getIDList():
        try:
            pos = traci.vehicle.getPosition(veh_id)
            spd = traci.vehicle.getSpeed(veh_id) * 3.6   # m/s → km/h
            result.extend([
                veh_hash(veh_id),
                int(round(pos[0])),
                int(round(pos[1])),
                int(round(spd)),
            ])
        except Exception:
            pass
    return result


# ── Fixed-time episode ────────────────────────────────────────────────────

def run_ft(cfg: dict, seed: int, steps: int):
    runner = FixedTimeRunner(
        sumocfg_path=cfg["sumocfg_path"],
        tl_ids=cfg["tl_ids"],
        seed=seed, label=f"anim_ft_{seed}",
    )
    runner.start(seed=seed)

    network  = capture_network(cfg["tl_ids"])
    vehicles = []
    per_tl   = {tl: {"stopped_ratio": [], "waiting_time": [],
                      "queue_length":  [], "phase":        []}
                for tl in cfg["tl_ids"]}
    agg      = {m: [] for m in METRICS}

    for t in range(steps):
        arrived = 0
        for _ in range(DELTA_TIME):
            traci.simulationStep()
            arrived += int(traci.simulation.getArrivedNumber())

        vehicles.append(capture_vehicles())

        # Per-TL metrics — read directly from TraCI (same logic as FixedTimeRunner)
        for tl in cfg["tl_ids"]:
            ratios, waits, queues = [], [], []
            for ln in runner._tl_lanes.get(tl, []):
                if not runner._is_veh_lane.get(ln, False):
                    continue
                halting = traci.lane.getLastStepHaltingNumber(ln)
                ratios.append(min(1.0, halting / runner._lane_cap[ln]))
                waits.append(traci.lane.getWaitingTime(ln))
                queues.append(float(halting))
            per_tl[tl]["stopped_ratio"].append(round(float(np.mean(ratios)) if ratios else 0.0, 4))
            per_tl[tl]["waiting_time"].append(round(float(np.mean(waits))   if waits  else 0.0, 3))
            per_tl[tl]["queue_length"].append(round(float(np.mean(queues))  if queues else 0.0, 3))
            per_tl[tl]["phase"].append(0)  # FT follows fixed SUMO programme

        if t >= WARMUP:
            m = runner._collect_step_metrics(arrived)
            for k, v in m.items():
                if k in agg:
                    agg[k].append(float(v))

        if traci.simulation.getMinExpectedNumber() <= 0:
            break

    runner.stop()
    return agg, per_tl, vehicles, network


# ── RL episode ─────────────────────────────────────────────────────────────

def run_rl(cfg: dict, seed: int, model_dir: str, steps: int):
    model_zip = os.path.join(model_dir, "ppo_best.zip")
    vn_path   = os.path.join(model_dir, "vec_normalize_best.pkl")

    base = BakuSUMOEnv(
        tl_ids=cfg["tl_ids"], sumocfg_path=cfg["sumocfg_path"],
        n_phases_per_tl=cfg["n_phases"], seed=seed, b0=0.0,
        meter_tls=cfg.get("meter_tls"), label=f"anim_rl_{seed}",
    )
    trimmed = TrimmedTrafficEnv(base)
    vec     = DummyVecEnv([lambda: trimmed])

    if os.path.exists(vn_path):
        vec = VecNormalize.load(vn_path, vec)
        vec.training = False; vec.norm_reward = False

    model = PPO.load(model_zip, env=vec)
    obs   = vec.reset()

    agg    = {m: [] for m in METRICS}
    per_tl = {tl: {"stopped_ratio": [], "waiting_time": [],
                    "queue_length":  [], "phase":        []}
              for tl in cfg["tl_ids"]}
    vehicles = []

    for t in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = vec.step(action)

        vehicles.append(capture_vehicles())

        m  = trimmed.get_metrics()
        pt = trimmed.get_per_tl_metrics()

        if t >= WARMUP:
            agg["waiting_time"].append(float(m["waiting_time"]))
            agg["queue_length"].append(float(m["queue_length"]))
            agg["stopped_ratio"].append(float(m["stopped_ratio"]))
            agg["throughput"].append(float(m["arrived"]))

        for tl in cfg["tl_ids"]:
            td = pt.get(tl, {})
            per_tl[tl]["stopped_ratio"].append(td.get("stopped_ratio", 0.0))
            per_tl[tl]["waiting_time"].append(td.get("waiting_time",  0.0))
            per_tl[tl]["queue_length"].append(td.get("queue_length",  0.0))
            per_tl[tl]["phase"].append(td.get("phase", 0))

        if bool(dones[0]):
            break

    vec.close()
    return agg, per_tl, vehicles


# ── Build highlights ────────────────────────────────────────────────────────

def build_highlights(ft_veh_steps, rl_veh_steps, ft_agg_all, rl_agg_all) -> list:
    """Pre-compute significant improvement events (step, metrics, veh counts)."""
    n = min(len(ft_agg_all["waiting_time"]) + WARMUP,
            len(rl_agg_all["waiting_time"]) + WARMUP, STEPS)

    # ft_agg_all / rl_agg_all only have post-warmup values; rebuild full arrays
    ft_wait_full = ([0.0] * WARMUP) + list(ft_agg_all["waiting_time"])
    rl_wait_full = ([0.0] * WARMUP) + list(rl_agg_all["waiting_time"])
    ft_q_full    = ([0.0] * WARMUP) + list(ft_agg_all["queue_length"])
    rl_q_full    = ([0.0] * WARMUP) + list(rl_agg_all["queue_length"])

    events, last = [], -999
    for t in range(WARMUP, min(n, len(ft_wait_full), len(rl_wait_full))):
        ft_w = ft_wait_full[t]; rl_w = rl_wait_full[t]
        if ft_w <= 0: continue
        pct = (ft_w - rl_w) / ft_w * 100
        if pct >= 35 and t - last >= 10:
            n_ft_veh = len(ft_veh_steps[t]) // 4 if t < len(ft_veh_steps) else 0
            n_rl_veh = len(rl_veh_steps[t]) // 4 if t < len(rl_veh_steps) else 0
            events.append({
                "step":        t,
                "sim_seconds": t * DELTA_TIME,
                "ft_wait":     round(ft_w, 2),
                "rl_wait":     round(rl_w, 2),
                "wait_pct":    round(pct,  1),
                "ft_queue":    round(ft_q_full[t] if t < len(ft_q_full) else 0, 2),
                "rl_queue":    round(rl_q_full[t] if t < len(rl_q_full) else 0, 2),
                "ft_vehicles": n_ft_veh,
                "rl_vehicles": n_rl_veh,
            })
            last = t
    return events[:25]


# ── Per-scenario + seed ─────────────────────────────────────────────────────

def generate(scenario: str, seed: int, out_dir: str):
    cfg       = get_scenario_config(scenario)
    model_dir = os.path.join("models", scenario)

    log.info("  %-12s seed=%-5d  FT ...", scenario, seed)
    ft_agg, ft_per_tl, ft_veh, network = run_ft(cfg, seed, STEPS)

    log.info("  %-12s seed=%-5d  RL ...", scenario, seed)
    rl_agg, rl_per_tl, rl_veh          = run_rl(cfg, seed, model_dir, STEPS)

    n_steps = min(len(ft_veh), len(rl_veh), STEPS)

    # Summary (post-warmup)
    def avg(seq): return float(np.mean(seq)) if seq else 0.0
    ft_sum = {m: avg(ft_agg[m]) for m in METRICS}
    rl_sum = {m: avg(rl_agg[m]) for m in METRICS}
    imp    = {m: pct_imp(ft_sum[m], rl_sum[m], m in MAXIMIZE) for m in METRICS}

    highlights = build_highlights(ft_veh, rl_veh, ft_agg, rl_agg)

    # Log vehicle counts for size awareness
    sample_ft = sum(len(ft_veh[t]) for t in range(min(10, len(ft_veh)))) // max(10, 1) // 4
    log.info("    avg vehicles/step (FT sample): %d  highlights: %d", sample_ft, len(highlights))

    data = {
        "scenario":     scenario,
        "seed":         seed,
        "n_steps":      n_steps,
        "warmup":       WARMUP,
        "delta_time":   DELTA_TIME,
        "tl_ids":       cfg["tl_ids"],
        "n_tls":        len(cfg["tl_ids"]),
        "demand_veh_h": cfg.get("demand_veh_h", 0),
        "network":      network,
        "fixed_time": {
            "waiting_time":  [float(v) for v in ft_agg["waiting_time"]],
            "queue_length":  [float(v) for v in ft_agg["queue_length"]],
            "stopped_ratio": [float(v) for v in ft_agg["stopped_ratio"]],
            "throughput":    [float(v) for v in ft_agg["throughput"]],
            "per_tl":        ft_per_tl,
            "vehicles":      ft_veh[:n_steps],
        },
        "ppo_best": {
            "waiting_time":  [float(v) for v in rl_agg["waiting_time"]],
            "queue_length":  [float(v) for v in rl_agg["queue_length"]],
            "stopped_ratio": [float(v) for v in rl_agg["stopped_ratio"]],
            "throughput":    [float(v) for v in rl_agg["throughput"]],
            "per_tl":        rl_per_tl,
            "vehicles":      rl_veh[:n_steps],
        },
        "summary":    {"fixed_time": ft_sum, "ppo_best": rl_sum, "improvement": imp},
        "highlights": highlights,
    }

    fname = f"anim_{scenario}_s{seed}.json"
    fpath = os.path.join(out_dir, fname)
    with open(fpath, "w") as f:
        json.dump(data, f, separators=(",", ":"))

    size_kb = os.path.getsize(fpath) / 1024
    log.info("    saved %s  (%.0f KB)", fname, size_kb)


# ── CLI ─────────────────────────────────────────────────────────────────────

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", choices=ALL_SCENARIOS + ["all"], default="all")
    p.add_argument("--seeds",   nargs="+", type=int, default=None)
    p.add_argument("--out-dir", default="website/data")
    args = p.parse_args(argv)

    scenarios = ALL_SCENARIOS if args.scenario == "all" else [args.scenario]
    os.makedirs(args.out_dir, exist_ok=True)

    for sc in scenarios:
        cfg   = get_scenario_config(sc)
        seeds = args.seeds or cfg["eval_seeds"]
        log.info("=== %s  seeds=%s ===", sc, seeds)
        for seed in seeds:
            generate(sc, seed, args.out_dir)

    log.info("Done.")


if __name__ == "__main__":
    main()
