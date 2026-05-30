"""
generate_site_data.py
=====================
Runs PPO best-checkpoint vs fixed-time baseline for all 4 scenarios and
all eval seeds. Saves per-step metric arrays + aggregate summary as JSON
under website/data/ for the static evaluation website.

Usage:
  python generate_site_data.py              # all scenarios, all seeds
  python generate_site_data.py --scenario main --seeds 42 123
  python generate_site_data.py --steps 300  # default (matches training eval window)
"""
from __future__ import annotations
import os, json, argparse, warnings, logging
from datetime import datetime
import numpy as np

os.environ.setdefault("SUMO_HOME", "/usr/share/sumo")
warnings.filterwarnings("ignore", category=UserWarning)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.baku_sumo_env    import BakuSUMOEnv
from envs.trimmed_env      import TrimmedTrafficEnv
from envs.scenario_configs import get_scenario_config
from baselines.fixed_time  import FixedTimeRunner

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("gen_site_data")

ALL_SCENARIOS = ["bottleneck", "main", "pedestrian", "hexagon"]
METRICS       = ["waiting_time", "queue_length", "stopped_ratio", "throughput"]
MAXIMIZE      = {"throughput"}
DEFAULT_STEPS  = 300
DEFAULT_WARMUP = 30

SCENARIO_META = {
    "bottleneck": {
        "title":       "Baku Bottleneck",
        "subtitle":    "Ramp-Metering at a 2-Junction Squeeze",
        "description": "Two signalised junctions (TL_Hasan_Aliyev, TL_Salamzade) "
                       "control a high-demand single-lane squeeze. The PPO agent learns "
                       "ramp-metering style control to prevent gridlock by creating vehicle "
                       "gaps into the downstream bottleneck.",
        "type":        "bottleneck",
    },
    "main": {
        "title":       "Baku Main Arterial",
        "subtitle":    "Coordinated Green-Wave on a 3-Intersection Corridor",
        "description": "Three signalised junctions (Int1, Int2, Int4) form an urban arterial "
                       "corridor. The agent learns green-wave coordination across the connected "
                       "network to minimise waiting time and prevent queue spillback.",
        "type":        "arterial",
    },
    "pedestrian": {
        "title":       "Baku Pedestrian Junction",
        "subtitle":    "Mixed Vehicle & Pedestrian Signal Control",
        "description": "A single complex junction (Central) with both vehicular lanes and "
                       "pedestrian crossings. The agent selects among 4 green phases "
                       "(2 pedestrian, 2 vehicle) to balance vehicle throughput against "
                       "pedestrian waiting pressure.",
        "type":        "pedestrian",
    },
    "hexagon": {
        "title":       "Baku Hexagon Highway",
        "subtitle":    "Multi-Agent Control of a 12-TL Highway Sub-Network",
        "description": "The largest benchmark: 12 traffic lights (Int1–6, L1–6) in a hexagonal "
                       "highway sub-network with a 156-dimensional observation space. The agent "
                       "must coordinate distributed signal control across the full network to "
                       "achieve free-flow conditions.",
        "type":        "highway",
    },
}


def pct_improvement(ft: float, rl: float, maximise: bool) -> float:
    if ft == 0.0:
        return 0.0
    return (rl - ft) / ft * 100.0 if maximise else (ft - rl) / ft * 100.0


def run_ft(cfg: dict, seed: int, steps: int) -> dict[str, list[float]]:
    runner = FixedTimeRunner(
        sumocfg_path=cfg["sumocfg_path"],
        tl_ids=cfg["tl_ids"],
        seed=seed,
        label=f"ft_gen_{seed}",
    )
    runner.start(seed=seed)
    raw = runner.run_episode(max_steps=steps, warmup_steps=0)
    runner.stop()
    return {k: [float(x) for x in v] for k, v in raw.items()}


def run_rl_best(cfg: dict, seed: int, model_dir: str, steps: int) -> dict[str, list[float]]:
    model_zip = os.path.join(model_dir, "ppo_best.zip")
    vn_path   = os.path.join(model_dir, "vec_normalize_best.pkl")

    base = BakuSUMOEnv(
        tl_ids          = cfg["tl_ids"],
        sumocfg_path    = cfg["sumocfg_path"],
        n_phases_per_tl = cfg["n_phases"],
        seed            = seed,
        b0              = 0.0,
        meter_tls       = cfg.get("meter_tls"),
        label           = f"rl_gen_{seed}",
    )
    trimmed = TrimmedTrafficEnv(base)
    vec     = DummyVecEnv([lambda: trimmed])

    if os.path.exists(vn_path):
        vec = VecNormalize.load(vn_path, vec)
        vec.training    = False
        vec.norm_reward = False

    model = PPO.load(model_zip, env=vec)
    obs   = vec.reset()

    per_step: dict[str, list[float]] = {m: [] for m in METRICS}

    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = vec.step(action)
        m = trimmed.get_metrics()
        per_step["waiting_time"].append(float(m["waiting_time"]))
        per_step["queue_length"].append(float(m["queue_length"]))
        per_step["stopped_ratio"].append(float(m["stopped_ratio"]))
        per_step["throughput"].append(float(m["arrived"]))
        if bool(dones[0]):
            break

    vec.close()
    return per_step


def agg(seq: list[float], warmup: int) -> float:
    vals = seq[warmup:]
    return float(np.mean(vals)) if vals else 0.0


def generate_seed(scenario: str, seed: int, steps: int, warmup: int) -> dict:
    cfg       = get_scenario_config(scenario)
    model_dir = os.path.join("models", scenario)

    log.info("  %-12s seed=%-5d  FT …", scenario, seed)
    ft = run_ft(cfg, seed, steps)

    log.info("  %-12s seed=%-5d  RL …", scenario, seed)
    rl = run_rl_best(cfg, seed, model_dir, steps)

    ft_agg  = {m: agg(ft[m], warmup) for m in METRICS}
    rl_agg  = {m: agg(rl[m], warmup) for m in METRICS}
    imp     = {m: pct_improvement(ft_agg[m], rl_agg[m], m in MAXIMIZE) for m in METRICS}

    n = len(ft["waiting_time"])

    return {
        "scenario":     scenario,
        "seed":         seed,
        "n_steps":      n,
        "warmup":       warmup,
        "tl_ids":       cfg["tl_ids"],
        "n_tls":        len(cfg["tl_ids"]),
        "demand_veh_h": cfg.get("demand_veh_h", 0),
        "fixed_time":   ft,
        "ppo_best":     rl,
        "summary": {
            "fixed_time": ft_agg,
            "ppo_best":   rl_agg,
            "improvement":imp,
        },
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", choices=ALL_SCENARIOS + ["all"], default="all")
    p.add_argument("--seeds",   nargs="+", type=int, default=None)
    p.add_argument("--steps",   type=int,  default=DEFAULT_STEPS)
    p.add_argument("--warmup",  type=int,  default=DEFAULT_WARMUP)
    p.add_argument("--out-dir", default="website/data")
    args = p.parse_args(argv)

    scenarios = ALL_SCENARIOS if args.scenario == "all" else [args.scenario]
    os.makedirs(args.out_dir, exist_ok=True)

    manifest: dict = {
        "generated": datetime.utcnow().strftime("%Y-%m-%d"),
        "steps":     args.steps,
        "warmup":    args.warmup,
        "scenarios": {},
    }

    for sc in scenarios:
        cfg   = get_scenario_config(sc)
        seeds = args.seeds or cfg["eval_seeds"]

        meta  = SCENARIO_META[sc].copy()
        meta.update({
            "seeds":        seeds,
            "tl_ids":       cfg["tl_ids"],
            "n_tls":        len(cfg["tl_ids"]),
            "demand_veh_h": cfg.get("demand_veh_h", 0),
        })

        log.info("=== %s  seeds=%s ===", sc, seeds)
        sc_summaries: list[dict] = []

        for seed in seeds:
            data  = generate_seed(sc, seed, args.steps, args.warmup)
            fname = f"{sc}_s{seed}.json"
            path  = os.path.join(args.out_dir, fname)
            with open(path, "w") as f:
                json.dump(data, f, separators=(",", ":"))
            log.info("    saved → %s", path)
            sc_summaries.append({
                "seed":        seed,
                "improvement": data["summary"]["improvement"],
                "fixed_time":  data["summary"]["fixed_time"],
                "ppo_best":    data["summary"]["ppo_best"],
            })

        mean_imp = {
            m: float(np.mean([s["improvement"][m] for s in sc_summaries]))
            for m in METRICS
        }
        meta["mean_improvement"] = mean_imp
        meta["seed_summaries"]   = sc_summaries
        manifest["scenarios"][sc] = meta

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("Manifest → %s", manifest_path)
    log.info("Done.")


if __name__ == "__main__":
    main()
