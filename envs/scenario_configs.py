"""
Scenario configurations for the four Baku SUMO benchmarks.

Each config specifies:
  - scenario_dir   : subdirectory containing the SUMO files
  - sumocfg        : filename of the SUMO configuration file
  - tl_ids         : ordered list of traffic-light IDs to control
  - n_phases       : number of selectable green phases per TL
                     (derived from SUMO phase discovery; hardcoded for fast init)
  - obs_dim        : trimmed observation dimension = len(tl_ids) × 13
  - demand_veh_h   : approximate total demand in vehicles per hour
  - eval_seeds     : demand seeds used in the matched-snapshot evaluation
"""

import os

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCENARIOS = {
    # ── Bottleneck ──────────────────────────────────────────────
    # 2 TLs, linear corridor, 15 000 veh/h, 2 × 13 = 26 obs dims
    "bottleneck": {
        "scenario_dir": os.path.join(_BASE, "baku_bottleneck"),
        "sumocfg": "baku.sumocfg",
        "tl_ids": ["TL_Hasan_Aliyev", "TL_Salamzade"],
        "n_phases": [1, 1],
        "obs_dim": 26,
        "demand_veh_h": 15_000,
        "eval_seeds": [42, 123, 456, 789, 1000],
    },

    # ── Main ─────────────────────────────────────────────────────
    # 3 TLs, signalised arterial, 1 900 veh/h, 3 × 13 = 39 obs dims
    # Primary 10-seed benchmark scenario
    "main": {
        "scenario_dir": os.path.join(_BASE, "baku_main"),
        "sumocfg": "baku.sumocfg",
        "tl_ids": ["Int1", "Int2", "Int4"],
        "n_phases": [2, 2, 2],
        "obs_dim": 39,
        "demand_veh_h": 1_900,
        "eval_seeds": [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000],
    },

    # ── Pedestrian ───────────────────────────────────────────────
    # 1 TL, mixed vehicle-pedestrian junction, 1 700 veh/h, 1 × 13 = 13 obs dims
    "pedestrian": {
        "scenario_dir": os.path.join(_BASE, "baku_pedestrian"),
        "sumocfg": "baku.sumocfg",
        "tl_ids": ["Central"],
        "n_phases": [6],
        "obs_dim": 13,
        "demand_veh_h": 1_700,
        "eval_seeds": [42, 123, 456, 789, 1000],
    },

    # ── Hexagon ──────────────────────────────────────────────────
    # 12 TLs, highway sub-network, 2 000 veh/h, 12 × 13 = 156 obs dims
    # TL order matches TraCI ID ordering for this network
    "hexagon": {
        "scenario_dir": os.path.join(_BASE, "baku_hexagon_highway"),
        "sumocfg": "baku.sumocfg",
        "tl_ids": [
            "Int1", "Int2", "Int3", "Int4", "Int5", "Int6",
            "L1",   "L2",   "L3",   "L4",   "L5",   "L6",
        ],
        "n_phases": [2, 2, 3, 3, 2, 3, 1, 1, 1, 1, 1, 1],
        "obs_dim": 156,
        "demand_veh_h": 2_000,
        "eval_seeds": [42, 123, 456, 789, 1000],
    },
}


def get_scenario_config(scenario_name: str) -> dict:
    if scenario_name not in SCENARIOS:
        raise ValueError(
            f"Unknown scenario '{scenario_name}'. "
            f"Available: {list(SCENARIOS.keys())}"
        )
    cfg = SCENARIOS[scenario_name].copy()
    cfg["sumocfg_path"] = os.path.join(cfg["scenario_dir"], cfg["sumocfg"])
    return cfg
