"""
Scenario configurations for the four Baku SUMO benchmarks.

Each config specifies:
  - scenario_dir   : subdirectory containing the SUMO files
  - sumocfg        : filename of the SUMO configuration file
  - tl_ids         : ordered list of traffic-light IDs to control
  - n_phases       : number of selectable green phases per TL
                     (derived from SUMO phase discovery; hardcoded for fast init)
  - obs_dim        : trimmed observation dimension = len(tl_ids) × 14
                     (14 = 10 lane ratios + phase + steps + starvation + ped_pressure)
  - demand_veh_h   : approximate total demand in vehicles per hour
  - eval_seeds     : demand seeds used in the matched-snapshot evaluation
"""

import os

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCENARIOS = {
    # ── Bottleneck ──────────────────────────────────────────────
    # 2 TLs, single-lane squeeze, metered. 3 000 veh/h. 2 × 13 = 26 obs dims
    # Both junctions are single-phase; metering (green/all-red, declared via
    # meter_tls) gives the agent a 2nd action to hold traffic and create gaps
    # into the squeeze (ramp-metering style), turning a no-action degenerate
    # scenario into a real control problem. See PLAN.md R2.
    "bottleneck": {
        "scenario_dir": os.path.join(_BASE, "scenarios", "baku_bottleneck"),
        "sumocfg": "baku.sumocfg",
        "tl_ids": ["TL_Hasan_Aliyev", "TL_Salamzade"],
        "n_phases": [2, 2],                                  # green + all-red (metering)
        "meter_tls": ["TL_Hasan_Aliyev", "TL_Salamzade"],
        "obs_dim": 26,
        "demand_veh_h": 3_000,   # recalibrated 2026-05-29 (was 15_000 ≈ 8x squeeze cap)
        "eval_seeds": [42, 123, 456, 789, 1000],
    },

    # ── Main ─────────────────────────────────────────────────────
    # 3 TLs, signalised arterial, 1 900 veh/h, 3 × 13 = 39 obs dims
    # Primary 10-seed benchmark scenario
    "main": {
        "scenario_dir": os.path.join(_BASE, "scenarios", "baku_main"),
        "sumocfg": "baku.sumocfg",
        "tl_ids": ["Int1", "Int2", "Int4"],
        "n_phases": [2, 2, 2],
        "obs_dim": 39,
        "demand_veh_h": 1_330,   # recalibrated 2026-05-29 (was 1_900; oversaturated, FT~343s wait)
        "eval_seeds": [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000],
    },

    # ── Pedestrian ───────────────────────────────────────────────
    # 1 TL, mixed vehicle+pedestrian junction. Rebuilt 2026-05-29 (PLAN.md R3)
    # with real <person> flows, signalised crossings and walkingareas. The
    # Central program has 4 selectable green phases: 2 serve pedestrian
    # crossings (idx 0,2), 2 are vehicle-only (idx 1,3). 1 × 14 = 14 obs dims.
    "pedestrian": {
        "scenario_dir": os.path.join(_BASE, "scenarios", "baku_pedestrian"),
        "sumocfg": "baku.sumocfg",
        "tl_ids": ["Central"],
        "n_phases": [4],
        "obs_dim": 14,
        "demand_veh_h": 1_000,   # ~1000 veh/h cars + ~700 persons/h
        "eval_seeds": [42, 123, 456, 789, 1000],
    },

    # ── Hexagon ──────────────────────────────────────────────────
    # 12 TLs, highway sub-network, 2 000 veh/h, 12 × 13 = 156 obs dims
    # TL order matches TraCI ID ordering for this network
    "hexagon": {
        "scenario_dir": os.path.join(_BASE, "scenarios", "baku_hexagon_highway"),
        "sumocfg": "baku.sumocfg",
        "tl_ids": [
            "Int1", "Int2", "Int3", "Int4", "Int5", "Int6",
            "L1",   "L2",   "L3",   "L4",   "L5",   "L6",
        ],
        "n_phases": [2, 2, 3, 3, 2, 3, 1, 1, 1, 1, 1, 1],
        "obs_dim": 156,
        "demand_veh_h": 2_600,   # recalibrated 2026-05-29 (was 2_000; under-loaded)
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
