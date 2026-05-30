"""
FixedTimeRunner
===============
Runs the SUMO fixed-time baseline WITHOUT issuing any TraCI phase-change
commands.  The simulation advances under the native TL programmes defined
in the .sumocfg (which references .tll.xml for the bottleneck, or uses
the SUMO-generated programmes for the other scenarios).

Used for:
  1. Computing the B2 baseline offset b_0 (before training).
  2. Matched-snapshot evaluation against the trained PPO policy.
"""

from __future__ import annotations
import os
import warnings
from typing import Dict, List, Optional

import numpy as np

os.environ.setdefault("SUMO_HOME", "/usr/share/sumo")
warnings.filterwarnings("ignore", category=UserWarning, module="traci")
import traci  # noqa: E402


class FixedTimeRunner:
    """
    Runs a SUMO scenario under its fixed-time TL programmes.

    Parameters
    ----------
    sumocfg_path : absolute path to the SUMO .sumocfg file
    tl_ids       : TLs to monitor (for metric aggregation)
    seed         : SUMO random seed
    label        : TraCI connection label
    min_veh_speed: speed threshold for vehicle-lane classification (m/s)
    veh_slot     : vehicle + min-gap length for capacity estimation (m)
    """

    DELTA_TIME    = 10       # seconds per evaluation step
    MIN_VEH_SPEED = 8.0
    VEH_SLOT      = 7.5

    def __init__(
        self,
        sumocfg_path: str,
        tl_ids: List[str],
        seed: int  = 42,
        label: str = "ft_runner",
    ):
        self.sumocfg_path = os.path.abspath(sumocfg_path)
        self.tl_ids       = list(tl_ids)
        self.seed         = seed
        self.label        = label

        self._tl_lanes:    Dict[str, List[str]] = {}
        self._lane_cap:    Dict[str, float]     = {}
        self._is_veh_lane: Dict[str, bool]      = {}
        self._running      = False

    # ── Lifecycle ────────────────────────────────────────────────────────

    def start(self, seed: Optional[int] = None):
        if seed is not None:
            self.seed = seed
        if self._running:
            self.stop()
        cmd = [
            "sumo",
            "-c",                   self.sumocfg_path,
            "--no-step-log",        "true",
            "--waiting-time-memory","10000",
            "--no-warnings",        "true",
            "--seed",               str(self.seed),
            "--time-to-teleport",   "-1",
        ]
        traci.start(cmd, label=self.label)
        self._running = True
        self._build_lane_index()
        traci.simulationStep()   # populate detector caches

    def stop(self):
        if self._running:
            try:
                traci.close(wait=False)
            except Exception:
                pass
            self._running = False

    # ── Episode runner ───────────────────────────────────────────────────

    def run_episode(
        self,
        max_steps: int  = 1000,
        warmup_steps: int = 0,
    ) -> Dict[str, List[float]]:
        """
        Advance the simulation for `max_steps` decision steps WITHOUT
        changing any TL phases.  Returns per-step metric lists.

        Parameters
        ----------
        max_steps    : total decision steps to run
        warmup_steps : initial steps discarded from metric collection
        """
        metrics: Dict[str, List[float]] = {
            "stopped_ratio": [],
            "waiting_time":  [],
            "queue_length":  [],
            "throughput":    [],
        }
        # per-TL buffer — populated every step (including warmup) for animation
        self._per_tl_buffer: Dict[str, Dict[str, List]] = {
            tl: {"stopped_ratio": [], "waiting_time": [], "queue_length": [], "phase": []}
            for tl in self.tl_ids
        }

        for t in range(max_steps):
            arrived_step = 0
            for _ in range(self.DELTA_TIME):
                traci.simulationStep()
                arrived_step += int(traci.simulation.getArrivedNumber())

            # Always collect per-TL (animation needs warmup steps too)
            for tl_id in self.tl_ids:
                ratios, waits, queues = [], [], []
                for ln in self._tl_lanes.get(tl_id, []):
                    if not self._is_veh_lane.get(ln, False):
                        continue
                    halting = traci.lane.getLastStepHaltingNumber(ln)
                    ratios.append(min(1.0, halting / self._lane_cap[ln]))
                    waits.append(traci.lane.getWaitingTime(ln))
                    queues.append(float(halting))
                self._per_tl_buffer[tl_id]["stopped_ratio"].append(
                    round(float(np.mean(ratios)) if ratios else 0.0, 4))
                self._per_tl_buffer[tl_id]["waiting_time"].append(
                    round(float(np.mean(waits))  if waits  else 0.0, 3))
                self._per_tl_buffer[tl_id]["queue_length"].append(
                    round(float(np.mean(queues)) if queues else 0.0, 3))
                self._per_tl_buffer[tl_id]["phase"].append(0)  # FT follows fixed SUMO programme

            if t < warmup_steps:
                continue

            m = self._collect_step_metrics(arrived_step)
            for k, v in m.items():
                metrics[k].append(v)

            if traci.simulation.getMinExpectedNumber() <= 0:
                break

        return metrics

    def run_and_aggregate(
        self,
        max_steps: int    = 1000,
        warmup_steps: int = 100,
    ) -> Dict[str, float]:
        """Run an episode and return mean values over the evaluation window."""
        raw = self.run_episode(max_steps=max_steps, warmup_steps=warmup_steps)
        return {
            k: float(np.mean(v)) if v else 0.0
            for k, v in raw.items()
        }

    # ── Metrics collection ───────────────────────────────────────────────

    def _collect_step_metrics(self, arrivals: int) -> Dict[str, float]:
        stopped_ratios, wait_times, queue_lengths = [], [], []

        for tl_id in self.tl_ids:
            for ln in self._tl_lanes.get(tl_id, []):
                if not self._is_veh_lane.get(ln, False):
                    continue
                halting = traci.lane.getLastStepHaltingNumber(ln)
                stopped_ratios.append(min(1.0, halting / self._lane_cap[ln]))
                wait_times.append(traci.lane.getWaitingTime(ln))
                queue_lengths.append(float(halting))

        return {
            "stopped_ratio": float(np.mean(stopped_ratios)) if stopped_ratios else 0.0,
            "waiting_time":  float(np.mean(wait_times))     if wait_times     else 0.0,
            "queue_length":  float(np.mean(queue_lengths))  if queue_lengths  else 0.0,
            "throughput":    float(arrivals),
        }

    def _build_lane_index(self):
        for tl_id in self.tl_ids:
            raw  = traci.trafficlight.getControlledLanes(tl_id)
            seen, lanes = set(), []
            for ln in raw:
                if ln not in seen:
                    seen.add(ln)
                    lanes.append(ln)
            self._tl_lanes[tl_id] = lanes

            for ln in lanes:
                length = traci.lane.getLength(ln)
                self._lane_cap[ln]    = max(1.0, length / self.VEH_SLOT)
                self._is_veh_lane[ln] = traci.lane.getMaxSpeed(ln) >= self.MIN_VEH_SPEED
