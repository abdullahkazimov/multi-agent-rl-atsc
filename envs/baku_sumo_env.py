"""
BakuSUMOEnv
===========
Gymnasium-compatible environment wrapping SUMO via TraCI for the four
Baku adaptive traffic signal control benchmarks.

Design (thesis §3.2 – §3.4):
  • Decision interval : 10 simulated seconds
  • Yellow clearance  : 3 simulated seconds (inserted on phase switch)
  • Min phase hold    : 3 decision steps (≥ 30 s)
  • Observation       : 12-slot × 13-value flat vector (full, 156-dim)
  • Action            : MultiDiscrete — one green-phase index per TL
  • Reward            : 7-term composite (thesis Eq. 1)
"""

import os
import warnings
import numpy as np
import gymnasium as gym
from gymnasium import spaces

os.environ.setdefault("SUMO_HOME", "/usr/share/sumo")
warnings.filterwarnings("ignore", category=UserWarning, module="traci")

import traci  # noqa: E402


class BakuSUMOEnv(gym.Env):
    metadata = {"render_modes": []}

    # ── Simulation constants ─────────────────────────────────────────────
    DELTA_TIME          = 10    # sim-seconds per RL decision step
    YELLOW_TIME         = 3     # sim-seconds for yellow clearance
    MIN_GREEN_STEPS     = 3     # decision steps before a phase switch is allowed

    # ── Observation layout ───────────────────────────────────────────────
    MAX_LANES           = 10    # lane ratio slots per TL
    OBS_PER_TL          = 13    # lane ratios (10) + phase_norm + steps_norm + starvation_norm
    MAX_TLS             = 12    # full observation covers 12 TL slots
    OBS_DIM             = MAX_TLS * OBS_PER_TL          # 156
    STARVATION_MAX_STEPS = 15   # phases unserved beyond this are penalised

    # ── Reward weights (thesis Eq. 1) ────────────────────────────────────
    BETA_TP = 0.15   # throughput increment
    BETA_SR = 0.30   # mean stopped-vehicle ratio
    BETA_WT = 0.55   # mean waiting time (primary target)
    BETA_QL = 0.12   # mean queue length
    BETA_LC = 0.25   # hotspot — worst-TL stopped ratio
    BETA_CC = 0.10   # MARL coordination balance penalty
    BETA_ST = 0.08   # phase starvation score
    BETA_SW = 0.05   # switching cost (phase chatter prevention)
    D_W     = 35.0   # waiting-time normaliser (seconds)
    D_Q     = 10.0   # queue-length normaliser (vehicles)

    # ── Lane classification ──────────────────────────────────────────────
    MIN_VEH_SPEED = 8.0   # m/s — lanes below this are pedestrian/slow lanes
    VEH_SLOT      = 7.5   # m   — vehicle + min-gap slot length (5.0 + 2.5)

    def __init__(
        self,
        tl_ids: list,
        sumocfg_path: str,
        n_phases_per_tl: list,
        seed: int       = 42,
        use_gui: bool   = False,
        b0: float       = 0.0,
        max_steps: int  = 1500,
        label: str      = "default",
    ):
        """
        Parameters
        ----------
        tl_ids          : ordered list of TL IDs to control
        sumocfg_path    : absolute path to the .sumocfg file
        n_phases_per_tl : number of selectable green phases per TL
                          (used to initialise action_space before SUMO starts)
        seed            : SUMO random seed (affects vehicle departure jitter)
        use_gui         : launch sumo-gui instead of headless sumo
        b0              : B2 baseline offset – negative mean FT reward per step
        max_steps       : episode truncation limit (decision steps)
        label           : TraCI connection label (must be unique per process)
        """
        super().__init__()

        self.tl_ids          = list(tl_ids)
        self.n_tls           = len(tl_ids)
        self.sumocfg_path    = os.path.abspath(sumocfg_path)
        self.n_phases_per_tl = list(n_phases_per_tl)
        self.seed            = seed
        self.use_gui         = use_gui
        self.b0              = b0
        self.max_steps       = max_steps
        self.label           = label

        # ── Spaces ──────────────────────────────────────────────────────
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.OBS_DIM,),
            dtype=np.float32,
        )
        n_phases_padded = list(n_phases_per_tl) + [1] * (self.MAX_TLS - self.n_tls)
        self.action_space = spaces.MultiDiscrete(n_phases_padded)

        # ── Runtime state (populated in _start_sumo) ────────────────────
        self._green_sumo_indices: dict = {}   # tl_id → [sumo_phase_idx, …]
        self._green_states:       dict = {}   # tl_id → [green_state_str, …]
        self._yellow_states:      dict = {}   # tl_id → [yellow_state_str, …]
        self._tl_lanes:           dict = {}   # tl_id → [lane_id, …]
        self._lane_cap:           dict = {}   # lane_id → float
        self._is_veh_lane:        dict = {}   # lane_id → bool

        self._cur_green_idx:    dict = {}   # tl_id → int
        self._steps_in_phase:   dict = {}   # tl_id → int
        self._steps_since_sw:   dict = {}   # tl_id → int
        self._phase_age:        dict = {}   # tl_id → np.ndarray

        self._step_count    = 0
        self._sumo_running  = False

    # ════════════════════════════════════════════════════════════════════
    #  Gymnasium API
    # ════════════════════════════════════════════════════════════════════

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed = seed
        self._stop_sumo()
        self._start_sumo()
        self._step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        self._step_count += 1

        # ── Determine phase switches ─────────────────────────────────
        switch_map: dict = {}   # tl_id → new_green_idx
        n_switches = 0
        for i, tl_id in enumerate(self.tl_ids):
            n_green   = len(self._green_sumo_indices[tl_id])
            requested = int(action[i]) % max(n_green, 1)
            can_sw    = self._steps_since_sw[tl_id] >= self.MIN_GREEN_STEPS
            if requested != self._cur_green_idx[tl_id] and can_sw and n_green > 1:
                switch_map[tl_id] = requested
                n_switches += 1
                # Immediately set yellow transition
                yw = self._yellow_states[tl_id][self._cur_green_idx[tl_id]]
                traci.trafficlight.setRedYellowGreenState(tl_id, yw)

        # ── Advance DELTA_TIME simulation steps ──────────────────────
        for t in range(self.DELTA_TIME):
            if t == self.YELLOW_TIME:
                for tl_id, new_idx in switch_map.items():
                    green_state = self._green_states[tl_id][new_idx]
                    traci.trafficlight.setRedYellowGreenState(tl_id, green_state)
            traci.simulationStep()

        # ── Collect metrics and build output ─────────────────────────
        arrived = traci.simulation.getArrivedNumber()
        obs     = self._get_obs()
        reward  = self._compute_reward(arrived, n_switches)

        # ── Update tracking state ────────────────────────────────────
        for tl_id in self.tl_ids:
            if tl_id in switch_map:
                self._cur_green_idx[tl_id]   = switch_map[tl_id]
                self._steps_in_phase[tl_id]  = 0
                self._steps_since_sw[tl_id]  = 0
            else:
                self._steps_in_phase[tl_id] += 1
                self._steps_since_sw[tl_id] += 1

            n_green = len(self._green_sumo_indices[tl_id])
            for j in range(n_green):
                if j == self._cur_green_idx[tl_id]:
                    self._phase_age[tl_id][j] = 0
                else:
                    self._phase_age[tl_id][j] += 1

        done = (
            traci.simulation.getMinExpectedNumber() <= 0
            or self._step_count >= self.max_steps
        )
        return obs, reward, done, False, {}

    def close(self):
        self._stop_sumo()

    def render(self):
        pass

    # ════════════════════════════════════════════════════════════════════
    #  SUMO lifecycle
    # ════════════════════════════════════════════════════════════════════

    def _start_sumo(self):
        binary = "sumo-gui" if self.use_gui else "sumo"
        cmd = [
            binary,
            "-c",                   self.sumocfg_path,
            "--no-step-log",        "true",
            "--waiting-time-memory","10000",
            "--no-warnings",        "true",
            "--seed",               str(self.seed),
            "--time-to-teleport",   "-1",
        ]
        traci.start(cmd, label=self.label)
        self._sumo_running = True
        self._discover_topology()
        self._reset_tracking()
        traci.simulationStep()          # populate detector values

    def _stop_sumo(self):
        if self._sumo_running:
            try:
                traci.close(wait=False)
            except Exception:
                pass
            self._sumo_running = False

    # ════════════════════════════════════════════════════════════════════
    #  Topology discovery
    # ════════════════════════════════════════════════════════════════════

    def _discover_topology(self):
        for tl_id in self.tl_ids:
            programs = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)
            phases   = programs[0].phases if programs else []

            green_idxs = [
                i for i, p in enumerate(phases)
                if self._is_pure_green(p.state)
            ]
            if not green_idxs:
                green_idxs = [0]

            self._green_sumo_indices[tl_id] = green_idxs
            self._green_states[tl_id]      = [phases[i].state for i in green_idxs]
            self._yellow_states[tl_id]     = [
                phases[i].state.replace("G", "y").replace("g", "y")
                for i in green_idxs
            ]

            # Lane list (deduplicated, order-preserving)
            raw = traci.trafficlight.getControlledLanes(tl_id)
            seen, lanes = set(), []
            for ln in raw:
                if ln not in seen:
                    seen.add(ln)
                    lanes.append(ln)
            self._tl_lanes[tl_id] = lanes

            for ln in lanes:
                length            = traci.lane.getLength(ln)
                self._lane_cap[ln]      = max(1.0, length / self.VEH_SLOT)
                self._is_veh_lane[ln]   = traci.lane.getMaxSpeed(ln) >= self.MIN_VEH_SPEED

        # Sync action_space with discovered phase counts
        n_phases = [len(self._green_sumo_indices[tl]) for tl in self.tl_ids]
        padded   = n_phases + [1] * (self.MAX_TLS - self.n_tls)
        self.action_space = spaces.MultiDiscrete(padded)

    def _reset_tracking(self):
        for tl_id in self.tl_ids:
            n_green = len(self._green_sumo_indices[tl_id])
            self._cur_green_idx[tl_id]   = 0
            self._steps_in_phase[tl_id]  = 0
            self._steps_since_sw[tl_id]  = self.MIN_GREEN_STEPS
            self._phase_age[tl_id]       = np.zeros(n_green, dtype=np.float32)
            # Set initial green state directly via state string
            init_state = self._green_states[tl_id][0]
            traci.trafficlight.setRedYellowGreenState(tl_id, init_state)

    @staticmethod
    def _is_pure_green(state: str) -> bool:
        """Green phase: at least one G or g, no yellow characters."""
        return ("G" in state or "g" in state) and "y" not in state.lower()

    # ════════════════════════════════════════════════════════════════════
    #  Observation
    # ════════════════════════════════════════════════════════════════════

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(self.OBS_DIM, dtype=np.float32)
        for i, tl_id in enumerate(self.tl_ids):
            base = i * self.OBS_PER_TL
            obs[base: base + self.OBS_PER_TL] = self._tl_obs(tl_id)
        return obs

    def _tl_obs(self, tl_id: str) -> np.ndarray:
        lanes = self._tl_lanes[tl_id]

        # 10 lane ratios — zero-padded beyond available lanes
        lane_ratios = np.zeros(self.MAX_LANES, dtype=np.float32)
        for j, ln in enumerate(lanes[: self.MAX_LANES]):
            halting        = traci.lane.getLastStepHaltingNumber(ln)
            lane_ratios[j] = min(1.0, halting / self._lane_cap[ln])

        # Phase index normalised to [0, 1]
        n_green    = len(self._green_sumo_indices[tl_id])
        phase_norm = (
            self._cur_green_idx[tl_id] / (n_green - 1)
            if n_green > 1 else 0.0
        )

        # Steps in current phase, normalised by 20
        steps_norm = min(1.0, self._steps_in_phase[tl_id] / 20.0)

        # Starvation: max unserved phase age vs. 2 × STARVATION_MAX_STEPS
        starvation = min(
            1.0,
            self._phase_age[tl_id].max() / (2.0 * self.STARVATION_MAX_STEPS),
        )

        return np.concatenate([lane_ratios, [phase_norm, steps_norm, starvation]])

    # ════════════════════════════════════════════════════════════════════
    #  Reward  (thesis Eq. 1)
    # ════════════════════════════════════════════════════════════════════

    def _compute_reward(self, arrived: int, n_switches: int) -> float:
        stopped_ratios: list = []
        wait_times:     list = []
        queue_lengths:  list = []
        tl_ratios:      dict = {tl: [] for tl in self.tl_ids}

        for tl_id in self.tl_ids:
            for ln in self._tl_lanes[tl_id]:
                if not self._is_veh_lane[ln]:
                    continue
                halting = traci.lane.getLastStepHaltingNumber(ln)
                ratio   = min(1.0, halting / self._lane_cap[ln])
                stopped_ratios.append(ratio)
                tl_ratios[tl_id].append(ratio)
                wait_times.append(traci.lane.getWaitingTime(ln))
                queue_lengths.append(float(halting))

        if not stopped_ratios:
            return float(self.b0)

        mean_sr = float(np.mean(stopped_ratios))
        mean_wt = float(np.mean(wait_times))
        mean_ql = float(np.mean(queue_lengths))

        tl_mean_sr = {
            tl: float(np.mean(v)) if v else 0.0
            for tl, v in tl_ratios.items()
        }
        max_sr = max(tl_mean_sr.values())

        # Starvation score ψ_t
        total_excess = sum(
            np.maximum(0, self._phase_age[tl] - self.STARVATION_MAX_STEPS).sum()
            for tl in self.tl_ids
        )
        psi = min(1.0, total_excess / (self.STARVATION_MAX_STEPS * max(1, self.n_tls)))

        reward = (
              self.BETA_TP * arrived
            - self.BETA_SR * mean_sr
            - self.BETA_WT * (mean_wt / self.D_W)
            - self.BETA_QL * (mean_ql / self.D_Q)
            - self.BETA_LC * max_sr
            - self.BETA_CC * (max_sr - mean_sr) ** 2
            - self.BETA_ST * psi
            - self.BETA_SW * n_switches
            + self.b0
        )
        return float(reward)

    # ════════════════════════════════════════════════════════════════════
    #  Evaluation helper
    # ════════════════════════════════════════════════════════════════════

    def get_metrics(self) -> dict:
        """Return per-step traffic metrics for evaluation reporting."""
        stopped_ratios, wait_times, queue_lengths = [], [], []

        for tl_id in self.tl_ids:
            for ln in self._tl_lanes[tl_id]:
                if not self._is_veh_lane[ln]:
                    continue
                halting = traci.lane.getLastStepHaltingNumber(ln)
                stopped_ratios.append(min(1.0, halting / self._lane_cap[ln]))
                wait_times.append(traci.lane.getWaitingTime(ln))
                queue_lengths.append(float(halting))

        return {
            "stopped_ratio":  float(np.mean(stopped_ratios)) if stopped_ratios else 0.0,
            "waiting_time":   float(np.mean(wait_times))     if wait_times     else 0.0,
            "queue_length":   float(np.mean(queue_lengths))  if queue_lengths  else 0.0,
            "arrived":        traci.simulation.getArrivedNumber(),
        }
