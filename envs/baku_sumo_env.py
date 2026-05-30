"""
BakuSUMOEnv
===========
Gymnasium-compatible environment wrapping SUMO via TraCI for the four
Baku adaptive traffic signal control benchmarks.

Design (thesis §3.2 – §3.4):
  • Decision interval : 10 simulated seconds
  • Yellow clearance  : 3 simulated seconds (inserted on phase switch)
  • Min phase hold    : 3 decision steps (≥ 30 s)
  • Observation       : 12-slot × 14-value flat vector (full, 168-dim)
  • Action            : MultiDiscrete — one green-phase index per TL
                        (+ optional all-red metering action via meter_tls)
  • Reward            : 8-term composite (thesis Eq. 1 + pedestrian term)
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
    OBS_PER_TL          = 14    # lane ratios (10) + phase + steps + starvation + ped_pressure
    MAX_TLS             = 12    # full observation covers 12 TL slots
    OBS_DIM             = MAX_TLS * OBS_PER_TL          # 168
    STARVATION_MAX_STEPS = 10   # phases unserved beyond this are penalised (lowered
                                #   2026-05-29 so cross-street starvation bites sooner)

    # ── Reward weights (thesis Eq. 1) ────────────────────────────────────
    BETA_TP = 0.15   # throughput increment
    BETA_SR = 0.30   # mean stopped-vehicle ratio
    BETA_WT = 0.55   # mean waiting time (primary target)
    BETA_QL = 0.12   # mean queue length
    BETA_LC = 0.60   # hotspot — worst-TL stopped ratio (raised 2026-05-29: directly
                     #   penalises a fully-jammed/starved approach on multi-junction nets)
    BETA_CC = 0.10   # MARL coordination balance penalty
    BETA_ST = 1.00   # phase-starvation score (raised 2026-05-29 to break phase-lock)
    BETA_SW = 0.00   # switching cost disabled (was preventing exploration)
    D_W     = 100.0  # waiting-time normaliser increased to dampen outlier penalties
    D_Q     = 10.0   # queue-length normaliser (vehicles)
    BETA_PED      = 0.50   # pedestrian-waiting penalty (active only where persons exist)
    PED_WAIT_NORM = 20.0   # persons-stopped-at-crossing normaliser

    # Gridlock guard (added 2026-05-29): end the episode with a penalty if the
    # network is jammed (≥ GRIDLOCK_MIN_VEH present but < 1 completing) for
    # GRIDLOCK_PATIENCE consecutive decision steps. Stops phase-lock trajectories
    # from poisoning the value function with a long all-bad tail.
    GRIDLOCK_MIN_VEH   = 30
    GRIDLOCK_MIN_QUEUE = 40    # total controlled-lane halting — distinguishes a real
                               #   jam from the harmless end-of-episode drain
    GRIDLOCK_PATIENCE  = 6     # decision steps (~60 s) of zero throughput
    GRIDLOCK_WARMUP    = 35    # don't arm the guard until vehicles can reach exits
    GRIDLOCK_PENALTY   = 10.0

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
        meter_tls: list = None,
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
        # TLs that receive a synthetic all-red action for ramp-style metering
        # (single-phase junctions controlling a downstream bottleneck).
        self.meter_tls       = set(meter_tls or [])

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
        self._last_arrivals = 0  # Σ vehicles arrived over last decision interval (thesis Δthr)
        self._ped_wait_count: dict = {}   # tl_id → #persons stopped at its crossings
        self._ped_wait_time:  dict = {}   # tl_id → Σ pedestrian waiting time
        self._gridlock_steps  = 0         # consecutive jammed decision steps

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
        arrived = 0
        for t in range(self.DELTA_TIME):
            if t == self.YELLOW_TIME:
                for tl_id, new_idx in switch_map.items():
                    green_state = self._green_states[tl_id][new_idx]
                    traci.trafficlight.setRedYellowGreenState(tl_id, green_state)
            traci.simulationStep()
            arrived += int(traci.simulation.getArrivedNumber())

        self._last_arrivals = arrived
        self._update_ped_pressure()
        # ── Collect metrics and build output ─────────────────────────
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

        # Gridlock guard — end the episode with a penalty when the network is
        # jammed (many vehicles present but ~none completing) for several
        # consecutive steps. Bounds the catastrophic tail and signals that
        # lock→gridlock trajectories are dead ends.
        try:
            veh_present = traci.vehicle.getIDCount()
            total_q = sum(traci.lane.getLastStepHaltingNumber(ln)
                          for lns in self._tl_lanes.values() for ln in lns)
        except Exception:
            veh_present, total_q = 0, 0
        if (self._step_count > self.GRIDLOCK_WARMUP
                and veh_present > self.GRIDLOCK_MIN_VEH
                and total_q    > self.GRIDLOCK_MIN_QUEUE
                and arrived < 1):
            self._gridlock_steps += 1
        else:
            self._gridlock_steps = 0
        gridlocked = self._gridlock_steps >= self.GRIDLOCK_PATIENCE
        if gridlocked:
            reward -= self.GRIDLOCK_PENALTY

        done = (
            gridlocked
            or traci.simulation.getMinExpectedNumber() <= 0
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
        self._last_arrivals = int(traci.simulation.getArrivedNumber())
        self._update_ped_pressure()

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

            # Metering: give a single-phase TL a 2nd action = all-red, so the
            # agent can hold traffic and create gaps into a downstream squeeze
            # (ramp-metering). Scoped via meter_tls so multi-phase TLs and other
            # scenarios (e.g. hexagon's single-phase L-nodes) are untouched.
            if tl_id in self.meter_tls and len(self._green_states[tl_id]) == 1:
                all_red = "r" * len(self._green_states[tl_id][0])
                self._green_sumo_indices[tl_id].append(-1)   # synthetic phase
                self._green_states[tl_id].append(all_red)
                self._yellow_states[tl_id].append(all_red)   # stay red in transition

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
        self._gridlock_steps = 0
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

        # Pedestrian pressure: #persons stopped at this TL's crossings (0 where none)
        ped = min(1.0, self._ped_wait_count.get(tl_id, 0) / self.PED_WAIT_NORM)

        return np.concatenate([lane_ratios, [phase_norm, steps_norm, starvation, ped]])

    def _update_ped_pressure(self):
        """Per-TL count + accumulated wait of pedestrians stopped at this TL's
        crossings/walkingareas (internal ':<tl>_*' edges). Naturally zero in
        scenarios without persons, so the other benchmarks are unaffected."""
        self._ped_wait_count = {tl: 0   for tl in self.tl_ids}
        self._ped_wait_time  = {tl: 0.0 for tl in self.tl_ids}
        try:
            pids = traci.person.getIDList()
        except Exception:
            return
        for p in pids:
            road = traci.person.getRoadID(p)
            if not road.startswith(":"):
                continue                       # walking a footpath, not at a junction
            if traci.person.getSpeed(p) >= 0.3:
                continue                       # moving, not waiting to cross
            for tl in self.tl_ids:
                if road.startswith(f":{tl}_"):
                    self._ped_wait_count[tl] += 1
                    self._ped_wait_time[tl]  += traci.person.getWaitingTime(p)
                    break

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

        # Pedestrian pressure: mean (over TLs) of normalised persons stopped at
        # crossings. Zero in car-only scenarios → leaves their reward unchanged.
        ped_pressure = (
            float(np.mean([
                min(1.0, self._ped_wait_count.get(tl, 0) / self.PED_WAIT_NORM)
                for tl in self.tl_ids
            ])) if self.tl_ids else 0.0
        )

        reward = (
              self.BETA_TP * arrived
            - self.BETA_SR * mean_sr
            - self.BETA_WT * (mean_wt / self.D_W)
            - self.BETA_QL * (mean_ql / self.D_Q)
            - self.BETA_LC * max_sr
            - self.BETA_CC * (max_sr - mean_sr) ** 2
            - self.BETA_ST * psi
            - self.BETA_SW * n_switches
            - self.BETA_PED * ped_pressure
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
            "arrived":        float(self._last_arrivals),
            "ped_waiting":    float(sum(self._ped_wait_count.values())),
        }

    def get_per_tl_metrics(self) -> dict:
        """Per-junction metrics for animation: stopped_ratio, waiting_time, queue_length, phase."""
        result = {}
        for tl_id in self.tl_ids:
            ratios, waits, queues = [], [], []
            for ln in self._tl_lanes.get(tl_id, []):
                if not self._is_veh_lane.get(ln, False):
                    continue
                halting = traci.lane.getLastStepHaltingNumber(ln)
                ratios.append(min(1.0, halting / self._lane_cap[ln]))
                waits.append(traci.lane.getWaitingTime(ln))
                queues.append(float(halting))
            result[tl_id] = {
                "stopped_ratio": round(float(np.mean(ratios)) if ratios else 0.0, 4),
                "waiting_time":  round(float(np.mean(waits))  if waits  else 0.0, 3),
                "queue_length":  round(float(np.mean(queues)) if queues else 0.0, 3),
                "phase":         int(self._cur_green_idx.get(tl_id, 0)),
            }
        return result
