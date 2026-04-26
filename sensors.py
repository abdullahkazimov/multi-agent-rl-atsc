"""
sensors.py
==========
Prototype sensor backend for real-world deployment (thesis §4.2).

In production, this module would interface with hardware sensor systems
(inductive loops, radar, camera-based detection) to provide the same
lane-level halting counts and waiting times that TraCI provides in
simulation.

For offline / simulation use, SumoSensorBackend reads directly from
a running TraCI session.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List
import os
import warnings

os.environ.setdefault("SUMO_HOME", "/usr/share/sumo")
warnings.filterwarnings("ignore", category=UserWarning, module="traci")


class SensorBackend(ABC):
    """Abstract base: must return per-lane halting counts and waiting times."""

    @abstractmethod
    def get_halting_counts(self, tl_id: str, lanes: List[str]) -> Dict[str, int]:
        """Return {lane_id: halting_vehicle_count} for the given TL's lanes."""

    @abstractmethod
    def get_waiting_times(self, tl_id: str, lanes: List[str]) -> Dict[str, float]:
        """Return {lane_id: cumulative_waiting_time_seconds}."""

    @abstractmethod
    def get_arrived_count(self) -> int:
        """Return the number of vehicles that completed trips in this step."""


class SumoSensorBackend(SensorBackend):
    """
    Reads sensor values directly from an active TraCI session.
    Used during simulation for both training and evaluation.
    """

    def get_halting_counts(self, tl_id: str, lanes: List[str]) -> Dict[str, int]:
        import traci
        return {ln: traci.lane.getLastStepHaltingNumber(ln) for ln in lanes}

    def get_waiting_times(self, tl_id: str, lanes: List[str]) -> Dict[str, float]:
        import traci
        return {ln: traci.lane.getWaitingTime(ln) for ln in lanes}

    def get_arrived_count(self) -> int:
        import traci
        return traci.simulation.getArrivedNumber()


class MockSensorBackend(SensorBackend):
    """
    In-memory mock backend for unit testing.

    Inject synthetic data via set_state() before each simulated step.
    """

    def __init__(self):
        self._halting:  Dict[str, int]   = {}
        self._waiting:  Dict[str, float] = {}
        self._arrived:  int              = 0

    def set_state(
        self,
        halting: Dict[str, int],
        waiting: Dict[str, float],
        arrived: int = 0,
    ):
        self._halting = halting
        self._waiting = waiting
        self._arrived = arrived

    def get_halting_counts(self, tl_id: str, lanes: List[str]) -> Dict[str, int]:
        return {ln: self._halting.get(ln, 0) for ln in lanes}

    def get_waiting_times(self, tl_id: str, lanes: List[str]) -> Dict[str, float]:
        return {ln: self._waiting.get(ln, 0.0) for ln in lanes}

    def get_arrived_count(self) -> int:
        return self._arrived


class HardwareSensorBackend(SensorBackend):
    """
    Stub for real hardware integration (OPC-UA / NTCIP).

    Replace each method body with calls to the actual hardware API.
    The interface contract (lane_id → value) remains identical to
    SumoSensorBackend so no other code needs to change.
    """

    def __init__(self, endpoint: str, timeout: float = 1.0):
        self.endpoint = endpoint
        self.timeout  = timeout
        # TODO: initialise OPC-UA / NTCIP client here

    def get_halting_counts(self, tl_id: str, lanes: List[str]) -> Dict[str, int]:
        # TODO: query inductive-loop detector counts via hardware API
        raise NotImplementedError("Hardware integration not yet implemented")

    def get_waiting_times(self, tl_id: str, lanes: List[str]) -> Dict[str, float]:
        # TODO: compute per-lane waiting times from detector timestamps
        raise NotImplementedError("Hardware integration not yet implemented")

    def get_arrived_count(self) -> int:
        # TODO: derive from exit-detector pulse counts
        raise NotImplementedError("Hardware integration not yet implemented")
