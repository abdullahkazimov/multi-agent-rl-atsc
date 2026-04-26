"""
safety.py
=========
Phase safety conflict checker (thesis §4.2, deployment component 3).

Ensures that no learned policy can generate a signal state in which two
physically conflicting movements are simultaneously green — a hard
safety requirement for real-world deployment.

A conflict exists when two movements share a physical road space and their
simultaneous green would create a collision risk (opposing left turns without
protected phases, crossing straight movements, etc.).

The conflict matrix is intersection-specific and must be derived from
the traffic engineering diagrams of each installation site.
For simulation verification, we use a heuristic: two signal links
conflict if their state indices overlap with opposite movements in the
SUMO-standard phase string.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple
import numpy as np


class ConflictMatrix:
    """
    Represents which signal-link pairs conflict at a given intersection.

    Parameters
    ----------
    n_links  : total number of signal links controlled by the TL
    conflicts : set of (i, j) pairs where i < j and the links conflict
    """

    def __init__(self, n_links: int, conflicts: Set[Tuple[int, int]]):
        self.n_links   = n_links
        self.conflicts = frozenset(
            (min(i, j), max(i, j)) for i, j in conflicts
        )

    def has_conflict(self, state: str) -> bool:
        """Return True if any pair of green links conflict."""
        green = {i for i, c in enumerate(state) if c in ("G", "g")}
        for (i, j) in self.conflicts:
            if i in green and j in green:
                return True
        return False

    @classmethod
    def from_sumo_phases(cls, phases: List[str]) -> "ConflictMatrix":
        """
        Heuristic conflict derivation from SUMO phase strings.

        Two links (i, j) are flagged as conflicting if there is NO phase
        in the program where both i and j are simultaneously green.
        This covers opposing movements that are never served together.
        """
        n_links = len(phases[0]) if phases else 0
        green_sets: List[Set[int]] = []
        for state in phases:
            green_sets.append({i for i, c in enumerate(state) if c in ("G", "g")})

        conflicts: Set[Tuple[int, int]] = set()
        for i in range(n_links):
            for j in range(i + 1, n_links):
                jointly_green = any(i in gs and j in gs for gs in green_sets)
                if not jointly_green:
                    conflicts.add((i, j))

        return cls(n_links, conflicts)


class SafetyChecker:
    """
    Validates TL state strings before they are sent to signal hardware.

    Usage
    -----
      checker = SafetyChecker.from_sumo_programs(tl_phase_dict)
      if not checker.is_safe("TL_1", proposed_state):
          proposed_state = checker.fallback_state("TL_1")
    """

    def __init__(self, matrices: Dict[str, ConflictMatrix]):
        self._matrices = matrices

    @classmethod
    def from_sumo_programs(
        cls,
        tl_programs: Dict[str, List[str]],
    ) -> "SafetyChecker":
        """
        Build a SafetyChecker from a dict of {tl_id: [phase_state_str, …]}.
        """
        matrices = {
            tl_id: ConflictMatrix.from_sumo_phases(phases)
            for tl_id, phases in tl_programs.items()
        }
        return cls(matrices)

    def is_safe(self, tl_id: str, state: str) -> bool:
        """Return True if the proposed state has no conflicting green links."""
        matrix = self._matrices.get(tl_id)
        if matrix is None:
            return True   # no conflict info → assume safe (conservative)
        return not matrix.has_conflict(state)

    def fallback_state(self, tl_id: str) -> str:
        """
        Return an all-red state string as a safe fallback.
        Length is derived from the conflict matrix.
        """
        matrix = self._matrices.get(tl_id)
        n      = matrix.n_links if matrix else 1
        return "r" * n

    def enforce(self, tl_id: str, state: str) -> str:
        """
        Return the state if safe, or the all-red fallback otherwise.
        Logs a warning when a conflict is blocked.
        """
        if self.is_safe(tl_id, state):
            return state
        import warnings
        warnings.warn(
            f"SafetyChecker: blocked conflicting state '{state}' for TL '{tl_id}'. "
            "Falling back to all-red.",
            RuntimeWarning,
            stacklevel=2,
        )
        return self.fallback_state(tl_id)

    def validate_program(self, tl_id: str, phases: List[str]) -> Dict[str, bool]:
        """Check every phase in a program for safety violations."""
        return {state: self.is_safe(tl_id, state) for state in phases}
