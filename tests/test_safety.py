"""Unit tests for the phase safety conflict checker."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from safety import ConflictMatrix, SafetyChecker


class TestConflictMatrix:
    def test_from_sumo_phases_basic(self):
        # Two phases: {0,1} green and {2,3} green — never share 0-2
        phases = ["GGrr", "rrGG"]
        cm = ConflictMatrix.from_sumo_phases(phases)
        assert (0, 2) in cm.conflicts
        assert (0, 1) not in cm.conflicts  # both green in phase 0
        assert (2, 3) not in cm.conflicts  # both green in phase 1

    def test_has_conflict_true(self):
        cm = ConflictMatrix(4, {(0, 2)})
        assert cm.has_conflict("GrGr")  # links 0 and 2 both green

    def test_has_conflict_false(self):
        cm = ConflictMatrix(4, {(0, 2)})
        assert not cm.has_conflict("GrrG")  # link 0 and 3 green — no conflict
        assert not cm.has_conflict("rGrG")  # link 1 and 3 green — no conflict

    def test_has_conflict_no_conflicts(self):
        cm = ConflictMatrix(4, set())
        assert not cm.has_conflict("GGGG")  # all green, no conflicts defined

    def test_small_lowercase_g(self):
        cm = ConflictMatrix(4, {(0, 1)})
        assert cm.has_conflict("ggGr")  # lowercase g also counts

    def test_empty_phases(self):
        cm = ConflictMatrix.from_sumo_phases([])
        assert cm.n_links == 0


class TestSafetyChecker:
    def _checker(self):
        programs = {
            "TL1": ["GGrr", "rrGG"],   # 0-2, 0-3, 1-2, 1-3 conflict
            "TL2": ["GGGG"],           # no conflicts (always all green)
        }
        return SafetyChecker.from_sumo_programs(programs)

    def test_is_safe_valid(self):
        checker = self._checker()
        assert checker.is_safe("TL1", "GGrr")   # phase 0 state
        assert checker.is_safe("TL1", "rrGG")   # phase 1 state

    def test_is_safe_invalid(self):
        checker = self._checker()
        # GrGr: links 0 and 2 green simultaneously — never appears in programs
        assert not checker.is_safe("TL1", "GrGr")

    def test_is_safe_unknown_tl(self):
        checker = self._checker()
        assert checker.is_safe("TL_UNKNOWN", "GGGG")   # unknown → assume safe

    def test_fallback_state(self):
        checker = self._checker()
        fb = checker.fallback_state("TL1")
        assert fb == "rrrr"
        assert "G" not in fb and "g" not in fb

    def test_enforce_passes_safe(self):
        checker = self._checker()
        state = checker.enforce("TL1", "GGrr")
        assert state == "GGrr"

    def test_enforce_blocks_unsafe(self):
        checker = self._checker()
        with pytest.warns(RuntimeWarning):
            state = checker.enforce("TL1", "GrGr")
        assert state == checker.fallback_state("TL1")

    def test_validate_program(self):
        checker = self._checker()
        results = checker.validate_program("TL1", ["GGrr", "rrGG", "GrGr"])
        assert results["GGrr"] is True
        assert results["rrGG"] is True
        assert results["GrGr"] is False

    def test_tl2_always_green_safe(self):
        checker = self._checker()
        # TL2 only has one phase (GGGG) so no pair of links is ever NOT jointly green
        assert checker.is_safe("TL2", "GGGG")
