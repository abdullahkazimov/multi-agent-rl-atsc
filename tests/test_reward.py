"""Unit tests for the 7-term composite reward function."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from rewards.reward_fn import RewardFunction, RewardComponents, compute_reward_components


class TestRewardComponents:
    def test_total_property(self):
        rc = RewardComponents(
            throughput    =  0.15,
            stopped_ratio =  0.30,
            waiting_time  =  0.55,
            queue_length  =  0.12,
            hotspot       =  0.25,
            coord_balance =  0.10,
            starvation    =  0.08,
            switching     =  0.05,
            baseline      =  0.50,
        )
        expected = 0.15 - 0.30 - 0.55 - 0.12 - 0.25 - 0.10 - 0.08 - 0.05 + 0.50
        assert abs(rc.total - expected) < 1e-9

    def test_zero_reward(self):
        rc = RewardComponents()
        assert rc.total == 0.0


class TestRewardFunction:
    def setup_method(self):
        self.rf = RewardFunction()

    def test_empty_lanes(self):
        """With no vehicle lanes, reward equals b0."""
        self.rf.b0 = -0.5
        r = self.rf(
            arrived=0,
            stopped_ratios=[],
            wait_times=[],
            queue_lengths=[],
            tl_ratios={},
            phase_ages={},
            n_switches=0,
        )
        assert r == -0.5

    def test_throughput_positive(self):
        """More arrived vehicles should increase reward."""
        base = self.rf(
            arrived=0,
            stopped_ratios=[0.0],
            wait_times=[0.0],
            queue_lengths=[0.0],
            tl_ratios={"TL1": [0.0]},
            phase_ages={"TL1": np.zeros(2)},
            n_switches=0,
        )
        high = self.rf(
            arrived=10,
            stopped_ratios=[0.0],
            wait_times=[0.0],
            queue_lengths=[0.0],
            tl_ratios={"TL1": [0.0]},
            phase_ages={"TL1": np.zeros(2)},
            n_switches=0,
        )
        assert high > base

    def test_congestion_penalised(self):
        """High stopped ratio should lower reward."""
        low = self.rf(
            arrived=0,
            stopped_ratios=[0.1],
            wait_times=[5.0],
            queue_lengths=[1.0],
            tl_ratios={"TL1": [0.1]},
            phase_ages={"TL1": np.zeros(1)},
            n_switches=0,
        )
        high = self.rf(
            arrived=0,
            stopped_ratios=[0.9],
            wait_times=[30.0],
            queue_lengths=[9.0],
            tl_ratios={"TL1": [0.9]},
            phase_ages={"TL1": np.zeros(1)},
            n_switches=0,
        )
        assert high < low

    def test_switching_cost(self):
        """Phase switches should reduce reward."""
        no_sw = self.rf(
            arrived=0,
            stopped_ratios=[0.5],
            wait_times=[10.0],
            queue_lengths=[5.0],
            tl_ratios={"TL1": [0.5]},
            phase_ages={"TL1": np.zeros(2)},
            n_switches=0,
        )
        with_sw = self.rf(
            arrived=0,
            stopped_ratios=[0.5],
            wait_times=[10.0],
            queue_lengths=[5.0],
            tl_ratios={"TL1": [0.5]},
            phase_ages={"TL1": np.zeros(2)},
            n_switches=3,
        )
        assert with_sw < no_sw

    def test_starvation_penalty(self):
        """Unserved phases beyond STARVATION_MAX_STEPS should lower reward."""
        fresh = self.rf.compute(
            arrived=0,
            stopped_ratios=[0.5],
            wait_times=[10.0],
            queue_lengths=[5.0],
            tl_ratios={"TL1": [0.5]},
            phase_ages={"TL1": np.zeros(2)},
            n_switches=0,
        )
        stale = self.rf.compute(
            arrived=0,
            stopped_ratios=[0.5],
            wait_times=[10.0],
            queue_lengths=[5.0],
            tl_ratios={"TL1": [0.5]},
            phase_ages={"TL1": np.array([30.0, 30.0])},  # far beyond threshold
            n_switches=0,
        )
        assert stale.total < fresh.total

    def test_coordination_balance(self):
        """Unequal per-TL congestion should increase the coordination penalty."""
        balanced = self.rf.compute(
            arrived=0,
            stopped_ratios=[0.5, 0.5],
            wait_times=[10.0, 10.0],
            queue_lengths=[5.0, 5.0],
            tl_ratios={"T1": [0.5], "T2": [0.5]},
            phase_ages={"T1": np.zeros(2), "T2": np.zeros(2)},
            n_switches=0,
        )
        unbalanced = self.rf.compute(
            arrived=0,
            stopped_ratios=[0.1, 0.9],
            wait_times=[10.0, 10.0],
            queue_lengths=[5.0, 5.0],
            tl_ratios={"T1": [0.1], "T2": [0.9]},
            phase_ages={"T1": np.zeros(2), "T2": np.zeros(2)},
            n_switches=0,
        )
        assert unbalanced.coord_balance > balanced.coord_balance

    def test_b2_baseline_shifts_reward(self):
        """b0 should shift the reward by its value."""
        rf_no_b0  = RewardFunction(b0=0.0)
        rf_with_b0 = RewardFunction(b0=1.0)
        kwargs = dict(
            arrived=5,
            stopped_ratios=[0.3],
            wait_times=[8.0],
            queue_lengths=[3.0],
            tl_ratios={"TL1": [0.3]},
            phase_ages={"TL1": np.zeros(2)},
            n_switches=1,
        )
        r1 = rf_no_b0(**kwargs)
        r2 = rf_with_b0(**kwargs)
        assert abs((r2 - r1) - 1.0) < 1e-9

    def test_per_sigma_gradients_shape(self):
        sigmas = {
            "throughput":    2.0,
            "stopped_ratio": 0.1,
            "waiting_time":  5.0,
            "queue_length":  3.0,
            "hotspot":       0.2,
            "coord_balance": 0.05,
            "starvation":    0.1,
        }
        g = self.rf.per_sigma_gradients(sigmas)
        # per_sigma_gradients returns entries for all non-switching terms
        assert "throughput" in g
        assert "waiting_time" in g
        assert "starvation" in g
        assert all(v >= 0 for v in g.values())

    def test_reward_weights_match_thesis(self):
        """Verify the default weights match thesis Table §3.4."""
        rf = RewardFunction()
        assert rf.beta_tp == pytest.approx(0.15)
        assert rf.beta_sr == pytest.approx(0.30)
        assert rf.beta_wt == pytest.approx(0.55)
        assert rf.beta_ql == pytest.approx(0.12)
        assert rf.beta_lc == pytest.approx(0.25)
        assert rf.beta_cc == pytest.approx(0.10)
        assert rf.beta_st == pytest.approx(0.08)
        assert rf.beta_sw == pytest.approx(0.05)
        assert rf.d_w     == pytest.approx(35.0)
        assert rf.d_q     == pytest.approx(10.0)


class TestComputeRewardComponents:
    def test_convenience_wrapper(self):
        rc = compute_reward_components(
            arrived=3,
            stopped_ratios=[0.2, 0.4],
            wait_times=[5.0, 10.0],
            queue_lengths=[2.0, 4.0],
            tl_ratios={"T1": [0.2], "T2": [0.4]},
            phase_ages={"T1": np.zeros(2), "T2": np.zeros(2)},
            n_switches=1,
        )
        assert isinstance(rc, RewardComponents)
        assert isinstance(rc.total, float)
