"""
Environment tests — SUMO required.

These tests actually start SUMO processes and exercise BakuSUMOEnv +
TrimmedTrafficEnv for a small number of steps.  They verify:
  • reset() returns the correct observation shape
  • step() returns valid obs, reward, done, truncated, info
  • Reward values are finite
  • TrimmedTrafficEnv slices obs correctly
  • Action space is consistent with scenario config

Each test is tagged with @pytest.mark.sumo so they can be skipped
in CI environments without SUMO installed:

  pytest tests/test_env.py -m "not sumo"
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np

# ── Check SUMO availability ────────────────────────────────────────────────────
import shutil
SUMO_AVAILABLE = shutil.which("sumo") is not None
sumo_required = pytest.mark.skipif(
    not SUMO_AVAILABLE,
    reason="SUMO binary not found in PATH",
)

from envs.baku_sumo_env    import BakuSUMOEnv
from envs.trimmed_env      import TrimmedTrafficEnv
from envs.scenario_configs  import get_scenario_config


N_STEPS = 5   # quick smoke-test: just a few steps per test


def make_trimmed_env(scenario_name: str, seed: int = 42) -> TrimmedTrafficEnv:
    cfg = get_scenario_config(scenario_name)
    base = BakuSUMOEnv(
        tl_ids          = cfg["tl_ids"],
        sumocfg_path    = cfg["sumocfg_path"],
        n_phases_per_tl = cfg["n_phases"],
        seed            = seed,
        b0              = 0.0,
        label           = f"test_{scenario_name}_{seed}",
    )
    return TrimmedTrafficEnv(base)


# ─────────────────────────────────────────────────────────────────────────────
#  Scenario configs (no SUMO needed)
# ─────────────────────────────────────────────────────────────────────────────

class TestScenarioConfigs:
    @pytest.mark.parametrize("scenario", ["bottleneck", "main", "pedestrian", "hexagon"])
    def test_obs_dim_matches_tl_count(self, scenario):
        cfg = get_scenario_config(scenario)
        expected = len(cfg["tl_ids"]) * BakuSUMOEnv.OBS_PER_TL
        assert cfg["obs_dim"] == expected

    @pytest.mark.parametrize("scenario", ["bottleneck", "main", "pedestrian", "hexagon"])
    def test_n_phases_length(self, scenario):
        cfg = get_scenario_config(scenario)
        assert len(cfg["n_phases"]) == len(cfg["tl_ids"])

    def test_main_has_10_eval_seeds(self):
        cfg = get_scenario_config("main")
        assert len(cfg["eval_seeds"]) == 10

    @pytest.mark.parametrize("scenario", ["bottleneck", "pedestrian", "hexagon"])
    def test_other_scenarios_have_5_seeds(self, scenario):
        cfg = get_scenario_config(scenario)
        assert len(cfg["eval_seeds"]) == 5


# ─────────────────────────────────────────────────────────────────────────────
#  Environment tests (SUMO required)
# ─────────────────────────────────────────────────────────────────────────────

@sumo_required
class TestBakuSUMOEnvBottleneck:
    def test_reset_obs_shape(self):
        env = make_trimmed_env("bottleneck")
        obs, info = env.reset()
        assert obs.shape == (26,), f"expected (26,), got {obs.shape}"
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0)
        env.close()

    def test_step_returns_valid_outputs(self):
        env = make_trimmed_env("bottleneck")
        env.reset()
        cfg = get_scenario_config("bottleneck")
        action = env.action_space.sample()
        obs, reward, done, trunc, info = env.step(action)
        assert obs.shape == (cfg["obs_dim"],)
        assert np.isfinite(reward)
        assert isinstance(done, (bool, np.bool_))
        env.close()

    def test_action_space_matches_config(self):
        env = make_trimmed_env("bottleneck")
        env.reset()
        cfg = get_scenario_config("bottleneck")
        assert len(env.action_space.nvec) == len(cfg["tl_ids"])
        env.close()

    def test_multiple_steps(self):
        env = make_trimmed_env("bottleneck")
        env.reset()
        for _ in range(N_STEPS):
            action = env.action_space.sample()
            obs, reward, done, trunc, _ = env.step(action)
            assert np.all(np.isfinite(obs))
            assert np.isfinite(reward)
            if done:
                break
        env.close()


@sumo_required
class TestBakuSUMOEnvMain:
    def test_reset_obs_shape(self):
        env = make_trimmed_env("main")
        obs, _ = env.reset()
        assert obs.shape == (39,)
        env.close()

    def test_action_space(self):
        env = make_trimmed_env("main")
        env.reset()
        # main: 3 TLs each with 2 green phases
        assert list(env.action_space.nvec) == [2, 2, 2]
        env.close()

    def test_multiple_steps(self):
        env = make_trimmed_env("main")
        env.reset()
        rewards = []
        for _ in range(N_STEPS):
            action = env.action_space.sample()
            obs, reward, done, _, _ = env.step(action)
            rewards.append(reward)
            if done:
                break
        assert all(np.isfinite(r) for r in rewards)
        env.close()


@sumo_required
class TestBakuSUMOEnvPedestrian:
    def test_reset_obs_shape(self):
        env = make_trimmed_env("pedestrian")
        obs, _ = env.reset()
        assert obs.shape == (13,)
        env.close()

    def test_pedestrian_has_6_green_phases(self):
        env = make_trimmed_env("pedestrian")
        env.reset()
        assert env.action_space.nvec[0] == 6
        env.close()

    def test_multiple_steps(self):
        env = make_trimmed_env("pedestrian")
        env.reset()
        for _ in range(N_STEPS):
            action = env.action_space.sample()
            _, reward, done, _, _ = env.step(action)
            assert np.isfinite(reward)
            if done:
                break
        env.close()


@sumo_required
class TestBakuSUMOEnvHexagon:
    def test_reset_obs_shape(self):
        env = make_trimmed_env("hexagon")
        obs, _ = env.reset()
        assert obs.shape == (156,)
        env.close()

    def test_action_space_size(self):
        env = make_trimmed_env("hexagon")
        env.reset()
        # 12 TLs
        assert len(env.action_space.nvec) == 12
        env.close()

    def test_multiple_steps(self):
        env = make_trimmed_env("hexagon")
        env.reset()
        for _ in range(N_STEPS):
            action = env.action_space.sample()
            _, reward, done, _, _ = env.step(action)
            assert np.isfinite(reward)
            if done:
                break
        env.close()


@sumo_required
class TestTrimmedEnvWrapper:
    def test_obs_sliced_correctly(self):
        cfg   = get_scenario_config("main")
        base  = BakuSUMOEnv(
            tl_ids          = cfg["tl_ids"],
            sumocfg_path    = cfg["sumocfg_path"],
            n_phases_per_tl = cfg["n_phases"],
            seed            = 99,
            label           = "test_trim_main",
        )
        trimmed = TrimmedTrafficEnv(base)
        obs, _ = trimmed.reset()
        # trimmed obs should have first n_tls × OBS_PER_TL elements
        assert obs.shape == (cfg["obs_dim"],)
        base.close()

    def test_action_padded_to_max_tls(self):
        """Action passed to base env should be padded to MAX_TLS."""
        cfg   = get_scenario_config("bottleneck")
        base  = BakuSUMOEnv(
            tl_ids          = cfg["tl_ids"],
            sumocfg_path    = cfg["sumocfg_path"],
            n_phases_per_tl = cfg["n_phases"],
            seed            = 7,
            label           = "test_trim_bn",
        )
        trimmed = TrimmedTrafficEnv(base)
        trimmed.reset()
        # bottleneck has 2 TLs — trimmed action space nvec should have 2 elements
        assert len(trimmed.action_space.nvec) == 2
        # step should not raise
        action = trimmed.action_space.sample()
        trimmed.step(action)
        trimmed.close()
