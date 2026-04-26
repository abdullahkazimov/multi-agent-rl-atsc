"""
test_run.py
===========
Quick integration smoke-test: runs each of the four Baku scenarios for
a small number of steps, verifying that the environment, observation,
action, and reward pipeline works end-to-end.

Run with:
  python test_run.py

Expected output: a table showing pass/fail for each scenario.
No training model required — random actions are used.
"""

from __future__ import annotations
import os
import sys
import warnings
import traceback
import time

os.environ.setdefault("SUMO_HOME", "/usr/share/sumo")
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np

from envs.baku_sumo_env    import BakuSUMOEnv
from envs.trimmed_env      import TrimmedTrafficEnv
from envs.scenario_configs  import get_scenario_config, SCENARIOS


N_STEPS       = 10     # steps per scenario smoke-test
WARMUP_STEPS  = 2      # steps before metrics are asserted


def run_scenario(scenario_name: str, n_steps: int = N_STEPS) -> dict:
    cfg = get_scenario_config(scenario_name)
    result = {
        "scenario":   scenario_name,
        "status":     "FAIL",
        "obs_shape":  None,
        "action_nvec": None,
        "rewards":    [],
        "error":      None,
        "elapsed_s":  0.0,
    }

    t0 = time.perf_counter()
    env = None
    try:
        base = BakuSUMOEnv(
            tl_ids          = cfg["tl_ids"],
            sumocfg_path    = cfg["sumocfg_path"],
            n_phases_per_tl = cfg["n_phases"],
            seed            = 42,
            b0              = 0.0,
            label           = f"smoke_{scenario_name}",
        )
        env = TrimmedTrafficEnv(base)
        obs, info = env.reset()

        result["obs_shape"]   = obs.shape
        result["action_nvec"] = list(env.action_space.nvec)

        # Validate observation
        assert obs.shape == (cfg["obs_dim"],), \
            f"obs shape mismatch: got {obs.shape}, expected ({cfg['obs_dim']},)"
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0), \
            "obs values out of [0, 1]"

        # Validate action space
        assert len(env.action_space.nvec) == len(cfg["tl_ids"]), \
            "action_space.nvec length mismatch"

        for step in range(n_steps):
            action = env.action_space.sample()
            obs, reward, done, trunc, info = env.step(action)

            assert obs.shape == (cfg["obs_dim"],), f"step {step}: bad obs shape"
            assert np.isfinite(reward), f"step {step}: non-finite reward {reward}"
            assert np.all(np.isfinite(obs)), f"step {step}: non-finite obs"

            if step >= WARMUP_STEPS:
                result["rewards"].append(reward)

            if done:
                break

        result["status"] = "PASS"

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        result["elapsed_s"] = time.perf_counter() - t0

    return result


def main():
    print("\n" + "═" * 70)
    print("  Baku MARL-ATSC — Integration Smoke Test")
    print("═" * 70)
    print(f"  {'Scenario':<14}  {'Status':>6}  {'Obs':>9}  "
          f"{'Actions':>18}  {'Mean Reward':>12}  {'Time':>6}")
    print("─" * 70)

    all_pass = True
    for scenario_name in SCENARIOS:
        r = run_scenario(scenario_name)
        status_str = r["status"]
        obs_str    = str(r["obs_shape"]) if r["obs_shape"] else "—"
        act_str    = str(r["action_nvec"]) if r["action_nvec"] else "—"
        mean_rew   = f"{np.mean(r['rewards']):.4f}" if r["rewards"] else "—"
        elapsed    = f"{r['elapsed_s']:.1f}s"

        print(f"  {scenario_name:<14}  {status_str:>6}  {obs_str:>9}  "
              f"{act_str:>18}  {mean_rew:>12}  {elapsed:>6}")

        if r["status"] != "PASS":
            all_pass = False
            print(f"\n  ERROR in {scenario_name}:")
            if r["error"]:
                for line in r["error"].splitlines():
                    print(f"    {line}")
            print()

    print("═" * 70)
    if all_pass:
        print("  ALL SCENARIOS PASSED ✓")
    else:
        print("  SOME SCENARIOS FAILED — see errors above")
    print("═" * 70 + "\n")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
