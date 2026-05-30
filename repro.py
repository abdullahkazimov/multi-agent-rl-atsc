"""
Reproducibility helpers for PPO + SUMO training.

Use before creating VecEnv, PPO, or any other RNG-backed objects.
For TensorBoard / episode-return curves to match as closely as possible:
  • Set a fixed --seed (default 42, thesis protocol).
  • On GPU, full bitwise determinism is not guaranteed; use --cpu for strict runs
    or set CUDA for deterministic matmuls (CUBLAS_WORKSPACE_CONFIG) below.
"""
from __future__ import annotations

import json
import os
import random
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np


def apply_reproducibility(seed: int, *, use_cuda: bool | None = None) -> None:
    """
    Seed Python / NumPy / PyTorch and enable deterministic CuDNN when on GPU.

    Call once at the start of training (and again at the start of a resumed run
    if you want the same initial RNG state before loading checkpoints — note
    that loading the checkpoint still restores optimizer state, so the rest of
    the run matches continuation semantics, not a fresh start).
    """
    if use_cuda is None:
        try:
            import torch

            use_cuda = bool(torch.cuda.is_available())
        except Exception:
            use_cuda = False

    # PyTorch: deterministic matmuls on GPU (2.x)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    from stable_baselines3.common.utils import set_random_seed

    set_random_seed(seed, using_cuda=use_cuda)

    try:
        import torch

        if use_cuda and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Some ops are still non-deterministic on GPU; fail soft.
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)  # type: ignore[call-overload]
    except Exception:
        pass


def run_manifest(
    out_dir: str,
    training_seed: int,
    n_envs: int,
    total_steps: int,
    resume_path: str | None,
    device: str,
    extra: dict[str, Any] | None = None,
) -> str:
    """Write repro_run_manifest.json next to the model. Returns path."""
    try:
        import torch
        import stable_baselines3 as sb3

        torch_v = torch.__version__
    except Exception:
        torch_v = "unknown"
    # SUMO version from binary if available
    sumo_v = "unknown"
    import shutil

    if shutil.which("sumo"):
        import subprocess

        try:
            sumo_v = (
                subprocess.check_output(["sumo", "--version"], text=True, timeout=5)
                .splitlines()[0]
            )
        except Exception:
            pass

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "training_seed": training_seed,
        "sumo_seeds_subproc": [training_seed + r for r in range(n_envs)],
        "n_envs": n_envs,
        "total_timesteps": total_steps,
        "resume_from": resume_path,
        "device": device,
        "torch": torch_v,
        "stable_baselines3": getattr(
            __import__("stable_baselines3", fromlist=["__version__"]),
            "__version__",
            "unknown",
        ),
        "sumo_cli_version": sumo_v,
    }
    if extra:
        payload["extra"] = extra

    path = os.path.join(out_dir, "repro_run_manifest.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path
