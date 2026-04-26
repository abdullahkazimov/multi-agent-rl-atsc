"""
7-term composite reward function (thesis Eq. 1).

R_t = +β_tp · Δthr_t
      - β_sr · ρ̄_t
      - β_wt · (w̄_t / D_w)
      - β_ql · (q̄_t / D_q)
      - β_lc · ρ^max_t
      - β_cc · (ρ^max_t − ρ̄_t)²
      - β_st · ψ_t
      - β_sw · n^sw_t
      + b_0

This module exposes a standalone RewardFunction class that can be used
independently of BakuSUMOEnv (e.g. for unit testing and ablation studies).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np


@dataclass
class RewardComponents:
    """Holds the individual terms of the reward for inspection/analysis."""
    throughput:    float = 0.0
    stopped_ratio: float = 0.0
    waiting_time:  float = 0.0
    queue_length:  float = 0.0
    hotspot:       float = 0.0
    coord_balance: float = 0.0
    starvation:    float = 0.0
    switching:     float = 0.0
    baseline:      float = 0.0

    @property
    def total(self) -> float:
        return (
              self.throughput
            - self.stopped_ratio
            - self.waiting_time
            - self.queue_length
            - self.hotspot
            - self.coord_balance
            - self.starvation
            - self.switching
            + self.baseline
        )


class RewardFunction:
    """
    Standalone 7-term composite reward, parameterised by the thesis weights.

    Parameters
    ----------
    beta_tp, beta_sr, … : reward weights (see thesis Table 2 / §3.4)
    d_w, d_q            : normalisation denominators for wait time and queue
    b0                  : B2 baseline offset (negative mean FT reward/step)
    starvation_max      : phase-age threshold for starvation penalty
    """

    def __init__(
        self,
        beta_tp: float = 0.15,
        beta_sr: float = 0.30,
        beta_wt: float = 0.55,
        beta_ql: float = 0.12,
        beta_lc: float = 0.25,
        beta_cc: float = 0.10,
        beta_st: float = 0.08,
        beta_sw: float = 0.05,
        d_w:     float = 35.0,
        d_q:     float = 10.0,
        b0:      float = 0.0,
        starvation_max: int = 15,
    ):
        self.beta_tp = beta_tp
        self.beta_sr = beta_sr
        self.beta_wt = beta_wt
        self.beta_ql = beta_ql
        self.beta_lc = beta_lc
        self.beta_cc = beta_cc
        self.beta_st = beta_st
        self.beta_sw = beta_sw
        self.d_w     = d_w
        self.d_q     = d_q
        self.b0      = b0
        self.starvation_max = starvation_max

    def __call__(
        self,
        arrived:       int,
        stopped_ratios: List[float],
        wait_times:    List[float],
        queue_lengths: List[float],
        tl_ratios:     Dict[str, List[float]],
        phase_ages:    Dict[str, np.ndarray],
        n_switches:    int,
    ) -> float:
        components = self.compute(
            arrived, stopped_ratios, wait_times,
            queue_lengths, tl_ratios, phase_ages, n_switches,
        )
        return components.total

    def compute(
        self,
        arrived:        int,
        stopped_ratios: List[float],
        wait_times:     List[float],
        queue_lengths:  List[float],
        tl_ratios:      Dict[str, List[float]],
        phase_ages:     Dict[str, np.ndarray],
        n_switches:     int,
    ) -> RewardComponents:
        """Return a RewardComponents object with each term filled in."""

        if not stopped_ratios:
            return RewardComponents(baseline=self.b0)

        mean_sr = float(np.mean(stopped_ratios))
        mean_wt = float(np.mean(wait_times))
        mean_ql = float(np.mean(queue_lengths))

        tl_mean_sr = {
            tl: float(np.mean(v)) if v else 0.0
            for tl, v in tl_ratios.items()
        }
        max_sr = max(tl_mean_sr.values()) if tl_mean_sr else 0.0

        # Starvation ψ_t — normalised excess age
        n_tls  = max(1, len(phase_ages))
        excess = sum(
            np.maximum(0, ages - self.starvation_max).sum()
            for ages in phase_ages.values()
        )
        psi = min(1.0, excess / (self.starvation_max * n_tls))

        return RewardComponents(
            throughput    =  self.beta_tp * arrived,
            stopped_ratio =  self.beta_sr * mean_sr,
            waiting_time  =  self.beta_wt * (mean_wt / self.d_w),
            queue_length  =  self.beta_ql * (mean_ql / self.d_q),
            hotspot       =  self.beta_lc * max_sr,
            coord_balance =  self.beta_cc * (max_sr - mean_sr) ** 2,
            starvation    =  self.beta_st * psi,
            switching     =  self.beta_sw * n_switches,
            baseline      =  self.b0,
        )

    def per_sigma_gradients(self, sigmas: Dict[str, float]) -> Dict[str, float]:
        """
        Compute per-σ gradient magnitude g_j = β_j · σ(f_j) for each term.
        Used in the weight-derivation analysis (thesis §3.4.1).
        """
        mapping = {
            "throughput":    (self.beta_tp, sigmas.get("throughput",    1.0)),
            "stopped_ratio": (self.beta_sr, sigmas.get("stopped_ratio", 1.0)),
            "waiting_time":  (self.beta_wt, sigmas.get("waiting_time",  1.0) / self.d_w),
            "queue_length":  (self.beta_ql, sigmas.get("queue_length",  1.0) / self.d_q),
            "hotspot":       (self.beta_lc, sigmas.get("hotspot",       1.0)),
            "coord_balance": (self.beta_cc, sigmas.get("coord_balance", 1.0)),
            "starvation":    (self.beta_st, sigmas.get("starvation",    1.0)),
        }
        return {name: beta * sigma for name, (beta, sigma) in mapping.items()}


def compute_reward_components(
    arrived:        int,
    stopped_ratios: List[float],
    wait_times:     List[float],
    queue_lengths:  List[float],
    tl_ratios:      Dict[str, List[float]],
    phase_ages:     Dict[str, np.ndarray],
    n_switches:     int,
    b0:             float = 0.0,
) -> RewardComponents:
    """Convenience wrapper around RewardFunction.compute()."""
    rf = RewardFunction(b0=b0)
    return rf.compute(
        arrived, stopped_ratios, wait_times,
        queue_lengths, tl_ratios, phase_ages, n_switches,
    )
