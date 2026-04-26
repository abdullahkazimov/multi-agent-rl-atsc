#!/usr/bin/env bash
# run_all.sh — Full training + evaluation pipeline for all four Baku scenarios.
#
# Usage:
#   bash run_all.sh              # train all four, then evaluate
#   bash run_all.sh bottleneck   # train + evaluate only bottleneck
#   SKIP_TRAIN=1 bash run_all.sh # evaluate only (requires pre-trained models)
#
# The script exits immediately on any error (set -e).

set -e

SCENARIOS="${1:-bottleneck main pedestrian hexagon}"
STEPS="${STEPS:-500000}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

export SUMO_HOME="${SUMO_HOME:-/usr/share/sumo}"
export PYTHONPATH="$(pwd):${PYTHONPATH}"

echo "============================================================"
echo "  Baku MARL-ATSC  —  Run-all pipeline"
echo "  Scenarios : ${SCENARIOS}"
echo "  Steps     : ${STEPS}"
echo "  SUMO HOME : ${SUMO_HOME}"
echo "============================================================"

# ── Step 0: Quick smoke-test ─────────────────────────────────────────────────
echo ""
echo "[ 0/3 ] Running integration smoke-test…"
python test_run.py
echo ""

# ── Step 1: Training ─────────────────────────────────────────────────────────
if [ "${SKIP_TRAIN}" = "0" ]; then
    echo "[ 1/3 ] Training PPO agents…"
    for scenario in ${SCENARIOS}; do
        echo ""
        echo "  — Training: ${scenario}"
        python train.py --scenario "${scenario}" --steps "${STEPS}"
    done
else
    echo "[ 1/3 ] Skipping training (SKIP_TRAIN=1)"
fi

# ── Step 2: Reward analysis ──────────────────────────────────────────────────
echo ""
echo "[ 2/3 ] Running reward gradient analysis…"
for scenario in ${SCENARIOS}; do
    echo "  — Analysis: ${scenario}"
    python analyze_reward.py --scenario "${scenario}" --steps 100
done

# ── Step 3: Evaluation ───────────────────────────────────────────────────────
echo ""
echo "[ 3/3 ] Evaluating PPO vs fixed-time…"
for scenario in ${SCENARIOS}; do
    echo "  — Evaluating: ${scenario}"
    python evaluate.py --scenario "${scenario}"
done

echo ""
echo "============================================================"
echo "  All done!  Results are in ./results/"
echo "============================================================"
