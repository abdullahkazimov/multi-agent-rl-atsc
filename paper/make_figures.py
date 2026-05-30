"""
make_figures.py
===============
Generates publication-quality (vector PDF) figures for the AICT 2026 paper
from the real evaluation data produced by the project.

Outputs → paper/figures/*.pdf
  fig_topologies.pdf  : schematic of the four Baku networks
  fig_training.pdf    : waiting-time improvement vs training timesteps
  fig_improvement.pdf : grouped bar — per-scenario % improvement (4 metrics)
  fig_timeseries.pdf  : FT vs RL waiting time over a representative episode
  fig_perseed.pdf     : per-seed waiting-time improvement distribution
  fig_sota.pdf        : SOTA comparison (% improvement over fixed-time)
"""
from __future__ import annotations
import os, csv, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle

# ── Global style (match LaTeX / IEEE) ────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "font.size":        8,
    "axes.titlesize":   8.5,
    "axes.labelsize":   8,
    "xtick.labelsize":  7,
    "ytick.labelsize":  7,
    "legend.fontsize":  7,
    "axes.linewidth":   0.6,
    "grid.linewidth":   0.4,
    "lines.linewidth":  1.1,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.02,
})

COL_FT   = "#d9700a"   # orange (fixed-time)
COL_RL   = "#0f9d63"   # green  (PPO)
COL_BLUE = "#1f6fb4"
COL_PURP = "#7b54c4"
COL_GRID = "#cccccc"

SCEN_ORDER  = ["bottleneck", "main", "pedestrian", "hexagon"]
SCEN_LABEL  = {
    "bottleneck": "Bottleneck\n(2 TL)",
    "main":       "Main\n(3 TL)",
    "pedestrian": "Pedestrian\n(1 TL)",
    "hexagon":    "Hexagon\n(12 TL)",
}
SCEN_SHORT  = {"bottleneck": "Bottleneck", "main": "Main",
               "pedestrian": "Pedestrian", "hexagon": "Hexagon"}

COL_W  = 3.45   # IEEE single column width (inches)
COL_2W = 7.16   # IEEE double column width (inches)

DATA = "website/data"   # overridden in __main__ via --data
FIG  = "paper/figures"


def load_manifest():
    return json.load(open(os.path.join(DATA, "manifest.json")))


# ════════════════════════════════════════════════════════════════════════════
#  Fig 1 — Network topologies (schematic)
# ════════════════════════════════════════════════════════════════════════════
def fig_topologies():
    fig, axes = plt.subplots(1, 4, figsize=(COL_2W, 1.85))

    def node(ax, x, y, label, r=0.13, c="#2b6cb0"):
        ax.add_patch(Circle((x, y), r, color=c, ec="white", lw=0.8, zorder=3))
        ax.text(x, y, label, ha="center", va="center", color="white",
                fontsize=5.5, fontweight="bold", zorder=4)

    def link(ax, x1, y1, x2, y2):
        ax.plot([x1, x2], [y1, y2], color="#888", lw=2.4, zorder=1,
                solid_capstyle="round")

    # ── Bottleneck: 2 nodes, squeeze ──
    ax = axes[0]
    link(ax, 0.30, 0.5, 0.70, 0.5)
    for sx, sy, tx, ty in [(0.05,0.5,0.30,0.5),(0.30,0.85,0.30,0.5),
                            (0.70,0.15,0.70,0.5),(0.95,0.5,0.70,0.5)]:
        ax.plot([sx,tx],[sy,ty], color="#bbb", lw=1.0, ls=":", zorder=0)
    node(ax, 0.30, 0.5, "H", c="#c0392b")
    node(ax, 0.70, 0.5, "S", c="#c0392b")
    ax.set_title("(a) Bottleneck\n2 TL, 3000 veh/h", fontsize=7)

    # ── Main: 3 nodes in a line (arterial) ──
    ax = axes[1]
    xs = [0.20, 0.50, 0.80]
    for i in range(len(xs)-1):
        link(ax, xs[i], 0.5, xs[i+1], 0.5)
    for x in xs:
        ax.plot([x,x],[0.16,0.84], color="#bbb", lw=1.0, ls=":", zorder=0)
    for x, lab in zip(xs, ["I1","I2","I4"]):
        node(ax, x, 0.5, lab)
    ax.set_title("(b) Main arterial\n3 TL, 1330 veh/h", fontsize=7)

    # ── Pedestrian: single node with crossings ──
    ax = axes[2]
    for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
        ax.plot([0.5,0.5+dx*0.38],[0.5,0.5+dy*0.38], color="#888", lw=2.4, zorder=1)
    # pedestrian crossing hashes
    for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
        ax.scatter([0.5+dx*0.30],[0.5+dy*0.30], s=14, color=COL_PURP, marker="s", zorder=2)
    node(ax, 0.5, 0.5, "C", r=0.16, c="#8e44ad")
    ax.set_title("(c) Pedestrian\n1 TL, 1000 veh/h", fontsize=7)

    # ── Hexagon: 6 outer + 6 inner ──
    ax = axes[3]
    import math
    outer, inner = [], []
    for k in range(6):
        a = math.pi/2 - k*math.pi/3
        outer.append((0.5+0.40*math.cos(a), 0.5+0.40*math.sin(a)))
        a2 = math.pi/2 - k*math.pi/3 - math.pi/6
        inner.append((0.5+0.20*math.cos(a2), 0.5+0.20*math.sin(a2)))
    for i in range(6):
        link(ax, *outer[i], *outer[(i+1)%6])
    for i in range(6):
        link(ax, *inner[i], *inner[(i+1)%6])
        link(ax, *outer[i], *inner[i])
    for (x,y) in outer:
        node(ax, x, y, "", r=0.075, c="#2b6cb0")
    for (x,y) in inner:
        node(ax, x, y, "", r=0.055, c="#3aa0d0")
    ax.set_title("(d) Hexagon highway\n12 TL, 2600 veh/h", fontsize=7)

    for ax in axes:
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(0.0, 1.05)
        ax.set_aspect("equal"); ax.axis("off")

    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIG, "fig_topologies.pdf"))
    plt.close(fig)
    print("  fig_topologies.pdf")


# ════════════════════════════════════════════════════════════════════════════
#  Fig 2 — Training convergence (waiting-time improvement vs timestep)
# ════════════════════════════════════════════════════════════════════════════
def fig_training():
    fig, ax = plt.subplots(figsize=(COL_W, 2.35))
    colors = {"bottleneck": COL_FT, "main": COL_BLUE,
              "pedestrian": COL_PURP, "hexagon": COL_RL}

    for sc in SCEN_ORDER:
        path = os.path.join("models", sc, "eval_progress.csv")
        if not os.path.exists(path):
            continue
        ts, imp = [], []
        with open(path) as f:
            for row in csv.DictReader(f):
                ts.append(int(row["timestep"]) / 1000.0)
                imp.append(float(row["imp_waiting_time"]))
        imp = np.clip(imp, -100, 110)   # clip early-training huge negatives for readability
        ax.plot(ts, imp, color=colors[sc], label=SCEN_SHORT[sc], lw=1.2)

    ax.axhline(0, color="#999", lw=0.7, ls="--")
    ax.set_xlabel("Training timesteps ($\\times 10^3$)")
    ax.set_ylabel("Waiting-time improvement (\\%)")
    ax.set_ylim(-105, 115)
    ax.grid(True, color=COL_GRID, alpha=0.6)
    ax.legend(loc="lower right", ncol=2, frameon=True, framealpha=0.9)
    ax.text(0.02, 0.96, "(clipped at $-100\\%$ for readability)",
            transform=ax.transAxes, fontsize=5.5, color="#888", va="top")
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIG, "fig_training.pdf"))
    plt.close(fig)
    print("  fig_training.pdf")


# ════════════════════════════════════════════════════════════════════════════
#  Fig 3 — Grouped bar of per-scenario improvement (4 metrics)
# ════════════════════════════════════════════════════════════════════════════
def fig_improvement(m):
    metrics = ["waiting_time", "queue_length", "stopped_ratio", "throughput"]
    mlabel  = ["Waiting time", "Queue length", "Stopped ratio", "Throughput"]
    mcolor  = [COL_RL, COL_BLUE, COL_PURP, COL_FT]

    fig, ax = plt.subplots(figsize=(COL_2W, 2.5))
    x = np.arange(len(SCEN_ORDER))
    w = 0.2
    for i, met in enumerate(metrics):
        vals = [m["scenarios"][sc]["mean_improvement"][met] for sc in SCEN_ORDER]
        bars = ax.bar(x + (i-1.5)*w, vals, w, label=mlabel[i], color=mcolor[i],
                      edgecolor="white", lw=0.4)
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, v + (1.5 if v>=0 else -3.5),
                    f"{v:+.0f}", ha="center", va="bottom" if v>=0 else "top",
                    fontsize=5.2)

    ax.axhline(0, color="#666", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([SCEN_LABEL[s] for s in SCEN_ORDER])
    ax.set_ylabel("Improvement over fixed-time (\\%)")
    ax.set_ylim(-12, 112)
    ax.grid(True, axis="y", color=COL_GRID, alpha=0.6)
    ax.legend(loc="upper left", ncol=4, frameon=True, framealpha=0.9,
              columnspacing=1.0, handletextpad=0.4)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIG, "fig_improvement.pdf"))
    plt.close(fig)
    print("  fig_improvement.pdf")


# ════════════════════════════════════════════════════════════════════════════
#  Fig 4 — FT vs RL waiting-time time series (representative scenarios)
# ════════════════════════════════════════════════════════════════════════════
def fig_timeseries():
    fig, axes = plt.subplots(1, 2, figsize=(COL_2W, 2.3))
    for ax, sc in zip(axes, ["main", "hexagon"]):
        d = json.load(open(os.path.join(DATA, f"{sc}_s42.json")))
        ft = np.array(d["fixed_time"]["waiting_time"])
        rl = np.array(d["ppo_best"]["waiting_time"])
        n  = min(len(ft), len(rl))
        t  = np.arange(n)
        warm = d.get("warmup", 30)

        ax.fill_between(t, rl[:n], ft[:n], where=(ft[:n] >= rl[:n]),
                        color=COL_RL, alpha=0.18, interpolate=True, lw=0)
        ax.fill_between(t, rl[:n], ft[:n], where=(ft[:n] < rl[:n]),
                        color="#d64545", alpha=0.18, interpolate=True, lw=0)
        ax.plot(t, ft[:n], color=COL_FT, label="Fixed-time", lw=1.1)
        ax.plot(t, rl[:n], color=COL_RL, label="PPO (best)", lw=1.1)
        ax.axvspan(0, warm, color="#000", alpha=0.05, lw=0)
        ax.axvline(warm, color="#999", ls=":", lw=0.7)

        ax.set_title(f"{SCEN_SHORT[sc]} — seed 42", fontsize=7.5)
        ax.set_xlabel("Decision step (10 s each)")
        ax.grid(True, color=COL_GRID, alpha=0.6)
        if sc == "main":
            ax.set_ylabel("Avg. waiting time (s)")
            ax.legend(loc="upper left", frameon=True, framealpha=0.9)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(FIG, "fig_timeseries.pdf"))
    plt.close(fig)
    print("  fig_timeseries.pdf")


# ════════════════════════════════════════════════════════════════════════════
#  Fig 5 — Per-seed waiting-time improvement distribution
# ════════════════════════════════════════════════════════════════════════════
def fig_perseed(m):
    fig, ax = plt.subplots(figsize=(COL_W, 2.35))
    data, labels = [], []
    for sc in SCEN_ORDER:
        vals = [s["improvement"]["waiting_time"]
                for s in m["scenarios"][sc]["seed_summaries"]]
        data.append(vals)
        labels.append(SCEN_SHORT[sc])

    bp = ax.boxplot(data, patch_artist=True, widths=0.55,
                    medianprops=dict(color="black", lw=1.0),
                    flierprops=dict(marker="o", markersize=2.5, alpha=0.6))
    colors = [COL_FT, COL_BLUE, COL_PURP, COL_RL]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.55); patch.set_edgecolor(c)
    # overlay individual seeds
    for i, vals in enumerate(data):
        jitter = np.random.RandomState(0).normal(0, 0.05, len(vals))
        ax.scatter(np.full(len(vals), i+1)+jitter, vals, s=8,
                   color="black", alpha=0.5, zorder=3)

    ax.set_xticklabels(labels)
    ax.set_ylabel("Waiting-time improvement (\\%)")
    ax.set_ylim(0, 108)
    ax.grid(True, axis="y", color=COL_GRID, alpha=0.6)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIG, "fig_perseed.pdf"))
    plt.close(fig)
    print("  fig_perseed.pdf")


# ════════════════════════════════════════════════════════════════════════════
#  Fig 6 — SOTA comparison (% improvement over fixed-time)
# ════════════════════════════════════════════════════════════════════════════
def fig_sota(m):
    # "Ours" waiting-time improvements are read from the active manifest so the
    # figure always matches the evaluation protocol used to build the tables.
    ours = {sc: m["scenarios"][sc]["mean_improvement"]["waiting_time"]
            for sc in SCEN_ORDER}
    bars = [
        ("Ours: Hexagon (12 TL)",         ours["hexagon"],    True),
        ("Ours: Main (3 TL)",             ours["main"],       True),
        ("Mahato 2025, 2$\\times$2 grid",  78.3,              False),
        ("Ours: Bottleneck (2 TL)",       ours["bottleneck"], True),
        ("Fed-PPO 2025, arterial",        37.0,               False),
        ("IDQN, Grid 4$\\times$4$^\\dagger$",   28.7,         False),
        ("MA-PPO 2025, 7-TL$^\\dagger$",   24.0,              False),
        ("Ours: Pedestrian (1 TL)",       ours["pedestrian"], True),
        ("IDQN, Ingolstadt$^\\dagger$",    17.7,              False),
        ("FMA2C, Ingolstadt$^\\dagger$",   14.4,              False),
        ("FMA2C, Cologne$^\\dagger$",     -15.9,              False),
        ("MPLight, Cologne$^\\dagger$",   -45.6,              False),
    ]
    bars.sort(key=lambda b: b[1])
    labels = [b[0] for b in bars]
    vals   = [b[1] for b in bars]
    ours   = [b[2] for b in bars]

    fig, ax = plt.subplots(figsize=(COL_2W, 3.0))
    y = np.arange(len(bars))
    colors = []
    for v, o in zip(vals, ours):
        if o:        colors.append(COL_RL)
        elif v < 0:  colors.append("#d64545")
        else:        colors.append("#9aa7b5")
    ax.barh(y, vals, color=colors, edgecolor="white", lw=0.4, height=0.7)
    for yi, v in zip(y, vals):
        ax.text(v + (1.2 if v >= 0 else -1.2), yi, f"{v:+.1f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=6)
    ax.axvline(0, color="#666", lw=0.7)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel("Improvement over fixed-time (\\%)  —  $\\dagger$ = travel time, others waiting time")
    ax.set_xlim(-58, 115)
    ax.grid(True, axis="x", color=COL_GRID, alpha=0.6)
    # legend proxies
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color=COL_RL, label="This work"),
        Patch(color="#9aa7b5", label="SOTA (RL better)"),
        Patch(color="#d64545", label="SOTA (RL worse than FT)"),
    ], loc="lower right", frameon=True, framealpha=0.9)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIG, "fig_sota.pdf"))
    plt.close(fig)
    print("  fig_sota.pdf")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="website/data",
                   help="directory with manifest.json + {sc}_s{seed}.json")
    p.add_argument("--out",  default="paper/figures", help="output directory")
    args = p.parse_args()
    DATA = args.data
    FIG  = args.out
    os.makedirs(FIG, exist_ok=True)

    m = load_manifest()
    print(f"Generating figures  (data={DATA}) →", FIG)
    fig_topologies()
    fig_training()
    fig_improvement(m)
    fig_timeseries()
    fig_perseed(m)
    fig_sota(m)
    print("Done.")
