#!/usr/bin/env python3
"""Generate the blog figures from real wandb data.

Run locally (needs network + matplotlib):
  cd taidnguyen.github.io && WANDB_API_KEY=... \
    uv run --no-project --with wandb --with matplotlib --with pandas \
    python assets/make_plots.py

Writes PNGs into assets/images/.
"""
import math
import os
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ENTITY_PROJECT = "taidnguyen/talkie-self-play"
OUT = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(OUT, exist_ok=True)

# --- tasteful, non-default style -------------------------------------------
plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#888",
    "axes.linewidth": 0.8,
    "axes.titlesize": 13,
    "axes.titleweight": "regular",
    "xtick.color": "#555", "ytick.color": "#555",
    "axes.labelcolor": "#333",
    "grid.color": "#e8e8e8", "grid.linewidth": 0.8,
})
ORACLE, SELF = "#2f6fb5", "#b5402f"
ARITH_KEY = "eval/arith_passN_rate"   # held-out arithmetic, pass@N (matches the post)


def get_run(name):
    runs = list(wandb.Api().runs(ENTITY_PROJECT, filters={"display_name": name}))
    if not runs:
        raise SystemExit(f"run not found: {name}")
    return runs[0]


def arith_series(run):
    """(_step, arith eval) pairs, NaNs dropped. Pinned to one key for all runs."""
    df = run.history(samples=5000)
    key = ARITH_KEY if ARITH_KEY in df.columns else None
    if key is None:  # fallback: shortest plain-arith column
        cands = [c for c in df.columns if "arith" in c.lower()
                 and "mmlu" not in c.lower() and "mcq" not in c.lower()
                 and "_mc" not in c.lower()]
        key = sorted(cands, key=len)[0] if cands else None
    if key is None:
        raise SystemExit("no arith eval column; have: "
                         + str([c for c in df.columns if "arith" in c.lower()]))
    print("  arith key:", key)
    d = df[["_step", key]].dropna()
    return d["_step"].tolist(), d[key].tolist()


def metric_series(run, *candidates):
    """(_step, value) for the first matching per-step metric key."""
    df = run.history(samples=8000)
    key = next((c for c in candidates if c in df.columns), None)
    if key is None:  # fuzzy: all tokens of the first candidate present
        toks = candidates[0].split("/")[-1].split("_")
        key = next((c for c in df.columns
                    if all(t in c.lower() for t in toks)), None)
    if key is None:
        raise SystemExit(f"metric not found: {candidates}")
    print("  metric:", key)
    d = df[["_step", key]].dropna()
    return d["_step"].tolist(), d[key].tolist()


def smooth(ys, w=6):
    """Trailing moving average for readable training curves."""
    out = []
    for i in range(len(ys)):
        seg = ys[max(0, i - w + 1):i + 1]
        out.append(sum(seg) / len(seg))
    return out


# --- figure 1: oracle vs self (real) ---------------------------------------
print("oracle vs self ...")
ox, oy = arith_series(get_run("arith_oracle_cpn_20260613"))
sx, sy = arith_series(get_run("arith_self_cpn_20260613"))
fig, ax = plt.subplots(figsize=(5.2, 3.2))
ax.plot(ox, oy, "-o", color=ORACLE, lw=2, ms=5, label="oracle (computed gold)")
ax.plot(sx, sy, "-o", color=SELF, lw=2, ms=5, label="self-grade (model grades itself)")
ax.set_xlabel("training step"); ax.set_ylabel("held-out arithmetic")
ax.set_ylim(0.2, 0.8); ax.grid(True, axis="y")
ax.legend(frameon=False, fontsize=10, loc="lower left")
ax.annotate(f"{oy[-1]:.2f}", (ox[-1], oy[-1]), color=ORACLE,
            fontsize=10, xytext=(6, 0), textcoords="offset points", va="center")
ax.annotate(f"{sy[-1]:.2f}", (sx[-1], sy[-1]), color=SELF,
            fontsize=10, xytext=(6, 0), textcoords="offset points", va="center")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "oracle_vs_self.png"), bbox_inches="tight")

# --- figure 2: variance reward (formula) -----------------------------------
print("variance reward ...")
ps = [i / 400 for i in range(401)]
rs = [math.exp(-((p * (1 - p) - 0.25) ** 2) / (2 * 0.01)) for p in ps]
fig, ax = plt.subplots(figsize=(5.2, 3.0))
ax.plot(ps, rs, color=SELF, lw=2.2)
ax.scatter([0.5], [1.0], color=SELF, zorder=5)
ax.annotate("frontier", (0.5, 1.0), color=SELF, fontsize=10,
            xytext=(0, 6), textcoords="offset points", ha="center")
ax.text(0.04, 0.10, "too hard", color="#999", fontsize=9)
ax.text(0.84, 0.10, "too easy", color="#999", fontsize=9)
ax.set_xlabel("Reasoner pass rate  p"); ax.set_ylabel("Challenger reward")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.08)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "variance_reward.png"), bbox_inches="tight")

# --- figure 3: capability climb (Results hero) ----------------------------
print("capability climb ...")
cx, cy = arith_series(get_run("arith_oracle_cpn_lr2e5_20260612"))
fig, ax = plt.subplots(figsize=(5.2, 3.2))
ax.plot(cx, cy, "-o", color=ORACLE, lw=2, ms=4)
ax.set_xlabel("training step"); ax.set_ylabel("held-out arithmetic")
ax.set_ylim(0.3, 0.8); ax.grid(True, axis="y")
ax.annotate(f"{cy[0]:.2f}", (cx[0], cy[0]), color="#999", fontsize=10,
            xytext=(0, -14), textcoords="offset points", ha="center")
ax.annotate(f"{cy[-1]:.2f}", (cx[-1], cy[-1]), color=ORACLE, fontsize=11,
            xytext=(6, 0), textcoords="offset points", va="center")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "capability_climb.png"), bbox_inches="tight")

# --- figure 4: gold integrity / collusion (self run, the mechanism) --------
print("gold integrity ...")
s = get_run("arith_self_cpn_20260613")
cx2, cy2 = metric_series(s, "train/collusion_rate", "collusion_rate")
gx, gy = metric_series(s, "train/gold_accuracy", "gold_accuracy")
fig, ax = plt.subplots(figsize=(5.2, 3.2))
ax.plot(cx2, cy2, color=SELF, lw=2, label="collusion rate")
ax.plot(gx, gy, color="#888", lw=2, label="gold accuracy")
ax.set_xlabel("training step"); ax.set_ylim(0, 1); ax.grid(True, axis="y")
ax.legend(frameon=False, fontsize=10, loc="center right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "gold_integrity.png"), bbox_inches="tight")

# --- figure 5: training dynamics (r_C, r_R, frontier on one axis) ----------
print("training dynamics ...")
run = get_run("arith_oracle_cpn_lr2e5_20260612")
lines = [
    ("train/r_C_mean", "Challenger reward", ORACLE),
    ("train/r_R_mean", "Reasoner reward", "#2e8b57"),
    ("train/frontier_rate", "frontier rate", "#888"),
]
fig, ax = plt.subplots(figsize=(7.8, 3.9))
for key, label, color in lines:
    xs, ys = metric_series(run, key, key.split("/")[-1])
    ax.plot(xs, smooth(ys), color=color, lw=2.2, label=label)
ax.set_xlabel("training step"); ax.set_ylabel("reward / rate")
ax.set_ylim(0, 1); ax.grid(True, axis="y")
ax.legend(frameon=False, fontsize=11, loc="best")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "training_dynamics.png"), bbox_inches="tight")

# --- eval table values: real step-0 / last-step numbers to paste in --------
print("eval table values ...")
def dump_eval_table(name):
    r = get_run(name)
    df = r.history(samples=8000)
    keep = ("arith_pass", "arith_mcq_acc", "arith_mcq_pgold", "humaneval_pass",
            "morse_pass", "vigenere_pass", "anagram_pass")
    cols = [c for c in df.columns if c.startswith("eval/") and any(t in c for t in keep)]
    print(f"  run: {name}")
    for c in sorted(cols):
        d = df[["_step", c]].dropna()
        if len(d):
            print(f"    {c:34s} step {int(d['_step'].iloc[0])}={d[c].iloc[0]:.3f}"
                  f"  ->  step {int(d['_step'].iloc[-1])}={d[c].iloc[-1]:.3f}")
dump_eval_table("arith_oracle_cpn_lr2e5_20260612")

# --- figure 6: self-guidance gate vs unguided scaling (arith + collusion) ---
# Axis-5 test: does a frozen-base gate stop collusion from compounding, and
# does held-out arith stop eroding? Unguided multi-corpus self (self_prose)
# runs away to ~0.99; the gate run (self_gate) should hold collusion low.
print("self-guidance gate ...")
try:
    pr = get_run("arith_self_prose_20260613")          # unguided, 4 corpora
    gt = get_run("self_gate_20260614")                 # frozen-base gate
    # left: held-out arithmetic, unguided vs gated
    px, py = arith_series(pr)
    gx, gy = arith_series(gt)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(px, py, "-o", color=SELF, lw=2, ms=4, label="unguided self (4 corpora)")
    ax.plot(gx, gy, "-o", color=ORACLE, lw=2, ms=4, label="self + frozen-base gate")
    ax.set_xlabel("training step"); ax.set_ylabel("held-out arithmetic")
    ax.set_ylim(0.2, 0.8); ax.grid(True, axis="y")
    ax.legend(frameon=False, fontsize=10, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "gate_arith.png"), bbox_inches="tight")
    # right: collusion rate, unguided vs gated
    cpx, cpy = metric_series(pr, "train/collusion_rate", "collusion_rate")
    cgx, cgy = metric_series(gt, "train/collusion_rate", "collusion_rate")
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(cpx, smooth(cpy), color=SELF, lw=2, label="unguided self (4 corpora)")
    ax.plot(cgx, smooth(cgy), color=ORACLE, lw=2, label="self + frozen-base gate")
    ax.set_xlabel("training step"); ax.set_ylabel("collusion rate")
    ax.set_ylim(0, 1); ax.grid(True, axis="y")
    ax.legend(frameon=False, fontsize=10, loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "self_guidance_gate.png"), bbox_inches="tight")
except SystemExit as e:
    print("  skipped gate figures:", e)

print("done ->", OUT)
