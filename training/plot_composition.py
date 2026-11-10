"""
plot_composition.py  -  wood composition (share of the wood per tissue), true vs predicted

Reads the volume_per_vine.csv that --replot writes in every run folder and
draws composition_per_group.png, composition_per_vine.png and
composition_difference_per_vine.png in the output folder.

USAGE
    python training/plot_composition.py <run folder>/volume_per_vine.csv <run folder> "run name"
"""
import csv, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src, out_dir = sys.argv[1], sys.argv[2]
RUN = sys.argv[3] if len(sys.argv) > 3 else "U-Net"
NAMES = ["healthy", "necrosis", "white_rot"]
LABELS = ["Healthy", "Necrosis", "White rot"]
COLORS = ["#1B5E20", "#F9A825", "#B71C1C"]   # healthy, necrosis, white rot
INK, MUTED, GRID = "#1F2421", "#5E6862", "#E4E7E5"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": INK})

vines = []
with open(src) as f:
    for r in csv.DictReader(f):
        vines.append({"name": r["vine"], "group": r["group"],
                      "true": [float(r[f"{n}_true_share_pct"]) for n in NAMES],
                      "pred": [float(r[f"{n}_pred_share_pct"]) for n in NAMES]})
groups = []                                   # mean share per plant group, in file order
for g in dict.fromkeys(v["group"] for v in vines):
    vs = [v for v in vines if v["group"] == g]
    groups.append({"name": f"{g} (n={len(vs)})", "group": g,
                   "true": [sum(v["true"][c] for v in vs) / len(vs) for c in range(3)],
                   "pred": [sum(v["pred"][c] for v in vs) / len(vs) for c in range(3)]})

def stacked(ax, rows, title):
    y, ticks, labels = 0, [], []
    gap_rows = []
    for i, r in enumerate(rows):
        for kind, yy in (("true", y), ("pred", y + 0.42)):
            left = 0
            for c in range(3):
                v = r[kind][c]
                ax.barh(yy, v, left=left, height=0.38, color=COLORS[c], edgecolor="white", linewidth=2)
                if v >= 6:
                    ax.text(left + v / 2, yy, f"{v:.0f}%", ha="center", va="center",
                            color=INK if c == 1 else "white", fontsize=9.5, fontweight="bold")
                left += v
            ax.text(-1.5, yy, "true" if kind == "true" else "predicted", ha="right", va="center",
                    fontsize=9, color=MUTED)
        ticks.append(y + 0.21); labels.append(r["name"])
        y += 1.25
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=11, fontweight="bold")
    ax.tick_params(axis="y", pad=62, length=0)
    ax.invert_yaxis()
    ax.set_xlim(0, 100); ax.set_xlabel("share of the wood (%)")
    ax.set_xticks(range(0, 101, 20)); ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold", pad=34)

def legend(fig, ax):
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS]
    ax.legend(handles, LABELS, ncol=3, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0),
              fontsize=10.5, handlelength=1.2)

# per group
fig, ax = plt.subplots(figsize=(11, 7.2))
stacked(ax, groups, "Wood composition: true vs predicted, per group")
legend(fig, ax)
fig.text(0.01, 0.01, f"{RUN}, 3 folds, every vine tested once by a model that never saw it. "
         "Shares = each tissue's total thickness / total wood thickness, about its share of the scanned volume.",
         fontsize=8.5, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(f"{out_dir}/composition_per_group.png", dpi=150); plt.close(fig)

# per vine
fig, ax = plt.subplots(figsize=(11, 15))
stacked(ax, vines, "Wood composition: true vs predicted, per vine")
legend(fig, ax)
y = 0
prev = None
for i, r in enumerate(vines):
    if r["group"] != prev:
        ax.text(101, i * 1.25 + 0.21, r["group"], va="center", fontsize=10, color=MUTED, style="italic")
        prev = r["group"]
fig.text(0.01, 0.005, f"{RUN}, 3 folds, every vine tested once by a model that never saw it.",
         fontsize=8.5, color=MUTED)
fig.tight_layout(rect=(0, 0.015, 0.93, 1))
fig.savefig(f"{out_dir}/composition_per_vine.png", dpi=150); plt.close(fig)

# difference, per vine
fig, ax = plt.subplots(figsize=(11, 7))
n = len(vines); w = 0.26
for c in range(3):
    d = [r["pred"][c] - r["true"][c] for r in vines]
    ax.bar([i + (c - 1) * w for i in range(n)], d, w, color=COLORS[c], label=LABELS[c],
           edgecolor="white", linewidth=1.5)
ax.axhline(0, color=INK, lw=1)
ax.set_xticks(range(n)); ax.set_xticklabels([r["name"] for r in vines], rotation=35, ha="right")
ax.set_ylabel("predicted share minus true share (percentage points)")
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.text(n - 0.5, ax.get_ylim()[1] * 0.92, "above 0: model predicts too much", ha="right", fontsize=9.5, color=MUTED)
ax.text(n - 0.5, ax.get_ylim()[0] * 0.92, "below 0: model predicts too little", ha="right", fontsize=9.5, color=MUTED)
ax.set_title("How far off the predicted composition is, per vine", loc="left", fontsize=14, fontweight="bold", pad=34)
ax.legend(ncol=3, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=10.5, handlelength=1.2)
fig.tight_layout()
fig.savefig(f"{out_dir}/composition_difference_per_vine.png", dpi=150); plt.close(fig)
print("done")