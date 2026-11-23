#!/usr/bin/env python3
"""
compare_runs.py  -  one table comparing the training runs, all 12 test vines

Reads what --summarize and --replot already wrote in each run folder:
    summary_per_group.csv   (MAE in mm, Dice, amount; row "ALL")
    volume_per_vine.csv     (true and predicted volume, cm3 and % of the wood)
Nothing is retrained or re-predicted.

USAGE (from the repo root, after --summarize and --replot of every run)
    python training/compare_runs.py
Writes results/comparison/compare_runs.csv, compare_runs_scores.png,
compare_runs_composition.png and compare_runs_composition_per_vine.png,
prints the table.
"""

import csv
import os

REPO = "/mnt/41d6c007-0c9e-41e2-b2eb-8d9c032e9e53/gargee/code_python/VineRadiologistAI"
RUNS = [   # (name in the table, results folder)
    ("baseline (cropped)",       "results/training_baseline_cropped"),
    ("augmented, on the fly",    "results/training_augmented"),
    ("stored views + flip",      "results/training_stored"),
    ("domain adaptation (λ 0.3)", "results/domain_adaptation/lam0.3"),
    ("class balancing",          "results/training_stored_balanced"),
    ("domain adaptation, soil ignored", "results/domain_adaptation/lam0.3_soil"),
    ("DA, soil ignored + D1 augmentation", "results/domain_adaptation/lam0.3_soil_d1aug"),
    ("same, λ 0.1", "results/domain_adaptation/lam0.1_soil_d1aug"),
    ("same, λ 1", "results/domain_adaptation/lam1_soil_d1aug"),
    ("DA adversarial + balancing", "results/domain_adaptation/lam0.3_soil_d1aug_bal"),
    ("DA contrastive + balancing", "results/domain_adaptation/contrast0.3_soil_bal"),
    ("DA MMD + balancing", "results/domain_adaptation/mmd0.3_soil_d1aug_bal"),
]
TISSUES = [("healthy", "Healthy"), ("necrosis", "Necrosis"), ("white_rot", "White rot")]
OUT_DIR = os.path.join(REPO, "results", "comparison")      # created if missing
OUT = os.path.join(OUT_DIR, "compare_runs.csv")
TISSUE_COLORS = ["#1B5E20", "#F9A825", "#B71C1C"]   # same as train_baseline.py
RUN_COLORS = ["#E8EBEE", "#D3D9DF", "#BEC6CF", "#A9B4C0", "#94A2B1", "#7F90A2", "#6A7E93", "#566C84", "#435A75", "#304866", "#1F3346", "#101C28"]  # light = first run, dark = latest
GROUP_COLORS = {"asymptomatic": "#1baf7a", "resistant": "#2a78d6",   # same as train_baseline.py
                "symptomatic": "#eb6834", "apoplectic": "#7a3fa0"}


def plot_per_vine(vols, path):
    """vols: {run name: rows of its volume_per_vine.csv}. One panel per vine
    (one row of panels per plant group): true, then each run, stacked shares."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    runs = list(vols)                        # the runs that have results
    keys = [t for t, _ in TISSUES]
    by_run = {r: {v["vine"]: v for v in vols[r]} for r in runs}
    first = vols[runs[0]]
    vines = [v["vine"] for v in first]
    group = {v["vine"]: v["group"] for v in first}
    groups = list(dict.fromkeys(group[v] for v in vines))
    ncol = max(sum(group[v] == g for v in vines) for g in groups)

    fig, ax = plt.subplots(len(groups), ncol, figsize=(7 * ncol, 3.1 * len(groups)),
                           layout="constrained", squeeze=False)
    fig.suptitle("Wood composition per vine, true vs predicted by each run",
                 fontsize=19, fontweight="bold")
    labels = ["true"] + runs
    for gi, g in enumerate(groups):
        gv = [v for v in vines if group[v] == g]
        for ci in range(ncol):
            a = ax[gi, ci]
            if ci >= len(gv):
                a.axis("off"); continue
            vine = gv[ci]
            for i, lab in enumerate(labels):
                row = by_run[runs[0]][vine] if lab == "true" else by_run[lab].get(vine)
                if row is None:                       # vine missing in this run
                    continue
                kind = "true" if lab == "true" else "pred"
                shares = [float(row[f"{t}_{kind}_share_pct"]) for t in keys]
                left = 0
                for c, v in enumerate(shares):
                    a.barh(i, v, left=left, color=TISSUE_COLORS[c], edgecolor="white",
                           linewidth=1.5, height=0.75)
                    if v >= 6:
                        a.text(left + v / 2, i, f"{v:.0f}", ha="center", va="center", fontsize=11,
                               fontweight="bold", color="#1F2421" if c == 1 else "white")
                    left += v
            a.set_yticks(range(len(labels)))
            a.set_yticklabels(labels if ci == 0 else [""] * len(labels), fontsize=12)
            if ci == 0:
                a.get_yticklabels()[0].set_fontweight("bold")
            a.invert_yaxis(); a.set_xlim(0, 100)
            a.set_title(vine, fontsize=15, fontweight="bold", color=GROUP_COLORS.get(g, "#333333"))
            a.tick_params(axis="x", labelsize=11)
            for sp in ("top", "right", "left"):
                a.spines[sp].set_visible(False)
            if gi == len(groups) - 1:
                a.set_xlabel("share of the wood (%)", fontsize=12)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in TISSUE_COLORS]
    names = [lab for _, lab in TISSUES]
    try:
        fig.legend(handles, names, loc="outside lower center", ncol=3, fontsize=14, frameon=False)
    except ValueError:
        ax[0, 0].legend(handles, names, fontsize=11)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot(rows, path_scores, path_comp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    runs = list(dict.fromkeys(r["run"] for r in rows))   # the runs that have results
    tissues = [label for _, label in TISSUES]
    get = lambda run, tis, key: next(r[key] for r in rows if r["run"] == run and r["tissue"] == tis)

    # one panel per metric (rows) and tissue (columns), horizontal bars, run names once
    panels = [("MAE_mm", "MAE (mm)", "%.2f", min),
              ("Dice", "Dice", "%.2f", max),
              ("share_error_pp", "error in share of the wood (pp)", "%.1f", min)]
    n = len(runs)
    fig, ax = plt.subplots(3, 3, figsize=(20, 3 * (1.2 + 0.32 * n)), layout="constrained",
                           sharey=True)
    fig.suptitle(f"Test scores of the training runs, 12 test vines (best value of each panel in bold)",
                 fontsize=19, fontweight="bold")
    y = list(range(n))
    for r, (key, title, fmt, best) in enumerate(panels):
        for c, tis in enumerate(tissues):
            a = ax[r, c]
            vals = [get(run, tis, key) for run in runs]
            a.barh(y, vals, color=[RUN_COLORS[k % len(RUN_COLORS)] for k in range(n)],
                   edgecolor="#9AA5B1", linewidth=0.6, height=0.75)
            b = best(vals)
            for k, v in enumerate(vals):
                a.text(v, k, " " + fmt % v, va="center", fontsize=11,
                       fontweight="bold" if v == b else "normal")
            lo = 0.6 if key == "Dice" else 0
            a.set_xlim(lo, max(vals) * 1.18 if key != "Dice" else 1.0)
            if r == 0:
                a.set_title(tis, fontsize=17, fontweight="bold", color=TISSUE_COLORS[c])
            a.set_xlabel(title, fontsize=12)
            a.tick_params(axis="x", labelsize=11)
            a.grid(axis="x", alpha=0.3); a.set_axisbelow(True)
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
    ax[0, 0].set_yticks(y)
    ax[0, 0].invert_yaxis()
    for r in range(3):
        ax[r, 0].set_yticklabels(runs, fontsize=12)
    fig.savefig(path_scores, dpi=100)
    plt.close(fig)

    # composition: true, then each run, stacked shares
    fig, a = plt.subplots(figsize=(12, 3.5 + 0.6 * len(runs)), layout="constrained")
    labels = ["true"] + runs
    for i, lab in enumerate(labels):
        shares = [get(runs[0], t, "true_share_pct") if lab == "true" else get(lab, t, "pred_share_pct")
                  for t in tissues]
        left = 0
        for c, v in enumerate(shares):
            a.barh(i, v, left=left, color=TISSUE_COLORS[c], edgecolor="white", linewidth=2, height=0.7)
            if v >= 4:
                a.text(left + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=12,
                       fontweight="bold", color="#1F2421" if c == 1 else "white")
            left += v
    a.set_yticks(range(len(labels)))
    a.set_yticklabels(labels, fontsize=13)
    a.get_yticklabels()[0].set_fontweight("bold")
    a.invert_yaxis(); a.set_xlim(0, 100)
    a.set_xlabel("share of the wood (%), mean over the vines", fontsize=13)
    a.set_title("Wood composition, true vs predicted by each run (12 test vines)",
                fontsize=17, fontweight="bold")
    a.tick_params(axis="x", labelsize=12)
    for sp in ("top", "right", "left"):
        a.spines[sp].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in TISSUE_COLORS]
    try:
        fig.legend(handles, tissues, loc="outside lower center", ncol=3, fontsize=14, frameon=False)
    except ValueError:
        a.legend(handles, tissues, fontsize=12)
    fig.savefig(path_comp, dpi=110)
    plt.close(fig)


def read_csv(path):
    if not os.path.exists(path):
        raise SystemExit(f"Missing {path}: run --summarize and --replot for this run first")
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    rows, vols = [], {}
    for name, folder in RUNS:
        d = os.path.join(REPO, folder)
        if not (os.path.exists(os.path.join(d, "summary_per_group.csv"))
                and os.path.exists(os.path.join(d, "volume_per_vine.csv"))):
            print(f"skip '{name}': no results yet in {folder} (train it, then --replot and --summarize)")
            continue
        allrow = [r for r in read_csv(os.path.join(d, "summary_per_group.csv")) if r["group"] == "ALL"][0]
        vol = read_csv(os.path.join(d, "volume_per_vine.csv"))
        vols[name] = vol
        n = len(vol)
        for t, label in TISSUES:
            tp = [float(v[f"{t}_true_share_pct"]) for v in vol]
            pp = [float(v[f"{t}_pred_share_pct"]) for v in vol]
            rows.append({
                "run": name, "tissue": label, "n_vines": n,
                "MAE_mm": float(allrow[f"{t}_mae_mm"]),
                "Dice": float(allrow[f"{t}_dice"]),
                "amount": float(allrow[f"{t}_amount_ratio"]),
                "true_share_pct": sum(tp) / n,
                "pred_share_pct": sum(pp) / n,
                "share_error_pp": sum(abs(p - q) for p, q in zip(pp, tp)) / n,
                "true_cm3": sum(float(v[f"{t}_true_cm3"]) for v in vol),
                "pred_cm3": sum(float(v[f"{t}_pred_cm3"]) for v in vol),
            })

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})

    print("Mean over the test vines (each vine tested by a model that never saw it)\n")
    head = (f"{'tissue':10s} {'run':32s} {'MAE mm':>7s} {'Dice':>5s} {'amount':>6s} "
            f"{'share true %':>12s} {'share pred %':>12s} {'|diff| pp':>9s} {'true cm3':>9s} {'pred cm3':>9s}")
    print(head); print("-" * len(head))
    for _, label in TISSUES:
        for r in rows:
            if r["tissue"] == label:
                print(f"{label:10s} {r['run']:32s} {r['MAE_mm']:7.2f} {r['Dice']:5.2f} {r['amount']:6.2f} "
                      f"{r['true_share_pct']:12.1f} {r['pred_share_pct']:12.1f} {r['share_error_pp']:9.1f} "
                      f"{r['true_cm3']:9.0f} {r['pred_cm3']:9.0f}")
        print()
    print("share = % of the wood; |diff| pp = mean |predicted - true| share per vine, in percentage points")
    print("cm3 = summed over the test vines, approximate (cone beam)")
    p1, p2 = OUT.replace(".csv", "_scores.png"), OUT.replace(".csv", "_composition.png")
    p3 = OUT.replace(".csv", "_composition_per_vine.png")
    plot(rows, p1, p2)
    plot_per_vine(vols, p3)
    print(f"written: {OUT}, {p1}, {p2}, {p3}")


if __name__ == "__main__":
    main()