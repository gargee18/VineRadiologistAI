#!/usr/bin/env python3
"""
analyze_errors.py  -  which tissue causes the error: a thickness confusion matrix

For every test vine (each predicted by the fold model that never saw it),
and at every pixel, the error of each tissue (predicted - true) is split into:
  * correct:  min(predicted, true) of that tissue
  * swaps:    where tissue A is over-estimated and tissue B under-estimated at
              the same pixel, that amount counts as "B read as A"
              (shared in proportion when several tissues are involved)
  * extra:    over-estimation left when nothing is under-estimated there
              (the model puts wood where there is less or none)
  * missed:   under-estimation left when nothing is over-estimated there
Everything is summed over the pixels and turned into volume (cm3, using the
pixel area at the vine, approximate for a cone beam).

Matrix: rows = true (healthy, necrosis, white rot, no wood),
columns = predicted (healthy, necrosis, white rot, missed).
Row sums = true volumes, column sums = predicted volumes.

No training: reloads each fold's model.pt and predicts its test vines.

Correlation (--correlation, no GPU, reads volume_per_vine.csv of each run):
per vine, healthy volume against necrosis volume, against white rot volume,
and total wood volume against necrosis + white rot; true and predicted, with
Spearman r. Volumes, not shares: shares add up to 100%, which would make
"more necrosis = less healthy" true by construction.

USAGE (from the repo root)
    python training/analyze_errors.py                          # all runs of compare_runs.py
    python training/analyze_errors.py --run-dir results/training_stored
    python training/analyze_errors.py --correlation            # correlation plots only
Output: results/error_analysis/<run name>/
    confusion_cm3.csv           all test vines together
    confusion_per_vine.csv      the same, one block of rows per vine
    confusion.png               the matrix, % of each true tissue + cm3
    correlation.png             healthy vs necrosis / white rot, per vine
"""

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_baseline as tb          # noqa: E402

OUT_ROOT = os.path.join(tb.REPO, "results", "error_analysis")
ROWS = ["Healthy", "Necrosis", "White rot", "No wood"]          # true
COLS = ["Healthy", "Necrosis", "White rot", "Missed"]           # predicted


def confusion(pred, gt):
    """(4, 4) matrix in mm x pixels: [true row, predicted column]."""
    e = pred - gt                                   # (3, H, W)
    over = np.clip(e, 0, None)                      # too much, per tissue
    under = np.clip(-e, 0, None)                    # too little, per tissue
    P, N = over.sum(0), under.sum(0)                # per pixel
    swap = np.minimum(P, N)                         # amount moved between tissues
    fo = np.divide(over, P, out=np.zeros_like(over), where=P > 0)    # share of each over
    fu = np.divide(under, N, out=np.zeros_like(under), where=N > 0)  # share of each under
    M = np.zeros((4, 4))
    for i in range(3):
        M[i, i] = np.minimum(pred[i], gt[i]).sum()                  # correct
        for j in range(3):
            if i != j:                                              # true i read as j
                M[i, j] = (swap * fu[i] * fo[j]).sum()
        M[i, 3] = ((N - swap) * fu[i]).sum()                        # true i missed
        M[3, i] = ((P - swap) * fo[i]).sum()                        # extra predicted i
    return M


def check_confusion(M, pred, gt):
    """Row sums must give the true totals and column sums the predicted ones."""
    ok_t = np.allclose(M[:3].sum(1), gt.reshape(3, -1).sum(1), rtol=1e-4, atol=1e-3)
    ok_p = np.allclose(M[:, :3].sum(0), pred.reshape(3, -1).sum(1), rtol=1e-4, atol=1e-3)
    if not (ok_t and ok_p):
        raise RuntimeError("confusion matrix does not add up to the true/predicted totals")


# readable model descriptions for the figure title (folder name -> text)
DESCRIPTIONS = {
    "training_baseline_cropped": "Baseline, no augmentation",
    "training_augmented": "3D augmentation, views rendered during training",
    "training_stored": "3D augmentation, 1200 stored views + left-right flip",
    "lam0.3": "Domain adaptation (λ = 0.3), stored views + flip",
    "training_stored_balanced": "Class balancing (white rot sampling + loss weights), stored views + flip",
}


def describe(run_dir):
    base = os.path.basename(os.path.normpath(run_dir))
    if base.startswith("lam") and base not in DESCRIPTIONS:
        return f"Domain adaptation (λ = {base[3:]}), stored views + flip"
    return DESCRIPTIONS.get(base, base)


def analyze_run(run_dir, name):
    dirs = sorted(glob.glob(os.path.join(run_dir, "fold_*", "model.pt")))
    if not dirs:
        print(f"skip {name}: no fold_*/model.pt in {run_dir}")
        return None
    to_cm3 = tb.PIXEL_AREA_AT_VINE_MM2 / 1000.0
    out = os.path.join(OUT_ROOT, name)
    os.makedirs(out, exist_ok=True)
    total, per_vine, seen = np.zeros((4, 4)), [], set()
    for mp in reversed(dirs):                       # latest fold first, a vine counted once
        fd = os.path.dirname(mp)
        test = json.load(open(os.path.join(fd, "metrics.json")))["test"]
        todo = [s for s in test if s not in seen]
        if not todo:
            continue
        model = tb.load_model(mp); model.eval()
        for s, d, l in tb.list_pairs(todo):
            img, gt = tb.load_pair(d, l)
            pred = tb.predict_full(model, img)
            M = confusion(pred, gt)
            check_confusion(M, pred, gt)
            per_vine.append((s, M * to_cm3))
            total += M * to_cm3
            seen.add(s)
            print(f"  {name}: {s}")
    per_vine.sort()
    write_csv(os.path.join(out, "confusion_cm3.csv"), [("ALL", total)])
    write_csv(os.path.join(out, "confusion_per_vine.csv"), per_vine)
    plot(total, f"Confusion matrix: true vs predicted tissue volume\n"
         f"Model: {describe(run_dir)}  ·  {len(per_vine)} test vines",
         os.path.join(out, "confusion.png"))
    print(f"written: {out}/")
    return total


def write_csv(path, blocks):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["vine", "true"] + [f"predicted {c} (cm3)" for c in COLS] + ["total (cm3)"])
        for vine, M in blocks:
            for i, r in enumerate(ROWS):
                w.writerow([vine, r] + [f"{x:.2f}" for x in M[i]] + [f"{M[i].sum():.2f}"])


NONE_COLOR = "#5E7288"     # slate, for "No wood" and "Missed"


def cell_color(i, j, pct):
    """White -> colour by the cell's %. Correct cells (diagonal): the tissue's
    own colour. Errors: an even mix of the true tissue (row) and the predicted
    tissue (column), e.g. necrosis predicted as healthy = yellow + green.
    "No wood" / "Missed" mix with slate."""
    from matplotlib.colors import to_rgb
    a = np.array(to_rgb(tb.TISSUE_COLORS[i] if i < 3 else NONE_COLOR))
    b = np.array(to_rgb(tb.TISSUE_COLORS[j] if j < 3 else NONE_COLOR))
    base = a if i == j else (a + b) / 2
    t = min(max(pct / 100.0, 0.0), 1.0)
    return tuple(1 - t * (1 - base))


def plot(M, title, path):
    plt = tb._plt()
    pct = np.zeros_like(M)
    rows_true = M[:3].sum(1, keepdims=True)
    pct[:3] = 100 * M[:3] / np.maximum(rows_true, 1e-9)             # % of each true tissue
    pct[3] = 100 * M[3] / max(M[:, :3].sum(), 1e-9)                 # % of all predicted volume
    fig, ax = plt.subplots(figsize=(10.5, 9), layout="constrained")
    for i in range(4):
        for j in range(4):
            if i == 3 and j == 3:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, color="#F2F2F2"))
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=12, color="#999999")
                continue
            c = cell_color(i, j, pct[i, j])
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, color=c))
            r, g, b = c
            dark = 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5
            v = f"{M[i, j]:.0f}" if M[i, j] >= 10 else f"{M[i, j]:.1f}"
            ax.text(j, i, f"{pct[i, j]:.0f}%\n{v} cm³", ha="center", va="center",
                    fontsize=14, fontweight="bold" if i == j else "normal",
                    color="white" if dark else "#1F2421")
    ax.set_xlim(-.5, 3.5); ax.set_ylim(3.5, -.5); ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.xaxis.set_label_position("top"); ax.xaxis.tick_top()
    ax.set_xticks(range(4)); ax.set_xticklabels(COLS, fontsize=13)
    ax.set_yticks(range(4)); ax.set_yticklabels(ROWS, fontsize=13)
    for lab, c in zip(ax.get_xticklabels()[:3], tb.TISSUE_COLORS):
        lab.set_color(c); lab.set_fontweight("bold")
    for lab, c in zip(ax.get_yticklabels()[:3], tb.TISSUE_COLORS):
        lab.set_color(c); lab.set_fontweight("bold")
    ax.set_xlabel("predicted", fontsize=14); ax.set_ylabel("true", fontsize=14)
    head, _, sub = title.partition("\n")              # bold title, lighter model line
    fig.suptitle(head, fontsize=16, fontweight="bold")
    if sub:
        ax.set_title(sub, fontsize=13, color="#444444", pad=12)
    fig.supxlabel("Rows 1-3: % of the true volume of that tissue (each row adds to 100%).\n"
             "Row 'No wood': extra predicted volume, % of all predicted volume. Volumes approximate (cone beam).",
             fontsize=10, color="#555555", ha="left", x=0.01)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def correlation_plot(run_dir, name):
    path = os.path.join(run_dir, "volume_per_vine.csv")
    if not os.path.exists(path):
        print(f"skip {name}: no volume_per_vine.csv (run --replot for it)")
        return
    rows = list(csv.DictReader(open(path)))
    vol = lambda r, t, k: float(r[f"{t}_{k}_cm3"])
    groups = [r["group"] for r in rows]
    data = {k: {t: np.array([vol(r, t, k) for r in rows]) for t in tb.CLASSES} for k in ("true", "pred")}
    panels = [("healthy", "necrosis", "Healthy", "Necrosis"),
              ("healthy", "white_rot", "Healthy", "White rot"),
              ("total", "disease", "Total wood", "Necrosis + white rot")]
    for k in data:
        data[k]["total"] = sum(data[k][t] for t in tb.CLASSES)
        data[k]["disease"] = data[k]["necrosis"] + data[k]["white_rot"]
    plt = tb._plt()
    fig, ax = plt.subplots(1, 3, figsize=(20, 7), layout="constrained")
    fig.suptitle("Is more diseased wood linked to less healthy wood?", fontsize=17, fontweight="bold")
    ax[1].set_title(f"Model: {describe(run_dir)}  ·  one point per test vine (volume, cm³)",
                    fontsize=12, color="#444444")
    for a, (xk, yk, xl, yl) in zip(ax, panels):
        for i, g in enumerate(groups):
            c = tb.GROUP_COLORS.get(g, "#555555")
            xt, yt, xp, yp = data["true"][xk][i], data["true"][yk][i], data["pred"][xk][i], data["pred"][yk][i]
            a.plot([xt, xp], [yt, yp], color=c, lw=1, alpha=0.5)
            a.scatter(xt, yt, s=90, color=c, zorder=3)
            a.scatter(xp, yp, s=90, facecolors="white", edgecolors=c, linewidths=2, zorder=3)
        rt = spearman(data["true"][xk], data["true"][yk])
        rp = spearman(data["pred"][xk], data["pred"][yk])
        a.set_xlabel(f"{xl} volume (cm³)\nSpearman r:  true {rt:+.2f}   predicted {rp:+.2f}   (n = {len(rows)})",
                     fontsize=13)
        a.set_ylabel(f"{yl} volume (cm³)", fontsize=13)
        a.margins(0.08)
        a.tick_params(labelsize=11); a.grid(alpha=0.3)
    h = [plt.Line2D([], [], marker="o", ls="", color="#555555", ms=9, label="true"),
         plt.Line2D([], [], marker="o", ls="", mfc="white", mec="#555555", mew=2, ms=9, label="predicted")]
    h += [plt.Line2D([], [], marker="o", ls="", color=c, ms=9, label=g)
          for g, c in tb.GROUP_COLORS.items() if g in groups]
    try:
        fig.legend(handles=h, loc="outside lower center", ncol=len(h), fontsize=12, frameon=False)
    except ValueError:
        ax[0].legend(handles=h, fontsize=10)
    out = os.path.join(OUT_ROOT, name)
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, "correlation.png"), dpi=100)
    plt.close(fig)
    print(f"  {name}: correlation.png written")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", help="one results folder (default: every run of compare_runs.py)")
    ap.add_argument("--correlation", action="store_true",
                    help="only the correlation plots (no GPU, reads volume_per_vine.csv)")
    args = ap.parse_args()
    if args.run_dir:
        runs = [(os.path.basename(os.path.normpath(args.run_dir)), args.run_dir)]
    else:
        import compare_runs
        runs = [(n.replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")
                 .replace("λ", "lam"), os.path.join(tb.REPO, f)) for n, f in compare_runs.RUNS]
    for name, d in runs:
        if not args.correlation:
            analyze_run(d, name)
        correlation_plot(d, name)


if __name__ == "__main__":
    main()