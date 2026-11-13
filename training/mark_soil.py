#!/usr/bin/env python3
"""
mark_soil.py  -  click the top of the soil on every D1 2026 DRR and PXR

Shows each image (registered DRRs and all 2026 PXRs, each on its own: the
soil sits differently in the CT, scanned lying down, and in the upright
PXR). Click on the highest point of the soil heap: everything below that
height is marked as soil and will be ignored in training (not in the image
itself: it is not cropped).

Keys (click on the window first):
    click       set the soil line (click again to move it)
    Enter       save and go to the next image
    Backspace   previous image
    n           no soil visible in this image (nothing ignored)
    q           save and quit (start again later: it continues where you left)

Saved after every image in results/domain_adaptation/soil/soil_lines.csv
(file, kind, vine, path, soil_row; soil_row empty = no soil).

USAGE (from the repo root; needs a screen)
    python training/mark_soil.py              # unmarked images first, then stops
    python training/mark_soil.py --all        # go through every image again
    python training/mark_soil.py --vine 322   # only one vine
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np
import tifffile
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_baseline as tb                  # noqa: E402
import train_domain_adaptation as da         # noqa: E402

CSV_PATH = os.path.join(da.DA_ROOT, "soil", "soil_lines.csv")
STEP = 2          # show every 2nd pixel (fast); the saved row is in full-size pixels
FIELDS = ["file", "kind", "vine", "path", "soil_row"]


def images():
    """(kind, vine, path) for every registered DRR and every 2026 PXR."""
    out = []
    for v in sorted(da.PAIRED_PXR):
        p = da.D1_DRR_PATTERN.format(v=v)
        if os.path.exists(p):
            out.append(("drr", v, p))
    for d in sorted(glob.glob(os.path.join(da.PXR_ROOT, "CEP_*"))):
        v = os.path.basename(d)[4:]
        out.extend(("pxr", v, p) for p in da.pxr_files(v))
    return out


def load_csv():
    if not os.path.exists(CSV_PATH):
        return {}
    with open(CSV_PATH) as f:
        return {r["file"]: r for r in csv.DictReader(f)}


def save_csv(marks):
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    tmp = CSV_PATH + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader()
        for k in sorted(marks):
            w.writerow(marks[k])
    os.replace(tmp, CSV_PATH)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="also show images already marked")
    ap.add_argument("--vine", nargs="+", help="only these vines (e.g. 322 1191)")
    args = ap.parse_args()
    if plt.get_backend().lower() == "agg":
        sys.exit("No screen available (matplotlib backend is Agg): run this where a window can open.")

    marks = load_csv()
    items = [it for it in images() if not args.vine or it[1] in args.vine]
    if not args.all:
        items = [it for it in items if os.path.basename(it[2]) not in marks]
    if not items:
        print(f"Nothing to mark ({len(marks)} images already in {CSV_PATH}). Use --all to revise.")
        return
    print(f"{len(items)} images to mark, {len(marks)} already done")

    fig, ax = plt.subplots(figsize=(10, 10), layout="constrained")
    st = {"i": 0, "row": None, "artists": []}

    def key(i):
        return os.path.basename(items[i][2])

    def draw_line():
        for a in st["artists"]:
            a.remove()
        st["artists"] = []
        if st["row"] is not None:
            st["artists"] = [ax.axhline(st["row"], color="#E53935", lw=2),
                             ax.axhspan(st["row"], st["H"], color="#E53935", alpha=0.18)]
        fig.canvas.draw_idle()

    def show():
        kind, v, path = items[st["i"]]
        img = tb.normalize(np.nan_to_num(tifffile.imread(path).astype(np.float32)))
        st["H"] = img.shape[0]
        ax.clear(); ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(img[::STEP, ::STEP], cmap="gray", vmin=0, vmax=1,
                  extent=(0, img.shape[1], img.shape[0], 0))          # axes in full-size pixels
        prev = marks.get(key(st["i"]), {}).get("soil_row", "")
        st["row"] = int(prev) if prev != "" else None
        st["artists"] = []
        done = sum(1 for j in range(len(items)) if key(j) in marks)
        ax.set_title(f"{kind.upper()}  CEP_{v}  {key(st['i'])}   ({st['i'] + 1}/{len(items)}, {done} saved)\n"
                     "click = top of the soil   Enter = save + next   n = no soil   "
                     "Backspace = previous   q = quit", fontsize=12)
        draw_line()

    def store(row):
        kind, v, path = items[st["i"]]
        marks[key(st["i"])] = {"file": key(st["i"]), "kind": kind, "vine": v,
                              "path": os.path.relpath(path, tb.REPO),
                              "soil_row": "" if row is None else int(row)}
        save_csv(marks)

    def go(step):
        j = st["i"] + step
        if 0 <= j < len(items):
            st["i"] = j; show()
        elif j >= len(items):
            print(f"All done: {len(marks)} images in {CSV_PATH}")
            plt.close(fig)

    def on_click(e):
        zooming = str(getattr(fig.canvas.toolbar, "mode", "")) != ""     # zoom/pan tool active
        if e.inaxes is ax and e.button == 1 and e.ydata is not None and not zooming:
            st["row"] = int(round(e.ydata)); draw_line()

    def on_key(e):
        if e.key == "enter":
            if st["row"] is None:
                ax.set_xlabel("Click on the top of the soil first, or press n for no soil",
                              color="#B71C1C", fontsize=12); fig.canvas.draw_idle(); return
            store(st["row"]); go(+1)
        elif e.key == "n":
            store(None); go(+1)
        elif e.key == "backspace":
            go(-1)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    show()
    plt.show()          # q closes the window (matplotlib default); everything is already saved
    print(f"{len(marks)} images saved in {CSV_PATH}")


if __name__ == "__main__":
    main()