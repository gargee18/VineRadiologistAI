#!/usr/bin/env python3
"""
browse_augmented.py  -  browse the stored augmented images, one at a time

Shows the DRR and its 3 thickness maps (healthy, necrosis, white rot) of each
view in VIEWS_DIR, with its augmentation settings from manifest.csv.

Keys (click on the window first):
    Enter       next view
    Backspace   previous view
    q           quit
The zoom and pan buttons of the window work as usual.

USAGE (from the repo root; needs a screen, e.g. not a plain SSH terminal)
    python training/browse_augmented.py                      # all vines, from the start
    python training/browse_augmented.py --vine CEP020_APO1   # one vine
    python training/browse_augmented.py --vine CEP020_APO1 --start 40
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_baseline import VIEWS_DIR, THICK_VMAX_MM   # noqa: E402

STEP = 2        # show every 2nd pixel: much faster to draw, enough to look at


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vine", nargs="+", help="only these vines (default: all)")
    ap.add_argument("--start", type=int, default=0, help="view number to start at")
    args = ap.parse_args()

    if plt.get_backend().lower() == "agg":
        sys.exit("No screen available (matplotlib backend is Agg): run this where a window can open.")

    vines = args.vine or sorted(d for d in os.listdir(VIEWS_DIR)
                                if os.path.isdir(os.path.join(VIEWS_DIR, d)))
    items = []
    for s in vines:
        for p in sorted(glob.glob(os.path.join(VIEWS_DIR, s, "*_img.npy"))):
            v = int(os.path.basename(p).split("_")[0])
            if v >= args.start:                    # --start applies to every vine shown
                items.append((s, v, p))
    if not items:
        sys.exit(f"No views found in {VIEWS_DIR} for {vines}")

    params = {}
    man = os.path.join(VIEWS_DIR, "manifest.csv")
    if os.path.exists(man):
        with open(man) as f:
            for r in csv.DictReader(f):
                params[(r["vine"], int(r["view"]))] = r

    fig, ax = plt.subplots(1, 4, figsize=(18, 7), layout="constrained")
    state = {"i": 0}

    def show():
        s, v, p = items[state["i"]]
        img = np.load(p, mmap_mode="r")[::STEP, ::STEP]
        lab = np.load(p.replace("_img.npy", "_lab.npy"), mmap_mode="r")[:, ::STEP, ::STEP]
        for a in ax:
            a.clear(); a.set_xticks([]); a.set_yticks([])
        ax[0].imshow(img, cmap="gray"); ax[0].set_title("DRR", fontsize=14)
        for k, name in enumerate(["healthy", "necrosis", "white rot"]):
            ax[k + 1].imshow(lab[k].astype(np.float32), cmap="magma", vmin=0, vmax=THICK_VMAX_MM)
            ax[k + 1].set_title(f"{name} (mm, 0 to {THICK_VMAX_MM:.0f})", fontsize=14)
        r = params.get((s, v))
        info = ("" if r is None else
                f"yaw {float(r['yaw']):.0f}°  pitch {float(r['pitch']):+.1f}°  roll {float(r['roll']):+.1f}°  "
                f"spd {float(r['spd']):.0f} mm  offset u {float(r['off_u']):+.0f} v {float(r['off_v']):+.0f} mm  "
                f"bend {float(r['bend_mm']):.1f} mm  wobble {float(r['elastic_mm']):.2f} mm  "
                f"noise {float(r['noise_std']):.4f}")
        fig.suptitle(f"{s}  view {v}   ({state['i'] + 1}/{len(items)})   "
                     "Enter: next, Backspace: previous, q: quit\n" + info, fontsize=13)
        fig.canvas.draw_idle()

    def on_key(e):
        if e.key == "enter" and state["i"] < len(items) - 1:
            state["i"] += 1; show()
        elif e.key == "backspace" and state["i"] > 0:
            state["i"] -= 1; show()

    fig.canvas.mpl_connect("key_press_event", on_key)
    show()
    plt.show()      # q closes the window (matplotlib default key)


if __name__ == "__main__":
    main()