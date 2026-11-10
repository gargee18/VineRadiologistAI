#!/usr/bin/env python3
"""
make_views.py  -  render the augmented D2 views ONCE and store them

Renders VIEWS_PER_VINE augmented views per D2 vine with augment.py and saves
them, so training reads them from disk instead of re-rendering (about 10 s
per view on the GPU). Every vine gets views; at training time each fold only
loads the views of its training vines, so a test vine is never seen.

Each view is cropped to its vine (+ AUG_CROP_MARGIN) and saved as two
uncompressed .npy files, which training reads with memory mapping (only the
patches it needs are read, the views never all sit in RAM):
    <VIEWS_DIR>/<vine>/<view>_img.npy   DRR, normalized 0..1, stored as uint16
    <VIEWS_DIR>/<vine>/<view>_lab.npy   (3, H, W) thickness in mm, float16
    <VIEWS_DIR>/<vine>/<view>_fg.npy    (N, 2) row, col of up to FG_POINTS vine
                                        pixels, for centring training patches
    <VIEWS_DIR>/manifest.csv            seed and parameters of every view

Seeds: zlib.crc32("<vine>-stored-<view>"), so any view can be rebuilt with
augment.py. Already-written views are skipped, so if the run stops, start it
again and it continues where it was.

After the first view, the script prints the size per view and the expected
total, and stops if the disk does not have room for it.

USAGE (from the repo root)
    python training/make_views.py --dry-run      # list the plan and the disk size
    python training/make_views.py                # make them all
    python training/make_views.py --vine CEP016_RES3   # one vine only
"""

import argparse
import os
import shutil
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import augment                                                   # noqa: E402
from train_baseline import (REPO, AUG_CROP_MARGIN, list_specimens,  # noqa: E402
                            normalize, crop_to_vine, LABEL_ROOT, LABEL_SUFFIXES,
                            VIEWS_DIR)   # set VIEWS_DIR in train_baseline.py only

VIEWS_PER_VINE = 100         # 12 vines x 100 = 1200 views
FG_POINTS = 20000            # vine pixels kept per view for patch centring
PARAM_KEYS = ("yaw", "pitch", "roll", "spd", "off_u", "off_v",
              "bend_mm", "bend_dir", "elastic_mm", "noise_std")


def view_paths(vine, v):
    d = os.path.join(VIEWS_DIR, vine)
    return tuple(os.path.join(d, f"{v:03d}_{k}.npy") for k in ("img", "lab", "fg"))


def view_seed(vine, v):
    return zlib.crc32(f"{vine}-stored-{v}".encode())


def save_view(vine, v, img, lab):
    paths = view_paths(vine, v)
    os.makedirs(os.path.dirname(paths[0]), exist_ok=True)
    img16 = np.round(np.clip(img, 0, 1) * 65535).astype(np.uint16)
    lab16 = lab.astype(np.float16)
    fg = np.argwhere(lab16.astype(np.float32).sum(0) > 0).astype(np.int32)
    if len(fg) > FG_POINTS:
        fg = fg[np.random.default_rng(view_seed(vine, v)).choice(len(fg), FG_POINTS, replace=False)]
    arrays = (img16, lab16, fg)
    # write to temp names first, then rename: a stopped run never leaves half a view
    for p, a in zip(paths, arrays):
        np.save(p + ".tmp.npy", a)
    for p in paths:
        os.replace(p + ".tmp.npy", p)
    for p, a in zip(paths, arrays):                 # read back and check
        if not np.array_equal(np.load(p), a):
            sys.exit(f"{p}: written file does not match, stopping")
    return sum(os.path.getsize(p) for p in paths)


def estimate_size(vines, n_views):
    """Rough disk size, without rendering: the vine box of each existing
    (un-augmented) label + AUG_CROP_MARGIN, 8 bytes per pixel. Rotated and
    bent views are somewhat wider or narrower, so treat it as approximate."""
    import tifffile
    total = 0
    for s in vines:
        lab = sum(tifffile.imread(os.path.join(LABEL_ROOT, s + suf)).astype(np.float32)
                  for suf in LABEL_SUFFIXES)
        ys, xs = np.nonzero(lab > 0)
        h = min(ys.max() - ys.min() + 1 + 2 * AUG_CROP_MARGIN, lab.shape[0])
        w = min(xs.max() - xs.min() + 1 + 2 * AUG_CROP_MARGIN, lab.shape[1])
        one = h * w * 8 + FG_POINTS * 8
        total += one * n_views
        print(f"  {s:12s} about {h} x {w} px -> {one / 1e6:.0f} MB per view")
    d = VIEWS_DIR                       # free space of the disk the views go to
    while not os.path.exists(d):
        d = os.path.dirname(d)
    free = shutil.disk_usage(d).free
    print(f"estimated total: about {total / 1e9:.0f} GB for {n_views} views per vine, "
          f"{free / 1e9:.0f} GB free")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vine", nargs="+", help="only these vines (default: all)")
    ap.add_argument("--dry-run", action="store_true", help="list the plan, render nothing")
    args = ap.parse_args()

    vines = args.vine or list_specimens()
    todo = [(s, v) for s in vines for v in range(VIEWS_PER_VINE)
            if not all(os.path.exists(p) for p in view_paths(s, v))]
    print(f"{len(vines)} vines x {VIEWS_PER_VINE} views; {len(todo)} still to make "
          f"-> {VIEWS_DIR}")
    if args.dry_run:
        estimate_size(vines, VIEWS_PER_VINE)
    if args.dry_run or not todo:
        return

    os.makedirs(VIEWS_DIR, exist_ok=True)
    manifest = os.path.join(VIEWS_DIR, "manifest.csv")
    if not os.path.exists(manifest):
        with open(manifest, "w") as f:
            f.write("vine,view,seed," + ",".join(PARAM_KEYS) + ",height_px,width_px\n")

    t0, done, checked_disk = time.perf_counter(), 0, False
    for s in vines:
        views = [v for (vs, v) in todo if vs == s]
        if not views:
            continue
        vols = augment.load_specimen(s)
        for v in views:
            seed = view_seed(s, v)
            p = augment.sample_params(seed)
            drr, lab = augment.render(vols, p)
            img = normalize(np.nan_to_num(drr))          # same scaling as the test images
            img, lab = crop_to_vine(img, np.clip(np.nan_to_num(lab), 0, None), AUG_CROP_MARGIN)
            nbytes = save_view(s, v, img, lab)
            with open(manifest, "a") as f:
                f.write(f"{s},{v},{seed}," + ",".join(f"{p[k]:.4f}" for k in PARAM_KEYS)
                        + f",{img.shape[0]},{img.shape[1]}\n")
            done += 1
            if not checked_disk:
                need = nbytes * (len(todo) - 1)
                free = shutil.disk_usage(VIEWS_DIR).free
                print(f"  one view = {nbytes / 1e6:.0f} MB -> about {need / 1e9:.0f} GB more "
                      f"needed, {free / 1e9:.0f} GB free")
                if need > 0.9 * free:
                    sys.exit("  not enough disk space, stopping (the view made so far is kept)")
                checked_disk = True
            el = time.perf_counter() - t0
            print(f"  {s} view {v:3d}  ({done}/{len(todo)}, "
                  f"{el / done:.1f} s/view, ~{el / done * (len(todo) - done) / 60:.0f} min left)")
        del vols
    print("done")


if __name__ == "__main__":
    main()