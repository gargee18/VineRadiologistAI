#!/usr/bin/env python3
"""
crop_ct_2019.py  -  keep only a range of slices of the 2019 CT stacks

For every specimen, crops the CT (_XR) and its 3 masks (_SAIN, _NECROSE,
_AMADOU) to the SAME slices, and saves them under the same file names in
OUT_DIR. The originals in IN_DIR are never modified.

Slice numbers are as Fiji shows them: the first slice is 1, and the range
is inclusive. (1, 338) keeps slices 1 to 338 and drops everything after.

Voxel size and unit (the ImageJ calibration) are copied to the cropped files.

USAGE (from the repo root)
    python crop_ct_2019.py --dry-run     # print what would happen, write nothing
    python crop_ct_2019.py               # crop and save
"""

import argparse
import os
import sys

import numpy as np
import tifffile

IN_DIR = "dataset/ct_2019"
OUT_DIR = "dataset/ct_2019_cropped"

FILE_TYPES = ["XR", "SAIN", "NECROSE", "AMADOU"]

# FILL IN: specimen -> (first slice to keep, last slice to keep), Fiji numbering.
CROP = {
    "CEP011_AS1":  (1, 390),
    "CEP012_AS2":  (1, 378),
    "CEP014_RES1": (1, 338),
    "CEP015_RES2": (1, 361),
    "CEP016_RES3": (1, 380),
    "CEP017_S1":   (1, 364),
    "CEP018_S2":   (1, 365),
    "CEP019_S3":   (1, 358),
    "CEP020_APO1": (1, 357),
    "CEP022_APO3": (1, 383),
    # CEP021_APO2: no change, copied as is (COPY_UNCROPPED below)
}

# Specimens this script must never write (e.g. already cropped by hand in OUT_DIR).
SKIP = set()

# Specimens not in CROP: True = copy them unchanged, so OUT_DIR holds all 12
# and the next scripts can read from one folder. False = leave them out.
COPY_UNCROPPED = True


def read_stack(path):
    """Returns the array and its ImageJ calibration (voxel size, unit)."""
    with tifffile.TiffFile(path) as tif:
        arr = tif.asarray()
        page = tif.pages[0]
        ij = tif.imagej_metadata or {}
        res = None
        if "XResolution" in page.tags and "YResolution" in page.tags:
            xr, yr = page.tags["XResolution"].value, page.tags["YResolution"].value
            res = (xr[0] / xr[1], yr[0] / yr[1])
    meta = {k: ij[k] for k in ("spacing", "unit") if k in ij}
    return arr, res, meta


def write_stack(path, arr, res, meta):
    kwargs = {"imagej": True, "metadata": dict(meta, axes="ZYX")}
    if res is not None:
        kwargs["resolution"] = res
    tifffile.imwrite(path, arr, **kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    args = ap.parse_args()

    if os.path.abspath(IN_DIR) == os.path.abspath(OUT_DIR):
        sys.exit("OUT_DIR must differ from IN_DIR, the originals are never overwritten.")

    specimens = sorted(f[:-len("_XR.tif")] for f in os.listdir(IN_DIR) if f.endswith("_XR.tif"))
    both = sorted(set(CROP) & SKIP)
    if both:
        sys.exit(f"In both CROP and SKIP: {both}")
    unknown = sorted(set(CROP) - set(specimens))
    if unknown:
        sys.exit(f"In CROP but not found in {IN_DIR}: {unknown}")
    print(f"{len(specimens)} specimens in {IN_DIR}")
    if not args.dry_run:
        os.makedirs(OUT_DIR, exist_ok=True)

    for s in specimens:
        if s in SKIP:
            print(f"  {s:12s} in SKIP, not touched")
            continue
        if s not in CROP and not COPY_UNCROPPED:
            print(f"  {s:12s} not in CROP, skipped")
            continue
        paths = {t: os.path.join(IN_DIR, f"{s}_{t}.tif") for t in FILE_TYPES}
        missing = [p for p in paths.values() if not os.path.exists(p)]
        if missing:
            sys.exit(f"{s}: missing {missing}")

        # check every file of this specimen has the same number of slices
        n = {}
        for t, p in paths.items():
            with tifffile.TiffFile(p) as tif:
                n[t] = tif.series[0].shape[0]
        if len(set(n.values())) != 1:
            sys.exit(f"{s}: files have different slice counts {n}, not cropping")
        nz = n["XR"]

        first, last = CROP.get(s, (1, nz))
        if not (1 <= first <= last <= nz):
            sys.exit(f"{s}: range ({first}, {last}) is outside slices 1 to {nz}")
        what = f"slices {first}-{last} of {nz}" if s in CROP else f"unchanged, {nz} slices"
        print(f"  {s:12s} {what}")
        if args.dry_run:
            continue

        for t, p in paths.items():
            arr, res, meta = read_stack(p)
            out = arr[first - 1:last]                      # Fiji 1-based inclusive -> numpy
            dst = os.path.join(OUT_DIR, os.path.basename(p))
            write_stack(dst, out, res, meta)
            back = tifffile.imread(dst)                    # verify what was written
            if back.shape != out.shape or back.dtype != arr.dtype or not np.array_equal(back, out):
                sys.exit(f"{dst}: written file does not match, stopping")
        print(f"  {'':12s} -> {OUT_DIR}, {len(paths)} files written and checked")

    print("dry run, nothing written" if args.dry_run else "done")


if __name__ == "__main__":
    main()