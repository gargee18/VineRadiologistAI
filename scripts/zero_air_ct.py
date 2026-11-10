#!/usr/bin/env python3
"""
zero_air_ct.py

Set every CT voxel below a cutoff to 0 (removes air noise, which otherwise
adds up along the DRR rays). Everything else is untouched:
  * same dtype (16-bit stays 16-bit, no rescaling)
  * same voxel size / unit / resolution tags (ImageJ metadata copied)
  * written to a NEW file, the input is never modified

USAGE
    # one volume
    python scripts/zero_air_ct.py IN.tif -o OUT.tif

    # every volume in a folder (glob pattern), outputs keep their file names
    python scripts/zero_air_ct.py "dataset/ct_volumes_table_removed/CEP_*_2026_XR.tif" \
        --out-dir dataset/ct_volumes_air_zeroed

    --cutoff 100 is the default.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import tifffile


def read_volume(path):
    """Read a TIFF stack plus the metadata needed to write it back identically."""
    with tifffile.TiffFile(path) as tf:
        vol = tf.asarray()
        page = tf.pages[0]
        ij = dict(tf.imagej_metadata or {})
        res = None
        if "XResolution" in page.tags and "YResolution" in page.tags:
            res = (page.tags["XResolution"].value, page.tags["YResolution"].value)
    return vol, ij, res


def write_volume(path, vol, ij, res):
    meta = {}
    for key in ("spacing", "unit", "finterval", "slices", "frames", "channels"):
        if key in ij:
            meta[key] = ij[key]
    meta.pop("slices", None)          # tifffile recomputes axes from the shape
    kw = {"imagej": True, "metadata": {**meta, "axes": "ZYX"}}
    if res is not None:
        kw["resolution"] = res
    tifffile.imwrite(path, vol, **kw)


def spacing_str(ij, res):
    xy = "?"
    if res is not None:
        num, den = res[0] if isinstance(res[0], tuple) else (res[0], 1)
        if num:
            xy = f"{den / num:.6f}"
    return f"xy={xy}, z={ij.get('spacing', '?')} {ij.get('unit', '')}".strip()


def process(src, dst, cutoff):
    vol, ij, res = read_volume(src)
    if vol.ndim != 3:
        print(f"  [skip] {src}: expected a 3D stack, got shape {vol.shape}")
        return False

    below = vol < cutoff
    n_changed = int(np.count_nonzero(below & (vol != 0)))
    out = vol.copy()
    out[below] = 0                     # in the original dtype, no conversion

    dst.parent.mkdir(parents=True, exist_ok=True)
    write_volume(dst, out, ij, res)

    # verify what was written
    chk, ij2, res2 = read_volume(dst)
    ok = (chk.dtype == vol.dtype and chk.shape == vol.shape
          and np.array_equal(chk[chk >= cutoff], vol[vol >= cutoff])
          and not np.any((chk > 0) & (chk < cutoff))
          and spacing_str(ij, res) == spacing_str(ij2, res2))
    print(f"  {Path(src).name}: {vol.dtype} {vol.shape}, "
          f"{n_changed:,} voxels set to 0 ({100 * n_changed / vol.size:.1f}%), "
          f"spacing {spacing_str(ij2, res2)}  ->  {'OK' if ok else 'CHECK FAILED'}")
    return ok


def main():
    p = argparse.ArgumentParser(description="Zero CT voxels below a cutoff (air noise).")
    p.add_argument("inputs", nargs="+", help="volume path(s) or glob pattern(s)")
    p.add_argument("-o", "--output", help="output file (single input only)")
    p.add_argument("--out-dir", help="output folder (keeps input file names)")
    p.add_argument("--cutoff", type=float, default=100)
    args = p.parse_args()

    files = sorted({f for pat in args.inputs for f in (glob.glob(pat) or [pat])})
    missing = [f for f in files if not Path(f).is_file()]
    if missing:
        sys.exit(f"not found: {missing}")
    if args.output and len(files) != 1:
        sys.exit("-o works with one input; use --out-dir for several")
    if not args.output and not args.out_dir:
        sys.exit("give -o OUT.tif or --out-dir DIR")

    print(f"cutoff: voxels < {args.cutoff:g} -> 0, {len(files)} volume(s)")
    all_ok = True
    for f in files:
        dst = Path(args.output) if args.output else Path(args.out_dir) / Path(f).name
        if dst.resolve() == Path(f).resolve():
            sys.exit(f"refusing to overwrite the input: {f}")
        all_ok &= process(f, dst, args.cutoff)
    print("done" if all_ok else "done, but at least one CHECK FAILED, see above")


if __name__ == "__main__":
    main()