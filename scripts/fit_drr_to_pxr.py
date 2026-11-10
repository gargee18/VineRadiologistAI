#!/usr/bin/env python3
"""
fit_drr_to_pxr.py

Learn a 1D monotonic mapping "DRR value -> PXR value" from registered
DRR/PXR pairs, then apply it to any other DRR.

The mapping is an isotonic regression: a curve that only ever goes one way
(always up or always down, detected automatically), since more material
should never make a pixel flip direction.

USAGE
    pip install numpy tifffile scikit-learn matplotlib

    # 1. fit on one or more registered pairs (same shape, same pixel grid)
    python fit_drr_to_pxr.py fit \
        --pair drr_318.tif pxr_318.tif \
        --pair drr_323.tif pxr_323.tif \
        -o drr_to_pxr_curve.npz

    # 2. apply to any DRR
    python fit_drr_to_pxr.py apply drr_to_pxr_curve.npz drr_in.tif -o drr_in_mapped.tif

NOTES
    * Pixel sampling is stratified by DRR value, so rare intensities (thick
      trunk, bark edges) get as many samples as the huge air background.
    * NaN / inf pixels are ignored.
    * Output is saved in the PXR dtype, with the input DRR's resolution tags.
    * `fit` also writes <output>_qc.png: sample cloud + fitted curve. Look at it.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import tifffile
from sklearn.isotonic import IsotonicRegression


def read_2d(path):
    """Read a single 2D image plus its X/Y resolution tags (if any)."""
    with tifffile.TiffFile(path) as tf:
        arr = np.squeeze(tf.asarray())
        page = tf.pages[0]
        res = None
        if "XResolution" in page.tags and "YResolution" in page.tags:
            res = (page.tags["XResolution"].value, page.tags["YResolution"].value)
        unit = page.tags["ResolutionUnit"].value if "ResolutionUnit" in page.tags else None
    if arr.ndim != 2:
        sys.exit(f"{path}: expected a 2D image, got shape {arr.shape}")
    return arr, res, unit


def sample_pair(drr, pxr, n_bins, per_bin, rng, mask=None):
    """Stratified sample of (drr, pxr) pixel pairs, equal count per DRR-value bin."""
    x = drr.astype(np.float64).ravel()
    y = pxr.astype(np.float64).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    if mask is not None:
        ok &= mask.ravel() > 0
    x, y = x[ok], y[ok]
    edges = np.linspace(x.min(), x.max(), n_bins + 1)
    bin_id = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    picks = []
    for b in range(n_bins):
        idx = np.flatnonzero(bin_id == b)
        if idx.size:
            picks.append(rng.choice(idx, min(per_bin, idx.size), replace=False))
    sel = np.concatenate(picks)
    return x[sel], y[sel]


def normalize_pxr(pxr, mask, lo_pct, hi_pct, label=""):
    """Scale one PXR by its OWN percentiles to 0..1 (clipped).

    Removes the per-image brightness scale (e.g. images with vs without the
    saturated strip). Percentiles are taken inside the mask when one is given,
    otherwise over all finite pixels.
    """
    vals = pxr[np.isfinite(pxr)]
    if mask is not None:
        vals = pxr[(mask > 0) & np.isfinite(pxr)]
    lo, hi = np.percentile(vals, [lo_pct, hi_pct])
    if hi <= lo:
        sys.exit(f"{label}: cannot normalize, percentile {hi_pct} <= percentile {lo_pct}")
    print(f"  normalize {label}: p{lo_pct:g}={lo:.0f} -> 0, p{hi_pct:g}={hi:.0f} -> 1")
    return np.clip((pxr.astype(np.float64) - lo) / (hi - lo), 0.0, 1.0)


def cmd_fit(args):
    rng = np.random.default_rng(args.seed)
    xs, ys, pxr_dtype = [], [], None
    dpool, ppool = [], []
    masks = args.mask or [None] * len(args.pair)
    if len(masks) != len(args.pair):
        sys.exit(f"got {len(masks)} --mask for {len(args.pair)} --pair; give one mask per pair, "
                 f"in the same order, or none at all")
    for (drr_path, pxr_path), mask_path in zip(args.pair, masks):
        drr, _, _ = read_2d(drr_path)
        pxr, _, _ = read_2d(pxr_path)
        if drr.shape != pxr.shape:
            sys.exit(f"shape mismatch: {drr_path} {drr.shape} vs {pxr_path} {pxr.shape}. "
                     f"The pair must be registered onto the same pixel grid.")
        mask = None
        if mask_path is not None:
            mask, _, _ = read_2d(mask_path)
            if mask.shape != drr.shape:
                sys.exit(f"shape mismatch: mask {mask_path} {mask.shape} vs DRR {drr.shape}")
            print(f"  mask {mask_path}: keeps {100 * np.mean(mask > 0):.1f}% of pixels "
                  f"(nonzero = used for fitting)")
        pxr_dtype = pxr_dtype or pxr.dtype
        if args.pxr_norm == "percentile":
            pxr = normalize_pxr(pxr, mask, args.norm_lo, args.norm_hi, label=pxr_path)
            pxr_dtype = np.dtype(np.float32)
        x, y = sample_pair(drr, pxr, args.bins, args.per_bin, rng, mask)
        xs.append(x)
        ys.append(y)
        if args.method in ("quantile", "median"):
            # pool pixels; the same index is used for both images, so pairs stay paired
            ok = np.isfinite(drr) & np.isfinite(pxr)
            if mask is not None:
                ok &= mask > 0
            if args.method == "median":
                ok &= drr != 0          # DRR == 0: rays missed the CT, no real pairing
            dv, pv = drr[ok].astype(np.float64), pxr[ok].astype(np.float64)
            if dv.size > args.max_pool:
                idx = rng.choice(dv.size, args.max_pool, replace=False)
                dv, pv = dv[idx], pv[idx]
            dpool.append(dv)
            ppool.append(pv)
        print(f"{drr_path} + {pxr_path}: {x.size:,} samples")

    x = np.concatenate(xs)
    y = np.concatenate(ys)

    if args.method == "isotonic":
        iso = IsotonicRegression(increasing="auto", out_of_bounds="clip")
        iso.fit(x, y)
        increasing = bool(iso.increasing_)
        grid = np.linspace(x.min(), x.max(), 4096)
        raw_curve = iso.predict(grid)
        # isotonic output is a staircase, which posterizes the mapped image.
        # A moving average with edge padding smooths the steps and keeps it monotonic.
        w = max(1, int(round(args.smooth * grid.size)))
        if w > 1:
            pad = np.pad(raw_curve, (w // 2, w - 1 - w // 2), mode="edge")
            curve = np.convolve(pad, np.ones(w) / w, mode="valid")
        else:
            curve = raw_curve
    elif args.method == "median":
        # median PXR value per DRR bin: follows the dense ridge of well-aligned
        # pixels and ignores scattered mismatched pairs (as long as they are < 50%
        # of a bin). Then made monotonic with a count-weighted isotonic fit.
        dpool, ppool = np.concatenate(dpool), np.concatenate(ppool)
        edges = np.linspace(dpool.min(), dpool.max(), args.median_bins + 1)
        bin_id = np.clip(np.digitize(dpool, edges) - 1, 0, args.median_bins - 1)
        order = np.argsort(bin_id, kind="stable")
        counts = np.bincount(bin_id, minlength=args.median_bins)
        groups = np.split(ppool[order], np.cumsum(counts)[:-1])
        centers = 0.5 * (edges[:-1] + edges[1:])
        keep = counts >= args.min_count
        med = np.array([np.median(g) if k else np.nan for g, k in zip(groups, keep)])
        print(f"  median: {keep.sum()} of {args.median_bins} bins have >= {args.min_count} pixels")
        iso = IsotonicRegression(increasing="auto", out_of_bounds="clip")
        iso.fit(centers[keep], med[keep], sample_weight=counts[keep])
        increasing = bool(iso.increasing_)
        grid = centers[keep]
        raw_curve = med[keep]            # per-bin medians, shown on the QC plot
        curve = iso.predict(grid)
    else:
        # match the k-th percentile of the DRR to the k-th percentile of the PXR.
        # Direction (is DRR air dark or bright?) comes from the paired samples.
        increasing = bool(np.corrcoef(x, y)[0, 1] >= 0)
        dpool, ppool = np.concatenate(dpool), np.concatenate(ppool)
        q = np.linspace(0.0, 1.0, 4096)
        grid = np.quantile(dpool, q)
        curve = np.quantile(ppool, q if increasing else 1.0 - q)
        # flat stretches of DRR values give repeated grid points; merge them
        grid, inv = np.unique(grid, return_inverse=True)
        curve = np.bincount(inv, weights=curve) / np.bincount(inv)
        raw_curve = None

    resid = y - np.interp(x, grid, curve)
    r2 = 1.0 - resid.var() / max(y.var(), 1e-12)
    print(f"\nfitted ({args.method}) on {x.size:,} paired samples")
    print(f"  direction: {'increasing' if increasing else 'decreasing'}")
    print(f"  DRR range {x.min():.1f} .. {x.max():.1f}  ->  PXR range {curve.min():.4g} .. {curve.max():.4g}"
          + ("  (normalized PXR units, 0..1)" if args.pxr_norm == "percentile" else ""))
    print(f"  R^2 on samples: {r2:.3f}  (low = the pairs disagree, check registration)")

    np.savez(args.output, grid=grid, curve=curve, pxr_dtype=str(pxr_dtype),
             pxr_norm=args.pxr_norm, norm_lo=args.norm_lo, norm_hi=args.norm_hi)
    print(f"curve saved: {args.output}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        qc = args.output.rsplit(".", 1)[0] + "_qc.png"
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.hexbin(x, y, gridsize=120, bins="log", cmap="Greys")
        if raw_curve is not None:
            lbl = "per-bin medians" if args.method == "median" else "isotonic (staircase)"
            ax.plot(grid, raw_curve, color="orange", lw=1, label=lbl)
        ax.plot(grid, curve, "r-", lw=2, label=f"{args.method} curve (used)")
        ax.set_xlabel("DRR value")
        ax.set_ylabel("PXR value (normalized 0..1)" if args.pxr_norm == "percentile"
                      else "PXR value")
        ax.set_title(f"DRR -> PXR mapping, {args.method} (R^2 = {r2:.3f})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(qc, dpi=110)
        print(f"QC plot saved: {qc}")
    except ImportError:
        print("matplotlib not installed, skipping QC plot")


def cmd_apply(args):
    d = np.load(args.curve)
    grid, curve = d["grid"], d["curve"]
    out_dtype = np.dtype(str(d["pxr_dtype"]))

    drr, res, unit = read_2d(args.drr)
    mapped = np.interp(drr.astype(np.float64), grid, curve)   # clips outside the fitted range

    outside = np.mean((drr < grid[0]) | (drr > grid[-1]))
    if outside > 0.01:
        print(f"  [warn] {outside*100:.1f}% of pixels fall outside the fitted DRR range "
              f"and were clamped to the curve's end values")

    if np.issubdtype(out_dtype, np.integer):
        info = np.iinfo(out_dtype)
        mapped = np.clip(np.rint(mapped), info.min, info.max)

    vals, counts = np.unique(drr, return_counts=True)
    print(f"  most common DRR value: {vals[np.argmax(counts)]}  "
          f"(often the air level, a good starting point for --air-thresh)")
    if args.air_thresh is not None:
        increasing = curve[-1] >= curve[0]          # air = the end that maps to dark PXR
        air = drr <= args.air_thresh if increasing else drr >= args.air_thresh
        mapped[air] = args.air_value
        print(f"  air: DRR {'<=' if increasing else '>='} {args.air_thresh} -> set to "
              f"{args.air_value}  ({100 * air.mean():.1f}% of pixels)")
    mapped = mapped.astype(out_dtype)

    assert mapped.shape == drr.shape
    kw = {}
    if res is not None:
        kw["resolution"] = res
        if unit is not None:
            kw["resolutionunit"] = unit
    tifffile.imwrite(args.output, mapped, **kw)
    print(f"written: {args.output}  shape {mapped.shape}  dtype {mapped.dtype}")


def main():
    p = argparse.ArgumentParser(description="Fit / apply a DRR -> PXR intensity mapping.")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit", help="learn the curve from registered pairs")
    f.add_argument("--pair", nargs=2, action="append", required=True,
                   metavar=("DRR", "PXR"), help="registered DRR and PXR (repeatable)")
    f.add_argument("--mask", action="append", default=None,
                   help="optional mask per pair, same order as --pair; nonzero pixels are used")
    f.add_argument("-o", "--output", default="drr_to_pxr_curve.npz")
    f.add_argument("--bins", type=int, default=64, help="DRR-value bins for stratified sampling")
    f.add_argument("--per-bin", type=int, default=200, help="max samples per bin per pair")
    f.add_argument("--smooth", type=float, default=0.02,
                   help="smoothing window as a fraction of the DRR range (0 = off; isotonic only)")
    f.add_argument("--method", choices=("isotonic", "quantile", "median"), default="isotonic",
                   help="isotonic = regression on paired pixels (needs good alignment); "
                        "quantile = histogram matching (no pixel pairing); "
                        "median = median PXR per DRR bin on paired pixels (robust to "
                        "misaligned outliers, skips DRR == 0)")
    f.add_argument("--pxr-norm", choices=("none", "percentile"), default="none",
                   help="percentile = scale each PXR by its own --norm-lo/--norm-hi "
                        "percentiles to 0..1 before fitting (inside the mask if given); "
                        "the curve then outputs normalized PXR units")
    f.add_argument("--norm-lo", type=float, default=0.1, help="low percentile -> 0")
    f.add_argument("--norm-hi", type=float, default=99.9, help="high percentile -> 1")
    f.add_argument("--max-pool", type=int, default=2_000_000,
                   help="quantile/median: max pixels pooled per pair")
    f.add_argument("--median-bins", type=int, default=256,
                   help="median only: number of DRR-value bins")
    f.add_argument("--min-count", type=int, default=200,
                   help="median only: ignore bins with fewer pixels than this")
    f.add_argument("--seed", type=int, default=0)
    f.set_defaults(func=cmd_fit)

    a = sub.add_parser("apply", help="map a DRR through a fitted curve")
    a.add_argument("curve", help=".npz from the fit step")
    a.add_argument("drr", help="DRR to convert")
    a.add_argument("-o", "--output", required=True)
    a.add_argument("--air-thresh", type=float, default=None,
                   help="DRR value marking air; air pixels get --air-value. The air side "
                        "(below or above) is taken from the curve's direction")
    a.add_argument("--air-value", type=float, default=0,
                   help="value written into air pixels (PXR air is 0)")
    a.set_defaults(func=cmd_apply)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()