#!/usr/bin/env python3
"""
masked_distributions.py

For each specimen: apply its mask to BOTH the PXR and the DRR (they are
registered, so one mask fits both), plot the two intensity distributions,
and print mean / std. At the end: one plot with the MEAN distribution over
all specimens (each specimen weighted equally), with a +-1 std band.

Paths use {spec} as the specimen placeholder, e.g. CEP_{spec}_mask.tif.

USAGE (from the repo root)
    python scripts/masked_distributions.py \
        --mask "results/masks/CEP_{spec}_mask_PXR.tif" \
        --drr  "results/CEP_{spec}_airzero_calibration_linear.tif" \
        --out-dir results/distributions_masked

    # same, but DRR passed through the fitted curve first (nothing is saved)
    python scripts/masked_distributions.py ... --curve results/curve_quantile_airzero_16spec.npz

    # only some specimens
    python scripts/masked_distributions.py ... --specs 322 1195 1193 323
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

PXR_DIR = "dataset/radiograph_portable_2026_tif"
PXR_FILE = {
    "313B": "69ca5d9ee42f5e088ff0f93b", "318": "69ca5db1e42f5e09df2bb4a6",
    "322": "69ca5dc4e42f5e09df2bb4ac", "323": "69ca5de6e42f5e09b4512919",
    "330": "69ca5df7e42f5e09df2bb4b0", "335": "69ca5e1be42f5e0a22520ec4",
    "368B": "69ca536b2ad9c81239913207", "378A": "69ca4b43e42f5e026f6dd0e7",
    "378B": "69ca5613e42f5e07eeeed92b", "380A": "69ca562de42f5e0682a2cf2d",
    "764B": "69ca5d80e42f5e088ff0f937", "988B": "69ca5645e42f5e0682a2cf31",
    "1181": "69ca6a532ad9c81239913234", "1186A": "69ca569de42f5e071ae9bcf3",
    "1189": "69ca67ad2ad9c8123991322c", "1191": "69ca5e44e42f5e0a22520ec8",
    "1193": "69ca5e6fe42f5e09df2bb4b4", "1195": "69ca56b6e42f5e071ae9bcfa",
    "1266A": "69ca56d2e42f5e0702f748da", "2184A": "69ca5676e42f5e071ae9bced",
}

N_BINS = 256
VALUE_MAX = 65535.0                      # 16-bit range, same axis for PXR and DRR
COLOR_PXR = "#2a78d6"                    # categorical slot 1 (blue)
COLOR_DRR = "#eb6834"                    # categorical slot 2 (orange)
TEXT_MUTED = "#52514e"


def read_2d(path):
    arr = np.squeeze(tifffile.imread(path)).astype(np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{path}: expected a 2D image, got shape {arr.shape}")
    return arr


def style(ax, title, xlabel="Stored 16-bit value"):
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_xlabel(xlabel, color=TEXT_MUTED)
    ax.set_ylabel("Probability density", color=TEXT_MUTED)
    ax.grid(axis="y", alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=TEXT_MUTED)
    ax.legend(frameon=False)


def main():
    p = argparse.ArgumentParser(description="Masked PXR vs DRR distributions.")
    p.add_argument("--mask", required=True, help="mask path with {spec}; nonzero = used")
    p.add_argument("--drr", required=True, help="DRR path with {spec}")
    p.add_argument("--curve", help="optional .npz from fit_drr_to_pxr.py; maps the DRR first")
    p.add_argument("--specs", nargs="+", default=list(PXR_FILE), help="default: all 20")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--exclude-background", action="store_true",
                   help="drop mask pixels where the raw PXR <= --bg-pxr or the raw DRR <= "
                        "--bg-drr (e.g. background leaking in through registration error), "
                        "from BOTH images; the count dropped is printed per specimen")
    p.add_argument("--bg-pxr", type=float, default=0, help="raw PXR value counted as background")
    p.add_argument("--bg-drr", type=float, default=0, help="raw DRR value counted as background")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    curve, norm = None, None
    if args.curve:
        c = np.load(args.curve)
        curve = (c["grid"], c["curve"])
        if "pxr_norm" in c.files and str(c["pxr_norm"]) == "percentile":
            norm = (float(c["norm_lo"]), float(c["norm_hi"]))
            print(f"Normalized curve: each PXR is scaled by its own p{norm[0]:g}/p{norm[1]:g} "
                  f"inside the mask, so both are compared on 0..1.")
    drr_label = "DRR (curve-mapped)" if curve else "DRR"

    value_max = 1.0 if norm else VALUE_MAX
    nf = ".3f" if norm else ".0f"
    xlabel = "Normalized value (each PXR by its own percentiles)" if norm else "Stored 16-bit value"
    bins = np.linspace(0.0, value_max, N_BINS + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    hist_pxr, hist_drr = [], []

    print(f"{'specimen':9s} {'pixels':>10s} {'PXR mean':>9s} {'PXR std':>8s} "
          f"{'DRR mean':>9s} {'DRR std':>8s} {'mean diff':>10s}")
    for spec in args.specs:
        if spec not in PXR_FILE:
            print(f"{spec:9s} SKIPPED: unknown specimen")
            continue
        paths = {
            "PXR": Path(PXR_DIR) / f"CEP_{spec}" / f"{PXR_FILE[spec]}.tif",
            "DRR": Path(args.drr.format(spec=spec)),
            "mask": Path(args.mask.format(spec=spec)),
        }
        missing = [f"{k}: {v}" for k, v in paths.items() if not v.is_file()]
        if missing:
            print(f"{spec:9s} SKIPPED, missing {'; '.join(missing)}")
            continue
        try:
            pxr, drr, mask = (read_2d(paths[k]) for k in ("PXR", "DRR", "mask"))
            if not (pxr.shape == drr.shape == mask.shape):
                raise ValueError(f"shapes differ: PXR {pxr.shape}, DRR {drr.shape}, "
                                 f"mask {mask.shape}")
        except ValueError as exc:
            print(f"{spec:9s} SKIPPED: {exc}")
            continue

        background = (pxr <= args.bg_pxr) | (drr <= args.bg_drr)   # raw values, before any mapping
        if curve is not None:
            drr = np.interp(drr, *curve)
        keep = (mask > 0) & np.isfinite(pxr) & np.isfinite(drr)
        if norm is not None and keep.any():
            # same region as the fit (whole mask), so the scaling matches the curve
            lo, hi = np.percentile(pxr[keep], norm)
            pxr = np.clip((pxr - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
        if args.exclude_background and keep.any():
            dropped = keep & background
            print(f"{spec:9s} background excluded: {dropped.sum():,} of {keep.sum():,} mask "
                  f"pixels ({100 * dropped.sum() / keep.sum():.1f}%)")
            keep &= ~background
        if not keep.any():
            print(f"{spec:9s} SKIPPED: no pixels left")
            continue
        p_vals, d_vals = pxr[keep], drr[keep]
        hp, _ = np.histogram(p_vals, bins=bins, density=True)
        hd, _ = np.histogram(d_vals, bins=bins, density=True)
        hist_pxr.append(hp)
        hist_drr.append(hd)

        print(f"{spec:9s} {keep.sum():10,d} {p_vals.mean():9{nf}} {p_vals.std():8{nf}} "
              f"{d_vals.mean():9{nf}} {d_vals.std():8{nf}} "
              f"{100 * (d_vals.mean() - p_vals.mean()) / p_vals.mean():+9.1f}%")

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(centers, hp, color=COLOR_PXR, lw=2,
                label=f"PXR  (mean {p_vals.mean():{nf}}, std {p_vals.std():{nf}})")
        ax.plot(centers, hd, color=COLOR_DRR, lw=2,
                label=f"{drr_label}  (mean {d_vals.mean():{nf}}, std {d_vals.std():{nf}})")
        style(ax, f"CEP_{spec}: inside mask ({keep.sum():,} pixels)", xlabel)
        fig.tight_layout()
        fig.savefig(out / f"CEP_{spec}_masked_dist.png", dpi=150)
        plt.close(fig)

    if not hist_pxr:
        print("No specimen could be processed, no summary plot.")
        return

    hp, hd = np.array(hist_pxr), np.array(hist_drr)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for h, color, label in ((hp, COLOR_PXR, "PXR"), (hd, COLOR_DRR, drr_label)):
        m, s = h.mean(axis=0), h.std(axis=0)
        ax.fill_between(centers, np.clip(m - s, 0, None), m + s, color=color, alpha=0.15, lw=0)
        ax.plot(centers, m, color=color, lw=2, label=f"{label}, mean over specimens")
    style(ax, f"Mean distribution over {len(hp)} specimens (band = ±1 std)", xlabel)
    fig.tight_layout()
    fig.savefig(out / "mean_masked_dist.png", dpi=150)
    plt.close(fig)
    print(f"\n{len(hp)} specimen plots + mean_masked_dist.png written to {out}")


if __name__ == "__main__":
    main()