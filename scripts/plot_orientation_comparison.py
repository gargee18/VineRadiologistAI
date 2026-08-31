"""
3-panel visual comparison: real PXR, DRR before orientation correction,
DRR after. Labels the found transform (flip, angle, NCC score) so the
figure is self-contained, no need to cross-reference console output.

Usage:
    python scripts/plot_orientation_comparison.py \
        --pxr dataset/radiograph_portable_2026_tif/CEP_318/69ca5db1e42f5e09df2bb4a6.tif \
        --drr-before results/sanity_check/CEP_318_before_orienting.tif \
        --drr-after results/sanity_check/CEP_318_orientation_test.tif \
        --angle -2.0 --flip True --ncc 0.2154 \
        --specimen 318 \
        --out results/sanity_check/CEP_318_orientation_comparison.png
"""

import argparse

import numpy as np
import tifffile as tiff
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def normalize01(img):
    img = img.astype(np.float64)
    lo, hi = img.min(), img.max()
    if hi <= lo:
        return np.zeros_like(img)
    return (img - lo) / (hi - lo)


def main(pxr_path, drr_before_path, drr_after_path, angle, flip, ncc, specimen, out_path,
         yaw=None, pitch=None):
    pxr = normalize01(tiff.imread(pxr_path))
    drr_before = normalize01(tiff.imread(drr_before_path))
    drr_after = normalize01(tiff.imread(drr_after_path))

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    axes[0].imshow(pxr, cmap="gray")
    axes[0].set_title("PXR (real)")
    axes[0].axis("off")

    axes[1].imshow(drr_before, cmap="gray")
    axes[1].set_title("DRR before orienting")
    axes[1].axis("off")

    axes[2].imshow(drr_after, cmap="gray")
    axes[2].set_title("DRR after orienting")
    axes[2].axis("off")

    title = f"CEP_{specimen} flip={flip}"
    if yaw is not None:
        title += f", yaw={yaw:.1f}deg"
    title += f", roll={angle:.1f}deg"
    if pitch is not None:
        title += f", pitch={pitch:.1f}deg"
    title += f", NCC={ncc:.4f}"

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved comparison figure to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pxr", required=True)
    parser.add_argument("--drr-before", required=True)
    parser.add_argument("--drr-after", required=True)
    parser.add_argument("--angle", type=float, required=True, help="roll")
    parser.add_argument("--yaw", type=float, default=None)
    parser.add_argument("--pitch", type=float, default=None)
    parser.add_argument("--flip", type=str, required=True, choices=["True", "False"])
    parser.add_argument("--ncc", type=float, required=True)
    parser.add_argument("--specimen", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args.pxr, args.drr_before, args.drr_after, args.angle,
         args.flip == "True", args.ncc, args.specimen, args.out, args.yaw, args.pitch)