"""
Compare pixel intensity distributions of a real PXR image against
multiple DRR variants on one overlaid histogram plot, plus print
per-variant stats and wasserstein distance to real.

Each image is independently min-max normalized to [0,1] before
comparison (same approach used throughout calibrate_drr.py), so raw
16-bit sensor counts and Beer-Lambert float output are on equal footing.

Usage:
    python scripts/compare_distributions_multi.py \
        --real raw_tiff_check/CEP_1191/69ca5e44e42f5e0a22520ec8.tif \
        --invert-real \
        --drr "no_gamma=test_drr_calibration/calibrated_1191/calibrated_1191_wasserstein.tif" \
        --drr "gamma_v1=test_drr_calibration/callibrated_1191_gamma1/calibrated_1191_gamma.tif" \
        --drr "gamma_v2=test_drr_calibration/callibrated_1191_gamma2/calibrated_1191_grid2.tif" \
        --out distribution_compare_4way.png

Each --drr takes a "label=path" pair, repeat the flag for as many DRR
variants as you want on the same plot, not fixed at any specific number.

--invert-real: pass this if the real image is a raw MONOCHROME1 export
(background bright, specimen dark), to match the DRR's convention before
comparing. Don't pass it if the real image is already inverted or in
DRR-matching polarity.
"""

import argparse

import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance


def normalize(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float64)
    lo, hi = img.min(), img.max()
    if hi <= lo:
        return np.zeros_like(img)
    return (img - lo) / (hi - lo)


def parse_drr_arg(arg: str):
    if "=" not in arg:
        raise ValueError(f"--drr expects 'label=path', got: {arg}")
    label, path = arg.split("=", 1)
    return label, path


def main(real_path, invert_real, drr_args, out_path):
    real = tiff.imread(real_path).astype(np.float64)
    if invert_real:
        real = real.max() - real
    real_n = normalize(real)

    entries = [("PXR (real)", real_n)]
    for arg in drr_args:
        label, path = parse_drr_arg(arg)
        drr = tiff.imread(path)
        entries.append((label, normalize(drr)))

    plt.figure(figsize=(9, 6))
    for label, data in entries:
        plt.hist(data.ravel(), bins=100, alpha=0.4, label=label, density=True)
    plt.legend()
    plt.xlabel("normalized intensity [0,1]")
    plt.ylabel("density")
    plt.title("Distribution comparison: real vs DRR variant(s)")
    plt.savefig(out_path, dpi=120)

    print("--- stats (all normalized independently) ---")
    for label, data in entries:
        dist_to_real = wasserstein_distance(real_n.ravel(), data.ravel()) if label != "PXR (real)" else 0.0
        print(f"{label}: mean={data.mean():.4f}  std={data.std():.4f}  "
              f"median={np.median(data):.4f}  wasserstein_to_real={dist_to_real:.4f}")

    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", required=True, help="path to real PXR tif")
    parser.add_argument("--invert-real", action="store_true",
                         help="invert real image before comparing (use for raw MONOCHROME1 exports)")
    parser.add_argument("--drr", action="append", required=True,
                         help="'label=path' pair, repeat for multiple DRR variants")
    parser.add_argument("--out", default="distribution_compare_multi.png")
    args = parser.parse_args()
    main(args.real, args.invert_real, args.drr, args.out)