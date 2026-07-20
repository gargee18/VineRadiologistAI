"""
Quantitative comparison of real vs synthetic radiograph pixel intensity
distributions. Deliberately simple: with a small number of real images
, something like FID isn't statistically reliable (it needs
hundreds+ samples to be stable), so this compares normalized pixel
intensity histograms directly, plus the Wasserstein (earth-mover's)
distance, which is meaningful at small sample sizes.

Outputs:
  - overlaid histogram plot (real vs synthetic)
  - printed summary: mean/std/median intensity for each group, and the
    Wasserstein distance between the two distributions (0 = identical
    distributions; larger = more different; there's no universal
    "good" threshold, it's a relative number to track as you tune the
    pipeline)

Usage:
    python scripts/compare_distributions.py \
        --real-dir ~/code_python/VineRadiologistAI/dataset/radiograph_tif/CEP011_AS1_radio \
        --synthetic-dir ~/code_python/VineRadiologistAI/dataset/train/CEP011_AS1
"""

import argparse
from pathlib import Path

import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from PIL import Image


def load_image(path):
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        return tiff.imread(path).astype(np.float64)
    else:
        return np.array(Image.open(path).convert("L"), dtype=np.float64)


def normalize(img):
    """Scale any image to [0, 1] based on its own min/max, so images with
    different bit depths (real 12-16 bit vs synthetic float32) become
    comparable on a common intensity scale."""
    img = img.astype(np.float64)
    lo, hi = img.min(), img.max()
    if hi <= lo:
        return np.zeros_like(img)
    return (img - lo) / (hi - lo)


def collect_pixels(paths):
    all_pixels = []
    for p in paths:
        img = normalize(load_image(p))
        all_pixels.append(img.ravel())
    return np.concatenate(all_pixels)


def main(real_dir, synthetic_dir, out_plot=None):
    real_dir = Path(real_dir)
    synth_dir = Path(synthetic_dir)

    real_files = sorted(list(real_dir.glob("*.jpg")) + list(real_dir.glob("*.jpeg")) +
                         list(real_dir.glob("*.tif")) + list(real_dir.glob("*.tiff")))
    # exclude mask files from synthetic comparison, only compare the actual images
    synth_files = sorted(f for f in synth_dir.glob("*.tif") if "_mask_" not in f.name)

    if not real_files:
        print(f"No real images found in {real_dir}")
        return
    if not synth_files:
        print(f"No synthetic images found in {synth_dir}")
        return

    print(f"Real images: {len(real_files)}, Synthetic images: {len(synth_files)}")

    real_pixels = collect_pixels(real_files)
    synth_pixels = collect_pixels(synth_files)

    w_dist = wasserstein_distance(real_pixels, synth_pixels)

    print("\n--- Pixel intensity summary (normalized 0-1 per image) ---")
    for name, pixels in [("REAL", real_pixels), ("SYNTHETIC", synth_pixels)]:
        print(f"{name}: mean={pixels.mean():.4f}  std={pixels.std():.4f}  "
              f"median={np.median(pixels):.4f}  n_pixels={len(pixels)}")

    print(f"\nWasserstein distance (real vs synthetic): {w_dist:.4f}")
    print("(0 = identical distributions; no universal 'good' threshold, "
          "track this number as you tune the pipeline, lower is closer)")

    # plot overlaid histograms
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 100)
    ax.hist(real_pixels, bins=bins, density=True, alpha=0.5, label=f"Real (n={len(real_files)} images)", color="tab:blue")
    ax.hist(synth_pixels, bins=bins, density=True, alpha=0.5, label=f"Synthetic (n={len(synth_files)} images)", color="tab:orange")
    ax.set_xlabel("Normalized pixel intensity")
    ax.set_ylabel("Density")
    ax.set_title(f"Pixel intensity distribution: real vs synthetic\nWasserstein distance = {w_dist:.4f}")
    ax.legend()
    plt.tight_layout()

    if out_plot:
        plt.savefig(out_plot, dpi=150)
        print(f"\nSaved plot: {out_plot}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-dir", required=True)
    parser.add_argument("--synthetic-dir", required=True)
    parser.add_argument("--out-plot", default=None, help="save plot to this path instead of showing it (useful on a headless server)")
    args = parser.parse_args()
    main(args.real_dir, args.synthetic_dir, out_plot=args.out_plot)