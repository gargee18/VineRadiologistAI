"""
Side-by-side comparison: parallel projection (generate_drr) vs cone-beam
projection (generate_cone_beam_drr) on the same specimen, plus each one's
Wasserstein distance to the real radiograph, so you can see whether
cone-beam actually moves the synthetic distribution closer to real.

Usage:
    python scripts/compare_parallel_vs_conebeam.py \
        --root /path/to/Dataset_Vitimage2019 \
        --specimen CEP011_AS1 \
        --real-dir dataset/radiograph_tif/CEP011_AS1_radio \
        --out-plot parallel_vs_conebeam.png
"""

import argparse
from pathlib import Path

import numpy as np
import tifffile as tiff
import matplotlib

import os
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from PIL import Image

from VineRadiologist import load_specimen, generate_drr
from VineRadiologist.cone_beam import ConeBeamGeometry, generate_cone_beam_drr


def load_real_pixels(real_dir):
    real_dir = Path(real_dir)
    files = sorted(list(real_dir.glob("*.jpg")) + list(real_dir.glob("*.jpeg")) +
                    list(real_dir.glob("*.tif")) + list(real_dir.glob("*.tiff")))
    all_pixels = []
    for f in files:
        if f.suffix.lower() in (".tif", ".tiff"):
            img = tiff.imread(f).astype(np.float64)
        else:
            img = np.array(Image.open(f).convert("L"), dtype=np.float64)
        lo, hi = img.min(), img.max()
        img = (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)
        all_pixels.append(img.ravel())
    return np.concatenate(all_pixels), len(files)


def normalize(img):
    lo, hi = img.min(), img.max()
    return (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)


def main(root, specimen, real_dir, out_plot=None):
    data = load_specimen(root, specimen)
    volume = data["volume"]

    parallel = generate_drr(volume, attenuation_scale=0.015, axis=1)

    # Size the detector so its PHYSICAL footprint (n_pixels * pixel_spacing)
    # actually covers the volume's real physical cross-section, not just
    # match the voxel grid's pixel count. Using pixel count directly with
    # the real (much finer) detector pixel spacing would zoom into a tiny
    # patch near the center instead of showing the whole trunk.
    voxel_spacing_mm = 0.7224
    pixel_spacing_mm = 0.148
    rows = int(volume.shape[0] * voxel_spacing_mm / pixel_spacing_mm)
    cols = int(volume.shape[2] * voxel_spacing_mm / pixel_spacing_mm)

    geometry = ConeBeamGeometry(detector_shape=(rows, cols),
                                 voxel_spacing_mm=voxel_spacing_mm,
                                 detector_pixel_spacing_mm=pixel_spacing_mm)
    cone = generate_cone_beam_drr(volume, geometry, attenuation_scale=0.015, beam_axis=1)

    real_pixels, n_real = load_real_pixels(real_dir)
    parallel_norm = normalize(parallel).ravel()
    cone_norm = normalize(cone).ravel()

    w_parallel = wasserstein_distance(real_pixels, parallel_norm)
    w_cone = wasserstein_distance(real_pixels, cone_norm)

    print(f"Wasserstein distance (real vs PARALLEL projection): {w_parallel:.4f}")
    print(f"Wasserstein distance (real vs CONE-BEAM projection): {w_cone:.4f}")
    if w_cone < w_parallel:
        print(f"-> cone-beam is CLOSER to real by {w_parallel - w_cone:.4f}")
    else:
        print(f"-> parallel is closer (or no improvement); cone-beam did not help here")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(parallel, cmap="gray")
    axes[0].set_title(f"Parallel projection\nWasserstein to real = {w_parallel:.4f}")
    axes[0].axis("off")

    axes[1].imshow(cone, cmap="gray")
    axes[1].set_title(f"Cone-beam projection\nWasserstein to real = {w_cone:.4f}")
    axes[1].axis("off")

    bins = np.linspace(0, 1, 100)
    axes[2].hist(real_pixels, bins=bins, density=True, alpha=0.5, label=f"Real (n={n_real})", color="tab:blue")
    axes[2].hist(parallel_norm, bins=bins, density=True, alpha=0.5, label="Parallel", color="tab:orange")
    axes[2].hist(cone_norm, bins=bins, density=True, alpha=0.5, label="Cone-beam", color="tab:green")
    axes[2].legend()
    axes[2].set_title("Pixel intensity distributions")

    fig.suptitle(f"{specimen}: parallel vs cone-beam vs real", fontweight="bold")
    plt.tight_layout()

    if out_plot:
        plt.savefig(out_plot, dpi=150)
        print(f"Saved plot: {out_plot}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--specimen", required=True)
    parser.add_argument("--real-dir", required=True)
    parser.add_argument("--out-plot", default=None)
    args = parser.parse_args()
    main(args.root, args.specimen, args.real_dir, out_plot=args.out_plot)