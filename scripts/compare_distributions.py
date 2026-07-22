"""
Quantitative comparison of real vs synthetic radiograph pixel intensity
distributions. Deliberately simple: with a small number of real images
(a few dozen), something like FID isn't statistically reliable (it needs
hundreds+ samples to be stable), so this compares normalized pixel
intensity histograms directly, plus the Wasserstein (earth-mover's)
distance, which IS meaningful at small sample sizes.

Two modes:
  1. Single specimen: --real-dir + --synthetic-dir (original behavior)
  2. All specimens: --real-root + --synthetic-root, auto-matches every
     specimen found under both, prints a per-specimen table plus a
     combined (pooled) Wasserstein distance across the whole dataset.

Usage (single specimen):
    python scripts/compare_distributions.py \
        --real-dir dataset/radiograph_tif/CEP011_AS1_radio \
        --synthetic-dir dataset/train/CEP011_AS1

Usage (all specimens, one PNG saved per specimen):
    python scripts/compare_distributions.py \
        --real-root dataset/radiograph_tif \
        --synthetic-root dataset \
        --out-dir specimen_comparisons
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


def get_real_files(real_dir):
    real_dir = Path(real_dir)
    return sorted(list(real_dir.glob("*.jpg")) + list(real_dir.glob("*.jpeg")) +
                  list(real_dir.glob("*.tif")) + list(real_dir.glob("*.tiff")))


def get_synth_files(synth_dir):
    synth_dir = Path(synth_dir)
    return sorted(f for f in synth_dir.glob("*.tif") if "_mask_" not in f.name)


def find_synthetic_dir(synthetic_root, specimen):
    """synthetic_root/{train,val,test}/<specimen> — search all three splits."""
    for split in ("train", "val", "test"):
        candidate = Path(synthetic_root) / split / specimen
        if candidate.exists():
            return candidate
    flat = Path(synthetic_root) / specimen
    if flat.exists():
        return flat
    return None


def main(real_dir=None, synthetic_dir=None, real_root=None, synthetic_root=None,
         out_plot=None, out_dir=None):
    if real_root and synthetic_root:
        run_all_specimens(real_root, synthetic_root, out_plot=out_plot, out_dir=out_dir)
    elif real_dir and synthetic_dir:
        run_single(real_dir, synthetic_dir, out_plot=out_plot)
    else:
        print("Provide either --real-dir/--synthetic-dir, or --real-root/--synthetic-root")


def run_single(real_dir, synthetic_dir, out_plot=None):
    real_dir = Path(real_dir)
    synth_dir = Path(synthetic_dir)

    real_files = get_real_files(real_dir)
    synth_files = get_synth_files(synth_dir)

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


def run_all_specimens(real_root, synthetic_root, out_plot=None, out_dir=None):
    real_root = Path(real_root)
    specimen_dirs = sorted(d for d in real_root.iterdir() if d.is_dir())

    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    all_real_pixels = []
    all_synth_pixels = []

    for real_dir in specimen_dirs:
        specimen = real_dir.name
        if specimen.endswith("_radio"):
            specimen = specimen[: -len("_radio")]

        synth_dir = find_synthetic_dir(synthetic_root, specimen)
        if synth_dir is None:
            print(f"{specimen}: no matching synthetic folder found under {synthetic_root}, skipping")
            continue

        real_files = get_real_files(real_dir)
        synth_files = get_synth_files(synth_dir)

        if not real_files or not synth_files:
            print(f"{specimen}: missing real ({len(real_files)}) or synthetic ({len(synth_files)}) images, skipping")
            continue

        real_pixels = collect_pixels(real_files)
        synth_pixels = collect_pixels(synth_files)
        w = wasserstein_distance(real_pixels, synth_pixels)

        results.append((specimen, len(real_files), len(synth_files), w))
        all_real_pixels.append(real_pixels)
        all_synth_pixels.append(synth_pixels)
        print(f"{specimen}: n_real={len(real_files)} n_synth={len(synth_files)} Wasserstein={w:.4f}")

        if out_dir:
            fig, ax = plt.subplots(figsize=(8, 5))
            bins = np.linspace(0, 1, 100)
            ax.hist(real_pixels, bins=bins, density=True, alpha=0.5,
                    label=f"Real (n={len(real_files)} images)", color="tab:blue")
            ax.hist(synth_pixels, bins=bins, density=True, alpha=0.5,
                    label=f"Synthetic (n={len(synth_files)} images)", color="tab:orange")
            ax.set_xlabel("Normalized pixel intensity")
            ax.set_ylabel("Density")
            ax.set_title(f"{specimen}: real vs synthetic\nWasserstein distance = {w:.4f}")
            ax.legend()
            plt.tight_layout()
            specimen_plot_path = out_dir / f"{specimen}.png"
            plt.savefig(specimen_plot_path, dpi=150)
            plt.close(fig)
            print(f"  saved: {specimen_plot_path}")

    if not results:
        print("No matched specimens found, nothing to compare.")
        return

    print("\n--- Summary across all specimens ---")
    print(f"{'Specimen':<15} {'n_real':>7} {'n_synth':>8} {'Wasserstein':>12}")
    for specimen, n_real, n_synth, w in results:
        print(f"{specimen:<15} {n_real:>7} {n_synth:>8} {w:>12.4f}")

    ws = [w for _, _, _, w in results]
    print(f"\nMean Wasserstein across {len(results)} specimens: {np.mean(ws):.4f}")
    print(f"Min: {min(ws):.4f}  Max: {max(ws):.4f}")

    combined_real = np.concatenate(all_real_pixels)
    combined_synth = np.concatenate(all_synth_pixels)
    combined_w = wasserstein_distance(combined_real, combined_synth)
    print(f"\nCombined (all specimens pooled together) Wasserstein distance: {combined_w:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    specimens = [r[0] for r in results]
    axes[0].bar(specimens, ws, color="tab:purple")
    axes[0].set_ylabel("Wasserstein distance")
    axes[0].set_title("Per-specimen real-vs-synthetic distance")
    axes[0].tick_params(axis="x", rotation=45)

    bins = np.linspace(0, 1, 100)
    axes[1].hist(combined_real, bins=bins, density=True, alpha=0.5, label="Real (all specimens)", color="tab:blue")
    axes[1].hist(combined_synth, bins=bins, density=True, alpha=0.5, label="Synthetic (all specimens)", color="tab:orange")
    axes[1].set_title(f"Combined distribution\nWasserstein = {combined_w:.4f}")
    axes[1].legend()

    plt.tight_layout()
    if out_plot:
        plt.savefig(out_plot, dpi=150)
        print(f"\nSaved summary plot: {out_plot}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-dir", default=None)
    parser.add_argument("--synthetic-dir", default=None)
    parser.add_argument("--real-root", default=None, help="folder containing per-specimen real radiograph subfolders (e.g. dataset/radiograph_tif)")
    parser.add_argument("--synthetic-root", default=None, help="folder containing train/val/test splits (e.g. dataset)")
    parser.add_argument("--out-plot", default=None, help="save summary bar chart + combined histogram to this path")
    parser.add_argument("--out-dir", default=None, help="save one individual histogram PNG per specimen into this folder")
    args = parser.parse_args()
    main(real_dir=args.real_dir, synthetic_dir=args.synthetic_dir,
         real_root=args.real_root, synthetic_root=args.synthetic_root,
         out_plot=args.out_plot, out_dir=args.out_dir)