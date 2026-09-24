"""
Quantitative comparison of real vs synthetic radiograph pixel intensity
distributions.

Modes
-----
1. Single specimen directory comparison:
   --real-dir + --synthetic-dir

2. All specimens:
   --real-root + --synthetic-root

3. Masked single-pair comparison:
   --real-image + --synthetic-image + --real-mask + --synthetic-mask

For the masked pair mode, only pixels inside the respective masks are used.
The PXR and DRR masks are applied independently, which is appropriate for
distribution-based metrics such as Wasserstein distance.

For the masked pair mode, both images are placed on the same fixed 16-bit
scale by dividing by 65535. They are NOT independently min-max normalized,
so attenuation-induced intensity differences are preserved.
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

    return np.array(
        Image.open(path).convert("L"),
        dtype=np.float64,
    )


def normalize(img):
    """
    Scale one image independently to [0, 1].

    This is retained for the original directory-comparison modes.
    The masked pair mode deliberately does NOT use this function.
    """
    img = img.astype(np.float64)

    lo = img.min()
    hi = img.max()

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

    return sorted(
        list(real_dir.glob("*.jpg"))
        + list(real_dir.glob("*.jpeg"))
        + list(real_dir.glob("*.tif"))
        + list(real_dir.glob("*.tiff"))
    )


def get_synth_files(synth_dir):
    synth_dir = Path(synth_dir)

    return sorted(
        f
        for f in synth_dir.glob("*.tif")
        if "_mask_" not in f.name
    )


def find_synthetic_dir(synthetic_root, specimen):
    """
    Search synthetic_root/{train,val,test}/<specimen>,
    then synthetic_root/<specimen>.
    """
    synthetic_root = Path(synthetic_root)

    for split in ("train", "val", "test"):
        candidate = synthetic_root / split / specimen

        if candidate.exists():
            return candidate

    flat = synthetic_root / specimen

    if flat.exists():
        return flat

    return None


def run_masked_pair(
    real_image,
    synthetic_image,
    real_mask,
    synthetic_mask,
    out_plot=None,
):
    """
    Compare the masked pixel-intensity distributions of one real PXR
    and one synthetic DRR.

    The masks are applied independently:
        PXR pixels = PXR[PXR_mask > 0]
        DRR pixels = DRR[DRR_mask > 0]

    This is appropriate for Wasserstein distribution comparison even
    when the two images are not perfectly registered.
    """

    real = load_image(real_image)
    synthetic = load_image(synthetic_image)

    real_mask_img = load_image(real_mask)
    synthetic_mask_img = load_image(synthetic_mask)

    if real.shape != real_mask_img.shape:
        raise ValueError(
            f"PXR shape {real.shape} != PXR mask shape {real_mask_img.shape}"
        )

    if synthetic.shape != synthetic_mask_img.shape:
        raise ValueError(
            f"DRR shape {synthetic.shape} != DRR mask shape "
            f"{synthetic_mask_img.shape}"
        )

    # Fixed 16-bit scaling.
    # Do not independently min-max normalize these images.
    real_n = np.clip(
        real.astype(np.float64) / 65535.0,
        0.0,
        1.0,
    )

    synthetic_n = np.clip(
        synthetic.astype(np.float64) / 65535.0,
        0.0,
        1.0,
    )

    real_pixels = real_n[real_mask_img > 0]
    synthetic_pixels = synthetic_n[synthetic_mask_img > 0]

    if real_pixels.size == 0:
        raise ValueError("PXR mask contains no foreground pixels.")

    if synthetic_pixels.size == 0:
        raise ValueError("DRR mask contains no foreground pixels.")

    w_dist = wasserstein_distance(
        real_pixels,
        synthetic_pixels,
    )

    print("\n--- Masked pixel intensity summary ---")
    print(
        f"PXR: mean={real_pixels.mean():.6f}  "
        f"std={real_pixels.std():.6f}  "
        f"median={np.median(real_pixels):.6f}  "
        f"n_pixels={len(real_pixels)}"
    )

    print(
        f"DRR: mean={synthetic_pixels.mean():.6f}  "
        f"std={synthetic_pixels.std():.6f}  "
        f"median={np.median(synthetic_pixels):.6f}  "
        f"n_pixels={len(synthetic_pixels)}"
    )

    print(
        f"\nMasked Wasserstein distance: {w_dist:.6f}"
    )

    print(
        "(0 = identical distributions; lower = closer)"
    )

    bins = np.linspace(
        0.0,
        1.0,
        150,
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.hist(
        real_pixels,
        bins=bins,
        density=True,
        alpha=0.5,
        label=f"PXR (n={len(real_pixels):,} pixels)",
    )

    ax.hist(
        synthetic_pixels,
        bins=bins,
        density=True,
        alpha=0.5,
        label=f"DRR (n={len(synthetic_pixels):,} pixels)",
    )

    ax.set_xlabel(
        "Pixel intensity (16-bit value / 65535)"
    )

    ax.set_ylabel(
        "Density"
    )

    ax.set_title(
        "Masked pixel intensity distributions\n"
        f"Wasserstein distance = {w_dist:.6f}"
    )

    ax.legend()

    plt.tight_layout()

    if out_plot:
        out_plot = Path(out_plot)
        out_plot.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.savefig(
            out_plot,
            dpi=150,
        )

        plt.close(fig)

        print(
            f"Saved plot: {out_plot}"
        )

    else:
        plt.show()


def run_single(
    real_dir,
    synthetic_dir,
    out_plot=None,
):
    real_dir = Path(real_dir)
    synth_dir = Path(synthetic_dir)

    real_files = get_real_files(
        real_dir
    )

    synth_files = get_synth_files(
        synth_dir
    )

    if not real_files:
        print(
            f"No real images found in {real_dir}"
        )
        return

    if not synth_files:
        print(
            f"No synthetic images found in {synth_dir}"
        )
        return

    print(
        f"Real images: {len(real_files)}, "
        f"Synthetic images: {len(synth_files)}"
    )

    real_pixels = collect_pixels(
        real_files
    )

    synth_pixels = collect_pixels(
        synth_files
    )

    w_dist = wasserstein_distance(
        real_pixels,
        synth_pixels,
    )

    print(
        "\n--- Pixel intensity summary "
        "(normalized 0-1 per image) ---"
    )

    for name, pixels in [
        ("REAL", real_pixels),
        ("SYNTHETIC", synth_pixels),
    ]:
        print(
            f"{name}: "
            f"mean={pixels.mean():.4f}  "
            f"std={pixels.std():.4f}  "
            f"median={np.median(pixels):.4f}  "
            f"n_pixels={len(pixels)}"
        )

    print(
        f"\nWasserstein distance "
        f"(real vs synthetic): {w_dist:.4f}"
    )

    print(
        "(0 = identical distributions; "
        "no universal 'good' threshold, "
        "lower is closer)"
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    bins = np.linspace(
        0,
        1,
        100,
    )

    ax.hist(
        real_pixels,
        bins=bins,
        density=True,
        alpha=0.5,
        label=f"Real (n={len(real_files)} images)",
    )

    ax.hist(
        synth_pixels,
        bins=bins,
        density=True,
        alpha=0.5,
        label=f"Synthetic (n={len(synth_files)} images)",
    )

    ax.set_xlabel(
        "Normalized pixel intensity"
    )

    ax.set_ylabel(
        "Density"
    )

    ax.set_title(
        "Pixel intensity distribution: "
        "real vs synthetic\n"
        f"Wasserstein distance = {w_dist:.4f}"
    )

    ax.legend()

    plt.tight_layout()

    if out_plot:
        out_plot = Path(out_plot)
        out_plot.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.savefig(
            out_plot,
            dpi=150,
        )

        plt.close(fig)

        print(
            f"\nSaved plot: {out_plot}"
        )

    else:
        plt.show()


def run_all_specimens(
    real_root,
    synthetic_root,
    out_plot=None,
    out_dir=None,
):
    real_root = Path(real_root)

    specimen_dirs = sorted(
        d
        for d in real_root.iterdir()
        if d.is_dir()
    )

    if out_dir:
        out_dir = Path(out_dir)

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    results = []
    all_real_pixels = []
    all_synth_pixels = []

    for real_dir in specimen_dirs:
        specimen = real_dir.name

        if specimen.endswith("_radio"):
            specimen = specimen[:-len("_radio")]

        synth_dir = find_synthetic_dir(
            synthetic_root,
            specimen,
        )

        if synth_dir is None:
            print(
                f"{specimen}: no matching synthetic folder "
                f"found under {synthetic_root}, skipping"
            )
            continue

        real_files = get_real_files(
            real_dir
        )

        synth_files = get_synth_files(
            synth_dir
        )

        if not real_files or not synth_files:
            print(
                f"{specimen}: missing real "
                f"({len(real_files)}) or synthetic "
                f"({len(synth_files)}) images, skipping"
            )
            continue

        real_pixels = collect_pixels(
            real_files
        )

        synth_pixels = collect_pixels(
            synth_files
        )

        w = wasserstein_distance(
            real_pixels,
            synth_pixels,
        )

        results.append(
            (
                specimen,
                len(real_files),
                len(synth_files),
                w,
            )
        )

        all_real_pixels.append(
            real_pixels
        )

        all_synth_pixels.append(
            synth_pixels
        )

        print(
            f"{specimen}: "
            f"n_real={len(real_files)} "
            f"n_synth={len(synth_files)} "
            f"Wasserstein={w:.4f}"
        )

        if out_dir:
            fig, ax = plt.subplots(
                figsize=(8, 5)
            )

            bins = np.linspace(
                0,
                1,
                100,
            )

            ax.hist(
                real_pixels,
                bins=bins,
                density=True,
                alpha=0.5,
                label=f"Real (n={len(real_files)} images)",
            )

            ax.hist(
                synth_pixels,
                bins=bins,
                density=True,
                alpha=0.5,
                label=f"Synthetic (n={len(synth_files)} images)",
            )

            ax.set_xlabel(
                "Normalized pixel intensity"
            )

            ax.set_ylabel(
                "Density"
            )

            ax.set_title(
                f"{specimen}: real vs synthetic\n"
                f"Wasserstein distance = {w:.4f}"
            )

            ax.legend()

            plt.tight_layout()

            specimen_plot_path = (
                out_dir / f"{specimen}.png"
            )

            plt.savefig(
                specimen_plot_path,
                dpi=150,
            )

            plt.close(fig)

            print(
                f"  saved: {specimen_plot_path}"
            )

    if not results:
        print(
            "No matched specimens found, "
            "nothing to compare."
        )
        return

    print(
        "\n--- Summary across all specimens ---"
    )

    print(
        f"{'Specimen':<15} "
        f"{'n_real':>7} "
        f"{'n_synth':>8} "
        f"{'Wasserstein':>12}"
    )

    for specimen, n_real, n_synth, w in results:
        print(
            f"{specimen:<15} "
            f"{n_real:>7} "
            f"{n_synth:>8} "
            f"{w:>12.4f}"
        )

    ws = [
        w
        for _, _, _, w in results
    ]

    print(
        f"\nMean Wasserstein across "
        f"{len(results)} specimens: "
        f"{np.mean(ws):.4f}"
    )

    print(
        f"Min: {min(ws):.4f}  "
        f"Max: {max(ws):.4f}"
    )

    combined_real = np.concatenate(
        all_real_pixels
    )

    combined_synth = np.concatenate(
        all_synth_pixels
    )

    combined_w = wasserstein_distance(
        combined_real,
        combined_synth,
    )

    print(
        f"\nCombined "
        f"(all specimens pooled together) "
        f"Wasserstein distance: "
        f"{combined_w:.4f}"
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5),
    )

    specimens = [
        r[0]
        for r in results
    ]

    axes[0].bar(
        specimens,
        ws,
    )

    axes[0].set_ylabel(
        "Wasserstein distance"
    )

    axes[0].set_title(
        "Per-specimen real-vs-synthetic distance"
    )

    axes[0].tick_params(
        axis="x",
        rotation=45,
    )

    bins = np.linspace(
        0,
        1,
        100,
    )

    axes[1].hist(
        combined_real,
        bins=bins,
        density=True,
        alpha=0.5,
        label="Real (all specimens)",
    )

    axes[1].hist(
        combined_synth,
        bins=bins,
        density=True,
        alpha=0.5,
        label="Synthetic (all specimens)",
    )

    axes[1].set_title(
        "Combined distribution\n"
        f"Wasserstein = {combined_w:.4f}"
    )

    axes[1].legend()

    plt.tight_layout()

    if out_plot:
        out_plot = Path(out_plot)
        out_plot.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.savefig(
            out_plot,
            dpi=150,
        )

        plt.close(fig)

        print(
            f"\nSaved summary plot: {out_plot}"
        )

    else:
        plt.show()


def main(
    real_dir=None,
    synthetic_dir=None,
    real_root=None,
    synthetic_root=None,
    real_image=None,
    synthetic_image=None,
    real_mask=None,
    synthetic_mask=None,
    out_plot=None,
    out_dir=None,
):
    if (
        real_image
        and synthetic_image
        and real_mask
        and synthetic_mask
    ):
        run_masked_pair(
            real_image=real_image,
            synthetic_image=synthetic_image,
            real_mask=real_mask,
            synthetic_mask=synthetic_mask,
            out_plot=out_plot,
        )

    elif real_root and synthetic_root:
        run_all_specimens(
            real_root,
            synthetic_root,
            out_plot=out_plot,
            out_dir=out_dir,
        )

    elif real_dir and synthetic_dir:
        run_single(
            real_dir,
            synthetic_dir,
            out_plot=out_plot,
        )

    else:
        print(
            "Provide one of:\n"
            "  --real-image + --synthetic-image + "
            "--real-mask + --synthetic-mask\n"
            "or\n"
            "  --real-dir + --synthetic-dir\n"
            "or\n"
            "  --real-root + --synthetic-root"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Original single-directory mode
    parser.add_argument(
        "--real-dir",
        default=None,
    )

    parser.add_argument(
        "--synthetic-dir",
        default=None,
    )

    # Original all-specimens mode
    parser.add_argument(
        "--real-root",
        default=None,
        help=(
            "Folder containing per-specimen real radiograph "
            "subfolders."
        ),
    )

    parser.add_argument(
        "--synthetic-root",
        default=None,
        help=(
            "Folder containing train/val/test splits "
            "or flat specimen folders."
        ),
    )

    # Masked single-pair mode
    parser.add_argument(
        "--real-image",
        default=None,
        help="Path to one real PXR image.",
    )

    parser.add_argument(
        "--synthetic-image",
        default=None,
        help="Path to one synthetic DRR image.",
    )

    parser.add_argument(
        "--real-mask",
        default=None,
        help="Path to the PXR specimen mask.",
    )

    parser.add_argument(
        "--synthetic-mask",
        default=None,
        help="Path to the DRR specimen mask.",
    )

    # Outputs
    parser.add_argument(
        "--out-plot",
        default=None,
        help="Save the plot to this path.",
    )

    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "For all-specimens mode, save one histogram "
            "PNG per specimen into this folder."
        ),
    )

    args = parser.parse_args()

    main(
        real_dir=args.real_dir,
        synthetic_dir=args.synthetic_dir,
        real_root=args.real_root,
        synthetic_root=args.synthetic_root,
        real_image=args.real_image,
        synthetic_image=args.synthetic_image,
        real_mask=args.real_mask,
        synthetic_mask=args.synthetic_mask,
        out_plot=args.out_plot,
        out_dir=args.out_dir,
    )
