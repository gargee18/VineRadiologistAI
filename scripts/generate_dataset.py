"""
Generate a batch of synthetic radiographs for one specimen using the
vineradiology pipeline: elastic/bend deformation -> pose sampling
(yaw/pitch/roll/distance) -> Beer-Lambert projection.

With --with-masks, the same deformation and pose are also applied to the
three tissue masks (white rot, necrosis, healthy) so they stay aligned with
the generated image, and both get displayed as a colored overlay.

Usage:
    python scripts/generate_dataset.py --root /path/to/data \
        --specimen CEP011_AS1 --n-samples 20 --out out/CEP011_AS1

    python scripts/generate_dataset.py --root /path/to/data \
        --specimen CEP011_AS1 --n-samples 6 --out out/CEP011_AS1 --with-masks
"""

import argparse
from pathlib import Path

import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

from vineradiology import (
    load_specimen, DEFAULT_CONFIG, random_bend_and_elastic,
    sample_pose, render_pose, render_pose_mask,
)

TISSUE_COLORS = {
    "whiterot": np.array([1.0, 0.0, 0.0]),
    "necrosis": np.array([0.0, 0.4, 1.0]),
    "healthy": np.array([0.0, 0.7, 0.0]),
}


def _overlay_masks(img, mask_projs, alpha=0.45):
    """Build an RGB overlay: grayscale image + colored tissue masks on top."""
    rgb = np.stack([img, img, img], axis=-1)
    rgb = np.clip(rgb, 0, 1)
    for name, proj in mask_projs.items():
        valid = proj > 0.5
        color = TISSUE_COLORS[name]
        rgb[valid] = (1 - alpha) * rgb[valid] + alpha * color
    return rgb


def generate(root, dataset, specimen, n_samples, out_dir, cfg=DEFAULT_CONFIG,
             seed=None, show=True, with_masks=False):
    rng = np.random.default_rng(seed)
    data = load_specimen(root, specimen, dataset)
    volume = data["volume"]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generated = []
    for i in range(n_samples):
        field_seed = int(rng.integers(0, 2**31 - 1))

        deformed = random_bend_and_elastic(volume, cfg.deformation, field_seed=field_seed)
        pose = sample_pose(cfg.projection, rng)
        img = render_pose(deformed, pose, axis=cfg.projection.projection_axis)
        tiff.imwrite(out_dir / f"sample_{i:04d}.tif", img.astype(np.float32))

        mask_projs = None
        if with_masks:
            mask_projs = {}
            for name in ("whiterot", "necrosis", "healthy"):
                deformed_mask = random_bend_and_elastic(data[name], cfg.deformation, field_seed=field_seed)
                proj = render_pose_mask(deformed_mask, pose, axis=cfg.projection.projection_axis)
                mask_projs[name] = proj
                tiff.imwrite(out_dir / f"sample_{i:04d}_mask_{name}.tif", proj.astype(np.float32))

        print(f"[{i+1}/{n_samples}] yaw={pose.yaw:.1f} pitch={pose.pitch:.1f} "
              f"roll={pose.roll:.1f} distance={pose.distance:.2f} -> "
              f"sample_{i:04d}.tif" + (" (+ masks)" if with_masks else ""))
        generated.append((img, pose, mask_projs))

    if show:
        fig, axes = plt.subplots(1, n_samples, figsize=(4 * n_samples, 4))
        if n_samples == 1:
            axes = [axes]
        for ax, (img, pose, mask_projs) in zip(axes, generated):
            if with_masks:
                ax.imshow(_overlay_masks(img, mask_projs))
            else:
                ax.imshow(img, cmap="gray")
            ax.set_title(f"yaw={pose.yaw:.0f} pitch={pose.pitch:.0f}\n"
                         f"roll={pose.roll:.1f} dist={pose.distance:.2f}", fontsize=9)
            ax.axis("off")
        fig.suptitle(f"{specimen}: {n_samples} synthetic samples" +
                     (" (red=white rot, blue=necrosis, green=healthy)" if with_masks else ""),
                     fontweight="bold", fontsize=11)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--specimen", required=True)
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-show", action="store_true", help="skip displaying results inline")
    parser.add_argument("--with-masks", action="store_true",
                         help="also deform/pose the tissue masks and overlay them on the display")
    args = parser.parse_args()

    generate(args.root, args.dataset, args.specimen, args.n_samples, args.out,
              seed=args.seed, show=not args.no_show, with_masks=args.with_masks)