"""
Show, for several specimens, the plain DRR (no deformation, yaw=0) next to
a deformed+posed synthetic version, side by side in one window.

Usage:
    python scripts/compare_deformations.py \
        --root /home/phukon/code_python/Dataset_Vitimage2019 \
        --n-specimens 5 --seed 0
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from VineRadiologist import (
    load_specimen, DEFAULT_CONFIG, random_bend_and_elastic,
    sample_pose, render_pose, generate_drr,
)


def main(root, specimens, seed=None):
    rng = np.random.default_rng(seed)

    fig, axes = plt.subplots(len(specimens), 2, figsize=(8, 4 * len(specimens)))
    if len(specimens) == 1:
        axes = axes.reshape(1, 2)

    for row, specimen in enumerate(specimens):
        data = load_specimen(root, specimen)
        volume = data["volume"]

        plain = generate_drr(volume, attenuation_scale=0.015, axis=1)

        field_seed = int(rng.integers(0, 2**31 - 1))
        deformed = random_bend_and_elastic(volume, DEFAULT_CONFIG.deformation, field_seed=field_seed)
        pose = sample_pose(DEFAULT_CONFIG.projection, rng)
        posed = render_pose(deformed, pose, axis=DEFAULT_CONFIG.projection.projection_axis)

        axes[row, 0].imshow(plain, cmap="gray")
        axes[row, 0].set_title(f"{specimen}\nplain DRR", fontsize=10)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(posed, cmap="gray")
        axes[row, 1].set_title(f"{specimen}\ndeformed, yaw={pose.yaw:.0f} pitch={pose.pitch:.0f} "
                                f"roll={pose.roll:.1f} dist={pose.distance:.2f}", fontsize=9)
        axes[row, 1].axis("off")

    fig.suptitle("Plain DRR vs deformed + posed synthetic radiograph", fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--specimens", nargs="+", default=None,
                         help="explicit specimen names; if omitted, auto-picks the first --n-specimens found under --root")
    parser.add_argument("--n-specimens", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.specimens:
        specimens = args.specimens
    else:
        root = Path(args.root)
        specimens = sorted(p.name for p in root.iterdir() if p.is_dir())[:args.n_specimens]
        print(f"Auto-selected specimens: {specimens}")

    main(args.root, specimens, seed=args.seed)