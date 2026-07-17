"""QA visualization: reproduces the original mask-projection-overlay figures
using the refactored VineRadiologist modules."""

import argparse
import matplotlib.pyplot as plt

from VineRadiologist import load_specimen, generate_drr, apply_yaw
from VineRadiologist.visualize import compute_mask_projs, plot_projections, plot_all_tissues_merged

DEFAULT_ANGLES = [0, 30, 60, 90]


def main(root, dataset, specimen, angles=DEFAULT_ANGLES):
    data = load_specimen(root, specimen, dataset)
    volume = data["volume"]

    ct_rots = [apply_yaw(volume, a) for a in angles]
    drr_projs = [generate_drr(ct, axis=1) for ct in ct_rots]

    wr = compute_mask_projs(data["whiterot"], angles)
    nc = compute_mask_projs(data["necrosis"], angles)
    hl = compute_mask_projs(data["healthy"], angles)

    plot_projections(drr_projs, wr, angles, "White Rot Mask Projections Overlaid on DRR")
    plot_projections(drr_projs, nc, angles, "Necrosis Mask Projections Overlaid on DRR")
    plot_projections(drr_projs, hl, angles, "Healthy Mask Projections Overlaid on DRR")

    fig, sliders = plot_all_tissues_merged(drr_projs, wr, nc, hl, angles, "All Tissues Overlaid on DRR")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--specimen", required=True)
    args = parser.parse_args()
    main(args.root, args.dataset, args.specimen)
