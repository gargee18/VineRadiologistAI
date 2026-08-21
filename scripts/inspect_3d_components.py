"""
Label 3D connected components in a CT volume and report their size and
shape, to identify a table/support artifact (typically large, flat/
planar) as distinct from the specimen (irregular, elongated, connected
to the pot).

Usage:
    python scripts/inspect_3d_components.py \
        --xr /mnt/.../CEP_368B_2026_XR.tif \
        --threshold 500

--threshold: HU-ish intensity cutoff for foreground (specimen + table +
pot all typically denser than background/air). Start around the same
ballpark you'd use for a simple foreground mask, tune based on the
printed component stats, real specimen should be one clearly identifiable
large elongated component, not the largest by raw voxel count necessarily
if the table is bigger.
"""

import argparse

import numpy as np
import tifffile as tiff
from scipy.ndimage import label, find_objects


def main(xr_path, threshold, min_voxels):
    vol = tiff.imread(xr_path).astype(np.float64)
    print(f"Volume shape: {vol.shape}, min={vol.min():.1f}, max={vol.max():.1f}, "
          f"mean={vol.mean():.1f}")

    fg = vol > threshold
    print(f"Foreground voxels (> {threshold}): {fg.sum()} ({100*fg.mean():.2f}% of volume)")

    labeled, n_components = label(fg)
    print(f"Found {n_components} connected component(s) total")

    slices = find_objects(labeled)
    components = []
    for i, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        size = (labeled[sl] == i).sum()
        if size < min_voxels:
            continue
        bbox_dims = tuple(s.stop - s.start for s in sl)
        # planarity: ratio of the smallest bbox dimension to the largest,
        # a thin flat slab (table) will have one dimension much smaller
        # than the other two, a trunk/branch structure won't be nearly
        # as flat
        planarity = min(bbox_dims) / max(bbox_dims)
        components.append((i, size, bbox_dims, planarity, sl))

    components.sort(key=lambda c: -c[1])  # largest first by voxel count

    print(f"\n{len(components)} component(s) with >= {min_voxels} voxels, "
          f"sorted by size (largest first):\n")
    print(f"{'label':>6} {'voxels':>10} {'bbox (z,y,x)':>20} {'planarity':>10}  note")
    for i, size, bbox_dims, planarity, sl in components:
        note = ""
        if planarity < 0.15:
            note = "<- likely FLAT/PLANAR (possible table artifact)"
        print(f"{i:>6} {size:>10} {str(bbox_dims):>20} {planarity:>10.3f}  {note}")

    print(f"\nLower planarity (closer to 0) = flatter/more slab-like. "
          f"The real specimen should be the component with LOW planarity "
          f"score relative to a thin table, even if the table has more "
          f"raw voxels. Confirm visually before trusting this automatically.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xr", required=True, help="path to CT volume tif")
    parser.add_argument("--threshold", type=float, default=500,
                         help="intensity threshold for foreground (tune based on output)")
    parser.add_argument("--min-voxels", type=int, default=1000,
                         help="ignore components smaller than this (noise)")
    args = parser.parse_args()
    main(args.xr, args.threshold, args.min_voxels)