"""
Remove specific 3D connected components (identified via
inspect_3d_components.py as table/support artifacts) from a CT volume,
by zeroing them out. Everything else (trunk, pot, soil, all fused
together) is left completely untouched.

Usage:
    python scripts/remove_3d_components.py \
        --xr /mnt/.../CEP_368B_2026_XR.tif \
        --threshold 500 \
        --remove-labels 1,4376 \
        --output /mnt/.../CEP_368B_2026_XR_cleaned.tif

Run inspect_3d_components.py first to find the label IDs to remove,
labels are only stable for a given --threshold value, use the SAME
threshold here that you used there.
"""

import argparse

import numpy as np
import tifffile as tiff
from scipy.ndimage import label


def main(xr_path, threshold, remove_labels, output_path):
    with tiff.TiffFile(xr_path) as tf:
        page = tf.pages[0]
        original_dtype = page.asarray().dtype
        xres_tag = page.tags.get("XResolution")
        yres_tag = page.tags.get("YResolution")
        xres = (xres_tag.value[0] / xres_tag.value[1]) if xres_tag else None
        yres = (yres_tag.value[0] / yres_tag.value[1]) if yres_tag else None
        imagej_meta = tf.imagej_metadata or {}

    vol = tiff.imread(xr_path).astype(np.float64)
    print(f"Volume shape: {vol.shape}")
    if xres:
        print(f"Preserving original resolution: {1/xres:.7f} mm/px, "
              f"spacing metadata: {imagej_meta.get('spacing')} {imagej_meta.get('unit')}")
    else:
        print("WARNING: no XResolution tag found in source file, output will NOT have "
              "calibrated spacing, check manually with Fiji Properties dialog.")

    fg = vol > threshold
    labeled, n_components = label(fg)
    print(f"Found {n_components} connected component(s) at threshold={threshold}")

    total_removed_voxels = 0
    for lbl in remove_labels:
        mask = labeled == lbl
        count = mask.sum()
        if count == 0:
            print(f"  WARNING: label {lbl} not found (0 voxels), check it matches "
                  f"the same --threshold used in inspect_3d_components.py")
            continue
        vol[mask] = 0
        total_removed_voxels += count
        print(f"  Removed label {lbl}: {count} voxels zeroed")

    print(f"\nTotal removed: {total_removed_voxels} voxels "
          f"({100*total_removed_voxels/fg.sum():.2f}% of original foreground)")

    from pathlib import Path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if xres:
        tiff.imwrite(output_path, vol.astype(original_dtype), imagej=True,
                     resolution=(xres, yres),
                     metadata={"spacing": imagej_meta.get("spacing"),
                               "unit": imagej_meta.get("unit", "mm"),
                               "axes": "ZYX"})
    else:
        tiff.imwrite(output_path, vol.astype(original_dtype))
    print(f"Saved cleaned volume to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xr", required=True, help="path to CT volume tif")
    parser.add_argument("--threshold", type=float, required=True,
                         help="MUST match the threshold used in inspect_3d_components.py, "
                              "labels are only meaningful for that specific threshold")
    parser.add_argument("--remove-labels", required=True,
                         help="comma-separated label IDs to remove, from inspect_3d_components.py output")
    parser.add_argument("--output", required=True, help="path to save the cleaned volume")
    args = parser.parse_args()

    labels_to_remove = [int(v) for v in args.remove_labels.split(",")]
    main(args.xr, args.threshold, labels_to_remove, args.output)