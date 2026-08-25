"""
One rule: per slice, if a connected component touches the image
border, remove it. Nothing else.

Usage:
    python scripts/remove_table.py \
        --xr /mnt/.../CEP_368B_2026_XR.tif \
        --threshold 250 \
        --border-margin 1 \
        --output /mnt/.../CEP_368B_2026_XR_cleaned.tif
"""

import argparse
from pathlib import Path

import numpy as np
import tifffile as tiff
from scipy.ndimage import label, find_objects


def remove_border_touching(vol, threshold, border_margin):
    n_slices, height, width = vol.shape
    vol_cleaned = np.zeros_like(vol)
    total_removed_components = 0
    total_removed_voxels = 0
    total_kept_voxels = 0

    for z in range(n_slices):
        fg2d = vol[z] > threshold
        if not fg2d.any():
            continue

        labeled, n = label(fg2d)
        if n == 0:
            continue

        mask2d = np.zeros(fg2d.shape, dtype=bool)
        for i, sl in enumerate(find_objects(labeled), start=1):
            if sl is None:
                continue
            r0, r1 = sl[0].start, sl[0].stop
            c0, c1 = sl[1].start, sl[1].stop
            touches_border = (r0 < border_margin or r1 > height - border_margin or
                               c0 < border_margin or c1 > width - border_margin)
            size = (labeled[sl] == i).sum()
            if touches_border:
                total_removed_components += 1
                total_removed_voxels += size
                continue
            mask2d[labeled == i] = True

        vol_cleaned[z] = np.where(mask2d, vol[z], 0)
        total_kept_voxels += mask2d.sum()

        if z % 200 == 0:
            print(f"  processed slice {z}/{n_slices}")

    fg_total = (vol > threshold).sum()
    pct_removed = 100 * total_removed_voxels / fg_total if fg_total else 0.0
    print(f"\nRemoved {total_removed_components} component(s), "
          f"{total_removed_voxels} voxels ({pct_removed:.2f}% of foreground)")
    print(f"Kept {total_kept_voxels} voxels "
          f"({100*total_kept_voxels/fg_total:.2f}% of foreground survived)")
    return vol_cleaned


def main(xr_path, threshold, border_margin, output_path):
    with tiff.TiffFile(xr_path) as tf:
        page = tf.pages[0]
        original_dtype = page.asarray().dtype
        xres_tag = page.tags.get("XResolution")
        yres_tag = page.tags.get("YResolution")
        xres = (xres_tag.value[0] / xres_tag.value[1]) if xres_tag else None
        yres = (yres_tag.value[0] / yres_tag.value[1]) if yres_tag else None
        imagej_meta = tf.imagej_metadata or {}

    vol = tiff.imread(xr_path)
    print(f"Volume shape: {vol.shape}")

    vol_final = remove_border_touching(vol, threshold, border_margin)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if xres:
        tiff.imwrite(output_path, vol_final.astype(original_dtype), imagej=True,
                     resolution=(xres, yres),
                     metadata={"spacing": imagej_meta.get("spacing"),
                               "unit": imagej_meta.get("unit", "mm"),
                               "axes": "ZYX"})
    else:
        tiff.imwrite(output_path, vol_final.astype(original_dtype))
    print(f"Saved cleaned volume to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xr", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--border-margin", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.xr, args.threshold, args.border_margin, args.output)