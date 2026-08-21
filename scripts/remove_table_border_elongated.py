"""
Per-slice filter: any connected component that BOTH touches the image
border AND is elongated (long/thin) gets removed. Matches what Cedric
described: "anything long that touches the border."

Usage:
    python scripts/remove_table_border_elongated.py \
        --xr /mnt/.../CEP_368B_2026_XR.tif \
        --threshold 250 \
        --border-margin 10 \
        --aspect-ratio-min 3.0 \
        --min-voxels 50 \
        --output /mnt/.../CEP_368B_2026_XR_cleaned.tif
"""

import argparse
from pathlib import Path

import numpy as np
import tifffile as tiff
from scipy.ndimage import label, find_objects


def touches_border(mask2d, sl, height, width, margin):
    r0, r1 = sl[0].start, sl[0].stop
    c0, c1 = sl[1].start, sl[1].stop
    return r0 < margin or r1 > height - margin or c0 < margin or c1 > width - margin


def main(xr_path, threshold, border_margin, aspect_ratio_min, min_voxels, output_path):
    with tiff.TiffFile(xr_path) as tf:
        page = tf.pages[0]
        original_dtype = page.asarray().dtype
        xres_tag = page.tags.get("XResolution")
        yres_tag = page.tags.get("YResolution")
        xres = (xres_tag.value[0] / xres_tag.value[1]) if xres_tag else None
        yres = (yres_tag.value[0] / yres_tag.value[1]) if yres_tag else None
        imagej_meta = tf.imagej_metadata or {}

    vol = tiff.imread(xr_path).astype(np.float64)
    n_slices, height, width = vol.shape
    print(f"Volume shape: {vol.shape}")

    vol_cleaned = vol.copy()
    total_removed_components = 0
    total_removed_voxels = 0

    for z in range(n_slices):
        fg2d = vol[z] > threshold
        if not fg2d.any():
            continue

        labeled, n = label(fg2d)
        if n == 0:
            continue

        for i, sl in enumerate(find_objects(labeled), start=1):
            if sl is None:
                continue
            size = (labeled[sl] == i).sum()
            if size < min_voxels:
                continue

            bbox_h = sl[0].stop - sl[0].start
            bbox_w = sl[1].stop - sl[1].start
            aspect_ratio = max(bbox_h, bbox_w) / max(min(bbox_h, bbox_w), 1)

            is_elongated = aspect_ratio > aspect_ratio_min
            is_at_border = touches_border(labeled == i, sl, height, width, border_margin)

            if is_elongated and is_at_border:
                vol_cleaned[z][labeled == i] = 0
                total_removed_components += 1
                total_removed_voxels += size

        if z % 200 == 0:
            print(f"  processed slice {z}/{n_slices}")

    fg_total = (vol > threshold).sum()
    pct = 100 * total_removed_voxels / fg_total if fg_total else 0.0
    print(f"\nRemoved {total_removed_components} component(s) across all slices, "
          f"{total_removed_voxels} voxels total ({pct:.2f}% of foreground)")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if xres:
        tiff.imwrite(output_path, vol_cleaned.astype(original_dtype), imagej=True,
                     resolution=(xres, yres),
                     metadata={"spacing": imagej_meta.get("spacing"),
                               "unit": imagej_meta.get("unit", "mm"),
                               "axes": "ZYX"})
    else:
        tiff.imwrite(output_path, vol_cleaned.astype(original_dtype))
    print(f"Saved cleaned volume to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xr", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--border-margin", type=int, default=10)
    parser.add_argument("--aspect-ratio-min", type=float, default=3.0)
    parser.add_argument("--min-voxels", type=int, default=50)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.xr, args.threshold, args.border_margin, args.aspect_ratio_min,
         args.min_voxels, args.output)