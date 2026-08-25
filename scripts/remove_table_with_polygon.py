"""
Rotoscoping-style panel removal: you draw a polygon tracing the panel
on TWO slices (one where it's small, one where it's big), add both to
the ROI Manager (press 't' on each), then Manager > More > Save...
saves both as one RoiSet.zip. This script reads that zip, auto-detects
each polygon's slice number, interpolates the shape for every slice in
between, and FILLS the INTERIOR of that interpolated polygon with the
local background mean. Everything outside the polygon is untouched.

Usage:
    python scripts/remove_table_with_polygon.py \
        --xr /mnt/.../CEP_368B_2026_XR.tif \
        --roi-zip polygonrois/368B_RoiSet.zip \
        --threshold 150 \
        --output /mnt/.../CEP_368B_2026_XR_filled.tif
"""

import argparse
from pathlib import Path

import numpy as np
import tifffile as tiff
from skimage.draw import polygon as sk_polygon
from roifile import roiread


def resample_polygon(x, y, n_points):
    """Resample a closed polygon to exactly n_points, evenly spaced by
    arc length, so two polygons with different vertex counts can be
    linearly interpolated point-to-point."""
    x = np.append(x, x[0])
    y = np.append(y, y[0])
    seg_lengths = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    cum_len = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_len = cum_len[-1]
    if total_len == 0:
        return np.full(n_points, x[0]), np.full(n_points, y[0])
    sample_at = np.linspace(0, total_len, n_points, endpoint=False)
    x_new = np.interp(sample_at, cum_len, x)
    y_new = np.interp(sample_at, cum_len, y)
    return x_new, y_new


def main(xr_path, roi_zip_path, threshold, n_resample, output_path):
    with tiff.TiffFile(xr_path) as tf:
        page = tf.pages[0]
        original_dtype = page.asarray().dtype
        xres_tag = page.tags.get("XResolution")
        yres_tag = page.tags.get("YResolution")
        xres = (xres_tag.value[0] / xres_tag.value[1]) if xres_tag else None
        yres = (yres_tag.value[0] / yres_tag.value[1]) if yres_tag else None
        imagej_meta = tf.imagej_metadata or {}

    vol = tiff.imread(xr_path)
    n_slices, height, width = vol.shape
    print(f"Volume shape: {vol.shape}")

    rois = roiread(roi_zip_path)
    if len(rois) != 2:
        raise ValueError(f"Expected exactly 2 ROIs in {roi_zip_path}, found {len(rois)}. "
                          f"Names/positions: {[(r.name, r.position) for r in rois]}")

    rois = sorted(rois, key=lambda r: r.position)
    roi_start, roi_end = rois[0], rois[1]
    z_start = roi_start.position - 1  # ImageJ is 1-indexed, our volume is 0-indexed
    z_end = roi_end.position - 1
    print(f"Found 2 ROIs: slice {z_start} (small) -> slice {z_end} (big)")

    coords_start = roi_start.coordinates()
    coords_end = roi_end.coordinates()
    x_start, y_start = resample_polygon(coords_start[:, 0].astype(np.float64),
                                         coords_start[:, 1].astype(np.float64), n_resample)
    x_end, y_end = resample_polygon(coords_end[:, 0].astype(np.float64),
                                     coords_end[:, 1].astype(np.float64), n_resample)

    vol_filled = vol.copy()
    n_filled_slices = 0

    for z in range(z_start, z_end + 1):
        t = (z - z_start) / (z_end - z_start) if z_end != z_start else 0
        x_interp = (1 - t) * x_start + t * x_end
        y_interp = (1 - t) * y_start + t * y_end

        rr, cc = sk_polygon(y_interp, x_interp, shape=(height, width))
        if rr.size == 0:
            continue

        slice_img = vol[z]
        background_pixels = slice_img[slice_img <= threshold]
        fill_value = background_pixels.mean() if background_pixels.size else 0

        slice_filled = slice_img.copy()
        slice_filled[rr, cc] = fill_value
        vol_filled[z] = slice_filled
        n_filled_slices += 1

        if z % 200 == 0:
            print(f"  filled slice {z}/{n_slices}")

    print(f"\nFilled polygon interior on {n_filled_slices} slice(s) "
          f"(z={z_start} to {z_end})")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if xres:
        tiff.imwrite(output_path, vol_filled.astype(original_dtype), imagej=True,
                     resolution=(xres, yres),
                     metadata={"spacing": imagej_meta.get("spacing"),
                               "unit": imagej_meta.get("unit", "mm"),
                               "axes": "ZYX"})
    else:
        tiff.imwrite(output_path, vol_filled.astype(original_dtype))
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xr", required=True)
    parser.add_argument("--roi-zip", required=True,
                         help="RoiSet.zip containing exactly 2 polygon ROIs, saved from "
                              "ROI Manager, positions auto-detected from each ROI's slice")
    parser.add_argument("--threshold", type=float, required=True,
                         help="used only to compute each slice's background mean fill value")
    parser.add_argument("--n-resample", type=int, default=100)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.xr, args.roi_zip, args.threshold, args.n_resample, args.output)