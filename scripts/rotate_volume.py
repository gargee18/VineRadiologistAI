"""
Applies a flip + rotation (found via search_orientation.py) to a CT
volume once, saves the result. Feed this output into calibrate_drr.py
instead of the original.

Usage:
    python scripts/rotate_volume.py \
        --xr /mnt/.../CEP_<specimen>_2026_XR.tif \
        --beam-axis 1 \
        --angle 8.0 \
        --flip \
        --output /mnt/.../CEP_<specimen>_2026_XR_oriented.tif
"""

import argparse
from pathlib import Path

import numpy as np
import tifffile as tiff
from scipy.ndimage import rotate as ndi_rotate


def main(xr_path, beam_axis, angle, flip, output_path):
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

    rot_axes = tuple(ax for ax in range(3) if ax != beam_axis)

    if flip:
        vol = np.flip(vol, axis=rot_axes[1])
        print(f"Flipped along axis {rot_axes[1]}")

    if angle != 0:
        vol = ndi_rotate(vol, angle, axes=rot_axes, reshape=False, order=1)
        print(f"Rotated {angle} degrees in plane {rot_axes}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if xres:
        tiff.imwrite(output_path, vol.astype(original_dtype), imagej=True,
                     resolution=(xres, yres),
                     metadata={"spacing": imagej_meta.get("spacing"),
                               "unit": imagej_meta.get("unit", "mm"),
                               "axes": "ZYX"})
    else:
        tiff.imwrite(output_path, vol.astype(original_dtype))
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xr", required=True)
    parser.add_argument("--beam-axis", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--angle", type=float, default=0)
    parser.add_argument("--flip", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.xr, args.beam_axis, args.angle, args.flip, args.output)