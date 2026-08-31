"""
Applies flip + yaw + roll + pitch (found via search_orientation_real_drr.py)
to a CT volume once, saves the result. Feed this output into
calibrate_drr.py instead of the original.

Rotation definitions, given beam_axis and the two detector-plane axes
(other_axes[0]="v"/vertical, other_axes[1]="u"/horizontal):
  - roll (--angle):  rotation WITHIN the detector plane (other_axes[0], other_axes[1]).
                      Tilts the final flat image, doesn't change what's visible.
  - yaw (--yaw):     rotation about the vertical axis (beam_axis, other_axes[1]).
                      "Walking around the pot", changes which side faces the beam.
  - pitch (--pitch): rotation about the horizontal axis (other_axes[0], beam_axis).
                      "Tipping the pot toward/away from the camera."

Applied in this FIXED order: flip -> yaw -> roll -> pitch. Must match
the order used in search_orientation_real_drr.py's staged search, or
the found angles won't reproduce the same result.

Usage:
    python scripts/rotate_volume.py \
        --xr /mnt/.../CEP_<specimen>_2026_XR.tif \
        --beam-axis 1 \
        --yaw 20.0 --angle 5.0 --pitch 0.0 \
        --flip \
        --output /mnt/.../CEP_<specimen>_2026_XR_oriented.tif
"""

import argparse
from pathlib import Path

import numpy as np
import tifffile as tiff
from scipy.ndimage import rotate as ndi_rotate


def apply_orientation(vol, beam_axis, flip, yaw, roll, pitch):
    other_axes = tuple(ax for ax in range(3) if ax != beam_axis)
    v_axis, u_axis = other_axes  # v=vertical, u=horizontal

    if flip:
        vol = np.flip(vol, axis=u_axis)

    if yaw != 0:
        vol = ndi_rotate(vol, yaw, axes=(beam_axis, u_axis), reshape=False, order=1)

    if roll != 0:
        vol = ndi_rotate(vol, roll, axes=(v_axis, u_axis), reshape=False, order=1)

    if pitch != 0:
        vol = ndi_rotate(vol, pitch, axes=(v_axis, beam_axis), reshape=False, order=1)

    return vol


def main(xr_path, beam_axis, angle, flip, output_path, yaw=0.0, pitch=0.0):
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

    vol = apply_orientation(vol, beam_axis, flip, yaw, angle, pitch)
    print(f"Applied: flip={flip}, yaw={yaw}, roll={angle}, pitch={pitch}")

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
    parser.add_argument("--angle", type=float, default=0, help="roll, in-plane rotation")
    parser.add_argument("--yaw", type=float, default=0)
    parser.add_argument("--pitch", type=float, default=0)
    parser.add_argument("--flip", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.xr, args.beam_axis, args.angle, args.flip, args.output, args.yaw, args.pitch)