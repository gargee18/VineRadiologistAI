"""
Register ONE DRR to its real PXR: finds the best small rotation + shift
(rigid alignment) that lines them up, saves the aligned DRR. Different
from calibrate_drr.py's offset-v-mm/offset-u-mm, which shift the 3D
projection geometry before rendering, this works on the two already-
rendered 2D images directly, correcting any small residual misalignment
left over after offset tuning.

Usage:
    python scripts/register_drr_to_pxr.py \
        --drr results/PXR_DRR_2026_final/CEP_318.tif \
        --pxr dataset/radiograph_portable_2026_tif/CEP_318/69ca5db1e42f5e09df2bb4a6.tif \
        --angle-range 10 --angle-step 1 \
        --out results/PXR_DRR_2026_final/CEP_318_registered.tif
"""

import argparse
from pathlib import Path

import numpy as np
import tifffile as tiff
from scipy.ndimage import rotate as ndi_rotate, shift as ndi_shift
from skimage.registration import phase_cross_correlation
from skimage.transform import resize


def normalize01(arr):
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr.astype(np.float64) - lo) / (hi - lo)


def estimate_rigid_transform(moving, fixed, angle_range, angle_step):
    moving_norm = normalize01(moving)
    fixed_norm = normalize01(fixed)
    moving_resized = resize(moving_norm, fixed_norm.shape, anti_aliasing=True)

    best = None
    for angle in np.arange(-angle_range, angle_range + angle_step, angle_step):
        rotated = ndi_rotate(moving_resized, angle, reshape=False, order=1)
        shift_est, _, _ = phase_cross_correlation(fixed_norm, rotated, upsample_factor=10)
        shifted = ndi_shift(rotated, shift_est, order=1)

        a = (shifted - shifted.mean()) / (shifted.std() + 1e-8)
        b = (fixed_norm - fixed_norm.mean()) / (fixed_norm.std() + 1e-8)
        score = float(np.mean(a * b))

        if best is None or score > best[2]:
            best = (angle, shift_est, score)

    return best  # (angle, (dy, dx), ncc_score)


def main(drr_path, pxr_path, angle_range, angle_step, out_path):
    with tiff.TiffFile(drr_path) as tf:
        page = tf.pages[0]
        original_dtype = page.asarray().dtype
        xres_tag = page.tags.get("XResolution")
        yres_tag = page.tags.get("YResolution")
        xres = (xres_tag.value[0] / xres_tag.value[1]) if xres_tag else None
        yres = (yres_tag.value[0] / yres_tag.value[1]) if yres_tag else None

    drr = tiff.imread(drr_path).astype(np.float64)
    pxr = tiff.imread(pxr_path).astype(np.float64)
    print(f"DRR shape: {drr.shape}, PXR shape: {pxr.shape}")

    angle, shift_vec, ncc_score = estimate_rigid_transform(drr, pxr, angle_range, angle_step)
    print(f"Best alignment: angle={angle:.2f}deg, shift=(dy={shift_vec[0]:.2f}, "
          f"dx={shift_vec[1]:.2f})px, NCC={ncc_score:.4f}")

    drr_norm = normalize01(drr)
    drr_resized = resize(drr_norm, pxr.shape, anti_aliasing=True)
    rotated = ndi_rotate(drr_resized, angle, reshape=False, order=1)
    aligned = ndi_shift(rotated, shift_vec, order=1)

    aligned_16bit = (np.clip(aligned, 0, 1) * 65535).astype(np.uint16)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if xres:
        tiff.imwrite(out_path, aligned_16bit, resolution=(xres, yres),
                     resolutionunit="CENTIMETER")
    else:
        tiff.imwrite(out_path, aligned_16bit)
    print(f"Saved registered DRR to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drr", required=True)
    parser.add_argument("--pxr", required=True)
    parser.add_argument("--angle-range", type=float, default=10)
    parser.add_argument("--angle-step", type=float, default=1)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args.drr, args.pxr, args.angle_range, args.angle_step, args.out)