"""
Cheap orientation search, run this BEFORE calibrate_drr.py. Instead of
rendering a full physics-based DRR at every candidate angle (slow),
this uses a simple sum-projection (just adding up voxel intensities
along the beam axis, no attenuation physics) as a fast stand-in for
shape/structure. It tries flip x rotation angle combinations and
scores each against the real PXR using normalized cross-correlation
(NCC), which is sensitive to spatial alignment, unlike wasserstein
which only looks at intensity distribution shape and can't tell a
mirrored image from a correct one.

Output: the best (flip, angle) combo. You then physically rotate/flip
the CT volume by that amount ONCE (see rotate_volume.py, companion
script) and feed the corrected volume into your normal calibrate_drr.py
+ attenuation sweep pipeline.

Usage:
    python scripts/search_orientation.py \
        --xr /mnt/.../CEP_<specimen>_2026_XR.tif \
        --pxr dataset/radiograph_portable_2026_tif/CEP_<specimen>/<file>.tif \
        --beam-axis 1 \
        --angle-range 20 --angle-step 2 \
        --threshold 150
"""

import argparse

import numpy as np
import tifffile as tiff
from scipy.ndimage import rotate as ndi_rotate
from skimage.transform import resize


def sum_projection(vol, beam_axis, threshold):
    masked = np.where(vol > threshold, vol.astype(np.float64), 0)
    return masked.sum(axis=beam_axis)


def normalized_cross_correlation(a, b):
    a = (a - a.mean()) / (a.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)
    return float(np.mean(a * b))


def main(xr_path, pxr_path, beam_axis, angle_range, angle_step, threshold):
    vol = tiff.imread(xr_path)
    real = tiff.imread(pxr_path).astype(np.float64)
    print(f"CT volume shape: {vol.shape}, PXR shape: {real.shape}")

    angles = np.arange(-angle_range, angle_range + angle_step, angle_step)
    # rotate within the plane perpendicular to the beam axis
    rot_axes = tuple(ax for ax in range(3) if ax != beam_axis)

    results = []
    for flip in (False, True):
        for angle in angles:
            vol_test = np.flip(vol, axis=rot_axes[1]) if flip else vol
            vol_rot = ndi_rotate(vol_test, angle, axes=rot_axes, reshape=False, order=1)
            proj = sum_projection(vol_rot, beam_axis, threshold)
            proj_resized = resize(proj, real.shape, anti_aliasing=True)
            score = normalized_cross_correlation(proj_resized, real)
            results.append((flip, angle, score))

    results.sort(key=lambda r: -r[2])
    print("\nTop 5 (flip, angle, NCC score):")
    for flip, angle, score in results[:5]:
        print(f"  flip={flip}, angle={angle:.1f}deg, NCC={score:.4f}")

    best_flip, best_angle, best_score = results[0]
    print(f"\nBest orientation: flip={best_flip}, angle={best_angle:.1f}deg "
          f"(NCC={best_score:.4f})")
    print("Apply this with rotate_volume.py before running calibrate_drr.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xr", required=True)
    parser.add_argument("--pxr", required=True)
    parser.add_argument("--beam-axis", type=int, default=1, choices=[0, 1, 2],
                         help="axis DRR is projected along, should match calibrate_drr.py's --beam-axis")
    parser.add_argument("--angle-range", type=float, default=20,
                         help="search from -angle-range to +angle-range degrees")
    parser.add_argument("--angle-step", type=float, default=2)
    parser.add_argument("--threshold", type=float, required=True)
    args = parser.parse_args()
    main(args.xr, args.pxr, args.beam_axis, args.angle_range, args.angle_step, args.threshold)