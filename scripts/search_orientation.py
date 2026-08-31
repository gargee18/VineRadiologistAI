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

Parallelized across CPU cores (default: all available), each (flip,
angle) combination is independent so this is embarrassingly parallel.
Uses fork (Linux default) so worker processes share the volume in
memory rather than re-serializing it per task.

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
        --threshold 150 \
        --workers 32
"""

import argparse
import multiprocessing as mp
import time

import numpy as np
import tifffile as tiff
from scipy.ndimage import rotate as ndi_rotate
from skimage.transform import resize

# module-level globals, set once before the Pool is created, inherited
# by forked worker processes without re-pickling the (large) volume
_VOL = None
_REAL = None
_BEAM_AXIS = None
_THRESHOLD = None
_ROT_AXES = None


def sum_projection(vol, beam_axis, threshold):
    masked = np.where(vol > threshold, vol.astype(np.float64), 0)
    return masked.sum(axis=beam_axis)


def normalized_cross_correlation(a, b):
    a = (a - a.mean()) / (a.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)
    return float(np.mean(a * b))


def _evaluate_one(args):
    flip, angle = args
    vol_test = np.flip(_VOL, axis=_ROT_AXES[1]) if flip else _VOL
    vol_rot = ndi_rotate(vol_test, angle, axes=_ROT_AXES, reshape=False, order=1)
    proj = sum_projection(vol_rot, _BEAM_AXIS, _THRESHOLD)
    proj_resized = resize(proj, _REAL.shape, anti_aliasing=True)
    score = normalized_cross_correlation(proj_resized, _REAL)
    return (flip, angle, score)


def main(xr_path, pxr_path, beam_axis, angle_range, angle_step, threshold, workers=None,
         downsample=4):
    global _VOL, _REAL, _BEAM_AXIS, _THRESHOLD, _ROT_AXES

    vol = tiff.imread(xr_path)
    real = tiff.imread(pxr_path).astype(np.float64)
    print(f"CT volume shape: {vol.shape}, PXR shape: {real.shape}")

    if downsample > 1:
        vol = vol[::downsample, ::downsample, ::downsample]
        print(f"Downsampled volume to {vol.shape} for the search "
              f"(factor {downsample}x per axis, ~{downsample**3}x fewer voxels), "
              f"apply the winning transform to the FULL-RES original with rotate_volume.py")

    angles = np.arange(-angle_range, angle_range + angle_step, angle_step)
    rot_axes = tuple(ax for ax in range(3) if ax != beam_axis)

    _VOL = vol
    _REAL = real
    _BEAM_AXIS = beam_axis
    _THRESHOLD = threshold
    _ROT_AXES = rot_axes

    tasks = [(flip, angle) for flip in (False, True) for angle in angles]

    n_workers = workers or mp.cpu_count()
    print(f"Evaluating {len(tasks)} (flip, angle) combinations across {n_workers} workers...")

    t0 = time.time()
    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(_evaluate_one, tasks)
    elapsed = time.time() - t0

    print(f"Done in {elapsed:.1f}s ({elapsed/len(tasks):.2f}s per combination on average, "
          f"{n_workers} workers)")

    results.sort(key=lambda r: -r[2])
    print("\nTop 5 (flip, angle, NCC score):")
    for flip, angle, score in results[:5]:
        print(f"  flip={flip}, angle={angle:.1f}deg, NCC={score:.4f}")

    best_flip, best_angle, best_score = results[0]
    print(f"\nBest orientation: flip={best_flip}, angle={best_angle:.1f}deg "
          f"(NCC={best_score:.4f})")
    print("Apply this with rotate_volume.py before running calibrate_drr.py")
    return best_flip, best_angle, best_score


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
    parser.add_argument("--workers", type=int, default=None,
                         help="number of parallel worker processes, defaults to all "
                              "available cores, set explicitly e.g. --workers 32")
    parser.add_argument("--downsample", type=int, default=4,
                         help="downsample the volume by this factor per axis before "
                              "searching (full res isn't needed to find an approximate "
                              "angle, this is the main speed lever). Set to 1 to disable.")
    args = parser.parse_args()
    main(args.xr, args.pxr, args.beam_axis, args.angle_range, args.angle_step,
         args.threshold, args.workers, args.downsample)