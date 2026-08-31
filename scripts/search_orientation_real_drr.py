"""
Orientation search using REAL cone-beam DRR renders (Beer-Lambert
physics), not the cheap sum-projection search_orientation.py uses.
Slower per-candidate, so this runs SEQUENTIALLY (one GPU).

STAGED 3-axis search (yaw, roll, pitch), not a full grid (too slow):
  Stage 1: search flip x YAW (the "walk around the pot" rotation,
           most likely to be wrong if the CT and PXR were taken from
           genuinely different viewing angles, not just tilted).
  Stage 2: fix flip+yaw, search ROLL (in-plane rotation, what the
           original version searched alone).
  Stage 3: fix flip+yaw+roll, search PITCH (tipping forward/back).

Not guaranteed to find the true global optimum (coordinate-descent
style, not exhaustive), but far more tractable than a full grid, and
covers the actual degrees of freedom, roll alone can't fix a genuine
viewing-angle mismatch, only yaw can.

Usage:
    python scripts/search_orientation_real_drr.py \
        --xr /mnt/.../CEP_<specimen>_2026_XR.tif \
        --pxr dataset/radiograph_portable_2026_tif/CEP_<specimen>/<file>.tif \
        --voxel-spacing-mm 0.585937 --voxel-spacing-z-mm 0.6 \
        --sid-mm 1230 --spd-mm 800 --offset-v-mm -60 \
        --beam-axis 1 \
        --yaw-range 180 --yaw-step 15 \
        --roll-range 30 --roll-step 5 \
        --pitch-range 20 --pitch-step 5 \
        --search-detector-size 384 --downsample 4 --gpu
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import rotate as ndi_rotate

sys.path.insert(0, str(Path(__file__).parent))
import calibrate_drr
from VineRadiologist.io import load_volume
from VineRadiologist.cone_beam import generate_cone_beam_drr, generate_cone_beam_drr_gpu


def normalize01(arr):
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr.astype(np.float64) - lo) / (hi - lo)


def masked_ncc(sim, real, threshold_frac=0.05):
    """NCC computed only over pixels that are foreground (above
    threshold_frac after normalizing to [0,1]) in EITHER image, not the
    whole cropped frame. Wide black background on either side of a
    narrow trunk can otherwise dominate the correlation, background
    matching background inflates the score even when actual structure
    (branch placement) is completely wrong."""
    sim_crop, real_crop = calibrate_drr.auto_crop_to_content(sim, real)
    sim_n = normalize01(sim_crop)
    real_n = normalize01(real_crop)

    mask = (sim_n > threshold_frac) | (real_n > threshold_frac)
    if mask.sum() < 20:  # not enough foreground to trust a correlation
        return 0.0

    sim_vals = sim_n[mask]
    real_vals = real_n[mask]
    a = (sim_vals - sim_vals.mean()) / (sim_vals.std() + 1e-8)
    b = (real_vals - real_vals.mean()) / (real_vals.std() + 1e-8)
    return float(np.mean(a * b))


def render_and_score(vol, geometry, beam_axis, use_gpu, row_chunk, attenuation_scale, real_img):
    if use_gpu:
        drr = generate_cone_beam_drr_gpu(vol, geometry, attenuation_scale=attenuation_scale,
                                          beam_axis=beam_axis, row_chunk=row_chunk, verbose=False)
    else:
        drr = generate_cone_beam_drr(vol, geometry, attenuation_scale=attenuation_scale,
                                      beam_axis=beam_axis)
    return masked_ncc(drr, real_img)


def apply_partial(vol, beam_axis, flip=False, yaw=0.0, roll=0.0, pitch=0.0):
    """Same fixed order as rotate_volume.py: flip -> yaw -> roll -> pitch."""
    other_axes = tuple(ax for ax in range(3) if ax != beam_axis)
    v_axis, u_axis = other_axes

    if flip:
        vol = np.flip(vol, axis=u_axis)
    if yaw != 0:
        vol = ndi_rotate(vol, yaw, axes=(beam_axis, u_axis), reshape=False, order=1)
    if roll != 0:
        vol = ndi_rotate(vol, roll, axes=(v_axis, u_axis), reshape=False, order=1)
    if pitch != 0:
        vol = ndi_rotate(vol, pitch, axes=(v_axis, beam_axis), reshape=False, order=1)
    return vol


def main(xr_path, pxr_path, beam_axis, yaw_range, yaw_step, roll_range, roll_step,
         pitch_range, pitch_step, voxel_spacing_mm, voxel_spacing_z_mm, sid_mm, spd_mm,
         offset_v_mm, search_detector_size, downsample, use_gpu, row_chunk,
         attenuation_scale, beam_width=3):

    vol = load_volume(xr_path)
    real_img = calibrate_drr.load_image(pxr_path)
    real_img, _ = calibrate_drr.strip_saturated_band(real_img)
    print(f"CT volume shape: {vol.shape}, PXR shape: {real_img.shape}")

    if downsample > 1:
        vol = vol[::downsample, ::downsample, ::downsample]
        eff_voxel_spacing_mm = voxel_spacing_mm * downsample
        eff_voxel_spacing_z_mm = voxel_spacing_z_mm * downsample
        print(f"Downsampled volume to {vol.shape} (factor {downsample}x), "
              f"scaled voxel spacing to {eff_voxel_spacing_mm:.4f}mm / "
              f"{eff_voxel_spacing_z_mm:.4f}mm")
    else:
        eff_voxel_spacing_mm = voxel_spacing_mm
        eff_voxel_spacing_z_mm = voxel_spacing_z_mm

    geometry = calibrate_drr.make_geometry(
        search_detector_size, eff_voxel_spacing_mm, eff_voxel_spacing_z_mm,
        sid_mm, spd_mm, offset_v_mm, 0.0)

    def score_of(flip, yaw, roll, pitch):
        vol_test = apply_partial(vol, beam_axis, flip, yaw, roll, pitch)
        return render_and_score(vol_test, geometry, beam_axis, use_gpu, row_chunk,
                                 attenuation_scale, real_img)

    t0 = time.time()
    n_total_rendered = 0

    # --- Stage 1: flip x yaw, keep top beam_width candidates, not just #1 ---
    # A strict greedy top-1 here can permanently lock in a wrong yaw, even
    # if a nearby, slightly-worse-scoring yaw would end up winning once
    # roll/pitch get optimized for it. Carrying multiple candidates
    # forward avoids that trap.
    yaws = np.arange(-yaw_range, yaw_range + yaw_step, yaw_step)
    stage1_tasks = [(flip, yaw) for flip in (False, True) for yaw in yaws]
    print(f"\nStage 1/3: flip x yaw ({len(stage1_tasks)} candidates, "
          f"keeping top {beam_width})...")
    stage1_results = []
    for flip, yaw in stage1_tasks:
        score = score_of(flip, yaw, 0.0, 0.0)
        stage1_results.append((flip, yaw, score))
        n_total_rendered += 1
    stage1_results.sort(key=lambda r: -r[2])
    beam = stage1_results[:beam_width]
    for flip, yaw, score in beam:
        print(f"  candidate: flip={flip}, yaw={yaw:.1f}deg, NCC={score:.4f}")

    # --- Stage 2: roll, refined independently for EACH beam candidate ---
    rolls = np.arange(-roll_range, roll_range + roll_step, roll_step)
    print(f"\nStage 2/3: roll ({len(rolls)} candidates x {len(beam)} beam entries)...")
    beam2 = []
    for flip, yaw, _ in beam:
        best_roll, best_score = 0.0, -np.inf
        for roll in rolls:
            score = score_of(flip, yaw, roll, 0.0)
            n_total_rendered += 1
            if score > best_score:
                best_roll, best_score = roll, score
        beam2.append((flip, yaw, best_roll, best_score))
        print(f"  flip={flip}, yaw={yaw:.1f}: best roll={best_roll:.1f}deg, "
              f"NCC={best_score:.4f}")
    beam2.sort(key=lambda r: -r[3])

    # --- Stage 3: pitch, refined independently for EACH beam candidate ---
    pitches = np.arange(-pitch_range, pitch_range + pitch_step, pitch_step)
    print(f"\nStage 3/3: pitch ({len(pitches)} candidates x {len(beam2)} beam entries)...")
    beam3 = []
    for flip, yaw, roll, _ in beam2:
        best_pitch, best_score = 0.0, -np.inf
        for pitch in pitches:
            score = score_of(flip, yaw, roll, pitch)
            n_total_rendered += 1
            if score > best_score:
                best_pitch, best_score = pitch, score
        beam3.append((flip, yaw, roll, best_pitch, best_score))
        print(f"  flip={flip}, yaw={yaw:.1f}, roll={roll:.1f}: "
              f"best pitch={best_pitch:.1f}deg, NCC={best_score:.4f}")
    beam3.sort(key=lambda r: -r[4])

    best_flip, best_yaw, best_roll, best_pitch, best_score = beam3[0]

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({n_total_rendered} renders total, "
          f"{elapsed/n_total_rendered:.2f}s/render average)")
    print(f"\nAll {len(beam3)} beam finalists, ranked:")
    for flip, yaw, roll, pitch, score in beam3:
        print(f"  flip={flip}, yaw={yaw:.1f}, roll={roll:.1f}, pitch={pitch:.1f}, "
              f"NCC={score:.4f}")
    print(f"\nFinal: flip={best_flip}, yaw={best_yaw:.1f}deg, roll={best_roll:.1f}deg, "
          f"pitch={best_pitch:.1f}deg, NCC={best_score:.4f}")
    print("Apply with rotate_volume.py (--flip --yaw --angle --pitch), then regenerate "
          "the FINAL DRR at full resolution with calibrate_drr.py")
    print("NOTE: still visually verify, this method has produced confident-looking "
          "but wrong answers before (background bias was one cause, ruled out by "
          "masked NCC, but not the only possible failure mode).")

    return best_flip, best_yaw, best_roll, best_pitch, best_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xr", required=True)
    parser.add_argument("--pxr", required=True)
    parser.add_argument("--voxel-spacing-mm", type=float, required=True)
    parser.add_argument("--voxel-spacing-z-mm", type=float, required=True)
    parser.add_argument("--sid-mm", type=float, default=1230.0)
    parser.add_argument("--spd-mm", type=float, default=800.0)
    parser.add_argument("--offset-v-mm", type=float, default=0.0)
    parser.add_argument("--beam-axis", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--yaw-range", type=float, default=180)
    parser.add_argument("--yaw-step", type=float, default=15)
    parser.add_argument("--roll-range", type=float, default=30)
    parser.add_argument("--roll-step", type=float, default=5)
    parser.add_argument("--pitch-range", type=float, default=20)
    parser.add_argument("--pitch-step", type=float, default=5)
    parser.add_argument("--search-detector-size", type=int, default=384)
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--gpu", dest="use_gpu", action="store_true")
    parser.add_argument("--row-chunk", type=int, default=16)
    parser.add_argument("--attenuation-scale", type=float, default=0.02)
    parser.add_argument("--beam-width", type=int, default=3,
                         help="number of top yaw candidates carried through all stages, "
                              "higher = more robust to greedy-search traps but slower")
    args = parser.parse_args()
    main(args.xr, args.pxr, args.beam_axis, args.yaw_range, args.yaw_step,
         args.roll_range, args.roll_step, args.pitch_range, args.pitch_step,
         args.voxel_spacing_mm, args.voxel_spacing_z_mm, args.sid_mm, args.spd_mm,
         args.offset_v_mm, args.search_detector_size, args.downsample,
         args.use_gpu, args.row_chunk, args.attenuation_scale, args.beam_width)