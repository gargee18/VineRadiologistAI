"""
Runs the full orientation-correction pipeline in one command instead of
four+ manual steps: staged 3-axis search (yaw, roll, pitch) using REAL
rendered DRRs, apply it, generate the final DRR, produce a comparison
figure, all threading found values through automatically.

Usage:
    python scripts/run_orientation_pipeline.py \
        --specimen 378A \
        --xr /mnt/.../dataset/ct_volumes_table_removed/CEP_378A_2026_XR.tif \
        --pxr dataset/radiograph_portable_2026_tif/CEP_378A/69ca4b43e42f5e026f6dd0e7.tif \
        --drr-before results/PXR_DRR_2026_final/CEP_378A.tif \
        --voxel-spacing-mm 0.617187 --voxel-spacing-z-mm 0.4 \
        --sid-mm 1230 --spd-mm 800 --offset-v-mm -10 \
        --detector-size 3072 --gpu --row-chunk 16 \
        --yaw-range 180 --yaw-step 15 --roll-range 30 --roll-step 5 \
        --pitch-range 20 --pitch-step 5 --search-detector-size 384 --downsample 4 \
        --oriented-dir dataset/ct_volumes_oriented \
        --drr-out-dir results/PXR_DRR_2026_registered/DRR \
        --plot-out-dir results/PXR_DRR_2026_registered/Comparison_Plots
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import search_orientation_real_drr
import rotate_volume
import calibrate_drr
import plot_orientation_comparison


LOW_NCC_WARNING_THRESHOLD = 0.3


def main(specimen, xr_path, pxr_path, drr_before_path, voxel_spacing_mm, voxel_spacing_z_mm,
         sid_mm, spd_mm, offset_v_mm, detector_size, use_gpu, row_chunk,
         yaw_range, yaw_step, roll_range, roll_step, pitch_range, pitch_step,
         search_detector_size, downsample, attenuation_scale,
         beam_axis, oriented_dir, drr_out_dir, plot_out_dir):

    print(f"\n{'='*20} STEP 1: orientation search, real DRRs ({specimen}) {'='*20}")
    flip, yaw, roll, pitch, ncc = search_orientation_real_drr.main(
        xr_path, pxr_path, beam_axis, yaw_range, yaw_step, roll_range, roll_step,
        pitch_range, pitch_step, voxel_spacing_mm, voxel_spacing_z_mm, sid_mm, spd_mm,
        offset_v_mm, search_detector_size, downsample, use_gpu, row_chunk, attenuation_scale)

    if ncc < LOW_NCC_WARNING_THRESHOLD:
        print(f"\n*** WARNING: best NCC ({ncc:.4f}) is below {LOW_NCC_WARNING_THRESHOLD}, "
              f"VISUALLY CHECK the comparison plot before trusting this result. ***\n")

    oriented_path = str(Path(oriented_dir) / f"CEP_{specimen}_2026_XR.tif")
    print(f"\n{'='*20} STEP 2: applying transform {'='*20}")
    rotate_volume.main(xr_path, beam_axis, roll, flip, oriented_path, yaw, pitch)

    drr_after_path = str(Path(drr_out_dir) / f"CEP_{specimen}.tif")
    print(f"\n{'='*20} STEP 3: generating DRR from oriented volume {'='*20}")
    calibrate_drr.main(
        oriented_path, pxr_path, "wasserstein", drr_after_path,
        detector_size=detector_size, voxel_spacing_mm=voxel_spacing_mm,
        voxel_spacing_z_mm=voxel_spacing_z_mm, fixed_attenuation=[attenuation_scale],
        beam_axis=beam_axis, sid_mm=sid_mm, spd_mm=spd_mm,
        offset_v_mm=offset_v_mm, use_gpu=use_gpu, row_chunk=row_chunk)

    plot_path = str(Path(plot_out_dir) / f"CEP_{specimen}.png")
    print(f"\n{'='*20} STEP 4: comparison plot {'='*20}")
    plot_orientation_comparison.main(
        pxr_path, drr_before_path, drr_after_path, roll, flip, ncc,
        specimen, plot_path, yaw, pitch)

    print(f"\n{'='*20} DONE ({specimen}) {'='*20}")
    print(f"  flip={flip}, yaw={yaw:.1f}, roll={roll:.1f}, pitch={pitch:.1f}, NCC={ncc:.4f}")
    print(f"  oriented volume: {oriented_path}")
    print(f"  DRR:             {drr_after_path}")
    print(f"  comparison plot: {plot_path}")
    if ncc < LOW_NCC_WARNING_THRESHOLD:
        print(f"  *** low-confidence result (NCC={ncc:.4f}), check the plot before using ***")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--specimen", required=True)
    parser.add_argument("--xr", required=True, dest="xr_path")
    parser.add_argument("--pxr", required=True, dest="pxr_path")
    parser.add_argument("--drr-before", required=True, dest="drr_before_path")
    parser.add_argument("--voxel-spacing-mm", type=float, required=True)
    parser.add_argument("--voxel-spacing-z-mm", type=float, required=True)
    parser.add_argument("--sid-mm", type=float, default=1230.0)
    parser.add_argument("--spd-mm", type=float, default=800.0)
    parser.add_argument("--offset-v-mm", type=float, default=0.0)
    parser.add_argument("--detector-size", type=int, default=3072)
    parser.add_argument("--gpu", dest="use_gpu", action="store_true")
    parser.add_argument("--row-chunk", type=int, default=16)
    parser.add_argument("--yaw-range", type=float, default=180)
    parser.add_argument("--yaw-step", type=float, default=15)
    parser.add_argument("--roll-range", type=float, default=30)
    parser.add_argument("--roll-step", type=float, default=5)
    parser.add_argument("--pitch-range", type=float, default=20)
    parser.add_argument("--pitch-step", type=float, default=5)
    parser.add_argument("--search-detector-size", type=int, default=384)
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--attenuation-scale", type=float, default=0.02)
    parser.add_argument("--beam-axis", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--oriented-dir", required=True)
    parser.add_argument("--drr-out-dir", required=True)
    parser.add_argument("--plot-out-dir", required=True)
    args = parser.parse_args()

    for d in (args.oriented_dir, args.drr_out_dir, args.plot_out_dir):
        Path(d).mkdir(parents=True, exist_ok=True)

    main(args.specimen, args.xr_path, args.pxr_path, args.drr_before_path,
         args.voxel_spacing_mm, args.voxel_spacing_z_mm, args.sid_mm, args.spd_mm,
         args.offset_v_mm, args.detector_size, args.use_gpu, args.row_chunk,
         args.yaw_range, args.yaw_step, args.roll_range, args.roll_step,
         args.pitch_range, args.pitch_step, args.search_detector_size, args.downsample,
         args.attenuation_scale, args.beam_axis, args.oriented_dir, args.drr_out_dir,
         args.plot_out_dir)