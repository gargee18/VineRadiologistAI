"""
Combined CT table removal.

PASS 1
------
Remove explicitly selected 3D connected components.

These labels must come from inspect_3d_components.py and MUST use the
same --component-threshold.

PASS 2
------
Remove residual table/support structures slice-by-slice using:

    side position
    + large vertical extent
    + low bounding-box fill fraction

Usage:

    python scripts/remove_table_by_position.py \
        --xr /path/input.tif \
        --component-threshold 500 \
        --remove-labels 1,4376 \
        --slice-threshold 200 \
        --side-frac 0.30 \
        --side both \
        --min-height 150 \
        --max-fill-frac 0.08 \
        --min-voxels 50 \
        --dilate 3 \
        --debug-slice 766 \
        --output /path/output.tif
"""

import argparse
from pathlib import Path

import numpy as np
import tifffile as tiff

from scipy.ndimage import (
    label,
    find_objects,
    binary_dilation,
)


def main(
    xr_path,
    component_threshold,
    remove_labels,
    slice_threshold,
    side_frac,
    side,
    min_height,
    max_fill_frac,
    min_voxels,
    dilate,
    debug_slice,
    output_path,
):

    # =========================================================
    # READ METADATA
    # =========================================================

    with tiff.TiffFile(xr_path) as tf:

        page = tf.pages[0]

        original_dtype = page.asarray().dtype

        xres_tag = page.tags.get("XResolution")
        yres_tag = page.tags.get("YResolution")

        xres = (
            xres_tag.value[0] / xres_tag.value[1]
            if xres_tag
            else None
        )

        yres = (
            yres_tag.value[0] / yres_tag.value[1]
            if yres_tag
            else None
        )

        imagej_meta = tf.imagej_metadata or {}

    # =========================================================
    # LOAD VOLUME
    # =========================================================

    vol = tiff.imread(xr_path)

    if vol.ndim != 3:
        raise ValueError(
            f"Expected 3D ZYX volume, got shape {vol.shape}"
        )

    n_slices, height, width = vol.shape

    print("=" * 70)
    print("INPUT")
    print("=" * 70)

    print(f"Volume shape: {vol.shape}")
    print(f"Data type: {vol.dtype}")
    print(f"Intensity range: {vol.min()} -> {vol.max()}")

    if xres:
        print(
            f"XY resolution: {1 / xres:.7f} mm/px"
        )

    print(
        f"Z spacing: {imagej_meta.get('spacing')} "
        f"{imagej_meta.get('unit')}"
    )

    vol_cleaned = vol.copy()

    # =========================================================
    # PASS 1
    # REMOVE SPECIFIC 3D COMPONENT LABELS
    # =========================================================

    print()
    print("=" * 70)
    print("PASS 1: EXPLICIT 3D COMPONENT REMOVAL")
    print("=" * 70)

    print(
        f"3D component threshold: "
        f"{component_threshold}"
    )

    print(
        f"Requested labels: {remove_labels}"
    )

    foreground_3d = (
        vol > component_threshold
    )

    original_3d_foreground = int(
        foreground_3d.sum()
    )

    print("Running 3D connected components...")

    labeled_3d, n_components = label(
        foreground_3d
    )

    print(
        f"Found {n_components} "
        f"3D connected component(s)"
    )

    removed_3d_voxels = 0

    for lbl in remove_labels:

        mask = (
            labeled_3d == lbl
        )

        count = int(mask.sum())

        if count == 0:

            print(
                f"WARNING: label {lbl} has 0 voxels."
            )

            print(
                "Check that --component-threshold "
                "matches inspect_3d_components.py."
            )

            continue

        vol_cleaned[mask] = 0

        removed_3d_voxels += count

        print(
            f"Removed 3D label {lbl}: "
            f"{count} voxels"
        )

    pct_3d = (
        100.0
        * removed_3d_voxels
        / original_3d_foreground
        if original_3d_foreground
        else 0.0
    )

    print(
        f"\nPASS 1 removed "
        f"{removed_3d_voxels} voxels "
        f"({pct_3d:.4f}% of thresholded foreground)"
    )

    # Free large arrays before pass 2
    del labeled_3d
    del foreground_3d

    # =========================================================
    # PASS 2
    # SLICE-WISE RESIDUAL TABLE REMOVAL
    # =========================================================

    print()
    print("=" * 70)
    print("PASS 2: SLICE-WISE RESIDUAL CLEANUP")
    print("=" * 70)

    print(f"Slice threshold: {slice_threshold}")
    print(f"Side: {side}")
    print(f"Side fraction: {side_frac}")
    print(f"Minimum height: {min_height}")
    print(f"Maximum fill fraction: {max_fill_frac}")
    print(f"Dilation: {dilate}")

    # ---------------------------------------------------------
    # Side ROI
    # ---------------------------------------------------------

    side_mask = np.zeros(
        (height, width),
        dtype=bool
    )

    cutoff = int(
        side_frac * width
    )

    if side == "left":

        side_mask[:, :cutoff] = True

        print(
            f"Search region: x = 0:{cutoff}"
        )

    elif side == "right":

        side_mask[
            :,
            width - cutoff:
        ] = True

        print(
            f"Search region: "
            f"x = {width - cutoff}:{width}"
        )

    elif side == "both":

        side_mask[:, :cutoff] = True

        side_mask[
            :,
            width - cutoff:
        ] = True

        print(
            f"Search regions: "
            f"x = 0:{cutoff} "
            f"and "
            f"x = {width - cutoff}:{width}"
        )

    else:

        raise ValueError(
            "--side must be left, right, or both"
        )

    removed_slice_components = 0
    removed_slice_voxels = 0

    # =========================================================
    # PROCESS EACH Z SLICE
    # =========================================================

    for z in range(n_slices):

        foreground = (
            vol_cleaned[z] > slice_threshold
        )

        candidate = (
            foreground
            & side_mask
        )

        if not candidate.any():
            continue

        labeled_2d, n_components_2d = label(
            candidate
        )

        objects = find_objects(
            labeled_2d
        )

        # -----------------------------------------------------
        # DEBUG SLICE
        # -----------------------------------------------------

        if z == debug_slice:

            print()
            print("=" * 80)

            print(
                f"DEBUG SLICE z = {z}"
            )

            print(
                f"Candidate components: "
                f"{n_components_2d}"
            )

            print("=" * 80)

        # -----------------------------------------------------
        # COMPONENT ANALYSIS
        # -----------------------------------------------------

        for component_id, sl in enumerate(
            objects,
            start=1
        ):

            if sl is None:
                continue

            local_labels = (
                labeled_2d[sl]
            )

            component_mask = (
                local_labels
                == component_id
            )

            size = int(
                component_mask.sum()
            )

            # ---------------------------------------------
            # Bounding box
            # ---------------------------------------------

            y0 = sl[0].start
            y1 = sl[0].stop

            x0 = sl[1].start
            x1 = sl[1].stop

            bbox_h = y1 - y0
            bbox_w = x1 - x0

            bbox_area = (
                bbox_h * bbox_w
            )

            if bbox_area == 0:
                continue

            # ---------------------------------------------
            # Bounding-box occupancy
            # ---------------------------------------------

            fill_fraction = (
                size / bbox_area
            )

            # ---------------------------------------------
            # Residual table criteria
            # ---------------------------------------------

            big_enough = (
                size >= min_voxels
            )

            tall_enough = (
                bbox_h >= min_height
            )

            sparse_enough = (
                fill_fraction
                <= max_fill_frac
            )

            is_table = (
                big_enough
                and tall_enough
                and sparse_enough
            )

            # ---------------------------------------------
            # DEBUG OUTPUT
            # ---------------------------------------------

            if z == debug_slice:

                print(
                    f"component "
                    f"{component_id:3d} | "
                    f"pixels={size:6d} | "
                    f"bbox={bbox_h:3d}x{bbox_w:3d} | "
                    f"x={x0:3d}:{x1:3d} | "
                    f"y={y0:3d}:{y1:3d} | "
                    f"fill={fill_fraction:7.4f} | "
                    f"{'REMOVE' if is_table else 'KEEP'}"
                )

            # ---------------------------------------------
            # REMOVE
            # ---------------------------------------------

            if is_table:

                remove_mask = np.zeros(
                    (height, width),
                    dtype=bool
                )

                remove_mask[sl] = (
                    component_mask
                )

                if dilate > 0:

                    remove_mask = binary_dilation(
                        remove_mask,
                        iterations=dilate
                    )

                # Do not allow dilation into center
                remove_mask &= side_mask

                actual_remove = (
                    remove_mask
                    & (vol_cleaned[z] != 0)
                )

                removed_count = int(
                    actual_remove.sum()
                )

                vol_cleaned[z][remove_mask] = 0

                removed_slice_components += 1
                removed_slice_voxels += removed_count

        if z % 200 == 0:

            print(
                f"processed slice "
                f"{z}/{n_slices}"
            )

    # =========================================================
    # FINAL STATISTICS
    # =========================================================

    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(
        f"PASS 1, explicit 3D components: "
        f"{removed_3d_voxels} voxels"
    )

    print(
        f"PASS 2, slice residual cleanup: "
        f"{removed_slice_voxels} voxels"
    )

    print(
        f"PASS 2 components removed: "
        f"{removed_slice_components}"
    )

    total_removed = (
        removed_3d_voxels
        + removed_slice_voxels
    )

    print(
        f"TOTAL removed: "
        f"{total_removed} voxels"
    )

    # =========================================================
    # SAVE
    # =========================================================

    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    metadata = {
        "axes": "ZYX"
    }

    if imagej_meta.get("spacing") is not None:

        metadata["spacing"] = (
            imagej_meta["spacing"]
        )

    metadata["unit"] = (
        imagej_meta.get(
            "unit",
            "mm"
        )
    )

    if xres is not None and yres is not None:

        tiff.imwrite(
            output_path,
            vol_cleaned.astype(original_dtype),
            imagej=True,
            resolution=(xres, yres),
            metadata=metadata,
        )

    else:

        tiff.imwrite(
            output_path,
            vol_cleaned.astype(original_dtype),
            imagej=True,
            metadata=metadata,
        )

    print()
    print(
        f"Saved cleaned volume to:\n"
        f"{output_path}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Input / output

    parser.add_argument(
        "--xr",
        required=True,
        help="Path to original CT TIFF"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output cleaned TIFF"
    )

    # PASS 1

    parser.add_argument(
        "--component-threshold",
        type=float,
        required=True,
        help=(
            "Threshold used for 3D connected components. "
            "MUST match inspect_3d_components.py"
        )
    )

    parser.add_argument(
        "--remove-labels",
        required=True,
        help=(
            "Comma-separated 3D label IDs from "
            "inspect_3d_components.py"
        )
    )

    # PASS 2

    parser.add_argument(
        "--slice-threshold",
        type=float,
        default=200
    )

    parser.add_argument(
        "--side-frac",
        type=float,
        default=0.30
    )

    parser.add_argument(
        "--side",
        choices=[
            "left",
            "right",
            "both"
        ],
        default="both"
    )

    parser.add_argument(
        "--min-height",
        type=int,
        default=150
    )

    parser.add_argument(
        "--max-fill-frac",
        type=float,
        default=0.08
    )

    parser.add_argument(
        "--min-voxels",
        type=int,
        default=50
    )

    parser.add_argument(
        "--dilate",
        type=int,
        default=3
    )

    parser.add_argument(
        "--debug-slice",
        type=int,
        default=766
    )

    args = parser.parse_args()

    labels_to_remove = [
        int(v.strip())
        for v in args.remove_labels.split(",")
    ]

    main(
        args.xr,
        args.component_threshold,
        labels_to_remove,
        args.slice_threshold,
        args.side_frac,
        args.side,
        args.min_height,
        args.max_fill_frac,
        args.min_voxels,
        args.dilate,
        args.debug_slice,
        args.output,
    )