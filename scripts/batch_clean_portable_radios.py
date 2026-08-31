"""
Batch version of clean_portable_radio.py's hard_crop method: applies the
same fixed diagonal-boundary crop to every TIFF in a folder, since the
band position was confirmed consistent across specimens (same device/
acquisition setup).

Output top-level folders are named by SPECIMEN, not the opaque DICOM
folder ID, using portable_xr_metadata.csv (same mapping logic as
sort_xr_folders.py: PatientName's specimen code, normalized to
CEP_xxx). Specimens with multiple raw folder IDs (confirmed duplicates:
CEP_330, CEP_378A) get _a/_b suffixes by acquisition time, same
convention as sort_xr_folders.py, so this stays consistent with your
existing RAW_data_sorted naming.

IMPORTANT:
Pixel-spacing / TIFF resolution metadata from the input TIFF is preserved
in the cleaned output TIFF.

Usage:
    python scripts/batch_clean_portable_radios.py \
        --input-dir /mnt/.../tiff_output \
        --output-dir /mnt/.../tiff_output_cleaned \
        --csv portable_xr_metadata.csv \
        --diagonal-mask 0,2895,3070,2702 \
        --mask-side below \
        --rotate-deg 180 \
        --invert
"""

import argparse
import csv as csv_module
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile as tiff
from scipy.ndimage import rotate as scipy_rotate


def normalize_specimen_code(patient_name: str) -> str:
    """
    Example:
    'IFV Cedric Moisy^cep_378A' -> 'CEP_378A'
    """
    code = patient_name.split("^")[-1].strip()

    if code.lower().startswith("cep_"):
        return "CEP_" + code[4:]

    return code.upper()


def build_folder_name_mapping(csv_path: str):
    """
    folder_id -> final output folder name.

    Specimens that have multiple raw folder IDs receive _a/_b/_c...
    suffixes ordered by acquisition time.
    """

    with open(csv_path, newline="") as f:
        rows = list(csv_module.DictReader(f))

    folder_info = {}

    for r in rows:
        folder_id = r["_specimen"]
        code = normalize_specimen_code(r["PatientName"])

        acq_key = (
            r.get("AcquisitionDate", "")
            + r.get("AcquisitionTime", "")
        )

        if (
            folder_id not in folder_info
            or acq_key < folder_info[folder_id][1]
        ):
            folder_info[folder_id] = (code, acq_key)

    by_specimen = defaultdict(list)

    for folder_id, (code, acq_key) in folder_info.items():
        by_specimen[code].append((acq_key, folder_id))

    mapping = {}

    for code, entries in by_specimen.items():
        entries.sort()

        suffixes = ["a", "b", "c", "d"] if len(entries) > 1 else [""]

        for (acq_key, folder_id), suffix in zip(entries, suffixes):
            if suffix:
                mapping[folder_id] = f"{code}_{suffix}"
            else:
                mapping[folder_id] = code

    return mapping


def apply_diagonal_hard_crop(
    img: np.ndarray,
    x1,
    y1,
    x2,
    y2,
    side: str,
) -> np.ndarray:
    """
    Hard crop using the vertical extent of the supplied diagonal boundary.

    This removes pixels completely rather than filling masked pixels.
    """

    slope_ys = [y1, y2]

    if side == "below":
        cut_y = min(slope_ys)
        return img[: int(np.floor(cut_y))]

    else:
        cut_y = max(slope_ys)
        return img[int(np.ceil(cut_y)) :]


def read_tiff_with_resolution(input_path):
    """
    Read TIFF image plus X/Y resolution metadata.

    Returns:
        img
        x_res
        y_res
        resolution_unit

    x_res and y_res are stored as pixels per resolution unit.
    """

    with tiff.TiffFile(str(input_path)) as tif:
        img = tif.asarray()

        page = tif.pages[0]

        x_res = None
        y_res = None
        resolution_unit = None

        if "XResolution" in page.tags:
            value = page.tags["XResolution"].value
            x_res = float(value[0]) / float(value[1])

        if "YResolution" in page.tags:
            value = page.tags["YResolution"].value
            y_res = float(value[0]) / float(value[1])

        if "ResolutionUnit" in page.tags:
            resolution_unit = page.tags["ResolutionUnit"].value

    return img, x_res, y_res, resolution_unit


def process_one(
    input_path,
    output_path,
    diagonal_mask,
    mask_side,
    rotate_deg,
    invert,
):
    # ------------------------------------------------------------
    # READ PIXELS + TIFF PHYSICAL RESOLUTION
    # ------------------------------------------------------------

    img, x_res, y_res, resolution_unit = read_tiff_with_resolution(
        input_path
    )

    original_rows = img.shape[0]

    x1, y1, x2, y2 = diagonal_mask

    # ------------------------------------------------------------
    # HARD CROP
    # ------------------------------------------------------------

    img = apply_diagonal_hard_crop(
        img,
        x1,
        y1,
        x2,
        y2,
        mask_side,
    )

    if img.shape[0] >= original_rows:
        raise ValueError(
            f"crop had NO effect "
            f"(shape unchanged: {original_rows} rows), "
            f"this image's dimensions likely don't match what "
            f"the diagonal coordinates assume. "
            f"Check manually rather than trusting this."
        )

    # ------------------------------------------------------------
    # ROTATION
    # ------------------------------------------------------------

    if rotate_deg:

        angle = float(rotate_deg) % 360

        # Exact 180-degree rotation.
        # No interpolation and no modification of original pixel values.
        if np.isclose(angle, 180.0):

            img = np.rot90(img, 2)

        elif np.isclose(angle, 90.0):

            img = np.rot90(img, 1)

            # X and Y axes swap after a 90-degree rotation.
            x_res, y_res = y_res, x_res

        elif np.isclose(angle, 270.0):

            img = np.rot90(img, 3)

            # X and Y axes swap after a 270-degree rotation.
            x_res, y_res = y_res, x_res

        elif not np.isclose(angle, 0.0):

            # Keep arbitrary-angle support.
            dtype = img.dtype

            img = scipy_rotate(
                img.astype(np.float64),
                angle=rotate_deg,
                reshape=True,
                order=1,
                cval=0.0,
            )

            img = img.astype(dtype)

            print(
                f"WARNING: arbitrary rotation {rotate_deg} degrees "
                f"uses interpolation."
            )

    # ------------------------------------------------------------
    # INVERT
    # ------------------------------------------------------------

    if invert:
        img = (img.max() - img).astype(img.dtype)

    # ------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        x_res is not None
        and y_res is not None
        and resolution_unit is not None
    ):
        tiff.imwrite(
            str(output_path),
            img,
            resolution=(x_res, y_res),
            resolutionunit=resolution_unit,
        )

    else:
        print(
            f"WARNING: no TIFF resolution metadata found in "
            f"{input_path.name}. Output will not contain "
            f"physical pixel spacing."
        )

        tiff.imwrite(
            str(output_path),
            img,
        )

    return img.shape, x_res, y_res, resolution_unit


def main(
    input_dir,
    output_dir,
    diagonal_mask,
    mask_side,
    rotate_deg,
    csv_path,
    invert,
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # ------------------------------------------------------------
    # BUILD DICOM-FOLDER-ID -> SPECIMEN MAPPING
    # ------------------------------------------------------------

    folder_mapping = {}

    if csv_path:
        folder_mapping = build_folder_name_mapping(csv_path)

        print(
            f"Loaded specimen mapping for "
            f"{len(folder_mapping)} folder IDs "
            f"from {csv_path}"
        )

    # ------------------------------------------------------------
    # FIND TIFF FILES
    # ------------------------------------------------------------

    files = sorted(
        list(input_dir.rglob("*.tif"))
        + list(input_dir.rglob("*.tiff"))
    )

    if not files:
        print(
            f"No .tif/.tiff files found under "
            f"{input_dir}"
        )
        return

    print(
        f"Found {len(files)} files. Processing..."
    )

    n_ok = 0
    n_failed = 0

    unmapped_folders = set()

    # ------------------------------------------------------------
    # PROCESS
    # ------------------------------------------------------------

    for f in files:

        rel_path = f.relative_to(input_dir)

        parts = list(rel_path.parts)

        top_folder = parts[0]

        if folder_mapping:

            if top_folder in folder_mapping:
                parts[0] = folder_mapping[top_folder]

            else:
                unmapped_folders.add(top_folder)

        out_rel_path = Path(*parts)

        out_path = output_dir / out_rel_path

        try:

            (
                shape,
                x_res,
                y_res,
                resolution_unit,
            ) = process_one(
                f,
                out_path,
                diagonal_mask,
                mask_side,
                rotate_deg,
                invert,
            )

            print(
                f"OK   {rel_path} "
                f"-> {out_rel_path} "
                f"shape={shape} "
                f"XResolution={x_res} "
                f"YResolution={y_res} "
                f"ResolutionUnit={resolution_unit}"
            )

            n_ok += 1

        except Exception as e:

            print(
                f"FAIL {rel_path}: {e}"
            )

            n_failed += 1

    # ------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------

    print(
        f"\nDone: {n_ok} succeeded, "
        f"{n_failed} failed, "
        f"out of {len(files)} total"
    )

    if n_failed > 0:
        print(
            "Check the FAIL lines above. "
            "Do not assume those specimens are fine."
        )

    if unmapped_folders:

        print(
            f"\nWARNING: {len(unmapped_folders)} "
            f"top-level folder(s) not found in the CSV mapping. "
            f"They were kept under their original opaque ID:"
        )

        print(
            sorted(unmapped_folders)
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        required=True,
        help="folder to search recursively for tif/tiff files",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="folder to write cleaned copies into",
    )

    parser.add_argument(
        "--diagonal-mask",
        type=str,
        required=True,
        help="'x1,y1,x2,y2' points on the band boundary",
    )

    parser.add_argument(
        "--mask-side",
        choices=["above", "below"],
        default="below",
    )

    parser.add_argument(
        "--rotate-deg",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--csv",
        default=None,
        help=(
            "portable_xr_metadata.csv, used to rename "
            "top-level output folders from opaque DICOM IDs "
            "to specimen names. Omit to keep opaque IDs as-is."
        ),
    )

    parser.add_argument(
        "--invert",
        action="store_true",
        help=(
            "invert pixel values (max - pixel) after crop/rotate "
            "for raw MONOCHROME1 exports"
        ),
    )

    args = parser.parse_args()

    parts = [
        float(v)
        for v in args.diagonal_mask.split(",")
    ]

    if len(parts) != 4:
        parser.error(
            "--diagonal-mask needs exactly 4 "
            "comma-separated values: x1,y1,x2,y2"
        )

    main(
        args.input_dir,
        args.output_dir,
        parts,
        args.mask_side,
        args.rotate_deg,
        args.csv,
        args.invert,
    )