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

Usage:
    python scripts/batch_clean_portable_radios.py \
        --input-dir /mnt/.../tiff_output \
        --output-dir /mnt/.../tiff_output_cleaned \
        --csv portable_xr_metadata.csv \
        --diagonal-mask 0,2895,3070,2702 \
        --mask-side below \
        --rotate-deg 180

--input-dir is searched recursively for *.tif / *.tiff files. Only the
TOP-LEVEL folder name (matching a DICOM folder ID) gets renamed to the
specimen code, everything below that (filenames, subfolders) is kept
as-is. Originals are never modified.

If a top-level folder isn't found in the CSV mapping at all, it's kept
under its original opaque ID (with a warning printed), rather than
silently dropping or guessing.

If you find a specimen where the crop coordinates DON'T match (band in
a different spot, or missing entirely), stop and handle that one
separately, don't force the same crop onto every image blindly.
"""

import argparse
import csv as csv_module
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile as tiff
from scipy.ndimage import rotate as scipy_rotate


def normalize_specimen_code(patient_name: str) -> str:
    """'IFV Cedric Moisy^cep_378A' -> 'CEP_378A' (same as sort_xr_folders.py)"""
    code = patient_name.split("^")[-1].strip()
    if code.lower().startswith("cep_"):
        return "CEP_" + code[4:]
    return code.upper()


def build_folder_name_mapping(csv_path: str):
    """folder_id -> final output folder name (specimen code, with _a/_b
    suffix for specimens that have multiple raw folder IDs, same
    convention as sort_xr_folders.py)."""
    rows = list(csv_module.DictReader(open(csv_path)))
    folder_info = {}
    for r in rows:
        folder_id = r["_specimen"]
        code = normalize_specimen_code(r["PatientName"])
        acq_key = r.get("AcquisitionDate", "") + r.get("AcquisitionTime", "")
        if folder_id not in folder_info or acq_key < folder_info[folder_id][1]:
            folder_info[folder_id] = (code, acq_key)

    by_specimen = defaultdict(list)
    for folder_id, (code, acq_key) in folder_info.items():
        by_specimen[code].append((acq_key, folder_id))

    mapping = {}
    for code, entries in by_specimen.items():
        entries.sort()
        suffixes = ["a", "b", "c", "d"] if len(entries) > 1 else [""]
        for (acq_key, folder_id), suffix in zip(entries, suffixes):
            mapping[folder_id] = f"{code}{'_' + suffix if suffix else ''}"

    return mapping


def apply_diagonal_hard_crop(img: np.ndarray, x1, y1, x2, y2, side: str) -> np.ndarray:
    slope_ys = [y1, y2]
    if side == "below":
        cut_y = min(slope_ys)
        return img[:int(np.floor(cut_y))]
    else:
        cut_y = max(slope_ys)
        return img[int(np.ceil(cut_y)):]


def process_one(input_path, output_path, diagonal_mask, mask_side, rotate_deg, invert):
    img = tiff.imread(str(input_path))
    original_rows = img.shape[0]
    x1, y1, x2, y2 = diagonal_mask
    img = apply_diagonal_hard_crop(img, x1, y1, x2, y2, mask_side)

    if img.shape[0] >= original_rows:
        raise ValueError(f"crop had NO effect (shape unchanged: {original_rows} rows), "
                          f"this image's dimensions likely don't match what the diagonal "
                          f"coordinates assume, check manually rather than trusting this")

    if rotate_deg:
        dtype = img.dtype
        img = scipy_rotate(img.astype(np.float64), angle=rotate_deg, reshape=True,
                            order=1, cval=0.0)
        img = img.astype(dtype)

    if invert:
        img = (img.max() - img).astype(img.dtype)  # per-file max, MONOCHROME1 -> DRR convention

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(str(output_path), img)
    return img.shape


def main(input_dir, output_dir, diagonal_mask, mask_side, rotate_deg, csv_path, invert):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    folder_mapping = {}
    if csv_path:
        folder_mapping = build_folder_name_mapping(csv_path)
        print(f"Loaded specimen mapping for {len(folder_mapping)} folder IDs from {csv_path}")

    files = sorted(list(input_dir.rglob("*.tif")) + list(input_dir.rglob("*.tiff")))
    if not files:
        print(f"No .tif/.tiff files found under {input_dir}")
        return

    print(f"Found {len(files)} files. Processing...")
    n_ok, n_failed = 0, 0
    unmapped_folders = set()

    for f in files:
        rel_path = f.relative_to(input_dir)
        parts = list(rel_path.parts)
        top_folder = parts[0]

        if folder_mapping:
            if top_folder in folder_mapping:
                parts[0] = folder_mapping[top_folder]
            else:
                unmapped_folders.add(top_folder)
                # keep original opaque ID, don't guess

        out_rel_path = Path(*parts)
        out_path = output_dir / out_rel_path

        try:
            shape = process_one(f, out_path, diagonal_mask, mask_side, rotate_deg, invert)
            print(f"OK   {rel_path}  ->  {out_rel_path}  shape={shape}")
            n_ok += 1
        except Exception as e:
            print(f"FAIL {rel_path}: {e}")
            n_failed += 1

    print(f"\nDone: {n_ok} succeeded, {n_failed} failed, out of {len(files)} total")
    if n_failed > 0:
        print("Check the FAIL lines above, don't assume those specimens are fine.")
    if unmapped_folders:
        print(f"\nWARNING: {len(unmapped_folders)} top-level folder(s) not found in the CSV "
              f"mapping, kept under their original opaque ID: {sorted(unmapped_folders)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="folder to search recursively for tif/tiff files")
    parser.add_argument("--output-dir", required=True, help="folder to write cleaned copies into")
    parser.add_argument("--diagonal-mask", type=str, required=True, help="'x1,y1,x2,y2' points on the band boundary")
    parser.add_argument("--mask-side", choices=["above", "below"], default="below")
    parser.add_argument("--rotate-deg", type=float, default=None)
    parser.add_argument("--csv", default=None,
                         help="portable_xr_metadata.csv, used to rename top-level output "
                              "folders from opaque DICOM IDs to specimen names. Omit to keep "
                              "opaque IDs as-is.")
    parser.add_argument("--invert", action="store_true",
                         help="invert pixel values (max - pixel) after crop/rotate, for raw "
                              "MONOCHROME1 exports where background is bright/low-value and "
                              "specimen is dark/high-value, the opposite of the DRR "
                              "convention calibrate_drr.py expects. Uses each file's own max, "
                              "not a shared value across files.")
    args = parser.parse_args()

    parts = [float(v) for v in args.diagonal_mask.split(",")]
    if len(parts) != 4:
        parser.error("--diagonal-mask needs exactly 4 comma-separated values: x1,y1,x2,y2")

    main(args.input_dir, args.output_dir, parts, args.mask_side, args.rotate_deg, args.csv,
         args.invert)