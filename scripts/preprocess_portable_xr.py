#!/usr/bin/env python3
"""
Batch preprocessing for portable grapevine radiographs.

Pipeline for every raw DICOM:
    1. Read the original DICOM pixel array.
    2. If PhotometricInterpretation == MONOCHROME1, invert it using
           inverted = pixels.max() - pixels
       This matches the inversion convention used by calibrate_drr.py.
    3. Remove the detector/panel artifact with a hard rectangular crop
       derived from a diagonal boundary line.
    4. Flip vertically (top <-> bottom). Left/right is preserved.
    5. Save as TIFF while keeping detector pixel-spacing metadata.
    6. Organize outputs into specimen-named folders using PatientName
       directly from the DICOM metadata.

No RescaleSlope/RescaleIntercept and no windowing are applied.

If one specimen has several independent raw acquisition folders, output
folders are suffixed _a, _b, ... in acquisition-time order.

Example:
    python scripts/preprocess_portable_xr.py \
        --raw-root /mnt/.../code_python/dcm \
        --out-root /mnt/.../dataset/radiograph_portable_2026_tif_vflip \
        --diagonal-crop 0,2895,3070,2702 \
        --crop-side below

Test only CEP_1191 first:
    python scripts/preprocess_portable_xr.py \
        --raw-root /mnt/.../code_python/dcm \
        --out-root /mnt/.../dataset/radiograph_portable_2026_tif_vflip \
        --specimen CEP_1191 \
        --diagonal-crop 0,2895,3070,2702 \
        --crop-side below
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pydicom
import tifffile as tiff


def normalize_specimen_code(patient_name: str) -> str:
    """
    Examples:
        'IFV Cedric Moisy^cep_378A' -> 'CEP_378A'
        'CEP_1191'                  -> 'CEP_1191'
    """
    code = str(patient_name).split("^")[-1].strip()

    if code.lower().startswith("cep_"):
        return "CEP_" + code[4:]

    return code.upper()


def acquisition_key(ds) -> str:
    """Sortable acquisition date/time string."""
    date = str(getattr(ds, "AcquisitionDate", "") or "")
    time = str(getattr(ds, "AcquisitionTime", "") or "")

    if not date:
        date = str(getattr(ds, "StudyDate", "") or "")
    if not time:
        time = str(getattr(ds, "StudyTime", "") or "")

    return date + time


def detector_pixel_spacing_mm(ds) -> float | None:
    """
    Prefer detector-plane spacing for projection radiographs.
    """
    for tag_name in (
        "ImagerPixelSpacing",
        "DetectorElementPhysicalSize",
        "PixelSpacing",
    ):
        spacing = getattr(ds, tag_name, None)

        if spacing is None:
            continue

        try:
            return float(spacing[0])
        except Exception:
            pass

    return None


def invert_monochrome1(pixels: np.ndarray, ds) -> np.ndarray:
    """
    Convert MONOCHROME1 to the same bright-object convention used later
    by calibrate_drr.py.

    MONOCHROME2 is left unchanged.
    """
    photo = str(
        getattr(ds, "PhotometricInterpretation", "")
    ).upper()

    if photo != "MONOCHROME1":
        print(f"      PhotometricInterpretation={photo or '?'} -> no inversion")
        return pixels

    dtype = pixels.dtype

    # Work in float/int-safe arithmetic, then cast back.
    x = pixels.astype(np.float64)
    x = x.max() - x

    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        x = np.clip(x, info.min, info.max)

    out = x.astype(dtype)

    print(
        f"      MONOCHROME1 inverted: "
        f"min={out.min()} max={out.max()}"
    )

    return out


def hard_crop_from_diagonal(
    img: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    side: str,
) -> np.ndarray:
    """
    Remove the entire artifact side of a diagonal boundary without
    synthesizing pixels.

    For side='below', keep only rows above the shallowest point of the
    diagonal line. For side='above', keep only rows below the shallowest
    valid point on the opposite side.

    x1/x2 are retained in the interface because the boundary is defined
    by two image points, even though a guaranteed rectangular hard crop
    only needs the two y coordinates.
    """
    del x1, x2

    rows = img.shape[0]

    if side == "below":
        cut_y = int(np.floor(min(y1, y2)))

        if not 0 < cut_y <= rows:
            raise ValueError(
                f"Invalid crop boundary: cut_y={cut_y}, image rows={rows}"
            )

        out = img[:cut_y, :]

        print(
            f"      hard crop below diagonal -> "
            f"kept rows 0:{cut_y}, shape={out.shape}"
        )

        return out

    if side == "above":
        cut_y = int(np.ceil(max(y1, y2)))

        if not 0 <= cut_y < rows:
            raise ValueError(
                f"Invalid crop boundary: cut_y={cut_y}, image rows={rows}"
            )

        out = img[cut_y:, :]

        print(
            f"      hard crop above diagonal -> "
            f"kept rows {cut_y}:{rows}, shape={out.shape}"
        )

        return out

    raise ValueError("crop side must be 'above' or 'below'")


def save_tiff(
    out_path: Path,
    img: np.ndarray,
    pixel_spacing_mm: float | None,
) -> None:
    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    kwargs = {
        "photometric": "minisblack",
    }

    if pixel_spacing_mm is not None and pixel_spacing_mm > 0:
        pixels_per_cm = 10.0 / float(pixel_spacing_mm)

        kwargs.update(
            resolution=(
                pixels_per_cm,
                pixels_per_cm,
            ),
            resolutionunit="CENTIMETER",
        )

    tiff.imwrite(
        str(out_path),
        img,
        **kwargs,
    )


def read_group_info(files: list[Path]):
    """
    Determine specimen code and earliest acquisition time for one raw
    acquisition folder.
    """
    codes = []
    times = []

    for path in files:
        ds = pydicom.dcmread(
            str(path),
            force=True,
            stop_before_pixels=True,
        )

        patient_name = getattr(ds, "PatientName", None)

        if patient_name is None:
            raise ValueError(
                f"No PatientName in {path}"
            )

        codes.append(
            normalize_specimen_code(
                str(patient_name)
            )
        )

        times.append(
            acquisition_key(ds)
        )

    unique_codes = sorted(set(codes))

    if len(unique_codes) != 1:
        raise ValueError(
            "One raw acquisition folder contains multiple specimen codes: "
            f"{unique_codes}"
        )

    valid_times = [
        value
        for value in times
        if value
    ]

    earliest = min(valid_times) if valid_times else ""

    return unique_codes[0], earliest


def discover_raw_groups(raw_root: Path):
    """
    Group DICOMs by their first directory below raw_root.

    If DICOMs live directly in raw_root, each file is treated as its own
    acquisition group.
    """
    dcm_files = sorted(
        path
        for path in raw_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() == ".dcm"
    )

    if not dcm_files:
        raise FileNotFoundError(
            f"No .dcm files found under {raw_root}"
        )

    groups = defaultdict(list)

    for path in dcm_files:
        rel = path.relative_to(raw_root)

        if len(rel.parts) == 1:
            group_id = path.stem
        else:
            group_id = rel.parts[0]

        groups[group_id].append(path)

    return groups


def build_output_group_names(groups):
    """
    Reproduce the specimen naming behavior of the previous sorting script:
    duplicate acquisitions become CEP_xxx_a, CEP_xxx_b, ...
    """
    by_specimen = defaultdict(list)

    for group_id, files in groups.items():
        code, acq = read_group_info(files)

        by_specimen[code].append(
            (
                acq,
                group_id,
                files,
            )
        )

    output_groups = []

    for code, entries in sorted(by_specimen.items()):
        entries.sort(
            key=lambda item: (
                item[0],
                item[1],
            )
        )

        multiple = len(entries) > 1

        for index, (acq, group_id, files) in enumerate(entries):
            if multiple:
                suffix = chr(ord("a") + index)
                output_name = f"{code}_{suffix}"
            else:
                output_name = code

            output_groups.append(
                {
                    "specimen": code,
                    "group_id": group_id,
                    "acquisition": acq,
                    "output_name": output_name,
                    "files": files,
                }
            )

    return output_groups


def preprocess_one(
    dcm_path: Path,
    out_path: Path,
    diagonal_crop: tuple[float, float, float, float] | None,
    crop_side: str,
    overwrite: bool,
):
    if out_path.exists() and not overwrite:
        print(f"    SKIP exists: {out_path}")
        return

    ds = pydicom.dcmread(
        str(dcm_path),
        force=True,
    )

    pixels = ds.pixel_array

    if pixels.ndim != 2:
        raise ValueError(
            f"Expected a 2D radiograph, got shape={pixels.shape} in {dcm_path}"
        )

    print(
        f"    {dcm_path.name}: "
        f"shape={pixels.shape} dtype={pixels.dtype} "
        f"min={pixels.min()} max={pixels.max()} "
        f"photo={getattr(ds, 'PhotometricInterpretation', '?')} "
        f"series={getattr(ds, 'SeriesDescription', '?')}"
    )

    # 1. Invert MONOCHROME1.
    img = invert_monochrome1(
        pixels,
        ds,
    )

    # 2. Remove the white/panel artifact.
    if diagonal_crop is not None:
        x1, y1, x2, y2 = diagonal_crop

        img = hard_crop_from_diagonal(
            img,
            x1,
            y1,
            x2,
            y2,
            crop_side,
        )

    # 3. Vertical flip only.
    img = np.flipud(img).copy()

    print(
        f"      vertical flip -> shape={img.shape}"
    )

    # 4. Save with detector-plane pixel spacing when available.
    px_mm = detector_pixel_spacing_mm(ds)

    save_tiff(
        out_path,
        img,
        px_mm,
    )

    print(
        f"      saved -> {out_path} "
        f"(pixel_spacing_mm={px_mm})"
    )


def parse_diagonal_crop(value: str | None):
    if value is None:
        return None

    parts = [
        float(v)
        for v in value.split(",")
    ]

    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "--diagonal-crop needs x1,y1,x2,y2"
        )

    return tuple(parts)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Raw DICOM -> MONOCHROME1 inversion -> artifact hard crop -> "
            "vertical flip -> specimen-organized TIFF."
        )
    )

    parser.add_argument(
        "--raw-root",
        required=True,
        help="Root folder containing the raw DICOM acquisition folders.",
    )

    parser.add_argument(
        "--out-root",
        required=True,
        help="Root folder for processed specimen-organized TIFFs.",
    )

    parser.add_argument(
        "--diagonal-crop",
        default="0,2895,3070,2702",
        help=(
            "Two points on the artifact boundary as x1,y1,x2,y2. "
            "Default: 0,2895,3070,2702. "
            "Use --no-crop to disable."
        ),
    )

    parser.add_argument(
        "--crop-side",
        choices=[
            "above",
            "below",
        ],
        default="below",
    )

    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Disable artifact crop.",
    )

    parser.add_argument(
        "--specimen",
        default=None,
        help=(
            "Optional specimen filter, e.g. CEP_1191. "
            "Useful for testing one specimen first."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show specimen/folder mapping without reading pixel data or writing TIFFs.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite TIFFs that already exist.",
    )

    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)

    if not raw_root.exists():
        raise FileNotFoundError(
            f"Raw DICOM root does not exist: {raw_root}"
        )

    groups = discover_raw_groups(
        raw_root
    )

    output_groups = build_output_group_names(
        groups
    )

    specimen_filter = None

    if args.specimen is not None:
        specimen_filter = normalize_specimen_code(
            args.specimen
        )

    selected = [
        group
        for group in output_groups
        if specimen_filter is None
        or group["specimen"] == specimen_filter
    ]

    if not selected:
        raise ValueError(
            f"No acquisition found for specimen {args.specimen}"
        )

    diagonal_crop = (
        None
        if args.no_crop
        else parse_diagonal_crop(
            args.diagonal_crop
        )
    )

    print()
    print("==============================================")
    print("PORTABLE XR PREPROCESSING")
    print("==============================================")
    print(f"Raw root       : {raw_root}")
    print(f"Output root    : {out_root}")
    print("MONOCHROME1    : invert")
    print(f"Artifact crop  : {diagonal_crop} side={args.crop_side}")
    print("Orientation    : vertical flip only")
    print(f"Groups selected: {len(selected)}")
    print()

    total = 0

    for group in selected:
        print(
            f"{group['group_id']} -> {group['output_name']} "
            f"(acq={group['acquisition'] or '?'}, "
            f"{len(group['files'])} DICOMs)"
        )

        if args.dry_run:
            continue

        specimen_out = (
            out_root
            / group["output_name"]
        )

        for dcm_path in group["files"]:
            out_path = (
                specimen_out
                / f"{dcm_path.stem}.tif"
            )

            preprocess_one(
                dcm_path,
                out_path,
                diagonal_crop=diagonal_crop,
                crop_side=args.crop_side,
                overwrite=args.overwrite,
            )

            total += 1

    if args.dry_run:
        print()
        print("[DRY RUN] Nothing was written.")
    else:
        print()
        print(
            f"Done. Saved {total} processed TIFF(s) under:"
        )
        print(out_root)


if __name__ == "__main__":
    main()
