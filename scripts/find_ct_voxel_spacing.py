"""
Search an entire specimen dataset tree for any DICOM files (not just the
known RADIO/DICOMOBJ radiograph files) and report which ones carry
PixelSpacing / SliceThickness, the real CT voxel spacing needed for
cone-beam projection geometry.

This does NOT assume where CT source DICOMs live, it just walks every file
under --root and tries to read it as DICOM, since registered.tif stacks
may not have kept the original DICOM metadata.

Usage:
    python scripts/find_ct_voxel_spacing.py \
        --root /home/phukon/code_python/Dataset_Vitimage2019
"""

import argparse
from pathlib import Path

import pydicom
from pydicom.errors import InvalidDicomError


def try_read_dicom(path):
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        return ds
    except Exception:
        return None


def main(root):
    root = Path(root)
    found_any_dicom = False
    found_spacing = []

    ct_dirs = [p for p in root.glob("*/CT") if p.is_dir()]
    if not ct_dirs:
        print(f"No CT/ subfolders found directly under specimen folders in {root}.")
        return

    for ct_dir in sorted(ct_dirs):
        for path in ct_dir.rglob("*"):
            if not path.is_file():
                continue
            ds = try_read_dicom(path)
            if ds is None:
                continue
            if not (hasattr(ds, "Modality") or hasattr(ds, "SOPClassUID")):
                continue

            found_any_dicom = True
            pixel_spacing = getattr(ds, "PixelSpacing", None)
            slice_thickness = getattr(ds, "SliceThickness", None)
            modality = getattr(ds, "Modality", "?")

            if pixel_spacing is not None or slice_thickness is not None:
                found_spacing.append({
                    "path": str(path),
                    "modality": modality,
                    "pixel_spacing": list(pixel_spacing) if pixel_spacing else None,
                    "slice_thickness": float(slice_thickness) if slice_thickness else None,
                })

    if not found_any_dicom:
        print(f"No DICOM files found anywhere under {root}.")
        print("Your CT volumes are likely stored only as .tif with no "
              "recoverable DICOM metadata, voxel spacing would need to "
              "come from another source (lab notes, scanner logs, or "
              "whoever ran the original CT acquisition).")
        return

    if not found_spacing:
        print(f"Found DICOM files under {root}, but none carry "
              f"PixelSpacing or SliceThickness.")
        return

    print(f"Found {len(found_spacing)} DICOM files with spacing info:\n")
    for entry in found_spacing:
        print(f"  {entry['path']}")
        print(f"    modality={entry['modality']}  "
              f"pixel_spacing={entry['pixel_spacing']}  "
              f"slice_thickness={entry['slice_thickness']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    main(args.root)