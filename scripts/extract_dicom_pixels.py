"""
Extract raw pixel data from real X-ray DICOM files and save as TIFF,
preserving original bit depth (no JPEG compression/quantization loss).

Usage:
    python scripts/extract_dicom_pixels.py \
        --root ~/code_python/VineRadiologistAI/dataset/radiograph \
        --out ~/code_python/VineRadiologistAI/dataset/radiograph_tif
"""

import argparse
from pathlib import Path

import numpy as np
import pydicom
import tifffile as tiff


def main(root, out_root):
    root = Path(root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    n_ok, n_fail = 0, 0
    for specimen_dir in sorted(root.iterdir()):
        if not specimen_dir.is_dir():
            continue
        dicomobj_dir = specimen_dir / "DICOMOBJ"
        if not dicomobj_dir.exists():
            continue

        specimen_out = out_root / specimen_dir.name
        specimen_out.mkdir(parents=True, exist_ok=True)

        for f in sorted(dicomobj_dir.iterdir()):
            try:
                ds = pydicom.dcmread(str(f), force=True)
                pixels = ds.pixel_array
            except Exception as e:
                print(f"  FAILED: {f}: {e}")
                n_fail += 1
                continue

            out_path = specimen_out / f"{f.name}.tif"
            tiff.imwrite(out_path, pixels)
            print(f"{f} -> {out_path}  shape={pixels.shape} dtype={pixels.dtype} "
                  f"min={pixels.min()} max={pixels.max()}")
            n_ok += 1

    print(f"\nDone: {n_ok} extracted, {n_fail} failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="folder containing per-specimen DICOMOBJ subfolders")
    parser.add_argument("--out", required=True, help="output folder for extracted TIFFs")
    args = parser.parse_args()
    main(args.root, args.out)