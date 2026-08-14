"""
Export DICOM pixel data straight to TIFF, unmodified.

No windowing, no photometric inversion, no RescaleSlope/Intercept applied,
exactly the raw sensor counts as stored in the file. This is deliberately
DIFFERENT from calibrate_drr.py's load_image(), which inverts MONOCHROME1
images for the Beer-Lambert comparison, that inversion is a modeling
choice for calibration, not "the real data."

Use this when you want to inspect actual pixel values (e.g. in ImageJ,
which doesn't auto-window like Weasis does) instead of trusting a
viewer's display stretch.

Usage (single file):
    python scripts/dicom_to_raw_tiff.py \
        --input path/to/file.dcm \
        --output path/to/out.tif

Usage (whole folder, e.g. one specimen's several views):
    python scripts/dicom_to_raw_tiff.py \
        --input /mnt/.../RAW_data_sorted/CEP_1191 \
        --output-dir /mnt/.../raw_tiff_check/CEP_1191
"""

import argparse
from pathlib import Path

import numpy as np
import pydicom
import tifffile as tiff


def export_one(dcm_path: Path, out_path: Path):
    ds = pydicom.dcmread(str(dcm_path), force=True)
    pixels = ds.pixel_array  # raw, as-stored, no modifications

    print(f"{dcm_path.name}: dtype={pixels.dtype} shape={pixels.shape} "
          f"min={pixels.min()} max={pixels.max()} "
          f"PhotometricInterpretation={getattr(ds, 'PhotometricInterpretation', '?')} "
          f"SeriesDescription={getattr(ds, 'SeriesDescription', '?')}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(str(out_path), pixels)  # writes at native dtype (uint16 here), no rescale
    print(f"  -> {out_path}")


def main(input_path, output, output_dir):
    input_path = Path(input_path)

    if input_path.is_dir():
        if output_dir is None:
            raise ValueError("--output-dir is required when --input is a folder")
        output_dir = Path(output_dir)
        dcm_files = sorted(input_path.glob("*.dcm"))
        if not dcm_files:
            print(f"No .dcm files found in {input_path}")
            return
        for f in dcm_files:
            out_path = output_dir / f"{f.stem}.tif"
            export_one(f, out_path)
    else:
        if output is None:
            raise ValueError("--output is required when --input is a single file")
        export_one(input_path, Path(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                         help="a single .dcm file, or a folder of .dcm files")
    parser.add_argument("--output", default=None,
                         help="output .tif path (required if --input is a single file)")
    parser.add_argument("--output-dir", default=None,
                         help="output folder (required if --input is a folder)")
    args = parser.parse_args()
    main(args.input, args.output, args.output_dir)