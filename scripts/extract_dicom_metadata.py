"""
Extract ALL available metadata fields from real portable X-ray DICOM files.

Writes:
  - a wide CSV with every tag seen across all files as a column
  - a summary printed to terminal: for each tag, how many files have it,
    and whether its value is constant or varies across files (varying
    tags are usually the more interesting ones to look at)

Usage:
    python scripts/extract_dicom_metadata_full.py \
        --root ~/code_python/VineRadiologistAI/dataset/radiograph \
        --out ~/code_python/VineRadiologistAI/dataset/radiograph/dicom_metadata_full.csv
"""

import argparse
import csv
from pathlib import Path
from collections import defaultdict

import pydicom
from pydicom.errors import InvalidDicomError
from pydicom.multival import MultiValue
from pydicom.valuerep import PersonName


def _stringify(value):
    """Make any DICOM value type safely writable to CSV."""
    if isinstance(value, (MultiValue, list, tuple)):
        return "[" + ", ".join(_stringify(v) for v in value) + "]"
    if isinstance(value, PersonName):
        return str(value)
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    return str(value)


def extract_all_fields(path):
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    except Exception as e:
        return None, str(e)

    row = {}
    for elem in ds:
        if elem.tag == 0x7FE00010:  # PixelData, skip even if present
            continue
        name = elem.keyword if elem.keyword else str(elem.tag)
        try:
            row[name] = _stringify(elem.value)
        except Exception:
            row[name] = "<unreadable>"
    return row, None


def main(root, out_csv):
    root = Path(root)
    all_rows = []
    errors = []

    for specimen_dir in sorted(root.iterdir()):
        if not specimen_dir.is_dir():
            continue
        dicomobj_dir = specimen_dir / "DICOMOBJ"
        if not dicomobj_dir.exists():
            continue
        for f in sorted(dicomobj_dir.iterdir()):
            row, err = extract_all_fields(f)
            if row is not None:
                row["_specimen"] = specimen_dir.name
                row["_file"] = str(f)
                all_rows.append(row)
            else:
                errors.append((str(f), err))

    if not all_rows:
        print("No DICOM files successfully read. Errors:")
        for f, e in errors:
            print(f"  {f}: {e}")
        return

    # union of every field name seen across all files
    all_fields = set()
    for row in all_rows:
        all_fields.update(row.keys())
    fieldnames = ["_specimen", "_file"] + sorted(all_fields - {"_specimen", "_file"})

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="not present")
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"Wrote {len(all_rows)} rows x {len(fieldnames)} fields to {out_csv}")
    if errors:
        print(f"{len(errors)} files failed to read:")
        for f, e in errors:
            print(f"  {f}: {e}")

    # summary: presence count + constant vs varying, per field
    print("\nField summary (name: n_present/n_total, constant or varying):")
    field_values = defaultdict(set)
    field_present = defaultdict(int)
    for row in all_rows:
        for field in fieldnames:
            if field in ("_specimen", "_file"):
                continue
            val = row.get(field, "not present")
            if val != "not present":
                field_present[field] += 1
                field_values[field].add(val)

    for field in sorted(fieldnames):
        if field in ("_specimen", "_file"):
            continue
        n = field_present[field]
        if n == 0:
            continue
        status = "constant" if len(field_values[field]) == 1 else f"varies ({len(field_values[field])} distinct values)"
        print(f"  {field}: {n}/{len(all_rows)} present, {status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="folder containing per-specimen DICOMOBJ subfolders")
    parser.add_argument("--out", required=True, help="output CSV path")
    args = parser.parse_args()
    main(args.root, args.out)