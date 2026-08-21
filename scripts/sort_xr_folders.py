"""
Rename/organize RAW_data_non_tri/<opaque_id>/ folders into specimen-named
folders, using the PatientName field (e.g. "IFV Cedric Moisy^cep_378A")
from portable_xr_metadata.csv to map opaque folder IDs to specimen codes.

Copies by default (safe, doesn't touch originals). Use --move to actually
move instead once you've checked the dry run looks right.

Two things this script surfaces and asks you about rather than guessing:
  1. Case inconsistency in specimen codes (cep_378A vs CEP_378A) -> normalized
     to uppercase CEP_xxx for the output folder name.
  2. Specimens with more than one raw folder ID (e.g. cep_330 has two,
     one appears to be a same-day re-export with different pixel encoding)
     -> both are kept, suffixed _a, _b in acquisition order, NOT merged,
     since I can't tell you which one is the "real" usable acquisition.

Usage:
    python sort_xr_folders.py \
        --csv portable_xr_metadata.csv \
        --raw-root /mnt/.../PortableXR_2026/RAW_data_non_tri \
        --out-root /mnt/.../PortableXR_2026/RAW_data_sorted \
        --dry-run
"""

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path


def normalize_specimen_code(patient_name: str) -> str:
    """'IFV Cedric Moisy^cep_378A' -> 'CEP_378A'"""
    code = patient_name.split("^")[-1].strip()
    if code.lower().startswith("cep_"):
        return "CEP_" + code[4:]
    return code.upper()


def build_mapping(csv_path: str):
    """folder_id -> (specimen_code, earliest AcquisitionDate+Time)"""
    rows = list(csv.DictReader(open(csv_path)))
    folder_info = {}
    for r in rows:
        folder_id = r["_specimen"]
        code = normalize_specimen_code(r["PatientName"])
        acq_key = r.get("AcquisitionDate", "") + r.get("AcquisitionTime", "")
        if folder_id not in folder_info or acq_key < folder_info[folder_id][1]:
            folder_info[folder_id] = (code, acq_key)
    return folder_info


def main(csv_path, raw_root, out_root, move, dry_run):
    raw_root = Path(raw_root)
    out_root = Path(out_root)
    folder_info = build_mapping(csv_path)

    # group folder ids by specimen code, sorted by acquisition time, so
    # duplicates get deterministic _a/_b/_c suffixes
    by_specimen = defaultdict(list)
    for folder_id, (code, acq_key) in folder_info.items():
        by_specimen[code].append((acq_key, folder_id))

    n_ok, n_missing, n_dupe = 0, 0, 0
    for code, entries in sorted(by_specimen.items()):
        entries.sort()  # by acquisition time
        suffixes = ["a", "b", "c", "d"] if len(entries) > 1 else [""]
        if len(entries) > 1:
            n_dupe += 1
            print(f"NOTE: {code} has {len(entries)} raw folders, "
                  f"suffixing _a/_b in acquisition-time order, verify which is correct")

        for (acq_key, folder_id), suffix in zip(entries, suffixes):
            src = raw_root / folder_id
            dst_name = f"{code}{'_' + suffix if suffix else ''}"
            dst = out_root / dst_name

            if not src.exists():
                print(f"MISSING source folder: {src}")
                n_missing += 1
                continue

            print(f"{folder_id}  ->  {dst_name}   (acq={acq_key})")
            if not dry_run:
                out_root.mkdir(parents=True, exist_ok=True)
                if move:
                    shutil.move(str(src), str(dst))
                else:
                    shutil.copytree(str(src), str(dst), dirs_exist_ok=True)
            n_ok += 1

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Done: {n_ok} folders processed, "
          f"{n_missing} missing, {n_dupe} specimens with multiple raw folders")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--move", action="store_true", help="move instead of copy")
    parser.add_argument("--dry-run", action="store_true", help="print planned actions only")
    args = parser.parse_args()
    main(args.csv, args.raw_root, args.out_root, args.move, args.dry_run)