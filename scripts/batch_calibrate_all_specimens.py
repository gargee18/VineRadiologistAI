"""
Batch-run DRR generation (fixed attenuation_scale=0.02) and distribution
comparison for every specimen with a known voxel spacing, using the
portable_xr_metadata.csv to identify each specimen's "Face" view file.

SAFETY: if a specimen has 0 or 2+ "Face"-labeled files (like CEP_322,
CEP_988B, CEP_378A did), it's SKIPPED and reported, not guessed at. Those
need the same manual visual confirmation you already did for the
specimens that had this ambiguity.

No vertical offset is applied here (offset_v_mm=0), since we've only
confirmed correct offsets for CEP_1191 (0), CEP_322 (-50), and CEP_988B
(-100) so far, not the rest. Expect some specimens to show the same
framing issues (too much pot, canopy cut off) that those needed manual
tuning to fix, this batch run establishes a baseline, not final results.

Usage:
    python scripts/batch_calibrate_all_specimens.py \
        --xr-dir /mnt/.../CEP/RegistrationHighRes \
        --pxr-dir dataset/radiograph_portable_2026_tif \
        --csv /mnt/.../portable_xr_metadata.csv \
        --results-dir results/PXR_DRR_2026 \
        --detector-size 512
"""

import argparse
import csv as csv_module
import subprocess
import sys
from pathlib import Path

# specimen -> (spacing_xy_mm, spacing_z_mm), from the confirmed table.
# CEP_1191 corrected to 0.5820315 (was mistakenly using CEP_378A's value
# earlier in the session), CEP_322/CEP_988B/CEP_378A already independently
# confirmed and match this table.
VOXEL_SPACING_TABLE = {
    "CEP_313B":  (0.578125,  0.4),
    "CEP_318":   (0.562500,  0.4),
    "CEP_322":   (0.851563,  0.4),
    "CEP_323":   (0.585937,  0.6),
    "CEP_330":   (0.697265,  0.4),
    "CEP_335":   (0.820312,  0.4),
    "CEP_368B":  (0.628906,  0.4),
    "CEP_378A":  (0.617187,  0.4),
    "CEP_378B":  (0.681640,  0.4),
    "CEP_380A":  (0.828125,  0.4),
    "CEP_764B":  (0.751953,  0.4),
    "CEP_988B":  (0.623047,  0.4),
    "CEP_1181":  (0.585937,  0.7),
    "CEP_1186A": (0.605469,  0.4),
    "CEP_1189":  (0.683593,  0.4),
    "CEP_1191":  (0.582031,  0.4),
    "CEP_1193":  (0.730469,  0.4),
    "CEP_1195":  (0.867187,  0.4),
    "CEP_1266A": (0.615234,  0.4),
    "CEP_2184A": (0.630859,  0.4),
}

# confirmed vertical offsets from manual tuning, everything else defaults to 0
KNOWN_OFFSETS = {
    "CEP_1191": 0.0,
    "CEP_322": -50.0,
    "CEP_988B": -100.0,
}

ATTENUATION_SCALE = 0.02


def normalize_specimen_code(patient_name: str) -> str:
    code = patient_name.split("^")[-1].strip()
    if code.lower().startswith("cep_"):
        return "CEP_" + code[4:]
    return code.upper()


def find_face_candidates(csv_path: str, specimen: str, pxr_dir: Path):
    """Return list of .tif filenames (stems matching real files on disk)
    that are labeled 'Face' for this specimen in the metadata CSV."""
    rows = list(csv_module.DictReader(open(csv_path)))
    candidates = []
    for r in rows:
        if normalize_specimen_code(r["PatientName"]) != specimen:
            continue
        if "face" not in r.get("SeriesDescription", "").lower():
            continue
        stem = Path(r["_file"]).stem
        tif_path = pxr_dir / specimen / f"{stem}.tif"
        if tif_path.exists():
            candidates.append(tif_path)
    return sorted(set(candidates))


def run(cmd):
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FAILED:\n{result.stderr[-2000:]}")
        return False
    print(result.stdout[-1500:])
    return True


def main(xr_dir, pxr_dir, csv_path, results_dir, detector_size):
    xr_dir = Path(xr_dir)
    pxr_dir = Path(pxr_dir)
    results_dir = Path(results_dir)

    skipped, succeeded, failed = [], [], []

    for specimen, (spacing_xy, spacing_z) in VOXEL_SPACING_TABLE.items():
        print(f"\n=== {specimen} ===")

        xr_path = xr_dir / f"{specimen}_2026_XR.tif"
        if not xr_path.exists():
            print(f"  SKIP: no CT volume found at {xr_path}")
            skipped.append((specimen, "no CT volume"))
            continue

        candidates = find_face_candidates(csv_path, specimen, pxr_dir)
        if len(candidates) != 1:
            print(f"  SKIP: found {len(candidates)} Face-view candidate(s), "
                  f"need exactly 1, confirm manually: {candidates}")
            skipped.append((specimen, f"{len(candidates)} Face candidates"))
            continue

        pxr_path = candidates[0]
        offset_v = KNOWN_OFFSETS.get(specimen, 0.0)

        specimen_dir = results_dir / specimen
        drr_out = specimen_dir / "DRR" / "test" / "calibrated.tif"
        dist_out = specimen_dir / "Distribution" / "dist_calibrated.png"

        ok = run([
            "python", "scripts/calibrate_drr.py",
            "--xr", str(xr_path),
            "--pxr", str(pxr_path),
            "--voxel-spacing-mm", str(spacing_xy),
            "--voxel-spacing-z-mm", str(spacing_z),
            "--offset-v-mm", str(offset_v),
            "--detector-size", str(detector_size),
            "--metric", "wasserstein",
            "--fixed-attenuation", str(ATTENUATION_SCALE),
            "--out-drr", str(drr_out),
        ])
        if not ok:
            failed.append((specimen, "calibrate_drr.py failed"))
            continue

        ok = run([
            "python", "scripts/compare_distributions_multi.py",
            "--real", str(pxr_path),
            "--drr", f"calibrated={drr_out}",
            "--out", str(dist_out),
        ])
        if not ok:
            failed.append((specimen, "compare_distributions_multi.py failed"))
            continue

        succeeded.append(specimen)

    print(f"\n\n=== SUMMARY ===")
    print(f"Succeeded ({len(succeeded)}): {succeeded}")
    print(f"Skipped ({len(skipped)}):")
    for s, reason in skipped:
        print(f"  {s}: {reason}")
    print(f"Failed ({len(failed)}):")
    for s, reason in failed:
        print(f"  {s}: {reason}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xr-dir", required=True, help="RegistrationHighRes folder with CT volumes")
    parser.add_argument("--pxr-dir", required=True, help="dataset/radiograph_portable_2026_tif folder")
    parser.add_argument("--csv", required=True, help="portable_xr_metadata.csv")
    parser.add_argument("--results-dir", default="results/PXR_DRR_2026")
    parser.add_argument("--detector-size", type=int, default=512)
    args = parser.parse_args()
    main(args.xr_dir, args.pxr_dir, args.csv, args.results_dir, args.detector_size)