"""
Generate a full synthetic training set across all specimens, with a
specimen-level train/val/test split. Specimens are generated in parallel
across CPU cores (each specimen is fully independent work).

Usage:
    python scripts/build_training_set.py \
        --root /home/phukon/code_python/Dataset_Vitimage2019 \
        --out ~/code_python/VineRadiologistAI/dataset \
        --n-samples-per-specimen 50 \
        --val-specimens CEP015_RES2 CEP019_S3 \
        --test-specimens CEP022_APO3 \
        --with-masks \
        --n-workers 12
"""

import argparse
import json
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import tifffile as tiff

from VineRadiologist import (
    load_specimen, DEFAULT_CONFIG, deform_batch,
    sample_pose, render_pose, render_pose_mask,
)


def generate_for_specimen(root, dataset, specimen, n_samples, out_dir, cfg, seed, with_masks):
    """Generate all samples for one specimen. Takes an integer seed (not a
    shared rng), so this can run safely in its own process."""
    rng = np.random.default_rng(seed)
    data = load_specimen(root, specimen, dataset)
    volume = data["volume"]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for i in range(n_samples):
        field_seed = int(rng.integers(0, 2**31 - 1))

        if with_masks:
            to_deform = {"volume": volume, "whiterot": data["whiterot"],
                         "necrosis": data["necrosis"], "healthy": data["healthy"]}
        else:
            to_deform = {"volume": volume}
        deformed = deform_batch(to_deform, cfg.deformation, field_seed=field_seed)

        pose = sample_pose(cfg.projection, rng)
        img = render_pose(deformed["volume"], pose, axis=cfg.projection.projection_axis)

        fname = f"sample_{i:04d}.tif"
        tiff.imwrite(out_dir / fname, img.astype(np.float32))

        record = {
            "file": fname, "specimen": specimen,
            "yaw": float(pose.yaw), "pitch": float(pose.pitch),
            "roll": float(pose.roll), "distance": float(pose.distance),
        }

        if with_masks:
            for name in ("whiterot", "necrosis", "healthy"):
                proj = render_pose_mask(deformed[name], pose, axis=cfg.projection.projection_axis)
                mask_fname = f"sample_{i:04d}_mask_{name}.tif"
                tiff.imwrite(out_dir / mask_fname, proj.astype(np.float32))
                record[f"mask_{name}"] = mask_fname

        manifest.append(record)

    return specimen, manifest


def _worker(args):
    (root, dataset, specimen, split_name, n_samples, out_root, seed, with_masks) = args
    out_dir = out_root / split_name / specimen
    print(f"[{split_name}] starting {specimen} -> {out_dir}", flush=True)
    _, records = generate_for_specimen(root, dataset, specimen, n_samples, out_dir,
                                        DEFAULT_CONFIG, seed, with_masks)
    print(f"[{split_name}] done {specimen} ({len(records)} samples)", flush=True)
    return split_name, specimen, records


def main(root, out_root, n_samples_per_specimen, val_specimens, test_specimens,
         with_masks, dataset="", seed=None, n_workers=None):
    root = Path(root)
    all_specimens = sorted(p.name for p in root.iterdir() if p.is_dir())

    val_specimens = set(val_specimens or [])
    test_specimens = set(test_specimens or [])
    train_specimens = [s for s in all_specimens if s not in val_specimens and s not in test_specimens]

    splits = {"train": train_specimens, "val": sorted(val_specimens), "test": sorted(test_specimens)}
    print("Split (specimen-level, no leakage):")
    for split_name, specs in splits.items():
        print(f"  {split_name}: {specs}")

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # build the full job list across all splits, each specimen gets its own
    # deterministic seed derived from the base seed (or from its name, if
    # no base seed given) so results are reproducible per-specimen even
    # though jobs run in parallel and finish in unpredictable order
    base_seed = seed if seed is not None else 0
    jobs = []
    for split_name, specimens in splits.items():
        for idx, specimen in enumerate(specimens):
            specimen_seed = base_seed + hash((split_name, specimen)) % (2**16)
            jobs.append((root, dataset, specimen, split_name, n_samples_per_specimen,
                         out_root, specimen_seed, with_masks))

    n_workers = n_workers or min(len(jobs), 12)
    print(f"\nRunning {len(jobs)} specimens across {n_workers} worker processes...\n")

    full_manifest = {k: [] for k in splits}
    manifest_path = out_root / "manifest.json"

    with Pool(processes=n_workers) as pool:
        for split_name, specimen, records in pool.imap_unordered(_worker, jobs):
            full_manifest[split_name].extend(records)
            # write manifest after every specimen completes, so an
            # interrupted run still leaves a valid partial manifest
            with open(manifest_path, "w") as f:
                json.dump({"splits": {k: v for k, v in splits.items()},
                           "samples": full_manifest,
                           "complete": False}, f, indent=2)
            total_so_far = sum(len(v) for v in full_manifest.values())
            print(f"  (manifest updated: {total_so_far} samples so far)")

    with open(manifest_path, "w") as f:
        json.dump({"splits": {k: v for k, v in splits.items()},
                   "samples": full_manifest,
                   "complete": True}, f, indent=2)
    print(f"\nWrote manifest: {manifest_path}")
    print(f"Total samples: {sum(len(v) for v in full_manifest.values())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-samples-per-specimen", type=int, default=50)
    parser.add_argument("--val-specimens", nargs="+", default=[])
    parser.add_argument("--test-specimens", nargs="+", default=[])
    parser.add_argument("--with-masks", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-workers", type=int, default=None,
                         help="number of parallel worker processes (default: min(n_specimens, 12))")
    args = parser.parse_args()

    main(args.root, args.out, args.n_samples_per_specimen,
         args.val_specimens, args.test_specimens, args.with_masks,
         dataset=args.dataset, seed=args.seed, n_workers=args.n_workers)