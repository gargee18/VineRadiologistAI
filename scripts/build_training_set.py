"""
Generate a full synthetic training set across all specimens, with a
specimen-level train/val/test split (never split within a specimen, so
validation actually measures generalization to unseen trunks, not just
unseen views of the same trunk).

Usage:
    python scripts/build_training_set.py \
        --root /home/phukon/code_python/Dataset_Vitimage2019 \
        --out ~/code_python/VineRadiologistAI/dataset \
        --n-samples-per-specimen 50 \
        --val-specimens CEP015_RES2 CEP019_S3 \
        --test-specimens CEP022_APO3 \
        --with-masks
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile as tiff

from VineRadiologist import (
    load_specimen, DEFAULT_CONFIG, random_bend_and_elastic,
    sample_pose, render_pose, render_pose_mask,
)


def generate_for_specimen(root, dataset, specimen, n_samples, out_dir, cfg, rng, with_masks):
    data = load_specimen(root, specimen, dataset)
    volume = data["volume"]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for i in range(n_samples):
        field_seed = int(rng.integers(0, 2**31 - 1))

        deformed = random_bend_and_elastic(volume, cfg.deformation, field_seed=field_seed)
        pose = sample_pose(cfg.projection, rng)
        img = render_pose(deformed, pose, axis=cfg.projection.projection_axis)

        fname = f"sample_{i:04d}.tif"
        tiff.imwrite(out_dir / fname, img.astype(np.float32))

        record = {
            "file": fname, "specimen": specimen,
            "yaw": float(pose.yaw), "pitch": float(pose.pitch),
            "roll": float(pose.roll), "distance": float(pose.distance),
        }

        if with_masks:
            for name in ("whiterot", "necrosis", "healthy"):
                deformed_mask = random_bend_and_elastic(data[name], cfg.deformation, field_seed=field_seed)
                proj = render_pose_mask(deformed_mask, pose, axis=cfg.projection.projection_axis)
                mask_fname = f"sample_{i:04d}_mask_{name}.tif"
                tiff.imwrite(out_dir / mask_fname, proj.astype(np.float32))
                record[f"mask_{name}"] = mask_fname

        manifest.append(record)

    return manifest


def main(root, out_root, n_samples_per_specimen, val_specimens, test_specimens,
         with_masks, dataset="", seed=None):
    root = Path(root)
    all_specimens = sorted(p.name for p in root.iterdir() if p.is_dir())

    val_specimens = set(val_specimens or [])
    test_specimens = set(test_specimens or [])
    train_specimens = [s for s in all_specimens if s not in val_specimens and s not in test_specimens]

    splits = {"train": train_specimens, "val": sorted(val_specimens), "test": sorted(test_specimens)}
    print("Split (specimen-level, no leakage):")
    for split_name, specs in splits.items():
        print(f"  {split_name}: {specs}")

    rng = np.random.default_rng(seed)
    out_root = Path(out_root)
    full_manifest = {}

    for split_name, specimens in splits.items():
        split_manifest = []
        for specimen in specimens:
            out_dir = out_root / split_name / specimen
            print(f"[{split_name}] generating {n_samples_per_specimen} samples for {specimen} -> {out_dir}")
            records = generate_for_specimen(
                root, dataset, specimen, n_samples_per_specimen, out_dir,
                DEFAULT_CONFIG, rng, with_masks,
            )
            split_manifest.extend(records)
        full_manifest[split_name] = split_manifest

    manifest_path = out_root / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({"splits": {k: v for k, v in splits.items()}, "samples": full_manifest}, f, indent=2)
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
    args = parser.parse_args()

    main(args.root, args.out, args.n_samples_per_specimen,
         args.val_specimens, args.test_specimens, args.with_masks,
         dataset=args.dataset, seed=args.seed)