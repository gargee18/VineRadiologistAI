"""
Save independently min-max normalized versions of a real PXR image and
any number of DRR variants into one folder, for manual side-by-side
comparison (e.g. in Fiji) on a consistent [0,1] scale.

Usage:
    python scripts/save_normalized_compare.py \
        --real raw_tiff_check/CEP_1191/69ca5e44e42f5e0a22520ec8_inverted.tif \
        --drr "no_gamma=test_drr_calibration/calibrated_1191/calibrated_1191_wasserstein.tif" \
        --drr "gamma_v1=test_drr_calibration/callibrated_1191_gamma1/calibrated_1191_gamma.tif" \
        --drr "gamma_v2=test_drr_calibration/callibrated_1191_gamma2/calibrated_1191_grid2.tif" \
        --out-dir normalized_compare

--invert-real: pass this if --real is a raw MONOCHROME1 export (background
bright, specimen dark), to match DRR polarity before normalizing. Don't
pass it if --real is already inverted (e.g. an "_inverted.tif" file).

Each --drr takes a "label=path" pair, repeat for as many variants as you
want saved, not fixed at any specific number.
"""

import argparse
from pathlib import Path

import numpy as np
import tifffile as tiff


def normalize(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float64)
    lo, hi = img.min(), img.max()
    if hi <= lo:
        return np.zeros_like(img)
    return (img - lo) / (hi - lo)


def parse_drr_arg(arg: str):
    if "=" not in arg:
        raise ValueError(f"--drr expects 'label=path', got: {arg}")
    label, path = arg.split("=", 1)
    return label, path


def main(real_path, invert_real, drr_args, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    real = tiff.imread(real_path).astype(np.float64)
    if invert_real:
        real = real.max() - real
    real_n = normalize(real)
    out_path = out_dir / "real_pxr_normalized.tif"
    tiff.imwrite(str(out_path), real_n.astype(np.float32))
    print(f"real: saved {out_path}  min={real_n.min():.3f} max={real_n.max():.3f}")

    for arg in drr_args:
        label, path = parse_drr_arg(arg)
        img = tiff.imread(path)
        img_n = normalize(img)
        out_path = out_dir / f"{label}_normalized.tif"
        tiff.imwrite(str(out_path), img_n.astype(np.float32))
        print(f"{label}: saved {out_path}  min={img_n.min():.3f} max={img_n.max():.3f}")

    print(f"\nAll normalized images saved to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", required=True, help="path to real PXR tif")
    parser.add_argument("--invert-real", action="store_true",
                         help="invert real image before normalizing (use for raw MONOCHROME1 exports)")
    parser.add_argument("--drr", action="append", required=True,
                         help="'label=path' pair, repeat for multiple DRR variants")
    parser.add_argument("--out-dir", default="normalized_compare")
    args = parser.parse_args()
    main(args.real, args.invert_real, args.drr, args.out_dir)