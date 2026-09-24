import argparse
from pathlib import Path

import numpy as np
import tifffile as tiff


PROJECT_ROOT = Path(
    "/mnt/41d6c007-0c9e-41e2-b2eb-8d9c032e9e53/"
    "gargee/code_python/VineRadiologistAI"
)

DRR_DIR = PROJECT_ROOT / "results/calibration_outputs"

PXR_DIR = (
    PROJECT_ROOT
    / "dataset/radiograph_portable_2026_tif"
)

MASK_DRR_DIR = PROJECT_ROOT / "results/masks/drr"
MASK_PXR_DIR = PROJECT_ROOT / "results/masks/pxr"


# ============================================================
# PXR FILE MAPPING
# ============================================================

PXR_FILES = {
    "313B": "69ca5d9ee42f5e088ff0f93b.tif",
    "318": "69ca5db1e42f5e09df2bb4a6.tif",
    "322": "69ca5dc4e42f5e09df2bb4ac.tif",
    "323": "69ca5de6e42f5e09b4512919.tif",
    "330": "69ca5df7e42f5e09df2bb4b0.tif",
    "335": "69ca5e1be42f5e0a22520ec4.tif",
    "368B": "69ca536b2ad9c81239913207.tif",
    "378A": "69ca4b43e42f5e026f6dd0e7.tif",
    "378B": "69ca5613e42f5e07eeeed92b.tif",
    "380A": "69ca562de42f5e0682a2cf2d.tif",
    "764B": "69ca5d80e42f5e088ff0f937.tif",
    "988B": "69ca5645e42f5e0682a2cf31.tif",
    "1181": "69ca6a532ad9c81239913234.tif",
    "1186A": "69ca569de42f5e071ae9bcf3.tif",
    "1189": "69ca67ad2ad9c8123991322c.tif",
    "1191": "69ca5e44e42f5e0a22520ec8.tif",
    "1193": "69ca5e6fe42f5e09df2bb4b4.tif",
    "1195": "69ca56b6e42f5e071ae9bcfa.tif",
    "1266A": "69ca56d2e42f5e0702f748da.tif",
    "2184A": "69ca5676e42f5e071ae9bced.tif",
}


# ============================================================
# MASK
# ============================================================

def make_mask(
    image: np.ndarray,
    threshold: float,
) -> np.ndarray:

    mask = image >= threshold

    # Save as normal 8-bit black/white image:
    # background = 0
    # foreground = 255
    return (
        mask.astype(np.uint8)
        * 255
    )


# ============================================================
# FIND DRR AUTOMATICALLY
# ============================================================

def find_drr(
    specimen: str,
) -> Path:

    preferred = (
        DRR_DIR
        / f"CEP_{specimen}_manual_pose_DRR_axis0.tif"
    )

    if preferred.exists():
        return preferred

    candidates = sorted(
        DRR_DIR.glob(
            f"CEP_{specimen}*.tif"
        )
    )

    if not candidates:

        raise FileNotFoundError(
            f"No DRR found for CEP_{specimen} "
            f"in {DRR_DIR}"
        )

    if len(candidates) > 1:

        print(
            f"WARNING: multiple DRRs found "
            f"for CEP_{specimen}:"
        )

        for candidate in candidates:
            print(
                f"  {candidate}"
            )

        print(
            f"Using: {candidates[0]}"
        )

    return candidates[0]


# ============================================================
# FIND PXR
# ============================================================

def find_pxr(
    specimen: str,
) -> Path:

    if specimen not in PXR_FILES:

        raise ValueError(
            f"No PXR mapping available "
            f"for specimen {specimen}"
        )

    path = (
        PXR_DIR
        / f"CEP_{specimen}"
        / PXR_FILES[specimen]
    )

    if not path.exists():

        raise FileNotFoundError(
            f"PXR not found: {path}"
        )

    return path


# ============================================================
# PROCESS ONE SPECIMEN
# ============================================================

def process_specimen(
    specimen: str,
    drr_threshold: float,
    pxr_threshold: float,
    drr_path: Path = None,
):

    print()
    print(
        "========================================"
    )

    print(
        f"CEP_{specimen}"
    )

    print(
        "========================================"
    )

    # --------------------------------------------------------
    # DRR
    # --------------------------------------------------------

    if drr_path is not None:

        current_drr_path = Path(
            drr_path
        )

        if not current_drr_path.exists():

            raise FileNotFoundError(
                f"Specified DRR does not exist: "
                f"{current_drr_path}"
            )

        print(
            "Using explicitly specified DRR:"
        )

    else:

        current_drr_path = find_drr(
            specimen
        )

        print(
            "Using automatically found DRR:"
        )

    print(
        current_drr_path
    )

    drr = tiff.imread(
        current_drr_path
    )

    drr_mask = make_mask(
        drr,
        drr_threshold,
    )

    MASK_DRR_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Keep the actual DRR filename so test DRRs
    # do not overwrite one another.
    drr_output = (
        MASK_DRR_DIR
        / f"{current_drr_path.stem}_mask.tif"
    )

    tiff.imwrite(
        drr_output,
        drr_mask,
    )

    print(
        f"DRR threshold: "
        f"{drr_threshold}"
    )

    print(
        f"DRR shape: "
        f"{drr.shape}"
    )

    print(
        f"DRR foreground pixels: "
        f"{np.count_nonzero(drr_mask)}"
    )

    print(
        f"Saved DRR mask: "
        f"{drr_output}"
    )

    # --------------------------------------------------------
    # PXR
    # --------------------------------------------------------

    pxr_path = find_pxr(
        specimen
    )

    print()
    print(
        "PXR:"
    )

    print(
        pxr_path
    )

    pxr = tiff.imread(
        pxr_path
    )

    pxr_mask = make_mask(
        pxr,
        pxr_threshold,
    )

    MASK_PXR_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pxr_output = (
        MASK_PXR_DIR
        / f"CEP_{specimen}_PXR_mask.tif"
    )

    tiff.imwrite(
        pxr_output,
        pxr_mask,
    )

    print(
        f"PXR threshold: "
        f"{pxr_threshold}"
    )

    print(
        f"PXR shape: "
        f"{pxr.shape}"
    )

    print(
        f"PXR foreground pixels: "
        f"{np.count_nonzero(pxr_mask)}"
    )

    print(
        f"Saved PXR mask: "
        f"{pxr_output}"
    )


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--drr-threshold",
        type=float,
        required=True,
        help="Threshold for the DRR.",
    )

    parser.add_argument(
        "--pxr-threshold",
        type=float,
        required=True,
        help="Threshold for the real PXR.",
    )

    parser.add_argument(
        "--specimen",
        default=None,
        help=(
            "Process one specimen only, "
            "for example --specimen 1191. "
            "If omitted, process all specimens."
        ),
    )

    parser.add_argument(
        "--drr-path",
        default=None,
        help=(
            "Optional explicit path to a DRR. "
            "Useful for test DRRs. "
            "Requires --specimen."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Explicit DRR
    # --------------------------------------------------------

    if args.drr_path is not None:

        if args.specimen is None:

            parser.error(
                "--drr-path requires --specimen"
            )

        process_specimen(
            specimen=args.specimen,
            drr_threshold=args.drr_threshold,
            pxr_threshold=args.pxr_threshold,
            drr_path=Path(args.drr_path),
        )

    # --------------------------------------------------------
    # One specimen
    # --------------------------------------------------------

    elif args.specimen is not None:

        process_specimen(
            specimen=args.specimen,
            drr_threshold=args.drr_threshold,
            pxr_threshold=args.pxr_threshold,
        )

    # --------------------------------------------------------
    # All specimens
    # --------------------------------------------------------

    else:

        for specimen in PXR_FILES:

            try:

                process_specimen(
                    specimen=specimen,
                    drr_threshold=args.drr_threshold,
                    pxr_threshold=args.pxr_threshold,
                )

            except Exception as e:

                print(
                    f"ERROR processing "
                    f"CEP_{specimen}: {e}"
                )