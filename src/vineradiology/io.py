"""IO helpers for loading CT volumes and segmentation masks."""

from pathlib import Path
import numpy as np
import tifffile as tiff


def load_volume(path: str, hu_scale: float = 1000.0) -> np.ndarray:
    """Load a registered CT volume TIFF and scale it to a working range.

    Parameters
    ----------
    path : path to the registered volume TIFF
    hu_scale : divisor applied after casting to float32 (matches the
        convention used elsewhere in the CEP/xvine pipelines)
    """
    vol = tiff.imread(path).astype(np.float32)
    return vol / hu_scale


def load_mask(path: str) -> np.ndarray:
    """Load a binary/segmentation mask TIFF as float32."""
    return tiff.imread(path).astype(np.float32)


def load_specimen(root: str, specimen: str, dataset: str = "") -> dict:
    """Load a CT volume plus the standard tissue masks for one specimen.

    Expects the same directory layout as the original CEP/2DProjection data:
        <root>/<dataset>/<specimen>/CT/registered.tif
        <root>/<dataset>/<specimen>/SEG/segmentation_AMADOU.tif   (white rot)
        <root>/<dataset>/<specimen>/SEG/segmentation_NECROSE.tif (necrosis)
        <root>/<dataset>/<specimen>/SEG/segmentation_SAIN.tif    (healthy)
    """
    base = Path(root) / dataset / specimen
    return {
        "volume": load_volume(str(base / "CT" / "registered.tif")),
        "whiterot": load_mask(str(base / "SEG" / "segmentation_AMADOU.tif")),
        "necrosis": load_mask(str(base / "SEG" / "segmentation_NECROSE.tif")),
        "healthy": load_mask(str(base / "SEG" / "segmentation_SAIN.tif")),
    }
