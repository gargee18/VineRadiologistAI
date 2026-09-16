"""
Reusable cone-beam DRR renderer + calibration utilities.

This file merges:
  1. the newer calibration workflow features:
       - attenuation sweeps
       - SSIM / NCC / PSNR / MI / Wasserstein metrics
       - optional DRR/PXR masks
       - CSV sweep output
       - optional manual Fiji transform
       - crop-to-PXR
       - GPU/CPU execution
  2. the DRR implementation currently used by manual registration:
       - detector pixel spacing = 0.139 mm at 3072 px
       - configurable SID / SPD
       - configurable beam axis
       - Fiji XYZ pose -> NumPy ZYX conversion
       - chunked CuPy ray casting
       - LOG / attenuation-domain output
       - NO 1 - exp(-A)

Important convention:
  - Legacy calibration default beam axis = 1
  - Manual registration should explicitly call make_geometry(..., beam_axis=0)

The DRR output is:

    A = attenuation_scale * integral(volume dl)

which corresponds to the log attenuation quantity:

    -log(I / I0) = A

No exponential opacity conversion is applied here.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from PIL import Image
from scipy.ndimage import map_coordinates as np_map_coordinates
from scipy.optimize import minimize_scalar
from scipy.stats import wasserstein_distance

from VineRadiologist.io import load_volume


# =============================================================================
# PROJECT DEFAULTS
# =============================================================================

DEFAULT_SID_MM = 1230.0
DEFAULT_SPD_MM = 800.0

FULL_DETECTOR_PIXELS = 3072
FULL_DETECTOR_PIXEL_SPACING_MM = 0.139

# Keep legacy calibration defaults here.
# Manual registration can explicitly pass OFFSET_V=-20 and beam_axis=0.
DEFAULT_OFFSET_V_MM = 0.0
DEFAULT_OFFSET_U_MM = 0.0
DEFAULT_BEAM_AXIS = 1

DEFAULT_ATTENUATION_SCALE = 0.015

DEFAULT_ROW_CHUNK = 16
DEFAULT_SAMPLE_CHUNK = 128


# =============================================================================
# OPTIONAL GPU BACKEND
# =============================================================================

try:
    import cupy as cp
    from cupyx.scipy.ndimage import map_coordinates as cp_map_coordinates

    CUPY_AVAILABLE = True
except Exception:
    cp = None
    cp_map_coordinates = None
    CUPY_AVAILABLE = False


# =============================================================================
# GEOMETRY
# =============================================================================

@dataclass
class ConeBeamGeometry:
    sid_mm: float = DEFAULT_SID_MM
    spd_mm: float = DEFAULT_SPD_MM
    detector_pixel_spacing_mm: float = FULL_DETECTOR_PIXEL_SPACING_MM
    voxel_spacing_mm: object = (1.0, 1.0, 1.0)  # NumPy Z,Y,X
    detector_shape: Tuple[int, int] = (512, 512)
    offset_v_mm: float = DEFAULT_OFFSET_V_MM
    offset_u_mm: float = DEFAULT_OFFSET_U_MM
    beam_axis: int = DEFAULT_BEAM_AXIS

    def spacing_for_axis(self, axis: int) -> float:
        spacing = _spacing_zyx(self.voxel_spacing_mm)
        return float(spacing[int(axis)])


def _spacing_zyx(spacing) -> np.ndarray:
    arr = np.asarray(spacing, dtype=np.float64)

    if arr.ndim == 0:
        return np.repeat(float(arr), 3)

    arr = arr.reshape(-1)

    if arr.size != 3:
        raise ValueError(
            "voxel spacing must be a scalar or a 3-value (Z,Y,X) sequence"
        )

    return arr


def detector_pixel_spacing(
    detector_size: int,
    full_detector_pixels: int = FULL_DETECTOR_PIXELS,
    full_detector_pixel_spacing_mm: float = FULL_DETECTOR_PIXEL_SPACING_MM,
) -> float:
    """
    Keep the detector's physical field of view fixed when rendering at
    another pixel resolution.
    """
    physical_extent_mm = (
        float(full_detector_pixels)
        * float(full_detector_pixel_spacing_mm)
    )
    return physical_extent_mm / float(detector_size)


def make_geometry(
    detector_size: int = 512,
    voxel_spacing_mm: float = None,
    voxel_spacing_z_mm: float = None,
    spacing_zyx: Optional[Sequence[float]] = None,
    sid_mm: float = DEFAULT_SID_MM,
    spd_mm: float = DEFAULT_SPD_MM,
    offset_v_mm: float = DEFAULT_OFFSET_V_MM,
    offset_u_mm: float = DEFAULT_OFFSET_U_MM,
    beam_axis: int = DEFAULT_BEAM_AXIS,
    full_detector_pixels: int = FULL_DETECTOR_PIXELS,
    full_detector_pixel_spacing_mm: float = FULL_DETECTOR_PIXEL_SPACING_MM,
) -> ConeBeamGeometry:
    """
    Build a cone-beam geometry.

    Preferred spacing input:
        spacing_zyx=(z_spacing, y_spacing, x_spacing)

    Legacy compatibility:
        voxel_spacing_mm + voxel_spacing_z_mm
    """
    if spacing_zyx is not None:
        spacing = _spacing_zyx(spacing_zyx)
    else:
        inplane = (
            float(voxel_spacing_mm)
            if voxel_spacing_mm is not None
            else 0.7224
        )

        z_spacing = (
            float(voxel_spacing_z_mm)
            if voxel_spacing_z_mm is not None
            else inplane
        )

        spacing = np.array(
            [z_spacing, inplane, inplane],
            dtype=np.float64,
        )

    scaled_detector_spacing = detector_pixel_spacing(
        detector_size,
        full_detector_pixels=full_detector_pixels,
        full_detector_pixel_spacing_mm=full_detector_pixel_spacing_mm,
    )

    return ConeBeamGeometry(
        sid_mm=float(sid_mm),
        spd_mm=float(spd_mm),
        detector_pixel_spacing_mm=float(scaled_detector_spacing),
        voxel_spacing_mm=tuple(map(float, spacing)),
        detector_shape=(int(detector_size), int(detector_size)),
        offset_v_mm=float(offset_v_mm),
        offset_u_mm=float(offset_u_mm),
        beam_axis=int(beam_axis),
    )


# =============================================================================
# FIJI / NUMPY TRANSFORM HELPERS
# =============================================================================

def fiji_pose_to_centered_zyx(
    rotation_xyz: np.ndarray,
    true_translation_xyz: np.ndarray,
):
    """
    Convert a pure centered Fiji pose from XYZ to NumPy ZYX coordinates.
    """
    permutation = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    rotation_xyz = np.asarray(
        rotation_xyz,
        dtype=np.float64,
    ).reshape(3, 3)

    true_translation_xyz = np.asarray(
        true_translation_xyz,
        dtype=np.float64,
    ).reshape(3)

    rotation_zyx = (
        permutation
        @ rotation_xyz
        @ permutation.T
    )

    translation_zyx = (
        permutation
        @ true_translation_xyz
    )

    return rotation_zyx, translation_zyx


def renderer_matrix_from_fiji_pose(
    rotation_xyz: np.ndarray,
    true_translation_xyz: np.ndarray,
    vol_shape_zyx,
    spacing_zyx,
) -> np.ndarray:
    """
    Build the full 4x4 Fiji-style transform used by manual registration.

        p_out = R (p - center) + center + translation
    """
    sz, sy, sx = map(
        float,
        _spacing_zyx(spacing_zyx),
    )
    nz, ny, nx = map(
        int,
        vol_shape_zyx,
    )

    center_xyz = np.array(
        [
            nx * sx / 2.0,
            ny * sy / 2.0,
            nz * sz / 2.0,
        ],
        dtype=np.float64,
    )

    rotation_xyz = np.asarray(
        rotation_xyz,
        dtype=np.float64,
    ).reshape(3, 3)

    true_translation_xyz = np.asarray(
        true_translation_xyz,
        dtype=np.float64,
    ).reshape(3)

    transform = np.eye(
        4,
        dtype=np.float64,
    )

    transform[:3, :3] = rotation_xyz
    transform[:3, 3] = (
        true_translation_xyz
        + center_xyz
        - rotation_xyz @ center_xyz
    )

    return transform


def fiji_matrix_to_pure_pose(
    transform_xyz: np.ndarray,
    vol_shape_zyx,
    spacing_zyx,
):
    """
    Recover centered Fiji rotation + true translation from the saved
    4x4 manual-registration transform.
    """
    transform_xyz = np.asarray(
        transform_xyz,
        dtype=np.float64,
    ).reshape(4, 4)

    rotation_xyz = transform_xyz[:3, :3].copy()

    # Clean tiny numerical drift.
    u_svd, _, vh_svd = np.linalg.svd(rotation_xyz)
    rotation_xyz = u_svd @ vh_svd

    sz, sy, sx = map(
        float,
        _spacing_zyx(spacing_zyx),
    )
    nz, ny, nx = map(
        int,
        vol_shape_zyx,
    )

    center_xyz = np.array(
        [
            nx * sx / 2.0,
            ny * sy / 2.0,
            nz * sz / 2.0,
        ],
        dtype=np.float64,
    )

    t_total_xyz = transform_xyz[:3, 3]

    true_translation_xyz = (
        rotation_xyz @ center_xyz
        + t_total_xyz
        - center_xyz
    )

    return rotation_xyz, true_translation_xyz


def load_transform_pose(
    transform_path,
    vol_shape_zyx,
    spacing_zyx,
):
    transform = np.loadtxt(
        transform_path,
        dtype=np.float64,
    )

    if transform.shape != (4, 4):
        raise ValueError(
            f"Expected 4x4 transform, got {transform.shape}"
        )

    return fiji_matrix_to_pure_pose(
        transform,
        vol_shape_zyx,
        spacing_zyx,
    )


# =============================================================================
# CORE DRR RENDERER
# =============================================================================

def _prepare_pose(
    rotation_xyz=None,
    true_translation_xyz=None,
):
    if rotation_xyz is None:
        rotation_xyz = np.eye(
            3,
            dtype=np.float64,
        )

    if true_translation_xyz is None:
        true_translation_xyz = np.zeros(
            3,
            dtype=np.float64,
        )

    return fiji_pose_to_centered_zyx(
        rotation_xyz,
        true_translation_xyz,
    )


def generate_cone_beam_drr_gpu(
    volume: np.ndarray,
    geometry: ConeBeamGeometry,
    attenuation_scale: float = DEFAULT_ATTENUATION_SCALE,
    beam_axis: Optional[int] = None,
    rotation_xyz: Optional[np.ndarray] = None,
    true_translation_xyz: Optional[np.ndarray] = None,
    object_transform: Optional[np.ndarray] = None,
    row_chunk: int = DEFAULT_ROW_CHUNK,
    sample_chunk: int = DEFAULT_SAMPLE_CHUNK,
    verbose: bool = False,
) -> np.ndarray:
    """
    Chunked CuPy cone-beam projector.

    Output remains in LOG / attenuation domain:

        A = attenuation_scale * integral(volume dl)

    object_transform:
        Optional 4x4 Fiji transform. This is accepted for compatibility with
        the newer calibration script. Do not also pass rotation_xyz /
        true_translation_xyz when using object_transform.
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError(
            "CuPy is not available. "
            "Use the CPU projector or install the matching CuPy CUDA package."
        )

    volume = np.asarray(volume)
    spacing_zyx = _spacing_zyx(
        geometry.voxel_spacing_mm
    )

    if beam_axis is None:
        beam_axis = int(geometry.beam_axis)
    else:
        beam_axis = int(beam_axis)

    if object_transform is not None:
        if (
            rotation_xyz is not None
            or true_translation_xyz is not None
        ):
            raise ValueError(
                "Pass either object_transform OR rotation/translation, not both."
            )

        rotation_xyz, true_translation_xyz = (
            fiji_matrix_to_pure_pose(
                object_transform,
                volume.shape,
                spacing_zyx,
            )
        )

    rows, cols = map(
        int,
        geometry.detector_shape,
    )

    if rows != cols:
        raise ValueError(
            "Projector expects a square detector before top-cropping."
        )

    other_axes = [
        axis
        for axis in range(3)
        if axis != beam_axis
    ]

    n_beam = int(
        volume.shape[beam_axis]
    )

    beam_spacing = float(
        spacing_zyx[beam_axis]
    )

    half_extent_mm = (
        n_beam / 2.0
    ) * beam_spacing

    src_pos = -float(
        geometry.spd_mm
    )

    det_pos = (
        float(geometry.sid_mm)
        - float(geometry.spd_mm)
    )

    source_to_detector = (
        det_pos - src_pos
    )

    rotation_zyx, translation_zyx = (
        _prepare_pose(
            rotation_xyz,
            true_translation_xyz,
        )
    )

    beam_center_mm = float(
        translation_zyx[beam_axis]
    )

    t_enter = (
        beam_center_mm
        - half_extent_mm
        - src_pos
    ) / source_to_detector

    t_exit = (
        beam_center_mm
        + half_extent_mm
        - src_pos
    ) / source_to_detector

    n_samples = n_beam * 2

    path_length_mm = (
        abs(t_exit - t_enter)
        * abs(source_to_detector)
    )

    step_mm = (
        path_length_mm
        / float(n_samples)
    )

    t_all = np.linspace(
        t_enter,
        t_exit,
        n_samples,
        dtype=np.float32,
    )

    det_u_all = (
        np.arange(
            cols,
            dtype=np.float32,
        )
        - cols / 2.0
    ) * float(
        geometry.detector_pixel_spacing_mm
    )

    det_v_all = (
        np.arange(
            rows,
            dtype=np.float32,
        )
        - rows / 2.0
    ) * float(
        geometry.detector_pixel_spacing_mm
    )

    vol_gpu = cp.asarray(
        volume.astype(
            np.float32,
            copy=False,
        )
    )

    rotation_inv_gpu = cp.asarray(
        rotation_zyx.T.astype(
            np.float32
        )
    )

    translation_gpu = cp.asarray(
        translation_zyx.astype(
            np.float32
        )
    )

    result = np.zeros(
        (rows, cols),
        dtype=np.float32,
    )

    if verbose:
        print(
            "GPU DRR: "
            f"detector={rows}x{cols}, "
            f"SID={geometry.sid_mm:g} mm, "
            f"SPD={geometry.spd_mm:g} mm, "
            f"pixel={geometry.detector_pixel_spacing_mm:.6f} mm, "
            f"beam_axis={beam_axis}, "
            f"samples={n_samples}"
        )

    for r0 in range(
        0,
        rows,
        int(row_chunk),
    ):
        r1 = min(
            r0 + int(row_chunk),
            rows,
        )

        det_v = cp.asarray(
            det_v_all[r0:r1]
        )
        det_u = cp.asarray(
            det_u_all
        )

        grid_v, grid_u = cp.meshgrid(
            det_v,
            det_u,
            indexing="ij",
        )

        accum = cp.zeros(
            (r1 - r0, cols),
            dtype=cp.float32,
        )

        for sample0 in range(
            0,
            n_samples,
            int(sample_chunk),
        ):
            sample1 = min(
                sample0 + int(sample_chunk),
                n_samples,
            )

            t = cp.asarray(
                t_all[sample0:sample1]
            )[:, None, None]

            beam_coord = (
                src_pos
                + t * source_to_detector
            )

            v_coord = (
                t * grid_v[None, :, :]
                + float(geometry.offset_v_mm)
            )

            u_coord = (
                t * grid_u[None, :, :]
                + float(geometry.offset_u_mm)
            )

            shape = (
                sample1 - sample0,
                r1 - r0,
                cols,
            )

            beam_coord = cp.broadcast_to(
                beam_coord,
                shape,
            )

            physical = [
                None,
                None,
                None,
            ]

            physical[beam_axis] = (
                beam_coord
            )

            physical[other_axes[0]] = (
                v_coord
            )

            physical[other_axes[1]] = (
                u_coord
            )

            p0 = cp.broadcast_to(
                physical[0],
                shape,
            )
            p1 = cp.broadcast_to(
                physical[1],
                shape,
            )
            p2 = cp.broadcast_to(
                physical[2],
                shape,
            )

            s0 = (
                p0
                - translation_gpu[0]
            )
            s1 = (
                p1
                - translation_gpu[1]
            )
            s2 = (
                p2
                - translation_gpu[2]
            )

            q0 = (
                rotation_inv_gpu[0, 0] * s0
                + rotation_inv_gpu[0, 1] * s1
                + rotation_inv_gpu[0, 2] * s2
            )

            q1 = (
                rotation_inv_gpu[1, 0] * s0
                + rotation_inv_gpu[1, 1] * s1
                + rotation_inv_gpu[1, 2] * s2
            )

            q2 = (
                rotation_inv_gpu[2, 0] * s0
                + rotation_inv_gpu[2, 1] * s1
                + rotation_inv_gpu[2, 2] * s2
            )

            z_idx = (
                q0 / float(spacing_zyx[0])
                + volume.shape[0] / 2.0
            )

            y_idx = (
                q1 / float(spacing_zyx[1])
                + volume.shape[1] / 2.0
            )

            x_idx = (
                q2 / float(spacing_zyx[2])
                + volume.shape[2] / 2.0
            )

            sample_coords = cp.stack(
                (
                    z_idx,
                    y_idx,
                    x_idx,
                ),
                axis=0,
            )

            sampled = cp_map_coordinates(
                vol_gpu,
                sample_coords,
                order=1,
                mode="constant",
                cval=0.0,
            )

            accum += cp.sum(
                sampled,
                axis=0,
            )

        attenuation = (
            accum
            * float(step_mm)
            * float(attenuation_scale)
        )

        # Log / attenuation domain.
        # No 1 - exp(-attenuation).
        result[r0:r1, :] = cp.asnumpy(
            attenuation
        )

    del (
        vol_gpu,
        rotation_inv_gpu,
        translation_gpu,
    )

    cp.get_default_memory_pool().free_all_blocks()

    return result


def generate_cone_beam_drr(
    volume: np.ndarray,
    geometry: ConeBeamGeometry,
    attenuation_scale: float = DEFAULT_ATTENUATION_SCALE,
    beam_axis: Optional[int] = None,
    rotation_xyz: Optional[np.ndarray] = None,
    true_translation_xyz: Optional[np.ndarray] = None,
    object_transform: Optional[np.ndarray] = None,
    row_chunk: int = 8,
    sample_chunk: int = 64,
    verbose: bool = False,
) -> np.ndarray:
    """
    CPU fallback with the same geometry and log-domain output as the GPU path.
    """
    volume = np.asarray(
        volume,
        dtype=np.float32,
    )

    spacing_zyx = _spacing_zyx(
        geometry.voxel_spacing_mm
    )

    if beam_axis is None:
        beam_axis = int(
            geometry.beam_axis
        )
    else:
        beam_axis = int(
            beam_axis
        )

    if object_transform is not None:
        if (
            rotation_xyz is not None
            or true_translation_xyz is not None
        ):
            raise ValueError(
                "Pass either object_transform OR rotation/translation, not both."
            )

        rotation_xyz, true_translation_xyz = (
            fiji_matrix_to_pure_pose(
                object_transform,
                volume.shape,
                spacing_zyx,
            )
        )

    rows, cols = map(
        int,
        geometry.detector_shape,
    )

    if rows != cols:
        raise ValueError(
            "Projector expects a square detector before top-cropping."
        )

    other_axes = [
        axis
        for axis in range(3)
        if axis != beam_axis
    ]

    n_beam = int(
        volume.shape[beam_axis]
    )

    beam_spacing = float(
        spacing_zyx[beam_axis]
    )

    half_extent_mm = (
        n_beam / 2.0
    ) * beam_spacing

    src_pos = -float(
        geometry.spd_mm
    )

    det_pos = (
        float(geometry.sid_mm)
        - float(geometry.spd_mm)
    )

    source_to_detector = (
        det_pos - src_pos
    )

    rotation_zyx, translation_zyx = (
        _prepare_pose(
            rotation_xyz,
            true_translation_xyz,
        )
    )

    beam_center_mm = float(
        translation_zyx[beam_axis]
    )

    t_enter = (
        beam_center_mm
        - half_extent_mm
        - src_pos
    ) / source_to_detector

    t_exit = (
        beam_center_mm
        + half_extent_mm
        - src_pos
    ) / source_to_detector

    n_samples = n_beam * 2

    path_length_mm = (
        abs(t_exit - t_enter)
        * abs(source_to_detector)
    )

    step_mm = (
        path_length_mm
        / float(n_samples)
    )

    t_all = np.linspace(
        t_enter,
        t_exit,
        n_samples,
        dtype=np.float32,
    )

    det_u_all = (
        np.arange(
            cols,
            dtype=np.float32,
        )
        - cols / 2.0
    ) * float(
        geometry.detector_pixel_spacing_mm
    )

    det_v_all = (
        np.arange(
            rows,
            dtype=np.float32,
        )
        - rows / 2.0
    ) * float(
        geometry.detector_pixel_spacing_mm
    )

    rotation_inv = (
        rotation_zyx.T
        .astype(np.float32)
    )

    translation = (
        translation_zyx
        .astype(np.float32)
    )

    result = np.zeros(
        (rows, cols),
        dtype=np.float32,
    )

    if verbose:
        print(
            "CPU DRR: "
            f"detector={rows}x{cols}, "
            f"SID={geometry.sid_mm:g} mm, "
            f"SPD={geometry.spd_mm:g} mm, "
            f"pixel={geometry.detector_pixel_spacing_mm:.6f} mm, "
            f"beam_axis={beam_axis}, "
            f"samples={n_samples}"
        )

    for r0 in range(
        0,
        rows,
        int(row_chunk),
    ):
        r1 = min(
            r0 + int(row_chunk),
            rows,
        )

        det_v = (
            det_v_all[r0:r1]
        )

        det_u = det_u_all

        grid_v, grid_u = np.meshgrid(
            det_v,
            det_u,
            indexing="ij",
        )

        accum = np.zeros(
            (r1 - r0, cols),
            dtype=np.float32,
        )

        for sample0 in range(
            0,
            n_samples,
            int(sample_chunk),
        ):
            sample1 = min(
                sample0 + int(sample_chunk),
                n_samples,
            )

            t = (
                t_all[sample0:sample1]
                [:, None, None]
            )

            beam_coord = (
                src_pos
                + t * source_to_detector
            )

            v_coord = (
                t * grid_v[None, :, :]
                + float(geometry.offset_v_mm)
            )

            u_coord = (
                t * grid_u[None, :, :]
                + float(geometry.offset_u_mm)
            )

            shape = (
                sample1 - sample0,
                r1 - r0,
                cols,
            )

            beam_coord = np.broadcast_to(
                beam_coord,
                shape,
            )

            physical = [
                None,
                None,
                None,
            ]

            physical[beam_axis] = (
                beam_coord
            )

            physical[other_axes[0]] = (
                v_coord
            )

            physical[other_axes[1]] = (
                u_coord
            )

            p0 = np.broadcast_to(
                physical[0],
                shape,
            )

            p1 = np.broadcast_to(
                physical[1],
                shape,
            )

            p2 = np.broadcast_to(
                physical[2],
                shape,
            )

            s0 = (
                p0 - translation[0]
            )

            s1 = (
                p1 - translation[1]
            )

            s2 = (
                p2 - translation[2]
            )

            q0 = (
                rotation_inv[0, 0] * s0
                + rotation_inv[0, 1] * s1
                + rotation_inv[0, 2] * s2
            )

            q1 = (
                rotation_inv[1, 0] * s0
                + rotation_inv[1, 1] * s1
                + rotation_inv[1, 2] * s2
            )

            q2 = (
                rotation_inv[2, 0] * s0
                + rotation_inv[2, 1] * s1
                + rotation_inv[2, 2] * s2
            )

            z_idx = (
                q0 / float(spacing_zyx[0])
                + volume.shape[0] / 2.0
            )

            y_idx = (
                q1 / float(spacing_zyx[1])
                + volume.shape[1] / 2.0
            )

            x_idx = (
                q2 / float(spacing_zyx[2])
                + volume.shape[2] / 2.0
            )

            sampled = np_map_coordinates(
                volume,
                [
                    z_idx,
                    y_idx,
                    x_idx,
                ],
                order=1,
                mode="constant",
                cval=0.0,
            )

            accum += np.sum(
                sampled,
                axis=0,
                dtype=np.float32,
            )

        attenuation = (
            accum
            * float(step_mm)
            * float(attenuation_scale)
        )

        result[r0:r1, :] = (
            attenuation
        )

    return result


def generate_drr(
    volume: np.ndarray,
    geometry: ConeBeamGeometry,
    attenuation_scale: float = DEFAULT_ATTENUATION_SCALE,
    rotation_xyz: Optional[np.ndarray] = None,
    true_translation_xyz: Optional[np.ndarray] = None,
    object_transform: Optional[np.ndarray] = None,
    use_gpu: bool = True,
    row_chunk: int = DEFAULT_ROW_CHUNK,
    sample_chunk: int = DEFAULT_SAMPLE_CHUNK,
    verbose: bool = False,
) -> np.ndarray:
    """
    Main reusable DRR entry point.
    """
    if use_gpu and CUPY_AVAILABLE:
        return generate_cone_beam_drr_gpu(
            volume,
            geometry,
            attenuation_scale=attenuation_scale,
            rotation_xyz=rotation_xyz,
            true_translation_xyz=true_translation_xyz,
            object_transform=object_transform,
            row_chunk=row_chunk,
            sample_chunk=sample_chunk,
            verbose=verbose,
        )

    return generate_cone_beam_drr(
        volume,
        geometry,
        attenuation_scale=attenuation_scale,
        rotation_xyz=rotation_xyz,
        true_translation_xyz=true_translation_xyz,
        object_transform=object_transform,
        row_chunk=max(
            1,
            min(int(row_chunk), 8),
        ),
        sample_chunk=max(
            1,
            min(int(sample_chunk), 64),
        ),
        verbose=verbose,
    )


def generate_drr_from_transform(
    volume: np.ndarray,
    geometry: ConeBeamGeometry,
    transform_xyz: np.ndarray,
    attenuation_scale: float = DEFAULT_ATTENUATION_SCALE,
    use_gpu: bool = True,
    row_chunk: int = DEFAULT_ROW_CHUNK,
    sample_chunk: int = DEFAULT_SAMPLE_CHUNK,
    verbose: bool = False,
) -> np.ndarray:
    """
    Generate a DRR directly from a 4x4 Fiji transform.
    """
    return generate_drr(
        volume,
        geometry,
        attenuation_scale=attenuation_scale,
        object_transform=transform_xyz,
        use_gpu=use_gpu,
        row_chunk=row_chunk,
        sample_chunk=sample_chunk,
        verbose=verbose,
    )


# =============================================================================
# PXR / IMAGE HELPERS
# =============================================================================

def load_image(path: str) -> np.ndarray:
    path = Path(path)

    if path.suffix.lower() == ".dcm":
        import pydicom

        ds = pydicom.dcmread(
            str(path),
            force=True,
        )

        pixels = (
            ds.pixel_array
            .astype(np.float64)
        )

        if (
            getattr(
                ds,
                "PhotometricInterpretation",
                "",
            )
            == "MONOCHROME1"
        ):
            pixels = (
                pixels.max()
                - pixels
            )

        return pixels

    if path.suffix.lower() in (
        ".tif",
        ".tiff",
    ):
        return (
            tiff.imread(path)
            .astype(np.float64)
        )

    return np.array(
        Image.open(path).convert("L"),
        dtype=np.float64,
    )


def find_view(
    specimen_dir: str,
    view: str = "Face",
) -> str:
    import pydicom

    specimen_dir = Path(
        specimen_dir
    )

    for file in sorted(
        specimen_dir.glob("*.dcm")
    ):
        try:
            ds = pydicom.dcmread(
                str(file),
                force=True,
                stop_before_pixels=True,
            )
        except Exception:
            continue

        desc = getattr(
            ds,
            "SeriesDescription",
            "",
        )

        if view.lower() in desc.lower():
            return str(file)

    raise FileNotFoundError(
        f"no '{view}' view found in {specimen_dir}"
    )


def strip_saturated_band(
    img: np.ndarray,
    sat_frac: float = 0.98,
    row_coverage: float = 0.5,
    debug: bool = True,
):
    """
    Return:
        cropped_image, number_of_top_rows_removed
    """
    max_val = img.max()

    if max_val <= 0:
        return img, 0

    sat_mask = (
        img
        >= sat_frac * max_val
    )

    row_sat_ratio = (
        sat_mask.mean(axis=1)
    )

    if debug:
        print(
            f"  [saturation check] image max={max_val:.2f}, "
            f"top-10 mean={img[:10].mean():.2f}, "
            f"top-10 saturation={row_sat_ratio[:10].mean():.3f}, "
            f"overall mean={img.mean():.2f}"
        )

    crop_row = 0

    for i, ratio in enumerate(
        row_sat_ratio
    ):
        if ratio < row_coverage:
            crop_row = i
            break
    else:
        return img, 0

    if crop_row > 0:
        print(
            f"Stripped {crop_row} saturated "
            f"top row(s)"
        )

        return (
            img[crop_row:],
            crop_row,
        )

    return img, 0


def crop_top_to_match_pxr(
    drr: np.ndarray,
    pxr_shape,
) -> np.ndarray:
    """
    Crop the square DRR from the TOP to match the PXR aspect ratio.

    This is preferable to comparing a 512x512 preview with a 2702x3072 PXR
    by raw row counts.
    """
    pxr_h = int(
        pxr_shape[0]
    )

    pxr_w = int(
        pxr_shape[1]
    )

    if pxr_w <= 0:
        return drr

    target_h = int(
        round(
            drr.shape[1]
            * pxr_h
            / pxr_w
        )
    )

    target_h = max(
        1,
        min(
            target_h,
            drr.shape[0],
        ),
    )

    crop_top = (
        drr.shape[0]
        - target_h
    )

    if crop_top > 0:
        return drr[
            crop_top:,
            :
        ]

    return drr


def normalize(
    img: np.ndarray,
) -> np.ndarray:
    img = np.asarray(
        img,
        dtype=np.float64,
    )

    lo = np.nanmin(img)
    hi = np.nanmax(img)

    if hi <= lo:
        return np.zeros_like(
            img
        )

    return (
        img - lo
    ) / (
        hi - lo
    )


def resize_to_match(
    img: np.ndarray,
    target_shape,
) -> np.ndarray:
    from scipy.ndimage import zoom

    if tuple(img.shape) == tuple(
        target_shape
    ):
        return img

    factors = (
        target_shape[0] / img.shape[0],
        target_shape[1] / img.shape[1],
    )

    return zoom(
        img,
        factors,
        order=1,
    )


def auto_crop_to_content(
    sim: np.ndarray,
    real: np.ndarray,
    threshold: float = 0.05,
):
    """
    Legacy content crop retained for compatibility with earlier optimizer mode.
    """
    sim_n = normalize(sim)
    mask = sim_n > float(
        threshold
    )

    if not mask.any():
        return sim, real

    rows = np.any(
        mask,
        axis=1,
    )
    cols = np.any(
        mask,
        axis=0,
    )

    r0, r1 = (
        np.where(rows)[0][[0, -1]]
    )
    c0, c1 = (
        np.where(cols)[0][[0, -1]]
    )

    pad = int(
        0.05 * max(sim.shape)
    )

    r0 = max(
        0,
        r0 - pad,
    )
    c0 = max(
        0,
        c0 - pad,
    )
    r1 = min(
        sim.shape[0],
        r1 + pad + 1,
    )
    c1 = min(
        sim.shape[1],
        c1 + pad + 1,
    )

    real_resized = resize_to_match(
        real,
        sim.shape,
    )

    return (
        sim[r0:r1, c0:c1],
        real_resized[r0:r1, c0:c1],
    )


# =============================================================================
# METRICS
# =============================================================================

def compute_distance(
    sim: np.ndarray,
    real: np.ndarray,
    metric: str = "ssim",
) -> float:
    """
    Legacy normalized distance.

    Lower is always better.

    This is useful for shape/structure comparison, but because each image is
    independently normalized it is not ideal for calibrating a global
    attenuation scale in the new log-domain model.
    """
    sim, real = auto_crop_to_content(
        sim,
        real,
    )

    sim_n = normalize(sim)
    real_n = normalize(real)

    if metric == "wasserstein":
        return float(
            wasserstein_distance(
                real_n.ravel(),
                sim_n.ravel(),
            )
        )

    if metric == "ncc":
        s = sim_n - sim_n.mean()
        r = real_n - real_n.mean()

        denom = np.sqrt(
            np.sum(s ** 2)
            * np.sum(r ** 2)
        )

        if denom == 0:
            return 1.0

        ncc = (
            np.sum(s * r)
            / denom
        )

        return float(
            1.0 - ncc
        )

    if metric == "ssim":
        from skimage.metrics import (
            structural_similarity,
        )

        score = structural_similarity(
            real_n,
            sim_n,
            data_range=1.0,
        )

        return float(
            1.0 - score
        )

    if metric == "psnr":
        from skimage.metrics import (
            peak_signal_noise_ratio,
        )

        score = peak_signal_noise_ratio(
            real_n,
            sim_n,
            data_range=1.0,
        )

        return -float(score)

    if metric == "mi":
        from sklearn.metrics import (
            normalized_mutual_info_score,
        )

        bins = 64
        edges = np.linspace(
            0,
            1,
            bins,
        )

        r_binned = np.digitize(
            real_n.ravel(),
            edges,
        )

        s_binned = np.digitize(
            sim_n.ravel(),
            edges,
        )

        nmi = normalized_mutual_info_score(
            r_binned,
            s_binned,
        )

        return float(
            1.0 - nmi
        )

    raise ValueError(
        f"unknown metric: {metric}"
    )


def _fixed_scale_images(
    sim: np.ndarray,
    real: np.ndarray,
):
    """
    Put simulated log-domain DRR and real 16-bit PXR on a common fixed scale.

    The DRR is expected to have been calibrated so useful values are near
    [0,1]. Values outside that range are clipped for fixed-range metrics.

    The real PXR is converted from 16-bit to [0,1].
    """
    sim_n = np.clip(
        np.asarray(
            sim,
            dtype=np.float64,
        ),
        0.0,
        1.0,
    )

    real_arr = np.asarray(
        real,
        dtype=np.float64,
    )

    if real_arr.max() > 1.0:
        real_n = np.clip(
            real_arr / 65535.0,
            0.0,
            1.0,
        )
    else:
        real_n = np.clip(
            real_arr,
            0.0,
            1.0,
        )

    return sim_n, real_n


def compute_sweep_metric(
    sim: np.ndarray,
    real: np.ndarray,
    metric: str,
    drr_mask: np.ndarray = None,
    pxr_mask: np.ndarray = None,
) -> float:
    """
    Fixed-scale metric used for attenuation sweeps.

    With masks:
      - Wasserstein compares foreground distributions independently.
      - SSIM/NCC/PSNR/MI use the intersection because they require spatial
        correspondence.

    Without masks:
      - the full images are compared.

    Return convention:
      - Wasserstein: lower is better
      - SSIM/NCC/PSNR/MI: higher is better
    """
    if sim.shape != real.shape:
        raise ValueError(
            "Sweep metric requires matching shapes, got "
            f"DRR={sim.shape}, PXR={real.shape}."
        )

    if (
        (drr_mask is None)
        != (pxr_mask is None)
    ):
        raise ValueError(
            "Provide both drr_mask and pxr_mask, or neither."
        )

    sim_n, real_n = (
        _fixed_scale_images(
            sim,
            real,
        )
    )

    if drr_mask is not None:
        if drr_mask.shape != sim.shape:
            raise ValueError(
                f"DRR mask shape {drr_mask.shape} "
                f"does not match DRR shape {sim.shape}"
            )

        if pxr_mask.shape != real.shape:
            raise ValueError(
                f"PXR mask shape {pxr_mask.shape} "
                f"does not match PXR shape {real.shape}"
            )

        drr_fg = (
            drr_mask > 0
        )

        pxr_fg = (
            pxr_mask > 0
        )

        if not drr_fg.any():
            raise ValueError(
                "DRR mask contains no foreground pixels."
            )

        if not pxr_fg.any():
            raise ValueError(
                "PXR mask contains no foreground pixels."
            )

        if metric == "wasserstein":
            return float(
                wasserstein_distance(
                    real_n[pxr_fg],
                    sim_n[drr_fg],
                )
            )

        common = (
            drr_fg
            & pxr_fg
        )

        if not common.any():
            raise ValueError(
                "DRR and PXR masks have no overlapping foreground pixels."
            )

        if metric == "ncc":
            s = sim_n[common]
            r = real_n[common]

            s = (
                s - s.mean()
            )
            r = (
                r - r.mean()
            )

            denom = np.sqrt(
                np.sum(s ** 2)
                * np.sum(r ** 2)
            )

            return (
                0.0
                if denom == 0
                else float(
                    np.sum(s * r)
                    / denom
                )
            )

        if metric == "psnr":
            from skimage.metrics import (
                peak_signal_noise_ratio,
            )

            return float(
                peak_signal_noise_ratio(
                    real_n[common],
                    sim_n[common],
                    data_range=1.0,
                )
            )

        if metric == "ssim":
            from skimage.metrics import (
                structural_similarity,
            )

            _, ssim_map = (
                structural_similarity(
                    real_n,
                    sim_n,
                    data_range=1.0,
                    full=True,
                )
            )

            return float(
                ssim_map[common]
                .mean()
            )

        if metric == "mi":
            from sklearn.metrics import (
                normalized_mutual_info_score,
            )

            bins = 64
            edges = np.linspace(
                0,
                1,
                bins,
            )

            r_binned = np.digitize(
                real_n[common],
                edges,
            )

            s_binned = np.digitize(
                sim_n[common],
                edges,
            )

            return float(
                normalized_mutual_info_score(
                    r_binned,
                    s_binned,
                )
            )

        raise ValueError(
            f"unknown metric: {metric}"
        )

    # No masks.
    if metric == "wasserstein":
        return float(
            wasserstein_distance(
                real_n.ravel(),
                sim_n.ravel(),
            )
        )

    if metric == "ncc":
        s = (
            sim_n
            - sim_n.mean()
        )
        r = (
            real_n
            - real_n.mean()
        )

        denom = np.sqrt(
            np.sum(s ** 2)
            * np.sum(r ** 2)
        )

        return (
            0.0
            if denom == 0
            else float(
                np.sum(s * r)
                / denom
            )
        )

    if metric == "psnr":
        from skimage.metrics import (
            peak_signal_noise_ratio,
        )

        return float(
            peak_signal_noise_ratio(
                real_n,
                sim_n,
                data_range=1.0,
            )
        )

    if metric == "ssim":
        from skimage.metrics import (
            structural_similarity,
        )

        return float(
            structural_similarity(
                real_n,
                sim_n,
                data_range=1.0,
            )
        )

    if metric == "mi":
        from sklearn.metrics import (
            normalized_mutual_info_score,
        )

        bins = 64
        edges = np.linspace(
            0,
            1,
            bins,
        )

        r_binned = np.digitize(
            real_n.ravel(),
            edges,
        )

        s_binned = np.digitize(
            sim_n.ravel(),
            edges,
        )

        return float(
            normalized_mutual_info_score(
                r_binned,
                s_binned,
            )
        )

    raise ValueError(
        f"unknown metric: {metric}"
    )


def _metric_loss(
    score: float,
    metric: str,
) -> float:
    """
    Convert a sweep score to a minimization loss.
    """
    if metric == "wasserstein":
        return float(score)

    return -float(score)


# =============================================================================
# SAVING
# =============================================================================

def save_drr_16bit(
    path,
    drr_float,
    pixel_spacing_mm=None,
):
    """
    Save a fixed-scale 16-bit calibration DRR.

    This preserves the same absolute [0,1] interpretation used by the
    attenuation sweep. Values outside [0,1] are clipped.
    """
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    scaled = (
        np.clip(
            np.asarray(
                drr_float,
                dtype=np.float32,
            ),
            0.0,
            1.0,
        )
        * 65535.0
    )

    kwargs = {
        "photometric": "minisblack",
    }

    if pixel_spacing_mm is not None:
        resolution = (
            10.0
            / float(pixel_spacing_mm)
        )

        kwargs.update(
            resolution=(
                resolution,
                resolution,
            ),
            resolutionunit="CENTIMETER",
        )

    tiff.imwrite(
        path,
        scaled.astype(np.uint16),
        **kwargs,
    )


def save_drr_float32(
    path,
    drr_float,
    pixel_spacing_mm=None,
):
    """
    Save the quantitative log-domain line-integral values without clipping.
    """
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    kwargs = {
        "photometric": "minisblack",
    }

    if pixel_spacing_mm is not None:
        resolution = (
            10.0
            / float(pixel_spacing_mm)
        )

        kwargs.update(
            resolution=(
                resolution,
                resolution,
            ),
            resolutionunit="CENTIMETER",
        )

    tiff.imwrite(
        path,
        np.asarray(
            drr_float,
            dtype=np.float32,
        ),
        **kwargs,
    )


# =============================================================================
# CALIBRATION
# =============================================================================

def _prepare_registered_drr(
    vol,
    geometry,
    attenuation_scale,
    use_gpu,
    row_chunk,
    sample_chunk,
    object_transform=None,
    rotation_xyz=None,
    true_translation_xyz=None,
    crop_to_pxr=False,
    real_img=None,
):
    drr = generate_drr(
        vol,
        geometry,
        attenuation_scale=attenuation_scale,
        object_transform=object_transform,
        rotation_xyz=rotation_xyz,
        true_translation_xyz=true_translation_xyz,
        use_gpu=use_gpu,
        row_chunk=row_chunk,
        sample_chunk=sample_chunk,
        verbose=False,
    )

    if crop_to_pxr:
        if real_img is None:
            raise ValueError(
                "crop_to_pxr=True requires real_img."
            )

        drr = crop_top_to_match_pxr(
            drr,
            real_img.shape,
        )

    return drr


def generate_fixed(
    vol: np.ndarray,
    real_img: np.ndarray,
    geometry: ConeBeamGeometry,
    attenuation_scale: float,
    metric: str = "ssim",
    use_gpu: bool = False,
    row_chunk: int = DEFAULT_ROW_CHUNK,
    sample_chunk: int = DEFAULT_SAMPLE_CHUNK,
    crop_to_pxr: bool = False,
    transform=None,
    drr_mask=None,
    pxr_mask=None,
) -> dict:
    drr = _prepare_registered_drr(
        vol,
        geometry,
        attenuation_scale,
        use_gpu,
        row_chunk,
        sample_chunk,
        object_transform=transform,
        crop_to_pxr=crop_to_pxr,
        real_img=real_img,
    )

    if drr.shape != real_img.shape:
        real_for_score = resize_to_match(
            real_img,
            drr.shape,
        )
    else:
        real_for_score = real_img

    if drr_mask is not None:
        if drr_mask.shape != drr.shape:
            raise ValueError(
                "For fixed metric mode, drr_mask must already match the DRR shape."
            )
        if pxr_mask.shape != real_for_score.shape:
            raise ValueError(
                "For fixed metric mode, pxr_mask must match the PXR comparison shape."
            )

    score = compute_sweep_metric(
        drr,
        real_for_score,
        metric=metric,
        drr_mask=drr_mask,
        pxr_mask=pxr_mask,
    )

    return {
        "attenuation_scale": float(attenuation_scale),
        "score": float(score),
        "metric": metric,
        "drr": drr,
    }


def calibrate(
    vol: np.ndarray,
    real_img: np.ndarray,
    geometry: ConeBeamGeometry,
    metric: str = "ssim",
    bounds=(0.002, 0.2),
    use_gpu: bool = False,
    row_chunk: int = DEFAULT_ROW_CHUNK,
    sample_chunk: int = DEFAULT_SAMPLE_CHUNK,
    crop_to_pxr: bool = False,
    transform=None,
    drr_mask=None,
    pxr_mask=None,
) -> dict:
    """
    Optimize attenuation_scale using FIXED-scale metrics.

    This is different from the old independent min-max normalization approach,
    so attenuation_scale remains identifiable in the log-domain model.
    """
    history = []

    def objective(attenuation_scale):
        result = generate_fixed(
            vol,
            real_img,
            geometry,
            attenuation_scale=float(attenuation_scale),
            metric=metric,
            use_gpu=use_gpu,
            row_chunk=row_chunk,
            sample_chunk=sample_chunk,
            crop_to_pxr=crop_to_pxr,
            transform=transform,
            drr_mask=drr_mask,
            pxr_mask=pxr_mask,
        )

        score = float(
            result["score"]
        )

        history.append(
            (
                float(attenuation_scale),
                score,
            )
        )

        return _metric_loss(
            score,
            metric,
        )

    result = minimize_scalar(
        objective,
        bounds=bounds,
        method="bounded",
        options={
            "xatol": 1e-5,
        },
    )

    best = generate_fixed(
        vol,
        real_img,
        geometry,
        attenuation_scale=float(result.x),
        metric=metric,
        use_gpu=use_gpu,
        row_chunk=row_chunk,
        sample_chunk=sample_chunk,
        crop_to_pxr=crop_to_pxr,
        transform=transform,
        drr_mask=drr_mask,
        pxr_mask=pxr_mask,
    )

    best["history"] = history
    best["optimizer_loss"] = float(
        result.fun
    )

    return best


def sweep(
    vol,
    geometry,
    values,
    out_dir,
    real_img=None,
    metric="wasserstein",
    save_all=False,
    save_float=False,
    use_gpu=False,
    row_chunk=DEFAULT_ROW_CHUNK,
    sample_chunk=DEFAULT_SAMPLE_CHUNK,
    crop_to_pxr=False,
    transform=None,
    drr_mask=None,
    pxr_mask=None,
):
    """
    Generate a NEW DRR from the CT for every attenuation value using the same
    geometry and optional manual-registration transform.

    When a PXR is supplied, the sweep writes the same multi-metric table used
    for attenuation selection. Wasserstein is reported as a similarity score:

        wasserstein_score = 1 - wasserstein_distance

    Therefore every score column in attenuation_metrics.csv is higher=better.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        (drr_mask is None)
        != (pxr_mask is None)
    ):
        raise ValueError(
            "Provide both --drr-mask and --pxr-mask, or neither."
        )

    results = []

    for attenuation_scale in values:
        print()
        print(
            "======================================"
        )
        print(
            "Generating attenuation_scale = "
            f"{attenuation_scale:.6f}"
        )
        print(
            "======================================"
        )

        # IMPORTANT: this renders a fresh DRR from the CT for every value.
        drr = _prepare_registered_drr(
            vol,
            geometry,
            float(attenuation_scale),
            use_gpu,
            row_chunk,
            sample_chunk,
            object_transform=transform,
            crop_to_pxr=crop_to_pxr,
            real_img=real_img,
        )

        row = {
            "attenuation": float(
                attenuation_scale
            ),
            "wasserstein_score": None,
            "ssim": None,
            "ncc": None,
            "psnr": None,
        }

        if real_img is not None:
            if drr.shape != real_img.shape:
                real_for_score = resize_to_match(
                    real_img,
                    drr.shape,
                )
            else:
                real_for_score = real_img

            wasserstein_distance_value = compute_sweep_metric(
                drr,
                real_for_score,
                metric="wasserstein",
                drr_mask=drr_mask,
                pxr_mask=pxr_mask,
            )

            row["wasserstein_score"] = float(
                1.0 - wasserstein_distance_value
            )

            row["ssim"] = float(
                compute_sweep_metric(
                    drr,
                    real_for_score,
                    metric="ssim",
                    drr_mask=drr_mask,
                    pxr_mask=pxr_mask,
                )
            )

            row["ncc"] = float(
                compute_sweep_metric(
                    drr,
                    real_for_score,
                    metric="ncc",
                    drr_mask=drr_mask,
                    pxr_mask=pxr_mask,
                )
            )

            row["psnr"] = float(
                compute_sweep_metric(
                    drr,
                    real_for_score,
                    metric="psnr",
                    drr_mask=drr_mask,
                    pxr_mask=pxr_mask,
                )
            )

            label = (
                "masked"
                if drr_mask is not None
                else "full-image"
            )

            print(
                f"{label} Wasserstein score (1 - distance): "
                f"{row['wasserstein_score']:.6f}  "

            )
            print(
                f"{label} SSIM: "
                f"{row['ssim']:.6f}  "
            
            )
            print(
                f"{label} NCC: "
                f"{row['ncc']:.6f}  "
            
            )
            print(
                f"{label} PSNR: "
                f"{row['psnr']:.3f} dB  "
            
            )

        results.append(row)

        if save_all:
            save_drr_16bit(
                out_dir
                / f"drr_atten_{attenuation_scale:.4f}.tif",
                drr,
                geometry.detector_pixel_spacing_mm,
            )

        if save_float:
            save_drr_float32(
                out_dir
                / f"drr_atten_{attenuation_scale:.4f}_float32.tif",
                drr,
                geometry.detector_pixel_spacing_mm,
            )

        # Avoid retaining full 3072 DRRs during a sweep.
        del drr

    csv_path = (
        out_dir
        / "attenuation_metrics.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
    ) as handle:
        writer = csv.writer(
            handle
        )

        writer.writerow(
            [
                "attenuation",
                "wasserstein_score",
                "ssim",
                "ncc",
                "psnr",
            ]
        )

        for row in results:
            writer.writerow(
                [
                    row["attenuation"],
                    row["wasserstein_score"],
                    row["ssim"],
                    row["ncc"],
                    row["psnr"],
                ]
            )

    print(
        f"\nMetrics written to: {csv_path}"
    )

    # -----------------------------------------------------------------
    # Line plot of attenuation sweep metrics
    # -----------------------------------------------------------------
    plot_rows = [
        row
        for row in results
        if row["wasserstein_score"] is not None
    ]

    if plot_rows:
        attenuation_values = [
            row["attenuation"]
            for row in plot_rows
        ]
        wasserstein_values = [
            row["wasserstein_score"]
            for row in plot_rows
        ]
        ssim_values = [
            row["ssim"]
            for row in plot_rows
        ]
        ncc_values = [
            row["ncc"]
            for row in plot_rows
        ]
        psnr_values = [
            row["psnr"]
            for row in plot_rows
        ]

        fig, ax1 = plt.subplots(
            figsize=(9, 5.5)
        )

        line_w, = ax1.plot(
            attenuation_values,
            wasserstein_values,
            marker="o",
            label="Wasserstein score (1 - distance)",
        )
        line_s, = ax1.plot(
            attenuation_values,
            ssim_values,
            marker="o",
            label="SSIM",
        )
        line_n, = ax1.plot(
            attenuation_values,
            ncc_values,
            marker="o",
            label="NCC",
        )

        ax1.set_xlabel(
            "Attenuation scale"
        )
        ax1.set_ylabel(
            "Similarity score"
        )
        ax1.set_ylim(
            0.0,
            1.02,
        )
        ax1.grid(
            True,
            alpha=0.25,
        )

        # PSNR is in dB and therefore needs a separate y-axis.
        ax2 = ax1.twinx()
        line_p, = ax2.plot(
            attenuation_values,
            psnr_values,
            marker="o",
            linestyle="--",
            label="PSNR",
        )
        ax2.set_ylabel(
            "PSNR (dB)"
        )

        # Mark the best point for every metric.
        for values, line in [
            (wasserstein_values, line_w),
            (ssim_values, line_s),
            (ncc_values, line_n),
        ]:
            best_idx = int(
                np.argmax(values)
            )
            ax1.scatter(
                attenuation_values[best_idx],
                values[best_idx],
                s=70,
                facecolors="none",
                edgecolors=line.get_color(),
                linewidths=1.8,
                zorder=5,
            )

        best_psnr_idx = int(
            np.argmax(psnr_values)
        )
        ax2.scatter(
            attenuation_values[best_psnr_idx],
            psnr_values[best_psnr_idx],
            s=70,
            facecolors="none",
            edgecolors=line_p.get_color(),
            linewidths=1.8,
            zorder=5,
        )

        lines = [
            line_w,
            line_s,
            line_n,
            line_p,
        ]
        ax1.legend(
            lines,
            [
                line.get_label()
                for line in lines
            ],
            loc="best",
        )

        ax1.set_title(
            "Attenuation coefficient sweep"
        )

        fig.tight_layout()

        plot_path = (
            out_dir
            / "attenuation_metrics.png"
        )

        fig.savefig(
            plot_path,
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

        print(
            f"Metrics plot written to: {plot_path}"
        )

    scored = [
        row
        for row in results
        if row["wasserstein_score"] is not None
    ]

    if scored:
        print()
        print(
            "======================================"
        )
        print(
            "BEST ATTENUATION BY METRIC"
        )
        print(
            "======================================"
        )

        metric_labels = [
            ("wasserstein_score", "Wasserstein score"),
            ("ssim", "SSIM"),
            ("ncc", "NCC"),
            ("psnr", "PSNR"),
        ]

        for key, label in metric_labels:
            best = max(
                scored,
                key=lambda row: row[key],
            )

            if key == "psnr":
                score_text = f"{best[key]:.3f} dB"
            else:
                score_text = f"{best[key]:.6f}"

            print(
                f"{label:<18}: "
                f"attenuation={best['attenuation']:.6f}  "
                f"score={score_text}"
            )


    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate and calibrate DRRs using the same "
            "log-domain cone-beam renderer as manual registration."
        )
    )

    parser.add_argument(
        "--xr",
        required=True,
        dest="xr_path",
    )

    parser.add_argument(
        "--pxr",
        required=False,
        default=None,
        dest="pxr_path",
    )

    parser.add_argument(
        "--view",
        default=None,
        choices=[
            "Face",
            "Profil",
        ],
    )

    parser.add_argument(
        "--transform",
        default=None,
        help=(
            "Optional path to the 4x4 Fiji transform "
            "saved by manual registration."
        ),
    )

    parser.add_argument(
        "--drr-mask",
        default=None,
    )

    parser.add_argument(
        "--pxr-mask",
        default=None,
    )

    parser.add_argument(
        "--detector-size",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--voxel-spacing-mm",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--voxel-spacing-z-mm",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--spacing-zyx",
        default=None,
        help=(
            "Explicit CT spacing as z,y,x in mm, "
            "e.g. 0.4,0.5820315,0.5820315"
        ),
    )

    parser.add_argument(
        "--sid-mm",
        type=float,
        default=DEFAULT_SID_MM,
    )

    parser.add_argument(
        "--spd-mm",
        type=float,
        default=DEFAULT_SPD_MM,
    )

    parser.add_argument(
        "--offset-v-mm",
        type=float,
        default=DEFAULT_OFFSET_V_MM,
    )

    parser.add_argument(
        "--offset-u-mm",
        type=float,
        default=DEFAULT_OFFSET_U_MM,
    )

    parser.add_argument(
        "--beam-axis",
        type=int,
        default=DEFAULT_BEAM_AXIS,
        choices=[
            0,
            1,
            2,
        ],
        help=(
            "Legacy calibration default is 1. "
            "Manual registration explicitly uses 0."
        ),
    )

    parser.add_argument(
        "--attenuation-scale",
        type=float,
        default=DEFAULT_ATTENUATION_SCALE,
    )

    parser.add_argument(
        "--fixed-attenuation",
        default=None,
        help=(
            "One value or comma-separated values. "
            "Example: 0.010,0.015,0.020"
        ),
    )

    parser.add_argument(
        "--atten-min",
        type=float,
        default=0.002,
    )

    parser.add_argument(
        "--atten-max",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--fit-attenuation",
        action="store_true",
    )

    parser.add_argument(
        "--metric",
        default="ssim",
        choices=[
            "ssim",
            "wasserstein",
            "ncc",
            "psnr",
            "mi",
        ],
    )

    parser.add_argument(
        "--sweep",
        default=None,
        help=(
            "Comma-separated attenuation values, "
            "e.g. 0.005,0.010,0.015,0.020"
        ),
    )

    parser.add_argument(
        "--sweep-out-dir",
        default="attenuation_sweep",
    )

    parser.add_argument(
        "--save-all-sweep",
        action="store_true",
    )

    parser.add_argument(
        "--save-float-sweep",
        action="store_true",
    )

    parser.add_argument(
        "--crop-to-pxr",
        choices=[
            "yes",
            "no",
        ],
        default="no",
    )

    parser.add_argument(
        "--gpu",
        action="store_true",
    )

    parser.add_argument(
        "--row-chunk",
        type=int,
        default=DEFAULT_ROW_CHUNK,
    )

    parser.add_argument(
        "--sample-chunk",
        type=int,
        default=DEFAULT_SAMPLE_CHUNK,
    )

    parser.add_argument(
        "--out-drr",
        default=None,
    )

    parser.add_argument(
        "--out-float",
        default=None,
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------
    # Spacing
    # -----------------------------------------------------------------

    if args.spacing_zyx is not None:
        spacing_zyx = tuple(
            float(value)
            for value in args.spacing_zyx.split(",")
        )

        if len(spacing_zyx) != 3:
            parser.error(
                "--spacing-zyx must contain exactly z,y,x"
            )
    else:
        spacing_zyx = None

    geometry = make_geometry(
        detector_size=args.detector_size,
        voxel_spacing_mm=args.voxel_spacing_mm,
        voxel_spacing_z_mm=args.voxel_spacing_z_mm,
        spacing_zyx=spacing_zyx,
        sid_mm=args.sid_mm,
        spd_mm=args.spd_mm,
        offset_v_mm=args.offset_v_mm,
        offset_u_mm=args.offset_u_mm,
        beam_axis=args.beam_axis,
    )

    # -----------------------------------------------------------------
    # Load CT + optional transform
    # -----------------------------------------------------------------

    vol = load_volume(
        args.xr_path
    )

    print(
        "Loaded CT shape:",
        vol.shape,
    )

    print(
        "Voxel spacing used:",
        geometry.voxel_spacing_mm,
    )

    print(
        "Physical CT size mm:",
        tuple(
            vol.shape[axis]
            * geometry.spacing_for_axis(axis)
            for axis in range(3)
        ),
    )

    transform = None

    if args.transform is not None:
        transform = np.loadtxt(
            args.transform
        )

        if transform.shape != (4, 4):
            raise ValueError(
                f"Expected 4x4 transform, got {transform.shape}"
            )

        print(
            "Loaded manual Fiji transform:"
        )
        print(
            transform
        )

    # -----------------------------------------------------------------
    # Load PXR
    # -----------------------------------------------------------------

    real_img = None

    if args.pxr_path is not None:
        pxr_path = args.pxr_path

        if args.view is not None:
            pxr_path = find_view(
                pxr_path,
                args.view,
            )

            print(
                f"Selected view file: {pxr_path}"
            )

        real_img = load_image(
            pxr_path
        )

        real_img, _ = (
            strip_saturated_band(
                real_img
            )
        )

        print(
            "PXR shape:",
            real_img.shape,
        )

    # -----------------------------------------------------------------
    # Masks
    # -----------------------------------------------------------------

    drr_mask = None
    pxr_mask = None

    if args.drr_mask is not None:
        drr_mask = tiff.imread(
            args.drr_mask
        )

        print(
            f"Loaded DRR mask: {args.drr_mask} "
            f"shape={drr_mask.shape}"
        )

    if args.pxr_mask is not None:
        pxr_mask = tiff.imread(
            args.pxr_mask
        )

        print(
            f"Loaded PXR mask: {args.pxr_mask} "
            f"shape={pxr_mask.shape}"
        )

    if (
        (drr_mask is None)
        != (pxr_mask is None)
    ):
        parser.error(
            "Use both --drr-mask and --pxr-mask together."
        )

    # -----------------------------------------------------------------
    # Execution settings
    # -----------------------------------------------------------------

    use_gpu = bool(
        args.gpu
        and CUPY_AVAILABLE
    )

    if args.gpu and not CUPY_AVAILABLE:
        print(
            "CuPy is unavailable. Falling back to CPU."
        )

    crop_to_pxr = (
        args.crop_to_pxr
        == "yes"
    )

    print(
        "Geometry: "
        f"SID={geometry.sid_mm:g} mm, "
        f"SPD={geometry.spd_mm:g} mm, "
        f"pixel_spacing="
        f"{geometry.detector_pixel_spacing_mm:.6f} mm, "
        f"detector={geometry.detector_shape}, "
        f"beam_axis={geometry.beam_axis}"
    )

    print(
        "Intensity domain: log / attenuation "
        "(no 1-exp(-A))"
    )

    # -----------------------------------------------------------------
    # Sweep mode
    # -----------------------------------------------------------------

    if args.sweep is not None:
        values = [
            float(value)
            for value in args.sweep.split(",")
        ]

        sweep(
            vol,
            geometry,
            values,
            args.sweep_out_dir,
            real_img=real_img,
            metric=args.metric,
            save_all=args.save_all_sweep,
            save_float=args.save_float_sweep,
            use_gpu=use_gpu,
            row_chunk=args.row_chunk,
            sample_chunk=args.sample_chunk,
            crop_to_pxr=crop_to_pxr,
            transform=transform,
            drr_mask=drr_mask,
            pxr_mask=pxr_mask,
        )

        return

    # -----------------------------------------------------------------
    # Non-sweep modes require a PXR only when scoring/calibrating.
    # -----------------------------------------------------------------

    if (
        args.fit_attenuation
        or args.fixed_attenuation is not None
    ) and real_img is None:
        parser.error(
            "attenuation calibration requires --pxr"
        )

    result = None

    # -----------------------------------------------------------------
    # Fixed attenuation values
    # -----------------------------------------------------------------

    if args.fixed_attenuation is not None:
        values = [
            float(value)
            for value in args.fixed_attenuation.split(",")
        ]

        fixed_results = []

        for value in values:
            current = generate_fixed(
                vol,
                real_img,
                geometry,
                attenuation_scale=value,
                metric=args.metric,
                use_gpu=use_gpu,
                row_chunk=args.row_chunk,
                sample_chunk=args.sample_chunk,
                crop_to_pxr=crop_to_pxr,
                transform=transform,
                drr_mask=drr_mask,
                pxr_mask=pxr_mask,
            )

            fixed_results.append(
                current
            )

            print(
                f"attenuation_scale: "
                f"{current['attenuation_scale']:.6f}"
            )

            if args.metric == "wasserstein":
                print(
                    f"{args.metric}: "
                    f"{current['score']:.6f} "
                    "(lower is better)"
                )
            else:
                print(
                    f"{args.metric}: "
                    f"{current['score']:.6f} "
    
                )

        if args.metric == "wasserstein":
            result = min(
                fixed_results,
                key=lambda item: item["score"],
            )
        else:
            result = max(
                fixed_results,
                key=lambda item: item["score"],
            )

    # -----------------------------------------------------------------
    # Optimizer
    # -----------------------------------------------------------------

    elif args.fit_attenuation:
        result = calibrate(
            vol,
            real_img,
            geometry,
            metric=args.metric,
            bounds=(
                args.atten_min,
                args.atten_max,
            ),
            use_gpu=use_gpu,
            row_chunk=args.row_chunk,
            sample_chunk=args.sample_chunk,
            crop_to_pxr=crop_to_pxr,
            transform=transform,
            drr_mask=drr_mask,
            pxr_mask=pxr_mask,
        )

        print(
            "\nBest attenuation_scale: "
            f"{result['attenuation_scale']:.6f}"
        )

        print(
            f"Best {args.metric}: "
            f"{result['score']:.6f}"
        )

        print(
            f"Iterations: "
            f"{len(result['history'])}"
        )

    # -----------------------------------------------------------------
    # Plain DRR generation
    # -----------------------------------------------------------------

    else:
        drr = _prepare_registered_drr(
            vol,
            geometry,
            args.attenuation_scale,
            use_gpu,
            args.row_chunk,
            args.sample_chunk,
            object_transform=transform,
            crop_to_pxr=crop_to_pxr,
            real_img=real_img,
        )

        result = {
            "attenuation_scale": float(
                args.attenuation_scale
            ),
            "drr": drr,
        }

        if real_img is not None:
            if drr.shape != real_img.shape:
                real_for_score = resize_to_match(
                    real_img,
                    drr.shape,
                )
            else:
                real_for_score = real_img

            result["score"] = (
                compute_sweep_metric(
                    drr,
                    real_for_score,
                    metric=args.metric,
                    drr_mask=drr_mask,
                    pxr_mask=pxr_mask,
                )
            )

            print(
                f"{args.metric}: "
                f"{result['score']:.6f}"
            )

    # -----------------------------------------------------------------
    # Save selected/final DRR
    # -----------------------------------------------------------------

    if args.out_drr is None:
        out_drr = (
            Path("calibration_outputs")
            / "calibrated_drr.tif"
        )
    else:
        out_drr = Path(
            args.out_drr
        )

    save_drr_16bit(
        out_drr,
        result["drr"],
        geometry.detector_pixel_spacing_mm,
    )

    print(
        f"16-bit DRR written to {out_drr}"
    )

    if args.out_float is not None:
        save_drr_float32(
            args.out_float,
            result["drr"],
            geometry.detector_pixel_spacing_mm,
        )

        print(
            f"Float32 DRR written to {args.out_float}"
        )


if __name__ == "__main__":
    main()
