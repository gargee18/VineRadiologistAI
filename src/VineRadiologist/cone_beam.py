"""
Cone-beam (point-source, divergent-ray) DRR projection, as opposed to the
parallel projection in projection.py's generate_drr.

Real X-ray sources are point sources emitting diverging rays, not parallel
rays. This produces magnification (objects closer to the source appear
larger) and geometric blur that a parallel projection cannot reproduce.
This module implements a straightforward ray-casting cone-beam projector
using the real acquisition geometry recovered from your DICOM metadata:
  - DistanceSourceToDetector (SID) ~= 1500mm
  - DistanceSourceToPatient (SPD)  ~= 1349mm

IMPORTANT: this requires the CT volume's real voxel spacing (mm per voxel)
to place it correctly in physical space between source and detector. If you
don't have a confirmed value for these specific CT volumes, the default
below is a placeholder, not a validated number, check your CT DICOM headers
or Java registration pipeline metadata (PixelSpacing/SliceThickness) before
trusting absolute distances/magnification from this function.
"""

from dataclasses import dataclass
import numpy as np
from scipy.ndimage import map_coordinates


@dataclass
class ConeBeamGeometry:
    sid_mm: float = 1500.0   # source-to-detector distance
    spd_mm: float = 1349.0   # source-to-patient(volume-center) distance
    detector_pixel_spacing_mm: float = 0.148  # real value from your radiograph DICOMs
    voxel_spacing_mm: float = 1.0  # PLACEHOLDER: confirm real CT voxel spacing before trusting absolute geometry
    detector_shape: tuple = (512, 512)  # (rows, cols) of the output image


def generate_cone_beam_drr(vol: np.ndarray, geometry: ConeBeamGeometry,
                            attenuation_scale: float = 0.015,
                            beam_axis: int = 1,
                            n_samples: int = None) -> np.ndarray:
    """Cast rays from a point source, through a flat detector grid, sampling
    `vol` along each ray, and apply Beer-Lambert attenuation.

    `beam_axis` is the volume axis the beam travels along (matches the
    `axis` convention used by generate_drr elsewhere in the package: 1).

    Only the portion of each ray that actually intersects the volume's
    bounding box is sampled (not the full source-to-detector distance),
    since most of that path is empty space between source, volume, and
    detector, sampling all of it would waste resolution and, for a small
    spd, could miss the volume slice entirely.
    """
    other_axes = [a for a in range(3) if a != beam_axis]
    n_beam = vol.shape[beam_axis]

    half_extent_mm = (n_beam / 2) * geometry.voxel_spacing_mm

    rows, cols = geometry.detector_shape
    det_v = (np.arange(rows) - rows / 2) * geometry.detector_pixel_spacing_mm
    det_u = (np.arange(cols) - cols / 2) * geometry.detector_pixel_spacing_mm
    det_grid_v, det_grid_u = np.meshgrid(det_v, det_u, indexing="ij")

    src_pos = -geometry.spd_mm
    det_pos = geometry.sid_mm - geometry.spd_mm

    if n_samples is None:
        n_samples = n_beam * 2

    t_enter = (-half_extent_mm - src_pos) / (det_pos - src_pos)
    t_exit = (half_extent_mm - src_pos) / (det_pos - src_pos)
    t = np.linspace(t_enter, t_exit, n_samples).reshape(-1, 1, 1)

    beam_coord_mm = src_pos + t * (det_pos - src_pos)
    beam_coord_mm = np.broadcast_to(beam_coord_mm, (n_samples, rows, cols))

    v_coord_mm = t * det_grid_v[np.newaxis, :, :]
    u_coord_mm = t * det_grid_u[np.newaxis, :, :]

    beam_idx = beam_coord_mm / geometry.voxel_spacing_mm + n_beam / 2
    v_idx = v_coord_mm / geometry.voxel_spacing_mm + vol.shape[other_axes[0]] / 2
    u_idx = u_coord_mm / geometry.voxel_spacing_mm + vol.shape[other_axes[1]] / 2

    coords = [None, None, None]
    coords[beam_axis] = beam_idx
    coords[other_axes[0]] = v_idx
    coords[other_axes[1]] = u_idx

    sampled = map_coordinates(vol, coords, order=1, mode="constant", cval=0.0)

    path_length_mm = abs(t_exit - t_enter) * abs(det_pos - src_pos)
    step_mm = path_length_mm / n_samples
    attenuation_sum = np.sum(sampled, axis=0) * step_mm * attenuation_scale

    return 1.0 - np.exp(-attenuation_sum)