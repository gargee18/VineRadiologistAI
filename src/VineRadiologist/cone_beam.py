"""
Cone-beam (point-source, divergent-ray) DRR projection, as opposed to the
parallel projection in projection.py's generate_drr.

Real X-ray sources are point sources emitting diverging rays, not parallel
rays. This produces magnification (objects closer to the source appear
larger) and geometric blur that a parallel projection cannot reproduce.
This module implements a straightforward ray-casting cone-beam projector.
sid_mm and spd_mm have NO default, every caller must pass the real measured
(or DICOM-confirmed) distances for their specific acquisition setup, don't
reuse a number from a different specimen/device/session.
"""

from dataclasses import dataclass
import numpy as np
from scipy.ndimage import map_coordinates


@dataclass
class ConeBeamGeometry:
    # NO DEFAULTS for sid_mm/spd_mm on purpose. These used to default to
    # 1500.0/1349.0 (stale DICOM-derived values from an earlier session,
    # a different acquisition setup than your measured d1/d2 for CEP_378A).
    # scripts/compare_parallel_vs_conebeam.py was silently relying on those
    # stale defaults, real distances were never explicitly passed there.
    # Making these required forces every caller to explicitly state the
    # real measured distances for whatever specimen/setup they're using,
    # instead of silently falling back to a number from a different context.
    sid_mm: float             # source-to-detector distance, MUST be measured/confirmed
    spd_mm: float             # source-to-patient(volume-center) distance, MUST be measured/confirmed
    detector_pixel_spacing_mm: float = 0.148  # real value from your radiograph DICOMs

    # Per-axis voxel spacing (mm), indexed to match vol.shape, i.e.
    # voxel_spacing_mm[i] is the real spacing along array axis i.
    # Pass a single float for the old isotropic behavior (broadcast to all
    # 3 axes), or a 3-tuple (spacing_axis0, spacing_axis1, spacing_axis2)
    # for real anisotropic volumes.
    #
    # CONFIRMED CASE (2022-2026 CEP RegistrationHighRes / Fijiyama pipeline,
    # e.g. CEP_378A_2026_XR.tif, shape (1908, 512, 512)):
    #   axis 0 (1908 slices) = Z, voxel depth = 0.4mm (from TIFF metadata dialog)
    #   axis 1, axis 2 (512x512) = in-plane XY = 0.6171876mm each (from
    #   the TIFF's own XResolution/YResolution tags, both equal)
    # This is a DIFFERENT dataset from the older 2019 Vitimage/CEP011-022
    # set, where 0.7224mm isotropic was confirmed separately, don't reuse
    # that number here, it does not apply to this pipeline's output.
    voxel_spacing_mm: object = 0.7224  # float (isotropic) or (z, y, x) tuple
    detector_shape: tuple = (512, 512)  # (rows, cols) of the output image

    # Vertical/horizontal centering offset (mm), in the DETECTOR plane,
    # not the beam axis. Physically: the code assumes the source/detector
    # were aimed at the volume's exact geometric center along both
    # detector-plane axes. Real acquisitions aren't always perfectly
    # centered (e.g. scanner aimed lower toward the pot, cutting off
    # canopy at the top), these offsets let you shift the effective
    # aim point without touching sid/spd (which only control distance/
    # magnification, not vertical framing).
    # offset_v_mm: shifts along other_axes[0] (the first non-beam axis,
    # typically the "vertical" axis in the rendered image, e.g. Z in the
    # CEP pipeline). offset_u_mm: shifts along other_axes[1].
    # Sign convention isn't derived from physical geometry here, this is
    # empirical: try a value, check if it moved the right direction, flip
    # sign if not.
    offset_v_mm: float = 0.0
    offset_u_mm: float = 0.0

    def spacing_for_axis(self, axis: int) -> float:
        s = self.voxel_spacing_mm
        if isinstance(s, (tuple, list)):
            return s[axis]
        return s  # scalar, same for every axis (old isotropic behavior)


def generate_cone_beam_drr(vol: np.ndarray, geometry: ConeBeamGeometry,
                            attenuation_scale: float = 0.015,
                            beam_axis: int = 1,
                            n_samples: int = None) -> np.ndarray:
    """Cast rays from a point source, through a flat detector grid, sampling
    `vol` along each ray, and apply Beer-Lambert attenuation.

    `beam_axis` is the volume axis the beam travels along (matches the
    `axis` convention used by generate_drr elsewhere in the package: 1).

    Only the portion of each ray that actually intersects the volume's
    bounding box is sampled (not the full source-to-detector distance).
    """
    other_axes = [a for a in range(3) if a != beam_axis]
    n_beam = vol.shape[beam_axis]

    beam_spacing = geometry.spacing_for_axis(beam_axis)
    v_spacing = geometry.spacing_for_axis(other_axes[0])
    u_spacing = geometry.spacing_for_axis(other_axes[1])

    half_extent_mm = (n_beam / 2) * beam_spacing

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

    beam_idx = beam_coord_mm / beam_spacing + n_beam / 2
    v_idx = v_coord_mm / v_spacing + vol.shape[other_axes[0]] / 2 + geometry.offset_v_mm / v_spacing
    u_idx = u_coord_mm / u_spacing + vol.shape[other_axes[1]] / 2 + geometry.offset_u_mm / u_spacing

    coords = [None, None, None]
    coords[beam_axis] = beam_idx
    coords[other_axes[0]] = v_idx
    coords[other_axes[1]] = u_idx

    sampled = map_coordinates(vol, coords, order=1, mode="constant", cval=0.0)

    path_length_mm = abs(t_exit - t_enter) * abs(det_pos - src_pos)
    step_mm = path_length_mm / n_samples
    attenuation_sum = np.sum(sampled, axis=0) * step_mm * attenuation_scale

    return 1.0 - np.exp(-attenuation_sum)