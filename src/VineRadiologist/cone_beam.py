from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates


@dataclass
class ConeBeamGeometry:
    sid_mm: float
    spd_mm: float
    detector_pixel_spacing_mm: float = 0.139
    voxel_spacing_mm: object = 0.7224
    detector_shape: tuple = (3072, 2702)
    offset_v_mm: float = 0.0
    offset_u_mm: float = 0.0

    def spacing_for_axis(self, axis: int) -> float:
        s = self.voxel_spacing_mm
        if isinstance(s, (tuple, list)):
            return s[axis]
        return s


def fiji_rotation_to_numpy(object_transform):
    """
    Fiji uses physical coordinates ordered (x, y, z).
    NumPy volume axes are ordered (z, y, x).

    Only the 3x3 rotation is used, translation intentionally discarded
    (matches the original, working baseline).
    """
    if object_transform is None:
        return None

    transform = np.asarray(object_transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected 4x4 transform, got {transform.shape}")

    r_xyz = transform[:3, :3]

    p = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    return p @ r_xyz @ p


def _half_extent_along_beam_mm(vol, geometry, beam_axis, rotation_np):
    if rotation_np is None:
        n_beam = vol.shape[beam_axis]
        return (n_beam / 2.0) * geometry.spacing_for_axis(beam_axis)

    half_lengths = np.array(
        [
            vol.shape[0] * geometry.spacing_for_axis(0) / 2.0,
            vol.shape[1] * geometry.spacing_for_axis(1) / 2.0,
            vol.shape[2] * geometry.spacing_for_axis(2) / 2.0,
        ],
        dtype=np.float64,
    )

    return float(np.sum(np.abs(rotation_np[beam_axis, :]) * half_lengths))


def _default_n_samples(vol, geometry, beam_axis, half_extent_mm, rotation_np):
    if rotation_np is None:
        return vol.shape[beam_axis] * 2

    min_spacing = min(geometry.spacing_for_axis(i) for i in range(3))
    target_step_mm = min_spacing / 2.0
    return max(2, int(np.ceil((2.0 * half_extent_mm) / target_step_mm)))


def generate_cone_beam_drr(
    vol,
    geometry,
    attenuation_scale=0.015,
    beam_axis=1,
    n_samples=None,
    object_transform=None,
    output_mode="line_integral",
):
    other_axes = [a for a in range(3) if a != beam_axis]

    beam_spacing = geometry.spacing_for_axis(beam_axis)
    v_spacing = geometry.spacing_for_axis(other_axes[0])
    u_spacing = geometry.spacing_for_axis(other_axes[1])

    rotation_np = fiji_rotation_to_numpy(object_transform)
    half_extent_mm = _half_extent_along_beam_mm(
        vol, geometry, beam_axis, rotation_np
    )

    rows, cols = geometry.detector_shape
    det_v = (
        np.arange(rows) - rows / 2
    ) * geometry.detector_pixel_spacing_mm
    det_u = (
        np.arange(cols) - cols / 2
    ) * geometry.detector_pixel_spacing_mm
    det_grid_v, det_grid_u = np.meshgrid(det_v, det_u, indexing="ij")

    src_pos = -geometry.spd_mm
    det_pos = geometry.sid_mm - geometry.spd_mm

    if n_samples is None:
        n_samples = _default_n_samples(
            vol, geometry, beam_axis, half_extent_mm, rotation_np
        )

    t_enter = (-half_extent_mm - src_pos) / (det_pos - src_pos)
    t_exit = (half_extent_mm - src_pos) / (det_pos - src_pos)

    t = np.linspace(t_enter, t_exit, n_samples).reshape(-1, 1, 1)

    beam_coord_mm = src_pos + t * (det_pos - src_pos)
    beam_coord_mm = np.broadcast_to(
        beam_coord_mm, (n_samples, rows, cols)
    )

    v_coord_mm = t * det_grid_v[np.newaxis, :, :]
    u_coord_mm = t * det_grid_u[np.newaxis, :, :]

    physical = [None, None, None]
    physical[beam_axis] = beam_coord_mm
    physical[other_axes[0]] = v_coord_mm + geometry.offset_v_mm
    physical[other_axes[1]] = u_coord_mm + geometry.offset_u_mm

    if rotation_np is not None:
        rotation_inv = rotation_np.T

        p0 = (
            rotation_inv[0, 0] * physical[0]
            + rotation_inv[0, 1] * physical[1]
            + rotation_inv[0, 2] * physical[2]
        )
        p1 = (
            rotation_inv[1, 0] * physical[0]
            + rotation_inv[1, 1] * physical[1]
            + rotation_inv[1, 2] * physical[2]
        )
        p2 = (
            rotation_inv[2, 0] * physical[0]
            + rotation_inv[2, 1] * physical[1]
            + rotation_inv[2, 2] * physical[2]
        )

        physical = [p0, p1, p2]

    coords = [
        physical[0] / geometry.spacing_for_axis(0) + vol.shape[0] / 2,
        physical[1] / geometry.spacing_for_axis(1) + vol.shape[1] / 2,
        physical[2] / geometry.spacing_for_axis(2) + vol.shape[2] / 2,
    ]

    sampled = map_coordinates(
        vol,
        coords,
        order=1,
        mode="constant",
        cval=0.0,
    )

    # FIX: per-pixel ray length, not one shared scalar. Rays to
    # off-center detector pixels are physically LONGER (diagonal) than
    # the central ray, the old code charged every ray the same short
    # central-ray length, undercounting attenuation toward the edges.
    # Confirmed real: at this project's actual geometry (sid=1230,
    # spd=800, ~213mm detector half-width), corner rays are ~3% longer
    # than the center, not negligible.
    dt = (t_exit - t_enter) / n_samples
    full_ray_length_mm = np.sqrt(
        (det_pos - src_pos) ** 2 + det_grid_v ** 2 + det_grid_u ** 2
    )
    step_mm_per_pixel = dt * full_ray_length_mm

    attenuation_sum = (
        np.sum(sampled, axis=0) * step_mm_per_pixel * attenuation_scale
    )

    if output_mode == "line_integral":
        # D = integral(mu dl)
        # Here attenuation_scale converts the sampled CT values into an
        # effective attenuation coefficient before integration.
        return attenuation_sum

    if output_mode == "opacity":
        # Previous display convention:
        # 1 - exp(-D)
        return 1.0 - np.exp(-attenuation_sum)

    raise ValueError(
        f"Unknown output_mode={output_mode!r}. "
        "Use 'line_integral' or 'opacity'."
    )


try:
    import cupy as cp
    from cupyx.scipy.ndimage import map_coordinates as gpu_map_coordinates

    GPU_AVAILABLE = True
except ImportError:
    cp = np
    from scipy.ndimage import map_coordinates as gpu_map_coordinates

    GPU_AVAILABLE = False


def estimate_chunk_memory_gb(
    row_chunk,
    cols,
    n_samples,
    dtype_bytes=4,
):
    core = row_chunk * cols * n_samples * dtype_bytes
    return (core * 7) / 1e9


def generate_cone_beam_drr_gpu(
    vol,
    geometry,
    attenuation_scale=0.015,
    beam_axis=1,
    n_samples=None,
    row_chunk=16,
    verbose=True,
    row_offset=0,
    object_transform=None,
    output_mode="line_integral",
):
    """
    row_offset skips rows at the top of the full detector while preserving
    the detector's original physical coordinate system.
    """
    other_axes = [a for a in range(3) if a != beam_axis]

    beam_spacing = geometry.spacing_for_axis(beam_axis)
    v_spacing = geometry.spacing_for_axis(other_axes[0])
    u_spacing = geometry.spacing_for_axis(other_axes[1])

    rotation_np = fiji_rotation_to_numpy(object_transform)
    half_extent_mm = _half_extent_along_beam_mm(
        vol, geometry, beam_axis, rotation_np
    )

    full_rows, cols = geometry.detector_shape
    output_rows = full_rows - row_offset

    det_v_full = (
        np.arange(row_offset, full_rows) - full_rows / 2
    ) * geometry.detector_pixel_spacing_mm
    det_u_full = (
        np.arange(cols) - cols / 2
    ) * geometry.detector_pixel_spacing_mm

    src_pos = -geometry.spd_mm
    det_pos = geometry.sid_mm - geometry.spd_mm

    if n_samples is None:
        n_samples = _default_n_samples(
            vol, geometry, beam_axis, half_extent_mm, rotation_np
        )

    t_enter = (-half_extent_mm - src_pos) / (det_pos - src_pos)
    t_exit = (half_extent_mm - src_pos) / (det_pos - src_pos)

    t_np = np.linspace(t_enter, t_exit, n_samples)
    dt = (t_exit - t_enter) / n_samples

    if verbose:
        est_gb = estimate_chunk_memory_gb(
            row_chunk, cols, n_samples
        )
        n_chunks = int(np.ceil(output_rows / row_chunk))
        print(
            f"[GPU DRR] backend={'cupy/GPU' if GPU_AVAILABLE else 'numpy/CPU fallback'}, "
            f"full_detector={full_rows}x{cols}, row_offset={row_offset}, "
            f"output={output_rows}x{cols}, row_chunk={row_chunk}, "
            f"~{est_gb:.2f} GB per chunk, {n_chunks} chunks total"
        )

    vol_gpu = cp.asarray(vol, dtype=cp.float32)
    t_gpu = cp.asarray(t_np, dtype=cp.float32).reshape(-1, 1, 1)
    output = cp.zeros((output_rows, cols), dtype=cp.float32)

    if rotation_np is not None:
        rotation_inv_gpu = cp.asarray(
            rotation_np.T, dtype=cp.float32
        )
    else:
        rotation_inv_gpu = None

    for r0 in range(0, output_rows, row_chunk):
        r1 = min(r0 + row_chunk, output_rows)
        chunk_rows = r1 - r0

        det_v_chunk = cp.asarray(
            det_v_full[r0:r1], dtype=cp.float32
        )
        det_u_chunk = cp.asarray(
            det_u_full, dtype=cp.float32
        )

        det_grid_v, det_grid_u = cp.meshgrid(
            det_v_chunk,
            det_u_chunk,
            indexing="ij",
        )

        beam_coord_mm = src_pos + t_gpu * (det_pos - src_pos)
        beam_coord_mm = cp.broadcast_to(
            beam_coord_mm,
            (n_samples, chunk_rows, cols),
        )

        v_coord_mm = t_gpu * det_grid_v[cp.newaxis, :, :]
        u_coord_mm = t_gpu * det_grid_u[cp.newaxis, :, :]

        physical = [None, None, None]
        physical[beam_axis] = beam_coord_mm
        physical[other_axes[0]] = (
            v_coord_mm + geometry.offset_v_mm
        )
        physical[other_axes[1]] = (
            u_coord_mm + geometry.offset_u_mm
        )

        if rotation_inv_gpu is not None:
            p0 = (
                rotation_inv_gpu[0, 0] * physical[0]
                + rotation_inv_gpu[0, 1] * physical[1]
                + rotation_inv_gpu[0, 2] * physical[2]
            )
            p1 = (
                rotation_inv_gpu[1, 0] * physical[0]
                + rotation_inv_gpu[1, 1] * physical[1]
                + rotation_inv_gpu[1, 2] * physical[2]
            )
            p2 = (
                rotation_inv_gpu[2, 0] * physical[0]
                + rotation_inv_gpu[2, 1] * physical[1]
                + rotation_inv_gpu[2, 2] * physical[2]
            )

            physical = [p0, p1, p2]

        coords = [
            physical[0] / geometry.spacing_for_axis(0)
            + vol.shape[0] / 2,
            physical[1] / geometry.spacing_for_axis(1)
            + vol.shape[1] / 2,
            physical[2] / geometry.spacing_for_axis(2)
            + vol.shape[2] / 2,
        ]

        coords = cp.stack(coords, axis=0)

        sampled = gpu_map_coordinates(
            vol_gpu,
            coords,
            order=1,
            mode="constant",
            cval=0.0,
        )

        # SAME FIX as the CPU version: per-pixel ray length instead of
        # one shared scalar, accounts for oblique/diagonal rays.
        full_ray_length_chunk = cp.sqrt(
            (det_pos - src_pos) ** 2 + det_grid_v ** 2 + det_grid_u ** 2
        )
        step_mm_chunk = dt * full_ray_length_chunk

        attenuation_sum_chunk = (
            cp.sum(sampled, axis=0)
            * step_mm_chunk
            * attenuation_scale
        )

        if output_mode == "line_integral":
            # D = integral(mu dl)
            output[r0:r1, :] = attenuation_sum_chunk
        elif output_mode == "opacity":
            # Previous display convention:
            # 1 - exp(-D)
            output[r0:r1, :] = (
                1.0 - cp.exp(-attenuation_sum_chunk)
            )
        else:
            raise ValueError(
                f"Unknown output_mode={output_mode!r}. "
                "Use 'line_integral' or 'opacity'."
            )

    if GPU_AVAILABLE:
        return cp.asnumpy(output)

    return output