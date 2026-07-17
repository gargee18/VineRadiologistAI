"""
Pre-projection 3D deformations.

These exist to push the synthetic volumes away from perfectly straight,
perfectly rigid geometry, since real trunks are neither. Applied to the
CT volume (and, with the same random field, to any co-registered masks)
before projection.
"""

import numpy as np
from scipy.ndimage import rotate, gaussian_filter, map_coordinates


def bend_volume(vol: np.ndarray, max_angle_deg: float, axis: int = 0) -> np.ndarray:
    """Bend a volume along `axis` by rotating each slice by an angle that
    varies linearly from -max_angle_deg/2 to +max_angle_deg/2 across the
    extent of `axis`. Approximates a trunk curving gradually along its
    length, rather than a rigid-body rotation of the whole volume.
    """
    n = vol.shape[axis]
    angles = np.linspace(-max_angle_deg / 2, max_angle_deg / 2, n)

    out = np.empty_like(vol)
    for i, angle in enumerate(angles):
        sl = [slice(None)] * vol.ndim
        sl[axis] = i
        sl = tuple(sl)
        out[sl] = rotate(vol[sl], angle=angle, axes=(0, 1), reshape=False, order=1)
    return out


def elastic_deform(vol: np.ndarray, alpha: float, sigma: float,
                    random_state: np.random.Generator = None,
                    field_seed: int = None) -> np.ndarray:
    """Classic random-displacement-field elastic deformation.

    Pass `field_seed` to regenerate the exact same displacement field on a
    second call (e.g. applying the same warp to a co-registered mask after
    warping the volume, so image and mask stay aligned).

    Note: this is a widely-used augmentation technique. I do not have a
    verified citation to give you for it, so don't quote one without
    checking it yourself.
    """
    rng = np.random.default_rng(field_seed) if field_seed is not None else (random_state or np.random.default_rng())
    shape = vol.shape

    displacements = []
    for _ in range(vol.ndim):
        d = rng.random(shape) * 2 - 1
        d = gaussian_filter(d, sigma, mode="constant", cval=0)
        d *= alpha
        displacements.append(d)

    grids = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    coords = [grids[i] + displacements[i] for i in range(vol.ndim)]

    return map_coordinates(vol, coords, order=1, mode="reflect").reshape(shape)


def random_bend_and_elastic(vol, cfg, rng=None, field_seed=None):
    rng = rng or np.random.default_rng()

    if field_seed is not None:
        seed_rng = np.random.default_rng(field_seed)
        bend_angle = seed_rng.uniform(*cfg.bend_max_angle_range)
        bent = bend_volume(vol, bend_angle, axis=cfg.bend_axis)
        alpha = seed_rng.uniform(*cfg.elastic_alpha_range)
        sigma = seed_rng.uniform(*cfg.elastic_sigma_range)
        return elastic_deform(bent, alpha, sigma, field_seed=field_seed)

    bend_angle = rng.uniform(*cfg.bend_max_angle_range)
    bent = bend_volume(vol, bend_angle, axis=cfg.bend_axis)
    alpha = rng.uniform(*cfg.elastic_alpha_range)
    sigma = rng.uniform(*cfg.elastic_sigma_range)
    return elastic_deform(bent, alpha, sigma, random_state=rng)
