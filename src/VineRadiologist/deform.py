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
    n = vol.shape[axis]
    angles = np.linspace(-max_angle_deg / 2, max_angle_deg / 2, n)

    out = np.empty_like(vol)
    for i, angle in enumerate(angles):
        sl = [slice(None)] * vol.ndim
        sl[axis] = i
        sl = tuple(sl)
        out[sl] = rotate(vol[sl], angle=angle, axes=(0, 1), reshape=False, order=1)
    return out


def compute_elastic_field(shape, alpha: float, sigma: float, field_seed: int = None,
                           rng: np.random.Generator = None):
    rng = np.random.default_rng(field_seed) if field_seed is not None else (rng or np.random.default_rng())

    displacements = []
    for _ in range(len(shape)):
        d = rng.random(shape) * 2 - 1
        d = gaussian_filter(d, sigma, mode="constant", cval=0)
        d *= alpha
        displacements.append(d)

    grids = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    coords = [grids[i] + displacements[i] for i in range(len(shape))]
    return coords


def apply_elastic_field(vol: np.ndarray, coords) -> np.ndarray:
    return map_coordinates(vol, coords, order=1, mode="reflect").reshape(vol.shape)


def elastic_deform(vol: np.ndarray, alpha: float, sigma: float,
                    random_state: np.random.Generator = None,
                    field_seed: int = None) -> np.ndarray:
    coords = compute_elastic_field(vol.shape, alpha, sigma, field_seed=field_seed, rng=random_state)
    return apply_elastic_field(vol, coords)


def random_bend_and_elastic(vol: np.ndarray, cfg, rng: np.random.Generator = None,
                             field_seed: int = None) -> np.ndarray:
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


def deform_batch(arrays: dict, cfg, rng: np.random.Generator = None,
                  field_seed: int = None) -> dict:
    rng = rng or np.random.default_rng()
    seed_rng = np.random.default_rng(field_seed) if field_seed is not None else rng

    bend_angle = seed_rng.uniform(*cfg.bend_max_angle_range)
    alpha = seed_rng.uniform(*cfg.elastic_alpha_range)
    sigma = seed_rng.uniform(*cfg.elastic_sigma_range)

    any_array = next(iter(arrays.values()))
    coords = compute_elastic_field(any_array.shape, alpha, sigma, field_seed=field_seed)

    out = {}
    for name, arr in arrays.items():
        bent = bend_volume(arr, bend_angle, axis=cfg.bend_axis)
        out[name] = apply_elastic_field(bent, coords)
    return out