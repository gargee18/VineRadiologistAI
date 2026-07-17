"""
Core projection logic: Beer-Lambert DRR generation plus full pose sampling
(yaw, pitch, roll, distance) used to synthesize realistic radiograph
geometry from a single CT volume.
"""

from dataclasses import dataclass
import numpy as np
from scipy.ndimage import rotate, sobel, zoom


def generate_drr(vol: np.ndarray, attenuation_scale: float = 0.015, axis: int = 1) -> np.ndarray:
    return 1.0 - np.exp(-np.sum(vol * attenuation_scale, axis=axis))


def apply_yaw(vol: np.ndarray, angle: float) -> np.ndarray:
    return rotate(vol, angle=angle, axes=(1, 2), reshape=False, order=1)


def apply_pitch(vol: np.ndarray, angle: float) -> np.ndarray:
    return rotate(vol, angle=angle, axes=(0, 2), reshape=False, order=1)


def apply_roll(img: np.ndarray, angle: float) -> np.ndarray:
    return rotate(img, angle=angle, reshape=False, order=1)


def apply_distance(img: np.ndarray, distance_factor: float) -> np.ndarray:
    if distance_factor == 1.0:
        return img

    zoomed = zoom(img, distance_factor, order=1)
    out = np.zeros_like(img)

    def _center_slices(target_len, source_len):
        if source_len >= target_len:
            start = (source_len - target_len) // 2
            return slice(0, target_len), slice(start, start + target_len)
        start = (target_len - source_len) // 2
        return slice(start, start + source_len), slice(0, source_len)

    out_r, in_r = _center_slices(img.shape[0], zoomed.shape[0])
    out_c, in_c = _center_slices(img.shape[1], zoomed.shape[1])
    out[out_r, out_c] = zoomed[in_r, in_c]
    return out


@dataclass
class Pose:
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    distance: float = 1.0
    attenuation_scale: float = 0.015


def render_pose(vol: np.ndarray, pose: Pose, axis: int = 1) -> np.ndarray:
    posed_vol = apply_pitch(apply_yaw(vol, pose.yaw), pose.pitch)
    drr = generate_drr(posed_vol, attenuation_scale=pose.attenuation_scale, axis=axis)
    drr = apply_roll(drr, pose.roll)
    drr = apply_distance(drr, pose.distance)
    return drr


def render_pose_mask(mask_vol: np.ndarray, pose: Pose, axis: int = 1,
                      mode: str = "silhouette") -> np.ndarray:
    """Same pose pipeline as render_pose, but for a binary tissue mask.
    Use with the *same* Pose and deformation field_seed as the volume."""
    posed = apply_pitch(apply_yaw(mask_vol, pose.yaw), pose.pitch)
    if mode == "silhouette":
        proj = silhouette_map(posed, axis=axis)
    elif mode == "thickness":
        proj = thickness_map(posed, axis=axis)
    else:
        raise ValueError(f"unknown mode: {mode}")
    proj = np.nan_to_num(proj, nan=0.0)
    proj = apply_roll(proj, pose.roll)
    proj = apply_distance(proj, pose.distance)
    return proj


def sample_pose(cfg, rng: np.random.Generator = None) -> Pose:
    rng = rng or np.random.default_rng()
    return Pose(
        yaw=rng.choice(cfg.yaw_angles),
        pitch=rng.uniform(*cfg.pitch_range),
        roll=rng.uniform(*cfg.roll_range),
        distance=rng.uniform(*cfg.distance_range),
        attenuation_scale=rng.uniform(*cfg.attenuation_scale_range),
    )


def depth_map(vol: np.ndarray, axis: int = 1) -> np.ndarray:
    coords = np.arange(vol.shape[axis]).reshape(
        [-1 if i == axis else 1 for i in range(vol.ndim)]
    )
    weights = vol.astype(np.float32)
    total = np.sum(weights, axis=axis)
    with np.errstate(invalid="ignore"):
        return np.where(total > 0, np.sum(weights * coords, axis=axis) / total, np.nan)


def thickness_map(vol: np.ndarray, axis: int = 1) -> np.ndarray:
    result = np.sum(vol > 0, axis=axis).astype(np.float32)
    return np.where(result > 0, result, np.nan)


def silhouette_map(vol: np.ndarray, axis: int = 1) -> np.ndarray:
    result = (np.sum(vol > 0, axis=axis) > 0).astype(np.float32)
    return np.where(result > 0, result, np.nan)


def contour_map(vol: np.ndarray, axis: int = 1) -> np.ndarray:
    silhouette = (np.sum(vol > 0, axis=axis) > 0).astype(np.float32)
    edges = np.hypot(sobel(silhouette, axis=0), sobel(silhouette, axis=1))
    return np.where(edges > 0, edges, np.nan)