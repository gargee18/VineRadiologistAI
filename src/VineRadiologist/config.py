"""
Central config for xvine synthetic radiograph generation.
Every randomization range used across the pipeline lives here so it can be
audited / tuned in one place (and overridden via configs/default.yaml).
"""

from dataclasses import dataclass, field
from typing import Tuple, List


@dataclass
class ProjectionConfig:
    # geometric pose sampling
    yaw_angles: List[float] = field(default_factory=lambda: [0, 30, 60, 90])
    pitch_range: Tuple[float, float] = (-30.0, 30.0)
    roll_range: Tuple[float, float] = (-10.0, 10.0)

    # source-to-detector distance, expressed as a relative scale factor
    # applied to the projection (1.0 = reference distance)
    distance_range: Tuple[float, float] = (0.6, 1.5)

    # Beer-Lambert attenuation
    attenuation_scale_range: Tuple[float, float] = (0.010, 0.020)

    # projection axis in the volume (1 = the axis used throughout xvine)
    projection_axis: int = 1


@dataclass
class DeformationConfig:
    # bending: max angular deflection applied along the trunk axis (degrees)
    bend_max_angle: float = 15.0
    bend_axis: int = 0  # axis along the trunk (e.g. vertical growth axis)

    # elastic deformation
    elastic_alpha_range: Tuple[float, float] = (200.0, 400.0)
    elastic_sigma_range: Tuple[float, float] = (8.0, 12.0)


@dataclass
class NoiseConfig:
    poisson_scale_range: Tuple[float, float] = (0.5, 2.0)
    gaussian_sigma_range: Tuple[float, float] = (0.0, 0.02)
    gamma_range: Tuple[float, float] = (0.8, 1.2)


@dataclass
class XvineConfig:
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    deformation: DeformationConfig = field(default_factory=DeformationConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    seed: int = None


DEFAULT_CONFIG = XvineConfig()
