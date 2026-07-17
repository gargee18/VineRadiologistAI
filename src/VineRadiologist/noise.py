"""
Detector-side noise / contrast simulation, applied to the 2D projected
image after geometric pose sampling.

These generally matter more than extra geometric augmentation for closing
the sim-to-real gap, since they target the image statistics a classifier
actually sees.
"""

import numpy as np


def add_poisson_noise(img: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Simulate quantum noise. `scale` controls the effective photon count:
    lower scale -> noisier (fewer counts), matches low-dose acquisitions."""
    img = np.clip(img, 0, None)
    counts = np.random.poisson(img * scale * 255.0) / (scale * 255.0)
    return np.clip(counts, 0, 1)


def add_gaussian_noise(img: np.ndarray, sigma: float = 0.01) -> np.ndarray:
    noise = np.random.normal(0, sigma, img.shape)
    return np.clip(img + noise, 0, 1)


def adjust_contrast(img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Gamma correction to simulate detector/processing contrast shifts."""
    img = np.clip(img, 0, 1)
    return np.power(img, gamma)


def simulate_detector(img: np.ndarray, cfg, rng: np.random.Generator = None) -> np.ndarray:
    """Apply the full noise chain using ranges from a NoiseConfig."""
    rng = rng or np.random.default_rng()
    out = adjust_contrast(img, rng.uniform(*cfg.gamma_range))
    out = add_poisson_noise(out, rng.uniform(*cfg.poisson_scale_range))
    out = add_gaussian_noise(out, rng.uniform(*cfg.gaussian_sigma_range))
    return out
