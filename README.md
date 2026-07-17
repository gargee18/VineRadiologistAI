# vineradiology

CT-to-portable-X-ray domain adaptation for grapevine trunk disease detection.

Ground-truth annotation on real portable X-ray images is not practically
feasible: tissue boundaries (white rot, necrosis, healthy wood) can't be
labeled with confidence on a real 2D radiograph. vineradiology instead annotates 3D
CT volumes, where structure is unambiguous, and projects them into
synthetic 2D radiographs that approximate what a real portable X-ray device
would produce.

## Why Beer-Lambert

A plain summed or averaged projection through the volume gives a flat
silhouette. Real X-ray attenuation is exponential in material density and
path length. vineradiology's `generate_drr` uses `1 - exp(-sum(attenuation))` so the
simulated image reflects that physics: denser wood attenuates more, matching
what a real detector would record. See `docs/methodology.md` for more detail.

## Closing the sim-to-real gap

Pure projection isn't enough on its own, since the goal is to train a model
that will later be tested on *real* radiographs, and real trunks are neither
straight nor rigid. vineradiology adds, before and after projection:

- **Pre-projection 3D deformation** (`vineradiology.deform`): trunk bending and
  elastic warps, so shapes vary beyond what rigid rotation alone gives you.
- **Full pose sampling** (`vineradiology.projection`): yaw (in-plane), pitch
  (out-of-plane tilt, -30° to 30°), roll (post-projection 2D rotation), and
  source-to-detector distance (approximated as image-plane magnification).
- **Detector-side noise** (`vineradiology.noise`): Poisson noise, Gaussian noise,
  gamma/contrast shifts, since these tend to matter more than extra
  geometric augmentation for closing the sim-to-real gap.

All randomization ranges live in one place: `vineradiology.config`.

## Install

```bash
pip install -e .
```

## Quickstart

```python
from vineradiology import load_specimen, DEFAULT_CONFIG, random_bend_and_elastic, sample_pose, render_pose, simulate_detector

data = load_specimen(root="/path/to/data", specimen="CEP011_AS1", dataset="Dataset_Vitimage2019")
volume = data["volume"]

deformed = random_bend_and_elastic(volume, DEFAULT_CONFIG.deformation)
pose = sample_pose(DEFAULT_CONFIG.projection)
img = render_pose(deformed, pose)
img = simulate_detector(img, DEFAULT_CONFIG.noise)
```

Or generate a batch directly:

```bash
python scripts/generate_dataset.py --root /path/to/data \
    --dataset Dataset_Vitimage2019 --specimen CEP011_AS1 \
    --n-samples 20 --out out/CEP011_AS1
```

## Repository layout

```
src/vineradiology/       core package (config, io, projection, deform, noise, visualize)
scripts/         dataset generation + QA visualization entry points
tests/           unit tests for the projection/deformation math
configs/         default.yaml — same ranges as config.py, for non-Python overrides
docs/            methodology notes
```

## Status

Early-stage. Geometric augmentation (deformation + full pose sampling) and
detector noise simulation are implemented; sim-to-real evaluation (e.g. FID
between DRRs and real portable X-rays, or feature-level domain alignment)
is not yet part of this repo.
