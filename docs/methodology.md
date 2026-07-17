# Methodology

## Problem

The end goal is a model that detects and quantifies grapevine trunk disease
(white rot, necrosis) from **portable X-ray radiographs taken in the field**.
There is no practical way to get ground-truth tissue annotations on a real
2D radiograph: tissue boundaries are only unambiguous in 3D CT. So instead:

1. Annotate 3D CT volumes (unambiguous, done once per specimen).
2. Project the CT volume and its masks into synthetic 2D radiographs (DRRs)
   using the same geometry.
3. Train on the synthetic (image, label) pairs; deploy on real portable
   X-rays.

This introduces a domain gap: no matter how the projection is done, a
synthetic DRR is not a real X-ray. Everything in vineradiology beyond the basic
projection exists to shrink that gap.

## Why Beer-Lambert projection

A naive projection (summing or averaging voxel intensities along a ray)
produces a flat silhouette: it discards the physical relationship between
material density, path length, and detected intensity. Real X-ray
attenuation follows an exponential law: intensity drops off exponentially
as attenuation accumulates along the ray. `generate_drr` implements
`1 - exp(-sum(attenuation))` for this reason, so dense wood attenuates more
than a simple linear projection would suggest, matching the physics a real
detector responds to.

## Why deformation before projection

Rotating the rigid volume (yaw/pitch) alone does not capture that real
trunks are curved and structurally variable, not perfectly straight rigid
bodies. `vineradiology.deform` adds:

- **Bending**: a per-slice rotation whose angle varies linearly along the
  trunk axis, approximating gradual curvature.
- **Elastic deformation**: a smooth random displacement field (Gaussian-
  smoothed noise), giving local shape variation beyond a single global bend.
  This is a widely-used augmentation technique; if you need a citation for
  it in a paper, verify one yourself rather than trusting one from this repo
  or from an LLM, since none is included here.

## Why full pose sampling (yaw, pitch, roll, distance)

A single fixed camera geometry only shows you one relationship between the
specimen and the X-ray source/detector. In the field, that relationship
varies:

- **Yaw**: in-plane rotation of the volume before projection.
- **Pitch** (-30° to 30°): out-of-plane tilt, i.e. the specimen isn't
  perfectly perpendicular to the beam.
- **Roll**: applied after projection, as a 2D rotation, approximating
  detector tilt rather than a volume-level effect.
- **Distance**: approximated as an image-plane magnification change, since
  source-to-detector distance affects apparent size/blur in a real
  radiograph.

## Why detector noise matters more than more geometry

Geometric randomization narrows the gap in pose/shape space, but the actual
pixel statistics a model sees (noise, contrast, dynamic range) are often a
bigger contributor to the sim-to-real gap than geometry. `vineradiology.noise`
simulates Poisson (quantum) noise, Gaussian sensor noise, and gamma/contrast
shifts on the projected image.

## What's not yet in this repo

- Quantitative sim-to-real evaluation, e.g. FID between DRRs and real
  portable X-rays.
- Feature-level domain adaptation (adversarial alignment, contrastive
  objectives) on top of the data-level augmentation implemented here.
- Any fine-tuning on real unlabeled portable X-rays, if/when available.
