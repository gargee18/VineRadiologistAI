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
synthetic DRR is not a real X-ray. Everything in VineRadiologist beyond the basic
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
bodies. `VineRadiologist.deform` adds:

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

### Distance baseline (from real DICOM metadata)

DICOM headers from the potted greenhouse radiograph acquisitions (extracted
via `scripts/extract_dicom_metadata.py`, `dataset/radiograph/`) give a real
reference point:

- **DistanceSourceToDetector (SID)**: ~1500mm, essentially constant across
  every scan (a handful read 1502mm; effectively 0.1% spread)
- **DistanceSourceToPatient (SPD)**: ~1349mm whenever present
- Implied magnification factor: SID / SPD ≈ 1.11

The greenhouse rig itself does not vary this distance scan to scan, it's a
fixed mechanical setup. That does **not** mean field acquisition will be
similarly fixed: a handheld portable device used directly on field vines,
with no rig, is expected to introduce real operator-to-operator and
scan-to-scan variation that this greenhouse data cannot measure. The
`distance_range` multiplier in `xvine.config.ProjectionConfig` (currently
0.85-1.15, applied around a baseline of 1.0) should be read as scaling
around this ~1349mm SPD reference point, not as a validated field range.

**Status: the reference point (1349mm SPD) is real and measured. The width
of variation expected in actual field handheld use is still an unvalidated
placeholder**, pending either real field acquisition data or input from
whoever operates the device in the field.

## Why detector noise matters more than more geometry

Geometric randomization narrows the gap in pose/shape space, but the actual
pixel statistics a model sees (noise, contrast, dynamic range) are often a
bigger contributor to the sim-to-real gap than geometry. `VineRadiologist.noise`
simulates Poisson (quantum) noise, Gaussian sensor noise, and gamma/contrast
shifts on the projected image.

## What's not yet in this repo

- Quantitative sim-to-real evaluation, e.g. FID between DRRs and real
  portable X-rays.
- Feature-level domain adaptation (adversarial alignment, contrastive
  objectives) on top of the data-level augmentation implemented here.
- Any fine-tuning on real unlabeled portable X-rays, if/when available.
