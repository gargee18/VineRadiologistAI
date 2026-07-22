# VineRadiologist: Synthetic Radiograph Generation Pipeline

**Project documentation — sim-to-real methodology, decisions, and status**

Repository: `github.com/gargee18/VineRadiologistAI`
Package: `VineRadiologist` (`src/VineRadiologist/`)

---

## 1. Purpose and motivation

The end goal is a model that detects and quantifies grapevine trunk disease
(white rot, necrosis) from **portable X-ray radiographs taken in the
field**. Ground-truth tissue annotation is not practically feasible on a
real 2D radiograph, tissue boundaries are only unambiguous in 3D CT.

The approach:
1. Annotate 3D CT volumes (unambiguous, done once per specimen) — masks
   already exist: `segmentation_AMADOU.tif` (white rot), `segmentation_NECROSE.tif`
   (necrosis), `segmentation_SAIN.tif` (healthy).
2. Project the CT volume and its masks into synthetic 2D radiographs (DRRs)
   that approximate what a real portable X-ray would produce.
3. Train on the synthetic (image, label) pairs.
4. Eventually validate/fine-tune against real radiographs.

Everything beyond the basic projection exists to shrink the gap between
step 2's synthetic output and step 4's real target.

---

## 2. Repository structure

```
VineRadiologistAI/
├── src/VineRadiologist/
│   ├── config.py       # all randomization ranges, single source of truth
│   ├── io.py            # load_specimen / load_volume / load_mask
│   ├── projection.py     # parallel Beer-Lambert DRR + pose sampling
│   ├── cone_beam.py       # cone-beam (point-source) DRR projection
│   ├── deform.py          # bending, elastic warp, deform_batch
│   ├── noise.py            # Poisson/Gaussian/gamma (built, not yet used)
│   └── visualize.py         # mask-overlay QA plots
├── scripts/
│   ├── generate_dataset.py         # single-specimen exploratory generation
│   ├── build_training_set.py        # full dataset build, specimen-level split, parallelized
│   ├── compare_deformations.py       # plain DRR vs deformed+posed (synthetic only)
│   ├── compare_real_vs_synthetic.py   # visual real vs synthetic grid
│   ├── compare_distributions.py        # quantitative real vs synthetic (Wasserstein)
│   ├── compare_parallel_vs_conebeam.py  # A/B test: projection geometry vs real
│   ├── extract_dicom_metadata.py         # curated DICOM geometry tags
│   ├── extract_dicom_metadata_full.py     # all DICOM fields, presence/variance summary
│   ├── extract_dicom_pixels.py             # lossless pixel extraction from real DICOMs
│   ├── inspect_tiff_spacing.py               # ImageJ/Fiji TIFF metadata reader
│   └── find_ct_voxel_spacing.py               # search CT/ folders for DICOM spacing tags
├── configs/default.yaml   # human-readable mirror of config.py
├── docs/methodology.md     # shorter in-repo rationale doc
└── tests/test_projection.py
```

---

## 3. Core projection method

### 3.1 Beer-Lambert attenuation (parallel projection)

A naive projection (summing voxel intensities along a ray) gives a flat
silhouette, discarding the physical relationship between material density,
path length, and detected intensity. Real X-ray attenuation is exponential:

```
I = 1 - exp(-Σ(attenuation_scale × voxel_density))
```

implemented in `projection.generate_drr()`. This is the correct **law**, but
originally implemented with **parallel rays** (all rays traveling in one
fixed direction), which is a geometric simplification, not physically how
an X-ray source works.

**Config parameter**: `attenuation_scale`, currently randomized in range
`(0.010, 0.020)` — **this range is an unvalidated guess**, not derived from
material physics or the real device's KVP/exposure settings. See §7.

### 3.2 Cone-beam projection (added later)

Real X-ray sources are point sources emitting diverging rays, producing
magnification (objects closer to the source appear larger) and geometric
blur that parallel projection cannot reproduce.

**Implementation** (`cone_beam.py`, `generate_cone_beam_drr`): for each
detector pixel, a ray is cast from a point source through that pixel; the
ray is sampled along its intersection with the volume's bounding box
(trilinear interpolation via `scipy.ndimage.map_coordinates`); samples are
summed and passed through the same Beer-Lambert exponential.

**Geometry parameters** (`ConeBeamGeometry`):
- `sid_mm` = 1500.0 — Source-to-Detector Distance
- `spd_mm` = 1349.0 — Source-to-Patient Distance
- `detector_pixel_spacing_mm` = 0.148
- `voxel_spacing_mm` = 0.7224 — **must match the volume being projected**
  (see §6.3, a real bug was caught and fixed here)

**A bug was found and fixed during development**: initial ray sampling
spanned the *entire* source-to-detector distance uniformly, wasting most
samples on empty space outside the volume; for small `spd`, most/all
samples could miss the volume entirely. Fixed by restricting sampling to
only the ray segment intersecting the volume's bounding box.

**A second bug was found and fixed**: the comparison script initially sized
the detector's pixel *count* to match the volume's voxel grid, which, at
the real (much finer) detector pixel spacing, made the detector's physical
footprint far smaller than the trunk itself, zooming into a meaningless
patch. Fixed by sizing detector pixel count so `n_pixels × pixel_spacing`
covers the volume's real physical extent.

**Empirical result (§8.2)**: cone-beam did NOT improve the match to real
data on tested specimen(s). Kept in the codebase for physical correctness
and future use, but not currently prioritized for further tuning, since the
real SID/SPD ratio (~1.11×) implies only mild magnification.

---

## 4. Pose randomization

Applied per generated sample via `Pose` dataclass and `sample_pose()`:

| Parameter | Range | Meaning | Basis |
|---|---|---|---|
| `yaw` | {0°, 30°, 60°, 90°} | in-plane rotation before projection | arbitrary discrete set |
| `pitch` | (-30°, 30°) | out-of-plane tilt before projection | arbitrary, deliberately modest |
| `roll` | (-10°, 10°) | 2D rotation of projected image (detector tilt) | arbitrary |
| `distance` | (0.85, 1.15) | magnification scale factor (see §7.2) | anchored to real SPD=1349mm reference but width unvalidated |
| `attenuation_scale` | (0.010, 0.020) | Beer-Lambert coefficient | unvalidated guess |

`apply_distance()` implements distance as an image-plane zoom + center
crop/pad, **not** true 3D perspective (that's what cone-beam adds).

---

## 5. Deformation

Real trunks are neither straight nor rigid. Two deformations are applied
to the CT volume (and, with the identical field, to co-registered masks)
before projection:

### 5.1 Bending (`bend_volume`)
Each slice along a chosen axis is rotated by an angle varying linearly
from `-max_angle/2` to `+max_angle/2` across the volume's extent,
approximating gradual trunk curvature (as opposed to one rigid rotation of
the whole volume).

`bend_max_angle_range = (5.0, 25.0)` — degrees, randomized per sample.
**Bug fixed**: this was originally a single fixed value (15°), producing
*identical* curvature across every specimen; caught via visual inspection
(`compare_deformations.py`) and changed to a randomized range.

### 5.2 Elastic deformation (`elastic_deform`)
A smooth random displacement field (Gaussian-filtered noise, scaled) is
built and applied via `map_coordinates`.

- `elastic_alpha_range = (200.0, 400.0)` — displacement strength
- `elastic_sigma_range = (8.0, 12.0)` — smoothness of the field

Both ranges are reasonable-sounding defaults, **not validated** against
any real measurement of trunk shape variability.

### 5.3 Batched deformation (`deform_batch`) — performance fix
Originally, deforming a volume plus 3 masks called the single-array
deformation function 4 times with the same `field_seed`, which
**regenerated the identical expensive displacement field from scratch each
time** (~4x wasted computation). `deform_batch()` computes the bend angle
and elastic field **once** and applies to all arrays, measured at
**~2.5-3x speedup**.

---

## 6. Real acquisition data — findings

### 6.1 Radiograph DICOM metadata (`RADIO/DICOMOBJ/`)

Extracted via `extract_dicom_metadata_full.py`. Key findings:

| Tag | Finding |
|---|---|
| `DistanceSourceToDetector` (SID) | ~1500mm, essentially constant (±0.1%) across all scans |
| `DistanceSourceToPatient` (SPD) | ~1349mm whenever present |
| `KVP` | **varies 50.8-78** across specimens |
| `Exposure`/`XRayTubeCurrent` | **varies ~900-7900** across specimens |
| `Grid` (anti-scatter) | **varies YES/NO** across specimens |
| `FilterType` | Cu 0.0mm, consistent |
| `ImagerPixelSpacing` | 0.148mm × 0.148mm, consistent |
| `SeriesDescription`/`BodyPartExamined` | e.g. "télérachis Face AP.", "RAD_Epaule F", "RAD_Poignet F avec grille" — **human clinical protocols** (spine/shoulder/wrist) repurposed for vine specimens, not a dedicated plant-imaging protocol |

**Implication**: the real acquisitions were taken on a **fixed clinical
rig on potted greenhouse vines**, not a portable field device, and used
**inconsistent protocols** across specimens (different KVP/exposure/grid
per specimen). There is no single "real target distribution", multiple
real conditions exist even within the reference dataset. Note also this
is NOT the target deployment domain (field, handheld); it is the closest
real-world anchor currently available.

### 6.2 Compressed pixel format
Real radiograph DICOMs use JPEG-Lossless compression (`pylibjpeg` +
`pylibjpeg-libjpeg` required to decode; `gdcm` is an alternative). Lossless
means no data loss, just requires the right decoder plugin.

### 6.3 CT voxel spacing — two different numbers, do not conflate

| Source | Voxel spacing | Confirmed how |
|---|---|---|
| **Raw CT reconstruction** (`unireconstruction.xml`, found in raw tar archives, e.g. `RX_CEP_011_a_022/*/CEP0XX.tar`) | **0.1777mm isotropic** (range 0.177104-0.178422mm across 11/12 specimens checked; CEP012 not recoverable) | Scanner's own reconstruction XML — ground truth |
| **`registered.tif`** (what `load_specimen()` actually loads) | **0.7224mm** in-plane (majority group); **~0.7145mm** for the 3 RES specimens (014, 015, 016) | ImageJ TIFF `XResolution` tag, confirmed directly |

The registration pipeline resampled the raw ~0.1777mm data to ~0.7224mm
(~4x coarser). **`ConeBeamGeometry.voxel_spacing_mm` must use 0.7224mm**,
matching what the pipeline actually processes — using the raw 0.1777mm
value here would silently produce geometry wrong by ~4x. This mistake was
caught before shipping (see §3.2).

**Unresolved**: Z-spacing (through-slice) for `registered.tif` was never
recorded in its metadata (ImageJ shows `voxel depth = 1.0` as an **unlabeled
placeholder**, not a real value — confirmed by the missing unit next to it
in the Fiji properties dialog). The current isotropic assumption (Z ≈ XY)
is an *inference* (the raw source was confirmed isotropic before
resampling), not a direct measurement.

### 6.4 Scan hardware context (`restore.macro`)
CT scan used: `voltage=70` kVp, `filter="Carbone 2mm"`, 7200 images over 5
turns. This is separate hardware/settings from the 2D radiograph
acquisitions (different clinical machine).

---

## 7. Parameters that remain unvalidated guesses

Being explicit about what is a guess vs. confirmed, per the earlier
parameter checklist (see attached spreadsheet `parameter_checklist.xlsx`
for the full comparison table):

| Parameter | Status |
|---|---|
| `attenuation_scale` range | ❌ guess; also found to under-use available dynamic range (synthetic max ~0.18-0.37 vs. real using full 0-4095 12-bit range) |
| `distance_range` width | ⚠️ center anchored to real SPD=1349mm, but width (0.85-1.15) is an unvalidated guess for actual field/handheld variance |
| `bend_max_angle_range`, `elastic_alpha_range`, `elastic_sigma_range` | ❌ reasonable-sounding guesses, not validated against real trunk shape data |
| KVP, exposure, grid effects | ❌ not modeled at all |
| Scatter, beam hardening | ❌ not modeled at all |
| Gain/offset (fixed-pattern detector calibration) | ❌ not modeled, lower priority than the above |

---

## 8. Quantitative comparison methodology

### 8.1 Why Wasserstein distance, not FID
With a small number of real images (a few dozen across specimens), FID
(Fréchet Inception Distance, the standard generative-model metric) is not
statistically reliable, it needs hundreds+ samples to stabilize. Instead,
`compare_distributions.py` computes:

- Each image normalized independently to its own [0,1] range (so different
  bit depths — real uint16 vs synthetic float32 — become comparable)
- **Wasserstein distance** (earth-mover's distance) between the pooled real
  pixel distribution and pooled synthetic pixel distribution:

  > Intuitively: the minimum "work" (mass × distance moved) needed to
  > reshape one distribution into the other. 0 = identical distributions.
  > No universal "good" threshold — track it as a relative number across
  > pipeline changes, not an absolute pass/fail score.

### 8.2 Results so far (see `experiment_log.xlsx` for the live-updated table)

| Specimen | Comparison | Wasserstein distance | Verdict |
|---|---|---|---|
| CEP011_AS1 | Real vs Parallel DRR | 0.1054 | baseline |
| CEP011_AS1 | Real vs Cone-beam DRR | 0.1368 | **worse**, not adopted for further tuning |

### 8.3 Key finding: background/FOV mismatch dominates the histogram
Both parallel and cone-beam synthetic histograms show a **large spike at
exactly pixel value 0.0** (>60 density) that real images do not have (real
images show a smooth single peak with no true-zero mass, consistent with
detector noise floor). This indicates synthetic images contain
significantly more solid-black background/empty space than real images,
either from field-of-view being too wide relative to the trunk, or
`apply_distance`'s zero-padding behavior at certain zoom factors.

**This is currently the single largest identified, unaddressed gap.**

---

## 9. Prioritized next steps

1. **Fix background/FOV framing** (§8.3) — measure foreground/background
   pixel fraction in real vs synthetic directly; tighten crop or adjust
   projection field of view to match. Likely single biggest lever.
2. **Increase/randomize `attenuation_scale`** so synthetic images use more
   of their native dynamic range before normalization (currently topping
   out around 20-37% of possible range).
3. **Re-run distributional comparison after each isolated change**, one
   variable at a time (the cone-beam test demonstrated why: a physically
   "more correct" change can still measure as worse; only isolated,
   measured changes tell you what's actually helping).
4. **Exposure-equivalent noise**: already implemented in `noise.py`
   (Poisson/Gaussian/gamma), currently deliberately deferred to
   training-time augmentation rather than baked into saved dataset files.
5. **Anti-scatter grid approximation**: moderate-complexity addition
   (blurred low-frequency haze layer when simulating grid=off), not yet
   implemented.
6. **KVP-dependent energy attenuation**: highest-complexity remaining item,
   would need real per-material attenuation coefficients at different
   X-ray energies (e.g. from NIST tables) — deferred until items 1-3 are
   resolved.
7. Once foreground/background is fixed, consider comparing **foreground-only**
   pixel statistics (masking out background in both real and synthetic)
   rather than whole-image histograms, since whole-image comparison is
   currently dominated by the background proportion rather than tissue
   structure.

---

## 10. Metadata field glossary

| Term | Definition |
|---|---|
| **SID** | Source-to-Detector Distance — total distance from X-ray source to detector plate |
| **SPD** | Source-to-Patient Distance — distance from X-ray source to subject |
| **KVP** | Kilovoltage peak — X-ray tube voltage, controls beam energy/penetration |
| **mAs / Exposure / XRayTubeCurrent** | Controls photon count — overall brightness and noise level |
| **Anti-scatter grid** | Physical grid blocking scattered X-rays before the detector; increases contrast, reduces haze |
| **Beer-Lambert law** | Physical law: X-ray intensity decays exponentially with attenuation × path length |
| **Voxel spacing** | Real-world size (mm) represented by one voxel/pixel |
| **Isotropic** | Equal spacing in all 3 spatial dimensions |
| **Wasserstein distance** | Earth-mover's distance between two probability distributions; 0 = identical |
| **Beam hardening** | Lower-energy X-ray photons are absorbed disproportionately as depth increases, changing effective attenuation with depth (not modeled currently) |
| **Scatter** | X-rays deflected off tissue before reaching the detector, adding low-contrast haze (not modeled currently) |
| **DRR** | Digitally Reconstructed Radiograph — a 2D projection synthesized from a 3D volume |
| **Field-level split (specimen-level split)** | Train/val/test split done by whole specimen, not by individual sample, to prevent data leakage from the same trunk appearing in both train and test |

---

## 11. Key formulas (reference)

**Beer-Lambert attenuation (parallel):**
```
I(x, y) = 1 - exp( -attenuation_scale × Σ_z density(x, y, z) )
```

**Wasserstein distance** (1D, as used here):
```
W(P, Q) = ∫ |F_P(x) - F_Q(x)| dx
```
where F_P, F_Q are the cumulative distribution functions of the real and
synthetic pixel-value distributions.

**Magnification factor (cone-beam):**
```
M = SID / SPD  ≈ 1500 / 1349 ≈ 1.11
```

**Bend angle field (per slice i along axis of length n):**
```
angle(i) = -max_angle/2 + i × (max_angle / (n-1))
```

---

## 12. Known limitations, stated plainly

- No quantitative comparison has been run across more than one specimen at
  a time; single-specimen results (n=2 real images) should not be treated
  as dataset-wide conclusions.
- Real reference data is from potted greenhouse vines on a fixed clinical
  rig, not field-deployed portable X-ray conditions — the actual
  deployment target domain remains unmeasured.
- Real acquisition protocols vary specimen-to-specimen (KVP, exposure,
  grid), so there is no single, consistent "ground truth distribution."
- Z-axis voxel spacing for `registered.tif` is inferred, not measured.
- Detector noise, scatter, beam hardening, and energy-dependent attenuation
  are all currently absent from the synthetic pipeline.
- Cone-beam projection is implemented and tested but empirically did not
  improve distributional match on the one specimen tested; this should be
  re-checked once background/FOV and attenuation_scale issues are resolved,
  since those confounds may currently be masking any real cone-beam benefit.

---

*Last updated: reflects pipeline state through cone-beam implementation and
first quantitative real-vs-synthetic comparison. Update this document as
further changes are made — see `experiment_log.xlsx` for the running,
per-change Wasserstein tracking table.*
