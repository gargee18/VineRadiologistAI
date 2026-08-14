"""
Step B calibration: fit the DRR engine's attenuation_scale so that a
cone-beam DRR generated from a D1_T20 CT volume matches the real portable
XR (PRX) radiograph for the same specimen as closely as possible.

Geometry (sid_mm, spd_mm, detector_pixel_spacing_mm, detector_shape) is
NOT fitted here, it's fixed from your measured/confirmed values (Step A).
Only attenuation_scale is free, since it's the one physical unknown left
(the Beer-Lambert scale factor has no direct DICOM-measurable analogue).

Usage:
    python scripts/calibrate_drr.py \
        --xr dataset/D1_T20/CEP_378A/CT/registered.tif \
        --pxr dataset/radiograph_tif/CEP_378A/face.tif \
        --metric ssim

Distance metrics (pick with --metric):
  - "ssim"        structural similarity (1 - SSIM, so 0 = identical). Good
                   default: sensitive to structure, less to raw intensity scale.
  - "wasserstein"  earth-mover's distance between intensity histograms.
                   Matches the metric already used in compare_distributions.py.
  - "ncc"          1 - normalized cross-correlation. Cheap, sensitive to
                   overall structural alignment.
  - "mi"           1 - normalized mutual information. Robust to nonlinear
                   intensity relationships between real and synthetic.

NOTE: this is not yet a verified choice, the whiteboard photo you shared
flagged this as an open question (histogram distance / SSIM / MAD / mutual
info). ssim is a reasonable default to start iterating with, not a final
answer, swap --metric and compare results before committing to one.
"""

import argparse
from pathlib import Path

import numpy as np
import tifffile as tiff
from PIL import Image
from scipy.optimize import minimize_scalar
from scipy.stats import wasserstein_distance

from VineRadiologist.cone_beam import ConeBeamGeometry, generate_cone_beam_drr
from VineRadiologist.io import load_volume


# --- confirmed geometry (Step A measurements + portable_xr_metadata.csv) ---
# d1 = source-to-specimen = 80cm -> spd_mm
# d2 = source-to-captor    = 123cm -> sid_mm
# pixel spacing and detector shape from DICOM metadata (PixelSpacing,
# Rows, Columns), NOT the old 0.148mm / 512x512 placeholders.
# Full DICOM resolution (0.139mm, 3072x3072) is physically correct but
# WILL OOM the ray caster: it allocates arrays of shape
# (n_samples, rows, cols) in float64, which at full resolution is ~77GB.
# Downsample the working grid, keeping the physical footprint (rows *
# pixel_spacing) constant, so the geometry stays correct, just coarser.
def make_geometry(detector_size: int = 512, voxel_spacing_mm: float = None,
                   voxel_spacing_z_mm: float = None) -> ConeBeamGeometry:
    full_res = 3072
    full_spacing = 0.139
    physical_extent_mm = full_res * full_spacing  # ~427mm, unchanged
    scaled_spacing = physical_extent_mm / detector_size

    # in-plane (X/Y, axes 1 and 2 for a (Z, Y, X)-shaped volume like
    # CEP_378A_2026_XR.tif, shape (1908, 512, 512))
    inplane = voxel_spacing_mm if voxel_spacing_mm is not None else 0.7224
    # through-slice (Z, axis 0). Confirmed 0.4mm for the CEP RegistrationHighRes
    # pipeline specifically, do NOT reuse for other datasets without checking.
    z = voxel_spacing_z_mm if voxel_spacing_z_mm is not None else inplane

    spacing = (z, inplane, inplane)  # matches (axis0, axis1, axis2)

    return ConeBeamGeometry(
        sid_mm=1230.0,
        spd_mm=800.0,
        detector_pixel_spacing_mm=scaled_spacing,
        detector_shape=(detector_size, detector_size),
        voxel_spacing_mm=spacing,
    )


def load_image(path: str) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".dcm":
        import pydicom
        ds = pydicom.dcmread(str(path), force=True)
        pixels = ds.pixel_array.astype(np.float64)
        # MONOCHROME1 = higher pixel value is DARKER, invert so higher = brighter,
        # matching MONOCHROME2 and the DRR convention (attenuation -> darker).
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            pixels = pixels.max() - pixels
        return pixels
    if path.suffix.lower() in (".tif", ".tiff"):
        return tiff.imread(path).astype(np.float64)
    return np.array(Image.open(path).convert("L"), dtype=np.float64)


def find_view(specimen_dir: str, view: str = "Face") -> str:
    """Pick the .dcm file matching a given SeriesDescription view
    ('Face' or 'Profil') out of a specimen folder that has several."""
    import pydicom
    specimen_dir = Path(specimen_dir)
    for f in sorted(specimen_dir.glob("*.dcm")):
        try:
            ds = pydicom.dcmread(str(f), force=True, stop_before_pixels=True)
        except Exception:
            continue
        desc = getattr(ds, "SeriesDescription", "")
        if view.lower() in desc.lower():
            return str(f)
    raise FileNotFoundError(f"no '{view}' view found in {specimen_dir}")


def strip_saturated_band(img: np.ndarray, sat_frac: float = 0.98, row_coverage: float = 0.5,
                          debug: bool = True) -> np.ndarray:
    """Detect and remove a saturated/overexposed band from the TOP of a
    real radiograph (e.g. a bright strip from sensor overexposure or a
    collimation edge, unrelated to the specimen). Crops rows from the top
    until a row is found where fewer than `row_coverage` fraction of
    pixels are at/near max intensity (>= sat_frac * max).

    Only checks from the top down, real specimen radiographs like these
    don't saturate large horizontal bands elsewhere, so this is a
    conservative, targeted fix rather than a general saturation filter.
    """
    max_val = img.max()
    if max_val <= 0:
        return img
    sat_mask = img >= (sat_frac * max_val)
    row_sat_ratio = sat_mask.mean(axis=1)

    if debug:
        print(f"  [saturation check] image max={max_val:.2f}, "
              f"top-10-rows mean intensity={img[:10].mean():.2f}, "
              f"top-10-rows saturation ratio (>=98% of max)={row_sat_ratio[:10].mean():.3f}, "
              f"overall image mean={img.mean():.2f}")

    crop_row = 0
    for i, ratio in enumerate(row_sat_ratio):
        if ratio < row_coverage:
            crop_row = i
            break
    else:
        return img  # entire image saturated, don't crop everything, bail out

    if crop_row > 0:
        print(f"Stripped {crop_row} saturated row(s) from top of real image "
              f"({crop_row}/{img.shape[0]} = {100*crop_row/img.shape[0]:.1f}%)")
        return img[crop_row:]
    return img


def normalize(img: np.ndarray) -> np.ndarray:
    """Scale to [0, 1] based on its own min/max so real (12-16 bit) and
    synthetic (float, Beer-Lambert output in [0,1]) images are comparable."""
    img = img.astype(np.float64)
    lo, hi = img.min(), img.max()
    if hi <= lo:
        return np.zeros_like(img)
    return (img - lo) / (hi - lo)


def resize_to_match(img: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Nearest-neighbour-free resize via scipy.ndimage.zoom to match shapes
    before comparison (DRR output shape = detector_shape, real radiograph
    may differ slightly after DICOM extraction/cropping)."""
    from scipy.ndimage import zoom
    if img.shape == target_shape:
        return img
    factors = (target_shape[0] / img.shape[0], target_shape[1] / img.shape[1])
    return zoom(img, factors, order=1)


def auto_crop_to_content(sim: np.ndarray, real: np.ndarray, threshold: float = 0.05):
    """Crop both images to the bounding box of 'non-background' pixels in
    the SIM image (the synthetic DRR), since that's the one we control and
    trust the shape of. Keeps SSIM from being dominated by matching faint
    background/pot noise instead of the actual specimen."""
    sim_n = normalize(sim)
    mask = sim_n > threshold
    if not mask.any():
        return sim, real  # nothing above threshold, don't crop blindly
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    pad = int(0.05 * max(sim.shape))  # small margin so we don't crop edges too tight
    r0, c0 = max(0, r0 - pad), max(0, c0 - pad)
    r1, c1 = min(sim.shape[0], r1 + pad), min(sim.shape[1], c1 + pad)

    real_resized = resize_to_match(real, sim.shape)
    return sim[r0:r1, c0:c1], real_resized[r0:r1, c0:c1]


def compute_distance(sim: np.ndarray, real: np.ndarray, metric: str = "ssim") -> float:
    sim, real = auto_crop_to_content(sim, real)
    sim_n = normalize(sim)
    real_n = normalize(real)

    if metric == "wasserstein":
        return wasserstein_distance(real_n.ravel(), sim_n.ravel())

    if metric == "ncc":
        s = sim_n - sim_n.mean()
        r = real_n - real_n.mean()
        denom = np.sqrt((s ** 2).sum() * (r ** 2).sum())
        if denom == 0:
            return 1.0
        ncc = (s * r).sum() / denom
        return 1.0 - ncc

    if metric == "ssim":
        from skimage.metrics import structural_similarity as ssim
        score = ssim(real_n, sim_n, data_range=1.0)
        return 1.0 - score

    if metric == "mi":
        from sklearn.metrics import normalized_mutual_info_score
        # discretize into bins for MI estimation
        bins = 64
        r_binned = np.digitize(real_n.ravel(), np.linspace(0, 1, bins))
        s_binned = np.digitize(sim_n.ravel(), np.linspace(0, 1, bins))
        nmi = normalized_mutual_info_score(r_binned, s_binned)
        return 1.0 - nmi

    raise ValueError(f"unknown metric: {metric}")


def generate_fixed(vol: np.ndarray, real_img: np.ndarray, geometry: ConeBeamGeometry,
                    attenuation_scale: float, metric: str = "ssim", beam_axis: int = 1) -> dict:
    """Generate a DRR at a fixed attenuation_scale (no optimization), still
    scoring it against real_img for reference so you know how it compares."""
    drr = generate_cone_beam_drr(vol, geometry, attenuation_scale=attenuation_scale, beam_axis=beam_axis)
    dist = compute_distance(drr, real_img, metric=metric)
    return {"attenuation_scale": attenuation_scale, "distance": dist, "metric": metric, "drr": drr}


def calibrate(vol: np.ndarray, real_img: np.ndarray, geometry: ConeBeamGeometry,
              metric: str = "ssim", bounds=(0.002, 0.2)) -> dict:
    """Fit attenuation_scale by minimizing distance(sim_drr, real_img).

    Returns dict with best attenuation_scale, final distance, and the
    generated DRR at that setting (so you can visually inspect it, don't
    just trust the number).
    """
    history = []

    def objective(atten_scale):
        drr = generate_cone_beam_drr(vol, geometry, attenuation_scale=atten_scale)
        dist = compute_distance(drr, real_img, metric=metric)
        history.append((atten_scale, dist))
        return dist

    result = minimize_scalar(objective, bounds=bounds, method="bounded",
                              options={"xatol": 1e-5})

    best_drr = generate_cone_beam_drr(vol, geometry, attenuation_scale=result.x)

    return {
        "attenuation_scale": result.x,
        "distance": result.fun,
        "metric": metric,
        "drr": best_drr,
        "history": history,
    }


def main(xr_path, pxr_path, metric, out_drr, view=None, detector_size=512,
         voxel_spacing_mm=None, atten_bounds=(0.002, 0.2), voxel_spacing_z_mm=None,
         fixed_attenuation=None, beam_axis=1):
    geometry = make_geometry(detector_size, voxel_spacing_mm, voxel_spacing_z_mm)

    vol = load_volume(xr_path)
    if view is not None:
        pxr_path = find_view(pxr_path, view)
        print(f"Selected view file: {pxr_path}")
    real_img = load_image(pxr_path)
    real_img = strip_saturated_band(real_img)

    n_beam_est = vol.shape[1] * 2  # approx n_samples used by generate_cone_beam_drr
    est_gb = n_beam_est * detector_size * detector_size * 8 / 1e9
    print(f"XR (CT volume): {xr_path}  shape={vol.shape}")
    print(f"PXR (real radiograph): {pxr_path}  shape={real_img.shape}")
    print(f"Geometry: sid={geometry.sid_mm}mm  spd={geometry.spd_mm}mm  "
          f"pixel_spacing={geometry.detector_pixel_spacing_mm:.4f}mm  "
          f"detector={geometry.detector_shape}")
    print(f"Estimated peak memory for ray casting: ~{est_gb:.1f} GB "
          f"(reduce --detector-size if this is too high for your machine)")
    if fixed_attenuation is not None:
        values = fixed_attenuation if isinstance(fixed_attenuation, list) else [fixed_attenuation]
        results = []
        for v in values:
            r = generate_fixed(vol, real_img, geometry, v, metric=metric, beam_axis=beam_axis)
            results.append(r)
            print(f"attenuation_scale: {r['attenuation_scale']:.6f}")
            print(f"Distance ({metric}): {r['distance']:.6f}")
            print()
        result = min(results, key=lambda r: r["distance"])
        if len(values) > 1:
            print(f"Best: attenuation_scale={result['attenuation_scale']:.6f}  "
                  f"distance={result['distance']:.6f}")
    else:
        print(f"Fitting attenuation_scale with metric={metric} ...")
        result = calibrate(vol, real_img, geometry, metric=metric, bounds=atten_bounds)
        if abs(result["attenuation_scale"] - atten_bounds[1]) < 1e-4:
            print(f"WARNING: fitted attenuation_scale ({result['attenuation_scale']:.4f}) is "
                  f"pinned at the upper search bound ({atten_bounds[1]}). The optimizer wants "
                  f"to go higher, raise --atten-max and rerun before trusting this result.")
        elif abs(result["attenuation_scale"] - atten_bounds[0]) < 1e-4:
            print(f"WARNING: fitted attenuation_scale ({result['attenuation_scale']:.4f}) is "
                  f"pinned at the lower search bound ({atten_bounds[0]}). Lower --atten-min "
                  f"and rerun before trusting this result.")

    if fixed_attenuation is None:
        print(f"\nattenuation_scale: {result['attenuation_scale']:.6f}")
        print(f"Distance ({metric}): {result['distance']:.6f}")
        if "history" in result:
            print(f"Iterations: {len(result['history'])}")

    if out_drr:
        out_path = Path(out_drr)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tiff.imwrite(out_path, result["drr"].astype(np.float32))
        print(f"Calibrated DRR written to {out_path}")


def sweep(vol, geometry, values, out_dir, beam_axis=1, real_img=None, metric="ssim",
          save_all=False):
    """Generate a DRR at each attenuation_scale value and score it against
    real_img (if given). By default nothing is written to disk except the
    single best-scoring DRR, pass save_all=True to keep every one.
    """
    results = []  # (value, distance_or_None, drr_array)
    for v in values:
        drr = generate_cone_beam_drr(vol, geometry, attenuation_scale=v, beam_axis=beam_axis)
        dist = compute_distance(drr, real_img, metric=metric) if real_img is not None else None
        results.append((v, dist, drr))

    out_dir = Path(out_dir)

    if real_img is not None:
        results.sort(key=lambda r: r[1])
        print(f"\nRanked by {metric} distance (lowest = closest match):\n")
        for v, dist, _ in results:
            print(f"attenuation_scale: {v:.6f}")
            print(f"Distance ({metric}): {dist:.6f}")
            print()
        best_v, best_dist, best_drr = results[0]
        print(f"Best: attenuation_scale: {best_v:.6f}")
        print(f"Best distance ({metric}): {best_dist:.6f}")
    else:
        best_v, best_dist, best_drr = results[0]
        print("No --pxr given, can't rank by distance, just generated each value, "
              "no file saved by default (use --save-all-sweep to keep them).")

    out_dir.mkdir(parents=True, exist_ok=True)
    if save_all:
        for v, dist, drr in results:
            out_path = out_dir / f"drr_atten_{v:.4f}_axis{beam_axis}.tif"
            tiff.imwrite(out_path, drr.astype(np.float32))
        print(f"Saved all {len(results)} DRRs to {out_dir}")
    elif real_img is not None:
        out_path = out_dir / f"drr_atten_{best_v:.4f}_axis{beam_axis}_BEST.tif"
        tiff.imwrite(out_path, best_drr.astype(np.float32))
        print(f"Saved only the best-scoring DRR to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xr", required=True, dest="xr_path",
                         help="path to registered.tif CT volume (the source projected into a synthetic XR)")
    parser.add_argument("--pxr", required=False, default=None, dest="pxr_path",
                         help="path to the real portable XR radiograph (tif/jpg/dcm), OR a "
                              "specimen folder of .dcm files if --view is given")
    parser.add_argument("--view", default=None, choices=["Face", "Profil"],
                         help="if --pxr is a folder with multiple views, pick this one "
                              "by SeriesDescription")
    parser.add_argument("--detector-size", type=int, default=512,
                         help="working detector resolution (square). Full DICOM res (3072) "
                              "will OOM, 512 is a safe default, raise cautiously and watch "
                              "the printed memory estimate.")
    parser.add_argument("--voxel-spacing-mm", type=float, default=None,
                         help="in-plane (X/Y) voxel spacing of the CT volume, in mm. "
                              "Read this from the volume's own TIFF resolution tag "
                              "(e.g. via ImageJ Image>Properties), do NOT reuse a value "
                              "from a different specimen. Overrides the ConeBeamGeometry "
                              "default (0.7224mm), which is likely wrong for your specimen.")
    parser.add_argument("--voxel-spacing-z-mm", type=float, default=None,
                         help="through-slice (Z) voxel spacing, in mm, if different from "
                              "the in-plane value (anisotropic volume). Defaults to the "
                              "same value as --voxel-spacing-mm if not given (isotropic).")
    parser.add_argument("--atten-min", type=float, default=0.002,
                         help="lower bound for the attenuation_scale search")
    parser.add_argument("--atten-max", type=float, default=0.2,
                         help="upper bound for the attenuation_scale search. Raise this if "
                              "the fitted value comes back pinned at the previous max.")
    parser.add_argument("--metric", default="ssim", choices=["ssim", "wasserstein", "ncc", "mi"])
    parser.add_argument("--sweep", default=None,
                         help="comma-separated attenuation_scale values to render and save "
                              "for manual visual comparison, skips the SSIM optimizer "
                              "entirely. e.g. --sweep 0.01,0.02,0.05,0.08,0.12,0.2")
    parser.add_argument("--fixed-attenuation", default=None,
                         help="one value, or comma-separated list of values, to test directly "
                              "instead of optimizing, e.g. --fixed-attenuation 0.02,0.04,0.06,0.08. "
                              "Skips the SSIM search, prints attenuation_scale/Distance for each, "
                              "then reports the best one. Only the best-scoring DRR is saved.")
    parser.add_argument("--sweep-out-dir", default="attenuation_sweep",
                         help="directory to save sweep outputs into")
    parser.add_argument("--save-all-sweep", action="store_true",
                         help="save every sweep value's DRR to disk, not just the best-scoring "
                              "one. Off by default to avoid cluttering the output dir.")
    parser.add_argument("--beam-axis", type=int, default=1, choices=[0, 1, 2],
                         help="which volume axis the beam travels along. Default (1) matches "
                              "the 'Face' projection convention used throughout the pipeline. "
                              "Try a different axis to approximate the 'Profil' view, this is "
                              "a guess at which axis corresponds to that real acquisition "
                              "angle, not confirmed, compare output shape against the real "
                              "Profil DICOM to check.")
    parser.add_argument("--out-drr", default=None,
                         help="path to save the calibrated DRR (parent dirs auto-created). "
                              "Defaults to calibration_outputs/<pxr folder name>_calibrated.tif")
    args = parser.parse_args()

    if not args.sweep and not args.pxr_path:
        parser.error("--pxr is required unless --sweep is given")

    if args.sweep:
        geometry = make_geometry(args.detector_size, args.voxel_spacing_mm, args.voxel_spacing_z_mm)
        vol = load_volume(args.xr_path)
        values = [float(v) for v in args.sweep.split(",")]

        real_img = None
        if args.pxr_path:
            pxr_path = args.pxr_path
            if args.view is not None:
                pxr_path = find_view(pxr_path, args.view)
                print(f"Selected view file: {pxr_path}")
            real_img = load_image(pxr_path)
            real_img = strip_saturated_band(real_img)
            print(f"Sweeping {len(values)} attenuation_scale values, scoring each "
                  f"against {pxr_path} with metric={args.metric}.")
        else:
            print(f"Sweeping {len(values)} attenuation_scale values, no --pxr given, "
                  f"no optimizer, just generating each for you to look at.")

        sweep(vol, geometry, values, args.sweep_out_dir, beam_axis=args.beam_axis,
              real_img=real_img, metric=args.metric, save_all=args.save_all_sweep)
    else:
        out_drr = args.out_drr
        if out_drr is None:
            specimen_name = Path(args.pxr_path).name  # works whether pxr_path is a folder or a file
            out_drr = str(Path("calibration_outputs") / f"{specimen_name}_calibrated.tif")

        fixed_atten = None
        if args.fixed_attenuation is not None:
            fixed_atten = [float(v) for v in args.fixed_attenuation.split(",")]

        main(args.xr_path, args.pxr_path, args.metric, out_drr, args.view,
             args.detector_size, args.voxel_spacing_mm, (args.atten_min, args.atten_max),
             args.voxel_spacing_z_mm, fixed_atten, args.beam_axis)