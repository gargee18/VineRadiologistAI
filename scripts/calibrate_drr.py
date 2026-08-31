import argparse
from pathlib import Path

import numpy as np
import tifffile as tiff
from PIL import Image
from scipy.optimize import minimize_scalar
from scipy.stats import wasserstein_distance

from VineRadiologist.cone_beam import ConeBeamGeometry, generate_cone_beam_drr, generate_cone_beam_drr_gpu
from VineRadiologist.io import load_volume


def make_geometry(detector_size: int = 512, voxel_spacing_mm: float = None,
                   voxel_spacing_z_mm: float = None, sid_mm: float = 1230.0,
                   spd_mm: float = 800.0, offset_v_mm: float = 0.0,
                   offset_u_mm: float = 0.0) -> ConeBeamGeometry:
    full_res = 3072
    full_spacing = 0.139
    physical_extent_mm = full_res * full_spacing
    scaled_spacing = physical_extent_mm / detector_size

    inplane = voxel_spacing_mm if voxel_spacing_mm is not None else 0.7224
    z = voxel_spacing_z_mm if voxel_spacing_z_mm is not None else inplane

    spacing = (z, inplane, inplane)

    return ConeBeamGeometry(
        sid_mm=sid_mm,
        spd_mm=spd_mm,
        detector_pixel_spacing_mm=scaled_spacing,
        detector_shape=(detector_size, detector_size),
        voxel_spacing_mm=spacing,
        offset_v_mm=offset_v_mm,
        offset_u_mm=offset_u_mm,
    )


def load_image(path: str) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".dcm":
        import pydicom
        ds = pydicom.dcmread(str(path), force=True)
        pixels = ds.pixel_array.astype(np.float64)
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            pixels = pixels.max() - pixels
        return pixels
    if path.suffix.lower() in (".tif", ".tiff"):
        return tiff.imread(path).astype(np.float64)
    return np.array(Image.open(path).convert("L"), dtype=np.float64)


def find_view(specimen_dir: str, view: str = "Face") -> str:
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
                          debug: bool = True):
    """Returns (cropped_img, crop_row). crop_row is how many rows got
    removed from the top, needed so the SAME crop can be applied to the
    DRR later, keeping both images genuinely aligned to the real
    detector's post-crop region, not just resized to match shapes."""
    max_val = img.max()
    if max_val <= 0:
        return img, 0
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
        return img, 0

    if crop_row > 0:
        print(f"Stripped {crop_row} saturated row(s) from top of real image "
              f"({crop_row}/{img.shape[0]} = {100*crop_row/img.shape[0]:.1f}%)")
        return img[crop_row:], crop_row
    return img, 0


def normalize(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float64)
    lo, hi = img.min(), img.max()
    if hi <= lo:
        return np.zeros_like(img)
    return (img - lo) / (hi - lo)


def resize_to_match(img: np.ndarray, target_shape: tuple) -> np.ndarray:
    from scipy.ndimage import zoom
    if img.shape == target_shape:
        return img
    factors = (target_shape[0] / img.shape[0], target_shape[1] / img.shape[1])
    return zoom(img, factors, order=1)


def auto_crop_to_content(sim: np.ndarray, real: np.ndarray, threshold: float = 0.05):
    sim_n = normalize(sim)
    mask = sim_n > threshold
    if not mask.any():
        return sim, real
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    pad = int(0.05 * max(sim.shape))
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
        bins = 64
        r_binned = np.digitize(real_n.ravel(), np.linspace(0, 1, bins))
        s_binned = np.digitize(sim_n.ravel(), np.linspace(0, 1, bins))
        nmi = normalized_mutual_info_score(r_binned, s_binned)
        return 1.0 - nmi

    raise ValueError(f"unknown metric: {metric}")


def generate_fixed(vol: np.ndarray, real_img: np.ndarray, geometry: ConeBeamGeometry,
                    attenuation_scale: float, metric: str = "ssim", beam_axis: int = 1,
                    use_gpu: bool = False, row_chunk: int = 16, crop_top_rows: int = 0) -> dict:
    if use_gpu:
        drr = generate_cone_beam_drr_gpu(vol, geometry, attenuation_scale=attenuation_scale,
                                          beam_axis=beam_axis, row_chunk=row_chunk)
    else:
        drr = generate_cone_beam_drr(vol, geometry, attenuation_scale=attenuation_scale, beam_axis=beam_axis)
    if crop_top_rows > 0:
        drr = drr[crop_top_rows:]
    dist = compute_distance(drr, real_img, metric=metric)
    return {"attenuation_scale": attenuation_scale, "distance": dist, "metric": metric, "drr": drr}


def calibrate(vol: np.ndarray, real_img: np.ndarray, geometry: ConeBeamGeometry,
              metric: str = "ssim", bounds=(0.002, 0.2), use_gpu: bool = False,
              row_chunk: int = 16, crop_top_rows: int = 0) -> dict:
    history = []

    def objective(atten_scale):
        if use_gpu:
            drr = generate_cone_beam_drr_gpu(vol, geometry, attenuation_scale=atten_scale,
                                              row_chunk=row_chunk, verbose=False)
        else:
            drr = generate_cone_beam_drr(vol, geometry, attenuation_scale=atten_scale)
        if crop_top_rows > 0:
            drr = drr[crop_top_rows:]
        dist = compute_distance(drr, real_img, metric=metric)
        history.append((atten_scale, dist))
        return dist

    result = minimize_scalar(objective, bounds=bounds, method="bounded",
                              options={"xatol": 1e-5})

    if use_gpu:
        best_drr = generate_cone_beam_drr_gpu(vol, geometry, attenuation_scale=result.x,
                                               row_chunk=row_chunk)
    else:
        best_drr = generate_cone_beam_drr(vol, geometry, attenuation_scale=result.x)
    if crop_top_rows > 0:
        best_drr = best_drr[crop_top_rows:]

    return {
        "attenuation_scale": result.x,
        "distance": result.fun,
        "metric": metric,
        "drr": best_drr,
        "history": history,
    }


def save_drr_16bit(path, drr_float, pixel_spacing_mm=None):
    """DRR values are floats in [0,1] (Beer-Lambert output). Real PXRs
    are 16-bit. Scale to the full 16-bit range so plot profiles are on
    a comparable intensity scale, don't just cast float->uint16 directly
    (that would truncate everything to 0 or 1).

    pixel_spacing_mm: writes real resolution metadata into the TIFF, so
    it doesn't come out as the placeholder 'Pixel width: 1.0000' seen
    before. Same fix as dicom_to_raw_tiff.py needed."""
    scaled = np.clip(drr_float, 0, 1) * 65535
    if pixel_spacing_mm:
        res = 10.0 / pixel_spacing_mm  # pixels per cm
        tiff.imwrite(path, scaled.astype(np.uint16), resolution=(res, res),
                     resolutionunit="CENTIMETER")
    else:
        print(f"  WARNING: no pixel_spacing_mm given, {path} will have no "
              f"resolution metadata (placeholder-only file)")
        tiff.imwrite(path, scaled.astype(np.uint16))


def main(xr_path, pxr_path, metric, out_drr, view=None, detector_size=512,
         voxel_spacing_mm=None, atten_bounds=(0.002, 0.2), voxel_spacing_z_mm=None,
         fixed_attenuation=None, beam_axis=1, sid_mm=1230.0, spd_mm=800.0,
         offset_v_mm=0.0, offset_u_mm=0.0, use_gpu=False, row_chunk=16):
    geometry = make_geometry(detector_size, voxel_spacing_mm, voxel_spacing_z_mm, sid_mm, spd_mm,
                              offset_v_mm, offset_u_mm)

    vol = load_volume(xr_path)
    if view is not None:
        pxr_path = find_view(pxr_path, view)
        print(f"Selected view file: {pxr_path}")
    real_img = load_image(pxr_path)
    real_img, _ = strip_saturated_band(real_img)
    crop_top_rows = max(0, detector_size - real_img.shape[0])
    if crop_top_rows > 0:
        print(f"Real PXR is already {real_img.shape[0]} rows, cropping "
              f"{crop_top_rows} rows off the top of the {detector_size}-row DRR to match.")

    n_beam_est = vol.shape[1] * 2
    est_gb = n_beam_est * detector_size * detector_size * 8 / 1e9
    print(f"XR (CT volume): {xr_path}  shape={vol.shape}")
    print(f"PXR (real radiograph): {pxr_path}  shape={real_img.shape}")
    print(f"Geometry: sid={geometry.sid_mm}mm  spd={geometry.spd_mm}mm  "
          f"pixel_spacing={geometry.detector_pixel_spacing_mm:.4f}mm  "
          f"detector={geometry.detector_shape}")
    if use_gpu:
        print(f"Using GPU path (row_chunk={row_chunk}), memory estimate printed per-call below.")
    else:
        print(f"Estimated peak memory for ray casting: ~{est_gb:.1f} GB "
              f"(reduce --detector-size if this is too high for your machine, "
              f"or pass --gpu to use the chunked GPU path instead)")
    if fixed_attenuation is not None:
        atten_values = fixed_attenuation if isinstance(fixed_attenuation, list) else [fixed_attenuation]
        results = []
        for v in atten_values:
            r = generate_fixed(vol, real_img, geometry, v, metric=metric, beam_axis=beam_axis,
                                use_gpu=use_gpu, row_chunk=row_chunk, crop_top_rows=crop_top_rows)
            results.append(r)
            print(f"attenuation_scale: {r['attenuation_scale']:.6f}")
            print(f"Distance ({metric}): {r['distance']:.6f}")
            print()
        result = min(results, key=lambda r: r["distance"])
        if len(results) > 1:
            print(f"Best: attenuation_scale={result['attenuation_scale']:.6f}  "
                  f"distance={result['distance']:.6f}")
    else:
        print(f"Fitting attenuation_scale with metric={metric} ...")
        result = calibrate(vol, real_img, geometry, metric=metric, bounds=atten_bounds,
                            use_gpu=use_gpu, row_chunk=row_chunk, crop_top_rows=crop_top_rows)
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
        save_drr_16bit(out_path, result["drr"], geometry.detector_pixel_spacing_mm)
        print(f"Calibrated DRR written to {out_path}")


def sweep(vol, geometry, values, out_dir, beam_axis=1, real_img=None, metric="ssim",
          save_all=False, use_gpu=False, row_chunk=16, crop_top_rows=0):
    results = []
    for v in values:
        if use_gpu:
            drr = generate_cone_beam_drr_gpu(vol, geometry, attenuation_scale=v,
                                              beam_axis=beam_axis, row_chunk=row_chunk)
        else:
            drr = generate_cone_beam_drr(vol, geometry, attenuation_scale=v, beam_axis=beam_axis)
        if crop_top_rows > 0:
            drr = drr[crop_top_rows:]
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
            out_path = out_dir / f"drr_atten_{v:.4f}.tif"
            save_drr_16bit(out_path, drr, geometry.detector_pixel_spacing_mm)
        print(f"Saved all {len(results)} DRRs to {out_dir}")
    elif real_img is not None:
        out_path = out_dir / f"drr_atten_{best_v:.4f}_BEST.tif"
        save_drr_16bit(out_path, best_drr, geometry.detector_pixel_spacing_mm)
        print(f"Saved only the best-scoring DRR to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xr", required=True, dest="xr_path")
    parser.add_argument("--pxr", required=False, default=None, dest="pxr_path")
    parser.add_argument("--view", default=None, choices=["Face", "Profil"])
    parser.add_argument("--detector-size", type=int, default=512)
    parser.add_argument("--voxel-spacing-mm", type=float, default=None)
    parser.add_argument("--voxel-spacing-z-mm", type=float, default=None)
    parser.add_argument("--sid-mm", type=float, default=1230.0)
    parser.add_argument("--spd-mm", type=float, default=800.0)
    parser.add_argument("--offset-v-mm", type=float, default=0.0)
    parser.add_argument("--offset-u-mm", type=float, default=0.0)
    parser.add_argument("--atten-min", type=float, default=0.002)
    parser.add_argument("--atten-max", type=float, default=0.2)
    parser.add_argument("--metric", default="ssim", choices=["ssim", "wasserstein", "ncc", "mi"])
    parser.add_argument("--sweep", default=None)
    parser.add_argument("--fixed-attenuation", default=None)
    parser.add_argument("--sweep-out-dir", default="attenuation_sweep")
    parser.add_argument("--save-all-sweep", action="store_true")
    parser.add_argument("--beam-axis", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--out-drr", default=None)
    parser.add_argument("--gpu", action="store_true",
                         help="use the GPU-chunked cone-beam projector (needs cupy installed) "
                              "instead of the CPU version, lets you run at full PXR resolution "
                              "(e.g. --detector-size 3072) without OOMing")
    parser.add_argument("--row-chunk", type=int, default=16,
                         help="only used with --gpu, how many detector rows to process at "
                              "once, lower = less memory per chunk but more chunks/slower, "
                              "raise if you have headroom, lower if you still OOM")
    args = parser.parse_args()

    if not args.sweep and not args.pxr_path:
        parser.error("--pxr is required unless --sweep is given")

    if args.sweep:
        geometry = make_geometry(args.detector_size, args.voxel_spacing_mm, args.voxel_spacing_z_mm,
                                  args.sid_mm, args.spd_mm, args.offset_v_mm, args.offset_u_mm)
        vol = load_volume(args.xr_path)
        values = [float(v) for v in args.sweep.split(",")]

        real_img = None
        if args.pxr_path:
            pxr_path = args.pxr_path
            if args.view is not None:
                pxr_path = find_view(pxr_path, args.view)
                print(f"Selected view file: {pxr_path}")
            real_img = load_image(pxr_path)
            real_img, _ = strip_saturated_band(real_img)
            print(f"Sweeping {len(values)} attenuation_scale values, scoring each "
                  f"against {pxr_path} with metric={args.metric}.")
        else:
            print(f"Sweeping {len(values)} attenuation_scale values, no --pxr given, "
                  f"no optimizer, just generating each for you to look at.")

        sweep(vol, geometry, values, args.sweep_out_dir, beam_axis=args.beam_axis,
              real_img=real_img, metric=args.metric, save_all=args.save_all_sweep,
              use_gpu=args.gpu, row_chunk=args.row_chunk)
    else:
        out_drr = args.out_drr
        if out_drr is None:
            specimen_name = Path(args.pxr_path).name
            out_drr = str(Path("calibration_outputs") / f"{specimen_name}_calibrated.tif")

        fixed_atten = None
        if args.fixed_attenuation is not None:
            fixed_atten = [float(v) for v in args.fixed_attenuation.split(",")]

        main(args.xr_path, args.pxr_path, args.metric, out_drr, args.view,
             args.detector_size, args.voxel_spacing_mm, (args.atten_min, args.atten_max),
             args.voxel_spacing_z_mm, fixed_atten, args.beam_axis,
             args.sid_mm, args.spd_mm, args.offset_v_mm, args.offset_u_mm,
             args.gpu, args.row_chunk)