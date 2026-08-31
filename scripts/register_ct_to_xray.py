"""
Ground-truth-free 2D/3D registration of a CT volume to a real portable
X-ray radiograph.

===============================================================================
COORDINATE CONVENTION 
===============================================================================
- CT array axes: (z, y, x) = (axis 0, axis 1, axis 2). This matches the
  physical mm axes of the same name, axis order is NEVER silently swapped.
- Beam travels along z. Source at physical z = -sod_mm. Detector plane at
  physical z = sdd_mm - sod_mm. Detector spans x (columns) and y (rows).
- Rotation: R = Rz @ Ry @ Rx, angles in degrees, right-hand convention,
  applied about the volume's physical CENTER, not the array corner.
- Pose semantics, VERIFIED empirically (see test in the accompanying
  development notes): pose.rz = +theta produces the same DRR as
  physically pre-rotating the CT volume by -theta using
  scipy.ndimage.rotate's convention. This is mathematically consistent
  (moving the object by +theta is equivalent to moving the sampling
  frame by -theta, and the ray sampler applies the INVERSE pose to map
  world coordinates back into the CT's own frame). Worth knowing if you
  ever cross-check against a manual scipy.ndimage.rotate call.
- Translation: pose.tx/ty/tz in mm, positive tx moves the object toward
  higher detector-column values (verified empirically, see dev notes).

===============================================================================
IDENTIFIABILITY
===============================================================================
With a SINGLE 2D radiograph:
  - Depth translation (tz, along the beam) is weakly constrained, it
    trades off against apparent magnification (an object moved closer
    to the source and made physically smaller can look similar to one
    left in place). Do not over-interpret a confident tz value.
  - Out-of-plane rotation (rx, ry if beam is along z) is much less
    constrained than in-plane rotation (rz), since many different
    out-of-plane poses can produce visually similar silhouettes.
  - A high similarity score does NOT prove correct registration, only
    that the projected silhouette matches well, which can happen for
    wrong poses too (this was observed directly during development,
    a coordinate-descent-style search on a real specimen locked onto a
    confidently-scoring but visually wrong orientation more than once).
  - Repeated convergence across independent restarts is evidence of
    STABILITY, not of ACCURACY. Report it as such.

===============================================================================
DEPENDENCIES
===============================================================================
Required: numpy, scipy, tifffile, scikit-image, scikit-learn, matplotlib
Optional: cma (pip install cma), preferred optimizer if available, falls
back to scipy.optimize.differential_evolution otherwise.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import tifffile as tiff
from scipy.ndimage import map_coordinates, gaussian_filter
from scipy.optimize import differential_evolution
from skimage.filters import sobel_h, sobel_v
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import cma
    CMA_AVAILABLE = True
except ImportError:
    CMA_AVAILABLE = False


# ==============================================================================
# 1. I/O
# ==============================================================================

def load_ct(path, spacing_x, spacing_y, spacing_z):
    if not Path(path).exists():
        raise FileNotFoundError(f"CT file not found: {path}")
    vol = tiff.imread(path).astype(np.float64)
    if vol.ndim != 3:
        raise ValueError(f"CT must be 3D, got shape {vol.shape}")
    for name, s in [("spacing_x", spacing_x), ("spacing_y", spacing_y), ("spacing_z", spacing_z)]:
        if s is None or s <= 0:
            raise ValueError(f"CT {name} must be a positive number, got {s}")
    if not np.all(np.isfinite(vol)):
        raise ValueError("CT contains NaN or Inf values")
    print(f"CT shape: {vol.shape}")
    print(f"CT dtype: {vol.dtype}")
    print(f"CT intensity range: [{vol.min():.2f}, {vol.max():.2f}]")
    print(f"CT voxel spacing (z,y,x): ({spacing_z}, {spacing_y}, {spacing_x}) mm")
    return vol, (spacing_z, spacing_y, spacing_x)


def load_xray(path, spacing_x, spacing_y):
    if not Path(path).exists():
        raise FileNotFoundError(f"X-ray file not found: {path}")
    img = tiff.imread(path).astype(np.float64)
    if img.ndim != 2:
        raise ValueError(f"X-ray must be 2D, got shape {img.shape}")
    for name, s in [("spacing_x", spacing_x), ("spacing_y", spacing_y)]:
        if s is None or s <= 0:
            raise ValueError(f"X-ray {name} must be a positive number, got {s}")
    if not np.all(np.isfinite(img)):
        raise ValueError("X-ray contains NaN or Inf values")
    print(f"X-ray shape: {img.shape}")
    print(f"X-ray dtype: {img.dtype}")
    print(f"X-ray intensity range: [{img.min():.2f}, {img.max():.2f}]")
    print(f"X-ray pixel spacing (y,x): ({spacing_y}, {spacing_x}) mm")
    return img, (spacing_y, spacing_x)


# ==============================================================================
# 3. CT preprocessing
# ==============================================================================

def preprocess_ct(vol, clip_percentiles=None, normalize=False, downsample=1, mask_background=None):
    vol_proc = vol  # keep original untouched, work on a reference/copy as needed
    if clip_percentiles is not None:
        lo, hi = np.percentile(vol_proc, clip_percentiles)
        vol_proc = np.clip(vol_proc, lo, hi)
    if mask_background is not None:
        vol_proc = np.where(vol_proc > mask_background, vol_proc, 0.0)
    if downsample > 1:
        vol_proc = vol_proc[::downsample, ::downsample, ::downsample]
    if normalize:
        lo, hi = vol_proc.min(), vol_proc.max()
        if hi > lo:
            vol_proc = (vol_proc - lo) / (hi - lo)
    return vol_proc.copy()


# ==============================================================================
# 4. X-ray preprocessing
# ==============================================================================

def preprocess_xray(img, spacing, normalize=False, crop=None, gaussian_sigma=None,
                     compute_gradient=False, foreground_mask_thresh=None):
    img_proc = img.copy()
    spacing_out = spacing

    if crop is not None:
        r0, r1, c0, c1 = crop
        img_proc = img_proc[r0:r1, c0:c1]

    if gaussian_sigma is not None:
        img_proc = gaussian_filter(img_proc, sigma=gaussian_sigma)

    if normalize:
        lo, hi = img_proc.min(), img_proc.max()
        if hi > lo:
            img_proc = (img_proc - lo) / (hi - lo)

    mask = None
    if foreground_mask_thresh is not None:
        mask = img_proc > foreground_mask_thresh

    grad = None
    if compute_gradient:
        grad = np.hypot(sobel_h(img_proc), sobel_v(img_proc))

    # spacing is NOT changed here since we don't resize, only crop/filter.
    # If a caller resizes elsewhere, they must update spacing_out themselves.
    return img_proc, spacing_out, mask, grad


# ==============================================================================
# 5. DRR generation (pose-parameterized ray casting, Beer-Lambert)
# ==============================================================================

def rotation_matrix(rx_deg, ry_deg, rz_deg):
    rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg])
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def pose_to_transform(pose):
    """Returns (R, t): rotation matrix and translation vector (mm),
    explicit, not hidden inside an opaque object."""
    R = rotation_matrix(pose["rx"], pose["ry"], pose["rz"])
    t = np.array([pose["tx"], pose["ty"], pose["tz"]])
    return R, t


def generate_drr(ct_volume, ct_spacing, pose, projection_geometry, output_shape, detector_spacing):
    """Real ray-casting Beer-Lambert projector, NOT np.sum(ct, axis=...).
    Pose applied via inverse transform on ray-sampling coordinates (see
    module docstring for the verified sign convention), not by
    resampling the whole volume, this keeps per-call cost tractable
    during optimization."""
    sz, sy, sx = ct_spacing
    nz, ny, nx = ct_volume.shape
    sdd = projection_geometry["sdd_mm"]
    sod = projection_geometry["sod_mm"]

    R, t = pose_to_transform(pose)
    R_inv = R.T

    rows, cols = output_shape
    det_y = (np.arange(rows) - rows / 2) * detector_spacing
    det_x = (np.arange(cols) - cols / 2) * detector_spacing
    grid_y, grid_x = np.meshgrid(det_y, det_x, indexing="ij")

    src = np.array([0.0, 0.0, -sod])
    det_z = sdd - sod

    half_extent_z = (nz / 2) * sz
    n_samples = int(nz * 1.5)

    t_enter = (-half_extent_z - src[2]) / (det_z - src[2])
    t_exit = (half_extent_z - src[2]) / (det_z - src[2])
    t_param = np.linspace(t_enter, t_exit, n_samples)

    accum = np.zeros((rows, cols), dtype=np.float64)
    step_mm = abs(t_exit - t_enter) * abs(det_z - src[2]) / n_samples

    for tp in t_param:
        world_x = src[0] + tp * (grid_x - src[0])
        world_y = src[1] + tp * (grid_y - src[1])
        world_z = np.full_like(world_x, src[2] + tp * (det_z - src[2]))

        pts_world = np.stack([world_x, world_y, world_z], axis=-1)
        pts_ct_phys = (pts_world - t) @ R_inv.T

        vox_x = pts_ct_phys[..., 0] / sx + nx / 2
        vox_y = pts_ct_phys[..., 1] / sy + ny / 2
        vox_z = pts_ct_phys[..., 2] / sz + nz / 2

        sampled = map_coordinates(ct_volume, [vox_z, vox_y, vox_x], order=1,
                                   mode="constant", cval=0.0)
        accum += sampled * step_mm

    attenuation_scale = projection_geometry.get("attenuation_scale", 0.02)
    return 1.0 - np.exp(-accum * attenuation_scale)


def create_projection_geometry(sdd_mm, sod_mm, attenuation_scale=0.02):
    if sdd_mm <= 0 or sod_mm <= 0:
        raise ValueError("sdd_mm and sod_mm must be positive")
    if sod_mm >= sdd_mm:
        raise ValueError("sod_mm (source-to-object) must be less than sdd_mm (source-to-detector)")
    return {"sdd_mm": sdd_mm, "sod_mm": sod_mm, "attenuation_scale": attenuation_scale}


# ==============================================================================
# 6. Similarity metrics
# ==============================================================================

def _resize_to_match(a, b):
    """Nearest-shape-match via simple zoom, only used to compare a DRR
    and X-ray of slightly different sizes, does not touch physical
    spacing bookkeeping elsewhere."""
    from scipy.ndimage import zoom
    if a.shape == b.shape:
        return a, b
    factors = (b.shape[0] / a.shape[0], b.shape[1] / a.shape[1])
    return zoom(a, factors, order=1), b


def _normalize01(img):
    lo, hi = img.min(), img.max()
    if hi <= lo:
        return np.zeros_like(img)
    return (img - lo) / (hi - lo)


def compute_ncc(img_a, img_b):
    a, b = _resize_to_match(img_a, img_b)
    a, b = _normalize01(a), _normalize01(b)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    if denom == 0:
        return 0.0
    return float((a * b).sum() / denom)


def compute_grad_ncc(img_a, img_b):
    a, b = _resize_to_match(img_a, img_b)
    a, b = _normalize01(a), _normalize01(b)
    grad_a = np.hypot(sobel_h(a), sobel_v(a))
    grad_b = np.hypot(sobel_h(b), sobel_v(b))
    return compute_ncc(grad_a, grad_b)


def compute_mutual_information(img_a, img_b, bins=64):
    from sklearn.metrics import normalized_mutual_info_score
    a, b = _resize_to_match(img_a, img_b)
    a, b = _normalize01(a), _normalize01(b)
    a_binned = np.digitize(a.ravel(), np.linspace(0, 1, bins))
    b_binned = np.digitize(b.ravel(), np.linspace(0, 1, bins))
    return float(normalized_mutual_info_score(a_binned, b_binned))


METRIC_FUNCS = {"grad-ncc": compute_grad_ncc, "ncc": compute_ncc, "mi": compute_mutual_information}


# ==============================================================================
# 7. Registration objective + optimization
# ==============================================================================

def registration_objective(pose_vec, param_names, fixed_pose, ct_volume, ct_spacing,
                            geometry, output_shape, detector_spacing, real_img, metric_fn):
    pose = dict(fixed_pose)
    for name, val in zip(param_names, pose_vec):
        pose[name] = val
    drr = generate_drr(ct_volume, ct_spacing, pose, geometry, output_shape, detector_spacing)
    score = metric_fn(drr, real_img)
    return -score  # minimize negative similarity = maximize similarity


# Module-level globals for the CMA parallel path. Set once before a Pool
# is created, inherited by forked worker processes (Linux default) so
# the (potentially large) ct_volume/real_img arrays aren't re-pickled
# per candidate, same pattern used in search_orientation.py.
_POOL_PARAM_NAMES = None
_POOL_FIXED_POSE = None
_POOL_CT_VOLUME = None
_POOL_CT_SPACING = None
_POOL_GEOMETRY = None
_POOL_OUTPUT_SHAPE = None
_POOL_DETECTOR_SPACING = None
_POOL_REAL_IMG = None
_POOL_METRIC_FN = None


def _pool_objective(pose_vec):
    return registration_objective(
        pose_vec, _POOL_PARAM_NAMES, _POOL_FIXED_POSE, _POOL_CT_VOLUME, _POOL_CT_SPACING,
        _POOL_GEOMETRY, _POOL_OUTPUT_SHAPE, _POOL_DETECTOR_SPACING, _POOL_REAL_IMG,
        _POOL_METRIC_FN)


def optimize_pose(param_names, bounds, fixed_pose, ct_volume, ct_spacing, geometry,
                   output_shape, detector_spacing, real_img, metric_fn, maxiter=60,
                   x0=None, optimizer="auto", workers=1):
    """Returns (best_pose_partial_dict, history, n_iters). history is a
    list of (iteration, best_score_so_far). workers>1 parallelizes
    candidate evaluation within each generation/iteration, not the
    overall optimization (which is inherently sequential, each
    generation depends on the last)."""
    global _POOL_PARAM_NAMES, _POOL_FIXED_POSE, _POOL_CT_VOLUME, _POOL_CT_SPACING
    global _POOL_GEOMETRY, _POOL_OUTPUT_SHAPE, _POOL_DETECTOR_SPACING, _POOL_REAL_IMG
    global _POOL_METRIC_FN

    history = []

    def objective(x):
        return registration_objective(x, param_names, fixed_pose, ct_volume, ct_spacing,
                                       geometry, output_shape, detector_spacing, real_img, metric_fn)

    if workers > 1:
        _POOL_PARAM_NAMES = param_names
        _POOL_FIXED_POSE = fixed_pose
        _POOL_CT_VOLUME = ct_volume
        _POOL_CT_SPACING = ct_spacing
        _POOL_GEOMETRY = geometry
        _POOL_OUTPUT_SHAPE = output_shape
        _POOL_DETECTOR_SPACING = detector_spacing
        _POOL_REAL_IMG = real_img
        _POOL_METRIC_FN = metric_fn

    use_cma = CMA_AVAILABLE and optimizer in ("auto", "cma")

    if use_cma:
        x0_arr = x0 if x0 is not None else [np.mean(b) for b in bounds]
        sigma0 = np.mean([b[1] - b[0] for b in bounds]) / 4
        es = cma.CMAEvolutionStrategy(x0_arr, sigma0, {
            "bounds": [[b[0] for b in bounds], [b[1] for b in bounds]],
            "maxiter": maxiter, "verbose": -9,
        })

        pool = None
        if workers > 1:
            import multiprocessing as mp
            # explicitly force 'fork' context: this pool is created (and
            # the _POOL_* globals set) in THIS same process, fork copies
            # current memory (COW) so workers see those globals. Do NOT
            # rely on whatever the environment's default start method
            # is, forkserver/spawn workers only see module-IMPORT-time
            # state, not runtime-set globals, confirmed by direct testing.
            pool = mp.get_context("fork").Pool(processes=workers)

        try:
            it = 0
            while not es.stop():
                solutions = es.ask()
                if pool is not None:
                    values = pool.map(_pool_objective, solutions)
                else:
                    values = [objective(s) for s in solutions]
                es.tell(solutions, values)
                it += 1
                history.append((it, -es.result.fbest))
        finally:
            if pool is not None:
                pool.close()
                pool.join()

        best_x = es.result.xbest
        n_iters = it
    else:
        de_workers = workers if workers > 1 else 1
        # Robust regardless of the environment's multiprocessing start
        # method: pass everything via args= (scipy pickles and sends
        # this properly to each worker), rather than relying on
        # module-level globals + fork's copy-on-write, which is NOT
        # guaranteed here, scipy's own internal Pool may use
        # forkserver/spawn depending on platform/environment.
        de_args = (param_names, fixed_pose, ct_volume, ct_spacing, geometry,
                   output_shape, detector_spacing, real_img, metric_fn)

        result = differential_evolution(
            registration_objective, bounds, args=de_args, maxiter=maxiter, polish=True,
            seed=None, workers=de_workers, updating="deferred" if de_workers > 1 else "immediate",
            callback=lambda xk, convergence: history.append(
                (len(history) + 1, -objective(xk))))
        best_x = result.x
        n_iters = result.nit

    best_pose_partial = {name: float(v) for name, v in zip(param_names, best_x)}
    return best_pose_partial, history, n_iters


# ==============================================================================
# 8/9. Multistage + multistart
# ==============================================================================

DEFAULT_BOUNDS = {
    "tx": (-20, 20), "ty": (-20, 20), "tz": (-30, 30),
    "rx": (-15, 15), "ry": (-180, 180), "rz": (-15, 15),
}


def run_multistart_registration(ct_volume, ct_spacing, geometry, real_img_full,
                                 detector_spacing_full, bounds, num_restarts, metric,
                                 downsample_stages, maxiter_per_stage, out_dir, workers=1):
    metric_fn = METRIC_FUNCS[metric]
    runs = []

    for run_id in range(num_restarts):
        print(f"\nRun {run_id + 1}/{num_restarts}")
        rng = np.random.default_rng(run_id)
        init_pose = {k: rng.uniform(lo, hi) for k, (lo, hi) in bounds.items()}
        print(f"Initial pose: {init_pose}")

        current_pose = dict(init_pose)
        run_start = time.time()
        combined_history = []

        for stage_i, (ds_factor, params_this_stage) in enumerate(downsample_stages, start=1):
            print(f"  Stage {stage_i}/{len(downsample_stages)} "
                  f"(downsample={ds_factor}x, optimizing {params_this_stage})...")
            vol_stage = ct_volume[::ds_factor, ::ds_factor, ::ds_factor] if ds_factor > 1 else ct_volume
            spacing_stage = tuple(s * ds_factor for s in ct_spacing)

            out_shape_stage = (max(32, real_img_full.shape[0] // ds_factor),
                                max(32, real_img_full.shape[1] // ds_factor))
            det_spacing_stage = detector_spacing_full * ds_factor
            real_stage = real_img_full[::ds_factor, ::ds_factor] if ds_factor > 1 else real_img_full

            stage_bounds = [bounds[p] for p in params_this_stage]
            x0 = [current_pose[p] for p in params_this_stage]

            best_partial, history, n_iters = optimize_pose(
                params_this_stage, stage_bounds, current_pose, vol_stage, spacing_stage,
                geometry, out_shape_stage, det_spacing_stage, real_stage, metric_fn,
                maxiter=maxiter_per_stage, x0=x0, workers=workers)

            current_pose.update(best_partial)
            combined_history.extend(history)
            print(f"    stage best: {best_partial}")

        run_time = time.time() - run_start

        initial_drr = generate_drr(ct_volume, ct_spacing, init_pose, geometry,
                                    real_img_full.shape, detector_spacing_full)
        final_drr = generate_drr(ct_volume, ct_spacing, current_pose, geometry,
                                  real_img_full.shape, detector_spacing_full)
        initial_sim = metric_fn(initial_drr, real_img_full)
        final_sim = metric_fn(final_drr, real_img_full)

        print(f"  Initial {metric}: {initial_sim:.4f}, Final {metric}: {final_sim:.4f}")

        runs.append({
            "run_id": run_id, "initial_pose": init_pose, "final_pose": current_pose,
            "initial_similarity": initial_sim, "final_similarity": final_sim,
            "n_iterations": sum(h[0] for h in [combined_history[-1]] if combined_history) or 0,
            "runtime_sec": run_time, "history": combined_history,
        })

        if out_dir:
            fig, ax = plt.subplots()
            if combined_history:
                iters, scores = zip(*combined_history)
                ax.plot(iters, scores)
            ax.set_xlabel("iteration")
            ax.set_ylabel(f"{metric} similarity")
            ax.set_title(f"Run {run_id} optimization curve")
            fig.savefig(Path(out_dir) / f"optimization_curve_run{run_id}.png", dpi=100)
            plt.close(fig)

    best_run = max(runs, key=lambda r: r["final_similarity"])
    return runs, best_run


# ==============================================================================
# 10B. Multi-start consistency
# ==============================================================================

def compute_multistart_consistency(runs):
    poses = [r["final_pose"] for r in runs]
    if len(poses) < 2:
        return {"note": "fewer than 2 runs, consistency not meaningful"}
    stats = {}
    for param in ["tx", "ty", "tz", "rx", "ry", "rz"]:
        vals = [p.get(param, 0.0) for p in poses]
        stats[param] = {"std": float(np.std(vals)), "values": vals}
    pairwise_diffs = []
    for i in range(len(poses)):
        for j in range(i + 1, len(poses)):
            diff = {p: poses[i].get(p, 0.0) - poses[j].get(p, 0.0)
                     for p in ["tx", "ty", "tz", "rx", "ry", "rz"]}
            pairwise_diffs.append({"run_i": i, "run_j": j, "diff": diff})
    return {"per_param_std": stats, "pairwise_diffs": pairwise_diffs}


# ==============================================================================
# 10C. Perturbation recovery / capture-range experiment
# ==============================================================================

def run_perturbation_test(reference_pose, ct_volume, ct_spacing, geometry, real_img,
                           detector_spacing, bounds, metric, translation_perturbations_mm,
                           rotation_perturbations_deg, maxiter, out_csv_path, workers=1):
    metric_fn = METRIC_FUNCS[metric]
    results = []

    perturbation_combos = []
    for t_mm in translation_perturbations_mm:
        perturbation_combos.append({"tx": t_mm, "ty": 0, "tz": 0, "rx": 0, "ry": 0, "rz": 0})
    for r_deg in rotation_perturbations_deg:
        perturbation_combos.append({"tx": 0, "ty": 0, "tz": 0, "rx": 0, "ry": r_deg, "rz": 0})

    for combo in perturbation_combos:
        perturbed_pose = {k: reference_pose.get(k, 0.0) + combo.get(k, 0.0)
                           for k in ["tx", "ty", "tz", "rx", "ry", "rz"]}

        params = list(bounds.keys())
        x0 = [perturbed_pose[p] for p in params]
        stage_bounds = [bounds[p] for p in params]

        best_partial, history, n_iters = optimize_pose(
            params, stage_bounds, perturbed_pose, ct_volume, ct_spacing, geometry,
            real_img.shape, detector_spacing, real_img, metric_fn, maxiter=maxiter, x0=x0,
            workers=workers)

        recovered_pose = dict(perturbed_pose)
        recovered_pose.update(best_partial)

        diff = {p: recovered_pose[p] - reference_pose.get(p, 0.0)
                for p in ["tx", "ty", "tz", "rx", "ry", "rz"]}
        diff_mag = np.sqrt(sum(v ** 2 for v in diff.values()))

        final_drr = generate_drr(ct_volume, ct_spacing, recovered_pose, geometry,
                                  real_img.shape, detector_spacing)
        final_sim = metric_fn(final_drr, real_img)

        success = diff_mag < 5.0  # arbitrary combined threshold, report raw diff regardless
        results.append({
            "applied_perturbation": combo, "recovered_pose": recovered_pose,
            "diff_from_reference": diff, "final_similarity": final_sim, "success": success,
        })
        print(f"  perturbation {combo} -> diff_mag={diff_mag:.2f}, "
              f"final_sim={final_sim:.4f}, success={success}")

    if out_csv_path:
        with open(out_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["perturbation", "recovered_pose", "diff", "final_similarity", "success"])
            for r in results:
                writer.writerow([json.dumps(r["applied_perturbation"]),
                                  json.dumps(r["recovered_pose"]),
                                  json.dumps(r["diff_from_reference"]),
                                  r["final_similarity"], r["success"]])

    return results


# ==============================================================================
# 11. Visual evaluation outputs
# ==============================================================================

def save_comparison_figures(real_img, initial_drr, final_drr, out_dir):
    out_dir = Path(out_dir)

    def overlay(a, b):
        a_n, b_n = _normalize01(a), _normalize01(b)
        a_n, b_n = _resize_to_match(a_n, b_n)
        rgb = np.zeros((*b_n.shape, 3))
        rgb[..., 0] = a_n
        rgb[..., 1] = b_n
        return rgb

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(_normalize01(real_img), cmap="gray"); ax[0].set_title("real X-ray"); ax[0].axis("off")
    ax[1].imshow(_normalize01(initial_drr), cmap="gray"); ax[1].set_title("initial DRR"); ax[1].axis("off")
    ax[2].imshow(overlay(initial_drr, real_img)); ax[2].set_title("overlay"); ax[2].axis("off")
    fig.savefig(out_dir / "overlay_before.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(_normalize01(real_img), cmap="gray"); ax[0].set_title("real X-ray"); ax[0].axis("off")
    ax[1].imshow(_normalize01(final_drr), cmap="gray"); ax[1].set_title("registered DRR"); ax[1].axis("off")
    ax[2].imshow(overlay(final_drr, real_img)); ax[2].set_title("overlay"); ax[2].axis("off")
    fig.savefig(out_dir / "overlay_after.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    def edge_fig(drr, real, path):
        drr_n, real_n = _resize_to_match(_normalize01(drr), _normalize01(real))
        edge_drr = np.hypot(sobel_h(drr_n), sobel_v(drr_n))
        edge_real = np.hypot(sobel_h(real_n), sobel_v(real_n))
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(edge_real, cmap="gray"); ax[0].set_title("real X-ray edges"); ax[0].axis("off")
        ax[1].imshow(edge_drr, cmap="gray"); ax[1].set_title("DRR edges"); ax[1].axis("off")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)

    edge_fig(initial_drr, real_img, out_dir / "edges_before.png")
    edge_fig(final_drr, real_img, out_dir / "edges_after.png")

    def diff_fig(drr, real, path):
        drr_n, real_n = _resize_to_match(_normalize01(drr), _normalize01(real))
        fig, ax = plt.subplots()
        im = ax.imshow(drr_n - real_n, cmap="RdBu", vmin=-1, vmax=1)
        ax.set_title("DRR - real X-ray"); ax.axis("off")
        fig.colorbar(im)
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)

    diff_fig(initial_drr, real_img, out_dir / "difference_before.png")
    diff_fig(final_drr, real_img, out_dir / "difference_after.png")


# ==============================================================================
# 10A / 13. Evaluation + saving
# ==============================================================================

def evaluate_registration(ct_volume, ct_spacing, geometry, real_img, detector_spacing,
                           initial_pose, final_pose):
    initial_drr = generate_drr(ct_volume, ct_spacing, initial_pose, geometry,
                                real_img.shape, detector_spacing)
    final_drr = generate_drr(ct_volume, ct_spacing, final_pose, geometry,
                              real_img.shape, detector_spacing)
    rows = []
    for name, fn in METRIC_FUNCS.items():
        rows.append({"metric": name, "before": fn(initial_drr, real_img),
                      "after": fn(final_drr, real_img)})
    return rows, initial_drr, final_drr


def save_results(out_dir, best_run, runs, consistency, perturbation_results,
                  final_drr, initial_drr, similarity_rows, geometry, ct_spacing,
                  xray_spacing, metric):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tiff.imwrite(out_dir / "registered_drr.tif", final_drr.astype(np.float32))
    tiff.imwrite(out_dir / "initial_drr.tif", initial_drr.astype(np.float32))

    pose_json = {
        "translation_mm": {"x": best_run["final_pose"].get("tx", 0.0),
                            "y": best_run["final_pose"].get("ty", 0.0),
                            "z": best_run["final_pose"].get("tz", 0.0)},
        "rotation_deg": {"x": best_run["final_pose"].get("rx", 0.0),
                          "y": best_run["final_pose"].get("ry", 0.0),
                          "z": best_run["final_pose"].get("rz", 0.0)},
    }
    with open(out_dir / "registered_pose.json", "w") as f:
        json.dump(pose_json, f, indent=2)

    with open(out_dir / "registration_runs.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "initial_pose", "final_pose", "initial_similarity",
                          "final_similarity", "n_iterations", "runtime_sec"])
        for r in runs:
            writer.writerow([r["run_id"], json.dumps(r["initial_pose"]),
                              json.dumps(r["final_pose"]), r["initial_similarity"],
                              r["final_similarity"], r["n_iterations"], r["runtime_sec"]])

    summary = {
        "best_run_id": best_run["run_id"], "best_final_pose": best_run["final_pose"],
        "best_final_similarity": best_run["final_similarity"], "metric_used": metric,
        "ct_spacing_mm": ct_spacing, "xray_spacing_mm": xray_spacing,
        "geometry": geometry, "multistart_consistency": consistency,
        "note": "high similarity does not prove anatomically correct registration, "
                "see module docstring for identifiability caveats",
    }
    with open(out_dir / "registration_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    with open(out_dir / "similarity_before_after.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "before", "after"])
        for row in similarity_rows:
            writer.writerow([row["metric"], row["before"], row["after"]])

    print(f"\nResults saved to {out_dir}")


# ==============================================================================
# main / CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ct", required=True)
    parser.add_argument("--xray", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--ct-spacing-x", type=float, required=True)
    parser.add_argument("--ct-spacing-y", type=float, required=True)
    parser.add_argument("--ct-spacing-z", type=float, required=True)
    parser.add_argument("--xray-spacing-x", type=float, required=True)
    parser.add_argument("--xray-spacing-y", type=float, required=True)

    parser.add_argument("--source-to-detector-distance", type=float, required=True)
    parser.add_argument("--source-to-object-distance", type=float, required=True)
    parser.add_argument("--attenuation-scale", type=float, default=0.02)

    parser.add_argument("--metric", choices=list(METRIC_FUNCS.keys()), default="grad-ncc")

    parser.add_argument("--tx-range", type=float, nargs=2, default=DEFAULT_BOUNDS["tx"])
    parser.add_argument("--ty-range", type=float, nargs=2, default=DEFAULT_BOUNDS["ty"])
    parser.add_argument("--tz-range", type=float, nargs=2, default=DEFAULT_BOUNDS["tz"])
    parser.add_argument("--rx-range", type=float, nargs=2, default=DEFAULT_BOUNDS["rx"])
    parser.add_argument("--ry-range", type=float, nargs=2, default=DEFAULT_BOUNDS["ry"])
    parser.add_argument("--rz-range", type=float, nargs=2, default=DEFAULT_BOUNDS["rz"])

    parser.add_argument("--num-restarts", type=int, default=5)
    parser.add_argument("--maxiter-per-stage", type=int, default=40)
    parser.add_argument("--workers", type=int, default=1,
                         help="parallel workers for candidate evaluation within each "
                              "optimizer generation/iteration. With CMA-ES this is "
                              "manually parallelized; with the differential_evolution "
                              "fallback it uses scipy's own built-in workers support.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                         help="cuda not implemented in this baseline, cpu only for now")

    parser.add_argument("--ct-downsample", type=int, default=1)
    parser.add_argument("--ct-clip-percentiles", type=float, nargs=2, default=None)
    parser.add_argument("--ct-normalize", action="store_true")
    parser.add_argument("--ct-mask-background", type=float, default=None)

    parser.add_argument("--xray-normalize", action="store_true")
    parser.add_argument("--xray-gaussian-sigma", type=float, default=None)

    parser.add_argument("--translation-perturbations-mm", type=float, nargs="+",
                         default=[2, 5, 10])
    parser.add_argument("--rotation-perturbations-deg", type=float, nargs="+",
                         default=[1, 3, 5, 10])

    args = parser.parse_args()

    if args.device == "cuda":
        print("NOTE: --device cuda requested but not implemented in this baseline, "
              "running on CPU. GPU support was not added here to keep the first "
              "implementation transparent and easy to verify.")

    bounds = {
        "tx": tuple(args.tx_range), "ty": tuple(args.ty_range), "tz": tuple(args.tz_range),
        "rx": tuple(args.rx_range), "ry": tuple(args.ry_range), "rz": tuple(args.rz_range),
    }

    print("REGISTERING SPECIMEN")
    print("====================\n")
    print(f"CT: {args.ct}")
    print(f"X-ray: {args.xray}\n")

    ct_volume, ct_spacing = load_ct(args.ct, args.ct_spacing_x, args.ct_spacing_y, args.ct_spacing_z)
    real_img, xray_spacing = load_xray(args.xray, args.xray_spacing_x, args.xray_spacing_y)

    ct_volume_proc = preprocess_ct(ct_volume, args.ct_clip_percentiles, args.ct_normalize,
                                    args.ct_downsample, args.ct_mask_background)
    ct_spacing_proc = tuple(s * args.ct_downsample for s in ct_spacing)

    real_img_proc, xray_spacing_proc, _, _ = preprocess_xray(
        real_img, xray_spacing, args.xray_normalize, None, args.xray_gaussian_sigma)

    if not CMA_AVAILABLE:
        print("\nNOTE: 'cma' package not found, falling back to "
              "scipy.optimize.differential_evolution. For the preferred optimizer, "
              "run: pip install cma\n")

    geometry = create_projection_geometry(args.source_to_detector_distance,
                                           args.source_to_object_distance,
                                           args.attenuation_scale)

    detector_spacing_full = xray_spacing_proc[1]  # assume square pixels (x spacing)

    downsample_stages = [
        (4, ["tx", "ty", "rz"]),
        (2, ["tx", "ty", "tz", "rx", "ry", "rz"]),
        (1, ["tx", "ty", "tz", "rx", "ry", "rz"]),
    ]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs, best_run = run_multistart_registration(
        ct_volume_proc, ct_spacing_proc, geometry, real_img_proc, detector_spacing_full,
        bounds, args.num_restarts, args.metric, downsample_stages, args.maxiter_per_stage,
        out_dir, workers=args.workers)

    consistency = compute_multistart_consistency(runs)

    zero_pose = {k: 0.0 for k in ["tx", "ty", "tz", "rx", "ry", "rz"]}
    similarity_rows, initial_drr, final_drr = evaluate_registration(
        ct_volume_proc, ct_spacing_proc, geometry, real_img_proc, detector_spacing_full,
        zero_pose, best_run["final_pose"])

    print("\nRunning perturbation recovery test around best pose...")
    perturbation_results = run_perturbation_test(
        best_run["final_pose"], ct_volume_proc, ct_spacing_proc, geometry, real_img_proc,
        detector_spacing_full, bounds, args.metric, args.translation_perturbations_mm,
        args.rotation_perturbations_deg, args.maxiter_per_stage,
        out_dir / "perturbation_recovery.csv", workers=args.workers)

    save_comparison_figures(real_img_proc, initial_drr, final_drr, out_dir)

    save_results(out_dir, best_run, runs, consistency, perturbation_results,
                 final_drr, initial_drr, similarity_rows, geometry, ct_spacing_proc,
                 xray_spacing_proc, args.metric)

    print(f"\nFinal pose: {best_run['final_pose']}")
    print(f"Final {args.metric}: {best_run['final_similarity']:.4f}")


if __name__ == "__main__":
    main()