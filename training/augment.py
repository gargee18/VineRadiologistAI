#!/usr/bin/env python3
"""
augment.py  -  seeded 3D augmentation for the D2 training views

Every augmented view is defined by ONE integer seed. The seed fixes all random
choices, so a view is never saved: it is regenerated from its seed whenever it
is needed, bit for bit the same.

What one view is made of, in this order:
  1. deformation of the 3D volume (CT and the 3 masks with the SAME field):
       bend   : the trunk curves sideways, ends shifted up to BEND_MAX_MM
       wobble : small smooth random displacement, up to ELASTIC_MAX_MM
  2. projection with random geometry (CT and masks with the SAME geometry):
       yaw (around the trunk), pitch and roll (small tilts),
       source distance spd, detector offsets u and v
  3. CT projection -> linear DRR -> 16-bit -> your DRR->PXR curve,
     exactly as calibrate_drr.py --linear + fit_drr_to_pxr.py apply do
     (USE_CURVE = False skips the curve: plain linear DRR, 0..1)
  4. 2D noise on the DRR only (never on the labels)

The volumes are padded with air so that any yaw fits inside the projector's
sampling range (calibrate_drr.py only samples the depth of the unrotated
volume).

USAGE (from the repo root)
    # look at augmented views; also prints how long one view takes
    python training/augment.py preview --vine CEP016_RES3 --seed 1 2 3

    # sanity check: the un-augmented view must match your existing DRR/labels
    python training/augment.py check --vine CEP016_RES3

    # measure the noise level of a real radiograph, to set NOISE_STD below
    python training/augment.py noise dataset/radiograph_portable_2026_tif/<file>.tif
"""

import argparse
import os
import sys
import time

import numpy as np
import tifffile

REPO = "/mnt/41d6c007-0c9e-41e2-b2eb-8d9c032e9e53/gargee/code_python/VineRadiologistAI"
sys.path.insert(0, os.path.join(REPO, "scripts"))
from calibrate_drr import make_geometry, generate_drr, CUPY_AVAILABLE  # noqa: E402
from VineRadiologist.io import load_volume                               # noqa: E402

# =========================================================================
# DATA AND PROJECTION SETTINGS (same as the D2 DRRs and labels)
# =========================================================================

CT_DIR = os.path.join(REPO, "dataset", "ct_2019_cropped")
CURVE = os.path.join(REPO, "results", "curves", "curve_quantile_airzero_20spec_norm.npz")
CANON_DRR_DIR = os.path.join(REPO, "results", "d2_cropped_drr")       # for `check`
CANON_LAB_DIR = os.path.join(REPO, "results", "d2_cropped_labels")    # for `check`
PREVIEW_DIR = os.path.join(REPO, "results", "augment_preview")

SPACING_MM = 0.7224                 # isotropic voxel size
SID_MM, SPD_MM = 1230.0, 800.0
BEAM_AXIS = 1
DETECTOR = 3072
ATT_CT = 0.0975                     # k6.5
ATT_MASK = 1000.0 / 255.0           # loader divides by 1000, masks are 0/255 -> mm
MASK_TYPES = ["SAIN", "NECROSE", "AMADOU"]   # healthy, necrosis, white rot

# True : linear DRR -> 16-bit -> D1-fitted DRR->PXR curve  (= *_k6.5_mapped_norm.tif)
# False: plain linear DRR 1 - exp(-A), 16-bit quantized, scaled 0..1 (= *_k6.5.tif / 65535)
# The training's test images must be made the same way: the matching file suffix
# is DRR_SUFFIX below.
USE_CURVE = True
DRR_SUFFIX = "_k6.5_mapped_norm.tif" if USE_CURVE else "_k6.5.tif"

# =========================================================================
# AUGMENTATION RANGES  (each value is drawn uniformly in its range)
# =========================================================================

YAW_DEG = (0.0, 360.0)              # around the trunk
PITCH_DEG = (-10.0, 10.0)           # tilts
ROLL_DEG = (-15.0, 15.0)
SPD_DELTA_MM = (-30.0, 30.0)        # added to SPD_MM; SID stays fixed
OFFSET_U_MM = (-20.0, 20.0)         # sideways shift
OFFSET_V_MM = (-20.0, 20.0)         # vertical shift
BEND_MAX_MM = 10.0                  # sideways shift of the trunk ends, 0..this
ELASTIC_MAX_MM = 1.5                # maximum wobble displacement, 0..this
ELASTIC_SCALE_MM = 25.0             # size of the wobble bumps (bigger = smoother)
NOISE_STD = (0.0011, 0.0053)            # in curve output units; measure with `noise`
                                    # (0, 0) = noise off

PAD_MARGIN_MM = 25.0                # extra air around the volume for bend and wobble;
                                    # along the beam the tilt is added on top, see
                                    # pad_for_rotation

# =========================================================================
# PARAMETERS FROM A SEED
# =========================================================================

CANONICAL = {"seed": None, "yaw": 0.0, "pitch": 0.0, "roll": 0.0, "spd": SPD_MM,
             "off_u": 0.0, "off_v": 0.0, "bend_mm": 0.0, "bend_dir": 0.0,
             "elastic_mm": 0.0, "noise_std": 0.0}


def sample_params(seed):
    """All random choices of one view. Same seed -> same view."""
    rng = np.random.default_rng(seed)
    u = lambda r: float(rng.uniform(*r))
    return {"seed": int(seed), "yaw": u(YAW_DEG), "pitch": u(PITCH_DEG), "roll": u(ROLL_DEG),
            "spd": SPD_MM + u(SPD_DELTA_MM), "off_u": u(OFFSET_U_MM), "off_v": u(OFFSET_V_MM),
            "bend_mm": u((0.0, BEND_MAX_MM)), "bend_dir": u((0.0, 360.0)),
            "elastic_mm": u((0.0, ELASTIC_MAX_MM)), "noise_std": u(NOISE_STD)}


# =========================================================================
# VOLUMES
# =========================================================================

def load_specimen(spec):
    """CT + 3 masks, as the projection loop loaded them, padded with air."""
    vols = [load_volume(os.path.join(CT_DIR, f"{spec}_XR.tif"))]
    vols += [load_volume(os.path.join(CT_DIR, f"{spec}_{m}.tif")) for m in MASK_TYPES]
    vols = [np.asarray(np.nan_to_num(v), dtype=np.float32) for v in vols]
    shapes = {v.shape for v in vols}
    if len(shapes) != 1:
        raise SystemExit(f"{spec}: CT and masks differ in shape {shapes}")
    return pad_for_rotation(vols)


def pad_for_rotation(vols):
    """Air around the volume. Along the beam (axis 1) it must cover the widest
    possible footprint, the diagonal of the y-x section, so any yaw fits, plus
    how far the trunk ends move along the beam at the largest tilt (half the
    height x sin(tilt)). The other axis gets a margin for bend and wobble.
    Symmetric, so the centre (the rotation centre) does not move."""
    nz, ny, nx = vols[0].shape
    m = int(np.ceil(PAD_MARGIN_MM / SPACING_MM))
    max_tilt = np.deg2rad(max(abs(a) for a in PITCH_DEG + ROLL_DEG))
    m_tilt = int(np.ceil(nz / 2 * np.sin(max_tilt)))          # voxels
    diag = int(np.ceil(np.hypot(ny, nx)))
    ty = max(diag, ny) + 2 * (m + m_tilt)
    tx = nx + 2 * m
    py, px = ty - ny, tx - nx
    pad = ((0, 0), (py // 2, py - py // 2), (px // 2, px - px // 2))
    return [np.pad(v, pad) for v in vols]


def _xp():
    if CUPY_AVAILABLE:
        import cupy as cp
        from cupyx.scipy.ndimage import map_coordinates, zoom
        return cp, map_coordinates, zoom
    from scipy.ndimage import map_coordinates, zoom
    return np, map_coordinates, zoom


def deform(vols, p):
    """Bend + wobble, one displacement field applied to every volume.
    Returns the volumes unchanged when both are zero."""
    if p["bend_mm"] <= 0 and p["elastic_mm"] <= 0:
        return vols
    xp, map_coordinates, zoom = _xp()
    nz, ny, nx = vols[0].shape
    f32 = xp.float32

    # bend: each slice shifts sideways by d(z) = A * s^2, s = -1..1 along the
    # trunk, so the middle stays and both ends move by A, in direction bend_dir
    s = (xp.arange(nz, dtype=f32) - (nz - 1) / 2) / max((nz - 1) / 2, 1)
    a = p["bend_mm"] / SPACING_MM
    th = np.deg2rad(p["bend_dir"])
    dy = (a * np.sin(th)) * s ** 2
    dx = (a * np.cos(th)) * s ** 2

    zc = xp.broadcast_to(xp.arange(nz, dtype=f32)[:, None, None], (nz, ny, nx)).copy()
    yc = (xp.arange(ny, dtype=f32)[None, :, None] - dy[:, None, None]).astype(f32)
    yc = xp.broadcast_to(yc, (nz, ny, nx)).copy()
    xc = (xp.arange(nx, dtype=f32)[None, None, :] - dx[:, None, None]).astype(f32)
    xc = xp.broadcast_to(xc, (nz, ny, nx)).copy()

    # wobble: random values on a coarse grid (one node every ELASTIC_SCALE_MM),
    # smoothly upsampled, scaled so the largest shift is exactly elastic_mm
    if p["elastic_mm"] > 0:
        rng = np.random.default_rng([p["seed"] or 0, 1])
        step = ELASTIC_SCALE_MM / SPACING_MM
        coarse_shape = tuple(int(np.ceil(n / step)) + 1 for n in (nz, ny, nx))
        coarse = rng.standard_normal((3,) + coarse_shape).astype(np.float32)
        fields = []
        for k in range(3):
            c = xp.asarray(coarse[k])
            fac = [(n - 1) / max(cn - 1, 1) for n, cn in zip((nz, ny, nx), coarse_shape)]
            up = zoom(c, fac, order=3, grid_mode=False)
            sl = tuple(slice(0, n) for n in (nz, ny, nx))
            up = up[sl]
            if up.shape != (nz, ny, nx):               # rounding: pad the last row/col
                up = xp.pad(up, [(0, n - u) for n, u in zip((nz, ny, nx), up.shape)], mode="edge")
            fields.append(up.astype(f32))
        mag = xp.sqrt(fields[0] ** 2 + fields[1] ** 2 + fields[2] ** 2).max()
        scale = (p["elastic_mm"] / SPACING_MM) / float(mag) if float(mag) > 0 else 0.0
        zc += fields[0] * scale
        yc += fields[1] * scale
        xc += fields[2] * scale
        del fields

    coords = xp.stack((zc, yc, xc))
    del zc, yc, xc
    out = []
    for v in vols:
        g = xp.asarray(v)
        w = map_coordinates(g, coords, order=1, mode="constant", cval=0.0)
        out.append(xp.asnumpy(w) if xp is not np else w)
        del g, w
    del coords
    if xp is not np:
        xp.get_default_memory_pool().free_all_blocks()
    return out


# =========================================================================
# PROJECTION
# =========================================================================

def rotation_xyz(yaw, pitch, roll):
    """Fiji xyz convention, as calibrate_drr.py expects. Fiji z = numpy axis 0
    = the trunk, so yaw turns the vine around its own axis."""
    def rz(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])

    def ry(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])

    def rx(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])

    return rz(np.deg2rad(yaw)) @ ry(np.deg2rad(pitch)) @ rx(np.deg2rad(roll))


_CURVE = None


def apply_curve(lin):
    """Same as calibrate_drr.py --linear (save_drr_16bit) followed by
    fit_drr_to_pxr.py apply: clip 0..1, x65535, cast to uint16, then
    interpolate through the fitted curve. With USE_CURVE = False it stops
    after the uint16 step and returns it scaled back to 0..1."""
    if not USE_CURVE:
        return (np.clip(lin, 0.0, 1.0) * 65535.0).astype(np.uint16).astype(np.float32) / 65535.0
    global _CURVE
    if _CURVE is None:
        d = np.load(CURVE)
        _CURVE = (d["grid"], d["curve"], np.dtype(str(d["pxr_dtype"])))
    grid, curve, out_dtype = _CURVE
    u16 = (np.clip(lin, 0.0, 1.0) * 65535.0).astype(np.uint16)
    mapped = np.interp(u16.astype(np.float64), grid, curve)
    if np.issubdtype(out_dtype, np.integer):
        info = np.iinfo(out_dtype)
        mapped = np.clip(np.rint(mapped), info.min, info.max)
    return mapped.astype(out_dtype)


def render(vols, p, timings=None):
    """One view. vols = [CT, healthy, necrosis, white rot], padded.
    Returns (drr, labels) with drr in curve units and labels (3, H, W) in mm."""
    t = time.perf_counter()
    vols = deform(vols, p)
    if timings is not None:
        timings["deform"] = time.perf_counter() - t

    geom = make_geometry(detector_size=DETECTOR, spacing_zyx=(SPACING_MM,) * 3,
                         sid_mm=SID_MM, spd_mm=p["spd"], offset_v_mm=p["off_v"],
                         offset_u_mm=p["off_u"], beam_axis=BEAM_AXIS)
    rot = rotation_xyz(p["yaw"], p["pitch"], p["roll"])
    zero = np.zeros(3)
    proj = lambda v, att: generate_drr(v, geom, attenuation_scale=att, rotation_xyz=rot,
                                       true_translation_xyz=zero, use_gpu=True)

    t = time.perf_counter()
    a = proj(vols[0], ATT_CT)
    lin = (-np.expm1(-np.maximum(a, 0.0))).astype(np.float32)   # 1 - exp(-A)
    drr = apply_curve(lin).astype(np.float32)
    if timings is not None:
        timings["project CT"] = time.perf_counter() - t

    t = time.perf_counter()
    labels = np.stack([proj(v, ATT_MASK).astype(np.float32) for v in vols[1:]])
    if timings is not None:
        timings["project 3 masks"] = time.perf_counter() - t

    if p["noise_std"] > 0:
        rng = np.random.default_rng([p["seed"] or 0, 2])
        drr = drr + rng.normal(0.0, p["noise_std"], drr.shape).astype(np.float32)
    return drr, labels


# =========================================================================
# COMMANDS
# =========================================================================

def _show(ax, img, title, cmap="gray", vmax=None):
    ax.imshow(img, cmap=cmap, vmin=0 if vmax else None, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


def cmd_preview(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(PREVIEW_DIR, exist_ok=True)
    if NOISE_STD == (0.0, 0.0):
        print("note: noise is off. Run `augment.py noise <radiograph>` and set NOISE_STD.")
    t = time.perf_counter()
    vols = load_specimen(args.vine)
    print(f"{args.vine}: loaded and padded to {vols[0].shape} in {time.perf_counter() - t:.1f} s")
    canon, canon_lab = render(vols, CANONICAL)
    for seed in args.seed:
        p = sample_params(seed)
        tm = {}
        t = time.perf_counter()
        drr, lab = render(vols, p, tm)
        total = time.perf_counter() - t
        fig, ax = plt.subplots(2, 4, figsize=(18, 10), constrained_layout=True)
        _show(ax[0, 0], canon, "original view (no augmentation)")
        for c, name in enumerate(["healthy", "necrosis", "white rot"]):
            _show(ax[0, c + 1], canon_lab[c], f"original {name} (mm)", "magma", 120)
        _show(ax[1, 0], drr, f"augmented, seed {seed}")
        for c, name in enumerate(["healthy", "necrosis", "white rot"]):
            _show(ax[1, c + 1], lab[c], f"augmented {name} (mm)", "magma", 120)
        txt = (f"yaw {p['yaw']:.0f} deg   pitch {p['pitch']:+.1f}   roll {p['roll']:+.1f}   "
               f"spd {p['spd']:.0f} mm   offset u {p['off_u']:+.0f} v {p['off_v']:+.0f} mm   "
               f"bend {p['bend_mm']:.1f} mm toward {p['bend_dir']:.0f} deg   "
               f"wobble {p['elastic_mm']:.2f} mm   noise {p['noise_std']:.4f}")
        fig.suptitle(f"{args.vine}, seed {seed}\n{txt}", fontsize=11)
        out = os.path.join(PREVIEW_DIR, f"{args.vine}_seed{seed}.png")
        fig.savefig(out, dpi=70)
        plt.close(fig)
        print(f"seed {seed}: {total:.1f} s  (" +
              ", ".join(f"{k} {v:.1f} s" for k, v in tm.items()) + f")  -> {out}")
        print(f"   {txt}")


def cmd_check(args):
    """Un-augmented render vs the DRR and labels made by the projection loop.
    Small differences are expected (the air padding moves the ray samples
    slightly); large ones mean a convention is wrong."""
    vols = load_specimen(args.vine)
    drr, lab = render(vols, CANONICAL)
    ref = tifffile.imread(os.path.join(CANON_DRR_DIR, f"{args.vine}{DRR_SUFFIX}")).astype(np.float32)
    if not USE_CURVE:
        ref = ref / 65535.0                       # the saved linear DRR is 16-bit
    print(f"DRR {'with' if USE_CURVE else 'WITHOUT'} the DRR->PXR curve, compared with *{DRR_SUFFIX}")

    def compare(name, a, b):
        if a.shape != b.shape:
            print(f"  {name:10s} SHAPE DIFFERS {a.shape} vs {b.shape}")
            return
        d = np.abs(a - b)
        r = np.corrcoef(a.ravel(), b.ravel())[0, 1]
        print(f"  {name:10s} max |diff| {d.max():9.4f}   mean |diff| {d.mean():9.5f}   "
              f"value range {b.min():.3f}..{b.max():.3f}   correlation {r:.5f}")

    print(f"{args.vine}: augment.py (no augmentation) vs existing files")
    compare("DRR", drr, ref)
    for c, m in enumerate(MASK_TYPES):
        compare(m, lab[c], tifffile.imread(os.path.join(CANON_LAB_DIR, f"{args.vine}_{m}_mm.tif")))
    print("correlation should be about 0.999 or higher, and mean |diff| small next to the value range")


def cmd_noise(args):
    """Noise of a real radiograph, in the curve's units: the image is scaled by
    its own 0.1 / 99.9 percentiles (as the curve's PXR side was), the smooth
    structure is removed with a Gaussian high-pass, and the noise is the spread
    of what remains in flat, unsaturated areas."""
    from scipy.ndimage import gaussian_filter
    img = tifffile.imread(args.radiograph).astype(np.float64)
    img = np.squeeze(img)
    lo, hi = np.percentile(img[np.isfinite(img)], [0.1, 99.9])
    img = np.clip((img - lo) / (hi - lo), 0, 1)
    resid = img - gaussian_filter(img, 3)
    smooth = gaussian_filter(img, 3)
    b = 32
    H, W = (img.shape[0] // b) * b, (img.shape[1] // b) * b
    blocks = lambda a: a[:H, :W].reshape(H // b, b, W // b, b).swapaxes(1, 2).reshape(-1, b * b)
    rs, ms = blocks(resid), blocks(smooth)
    keep = (ms.mean(1) > 0.05) & (ms.mean(1) < 0.95)    # skip air and saturated blocks
    stds = rs[keep].std(1)
    est = float(np.percentile(stds, 10))                 # flat blocks: noise only
    print(f"{args.radiograph}")
    print(f"  {keep.sum()} usable 32x32 blocks")
    print(f"  noise std, flattest 10% of blocks: {est:.4f}   median block: {np.median(stds):.4f}")
    print(f"  suggestion: NOISE_STD = ({0.5 * est:.4f}, {1.5 * est:.4f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("preview"); p.add_argument("--vine", required=True)
    p.add_argument("--seed", type=int, nargs="+", default=[1])
    p.set_defaults(func=cmd_preview)
    c = sub.add_parser("check"); c.add_argument("--vine", required=True)
    c.set_defaults(func=cmd_check)
    n = sub.add_parser("noise"); n.add_argument("radiograph")
    n.set_defaults(func=cmd_noise)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()