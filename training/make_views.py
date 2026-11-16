#!/usr/bin/env python3
"""
make_views.py  -  render augmented views ONCE and store them (D2 and D1)

Renders VIEWS_PER_VINE augmented views per vine and saves them, so training
reads them from disk instead of re-rendering. Views are saved as uncompressed
.npy files, which training reads with memory mapping.

D2 (default): the 12 labelled vines, rendered with augment.py. Every vine gets
views; at training time each fold only loads the views of its training vines.
Each view is cropped to its vine (+ AUG_CROP_MARGIN):
    <VIEWS_DIR>/<vine>/<view>_img.npy   DRR, normalized 0..1, stored as uint16
    <VIEWS_DIR>/<vine>/<view>_lab.npy   (3, H, W) thickness in mm, float16
    <VIEWS_DIR>/<vine>/<view>_fg.npy    (N, 2) row, col of up to FG_POINTS vine
                                        pixels, for centring training patches
    <VIEWS_DIR>/manifest.csv            seed and parameters of every view
Seeds: zlib.crc32("<vine>-stored-<view>").

D1 (--d1): all 20 monitoring vines of 2026. The 5 held-out vines get views too,
but training must only load the views of its training vines. For each vine:
  1. CT (dataset/ct_volumes_table_removed/CEP_<v>_2026_XR.tif), air zeroed:
     every voxel below AIR_BELOW (raw value) -> 0
  2. bend + wobble (same ranges as augment.py)
  3. pose = registration transform (dataset/transformation_matrix/CEP_<v>_transform.txt), then extra
     yaw / pitch / roll around the vine's own centre
  4. projection with random distance and offsets (same ranges as augment.py)
  5. linear DRR 1 - exp(-A) -> DRR->PXR curve -> normalize
No noise (train_domain_adaptation.py adds it to D1 DRR patches), no labels:
    <D1_VIEWS_DIR>/CEP_<v>/<view>_img.npy
    <D1_VIEWS_DIR>/manifest.csv
Seeds: zlib.crc32("<vine>-d1-<view>").
The un-augmented view must reproduce results/CEP_<v>_airzero_calibration_linear.tif:
run --check first.

Already-written views are skipped, so a stopped run continues where it was.
After the first view, the script prints the expected total size and stops if
the disk does not have room for it.

USAGE (from the repo root)
    python training/make_views.py --dry-run             # D2: plan and disk size
    python training/make_views.py                       # D2: make them all
    python training/make_views.py --vine CEP016_RES3    # D2: one vine only

    python training/make_views.py --d1 --check --vine 1191   # D1: match + 3 examples
    python training/make_views.py --d1 --dry-run
    python training/make_views.py --d1
    python training/make_views.py --d1 --soil      # after the views: soil lines + vine pixels
"""

import argparse
import os
import shutil
import sys
import time
import zlib

import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import augment                                                   # noqa: E402
from train_baseline import (REPO, AUG_CROP_MARGIN, list_specimens,  # noqa: E402
                            normalize, crop_to_vine, LABEL_ROOT, LABEL_SUFFIXES,
                            VIEWS_DIR)   # set VIEWS_DIR in train_baseline.py only
from calibrate_drr import (make_geometry, generate_drr,          # noqa: E402
                           fiji_matrix_to_pure_pose, crop_top_to_match_pxr,
                           fiji_pose_to_centered_zyx)
from VineRadiologist.io import load_volume                       # noqa: E402

VIEWS_PER_VINE = 100         # per vine, D2 and D1
FG_POINTS = 20000            # D2: vine pixels kept per view for patch centring
PARAM_KEYS = ("yaw", "pitch", "roll", "spd", "off_u", "off_v",
              "bend_mm", "bend_dir", "elastic_mm", "noise_std")

# ---- D1 settings ----------------------------------------------------------
D1_VIEWS_DIR = os.path.join(os.path.dirname(VIEWS_DIR), "d1_views")   # next to d2_views
D1_CT_PATTERN = os.path.join(REPO, "dataset", "ct_volumes_table_removed", "CEP_{v}_2026_XR.tif")
D1_TRANSFORM_PATTERN = os.path.join(REPO, "dataset", "transformation_matrix", "CEP_{v}_transform.txt")
D1_CHECK_DIR = os.path.join(REPO, "results", "augment_preview_d1")
AIR_BELOW = 100              # raw CT value; below this = air, set to 0
D1_ATT = 0.015               # attenuation used for the D1 airzero DRRs
D1_BEAM_AXIS = 0             # manual registration convention
CHUNK_Z = 64                 # slices deformed at once on the GPU (memory)
D1_PARAM_KEYS = PARAM_KEYS[:-1]        # no noise stored for D1

# voxel spacing (z, y, x) in mm, from specimen_table.txt
D1_SPACING_ZYX = {
    "313B": (0.4, 0.578125, 0.578125), "318": (0.4, 0.5625, 0.5625),
    "322": (0.4, 0.851563, 0.851563), "323": (0.6, 0.585937, 0.585937),
    "330": (0.4, 0.697265, 0.697265), "335": (0.4, 0.820312, 0.820312),
    "368B": (0.4, 0.628906, 0.628906), "378A": (0.4, 0.617187, 0.617187),
    "378B": (0.4, 0.68164, 0.68164), "380A": (0.4, 0.828125, 0.828125),
    "764B": (0.4, 0.751953, 0.751953), "988B": (0.4, 0.623047, 0.623047),
    "1181": (0.7, 0.585937, 0.585937), "1186A": (0.4, 0.605469, 0.605469),
    "1189": (0.4, 0.683593, 0.683593), "1191": (0.4, 0.582031, 0.582031),
    "1193": (0.4, 0.730469, 0.730469), "1195": (0.4, 0.867187, 0.867187),
    "1266A": (0.4, 0.615234, 0.615234), "2184A": (0.4, 0.630859, 0.630859),
}


# =========================================================================
# SHARED
# =========================================================================

def save_arrays(paths, arrays):
    """Temp names first, then rename (a stopped run never leaves half a view),
    then read back and check. Returns the bytes written."""
    os.makedirs(os.path.dirname(paths[0]), exist_ok=True)
    for p, a in zip(paths, arrays):
        np.save(p + ".tmp.npy", a)
    for p in paths:
        os.replace(p + ".tmp.npy", p)
    for p, a in zip(paths, arrays):
        if not np.array_equal(np.load(p), a):
            sys.exit(f"{p}: written file does not match, stopping")
    return sum(os.path.getsize(p) for p in paths)


def free_space(d):
    while not os.path.exists(d):
        d = os.path.dirname(d)
    return shutil.disk_usage(d).free


def run(ds, vines, dry_run):
    """The render-and-save loop, the same for D2 and D1. ds is D2 or D1 below."""
    todo = [(s, v) for s in vines for v in range(VIEWS_PER_VINE)
            if not all(os.path.exists(p) for p in ds.paths(s, v))]
    print(f"{len(vines)} vines x {VIEWS_PER_VINE} views; {len(todo)} still to make "
          f"-> {ds.root}")
    if dry_run:
        ds.estimate_size(vines)
    if dry_run or not todo:
        return

    os.makedirs(ds.root, exist_ok=True)
    manifest = os.path.join(ds.root, "manifest.csv")
    if not os.path.exists(manifest):
        with open(manifest, "w") as f:
            f.write("vine,view,seed," + ",".join(ds.keys) + ",height_px,width_px\n")

    t0, done, checked_disk = time.perf_counter(), 0, False
    for s in vines:
        views = [v for (vs, v) in todo if vs == s]
        if not views:
            continue
        data = ds.load(s)
        for v in views:
            seed = ds.seed(s, v)
            p = augment.sample_params(seed)
            arrays = ds.make(data, s, v, p)
            nbytes = save_arrays(ds.paths(s, v), arrays)
            with open(manifest, "a") as f:
                f.write(f"{s},{v},{seed}," + ",".join(f"{p[k]:.4f}" for k in ds.keys)
                        + f",{arrays[0].shape[0]},{arrays[0].shape[1]}\n")
            done += 1
            if not checked_disk:
                need = nbytes * (len(todo) - 1)
                free = free_space(ds.root)
                print(f"  one view = {nbytes / 1e6:.0f} MB -> about {need / 1e9:.0f} GB more "
                      f"needed, {free / 1e9:.0f} GB free")
                if need > 0.9 * free:
                    sys.exit("  not enough disk space, stopping (the view made so far is kept)")
                checked_disk = True
            el = time.perf_counter() - t0
            print(f"  {s} view {v:3d}  ({done}/{len(todo)}, "
                  f"{el / done:.1f} s/view, ~{el / done * (len(todo) - done) / 60:.0f} min left)")
        del data
    print("done")


# =========================================================================
# D2
# =========================================================================

class D2:
    root = VIEWS_DIR
    keys = PARAM_KEYS

    @staticmethod
    def paths(vine, v):
        d = os.path.join(VIEWS_DIR, vine)
        return tuple(os.path.join(d, f"{v:03d}_{k}.npy") for k in ("img", "lab", "fg"))

    @staticmethod
    def seed(vine, v):
        return zlib.crc32(f"{vine}-stored-{v}".encode())

    @staticmethod
    def load(vine):
        return augment.load_specimen(vine)

    @staticmethod
    def make(vols, vine, v, p):
        drr, lab = augment.render(vols, p)
        img = normalize(np.nan_to_num(drr))          # same scaling as the test images
        img, lab = crop_to_vine(img, np.clip(np.nan_to_num(lab), 0, None), AUG_CROP_MARGIN)
        img16 = np.round(np.clip(img, 0, 1) * 65535).astype(np.uint16)
        lab16 = lab.astype(np.float16)
        fg = np.argwhere(lab16.astype(np.float32).sum(0) > 0).astype(np.int32)
        if len(fg) > FG_POINTS:
            fg = fg[np.random.default_rng(D2.seed(vine, v)).choice(len(fg), FG_POINTS,
                                                                     replace=False)]
        return img16, lab16, fg

    @staticmethod
    def estimate_size(vines):
        """Rough disk size, without rendering: the vine box of each existing
        (un-augmented) label + AUG_CROP_MARGIN, 8 bytes per pixel. Rotated and
        bent views are somewhat wider or narrower, so treat it as approximate."""
        total = 0
        for s in vines:
            lab = sum(tifffile.imread(os.path.join(LABEL_ROOT, s + suf)).astype(np.float32)
                      for suf in LABEL_SUFFIXES)
            ys, xs = np.nonzero(lab > 0)
            h = min(ys.max() - ys.min() + 1 + 2 * AUG_CROP_MARGIN, lab.shape[0])
            w = min(xs.max() - xs.min() + 1 + 2 * AUG_CROP_MARGIN, lab.shape[1])
            one = h * w * 8 + FG_POINTS * 8
            total += one * VIEWS_PER_VINE
            print(f"  {s:12s} about {h} x {w} px -> {one / 1e6:.0f} MB per view")
        print(f"estimated total: about {total / 1e9:.0f} GB for {VIEWS_PER_VINE} views per "
              f"vine, {free_space(VIEWS_DIR) / 1e9:.0f} GB free")


# =========================================================================
# D1
# =========================================================================

def d1_ref_path(v):
    from train_domain_adaptation import D1_DRR_PATTERN
    return D1_DRR_PATTERN.format(v=v)


def d1_deform(vol, p, sp):
    """Bend + wobble, like augment.deform, but in z chunks so the big D1
    volumes fit on the GPU, and with anisotropic voxels."""
    if p["bend_mm"] <= 0 and p["elastic_mm"] <= 0:
        return vol
    xp, map_coordinates, _ = augment._xp()
    nz, ny, nx = vol.shape
    f32 = xp.float32
    g = xp.asarray(vol, dtype=f32)
    out = xp.empty_like(g)

    # bend: slice z shifts sideways by A * s^2 (s = -1..1 along the trunk)
    th = np.deg2rad(p["bend_dir"])
    ay = p["bend_mm"] * np.sin(th) / sp[1]          # voxels
    ax = p["bend_mm"] * np.cos(th) / sp[2]

    # wobble: random values on a coarse grid (one node every ELASTIC_SCALE_MM),
    # smooth cubic interpolation, scaled so the largest shift is about elastic_mm
    coarse, step, scale = None, None, 0.0
    if p["elastic_mm"] > 0:
        step = [augment.ELASTIC_SCALE_MM / s for s in sp]           # voxels per node
        cshape = tuple(int(np.ceil(n / st)) + 1 for n, st in zip(vol.shape, step))
        rng = np.random.default_rng([p["seed"] or 0, 1])
        coarse = xp.asarray(rng.standard_normal((3,) + cshape).astype(np.float32))

    def field(z, y, x):                              # 3 x chunk, unscaled
        c = xp.stack([z / step[0], y / step[1], x / step[2]])
        return [map_coordinates(coarse[k], c, order=3, mode="nearest") for k in range(3)]

    def grid(z0, z1, every=1):
        z = xp.arange(z0, z1, every, dtype=f32)[:, None, None]
        y = xp.arange(0, ny, every, dtype=f32)[None, :, None]
        x = xp.arange(0, nx, every, dtype=f32)[None, None, :]
        shp = (z.shape[0], y.shape[1], x.shape[2])
        return [xp.broadcast_to(a, shp).astype(f32) for a in (z, y, x)]

    if coarse is not None:                           # largest shift, on a sparse grid
        f = field(*grid(0, nz, 8))
        mag = float(xp.sqrt(f[0] ** 2 + f[1] ** 2 + f[2] ** 2).max())
        scale = p["elastic_mm"] / mag if mag > 0 else 0.0
        del f

    for z0 in range(0, nz, CHUNK_Z):
        z1 = min(z0 + CHUNK_Z, nz)
        zc, yc, xc = grid(z0, z1)
        s = (zc - (nz - 1) / 2) / max((nz - 1) / 2, 1)
        yc = yc - ay * s ** 2
        xc = xc - ax * s ** 2
        if coarse is not None and scale > 0:
            f = field(zc, yc, xc)
            zc = zc + f[0] * (scale / sp[0])
            yc = yc + f[1] * (scale / sp[1])
            xc = xc + f[2] * (scale / sp[2])
            del f
        out[z0:z1] = map_coordinates(g, xp.stack((zc, yc, xc)), order=1,
                                     mode="constant", cval=0.0)
    res = xp.asnumpy(out) if xp is not np else out
    del g, out, coarse
    if xp is not np:
        xp.get_default_memory_pool().free_all_blocks()
    return res


def d1_render_linear(data, p):
    """Linear DRR 1 - exp(-A), same detector size and crop as the reference DRR."""
    vol, T, ref, sp = data
    vol = d1_deform(vol, p, sp)
    geom = make_geometry(detector_size=int(ref.shape[1]), spacing_zyx=sp,
                         sid_mm=augment.SID_MM, spd_mm=p["spd"], offset_v_mm=p["off_v"],
                         offset_u_mm=p["off_u"], beam_axis=D1_BEAM_AXIS)
    r_reg, t_reg = fiji_matrix_to_pure_pose(T, vol.shape, sp)
    # extra rotation in the vine's own frame (Fiji z = trunk), then the registration
    rot = r_reg @ augment.rotation_xyz(p["yaw"], p["pitch"], p["roll"])
    a = generate_drr(vol, geom, attenuation_scale=D1_ATT, rotation_xyz=rot,
                     true_translation_xyz=t_reg, use_gpu=True)
    lin = (-np.expm1(-np.maximum(a, 0.0))).astype(np.float32)
    if ref.shape[0] != ref.shape[1]:
        lin = crop_top_to_match_pxr(lin, ref.shape)
    return lin


def d1_to_image(lin):
    """Same as load_d1_drr in train_domain_adaptation.py: curve, then normalize."""
    return normalize(augment.apply_curve(lin).astype(np.float32))


class D1:
    root = D1_VIEWS_DIR
    keys = D1_PARAM_KEYS

    @staticmethod
    def paths(v, k):
        return (os.path.join(D1_VIEWS_DIR, f"CEP_{v}", f"{k:03d}_img.npy"),)

    @staticmethod
    def seed(v, k):
        return zlib.crc32(f"{v}-d1-{k}".encode())

    @staticmethod
    def load(v):
        """CT with air zeroed, registration transform, reference DRR (0..1), spacing."""
        path = D1_CT_PATTERN.format(v=v)
        raw = tifffile.imread(path)
        vol = np.asarray(np.nan_to_num(load_volume(path)), dtype=np.float32)
        if raw.shape != vol.shape:
            sys.exit(f"{path}: tifffile {raw.shape} and load_volume {vol.shape} differ")
        vol[raw < AIR_BELOW] = 0.0
        del raw
        T = np.loadtxt(D1_TRANSFORM_PATTERN.format(v=v))
        if T.shape != (4, 4):
            sys.exit(f"CEP_{v}: transform is {T.shape}, expected 4x4")
        ref = np.nan_to_num(tifffile.imread(d1_ref_path(v)).astype(np.float32))
        if ref.max() > 1.5:                     # stored as 16-bit
            ref = ref / 65535.0
        return vol, T, ref, D1_SPACING_ZYX[v]

    @staticmethod
    def make(data, v, k, p):
        img = d1_to_image(d1_render_linear(data, p))
        return (np.round(np.clip(img, 0, 1) * 65535).astype(np.uint16),)

    @staticmethod
    def estimate_size(vines):
        total = 0
        for v in vines:
            shp = tifffile.TiffFile(d1_ref_path(v)).pages[0].shape
            one = shp[0] * shp[1] * 2
            total += one * VIEWS_PER_VINE
            print(f"  CEP_{v:6s} {shp[0]} x {shp[1]} px -> {one / 1e6:.0f} MB per view")
        print(f"estimated total: about {total / 1e9:.0f} GB, "
              f"{free_space(D1_VIEWS_DIR) / 1e9:.0f} GB free")


def d1_check(vines):
    """Un-augmented view vs the existing registered DRR, plus 3 augmented examples."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(D1_CHECK_DIR, exist_ok=True)
    canon = dict(augment.CANONICAL, spd=augment.SPD_MM)
    for v in vines:
        t = time.perf_counter()
        data = D1.load(v)
        vol, _, ref, sp = data
        print(f"CEP_{v}: CT {vol.shape}, spacing zyx {sp}, reference DRR {ref.shape}, "
              f"loaded in {time.perf_counter() - t:.0f} s")
        lin = d1_render_linear(data, canon)
        if lin.shape != ref.shape:
            print(f"  SHAPE DIFFERS: rendered {lin.shape} vs reference {ref.shape}")
            continue
        d = np.abs(lin - ref)
        r = np.corrcoef(lin.ravel(), ref.ravel())[0, 1]
        print(f"  un-augmented vs reference: mean |diff| {d.mean() * 65535:.0f}, "
              f"max {d.max() * 65535:.0f} (16-bit units), correlation {r:.5f}")

        fig, ax = plt.subplots(1, 6, figsize=(30, 6), constrained_layout=True)
        for a, img, title, cmap in ((ax[0], ref, "reference DRR", "gray"),
                                    (ax[1], lin, "re-rendered", "gray"),
                                    (ax[2], d, "|difference|", "magma")):
            a.imshow(img, cmap=cmap)
            a.set_title(title, fontsize=14)
            a.axis("off")
        for i, seed in enumerate((1, 2, 3)):
            p = augment.sample_params(seed)
            t = time.perf_counter()
            img = d1_to_image(d1_render_linear(data, p))
            print(f"  augmented seed {seed}: {time.perf_counter() - t:.1f} s")
            ax[3 + i].imshow(img, cmap="gray")
            ax[3 + i].set_title(f"yaw {p['yaw']:.0f}  pitch {p['pitch']:+.0f}  "
                                f"roll {p['roll']:+.0f}", fontsize=14)
            ax[3 + i].axis("off")
        out = os.path.join(D1_CHECK_DIR, f"CEP_{v}_check.png")
        fig.savefig(out, dpi=80)
        plt.close(fig)
        print(f"  -> {out}")
        del data


# =========================================================================
# D1 SOIL LINES AND VINE PIXELS OF THE STORED VIEWS (--d1 --soil)
# =========================================================================

SOIL_BAND_ROWS = 150         # rows below the clicked line taken as "the soil" (~14 mm)
SOIL_POINTS = 50000          # soil voxels kept per vine for projecting
SOIL_STRIDE = 3              # voxel subsampling when looking for the soil voxels


def d1_project_rows(pts, rot_xyz, t_xyz, spd, off_v, det, crop, pix):
    """Detector rows (in the cropped image) of object points pts (N, 3), given
    in mm zyx from the volume centre: the same geometry as the projector in
    calibrate_drr.py (beam = world axis 0, rows = world axis 1)."""
    r, t = fiji_pose_to_centered_zyx(rot_xyz, t_xyz)
    w = pts @ r.T + t
    tpar = (w[:, 0] + spd) / augment.SID_MM            # 0 at the source, 1 at the detector
    gv = (w[:, 1] - off_v) / tpar
    return gv / pix + det / 2.0 - crop


def d1_soil(vines):
    """For every stored D1 view: its soil row and its vine pixels.

    Soil: the clicked line of the registered DRR (mark_soil.py) marks the top of
    the soil. The CT voxels that project into the band just below that line are
    the soil. Each view moves them with its own yaw, pitch, roll, distance and
    offsets (from its seed, as in the manifest) and projects them again; the
    highest of them (1st percentile row) is that view's soil line. Bend and
    wobble are ignored (sideways, a few mm).
    Vine pixels: as for the registered DRR (Otsu, above the soil).
    Writes <D1_VIEWS_DIR>/soil_rows.csv and <view>_fg.npy next to each view."""
    from train_domain_adaptation import soil_row, otsu, FG_POINTS
    out_csv = os.path.join(D1_VIEWS_DIR, "soil_rows.csv")
    rows_out = []
    for v in vines:
        files = sorted(f for f in os.listdir(os.path.join(D1_VIEWS_DIR, f"CEP_{v}"))
                       if f.endswith("_img.npy"))
        if len(files) < VIEWS_PER_VINE:
            sys.exit(f"CEP_{v}: only {len(files)} views, run make_views.py --d1 first")
        vol, T, ref, sp = D1.load(v)
        det = int(ref.shape[1])
        crop = det - int(ref.shape[0])
        pix = make_geometry(detector_size=det).detector_pixel_spacing_mm
        r_reg, t_reg = fiji_matrix_to_pure_pose(T, vol.shape, sp)
        clicked = soil_row(d1_ref_path(v))

        pts = None
        if clicked is not None:
            idx = np.argwhere(vol[::SOIL_STRIDE, ::SOIL_STRIDE, ::SOIL_STRIDE] > 0) * SOIL_STRIDE
            obj = (idx - np.array(vol.shape) / 2.0) * np.array(sp)        # mm from the centre
            rows0 = d1_project_rows(obj, r_reg, t_reg, augment.SPD_MM, 0.0, det, crop, pix)
            band = (rows0 >= clicked) & (rows0 < clicked + SOIL_BAND_ROWS)
            pts = obj[band]
            if len(pts) > SOIL_POINTS:
                pts = pts[np.random.default_rng(0).choice(len(pts), SOIL_POINTS, replace=False)]
            back = np.percentile(d1_project_rows(pts, r_reg, t_reg, augment.SPD_MM, 0.0,
                                                 det, crop, pix), 1)
            print(f"CEP_{v}: clicked soil row {clicked}, recomputed {back:.0f} "
                  f"({len(pts)} soil points)")
        else:
            print(f"CEP_{v}: no soil line clicked, views get none")
        del vol

        for f in files:
            k = int(f.split("_")[0])
            soil = ""
            if pts is not None:
                p = augment.sample_params(D1.seed(v, k))
                rot = r_reg @ augment.rotation_xyz(p["yaw"], p["pitch"], p["roll"])
                rows = d1_project_rows(pts, rot, t_reg, p["spd"], p["off_v"], det, crop, pix)
                soil = int(np.clip(np.floor(np.percentile(rows, 1)), 0, ref.shape[0]))
            ip = os.path.join(D1_VIEWS_DIR, f"CEP_{v}", f)
            img = np.load(ip, mmap_mode="r")
            small = np.asarray(img[::4, ::4], dtype=np.float32) / 65535.0
            mask = np.asarray(img, dtype=np.float32) / 65535.0 > otsu(small)
            if soil != "":
                mask[soil:] = False
            fg = np.argwhere(mask).astype(np.int32)
            if len(fg) > FG_POINTS:
                fg = fg[np.random.default_rng(D1.seed(v, k)).choice(len(fg), FG_POINTS,
                                                                     replace=False)]
            save_arrays((ip.replace("_img.npy", "_fg.npy"),), (fg,))
            rows_out.append((f"CEP_{v}", k, soil))
        print(f"  {len(files)} views done")

    old = {}
    if os.path.exists(out_csv):                      # keep vines not redone this time
        with open(out_csv) as fh:
            next(fh)
            for line in fh:
                a, b, c = line.rstrip("\n").split(",")
                old[(a, int(b))] = c
    for a, b, c in rows_out:
        old[(a, b)] = c
    with open(out_csv, "w") as fh:
        fh.write("vine,view,soil_row\n")
        for (a, b), c in sorted(old.items()):
            fh.write(f"{a},{b},{c}\n")
    print(f"-> {out_csv}")


# =========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d1", action="store_true", help="D1 2026 vines instead of D2")
    ap.add_argument("--check", action="store_true",
                    help="D1 only: un-augmented view vs the existing DRR + 3 examples")
    ap.add_argument("--soil", action="store_true",
                    help="D1 only, after the views: soil line and vine pixels of every view")
    ap.add_argument("--vine", nargs="+", help="only these vines (default: all of the set)")
    ap.add_argument("--dry-run", action="store_true", help="list the plan, render nothing")
    args = ap.parse_args()

    if not args.d1:
        if args.check:
            sys.exit("--check is for D1 (use augment.py check for D2)")
        run(D2, args.vine or list_specimens(), args.dry_run)
        return

    from train_domain_adaptation import d1_split
    train, held = d1_split()
    if args.check:
        d1_check(args.vine or train[:1])
        return
    if args.soil:
        d1_soil(args.vine or sorted(train + held))
        return
    # all 20 vines get views; training must only load the views of `train`
    vines = args.vine or sorted(train + held)
    print(f"held-out vines (views made, never to be used in training): {held}")
    run(D1, vines, args.dry_run)


if __name__ == "__main__":
    main()