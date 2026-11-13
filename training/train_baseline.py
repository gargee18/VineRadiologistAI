#!/usr/bin/env python3
"""
train_baseline.py  -  Stage 2 baseline for VineRadiologistAI

ResNet-50 U-Net, 1 input channel (DRR), 3 output channels (per-class
thickness in mm: healthy, necrosis, white rot). D2 only. No randomization.
Leave-specimen-out: you pick the test specimens, everything else trains.

No validation set and no early stopping on purpose: the model trains for a
fixed number of epochs and the test specimens are only touched once, at the
end. Otherwise the test score is not honest.

USAGE
    pip install segmentation-models-pytorch tifffile matplotlib

    # one fold: test on some specimens, train on the rest
    python train_baseline.py --test SPEC_A SPEC_B

    # all folds, leave-one-specimen-out (12 trainings)
    python train_baseline.py --all-folds

    # after several folds: per-vine and per-group summary, no training
    python train_baseline.py --summarize

    # redraw every plot from the saved models, no training
    python train_baseline.py --replot

    # same, with 3D augmentation (needs training/augment.py next to this file);
    # results go to OUT_DIR_AUG, so pass --augment to --summarize/--replot too
    python train_baseline.py --augment --test SPEC_A SPEC_B
    python train_baseline.py --augment --summarize

    # same, with the views stored by training/make_views.py (run that first);
    # adds a random left-right flip; results go to OUT_DIR_STORED
    python train_baseline.py --stored --test SPEC_A SPEC_B
    python train_baseline.py --stored --summarize

OUTPUT (in OUT_DIR/fold_<test specimens>/)
    model.pt              final weights
    metrics.json          per-image and mean metrics, plus volumes
    loss.csv              training L1 per epoch
    loss_curve.png        the same, as a plot
    preview_*.png         per test vine, cropped to the vine. Columns = classes.
                          Rows: true thickness, predicted thickness, true region,
                          predicted region, error. One colour scale for all
                          thickness maps, one for all error maps.
    pred_vs_true_<tissue>.png  predicted vs true thickness per pixel, one file
                          per tissue, one hexbin panel per plant group
    metrics_per_vine.png  MAE, Dice and amount ratio per test vine
    volume_per_vine.png   tissue volume (cm3) and share (%), true vs predicted
    volume_per_vine.csv   the same numbers
    profile.png           share of the wood along the trunk (height from the
                          bottom of the vine), one panel per group, all three
                          tissues, solid = true, dashed = predicted

OUTPUT of --summarize (in OUT_DIR/)
    summary_per_vine.csv, summary_per_group.csv, summary_per_group.png,
    loss_curves_all.png

OUTPUT of --replot (in OUT_DIR/), on top of the fold plots above
    pred_vs_true_all_<tissue>.png, profile_all.png, volume_all.png, dumbbell_<share|volume>.png,
    volume_per_vine.csv  (all test vines)
"""

import argparse
import glob
import json
import os
import random
import warnings

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp

# =========================================================================
# PATHS  (fill these in)
# =========================================================================

REPO = "/mnt/41d6c007-0c9e-41e2-b2eb-8d9c032e9e53/gargee/code_python/VineRadiologistAI"

DRR_ROOT = os.path.join(REPO, "results", "d2_cropped_drr")
# The D2 DRRs, flat:  <SPECIMEN>_k6.5_mapped_norm.tif, e.g.
#   CEP011_AS1_k6.5_mapped_norm.tif       shape (H, W)
DRR_SUFFIX = "_k6.5_mapped_norm.tif"

LABEL_ROOT = os.path.join(REPO, "results", "d2_cropped_labels")
# Output of the projection loop, flat, 3 files per specimen, each (H, W), mm:
#   CEP011_AS1_SAIN_mm.tif      -> channel 0, healthy
#   CEP011_AS1_NECROSE_mm.tif   -> channel 1, necrosis
#   CEP011_AS1_AMADOU_mm.tif    -> channel 2, white rot
LABEL_SUFFIXES = ["_SAIN_mm.tif", "_NECROSE_mm.tif", "_AMADOU_mm.tif"]

OUT_DIR = os.path.join(REPO, "results", "training_baseline_cropped")
# With --augment, every command (training, --summarize, --replot) uses
# OUT_DIR_AUG instead, so augmented and baseline results never mix.
OUT_DIR_AUG = os.path.join(REPO, "results", "training_augmented")

# ---- augmentation (only with --augment; see training/augment.py) ---------
AUG_EVERY_EPOCHS = 10    # a fresh set of augmented views every N epochs
AUG_VIEWS_PER_VINE = 5   # augmented views made per training vine, per set
AUG_CROP_MARGIN = 256    # views are kept cropped to the vine + this margin (px)

# ---- stored views (only with --stored; see training/make_views.py) --------
VIEWS_DIR = "/mnt/e823c70f-4136-47c9-91be-1ca7901a37b5/gargee/d2_views"   # other disk, ~83 GB
OUT_DIR_STORED = os.path.join(REPO, "results", "training_stored")
# ---- class balancing (only with --balance, which needs --stored) ----------
OUT_DIR_BALANCED = os.path.join(REPO, "results", "training_stored_balanced")
BALANCE_WR_PROB = 0.3    # share of patches centred on a white rot pixel
BALANCE_POWER = 0.5      # loss weight of a tissue = (1 / its mean thickness) ** power,
                         # scaled to a mean of 1. 0.5 = square root (mild), 1 = full
WR_POINTS = 20000        # white rot pixels kept per view for patch centring
_WR_PROB = 0.0           # set by run_fold: 0 = no white rot centring (all earlier runs)
FLIP_LR = True           # --stored only: flip each training patch left-right
                         # with probability 0.5 (test images are never flipped)
# Where checkpoints, metrics and preview images are written.

# =========================================================================
# SETTINGS
# =========================================================================

CLASSES = ["healthy", "necrosis", "white_rot"]
PATCH = 512              # training patch size (multiple of 32)
BATCH = 8                # lower it if the GPU runs out of memory
EPOCHS = 100
STEPS_PER_EPOCH = 200    # random patches drawn per epoch = STEPS * BATCH
LR = 1e-4
FG_PROB = 0.7            # chance a patch is centred on vine instead of anywhere
DICE_THR_MM = 0.5        # thickness above this counts as "class present" for
                         # Dice and for the region (silhouette) plots.
                         # Arbitrary starting value, adjust to your data.

# Plot scales: ONE scale for every thickness map and ONE for every error map,
# across classes and vines, so the same colour always means the same mm.
THICK_VMAX_MM = 120      # labels go up to ~135 mm; brighter is clipped
ERR_VMAX_MM = 40         # errors beyond +-40 mm are clipped

# Geometry, only used to turn summed thickness into an approximate volume.
# Pixel size at the vine = detector pixel * SPD / SID (cone-beam magnification).
DET_PIXEL_MM, SID_MM, SPD_MM = 0.139, 1230.0, 800.0
PIXEL_AREA_AT_VINE_MM2 = (DET_PIXEL_MM * SPD_MM / SID_MM) ** 2

TISSUE_COLORS = ["#1B5E20", "#F9A825", "#B71C1C"]   # healthy, necrosis, white rot
GROUP_COLORS = {"asymptomatic": "#1baf7a", "resistant": "#2a78d6",
                "symptomatic": "#eb6834", "apoplectic": "#7a3fa0"}
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================================
# DATA
# =========================================================================

def list_specimens():
    files = glob.glob(os.path.join(DRR_ROOT, "*" + DRR_SUFFIX))
    return sorted(os.path.basename(f)[:-len(DRR_SUFFIX)] for f in files)


def list_pairs(specimens):
    """One DRR per specimen. `lab` is the list of 3 per-class label files."""
    pairs = []
    for s in specimens:
        drr = os.path.join(DRR_ROOT, s + DRR_SUFFIX)
        lab = [os.path.join(LABEL_ROOT, s + suf) for suf in LABEL_SUFFIXES]
        missing = [p for p in [drr] + lab if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"{s}: missing {missing}")
        pairs.append((s, drr, lab))
    if not pairs:
        raise RuntimeError(f"No DRRs found for {specimens}")
    return pairs


def normalize(img):
    """Per-image 1-99 percentile scaling to [0, 1]. Use the SAME function on
    PXRs later, or the model learns to tell the two domains apart."""
    lo, hi = np.percentile(img, [1, 99])
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def load_pair(drr_path, lab_paths):
    img = np.nan_to_num(tifffile.imread(drr_path).astype(np.float32))
    assert img.ndim == 2, f"{drr_path}: expected (H, W), got {img.shape}"
    chans = []
    for p in lab_paths:
        c = np.nan_to_num(tifffile.imread(p).astype(np.float32))
        assert c.shape == img.shape, \
            f"{p}: shape {c.shape} does not match DRR {img.shape}"
        chans.append(c)
    lab = np.stack(chans, axis=0)                # (3, H, W)
    return normalize(img), np.clip(lab, 0, None)


def pad_to(arr, h, w):
    ph, pw = max(h - arr.shape[-2], 0), max(w - arr.shape[-1], 0)
    if ph == 0 and pw == 0:
        return arr
    pad = [(0, 0)] * (arr.ndim - 2) + [(0, ph), (0, pw)]
    return np.pad(arr, pad)


def random_patch(img, lab, fg_pts=None, wr_pts=None):
    """fg_pts: (N, 2) vine pixels, given for stored views (memory-mapped
    uint16 / float16 arrays); None = find them in lab, as before.
    wr_pts: (N, 2) white rot pixels, only with --balance: a share _WR_PROB of
    the patches is centred on one of them."""
    img, lab = pad_to(img, PATCH, PATCH), pad_to(lab, PATCH, PATCH)
    H, W = img.shape
    if wr_pts is not None and len(wr_pts) and random.random() < _WR_PROB:
        cy, cx = wr_pts[random.randrange(len(wr_pts))]
        y0 = int(np.clip(cy - PATCH // 2, 0, H - PATCH))
        x0 = int(np.clip(cx - PATCH // 2, 0, W - PATCH))
        return _cut_patch(img, lab, y0, x0)
    if fg_pts is None:
        fg = lab.sum(0) > 0
        has_fg = fg.any()
    else:
        has_fg = len(fg_pts) > 0
    if random.random() < FG_PROB and has_fg:
        if fg_pts is None:
            ys, xs = np.nonzero(fg)
            i = random.randrange(len(ys))
            cy, cx = ys[i], xs[i]
        else:
            cy, cx = fg_pts[random.randrange(len(fg_pts))]
        y0 = int(np.clip(cy - PATCH // 2, 0, H - PATCH))
        x0 = int(np.clip(cx - PATCH // 2, 0, W - PATCH))
    else:
        y0, x0 = random.randint(0, H - PATCH), random.randint(0, W - PATCH)
    return _cut_patch(img, lab, y0, x0)


def _cut_patch(img, lab, y0, x0):
    x = img[y0:y0 + PATCH, x0:x0 + PATCH]
    y = lab[:, y0:y0 + PATCH, x0:x0 + PATCH]
    if x.dtype == np.uint16:                     # stored view
        x = x.astype(np.float32) / 65535.0
    return x.astype(np.float32, copy=False), y.astype(np.float32, copy=False)


def white_rot_points(lab, seed):
    """Up to WR_POINTS pixels with white rot thicker than DICE_THR_MM."""
    pts = np.argwhere(np.asarray(lab[2], dtype=np.float32) > DICE_THR_MM).astype(np.int32)
    if len(pts) > WR_POINTS:
        pts = pts[np.random.default_rng(seed).choice(len(pts), WR_POINTS, replace=False)]
    return pts


def class_weights(labs):
    """Loss weight per tissue from the training labels: (1 / mean thickness
    inside the vine) ** BALANCE_POWER, scaled to a mean of 1."""
    tot = np.zeros(len(CLASSES)); n = 0
    for lab in labs:
        vine = lab.sum(0) > 0
        tot += lab[:, vine].sum(1); n += vine.sum()
    mean = tot / max(n, 1)
    w = (1.0 / np.maximum(mean, 1e-6)) ** BALANCE_POWER
    return w / w.mean(), mean


def batch_iter(data, flip_lr=False):
    """data: list of (img, lab) arrays in RAM, or (img, lab, fg_pts) for
    stored views. flip_lr: flip each patch left-right with probability 0.5."""
    while True:
        xs, ys = [], []
        for _ in range(BATCH):
            item = random.choice(data)
            x, y = random_patch(*item)
            if flip_lr and random.random() < 0.5:
                x, y = x[:, ::-1], y[:, :, ::-1]
            x, y = np.ascontiguousarray(x), np.ascontiguousarray(y)
            xs.append(x[None])
            ys.append(y)
        yield (torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys)))


# =========================================================================
# MODEL
# =========================================================================

def build_model():
    # Verify these arguments against the current segmentation_models_pytorch
    # docs if this line errors. in_channels=1 adapts the ImageNet first layer
    # to grayscale input.
    return smp.Unet(encoder_name="resnet50", encoder_weights="imagenet",
                    in_channels=1, classes=len(CLASSES))


def forward(model, x):
    return F.softplus(model(x))                  # thickness is never negative


def _tile_weight():
    """Weight of each pixel of a tile when tiles are blended: 1 in the centre,
    falling smoothly towards the edges (never exactly 0), so the less reliable
    tile borders count little and no seams appear between tiles."""
    w1 = np.sin(np.pi * (np.arange(PATCH) + 0.5) / PATCH).astype(np.float32)
    return np.outer(w1, w1)


@torch.no_grad()
def predict_full(model, img):
    """Sliding-window prediction on a full image: tiles overlap by 50% and are
    blended with _tile_weight (centre counts most), so tile borders leave no
    visible seams."""
    H, W = img.shape
    stride = PATCH // 2                          # 50% overlap between tiles
    Hp = PATCH + int(np.ceil(max(H - PATCH, 0) / stride)) * stride
    Wp = PATCH + int(np.ceil(max(W - PATCH, 0) / stride)) * stride
    x = pad_to(img, Hp, Wp)
    wt = _tile_weight()
    acc = np.zeros((len(CLASSES), Hp, Wp), np.float32)
    cnt = np.zeros((Hp, Wp), np.float32)
    for y0 in range(0, Hp - PATCH + 1, stride):
        for x0 in range(0, Wp - PATCH + 1, stride):
            tile = torch.from_numpy(x[y0:y0 + PATCH, x0:x0 + PATCH][None, None]).to(DEVICE)
            acc[:, y0:y0 + PATCH, x0:x0 + PATCH] += forward(model, tile)[0].cpu().numpy() * wt
            cnt[y0:y0 + PATCH, x0:x0 + PATCH] += wt
    return (acc / cnt)[:, :H, :W]


# =========================================================================
# METRICS
# =========================================================================

def metrics(pred, gt):
    """Per class, computed inside the vine footprint (where any GT tissue > 0),
    so the empty background does not inflate the scores."""
    vine = gt.sum(0) > 0
    out = {}
    for c, name in enumerate(CLASSES):
        p, g = pred[c][vine], gt[c][vine]
        pb, gb = p > DICE_THR_MM, g > DICE_THR_MM
        denom = pb.sum() + gb.sum()
        out[name] = {
            "mae_mm": float(np.abs(p - g).mean()) if vine.any() else None,
            "dice": float(2 * (pb & gb).sum() / denom) if denom else None,
            # summed thickness, predicted / true. 1.0 = right amount of tissue
            "amount_ratio": float(p.sum() / g.sum()) if g.sum() > 0 else None,
        }
    return out


def mean_metrics(per_image):
    out = {}
    for name in CLASSES:
        out[name] = {}
        for k in ("mae_mm", "dice", "amount_ratio"):
            vals = [m[name][k] for m in per_image if m[name][k] is not None]
            out[name][k] = float(np.mean(vals)) if vals else None
    return out


# =========================================================================
# PLOTS
# =========================================================================

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def vine_crop(gt, margin=60):
    """Bounding box of the vine (any true tissue), plus a margin, so the plots
    show the vine instead of a 3072 x 3072 frame of mostly air."""
    ys, xs = np.nonzero(gt.sum(0) > 0)
    if ys.size == 0:
        return slice(None), slice(None)
    H, W = gt.shape[1:]
    return (slice(max(ys.min() - margin, 0), min(ys.max() + margin + 1, H)),
            slice(max(xs.min() - margin, 0), min(xs.max() + margin + 1, W)))


def _hex_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255.0


def _region_rgb(region, vine, color):
    """White background, vine in light grey, the class region in its colour."""
    rgb = np.ones(region.shape + (3,), np.float32)
    rgb[vine] = 0.90
    rgb[region] = _hex_rgb(color)
    return rgb


def save_preview(img, gt, pred, m, title, path):
    """One figure per test vine, landscape. One row per tissue. Columns: true
    thickness, predicted thickness, error, true region, predicted region.
    Left column: the input DRR and the scores. One colour scale for all
    thickness maps, one for all error maps (THICK_VMAX_MM, ERR_VMAX_MM)."""
    plt = _plt()
    ry, rx = vine_crop(gt)
    img, gt, pred = img[ry, rx], gt[:, ry, rx], pred[:, ry, rx]
    vine = gt.sum(0) > 0
    names = {"healthy": "Healthy", "necrosis": "Necrosis", "white_rot": "White rot"}
    f = lambda x, fmt: (fmt % x) if x is not None else "n/a"

    fig = plt.figure(figsize=(22, 15), layout="constrained")
    # last row = colour scales only, so they do not squeeze some panels more than others
    gs = fig.add_gridspec(4, 6, width_ratios=[1.35, 1, 1, 1, 1, 1], height_ratios=[1, 1, 1, 0.14])
    fig.suptitle(title, fontsize=22, fontweight="bold")

    # left: input image, then the scores
    a = fig.add_subplot(gs[0, 0])
    a.imshow(img, cmap="gray"); a.set_title("Input DRR", fontsize=16)
    a.set_xticks([]); a.set_yticks([])
    a = fig.add_subplot(gs[1:3, 0]); a.axis("off")
    rows = [names[n] for n in CLASSES]
    t1 = a.table(cellText=[[f(m[n]["mae_mm"], "%.1f"), f(m[n]["dice"], "%.2f"),
                            f(m[n]["amount_ratio"], "%.2f")] for n in CLASSES],
                 rowLabels=rows, colLabels=["MAE (mm)", "Dice", "amount"],
                 cellLoc="center", bbox=[0.3, 0.68, 0.7, 0.3])
    t2 = a.table(cellText=[[f"{m['volume'][n]['true_cm3']:.0f}", f"{m['volume'][n]['pred_cm3']:.0f}",
                            f"{m['volume'][n]['true_share_pct']:.0f}", f"{m['volume'][n]['pred_share_pct']:.0f}"]
                           for n in CLASSES],
                 rowLabels=rows, colLabels=["true\ncm\u00b3", "pred.\ncm\u00b3", "true\n%", "pred.\n%"],
                 cellLoc="center", bbox=[0.3, 0.27, 0.7, 0.32])
    for t in (t1, t2):
        t.auto_set_font_size(False); t.set_fontsize(14)
    a.text(0.0, 1.0, "Scores", fontsize=16, fontweight="bold", va="bottom", transform=a.transAxes)
    a.text(0.0, 0.61, "Volume and share of the wood", fontsize=16, fontweight="bold",
           va="bottom", transform=a.transAxes)
    a.text(0.0, 0.0, "MAE: mean error inside the vine\n"
           "amount: predicted / true total\n"
           f"region: thickness > {DICE_THR_MM} mm,\n   grey = rest of the vine\n"
           "error: red = too much,\n   blue = too little\n"
           "volumes are approximate (cone beam)",
           fontsize=13, va="bottom", color="#444444", transform=a.transAxes)

    # right: one row per tissue
    cols = ["True thickness", "Predicted thickness", "Error (pred. - true)",
            "True region", "Predicted region"]
    for c, name in enumerate(CLASSES):
        ax = [fig.add_subplot(gs[c, k + 1]) for k in range(5)]
        im_t = ax[0].imshow(gt[c], cmap="magma", vmin=0, vmax=THICK_VMAX_MM)
        ax[1].imshow(pred[c], cmap="magma", vmin=0, vmax=THICK_VMAX_MM)
        err = np.where(vine, pred[c] - gt[c], np.nan)
        im_e = ax[2].imshow(err, cmap="RdBu_r", vmin=-ERR_VMAX_MM, vmax=ERR_VMAX_MM)
        ax[3].imshow(_region_rgb(gt[c] > DICE_THR_MM, vine, TISSUE_COLORS[c]))
        ax[4].imshow(_region_rgb(pred[c] > DICE_THR_MM, vine, TISSUE_COLORS[c]))
        ax[2].text(0.03, 0.97, f"MAE {f(m[name]['mae_mm'], '%.1f')} mm", transform=ax[2].transAxes,
                   fontsize=13, va="top", bbox=dict(fc="white", ec="none", alpha=0.8, pad=2))
        ax[4].text(0.03, 0.97, f"Dice {f(m[name]['dice'], '%.2f')}", transform=ax[4].transAxes,
                   fontsize=13, va="top", bbox=dict(fc="white", ec="none", alpha=0.8, pad=2))
        ax[0].set_ylabel(names[name], fontsize=18, fontweight="bold", color=TISSUE_COLORS[c])
        for k, a in enumerate(ax):
            a.set_xticks([]); a.set_yticks([])
            if c == 0:
                a.set_title(cols[k], fontsize=16)

    cax_t = fig.add_subplot(gs[3, 1:3])
    cax_e = fig.add_subplot(gs[3, 3])
    cb = fig.colorbar(im_t, cax=cax_t, orientation="horizontal", extend="max")
    cb.set_label("thickness (mm), same scale for all", fontsize=13); cb.ax.tick_params(labelsize=12)
    cb = fig.colorbar(im_e, cax=cax_e, orientation="horizontal", extend="both")
    cb.set_label("error (mm)", fontsize=13); cb.ax.tick_params(labelsize=12)
    fig.savefig(path, dpi=100, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def volumes(gt, pred):
    """Approximate tissue volume: summed thickness (mm) x pixel area at the vine
    (mm2) -> mm3 -> cm3. Exact for parallel rays; approximate for a cone beam,
    since the area a pixel covers changes with depth. Shares are each tissue's
    volume divided by the total of the three."""
    vine = gt.sum(0) > 0
    t = [float(gt[c][vine].sum()) * PIXEL_AREA_AT_VINE_MM2 / 1000.0 for c in range(len(CLASSES))]
    p = [float(pred[c][vine].sum()) * PIXEL_AREA_AT_VINE_MM2 / 1000.0 for c in range(len(CLASSES))]
    st, sp = max(sum(t), 1e-9), max(sum(p), 1e-9)
    return {name: {"true_cm3": t[c], "pred_cm3": p[c],
                   "true_share_pct": 100 * t[c] / st, "pred_share_pct": 100 * p[c] / sp}
            for c, name in enumerate(CLASSES)}


def sample_points(specimen, gt, pred, n=4000, seed=0):
    """A random subset of vine pixels, for the pred-vs-true scatter."""
    vine = np.flatnonzero(gt.sum(0) > 0)
    rng = np.random.default_rng(seed)
    idx = rng.choice(vine, min(n, vine.size), replace=False) if vine.size else vine
    g = gt.reshape(len(CLASSES), -1)[:, idx]
    p = pred.reshape(len(CLASSES), -1)[:, idx]
    return {"specimen": specimen, "group": group_of(specimen), "gt": g, "pred": p}


def save_pred_vs_true(samples, path, title="Predicted vs true thickness, per pixel"):
    """samples: list from sample_points. One figure PER TISSUE, saved as
    <path stem>_<tissue>.png. Each figure has one hexbin panel per plant group,
    in that group's colour (light = few pixels, dark = many). Dashed line =
    perfect prediction; above it = too much predicted."""
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap
    groups = [g for g in GROUP_COLORS if any(s["group"] == g for s in samples)]
    groups += sorted({s["group"] for s in samples} - set(groups))
    stem, ext = os.path.splitext(path)

    for c, name in enumerate(CLASSES):
        # axis range covers 99.5% of this tissue's pixels, so rare extreme
        # values do not squash the plot
        v = np.concatenate([np.concatenate([s["gt"][c], s["pred"][c]]) for s in samples])
        hi = max(float(np.percentile(v, 99.5)) if v.size else 1.0, 1.0)

        fig, ax = plt.subplots(1, len(groups), figsize=(4.6 * len(groups), 5.2),
                               squeeze=False, constrained_layout=True)
        fig.suptitle(f"{title}: {name}   (dashed line = perfect prediction, above it = too much predicted)",
                     fontweight="bold")
        for k, g in enumerate(groups):
            a = ax[0, k]
            sel = [s for s in samples if s["group"] == g]
            col = GROUP_COLORS.get(g, "#777777")
            cmap = LinearSegmentedColormap.from_list(g, ["#F4F4F2", col, "#1A1A1A"])
            x = np.concatenate([s["gt"][c] for s in sel])
            y = np.concatenate([s["pred"][c] for s in sel])
            hb = a.hexbin(x, y, gridsize=45, bins="log", mincnt=1, cmap=cmap,
                          extent=(0, hi, 0, hi), linewidths=0)
            a.plot([0, hi], [0, hi], "--", color="#333333", lw=1)
            a.set(xlim=(0, hi), ylim=(0, hi), xlabel="true thickness (mm)",
                  title=f"{g} ({len(sel)} vine{'s' if len(sel) > 1 else ''})")
            if k == 0:
                a.set_ylabel("predicted thickness (mm)")
            a.set_aspect("equal"); a.grid(alpha=0.25)
            fig.colorbar(hb, ax=a, shrink=0.8, label="number of pixels (log)")
        fig.savefig(f"{stem}_{name}{ext}", dpi=100)
        plt.close(fig)


PROFILE_BIN_MM = 2.0       # height step of the trunk profiles
PROFILE_SMOOTH_CM = 1.0    # smoothing along the height (Gaussian sigma); 0 = none
PROFILE_MIN_WOOD = 0.05    # hide heights with less wood than 5% of the vine's
                           # widest part: too little wood, shares jump around


def height_profile(specimen, gt, pred, bin_mm=PROFILE_BIN_MM, smooth_cm=PROFILE_SMOOTH_CM,
                   min_wood=PROFILE_MIN_WOOD):
    """Wood composition along the trunk. Image rows = heights. 0 = the lowest
    row that contains wood (bottom of the vine), heights go up from there.
    Rows are grouped in bins of bin_mm. The summed thickness of each tissue is
    smoothed along the height, then share of a tissue = its thickness / the
    thickness of all three tissues, in %. Only pixels inside the true vine are
    used, for true and predicted alike."""
    from scipy.ndimage import gaussian_filter1d
    vine = gt.sum(0) > 0
    rows = np.flatnonzero(vine.any(1))
    if rows.size == 0:
        return {"specimen": specimen, "group": group_of(specimen),
                "height_mm": np.zeros(0), "true": np.zeros((3, 0)), "pred": np.zeros((3, 0)),
                "true_amount": np.zeros((3, 0)), "pred_amount": np.zeros((3, 0))}
    bottom = rows.max()
    px_mm = DET_PIXEL_MM * SPD_MM / SID_MM                     # pixel size at the vine
    t_row = np.stack([np.where(vine, gt[c], 0).sum(1) for c in range(len(CLASSES))])   # (3, H)
    p_row = np.stack([np.where(vine, pred[c], 0).sum(1) for c in range(len(CLASSES))])
    height = (bottom - np.arange(gt.shape[1])) * px_mm         # mm above the bottom
    keep = (height >= 0) & vine.any(1)
    b = np.floor(height[keep] / bin_mm).astype(int)
    nb = b.max() + 1
    T = np.zeros((3, nb)); P = np.zeros((3, nb))
    for c in range(len(CLASSES)):
        np.add.at(T[c], b, t_row[c][keep]); np.add.at(P[c], b, p_row[c][keep])
    T_raw, P_raw = T.copy(), P.copy()          # unsmoothed amounts, for the correlations
    if smooth_cm > 0:
        sigma = smooth_cm * 10.0 / bin_mm
        T = gaussian_filter1d(T, sigma, axis=1, mode="nearest")
        P = gaussian_filter1d(P, sigma, axis=1, mode="nearest")
    wood = T.sum(0)
    ok = wood >= min_wood * wood.max()
    with np.errstate(invalid="ignore", divide="ignore"):
        ts = np.where(ok, 100 * T / T.sum(0, keepdims=True), np.nan)
        ps = np.where(ok, 100 * P / P.sum(0, keepdims=True), np.nan)
    return {"specimen": specimen, "group": group_of(specimen),
            "height_mm": (np.arange(nb) + 0.5) * bin_mm, "true": ts, "pred": ps,
            "true_amount": np.where(ok, T_raw, np.nan), "pred_amount": np.where(ok, P_raw, np.nan)}


PROFILE_PAIRS = [(0, 1, "healthy–necrosis"), (0, 2, "healthy–white rot"), (1, 2, "necrosis–white rot")]


def profile_correlations(sel, key):
    """Spearman r between the amounts of two tissues along the trunk, pooled
    over the height bins of the vines in sel. Amounts (summed thickness per
    height, unsmoothed), not shares: shares add up to 100% at every height,
    which would make the healthy-disease correlation negative by construction."""
    out = {}
    for i, j, name in PROFILE_PAIRS:
        x = np.concatenate([p[key][i] for p in sel if p.get(key) is not None])
        y = np.concatenate([p[key][j] for p in sel if p.get(key) is not None])
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 5 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
            out[name] = np.nan
            continue
        rx = np.argsort(np.argsort(x[ok])); ry = np.argsort(np.argsort(y[ok]))
        out[name] = float(np.corrcoef(rx, ry)[0, 1])
    return out


def save_profiles(profiles, path, title="Wood composition along the trunk"):
    """One panel per plant group, side by side, all three tissues in each:
    solid = true, dashed = predicted, one colour per tissue. Vertical axis:
    height above the bottom of the vine. Horizontal axis: share of the wood at
    that height. With several vines in a group, the line is their mean."""
    plt = _plt()
    groups = [g for g in GROUP_COLORS if any(p["group"] == g for p in profiles)]
    groups += sorted({p["group"] for p in profiles} - set(groups))
    nb = max((p["height_mm"].size for p in profiles), default=0)
    if nb == 0:
        return
    h_cm = (np.arange(nb) + 0.5) * PROFILE_BIN_MM / 10.0
    labels = {"healthy": "healthy", "necrosis": "necrosis", "white_rot": "white rot"}

    def mean_over(sel, key, c):
        v = np.full((len(sel), nb), np.nan)
        for i, p in enumerate(sel):
            v[i, :p[key].shape[1]] = p[key][c]
        with warnings.catch_warnings():               # heights no vine reaches -> NaN, fine
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(v, 0)

    fig, ax = plt.subplots(1, len(groups), figsize=(4.4 * len(groups), 6.6),
                           squeeze=False, sharey=True, constrained_layout=True)
    fig.suptitle(f"{title}   (solid = true, dashed = predicted; smoothed over "
                 f"{PROFILE_SMOOTH_CM:g} cm)", fontweight="bold")
    top = 0.0
    for k, g in enumerate(groups):
        a = ax[0, k]
        sel = [p for p in profiles if p["group"] == g]
        for c, name in enumerate(CLASSES):
            t, p = mean_over(sel, "true", c), mean_over(sel, "pred", c)
            a.plot(t, h_cm, color=TISSUE_COLORS[c], lw=2.2, ls="-", label=f"{labels[name]}, true")
            a.plot(p, h_cm, color=TISSUE_COLORS[c], lw=2.2, ls="--", label=f"{labels[name]}, predicted")
            ok = np.isfinite(t)
            if ok.any():
                top = max(top, h_cm[ok].max())
        rt, rp = profile_correlations(sel, "true_amount"), profile_correlations(sel, "pred_amount")
        corr = "\n".join(f"{n}:  {rt[n]:+.2f} true,  {rp[n]:+.2f} pred." for _, _, n in PROFILE_PAIRS)
        a.set(xlim=(0, 100), title=f"{g} ({len(sel)} vine{'s' if len(sel) > 1 else ''})")
        a.set_xlabel("share of the wood (%)\n\nSpearman r along the trunk (amounts):\n" + corr,
                     fontsize=9.5)
        if k == 0:
            a.set_ylabel("height above the bottom of the vine (cm)")
        a.grid(alpha=0.25)
    for a in ax[0]:
        a.set_ylim(0, top + 0.5)
    handles, labs = ax[0, 0].get_legend_handles_labels()
    try:                                   # matplotlib >= 3.7: legend below the panels
        fig.legend(handles, labs, loc="outside lower center", ncol=3, fontsize=10, frameon=False)
    except ValueError:                     # older matplotlib: legend inside the last panel
        ax[0, -1].legend(handles, labs, loc="upper right", fontsize=8.5, framealpha=0.9)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def save_volume_plot(per_image, path, title):
    """Row 1: volume (cm3), row 2: share of the wood (%), true vs predicted,
    one panel per tissue, one pair of bars per vine."""
    plt = _plt()
    names = {"healthy": "Healthy", "necrosis": "Necrosis", "white_rot": "White rot"}
    vines = [m["specimen"] for m in per_image]
    x = np.arange(len(vines)); w = 0.38
    fig, ax = plt.subplots(2, 3, figsize=(max(18, 1.5 * len(vines) * 3), 11), constrained_layout=True)
    fig.suptitle(title + "  (grey = true, coloured = predicted)", fontsize=18, fontweight="bold")
    for c, name in enumerate(CLASSES):
        for r, (kt, kp, lab) in enumerate((("true_cm3", "pred_cm3", "volume (cm\u00b3, approximate)"),
                                           ("true_share_pct", "pred_share_pct", "share of the wood (%)"))):
            a = ax[r, c]
            t = [m["volume"][name][kt] for m in per_image]
            p = [m["volume"][name][kp] for m in per_image]
            a.bar(x - w / 2, t, w, color="#B9BFBB", label="true")
            a.bar(x + w / 2, p, w, color=TISSUE_COLORS[c], label="predicted")
            a.set_xticks(x)
            a.set_xticklabels(vines, rotation=35, ha="right", fontsize=12)
            a.tick_params(axis="y", labelsize=12)
            a.set_title(names[name], fontsize=16, fontweight="bold", color=TISSUE_COLORS[c])
            a.set_ylabel(lab, fontsize=13)
            a.grid(axis="y", alpha=0.3)
            if r == 1:
                a.set_ylim(0, 100)
        ax[0, c].legend(fontsize=12)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def save_dumbbell(per_image, path, key="share"):
    """Dumbbell plot, one panel per tissue, one row per test vine (grouped by
    plant group): grey dot = true, coloured dot = predicted, line between
    them = the error. key: "share" (% of the wood) or "volume" (cm3)."""
    plt = _plt()
    kt, kp, xlab = {"share": ("true_share_pct", "pred_share_pct", "share of the wood (%)"),
                    "volume": ("true_cm3", "pred_cm3", "volume (cm\u00b3, approximate)")}[key]
    names = {"healthy": "Healthy", "necrosis": "Necrosis", "white_rot": "White rot"}
    vines = [m["specimen"] for m in per_image]
    groups = [group_of(v) for v in vines]
    y = np.arange(len(vines))[::-1]                # first vine at the top
    fig, ax = plt.subplots(1, 3, figsize=(18, 1.6 + 0.55 * len(vines)), sharey=True,
                           constrained_layout=True)
    fig.suptitle(f"{xlab[0].upper() + xlab[1:]}, true vs predicted per vine   "
                 "(grey = true, coloured = predicted)", fontsize=18, fontweight="bold")
    for c, name in enumerate(CLASSES):
        a = ax[c]
        t = np.array([m["volume"][name][kt] for m in per_image])
        p = np.array([m["volume"][name][kp] for m in per_image])
        a.hlines(y, np.minimum(t, p), np.maximum(t, p), color=TISSUE_COLORS[c], lw=3, alpha=0.45)
        a.scatter(t, y, s=110, color="#9AA19D", zorder=3, label="true")
        a.scatter(p, y, s=110, color=TISSUE_COLORS[c], zorder=3, label="predicted (tissue colour)")
        for k in range(1, len(vines)):             # thin line between plant groups
            if groups[k] != groups[k - 1]:
                a.axhline(y[k] + 0.5, color="#CCCCCC", lw=1)
        a.set_title(names[name], fontsize=16, fontweight="bold", color=TISSUE_COLORS[c])
        a.set_xlabel(xlab, fontsize=13)
        a.tick_params(labelsize=12)
        a.grid(axis="x", alpha=0.3)
        a.set_xlim(left=0)
        if key == "share":
            a.set_xlim(0, 100)
    h, l = ax[0].get_legend_handles_labels()
    try:                                   # matplotlib >= 3.7: one legend below the panels
        fig.legend(h, l, loc="outside lower center", ncol=2, fontsize=13, frameon=False)
    except ValueError:                     # older matplotlib: inside the first panel
        ax[0].legend(h, l, fontsize=12, loc="lower right")
    ax[0].set_yticks(y)
    ax[0].set_yticklabels(vines, fontsize=12)
    for lab, g in zip(ax[0].get_yticklabels(), groups):
        lab.set_color(GROUP_COLORS.get(g, "#333333"))
    ax[0].set_ylim(-0.6, len(vines) - 0.4)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def save_volume_csv(per_image, path):
    with open(path, "w") as f:
        f.write("vine,group," + ",".join(
            f"{n}_{k}" for n in CLASSES
            for k in ("true_cm3", "pred_cm3", "true_share_pct", "pred_share_pct")) + "\n")
        for m in per_image:
            f.write(f"{m['specimen']},{group_of(m['specimen'])}," + ",".join(
                f"{m['volume'][n][k]:.3f}" for n in CLASSES
                for k in ("true_cm3", "pred_cm3", "true_share_pct", "pred_share_pct")) + "\n")


def save_loss(losses, fold_dir):
    with open(os.path.join(fold_dir, "loss.csv"), "w") as f:
        f.write("epoch,train_l1_mm\n")
        for i, v in enumerate(losses, 1):
            f.write(f"{i},{v:.5f}\n")
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(1, len(losses) + 1), losses, color="#3B5B7A")
    ax.set(xlabel="epoch", ylabel="train L1 (mm)", title="Training loss")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fold_dir, "loss_curve.png"), dpi=100)
    plt.close(fig)


def bar_metrics(rows, labels, path, title):
    """rows: list of metric dicts (one per vine or group), labels: x labels."""
    plt = _plt()
    colors = TISSUE_COLORS
    keys = [("mae_mm", "MAE (mm)"),
            ("dice", "Dice"),
            ("amount_ratio", "amount predicted / true, 1 is right")]
    names = {"healthy": "Healthy", "necrosis": "Necrosis", "white_rot": "White rot"}
    x = np.arange(len(labels)); w = 0.26
    fig, ax = plt.subplots(1, 3, figsize=(max(18, 1.9 * len(labels) * 3), 6.5), layout="constrained")
    fig.suptitle(title, fontsize=18, fontweight="bold")
    for k, (key, lab) in enumerate(keys):
        for c, name in enumerate(CLASSES):
            vals = [r[name][key] if r[name][key] is not None else np.nan for r in rows]
            ax[k].bar(x + (c - 1) * w, vals, w, label=names[name], color=colors[c])
        if key == "amount_ratio":
            ax[k].axhline(1.0, color="black", lw=1, ls="--")
        if key == "dice":
            ax[k].set_ylim(0, 1)
        ax[k].set_xticks(x); ax[k].set_xticklabels(labels, rotation=30, ha="right", fontsize=12)
        ax[k].tick_params(axis="y", labelsize=12)
        ax[k].set_title(lab, fontsize=15); ax[k].grid(axis="y", alpha=0.3)
    h, l = ax[0].get_legend_handles_labels()
    try:                                   # matplotlib >= 3.7: legend below the panels
        fig.legend(h, l, loc="outside lower center", ncol=3, fontsize=13, frameon=False)
    except ValueError:
        ax[0].legend(h, l, fontsize=12)
    fig.savefig(path, dpi=100)
    plt.close(fig)


# =========================================================================
# EVALUATION (used after training, and by --replot)
# =========================================================================

def evaluate_fold(model, test_specimens, train_specimens, fold_dir):
    """Predicts every test vine, writes metrics.json and all fold plots.
    Returns (summary, samples) where samples feed the global scatter."""
    model.eval()
    per_image, samples, profiles = [], [], []
    for s, d, l in list_pairs(test_specimens):
        img, gt = load_pair(d, l)
        pred = predict_full(model, img)
        m = metrics(pred, gt)
        m["volume"] = volumes(gt, pred)
        m["specimen"], m["file"] = s, os.path.basename(d)
        per_image.append(m)
        samples.append(sample_points(s, gt, pred))
        profiles.append(height_profile(s, gt, pred))
        save_preview(img, gt, pred, m, f"{s}  ({group_of(s)})",
                     os.path.join(fold_dir, f"preview_{s}.png"))
        del img, gt, pred

    summary = {"test": test_specimens, "train": train_specimens,
               "mean": mean_metrics(per_image), "per_image": per_image}
    with open(os.path.join(fold_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    save_pred_vs_true(samples, os.path.join(fold_dir, "pred_vs_true.png"))
    bar_metrics(per_image, [m["specimen"] for m in per_image],
                os.path.join(fold_dir, "metrics_per_vine.png"), "Test scores per vine")
    save_volume_plot(per_image, os.path.join(fold_dir, "volume_per_vine.png"),
                     "Tissue volume, true vs predicted")
    save_volume_csv(per_image, os.path.join(fold_dir, "volume_per_vine.csv"))
    save_profiles(profiles, os.path.join(fold_dir, "profile.png"))
    return summary, samples, profiles


def load_model(path):
    model = smp.Unet(encoder_name="resnet50", encoder_weights=None,
                     in_channels=1, classes=len(CLASSES))
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model.to(DEVICE)


def fold_dirs_by_training_time():
    """Fold folders that have a trained model, oldest training first. Uses
    model.pt's time, which --replot never changes, so 'latest' stays stable."""
    dirs = [os.path.dirname(p) for p in glob.glob(os.path.join(OUT_DIR, "fold_*", "model.pt"))]
    return sorted(dirs, key=lambda d: os.path.getmtime(os.path.join(d, "model.pt")))


def replot():
    """No training. Reloads each fold's model.pt, predicts its test vines again
    and redraws every plot. A fold whose test vines were all tested again by a
    later fold is skipped. Then writes the all-vine plots in OUT_DIR."""
    dirs = fold_dirs_by_training_time()
    if not dirs:
        raise SystemExit(f"No fold_*/model.pt in {OUT_DIR}")
    tests = {d: json.load(open(os.path.join(d, "metrics.json")))["test"] for d in dirs}
    latest = {}
    for d in dirs:
        for s in tests[d]:
            latest[s] = d
    by_vine, samples_by_vine, profiles_by_vine = {}, {}, {}
    for d in dirs:
        if not any(latest[s] == d for s in tests[d]):
            print(f"skip {os.path.basename(d)}: all its test vines were tested again later")
            continue
        meta = json.load(open(os.path.join(d, "metrics.json")))
        print(f"replot {os.path.basename(d)}")
        summary, samples, profiles = evaluate_fold(load_model(os.path.join(d, "model.pt")),
                                                   meta["test"], meta["train"], d)
        for m, smp_, prof in zip(summary["per_image"], samples, profiles):
            if latest[m["specimen"]] == d:
                by_vine[m["specimen"]] = m
                samples_by_vine[m["specimen"]] = smp_
                profiles_by_vine[m["specimen"]] = prof

    vines = sorted(by_vine, key=lambda s: (list(GROUP_COLORS).index(group_of(s))
                                           if group_of(s) in GROUP_COLORS else 99, s))
    per_image = [by_vine[s] for s in vines]
    save_pred_vs_true([samples_by_vine[s] for s in vines],
                      os.path.join(OUT_DIR, "pred_vs_true_all.png"),
                      f"Predicted vs true thickness, all {len(vines)} test vines")
    save_volume_plot(per_image, os.path.join(OUT_DIR, "volume_all.png"),
                     f"Tissue volume, true vs predicted, all {len(vines)} test vines")
    save_volume_csv(per_image, os.path.join(OUT_DIR, "volume_per_vine.csv"))
    for key in ("share", "volume"):
        save_dumbbell(per_image, os.path.join(OUT_DIR, f"dumbbell_{key}.png"), key)
    save_profiles([profiles_by_vine[s] for s in vines], os.path.join(OUT_DIR, "profile_all.png"),
                  f"Wood composition along the trunk, all {len(vines)} test vines")
    print(f"\nWritten to {OUT_DIR}: pred_vs_true_all_<tissue>.png (3 files), profile_all.png, "
          "volume_all.png, volume_per_vine.csv, dumbbell_share.png, dumbbell_volume.png")
    print("Each fold folder: new preview_*.png, pred_vs_true_<tissue>.png, profile.png, "
          "volume_per_vine.png/.csv")


# =========================================================================
# ONE FOLD
# =========================================================================

def crop_to_vine(img, lab, margin=AUG_CROP_MARGIN):
    """Keeps memory down: an augmented view is stored cropped to its vine."""
    ys, xs = np.nonzero(lab.sum(0) > 0)
    if ys.size == 0:
        return img, lab
    y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin + 1, img.shape[0])
    x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin + 1, img.shape[1])
    return img[y0:y1, x0:x1].copy(), lab[:, y0:y1, x0:x1].copy()


def make_augmented_set(train_specimens, round_idx, fold_dir, epoch):
    """AUG_VIEWS_PER_VINE new views per training vine. Each view's seed is
    derived from (vine, round, view), and every view is logged in
    augment_manifest.csv, so any of them can be rebuilt with augment.py."""
    import time
    import zlib
    import augment                                   # training/augment.py
    torch.cuda.empty_cache()                         # leave GPU room for the projector
    t0 = time.perf_counter()
    views = []
    manifest = os.path.join(fold_dir, "augment_manifest.csv")
    new_file = not os.path.exists(manifest) or round_idx == 0
    with open(manifest, "w" if new_file else "a") as f:
        if new_file:
            f.write("round,first_epoch,vine,seed,yaw,pitch,roll,spd,off_u,off_v,"
                    "bend_mm,bend_dir,elastic_mm,noise_std\n")
        for s in train_specimens:
            vols = augment.load_specimen(s)
            for v in range(AUG_VIEWS_PER_VINE):
                seed = zlib.crc32(f"{s}-{round_idx}-{v}".encode())
                p = augment.sample_params(seed)
                drr, lab = augment.render(vols, p)
                img = normalize(np.nan_to_num(drr))     # same scaling as the test images
                views.append(crop_to_vine(img, np.clip(np.nan_to_num(lab), 0, None)))
                f.write(f"{round_idx},{epoch + 1},{s},{seed}," + ",".join(
                    f"{p[k]:.4f}" for k in ("yaw", "pitch", "roll", "spd", "off_u", "off_v",
                                           "bend_mm", "bend_dir", "elastic_mm", "noise_std")) + "\n")
            del vols
    print(f"  augmented set {round_idx}: {len(views)} views in {time.perf_counter() - t0:.0f} s")
    return views


def load_stored_views(train_specimens, with_wr=False):
    """Stored views of the TRAINING vines only, memory-mapped (read from disk
    patch by patch). Made by training/make_views.py. with_wr (--balance): also
    the white rot pixels of each view, computed once and cached as *_wr.npy."""
    import time
    t0 = time.perf_counter()
    views = []
    for s in train_specimens:
        imgs = sorted(glob.glob(os.path.join(VIEWS_DIR, s, "*_img.npy")))
        if not imgs:
            raise SystemExit(f"No stored views for {s} in {VIEWS_DIR}: run make_views.py first")
        for ip in imgs:
            img = np.load(ip, mmap_mode="r")
            lab = np.load(ip.replace("_img.npy", "_lab.npy"), mmap_mode="r")
            fg = np.load(ip.replace("_img.npy", "_fg.npy"))
            if img.shape[0] < PATCH or img.shape[1] < PATCH:   # too small: pad in RAM
                img, lab = pad_to(np.asarray(img), PATCH, PATCH), pad_to(np.asarray(lab), PATCH, PATCH)
            if with_wr:
                wp = ip.replace("_img.npy", "_wr.npy")
                if os.path.exists(wp):
                    wr = np.load(wp)
                else:
                    wr = white_rot_points(lab, len(views))
                    np.save(wp, wr)
                views.append((img, lab, fg, wr))
            else:
                views.append((img, lab, fg))
    print(f"  stored views: {len(views)} from {len(train_specimens)} vines "
          f"({time.perf_counter() - t0:.0f} s)")
    return views


def run_fold(test_specimens, all_specimens, augment_on=False, stored_on=False, balance_on=False):
    global _WR_PROB
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    _WR_PROB = BALANCE_WR_PROB if balance_on else 0.0
    train_specimens = [s for s in all_specimens if s not in test_specimens]
    fold_dir = os.path.join(OUT_DIR, "fold_" + "_".join(test_specimens))
    os.makedirs(fold_dir, exist_ok=True)
    print(f"\n=== test: {test_specimens}   train: {train_specimens}")

    train_pairs = list_pairs(train_specimens)
    test_pairs = list_pairs(test_specimens)
    print(f"  {len(train_pairs)} train images, {len(test_pairs)} test images")
    originals = [load_pair(d, l) for _, d, l in train_pairs]
    train_data = list(originals)          # the batch generator reads this list
    weights = None
    if balance_on:
        w, mean = class_weights([lab for _, lab in originals])
        weights = torch.tensor(w, dtype=torch.float32, device=DEVICE).view(1, -1, 1, 1)
        train_data = [(img, lab, None, white_rot_points(lab, i)) for i, (img, lab) in enumerate(originals)]
        info = {"mean_thickness_mm": dict(zip(CLASSES, mean.round(3).tolist())),
                "loss_weight": dict(zip(CLASSES, w.round(3).tolist())),
                "power": BALANCE_POWER, "white_rot_patch_share": BALANCE_WR_PROB}
        with open(os.path.join(fold_dir, "balance.json"), "w") as f:
            json.dump(info, f, indent=2)
        print(f"  class balancing: loss weights {info['loss_weight']}, "
              f"{BALANCE_WR_PROB:.0%} of patches centred on white rot")
    if stored_on:
        train_data += load_stored_views(train_specimens, with_wr=balance_on)
        print(f"  + the {len(originals)} original views; left-right flip "
              f"{'on' if FLIP_LR else 'off'}")
    if augment_on:
        print(f"  augmentation on: {AUG_VIEWS_PER_VINE} views per vine, new set every "
              f"{AUG_EVERY_EPOCHS} epochs (+ the {len(originals)} original views)")

    model = build_model().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    use_amp = DEVICE == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    batches = batch_iter(train_data, flip_lr=stored_on and FLIP_LR)
    losses = []

    for ep in range(EPOCHS):
        if augment_on and ep % AUG_EVERY_EPOCHS == 0:
            # replace the previous augmented set, keep the originals
            train_data[:] = originals + make_augmented_set(
                train_specimens, ep // AUG_EVERY_EPOCHS, fold_dir, ep)
        model.train()
        total = total_w = 0.0
        for _ in range(STEPS_PER_EPOCH):
            x, y = next(batches)
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                err = (forward(model, x) - y).abs()
                l1 = err.mean()                                  # plain L1, as in every run
                loss = (err * weights).mean() if weights is not None else l1
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += l1.item(); total_w += loss.item()
        sched.step()
        losses.append(total / STEPS_PER_EPOCH)                   # plain L1: comparable across runs
        extra = f"  (weighted {total_w / STEPS_PER_EPOCH:.4f})" if weights is not None else ""
        print(f"  epoch {ep + 1:3d}/{EPOCHS}  train L1 = {losses[-1]:.4f} mm{extra}")

    torch.save(model.state_dict(), os.path.join(fold_dir, "model.pt"))
    save_loss(losses, fold_dir)

    # ---- test, touched once ------------------------------------------
    summary, _, _ = evaluate_fold(model, test_specimens, train_specimens, fold_dir)

    print("  test results (mean over test images):")
    for name, v in summary["mean"].items():
        print(f"    {name:10s} " + "  ".join(
            f"{k}={v[k]:.3f}" if v[k] is not None else f"{k}=n/a" for k in v))
    return summary


# =========================================================================
# SUMMARY OVER FOLDS
# =========================================================================

GROUP_NAMES = {"AS": "asymptomatic", "RES": "resistant", "S": "symptomatic", "APO": "apoplectic"}


def group_of(specimen):
    """CEP011_AS1 -> 'asymptomatic'. Unknown codes are returned as they are."""
    code = specimen.split("_", 1)[-1].rstrip("0123456789")
    return GROUP_NAMES.get(code, code)


def summarize():
    """Reads every OUT_DIR/fold_*/metrics.json. If a vine was tested in more
    than one fold, the most recent fold wins (and a warning is printed)."""
    files = [os.path.join(d, "metrics.json") for d in fold_dirs_by_training_time()
             if os.path.exists(os.path.join(d, "metrics.json"))]
    if not files:
        raise SystemExit(f"No fold_*/metrics.json found in {OUT_DIR}")
    per_vine, source = {}, {}
    for f in files:
        for m in json.load(open(f))["per_image"]:
            s = m["specimen"]
            if s in per_vine:
                print(f"  [warning] {s} tested in {source[s]} and {os.path.basename(os.path.dirname(f))}; keeping the latter")
            per_vine[s], source[s] = m, os.path.basename(os.path.dirname(f))
    vines = sorted(per_vine)
    print(f"{len(vines)} vines with test scores: {vines}")

    keys = ("mae_mm", "dice", "amount_ratio")
    fmt = lambda x: f"{x:.4f}" if x is not None else ""
    with open(os.path.join(OUT_DIR, "summary_per_vine.csv"), "w") as f:
        f.write("vine,group,fold," + ",".join(f"{c}_{k}" for c in CLASSES for k in keys) + "\n")
        for s in vines:
            f.write(f"{s},{group_of(s)},{source[s]}," + ",".join(
                fmt(per_vine[s][c][k]) for c in CLASSES for k in keys) + "\n")

    groups = [g for g in GROUP_NAMES.values() if any(group_of(s) == g for s in vines)]
    groups += sorted({group_of(s) for s in vines} - set(groups))
    by_group = [mean_metrics([per_vine[s] for s in vines if group_of(s) == g]) for g in groups]
    counts = [sum(group_of(s) == g for s in vines) for g in groups]
    overall = mean_metrics([per_vine[s] for s in vines])

    with open(os.path.join(OUT_DIR, "summary_per_group.csv"), "w") as f:
        f.write("group,n_vines," + ",".join(f"{c}_{k}" for c in CLASSES for k in keys) + "\n")
        for g, n, r in zip(groups + ["ALL"], counts + [len(vines)], by_group + [overall]):
            f.write(f"{g},{n}," + ",".join(fmt(r[c][k]) for c in CLASSES for k in keys) + "\n")

    bar_metrics(by_group + [overall], [f"{g} (n={n})" for g, n in zip(groups, counts)] + [f"all (n={len(vines)})"],
                os.path.join(OUT_DIR, "summary_per_group.png"), "Test scores per group")

    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for f in files:
        lc = os.path.join(os.path.dirname(f), "loss.csv")
        if os.path.exists(lc):
            v = np.loadtxt(lc, delimiter=",", skiprows=1)
            ax.plot(v[:, 0], v[:, 1], label=os.path.basename(os.path.dirname(f))[5:])
    ax.set(xlabel="epoch", ylabel="train L1 (mm)", title="Training loss per fold")
    ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "loss_curves_all.png"), dpi=100)
    plt.close(fig)

    print("\n=== per group (mean over vines)")
    for g, n, r in zip(groups + ["ALL"], counts + [len(vines)], by_group + [overall]):
        print(f"  {g:13s} n={n}  " + "  ".join(
            f"{c}: amount={r[c]['amount_ratio']:.2f}" if r[c]["amount_ratio"] is not None
            else f"{c}: amount=n/a" for c in CLASSES))
    print(f"\nWritten to {OUT_DIR}: summary_per_vine.csv, summary_per_group.csv, "
          "summary_per_group.png, loss_curves_all.png")


# =========================================================================

def main():
    global OUT_DIR
    for name, p in (("DRR_ROOT", DRR_ROOT), ("LABEL_ROOT", LABEL_ROOT), ("OUT_DIR", OUT_DIR)):
        if not p:
            raise SystemExit(f"Fill in {name} at the top of the script.")

    ap = argparse.ArgumentParser()
    ap.add_argument("--augment", action="store_true",
                    help="train with 3D augmentation (training/augment.py); results go to "
                         "OUT_DIR_AUG. Also give it to --summarize / --replot of those results.")
    ap.add_argument("--stored", action="store_true",
                    help="train with the views stored by training/make_views.py, plus a "
                         "left-right flip; results go to OUT_DIR_STORED. Also give it to "
                         "--summarize / --replot of those results.")
    ap.add_argument("--balance", action="store_true",
                    help="with --stored: class balancing (white rot patch sampling + loss "
                         "weights per tissue); results go to OUT_DIR_BALANCED. Also give it "
                         "to --summarize / --replot of those results.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--test", nargs="+", help="specimen names to hold out")
    g.add_argument("--all-folds", action="store_true",
                   help="leave-one-specimen-out over every specimen")
    g.add_argument("--summarize", action="store_true",
                   help="no training: summarize all folds already in OUT_DIR")
    g.add_argument("--replot", action="store_true",
                   help="no training: reload each fold's model.pt and redraw all plots")
    args = ap.parse_args()
    if args.augment and args.stored:
        raise SystemExit("Use --augment or --stored, not both.")
    if args.augment:
        OUT_DIR = OUT_DIR_AUG
    if args.stored:
        OUT_DIR = OUT_DIR_STORED
    if args.balance:
        if not args.stored:
            raise SystemExit("--balance works on top of --stored: give both.")
        OUT_DIR = OUT_DIR_BALANCED
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"results folder: {OUT_DIR}")

    if args.summarize:
        summarize()
        return
    if args.replot:
        replot()
        return

    specimens = list_specimens()
    print(f"Found {len(specimens)} specimens: {specimens}")

    if args.test:
        missing = [s for s in args.test if s not in specimens]
        if missing:
            raise SystemExit(f"Not found in DRR_ROOT: {missing}")
        run_fold(args.test, specimens, augment_on=args.augment, stored_on=args.stored,
                 balance_on=args.balance)
    else:
        results = [run_fold([s], specimens, augment_on=args.augment, stored_on=args.stored,
                            balance_on=args.balance)
                   for s in specimens]
        overall = mean_metrics([r["mean"] for r in results])
        with open(os.path.join(OUT_DIR, "all_folds.json"), "w") as f:
            json.dump({"mean_over_folds": overall, "folds": results}, f, indent=2)
        print("\n=== mean over all folds")
        for name, v in overall.items():
            print(f"  {name:10s} " + "  ".join(
                f"{k}={v[k]:.3f}" if v[k] is not None else f"{k}=n/a" for k in v))


if __name__ == "__main__":
    main()