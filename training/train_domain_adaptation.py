#!/usr/bin/env python3
"""
train_domain_adaptation.py  -  domain adaptation (DANN) on top of the stored-views training

Same U-Net and D2 training as `train_baseline.py --stored`, plus:
  * D1 2026 images in every step: registered DRRs (labelled "simulated")
    and portable radiographs, PXR (labelled "real"). No D1 thickness labels.
  * Two per-location domain classifiers, on encoder stage 3 (1024 channels,
    1/16 of the patch: 32 x 32 for 512 px) and stage 4 (2048 channels, 1/32:
    16 x 16). Each says, for every location, "simulated or real".
  * A gradient reversal layer between the encoder and each classifier:
    the classifiers learn to tell the domains apart, the encoder is pushed
    the other way, to make them impossible to tell apart.
  * Loss = L1 (D2 thickness) + domain loss (binary cross-entropy, stage 3 +
    stage 4). The simulated and real sides are averaged separately, then
    together, so they count equally. lambda scales the reversed gradient
    that reaches the encoder; it rises from 0 to LAMBDA_MAX during training
    (standard DANN schedule).
The classifiers are only used for training. model.pt holds the plain U-Net,
so --replot, --summarize and compare_runs.py work on it as on any other run.

D1 split (fixed, written to d1_split.txt in the results folder):
  * the 20 vines with a registered DRR: D1_HELD_OUT_N picked at random
    (seed D1_SPLIT_SEED) are never used in training, only for the agreement
    check; the others are used in training (registered DRR + all PXRs).
  * vines with PXRs but no DRR (not in the monitoring dataset) are not used.

USAGE (from the repo root)
    python training/train_domain_adaptation.py --check-d1            # look at the inputs first
    python training/train_domain_adaptation.py --folds               # the 3 usual folds
    python training/train_domain_adaptation.py --test A B C D        # one fold
    python training/train_domain_adaptation.py --summarize           # as train_baseline --summarize
    python training/train_domain_adaptation.py --replot
    python training/train_domain_adaptation.py --eval-d1             # agreement on held-out D1 vines
    python training/train_domain_adaptation.py --eval-d1 --run-dir results/training_stored   # same, other run
Add --lambda-max 0.1 (or 1.0) to any command for another lambda; each lambda
has its own results folder.

OUTPUT, all under results/domain_adaptation/
    lam0.3/                         folds, model.pt, plots, summaries (like any run)
    check_inputs.png                from --check-d1
    d1_agreement_lam0.3/            from --eval-d1
    d1_agreement_training_stored/   from --eval-d1 --run-dir results/training_stored
"""

import argparse
import csv
import glob
import inspect
import json
import math
import os
import random
import sys

import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_baseline as tb          # noqa: E402
import augment                       # noqa: E402

# ---- settings -------------------------------------------------------------
REPO = tb.REPO
PXR_ROOT = os.path.join(REPO, "dataset", "radiograph_portable_2026_tif")
D1_DRR_PATTERN = os.path.join(REPO, "results", "CEP_{v}_airzero_calibration_linear.tif")
LAMBDA_MAX = 0.3
N_D1_DRR = 8                 # D1 DRR patches per step (D2 patches: tb.BATCH)
N_D1_PXR = 8                 # PXR patches per step
D1_HELD_OUT_N = 5
D1_SPLIT_SEED = 0
FG_POINTS = 20000            # vine pixels kept per D1 image for patch centring
CLF_STAGES = (1024, 2048)    # encoder stages the classifiers look at (channels)
STAGE_NAME = {1024: "stage3", 2048: "stage4"}
FOLDS = [                    # same folds as the earlier runs
    ["CEP011_AS1", "CEP014_RES1", "CEP017_S1", "CEP020_APO1"],
    ["CEP012_AS2", "CEP015_RES2", "CEP018_S2", "CEP021_APO2"],
    ["CEP013_AS3", "CEP016_RES3", "CEP019_S3", "CEP022_APO3"],
]

# the 20 D1 vines with a registered DRR, and the PXR each DRR is registered to
# (from specimen_table.txt)
PAIRED_PXR = {
    "313B": "69ca5d9ee42f5e088ff0f93b.tif", "318": "69ca5db1e42f5e09df2bb4a6.tif",
    "322": "69ca5dc4e42f5e09df2bb4ac.tif", "323": "69ca5de6e42f5e09b4512919.tif",
    "330": "69ca5df7e42f5e09df2bb4b0.tif", "335": "69ca5e1be42f5e0a22520ec4.tif",
    "368B": "69ca536b2ad9c81239913207.tif", "378A": "69ca4b43e42f5e026f6dd0e7.tif",
    "378B": "69ca5613e42f5e07eeeed92b.tif", "380A": "69ca562de42f5e0682a2cf2d.tif",
    "764B": "69ca5d80e42f5e088ff0f937.tif", "988B": "69ca5645e42f5e0682a2cf31.tif",
    "1181": "69ca6a532ad9c81239913234.tif", "1186A": "69ca569de42f5e071ae9bcf3.tif",
    "1189": "69ca67ad2ad9c8123991322c.tif", "1191": "69ca5e44e42f5e0a22520ec8.tif",
    "1193": "69ca5e6fe42f5e09df2bb4b4.tif", "1195": "69ca56b6e42f5e071ae9bcfa.tif",
    "1266A": "69ca56d2e42f5e0702f748da.tif", "2184A": "69ca5676e42f5e071ae9bced.tif",
}


DA_ROOT = os.path.join(REPO, "results", "domain_adaptation")   # everything of this script


def out_dir(lam):
    return os.path.join(DA_ROOT, f"lam{lam:g}")


# =========================================================================
# D1 DATA
# =========================================================================

def d1_split():
    vines = sorted(PAIRED_PXR)
    held = sorted(str(v) for v in
                  np.random.default_rng(D1_SPLIT_SEED).choice(vines, D1_HELD_OUT_N, replace=False))
    return [v for v in vines if v not in held], held


def pxr_files(v):
    """All 2026 PXRs of a vine (not the line_integral files)."""
    fs = sorted(glob.glob(os.path.join(PXR_ROOT, f"CEP_{v}", "*.tif")))
    return [f for f in fs if "line_integral" not in os.path.basename(f)]


def load_d1_drr(v):
    """Registered linear DRR -> the same DRR->PXR curve as D2 -> normalize."""
    raw = np.nan_to_num(tifffile.imread(D1_DRR_PATTERN.format(v=v)).astype(np.float32))
    if raw.max() > 1.5:                     # stored as 16-bit (0..65535)
        raw = raw / 65535.0
    return tb.normalize(augment.apply_curve(raw).astype(np.float32))


def load_pxr(path):
    return tb.normalize(np.nan_to_num(tifffile.imread(path).astype(np.float32)))


def otsu(img):
    h, edges = np.histogram(img, bins=256, range=(0, 1))
    c = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(h); w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1)
    m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    return c[np.argmax(w0 * w1 * (m0 - m1) ** 2)]


def vine_mask(img):
    """No labels on D1: the vine = pixels brighter than the Otsu threshold."""
    return img > otsu(img)


def with_fg(img, seed):
    fg = np.argwhere(vine_mask(img)).astype(np.int32)
    if len(fg) > FG_POINTS:
        fg = fg[np.random.default_rng(seed).choice(len(fg), FG_POINTS, replace=False)]
    return img, fg


def d1_patches(data, n, noise=False):
    """n patches from a list of (img, fg) D1 images, same sampling as D2
    (FG_PROB centred on the vine), random left-right flip; noise=True adds
    the same 2D noise as the D2 views (for the D1 DRRs)."""
    xs = []
    for _ in range(n):
        img, fg = random.choice(data)
        x = _patch(img, fg)
        if noise:
            x = x + np.random.normal(0.0, random.uniform(*augment.NOISE_STD), x.shape).astype(np.float32)
        if random.random() < 0.5:
            x = x[:, ::-1]
        xs.append(np.ascontiguousarray(x)[None])
    return torch.from_numpy(np.stack(xs))


def _patch(img, fg):
    P = tb.PATCH
    img = tb.pad_to(img, P, P)
    H, W = img.shape
    if random.random() < tb.FG_PROB and len(fg):
        cy, cx = fg[random.randrange(len(fg))]
        y0 = int(np.clip(cy - P // 2, 0, H - P)); x0 = int(np.clip(cx - P // 2, 0, W - P))
    else:
        y0, x0 = random.randint(0, H - P), random.randint(0, W - P)
    return img[y0:y0 + P, x0:x0 + P].astype(np.float32)


def load_d1_train(vines):
    drrs, pxrs = [], []
    for i, v in enumerate(vines):
        drrs.append(with_fg(load_d1_drr(v), i))
        for j, f in enumerate(pxr_files(v)):
            pxrs.append(with_fg(load_pxr(f), 1000 * i + j))
    return drrs, pxrs


# =========================================================================
# MODEL PARTS
# =========================================================================

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class DomainClassifier(nn.Module):
    """Per-location: one simulated/real logit for every location of the map."""
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, 256, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 128, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, 1))

    def forward(self, f, lam):
        return self.net(GradReverse.apply(f, lam))


def lambda_at(step, total, lam_max):
    p = step / max(total, 1)
    return lam_max * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)


def split_forward(model, x, n_lab):
    """Encoder on the whole batch (shared BatchNorm statistics), decoder only
    on the first n_lab patches (the D2 ones, the only ones with labels).
    Returns (thickness of the D2 patches, {channels: encoder feature map})."""
    feats = model.encoder(x)
    stage = {f.shape[1]: f for f in feats if f.dim() == 4 and f.shape[1] in CLF_STAGES}
    lab_feats = [f[:n_lab] for f in feats]
    if any(p.kind == inspect.Parameter.VAR_POSITIONAL
           for p in inspect.signature(model.decoder.forward).parameters.values()):
        dec = model.decoder(*lab_feats)          # older segmentation_models_pytorch
    else:
        dec = model.decoder(lab_feats)
    return F.softplus(model.segmentation_head(dec)), stage


@torch.no_grad()
def check_split_forward(model):
    """split_forward must give the same thickness as the normal forward."""
    model.eval()
    x = torch.randn(2, 1, tb.PATCH, tb.PATCH, device=tb.DEVICE)
    a = tb.forward(model, x)
    b, stage = split_forward(model, x, 2)
    err = (a - b).abs().max().item()
    shapes = {c: tuple(f.shape[-2:]) for c, f in stage.items()}
    if err > 1e-4 or sorted(stage) != sorted(CLF_STAGES):
        raise SystemExit(f"split_forward check failed: max diff {err}, stages {shapes}")
    print(f"  split forward checked (max diff {err:.1e}); classifier maps {shapes}")
    model.train()


def domain_loss(logits, n_sim):
    """BCE per location; simulated (first n_sim, target 0) and real (target 1)
    averaged separately, then together. Also returns the balanced accuracy."""
    t = torch.zeros_like(logits); t[n_sim:] = 1
    l = F.binary_cross_entropy_with_logits(logits.float(), t, reduction="none")
    loss = 0.5 * (l[:n_sim].mean() + l[n_sim:].mean())
    pred = logits.detach() > 0
    acc = 0.5 * ((~pred[:n_sim]).float().mean() + pred[n_sim:].float().mean())
    return loss, acc.item()


# =========================================================================
# TRAINING
# =========================================================================

def run_fold(test_specimens, all_specimens, lam_max):
    random.seed(tb.SEED); np.random.seed(tb.SEED); torch.manual_seed(tb.SEED)
    train_specimens = [s for s in all_specimens if s not in test_specimens]
    fold_dir = os.path.join(tb.OUT_DIR, "fold_" + "_".join(test_specimens))
    os.makedirs(fold_dir, exist_ok=True)
    print(f"\n=== test: {test_specimens}   train: {train_specimens}   lambda_max {lam_max}")

    # D2, exactly as --stored
    originals = [tb.load_pair(d, l) for _, d, l in tb.list_pairs(train_specimens)]
    d2_data = originals + tb.load_stored_views(train_specimens)
    d2_batches = tb.batch_iter(d2_data, flip_lr=tb.FLIP_LR)

    # D1
    d1_train, d1_held = d1_split()
    d1_drr, d1_pxr = load_d1_train(d1_train)
    print(f"  D1: {len(d1_drr)} DRRs + {len(d1_pxr)} PXRs from {len(d1_train)} vines; "
          f"held out: {d1_held}")

    model = tb.build_model().to(tb.DEVICE)
    check_split_forward(model)
    clfs = nn.ModuleDict({str(c): DomainClassifier(c) for c in CLF_STAGES}).to(tb.DEVICE)
    opt = torch.optim.AdamW(list(model.parameters()) + list(clfs.parameters()), lr=tb.LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, tb.EPOCHS)
    use_amp = tb.DEVICE == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    total_steps = tb.EPOCHS * tb.STEPS_PER_EPOCH
    n_lab = tb.BATCH
    n_sim = n_lab + N_D1_DRR
    log, step = [], 0

    for ep in range(tb.EPOCHS):
        model.train(); clfs.train()
        s_l1 = s_dom = 0.0
        s_acc = {c: 0.0 for c in CLF_STAGES}
        for _ in range(tb.STEPS_PER_EPOCH):
            lam = lambda_at(step, total_steps, lam_max)
            x2, y2 = next(d2_batches)
            x = torch.cat([x2, d1_patches(d1_drr, N_D1_DRR, noise=True),
                           d1_patches(d1_pxr, N_D1_PXR)]).to(tb.DEVICE)
            y2 = y2.to(tb.DEVICE)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                pred, stage = split_forward(model, x, n_lab)
                l1 = F.l1_loss(pred, y2)
                dom = 0.0
                for c in CLF_STAGES:
                    d, acc = domain_loss(clfs[str(c)](stage[c], lam), n_sim)
                    dom = dom + d
                    s_acc[c] += acc
                loss = l1 + dom
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            s_l1 += l1.item(); s_dom += dom.item()
            step += 1
        sched.step()
        n = tb.STEPS_PER_EPOCH
        row = {"epoch": ep + 1, "lambda": lam, "train_L1_mm": s_l1 / n, "domain_loss": s_dom / n}
        row.update({f"domain_acc_{STAGE_NAME[c]}": s_acc[c] / n for c in CLF_STAGES})
        log.append(row)
        print(f"  epoch {ep + 1:3d}/{tb.EPOCHS}  L1 {row['train_L1_mm']:.3f} mm  "
              f"domain {row['domain_loss']:.3f}  acc " +
              " ".join(f"{row[f'domain_acc_{STAGE_NAME[c]}']:.2f}" for c in CLF_STAGES) +
              f"  lambda {lam:.3f}")

    torch.save(model.state_dict(), os.path.join(fold_dir, "model.pt"))      # plain U-Net
    torch.save(clfs.state_dict(), os.path.join(fold_dir, "domain_classifiers.pt"))
    with open(os.path.join(fold_dir, "training_log.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0])); w.writeheader(); w.writerows(log)
    tb.save_loss([r["train_L1_mm"] for r in log], fold_dir)
    save_domain_plot(log, fold_dir)

    summary, _, _ = tb.evaluate_fold(model, test_specimens, train_specimens, fold_dir)
    print("  D2 test results (mean over test images):")
    for name, v in summary["mean"].items():
        print(f"    {name:10s} " + "  ".join(
            f"{k}={v[k]:.3f}" if v[k] is not None else f"{k}=n/a" for k in v))
    return summary


def save_domain_plot(log, fold_dir):
    plt = tb._plt()
    ep = [r["epoch"] for r in log]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    ax[0].plot(ep, [r["domain_loss"] for r in log], color="#2F4558", lw=2)
    ax[0].set_title("Domain loss (stage 3 + stage 4)", fontsize=14); ax[0].set_xlabel("epoch")
    for c, col in zip(CLF_STAGES, ("#7D8FA3", "#2F4558")):
        ax[1].plot(ep, [r[f"domain_acc_{STAGE_NAME[c]}"] for r in log], color=col, lw=2,
                   label=STAGE_NAME[c].replace("stage", "stage "))
    ax[1].axhline(0.5, color="#B71C1C", ls="--", lw=1, label="chance (0.5)")
    ax[1].set_ylim(0.4, 1.0); ax[1].legend()
    ax[1].set_title("Domain classifier accuracy (balanced)", fontsize=14); ax[1].set_xlabel("epoch")
    fig.savefig(os.path.join(fold_dir, "domain_training.png"), dpi=100)
    plt.close(fig)


# =========================================================================
# CHECKS AND D1 EVALUATION
# =========================================================================

def check_d1():
    """One figure to compare the inputs before training: a D2 view, a D1 DRR
    and its registered PXR, all as the model sees them, with histograms."""
    train, held = d1_split()
    v = train[0]
    d2_img, _ = tb.load_pair(*tb.list_pairs([tb.list_specimens()[0]])[0][1:])
    drr = load_d1_drr(v)
    pxr = load_pxr(os.path.join(PXR_ROOT, f"CEP_{v}", PAIRED_PXR[v]))
    plt = tb._plt()
    fig, ax = plt.subplots(2, 3, figsize=(18, 11), layout="constrained")
    items = [("D2 DRR (simulated)", d2_img), (f"D1 DRR CEP_{v} (simulated)", drr),
             (f"D1 PXR CEP_{v} (real)", pxr)]
    for k, (t, im) in enumerate(items):
        ax[0, k].imshow(im, cmap="gray", vmin=0, vmax=1); ax[0, k].set_title(t, fontsize=14)
        ax[0, k].contour(vine_mask(im), levels=[0.5], colors="#F9A825", linewidths=0.8)
        ax[0, k].set_xticks([]); ax[0, k].set_yticks([])
        m = vine_mask(im)
        ax[1, k].hist(im[m], bins=100, range=(0, 1), color="#2F4558")
        ax[1, k].set_title("histogram inside the vine (yellow outline)", fontsize=12)
        ax[1, k].set_xlabel("normalized intensity")
    fig.suptitle("Check before domain adaptation: same polarity (wood bright?), similar range, "
                 "vine outline correct?", fontsize=15, fontweight="bold")
    os.makedirs(DA_ROOT, exist_ok=True)
    path = os.path.join(DA_ROOT, "check_inputs.png")
    fig.savefig(path, dpi=90); plt.close(fig)
    print(f"D1 train vines: {train}\nD1 held-out vines: {held}")
    print("PXRs per training vine: " + ", ".join(f"{v}:{len(pxr_files(v))}" for v in train))
    print(f"written: {path}")


@torch.no_grad()
def eval_d1(run_dir):
    """For each held-out D1 vine: predict on the registered DRR and on its PXR
    with every fold model of run_dir, compare the two (agreement, NOT accuracy:
    there is no ground truth on D1)."""
    _, held = d1_split()
    models = sorted(glob.glob(os.path.join(run_dir, "fold_*", "model.pt")))
    if not models:
        raise SystemExit(f"No fold_*/model.pt in {run_dir}")
    out = os.path.join(DA_ROOT, "d1_agreement_" + os.path.basename(os.path.normpath(run_dir)))
    os.makedirs(out, exist_ok=True)
    rows = []
    for v in held:
        drr = load_d1_drr(v)
        pxr = load_pxr(os.path.join(PXR_ROOT, f"CEP_{v}", PAIRED_PXR[v]))
        mask = vine_mask(drr)
        p_d, p_p = [], []
        for mp in models:
            m = tb.load_model(mp); m.eval()
            p_d.append(tb.predict_full(m, drr)); p_p.append(tb.predict_full(m, pxr))
        p_d, p_p = np.mean(p_d, 0), np.mean(p_p, 0)          # mean of the fold models
        row = {"vine": v}
        tot_d, tot_p = p_d[:, mask].sum(), p_p[:, mask].sum()
        for c, name in enumerate(tb.CLASSES):
            row[f"{name}_mean_abs_diff_mm"] = float(np.abs(p_d[c] - p_p[c])[mask].mean())
            row[f"{name}_share_on_drr_pct"] = float(100 * p_d[c][mask].sum() / max(tot_d, 1e-6))
            row[f"{name}_share_on_pxr_pct"] = float(100 * p_p[c][mask].sum() / max(tot_p, 1e-6))
        rows.append(row)
        save_d1_figure(v, drr, pxr, p_d, p_p, mask, row, os.path.join(out, f"CEP_{v}.png"))
        print(f"  CEP_{v}: " + "  ".join(
            f"{n} |diff| {row[f'{n}_mean_abs_diff_mm']:.1f} mm, share DRR/PXR "
            f"{row[f'{n}_share_on_drr_pct']:.0f}/{row[f'{n}_share_on_pxr_pct']:.0f}%"
            for n in tb.CLASSES))
    mean = {"vine": "MEAN"}
    for k in rows[0]:
        if k != "vine":
            mean[k] = float(np.mean([r[k] for r in rows]))
    rows.append(mean)
    with open(os.path.join(out, "d1_agreement.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
        for r in rows:
            w.writerow({k: (f"{x:.3f}" if isinstance(x, float) else x) for k, x in r.items()})
    print(f"{len(models)} fold models, mean prediction. Written to {out}/")


def save_d1_figure(v, drr, pxr, p_d, p_p, mask, row, path):
    plt = tb._plt()
    names = {"healthy": "Healthy", "necrosis": "Necrosis", "white_rot": "White rot"}
    ry, rx = tb.vine_crop(mask[None].astype(np.float32))
    fig = plt.figure(figsize=(18, 15), layout="constrained")
    gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1, 0.12])
    fig.suptitle(f"CEP_{v} (held out): prediction on the registered DRR vs on the real PXR",
                 fontsize=18, fontweight="bold")
    for k, (t, im) in enumerate([("DRR (simulated)", drr), ("PXR (real)", pxr)]):
        a = fig.add_subplot(gs[k, 0]); a.imshow(im[ry, rx], cmap="gray", vmin=0, vmax=1)
        a.set_title(t, fontsize=14); a.set_xticks([]); a.set_yticks([])
    for c, name in enumerate(tb.CLASSES):
        a1 = fig.add_subplot(gs[c, 1]); a2 = fig.add_subplot(gs[c, 2]); a3 = fig.add_subplot(gs[c, 3])
        im_t = a1.imshow(p_d[c][ry, rx], cmap="magma", vmin=0, vmax=tb.THICK_VMAX_MM)
        a2.imshow(p_p[c][ry, rx], cmap="magma", vmin=0, vmax=tb.THICK_VMAX_MM)
        im_e = a3.imshow(np.where(mask, p_p[c] - p_d[c], np.nan)[ry, rx], cmap="RdBu_r",
                         vmin=-tb.ERR_VMAX_MM, vmax=tb.ERR_VMAX_MM)
        a3.text(0.03, 0.97, f"mean |diff| {row[f'{name}_mean_abs_diff_mm']:.1f} mm",
                transform=a3.transAxes, fontsize=12, va="top",
                bbox=dict(fc="white", ec="none", alpha=0.8, pad=2))
        a1.set_ylabel(names[name], fontsize=16, fontweight="bold", color=tb.TISSUE_COLORS[c])
        for a, t in zip((a1, a2, a3), ("predicted on DRR", "predicted on PXR", "PXR - DRR (mm)")):
            a.set_xticks([]); a.set_yticks([])
            if c == 0:
                a.set_title(t, fontsize=14)
    a = fig.add_subplot(gs[2, 0]); a.axis("off")
    a.text(0, 1, "\n".join(f"{names[n]}: share {row[f'{n}_share_on_drr_pct']:.0f}% (DRR) vs "
                           f"{row[f'{n}_share_on_pxr_pct']:.0f}% (PXR)" for n in tb.CLASSES)
           + "\n\nAgreement between the two inputs,\nnot accuracy (no ground truth on D1).",
           fontsize=13, va="top", transform=a.transAxes)
    cb = fig.colorbar(im_t, cax=fig.add_subplot(gs[3, 1:3]), orientation="horizontal", extend="max")
    cb.set_label("thickness (mm)", fontsize=13)
    cb = fig.colorbar(im_e, cax=fig.add_subplot(gs[3, 3]), orientation="horizontal", extend="both")
    cb.set_label("difference (mm)", fontsize=13)
    fig.savefig(path, dpi=90, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


# =========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambda-max", type=float, default=LAMBDA_MAX)
    ap.add_argument("--test", nargs="+", help="test vines of one fold")
    ap.add_argument("--folds", action="store_true", help="train the 3 usual folds, one after another")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--replot", action="store_true")
    ap.add_argument("--check-d1", action="store_true", help="figure of the inputs, no training")
    ap.add_argument("--eval-d1", action="store_true", help="agreement on the held-out D1 vines")
    ap.add_argument("--run-dir", help="with --eval-d1: another results folder (default: this lambda)")
    args = ap.parse_args()

    tb.OUT_DIR = out_dir(args.lambda_max)
    os.makedirs(tb.OUT_DIR, exist_ok=True)
    train, held = d1_split()
    with open(os.path.join(tb.OUT_DIR, "d1_split.txt"), "w") as f:
        f.write(f"seed {D1_SPLIT_SEED}\ntrain: {' '.join(train)}\nheld out: {' '.join(held)}\n")

    if args.check_d1:
        check_d1(); return
    if args.eval_d1:
        eval_d1(args.run_dir or tb.OUT_DIR); return
    if args.summarize:
        tb.summarize(); return
    if args.replot:
        tb.replot(); return
    specimens = tb.list_specimens()
    if args.test:
        run_fold(args.test, specimens, args.lambda_max)
    elif args.folds:
        for fold in FOLDS:
            run_fold(fold, specimens, args.lambda_max)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()