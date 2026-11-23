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
Pot and soil (D1): the images are not cropped. The top of the soil, clicked
per image with mark_soil.py (results/domain_adaptation/soil/soil_lines.csv),
marks what lies below as ignored: no patch is centred there, the domain loss
leaves those locations out, and --eval-d1 leaves those pixels out.
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
D2_VIEWS_PER_VINE = None     # plateau check (--views-per-vine N): only N stored D2
                             # views per vine (0 = the original views only); None = all 100
USE_D1_VIEWS = True          # also train on the stored augmented D1 DRRs
                             # (make_views.py --d1, then --d1 --soil)
CLF_STAGES = (1024, 2048)    # encoder stages the classifiers look at (channels)
# --method contrastive: instead of the domain classifier, the registered DRR and its
# real PXR give positive pairs (same patch location); other pairs in the batch are
# negatives (InfoNCE on the pooled stage-4 features, through a projection head)
CONTRAST_WEIGHT = 0.3        # weight of the contrastive loss, ramped up like lambda
CONTRAST_TEMP = 0.1          # InfoNCE temperature
N_PAIRS = 8                  # DRR/PXR patch pairs per step (8 DRR + 8 PXR patches)
PAIRS_PER_VINE = 2           # pairs from the same vine per step (different places), so
                             # the model cannot match a pair just by recognising the vine
# --method mmd: the encoder directly shrinks the Maximum Mean Discrepancy between
# the pooled features of the simulated patches (D2 + D1 DRR) and the real PXR
# patches, at the same stages as the domain classifier (as in DRR4Covid)
MMD_WEIGHT = 0.3             # weight of the MMD loss, ramped up like lambda
MMD_SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)   # Gaussian kernels, x the median distance
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


SOIL_CSV = os.path.join(DA_ROOT, "soil", "soil_lines.csv")      # from mark_soil.py


def out_dir(lam, method="adversarial"):
    """lam0.3 = the first run (soil not ignored); lam0.3_soil once soil lines exist,
    so a new run never overwrites the earlier results. Contrastive runs:
    contrast0.3_soil_bal (the stored D1 views are not used: they have no PXR)."""
    tag = "_soil" if os.path.exists(SOIL_CSV) else ""
    if method == "contrastive":
        return os.path.join(DA_ROOT, f"contrast{CONTRAST_WEIGHT:g}{tag}_bal")
    if method == "mmd":
        tag += "_d1aug" if USE_D1_VIEWS else ""
        return os.path.join(DA_ROOT, f"mmd{MMD_WEIGHT:g}{tag}_bal")
    tag += "_d1aug" if USE_D1_VIEWS else ""
    tag += "_bal"                                   # class balancing, always on
    if D2_VIEWS_PER_VINE is not None:
        tag += f"_v{D2_VIEWS_PER_VINE}"             # plateau check
    return os.path.join(DA_ROOT, f"lam{lam:g}{tag}")


# =========================================================================
# D1 DATA
# =========================================================================

def d1_split():
    vines = sorted(PAIRED_PXR)
    held = sorted(str(v) for v in
                  np.random.default_rng(D1_SPLIT_SEED).choice(vines, D1_HELD_OUT_N, replace=False))
    return [v for v in vines if v not in held], held


# vines with more PXRs than the usual 3: keep only these (CEP_1181 was X-rayed
# from more angles; keep the registered one and the two taken with it)
PXR_KEEP = {
    "1181": ["69ca6a532ad9c81239913234.tif", "69ca6a542ad9c81239913235.tif",
             "69ca6a542ad9c81239913236.tif"],
}


def pxr_files(v):
    """The 2026 PXRs of a vine (not the line_integral files; PXR_KEEP if listed)."""
    fs = sorted(glob.glob(os.path.join(PXR_ROOT, f"CEP_{v}", "*.tif")))
    fs = [f for f in fs if "line_integral" not in os.path.basename(f)]
    if v in PXR_KEEP:
        fs = [f for f in fs if os.path.basename(f) in PXR_KEEP[v]]
    return fs


_SOIL = None


def soil_row(path):
    """Top of the soil (clicked with mark_soil.py), or None = no soil. The image
    is NOT cropped: from this row down (soil, pot) is only ignored: no patch is
    centred there, the domain loss leaves it out, and so does --eval-d1."""
    global _SOIL
    if _SOIL is None:
        _SOIL = {}
        if os.path.exists(SOIL_CSV):
            with open(SOIL_CSV) as f:
                for r in csv.DictReader(f):
                    if r["soil_row"] != "":
                        _SOIL[r["file"]] = int(r["soil_row"])
            print(f"  soil: {len(_SOIL)} soil lines from {SOIL_CSV}")
        else:
            print(f"  soil: {SOIL_CSV} not found, soil and pot are NOT ignored "
                  "(run training/mark_soil.py)")
    return _SOIL.get(os.path.basename(path))


def load_d1_drr(v):
    """Registered linear DRR -> the same DRR->PXR curve as D2 -> normalize."""
    raw = np.nan_to_num(tifffile.imread(D1_DRR_PATTERN.format(v=v)).astype(np.float32))
    if raw.max() > 1.5:                     # stored as 16-bit (0..65535)
        raw = raw / 65535.0
    return tb.normalize(augment.apply_curve(raw).astype(np.float32))


def load_pxr(path):
    return tb.normalize(np.nan_to_num(tifffile.imread(path).astype(np.float32)))


def above_soil(shape, soil):
    """Pixels above the soil line (all True if no soil line)."""
    ok = np.ones(shape, bool)
    if soil is not None:
        ok[soil:] = False
    return ok


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


def with_fg(img, seed, soil=None):
    """(img, vine pixels above the soil, soil row) for patch sampling."""
    fg = np.argwhere(vine_mask(img) & above_soil(img.shape, soil)).astype(np.int32)
    if len(fg) > FG_POINTS:
        fg = fg[np.random.default_rng(seed).choice(len(fg), FG_POINTS, replace=False)]
    return img, fg, soil


def d1_patches(data, n, noise=False):
    """n patches from a list of (img, fg, soil) D1 images, same sampling as D2
    (FG_PROB centred on the vine, never on the soil), random left-right flip;
    noise=True adds the same 2D noise as the D2 views (for the D1 DRRs).
    Returns the patches and, per patch, which of its rows are above the soil."""
    xs, rows_ok = [], []
    for _ in range(n):
        img, fg, soil = random.choice(data)
        x, y0 = _patch(img, fg, soil)
        if noise:
            x = x + np.random.normal(0.0, random.uniform(*augment.NOISE_STD), x.shape).astype(np.float32)
        if random.random() < 0.5:
            x = x[:, ::-1]                       # left-right: rows unchanged
        xs.append(np.ascontiguousarray(x)[None])
        r = np.arange(y0, y0 + tb.PATCH)
        rows_ok.append(r < soil if soil is not None else np.ones(tb.PATCH, bool))
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(rows_ok))


def _patch(img, fg, soil=None):
    """A patch and its top row. Random (non-centred) patches are drawn with
    their centre above the soil line too."""
    P = tb.PATCH
    img = tb.pad_to(img, P, P)
    H, W = img.shape
    if random.random() < tb.FG_PROB and len(fg):
        cy, cx = fg[random.randrange(len(fg))]
    else:
        top = H if soil is None else max(min(soil, H), 1)
        cy, cx = random.randrange(top), random.randrange(W)
    y0 = int(np.clip(cy - P // 2, 0, H - P)); x0 = int(np.clip(cx - P // 2, 0, W - P))
    x = img[y0:y0 + P, x0:x0 + P]
    if x.dtype == np.uint16:                     # stored view
        return x.astype(np.float32) / 65535.0, y0
    return x.astype(np.float32), y0


def load_d1_train(vines):
    drrs, pxrs = [], []
    for i, v in enumerate(vines):
        p = D1_DRR_PATTERN.format(v=v)
        drrs.append(with_fg(load_d1_drr(v), i, soil_row(p)))
        for j, f in enumerate(pxr_files(v)):
            pxrs.append(with_fg(load_pxr(f), 1000 * i + j, soil_row(f)))
    if USE_D1_VIEWS:
        drrs += load_d1_views(vines)
    return drrs, pxrs


def load_d1_views(vines):
    """Stored augmented D1 DRRs of these vines, memory-mapped, with the vine
    pixels and soil rows written by make_views.py --d1 --soil."""
    import resource
    import make_views
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)   # one open file per view
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(hard, 65536), hard))
    root = make_views.D1_VIEWS_DIR
    soil_csv = os.path.join(root, "soil_rows.csv")
    if not os.path.exists(soil_csv):
        raise SystemExit(f"{soil_csv} not found: run make_views.py --d1 --soil first")
    with open(soil_csv) as f:
        soil = {(r["vine"], int(r["view"])): (int(r["soil_row"]) if r["soil_row"] else None)
                for r in csv.DictReader(f)}
    views = []
    for v in vines:
        imgs = sorted(glob.glob(os.path.join(root, f"CEP_{v}", "*_img.npy")))
        if not imgs:
            raise SystemExit(f"No stored D1 views for CEP_{v} in {root}")
        for ip in imgs:
            k = int(os.path.basename(ip).split("_")[0])
            views.append((np.load(ip, mmap_mode="r"),
                          np.load(ip.replace("_img.npy", "_fg.npy")),
                          soil[(f"CEP_{v}", k)]))
    print(f"  D1 stored views: {len(views)} from {len(vines)} vines")
    return views


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


def load_d1_pairs(vines):
    """(registered DRR, its paired real PXR, vine pixels, soil row) per vine.
    The DRR was rendered with the registration and cropped to the PXR, so both
    images are aligned pixel for pixel (up to the registration error)."""
    pairs = []
    for v in vines:
        drr_path = D1_DRR_PATTERN.format(v=v)
        pxr_path = os.path.join(PXR_ROOT, f"CEP_{v}", PAIRED_PXR[v])
        drr, pxr = load_d1_drr(v), load_pxr(pxr_path)
        if drr.shape != pxr.shape:
            print(f"  CEP_{v}: DRR {drr.shape} and PXR {pxr.shape} differ, pair skipped")
            continue
        sd, sp = soil_row(drr_path), soil_row(pxr_path)
        soil = min([x for x in (sd, sp) if x is not None], default=None)
        fg = np.argwhere(vine_mask(drr) & above_soil(drr.shape, soil)).astype(np.int32)
        if len(fg) == 0:
            continue
        pairs.append((drr, pxr, fg, soil))
    return pairs


def pair_patches(pairs, n):
    """n patch pairs: n / PAIRS_PER_VINE vines, PAIRS_PER_VINE places on each
    (at least half a patch apart), the same window of the DRR (with noise) and
    of its PXR, centred on the vine, same flip. Wrong answers from the same vine
    force the model to look at the wood in the patch, not at which vine it is.
    Returns DRR patches, PXR patches and the rows above the soil (both images)."""
    P = tb.PATCH
    k = max(n // PAIRS_PER_VINE, 1)
    vines = random.sample(pairs, k) if len(pairs) >= k else [random.choice(pairs) for _ in range(k)]
    chosen = []
    for pr in vines:
        fg = pr[2]
        centres = []
        for _ in range(PAIRS_PER_VINE):
            for _try in range(20):                       # a place away from the previous ones
                c = fg[random.randrange(len(fg))]
                if all(max(abs(int(c[0]) - a), abs(int(c[1]) - b)) >= P // 2 for a, b in centres):
                    break
            centres.append((int(c[0]), int(c[1])))
            chosen.append((pr, centres[-1]))
    chosen = chosen[:n]
    xd, xp, rows_ok = [], [], []
    for (drr, pxr, fg, soil), (cy, cx) in chosen:
        H, W = drr.shape
        y0 = int(np.clip(cy - P // 2, 0, H - P)); x0 = int(np.clip(cx - P // 2, 0, W - P))
        a = drr[y0:y0 + P, x0:x0 + P].astype(np.float32)
        a = a + np.random.normal(0.0, random.uniform(*augment.NOISE_STD), a.shape).astype(np.float32)
        b = pxr[y0:y0 + P, x0:x0 + P].astype(np.float32)
        if random.random() < 0.5:
            a, b = a[:, ::-1], b[:, ::-1]
        xd.append(np.ascontiguousarray(a)[None]); xp.append(np.ascontiguousarray(b)[None])
        r = np.arange(y0, y0 + P)
        rows_ok.append(r < soil if soil is not None else np.ones(P, bool))
    return (torch.from_numpy(np.stack(xd)), torch.from_numpy(np.stack(xp)),
            torch.from_numpy(np.stack(rows_ok)))


def masked_pool(feat, rows_ok):
    """Average a (B, C, h, w) feature map over the rows above the soil -> (B, C)."""
    f = feat.float()
    h = f.shape[2]
    m = rows_ok[:, ::tb.PATCH // h][:, :h].float().view(len(f), 1, h, 1)
    return (f * m).sum((2, 3)) / (m.sum((2, 3)) * f.shape[3]).clamp(min=1.0)


def mmd_loss(z, n_sim):
    """Squared Maximum Mean Discrepancy between z[:n_sim] (simulated) and
    z[n_sim:] (real), sum of Gaussian kernels whose widths are MMD_SCALES x the
    median pairwise distance (biased estimate)."""
    d = torch.cdist(z, z).pow(2)
    med = d[d > 0].median().detach() if (d > 0).any() else d.new_tensor(1.0)
    k = sum(torch.exp(-d / (s * med)) for s in MMD_SCALES)
    xx, yy, xy = k[:n_sim, :n_sim], k[n_sim:, n_sim:], k[:n_sim, n_sim:]
    return xx.mean() + yy.mean() - 2 * xy.mean()


class ProjectionHead(nn.Module):
    """Pooled 2048-channel stage-4 features -> 128-d unit vector (used only for the loss)."""
    def __init__(self, ch=2048):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(ch, 512), nn.ReLU(inplace=True), nn.Linear(512, 128))

    def forward(self, z):
        return F.normalize(self.net(z), dim=1)


def contrastive_loss(head, feat, start, n, rows_ok):
    """InfoNCE between the DRR patches feat[start:start+n] and the PXR patches
    feat[start+n:start+2n]: patch i of the DRR must match patch i of the PXR.
    Features are averaged over the rows above the soil. Returns the loss and
    the share of pairs matched correctly (chance = 1/n)."""
    f = feat[start:start + 2 * n].float()
    h = f.shape[2]
    m = rows_ok[:, ::tb.PATCH // h][:, :h].float()                       # (n, h)
    m = torch.cat([m, m]).view(2 * n, 1, h, 1)
    z = (f * m).sum((2, 3)) / (m.sum((2, 3)) * f.shape[3]).clamp(min=1.0)
    z = head(z)
    zd, zp = z[:n], z[n:]
    logits = zd @ zp.t() / CONTRAST_TEMP
    target = torch.arange(n, device=logits.device)
    loss = 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))
    acc = (logits.argmax(1) == target).float().mean().item()
    return loss, acc


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


def domain_loss(logits, n_sim, rows_ok=None):
    """BCE per location; simulated (first n_sim, target 0) and real (target 1)
    averaged separately, then together. Also returns the balanced accuracy.
    rows_ok (B, PATCH): rows above the soil; a location counts only if all the
    patch rows it covers are above the soil."""
    B, _, h, w = logits.shape
    if rows_ok is None:
        m = torch.ones_like(logits)
    else:
        cell = rows_ok.view(B, h, -1).all(-1)                       # (B, h)
        m = cell[:, None, :, None].expand(B, 1, h, w).float()
    t = torch.zeros_like(logits); t[n_sim:] = 1
    l = F.binary_cross_entropy_with_logits(logits.float(), t, reduction="none")
    mean = lambda x, mm: (x * mm).sum() / mm.sum().clamp(min=1)
    loss = 0.5 * (mean(l[:n_sim], m[:n_sim]) + mean(l[n_sim:], m[n_sim:]))
    pred = (logits.detach() > 0).float()
    acc = 0.5 * (mean(1 - pred[:n_sim], m[:n_sim]) + mean(pred[n_sim:], m[n_sim:]))
    return loss, acc.item()


# =========================================================================
# TRAINING
# =========================================================================

def run_fold(test_specimens, all_specimens, lam_max, method="adversarial"):
    random.seed(tb.SEED); np.random.seed(tb.SEED); torch.manual_seed(tb.SEED)
    train_specimens = [s for s in all_specimens if s not in test_specimens]
    fold_dir = os.path.join(tb.OUT_DIR, "fold_" + "_".join(test_specimens))
    os.makedirs(fold_dir, exist_ok=True)
    print(f"\n=== test: {test_specimens}   train: {train_specimens}   method {method}, "
          + {"adversarial": f"lambda_max {lam_max}", "contrastive": f"weight {CONTRAST_WEIGHT}",
             "mmd": f"weight {MMD_WEIGHT}"}[method])

    # D2: stored augmented views + flip + class balancing (always on), as
    # train_baseline.py --stored --balance
    originals = [tb.load_pair(d, l) for _, d, l in tb.list_pairs(train_specimens)]
    w, mean = tb.class_weights([lab for _, lab in originals])
    weights = torch.tensor(w, dtype=torch.float32, device=tb.DEVICE).view(1, -1, 1, 1)
    tb._WR_PROB = tb.BALANCE_WR_PROB
    d2_data = [(img, lab, None, tb.white_rot_points(lab, i)) for i, (img, lab) in enumerate(originals)]
    if D2_VIEWS_PER_VINE != 0:
        d2_data += tb.load_stored_views(train_specimens, with_wr=True, max_per_vine=D2_VIEWS_PER_VINE)
    d2_batches = tb.batch_iter(d2_data, flip_lr=tb.FLIP_LR)
    info = {"mean_thickness_mm": dict(zip(tb.CLASSES, mean.round(3).tolist())),
            "loss_weight": dict(zip(tb.CLASSES, w.round(3).tolist())),
            "power": tb.BALANCE_POWER, "white_rot_patch_share": tb.BALANCE_WR_PROB}
    with open(os.path.join(fold_dir, "balance.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"  D2: {len(d2_data)} views, flip {'on' if tb.FLIP_LR else 'off'}, class balancing: "
          f"loss weights {info['loss_weight']}, {tb.BALANCE_WR_PROB:.0%} of patches on white rot")

    # D1
    d1_train, d1_held = d1_split()
    if method == "contrastive":
        pairs = load_d1_pairs(d1_train)
        print(f"  D1: {len(pairs)} registered DRR/PXR pairs from {len(d1_train)} vines; "
              f"held out: {d1_held}")
    else:
        d1_drr, d1_pxr = load_d1_train(d1_train)
        print(f"  D1: {len(d1_drr)} DRRs + {len(d1_pxr)} PXRs from {len(d1_train)} vines; "
              f"held out: {d1_held}")

    model = tb.build_model().to(tb.DEVICE)
    check_split_forward(model)
    if method == "contrastive":
        clfs = nn.ModuleDict({"head": ProjectionHead()}).to(tb.DEVICE)
    elif method == "mmd":
        clfs = nn.ModuleDict()                      # no extra network
    else:
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
            x2, y2 = next(d2_batches)
            if method == "contrastive":
                lam = lambda_at(step, total_steps, CONTRAST_WEIGHT)
                xs, xr, ok_pair = pair_patches(pairs, N_PAIRS)
                rows_ok = ok_pair.to(tb.DEVICE)
            else:
                lam = lambda_at(step, total_steps, MMD_WEIGHT if method == "mmd" else lam_max)
                xs, ok_s = d1_patches(d1_drr, N_D1_DRR, noise=True)
                xr, ok_r = d1_patches(d1_pxr, N_D1_PXR)
                rows_ok = torch.cat([torch.ones(len(x2), tb.PATCH, dtype=torch.bool), ok_s, ok_r]).to(tb.DEVICE)
            x = torch.cat([x2, xs, xr]).to(tb.DEVICE)
            y2 = y2.to(tb.DEVICE)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                pred, stage = split_forward(model, x, n_lab)
                err = (pred - y2).abs()
                l1 = err.mean()                          # plain L1, logged (comparable across runs)
                l1_w = (err * weights).mean()            # class-balanced L1, trained on
                dom = 0.0
                if method == "contrastive":
                    con, acc = contrastive_loss(clfs["head"], stage[2048], n_lab, N_PAIRS, rows_ok)
                    dom = lam * con
                    s_acc[2048] += acc
                elif method == "mmd":
                    for c in CLF_STAGES:                # logged per stage in the acc columns
                        m = mmd_loss(masked_pool(stage[c], rows_ok), n_sim)
                        dom = dom + lam * m
                        s_acc[c] += m.item()
                else:
                    for c in CLF_STAGES:
                        d, acc = domain_loss(clfs[str(c)](stage[c], lam), n_sim, rows_ok)
                        dom = dom + d
                        s_acc[c] += acc
                loss = l1_w + dom
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
    save_domain_plot(log, fold_dir, method)

    summary, _, _ = tb.evaluate_fold(model, test_specimens, train_specimens, fold_dir)
    print("  D2 test results (mean over test images):")
    for name, v in summary["mean"].items():
        print(f"    {name:10s} " + "  ".join(
            f"{k}={v[k]:.3f}" if v[k] is not None else f"{k}=n/a" for k in v))
    return summary


def save_domain_plot(log, fold_dir, method="adversarial"):
    plt = tb._plt()
    if method == "mmd":
        ep = [r["epoch"] for r in log]
        fig, ax = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
        ax[0].plot(ep, [r["domain_loss"] for r in log], color="#2F4558", lw=2)
        ax[0].set_title("MMD loss (weighted, stage 3 + stage 4)", fontsize=14); ax[0].set_xlabel("epoch")
        for c, col in zip(CLF_STAGES, ("#7D8FA3", "#2F4558")):
            ax[1].plot(ep, [r[f"domain_acc_{STAGE_NAME[c]}"] for r in log], color=col, lw=2,
                       label=STAGE_NAME[c].replace("stage", "stage "))
        ax[1].legend()
        ax[1].set_title("MMD between simulated and real (lower = closer)", fontsize=14)
        ax[1].set_xlabel("epoch")
        fig.savefig(os.path.join(fold_dir, "domain_training.png"), dpi=100)
        plt.close(fig)
        return
    if method == "contrastive":
        ep = [r["epoch"] for r in log]
        fig, ax = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
        ax[0].plot(ep, [r["domain_loss"] for r in log], color="#2F4558", lw=2)
        ax[0].set_title("Contrastive loss (weighted)", fontsize=14); ax[0].set_xlabel("epoch")
        ax[1].plot(ep, [r["domain_acc_stage4"] for r in log], color="#2F4558", lw=2)
        ax[1].axhline(1 / N_PAIRS, color="#B71C1C", ls="--", lw=1, label=f"chance (1/{N_PAIRS})")
        ax[1].set_ylim(0, 1.0); ax[1].legend()
        ax[1].set_title("DRR/PXR pairs matched correctly", fontsize=14); ax[1].set_xlabel("epoch")
        fig.savefig(os.path.join(fold_dir, "domain_training.png"), dpi=100)
        plt.close(fig)
        return
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
        soil = None if k == 0 else soil_row(D1_DRR_PATTERN.format(v=v) if k == 1 else
                                            os.path.join(PXR_ROOT, f"CEP_{v}", PAIRED_PXR[v]))
        if soil is not None:
            ax[0, k].axhline(soil, color="#E53935", lw=1.5)
            ax[0, k].axhspan(soil, im.shape[0], color="#E53935", alpha=0.15)
        m = vine_mask(im) & above_soil(im.shape, soil)
        ax[1, k].hist(im[m], bins=100, range=(0, 1), color="#2F4558")
        ax[1, k].set_title("histogram inside the vine, above the soil (red)", fontsize=12)
        ax[1, k].set_xlabel("normalized intensity")
    fig.suptitle("Check before domain adaptation: same polarity (wood bright?), similar range, "
                 "vine outline correct?", fontsize=15, fontweight="bold")
    os.makedirs(DA_ROOT, exist_ok=True)
    path = os.path.join(DA_ROOT, "check_inputs.png")
    fig.savefig(path, dpi=90); plt.close(fig)
    print(f"D1 train vines: {train}\nD1 held-out vines: {held}")
    print("PXRs per training vine: " + ", ".join(f"{v}:{len(pxr_files(v))}" for v in train))
    print(f"written: {path}")


DETECTOR_PIX_MM = 0.139                       # D1 and D2 images: full-detector pixel size
MAGNIFICATION = augment.SID_MM / augment.SPD_MM  # ~1.54 at the vine centre (approximate)


def total_vs_width(total, mask, min_px=20):
    """Sanity check of the predicted amounts: in each image row, the longest run
    of vine pixels is taken as the trunk width (mm, corrected for magnification)
    and compared with the largest total thickness (3 tissues) in that run. A
    round trunk gives about 1 at its centre. Returns the median over the rows."""
    ratios = []
    for r in np.nonzero(mask.any(1))[0]:
        m = np.concatenate(([0], mask[r].astype(np.int8), [0]))
        edges = np.flatnonzero(np.diff(m))
        starts, ends = edges[::2], edges[1::2]
        k = int(np.argmax(ends - starts))
        a, b = starts[k], ends[k]
        if b - a < min_px:
            continue
        width_mm = (b - a) * DETECTOR_PIX_MM / MAGNIFICATION
        ratios.append(float(total[r, a:b].max()) / width_mm)
    return float(np.median(ratios)) if ratios else float("nan")


def d2_width_reference():
    """The same ratio on the TRUE D2 thickness maps: what a correct prediction
    gives, since trunks are not perfect cylinders (and bark is not labelled)."""
    vals = []
    for s, d, l in tb.list_pairs(tb.list_specimens()):
        _, lab = tb.load_pair(d, l)
        tot = lab.sum(0)
        vals.append(total_vs_width(tot, tot > 0.5))
    return float(np.median(vals))


@torch.no_grad()
def angle_maps(v, run_dir, yaws=(0, 30, 60, 90)):
    """One figure: the stored views of a vine closest to these yaws (preferring
    small pitch and roll), each with its predicted thickness maps, so the same
    tissue can be followed from angle to angle."""
    import make_views
    models = sorted(glob.glob(os.path.join(run_dir, "fold_*", "model.pt")))
    if not models:
        raise SystemExit(f"No fold_*/model.pt in {run_dir}")
    models = [tb.load_model(mp).eval() for mp in models]
    root = make_views.D1_VIEWS_DIR
    with open(os.path.join(root, "soil_rows.csv")) as f:
        soil = {int(r["view"]): (int(r["soil_row"]) if r["soil_row"] else None)
                for r in csv.DictReader(f) if r["vine"] == f"CEP_{v}"}
    views = []
    for ip in sorted(glob.glob(os.path.join(root, f"CEP_{v}", "*_img.npy"))):
        k = int(os.path.basename(ip).split("_")[0])
        views.append((k, ip, augment.sample_params(make_views.D1.seed(v, k))))
    if not views:
        raise SystemExit(f"No stored views for CEP_{v} in {root}")

    def pick(target):
        def cost(x):
            p = x[2]
            d = abs((p["yaw"] - target + 180) % 360 - 180)
            return d + 2 * (abs(p["pitch"]) + abs(p["roll"]))
        return min(views, key=cost)

    chosen = [pick(t) for t in yaws]
    names = {"healthy": "Healthy", "necrosis": "Necrosis", "white_rot": "White rot"}
    plt = tb._plt()
    fig = plt.figure(figsize=(5 * len(chosen), 21), layout="constrained")
    gs = fig.add_gridspec(5, len(chosen), height_ratios=[1, 1, 1, 1, 0.06])
    fig.suptitle(f"CEP_{v} (held out): the same vine from {len(chosen)} angles",
                 fontsize=20, fontweight="bold")
    for j, (k, ip, p) in enumerate(chosen):
        img = np.load(ip).astype(np.float32) / 65535.0
        mask = vine_mask(img) & above_soil(img.shape, soil.get(k))
        pred = np.mean([tb.predict_full(m, img) for m in models], 0)
        ry, rx = tb.vine_crop(mask[None].astype(np.float32))
        tot = max(sum(float(pred[c][mask].sum()) for c in range(3)), 1e-6)
        a = fig.add_subplot(gs[0, j])
        a.imshow(img[ry, rx], cmap="gray", vmin=0, vmax=1)
        a.set_title(f"yaw {p['yaw']:.0f}°  (pitch {p['pitch']:+.0f}°, roll {p['roll']:+.0f}°)\n"
                    f"view {k}", fontsize=14)
        a.set_xticks([]); a.set_yticks([])
        for c, n in enumerate(tb.CLASSES):
            a = fig.add_subplot(gs[1 + c, j])
            im = a.imshow(np.where(mask, pred[c], 0)[ry, rx], cmap="magma",
                          vmin=0, vmax=D1_THICK_VMAX_MM)
            share = 100 * float(pred[c][mask].sum()) / tot
            a.text(0.03, 0.97, f"{share:.0f}% of the wood", transform=a.transAxes, fontsize=13,
                   va="top", color="white")
            a.set_xticks([]); a.set_yticks([])
            if j == 0:
                a.set_ylabel(names[n], fontsize=18, fontweight="bold", color=tb.TISSUE_COLORS[c])
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[4, :]), orientation="horizontal", extend="max")
    cb.set_label("predicted thickness (mm)", fontsize=14)
    out = os.path.join(DA_ROOT, f"angles_CEP_{v}_" + os.path.basename(os.path.normpath(run_dir)))
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "angle_maps.png")
    fig.savefig(path, dpi=80, bbox_inches="tight")
    plt.close(fig)
    print(f"views {[k for k, _, _ in chosen]} -> {path}")


@torch.no_grad()
def eval_d1(run_dir):
    """For each held-out D1 vine: predict on the registered DRR and on its PXR
    with every fold model of run_dir, compare the two (agreement, NOT accuracy:
    there is no ground truth on D1)."""
    _, held = d1_split()
    models = sorted(glob.glob(os.path.join(run_dir, "fold_*", "model.pt")))
    if not models:
        raise SystemExit(f"No fold_*/model.pt in {run_dir}")
    ref = d2_width_reference()
    print(f"  total thickness / trunk width on the TRUE D2 maps (reference): {ref:.2f}")
    out = os.path.join(DA_ROOT, "d1_agreement_" + os.path.basename(os.path.normpath(run_dir)))
    os.makedirs(out, exist_ok=True)
    rows = []
    for v in held:
        drr = load_d1_drr(v)
        pxr = load_pxr(os.path.join(PXR_ROOT, f"CEP_{v}", PAIRED_PXR[v]))
        pxr_path = os.path.join(PXR_ROOT, f"CEP_{v}", PAIRED_PXR[v])
        mask = (vine_mask(drr) & above_soil(drr.shape, soil_row(D1_DRR_PATTERN.format(v=v)))
                & above_soil(drr.shape, soil_row(pxr_path)))          # soil of either image left out
        p_d, p_p = [], []
        for mp in models:
            m = tb.load_model(mp); m.eval()
            p_d.append(tb.predict_full(m, drr)); p_p.append(tb.predict_full(m, pxr))
        p_d, p_p = np.mean(p_d, 0), np.mean(p_p, 0)          # mean of the fold models
        row = {"vine": v}
        row["total_vs_width_drr"] = total_vs_width(p_d.sum(0), mask)
        row["total_vs_width_pxr"] = total_vs_width(p_p.sum(0), mask)
        tot_d, tot_p = p_d[:, mask].sum(), p_p[:, mask].sum()
        for c, name in enumerate(tb.CLASSES):
            row[f"{name}_mean_abs_diff_mm"] = float(np.abs(p_d[c] - p_p[c])[mask].mean())
            row[f"{name}_share_on_drr_pct"] = float(100 * p_d[c][mask].sum() / max(tot_d, 1e-6))
            row[f"{name}_share_on_pxr_pct"] = float(100 * p_p[c][mask].sum() / max(tot_p, 1e-6))
        rows.append(row)
        save_d1_figure(v, drr, pxr, p_d, p_p, mask, row, os.path.join(out, f"CEP_{v}.png"), ref)
        print(f"  CEP_{v}: " + "  ".join(
            f"{n} |diff| {row[f'{n}_mean_abs_diff_mm']:.1f} mm, share DRR/PXR "
            f"{row[f'{n}_share_on_drr_pct']:.0f}/{row[f'{n}_share_on_pxr_pct']:.0f}%"
            for n in tb.CLASSES)
            + f"  total/width DRR {row['total_vs_width_drr']:.2f} PXR {row['total_vs_width_pxr']:.2f}")
    mean = {"vine": "MEAN"}
    for k in rows[0]:
        if k != "vine":
            mean[k] = float(np.mean([r[k] for r in rows]))
    rows.append(mean)
    rows.append({k: (ref if k.startswith("total_vs_width") else "") for k in rows[0]} | {"vine": "D2_TRUE_REFERENCE"})
    with open(os.path.join(out, "d1_agreement.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
        for r in rows:
            w.writerow({k: (f"{x:.3f}" if isinstance(x, float) else x) for k, x in r.items()})
    print(f"{len(models)} fold models, mean prediction. Written to {out}/")


D1_THICK_VMAX_MM = 60         # colour scale of the D1 thickness maps (D1 values are lower than D2)


def save_d1_figure(v, drr, pxr, p_d, p_p, mask, row, path, ref=None):
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
        im_t = a1.imshow(p_d[c][ry, rx], cmap="magma", vmin=0, vmax=D1_THICK_VMAX_MM)
        a2.imshow(p_p[c][ry, rx], cmap="magma", vmin=0, vmax=D1_THICK_VMAX_MM)
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
    txt = "\n".join(f"{names[n]}: share {row[f'{n}_share_on_drr_pct']:.0f}% (DRR) vs "
                    f"{row[f'{n}_share_on_pxr_pct']:.0f}% (PXR)" for n in tb.CLASSES)
    txt += (f"\n\nTotal thickness / trunk width:\n  DRR {row['total_vs_width_drr']:.2f},  "
            f"PXR {row['total_vs_width_pxr']:.2f}")
    if ref is not None:
        txt += f"\n  true D2 maps: {ref:.2f} (reference)"
    txt += "\n\nAgreement between the two inputs,\nnot accuracy (no ground truth on D1)."
    a.text(0, 1, txt, fontsize=13, va="top", transform=a.transAxes)
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
    ap.add_argument("--views-per-vine", type=int, default=None,
                    help="plateau check: only N stored D2 views per vine (0 = originals only)")
    ap.add_argument("--method", choices=["adversarial", "contrastive", "mmd"], default="adversarial",
                    help="domain adaptation: domain classifier (default), DRR/PXR contrastive "
                         "pairs, or MMD between simulated and real features")
    ap.add_argument("--test", nargs="+", help="test vines of one fold")
    ap.add_argument("--folds", action="store_true", help="train the 3 usual folds, one after another")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--replot", action="store_true")
    ap.add_argument("--check-d1", action="store_true", help="figure of the inputs, no training")
    ap.add_argument("--eval-d1", action="store_true", help="agreement on the held-out D1 vines")
    ap.add_argument("--run-dir", help="with --eval-d1: another results folder (default: this lambda)")
    ap.add_argument("--angle-maps", metavar="VINE",
                    help="predicted maps of one D1 vine at yaw 0, 30, 60 and 90 degrees")
    args = ap.parse_args()

    global D2_VIEWS_PER_VINE
    D2_VIEWS_PER_VINE = args.views_per_vine
    tb.OUT_DIR = out_dir(args.lambda_max, args.method)
    os.makedirs(tb.OUT_DIR, exist_ok=True)
    train, held = d1_split()
    with open(os.path.join(tb.OUT_DIR, "d1_split.txt"), "w") as f:
        f.write(f"seed {D1_SPLIT_SEED}\ntrain: {' '.join(train)}\nheld out: {' '.join(held)}\n")

    if args.check_d1:
        check_d1(); return
    if args.eval_d1:
        eval_d1(args.run_dir or tb.OUT_DIR); return
    if args.angle_maps:
        angle_maps(args.angle_maps, args.run_dir or tb.OUT_DIR); return
    if args.summarize:
        tb.summarize(); return
    if args.replot:
        tb.replot(); return
    specimens = tb.list_specimens()
    if args.test:
        run_fold(args.test, specimens, args.lambda_max, args.method)
    elif args.folds:
        for fold in FOLDS:
            done = os.path.join(tb.OUT_DIR, "fold_" + "_".join(fold), "metrics.json")
            if os.path.exists(done):            # finished in an earlier (crashed) run
                print(f"\n=== fold {fold}: already done, skipped")
                continue
            run_fold(fold, specimens, args.lambda_max, args.method)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()