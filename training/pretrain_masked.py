#!/usr/bin/env python3
"""
pretrain_masked.py  -  masked image pretraining of the ResNet-50 encoder

Random squares of unlabelled X-ray patches are hidden (set to 0) and the
U-Net learns to fill them in; the loss only counts the hidden pixels. To do
that the encoder must learn what vine wood looks like in X-rays, simulated
and real. The encoder is then saved and used as the starting point of the
normal training instead of ImageNet:
    python training/train_domain_adaptation.py --init-encoder results/pretrain/mim_d1/encoder.pt ...

Data (no labels used, no test vine seen): the 15 D1 TRAINING vines only, their
stored augmented DRR views + registered DRRs (simulated, with noise) and their
real PXRs, half and half in every batch. The 5 held-out D1 vines, all D2 vines
and the 13 extra vines are left out. Rows below the soil line are not counted.

The encoder is the same ResNet-50 as in training, starting from ImageNet, so
the only difference with the default model is this pretraining.

USAGE (from the repo root)
    python training/pretrain_masked.py            # writes results/pretrain/mim_d1/
"""

import csv
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_baseline as tb                      # noqa: E402
import train_domain_adaptation as da             # noqa: E402

OUT = os.path.join(tb.REPO, "results", "pretrain", "mim_d1")
EPOCHS = 100
STEPS_PER_EPOCH = 200
N_SIM, N_REAL = 8, 8          # patches per step: simulated DRRs, real PXRs
CELL = 32                     # masked squares are CELL x CELL pixels
MASK_RATIO = 0.6              # share of squares hidden
LR = 1e-4


def random_mask(n, size, ratio, device):
    """(n, 1, size, size) with 1 on hidden squares."""
    g = size // CELL
    m = (torch.rand(n, 1, g, g, device=device) < ratio).float()
    return F.interpolate(m, scale_factor=CELL, mode="nearest")


def save_preview(x, mask, rec, path):
    plt = tb._plt()
    k = min(4, len(x))
    fig, ax = plt.subplots(k, 3, figsize=(10, 3.3 * k), layout="constrained", squeeze=False)
    for i in range(k):
        for j, (img, t) in enumerate(((x[i] * (1 - mask[i]), "input (hidden squares)"),
                                      (rec[i], "filled in"), (x[i], "original"))):
            ax[i, j].imshow(img[0], cmap="gray", vmin=0, vmax=1)
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            if i == 0:
                ax[i, j].set_title(t, fontsize=13)
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    train, held = da.d1_split()
    print(f"masked pretraining on the {len(train)} D1 training vines; left out: held-out D1 {held}, "
          f"all D2 vines, the extra vines")
    sim, real = da.load_d1_train(train)
    print(f"  {len(sim)} simulated DRRs (views + registered), {len(real)} real PXRs")

    model = smp.Unet(encoder_name="resnet50", encoder_weights="imagenet",
                     in_channels=1, classes=1).to(tb.DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    total = EPOCHS * STEPS_PER_EPOCH
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / total)))
    scaler = torch.amp.GradScaler("cuda", enabled=tb.DEVICE == "cuda")
    log = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        s_loss = 0.0
        for _ in range(STEPS_PER_EPOCH):
            xs, ok_s = da.d1_patches(sim, N_SIM, noise=True)
            xr, ok_r = da.d1_patches(real, N_REAL)
            x = torch.cat([xs, xr]).to(tb.DEVICE)
            ok = torch.cat([ok_s, ok_r]).to(tb.DEVICE).float()          # (n, P) rows above soil
            mask = random_mask(len(x), tb.PATCH, MASK_RATIO, tb.DEVICE)
            weight = mask * ok[:, None, :, None]                          # hidden AND above soil
            with torch.autocast("cuda", dtype=torch.float16, enabled=tb.DEVICE == "cuda"):
                rec = model(x * (1 - mask))
            err = (rec.float() - x).abs() * weight
            loss = err.sum() / weight.sum().clamp(min=1.0)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            s_loss += loss.item()
        log.append((epoch, s_loss / STEPS_PER_EPOCH))
        print(f"  epoch {epoch:3d}/{EPOCHS}  masked L1 {log[-1][1]:.4f}", flush=True)
        torch.save(model.encoder.state_dict(), os.path.join(OUT, "encoder.pt"))
        if epoch == 1 or epoch % 25 == 0:
            model.eval()
            with torch.no_grad():
                r = model(x * (1 - mask)).float().clamp(0, 1)
            save_preview(x.cpu().numpy(), mask.cpu().numpy(), r.cpu().numpy(),
                         os.path.join(OUT, f"preview_epoch{epoch:03d}.png"))

    with open(os.path.join(OUT, "loss.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["epoch", "masked_l1"]); w.writerows(log)
    with open(os.path.join(OUT, "info.txt"), "w") as f:
        f.write(f"D1 training vines: {' '.join(train)}\nleft out: {' '.join(held)}, all D2, extra vines\n"
                f"cell {CELL}, mask ratio {MASK_RATIO}, {EPOCHS} x {STEPS_PER_EPOCH} steps, "
                f"{N_SIM} sim + {N_REAL} real per step\n")
    print(f"encoder saved: {os.path.join(OUT, 'encoder.pt')}")


if __name__ == "__main__":
    main()