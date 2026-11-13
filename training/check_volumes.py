#!/usr/bin/env python3
"""
check_volumes.py  -  are the "true" volumes from the projections right?

All training results give tissue volumes as: summed projected thickness (mm)
x pixel area at the vine (mm2). That is exact for parallel rays but only
approximate for the cone beam. This compares it, for every D2 vine and tissue,
with the real 3D volume from the CT masks: voxel count x voxel volume.

CPU only, no model. Reads:
    dataset/ct_2019_cropped/<vine>_{SAIN,NECROSE,AMADOU}.tif     3D masks (0 / 255)
    results/d2_cropped_labels/<vine>_<tissue>_mm.tif             projected thickness (mm)

USAGE (from the repo root)
    python training/check_volumes.py
Output: results/volume_check/
    volumes_3d_vs_projected.csv     per vine and tissue: 3D cm3, projected cm3, ratio
    volumes_3d_vs_projected.png     projected vs 3D (one point per vine), and the ratio
"""

import csv
import os
import sys

import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_baseline as tb          # noqa: E402

CT_DIR = os.path.join(tb.REPO, "dataset", "ct_2019_cropped")
MASKS = ["SAIN", "NECROSE", "AMADOU"]                     # same order as tb.CLASSES
SPACING_FALLBACK_MM = 0.7224                              # isotropic, used if the file has none
OUT = os.path.join(tb.REPO, "results", "volume_check")


def voxel_size_mm(path):
    """(z, y, x) voxel size from the ImageJ calibration of the tif."""
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        ij = tif.imagej_metadata or {}
        xy = None
        if "XResolution" in page.tags and "YResolution" in page.tags:
            xr, yr = page.tags["XResolution"].value, page.tags["YResolution"].value
            if xr[0] and yr[0]:
                xy = (yr[1] / yr[0], xr[1] / xr[0])           # pixels per unit -> unit per pixel
        z = ij.get("spacing")
    if xy is None or z is None:
        return (SPACING_FALLBACK_MM,) * 3, "fallback"
    return (float(z), float(xy[0]), float(xy[1])), "file"


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for s in tb.list_specimens():
        for c, (name, m) in enumerate(zip(tb.CLASSES, MASKS)):
            mpath = os.path.join(CT_DIR, f"{s}_{m}.tif")
            lpath = os.path.join(tb.LABEL_ROOT, s + tb.LABEL_SUFFIXES[c])
            if not (os.path.exists(mpath) and os.path.exists(lpath)):
                print(f"  skip {s} {name}: missing {mpath if not os.path.exists(mpath) else lpath}")
                continue
            vox, src = voxel_size_mm(mpath)
            n = int(np.count_nonzero(tifffile.imread(mpath)))
            v3d = n * vox[0] * vox[1] * vox[2] / 1000.0
            proj = np.nan_to_num(tifffile.imread(lpath).astype(np.float64))
            vproj = float(proj.sum()) * tb.PIXEL_AREA_AT_VINE_MM2 / 1000.0
            rows.append({"vine": s, "group": tb.group_of(s), "tissue": name,
                         "volume_3d_cm3": v3d, "volume_projected_cm3": vproj,
                         "ratio_projected_over_3d": vproj / v3d if v3d > 0 else np.nan,
                         "voxel_mm": "x".join(f"{x:.4f}" for x in vox), "voxel_size_from": src})
            print(f"  {s:12s} {name:9s} 3D {v3d:8.1f} cm3   projected {vproj:8.1f} cm3   "
                  f"ratio {rows[-1]['ratio_projected_over_3d']:.3f}   (voxel {rows[-1]['voxel_mm']} mm, {src})")
    if not rows:
        raise SystemExit("Nothing compared: check CT_DIR and LABEL_ROOT.")

    path = os.path.join(OUT, "volumes_3d_vs_projected.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})

    print("\nratio projected / 3D, over the vines:")
    for name in tb.CLASSES:
        r = np.array([x["ratio_projected_over_3d"] for x in rows if x["tissue"] == name])
        print(f"  {name:9s} mean {np.nanmean(r):.3f}   min {np.nanmin(r):.3f}   max {np.nanmax(r):.3f}")
    plot(rows, os.path.join(OUT, "volumes_3d_vs_projected.png"))
    print(f"written: {OUT}/")


def plot(rows, path):
    plt = tb._plt()
    names = {"healthy": "Healthy", "necrosis": "Necrosis", "white_rot": "White rot"}
    fig, ax = plt.subplots(1, 2, figsize=(17, 7.5), layout="constrained")
    fig.suptitle("Tissue volume: projected (used in all results) vs real 3D volume from the CT masks",
                 fontsize=16, fontweight="bold")
    a = ax[0]
    hi = max(max(r["volume_3d_cm3"], r["volume_projected_cm3"]) for r in rows) * 1.05
    a.plot([0, hi], [0, hi], color="#999999", ls="--", lw=1, label="equal")
    for c, name in enumerate(tb.CLASSES):
        sel = [r for r in rows if r["tissue"] == name]
        a.scatter([r["volume_3d_cm3"] for r in sel], [r["volume_projected_cm3"] for r in sel],
                  s=80, color=tb.TISSUE_COLORS[c], label=names[name], zorder=3)
    a.set_xlim(0, hi); a.set_ylim(0, hi); a.set_aspect("equal")
    a.set_xlabel("3D volume from the CT masks (cm³)", fontsize=13)
    a.set_ylabel("projected volume (cm³)", fontsize=13)
    a.set_title("one point per vine and tissue", fontsize=13); a.grid(alpha=.3); a.legend(fontsize=12)
    a = ax[1]
    vines = list(dict.fromkeys(r["vine"] for r in rows)); x = np.arange(len(vines)); w = 0.27
    for c, name in enumerate(tb.CLASSES):
        vals = [next((r["ratio_projected_over_3d"] for r in rows if r["vine"] == v and r["tissue"] == name), np.nan)
                for v in vines]
        a.bar(x + (c - 1) * w, vals, w, color=tb.TISSUE_COLORS[c], label=names[name])
    a.axhline(1.0, color="#333333", ls="--", lw=1)
    a.set_xticks(x); a.set_xticklabels(vines, rotation=35, ha="right", fontsize=11)
    a.set_ylabel("projected / 3D  (1 = equal)", fontsize=13)
    a.set_title("ratio per vine", fontsize=13); a.grid(axis="y", alpha=.3)
    fig.savefig(path, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    main()