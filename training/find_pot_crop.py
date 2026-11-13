#!/usr/bin/env python3
"""
find_pot_crop.py  -  find where the pot starts in the D1 2026 images

For every D1 2026 registered DRR and every 2026 PXR, finds the row where the
pot begins and saves a "cut row": everything from that row down (pot rim,
pot, soil) is dropped when the image is used. Nothing is cropped on disk; the
cut rows are saved in a CSV that train_domain_adaptation.py applies.

How the pot is found: above the pot, a row of the image only crosses the
trunk, so the wood in that row is narrow. From the pot rim down, a row also
crosses the rim, the pot walls and the soil, so it spans almost the whole
width. In the lower part of the image (below SEARCH_FROM), the script takes
the highest row whose wood spans more than WIDE of the width, for at least
SUSTAIN rows in a row, and cuts MARGIN_PX above it. No wide row found = no
pot = no cut.

Checks:
  * contact sheets (pot_crop_sheet_*.png): every image with its cut line
  * pair check: a registered DRR and its PXR should get about the same row;
    a difference above PAIR_TOL_PX is flagged in the CSV and on the sheet
  * MANUAL_CUT below: type a row for any image where the detection is wrong,
    then run the script again

USAGE (from the repo root)
    python training/find_pot_crop.py
Output: results/domain_adaptation/pot_crop/
    pot_crop_rows.csv           one row per image: file, cut row, flags
    pot_crop_sheet_01.png ...   contact sheets to check
"""

import csv
import glob
import os
import sys

import numpy as np
import tifffile
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_baseline as tb                  # noqa: E402
import train_domain_adaptation as da         # noqa: E402

OUT = os.path.join(da.DA_ROOT, "pot_crop")
CSV_PATH = os.path.join(OUT, "pot_crop_rows.csv")
SEARCH_FROM = 0.35      # only look for the pot below this fraction of the height
WIDE = 0.75             # a row is "wide" if its wood spans more than this of the width
SUSTAIN = 25            # ... for at least this many rows in a row
MARGIN_PX = 40          # cut this many rows above the first wide row
PAIR_TOL_PX = 80        # flag a DRR/PXR pair whose cut rows differ more than this
PER_SHEET = 24

# file name (as in the CSV) -> cut row, to override the detection
MANUAL_CUT = {
    # "69ca568be42f5e0702f748d5.tif": 2400,
}


def foreground(img):
    """Wood, pot and soil: brighter than the air, small specks removed."""
    t = 0.5 * da.otsu(img)
    fg = ndimage.binary_opening(img > t, iterations=2)
    lab, n = ndimage.label(fg)
    if n:
        sizes = ndimage.sum(fg, lab, range(1, n + 1))
        fg = np.isin(lab, 1 + np.flatnonzero(sizes >= 500))
    return fg


def row_extent(fg):
    """Per row: (rightmost - leftmost foreground column) / width, 0 if empty."""
    H, W = fg.shape
    any_ = fg.any(1)
    left = np.argmax(fg, 1)
    right = W - 1 - np.argmax(fg[:, ::-1], 1)
    return np.where(any_, (right - left + 1) / W, 0.0)


def detect_cut(img):
    """Returns (cut_row or None, extent profile)."""
    ext = row_extent(foreground(img))
    H = len(ext)
    wide = ext > WIDE
    run = np.convolve(wide.astype(int), np.ones(SUSTAIN, int), "valid") == SUSTAIN   # run[r]: rows r..r+SUSTAIN-1 wide
    start = int(SEARCH_FROM * H)
    hits = np.flatnonzero(run[start:]) + start
    if hits.size == 0:
        return None, ext
    return max(int(hits[0]) - MARGIN_PX, 0), ext


def images():
    """(kind, vine, path, name in CSV) for every registered DRR and every 2026 PXR."""
    out = []
    for v in sorted(da.PAIRED_PXR):
        p = da.D1_DRR_PATTERN.format(v=v)
        if os.path.exists(p):
            out.append(("drr", v, p, os.path.basename(p)))
    for d in sorted(glob.glob(os.path.join(da.PXR_ROOT, "CEP_*"))):
        v = os.path.basename(d)[4:]
        for p in da.pxr_files(v):
            out.append(("pxr", v, p, os.path.basename(p)))
    return out


def load_for_detection(kind, path):
    """The full image, never cut (da.load_pxr would apply an existing CSV).
    No DRR->PXR curve for the DRRs: it is monotonic, so the pot is in the same place."""
    return tb.normalize(np.nan_to_num(tifffile.imread(path).astype(np.float32)))


def main():
    os.makedirs(OUT, exist_ok=True)
    items = images()
    print(f"{len(items)} images ({sum(k == 'drr' for k, *_ in items)} DRRs, "
          f"{sum(k == 'pxr' for k, *_ in items)} PXRs)")
    rows, thumbs = [], []
    for kind, v, path, name in items:
        img = load_for_detection(kind, path)
        cut, _ = detect_cut(img)
        method = "auto"
        if name in MANUAL_CUT:
            cut, method = int(MANUAL_CUT[name]), "manual"
        rows.append({"file": name, "kind": kind, "vine": v, "path": os.path.relpath(path, tb.REPO),
                     "height": img.shape[0], "width": img.shape[1],
                     "cut_row": "" if cut is None else cut, "method": method if cut is not None else "none",
                     "paired": kind == "drr" or (v in da.PAIRED_PXR and name == da.PAIRED_PXR[v]),
                     "flag": ""})
        thumbs.append((img[::8, ::8], cut, f"{kind.upper()} CEP_{v}\n{name[:24]}", len(rows) - 1))
        print(f"  {kind} CEP_{v:6s} {name:45s} cut {cut} ({method})")

    # pair check: registered DRR vs its PXR
    by_name = {r["file"]: r for r in rows}
    for r in rows:
        if r["kind"] != "drr":
            continue
        p = by_name.get(da.PAIRED_PXR.get(r["vine"], ""))
        if p is None:
            continue
        a, b = r["cut_row"], p["cut_row"]
        if (a == "") != (b == "") or (a != "" and abs(a - b) > PAIR_TOL_PX):
            msg = f"pair differs: DRR {a or 'none'} vs PXR {b or 'none'}"
            r["flag"] = p["flag"] = msg
            print(f"  FLAG CEP_{r['vine']}: {msg}")
    for r in rows:
        if r["cut_row"] == "" and not r["flag"]:
            r["flag"] = "no pot found"

    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    make_sheets(thumbs, rows)
    n_flag = sum(bool(r["flag"]) for r in rows)
    print(f"written: {CSV_PATH}\n{n_flag} image(s) flagged: check them on the contact sheets "
          f"in {OUT}, fix with MANUAL_CUT if needed")


def make_sheets(thumbs, rows):
    plt = tb._plt()
    for s in range(0, len(thumbs), PER_SHEET):
        chunk = thumbs[s:s + PER_SHEET]
        fig, ax = plt.subplots(4, 6, figsize=(24, 20), layout="constrained")
        fig.suptitle(f"Pot crop check, sheet {s // PER_SHEET + 1}: the part above the red line is kept "
                     f"(red frame = flagged)", fontsize=18, fontweight="bold")
        for a in ax.flat:
            a.axis("off")
        for a, (th, cut, label, i) in zip(ax.flat, chunk):
            a.imshow(th, cmap="gray", vmin=0, vmax=1)
            if cut is not None:
                a.axhline(cut / 8, color="#E53935", lw=2.5)
                a.axhspan(cut / 8, th.shape[0], color="#E53935", alpha=0.15)
            flag = rows[i]["flag"]
            a.set_title(label + (f"\n{flag}" if flag else ""), fontsize=10,
                        color="#B71C1C" if flag else "#1F2421")
            if flag:
                a.axis("on"); a.set_xticks([]); a.set_yticks([])
                for sp in a.spines.values():
                    sp.set_color("#B71C1C"); sp.set_linewidth(4)
        fig.savefig(os.path.join(OUT, f"pot_crop_sheet_{s // PER_SHEET + 1:02d}.png"), dpi=60)
        plt.close(fig)


if __name__ == "__main__":
    main()