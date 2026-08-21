"""
Direct port of Cedric's masquer_tronc() v10 (from
etape_07_GENERATION_DE_STACKS_PROPRES..._v10.txt), applied per-slice to
a CT volume, with ONE addition: auto-detection of where the pot begins,
after which the border/table filters are turned OFF (they were eating
real trunk once the pot's much wider cross-section triggered the
border-touch and table-position rules).

Same base logic as before:
  1. threshold at SEUIL_ETAPE1
  2. drop components smaller than SEUIL_TAILLE_MIN
  3. drop components touching the border (MARGE_BORD)          <- pot-region only
  4. drop components that are low + elongated (TABLE)           <- pot-region only
  5. thin curved (low extent) components set aside as "arc candidates"
  6. everything else: keep only if thick enough, with fill-holes rescue
  7. arc candidates rescued at the end if not low and thick enough

POT DETECTION: per-slice foreground area (raw threshold, before any
filtering) is tracked. A baseline is taken from the first
POT_BASELINE_SLICES valid slices (near the apex, trunk-only, small
area). The pot start is the first slice where area exceeds
POT_AREA_FACTOR x baseline and STAYS above it for POT_SUSTAIN_SLICES
consecutive slices (avoids false-triggering on a single noisy slice).
From that slice to the end of the volume, masquer_tronc() skips steps
3 and 4 (border-touch and TABLE removal) entirely -- everything else
(size, extent/thickness filtering) still applies, so real noise still
gets cleaned, just not the two rules that assume "small trunk in a
mostly-empty frame."

Usage:
    python scripts/remove_table_cedric_masking.py \
        --xr /mnt/.../CEP_368B_2026_XR.tif \
        --output /mnt/.../CEP_368B_2026_XR_cleaned.tif
"""

import argparse
from pathlib import Path

import numpy as np
import tifffile as tiff
from skimage import measure
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt, binary_fill_holes

SEUIL_ETAPE1 = 250
SEUIL_TAILLE_MIN = 50
MARGE_BORD = 10
RATIO_ALLONGEMENT_TABLE = 3.0
FRACTION_BAS_TABLE = 0.75
SEUIL_EXTENT_ARC = 0.30
SEUIL_EPAISSEUR_MAX_FILAMENT = 8
SEUIL_POSITION_RECUPERATION = 0.65

POT_BASELINE_SLICES = 50     # how many early (apex) slices define "normal trunk area"
POT_AREA_FACTOR = 3.0        # area must exceed baseline x this to count as "pot-sized"
POT_SUSTAIN_SLICES = 15      # must stay above threshold this many consecutive slices


def _touche_bord(masque_region, marge=MARGE_BORD):
    return (
        np.any(masque_region[:marge, :]) or
        np.any(masque_region[-marge:, :]) or
        np.any(masque_region[:, :marge]) or
        np.any(masque_region[:, -marge:])
    )


def _crop_composante(labels, p):
    min_row, min_col, max_row, max_col = p.bbox
    r0, c0 = max(min_row - 1, 0), max(min_col - 1, 0)
    r1, c1 = min(max_row + 1, labels.shape[0]), min(max_col + 1, labels.shape[1])
    return labels[r0:r1, c0:c1] == p.label


def _epaisseur_typique(masque_region_crop):
    dt = distance_transform_edt(masque_region_crop)
    squelette = skeletonize(masque_region_crop)
    valeurs_dt_squelette = dt[squelette]
    if valeurs_dt_squelette.size == 0:
        return 2 * dt.max()
    return 2 * np.percentile(valeurs_dt_squelette, 90)


def masquer_tronc(coupe, skip_border_table=False):
    masque_brut = coupe > SEUIL_ETAPE1
    labels = measure.label(masque_brut)
    props = measure.regionprops(labels)
    h, w = coupe.shape
    masque_final = np.zeros(coupe.shape, dtype=bool)
    candidats_recuperation_arc_fin = []

    for p in props:
        if p.area < SEUIL_TAILLE_MIN:
            continue
        if not skip_border_table and _touche_bord(labels == p.label):
            continue
        min_row, min_col, max_row, max_col = p.bbox
        hauteur_bbox = max_row - min_row
        largeur_bbox = max_col - min_col
        allongement = max(largeur_bbox, hauteur_bbox) / max(min(largeur_bbox, hauteur_bbox), 1)
        position_verticale_rel = p.centroid[0] / h
        if (not skip_border_table and
                position_verticale_rel > FRACTION_BAS_TABLE and
                allongement > RATIO_ALLONGEMENT_TABLE):
            continue
        aire_bbox = hauteur_bbox * largeur_bbox
        extent = p.area / max(aire_bbox, 1)
        if extent < SEUIL_EXTENT_ARC:
            candidats_recuperation_arc_fin.append((p, position_verticale_rel))
            continue
        masque_region_crop = _crop_composante(labels, p)
        epaisseur_typique = _epaisseur_typique(masque_region_crop)
        if epaisseur_typique < SEUIL_EPAISSEUR_MAX_FILAMENT:
            if p.filled_area > p.area:
                masque_rempli = binary_fill_holes(masque_region_crop)
                epaisseur_remplie = _epaisseur_typique(masque_rempli)
                if epaisseur_remplie >= SEUIL_EPAISSEUR_MAX_FILAMENT:
                    masque_final[labels == p.label] = True
            continue
        masque_final[labels == p.label] = True

    for p, position_verticale_rel in candidats_recuperation_arc_fin:
        if position_verticale_rel >= SEUIL_POSITION_RECUPERATION:
            continue
        masque_region_crop = _crop_composante(labels, p)
        if _epaisseur_typique(masque_region_crop) >= SEUIL_EPAISSEUR_MAX_FILAMENT:
            masque_final[labels == p.label] = True

    return binary_fill_holes(masque_final)


def detect_pot_start(vol, threshold):
    """Returns the first Z index where the pot begins, or None if never
    detected (in which case border/table filtering stays on for the
    whole volume, i.e. old behavior)."""
    n_slices = vol.shape[0]
    areas = np.array([(vol[z] > threshold).sum() for z in range(n_slices)])

    valid_areas = areas[areas > 0]
    if len(valid_areas) < POT_BASELINE_SLICES:
        print("  [pot-detect] not enough foreground slices to establish a baseline, "
              "skipping pot detection (filters stay on for entire volume)")
        return None
    baseline = np.median(valid_areas[:POT_BASELINE_SLICES])
    cutoff_area = baseline * POT_AREA_FACTOR
    print(f"  [pot-detect] baseline trunk area ~{baseline:.0f} px (first "
          f"{POT_BASELINE_SLICES} slices), pot cutoff area = {cutoff_area:.0f} px")

    above = areas > cutoff_area
    for z in range(n_slices - POT_SUSTAIN_SLICES):
        if above[z:z + POT_SUSTAIN_SLICES].all():
            print(f"  [pot-detect] pot starts at slice {z} (area {areas[z]:.0f} px, "
                  f"sustained for {POT_SUSTAIN_SLICES}+ slices)")
            return z

    print("  [pot-detect] no sustained area jump found, pot never detected, "
          "filters stay on for entire volume")
    return None


def main(xr_path, output_path):
    with tiff.TiffFile(xr_path) as tf:
        page = tf.pages[0]
        original_dtype = page.asarray().dtype
        xres_tag = page.tags.get("XResolution")
        yres_tag = page.tags.get("YResolution")
        xres = (xres_tag.value[0] / xres_tag.value[1]) if xres_tag else None
        yres = (yres_tag.value[0] / yres_tag.value[1]) if yres_tag else None
        imagej_meta = tf.imagej_metadata or {}

    vol = tiff.imread(xr_path)
    n_slices, height, width = vol.shape
    print(f"Volume shape: {vol.shape}")

    pot_start_z = detect_pot_start(vol, SEUIL_ETAPE1)

    vol_cleaned = np.zeros_like(vol)
    total_kept_voxels = 0

    for z in range(n_slices):
        skip_border_table = pot_start_z is not None and z >= pot_start_z
        masque = masquer_tronc(vol[z], skip_border_table=skip_border_table)
        vol_cleaned[z] = np.where(masque, vol[z], 0)
        total_kept_voxels += masque.sum()
        if z % 200 == 0:
            print(f"  processed slice {z}/{n_slices}")

    fg_total = (vol > SEUIL_ETAPE1).sum()
    pct = 100 * total_kept_voxels / fg_total if fg_total else 0.0
    print(f"\nKept {total_kept_voxels} voxels across all slices "
          f"({pct:.2f}% of original foreground survived masking)")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if xres:
        tiff.imwrite(output_path, vol_cleaned.astype(original_dtype), imagej=True,
                     resolution=(xres, yres),
                     metadata={"spacing": imagej_meta.get("spacing"),
                               "unit": imagej_meta.get("unit", "mm"),
                               "axes": "ZYX"})
    else:
        tiff.imwrite(output_path, vol_cleaned.astype(original_dtype))
    print(f"Saved cleaned volume to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xr", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.xr, args.output)