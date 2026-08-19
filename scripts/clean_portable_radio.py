"""
Clean up a real radiograph before calibration: mask out a bright
band/artifact (which may be a straight horizontal edge OR a tilted
diagonal edge unrelated to the trunk's own tilt), and optionally rotate
to straighten the trunk separately. Doesn't touch the original file, and
doesn't touch calibrate_drr.py, use the cleaned output as your --pxr
input afterward.

TWO WAYS TO REMOVE THE BAND, pick whichever matches what you see:

1) --crop-px (+ --side): use this if the band's edge is a straight
   horizontal line. Removes whole rows.

2) --diagonal-mask x1,y1,x2,y2 (+ --mask-side): use this if the band's
   edge is tilted independent of the trunk's own tilt (a straight rotate
   won't fix it, since it's tilted for a different reason than the
   specimen). Find two points ON the edge line in Fiji (Point tool,
   note x,y from the status bar for two spots along the boundary), pass
   them here. Everything on --mask-side of that line gets zeroed out
   (set to 0, matching background convention) rather than cropped, since
   a diagonal boundary doesn't crop cleanly into a rectangle.

--rotate-deg is independent of both, use it separately if you also want
to straighten the trunk itself. Order of operations: crop/mask first,
then rotate, both applied to the same output if given together.

Usage (diagonal mask example, band at bottom):
    python scripts/clean_portable_radio.py \
        --input raw_tiff_check/CEP_1191/69ca5e44e42f5e0a22520ec8_inverted.tif \
        --diagonal-mask 0,2650,3072,2820 \
        --mask-side below \
        --output raw_tiff_check/CEP_1191/69ca5e44e42f5e0a22520ec8_cleaned.tif

Usage (straight crop example):
    python scripts/clean_portable_radio.py \
        --input raw_tiff_check/CEP_1191/69ca5e44e42f5e0a22520ec8_inverted.tif \
        --side bottom --crop-px 450 \
        --output raw_tiff_check/CEP_1191/69ca5e44e42f5e0a22520ec8_cleaned.tif
"""

import argparse
from pathlib import Path

import numpy as np
import tifffile as tiff
from scipy.ndimage import rotate as scipy_rotate


def apply_diagonal_inpaint(img: np.ndarray, x1, y1, x2, y2, side: str,
                            inpaint_radius: int = 15) -> np.ndarray:
    """Remove the region on `side` of the line through (x1,y1)-(x2,y2)
    using OpenCV inpainting (Telea's method), which extends the
    surrounding texture/gradient naturally into the masked region,
    instead of filling with a flat statistical estimate.

    cv2.inpaint only supports 8-bit images, so the image is normalized
    to [0,255] for the inpainting step, then the result is rescaled back
    to the original data range before being written into the masked
    pixels. The KEPT region is left completely untouched, at full
    original precision, only the masked pixels get the (rescaled)
    inpainted result.
    """
    import cv2

    rows, cols = img.shape[0], img.shape[1]
    if x2 == x1:
        raise ValueError("x1 and x2 must differ to define a line slope")
    slope = (y2 - y1) / (x2 - x1)

    col_idx = np.arange(cols)
    line_y = y1 + slope * (col_idx - x1)
    row_idx = np.arange(rows).reshape(-1, 1)
    binary_mask = (row_idx > line_y.reshape(1, -1)) if side == "below" \
        else (row_idx < line_y.reshape(1, -1))

    print(f"Inpainting {binary_mask.sum()}/{binary_mask.size} pixels "
          f"({100*binary_mask.mean():.1f}%) {side} the boundary line")

    img_f = img.astype(np.float64)
    lo, hi = img_f.min(), img_f.max()
    img_8u = ((img_f - lo) / (hi - lo) * 255).astype(np.uint8)

    mask_8u = (binary_mask.astype(np.uint8)) * 255
    inpainted_8u = cv2.inpaint(img_8u, mask_8u, inpaint_radius, cv2.INPAINT_TELEA)

    # rescale the inpainted result back to the original data range
    inpainted_full = inpainted_8u.astype(np.float64) / 255.0 * (hi - lo) + lo

    out = img.copy().astype(np.float64)
    out[binary_mask] = inpainted_full[binary_mask]  # only touch masked pixels
    return out.astype(img.dtype)


def apply_diagonal_hard_crop(img: np.ndarray, x1, y1, x2, y2, side: str) -> np.ndarray:
    """Genuinely remove the band, no fill, no inpaint, just delete the
    rows. Since the boundary is diagonal, a straight rectangular crop has
    to cut at the SHALLOWEST point of the line (closest to the kept
    side), guaranteeing every column is fully clean, at the cost of
    losing a bit of extra real image on the side where the line was
    deeper. No fake pixels anywhere in the result, so no seam is
    possible, this is the only way to have literally zero trace.
    """
    slope_ys = [y1, y2]
    if side == "below":
        cut_y = min(slope_ys)  # shallowest = smallest y kept, cut here so nothing below survives
        out = img[:int(np.floor(cut_y))]
        print(f"Hard-cropped to row {int(np.floor(cut_y))} (shallowest point of the line), "
              f"guarantees zero band pixels remain, at the cost of trimming a bit of real "
              f"image on the deeper side")
    else:
        cut_y = max(slope_ys)  # shallowest for 'above' case = largest y kept
        out = img[int(np.ceil(cut_y)):]
        print(f"Hard-cropped from row {int(np.ceil(cut_y))} onward (shallowest point of the "
              f"line), guarantees zero band pixels remain")
    return out


def apply_diagonal_mask(img: np.ndarray, x1, y1, x2, y2, side: str,
                         feather_px: float = 25.0, local_strip_px: float = 150.0) -> np.ndarray:
    """Fill the region on `side` of the line through (x1,y1)-(x2,y2) with
    a LOCAL background estimate (a strip of pixels right next to the
    boundary, on the kept side, not a global percentile over the whole
    image), blended in with a soft feathered edge instead of a sharp cut.

    Using a global background estimate is wrong here: a 3072x3072 image
    is mostly pure-black background far from the specimen, so a global
    5th percentile just returns ~0, nothing like the actual local
    brightness right next to the boundary. Sampling a strip immediately
    adjacent to the line instead gives a realistic local estimate.

    'below' = pixels with row index greater than the line's y at that
    column (i.e. physically below the line); 'above' = the opposite.
    """
    rows, cols = img.shape[0], img.shape[1]
    if x2 == x1:
        raise ValueError("x1 and x2 must differ to define a line slope")
    slope = (y2 - y1) / (x2 - x1)

    col_idx = np.arange(cols)
    line_y = y1 + slope * (col_idx - x1)  # y on the line for every column

    row_idx = np.arange(rows).reshape(-1, 1)
    binary_mask = (row_idx > line_y.reshape(1, -1)) if side == "below" \
        else (row_idx < line_y.reshape(1, -1))

    # LOCAL background estimate: a strip of `local_strip_px` rows
    # immediately adjacent to the boundary line, on the KEPT side only.
    if side == "below":
        strip_mask = (row_idx <= line_y.reshape(1, -1)) & \
                     (row_idx > line_y.reshape(1, -1) - local_strip_px)
    else:
        strip_mask = (row_idx >= line_y.reshape(1, -1)) & \
                     (row_idx < line_y.reshape(1, -1) + local_strip_px)

    strip_pixels = img[strip_mask]
    if strip_pixels.size == 0:
        raise ValueError("local_strip_px produced an empty sample, increase it")
    background_level = np.median(strip_pixels)
    # robust std via median absolute deviation, NOT raw .std(): a raw std
    # is very sensitive to even a small fraction of the strip
    # accidentally overlapping something bright (e.g. part of the trunk
    # near where the boundary line passes close to the specimen), MAD is
    # far more resistant to that kind of contamination
    mad = np.median(np.abs(strip_pixels - background_level))
    background_std = 1.4826 * mad  # scaling factor makes MAD ~= std for a normal distribution

    print(f"Estimated LOCAL background (strip of {local_strip_px}px next to the boundary): "
          f"level={background_level:.2f}  noise_std={background_std:.2f}  "
          f"(from {strip_pixels.size} pixels)")

    rng = np.random.default_rng(0)
    fill = background_level + rng.normal(0, max(background_std, 1e-6), size=img.shape)
    fill = np.clip(fill, 0, None)

    # ONE-SIDED feather: never mix in the original band value, it's often
    # extremely bright (e.g. ~25000) and even a small blend weight of it
    # produces a visible glow/bright line right at the boundary. Instead:
    #   - masked side: ALWAYS pure fill, no original value involved at all
    #   - kept side: fades from pure original (far from line) to pure
    #     fill (right at the line), so the transition only ever touches
    #     real background/specimen values and the fill, never the band
    signed_dist = line_y.reshape(1, -1) - row_idx  # positive on kept side ("below" case)
    if side == "above":
        signed_dist = -signed_dist
    # signed_dist > 0 => kept side, distance from line
    # signed_dist <= 0 => masked side

    kept_alpha = np.clip(1.0 - signed_dist / feather_px, 0.0, 1.0)  # 1=full fill, 0=full original
    kept_alpha = np.where(signed_dist > 0, kept_alpha, 1.0)  # masked side always alpha=1 (pure fill)

    out = img.astype(np.float64) * (1 - kept_alpha) + fill * kept_alpha
    out = out.astype(img.dtype)

    print(f"Diagonal region filled with background level (feather sigma={feather_px}px), "
          f"{binary_mask.sum()}/{binary_mask.size} pixels affected ({100*binary_mask.mean():.1f}%)")
    return out


def main(input_path, side, crop_px, diagonal_mask, mask_side, rotate_deg, output_path,
         feather_px=25.0, local_strip_px=150.0, method="fill", inpaint_radius=15):
    img = tiff.imread(input_path)
    print(f"Input:  shape={img.shape}  dtype={img.dtype}")

    if crop_px is not None:
        if crop_px <= 0 or crop_px >= img.shape[0]:
            raise ValueError(f"--crop-px must be between 1 and {img.shape[0]-1}, got {crop_px}")
        if side == "top":
            img = img[crop_px:]
            print(f"Cropped {crop_px} rows from top, {img.shape[0]} rows remain")
        else:
            img = img[:-crop_px]
            print(f"Cropped {crop_px} rows from bottom, {img.shape[0]} rows remain")

    if diagonal_mask is not None:
        x1, y1, x2, y2 = diagonal_mask
        if method == "hard_crop":
            img = apply_diagonal_hard_crop(img, x1, y1, x2, y2, mask_side)
        elif method == "inpaint":
            img = apply_diagonal_inpaint(img, x1, y1, x2, y2, mask_side,
                                          inpaint_radius=inpaint_radius)
        else:
            img = apply_diagonal_mask(img, x1, y1, x2, y2, mask_side, feather_px=feather_px,
                                       local_strip_px=local_strip_px)

    if rotate_deg is not None and rotate_deg != 0:
        dtype = img.dtype
        img = scipy_rotate(img.astype(np.float64), angle=rotate_deg, reshape=True,
                            order=1, cval=0.0)
        img = img.astype(dtype)
        print(f"Rotated {rotate_deg} degrees, new shape={img.shape}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(str(output_path), img)
    print(f"Saved: {output_path}  final shape={img.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="path to the real radiograph tif")
    parser.add_argument("--side", choices=["top", "bottom"], default="top",
                         help="which edge --crop-px removes rows from")
    parser.add_argument("--crop-px", type=int, default=None,
                         help="number of rows to remove (straight horizontal edge only)")
    parser.add_argument("--diagonal-mask", type=str, default=None,
                         help="'x1,y1,x2,y2' two points on the tilted band edge, found in Fiji")
    parser.add_argument("--mask-side", choices=["above", "below"], default="below",
                         help="which side of --diagonal-mask's line to fill")
    parser.add_argument("--feather-px", type=float, default=25.0,
                         help="Gaussian blur radius (px) for the mask's edge transition, "
                              "so it blends smoothly instead of a hard cut. Larger = softer "
                              "transition. Only used with --diagonal-mask.")
    parser.add_argument("--local-strip-px", type=float, default=150.0,
                         help="width (px) of the strip right next to the boundary line, on "
                              "the kept side, used to estimate LOCAL background level. Don't "
                              "use a global image-wide estimate, most of a 3072x3072 image is "
                              "far-field black background that has nothing to do with the "
                              "actual brightness near the boundary. Increase this if the "
                              "estimate still looks off.")
    parser.add_argument("--method", choices=["fill", "inpaint", "hard_crop"], default="fill",
                         help="'fill' = local background level + noise, feathered edge. "
                              "'inpaint' = OpenCV Telea inpainting, extends surrounding "
                              "texture/gradient naturally. 'hard_crop' = genuinely delete the "
                              "rows, no fill at all, guarantees zero trace since there's "
                              "nothing fake in the result, but loses a bit of real image on "
                              "the shallower side of the diagonal and changes image dimensions.")
    parser.add_argument("--inpaint-radius", type=int, default=15,
                         help="inpainting radius in px (only used with --method inpaint), "
                              "how far around each masked pixel OpenCV looks for source "
                              "texture to extend inward")
    parser.add_argument("--rotate-deg", type=float, default=None,
                         help="degrees to rotate so the trunk is vertical, applied "
                              "independently of crop/mask")
    parser.add_argument("--output", required=True, help="path to save the cleaned copy")
    args = parser.parse_args()

    diag = None
    if args.diagonal_mask is not None:
        parts = [float(v) for v in args.diagonal_mask.split(",")]
        if len(parts) != 4:
            parser.error("--diagonal-mask needs exactly 4 comma-separated values: x1,y1,x2,y2")
        diag = parts

    main(args.input, args.side, args.crop_px, diag, args.mask_side, args.rotate_deg,
         args.output, args.feather_px, args.local_strip_px, args.method, args.inpaint_radius)
    