import sys
import time
import threading
from pathlib import Path

import imagej
import jpype
import numpy as np
import tifffile as tiff
from scyjava import jimport

# =============================================================================
# PATHS
# =============================================================================

FIJI_PATH = "/home/phukon/Fiji.app"

PROJECT_ROOT = Path("/home/phukon/code_python/VineRadiologistAI")
SCRIPT_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

SPECIMEN = "CEP_313B"

OUT_DIR = Path(
    "/home/phukon/Desktop/Ct_PXR_manual_registration"
)

CT_PATH = OUT_DIR / f"{SPECIMEN}_2026_XR.tif"
PXR_PATH = OUT_DIR / f"{SPECIMEN}_2026_PXR.tif"

TRANSFORM_PATH = OUT_DIR / f"{SPECIMEN}_transform_test.txt"
REGISTRATION_INFO_PATH = OUT_DIR / f"{SPECIMEN}_registration_info.txt"
PREVIEW_PATH = OUT_DIR / f"{SPECIMEN}_DRR_preview.tif"
OVERLAY_PATH = OUT_DIR / f"{SPECIMEN}_PXR_DRR_overlay.tif"
FINAL_DRR_PATH = OUT_DIR / f"{SPECIMEN}_DRR_test.tif"

# Resume an existing registration, or start from identity.
# "ask" = ask in the terminal if a transform exists; "resume" = load it;
# "new" = always start from identity, without touching existing files.
START_POSE_MODE = "ask"

# None: look for both the current *_transform_test.txt and the earlier
# *_transform.txt for this specimen. Set a Path here to load a specific file.
RESUME_TRANSFORM_PATH = None


# =============================================================================
# DESKTOP WINDOW LAYOUT
# =============================================================================
#
# Window bounds are computed from the actual screen size after Fiji starts.
# Layout:
#
#   PXR | DRR        | 3D Viewer
#   -----------      |
#   PXR + DRR fuse   |
#
# Every 2D image window is automatically fit-to-window so the complete image
# is visible instead of opening at an overly zoomed 1:1 view.
PXR_WINDOW_BOUNDS = None
DRR_WINDOW_BOUNDS = None
OVERLAY_WINDOW_BOUNDS = None
VIEWER_WINDOW_BOUNDS = None


# =============================================================================
# OPTIONAL PXR / DRR ALIGNMENT GRID
# =============================================================================
#
# Press G to toggle the same overlay grid on BOTH the PXR and DRR.
# It is only an ImageJ overlay:
#   - it does NOT modify image pixels
#   - it is NOT written into the saved PXR / DRR TIFF
#   - it does NOT affect registration or transforms
#
# Fractions are relative to the displayed image dimensions.
# 0.50 is the image centre.
GRID_FRACTIONS = (0.25, 0.50, 0.75)
GRID_LINE_WIDTH = 1.0
GRID_CENTER_LINE_WIDTH = 2.0


# =============================================================================
# MANUAL REGISTRATION
# =============================================================================

TRANSLATION_STEP_MM = 1.0
ROTATION_STEP_DEG = 0.5


# =============================================================================
# DRR SETTINGS
# =============================================================================

SID_MM = 1230.0
SPD_MM = 800.0

OFFSET_V_MM = 0.0
OFFSET_U_MM = 0.0

# Linear gain applied to the integrated attenuation.
# Because the output is now in log/attenuation domain, this is a linear
# intensity calibration factor, not an exponential-opacity parameter.
ATTENUATION_SCALE = 0.015
BEAM_AXIS = 0

PREVIEW_SIZE = 512
FINAL_SIZE = 3072

# Preview uses a lower-resolution CT but preserves its physical dimensions.
# This makes ENTER much faster.
PREVIEW_CT_DOWNSAMPLE = 4

PREVIEW_ROW_CHUNK = 32
PREVIEW_SAMPLE_CHUNK = 256

# Safer/smaller chunks for CPU fallback.
CPU_PREVIEW_ROW_CHUNK = 8
CPU_PREVIEW_SAMPLE_CHUNK = 64

FINAL_ROW_CHUNK = 8
FINAL_SAMPLE_CHUNK = 128

# Live preview: mouse/keyboard changes to the PHYSICAL XYZ pose automatically
# trigger a new preview, including front/back beam-depth motion.
LIVE_PREVIEW_POLL_SEC = 0.08
POSE_ROT_ATOL = 1e-6
POSE_TRANS_ATOL_MM = 1e-4

# Display-only gains for the red/green fused registration window.
# These do NOT affect the PXR, DRR values, transform, or saved final DRR.
#
# Keep these between 0 and 1:
#   lower = darker channel
#   higher = brighter channel
OVERLAY_PXR_GAIN = 0.55
OVERLAY_DRR_GAIN = 0.55


# =============================================================================
# PROJECT / SHARED DRR IMPORTS
# =============================================================================

from VineRadiologist.io import load_volume  # noqa: E402

# IMPORTANT:
# calibrate_drr.py is now the single source of truth for the cone-beam DRR.
# Put calibrate_drr.py in the same scripts directory as this file.
from calibrate_drr import (  # noqa: E402
    CUPY_AVAILABLE,
    crop_top_to_match_pxr,
    detector_pixel_spacing,
    generate_drr,
    make_geometry,
    save_drr_16bit,
)


# =============================================================================
# TRANSFORM HELPERS
# =============================================================================

def transform3d_to_numpy(transform, Matrix4d):
    m = Matrix4d()
    transform.get(m)

    return np.array(
        [
            [m.m00, m.m01, m.m02, m.m03],
            [m.m10, m.m11, m.m12, m.m13],
            [m.m20, m.m21, m.m22, m.m23],
            [m.m30, m.m31, m.m32, m.m33],
        ],
        dtype=np.float64,
    )


def numpy_to_transform3d(matrix, Transform3D, Matrix4d):
    """Convert a 4x4 NumPy matrix into Java3D Transform3D."""
    a = np.asarray(matrix, dtype=np.float64)

    m = Matrix4d()
    m.m00, m.m01, m.m02, m.m03 = map(float, a[0])
    m.m10, m.m11, m.m12, m.m13 = map(float, a[1])
    m.m20, m.m21, m.m22, m.m23 = map(float, a[2])
    m.m30, m.m31, m.m32, m.m33 = map(float, a[3])

    return Transform3D(m)


def rotation_matrix_xyz(axis, angle_rad):
    """Incremental physical XYZ rotation matrix."""
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))

    if axis == "x":
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, c, -s],
                [0.0, s, c],
            ],
            dtype=np.float64,
        )

    if axis == "y":
        return np.array(
            [
                [c, 0.0, s],
                [0.0, 1.0, 0.0],
                [-s, 0.0, c],
            ],
            dtype=np.float64,
        )

    if axis == "z":
        return np.array(
            [
                [c, -s, 0.0],
                [s, c, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    raise ValueError("axis must be x, y, or z")


def capture_fiji_transform(content, Transform3D, Matrix4d):
    """
    Capture the current selected CT pose from Fiji.

    Fiji keeps the genuine user translation in localTranslate, while
    localRotate also contains the center-of-rotation compensation.
    """
    local_translate = Transform3D()
    local_rotate = Transform3D()

    content.getLocalTranslate(local_translate)
    content.getLocalRotate(local_rotate)

    T_translate = transform3d_to_numpy(local_translate, Matrix4d)
    T_rotate = transform3d_to_numpy(local_rotate, Matrix4d)

    exact_total = T_translate @ T_rotate
    rotation_xyz = T_rotate[:3, :3].copy()
    true_translation_xyz = T_translate[:3, 3].copy()

    return exact_total, rotation_xyz, true_translation_xyz


def renderer_matrix_from_fiji_pose(
    rotation_xyz,
    true_translation_xyz,
    vol_shape_zyx,
    spacing_zyx,
):
    """
    Rebuild the complete Fiji-style 4x4 matrix for the CT representation
    actually being rendered.

    t_total = true_translation + center - R @ center
    """
    sz, sy, sx = map(float, spacing_zyx)
    nz, ny, nx = map(int, vol_shape_zyx)

    center_xyz = np.array(
        [
            nx * sx / 2.0,
            ny * sy / 2.0,
            nz * sz / 2.0,
        ],
        dtype=np.float64,
    )

    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rotation_xyz
    out[:3, 3] = (
        true_translation_xyz
        + center_xyz
        - rotation_xyz @ center_xyz
    )

    return out


# =============================================================================
# DRR HELPERS
# =============================================================================

def display_uint16(img):
    """
    Contrast-scale a DRR for Fiji display.
    The cone-beam calculation itself lives in calibrate_drr.py.
    """
    x = np.asarray(img, dtype=np.float32)
    finite = np.isfinite(x)

    if not finite.any():
        return np.zeros(x.shape, dtype=np.uint16)

    candidate = x[finite & (x > 0)]

    if candidate.size < 100:
        candidate = x[finite]

    lo, hi = np.percentile(candidate, [1.0, 99.5])

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(candidate))
        hi = float(np.max(candidate))

    if hi <= lo:
        return np.zeros(x.shape, dtype=np.uint16)

    y = np.clip(
        (x - lo) / (hi - lo),
        0.0,
        1.0,
    )
    y[~finite] = 0.0

    return np.round(y * 65535.0).astype(np.uint16)


def save_preview_display(path, drr, pixel_spacing_mm):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    out = display_uint16(drr)
    pixels_per_cm = 10.0 / float(pixel_spacing_mm)

    tiff.imwrite(
        path,
        out,
        photometric="minisblack",
        resolution=(pixels_per_cm, pixels_per_cm),
        resolutionunit="CENTIMETER",
    )


# NOTE:
# Final DRRs are saved with calibrate_drr.save_drr_16bit(), so manual
# registration and calibrate_drr.py use the exact same quantitative
# float -> uint16 mapping.
#
# Only ENTER previews use the local percentile display stretch above.


def make_manual_geometry(detector_size, spacing):
    """
    Manual registration deliberately uses beam_axis=0 even though
    calibrate_drr.py retains beam_axis=1 as its legacy/default calibration
    convention.
    """
    return make_geometry(
        detector_size=int(detector_size),
        spacing_zyx=tuple(float(v) for v in spacing),
        sid_mm=SID_MM,
        spd_mm=SPD_MM,
        offset_v_mm=OFFSET_V_MM,
        offset_u_mm=OFFSET_U_MM,
        beam_axis=0,
    )


def render_preview(
    preview_volume,
    preview_spacing,
    rotation_xyz,
    true_translation_xyz,
    pxr_shape,
):
    geometry = make_manual_geometry(
        PREVIEW_SIZE,
        preview_spacing,
    )

    drr = generate_drr(
        preview_volume,
        geometry,
        attenuation_scale=ATTENUATION_SCALE,
        rotation_xyz=rotation_xyz,
        true_translation_xyz=true_translation_xyz,
        use_gpu=CUPY_AVAILABLE,
        row_chunk=PREVIEW_ROW_CHUNK,
        sample_chunk=PREVIEW_SAMPLE_CHUNK,
        verbose=False,
    )

    return crop_top_to_match_pxr(
        drr,
        pxr_shape,
    )


def render_final(
    full_volume,
    full_spacing,
    rotation_xyz,
    true_translation_xyz,
    pxr_shape,
):
    if not CUPY_AVAILABLE:
        return None

    geometry = make_manual_geometry(
        FINAL_SIZE,
        full_spacing,
    )

    drr = generate_drr(
        full_volume,
        geometry,
        attenuation_scale=ATTENUATION_SCALE,
        rotation_xyz=rotation_xyz,
        true_translation_xyz=true_translation_xyz,
        use_gpu=True,
        row_chunk=FINAL_ROW_CHUNK,
        sample_chunk=FINAL_SAMPLE_CHUNK,
        verbose=False,
    )

    return crop_top_to_match_pxr(
        drr,
        pxr_shape,
    )


# =============================================================================
# START FIJI
# =============================================================================

ij = imagej.init(
    FIJI_PATH,
    mode="interactive",
)

IJ = jimport("ij.IJ")
Image3DUniverse = jimport("ij3d.Image3DUniverse")
StackConverter = jimport("ij.process.StackConverter")

Transform3D = jimport("org.jogamp.java3d.Transform3D")
Matrix4d = jimport("org.jogamp.vecmath.Matrix4d")
KeyEvent = jimport("java.awt.event.KeyEvent")
Overlay = jimport("ij.gui.Overlay")
Line = jimport("ij.gui.Line")
Color = jimport("java.awt.Color")
Toolkit = jimport("java.awt.Toolkit")


# =============================================================================
# LOAD CT
# =============================================================================

volume = load_volume(str(CT_PATH))

if volume.ndim != 3:
    raise RuntimeError(
        f"Expected 3D CT, got {volume.shape}"
    )

ct = IJ.openImage(str(CT_PATH))

if ct is None:
    raise RuntimeError(
        f"Could not open CT: {CT_PATH}"
    )

cal = ct.getCalibration()

spacing_zyx = np.array(
    [
        float(cal.pixelDepth),
        float(cal.pixelHeight),
        float(cal.pixelWidth),
    ],
    dtype=np.float64,
)

# Fast preview representation of the SAME physical CT.
f = int(PREVIEW_CT_DOWNSAMPLE)

preview_volume = np.ascontiguousarray(
    volume[::f, ::f, ::f]
)

preview_spacing = (
    spacing_zyx
    * float(f)
)

print(
    "CT DRR volume:",
    f"shape={volume.shape}, "
    f"min={float(np.nanmin(volume)):.3f}, "
    f"max={float(np.nanmax(volume)):.3f}, "
    f"nonzero={int(np.count_nonzero(volume))}"
)
print(
    "CT spacing ZYX mm:",
    tuple(float(v) for v in spacing_zyx)
)
print(
    "Preview CT:",
    f"shape={preview_volume.shape}, "
    f"spacing={tuple(float(v) for v in preview_spacing)}"
)
print(
    "Shared DRR module: calibrate_drr.py | "
    "manual beam_axis=0 | "
    f"SID={SID_MM:g} mm | SPD={SPD_MM:g} mm | "
    "full detector pixel spacing=0.139 mm | "
    "log/attenuation output"
)

# Viewer copy only.
ct.setDisplayRange(20, 800)
StackConverter(ct).convertToGray8()


def compute_desktop_layout():
    """
    Build a screen-aware four-window layout.

    Left side:
      top-left  = PXR
      top-right = live DRR
      bottom    = fused PXR/DRR

    Right side:
      full height = 3D Viewer
    """
    screen = Toolkit.getDefaultToolkit().getScreenSize()

    screen_w = int(screen.width)
    screen_h = int(screen.height)

    # Leave a little room for desktop panels/window decorations.
    usable_h = max(600, screen_h - 50)

    left_w = int(round(screen_w * 0.56))
    right_w = max(500, screen_w - left_w)

    top_h = int(round(usable_h * 0.50))
    bottom_h = usable_h - top_h

    half_left = left_w // 2

    return (
        (0, 0, half_left, top_h),
        (half_left, 0, left_w - half_left, top_h),
        (0, top_h, left_w, bottom_h),
        (left_w, 0, right_w, usable_h),
    )


(
    PXR_WINDOW_BOUNDS,
    DRR_WINDOW_BOUNDS,
    OVERLAY_WINDOW_BOUNDS,
    VIEWER_WINDOW_BOUNDS,
) = compute_desktop_layout()


def fit_image_to_window(imp):
    """
    Show the whole 2D image in its current ImageJ window.

    ImageJ otherwise tends to preserve a 1:1 magnification when a large image
    is put into a smaller desktop window, which looks excessively zoomed-in.
    """
    if imp is None:
        return

    try:
        canvas = imp.getCanvas()
        if canvas is not None:
            canvas.fitToWindow()
        imp.updateAndDraw()
    except Exception:
        pass


def set_window_bounds(window, bounds, name="window"):
    """
    Put an ImageJ/AWT window at a fixed desktop location.

    bounds = (x, y, width, height) in screen pixels.
    This affects only desktop layout, never registration or DRR geometry.
    """
    if window is None:
        return

    x, y, width, height = map(int, bounds)

    try:
        window.setBounds(x, y, width, height)
        return
    except Exception:
        pass

    try:
        window.setLocation(x, y)
        window.setSize(width, height)
    except Exception as exc:
        print(
            f"Could not position {name}: "
            f"{type(exc).__name__}: {exc}"
        )


# =============================================================================
# LOAD PXR
# =============================================================================

pxr = IJ.openImage(str(PXR_PATH))

if pxr is None:
    raise RuntimeError(
        f"Could not open PXR: {PXR_PATH}"
    )

pxr.setTitle("PXR_REFERENCE")
pxr.show()

set_window_bounds(
    pxr.getWindow(),
    PXR_WINDOW_BOUNDS,
    "PXR_REFERENCE",
)

fit_image_to_window(pxr)

pxr_shape = (
    int(pxr.getHeight()),
    int(pxr.getWidth()),
)

pxr_np = np.squeeze(
    tiff.imread(str(PXR_PATH))
)

if pxr_np.ndim != 2:
    raise RuntimeError(
        f"Expected 2D PXR for overlay, got shape {pxr_np.shape}"
    )

# Display-normalized PXR used only for the live fused visualization.
# It does not modify the source PXR and is never used by the DRR renderer.
pxr_display_full = display_uint16(pxr_np)


def resize_nearest_2d(img, target_shape):
    """Fast display-only nearest-neighbour resize for the fused view."""
    src = np.asarray(img)

    target_h, target_w = map(int, target_shape)

    if src.shape == (target_h, target_w):
        return src

    yy = np.rint(
        np.linspace(
            0,
            src.shape[0] - 1,
            target_h,
        )
    ).astype(np.int64)

    xx = np.rint(
        np.linspace(
            0,
            src.shape[1] - 1,
            target_w,
        )
    ).astype(np.int64)

    return src[np.ix_(yy, xx)]


def scale_overlay_channel(channel16, gain):
    """
    Simple display-only channel scaling.

    gain = 1.0 keeps the current display normalization.
    gain < 1.0 darkens the channel without changing contrast/gamma.
    """
    x = np.asarray(
        channel16,
        dtype=np.float32,
    ) / 65535.0

    x = np.clip(
        x * float(gain),
        0.0,
        1.0,
    )

    return np.round(
        x * 255.0
    ).astype(np.uint8)


def make_registration_overlay(drr):
    """
    False-colour registration fusion.

    PXR = red
    DRR = green

    Good agreement tends toward neutral/white-grey; misregistration appears
    as separated red/green structures.
    """
    drr16 = display_uint16(drr)

    pxr16 = resize_nearest_2d(
        pxr_display_full,
        drr16.shape,
    )

    pxr8 = scale_overlay_channel(
        pxr16,
        OVERLAY_PXR_GAIN,
    )

    drr8 = scale_overlay_channel(
        drr16,
        OVERLAY_DRR_GAIN,
    )

    rgb = np.empty(
        drr8.shape + (3,),
        dtype=np.uint8,
    )

    # Red PXR + green DRR.
    rgb[..., 0] = pxr8
    rgb[..., 1] = drr8
    rgb[..., 2] = 0

    return rgb


def save_registration_overlay(path, drr):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rgb = make_registration_overlay(drr)

    tiff.imwrite(
        path,
        rgb,
        photometric="rgb",
    )


# =============================================================================
# OPTIONAL ALIGNMENT GRID
# =============================================================================

grid_enabled = False


def make_alignment_grid(imp):
    """
    Create a non-destructive ImageJ overlay grid.

    The same fractional coordinates are used for PXR and DRR, so a line at
    25%, 50%, or 75% refers to the same relative detector location in both.
    """
    overlay = Overlay()

    width = int(imp.getWidth())
    height = int(imp.getHeight())

    if width <= 1 or height <= 1:
        return overlay

    for fraction in GRID_FRACTIONS:
        fraction = float(fraction)

        x = fraction * float(width - 1)
        y = fraction * float(height - 1)

        vertical = Line(
            x,
            0.0,
            x,
            float(height - 1),
        )
        horizontal = Line(
            0.0,
            y,
            float(width - 1),
            y,
        )

        # Cyan is visible on both the PXR and DRR without looking like
        # actual image content.
        vertical.setStrokeColor(Color.CYAN)
        horizontal.setStrokeColor(Color.CYAN)

        line_width = (
            GRID_CENTER_LINE_WIDTH
            if abs(fraction - 0.5) < 1e-9
            else GRID_LINE_WIDTH
        )

        vertical.setStrokeWidth(float(line_width))
        horizontal.setStrokeWidth(float(line_width))

        overlay.add(vertical)
        overlay.add(horizontal)

    return overlay


def apply_grid_overlay(imp):
    """
    Apply/remove the grid from one ImagePlus without touching its pixels.
    """
    if imp is None:
        return

    if grid_enabled:
        imp.setOverlay(
            make_alignment_grid(imp)
        )
    else:
        imp.setOverlay(None)

    imp.updateAndDraw()


def toggle_grid():
    """
    Toggle the grid simultaneously on PXR and the currently displayed DRR.
    """
    global grid_enabled

    grid_enabled = not grid_enabled

    apply_grid_overlay(pxr)

    if preview_imp is not None:
        apply_grid_overlay(preview_imp)

    if overlay_imp is not None:
        apply_grid_overlay(overlay_imp)

    print(
        "Alignment grid: "
        + ("ON" if grid_enabled else "OFF")
    )


# =============================================================================
# 3D VIEWER
# =============================================================================

univ = Image3DUniverse()
univ.show()

try:
    set_window_bounds(
        univ.getWindow(),
        VIEWER_WINDOW_BOUNDS,
        "ImageJ 3D Viewer",
    )
except Exception as exc:
    print(
        "Could not position ImageJ 3D Viewer: "
        f"{type(exc).__name__}: {exc}"
    )

content = univ.addVoltex(ct)

if content is None:
    raise RuntimeError(
        "Could not add CT to Fiji 3D Viewer"
    )

content.setThreshold(1)
content.setTransparency(0.0)

try:
    content.setLocked(False)
except Exception:
    pass

univ.select(content)


# =============================================================================
# MANUAL MOVEMENT
# =============================================================================
#
# IMPORTANT:
# We keep the registration pose ourselves instead of trying to recover it
# from Fiji's internal localRotate/localTranslate decomposition.
#
# pose_R_xyz is the true object rotation.
# pose_t_xyz is the true manual translation in mm.
#
# The Fiji display transform is rebuilt from these values after every key:
#
#     p_out = R @ p + (u + c - R @ c)
#
# so rotation is always about the CURRENT CT center and does NOT create
# hundreds of millimetres of fake translation.
#

pose_lock = threading.Lock()

# PHYSICAL registration pose.
# This is what goes into the DRR and the saved transform.
pose_R_xyz = np.eye(3, dtype=np.float64)
pose_t_xyz = np.zeros(3, dtype=np.float64)

# BEAM_AXIS=0 in NumPy ZYX corresponds to Fiji Z.
# Front/back motion is PHYSICAL again:
#   Fiji Z translation is sent to the DRR and saved in the final transform.
#
# With the current geometry:
#   effective SPD = SPD_MM + pose_t_xyz[2]
#
# Positive Z moves the CT farther from the source.
# Negative Z moves the CT closer to the source.

viewer_center_xyz = np.array(
    [
        ct.getWidth() * float(cal.pixelWidth) / 2.0,
        ct.getHeight() * float(cal.pixelHeight) / 2.0,
        ct.getNSlices() * float(cal.pixelDepth) / 2.0,
    ],
    dtype=np.float64,
)


def sync_pose_from_fiji():
    """
    Synchronize the live Fiji object pose with our tracked registration pose.

    Mouse rotations are PHYSICAL.
    Mouse X/Y/Z translations are PHYSICAL.

    For BEAM_AXIS=0, Fiji Z is the beam-depth direction, so front/back
    translation changes the effective source-to-object-centre distance and
    therefore the DRR magnification.

    The full XYZ translation is sent to the DRR and saved in the transform.
    """
    global pose_R_xyz, pose_t_xyz

    local_translate = Transform3D()
    local_rotate = Transform3D()

    content.getLocalTranslate(local_translate)
    content.getLocalRotate(local_rotate)

    T_translate = transform3d_to_numpy(
        local_translate,
        Matrix4d,
    )
    T_rotate = transform3d_to_numpy(
        local_rotate,
        Matrix4d,
    )

    exact_total = T_translate @ T_rotate

    R = exact_total[:3, :3].copy()
    t_total = exact_total[:3, 3].copy()

    # Remove tiny numerical drift.
    u_svd, _, vh_svd = np.linalg.svd(R)
    R = u_svd @ vh_svd

    # Centered live object translation in Fiji XYZ.
    live_u_xyz = (
        R @ viewer_center_xyz
        + t_total
        - viewer_center_xyz
    )

    with pose_lock:
        pose_R_xyz = R
        pose_t_xyz[:] = live_u_xyz

        return (
            pose_R_xyz.copy(),
            pose_t_xyz.copy(),
        )


def current_pose(sync_from_fiji=True):
    if sync_from_fiji:
        return sync_pose_from_fiji()

    with pose_lock:
        return (
            pose_R_xyz.copy(),
            pose_t_xyz.copy(),
        )


def effective_spd_mm(translation_xyz=None):
    if translation_xyz is None:
        with pose_lock:
            translation_xyz = pose_t_xyz.copy()

    # Fiji Z maps to NumPy beam axis 0 in this project.
    return float(SPD_MM + float(translation_xyz[2]))



def update_fiji_from_pose():
    """
    Display exactly the same physical registration pose used by the DRR.
    """
    with pose_lock:
        R = pose_R_xyz.copy()
        u_display = pose_t_xyz.copy()

    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3] = (
        u_display
        + viewer_center_xyz
        - R @ viewer_center_xyz
    )

    transform = numpy_to_transform3d(
        M,
        Transform3D,
        Matrix4d,
    )

    content.setTransform(transform)
    univ.fireTransformationUpdated()


def apply_translation(dx, dy, dz):
    global pose_t_xyz

    # First absorb any mouse changes.
    sync_pose_from_fiji()

    with pose_lock:
        pose_t_xyz[0] += float(dx)
        pose_t_xyz[1] += float(dy)
        pose_t_xyz[2] += float(dz)

    update_fiji_from_pose()

    # X/Y/Z are all physical now, so every translation changes the DRR.
    if (
        abs(float(dx)) > 0.0
        or abs(float(dy)) > 0.0
        or abs(float(dz)) > 0.0
    ):
        preview_event.set()


def apply_rotation(axis, angle_rad):
    global pose_R_xyz

    # Pull in any mouse movement/rotation that happened since the last keypress.
    sync_pose_from_fiji()

    R_inc = rotation_matrix_xyz(
        axis,
        angle_rad,
    )

    with pose_lock:
        # World/extrinsic X,Y,Z rotations, which behave like
        # pitch/yaw/roll controls in the fixed viewer coordinate system.
        pose_R_xyz = R_inc @ pose_R_xyz

        # Remove tiny floating point drift.
        u, _, vh = np.linalg.svd(pose_R_xyz)
        pose_R_xyz = u @ vh

    update_fiji_from_pose()

    # Every physical rotation changes the projection.
    preview_event.set()


def choose_starting_transform():
    """Ask whether to resume, without silently choosing the wrong specimen pose."""
    mode = START_POSE_MODE.strip().lower()
    if mode not in {"ask", "resume", "new"}:
        raise ValueError('START_POSE_MODE must be "ask", "resume", or "new"')
    if mode == "new":
        return None

    if RESUME_TRANSFORM_PATH is not None:
        candidates = [Path(RESUME_TRANSFORM_PATH)]
    else:
        candidates = list(dict.fromkeys([
            TRANSFORM_PATH,
            OUT_DIR / f"{SPECIMEN}_transform.txt",
        ]))

    existing = [p for p in candidates if p.is_file()]
    if not existing:
        if mode == "resume":
            raise FileNotFoundError(
                "Resume requested, but no transform exists at: "
                + ", ".join(str(p) for p in candidates)
            )
        print(f"No existing transform for {SPECIMEN}; starting from identity.")
        return None

    if mode == "ask":
        print("\nExisting registration transform(s):")
        for i, path in enumerate(existing, 1):
            print(f"  {i}. {path}")
        while True:
            answer = input("Resume registration from a saved transform? [y/n]: ").strip().lower()
            if answer in {"n", "no"}:
                print("Starting a new registration from identity.")
                return None
            if answer in {"y", "yes"}:
                break
            print("Please enter y or n.")

    if len(existing) == 1:
        return existing[0]

    while True:
        answer = input(f"Which transform to load? [1-{len(existing)}]: ").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(existing):
            return existing[int(answer) - 1]
        print("Enter the number of the desired transform.")


def load_existing_registration(path):
    """Recover the centered pose from the actual saved Fiji 4x4 matrix."""
    global pose_R_xyz, pose_t_xyz

    saved = np.loadtxt(path, dtype=np.float64)
    if saved.shape != (4, 4) or not np.isfinite(saved).all():
        raise ValueError(f"Invalid 4x4 transform: {path}")
    if not np.allclose(saved[3], [0., 0., 0., 1.], atol=1e-6):
        raise ValueError(f"Invalid homogeneous matrix last row: {path}")

    R = saved[:3, :3].copy()
    if (not np.allclose(R.T @ R, np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(R), 1.0, atol=1e-4)):
        raise ValueError(f"Transform rotation is not a proper rigid rotation: {path}")

    sz, sy, sx = map(float, spacing_zyx)
    nz, ny, nx = map(int, volume.shape)
    volume_center_xyz = np.array(
        [nx * sx / 2., ny * sy / 2., nz * sz / 2.], dtype=np.float64,
    )
    if not np.allclose(volume_center_xyz, viewer_center_xyz, atol=1e-3, rtol=0):
        raise ValueError(
            "Fiji CT center differs from renderer CT center. Check CT shape/"
            "voxel-spacing metadata before restoring this transform."
        )

    # Saved matrix: p_out = R @ p + t_total.
    # Manual pose:  p_out = R @ (p-center) + center + true_translation.
    true_translation = R @ volume_center_xyz + saved[:3, 3] - volume_center_xyz
    if SPD_MM + true_translation[2] <= 0:
        raise ValueError("Saved Z translation places the CT at/behind the source")

    # A sidecar written by the original manual script records initial SPD.
    # Effective SPD already includes the physical Z translation: do not add it again.
    if REGISTRATION_INFO_PATH.exists():
        for line in REGISTRATION_INFO_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("initial_spd_mm") and "=" in line:
                previous_spd = float(line.split("=", 1)[1].strip())
                if not np.isclose(previous_spd, SPD_MM, atol=1e-6):
                    raise ValueError(
                        f"Registration used initial SPD={previous_spd:g} mm, "
                        f"but script uses SPD_MM={SPD_MM:g} mm. "
                        "Use the original initial SPD when resuming."
                    )
                break

    with pose_lock:
        pose_R_xyz = R
        pose_t_xyz = true_translation.copy()

    update_fiji_from_pose()

    # Verify that the pose recovered from Fiji after installing the matrix is
    # the pose we loaded, before starting the first DRR or accepting movements.
    restored_R, restored_t = sync_pose_from_fiji()
    if (not np.allclose(restored_R, R, rtol=0, atol=1e-5)
            or not np.allclose(restored_t, true_translation, rtol=0, atol=1e-3)):
        raise RuntimeError(
            "Fiji did not retain the loaded registration pose. "
            "Stopping rather than rendering/saving an incorrect transform."
        )

    print(f"Resumed {SPECIMEN} from: {path}")
    print("Restored physical XYZ translation (mm):", np.round(restored_t, 3))
    print(f"Effective SPD: {effective_spd_mm(restored_t):.3f} mm "
          f"(initial SPD={SPD_MM:g} mm)")
    if path != TRANSFORM_PATH:
        print(f"Press S to save the updated pose to: {TRANSFORM_PATH}")


selected_transform = choose_starting_transform()
if selected_transform is None:
    # Original behavior: begin with the identity pose.
    update_fiji_from_pose()
else:
    load_existing_registration(selected_transform)


# =============================================================================
# EVENTS
# =============================================================================

preview_event = threading.Event()
save_event = threading.Event()
quit_event = threading.Event()
rendering_event = threading.Event()

rotation_step_rad = (
    ROTATION_STEP_DEG
    * np.pi
    / 180.0
)


# =============================================================================
# KEYBOARD
# =============================================================================

def key_pressed(event):
    ch = str(
        event.getKeyChar()
    ).lower()

    key_code = int(
        event.getKeyCode()
    )

    step = float(
        TRANSLATION_STEP_MM
    )

    angle = float(
        rotation_step_rad
    )

    # Translation
    if ch == "4":
        apply_translation(-step, 0.0, 0.0)
        event.consume()
        return

    if ch == "6":
        apply_translation(+step, 0.0, 0.0)
        event.consume()
        return

    if ch == "8":
        apply_translation(0.0, -step, 0.0)
        event.consume()
        return

    if ch == "2":
        apply_translation(0.0, +step, 0.0)
        event.consume()
        return

    if ch == "0":
        apply_translation(0.0, 0.0, +step)
        event.consume()
        return

    if ch == "5":
        apply_translation(0.0, 0.0, -step)
        event.consume()
        return

    # Rotation
    #
    # 7/9 = pitch around X
    # O/P = yaw   around Y
    # 1/3 = roll  around Z
    if ch == "7":
        apply_rotation("x", +angle)
        event.consume()
        return

    if ch == "9":
        apply_rotation("x", -angle)
        event.consume()
        return

    if key_code == KeyEvent.VK_O or ch == "o":
        apply_rotation("y", +angle)
        event.consume()
        return

    if key_code == KeyEvent.VK_P or ch == "p":
        apply_rotation("y", -angle)
        event.consume()
        return

    if ch == "1":
        apply_rotation("z", -angle)
        event.consume()
        return

    if ch == "3":
        apply_rotation("z", +angle)
        event.consume()
        return

    # Toggle alignment grid on PXR + DRR
    if key_code == KeyEvent.VK_G or ch == "g":
        toggle_grid()
        event.consume()
        return

    # Preview
    if key_code == KeyEvent.VK_ENTER:
        preview_event.set()
        event.consume()
        return

    # Final save
    if key_code == KeyEvent.VK_S or ch == "s":
        save_event.set()
        event.consume()
        return

    # Quit
    if key_code == KeyEvent.VK_Q or ch == "q":
        quit_event.set()
        event.consume()
        return


def key_released(event):
    pass


def key_typed(event):
    pass


listener = jpype.JProxy(
    "java.awt.event.KeyListener",
    dict(
        keyPressed=key_pressed,
        keyReleased=key_released,
        keyTyped=key_typed,
    ),
)

canvas = univ.getCanvas()
canvas.addKeyListener(listener)

# Let the same shortcuts, especially G, work when the PXR window has focus.
try:
    pxr.getCanvas().addKeyListener(listener)
except Exception:
    pass

try:
    canvas.setFocusable(True)
    canvas.requestFocusInWindow()
except Exception:
    pass


# =============================================================================
# LIVE DRR + FUSED DISPLAY
# =============================================================================

preview_imp = None
overlay_imp = None


def _show_or_update_imageplus(
    current_imp,
    path,
    title,
    bounds,
    apply_grid=True,
):
    """
    Keep one ImageJ window alive and replace only its pixel processor.

    This avoids the old close/re-open behaviour, so window position, focus,
    and zoom remain stable while the registration is adjusted.
    """
    fresh = IJ.openImage(str(path))

    if fresh is None:
        raise RuntimeError(
            f"Fiji could not open rendered image: {path}"
        )

    if current_imp is None:
        current_imp = fresh
        current_imp.setTitle(title)
        current_imp.resetDisplayRange()
        current_imp.show()

        set_window_bounds(
            current_imp.getWindow(),
            bounds,
            title,
        )

        fit_image_to_window(
            current_imp
        )

        try:
            current_imp.getCanvas().addKeyListener(
                listener
            )
        except Exception:
            pass

    else:
        # Duplicate the processor so it remains valid after the temporary
        # ImagePlus is closed.
        processor = fresh.getProcessor().duplicate()

        current_imp.setProcessor(
            title,
            processor,
        )

        try:
            current_imp.setCalibration(
                fresh.getCalibration().copy()
            )
        except Exception:
            pass

        current_imp.resetDisplayRange()
        current_imp.updateAndDraw()

        fresh.changes = False
        try:
            fresh.close()
        except Exception:
            pass

    if apply_grid:
        apply_grid_overlay(
            current_imp
        )

    return current_imp


def show_or_update_preview(path, title="DRR_LIVE"):
    global preview_imp

    preview_imp = _show_or_update_imageplus(
        preview_imp,
        path,
        title,
        DRR_WINDOW_BOUNDS,
        apply_grid=True,
    )


def show_or_update_overlay(path, title="PXR_RED__DRR_GREEN"):
    global overlay_imp

    overlay_imp = _show_or_update_imageplus(
        overlay_imp,
        path,
        title,
        OVERLAY_WINDOW_BOUNDS,
        apply_grid=True,
    )


# =============================================================================
# INSTRUCTIONS
# =============================================================================

print(
    "DRR backend: " + (
        "CuPy GPU"
        if CUPY_AVAILABLE
        else "SciPy CPU fallback for 512px previews"
    )
)

print(
    f"""
============================================================
MANUAL CT -> PXR REGISTRATION
============================================================

Translation
  4 / 6   X - / +
  8 / 2   Y - / +
  0 / 5   Z front/back + / -        [physical]

Rotation
  7 / 9   pitch -/+
  O / P   yaw   -/+
  1 / 3   roll  -/+

ENTER   force an immediate DRR refresh
G       toggle alignment grid on PXR + DRR + fused view
S       save transform + render final DRR [quantitative fixed-scale]
Q       quit

DRR preview updates automatically after every physical keyboard move
and after mouse rotations/translations in the 3D Viewer.

The DRR window stays open; only its pixels are updated.
A second live window fuses PXR=red with DRR=green.
Its red/green channel brightness is controlled by:
  OVERLAY_PXR_GAIN
  OVERLAY_DRR_GAIN

Alignment grid:
  G toggles cyan grid lines at 25%, 50%, 75%
  on the PXR, live DRR, and fused registration view.
  Grid is overlay-only and is never saved into image pixels.

Desktop layout:
  PXR -> top-left
  live DRR -> top-middle
  PXR/DRR fused view -> bottom-left
  3D Viewer -> right

All 2D images are automatically fit to their windows so the complete
image is visible instead of opening overly zoomed-in.

This run starts from the selected saved transform (or identity if new).
Mouse edits and keyboard edits are synchronized.
Rotations are about the CT center, so they no longer create fake translation.

Depth handling:
  mouse front/back shift -> PHYSICAL
  keyboard 0 / 5 shift  -> PHYSICAL

Initial SPD = 800 mm.

For BEAM_AXIS=0:
  effective SPD = 800 mm + Fiji Z translation

Positive Z moves the CT farther from the source and reduces magnification.
Negative Z moves the CT closer to the source and increases magnification.

The Z translation is included in:
  DRR
  effective SPD / magnification
  saved transform
============================================================
"""
)


# =============================================================================
# MAIN LOOP
# =============================================================================

preview_count = 0

last_rendered_R = None
last_rendered_t = None


def physical_pose_changed(R, t):
    if last_rendered_R is None or last_rendered_t is None:
        return True

    rotation_changed = not np.allclose(
        R,
        last_rendered_R,
        rtol=0.0,
        atol=POSE_ROT_ATOL,
    )

    translation_changed = not np.allclose(
        t,
        last_rendered_t,
        rtol=0.0,
        atol=POSE_TRANS_ATOL_MM,
    )

    return (
        rotation_changed
        or translation_changed
    )


# Render the chosen initial pose immediately (saved transform or identity).
# ENTER is not required to create the first DRR window.
preview_event.set()

while not quit_event.is_set():

    # Poll the live Fiji object so mouse rotations and all XYZ translations
    # trigger the DRR automatically. Z/front-back is physical again.
    if (
        not preview_event.is_set()
        and not save_event.is_set()
        and not rendering_event.is_set()
    ):
        try:
            live_R, live_t = current_pose(
                sync_from_fiji=True
            )

            if physical_pose_changed(
                live_R,
                live_t,
            ):
                preview_event.set()

        except Exception:
            pass

    # -------------------------------------------------------------------------
    # PREVIEW
    # -------------------------------------------------------------------------

    if preview_event.is_set():
        preview_event.clear()
        rendering_event.set()

        try:
            preview_count += 1

            rotation_xyz, true_translation_xyz = current_pose(
                sync_from_fiji=True
            )

            print(
                f"Rendering DRR preview #{preview_count}..."
            )
            print(
                "Physical translation XYZ mm:",
                np.array2string(
                    true_translation_xyz,
                    precision=3,
                    suppress_small=True,
                ),
            )
            print(
                f"Beam-depth translation Z: "
                f"{true_translation_xyz[2]:+.3f} mm"
            )
            print(
                f"Effective SPD: "
                f"{effective_spd_mm(true_translation_xyz):.3f} mm "
                f"(initial SPD={SPD_MM:.1f} mm)"
            )

            if np.max(np.abs(true_translation_xyz)) > 300.0:
                print(
                    "WARNING: translation exceeds 300 mm; "
                    "the CT may be outside the detector field."
                )
            print(
                f"Rotation det={np.linalg.det(rotation_xyz):.6f}"
            )

            t0 = time.time()

            drr = render_preview(
                preview_volume,
                preview_spacing,
                rotation_xyz,
                true_translation_xyz,
                pxr_shape,
            )

            drr_min = float(np.nanmin(drr))
            drr_max = float(np.nanmax(drr))
            drr_nonzero = int(np.count_nonzero(drr))

            print(
                f"DRR values: min={drr_min:.6g}, "
                f"max={drr_max:.6g}, nonzero={drr_nonzero}"
            )

            if drr_nonzero == 0 or drr_max <= 0:
                raise RuntimeError(
                    "DRR is all zero. No preview file was written."
                )

            save_preview_display(
                PREVIEW_PATH,
                drr,
                detector_pixel_spacing(PREVIEW_SIZE),
            )

            show_or_update_preview(
                PREVIEW_PATH,
                "DRR_LIVE",
            )

            save_registration_overlay(
                OVERLAY_PATH,
                drr,
            )

            show_or_update_overlay(
                OVERLAY_PATH,
                "PXR_RED__DRR_GREEN",
            )

            last_rendered_R = rotation_xyz.copy()
            last_rendered_t = true_translation_xyz.copy()

            print(
                f"Preview displayed ({time.time() - t0:.1f} s). "
                "Compare with PXR_REFERENCE."
            )

        except Exception as exc:
            print(
                f"DRR preview failed: {type(exc).__name__}: {exc}"
            )

        finally:
            rendering_event.clear()

        continue

    # -------------------------------------------------------------------------
    # FINAL
    # -------------------------------------------------------------------------

    if save_event.is_set():
        save_event.clear()
        rendering_event.set()

        try:
            rotation_xyz, true_translation_xyz = current_pose(
                sync_from_fiji=True
            )

            final_transform = renderer_matrix_from_fiji_pose(
                rotation_xyz,
                true_translation_xyz,
                volume.shape,
                spacing_zyx,
            )

            np.savetxt(
                TRANSFORM_PATH,
                final_transform,
                fmt="%.12f",
            )

            print(
                f"Saved transform: {TRANSFORM_PATH}"
            )

            effective_spd = effective_spd_mm(
                true_translation_xyz
            )

            with open(
                REGISTRATION_INFO_PATH,
                "w",
                encoding="utf-8",
            ) as f:
                f.write(
                    f"specimen = {SPECIMEN}\n"
                )
                f.write(
                    f"translation_x_mm = "
                    f"{true_translation_xyz[0]:.6f}\n"
                )
                f.write(
                    f"translation_y_mm = "
                    f"{true_translation_xyz[1]:.6f}\n"
                )
                f.write(
                    f"translation_z_mm = "
                    f"{true_translation_xyz[2]:.6f}\n"
                )
                f.write(
                    f"initial_spd_mm = "
                    f"{SPD_MM:.6f}\n"
                )
                f.write(
                    f"effective_spd_mm = "
                    f"{effective_spd:.6f}\n"
                )

            print(
                f"Saved registration info: "
                f"{REGISTRATION_INFO_PATH}"
            )
            print(
                f"Physical translation XYZ: "
                f"({true_translation_xyz[0]:+.3f}, "
                f"{true_translation_xyz[1]:+.3f}, "
                f"{true_translation_xyz[2]:+.3f}) mm"
            )
            print(
                f"Effective SPD: "
                f"{effective_spd:.3f} mm "
                f"(initial SPD={SPD_MM:.1f} mm)"
            )

            if CUPY_AVAILABLE:
                print(
                    "Rendering final DRR..."
                )

                drr = render_final(
                    volume,
                    spacing_zyx,
                    rotation_xyz,
                    true_translation_xyz,
                    pxr_shape,
                )

                # IMPORTANT:
                # Use the shared calibrate_drr.py saver here.
                # This keeps the final manual-registration DRR numerically
                # consistent with a DRR generated later by calibrate_drr.py.
                save_drr_16bit(
                    FINAL_DRR_PATH,
                    drr,
                    detector_pixel_spacing(FINAL_SIZE),
                )

                show_or_update_preview(
                    FINAL_DRR_PATH,
                    "DRR_FINAL",
                )

                save_registration_overlay(
                    OVERLAY_PATH,
                    drr,
                )

                show_or_update_overlay(
                    OVERLAY_PATH,
                    "PXR_RED__DRR_GREEN",
                )

                last_rendered_R = rotation_xyz.copy()
                last_rendered_t = true_translation_xyz.copy()

                print(
                    f"Saved final DRR: {FINAL_DRR_PATH}"
                )
            else:
                print(
                    "CuPy is not installed, so the 3072px final DRR was "
                    "not rendered here. The final transform was saved."
                )

        except Exception as exc:
            print(
                f"Final save failed: {type(exc).__name__}: {exc}"
            )

        finally:
            rendering_event.clear()

        continue

    time.sleep(LIVE_PREVIEW_POLL_SEC)


# =============================================================================
# CLEANUP
# =============================================================================

try:
    canvas.removeKeyListener(listener)
except Exception:
    pass

try:
    pxr.getCanvas().removeKeyListener(listener)
except Exception:
    pass

try:
    if preview_imp is not None:
        preview_imp.getCanvas().removeKeyListener(listener)
except Exception:
    pass

try:
    if overlay_imp is not None:
        overlay_imp.getCanvas().removeKeyListener(listener)
except Exception:
    pass

try:
    univ.close()
except Exception:
    pass

try:
    if preview_imp is not None:
        preview_imp.changes = False
        preview_imp.close()
except Exception:
    pass

try:
    if overlay_imp is not None:
        overlay_imp.changes = False
        overlay_imp.close()
except Exception:
    pass

try:
    pxr.changes = False
    pxr.close()
except Exception:
    pass

try:
    ct.changes = False
    ct.close()
except Exception:
    pass

print("Closed.")