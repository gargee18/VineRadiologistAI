import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import rotate, sobel
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider
from matplotlib.colors import LinearSegmentedColormap

# ── Load volumes ──────────────────────────────────────────────────────────────

main_path = "/mnt/41d6c007-0c9e-41e2-b2eb-8d9c032e9e53/gargee/2DProjection/"
dataset = "Dataset_Vitimage2019/"
specimen = "CEP011_AS1/"

volume   = tiff.imread(main_path+dataset+specimen+"CT/registered.tif").astype(np.float32)
volume   = volume / 1000.0
whiterot = tiff.imread(main_path+dataset+specimen+"/SEG/segmentation_AMADOU.tif").astype(np.float32)
necrosis = tiff.imread(main_path+dataset+specimen+"/SEG/segmentation_NECROSE.tif").astype(np.float32)
healthy  = tiff.imread(main_path+dataset+specimen+"/SEG/segmentation_SAIN.tif").astype(np.float32)

print("Volume shape:  ", volume.shape)
print("White rot:     ", whiterot.shape)
print("Necrosis:      ", necrosis.shape)
print("Healthy:       ", healthy.shape)

# ── Projection functions ──────────────────────────────────────────────────────

def generate_drr(vol, attenuation_scale=0.015, axis=1):
    return 1.0 - np.exp(-np.sum(vol * attenuation_scale, axis=axis))

def depth_map(vol, axis=1):
    coords  = np.arange(vol.shape[axis]).reshape([-1 if i == axis else 1 for i in range(vol.ndim)])
    weights = vol.astype(np.float32)
    total   = np.sum(weights, axis=axis)
    with np.errstate(invalid='ignore'):
        return np.where(total > 0, np.sum(weights * coords, axis=axis) / total, np.nan)

def thickness_map(vol, axis=1):
    result = np.sum(vol > 0, axis=axis).astype(np.float32)
    return np.where(result > 0, result, np.nan)

def silhouette_map(vol, axis=1):
    result = (np.sum(vol > 0, axis=axis) > 0).astype(np.float32)
    return np.where(result > 0, result, np.nan)

def contour_map(vol, axis=1):
    silhouette = (np.sum(vol > 0, axis=axis) > 0).astype(np.float32)
    edges = np.hypot(sobel(silhouette, axis=0), sobel(silhouette, axis=1))
    return np.where(edges > 0, edges, np.nan)

def rotated_vol(vol, angle):
    return rotate(vol, angle=angle, axes=(1, 2), reshape=False, order=1)

# ── Precompute rotations and projections ──────────────────────────────────────

angles   = [0, 30, 60, 90]
map_fns  = [depth_map, thickness_map, silhouette_map, contour_map]
map_names = ["Depth\n(wt. avg position along ray)",
             "Thickness\n(voxel count along ray)",
             "Silhouette\n(filled binary mask)",
             "Contour\n(silhouette boundary)"]
cmaps    = ['rainbow', 'magma', 'bone', 'autumn']

ct_rots  = [rotated_vol(volume, a) for a in angles]
drr_projs = [generate_drr(ct, axis=1) for ct in ct_rots]

def compute_mask_projs(seg_vol):
    rots = [rotated_vol(seg_vol, a) for a in angles]
    return [[fn(r, axis=1) for r in rots] for fn in map_fns]

wr_mask_projs = compute_mask_projs(whiterot)
nc_mask_projs = compute_mask_projs(necrosis)
hl_mask_projs = compute_mask_projs(healthy)

# ── Plotting function ─────────────────────────────────────────────────────────

def plot_projections(mask_projs, title):
    n_rows = len(map_fns)
    n_cols = len(angles)
    fig = plt.figure(figsize=(24, 24))
    gs  = GridSpec(n_rows, n_cols + 1, figure=fig,
                   width_ratios=[1] * n_cols + [0.05],
                   hspace=0.15, wspace=0.25)

    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(n_cols)] for r in range(n_rows)])
    cax0 = fig.add_subplot(gs[0, n_cols])
    cax1 = fig.add_subplot(gs[1, n_cols])

    row_ims = {}

    for row, (map_name, projs, cmap) in enumerate(zip(map_names, mask_projs, cmaps)):
        for col, (angle, drr, mask) in enumerate(zip(angles, drr_projs, projs)):
            ax = axes[row, col]
            ax.imshow(drr, cmap='gray')
            im = ax.imshow(np.ma.masked_invalid(mask), cmap=cmap, alpha=0.5)

            if row == 0:
                ax.set_title(f"{angle}°", fontsize=12, fontweight='bold', pad=6)

            if col == len(angles) - 1 and row in (0, 1):
                row_ims[row] = im

        # Place row label using fig.text — positioned to the left of the first column
        pos = axes[row, 0].get_position()
        fig.text(pos.x0 - 0.02, pos.y0 + pos.height / 2, map_name,
                 ha='right', va='center', fontsize=10, fontweight='bold',
                 rotation=90, multialignment='center')

    cbar0 = fig.colorbar(row_ims[0], cax=cax0)
    cbar0.set_label("depth (voxels)", fontsize=9)
    cbar0.ax.tick_params(labelsize=8)

    cbar1 = fig.colorbar(row_ims[1], cax=cax1)
    cbar1.set_label("thickness (voxels)", fontsize=9)
    cbar1.ax.tick_params(labelsize=8)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.99)
# ── Generate all three figures ────────────────────────────────────────────────

plot_projections(wr_mask_projs, "White Rot Mask Projections Overlaid on DRR")
plot_projections(nc_mask_projs, "Necrosis Mask Projections Overlaid on DRR")
plot_projections(hl_mask_projs, "Healthy Mask Projections Overlaid on DRR")

def trim_cmap(cmap_name, minval=0.3, maxval=1.0):
    """Cut off the light/white end of a colormap."""
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(minval, maxval, 256))
    return LinearSegmentedColormap.from_list(f'{cmap_name}_trimmed', colors)


def plot_all_tissues_merged(wr_projs, nc_projs, hl_projs, title="All Tissues Merged"):
    n_rows = len(map_fns)
    n_cols = len(angles)
    fig = plt.figure(figsize=(24, 26))
    gs   = GridSpec(n_rows, n_cols, figure=fig, hspace=0.15, wspace=0.1, top=0.92, bottom=0.05, left=0.08, right=0.80)
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(n_cols)] for r in range(n_rows)])

    tissue_cmaps  = [trim_cmap('Reds'), trim_cmap('Blues'), trim_cmap('Greens')]
    tissue_colors = [np.array([1, 0, 0]), np.array([0, 0.4, 1]), np.array([0, 0.7, 0])]
    binary_rows   = {2, 3}
    alphas = [0.6, 0.6, 0.6]

    overlays_ims = {t: [] for t in range(3)}

    for row, map_name in enumerate(map_names):
        for col, (angle, drr) in enumerate(zip(angles, drr_projs)):
            ax = axes[row, col]
            ax.imshow(drr, cmap='gray')

            for t, (projs, cmap, color) in enumerate(zip([wr_projs, nc_projs, hl_projs], tissue_cmaps, tissue_colors)):
                mask = projs[row][col]
                if row in binary_rows:
                    valid = np.isfinite(mask)
                    rgba  = np.zeros((*mask.shape, 4), dtype=np.float32)
                    rgba[valid, :3] = color
                    rgba[valid,  3] = alphas[t]
                    im = ax.imshow(rgba)
                else:
                    im = ax.imshow(np.ma.masked_invalid(mask), cmap=cmap, alpha=0.9)
                overlays_ims[t].append((im, mask, row, color))

            ax.axis("off")
            if row == 0:
                ax.set_title(f"{angle}°", fontsize=12, fontweight='bold', pad=6)

        pos = axes[row, 0].get_position()
        fig.text(pos.x0 - 0.02, pos.y0 + pos.height / 2, map_name,
                 ha='right', va='center', fontsize=10, fontweight='bold',
                 rotation=90, multialignment='center')

    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.97)

    # Vertical sliders on the right — evenly spaced, well separated
    slider_height = 0.18
    slider_gap    = 0.08
    total_block   = 3 * slider_height + 2 * slider_gap   # ~0.7
    start_y       = 0.5 + total_block / 2                 # centre vertically

    slider_specs = [
        ('White Rot', 'red'),
        ('Necrosis',  'royalblue'),
        ('Healthy',   'limegreen'),
    ]
    sliders = []
    for i, (label, color) in enumerate(slider_specs):
        bottom = start_y - i * (slider_height + slider_gap) - slider_height
        ax_sl  = fig.add_axes([0.855, bottom, 0.02, slider_height])
        sl     = Slider(ax_sl, '', 0, 1, valinit=0.5, color=color, orientation='vertical')
        # Label above the slider
        fig.text(0.865, bottom + slider_height + 0.01, label,
                 ha='center', va='bottom', fontsize=10, fontweight='bold', color=color)
        sliders.append(sl)

    def update(val):
        for t, sl in enumerate(sliders):
            new_alpha = sl.val
            for im, mask, row, color in overlays_ims[t]:
                if row in binary_rows:
                    valid = np.isfinite(mask)
                    rgba  = np.zeros((*mask.shape, 4), dtype=np.float32)
                    rgba[valid, :3] = color
                    rgba[valid,  3] = new_alpha
                    im.set_array(rgba)
                else:
                    im.set_alpha(new_alpha)
        fig.canvas.draw_idle()

    for sl in sliders:
        sl.on_changed(update)
    return fig, sliders

fig, sliders = plot_all_tissues_merged(wr_mask_projs, nc_mask_projs, hl_mask_projs, "All Tissues Overlaid on DRR")
plt.show()