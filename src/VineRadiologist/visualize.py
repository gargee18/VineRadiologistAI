"""Plotting helpers for QA: mask projections overlaid on DRRs, with
interactive alpha sliders for multi-tissue overlays."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider
from matplotlib.colors import LinearSegmentedColormap

from .projection import depth_map, thickness_map, silhouette_map, contour_map, apply_yaw, generate_drr

MAP_FNS = [depth_map, thickness_map, silhouette_map, contour_map]
MAP_NAMES = [
    "Depth\n(wt. avg position along ray)",
    "Thickness\n(voxel count along ray)",
    "Silhouette\n(filled binary mask)",
    "Contour\n(silhouette boundary)",
]
CMAPS = ["rainbow", "magma", "bone", "autumn"]


def compute_mask_projs(seg_vol: np.ndarray, angles, axis: int = 1):
    rots = [apply_yaw(seg_vol, a) for a in angles]
    return [[fn(r, axis=axis) for r in rots] for fn in MAP_FNS]


def plot_projections(drr_projs, mask_projs, angles, title):
    n_rows, n_cols = len(MAP_FNS), len(angles)
    fig = plt.figure(figsize=(24, 24))
    gs = GridSpec(n_rows, n_cols + 1, figure=fig,
                  width_ratios=[1] * n_cols + [0.05], hspace=0.15, wspace=0.25)

    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(n_cols)] for r in range(n_rows)])
    cax0 = fig.add_subplot(gs[0, n_cols])
    cax1 = fig.add_subplot(gs[1, n_cols])
    row_ims = {}

    for row, (map_name, projs, cmap) in enumerate(zip(MAP_NAMES, mask_projs, CMAPS)):
        for col, (angle, drr, mask) in enumerate(zip(angles, drr_projs, projs)):
            ax = axes[row, col]
            ax.imshow(drr, cmap="gray")
            im = ax.imshow(np.ma.masked_invalid(mask), cmap=cmap, alpha=0.5)
            if row == 0:
                ax.set_title(f"{angle}\u00b0", fontsize=12, fontweight="bold", pad=6)
            if col == len(angles) - 1 and row in (0, 1):
                row_ims[row] = im

        pos = axes[row, 0].get_position()
        fig.text(pos.x0 - 0.02, pos.y0 + pos.height / 2, map_name,
                  ha="right", va="center", fontsize=10, fontweight="bold",
                  rotation=90, multialignment="center")

    cbar0 = fig.colorbar(row_ims[0], cax=cax0)
    cbar0.set_label("depth (voxels)", fontsize=9)
    cbar0.ax.tick_params(labelsize=8)
    cbar1 = fig.colorbar(row_ims[1], cax=cax1)
    cbar1.set_label("thickness (voxels)", fontsize=9)
    cbar1.ax.tick_params(labelsize=8)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.99)
    return fig


def _trim_cmap(cmap_name, minval=0.3, maxval=1.0):
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(minval, maxval, 256))
    return LinearSegmentedColormap.from_list(f"{cmap_name}_trimmed", colors)


def plot_all_tissues_merged(drr_projs, wr_projs, nc_projs, hl_projs, angles,
                             title="All Tissues Merged"):
    n_rows, n_cols = len(MAP_FNS), len(angles)
    fig = plt.figure(figsize=(24, 26))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.15, wspace=0.1,
                  top=0.92, bottom=0.05, left=0.08, right=0.80)
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(n_cols)] for r in range(n_rows)])

    tissue_cmaps = [_trim_cmap("Reds"), _trim_cmap("Blues"), _trim_cmap("Greens")]
    tissue_colors = [np.array([1, 0, 0]), np.array([0, 0.4, 1]), np.array([0, 0.7, 0])]
    binary_rows = {2, 3}
    alphas = [0.6, 0.6, 0.6]
    overlays_ims = {t: [] for t in range(3)}

    for row, map_name in enumerate(MAP_NAMES):
        for col, (angle, drr) in enumerate(zip(angles, drr_projs)):
            ax = axes[row, col]
            ax.imshow(drr, cmap="gray")
            for t, (projs, cmap, color) in enumerate(zip([wr_projs, nc_projs, hl_projs], tissue_cmaps, tissue_colors)):
                mask = projs[row][col]
                if row in binary_rows:
                    valid = np.isfinite(mask)
                    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
                    rgba[valid, :3] = color
                    rgba[valid, 3] = alphas[t]
                    im = ax.imshow(rgba)
                else:
                    im = ax.imshow(np.ma.masked_invalid(mask), cmap=cmap, alpha=0.9)
                overlays_ims[t].append((im, mask, row, color))
            ax.axis("off")
            if row == 0:
                ax.set_title(f"{angle}\u00b0", fontsize=12, fontweight="bold", pad=6)
        pos = axes[row, 0].get_position()
        fig.text(pos.x0 - 0.02, pos.y0 + pos.height / 2, map_name,
                  ha="right", va="center", fontsize=10, fontweight="bold",
                  rotation=90, multialignment="center")

    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.97)

    slider_height, slider_gap = 0.18, 0.08
    total_block = 3 * slider_height + 2 * slider_gap
    start_y = 0.5 + total_block / 2
    slider_specs = [("White Rot", "red"), ("Necrosis", "royalblue"), ("Healthy", "limegreen")]
    sliders = []
    for i, (label, color) in enumerate(slider_specs):
        bottom = start_y - i * (slider_height + slider_gap) - slider_height
        ax_sl = fig.add_axes([0.855, bottom, 0.02, slider_height])
        sl = Slider(ax_sl, "", 0, 1, valinit=0.5, color=color, orientation="vertical")
        fig.text(0.865, bottom + slider_height + 0.01, label,
                  ha="center", va="bottom", fontsize=10, fontweight="bold", color=color)
        sliders.append(sl)

    def update(val):
        for t, sl in enumerate(sliders):
            new_alpha = sl.val
            for im, mask, row, color in overlays_ims[t]:
                if row in binary_rows:
                    valid = np.isfinite(mask)
                    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
                    rgba[valid, :3] = color
                    rgba[valid, 3] = new_alpha
                    im.set_array(rgba)
                else:
                    im.set_alpha(new_alpha)
        fig.canvas.draw_idle()

    for sl in sliders:
        sl.on_changed(update)
    return fig, sliders
