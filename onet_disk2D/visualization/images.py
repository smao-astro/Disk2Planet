import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray
from mpl_toolkits.axes_grid1 import ImageGrid


def get_sigma_pred_compare_images(
    pred: xarray.DataArray,
    s_truth: xarray.DataArray,
    zmin=None,
    zmax=None,
    emin=None,
    emax=None,
    figsize=(15, 5),
    aspect="auto",
    axes_pad=0.15,
    cmap="RdBu_r",
    error_cmap="RdBu_r",
    title_fontsize=28,
    axes_tick_fontsize=22,
    cbar_size="5%",
    cbar_pad="5%",  # "5%"
    cbar_tick_fontsize=22,
):
    zmin = np.nanmin([np.nanmin(pred), np.nanmin(s_truth)]) if zmin is None else zmin
    zmax = np.nanmax([np.nanmax(pred), np.nanmax(s_truth)]) if zmax is None else zmax
    norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=zmin, vmax=zmax)

    error = pred - s_truth
    emin = np.nanmin(error) if emin is None else emin
    emax = np.nanmax(error) if emax is None else emax
    error_norm = mcolors.TwoSlopeNorm(vmin=emin, vcenter=0, vmax=emax)

    return get_pred_compare_images(
        pred=pred,
        s_truth=s_truth,
        norm=norm,
        error_norm=error_norm,
        figsize=figsize,
        aspect=aspect,
        axes_pad=axes_pad,
        cmap=cmap,
        error_cmap=error_cmap,
        title_fontsize=title_fontsize,
        axes_tick_fontsize=axes_tick_fontsize,
        cbar_size=cbar_size,
        cbar_pad=cbar_pad,
        cbar_tick_fontsize=cbar_tick_fontsize,
    )


def get_pred_compare_images(
    pred: xarray.DataArray,
    s_truth: xarray.DataArray,
    norm,
    error_norm,
    cmap,
    error_cmap,
    figsize=(15, 5),
    aspect="auto",
    axes_pad=0.15,
    title_fontsize=28,
    axes_tick_fontsize=22,
    cbar_size="5%",
    cbar_pad="5%",  # "5%"
    cbar_tick_fontsize=22,
):
    error = pred - s_truth

    coord_keys = list(pred.coords.keys())

    fig = plt.figure(figsize=figsize)

    grid = ImageGrid(
        fig,
        111,  # similar to subplot(111)
        nrows_ncols=(1, 3),  # creates 1x3 grid of axes
        axes_pad=axes_pad,  # pad between axes in inch.
        share_all=True,
        cbar_location="bottom",
        cbar_mode="each",
        cbar_size=cbar_size,
        cbar_pad=cbar_pad,
    )

    for i, data in enumerate([pred, s_truth]):
        im = grid[i].pcolormesh(
            data.coords[coord_keys[1]].values,
            data.coords[coord_keys[0]].values,
            data,
            shading="auto",
            cmap=cmap,
            norm=norm,
        )
        grid.cbar_axes[i].colorbar(im)

    im = grid[2].pcolormesh(
        error.coords[coord_keys[1]].values,
        error.coords[coord_keys[0]].values,
        error,
        shading="auto",
        cmap=error_cmap,
        norm=error_norm,
    )
    grid.cbar_axes[2].colorbar(im)

    # set cbar tick fontsize
    for cax in grid.cbar_axes:
        cax.tick_params(labelsize=cbar_tick_fontsize)
    for ax, title in zip(grid, ["pred", "truth", "error"]):
        ax.set_title(title, fontsize=title_fontsize)
    for ax in grid:
        ax.grid(False)
        ax.set_aspect(aspect)
        ax.tick_params(axis="both", which="major", labelsize=axes_tick_fontsize)

    plt.tight_layout()

    return fig
