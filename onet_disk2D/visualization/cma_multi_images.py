import typing

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from mpl_toolkits.axes_grid1 import ImageGrid


def draw_pred_from_list(
    pred_cart_ls: list[xr.DataArray],
    s_truth_cart_ls: list[xr.DataArray],
    r_p: float,
    norm,
    err_norm,
    column_titles: typing.Optional[list[str]] = None,
    row_titles: typing.Optional[list[str]] = None,
    cmap="RdBu_r",
    err_cmap="RdBu_r",
    figsize=(15, 8),
    axes_pad=0.15,
    cbar_size="7%",
    cbar_pad="2%",
    title_fontsize=28,
    axes_tick_fontsize=22,
    cbar_tick_fontsize=22,
):
    # Here is your plot drawing code:
    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(
        fig,
        rect=111,
        nrows_ncols=(3, len(pred_cart_ls)),
        share_all=True,
        axes_pad=axes_pad,
        cbar_location="right",
        cbar_mode="edge",
        cbar_size=cbar_size,
        cbar_pad=cbar_pad,
    )

    for i, (pred_cart, truth_cart) in enumerate(zip(pred_cart_ls, s_truth_cart_ls)):
        # convert the lines below to pcolormesh
        im1 = grid[i].pcolormesh(
            truth_cart.x.values,
            truth_cart.y.values,
            truth_cart,
            shading="auto",
            cmap=cmap,
            norm=norm,
        )
        grid[i].grid(False)
        if i == len(pred_cart_ls) - 1:
            grid.cbar_axes[0].colorbar(im1)
            # cbar font size
            grid.cbar_axes[0].tick_params(labelsize=cbar_tick_fontsize)

        # column titles
        if column_titles is not None:
            grid[i].set_title(column_titles[i], fontsize=title_fontsize)

        im2 = grid[i + len(pred_cart_ls)].pcolormesh(
            pred_cart.x.values,
            pred_cart.y.values,
            pred_cart,
            shading="auto",
            cmap=cmap,
            norm=norm,
        )
        grid[i + len(pred_cart_ls)].grid(False)
        if i == len(pred_cart_ls) - 1:
            grid.cbar_axes[1].colorbar(im2)
            # cbar font size
            grid.cbar_axes[1].tick_params(labelsize=cbar_tick_fontsize)

        # the s_truth_cart_ls[0] is the full ground truth image, we use it to calculate the error
        error_cart = pred_cart - s_truth_cart_ls[0]
        im3 = grid[i + 2 * len(pred_cart_ls)].pcolormesh(
            error_cart.x.values,
            error_cart.y.values,
            error_cart,
            shading="auto",
            cmap=err_cmap,
            norm=err_norm,
        )
        grid[i + 2 * len(pred_cart_ls)].grid(False)
        if i == len(pred_cart_ls) - 1:
            grid.cbar_axes[2].colorbar(im3)
            # cbar font size
            grid.cbar_axes[2].tick_params(labelsize=cbar_tick_fontsize)

    # row titles
    if row_titles is not None:
        for ax, title in zip(grid[0 :: len(pred_cart_ls)], row_titles):
            ax.set_ylabel(title, fontsize=title_fontsize)

    # set axes tick fontsize
    for ax in grid:
        # x and y ticks at 1*r_p
        ax.set_xticks(np.array([-1, 0, 1]) * r_p)
        ax.set_yticks(np.array([-1, 0, 1]) * r_p)
        # x and y tick labels in latex -2r_p, -r_p, 0, r_p, 2r_p
        ax.set_xticklabels(["$-r_p$", "$0$", "$r_p$"], fontsize=axes_tick_fontsize)
        ax.set_yticklabels(["$-r_p$", "$0$", "$r_p$"], fontsize=axes_tick_fontsize)

    plt.grid(False)
    return fig, grid
