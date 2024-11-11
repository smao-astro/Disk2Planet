from typing import Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import xarray as xr

KEY_MAP = {
    "alpha": "ALPHA",
    "h0": "ASPECTRATIO",
    "q": "PLANETMASS",
    "r_p": "r_p",
    "theta_p": "theta_p",
}


def df_cross_match(
    df_opt: pd.DataFrame, df_loss_surf: pd.DataFrame, opt_run_id: str
) -> str:
    if "run" in df_opt.columns:
        df_selected = df_opt.loc[df_opt["run"] == opt_run_id]
    elif "id" in df_opt.columns:
        df_selected = df_opt.loc[df_opt["id"] == opt_run_id]
    else:
        raise ValueError("df_opt must have either 'run' or 'id' column")
    fargo_run_id = df_selected["run_id"].values[0]
    metric = df_selected["metric"].values[0]

    cri = [df_loss_surf["fargo_run_id"] == fargo_run_id]
    if "metric" in df_loss_surf.columns:
        cri.append(df_loss_surf["metric"] == metric)
    cri = np.all(cri, axis=0)
    loss_surf_run_id = df_loss_surf.loc[cri, "run"].values[0]
    return loss_surf_run_id


def draw_loss_surface(
    loss_surf: xr.DataArray,
    minimize: bool = True,
    min_marker_config=None,
) -> Tuple[go.Figure, str, str]:
    """Loss surface with minimum"""
    if min_marker_config is None:
        min_marker_config = dict(color="white", symbol="x", size=10)

    axes = {
        i: key
        for i, (length, key) in enumerate(zip(loss_surf.shape, loss_surf.dims))
        if length > 1
    }
    assert len(axes) == 2
    y_key, x_key = axes.values()

    # draw the surface
    y = loss_surf[y_key].values
    x = loss_surf[x_key].values
    # loss_surf shape: (..., len(y), ..., len(x), ...)
    # y should be the first dimension as required by imshow
    fig = px.imshow(
        loss_surf.values.reshape(len(y), len(x)), x=x, y=y, origin="lower", aspect=3
    )
    # log axis
    if y_key in ["alpha", "q"]:
        fig.update_yaxes(
            type="log",
            exponentformat="power",
            range=[np.log10(y.min()), np.log10(y.max())],
        )
    else:
        fig.update_yaxes(range=[y.min(), y.max()])
    if x_key in ["alpha", "q"]:
        fig.update_xaxes(
            type="log",
            exponentformat="power",
            range=[np.log10(x.min()), np.log10(x.max())],
        )
    else:
        fig.update_xaxes(range=[x.min(), x.max()])
    # axis labels
    fig.update_yaxes(title_text=y_key)
    fig.update_xaxes(title_text=x_key)

    # draw minimum
    loss_min = loss_surf.where(
        loss_surf == (loss_surf.min() if minimize else loss_surf.max()), drop=True
    ).squeeze()
    print(f"loss_min: {float(loss_min.values):.3g}")
    loss_min_x = float(loss_min[x_key].values)
    loss_min_y = float(loss_min[y_key].values)
    hover_name = "Min" if minimize else "Max"
    fig.add_trace(
        go.Scatter(
            x=[loss_min_x],
            y=[loss_min_y],
            marker=min_marker_config,
            customdata=np.array([loss_min.values]).reshape((-1, 1)),
            hovertemplate=f"{hover_name}: {x_key}={loss_min_x:.2g}, {y_key}={loss_min_y:.2g}<br>"
            + "Score: %{customdata[0]:.2g}",
            showlegend=False,
        )
    )
    return fig, y_key, x_key


def add_ground_truth2loss_surface(
    loss_surf: xr.DataArray,
    fig: go.Figure,
    y_key: str,
    x_key: str,
    truth_marker_config=None,
) -> go.Figure:
    if truth_marker_config is None:
        truth_marker_config = dict(color="green", symbol="cross", size=10)
    # add truth
    x_truth = loss_surf.attrs[f"{x_key}_truth"]
    y_truth = loss_surf.attrs[f"{y_key}_truth"]
    fig.add_trace(
        go.Scatter(
            x=[x_truth],
            y=[y_truth],
            mode="markers",
            marker=truth_marker_config,
            customdata=np.array([loss_surf.attrs["truth_score"]])[:, None],
            hovertemplate=f"Truth: {x_key}={x_truth:.2g}, {y_key}={y_truth:.2g}<br>"
            + "Score: %{customdata[0]:.2g}",
            showlegend=False,
        )
    )
    return fig


def add_trace2loss_surface(
    fig: go.Figure,
    log: xr.Dataset,
    y_key,
    x_key,
    line_config=None,
) -> go.Figure:
    if line_config is None:
        line_config = dict(color="lightgreen", width=2)

    # draw the trace
    y_log = log["log_params"].sel(params=KEY_MAP[y_key]).values
    x_log = log["log_params"].sel(params=KEY_MAP[x_key]).values

    fig.add_trace(
        go.Scatter(
            x=x_log,
            y=y_log,
            mode="lines",
            line=line_config,
            showlegend=False,
        )
    )

    return fig


def draw_trace_on_surface(
    loss_surf: xr.DataArray,
    log: xr.Dataset,
    minimize: bool = True,
    truth_marker_config=None,
    min_marker_config=None,
    line_config=None,
) -> go.Figure:
    fig, y_key, x_key = draw_loss_surface(
        loss_surf=loss_surf, minimize=minimize, min_marker_config=min_marker_config
    )
    fig = add_ground_truth2loss_surface(
        loss_surf=loss_surf,
        fig=fig,
        y_key=y_key,
        x_key=x_key,
        truth_marker_config=truth_marker_config,
    )
    fig = add_trace2loss_surface(
        fig=fig, log=log, y_key=y_key, x_key=x_key, line_config=line_config
    )
    return fig
