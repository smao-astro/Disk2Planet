from typing import Union

import numpy as np
import plotly.subplots
from plotly import express as px, graph_objects as go


def line_plot(
    log,
    key,
    log_y=True,
    width=400,
    height=400,
    gen: Union[int, None] = None,
    font_size=15,
    title_font_size=15,
    marker_size=10,
):
    fig = px.line(
        x=log["gen"].values,
        y=log["log_params"].sel(params=key).values,
        log_y=log_y,
        width=width,
        height=height,
    )
    # add horizontal line
    truth = float(log["truth_params"].sel(params=key).values)
    fig.add_hline(y=truth, line_dash="dash", line_color="red")
    # title, title font size, title in the middle
    fig.update_layout(font_size=font_size)
    fig.update_layout(
        title=f"{key} truth={truth:.2g}", title_font_size=title_font_size, title_x=0.5
    )
    if log_y:
        # y axis exponent format
        fig.update_yaxes(exponentformat="power", showexponent="all")
    if gen is not None:
        fig.add_trace(
            go.Scatter(
                x=[gen],
                y=[log["log_params"].sel(params=key).isel(gen=gen).values],
                mode="markers",
                marker=dict(color="green", size=marker_size),
            )
        )
    # legend off
    fig.update_layout(showlegend=False)
    # margin
    fig.update_layout(margin=dict(l=0, r=0))
    # automargin
    fig.update_layout(autosize=True)
    return fig


def update_line_plot_marker(log, key, fig, gen):
    """Implement only if you want to improve the speed of the dash app."""
    pass


def build_line_plots(
    logs,
    metric,
    log_y,
    horizontal_spacing=0.04,
    show_planet_location=False,
):
    subplot_titles = ["metric"]
    params = (
        ["ALPHA", "ASPECTRATIO", "PLANETMASS"]
        if not show_planet_location
        else ["ALPHA", "ASPECTRATIO", "PLANETMASS", "r_p", "theta_p"]
    )
    title_name = (
        ["alpha", "h0", "q"]
        if not show_planet_location
        else ["alpha", "h0", "q", "r_p", "theta_p"]
    )
    # get truth
    truth = [
        {float(log["truth_params"].sel(params=key).values) for log in logs.values()}
        for key in params
    ]
    subplot_titles += [
        f"{key}={next(iter(value)):.2g}" if len(value) == 1 else key
        for key, value in zip(title_name, truth)
    ]

    ymin = [min(truth_) for truth_ in truth]
    ymax = [max(truth_) for truth_ in truth]
    range_lo = 0.9
    range_hi = 1.1
    fig_ = plotly.subplots.make_subplots(
        rows=1,
        cols=6 if show_planet_location else 4,
        subplot_titles=subplot_titles,
        horizontal_spacing=horizontal_spacing,
    )
    color_palette = px.colors.sequential.Blackbody
    if len(logs) > len(color_palette):
        color_palette = px.colors.sample_colorscale(color_palette, len(logs))
    # loss plot
    for i, (job_id, log) in enumerate(logs.items(), start=1):
        fig_.add_trace(
            go.Scatter(
                x=log["gen"],
                y=log[metric].values,
                name=i,
                legendgroup=i,
                line=dict(color=color_palette[i - 1]),
            ),
            row=1,
            col=1,
        )
    # params plot
    for i_col, params in enumerate(params, start=2):
        for i, (job_id, log) in enumerate(logs.items(), start=1):
            y = log["log_params"].sel(params=params).values
            ymin[i_col - 2] = min(ymin[i_col - 2], float(y.min()))
            ymax[i_col - 2] = max(ymax[i_col - 2], float(y.max()))
            fig_.add_trace(
                go.Scatter(
                    x=log["gen"],
                    y=y,
                    name=i,
                    legendgroup=i,
                    line=dict(color=color_palette[i - 1]),
                    showlegend=False,
                ),
                row=1,
                col=i_col,
            )
    # add hline
    for i_col, truth_ in enumerate(truth, start=2):
        for truth__ in truth_:
            fig_.add_hline(
                y=truth__, row=1, col=i_col, line_dash="dash", line_color="black"
            )

    # to log scale
    if log_y[0]:
        fig_.update_layout({"yaxis": {"type": "log", "exponentformat": "power"}})
    for i_col, (log_y_, ymin_, ymax_) in enumerate(zip(log_y[1:], ymin, ymax), start=2):
        if log_y_:
            fig_.update_layout(
                {
                    f"yaxis{i_col}": {
                        "type": "log",
                        "range": [
                            np.log10(range_lo * ymin_),
                            np.log10(range_hi * ymax_),
                        ],
                        "dtick": 1,
                        "exponentformat": "power",
                    }
                }
            )
        else:
            fig_.update_layout(
                {f"yaxis{i_col}": {"range": [range_lo * ymin_, range_hi * ymax_]}}
            )
    # font size
    fig_.update_layout(font_size=20)
    fig_.update_annotations(font_size=20)

    return fig_
