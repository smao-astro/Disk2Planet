from plotly import express as px


def build_scatter_3d_plot(df_fargo_):
    fig = px.scatter_3d(
        df_fargo_,
        x="ALPHA truth",
        y="h0 truth",
        z="q truth",
        color="ssim",
        color_continuous_scale="Viridis",
        log_x=True,
        log_y=False,
        log_z=True,
        custom_data=["run_id"],
        hover_data="run_id",
        width=800,
        height=700,
    )
    # colorbar horizontal at bottom
    fig.update_layout(
        {
            "coloraxis": {
                "colorbar": {
                    "orientation": "h",
                    "len": 0.8,
                    "thickness": 20,
                    "x": 0.5,
                    "xanchor": "center",
                    "y": -0.2,
                    "yanchor": "top",
                }
            }
        }
    )
    fig.update_layout(
        {
            "scene": {
                "xaxis": {"exponentformat": "e", "showexponent": "all"},
                "zaxis": {"exponentformat": "e", "showexponent": "all"},
            },
            "margin": {"l": 0, "r": 0, "b": 0, "t": 0},
        }
    )
    return fig


def build_scatter_3d_plot_filtered(df_):
    fig = px.scatter_3d(
        df_,
        x="ALPHA truth",
        y="h0 truth",
        z="q truth",
        color="converge",
        color_discrete_map={
            True: "green",
            False: "red",
        },
        log_x=True,
        log_y=False,
        log_z=True,
        custom_data=["run_id"],
        hover_data="run_id",
    )
    # reduce marker size
    fig.update_traces(marker=dict(size=10))
    # colorbar horizontal at bottom
    fig.update_layout(
        {
            "coloraxis": {
                "colorbar": {
                    "orientation": "h",
                    "len": 0.8,
                    "thickness": 20,
                    "x": 0.5,
                    "xanchor": "center",
                    "y": -0.2,
                    "yanchor": "top",
                }
            }
        }
    )
    # legend horizontal at top
    fig.update_layout(
        {
            "legend": {
                "orientation": "h",
                "yanchor": "top",
                "y": 1.1,
                "xanchor": "left",
                "x": 0.1,
                # font size
                "font": {"size": 20},
            }
        }
    )
    fig.update_layout(
        {
            "scene": {
                "xaxis": {"exponentformat": "e", "showexponent": "all"},
                "zaxis": {"exponentformat": "e", "showexponent": "all"},
            }
        }
    )
    return fig
