import dash
import dash.dash_table.Format as Format
import numpy as np
import pandas as pd

import onet_disk2D.run.inverse_job


def rename_run_to_id(df: pd.DataFrame):
    """For dataTable convenience, rename the 'run' column to 'id' and set it as the index.

    Args:
        df:

    Returns:

    """
    new_df = df.copy()
    # check if new_df has a column named 'run'
    if "run" not in new_df.columns:
        raise ValueError("The DataFrame does not have a column named 'run'.")
    new_df.rename(columns={"run": "id"}, inplace=True)
    new_df.set_index("id", inplace=True, drop=False)
    # do not drop, as required by dash_table.DataTable
    return new_df


def rename_pred_truth_cols(df: pd.DataFrame):
    """Rename the cols with '_pred' or '_truth' to 'pred' or 'truth', respectively."""
    new_df = df.copy()
    new_df.rename(
        columns={
            i: i.replace("_truth", " truth").replace("_pred", " pred")
            for i in new_df.columns
        },
        inplace=True,
    )
    return new_df


def reorder_cols(df: pd.DataFrame, to_delete=("run", "status", "label", "step")):
    """Delete the cols in `to_delete` and reorder the rest by make truth and prediction columns at the end."""
    df = df.copy()
    show_cols = [i for i in df.columns if i not in to_delete]
    front_cols = [
        i
        for i in show_cols
        if not any(
            [i.endswith(key) for key in ("mse", "ssim", "loss", "pred", "truth")]
        )
    ]
    back_cols = [i for i in show_cols if (i.endswith("truth")) or (i.endswith("pred"))]
    back_cols.sort()
    df = df.reindex(columns=front_cols + ["loss", "mse", "ssim"] + back_cols)
    return df


def rename_and_reorder(df: pd.DataFrame, r_p_truth: float, theta_p_truth: float):
    if "status" in df.columns:
        df = df[df["status"] == "completed"].copy()
    df = rename_run_to_id(df)
    df = rename_pred_truth_cols(df)
    if "r_p truth" not in df.columns:
        df["r_p truth"] = r_p_truth
    if "theta_p truth" not in df.columns:
        df["theta_p truth"] = theta_p_truth
    df = reorder_cols(
        df,
        (
            "run",
            "status",
            "label",
            "step",
            "arg_groups_file",
            "args_file",
            "fargo_setup_file",
        ),
    )
    return df


def flag_converge_status(
    df: pd.DataFrame,
    relative_error_threshold=0.2,
    angle_threshold=0.05 * np.pi,
):
    df_converge = df.copy()
    for key in ["h0", "r_p"]:
        df_converge[key + " diff"] = (
            np.abs(df_converge[key + " pred"] - df_converge[key + " truth"])
            / df_converge[key + " truth"]
        )
    for key in ["ALPHA", "q"]:
        df_converge[key + " diff"] = np.abs(
            np.log10(df_converge[key + " pred"]) - np.log10(df_converge[key + " truth"])
        )
    theta_diff = df_converge["theta_p pred"] - df_converge["theta_p truth"]
    theta_diff = np.mod(theta_diff + np.pi, 2 * np.pi) - np.pi
    df_converge["theta_p diff"] = np.abs(theta_diff)

    df_converge["converge"] = np.all(
        df_converge[[key + " diff" for key in ["ALPHA", "h0", "q", "r_p", "theta_p"]]]
        < np.array([relative_error_threshold] * 4 + [angle_threshold]),
        axis=1,
    )
    return df_converge


def build_datatable(df_: pd.DataFrame, numeric_cols=None, col_width=120, page_size=6):
    """

    Notes:
        you can select multiple rows to compare their optimization history. If you do so while still want to check the 2D maps one by one, you can use two independent selecting methods, one for the 2D maps and one for the optimization history. (available selecting methods: `derived_virtual_selected_row_ids` and `active_cell`)
    References:
        https://dash.plotly.com/datatable/reference

    Returns:

    """
    if numeric_cols is None:
        numeric_cols = (
            ["loss"]
            + list(onet_disk2D.run.inverse_job.METRICS)
            + [i for i in df_.columns if (i.endswith("pred") or i.endswith("truth"))]
        )
    numeric_format = Format.Format(
        precision=2, scheme=Format.Scheme.decimal_or_exponent
    )
    columns = [
        {
            "name": i,
            "id": i,
            "deletable": False,
            "hideable": True,
            "type": "numeric",
            "format": numeric_format,
        }
        if i in numeric_cols
        else {"name": i, "id": i, "deletable": False, "hideable": True}
        for i in df_.columns
    ]
    data_table = dash.dash_table.DataTable(
        columns=columns,
        data=df_.to_dict("records"),
        editable=False,
        filter_action="native",  # allow filtering of data by user ('native') or not ('none')
        sort_action="native",
        sort_mode="multi",
        row_selectable="multi",
        row_deletable=False,
        page_size=page_size,
        style_header={
            "whiteSpace": "normal",
            "height": "auto",
        },
        style_cell={  # ensure adequate header width when text is shorter than cell's text
            "minWidth": col_width,
            "maxWidth": col_width,
            "width": col_width,
            "font-size": "1.2em",
        },
        style_data={  # overflow cells' content into multiple lines
            "whiteSpace": "normal",
            "height": "auto",
        },
    )
    return data_table
