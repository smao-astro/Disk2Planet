import functools
import pathlib
import subprocess
import typing

import IPython.display
import IPython.display
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import xarray as xr

import onet_disk2D.run.inverse_job
import onet_disk2D.transformed_subset_creator
import onet_disk2D.utils
import onet_disk2D.utils
import onet_disk2D.visualization
import onet_disk2D.visualization.images
import onet_disk2D.visualization.loss_surface

NETWORKS = {"sigma": "log_sigma_model", "v_r": "v_r_model", "v_theta": "v_theta_model"}
DATASET_IDS = {"full": "cdd269f6", "-90to90": "801a6ff6", "90to-90": "4c770dc7"}
PARAMETER_NAMES = ("alpha", "h0", "q", "r_p", "theta_p")


class GuildCMARst:
    def __init__(
        self,
        name: str,
        cluster: str = "cedar",
        log_alpha_range=(-3.52, -1),
        h0_range=(0.05, 0.1),
        log_q_range=(-4.3, -2.7),
        r_p_range=(0.40096430227206814, 2.4940020300796846),
        theta_p_range=(-np.pi, np.pi),
    ):
        """

        Args:
            name: e.g. v_r_15

        Note:
            1. Only works for pure SSIM experiments

        Returns:

        """
        self.name = name
        self.cluster = cluster
        self.guild_dir = "cma/" + name
        self.run_root_dir = pathlib.Path(
            f"/Users/kyika/project/pinn/onet-disk2D-single/{self.cluster}/{self.guild_dir}/runs"
        )
        self.log_alpha_range = log_alpha_range
        self.h0_range = h0_range
        self.log_q_range = log_q_range
        self.r_p_range = r_p_range
        self.theta_p_range = theta_p_range

    @functools.cached_property
    def df(self):
        df = pd.read_csv(
            f"/Users/kyika/project/pinn/onet-disk2D-single/{self.cluster}/{self.guild_dir}/cma_opt.csv"
        )
        for drop_cols in ["label", "data_root_dir"]:
            if drop_cols in df.columns:
                df.drop(columns=[drop_cols], inplace=True)
        if "started" in df.columns:
            df["started"] = pd.to_datetime(df["started"])
        if "time" in df.columns:
            # convert to time delta
            df["time"] = pd.to_timedelta(df["time"])
        return df

    @functools.cached_property
    def df_loss_surf(self):
        try:
            df = pd.read_csv(
                f"/Users/kyika/project/pinn/onet-disk2D-single/{self.cluster}/{self.guild_dir}/draw_loss_surface.csv"
            )
        except FileNotFoundError:
            return None
        return df

    @functools.cached_property
    def df_staged(self):
        if "status" not in self.df.columns:
            return None
        return self.df[self.df["status"] == "staged"].copy()

    @functools.cached_property
    def df_error(self):
        if "status" not in self.df.columns:
            return None
        return self.df[self.df["status"] == "error"].copy()

    @functools.cached_property
    def df_error_and_terminated(self):
        if "status" not in self.df.columns:
            return None
        return self.df[self.df["status"].isin(["error", "terminated"])].copy()

    @functools.cached_property
    def df_completed(self):
        if "status" not in self.df.columns:
            df = self.df.copy()
        else:
            df = self.df[self.df["status"] == "completed"].copy()

        if (self.df_loss_surf is not None) and (
            "loss_min" in self.df_loss_surf.columns
        ):
            df_loss_surf = self.df_loss_surf.copy()
            df_loss_surf.rename(
                columns={
                    "fargo_run_id": "run_id",
                    "loss_min": "loss_opt",
                    "run": "loss_surf_id",
                },
                inplace=True,
            )
            df = df.merge(
                df_loss_surf[["run_id", "loss_opt", "loss_surf_id"]],
                on="run_id",
                how="left",
            )
            df["loss_diff"] = df["loss"] - df["loss_opt"]

        for key in ["pred", "truth"]:
            df[f"alpha_rlt_{key}"] = (
                np.log10(df[f"alpha_{key}"]) - self.log_alpha_range[0]
            ) / (self.log_alpha_range[1] - self.log_alpha_range[0])
            df[f"h0_rlt_{key}"] = (df[f"h0_{key}"] - self.h0_range[0]) / (
                self.h0_range[1] - self.h0_range[0]
            )
            df[f"q_rlt_{key}"] = (np.log10(df[f"q_{key}"]) - self.log_q_range[0]) / (
                self.log_q_range[1] - self.log_q_range[0]
            )
            # below is not a bug,
            df[f"r_p_rlt_{key}"] = (
                df[f"r_p_{key}"] / df["r_p_truth"] - self.r_p_range[0]
            ) / (self.r_p_range[1] - self.r_p_range[0])
            df[f"theta_p_rlt_{key}"] = (
                df[f"theta_p_{key}"] - self.theta_p_range[0]
            ) / (self.theta_p_range[1] - self.theta_p_range[0])

        df["alpha_diff"] = np.log10(df["alpha_pred"]) - np.log10(df["alpha_truth"])
        df["h0_diff"] = (df["h0_pred"] - df["h0_truth"]) / df["h0_truth"]
        df["q_diff"] = np.log10(df["q_pred"]) - np.log10(df["q_truth"])
        df["r_p_diff"] = (df["r_p_pred"] - df["r_p_truth"]) / df["r_p_truth"]
        theta_diff = df["theta_p_pred"] - df["theta_p_truth"]
        # wrap to [-pi, pi]
        df["theta_p_diff"] = np.mod(theta_diff + np.pi, 2 * np.pi) - np.pi
        # relative difference
        for pname in PARAMETER_NAMES:
            df[f"{pname}_rlt_diff"] = df[f"{pname}_rlt_pred"] - df[f"{pname}_rlt_truth"]

        return df

    @functools.cached_property
    def df_min(self):
        idxmin = self.df_completed.groupby("run_id")["loss"].idxmin()
        df_min = self.df_completed.loc[remove_nan(idxmin)].copy()
        df_loss_std = self.df_completed.groupby("run_id")["loss"].std().reset_index()
        df_loss_std.rename(columns={"loss": "loss_std"}, inplace=True)
        df_min = df_min.merge(df_loss_std, on="run_id", how="left")
        return df_min

    @functools.cached_property
    def df_diff(self):
        df_diff = self.df_min[self.df_min["loss_diff"] <= 0][
            ["alpha_diff", "h0_diff", "q_diff", "r_p_diff", "theta_p_diff"]
        ]
        return df_diff

    @functools.cached_property
    def df_vanilla_mean(self):
        mean = self.df_diff.mean()
        mean.name = self.name
        return mean

    @functools.cached_property
    def df_vanilla_std(self):
        """vanilla standard deviation, influenced by outliers"""
        std = self.df_diff.std()
        std.name = self.name
        return std

    @functools.cached_property
    def df_median(self):
        median = self.df_diff.median()
        median.name = self.name
        return median

    @functools.cached_property
    def df_percentile_sigma(self):
        # estimate the sigma from the percentile of one sigma
        sigma_minus = self.df_diff.quantile(0.158655)
        sigma_plus = self.df_diff.quantile(0.841345)
        sigma = (sigma_plus - sigma_minus) / 2
        sigma.name = self.name
        return sigma

    @functools.cached_property
    def df_robust_sigma(self):
        """robust standard deviation, less influenced by outliers"""
        # calculating robust sigma from median absolute deviation
        mad: pd.Series = (self.df_diff - self.df_median).abs().median()
        robust_sigma: pd.Series = mad * 1.4826
        robust_sigma.name = self.name
        return robust_sigma

    @functools.cached_property
    def df_robust_filter_mask(self):
        median: pd.Series = self.df_diff.median()
        df = np.all(np.abs((self.df_diff - median) / self.df_robust_sigma) <= 5, axis=1)
        return df

    @functools.cached_property
    def df_robust_mean(self):
        robust_mean = self.df_diff[self.df_robust_filter_mask].mean()
        robust_mean.name = self.name
        return robust_mean

    @functools.cached_property
    def df_robust_std(self):
        """robust standard deviation, less influenced by outliers"""
        # calculating robust standard deviation
        robust_std = self.df_diff[self.df_robust_filter_mask].std()
        robust_std.name = self.name
        return robust_std

    @functools.cached_property
    def df_unused_runs(self):
        runs = list(set(self.df["run"].tolist()) - set(self.df_min["run"].tolist()))
        df = pd.DataFrame(runs, columns=["run"])
        return df

    def upload_unused_runs(self):
        self.df_unused_runs.to_csv(
            "unused_runs.csv",
            index=False,
            header=False,
        )
        subprocess.run(
            [
                "rsync",
                "-Ruv",
                "unused_runs.csv",
                f'symao@{self.cluster}.computecanada.ca:"/scratch/symao/onet-disk2D-single/guild_home/'
                + self.guild_dir
                + '"',
            ]
        )

    def display_df(self):
        IPython.display.display(self.df.drop(columns=["label"], inplace=False))

    def display_cma_plot(self, run: str):
        IPython.display.display(
            IPython.display.Image(
                onet_disk2D.utils.match_run_dir(self.run_root_dir, run) / "cmalog.png"
            )
        )

    def show_run_status(self):
        if "status" not in self.df.columns:
            print("No status column")
        else:
            IPython.display.display(self.df.groupby("status").count())

    def stage_failed_runs(self, default_key=10, default_sigma0=0.5):
        if "status" not in self.df.columns:
            print("No status column")
            return
        for _, row in self.df_error_and_terminated.iterrows():
            print(
                f"guild run onet:cma_opt network_root_dir=$CMA_NETWORK_ROOT network_id=$NETWORK_ID data_root_dir=$CMA_DATA_ROOT dataset_id=$DATASET_ID save_dir=. run_id={row['run_id']} --proto {row['run']} --force-sourcecode --stage -y && guild runs rm {row['run']} -y;"
            )

    def get_df_min_data(self, sort=None, label_cols=("key",)):
        keys = ["run", "run_id"] + list(label_cols) + ["loss"]
        if "loss_opt" in self.df_min.columns:
            keys.append("loss_opt")
        df = self.df_min[
            keys
            + [key for key in self.df_min.columns if key.endswith("diff")]
            + [
                key
                for key in self.df_min.columns
                if (key.endswith("truth") or key.endswith("pred"))
            ]
        ].copy()
        if sort is not None:
            return df.sort_values(
                by=sort,
                key=(None if sort == "loss_diff" else abs),
                ascending=False,
            )

    def get_df_min_description(self):
        return self.df_min[
            [
                "time",
                "loss",
                "alpha_diff",
                "h0_diff",
                "q_diff",
                "r_p_diff",
                "theta_p_diff",
            ]
        ].describe()

    def show_hist(self, keyword: str, n_bins: int = 30):
        plt.hist(self.df_min[keyword], bins=np.linspace(-1, 1, n_bins))
        plt.title(f"{keyword} histogram")
        plt.show()
        IPython.display.display(self.df_min[keyword].describe())

    def get_loss_surf_fig(self, run: str, cmin=None, cmax=None):
        loss_surf_id = self.df_completed[self.df_completed["run"] == run][
            "loss_surf_id"
        ].values[0]
        print(f"loss_surf_id: {loss_surf_id}")
        loss_surf = xr.load_dataarray(
            onet_disk2D.utils.match_run_dir(self.run_root_dir, loss_surf_id)
            / "loss_surf.nc"
        )
        fig, y_key, x_key = onet_disk2D.visualization.loss_surface.draw_loss_surface(
            loss_surf=loss_surf,
            minimize=True,
        )
        fig = onet_disk2D.visualization.loss_surface.add_ground_truth2loss_surface(
            loss_surf=loss_surf,
            fig=fig,
            y_key=y_key,
            x_key=x_key,
        )
        # add prediction
        fig.add_trace(
            go.Scatter(
                x=self.df_completed[self.df_completed["run"] == run][f"{x_key}_pred"],
                y=self.df_completed[self.df_completed["run"] == run][f"{y_key}_pred"],
                mode="markers",
                marker=dict(color="lightgreen", size=10, opacity=1.0),
                customdata=self.df_completed[self.df_completed["run"] == run][
                    "loss"
                ].values,
                hovertemplate=f"Pred: {x_key}=%{{x:.3f}}, {y_key}=%{{y:.3f}}<br>Loss=%{{customdata:.3f}}<extra></extra>",
                showlegend=False,
            )
        )

        if cmin is not None:
            fig.update_layout({"coloraxis": {"cmin": cmin}})
        if cmax is not None:
            fig.update_layout({"coloraxis": {"cmax": cmax}})
        fig.update_layout(width=460, height=460, template="simple_white")
        return fig

    def get_error_distribution_2d(
        self,
        x_key="q_diff",
        y_key="alpha_diff",
        bins=40,
        x_range: tuple = (-0.05, 0.05),
        y_range: tuple = (-0.05, 0.05),
        vmin=None,
        vmax=None,
    ):
        y = self.df_min[y_key]
        A = np.vstack([self.df_min[x_key], np.ones(len(self.df_min[x_key]))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]

        fig = plt.hist2d(
            self.df_min[x_key],
            self.df_min[y_key],
            bins=bins,
            range=(x_range, y_range),
            cmap="Blues",
            vmin=vmin if vmin is not None else 0,
            vmax=vmax if vmax is not None else 14,
        )
        # overplot the least squares line
        x = np.linspace(*x_range, 100)
        plt.plot(x, m * x + c, "r", label="Fitted line")
        plt.xlabel(
            r"$log_{10}(q_{predict})-log_{10}(q_{truth})$"
            if x_key == "q_diff"
            else rf"$\Delta {x_key}$"
        )
        plt.ylabel(
            r"$log_{10}(\alpha_{predict})-log_{10}(\alpha_{truth})$"
            if y_key == "alpha_diff"
            else rf"$\Delta {y_key}$"
        )
        plt.colorbar()

        return fig, m, c

    def get_error_hexbin_2d(self):
        raise NotImplementedError


def remove_nan(df: pd.DataFrame):
    return df[~pd.isna(df)]


def modify_list(input_list, key, pnames: list = PARAMETER_NAMES):
    """
    The function locates the 'key' in the 'input_list', deletes it, then inserts 'key_alpha' to 'key_theta_p'
    at the same location where 'key' was, and returns the modified list.

    :param input_list: List of strings.
    :param key: The key string to be located and deleted.
    :param pnames: The string appendix to be inserted.
    :return: Modified list.
    """

    if key not in input_list:
        return input_list  # Return the original list if the key is not found

    index = input_list.index(key)  # Find the index of the key
    del input_list[index]  # Delete the key

    # Insert 'key_alpha' to 'key_theta_p' at the located index
    for i, pname in enumerate(pnames):
        input_list.insert(index + i, f"{key}_{pname}")

    return input_list


def parse_header_line(line: str, x_names: typing.Iterable = None):
    columns = line.split('columns="')[1].split('"')[0].split(", ")
    # strip the leading and trailing spaces
    columns = [col.strip() for col in columns]
    # Dataframes do not allow duplicate column names
    for void_key in ["void", "0"]:
        i = 0
        for j, col in enumerate(columns):
            if col == void_key:
                columns[j] = f"{void_key}_{i}"
                i += 1
    if x_names is not None:
        for x_name in x_names:
            columns = modify_list(columns, x_name)
    return columns


def plot_hexbin(
    data,
    x_col,
    y_col,
    xlim=(-0.06, 0.06),
    ylim=(-0.06, 0.06),
    vmin=0,
    vmax=15,
    style="ticks",
    height=5,
    gridsize=30,
    adjust_params=(0.2, 0.8, 0.8, 0.2),
    cbar_dims=(0.85, 0.25, 0.03, 0.4),
    x_axis_label=r"$\log_{10}(q^{\mathrm{inferred}}) - \log_{10}(q^{\mathrm{true}})$",
    y_axis_label=r"$\log_{10}(\alpha^{\mathrm{inferred}}) - \log_{10}(\alpha^{\mathrm{true}})$",
    xticks=(-0.05, 0, 0.05),
    yticks=(-0.05, 0, 0.05),
    cbar_ticks=(0, 4, 9, 14),
):
    sns.set_theme(style=style)

    hexplot = sns.jointplot(
        data=data,
        x=x_col,
        y=y_col,
        kind="hex",
        xlim=xlim,
        ylim=ylim,
        height=height,
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        gridsize=gridsize,
        vmin=vmin,
        vmax=vmax,
    )

    hexplot.set_axis_labels(x_axis_label, y_axis_label)

    hexplot.ax_joint.set_xticks(xticks)
    hexplot.ax_joint.set_yticks(yticks)

    plt.subplots_adjust(
        left=adjust_params[0],
        right=adjust_params[1],
        top=adjust_params[2],
        bottom=adjust_params[3],
    )

    cbar_ax = hexplot.fig.add_axes(cbar_dims)
    cbar = plt.colorbar(cax=cbar_ax, ticks=cbar_ticks, label="Percentage")
    cbar.ax.set_yticklabels([f"{tick/len(data):.1%}" for tick in cbar_ticks])

    return hexplot


class CMARst:
    def __init__(self, file_dir):
        self.file_dir = pathlib.Path(file_dir)

    @functools.cached_property
    def xmean(self):
        with open(self.file_dir / "xmean.dat") as f:
            first_line = f.readline()
            print(first_line)
        columns = parse_header_line(first_line, ("xmean",))
        df = pd.read_csv(
            self.file_dir / "xmean.dat",
            skiprows=1,
            sep="\s+",
            header=None,
            names=columns,
        )
        # drop void columns
        df.drop(
            columns=[col for col in df.columns if col.startswith("void")], inplace=True
        )

        return df

    @functools.cached_property
    def xrecentbest(self):
        with open(self.file_dir / "xrecentbest.dat") as f:
            first_line = f.readline()
            print(first_line)
        columns = parse_header_line(first_line, ("xbest",))
        df = pd.read_csv(
            self.file_dir / "xrecentbest.dat",
            skiprows=1,
            sep="\s+",
            header=None,
            names=columns,
        )
        # drop void columns
        df.drop(
            columns=[col for col in df.columns if col.startswith("0")], inplace=True
        )

        return df

    @functools.cached_property
    def fit(self):
        with open(self.file_dir / "fit.dat") as f:
            first_line = f.readline()
            print(first_line)
        columns = parse_header_line(first_line, None)
        df = pd.read_csv(
            self.file_dir / "fit.dat",
            skiprows=1,
            sep="\s+",
            header=None,
            names=columns,
        )
        return df

    @functools.cached_property
    def stddev(self):
        with open(self.file_dir / "stddev.dat") as f:
            first_line = f.readline()
            print(first_line)
        columns = parse_header_line(
            first_line, ("stds==sigma*sigma_vec.scaling*sqrt(diag(C))",)
        )
        df = pd.read_csv(
            self.file_dir / "stddev.dat",
            skiprows=1,
            sep="\s+",
            header=None,
            names=columns,
        )
        return df

    @functools.cached_property
    def std_ratio(self):
        arrays = []
        for pname in PARAMETER_NAMES[1:]:
            arrays.append(
                self.stddev[f"stds==sigma*sigma_vec.scaling*sqrt(diag(C))_alpha"]
                / self.stddev[f"stds==sigma*sigma_vec.scaling*sqrt(diag(C))_{pname}"]
            )
        arrays = np.array(arrays).T
        df = pd.DataFrame(
            arrays, columns=[f"alpha/{pname}" for pname in PARAMETER_NAMES[1:]]
        )
        return df


def get_pred_truth(
    opt_run: str,
    guild_run_id: str,
    network_id: str,
    dataset_id: str,
    guild_dir: str,
    parameter_type: typing.Union[str, tuple[str, str, str, str, str]],
    remove_radial_background: bool = False,
    xylim: float = 2.0,
    xygrid: int = 256,
    network_root_dir="/Users/kyika/project/pinn/onet-disk2D-single/cedar/pm_al_ar_fung_gap2steady4_2/runs/",
    data_root_dir="/Users/kyika/project/pinn/onet-disk2D-single/cedar/pm_al_ar_fung_gap2steady4test/runs/",
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """

    Args:
        opt_run: e.g. 018b04e8
        guild_run_id: e.g. 3714094c
        network_id: e.g. log_sigma_model
        dataset_id: e.g. 7fc6649a
        guild_dir: e.g. cma/sigma_noise03
        parameter_type: e.g. pred | truth | (pred, pred, pred, truth, pred)
        remove_radial_background:
        network_root_dir:
        data_root_dir:

    Returns:

    """
    data_dir = onet_disk2D.utils.match_run_dir(data_root_dir, dataset_id)
    guild_csv_file = pathlib.Path(
        f"/Users/kyika/project/pinn/onet-disk2D-single/cedar/{guild_dir}/cma_opt.csv"
    )

    df = pd.read_csv(guild_csv_file)
    df_opt_run = df[df["run"] == opt_run]

    job, data_obj = onet_disk2D.run.inverse_job.get_job_and_data_obj(
        run_id=guild_run_id,
        network_id=network_id,
        network_root_dir=network_root_dir,
        args_file="args.yml",
        arg_groups_file="arg_groups.yml",
        fargo_setup_file="fargo_setups.yml",
        metric="l2",
        data_dir=data_dir,
    )

    if isinstance(parameter_type, str):
        if parameter_type not in ["pred", "truth"]:
            raise ValueError
        parameter_type = [parameter_type] * 5
    u = df_opt_run[
        [f"{pname}_{ptype}" for pname, ptype in zip(PARAMETER_NAMES, parameter_type)]
    ].values[0]
    u = jnp.array(u.flatten())

    pred = job.inv_pred_fn(u, data_obj.y_net)
    s_truth = data_obj.truth
    if remove_radial_background:
        pred, s_truth = job.remove_radial_background(
            pred, s_truth, data_obj.y_net[:, 0]
        )
    pred = pred.reshape(job.nr, job.ntheta)
    s_truth = s_truth.reshape(job.nr, job.ntheta)

    pred = xr.DataArray(
        pred,
        coords=[("r", data_obj.data.r.values), ("theta", data_obj.data.theta.values)],
    )
    s_truth = xr.DataArray(
        s_truth,
        coords=[("r", data_obj.data.r.values), ("theta", data_obj.data.theta.values)],
    )

    x = y = np.linspace(-xylim, xylim, xygrid) * df_opt_run["r_p_truth"].values[0]
    pred_cart = onet_disk2D.utils.xarray_polar_to_cartesian(pred, x, y)
    s_truth_cart = onet_disk2D.utils.xarray_polar_to_cartesian(s_truth, x, y)

    return pred, s_truth, pred_cart, s_truth_cart
