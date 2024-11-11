import argparse
import functools
import pathlib
from functools import partial
from typing import Union, Tuple, List

import chex
import dm_pix
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

import onet_disk2D.data
import onet_disk2D.transformed_subset_creator
import onet_disk2D.utils
from . import job as job_module

METRICS = ("l2", "mse", "ssim", "dssim")
R_UNIT = 100  # AU


def load_data_array(
    data_dir: Union[str, pathlib.Path], unknown_type: str, unknown: str
):
    """

    if unknown_type == "sigma" and unknown == "log_sigma":
        the value is in linear scale, this function will convert it to log10 scale.
    if unknown_type == "v_theta" and unknown == "v_theta":
        the value should already be in non-rotating frame, this function DOES NOT take care of it.

    Args:
        data_dir:
        unknown_type:
        unknown:

    Returns:

    """
    data_dir = pathlib.Path(data_dir)
    data_file = f"batch_truth_{unknown_type}.nc"
    dataarray = xr.open_dataarray(data_dir / data_file)
    # the operation below should have been done
    # if unknown_type == "v_theta":
    #     # convert back to non-rotating frame
    #     dataarray = dataarray + dataarray.r
    if unknown == "log_sigma":
        dataarray = np.log10(dataarray)
    return dataarray


class Data:
    def __init__(self, data: xr.DataArray):
        self.data = data
        self.y_net = onet_disk2D.utils.get_y_from_faked_data(self.data)
        # Warning: you should be very careful here to make sure the y_net and truth are consistent on coordinates.
        self.truth = jnp.array(self.data.values).flatten()
        # the values below are used to determine the inverse prob search space (parameter space)
        self.r_min = np.min(self.data.r.values)
        self.r_max = np.max(self.data.r.values)

    @functools.cached_property
    def params_truth(self) -> np.array:
        params_truth = np.array(
            [
                self.data[key].values
                for key in ["ALPHA", "ASPECTRATIO", "PLANETMASS", "r_p", "theta_p"]
            ]
        )
        """ALPHA, ASPECTRATIO, PLANETMASS, r_p, theta_p"""
        return params_truth


@jax.jit
def shift_back_angle(x, angle):
    # We guess the planet is at angle, so to make prediction match the grond truth distribution, we want to calculate truth(r, theta) - pred(r, theta - angle). For example, for the density at the planet, we want to give truth(r, angle) - pred(r, theta - angle).
    return jnp.mod(x - angle + jnp.pi, 2 * jnp.pi) - jnp.pi


@jax.jit
def transform_back_inputs(u_guess, y_net):
    """Calculate the corresponding coordinates of `y_net` in the original coordinate system --- the radial unit is planet radius, and the planet is at angle 0.

    This transform the coordinates from a system where the planet locate at (u_guess[-2], u_guess[-1]) to the (1, 0) coordinate system. So that the prediction of the network can be 1) compared with the ground truth, 2) and easily displayed in the real coordinate system.

    apply radial stretch to the coordinates.

    Args:
        u_guess: shape (Np,) or (Nu, Np). The guess of the parameters.
        y_net: shape (Ny, 2). The coordinates of the sample points of a disk without knowing the planet location.

    Returns:
        inputs: dict. The inputs of the network.
            u_net: shape (Np-2,) or (Nu, Np-2). The guess of the parameters.
            y_net: shape (Ny, 2) or (Nu, Ny, 2).
    """
    u_guess, r, angle = u_guess[..., :-2], u_guess[..., -2:-1], u_guess[..., -1:]
    r = y_net[:, 0] / r
    theta = shift_back_angle(y_net[:, 1], angle)
    y_net = jnp.stack([r, theta], axis=-1)
    inputs = {"u_net": u_guess, "y_net": y_net}
    return inputs


@jax.jit
def log_sigma_remove_radial_background(data, r):
    """

    Args:
        data: shape (Nu, Ny,) must have been transformed to log10 scale
        r: shape (Ny,), in AU

    Returns:

    """
    # unit below is in A*M_*/ R_UNIT^2
    background = (r / R_UNIT) ** -0.5
    data = data - jnp.log10(background)
    return data


@jax.jit
def sigma_remove_radial_background(data, r):
    """

    Args:
        data: shape (Nu, Ny,)
        r: shape (Ny,)

    Returns:

    """
    # unit below is in A*M_*/ R_UNIT^2
    background = (r / R_UNIT) ** -0.5
    data = data / background
    return data


@jax.jit
def azimuthal_vel_remove_radial_background(data, r):
    """

    Args:
        data: shape (Ny,) or shape (Nu, Ny,)
        r: shape (Ny,)

    Returns:

    """
    # unit below is in √(GM_*/R_UNIT)
    background = (r / R_UNIT) ** -0.5
    data = data - background
    return data


class InversePred(job_module.JOB):
    """Inverse problem class.

    Easily reproduce the prediction in inverse problems, use `InverseOpt` instead for optimization.

    About naming convention: do not overwrite name defined in parent class. For names that easy to confuse, add inv_ prefix to the name.
    """

    def __init__(self, args):
        super().__init__(args)

        # for backward compatibility
        self.vmap_inv_pred_fn = self.inv_pred_fn
        self.pmap_inv_pred_fn = jax.pmap(self.inv_pred_fn, in_axes=(0, None))
        self.r_min = float(self.fargo_setups["ymin"])
        self.r_max = float(self.fargo_setups["ymax"])

        self.inv_col_idx_to_log = jnp.concatenate(
            # u, r, theta
            [self.col_idx_to_log, jnp.array([False, False], dtype=bool)]
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def inv_pred_fn(self, u_guess: chex.Array, y_net: chex.Array):
        """Predict the solution considering that the planet location is at (u_guess[Nu, -2], u_guess[Nu, -1]).

        Args:
            u_guess: shape (Nu, Np+2,). The last dimension represent PDE parameters + planet angle. The PDE parameters are in log10 scale, linear scale, and log10 scale
            y_net: The coordinates of pixels in the reference system of the disk to constrain. shape (Ny, 2), in AU

        Returns:
            (Nu, Ny,)
        """
        inputs = transform_back_inputs(u_guess, y_net)
        # the coordinates are in the reference system whose planet is at (1.0, 0.0).
        r = inputs["y_net"][..., 0]
        # only predict if r_min < r < r_max
        exclude = jnp.logical_or(r < self.r_min, r > self.r_max)
        pred = self.s_pred_fn(self.model.params, self.state, inputs)
        if self.unknown_type == "v_theta":
            # convert back to non-rotating frame
            pred = pred + r
        # scale the output
        r_p_guess = u_guess[..., -2:-1]
        if self.args["unknown"] == "log_sigma":
            # unit below is in A*M_*/ R_UNIT^2, background profile is (r/R_UNIT)^-0.5, equal to one at r = R_UNIT
            pred = pred + jnp.log10((r_p_guess / R_UNIT) ** -0.5)
        else:
            # unit below is in A*M_*/ R_UNIT^2, background profile is (r/R_UNIT)^-0.5, equal to one at r = R_UNIT
            pred = pred * (r_p_guess / R_UNIT) ** -0.5
        return jnp.where(exclude, jnp.nan, pred)


class InverseLoss(InversePred):
    def __init__(
        self,
        args,
        loss_metric: str,
        nr=None,
        ntheta=None,
    ):
        super().__init__(args)
        self.loss_metric = loss_metric
        self.nr = nr if nr else int(self.fargo_setups["ny"])
        self.ntheta = ntheta if ntheta else int(self.fargo_setups["nx"])

        self.loss_metrics = {
            "l2": self.l2_fn,
            "mse": self.mse_fn,
            "ssim": self.ssimilarity_fn,
            "dssim": self.dissimilarity_fn,
        }
        self.inv_loss_fn = self.loss_metrics[self.loss_metric]
        # for backward compatibility
        self.vmap_inv_loss_fn = self.inv_loss_fn
        self.pmap_inv_loss_fn = jax.pmap(self.inv_loss_fn, in_axes=(0, None, None))

    @partial(jax.jit, static_argnums=(0,))
    def remove_radial_background(
        self, pred: chex.Array, s_truth: chex.Array, r: chex.Array
    ):
        """

        Args:
            pred: shape (Nu, Ny,)
            s_truth: shape (Nu, Ny,)
            r: shape (Ny,) in AU
        Returns:

        """
        # remove background
        if self.args["unknown"] == "log_sigma":
            # remove background
            pred = log_sigma_remove_radial_background(pred, r)
            s_truth = log_sigma_remove_radial_background(s_truth, r)
        elif self.args["unknown"] == "sigma":
            # remove background
            pred = sigma_remove_radial_background(pred, r)
            s_truth = sigma_remove_radial_background(s_truth, r)
        elif self.unknown_type == "v_theta":
            # remove background
            pred = azimuthal_vel_remove_radial_background(pred, r)
            s_truth = azimuthal_vel_remove_radial_background(s_truth, r)
        return pred, s_truth

    @partial(jax.jit, static_argnums=(0,))
    def l2_fn(
        self, u_guess: chex.Array, y_net: chex.Array, s_truth: chex.Array
    ) -> chex.Array:
        """

        Args:
            u_guess: shape (Np,) or (Nu, Np)
            y_net: shape (Ny, 2) in AU
            s_truth: shape (Ny,)

        Returns:
            shape () or (Nu,)
        """
        pred = self.inv_pred_fn(u_guess, y_net)
        """pred: shape (Ny,) or (Nu, Ny,)"""
        # the input r: whether we scale it or not does not matter.
        pred, s_truth = self.remove_radial_background(pred, s_truth, y_net[:, 0])

        # the denominator does not depend on u_guess, so it does not influence the optimization when CMA-ES is used and there is only one physical variable. However, it should help to balance the optimization when multiple physical variables are envolved.
        loss = jnp.sqrt(jnp.nanmean((pred - s_truth) ** 2, axis=-1)) / jnp.sqrt(
            jnp.nanmean(s_truth**2)
        )
        return loss

    @partial(jax.jit, static_argnums=(0,))
    def mse_fn(
        self, u_guess: chex.Array, y_net: chex.Array, s_truth: chex.Array
    ) -> chex.Array:
        """

        Args:
            u_guess: shape (Np,) or (Nu, Np)
            y_net: shape (Ny, 2)
            s_truth: shape (Ny,)

        Returns:
            shape () or (Nu,)
        """
        pred = self.inv_pred_fn(u_guess, y_net)
        """pred: shape (Ny,) or (Nu, Ny,)"""

        loss = jnp.nanmean((pred - s_truth) ** 2, axis=-1)
        return loss

    @partial(jax.jit, static_argnums=(0,))
    def ssimilarity_fn(
        self, u_guess: chex.Array, y_net: chex.Array, s_truth: chex.Array
    ) -> chex.Array:
        """The loss function when the metric == "ssim".

        Args:
            u_guess: shape (Np,) or (Nu, Np)
            y_net: shape (Ny, 2)
            s_truth: shape (Ny,)

        Returns:
            shape () or (Nu,)
        """
        pred = self.inv_pred_fn(u_guess, y_net)
        """pred: shape (Ny,) or (Nu, Ny,)"""

        pred, s_truth = self.remove_radial_background(pred, s_truth, y_net[..., 0])

        # reshape to (nr, ntheta)
        shape = pred.shape[:-1] + (self.nr, self.ntheta)
        pred = pred.reshape(*shape)
        s_truth = s_truth.reshape(self.nr, self.ntheta)

        if pred.ndim > 2:
            ssim = jax.vmap(cal_ssim, in_axes=(0, None))(pred, s_truth)
        else:
            ssim = cal_ssim(pred, s_truth)
        return ssim

    @partial(jax.jit, static_argnums=(0,))
    def dissimilarity_fn(
        self, u_guess: chex.Array, y_net: chex.Array, s_truth: chex.Array
    ) -> chex.Array:
        """The loss function when the metric == "ssim".

        Args:
            u_guess: shape (Np,) or (Nu, Np)
            y_net: shape (Ny, 2)
            s_truth: shape (Ny,)

        Returns:
            shape () or (Nu,
        """
        ssim = self.ssimilarity_fn(u_guess, y_net, s_truth)
        return (1 - ssim) / 2.0


class InverseLossLineOfSightVelocity:
    def __init__(
        self,
        job_v_r: InversePred,
        job_v_theta: InversePred,
        loss_metric: str,
        nr=None,
        ntheta=None,
    ):
        self.job_v_r = job_v_r
        self.job_v_theta = job_v_theta
        self.loss_metric = loss_metric
        self.nr = nr if nr else int(self.job_v_r.fargo_setups["ny"])
        self.ntheta = ntheta if ntheta else int(self.job_v_r.fargo_setups["nx"])

        # for backward compatibility
        self.vmap_inv_pred_fn = self.inv_pred_fn
        self.pmap_inv_pred_fn = jax.pmap(self.inv_pred_fn, in_axes=(0, None))
        self.loss_metrics = {
            "l2": self.l2_fn,
            "mse": self.mse_fn,
            "ssim": self.ssimilarity_fn,
            "dssim": self.dissimilarity_fn,
        }
        self.inv_loss_fn = self.loss_metrics[self.loss_metric]
        self.vmap_inv_loss_fn = self.inv_loss_fn
        self.pmap_inv_loss_fn = jax.pmap(self.inv_loss_fn, in_axes=(0, None, None))
        self.args = {
            "u_min": self.job_v_r.args["u_min"],
            "u_max": self.job_v_r.args["u_max"],
        }

    @functools.partial(jax.jit, static_argnums=(0,))
    def inv_pred_fn(self, u_guess: chex.Array, y_net: chex.Array):
        pred_v_r = self.job_v_r.inv_pred_fn(u_guess, y_net)
        pred_v_theta = self.job_v_theta.inv_pred_fn(u_guess, y_net)
        pred_v_los = pred_v_r * jnp.cos(y_net[:, 1]) - pred_v_theta * jnp.sin(
            y_net[:, 1]
        )
        return pred_v_los

    @functools.partial(jax.jit, static_argnums=(0,))
    def l2_fn(
        self, u_guess: chex.Array, y_net: chex.Array, s_truth: chex.Array
    ) -> chex.Array:
        pred_v_los = self.inv_pred_fn(u_guess, y_net)

        pred_v_los = v_los_remove_radial_background(pred_v_los, y_net)
        s_truth = v_los_remove_radial_background(s_truth, y_net)

        loss = jnp.sqrt(jnp.nanmean((pred_v_los - s_truth) ** 2, axis=-1)) / jnp.sqrt(
            jnp.nanmean(s_truth**2)
        )

        return loss

    @partial(jax.jit, static_argnums=(0,))
    def mse_fn(
        self, u_guess: chex.Array, y_net: chex.Array, s_truth: chex.Array
    ) -> chex.Array:
        """

        Args:
            u_guess: shape (Np,) or (Nu, Np)
            y_net: shape (Ny, 2)
            s_truth: shape (Ny,)

        Returns:
            shape () or (Nu,)
        """
        pred = self.inv_pred_fn(u_guess, y_net)
        """pred: shape (Ny,) or (Nu, Ny,)"""

        loss = jnp.nanmean((pred - s_truth) ** 2, axis=-1)
        return loss

    @partial(jax.jit, static_argnums=(0,))
    def ssimilarity_fn(
        self, u_guess: chex.Array, y_net: chex.Array, s_truth: chex.Array
    ) -> chex.Array:
        """The loss function when the metric == "ssim".

        Args:
            u_guess: shape (Np,) or (Nu, Np)
            y_net: shape (Ny, 2)
            s_truth: shape (Ny,)

        Returns:
            shape () or (Nu,)
        """
        pred = self.inv_pred_fn(u_guess, y_net)
        """pred: shape (Ny,) or (Nu, Ny,)"""

        pred = v_los_remove_radial_background(pred, y_net)
        s_truth = v_los_remove_radial_background(s_truth, y_net)

        # reshape to (nr, ntheta)
        shape = pred.shape[:-1] + (self.nr, self.ntheta)
        pred = pred.reshape(*shape)
        s_truth = s_truth.reshape(self.nr, self.ntheta)

        if pred.ndim > 2:
            ssim = jax.vmap(cal_ssim, in_axes=(0, None))(pred, s_truth)
        else:
            ssim = cal_ssim(pred, s_truth)
        return ssim

    @partial(jax.jit, static_argnums=(0,))
    def dissimilarity_fn(
        self, u_guess: chex.Array, y_net: chex.Array, s_truth: chex.Array
    ) -> chex.Array:
        """The loss function when the metric == "ssim".

        Args:
            u_guess: shape (Np,) or (Nu, Np)
            y_net: shape (Ny, 2)
            s_truth: shape (Ny,)

        Returns:
            shape () or (Nu,
        """
        ssim = self.ssimilarity_fn(u_guess, y_net, s_truth)
        return (1 - ssim) / 2.0


@jax.jit
def v_los_remove_radial_background(v_los: chex.Array, y_net: chex.Array):
    """Remove the background of the line-of-sight velocity.

    Args:
        v_los: shape (Nu, Ny,)
        y_net: shape (Ny, 2)

    Returns:

    """
    background = -((y_net[:, 0] / R_UNIT) ** -0.5) * jnp.sin(y_net[:, 1])
    v_los = v_los - background
    return v_los


# todo test
@jax.jit
def cal_ssim(pred, truth):
    """

    Args:
        pred: (nr, ntheta)
        truth: (nr, ntheta)

    Returns:

    """
    vmax = jnp.nanmax(jnp.array([jnp.nanmax(truth), jnp.nanmax(pred)]))
    # ssim's inputs should have rank at least 3
    ssim_map = dm_pix.ssim(
        pred[..., jnp.newaxis], truth[..., jnp.newaxis], max_val=vmax, return_map=True
    )
    return jnp.nanmean(ssim_map, tuple(range(-3, 0)))


def log_to_dataset(
    log: dict,
    run_id: str,
    param_labels=("ALPHA", "ASPECTRATIO", "PLANETMASS", "r_p", "theta_p"),
    metrics: tuple = METRICS,
) -> xr.Dataset:
    """Convert the logging storage to xarray dataset."""

    # check data completeness
    necessary_keys = [
        "gen_counter",
        "log_gen_1",
        "log_gen_mean",
        "log_gen_std",
        "log_top_1",
        "log_top_mean",
        "log_top_std",
        "top_fitness",
        "top_params",
    ]
    if not all([key in log.keys() for key in necessary_keys]):
        raise ValueError(f"Log is missing some keys. Necessary keys: {necessary_keys}")
    # we do not need manually check the compatibility of the shape of the arrays. They will be checked when dataset is created.
    num_gens = log["log_gen_1"].shape[0]
    top_k, num_dims = log["top_params"].shape
    # create dataset
    datadict = {
        "log_gen_1": (["gen"], log["log_gen_1"]),
        "log_gen_mean": (["gen"], log["log_gen_mean"]),
        "log_gen_std": (["gen"], log["log_gen_std"]),
        "log_top_1": (["gen"], log["log_top_1"]),
        "log_top_mean": (["gen"], log["log_top_mean"]),
        "log_top_std": (["gen"], log["log_top_std"]),
        "top_fitness": (["top_k"], log["top_fitness"]),
        "top_params": (["top_k", "params"], log["top_params"]),
    }
    # add metrics
    for metric in metrics:
        datadict[metric] = (["gen"], log[metric])
    if "log_params" in log.keys():
        datadict["log_params"] = (["gen", "params"], log["log_params"])
    coords = {
        "gen": np.arange(num_gens),
        "top_k": np.arange(top_k),
        "params": list(param_labels),
    }
    if "log_all_params" in log.keys():
        datadict["log_all_params"] = (["gen", "pop", "params"], log["log_all_params"])
        datadict["all_fitness"] = (["gen", "pop"], log["all_fitness"])
        # we might do not need number the particles in the population, because they are not sorted and thus do not have a meaning
        coords["pop"] = np.arange(log["log_all_params"].shape[1])
    log_dataset = xr.Dataset(
        datadict,
        coords=coords,
        attrs={"run": run_id},
    )
    return log_dataset


def get_v_los_job(
    v_r_network_id: str,
    v_theta_network_id: str,
    network_root_dir: str,
    args_file: str,
    arg_groups_file: str,
    fargo_setup_file: str,
    loss_metric: str,
    nr: int = None,
    ntheta: int = None,
) -> InverseLossLineOfSightVelocity:
    jobs = []
    for network_id in [v_r_network_id, v_theta_network_id]:
        run_dir = onet_disk2D.utils.match_run_dir(network_root_dir, network_id)
        job_args = onet_disk2D.run.load_job_args(
            run_dir,
            args_file,
            arg_groups_file,
            fargo_setup_file,
        )
        job = InversePred(job_args)
        job.load_model(run_dir)
        jobs.append(job)
    job_v_los = InverseLossLineOfSightVelocity(
        job_v_r=jobs[0],
        job_v_theta=jobs[1],
        loss_metric=loss_metric,
        nr=nr,
        ntheta=ntheta,
    )

    return job_v_los


def get_job_and_dataarray(
    network_id: str,
    network_root_dir: str,
    args_file: str,
    arg_groups_file: str,
    fargo_setup_file: str,
    metric: str,
    data_dir: Union[str, pathlib.Path],
) -> Tuple[Union[InverseLoss, InverseLossLineOfSightVelocity], xr.DataArray]:
    if network_id == "v_los_model":
        job = get_v_los_job(
            "v_r_model",
            "v_theta_model",
            network_root_dir=network_root_dir,
            args_file=args_file,
            arg_groups_file=arg_groups_file,
            fargo_setup_file=fargo_setup_file,
            loss_metric=metric,
        )
        dataarray = load_data_array(
            data_dir=data_dir,
            unknown_type="v_los",
            unknown="v_los",
        )
        return job, dataarray
    run_dir = onet_disk2D.utils.match_run_dir(network_root_dir, network_id)
    job_args = onet_disk2D.run.load_job_args(
        run_dir,
        args_file,
        arg_groups_file,
        fargo_setup_file,
    )
    job = InverseLoss(job_args, loss_metric=metric)
    job.load_model(run_dir)

    dataarray = load_data_array(
        data_dir=data_dir,
        unknown_type=job.unknown_type,
        unknown=job.args["unknown"],
    )
    return job, dataarray


def get_job_and_data_obj(
    run_id: str,
    network_id: str,
    network_root_dir: str,
    args_file: str,
    arg_groups_file: str,
    fargo_setup_file: str,
    metric: str,
    data_dir: Union[str, pathlib.Path],
) -> Tuple[Union[InverseLoss, InverseLossLineOfSightVelocity], Data]:
    job, dataarray = get_job_and_dataarray(
        network_id,
        network_root_dir,
        args_file,
        arg_groups_file,
        fargo_setup_file,
        metric,
        data_dir,
    )
    data = dataarray.sel(run=run_id)
    data["r"] = data["r"] * data["r_p"]
    data = data.transpose("r", "theta")
    data_obj = Data(data)
    return job, data_obj


def get_job_and_dataarray_from_args(
    network_id: str, args: argparse.Namespace
) -> Tuple[InverseLoss, xr.DataArray]:
    # check if args_file in args
    args_file = args.args_file if hasattr(args, "args_file") else "args.yml"
    arg_groups_file = (
        args.arg_groups_file if hasattr(args, "arg_groups_file") else "arg_groups.yml"
    )
    fargo_setup_file = (
        args.fargo_setup_file
        if hasattr(args, "fargo_setup_file")
        else "fargo_setups.yml"
    )
    data_dir = onet_disk2D.utils.match_run_dir(args.data_root_dir, args.dataset_id)

    job, dataarray = get_job_and_dataarray(
        network_id,
        network_root_dir=args.network_root_dir,
        args_file=args_file,
        arg_groups_file=arg_groups_file,
        fargo_setup_file=fargo_setup_file,
        metric=args.metric,
        data_dir=data_dir,
    )
    return job, dataarray


def get_job_and_data_obj_from_args(
    network_id: str, args: argparse.Namespace
) -> Tuple[Union[InverseLoss, InverseLossLineOfSightVelocity], Data]:
    # check if args_file in args
    args_file = args.args_file if hasattr(args, "args_file") else "args.yml"
    arg_groups_file = (
        args.arg_groups_file if hasattr(args, "arg_groups_file") else "arg_groups.yml"
    )
    fargo_setup_file = (
        args.fargo_setup_file
        if hasattr(args, "fargo_setup_file")
        else "fargo_setups.yml"
    )
    data_dir = onet_disk2D.utils.match_run_dir(args.data_root_dir, args.dataset_id)
    # the line below is to make it compatible with inv_loss_surface.py
    run_id = args.run_id if hasattr(args, "run_id") else args.fargo_run_id

    job, data_obj = get_job_and_data_obj(
        run_id=run_id,
        network_id=network_id,
        network_root_dir=args.network_root_dir,
        args_file=args_file,
        arg_groups_file=arg_groups_file,
        fargo_setup_file=fargo_setup_file,
        metric=args.metric,
        data_dir=data_dir,
    )
    return job, data_obj


def get_jobs_and_dataarrays_from_args(
    args: argparse.Namespace,
) -> Tuple[List[InverseLoss], List[xr.DataArray]]:
    jobs = []
    dataarrays = []
    for net_id in args.network_id:
        job, dataarray = get_job_and_dataarray_from_args(net_id, args)
        jobs.append(job)
        dataarrays.append(dataarray)
    return jobs, dataarrays


def get_jobs_and_data_objs_from_args(
    args: argparse.Namespace,
) -> Tuple[List[Union[InverseLoss, InverseLossLineOfSightVelocity]], List[Data]]:
    jobs = []
    data_objs = []
    for net_id in args.network_id:
        job, data_obj = get_job_and_data_obj_from_args(net_id, args)
        jobs.append(job)
        data_objs.append(data_obj)
    return jobs, data_objs
