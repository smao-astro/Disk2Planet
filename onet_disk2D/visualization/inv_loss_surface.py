import argparse
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import tqdm
import xarray as xr

import onet_disk2D.run.inverse_job
import onet_disk2D.transformed_subset_creator
import onet_disk2D.utils
import onet_disk2D.visualization.loss_surface as draw_loss_surface


def get_parser():
    parser = argparse.ArgumentParser(description="Parameters for the script")
    parser.add_argument("--network_root_dir", type=str, help="Network Root Directory")
    parser.add_argument(
        "--network_id", type=onet_disk2D.utils.list_of_str, help="Network ID"
    )
    parser.add_argument("--data_root_dir", type=str, help="Data Root Directory")
    parser.add_argument("--dataset_id", type=str, help="Dataset ID")
    parser.add_argument("--fargo_run_id", type=str, help="Fargo Run ID")
    parser.add_argument(
        "--opt_guild_dir",
        type=str,
        help="The guild run dir that keeps the results of optimization.",
    )
    parser.add_argument("--opt_run_id", type=str)
    # one of keys from "alpha", "h0", "q", "r_p", "theta_p"
    parser.add_argument("--var1", type=str)
    parser.add_argument("--var2", type=str)
    parser.add_argument("--nvar1", type=int, default=10)
    parser.add_argument("--nvar2", type=int, default=10)
    parser.add_argument(
        "--metric", choices=onet_disk2D.run.inverse_job.METRICS, type=str
    )
    parser.add_argument("--save_dir", type=str, required=True, help="Save Directory")
    return parser


def main(n_cpu=1):
    """TODO: Warning: this function is hard to maintain and debug, when you have time, please refactor it."""
    args = get_parser().parse_args()
    # load model (job)
    jobs, data_objs = onet_disk2D.run.inverse_job.get_jobs_and_data_objs_from_args(args)
    # for shared information
    job = jobs[0]
    data_obj = data_objs[0]

    if args.save_dir:
        save_dir = pathlib.Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    elif args.opt_guild_dir is not None and args.opt_run_id is not None:
        save_dir = onet_disk2D.utils.match_run_dir(args.opt_guild_dir, args.opt_run_id)
    else:
        raise ValueError("save_dir or opt_guild_dir and opt_run_id must be provided.")
    print(f"save_dir: {save_dir}")

    # ground truth parameters
    keys_short = ["alpha", "h0", "q", "r_p", "theta_p"]
    for k, v in zip(keys_short, data_obj.params_truth):
        print(f"{k}_truth: {v:.3g}")

    # ground truth score
    truth_score = float(
        np.mean(
            [
                job.inv_loss_fn(data_obj.params_truth, data_obj.y_net, data_obj.truth)
                for job, data_obj in zip(jobs, data_objs)
            ]
        )
    )
    print(f"truth_score: {truth_score:.3g}")

    # grid
    # if log not None, input parameters should use the optimized parameters
    # load optimization result
    try:
        log = onet_disk2D.visualization.load_log(args.opt_guild_dir, args.opt_run_id)
    except (TypeError, FileNotFoundError, NotADirectoryError):
        # the input values are the ground truth values, except for the x and y
        coords = {
            k: np.array(v).reshape((1,))
            for k, v in zip(keys_short, data_obj.params_truth)
        }
    else:
        coords = {
            k: np.array(v).reshape((1,))
            for k, v in zip(keys_short, log["log_params"].values[-1])
        }
    idx_has_data = np.any(~np.isnan(data_obj.data.values), axis=1)
    r_min = data_obj.data.r.values[idx_has_data].min()
    r_max = data_obj.data.r.values[idx_has_data].max()
    print(f"r_min={r_min:.1f}, r_max={r_max:.1f}")
    vmin = job.args["u_min"] + [r_min, -np.pi]
    vmax = job.args["u_max"] + [r_max, np.pi]
    for var_name, n_sample in zip([args.var1, args.var2], [args.nvar1, args.nvar2]):
        index = keys_short.index(var_name)
        axis_min = vmin[index]
        axis_max = vmax[index]
        if var_name in ["alpha", "q"]:
            coords[var_name] = np.logspace(axis_min, axis_max, n_sample)
        else:
            coords[var_name] = np.linspace(axis_min, axis_max, n_sample)

    inputs = np.stack(np.meshgrid(*coords.values(), indexing="ij"), axis=-1)
    """shape: (n_alpha, n_h0, n_q, n_r_p, n_theta_p, 5)"""
    inputs_shape = inputs.shape[:-1]
    inputs = jnp.array(inputs)

    inputs = inputs.reshape((-1, 5))
    n_inputs = inputs.shape[0]

    # split inputs for parallelization over CPUs
    if jax.devices()[0].platform == "cpu":
        print("Using CPU, Parallelizing...")
        n_splits = int(np.ceil(n_inputs / n_cpu))
        inputs = jnp.array_split(inputs, n_splits)
        loss = []
        for ipt in tqdm.tqdm(inputs):
            losses = jnp.array(
                [
                    job.pmap_inv_loss_fn(ipt, data_obj.y_net, data_obj.truth)
                    for job, data_obj in zip(jobs, data_objs)
                ]
            )
            loss.append(jnp.mean(losses, axis=0))
    else:
        print("Using GPU")
        n_splits = onet_disk2D.utils.find_minimum_splits_maximum_memory_usage(
            job.vmap_inv_loss_fn, inputs, data_obj.y_net, data_obj.truth
        )
        inputs = jnp.array_split(inputs, n_splits)
        loss = []
        for ipt in tqdm.tqdm(inputs):
            # print(x.shape)
            losses = jnp.array(
                [
                    job.vmap_inv_loss_fn(ipt, data_obj.y_net, data_obj.truth)
                    for job, data_obj in zip(jobs, data_objs)
                ]
            )
            loss.append(jnp.mean(losses, axis=0))
        loss = np.concatenate(loss)

    loss = jnp.array(loss).reshape(inputs_shape)

    # prepare attrs
    attrs = {k + "_truth": v for k, v in zip(keys_short, data_obj.params_truth)}
    attrs["fargo_run_id"] = args.fargo_run_id
    attrs["truth_score"] = truth_score
    if args.opt_run_id is not None:
        attrs["opt_run_id"] = args.opt_run_id
    loss = xr.DataArray(
        loss,
        coords=coords,
        dims=keys_short,
        attrs=attrs,
    )
    # save
    print(f"Saving loss surface data to {save_dir}")
    loss.to_netcdf(save_dir / "loss_surf.nc")

    fig, y_key, x_key = draw_loss_surface.draw_loss_surface(
        loss,
        minimize=(False if args.metric == "ssim" else True),
        min_marker_config=dict(color="green", symbol="x", size=10),
    )
    if (args.var1 != y_key) or (args.var2 != x_key):
        raise ValueError(
            f"args.var2 ({args.var2}) != x_key ({x_key}) or args.var1 ({args.var1}) != y_key ({y_key})"
        )
    # add ground truth
    fig = draw_loss_surface.add_ground_truth2loss_surface(
        loss, fig, y_key=y_key, x_key=x_key
    )
    # increase font size
    fig.update_layout(font=dict(size=24))
    # fig.show()
    print(f"Saving loss surface plot to {save_dir}")
    fig.write_html(save_dir / f"loss.html")
