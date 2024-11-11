import argparse
import os
import tqdm

n_cpu = 8
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={n_cpu}"

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import onet_disk2D.run.inverse_job
import onet_disk2D.utils
import onet_disk2D.visualization.cma_multi_images

DATASET_ID_LS = ["37dbc978", "4a621e47", "8f2c6356", "02c8ecf1"]
RUN_DIR_ID_LS = [
    "sigma",
    "sigma_noise",
    "sigma_r_crop",
    "sigma_theta_crop",
]
MJ = 0.000955


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network_root_dir", type=str, required=True)
    parser.add_argument("--network_id", type=str, required=True)
    parser.add_argument("--data_root_dir", type=str, required=True)
    parser.add_argument("--guild_run_id", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=80)
    args = parser.parse_args()
    return args


def get_scores(
    parameters,
    q,
    u_truth,
    data_root_dir,
    guild_run_id,
    network_id,
    network_root_dir,
) -> xr.DataArray:
    score_ls = []
    # tqdm

    for i, (dataset_id) in enumerate(tqdm.tqdm(DATASET_ID_LS)):
        data_dir = onet_disk2D.utils.match_run_dir(data_root_dir, dataset_id)
        job, data_obj = onet_disk2D.run.inverse_job.get_job_and_data_obj(
            guild_run_id,
            network_id,
            network_root_dir,
            args_file="args.yml",
            arg_groups_file="arg_groups.yml",
            fargo_setup_file="fargo_setups.yml",
            metric="l2",
            data_dir=data_dir,
        )

        parameters = parameters.reshape((-1, 8, 5))
        score = [
            job.pmap_inv_loss_fn(parameters_subset, data_obj.y_net, data_obj.truth)
            for parameters_subset in parameters
        ]
        score = np.array(score).flatten()

        score_ls.append(score)

    scores = xr.DataArray(
        np.array(score_ls),
        dims=("run_dir_id", "q"),
        coords={"run_dir_id": RUN_DIR_ID_LS, "q": q},
        attrs={
            "guild_run_id": guild_run_id,
            "alpha_truth": u_truth[0],
            "h0_truth": u_truth[1],
            "q_truth": u_truth[2],
            "r_p_truth": u_truth[3],
            "theta_p_truth": u_truth[4],
        },
    )

    return scores


def draw(
    q: np.ndarray,
    q_truth: float,
    scores: xr.DataArray,
):
    # matplotlib font size set to times new roman
    plt.rcParams["font.family"] = "Times New Roman"
    # plot
    fontsize = 14
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
    for i, (score, run_dir_id) in enumerate(zip(scores, RUN_DIR_ID_LS)):
        ax.plot(q / MJ, score, label=run_dir_id)
    # vertical line for q_truth
    ax.axvline(q_truth / MJ, color="black", linestyle="--", label="truth")
    plt.legend(fontsize=fontsize)
    # ax.set_xscale("log")
    plt.xlabel(r"$\mathrm{M}_\mathrm{p}/\mathrm{M}_\mathrm{J}$", fontsize=fontsize)
    plt.ylabel("Score", fontsize=fontsize)
    return fig


def main():
    args = get_args()

    # rst_ls = [guild_cma_opt.GuildCMARst(name=key) for key in RUN_DIR_ID_LS]

    data_dir = onet_disk2D.utils.match_run_dir(args.data_root_dir, "37dbc978")
    job, data_obj = onet_disk2D.run.inverse_job.get_job_and_data_obj(
        args.guild_run_id,
        args.network_id,
        args.network_root_dir,
        args_file="args.yml",
        arg_groups_file="arg_groups.yml",
        fargo_setup_file="fargo_setups.yml",
        metric="l2",
        data_dir=data_dir,
    )
    u_truth = data_obj.params_truth
    q = np.logspace(job.args["u_min"][2], job.args["u_max"][2], 80)
    parameters = np.full((80, 5), u_truth)
    parameters[:, 2] = q

    scores = get_scores(
        parameters,
        q,
        u_truth,
        args.data_root_dir,
        args.guild_run_id,
        args.network_id,
        args.network_root_dir,
    )
    # save scores
    scores.to_netcdf(os.path.join(args.output_dir, "q-l2-scores.nc"))

    fig = draw(q, u_truth[2], scores)
    fig.savefig(
        os.path.join(args.output_dir, "q-l2-scores.png"), dpi=300, bbox_inches="tight"
    )


if __name__ == "__main__":
    main()
