import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import onet_disk2D.summary.guild_cma_opt as guild_cma_opt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True)
    args = parser.parse_args()
    return args


def main(args: argparse.Namespace):
    name_append_ls = [
        "sigma_v_r_v_theta",
        "sigma_v_los",
        "sigma",
        "sigma_noise_v_los_noise",
        "sigma_noise",
        "sigma_r_crop",
        "sigma_theta_crop",
    ]

    parameter_list = ["alpha", "h0", "q", "r_p", "theta_p"]

    rst_dict = {
        name_append: guild_cma_opt.GuildCMARst(name=name_append)
        for name_append in name_append_ls
    }

    # draw the error histogram, with len(name_append_ls) * len(parameter_list) subplots
    fig, axes = plt.subplots(
        len(name_append_ls),
        len(parameter_list),
        figsize=(20, 20),
        # row spacing and column spacing
        gridspec_kw={"hspace": 0.33, "wspace": 0.2},
    )
    for i, (name, ax_row) in enumerate(zip(name_append_ls, axes)):
        for j, (parameter, ax) in enumerate(zip(parameter_list, ax_row)):
            plt.sca(ax)
            sns.histplot(
                rst_dict[name].df_min[f"{parameter}_diff"],
                label="hist",
                stat="density",
                color="yellow",
            )
            vanilla_mean = rst_dict[name].df_vanilla_mean[f"{parameter}_diff"]
            vanilla_std = rst_dict[name].df_vanilla_std[f"{parameter}_diff"]
            x0 = rst_dict[name].df_median[f"{parameter}_diff"]
            sigma = rst_dict[name].df_robust_sigma[f"{parameter}_diff"]
            x = np.linspace(
                x0 - 10 * sigma, x0 + 10 * sigma, 300
            )
            ax.plot(
                x,
                1
                / (vanilla_std * np.sqrt(2 * np.pi))
                * np.exp(-((x - vanilla_mean) ** 2) / (2 * vanilla_std**2)),
                "--",
                label="vanilla",
            )
            ax.plot(
                x,
                1
                / (sigma * np.sqrt(2 * np.pi))
                * np.exp(-((x - x0) ** 2) / (2 * sigma**2)),
                "-.",
                label="robust",
            )
            # Adding labels and title
            ax.set_title(f"{parameter} x0={x0:.1e}, sigma={sigma:.1e}")
            if j == 0:
                ax.set_ylabel(f"{name}")
            else:
                ax.set_ylabel("")
            ax.set_xlabel("")
            ax.set_xlim(x0 - 10 * sigma, x0 + 10 * sigma)
            ax.legend()
    # save the figure
    plt.savefig(
        os.path.join(args.save_dir, "cma_error_hist.png"),
        dpi=300,
    )


if __name__ == "__main__":
    args = get_args()
    main(args)
