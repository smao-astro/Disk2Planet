import argparse
import pathlib

import cma
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from scipy.stats import qmc

import onet_disk2D.run
import onet_disk2D.run.inverse_job
import onet_disk2D.utils
from onet_disk2D.utils import list_of_str

# we do not need the line below since cma-es runs on CPU
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".25"

NUM_DIMS = 5
VARIABLES = ["sigma", "v_r", "v_theta"]
PARAMETER_NAMES = ("alpha", "h0", "q", "r_p", "theta_p")
LONG_PARAMETER_NAMES = ["ALPHA", "ASPECTRATIO", "PLANETMASS", "r_p", "theta_p"]
METRIC_NAMES = ("dssim", "l2")
col_idx_to_apply = jnp.array([True, False, True, False, False])


class tqdm_skopt(object):
    def __init__(self, **kwargs):
        self._bar = tqdm.tqdm(**kwargs)

    def __call__(self, res):
        self._bar.update()


def get_parser():
    parser = argparse.ArgumentParser()
    # IO
    # Network
    parser.add_argument(
        "--network_root_dir",
        type=str,
        required=True,
        help="The root directory of ML models.",
    )
    parser.add_argument(
        "--network_id",
        type=list_of_str,
        required=True,
        help="The short/full id of the trained network.",
    )
    parser.add_argument(
        "--args_file",
        type=str,
        default="args.yml",
        help="file that logs training args.",
    )
    parser.add_argument("--arg_groups_file", type=str, default="arg_groups.yml")
    parser.add_argument("--fargo_setup_file", type=str, default="fargo_setups.yml")
    # Data
    parser.add_argument(
        "--data_root_dir",
        type=str,
        required=True,
        help="directory of fargo:collect_xarrays output",
    )
    parser.add_argument(
        "--dataset_id",
        type=str,
        required=True,
        help="The short/full id of the fargo simulation.",
    )
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument(
        "--restart_dir",
        type=str,
        default="",
        help="Directory that store restart files.",
    )
    parser.add_argument("--metric", type=str, choices=METRIC_NAMES)
    # optimization algorithm
    parser.add_argument(
        "--x0_search",
        type=str,
        choices=["center", "sobol_no_scramble", "sobol_scramble"],
        default="sobol_no_scramble",
    )
    parser.add_argument("--init_sample_m", type=int, default=10)
    parser.add_argument("--sigma0", type=float, default=0.3)
    parser.add_argument("--std_alpha", type=float, default=1)
    parser.add_argument("--std_h0", type=float, default=1)
    parser.add_argument("--std_q", type=float, default=1)
    parser.add_argument("--std_r_p", type=float, default=1)
    parser.add_argument("--std_theta_p", type=float, default=1)
    parser.add_argument("--popsize", type=int, default=32)
    parser.add_argument("--maxiter", type=int, default=256)
    parser.add_argument("--save_every", type=int, default=4)
    parser.add_argument(
        "--key", type=int, default=99, help="key for random number generator"
    )
    return parser


import jax.numpy as jnp


def compute_mean_scores(jobs, data_objs, p, n_splits):
    """
    Compute the mean scores for given jobs and data objects by splitting the p array.

    Parameters:
    - jobs: List of jobs.
    - data_objs: Corresponding data objects for the jobs.
    - p: Array to split.
    - n_splits: Number of splits for the p array.

    Returns:
    - Mean scores as a numpy array.
    """

    scores_list = []

    for job, data_obj in zip(jobs, data_objs):
        split_inputs = jnp.array_split(p, n_splits)
        score = jnp.concatenate(
            [
                job.vmap_inv_loss_fn(ipt, data_obj.y_net, data_obj.truth)
                for ipt in split_inputs
            ]
        )
        scores_list.append(score)

    return jnp.mean(jnp.stack(scores_list), axis=0)


# Usage:
# mean_scores = compute_mean_scores(jobs, data_objs, p, n_splits)


if __name__ == "__main__":
    args = get_parser().parse_args()
    # todo implement restart
    if args.restart_dir:
        raise NotImplementedError
    save_dir = pathlib.Path(args.save_dir).resolve()

    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    # save args
    onet_disk2D.utils.save_args_to_yaml(args, inv_args_file="inv_args.yml")

    # load model (job)
    jobs, data_objs = onet_disk2D.run.inverse_job.get_jobs_and_data_objs_from_args(args)
    job = jobs[0]
    data_obj = data_objs[0]

    idx_has_data = np.any(~np.isnan(data_obj.data.values), axis=1)
    r_min = data_obj.data.r.values[idx_has_data].min()
    r_max = data_obj.data.r.values[idx_has_data].max()
    print(f"r_min={r_min:.1f}, r_max={r_max:.1f}")
    params_min = np.array(job.args["u_min"] + [r_min, -np.pi])
    """log: True, False, True, False, False"""
    params_max = np.array(job.args["u_max"] + [r_max, np.pi])
    """log: True, False, True, False, False"""

    # figure out n_splits
    if jax.devices()[0].platform == "cpu":
        n_splits = 2
    else:
        n_splits = None

    # determine x0
    if args.x0_search == "center":
        x0 = np.array([0.5] * NUM_DIMS)
    elif args.x0_search.startswith("sobol"):
        if args.x0_search == "sobol_no_scramble":
            scramble = False
        elif args.x0_search == "sobol_scramble":
            scramble = True
        else:
            raise ValueError(f"Unknown x0_search: {args.x0_search}")
        init_x = qmc.Sobol(d=NUM_DIMS, scramble=scramble).random_base2(
            m=args.init_sample_m
        )
        init_p = init_x * (params_max - params_min) + params_min
        init_p = onet_disk2D.utils.to_linear(
            jnp.array(init_p), col_idx_to_apply=col_idx_to_apply
        )
        if n_splits is None:
            n_splits_init = onet_disk2D.utils.find_minimum_splits_maximum_memory_usage(
                job.vmap_inv_loss_fn, init_p, data_obj.y_net, data_obj.truth
            )
            n_splits = args.popsize // (len(init_p) // n_splits_init)
            print(f"n_splits_init={n_splits_init}, n_splits={n_splits}")
        else:
            n_splits_init = n_splits
        init_solutions = compute_mean_scores(jobs, data_objs, init_p, n_splits_init)
        # select the minimum
        x0 = init_x[np.argmin(init_solutions)]
    else:
        raise ValueError(f"Unknown x0_search: {args.x0_search}")
    print("x0 is \n", x0.tolist())

    es = cma.CMAEvolutionStrategy(
        x0=x0.tolist(),
        sigma0=args.sigma0,
        inopts={
            "seed": args.key,
            "popsize": args.popsize,
            "bounds": [0, 1],
            "CMA_stds": [
                args.std_alpha,
                args.std_h0,
                args.std_q,
                args.std_r_p,
                args.std_theta_p,
            ],
            "maxiter": args.maxiter,
            "verb_filenameprefix": (save_dir / "cmaes").as_posix() + "/",
            "verb_disp": 5,
            "verb_log": args.save_every,
        },
    )

    while not es.stop():
        x = es.ask()
        # convert solutions to linear scale
        p = np.array(x) * (params_max - params_min) + params_min
        p = onet_disk2D.utils.to_linear(jnp.array(p), col_idx_to_apply=col_idx_to_apply)
        if n_splits is None:
            n_splits = onet_disk2D.utils.find_minimum_splits_maximum_memory_usage(
                job.vmap_inv_loss_fn, p, data_obj.y_net, data_obj.truth
            )
            print(f"n_splits={n_splits}")
        scores = compute_mean_scores(jobs, data_objs, p, n_splits)
        es.tell(x, scores.tolist())
        es.logger.add()
        es.disp()
    es.result_pretty()
    es.logger.plot()
    cma.s.figsave((save_dir / "cmalog.png").as_posix())

    print(f"loss: {es.result.fbest:.3g}")
    pbest = np.array(es.result.xbest) * (params_max - params_min) + params_min
    pbest = onet_disk2D.utils.to_linear(
        jnp.array(pbest), col_idx_to_apply=col_idx_to_apply
    )
    for i, short_name in enumerate(PARAMETER_NAMES, start=0):
        truth = data_objs[0].params_truth[i]
        pred = pbest[i]
        print(f"{short_name}_truth: {truth}")
        print(f"{short_name}_pred: {pred}")
