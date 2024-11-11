import pathlib
from typing import Union, Tuple

import numpy as np
import pandas as pd

from . import load_log


def log_summary(
    guild_run_dir: Union[str, pathlib.Path], df: pd.DataFrame, length: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Summarize multiple `log.nc` files into one dataset.

    Args:
        guild_run_dir:
        df:
        length:

    Returns:

    """
    scores = []
    params = []
    truths = []
    for _, row in df.iterrows():
        job_id = row["run"]
        log = load_log(guild_run_dir, job_id)
        score = log["log_top_1"].values
        score = np.pad(score, (0, length - len(score)), constant_values=np.nan)
        scores.append(score)
        param = log["log_params"]
        param = np.pad(
            param, ((0, length - len(param)), (0, 0)), constant_values=np.nan
        )
        params.append(param)
        truth = log["truth_params"]
        truths.append(truth)

    scores = np.array(scores)
    params = np.array(params)
    truths = np.array(truths)

    return scores, params, truths
