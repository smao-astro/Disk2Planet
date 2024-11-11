import base64
from io import BytesIO

import xarray as xr

import onet_disk2D.utils


def mpl_to_uri(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return "data:image/png;base64,{}".format(
        base64.b64encode(buf.getvalue()).decode("utf-8")
    )


def load_log(guild_run_dir, job_id):
    log = xr.load_dataset(
        onet_disk2D.utils.match_run_dir(guild_run_dir, job_id) / "log.nc"
    )
    return log
