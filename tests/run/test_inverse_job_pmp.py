NETWORK_ROOT_DIR = "/Users/kyika/project/pinn/onet-disk2D-single/cedar/pm_al_ar_fung_gap2steady4_2/runs/"
NETWORK_ID = "c7f3f6da"
import os

n_cpu = 8
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={n_cpu}"
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import onet_disk2D.run
import onet_disk2D.run.inverse_job
import onet_disk2D.utils


@pytest.fixture
def job():
    run_dir = onet_disk2D.utils.match_run_dir(NETWORK_ROOT_DIR, NETWORK_ID)
    job_args = onet_disk2D.run.load_job_args(
        run_dir,
        "args.yml",
        "arg_groups.yml",
        "fargo_setups.yml",
    )
    job = onet_disk2D.run.inverse_job.InverseLoss(
        job_args,
        loss_metric="ssim",
        nr=2,
        ntheta=7,
    )
    job.load_model(run_dir)
    return job


@pytest.fixture
def params():
    return jnp.array([1e-2, 0.1, 1e-3, 1.0, 0.0])


@pytest.fixture
def u(params):
    return jnp.vstack([params] * 8)


@pytest.fixture
def r():
    return jnp.linspace(0.4, 2.5, 2)


@pytest.fixture
def theta():
    return jnp.linspace(-jnp.pi, jnp.pi, 7)


@pytest.fixture
def y_net(r, theta):
    r_grid, theta_grid = jnp.meshgrid(r, theta, indexing="ij")
    return jnp.stack([r_grid.flatten(), theta_grid.flatten()], axis=1)


@pytest.fixture
def truth(y_net):
    return y_net[:, 0] * jnp.cos(y_net[:, 1])


def test_pmap_inv_pred_fn(job, u, y_net):
    pred = job.pmap_inv_pred_fn(u, y_net)
    assert pred.shape == (8, 14)


def test_pmap_inv_loss_fn(job, u, y_net, truth):
    loss = job.pmap_inv_loss_fn(u, y_net, truth)
    assert loss.shape == (8,)
