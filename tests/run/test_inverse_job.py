NETWORK_ROOT_DIR = "/Users/kyika/project/pinn/onet-disk2D-single/cedar/pm_al_ar_fung_gap2steady4_2/runs/"
DATA_DIR = "/Users/kyika/project/pinn/fargo_utils/tmp_rotate_and_stretch"
RUN_ID = "b38f0766"
NETWORK_ID = ("log_sigma_model", "v_r_model", "v_theta_model", "v_los_model")
METRICS = ("l2", "mse", "ssim", "dssim")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import onet_disk2D.run
import onet_disk2D.run.inverse_job
import onet_disk2D.utils
import matplotlib.pyplot as plt
import onet_disk2D.transformed_subset_creator
import itertools

# todo test that prediction also converted to the unit above, so that the background matches the data

@pytest.fixture
def log_sigma_dataarray():
    return onet_disk2D.run.inverse_job.load_data_array(
        DATA_DIR, unknown_type="sigma", unknown="log_sigma"
    )


@pytest.fixture
def v_theta_dataarray():
    return onet_disk2D.run.inverse_job.load_data_array(
        DATA_DIR, unknown_type="v_theta", unknown="v_theta"
    )


def test_load_data_array_v_theta(v_theta_dataarray):
    is_positive = v_theta_dataarray.data > 0.0
    assert np.all(is_positive)


def test_log_sigma_profile(log_sigma_dataarray):
    data = log_sigma_dataarray.sel(run=RUN_ID)
    plt.plot(data.r * data["r_p"].values, data.mean("theta"))
    plt.xlabel("r [AU]")
    plt.show()
    assert True


def test_log_sigma_remove_radial_background():
    pass


def test_sigma_remove_radial_background():
    pass


def test_azimuthal_vel_remove_radial_background():
    pass


def test_transform_back_inputs():
    # planet position candidate: (r, theta) = (3.0, pi/2)
    u_guess = np.array(
        [[1e-2, 0.1, 1e-3, 3.0, np.pi / 2], [1e-2, 0.1, 1e-3, 2.0, -np.pi / 2]]
    )
    # coordinate grid, though we only selected two points
    y_net = np.array([[3.0, np.pi / 2], [2.0, -np.pi / 2]])
    # we transfer the data image grid to the grid in a coordinate system where planet is at (1.0, 0.0)
    inputs = onet_disk2D.run.inverse_job.transform_back_inputs(u_guess, y_net)
    new_y_net = np.array(
        [
            # after transformation, the two points should be at (1.0, 0.0) and (2.0/3.0, -pi/2)
            [[1.0, 0.0], [2.0 / 3.0, -np.pi]],
            [[3 / 2, jnp.mod(np.pi + jnp.pi, 2 * jnp.pi) - jnp.pi], [1.0, 0.0]],
        ]
    )
    cri = [
        inputs["u_net"].shape == u_guess.shape[:-1] + (3,),
        np.allclose(inputs["u_net"], u_guess[:, :3]),
        inputs["y_net"].shape == u_guess.shape[:1] + y_net.shape,
        np.allclose(inputs["y_net"], new_y_net),
    ]
    assert all(cri)


@pytest.fixture(params=list(itertools.product(NETWORK_ID, METRICS)))
def job(request):
    if (request.param[0] == "v_los_model") & (request.param[1] != "l2"):
        pytest.skip("v_los_model does not have other metrics")
    network_id = request.param[0]
    metric = request.param[1]
    if network_id == "v_los_model":
        job = onet_disk2D.run.inverse_job.get_v_los_job(
            v_r_network_id="v_r_model",
            v_theta_network_id="v_theta_model",
            network_root_dir=NETWORK_ROOT_DIR,
            args_file="args.yml",
            arg_groups_file="arg_groups.yml",
            fargo_setup_file="fargo_setups.yml",
            loss_metric=metric,
            nr=2,
            ntheta=7,
        )
    else:
        run_dir = onet_disk2D.utils.match_run_dir(NETWORK_ROOT_DIR, network_id)
        job_args = onet_disk2D.run.load_job_args(
            run_dir,
            "args.yml",
            "arg_groups.yml",
            "fargo_setups.yml",
        )
        job = onet_disk2D.run.inverse_job.InverseLoss(
            job_args,
            loss_metric=metric,
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
    return jnp.vstack([params] * 10)


@pytest.fixture
def r():
    return jnp.linspace(0.4, 2.5, 2) * onet_disk2D.run.inverse_job.R_UNIT


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


def test_y_net(y_net):
    assert y_net.shape == (14, 2)


def test_inv_pred_fn(job, params, y_net):
    pred = job.inv_pred_fn(params, y_net)
    assert pred.shape == (14,)


def test_vmap_inv_pred_fn(job, u, y_net):
    pred = job.inv_pred_fn(u, y_net)
    assert pred.shape == (10, 14)


def test_inv_loss_fn(job, params, y_net, truth):
    loss = job.inv_loss_fn(params, y_net, truth)
    assert loss.shape == ()


def test_vmap_inv_loss_fn(job, u, y_net, truth):
    loss = job.inv_loss_fn(u, y_net, truth)
    assert loss.shape == (10,)
