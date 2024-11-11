import jax.numpy as jnp
import numpy as np
import xarray as xr

import onet_disk2D.utils


def test_to_log_1():
    u = jnp.array([1.0, 10.0, 100.0])
    col_idx_to_apply = jnp.array([True, False, True])
    u = onet_disk2D.utils.to_log(u, col_idx_to_apply)
    assert jnp.allclose(u, jnp.array([0.0, 10.0, 2.0]))


def test_to_log_2():
    u = jnp.array([1.0, 10.0, 100.0])
    u = jnp.stack([u, u, u], axis=0)

    col_idx_to_apply = jnp.array([True, False, True])
    log_u = jnp.array([0.0, 10.0, 2.0])
    log_u = jnp.stack([log_u, log_u, log_u], axis=0)

    u = onet_disk2D.utils.to_log(u, col_idx_to_apply)
    assert jnp.allclose(u, log_u)


def test_to_linear_1():
    u = jnp.array([0.0, 10.0, 2.0])
    col_idx_to_apply = jnp.array([True, False, True])
    u = onet_disk2D.utils.to_linear(u, col_idx_to_apply)
    assert jnp.allclose(u, jnp.array([1.0, 10.0, 100.0]))


def test_to_linear_2():
    log_u = jnp.array([0.0, 10.0, 2.0])
    log_u = jnp.stack([log_u, log_u, log_u], axis=0)

    col_idx_to_apply = jnp.array([True, False, True])
    u = jnp.array([1.0, 10.0, 100.0])
    u = jnp.stack([u, u, u], axis=0)

    log_u = onet_disk2D.utils.to_linear(log_u, col_idx_to_apply)
    assert jnp.allclose(log_u, u)


def test_rotate_dataarray():
    theta = np.array([-np.pi, -np.pi / 2, 0, np.pi / 2])
    old_data = np.sin(theta)
    old_data = xr.DataArray(old_data, dims=["theta"], coords={"theta": theta})
    # Rotate by pi/2
    delta = np.pi / 2
    # [1, 0, -1, 0]
    new_data_truth = np.sin([np.pi / 2, -np.pi, -np.pi / 2, 0])
    new_data_real = onet_disk2D.utils.rotate_dataarray(old_data, delta)
    assert np.allclose(new_data_real.values, new_data_truth)

    # Rotate by pi/4
    delta = np.pi / 4
    new_data_truth = np.array(
        [
            (np.sin(np.pi / 2) + np.sin(-np.pi)) / 2,
            (np.sin(-np.pi) + np.sin(-np.pi / 2)) / 2,
            (np.sin(-np.pi / 2) + np.sin(0)) / 2,
            (np.sin(0) + np.sin(np.pi / 2)) / 2,
        ]
    )
    new_data_real = onet_disk2D.utils.rotate_dataarray(old_data, delta)
    assert np.allclose(new_data_real.values, new_data_truth)

    # 2D data using meshgrid
    r = np.arange(4)
    old_data = r[:, None] + np.sin(theta)
    old_data = xr.DataArray(
        old_data, dims=["r", "theta"], coords={"r": r, "theta": theta}
    )
    # Rotate by pi/2
    delta = np.pi / 2
    new_data_truth = np.sin([np.pi / 2, -np.pi, -np.pi / 2, 0])
    new_data_truth = r[:, None] + new_data_truth
    new_data_real = onet_disk2D.utils.rotate_dataarray(old_data, delta)
    assert np.allclose(new_data_real.values, new_data_truth)


def test_cal_theta_mean():
    theta = jnp.array([355, 5, 15]) / 180 * jnp.pi
    mean = onet_disk2D.utils.cal_theta_mean(theta)
    expected_mean = 5 * jnp.pi / 180
    assert jnp.isclose(mean, expected_mean)
    theta = jnp.array([0, 10, 20]) / 180 * jnp.pi
    mean = onet_disk2D.utils.cal_theta_mean(theta)
    expected_mean = 10 * jnp.pi / 180
    assert jnp.isclose(mean, expected_mean)
