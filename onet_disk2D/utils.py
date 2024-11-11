import argparse
import os
import pathlib
import functools
from typing import Union, Iterable

import chex
import jax
import jax.numpy as jnp
import jaxlib.xla_extension
import numpy as np
import scipy.interpolate as si
import xarray as xr
import yaml
import warnings

def deprecated(func):
    """This is a decorator which can be used to mark functions as deprecated.
    It will result in a warning being emitted when the function is used."""
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.warn(f"Call to deprecated function {func.__name__}.",
                      category=DeprecationWarning,
                      stacklevel=2)
        return func(*args, **kwargs)
    return new_func

@jax.jit
def to_log(u: chex.Array, col_idx_to_apply: chex.Array) -> chex.Array:
    """Convert the second axis of u to log scale.

    Args:
        u: shape (n_samples, n_features)
        col_idx_to_apply: jax boolean array of which column to apply the transformation. E.g. [True, False, True] for the first and third columns.

    Returns:

    """

    def apply_log_transform(col, apply_log):
        return jnp.where(apply_log, jnp.log10(col), col)

    u = jax.vmap(apply_log_transform, in_axes=(-1, 0), out_axes=-1)(u, col_idx_to_apply)

    return u


@jax.jit
def to_linear(u: chex.Array, col_idx_to_apply: chex.Array) -> chex.Array:
    """Convert the columns of u to linear scale.

    Args:
        u:
        col_idx_to_apply: jax boolean array of which column to apply the transformation. E.g. [True, False, True] for the first and third columns.

    """

    def apply_log_transform(col, apply_log):
        return jnp.where(apply_log, 10.0**col, col)

    u = jax.vmap(apply_log_transform, in_axes=(-1, 0), out_axes=-1)(u, col_idx_to_apply)

    return u


@functools.partial(jax.jit, static_argnums=(1,))
def circle_samples_to_polar_ring(x_and_y: chex.Array, r_min: float):
    x, y = x_and_y[..., 0], x_and_y[..., 1]
    r = jnp.sqrt(x**2 + y**2) + r_min
    theta = jnp.arctan2(y, x)
    return jnp.stack([r, theta], axis=-1)


@functools.partial(jax.jit, static_argnums=(1,))
def inputs_to_polar_ring(inputs: chex.Array, r_min: float):
    x_and_y = inputs[..., 3:]
    r_and_theta = circle_samples_to_polar_ring(x_and_y, r_min)
    inputs = inputs.at[..., 3:].set(r_and_theta)
    return inputs


def cal_theta_mean(theta, axis=-1):
    """Calculate the theta mean.

    Args:
        theta: shape (..., n_samples,)

    Returns:
        theta_mean: shape (...,)

    References:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.circmean.html

    """
    mean = jnp.arctan2(
        jnp.mean(jnp.sin(theta), axis=axis), jnp.mean(jnp.cos(theta), axis=axis)
    )
    # constrain theta_mean to [-pi, pi). Side note: I am not sure whether this is necessary and good practice.
    mean = jnp.mod(mean + jnp.pi, 2 * jnp.pi) - jnp.pi
    return mean


@jax.jit
def cal_circular_mean(array):
    """Calculate the circular mean among batch/population.

    The last axis is the coordinates axis: (u_i, r, theta).

    Args:
        array: shape (..., popsize, Nu+2)

    Returns:
        array: shape (..., Nu+2)

    """
    u_and_r, theta = array[..., :-1], array[..., -1:]
    # mean over population
    u_and_r_mean = jnp.mean(u_and_r, axis=-2)
    """shape (..., Nu+1)"""
    # circular mean over population
    theta_mean = cal_theta_mean(theta, axis=-2)
    """shape (..., 1)"""
    return jnp.concatenate([u_and_r_mean, theta_mean], axis=-1)


@jax.jit
def cal_circular_diff(x, y):
    """Calculate the circular difference of an array."""
    u_and_r_diff = x[..., :-1] - y[..., :-1]
    theta_diff = x[..., -1:] - y[..., -1:]
    # constrain theta_diff to [-pi, pi).
    theta_diff = jnp.mod(theta_diff + jnp.pi, 2 * jnp.pi) - jnp.pi
    return jnp.concatenate([u_and_r_diff, theta_diff], axis=-1)


@jax.jit
def cal_circular_clip(array, clip_min, clip_max):
    """Clip radius and theta of an array.

    Clip u_and_r to [clip_min, clip_max] and theta to [-pi, pi).
    """
    u_and_r, theta = array[..., :-1], array[..., -1:]
    u_and_r = jnp.clip(u_and_r, clip_min, clip_max)
    theta = jnp.mod(theta + jnp.pi, 2 * jnp.pi) - jnp.pi
    return jnp.concatenate([u_and_r, theta], axis=-1)


def rotate_dataarray(data_array, delta):
    """
    Rotates a given xarray DataArray along the theta dimension by a given angle delta.

    Args:
        data_array (xr.DataArray): Input xarray DataArray with the last dimension being the theta dimension.
        delta (float): Rotation angle in radians (-pi to pi).

    Notes:
        Theoretically the rotation angle can be different among fargo cases (along run dimension), but that would cause the outputs have different theta grid, thus we can not stack them together to one DataArray. Additional care must be taken.

    Returns:
        xr.DataArray: Rotated xarray DataArray.
    """
    # Check if the data_array has a theta dimension
    if "theta" not in data_array.dims:
        raise ValueError("Input data_array must have a 'theta' dimension.")

    # Ensure delta is within the valid range
    if not -np.pi <= delta <= np.pi:
        raise ValueError("Rotation angle 'delta' must be in the range [-pi, pi].")
    # convert delta to range [-pi, pi). I am not sure if this is necessary, but might be safer.
    delta = np.mod(delta + np.pi, 2 * np.pi) - np.pi

    # sort the theta dimension
    data_array = data_array.sortby("theta")

    values = data_array.values
    theta = data_array.theta.values
    # the last dimension is the theta dimension

    # shifted theta values, limit to [-pi, pi)
    theta_shifted = np.mod(theta + delta + np.pi, 2 * np.pi) - np.pi
    # sort values and theta_shifted by theta_shifted
    idx = np.argsort(theta_shifted)
    theta_shifted = theta_shifted[idx]
    values = values[..., idx]

    # pad the values array with first and last value to avoid interpolation errors at the edges
    theta_shifted = np.concatenate(
        [
            # in range [-3pi, -pi)
            np.mod(theta_shifted[-1:] + np.pi, 2 * np.pi) - 3 * np.pi,
            theta_shifted,
            # in range [pi, 3pi)
            np.mod(theta_shifted[:1] + np.pi, 2 * np.pi) + np.pi,
        ]
    )
    values = np.concatenate((values[..., -1:], values, values[..., :1]), axis=-1)

    # Interpolate the data_array along the theta dimension to theta
    # the last dimension is the theta dimension
    values = si.interp1d(theta_shifted, values, axis=-1)(theta)

    # assemble the new data_array
    values = xr.DataArray(
        values, dims=data_array.dims, coords=data_array.coords, attrs=data_array.attrs
    )
    return values


def xarray_polar_to_cartesian(data_array: xr.DataArray, x, y):
    x = np.array(x)
    y = np.array(y)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D array.")

    # y is outer axis, x is inner axis
    xgrid, ygrid = np.meshgrid(x, y, indexing="xy")
    # pseudo dimension
    r = xr.DataArray(np.sqrt(xgrid.flatten() ** 2 + ygrid.flatten() ** 2), dims="i")
    theta = xr.DataArray(np.arctan2(ygrid.flatten(), xgrid.flatten()), dims="i")

    coords = {k: data_array[k] for k in set(data_array.coords) - {"r", "theta"}}
    coords.update({"y": ("y", y), "x": ("x", x)})

    data_array = data_array.interp({"r": r, "theta": theta}).values

    if data_array.ndim == 2:
        data_array = xr.DataArray(
            data_array.reshape((-1, len(y), len(x))),
            coords=coords,
            dims=["run", "y", "x"],
        )
    else:
        data_array = xr.DataArray(
            data_array.reshape((len(y), len(x))),
            coords=coords,
            dims=["y", "x"],
        )

    return data_array


def xarray_cartesian_to_polar(data_array: xr.DataArray, r, theta):
    r = np.array(r)
    theta = np.array(theta)
    if r.ndim != 1 or theta.ndim != 1:
        raise ValueError("r and theta must be 1D array.")

    # theta is outer axis, r is inner axis
    rgrid, thetagrid = np.meshgrid(r, theta, indexing="ij")
    # pseudo dimension
    x = xr.DataArray(rgrid.flatten() * np.cos(thetagrid.flatten()), dims="i")
    y = xr.DataArray(rgrid.flatten() * np.sin(thetagrid.flatten()), dims="i")

    coords = {k: data_array[k] for k in set(data_array.coords) - {"x", "y"}}
    coords.update({"r": ("r", r), "theta": ("theta", theta)})

    data_array = data_array.interp({"x": x, "y": y}).values

    if data_array.ndim == 2:
        data_array = xr.DataArray(
            data_array.reshape((-1, len(r), len(theta))),
            coords=coords,
            dims=["run", "r", "theta"],
        )
    else:
        data_array = xr.DataArray(
            data_array.reshape((len(r), len(theta))),
            coords=coords,
            dims=["r", "theta"],
        )

    return data_array


def match_run_dir(runs_dir: Union[str, pathlib.Path], run_id: str) -> pathlib.PosixPath:
    runs_dir = pathlib.Path(runs_dir)
    run_dir = list(runs_dir.glob(run_id + "*"))
    if len(run_dir) == 1:
        return run_dir[0]
    else:
        raise ValueError


def load_args_from_yaml(inv_args_file: Union[str, pathlib.Path]):
    inv_args_file = pathlib.Path(inv_args_file)
    with inv_args_file.open("r") as f:
        args_loaded = yaml.safe_load(f)
    # to argparse.Namespace
    args_loaded = argparse.Namespace(**args_loaded)
    return args_loaded


def save_args_to_yaml(args_to_save: argparse.Namespace, inv_args_file="inv_args.yml"):
    save_path = pathlib.Path(os.path.join(args_to_save.save_dir, inv_args_file))
    args_to_save = vars(args_to_save)
    with save_path.open("w") as f:
        yaml.safe_dump(args_to_save, f)


def get_y_from_faked_data(data_):
    """

    r changes along the first dimension, theta changes along the second dimension. r unit: AU, theta unit: radian.

    Returns:
        shape (n, 2) array, where n = len(r) * len(theta)
    """
    r = data_.r.values
    theta = data_.theta.values
    # grid, r change along the first dimension, theta change along the second dimension
    rgrid, thetagrid = jnp.meshgrid(r, theta, indexing="ij")
    y = jnp.stack([rgrid, thetagrid], axis=-1).reshape((-1, 2))
    return y


@deprecated
def check_faked_data(data):
    """
    data should include "r_p" and "theta_p" (ground truth values) in attributes.
    """
    for key in ["r_p", "theta_p"]:
        if key not in data.attrs:
            raise ValueError(f"data should have {key} as attribute.")


def restart_check_args(
    args_: argparse.Namespace, args_old_: argparse.Namespace, keys_to_check: Iterable
):
    print("====== Check args ======")
    for key_ in args_.__dict__.keys():
        current_value = getattr(args_, key_)
        old_value = getattr(args_old_, key_)
        if (key_ in keys_to_check) and (current_value != old_value):
            raise ValueError(
                f"key {key_} is different between args and args_old: {current_value} vs {old_value}"
            )
        else:
            print(f"{key_}: {current_value}")
    print("====== Check args ======")


def find_minimum_splits_maximum_memory_usage(f, x, *args, **kwargs):
    """
    Find the minimum number of splits such that the memory usage is below the maximum memory usage.
    """
    i = 0
    while (n_splits := 2**i) < (len_ := len(x)):
        try:
            n = len_ // n_splits
            print(f"n_splits={n_splits}, n={n}")
            _ = f(x[:n], *args, **kwargs)
        except jaxlib.xla_extension.XlaRuntimeError:
            print(f"n_splits={n_splits} failed")
            i += 1
            continue
        else:
            print(f"n_splits={n_splits} succeed")
            break
    return n_splits


def list_of_str(raw_inputs: str) -> list[str]:
    return [value for value in raw_inputs.split(",")]
