"""
This script is used to create the transformed data from one fargo case.
The transformation will create a disk with a planet at any position, specificed by the user.
"""
import argparse
import pathlib

import numpy as np
import xarray as xr

import onet_disk2D.utils


def get_parser():
    parser = argparse.ArgumentParser()
    # data dir
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument(
        "--r_scale",
        type=float,
        default=1.0,
        help="Radial scale of the simulated (transformed, or faked, whatever you call it) disk.",
    )
    parser.add_argument(
        "--planet_angle",
        type=float,
        default=0.0,
        help="planet angle in degree, measured from the x-axis",
    )
    # save dir
    parser.add_argument(
        "--save_dir",
        type=str,
        default=".",
        help="If not specified, save to current directory.",
    )
    return parser


@onet_disk2D.utils.deprecated
def generate_transformed_data(dataarray_, r_scale_, angle_):
    """

    Args:
        dataarray_: with selected run_id
        r_scale_:
        angle_: in degree

    Returns:

    """
    # from angle to radian
    if not (-180 <= angle_ <= 180):
        raise ValueError("angle_ must be in [-180, 180] degree.")
    angle_ = angle_ * np.pi / 180.0
    transformed_data_ = onet_disk2D.utils.rotate_dataarray(dataarray_, angle_)

    # fake the radial distance
    # do we really need to do so?
    # for the fargo3d data, r_min=0.4, r_p=1.0, r_max=2.5
    # if we get an image range from r_min to r_max, and we guess the planet is at r_p_guess (r_min < r_p_guess < r_max), then we can always scale all coordinates by r_p_guess, then we get the image that range from r_min/r_p_guess to r_max/r_p_guess, and the planet is at 1.0
    # anyway, the design will allow this, but all r_p should be the same to combine the results into one dataarray.
    transformed_data_["r"] = transformed_data_["r"] * r_scale_

    # record r_scale, theta_p, and parameters to the dataarray
    transformed_data_.attrs["r_p"] = r_scale_
    transformed_data_.attrs["theta_p"] = angle_

    return transformed_data_


@onet_disk2D.utils.deprecated
def generate_transformed_data_without_rotation(dataarray_, r_scale_, angle_):
    """

    Args:
        dataarray_: with selected run_id
        r_scale_:
        angle_: in degree

    Returns:

    """
    # from angle to radian
    if not (-180 <= angle_ <= 180):
        raise ValueError("angle_ must be in [-180, 180] degree.")
    angle_ = angle_ * np.pi / 180.0
    # transformed_data_ = onet_disk2D.utils.rotate_dataarray(dataarray_, angle_)
    transformed_data_ = dataarray_.copy()

    # fake the radial distance
    # do we really need to do so?
    # for the fargo3d data, r_min=0.4, r_p=1.0, r_max=2.5
    # if we get an image range from r_min to r_max, and we guess the planet is at r_p_guess (r_min < r_p_guess < r_max), then we can always scale all coordinates by r_p_guess, then we get the image that range from r_min/r_p_guess to r_max/r_p_guess, and the planet is at 1.0
    # anyway, the design will allow this, but all r_p should be the same to combine the results into one dataarray.
    transformed_data_["r"] = transformed_data_["r"] * r_scale_

    # record r_scale, theta_p, and parameters to the dataarray
    transformed_data_.attrs["r_p"] = r_scale_
    transformed_data_.attrs["theta_p"] = angle_

    return transformed_data_


if __name__ == "__main__":
    args = get_parser().parse_args()
    data_dir = pathlib.Path(args.data_dir)
    save_dir = pathlib.Path(args.save_dir)

    for data_file in [
        "batch_truth_sigma.nc",
        "batch_truth_v_r.nc",
        "batch_truth_v_theta.nc",
    ]:
        dataarray = xr.open_dataarray(data_dir / data_file).sel(run=args.run_id)
        transformed_data = generate_transformed_data(
            dataarray, args.r_scale, args.planet_angle
        )
        # save the data
        # give a new name to differentiate from the original data
        transformed_data.to_netcdf(save_dir / ("transformed_" + data_file))

        dataarray.close()
