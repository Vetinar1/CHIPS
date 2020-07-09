import os
import numpy as np
import pandas as pd
from cloudy_optimizer import load_all_points_from_cache

def prod(iterable):
    """
    Return the product of all elements of the iterable. Analogous to sum().
    """
    prod = 1
    for i in iterable:
        prod *= i

    return prod


def get_folder_size(folder):
    """
    Returns size of folder in bytes. Does not consider subdirectories, links, etc.
    https://stackoverflow.com/a/1392549

    :param folder:
    :return:
    """
    return sum(os.path.getsize(f) for f in os.listdir(folder) if os.path.isfile(f))


def simple_print(var):
    """
    Custom print with no line separator, analogous to file.write().

    :param var:         Variable to be printed.
    """
    print(var, sep="")


def get_normalization_transform(min, max):
    """
    returns two functions, one to transform points in an interval linearly on [0, 1]
    and one to transform back
    """
    def transform(x):
        return (x - min) / abs(max - min)

    def inv_transform(x):
        return x * abs(max - min) + min

    return transform, inv_transform


def get_pruning_function(dims):
    """
    Returns a function to prune an array of points to an Nd cuboid.

    :param dims:    List: [[min, max], [min, max], ...]
                    len(list) <= points.shape[1]!
                    min_i < max_i!
    :return:        pruning function
    """
    def prune(points):
        mask = np.ones(points.shape[0], dtype=bool)
        for i, dim in enumerate(dims):
            mask[(points[:, i] < dim[0]) | (points[:, i] > dim[1])] = 0

        return points[mask]

    return prune


def drop_all_rows_containing_nan(array):
    """
    Returns a copy of the array that lacks all rows that originally contained a np.nan value.

    :param array:
    :return:
    """
    # https://stackoverflow.com/questions/22032668/numpy-drop-rows-with-all-nan-or-0-values
    return array[np.all(~np.isnan(array), axis=1)]


def compile_to_dataframe(num_coords, folder):
    """
    Loads all points from the files in the given folder and returns them as a dataframe.

    :param folder:
    :return:
    """
    dim_names = ["Temperature", "n_H", "Metallicity", "Redshift"]
    points = load_all_points_from_cache(num_coords, folder)
    df_points = pd.DataFrame(
        points,
        columns=[dim_names[i] for i in range(num_coords)] + ["Value"]
    )

    return df_points


def seconds_to_human_readable(seconds):
    """
    Convert a number of seconds into a human readable format (string) like Xh Ym Zs

    :param seconds:
    :return:
    """
    hours = (seconds - (seconds % 3600)) / 3600
    seconds_no_hours = seconds % 3600
    minutes = (seconds_no_hours - (seconds_no_hours % 60)) / 60
    seconds_left = seconds_no_hours % 60

    return str(int(hours)) + "h " + str(int(minutes)) + "m " + str(int(seconds_left)) + "s"