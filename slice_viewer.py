import subprocess
import os
from scipy import spatial, stats
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import re
from cloudy_optimizer import *


def get_ndim_linear_interp(point1, point2):
    point1 = np.array(point1)
    point2 = np.array(point2)

    def ndim_linear_interp(t):
        """

        :param t:       in [0, 1]
        :return:
        """
        return np.array([point1 + t * (point2 - point1) for t in list(t)])

    return ndim_linear_interp


def view_slice(points, begin, end, steps, filename=None, title=None):
    """
    Interpolate steps points on a line between begin coordinates and end coordinates. Plot results.

    :param points:      Points to use for interpolation
    :param begin:       Start coordinates of slice to interpolate over. List/Iterable.
    :param end:         End coordinates of slice to interpolate over. List/Iterable.
    :param steps:       Number of points to evenly distribute over slice for interpolation. Int.
    :param filename:    If given, save plot to file instead of displaying it directly.
    :return:
    """

    # coord interpolator gets the coordinates along the line via linear interpolation
    coord_interpolator = get_ndim_linear_interp(begin, end)

    # interp coordinates are these coordinates along the line
    interp_coordinates = coord_interpolator(np.linspace(0, 1, steps))

    # use the delaunay interpolator to get the values at the interp coordinates
    interpolated = interpolate_delaunay(
        points,
        interp_coordinates
    )[0]

    plt.plot(np.linspace(0, 1, steps), interpolated[:,-1])
    if title:
        plt.title(title)
    plt.xlabel("x: " + str(begin) + " to " + str(end))
    plt.ylabel("Ctot")
    plt.yscale("log")
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    points = load_all_points_from_cache(cache_folder="run4/cache/")
    for i in range(-3, 3):
        view_slice(points, [2.01, i], [5.99, i], 100, title=r"$n_H = " + str(i) + "$", filename="n_H_" + str(i))