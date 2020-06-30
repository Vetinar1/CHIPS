import subprocess
import os
from scipy import spatial, stats
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import re
from cloudy_optimizer import initialize_points, interpolate_delaunay


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


def view_slice(points, begin, end, steps, filename=None):
    """
    Interpolate steps points on a line between begin coordinates and end coordinates. Plot results.

    :param points:      Points to use for interpolation
    :param begin:       Start coordinates of slice to interpolate over. List/Iterable.
    :param end:         End coordinates of slice to interpolate over. List/Iterable.
    :param steps:       Number of points to evenly distribute over slice for interpolation. Int.
    :param filename:    If given, save plot to file instead of displaying it directly.
    :return:
    """

    coord_interpolator = get_ndim_linear_interp(begin, end)
    interp_coordinates = coord_interpolator(np.linspace(0, 1, steps))
    interpolated = interpolate_delaunay(
        points,
        interp_coordinates
    )

    plt.plot(np.linspace(0, 1, steps), interpolated)
    plt.xlabel("x: " + str(begin) + " to " + str(end))
    plt.ylabel("Ctot")
    plt.yscale("log")
    plt.show()


if __name__ == "__main__":
    points = initialize_points(add_grid=False, cache_overwrite="run1/cache/")
    # view_slice(points)

    interp = get_ndim_linear_interp([5, 9, 3], [20, 14, 4])
    points = interp(np.linspace(0, 1, 100))
    print(points)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(points[:,0], points[:,1], points[:,2])
    fig.show()