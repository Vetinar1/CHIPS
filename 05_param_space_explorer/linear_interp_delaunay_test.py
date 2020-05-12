from scipy import spatial, stats
import numpy as np
import itertools
import matplotlib.pyplot as plt
import time


def get_test_point(x, y):
    return np.cos(x*y) ** 2


def IDW(points, values, x, exp=1):
    """
    Simple inverse distance weighting interpolation.

    :param points:      Iterable of points to use for weighting, shape (npoints, ncoord +1)
    :param values:      Values at points, 1d array
    :param x:           coordinates of point to interpolate at
    :param exp:         Exponent of weighting function; 1/2 = euclidean, 1 = squarded euclidean etc
    :return:
    """
    # Equalize shapes
    x = np.reshape(x, (1, x.shape[0]))
    x = np.repeat(x, points.shape[0], axis=0)

    # Distances -> weights
    weights = 1/np.sum( np.power((points - x)**2, exp), axis=1)
    #assert(weights.shape[0] == points.shape[0])

    interp = sum([weights[i] * values[i] for i in range(weights.shape[0])]) / np.sum(weights)

    if interp > 1:
        print("Warning! Bad interpolation:", interp)
        print(weights)
        print(values)
        print()

    return interp


plt.rcParams['axes.facecolor'] = (0.5, 0.5, 0.5)

point_generation = "random"

if point_generation == "grid":
    x, y = np.meshgrid(
        np.arange(-2.2, 2.2, 0.1), # go over boundaries to avoid edge effects
        np.arange(-2.2, 2.2, 0.1)
    )

    x = x.flatten()
    y = y.flatten()

    x = np.reshape(x, (x.shape[0], 1))
    y = np.reshape(y, (y.shape[0], 1))

    points = np.hstack((x, y))
    points = np.hstack(
        (points, get_test_point(x, y))
    )
elif point_generation == "random":
    x = 4.2*np.random.random(1000) - 2.1 # Again go over edges
    y = 4.2*np.random.random(1000) - 2.1

    x = np.reshape(x, (x.shape[0], 1))
    y = np.reshape(y, (y.shape[0], 1))

    points = np.hstack((x, y))
    points = np.hstack(
        (points, get_test_point(x, y))
    )
else:
    raise RuntimeError("invalid point_generation")

plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.scatter(points[:, 0], points[:, 1], c=points[:, 2], marker=".", s=0.5, cmap="jet")
plt.colorbar()
plt.clim(0, 1)
plt.show()



tri = spatial.Delaunay(
    points[:, :-1]
)

tree = spatial.cKDTree(
    points[:, :-1]
)

x, y = np.meshgrid(
    np.arange(-2, 2, 0.05),
    np.arange(-2, 2, 0.05)
)

x = x.flatten()
y = y.flatten()

x = np.reshape(x, (x.shape[0], 1))
y = np.reshape(y, (y.shape[0], 1))

fine_points = np.hstack((x, y))
values = []

interp_method = "IDW"

if interp_method == "delaunay":
    for i, p in enumerate(fine_points):
        simplex_index = tri.find_simplex(p)
        simplex = tri.simplices[simplex_index]

        # Because we only look at one point, trafo has shape (ndim+1, ndim) (instead of (1, ndim+1, ndim)
        # contains inverse transformation matrix and the simplex vertex belonging to this matrix
        trafo = tri.transform[simplex_index]

        n = len(p)
        barycentric_coords = trafo[:n, :n].dot(p - trafo[n, :])
        # Add missing last (dependent) coordinate
        barycentric_coords = np.concatenate((barycentric_coords, np.array([1-np.sum(barycentric_coords)])))

        # The interpolated value is the values at the simplex vertices, weighted by the barycentric coordinates
        values.append(np.dot(barycentric_coords, points[simplex, -1]))
elif interp_method == "IDW":
    for i, p in enumerate(fine_points):
        _, p_neigh = tree.query(
            p,
            4
        )

        p_neigh = list(p_neigh)  # list of indices including orig point
        #p_neigh.remove(i)  # list of indices of neighbors
        p_neigh = points[p_neigh]  # list of points (+ values)

        values.append(IDW(p_neigh[:, :-1], p_neigh[:, -1], p))
else:
    raise RuntimeError("invalid interp method")


plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.scatter(fine_points[:, 0], fine_points[:, 1], c=values, marker=".", s=0.5, cmap="jet")
plt.colorbar()
plt.clim(0, 1)
plt.show()


diffs = np.array(get_test_point(fine_points[:,0], fine_points[:,1])) - np.array(values)

plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.scatter(fine_points[:, 0], fine_points[:, 1], c=diffs, marker=".", s=0.5, cmap="nipy_spectral")
plt.colorbar()
plt.clim(0, 0.2)
plt.show()

diffs = diffs[diffs > 0.01]
print(len(diffs), "/", len(values))