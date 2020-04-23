from scipy import spatial
import numpy as np
import itertools
import matplotlib.pyplot as plt

def prod(iterable):
    prod = 1
    for i in iterable:
        prod *= i

    return prod


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




def explore_parameter_space(
        dimensions,
        get_new_point,
        threshold=0.05,
        point_threshold=0.05,
        random_new_points=10,
        max_diff=None
):
    """

    :param dimensions:      List of list
                            [[start, stop, number], ...}
    :param get_new_point:   Function to get new point in space. Must match dimensions
    :param threshold:       Relative maximum threshold for points to differ from analytic RIGHT NOW ABSOLUTE
    :param point_threshold: Relative number of points who may be over threshold for interpolation to quit
    :param random_new_points: How many random new points to add each iteration to avoid "local optimum"
    :param: max_diff:       Maximum difference any point may have from an analytic
    :return:
    """
    # Establish initial grid
    axes = []
    shape = []
    volume = 1
    for dim in dimensions:
        volume *= abs(dim[0] - dim[1])
        axes.append(np.linspace(*dim))
        shape.append(dim[-1])

    # Shape: N points, M coordinates + 1 value
    points = np.zeros((prod(shape), len(shape)+1))

    for i, comb in enumerate(itertools.product(*axes)):
        points[i] = np.array([*comb, get_new_point(*comb)])

    finished = False
    count = 0

    while not finished:
        finished = True
        count += 1
        print(count)
        point_count = points.shape[0]
        print("Points this iteration:", point_count)
        thresh_points = 0

        # Build k-d tree
        tree = spatial.cKDTree(
            points[:, :-1],             # exclude value column
            copy_data=True              # Just a precaution while using small trees
        )

        diffs = []
        orig_points = points[:] #help, hacks


        # TODO: Optimal way of finding neighbors?
        # -> Looking in a d+1 dimensional ball with a radius slightly bigger than the distances in the original grid
        #    guarantees finding neighbors in all dimensions. not efficient in later iterations.
        #    is finding neighbors in all dimensions important?
        for i, p in enumerate(points[:]): # enumerate over copy so we can modify original
            skip = False
            for j, dim in enumerate(dimensions):
                if not (dim[0] < p[j] < dim[1] or dim[1] < p[j] < dim[0]):
                    # do not consider this point for interpolation
                    skip = True
                    break
            if skip:
                continue


            # Determine neighbors of point
            _, p_neigh = tree.query(
                p[:-1],
                4
            )

            p_neigh = list(p_neigh)     # list of indices including orig point
            p_neigh.remove(i)           # list of indices of neighbors
            p_neigh = points[p_neigh]   # list of points (+ values)

            interp_value = IDW(p_neigh[:, :-1], p_neigh[:, -1], p[:-1])
            # print("Analytic: ", points[i, -1], "\tInterpolated", interp_value, "\tDiff", points[i, -1] - interp_value)

            # Check if we need to add more points in this area
            # If yes, add point halfway between all neighbors
            #loss = interp_value**2 - p[-1]**2
            if abs(interp_value - p[-1]) >= threshold * abs(p[-1]):
                diffs.append(interp_value - p[-1])
                thresh_points += 1
                finished = False
                x = np.reshape(p, (1, p.shape[0]))
                x = np.repeat(x, p_neigh.shape[0], axis=0)
                midpoints = (x + p_neigh) / 2# + np.random.normal(0, 0.05, p_neigh.shape)
                for c in range(midpoints.shape[1]):
                    # columns
                    midpoints[:,c] += np.random.normal(
                        0,
                        np.sqrt(volume / points.shape[0]), #np.std(midpoints[:,c]),
                        midpoints.shape[0]
                    )
                midpoints[:, -1] = get_new_point(midpoints[:, 0], midpoints[:, 1])
                points = np.vstack((points, midpoints))
            else:
                diffs.append(0)

        print("Points not fulfilling condition this iteration:", thresh_points)
        print("Relative:", thresh_points/point_count)

        if thresh_points/point_count < point_threshold:
            finished = True

        #print(points)
        plt.scatter(points[:,0], points[:,1], c=points[:,2], marker=".", s=0.5)
        plt.colorbar()
        plt.show()

        print("Max diff", max(diffs))
        print()

        if max(diffs) > max_diff:
            finished = False

        for j in range(random_new_points):
            coord = [(dim[1] - dim[0]) * np.random.random_sample() + dim[0] for dim in dimensions]
            points = np.vstack(
                (
                    points,
                    np.array(coord + [get_new_point(*coord)])
                )
            )

            #print(coord + [get_new_point(*coord)])

        if count > 30:
            finished = True



explore_parameter_space(
    [
        [-2, 2, 11],
        [-2, 2, 11]
    ],
    get_test_point,
    threshold=0.05,
    point_threshold=0.05,
    random_new_points=20,
    max_diff=999
)
