from scipy import spatial, stats
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

plt.rcParams['axes.facecolor'] = (0.5, 0.5, 0.5)
#plt.rcParams['image.cmap'] = 'jet'

def prod(iterable):
    prod = 1
    for i in iterable:
        prod *= i

    return prod


def cos2d_squared(x, y):
    return np.cos(x*y) ** 2


def complex_sine(x, y):
    return np.sin(x + y)


def wedge(x, y):
    return np.clip(1-2*np.abs(x), a_min=0, a_max=1)


# breaks the algorithm (doesnt terminate)
def theta(x, y):
    return np.array(x > 0).astype(int)


def exp2d(x, y):
    return np.exp(x*y)





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


        time1 = time.time()
        # Build k-d tree
        tree = spatial.cKDTree(
            points[:, :-1],             # exclude value column
            copy_data=True              # Just a precaution while using small trees
        )
        time2 = time.time()
        print(round(time2 - time1, 2), "to build tree")

        diffs = []
        orig_points = points[:] #help, hacks

        points[:, -1] = get_new_point(points[:, 0], points[:, 1])
        # A copy of points that will hold the differences instead of values
        # Inefficient but not nearly as inefficient as cloudy evaluations at each point are anyway...
        raw_diffs = points.copy()
        skipcount = 0


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
                skipcount += 1
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
            raw_diffs[i,-1] = interp_value - p[-1]
            if abs(interp_value - p[-1]) >= threshold:# * abs(p[-1]):
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

        time3 = time.time()
        print(round(time3 - time2, 2), "to check interpolation")
        print("Points skipped due to out of bounds:", skipcount)
        outlier_count = raw_diffs[raw_diffs[:,-1] >= threshold].shape[0]
        print("Points not fulfilling condition this iteration:", outlier_count, "(", round(outlier_count/point_count, 2), ")")
        print("Max diff", round(max(raw_diffs[:,-1]), 2))

        if thresh_points/point_count < point_threshold:
            finished = True

        plt.title("Samples with values")
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.scatter(points[:,0], points[:,1], c=points[:,2], marker=".", s=0.5, cmap="jet")
        plt.colorbar()
        plt.clim(0, 1)
        plt.show()
        plt.close()

        plt.title("Samples with differences")
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.scatter(raw_diffs[:, 0], raw_diffs[:, 1], c=raw_diffs[:, 2], marker=".", s=0.5, cmap="nipy_spectral")
        plt.colorbar()
        plt.clim(0, 0.1)
        plt.show()
        plt.close()

        time4 = time.time()
        print(round(time4 - time3, 2), "to plot scatterplot")
        print()

        if max(raw_diffs[:,-1]) > max_diff:
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


def explore_parameter_space_v2(
        dimensions,
        get_new_point,
        threshold=0.05,
        point_threshold=0.05,
        random_new_points=10,
        max_diff=None,
        min_dist=0.1,
        max_iterations=30
):
    """
    Version where I wanted to improve Gaussian, incomplete

    :param dimensions:      List of list
                            [[start, stop, number], ...}
    :param get_new_point:   Function to get new point in space. Must match dimensions
    :param threshold:       Relative maximum threshold for points to differ from analytic RIGHT NOW ABSOLUTE
    :param point_threshold: Relative number of points who may be over threshold for interpolation to quit
    :param random_new_points: How many random new points to add each iteration to avoid "local optimum"
    :param: max_diff:       Maximum difference any point may have from an analytic
    :param: min_dist        Minimum distance between samples to draw new points at
    :param max_iterations:  Maximum number of iterations
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

        time1 = time.time()
        # Build k-d tree
        tree = spatial.cKDTree(
            points[:, :-1],             # exclude value column
            copy_data=True              # Just a precaution while using small trees
        )
        time2 = time.time()
        print(round(time2 - time1, 2), "to build tree")


        # Contains points that are over threshold, but last element will be difference instead of value
        diff_points = np.zeros((1, points.shape[1]))
        skipcount = 0

        for i, p in enumerate(points[:]): # enumerate over copy so we can modify original
            skip = False
            for j, dim in enumerate(dimensions):
                if not (dim[0] < p[j] < dim[1] or dim[1] < p[j] < dim[0]):
                    # do not consider this point for interpolation
                    skip = True
                    skipcount += 1
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

            diff = abs(interp_value - p[-1])
            if diff >= threshold:
                diff_points = np.vstack((diff_points, p))
                diff_points[-1][-1] = diff
                finished = False

        # Remove 0 point
        diff_points = diff_points[1:]

        time3 = time.time()
        print(round(time3 - time2, 2), "to check interpolation")
        print("Points skipped due to out of bounds:", skipcount)
        print("Points not fulfilling condition this iteration:", diff_points.shape[0], "(", round(diff_points.shape[0]/point_count, 2), ")")
        print("Max diff", round(max(diff_points[:,-1]), 2))

        if diff_points.shape[0]/point_count < point_threshold:
            finished = True
            print("Number of points outside", threshold, "within", point_threshold)

        plt.scatter(points[:,0], points[:,1], c=points[:,2], marker=".", s=0.5)
        plt.colorbar()
        plt.show()

        time4 = time.time()
        print(round(time4 - time3, 2), "to plot scatterplot")

        if max(diff_points[:,-1]) > max_diff:
            finished = False

        diff_points = diff_points[diff_points[:,-1].argsort()]
        # Add samples in the neighborhoods of the 100 worst points
        for i, p in enumerate(diff_points):
            # Determine neighbors of point
            _, p_neigh = tree.query(
                p[:-1],
                4
            )

            p_neigh = list(p_neigh)     # list of indices including orig point
            p_neigh = points[p_neigh]   # list of points (+ values), not incl. original?

            x = np.reshape(p, (1, p.shape[0]))
            x = np.repeat(x, p_neigh.shape[0], axis=0)
            midpoints = (x + p_neigh) / 2
            for c in range(midpoints.shape[1]):
                # columns
                midpoints[:, c] += np.random.normal(
                    0,
                    np.sqrt(volume / points.shape[0]),  # np.std(midpoints[:,c]),
                    midpoints.shape[0]
                )
            midpoints[:, -1] = get_new_point(midpoints[:, 0], midpoints[:, 1])
            points = np.vstack((points, midpoints))

            if i >= 100:
                break

        time5 = time.time()
        print(round(time5 - time4, 2), "to sort diff_points and add new samples")

        for j in range(random_new_points):
            coord = [(dim[1] - dim[0]) * np.random.random_sample() + dim[0] for dim in dimensions]
            points = np.vstack(
                (
                    points,
                    np.array(coord + [get_new_point(*coord)])
                )
            )

        if count > max_iterations:
            finished = True
            print("Finished: Maximum number of iterations reached.")

        print()


def explore_parameter_space_v3(
        dimensions,
        get_new_point,
        threshold=0.05,
        point_threshold=0.05,
        random_new_points=10,
        max_diff=None,
        max_iterations=30
):
    """
    Version with KDE sampling

    :param dimensions:      List of list
                            [[start, stop, number], ...}
    :param get_new_point:   Function to get new point in space. Must match dimensions
    :param threshold:       Relative maximum threshold for points to differ from analytic RIGHT NOW ABSOLUTE
    :param point_threshold: Relative number of points who may be over threshold for interpolation to quit
    :param random_new_points: How many random new points to add each iteration to avoid "local optimum"
                            In this version not uniformly distributed but according to KDE
    :param: max_diff:       Maximum difference any point may have from an analytic
    :param: min_dist        Minimum distance between samples to draw new points at
    :param max_iterations:  Maximum number of iterations
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

        time1 = time.time()
        # Build k-d tree
        tree = spatial.cKDTree(
            points[:, :-1],             # exclude value column
            copy_data=True              # Just a precaution while using small trees
        )
        time2 = time.time()
        print(round(time2 - time1, 2), "to build tree")


        # A copy of points that will hold the differences instead of values
        # Inefficient but not nearly as inefficient as cloudy evaluations at each point are anyway...
        raw_diffs = points.copy()
        diffs = points.copy()
        skipcount = 0

        for i, p in enumerate(points[:]): # enumerate over copy so we can modify original
            skip = False
            for j, dim in enumerate(dimensions):
                if not (dim[0] < p[j] < dim[1] or dim[1] < p[j] < dim[0]):
                    # do not consider this point for interpolation because it is out of bounds
                    skip = True
                    skipcount += 1
                    diffs[i,-1] = 0 # Weight of this point is always 0
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

            diff = abs(interp_value - p[-1])
            raw_diffs[i,-1] = diff

            if diff >= threshold:
                # Difference times the maximum difference of value at this point to value at a neighboring point
                diffs[i,-1] = diff * max(np.abs(p[-1] - p_neigh[:,-1]))
            else:
                diffs[i,-1] = 0

        time3 = time.time()
        print(round(time3 - time2, 2), "to check interpolation")
        print("Points skipped due to out of bounds:", skipcount)
        outlier_count = raw_diffs[raw_diffs[:,-1] >= threshold].shape[0]
        print("Points not fulfilling condition this iteration:", outlier_count, "(", round(outlier_count/point_count, 2), ")")
        print("Max diff", round(max(raw_diffs[:,-1]), 2))

        if outlier_count/point_count < point_threshold:
            finished = True
            print("Number of points outside", threshold, "within", point_threshold)


        if not count % 3:
            plt.title("Samples with values")
            plt.xlim(-2, 2)
            plt.ylim(-2, 2)
            plt.scatter(points[:,0], points[:,1], c=points[:,2], marker=".", s=0.5, cmap="jet")
            plt.colorbar(cmap="jet")
            plt.clim(0, 1)
            plt.show()
            plt.close()

            plt.title("Samples with differences")
            plt.xlim(-2, 2)
            plt.ylim(-2, 2)
            plt.scatter(raw_diffs[:, 0], raw_diffs[:, 1], c=raw_diffs[:, 2], marker=".", s=0.5, cmap="nipy_spectral")
            plt.colorbar(cmap="jet")
            plt.clim(0, 1)
            plt.show()
            plt.close()

        time4 = time.time()
        print(round(time4 - time3, 2), "to plot scatterplot")

        if max(raw_diffs[:,-1]) > max_diff:
            finished = False

        # Construct kernel density estimator weighed with differences, draw new samples from it
        # Need to transpose our input first because scipy.stats.gaussian_kde is weird
        diffs = np.transpose(diffs)
        print(diffs[-1])
        print(diffs[-1]**2)
        kde = stats.gaussian_kde(
            dataset=diffs[:-1],
            bw_method=0.1,#"scott",
            weights=diffs[-1]**2
        )

        # plot kde pdf
        x, y = np.meshgrid(
            np.linspace(-2, 2, 71),
            np.linspace(-2, 2, 71)
        )

        vals = kde.evaluate(np.array([x.flatten(), y.flatten()]))

        if not count % 3:
            plt.title("KDE PDF")
            plt.xlim(-2, 2)
            plt.ylim(-2, 2)
            plt.scatter(x, y, c=vals, marker=".", s=2, cmap="jet")
            plt.colorbar()
            plt.show()
            plt.close()

        samples = kde.resample(random_new_points)
        samples = np.transpose(samples)

        time5 = time.time()
        print(round(time5 - time4, 2), "to do Kernel Density Estimation and draw new samples")

        new_points = np.zeros((samples.shape[0], samples.shape[1]+1))
        new_points[:,:-1] = samples
        new_points[:,-1] = get_new_point(new_points[:,0], new_points[:,1])

        time6 = time.time()
        print(round(time6 - time5, 2), "to evaluate new points")

        points = np.vstack((points, new_points))

        if count > max_iterations:
            finished = True
            print("Finished: Maximum number of iterations reached.")

        print()


def explore_parameter_space_delaunay(
        dimensions,
        get_new_point,
        threshold=0.05,
        point_threshold=0.05,
        random_new_points=10,
        max_diff=None
):
    """
    Like explore_parameter_space, but uses delaunay triangulation and barycentric coordinates for interpolation

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
        shape.append(dim[2])

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


        time1 = time.time()
        # Build k-d tree
        tree = spatial.cKDTree(
            points[:, :-1],             # exclude value column
            copy_data=True              # Just a precaution while using small trees
        )
        time2 = time.time()
        print(round(time2 - time1, 2), "to build tree")

        diffs = []

        points[:, -1] = get_new_point(points[:, 0], points[:, 1])
        # A copy of points that will hold the differences instead of values
        # Inefficient but not nearly as inefficient as cloudy evaluations at each point are anyway...
        raw_diffs = points.copy()
        skipcount = 0


        outside_of_tri_counter = 0
        for i, p in enumerate(points[:]): # enumerate over copy so we can modify original
            skip = False
            for j, dim in enumerate(dimensions):
                if not (dim[0] < p[j] < dim[1] or dim[1] < p[j] < dim[0]):
                    # do not consider this point for interpolation
                    skip = True
                    break
            if skip:
                skipcount += 1
                continue


            # Determine neighbors of point
            _, p_neigh = tree.query(
                p[:-1],
                4
            )

            p_neigh = list(p_neigh)     # list of indices including orig point
            p_neigh.remove(i)           # list of indices of neighbors
            p_neigh = points[p_neigh]   # list of points (+ values)

            # Construct delaunay triangles from neighbors
            # TODO: Sample neighbors in such a way that point is always within triangles if possible
            tri = spatial.Delaunay(
                p_neigh[:,:-1]
            )

            #interp_value = IDW(p_neigh[:, :-1], p_neigh[:, -1], p[:-1])
            # print("Analytic: ", points[i, -1], "\tInterpolated", interp_value, "\tDiff", points[i, -1] - interp_value)

            # Instead of interpolating from neighbors using IDW:
            # - Find simplex of delaunay triangulation containing this point
            # - Find coordinates of point in local barycentric coordinates
            # - use those for linear interpolation
            # - In case we are outside of triangulation area, use IDW as backup.
            #
            # Transformation and interpolation in general: https://en.wikipedia.org/wiki/Barycentric_coordinate_system
            # Scipy specifically: https://stackoverflow.com/a/30401693/10934472
            simplex_index = tri.find_simplex(p[:-1])
            if simplex_index == -1:
                outside_of_tri_counter += 1
                interp_value = IDW(p_neigh[:, :-1], p_neigh[:, -1], p[:-1])
            else:
                simplex = tri.simplices[simplex_index]

                # Because we only look at one point, trafo has shape (ndim+1, ndim) (instead of (1, ndim+1, ndim)
                # contains inverse transformation matrix and the simplex vertex belonging to this matrix
                trafo = tri.transform[simplex_index]

                n = len(p[:-1])
                barycentric_coords = trafo[:n, :n].dot(p[:-1] - trafo[n, :])
                # Add missing last (dependent) coordinate
                barycentric_coords = np.concatenate((barycentric_coords, np.array([1-np.sum(barycentric_coords)])))

                # The interpolated value is the values at the simplex vertices, weighted by the barycentric coordinates
                interp_value = np.dot(barycentric_coords, points[simplex, -1])





            # Check if we need to add more points in this area
            # If yes, add point halfway between all neighbors
            #loss = interp_value**2 - p[-1]**2
            raw_diffs[i,-1] = interp_value - p[-1]
            if abs(interp_value - p[-1]) >= threshold:# * abs(p[-1]):
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

        time3 = time.time()
        print(round(time3 - time2, 2), "to check interpolation")
        print("Points skipped due to out of bounds:", skipcount)
        print("Points requiring IDW fallback due to out of Delaunay Bounds:", outside_of_tri_counter)
        outlier_count = raw_diffs[raw_diffs[:,-1] >= threshold].shape[0]
        print("Points not fulfilling condition this iteration:", outlier_count, "(", round(outlier_count/point_count, 2), ")")
        print("Max diff", round(max(raw_diffs[:,-1]), 2))

        if thresh_points/point_count < point_threshold:
            finished = True

        plt.title("Samples with values")
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.scatter(points[:,0], points[:,1], c=points[:,2], marker=".", s=0.5, cmap="jet")
        plt.colorbar()
        plt.clim(0, 1)
        plt.show()
        plt.close()

        plt.title("Samples with differences")
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.scatter(raw_diffs[:, 0], raw_diffs[:, 1], c=raw_diffs[:, 2], marker=".", s=0.5, cmap="jet")#"nipy_spectral")
        plt.colorbar()
        plt.clim(0, 0.1)
        plt.show()
        plt.close()

        time4 = time.time()
        print(round(time4 - time3, 2), "to plot scatterplot")
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



def explore_parameter_space_delaunay_KDE(
        dimensions,
        get_new_point,
        threshold=0.05,
        point_threshold=0.05,
        random_new_points=1000,
        sample_multiplier=4,
        max_diff=None,
        number_of_partitions=5,
        max_iterations=30
):
    """
    This variant partitions the points into n partitions each step, then successivley triangulates using all
    partitions except one, and checks the points in the remaining partition. In the end new points are sampled
    using kernel density estimation.
    This version does not actually require a kd tree.

    :param dimensions:      List of list
                            [[start, stop, number], ...}
    :param get_new_point:   Function to get new point in space. Must match dimensions
    :param threshold:       Relative maximum threshold for points to differ from analytic RIGHT NOW ABSOLUTE
    :param point_threshold: Relative number of points who may be over threshold for interpolation to quit
    :param random_new_points: How many random new points to add each iteration to avoid "local optimum"
    :param sample_multiplier:   For each point not fulfilling the conditions, sample_multiplier new points will be
                                drawn (not necessarily in the vicinity of specific points)
    :param max_diff:        Maximum difference any point may have from an analytic
    :param number_of_partitions:    Number of partitions to divide the dataset into each iteration
                                    Equivalent to number of delaunay triangulations that are done each iteration
    :param max_iterations:  Maximum number of iterations
    :return:
    """
    # Establish initial grid
    axes = []
    shape = []
    volume = 1
    for dim in dimensions:
        volume *= abs(dim[0] - dim[1])
        axes.append(np.linspace(*dim))
        shape.append(dim[2])

    # Shape: N points, M coordinates + 1 value
    points = np.zeros((prod(shape), len(shape)+1))

    for i, comb in enumerate(itertools.product(*axes)):
        points[i] = np.array([*comb, get_new_point(*comb)])

    finished = False
    count = 0

    while not finished:
        count += 1
        print(count)
        point_count = points.shape[0]
        print("Points this iteration:", point_count)
        thresh_points = 0


        time1 = time.time()

        points[:, -1] = get_new_point(points[:, 0], points[:, 1])
        # A copy of points that will hold the differences instead of values
        # Inefficient but not nearly as inefficient as cloudy evaluations at each point are anyway...
        raw_diffs = points.copy()

        # TODO: Find diffs here
        outside_tri_counter = 0
        for i in range(number_of_partitions):
            mask = np.ones(points.shape[0], dtype=bool)
            mask[i::number_of_partitions] = False

            a = points[mask]    # all points EXCEPT some
            b = points[~mask]   # only SOME of the points

            tri = spatial.Delaunay(
                a[:, :-1]
            )

            # number of dimensions; subtract one because of value
            n = points.shape[1] - 1

            simplex_indices = tri.find_simplex(b[:, :-1])

            # we will deal with points outside triangulation later
            idw_points = b[simplex_indices == -1].copy()
            del_points = b[simplex_indices != -1].copy()

            # Cutting out indices -1
            simplex_indices_cleaned = simplex_indices[simplex_indices != -1]
            simplices = tri.simplices[simplex_indices_cleaned]

            # transforms of the points that are within triangles
            transforms = tri.transform[simplex_indices_cleaned]

            # This generalisation of the version in explore_parameter_space_delaunay is shamelessly copied from
            # https://stackoverflow.com/questions/30373912/interpolation-with-delaunay-triangulation-n-dim/30401693#30401693
            bary = np.einsum('ijk,ik->ij', transforms[:, :n, :n], del_points[:, :-1] - transforms[:, n, :])

            # the barycentric coordinates obtained this way are not complete. obtain last linearly dependent coordinate:
            weights = np.c_[bary, 1 - bary.sum(axis=1)]


            for i in range(del_points.shape[0]):
                # Attention! Overwriting previous analytic values with interpolated ones
                del_points[i, -1] = np.inner(a[simplices[i], -1], weights[i])


            # Build a tree and interpolate the remaining points using IDW
            tree = spatial.cKDTree(
                a[:, :-1]  # exclude value column
            )

            outside_tri_counter += idw_points.shape[0]

            for i, p in enumerate(idw_points):
                # Determine neighbors of point
                _, p_neigh = tree.query(
                    p[:-1],
                    4
                )

                p_neigh = list(p_neigh)  # list of indices including orig point
                # inconsistent behavior - sometimes i is in p_neigh, sometimes not
                if i in p_neigh:
                    p_neigh.remove(i)  # list of indices of neighbors
                p_neigh = points[p_neigh]  # list of points (+ values)

                idw_points[i, -1] = IDW(p_neigh[:, :-1], p_neigh[:, -1], p[:-1])

            # Write back into raw_diffs
            # This strange construct is required because of how numpy handles assignment to views/multi-masking
            raw_diffs_view = raw_diffs[~mask]
            raw_diffs_view[simplex_indices == -1, -1] -= idw_points[:, -1]
            raw_diffs_view[simplex_indices != -1, -1] -= del_points[:, -1]
            raw_diffs[~mask] = raw_diffs_view

        raw_diffs[:, -1] = np.abs(raw_diffs[:, -1])

        time3 = time.time()
        print(round(time3 - time1, 2), "to check interpolation")

        oob_mask = np.zeros(points.shape[0], dtype=bool)
        for i, dim in enumerate(dimensions):
            oob_mask[(points[:, i] < dim[0]) | (points[:, i] > dim[1])] = 1

        oob_count = points[oob_mask].shape[0]
        inbounds_count = points.shape[0] - oob_count

        print("Points skipped due to out of bounds:", oob_count)
        print("Points requiring IDW fallback: ", outside_tri_counter)
        outlier_count = raw_diffs[(~oob_mask) & (raw_diffs[:, -1] >= threshold)].shape[0]
        print("Points in bounds not fulfilling condition this iteration:", outlier_count, "(",
              round(outlier_count / inbounds_count, 2), ")")
        print("Max diff", round(max(raw_diffs[(~oob_mask), -1]), 2))

        primary_condition_fulfilled = False
        if outlier_count / inbounds_count < point_threshold:
            primary_condition_fulfilled = True
            print("Number of points in bounds outside", threshold, "within", point_threshold)

        plt.title("Samples with values")
        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.scatter(points[:, 0], points[:, 1], c=points[:, 2], marker=".", s=0.5, cmap="jet")
        plt.colorbar(cmap="jet")
        plt.clim(0, 1)
        rect = patches.Rectangle((-2, -2), 4, 4, linewidth=1, edgecolor='k', facecolor='none')
        plt.gca().add_patch(rect)
        plt.show()
        plt.close()

        # plt.title("Samples with differences")
        # plt.xlim(-2, 2)
        # plt.ylim(-2, 2)
        # plt.scatter(raw_diffs[:, 0], raw_diffs[:, 1], c=raw_diffs[:, 2], marker=".", s=0.5, cmap="nipy_spectral")
        # plt.colorbar(cmap="jet")
        # plt.clim(0, 1)
        # plt.show()
        # plt.close()

        time4 = time.time()
        print(round(time4 - time3, 2), "to plot scatterplot")

        if max(raw_diffs[(~oob_mask), -1]) < max_diff and primary_condition_fulfilled:
            return points

        # Construct kernel density estimator weighed with differences, draw new samples from it
        # Need to transpose our input first because scipy.stats.gaussian_kde is weird
        # TODO: Consider only using raw_diffs above threshold
        raw_diffs = raw_diffs[(~oob_mask) & (raw_diffs[:, -1] >= threshold)]
        raw_diffs = np.transpose(raw_diffs)
        #print(raw_diffs[-1] ** 2)
        kde = stats.gaussian_kde(
            dataset=raw_diffs[:-1],
            bw_method=0.1,
            weights=raw_diffs[-1] ** 2
        )

        # # plot kde pdf
        # x, y = np.meshgrid(
        #     np.linspace(-2, 2, 71),
        #     np.linspace(-2, 2, 71)
        # )
        #
        # vals = kde.evaluate(np.array([x.flatten(), y.flatten()]))
        #
        # plt.title("KDE PDF")
        # plt.xlim(-2, 2)
        # plt.ylim(-2, 2)
        # plt.scatter(x, y, c=vals, marker=".", s=2, cmap="jet")
        # plt.colorbar()
        # plt.show()
        # plt.close()

        samples = kde.resample(raw_diffs.shape[1] * sample_multiplier)
        samples = np.transpose(samples)

        time5 = time.time()
        print(round(time5 - time4, 2), "to do Kernel Density Estimation and draw new samples")

        new_points = np.zeros((samples.shape[0], samples.shape[1] + 1))
        new_points[:, :-1] = samples
        new_points[:, -1] = get_new_point(new_points[:, 0], new_points[:, 1])

        time6 = time.time()
        print(round(time6 - time5, 2), "to evaluate new points")

        points = np.vstack((points, new_points))

        for j in range(random_new_points):
            coord = [(dim[1] - dim[0]) * np.random.random_sample() + dim[0] for dim in dimensions]
            points = np.vstack(
                (
                    points,
                    np.array(coord + [get_new_point(*coord)])
                )
            )

        if count > max_iterations:
            finished = True
            print("Finished: Maximum number of iterations reached.")

        print()


def explore_parameter_space_delaunay_uniform(
        dimensions,
        get_new_point,
        threshold=0.05,
        point_threshold=0.05,
        random_new_points=1000,
        sample_multiplier=4,
        max_diff=None,
        number_of_partitions=5,
        max_iterations=30
):
    """
    This variant partitions the points into n partitions each step, then successivley triangulates using all
    partitions except one, and checks the points in the remaining partition.
    New points are sampled uniformly (in barycentric coordinates) in the triangle of their "parent" points.
    This version does not actually require a kd tree.

    Hacky proof of concept implementation

    Note: max_diff parameter is broken in this one, dont know why

    just for testing purposes!

    :param dimensions:      List of list
                            [[start, stop, number], ...}
    :param get_new_point:   Function to get new point in space. Must match dimensions
    :param threshold:       Relative maximum threshold for points to differ from analytic RIGHT NOW ABSOLUTE
    :param point_threshold: Relative number of points who may be over threshold for interpolation to quit
    :param random_new_points: How many random new points to add each iteration to avoid "local optimum"
    :param sample_multiplier:   For each point not fulfilling the conditions, sample_multiplier new points will be
                                drawn (not necessarily in the vicinity of specific points)
    :param max_diff:        Maximum difference any point may have from an analytic
    :param number_of_partitions:    Number of partitions to divide the dataset into each iteration
                                    Equivalent to number of delaunay triangulations that are done each iteration
    :param max_iterations:  Maximum number of iterations
    :return:
    """
    # Establish initial grid
    axes = []
    shape = []
    volume = 1
    for dim in dimensions:
        volume *= abs(dim[0] - dim[1])
        axes.append(np.linspace(*dim))
        shape.append(dim[2])

    # Shape: N points, M coordinates + 1 value
    points = np.zeros((prod(shape), len(shape)+1))

    for i, comb in enumerate(itertools.product(*axes)):
        points[i] = np.array([*comb, get_new_point(*comb)])

    finished = False
    count = 0

    while not finished:
        count += 1
        print(count)
        point_count = points.shape[0]
        print("Points this iteration:", point_count)
        thresh_points = 0


        time1 = time.time()

        points[:, -1] = get_new_point(points[:, 0], points[:, 1])
        # A copy of points that will hold the differences instead of values
        # Inefficient but not nearly as inefficient as cloudy evaluations at each point are anyway...
        raw_diffs = points.copy()

        # TODO: Find diffs here
        outside_tri_counter = 0
        samples = np.zeros((1, points.shape[1]))
        for i in range(number_of_partitions):
            mask = np.ones(points.shape[0], dtype=bool)
            mask[i::number_of_partitions] = False

            a = points[mask]    # all points EXCEPT some
            b = points[~mask]   # only SOME of the points

            tri = spatial.Delaunay(
                a[:, :-1]
            )

            # number of dimensions; subtract one because of value
            n = points.shape[1] - 1

            simplex_indices = tri.find_simplex(b[:, :-1])

            # we will deal with points outside triangulation later
            idw_points = b[simplex_indices == -1].copy()
            del_points = b[simplex_indices != -1].copy()

            # Cutting out indices -1
            simplex_indices_cleaned = simplex_indices[simplex_indices != -1]
            simplices = tri.simplices[simplex_indices_cleaned]

            # transforms of the points that are within triangles
            transforms = tri.transform[simplex_indices_cleaned]

            # This generalisation of the version in explore_parameter_space_delaunay is shamelessly copied from
            # https://stackoverflow.com/questions/30373912/interpolation-with-delaunay-triangulation-n-dim/30401693#30401693
            bary = np.einsum('ijk,ik->ij', transforms[:, :n, :n], del_points[:, :-1] - transforms[:, n, :])

            # the barycentric coordinates obtained this way are not complete. obtain last linearly dependent coordinate:
            weights = np.c_[bary, 1 - bary.sum(axis=1)]


            for j in range(del_points.shape[0]):
                orig_value = del_points[j, -1]
                # Attention! Overwriting previous analytic values with interpolated ones
                del_points[j, -1] = np.inner(a[simplices[j], -1], weights[j])

                if abs(del_points[j, -1] - orig_value) > threshold * orig_value:
                    if not (dimensions[0][0] < del_points[j, 0 ] < dimensions[0][1]) or not (dimensions[1][0] < del_points[j, 1] < dimensions[1][1]):
                        continue
                    lambda1 = np.random.random()
                    lambda2 = np.random.random() * lambda1
                    lambda3 = 1 - lambda1 - lambda2

                    x = lambda1 * a[simplices[j, 0], 0] + lambda2 * a[simplices[j, 1], 0] + lambda3 * a[simplices[j, 2], 0]
                    y = lambda1 * a[simplices[j, 0], 1] + lambda2 * a[simplices[j, 1], 1] + lambda3 * a[simplices[j, 2], 1]

                    v = cos2d_squared(x, y)

                    samples = np.vstack((samples, np.array([x, y, v])))

            # Build a tree and interpolate the remaining points using IDW
            tree = spatial.cKDTree(
                a[:, :-1]  # exclude value column
            )

            outside_tri_counter += idw_points.shape[0]

            for j, p in enumerate(idw_points):
                # Determine neighbors of point
                _, p_neigh = tree.query(
                    p[:-1],
                    4
                )

                p_neigh = list(p_neigh)  # list of indices including orig point
                # inconsistent behavior - sometimes i is in p_neigh, sometimes not
                if j in p_neigh:
                    p_neigh.remove(j)  # list of indices of neighbors
                p_neigh = points[p_neigh]  # list of points (+ values)

                idw_points[j, -1] = IDW(p_neigh[:, :-1], p_neigh[:, -1], p[:-1])

            # Write back into raw_diffs
            # This strange construct is required because of how numpy handles assignment to views/multi-masking
            raw_diffs_view = raw_diffs[~mask]
            raw_diffs_view[simplex_indices == -1, -1] -= idw_points[:, -1]
            raw_diffs_view[simplex_indices != -1, -1] -= del_points[:, -1]
            raw_diffs[~mask] = raw_diffs_view



        raw_diffs[:, -1] = np.abs(raw_diffs[:, -1])

        time3 = time.time()
        print(round(time3 - time1, 2), "to check interpolation")

        oob_mask = np.zeros(points.shape[0], dtype=bool)
        for i, dim in enumerate(dimensions):
            oob_mask[(points[:, i] < dim[0]) | (points[:, i] > dim[1])] = 1

        oob_count = points[oob_mask].shape[0]
        inbounds_count = points.shape[0] - oob_count

        print("Points skipped due to out of bounds:", oob_count)
        print("Points requiring IDW fallback: ", outside_tri_counter)
        outlier_count = raw_diffs[(~oob_mask) & (raw_diffs[:, -1] >= threshold)].shape[0]
        print("Points in bounds not fulfilling condition this iteration:", outlier_count, "(",
              round(outlier_count / inbounds_count, 2), ")")
        print("Max diff", round(max(raw_diffs[(~oob_mask), -1]), 2))

        primary_condition_fulfilled = False
        if outlier_count / inbounds_count < point_threshold:
            primary_condition_fulfilled = True
            print("Number of points in bounds outside", threshold, "within", point_threshold)

        plt.title("Samples with values")
        #plt.xlim(-2.5, 2.5)
        #plt.ylim(-2.5, 2.5)
        plt.scatter(points[:, 0], points[:, 1], c=points[:, 2], marker=".", s=0.5, cmap="jet")
        plt.colorbar(cmap="jet")
        plt.clim(0, 1)
        rect = patches.Rectangle((-2, -2), 4, 4, linewidth=1, edgecolor='k', facecolor='none')
        plt.gca().add_patch(rect)
        plt.show()
        plt.close()

        plt.title("Samples with differences")
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.scatter(raw_diffs[:, 0], raw_diffs[:, 1], c=raw_diffs[:, 2], marker=".", s=0.5, cmap="nipy_spectral")
        plt.colorbar(cmap="jet")
        plt.clim(0, 1)
        plt.show()
        plt.close()

        time4 = time.time()
        print(round(time4 - time3, 2), "to plot scatterplot")

        if max(raw_diffs[(~oob_mask), -1]) < max_diff and primary_condition_fulfilled:
            return points

        points = np.vstack((points, samples))

        for j in range(random_new_points):
            coord = [(dim[1] - dim[0]) * np.random.random_sample() + dim[0] for dim in dimensions]
            points = np.vstack(
                (
                    points,
                    np.array(coord + [get_new_point(*coord)])
                )
            )

        if count > max_iterations:
            finished = True
            print("Finished: Maximum number of iterations reached.")

        print()

    return points



def test_interp_2d(points, analytic_fct, grid_side=100):

    print("\n==============================")
    print("Testing Interpolation")
    print("==============================")
    print()
    print("Testing at", grid_side**2, "evenly spaced points")

    tri = spatial.Delaunay(
        points[:, :-1]
    )

    x, y = np.meshgrid(
        np.arange(-2, 2, 4/grid_side),
        np.arange(-2, 2, 4/grid_side)
    )

    x = x.flatten()
    y = y.flatten()

    x = np.reshape(x, (x.shape[0], 1))
    y = np.reshape(y, (y.shape[0], 1))

    test_points = np.hstack((x, y))

    # number of dimensions; subtract one because of value
    n = points.shape[1] - 1

    simplex_indices = tri.find_simplex(test_points)

    out_points  = test_points[simplex_indices == -1].copy()
    test_points = test_points[simplex_indices != -1].copy()

    print("Number of points outside of triangulation area: ", out_points.shape[0])

    simplex_indices = simplex_indices[simplex_indices != -1]
    simplices = tri.simplices[simplex_indices]

    # transforms of the points that are within triangles
    transforms = tri.transform[simplex_indices]

    # This generalisation of the version in explore_parameter_space_delaunay is shamelessly copied from
    # https://stackoverflow.com/questions/30373912/interpolation-with-delaunay-triangulation-n-dim/30401693#30401693
    bary = np.einsum('ijk,ik->ij', transforms[:, :n, :n], test_points - transforms[:, n, :])

    # the barycentric coordinates obtained this way are not complete. obtain last linearly dependent coordinate:
    weights = np.c_[bary, 1 - bary.sum(axis=1)]

    # Make space for interpolated values
    test_points = np.hstack(
        (
            test_points,
            np.reshape(np.zeros(test_points.shape[0]), (test_points.shape[0], 1))
        )
    )

    for i in range(test_points.shape[0]):
        # Overwriting column of 0s we just created
        test_points[i, -1] = np.inner(points[simplices[i], -1], weights[i])

    # Make yet another column with analytic values
    test_points = np.hstack(
        (
            test_points,
            np.reshape(analytic_fct(test_points[:, 0], test_points[:, 1]), (test_points.shape[0], 1))
        )
    )

    diffs = np.abs(test_points[:, -1] - test_points[:, -2])

    print("Average difference:", np.average(diffs))
    print("Standard deviation of averages:", np.std(diffs))
    print("Smallest / Largest difference:", np.min(diffs), "/", np.max(diffs))

    plt.title("Interpolation evaluation - differences")
    plt.hist(diffs, bins=grid_side**1)
    plt.show()
    plt.close()

    plt.title("Interpolation evaluation - analytic values")
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.scatter(test_points[:, 0], test_points[:, 1], c=test_points[:, -1], marker=".", s=0.5, cmap="jet")
    plt.colorbar(cmap="jet")
    #plt.clim(0, 0.2)
    plt.show()
    plt.close()

    plt.title("Interpolation evaluation - interp values")
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.scatter(test_points[:, 0], test_points[:, 1], c=test_points[:, -2], marker=".", s=0.5, cmap="jet")
    plt.colorbar(cmap="jet")
    #plt.clim(0, 0.2)
    plt.show()
    plt.close()

    plt.title("Interpolation evaluation - differences")
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.scatter(test_points[:, 0], test_points[:, 1], c=diffs, marker=".", s=0.5, cmap="jet")
    plt.colorbar(cmap="jet")
    #plt.clim(0, 0.2)
    plt.show()
    plt.close()






arg_fct = cos2d_squared

points = explore_parameter_space_delaunay_KDE(
    [
        [-2, 2, 11],
        [-2, 2, 11]
    ],
    arg_fct,
    threshold=0.01,
    point_threshold=0.1,
    random_new_points=10,
    sample_multiplier=1,
    max_diff=999,
    number_of_partitions=10
)


test_interp_2d(points, arg_fct, grid_side=100)

















