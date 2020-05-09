from scipy import spatial, stats
import numpy as np
import itertools
import matplotlib.pyplot as plt
import time

#plt.rcParams['image.cmap'] = 'jet'

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



# explore_parameter_space_delaunay(
#     [
#         [-2, 2, 11],
#         [-2, 2, 11]
#     ],
#     get_test_point,
#     threshold=0.05,
#     point_threshold=0.05,
#     random_new_points=50,
#     max_diff=99999999
# )




















