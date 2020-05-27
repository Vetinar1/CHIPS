import subprocess
import os
from scipy import spatial, stats
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import re

def prod(iterable):
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


def get_base_filename_from_parameters(T, nH, Z, z):
    """
    Returns filenames for cloudy to use. Does not append suffixes, i.e. returns "filename" instead of "filename.in",
    "filename.out" etc.

    Note: In the current implementation the excessive underscores are a precaution in case I want to parse these with
    regex.

    :param T:
    :param nH:
    :param Z: metallicity
    :param z: redshift
    :return:
    """
    return "T_" + str(T) + "__nH_" + str(nH) + "__Z_" + str(Z) + "__z_" + str(z) + "_"


def load_point_from_cache(filename):
    """
    Loads point from filename in CACHE_FOLDER.

    :param filename:
    :return:        A (1, M+1) numpy array containing the point
    """
    point = []
    # positive lookbehinds and lookaheads
    point.append(
        float(re.search(r"(?<=T_)[^_]+(?=_)", filename)[0])
    )
    point.append(
        float(re.search(r"(?<=nH_)[^_]+(?=_)", filename)[0])
    )

    # Right now I am only using Ctot for testing purposes
    point.append(
        np.loadtxt(CACHE_FOLDER + filename + ".cool", usecols=3)
    )

    return np.array(point)


def initialize_points(dimensions=None, logfile=None):
    """
    Loads all points from cache. If less points than specified in dimensions are loaded, a grid will be generated
    according to dimensions, and the points at this grid evaluated.

    Note: This grid is *independent* of how many points were already loaded, provided they were less than would have
    been created by the dimensions parameter!

    :param dimensions:      List of form [[start, stop, steps], [start, stop, steps], ...]
                            By convention always in order T, nH, Z, z
    :return:
    """
    if logfile:
        log = logfile.write
    else:
        log = simple_print

    # Establish initial grid; is only actually used if we dont find enough points in the cache, but it conveniently
    # gives us the shape so i do it first
    axes = []
    shape = []

    axes.append(np.linspace(T_min, T_max, T_init_steps))
    axes.append(np.linspace(nH_min, nH_max, nH_init_steps))

    for axis in axes:
        shape.append(axis.shape[0])

    # Shape: N points, M coordinates + 1 value
    points = np.zeros((1, len(shape) + 1))

    # https://stackoverflow.com/a/10378012
    directory = os.fsencode(CACHE_FOLDER)

    time1 = time.time()
    log("Initializing points\n")
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".cool"):
            points = np.vstack(
                (
                    points,
                    load_point_from_cache(filename[:-len(".cool")])
                )
            )

    # Remove zeros
    points = points[1:]

    time2 = time.time()
    log(str(round(time2-time1, 2)) + "s to load points from cache\n")

    # Create a grid points as specified in dimensions
    if points.shape[0] < prod(shape):
        grid_points = np.zeros((prod(shape), len(shape) + 1))

        for i, comb in enumerate(itertools.product(*axes)):
            grid_points[i, :-1] = np.array(comb)

        grid_points = cloudy_evaluate_points(grid_points)
        points = np.vstack((points, grid_points))

        time3 = time.time()
        log(str(round(time3-time2, 2)) + "s to generate and evaluate an additional " + str(prod(shape)) + " points\n")

    log("\n\n")

    return points




def cloudy_evaluate_points(points, Z=0, z=0, jobs=12):
    """

    :param points:      Numpy array of points in parameter space to evaluate
                        Right now: (npoints, [T, hden])
    :param z            redshift
    :param jobs         no of parallel executions
    :return:
    """
    # Step 1: Read radiation field
    with open("rad", "r") as file:
        rad = file.read()

    # Step 2: Build cloudy files
    input_files = []
    for point in points:
        filename = get_base_filename_from_parameters(point[0], point[1], Z, z)
        input_files.append(filename)

        with open(filename + ".in", "w") as file:
            file.write('CMB redshift %.2f\n' % z)
            file.write('table HM12 redshift %.2f\n' % z)
            file.write("metals " + str(Z) + "\n")
            file.write("hden " + str(point[1]) + "\n")
            file.write("constant temperature " + str(point[0]) +"\n")
            file.write('stop zone 1\n')
            file.write("iterate to convergence\n")
            file.write('print last\n')
            file.write('print short\n')
            file.write('set save prefix "%s"\n' % filename)
            #file.write('save grid last ".grid"\n')
            file.write('save overview last ".overview"\n')
            file.write('save cooling last ".cool"\n')
            file.write('save heating last ".heat"\n')
            file.write('save cooling each last ".cool_by_element"\n')
            file.write("""save element hydrogen last ".H_ionf"
save element helium last ".He_ionf"
save element oxygen last ".O_ionf"
save element carbon last ".C_ionf"
save element neon last ".Ne_ionf"
save element magnesium last ".Mg_ionf"
save element silicon last ".Si_ionf"
save last line emissivity ".lines"
H 1 1215.67A
H 1 1025.72A
He 2 1640.43A
C 3 977.020A
C 4 1548.19A
C 4 1550.78A
N 5 1238.82A
N 5 1242.80A
O 6 1031.91A
O 6 1037.62A
Si 3 1206.50A
Si 4 1393.75A
Si 4 1402.77A
end of line
""")
            file.write(rad)

    with open("filenames", "w") as file:
        for filename in input_files:
            file.write(filename + "\n")

    # Step 3: Execute cloudy runs
    #result = subprocess.run(["parallel", "-j12", "--progress", "'source/cloudy.exe -r'", ":::: filenames"])
    os.system("parallel -j12 'source/cloudy.exe -r' :::: filenames")

    # Step 4: Read cooling data
    # for now only Ctot
    for i, filename in enumerate(input_files):
        points[i, -1] = np.loadtxt(filename + ".cool", usecols=3)

    # Step 5: Move files to cache
    pattern = get_base_filename_from_parameters("*", "*", "*", "*") + "*"
    os.system("mv " + pattern + " " + CACHE_FOLDER)

    return points


def simple_print(var):
    """
    Custom print with no line separator, analogous to file.write().

    :param var:         Variable to be printed.
    """
    print(var, sep="")


def interpolate_and_sample_delaunay(points, threshold, partitions=5, prune=None, logfile=None):
    """
    Interpolates the given set of points using delaunay triangulation. The interpolated values are checked against
    threshold (absolute value). If the difference between interpolated and analytic value is not within threshold,
    a new samples is drawn in the respective triangle (uniformly).
    Returns the coordinates of these new samples.

    Some points may be outside of the triangulation area. This particularly happens at corners of the parameter
    space. For these edge cases Inverse Distance Weighting (IDW) is used for interpolation instead.

    :param points:      Numpy array; Shape (N, M+1), where N is the number of points, M is the dimensionality of these
                        points. The last column should contain the values associated with the points.
    :param threshold:   The (absolute) difference to check the interpolation against.
    :param partitions:  The number of partitions to divide the points into. For each partition, one interpolation will
                        be done, with this partition being removed from the set of points and the triangulation being
                        made on the remaining points. More partitions means better accuracy, but slightly longer
                        runtime.
    :param prune:       Pruning function. Should take points in and return points. Use to remove points outside of
                        parameter space.
    :param logfile:     File object to write into. If none given, print instead.
    :return:            1. Numpy array; Shape (N, M). Coordinates of new points. May be None if no new points.
                        2. Number of points within pruning bounds that do not fulfill threshold condition
                        3. The maximum difference between analytic and interpolated points
    """
    start = time.time()
    new_points = np.zeros((1, points.shape[1]-1))
    outside = 0
    over_thresh_count = 0
    diffs = points.copy()   # Yes, ugly, but makes things easier

    if logfile:
        log = logfile.write
    else:
        log = simple_print

    for i in range(partitions):
        mask = np.ones(points.shape[0], dtype=bool)
        mask[i::partitions] = False     # extract every nth point starting at i

        a = points[mask]    # all points EXCEPT some
        b = points[~mask]   # only SOME of the points

        tri = spatial.Delaunay(a[:, :-1])
        simplex_indices = tri.find_simplex(b[:, :-1])   # For all points b find the simplices that contain them

        # Points for IDW interp/Points for barycentric (delaunay) interp
        idw_points = b[simplex_indices == -1].copy()
        del_points = b[simplex_indices != -1].copy()

        # Cutting out indices -1
        simplex_indices_cleaned = simplex_indices[simplex_indices != -1]
        simplices = tri.simplices[simplex_indices_cleaned]
        transforms = tri.transform[simplex_indices_cleaned]

        # The following implementation is adapted from
        # https://stackoverflow.com/questions/30373912/interpolation-with-delaunay-triangulation-n-dim/30401693#30401693
        n = points.shape[1] - 1     # number of coordinate dimensions; subtract one because of value

        # barycentric coordinates of points; N-1
        bary = np.einsum('ijk,ik->ij', transforms[:, :n, :n], del_points[:, :-1] - transforms[:, n, :])

        # Weights of points; N; (the coordinates sum up to 1, last one is dependent)
        weights = np.c_[bary, 1 - bary.sum(axis=1)]

        # The actual interpolation step
        interpolated = []
        for j in range(del_points.shape[0]):
            interpolated.append(
                np.inner(a[simplices[j], -1], weights[j])
            )

        # Points which are over given threshold
        over_thresh_unpruned = del_points[np.abs(np.array(interpolated) - del_points[:,-1]) > (threshold * del_points[:,-1])]
        if prune:
            over_thresh = prune(over_thresh_unpruned)
        else:
            over_thresh = over_thresh_unpruned

        over_thresh_count += over_thresh.shape[0]

        # Draw coordinates for new samples and convert them to cartesian
        if over_thresh.size > 0:
            bcoords = np.random.random((over_thresh.shape[0], n))
            for dim in range(1, bcoords.shape[1]):
                # Make sure the coordinates do not sum up to more than 1
                bcoords[:, dim] *= 1 - np.sum(bcoords[:, :dim], axis=1)

            #print(bcoords)
            assert(max(np.sum(bcoords, axis=1)) < 1)

            bcoords = np.hstack((bcoords, np.reshape(1 - np.sum(bcoords, axis=1), (bcoords.shape[0], 1))))

            # At this point I gave up trying to find a vectorized solution.
            ccoords = []
            for j in range(bcoords.shape[0]):
                ccoords.append([])

                for k in range(n):
                    ccoords[-1].append(
                        np.sum(bcoords[j] * a[simplices[j], k])
                    )

            new_points = np.vstack((new_points, ccoords))
        else:
            log("WARNING: No points over threshold. Are you sure your parameters are tight enough? (partition " +
                str(i) + "/" + str(partitions) + ")\n")
            new_points = None



        # IDW
        # NOTE: Currently, we do not draw new samples near IDW points. This should be okay since they are very few
        outside += idw_points.shape[0]

        # KDTree for finding nearest neighbors to use for IDW
        tree = spatial.cKDTree(
            a[:, :-1]  # exclude value column
        )

        for j, p in enumerate(idw_points):
            # Determine closest 4 neighbors of point
            _, p_neigh = tree.query(
                p[:-1],
                4
            )

            p_neigh = list(p_neigh)  # list of indices including orig point
            # inconsistent behavior - sometimes i is in p_neigh, sometimes not
            # I just realized j is indexing something different. Original issue still a thing?
            # while j in p_neigh:
            #     p_neigh.remove(j)  # list of indices of neighbors
            p_neigh = points[p_neigh]  # list of points (+ values)

            idw_points[j, -1] = IDW(p_neigh[:, :-1], p_neigh[:, -1], p[:-1])


        # This strange construct is required because of how numpy handles assignment to views/multi-masking
        diffs_view = diffs[~mask]
        diffs_view[simplex_indices == -1, -1] = np.abs(diffs_view[simplex_indices == -1, -1] / idw_points[:, -1] - 1)
        diffs_view[simplex_indices != -1, -1] = np.abs(diffs_view[simplex_indices != -1, -1] / del_points[:, -1] - 1)
        diffs[~mask] = diffs_view


    if prune:
        diffs_pruned = prune(diffs)
    else:
        diffs_pruned = diffs


    oob_count = over_thresh_unpruned.shape[0] - over_thresh.shape[0]
    new_point_count = 0
    if new_points is not None:
        new_point_count = new_points.shape[0]
    max_diff = max(diffs_pruned[:,-1])

    log("Interpolated and sampled using Delaunay Triangulation\n")
    log("\tTime:".ljust(50) + str(round(time.time() - start, 2)) + "s\n")
    log("\tOut of bounds points (skipped)".ljust(50) + str(oob_count) + "\n")
    log("\tPoints requiring IDW fallback:".ljust(50) + str(outside) + "\n")
    log("\tPoints not within threshold:".ljust(50) +
        str(new_point_count) + "/" +  # Equal to number of new samples
        str(points.shape[0]) + " (" +
        str(round(new_point_count / points.shape[0], 2)) + ")\n"
    )
    log("\tMaximum difference:".ljust(50) + str(max_diff) + "\n")

    if new_points is not None:
        new_points = new_points[1:] # Remove zeros used to create array

    return new_points, over_thresh_count, max_diff



# TODO: I wonder if normalization is even necessary when not using KDE?
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




NUMBER_OF_JOBS = 40
NUMBER_OF_PARTITIONS = 10
THRESHOLD = 0.1                 # Max difference between interpolated and analytic values
OVER_THRESH_MAX_FRACTION = 0.1  # Fraction of points for which THRESHOLD may not hold at maximum
MAX_DIFF = 0.5                  # Maximum difference that may exist between interpolated and analytic values anywhere
MAX_ITERATIONS = None           # Maximum number of iterations before aborting
MAX_STORAGE = 20                # Maximum storage that may be taken up by data before aborting; in GB
MAX_TIME = 24*3600              # Maximum runtime in seconds
PLOT_RESULTS = True
RANDOM_NEW_POINTS = 10          # How many completely random new points to add each iteration
CACHE_FOLDER = "cache/"

# TODO: Fix divide by 0 in IDW
if __name__ == "__main__":
    time_start = time.time()
    logfile = open("logfile", "w")

    # Cooling function parameter space:
    # T from 1 to 9
    # n_h from -4 to 4
    # Metallicity 0.01, 0.1, and 1
    # Element abundances: Can be very easily extra/interpolated linearly/analytically, can be done after the fact
    # Radiation background: Unsolved problem
    # By convention these are always used in this order, i.e. a point is given by
    # [T, nH, value] right now and will later be given by [T, nH, Z, z, value] for example
    T_min = 1
    T_max = 9
    T_init_steps = 7

    nH_min = -4
    nH_max = 4
    nH_init_steps = 7

    dimensions = [[T_min, T_max, T_init_steps], [nH_min, nH_max, nH_init_steps]]
    points = initialize_points(dimensions, logfile)
    prune = get_pruning_function(dimensions)
    init_point_count = points.shape[0]

    iteration = 0

    while True:
        iteration += 1
        point_count = points.shape[0]
        logfile.write("Iteration ".ljust(50) + str(iteration) + "\n")
        logfile.write("Number of points:".ljust(50) + str(point_count) + "\n")
        thresh_points = 0

        time1 = time.time()
        prev_length = points.shape[0]
        points = np.unique(points, axis=0)
        if points.shape[0] < prev_length:
            logfile.write("Removed duplicates:".ljust(50) + str(prev_length - points.shape[0]) + "\n")

        time2 = time.time()

        new_points, over_thresh_count, max_diff = interpolate_and_sample_delaunay(
            points,
            THRESHOLD,
            partitions=10,
            logfile=logfile
        )

        in_bounds_points = prune(points)
        in_bounds_count = in_bounds_points.shape[0]

        if PLOT_RESULTS:
            # TODO: Does not generalize
            plt.title(r"$C_{tot}$ in erg/cm$^3$/s")
            plt.xlim(T_min-1, T_max+1)
            plt.xlabel("log T/K")
            plt.ylim(nH_min-1, nH_max+1)
            plt.ylabel(r"log $n_H$/cm$^{-3}$")
            plt.scatter(points[:, 0], points[:, 1], c=points[:, 2], marker=".", s=0.5, cmap="jet")
            plt.colorbar(cmap="jet")
            rect = patches.Rectangle((T_min, nH_min), T_max - T_min, nH_max - nH_min, linewidth=1, edgecolor='k', facecolor='none')
            plt.gca().add_patch(rect)
            plt.savefig("iteration" + str(iteration) + ".png")
            plt.close()


        threshold_condition = False
        if OVER_THRESH_MAX_FRACTION and over_thresh_count / in_bounds_count < OVER_THRESH_MAX_FRACTION:
            threshold_condition = True

        max_diff_condition = False
        if MAX_DIFF and max_diff < MAX_DIFF:     # Yes, unfortunate naming, but it should be clear that CAPS = constant
            max_diff_condition = True

        max_iteration_condition = False
        if MAX_ITERATIONS and iteration > MAX_ITERATIONS:
            max_iteration_condition = True

        max_storage_condition = False
        cache_size_gb = get_folder_size(CACHE_FOLDER) / 1e9     # Does not account for base 2
        if MAX_STORAGE and cache_size_gb > MAX_STORAGE:
            max_storage_condition = True

        max_time_condition = False
        if MAX_TIME and time.time() - time_start > MAX_TIME:
            max_time_condition = True

        if threshold_condition and max_diff_condition:
            logfile.write("\n\nReached desired accuracy. Quitting.\n\n")
            break

        if max_iteration_condition:
            logfile.write("\n\nReached maximum number of iterations (" + str(MAX_ITERATIONS) + "). Quitting.\n\n")
            break

        if max_storage_condition:
            logfile.write("\n\nReached maximum allowed storage (" + str(MAX_STORAGE) + "GB). Quitting.\n\n")
            break

        if max_time_condition:
            logfile.write("\n\nReached maximum calculation time (" + str(MAX_TIME) + "s). Quitting.\n\n")
            break


        time3 = time.time()
        if new_points is not None:
            new_points = np.hstack((new_points, np.zeros((new_points.shape[0], 1))))
            new_points = cloudy_evaluate_points(new_points)
            time4 = time.time()
            points = np.vstack((points, new_points))

            logfile.write(str(round(time4-time3, 2)) + "s to evaluate " + str(new_points.shape[0]) +  " new points\n")
        else:
            time4 = time.time()

        # random points
        random_points = np.zeros((1, points.shape[1]-1))

        for j in range(RANDOM_NEW_POINTS):
            coord = [(dim[1] - dim[0]) * np.random.random() + dim[0] for dim in dimensions]
            random_points = np.vstack((random_points, np.array(coord)))

        random_points = np.hstack((random_points, np.zeros((random_points.shape[0], 1))))
        random_points = random_points[1:]   # remove zeros used to create array
        random_points = cloudy_evaluate_points(random_points)
        points = np.vstack(
            (
                points,
                random_points
            )
        )

        time5 = time.time()
        logfile.write(str(round(time5 - time4, 2)) + "s to add " + str(RANDOM_NEW_POINTS) + " random new points\n")

        logfile.write("\n\n")





    time_end = time.time()
    elapsed = time_end - time_start

    hours = (elapsed - (elapsed % 3600)) / 3600
    seconds_no_hours = elapsed % 3600
    minutes = (seconds_no_hours - (seconds_no_hours % 60)) / 60
    seconds = seconds_no_hours % 60

    logfile.write("Run complete; Calculated at least " +
                  str(points.shape[0] - init_point_count) + " new points (" +
                  str(points.shape[0]) + " total) in " +
                  str(round(elapsed, 2)) + "s / " +
                  str(int(hours)) + "h " +
                  str(int(minutes)) + "m " +
                  str(round(seconds, 2)) + "s"
    )

    logfile.close()







