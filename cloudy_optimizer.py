import subprocess
import os
from scipy import spatial, stats
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import re
from util import *

CLOUDY_LOCATION = "cloudy/source/cloudy.exe"


def IDW(points, values, x, exp=1):
    """
    Simple inverse distance weighting interpolation.

    :param points:      Iterable of points to use for weighting, shape (npoints, ncoord +1)
    :param values:      Values at points, 1d array
    :param x:           coordinates of point to interpolate at
    :param exp:         Exponent of weighting function; 1/2 = euclidean, 1 = squared euclidean etc
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


def load_point_from_cache(filename, num_coords, cache_folder="cache/"):
    """
    Loads point from filename in cache_folder. Does not check subfolders.

    TODO: Generalize to arbitrary coordinates

    :param filename:
    :param num_coords:      Number of coordinates to use. Order T, nH, Z, z, ...
                            1 = only T, 2 = T and nH, ...
    :return:                A (1, num_coords+1) numpy array containing the point
    """

    point = []
    # positive lookbehinds and lookaheads

    while True:
        point.append(
            float(re.search(r"(?<=T_)[^_]+(?=_)", filename)[0])
        )

        if num_coords == 1:
            break

        point.append(
            float(re.search(r"(?<=nH_)[^_]+(?=_)", filename)[0])
        )

        if num_coords == 2:
            break

        point.append(
            float(re.search(r"(?<=Z_)[^_]+(?=_)", filename)[0])
        )

        if num_coords == 3:
            break

        break



    # Right now I am only using Ctot for testing purposes
    point.append(
        np.loadtxt(cache_folder + filename + ".cool", usecols=3)
    )

    return np.array(point)


def load_all_points_from_cache(num_coords, cache_folder="cache/"):
    """
    Load all points from all valid files in cache_folder. Includes subfolders.

    TODO: Generalize to arbitrary coordinates

    :param num_coords:      Number of coordinates to use. Order T, nH, Z, z, ...
                            1 = only T, 2 = T and nH, ...
    :param cache_folder:
    :return:
    """
    filenames = []
    for dirpath, dirnames, fnames in os.walk(cache_folder):
        filenames += [os.path.join(dirpath, fname) for fname in fnames]

    points = np.zeros((1, num_coords+1))

    for file in filenames:
        filename = os.fsdecode(file)
        if filename.endswith(".cool"):
            points = np.vstack(
                (
                    points,
                    load_point_from_cache(filename[:-len(".cool")], num_coords, cache_folder="") # folder contained in names
                )
            )

    # Remove zeros
    points = points[1:]

    return points


def initialize_points(dimensions, logfile=None, add_grid=False, cache_folder="cache/"):
    """
    Loads all points from cache. If less points than specified in dimensions are loaded, a grid will be generated
    according to dimensions, and the points at this grid evaluated.

    Note: This grid is *independent* of how many points were already loaded, provided they were less than would have
    been created by the dimensions parameter!

    :param dimensions:      List of form [[start, stop, steps], [start, stop, steps], ...]
                            By convention always in order T, nH, Z, z
    :param logfile:         Logfile to write into. If none, print instead.
    :param add_grid:        If True a grid will be added like described, otherwise this step is skipped.
    :param cache_folder:    Folder containing the files
    :return:
    """
    if logfile:
        log = logfile.write
    else:
        log = simple_print


    # Establish initial grid; is only actually used if we dont find enough points in the cache, but it conveniently
    # gives us the shape so i do it first
    axes = []
    grid_shape = []     # not a numpy shape

    for dim in dimensions:
        axes.append(np.linspace(*dim))

    for axis in axes:
        grid_shape.append(axis.shape[0])


    time1 = time.time()
    log("Initializing points\n")
    points = load_all_points_from_cache(len(dimensions), cache_folder)

    time2 = time.time()
    log(str(round(time2-time1, 2)) + "s to load points from cache\n")

    # Create a grid points as specified in dimensions
    if add_grid and points.shape[0] < prod(grid_shape):
        grid_points = np.zeros((prod(grid_shape), len(grid_shape) + 1))

        for i, comb in enumerate(itertools.product(*axes)):
            grid_points[i, :-1] = np.array(comb)

        if not os.path.exists(cache_folder + "iteration0"):
            os.system("mkdir {}iteration0".format(cache_folder))

        grid_points = cloudy_evaluate_points(grid_points, cache_folder=cache_folder + "iteration0")
        points = np.vstack((points, grid_points))

        time3 = time.time()
        log(str(round(time3-time2, 2)) + "s to generate and evaluate an additional " + str(prod(grid_shape)) + " points\n")

    log("\n\n")

    return points




def cloudy_evaluate_points(points, jobs=12, cache_folder="cache/", cloudy_exe=CLOUDY_LOCATION):
    """

    :param points:      Numpy array of points in parameter space to evaluate
                        Shape (N, M+1)
                        Coordinate order: T, nH, Z, z
    :param z            redshift
    :param jobs         no of parallel executions
    :param cache_folder
    :param cloudy_exe   Location of cloudy.exe
    :return:
    """
    T = np.zeros(points.shape[0])
    if points.shape[1] - 1 > 0:     # always subtract one to account for value field
        T = points[:,0]

    nH = np.zeros(points.shape[0])
    if points.shape[1] - 1 > 1:
        nH = points[:,1]

    Z = np.zeros(points.shape[0])
    if points.shape[1] - 1 > 2:
        Z = points[:,2]

    z = np.zeros(points.shape[0])

    # Step 1: Read radiation field
    with open("rad", "r") as file:
        rad = file.read()

    # Step 2: Build cloudy files
    input_files = []
    for i, point in enumerate(points):
        filename = get_base_filename_from_parameters(T[i], nH[i], Z[i], z[i])
        input_files.append(filename)

        with open(filename + ".in", "w") as file:
            file.write('CMB redshift %.2f\n' % z[i])
            #file.write('table HM12 redshift %.2f\n' % z)
            #file.write("metals " + str(Z) + "\n")
            file.write("metals " + str(Z[i]) + "\n")
            file.write("hden " + str(nH[i]) + "\n")
            file.write("constant temperature " + str(T[i]) +"\n")
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
    os.system("parallel -j" + str(jobs) + " '" + cloudy_exe + " -r' :::: filenames")

    # Step 4: Read cooling data
    # for now only Ctot
    for i, filename in enumerate(input_files):
        points[i, -1] = np.loadtxt(filename + ".cool", usecols=3)

    # Step 5: Move files to cache
    pattern = get_base_filename_from_parameters("-?[0-9\.]+", "-?[0-9\.]+", "-?[0-9\.]+", "-?[0-9\.]+") + "-?[0-9\.]+"
    #os.system("mv " + pattern + " " + cache_folder)
    os.system("ls | grep -P {} | xargs -I % -n 1000 mv -t {} %".format(pattern, cache_folder))

    return points


def interpolate_delaunay(points, interp_coords):
    """
    Use points to interpolate at positions interp_coords using Delaunay triangulation.
    For a commented version see interpolate_and_sample_delaunay

    :param points:          Points; coords + values; Shape (N, M+1)
    :param interp_coords:   Coords only; Shape (N', M)
    :return:                1. Array of successfully interpolated points with values at interp_coords; Shape (N'', M+1)
                            2. Mask for interp_coords that is true for each element that couldnt be interpolated/
                            was ignored; Shape (N)
                            3. The triangulation object
    """

    tri = spatial.Delaunay(points[:, :-1])              # Triangulation
    simplex_indices = tri.find_simplex(interp_coords)   # Find the indices of the simplices containing the interp_coords

    # Only consider those points which are inside of the triangulation area space.
    valid_coords   = interp_coords[simplex_indices != -1]
    #ignored_coords = interp_coords[simplex_indices == -1]

    # Get the simplices and transforms containing the valid points
    simplex_indices_cleaned = simplex_indices[simplex_indices != -1]
    simplices  = tri.simplices[simplex_indices_cleaned]
    transforms = tri.transform[simplex_indices_cleaned]

    # Adapted from
    # https://stackoverflow.com/questions/30373912/interpolation-with-delaunay-triangulation-n-dim/30401693#30401693
    # Get number of coordinate dimensions
    n = interp_coords.shape[1]

    # barycentric coordinates of points; N-1
    bary = np.einsum('ijk,ik->ij', transforms[:, :n, :n], valid_coords - transforms[:, n, :])

    # Weights of points/complete barycentric coordinates (including the last dependent one); N
    weights = np.c_[bary, 1 - bary.sum(axis=1)]

    # The actual interpolation step
    interpolated = []
    for j in range(valid_coords.shape[0]):
        interpolated.append(
            np.inner(points[simplices[j], -1], weights[j])
        )

    interpolated = np.array(interpolated)
    # reshape for hstack
    interpolated = np.reshape(interpolated, (interpolated.shape[0], 1))

    ignored_mask = simplex_indices == -1

    return np.hstack((valid_coords, interpolated)), ignored_mask, tri


def sign(x1, y1, x2, y2, x3, y3):
    return (x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3)


def interpolate_and_sample_delaunay(points, threshold, partitions=5, prune=None, logfile=None, iteration=0):
    """
    Interpolates the given set of points using delaunay triangulation. The interpolated values are checked against
    threshold (absolute value). If the difference between interpolated and analytic value is not within threshold,
    a new samples is drawn in the respective triangle (uniformly).
    Returns the coordinates of these new samples.

    Some points may be outside of the triangulation area. This particularly happens at corners of the parameter
    space. For these edge cases Inverse Distance Weighting (IDW) is used for interpolation instead.

    :param points:      Numpy array; Shape (N, M+1), where N is the number of points, M is the dimensionality of these
                        points. The last column should contain the values associated with the points.
    :param threshold:   The (absolute) difference to check the interpolation against, in dex.
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
    N = points.shape[1] - 1 # dimensions
    new_points = np.zeros((1, N))
    outside = 0
    over_thresh_count = 0

    # Yes, ugly, but makes things easier TODO: Potential vector for optimization
    # Will hold relative differences
    diffs = points.copy()

    if logfile:
        log = logfile.write
    else:
        log = simple_print

    fig, ax = plt.subplots(1, partitions, figsize=(10*partitions, 10))

    for partition in range(partitions):
        mask = np.ones(points.shape[0], dtype=bool)
        mask[partition::partitions] = False     # extract every nth point starting at i

        a = points[mask]    # all points EXCEPT some
        b = points[~mask]   # only SOME of the points

        interp_points, ignored_mask, tri = interpolate_delaunay(a, b[:,:-1])
        orig_points = b[~ignored_mask]  # for comparison against interp_points


        # Points which are over given threshold
        # log10s because I want to use the threshold in dex
        over_thresh_unpruned = orig_points[
            np.abs(
                np.log10(interp_points[:,-1]) - np.log10(orig_points[:,-1])
            ) > threshold
        ]

        # Remove points outside considered parameter space
        if prune:
            over_thresh = prune(over_thresh_unpruned)
        else:
            over_thresh = over_thresh_unpruned

        over_thresh_count += over_thresh.shape[0]

        # Draw coordinates for new samples
        # How to uniformly sample over a triangle:
        # https://math.stackexchange.com/questions/18686/uniform-random-point-in-triangle
        # TODO Make separate function?
        if over_thresh.size > 0:
            simplex_indices = tri.find_simplex(over_thresh[:,:-1])
            #simplex_indices = simplex_indices[~ignored_mask]
            simplices = tri.simplices[simplex_indices]

            simplex_points = np.zeros((simplices.shape[0], simplices.shape[1], simplices.shape[1]-1))

            for i in range(simplices.shape[0]):
                simplex_points[i] = a[simplices[i], :-1]

            samples = sample_simplices(simplex_points)
            print(iteration, partition)
            print(samples)

            new_points = np.vstack((new_points, samples))
        else:
            log("WARNING: No points over threshold. Are you sure your parameters are tight enough? (partition " +
                str(partition) + "/" + str(partitions) + ")\n")

        # end TODO

        ax[partition].triplot(a[:,0], a[:,1], tri.simplices)
        ax[partition].plot(a[:,0], a[:,1], "ko")
        for j in range(over_thresh.shape[0]):
            ax[partition].plot([over_thresh[j,0], samples[j][0]], [over_thresh[j,1], samples[j][1]], "red")
            ax[partition].plot([over_thresh[j,0]], [over_thresh[j,1]], "ro")

        ax[partition].set_title("Delaunay Partition " + str(partitions))
        ax[partition].set_xlabel("T")
        ax[partition].set_ylabel("nH")

        # # IDW
        # # NOTE: Currently, we do not draw new samples near IDW points. This should be okay since they are very few
        # outside += idw_points.shape[0]
        #
        # # KDTree for finding nearest neighbors to use for IDW
        # tree = spatial.cKDTree(
        #     a[:, :-1]  # exclude value column
        # )
        #
        # for j, p in enumerate(idw_points):
        #     # Determine closest 4 neighbors of point
        #     _, p_neigh = tree.query(
        #         p[:-1],
        #         4
        #     )
        #
        #     p_neigh = list(p_neigh)  # list of indices including orig point
        #     # inconsistent behavior - sometimes i is in p_neigh, sometimes not
        #     # I just realized j is indexing something different. Original issue still a thing?
        #     # while j in p_neigh:
        #     #     p_neigh.remove(j)  # list of indices of neighbors
        #     p_neigh = points[p_neigh]  # list of points (+ values)
        #
        #     idw_points[j, -1] = IDW(p_neigh[:, :-1], p_neigh[:, -1], p[:-1])


        # Calculate relative differences between original values and interpolation
        # Note: Currently, points that could not be interpolated (previously using IDW) are dropped silently
        # This strange construct is required because of how numpy handles assignment to views/multi-masking
        diffs_view = diffs[~mask]
        # diffs_view[ ignored_mask, -1] = np.abs(diffs_view[ ignored_mask, -1] / idw_points[:, -1] - 1)
        diffs_view[ ignored_mask, -1] = np.nan
        #diffs_view[~ignored_mask, -1] = np.abs(diffs_view[~ignored_mask, -1] / interp_points[:, -1] - 1)
        diffs_view[~ignored_mask, -1] = np.abs(
            np.log10(interp_points[:,-1]) - np.log10(orig_points[:,-1])
        )
        diffs[~mask] = diffs_view

    fig.savefig("Partitions" + str(iteration) + ".png")

    # Note: Continuation of the silent dropping of non-interpolated values
    diffs = drop_all_rows_containing_nan(diffs)

    diffs_length_unpruned = diffs.shape[0]
    if prune:
        diffs_pruned = prune(diffs)
    else:
        diffs_pruned = diffs

    diffs_length_pruned = diffs.shape[0]
    # Not very clean, but since pruning is only considering coordinates (and not values) this gives an accurate number
    # for the out of bounds points (ignoring silently dropped ones)
    oob_count = diffs_length_unpruned - diffs_length_pruned

    new_point_count = 0
    if not np.array_equal(new_points, np.zeros((1, N))):
        new_point_count = new_points.shape[0] - 1 # account for the "starter" made of zeros

    if new_points is not None:
        new_points = new_points[1:]               # Remove zeros used to create array

    # To determine the maximum difference, we need to consider "both directions"
    # i.e. a relative difference of 0.1 is larger than a relative difference of 2
    # -> take the log, use the absolute biggest log, determine original difference from that
    #print(diffs_pruned)
    # logdiff = np.log10(diffs_pruned[:,-1])
    # log_max_diff = max(np.abs(logdiff))
    # max_diff = 10**log_max_diff
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
    log("\tMaximum (log) difference:".ljust(50) + str(max_diff) + "\n")

    return new_points, over_thresh_count, max_diff


def draw_points_not_in_threshold(dimensions, points, threshold, prune=None, partitions=5):
    """
    Plots several diagram helping to understand which points are over threshold, out of bounds etc.

    Contains mostly code copied from interpolate_and_sample_delaunay

    :param dimensions
    :param points:
    :param threshold:
    :param prune:
    :return:
    """
    over_thresh_total = np.zeros((1, points.shape[1]))

    for i in range(partitions):
        mask = np.ones(points.shape[0], dtype=bool)
        mask[i::partitions] = False     # extract every nth point starting at i

        a = points[mask]    # all points EXCEPT some
        b = points[~mask]   # only SOME of the points

        interp_points, ignored_mask, tri = interpolate_delaunay(a, b[:,:-1])
        orig_points = b[~ignored_mask]  # for comparison against interp_points


        # Points which are over given threshold
        # log10s because I want to use the threshold in dex
        over_thresh_unpruned = orig_points[
            np.abs(
                np.log10(interp_points[:,-1]) - np.log10(orig_points[:,-1])
            ) > threshold
        ]

        # Remove points outside considered parameter space
        if prune:
            over_thresh = prune(over_thresh_unpruned)
        else:
            over_thresh = over_thresh_unpruned

        over_thresh_total = np.vstack((over_thresh_total, over_thresh))

    over_thresh_total = over_thresh_total[1:]

    plt.title(r"Over threshold only")
    plt.xlim(dimensions[0][0] - 1, dimensions[0][1] + 1) # T
    plt.xlabel("log T/K")
    plt.ylim(dimensions[1][0] - 1, dimensions[1][1] + 1) # nH
    plt.ylabel(r"log $n_H$/cm$^{-3}$")
    plt.scatter(over_thresh_total[:, 0], over_thresh_total[:, 1], c=over_thresh_total[:, 2], marker=".", s=0.5, cmap="jet")
    rect = patches.Rectangle((dimensions[0][0], dimensions[1][0]),
                             dimensions[0][1] - dimensions[0][0],
                             dimensions[1][1] - dimensions[1][0], linewidth=1, edgecolor='k',
                             facecolor='none')
    plt.gca().add_patch(rect)
    plt.show()
    plt.close()


    plt.title(r"All points, red over threshold")
    plt.xlim(dimensions[0][0] - 1, dimensions[0][1] + 1) # T
    plt.xlabel("log T/K")
    plt.ylim(dimensions[1][0] - 1, dimensions[1][1] + 1) # nH
    plt.ylabel(r"log $n_H$/cm$^{-3}$")
    plt.scatter(points[:, 0], points[:, 1], c="blue", marker=".", s=0.5)
    plt.scatter(over_thresh_total[:, 0], over_thresh_total[:, 1], c="red", marker=".", s=0.5)
    rect = patches.Rectangle((dimensions[0][0], dimensions[1][0]),
                             dimensions[0][1] - dimensions[0][0],
                             dimensions[1][1] - dimensions[1][0], linewidth=1, edgecolor='k',
                             facecolor='none')
    plt.gca().add_patch(rect)
    plt.show()
    plt.close()










