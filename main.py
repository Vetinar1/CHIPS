import numpy as np
import matplotlib.pyplot as plt
import time
from util import *
from cloudy_optimizer import *


NUMBER_OF_JOBS = 12
NUMBER_OF_PARTITIONS = 10
THRESHOLD = 0.1                 # Max difference between interpolated and analytic values
OVER_THRESH_MAX_FRACTION = 0.1  # Fraction of points for which THRESHOLD may not hold at maximum
MAX_DIFF = 0.5                  # Maximum difference that may exist between interpolated and analytic values anywhere
                                # in dex
MAX_ITERATIONS = 20             # Maximum number of iterations before aborting
MAX_STORAGE = 20                # Maximum storage that may be taken up by data before aborting; in GB
MAX_TIME = 0.33*3600                 # Maximum runtime in seconds
PLOT_RESULTS = True
RANDOM_NEW_POINTS = 10          # How many completely random new points to add each iteration
CACHE_FOLDER = "cache/"



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
    T_min = 2
    T_max = 6
    T_init_steps = 7

    nH_min = -4
    nH_max = 4
    nH_init_steps = 7

    dimensions = [[T_min, T_max, T_init_steps], [nH_min, nH_max, nH_init_steps]]


    points = load_all_points_from_cache("run4/cache/")
    prune = get_pruning_function(dimensions)

    draw_points_not_in_threshold(dimensions, points, THRESHOLD, prune, 10)

    exit()


    points = initialize_points(dimensions, logfile, add_grid=True)
    prune = get_pruning_function(dimensions)
    init_point_count = points.shape[0]

    iteration = 0

    while True:
        iteration += 1
        point_count = points.shape[0]
        logfile.write("Iteration ".ljust(50) + str(iteration) + "\n")
        logfile.write("Number of points:".ljust(50) + str(point_count) + "\n")

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
            logfile=logfile,
            prune=prune
        )

        # Note: Double pruning...
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
