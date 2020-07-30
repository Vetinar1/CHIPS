import numpy as np
import matplotlib.pyplot as plt
import time
from util import *
from cloudy_optimizer import *
import seaborn as sns
import pandas as pd

sns.set()


NUMBER_OF_JOBS = 40
NUMBER_OF_PARTITIONS = 10
THRESHOLD = 0.1                 # Max difference between interpolated and analytic values in dex
OVER_THRESH_MAX_FRACTION = 0.1  # Fraction of points for which THRESHOLD may not hold at maximum
MAX_DIFF = 0.5                  # Maximum difference that may exist between interpolated and analytic values anywhere
                                # in dex
MAX_ITERATIONS = 3             # Maximum number of iterations before aborting
MAX_STORAGE = 20                # Maximum storage that may be taken up by data before aborting; in GB
MAX_TIME = 0.1*3600                 # Maximum runtime in seconds
PLOT_RESULTS = True
RANDOM_NEW_POINTS = 20          # How many completely random new points to add each iteration
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
    dim_names = ["Temperature", "n_H", "Metallicity", "Redshift"]
    T_min = 2
    T_max = 6
    T_init_steps = 7

    nH_min = -4
    nH_max = 4
    nH_init_steps = 7

    Z_min = -2
    Z_max = 0
    Z_init_steps = 3

    dimensions = [
        [T_min, T_max, T_init_steps],
        [nH_min, nH_max, nH_init_steps],
        #[Z_min, Z_max, Z_init_steps]
    ]


    margins = 0.1


    # points = load_all_points_from_cache("run4/cache/")
    # prune = get_pruning_function(dimensions)
    #
    # draw_points_not_in_threshold(dimensions, points, THRESHOLD, prune, 10)
    #
    # exit()


    points = initialize_points(dimensions, logfile, add_grid=True, margins=margins)
    prune = get_pruning_function(dimensions)
    init_point_count = points.shape[0]


    logfile.write("NUMBER_OF_PARTITIONS".ljust(50) + str(NUMBER_OF_PARTITIONS) + "\n")
    logfile.write("THRESHOLD (dex)".ljust(50) + str(THRESHOLD) + "\n")
    logfile.write("OVER_THRESH_MAX_FRACTION".ljust(50) + str(OVER_THRESH_MAX_FRACTION) + "\n")
    logfile.write("MAX_DIFF (dex)".ljust(50) + str(MAX_DIFF) + "\n")
    logfile.write("RANDOM_NEW_POINTS".ljust(50) + str(RANDOM_NEW_POINTS) + "\n")
    logfile.write("\n")
    logfile.write("NUMBER_OF_JOBS".ljust(50) + str(NUMBER_OF_JOBS) + "\n")
    logfile.write("MAX_ITERATIONS".ljust(50) + str(MAX_ITERATIONS) + "\n")
    logfile.write("MAX_STORAGE (GB)".ljust(50) + str(MAX_STORAGE) + "\n")
    logfile.write("MAX_TIME (h)".ljust(50) + str(MAX_TIME / 3600) + "\n")
    logfile.write("\n")
    logfile.write("Points loaded from cache:".ljust(50) + str(init_point_count) + "\n")
    logfile.write("dimensions (T, nH, Z, z) (start, stop, step) ".ljust(50) + str(dimensions) + "\n")


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
            prune=prune,
            iteration=iteration,
            dimensions=dimensions
        )

        # Note: Double pruning...
        in_bounds_points = prune(points)
        in_bounds_count = in_bounds_points.shape[0]

        if PLOT_RESULTS:
            timeA = time.time()
            # Seaborn wants a dataframe so lets convert...
            df_points = pd.DataFrame(
                points,
                columns=[dim_names[i] for i in range(len(dimensions))] + ["Value"]
            )
            df_points_new = compile_to_dataframe(
                len(dimensions),
                "cache/iteration{}".format(iteration-1)
            )
            old_length = len(df_points.index)
            df_points = pd.concat([df_points, df_points_new], ignore_index=True)
            df_points["New Points"] = df_points.duplicated(keep=False)
            df_points = df_points.iloc[:old_length]

            df_points.loc[df_points["New Points"] == True, "New Points"] = "Latest Iteration"
            df_points.loc[df_points["New Points"] == False, "New Points"] = "Older Iterations"


            grid = sns.pairplot(
                df_points,
                # hue="Value",
                diag_kind="hist",
                vars=[dim_names[i] for i in range(len(dimensions))],
                hue="New Points",
                markers=".",
                plot_kws={
                    "s": 1,
                    "marker": ".",
                    "edgecolor": None
                },
                diag_kws={
                    "bins":50
                },
                height=6
            )
            grid.savefig("iteration" + str(iteration) + ".png")

            del df_points
            del df_points_new

            timeB = time.time()
            logfile.write(str(round(timeB - timeA, 2)) + "s to plot current iteration\n")



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
        if not os.path.exists("cache/iteration{}".format(iteration)):
            os.system("mkdir cache/iteration{}".format(iteration))

        if new_points is not None:
            new_points = np.hstack((new_points, np.zeros((new_points.shape[0], 1))))
            new_points = cloudy_evaluate_points(new_points, cache_folder="cache/iteration{}".format(iteration))
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
        random_points = cloudy_evaluate_points(random_points, cache_folder="cache/iteration{}".format(iteration))
        points = np.vstack(
            (
                points,
                random_points
            )
        )

        time5 = time.time()
        logfile.write(str(round(time5 - time4, 2)) + "s to add " + str(RANDOM_NEW_POINTS) + " random new points\n")
        logfile.write("Total elapsed time so far: " + str(round(time5 - time_start, 2)) + "s / " +
                      seconds_to_human_readable(time5 - time_start))

        logfile.write("\n\n")





    time_end = time.time()
    elapsed = time_end - time_start

    elapsed_readable = seconds_to_human_readable(elapsed)

    logfile.write("Run complete; Calculated at least " +
                  str(points.shape[0] - init_point_count) + " new points (" +
                  str(points.shape[0]) + " total) in " +
                  str(round(elapsed, 2)) + "s / " +
                  elapsed_readable
    )

    logfile.close()
