import os
from chips.utils import *
import pandas as pd
import itertools
from scipy import interpolate, spatial
import matplotlib.pyplot as plt
import seaborn as sns
import time
from parse import parse
from matplotlib import patches

sns.set()

# TODO: Options for cloudy to clean up after itself
# TODO: Check existence of radiation files
def sample(
        cloudy_input,
        cloudy_source_path,
        output_folder,
        param_space,
        param_space_margins,

        rad_params=None,
        rad_params_margins=None,

        existing_data=None, # TODO: Rename
        initial_grid=10,
        perturbation_scale=None,

        filename_pattern=None,

        dex_threshold=0.1,
        over_thresh_max_fraction=0.1,
        dex_max_allowed_diff=0.5,

        random_samples_per_iteration=30,

        n_jobs=4,
        n_partitions=10,

        max_iterations=20,
        max_storage_gb=10,
        max_time=20,

        plot_iterations=True,
        debug_plot_2d=False
):
    """
    Main function of the library.
    Iteratively:
    1. Triangulate all current samples
    2. Interpolate all sample positions based on neighbors; get difference between interpolation and correct value
    3. Draw new samples based on interpolation errors
    
    ...until exit condition is reached, e.g. desired accuracy.

    :param cloudy_input:                    String. Cloudy input or path to cloudy input file.
    :param cloudy_source_path:              String. Path to cloudy's source/ folder
    :param output_folder:                   Path to output folder. Output folder should be empty/nonexistent.
    :param param_space:                     "key":(min, max)
                                            key: The variable to fill in in the cloudy file, e.g. z, T, hden, etc
                                                 Must match the string format syntax in the cloudy input string
                                            (min, max): The edges of the parameter space along that parameter axis
                                                        Not accounting for margins
    :param param_space_margins:             How much buffer to add around the parameter space for interpolation,
                                            per parameter. If one value per parameter: Relative, if two: Absolute
    :param rad_params:                      "key":(min, max).
                                            key: path to radiation input file; must contain data in format f(x) x
    :param rad_params_margins:              2 values: absolute. 1 value: relative
    :param rad_params_names:                TODO optional prettier names
    :param existing_data:                   Path to folder containing results of previous run to load. Should use same
                                            parameters and filename_pattern as current run.
    :param initial_grid:                    How many samples in each dimension in initial grid. 0 to disable
    :param dex_threshold:                   How close an interpolation has to be to the real value to be considered
                                            acceptable. Absolute value, in dex
    :param over_thresh_max_fraction:        Fraction of interpolations that may be over dex_threshold for interpolation
                                            to be considered "good enough".
    :param dex_max_allowed_diff:            Largest interpolation error permitted.
    :param random_samples_per_iteration:    How many completely random samples to add each iteration
    :param n_jobs:                          How many cloudy jobs to run in parallel
    :param n_partitions:                    How many partitions to use during Triangulation+Interpolation
    :param max_iterations:                  Exit condition: Quit after this many iterations
    :param max_storage_gb:                  Exit condition: Quit after this much storage space has been used
    :param max_time:                        Exit condition: Quit after this much time.
    :param plot_iterations:                 Whether to save plots of the parameter space each iteration.
    :return:
    """
    ####################################################################################################################
    ###############################################    Safety checks    ################################################
    ####################################################################################################################
    # Python does not understand that ~ is home
    cloudy_input = os.path.expanduser(cloudy_input)
    cloudy_source_path = os.path.expanduser(cloudy_source_path)
    output_folder = os.path.expanduser(output_folder)
    if existing_data is not None:
        existing_data = os.path.expanduser(existing_data)


    # Set up Output folder
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    elif len(os.listdir(output_folder)) != 0:
        # choice = input("Chosen output folder is not empty. Proceed? (y/n)")
        # if choice not in ["y", "Y", "yes", "Yes"]:
        #     print("Aborting...")
        #     exit()
        raise RuntimeWarning(f"Chosen Output {output_folder} folder is not empty")


    # Make sure cloudy exists
    if not os.path.exists(os.path.join(cloudy_source_path, "cloudy.exe")):
        raise RuntimeError("No cloudy.exe found at: " + str(cloudy_source_path))

    # Make sure all margins are given
    for key in param_space.keys():
        if key not in list(param_space_margins.keys()):
            raise RuntimeError("No margins given for: " + str(key))

    # Make sure we have a starting point - grid or preexisting data
    if not initial_grid and not existing_data:
        raise RuntimeError("Need to specify existing data or use initial_grid to set up a run")

    # Make sure cloudy input is valid, either as input file or as input string
    if os.path.exists(cloudy_input):
        with open(cloudy_input, "r") as file:
            cloudy_input = file.read()

        print("Interpreting cloudy input as file")
    else:
        print("Interpreting cloudy input directly")

    # Check if all "slots" in the cloudy template can be filled with the given parameters
    try:
        input_filled = cloudy_input.format_map({**{k : v[0] for k, v in param_space.items()}, **{"fname":""}})
    except SyntaxError:
        print("Error while filling cloudy input: Missing parameter or syntax error")

    ####################################################################################################################
    ################################################    Diagnostics    #################################################
    ####################################################################################################################
    print("\nCloudy template:")
    print(cloudy_input)
    print("\n")
    print("Output folder".ljust(50) + output_folder)
    print("Parameter space:")
    for dim, edges in param_space.items():
        print(f"\t{dim}:\t{edges}\t\tMargins: {param_space_margins[dim]}")

    print("Radiation background parameters:")
    if rad_params:
        for k, v in rad_params.items():
            print(f"\t{k}:\t{v[1]}\t\tMargins: {rad_params_margins[k]}\t\tSource file: {v[0]}")
    else:
        print("\tNone")

    print("Existing data".ljust(50) + str(existing_data))
    print("Points per dimension of initial grid ".ljust(50) + str(initial_grid))
    print("Filename pattern ".ljust(50) + str(filename_pattern))
    print()
    print("Threshold (dex)".ljust(50) + str(dex_threshold))
    print("Maximum difference (dex) ".ljust(50) + str(dex_max_allowed_diff))
    print("Maximum fraction of samples over threshold ".ljust(50) + str(over_thresh_max_fraction))
    print("Number of random samples per iteration ".ljust(50) + str(random_samples_per_iteration))
    print()
    print("Maximum number of iterations".ljust(50) + str(max_iterations))
    print("Maximum amount of storage (GB) ".ljust(50) + str(max_storage_gb))
    print("Maximum runtime ".ljust(50) + str(seconds_to_human_readable(max_time)))
    print()
    print("Maximum number of parallel jobs ".ljust(50) + str(n_jobs))
    print("Plot parameter space ".ljust(50) + str(plot_iterations))


    ####################################################################################################################
    ###################################################    Setup    ####################################################
    ####################################################################################################################
    time1 = time.time()
    rad_bg = _get_rad_bg_as_function(rad_params)

    # Compile coordinates and values into easily accessible lists
    coordinates = { # TODO Rename to core
        **param_space,
        **{k:v[1] for k, v in rad_params.items()}
    } #list(param_space.keys()) + list(rad_params.keys())
    margins = {**param_space_margins, **rad_params_margins}
    coord_list  = list(coordinates.keys())  # guaranteed order
    values = ["values"]  # TODO
    points = pd.DataFrame(columns=coord_list + values)

    if not filename_pattern:
        filename_pattern = "__".join(
            ["_".join(
                [c, "{" + str(c) + "}"]
            ) for c in coordinates.keys()]
        )
    else:
        for c in coordinates.keys():
            if "{" + str(c) + "}" not in filename_pattern:
                raise RuntimeError(f"Invalid file pattern: Missing {c}")

    # Load existing data if applicable
    existing_point_count = 0
    if existing_data:
        if not os.path.isdir(existing_data):
            raise RuntimeError("Specified existing data at " + str(existing_data) + " but is not a dir or does not exist")

        points = pd.concat((
            points,
            _load_existing_data(existing_data, filename_pattern, coordinates)
        ))

        existing_point_count = len(points.index)

    if initial_grid:
        print("Setting up grid")
        grid_points = _set_up_amorphous_grid(initial_grid, coordinates, margins, perturbation_scale)
        points = pd.concat((points, grid_points))

    # The "corners" of the parameter space will work as convex hull, ensuring that all other points are contained
    # within and are thus valid in the Delaunay interpolation
    corners = _get_corners(
        _get_param_space_with_margins(
            coordinates,
            margins,
        )
    )

    it0folder = os.path.join(output_folder, "iteration0/")
    if not os.path.exists(it0folder):
        os.mkdir(it0folder)

    corners = _cloudy_evaluate(cloudy_input, cloudy_source_path, it0folder, filename_pattern, corners, rad_bg, n_jobs)
    points = pd.concat((points, corners), ignore_index=True)
    points = points.drop_duplicates(subset=coord_list, keep="last", ignore_index=True)
    assert(not points.index.duplicated().any())
    points  = _cloudy_evaluate(cloudy_input, cloudy_source_path, it0folder, filename_pattern, points, rad_bg, n_jobs)
    time2 = time.time()
    print(round(time2 - time1, 2), "s to do initial setup")
    print("\n\n")


    assert (not points[values].isnull().to_numpy().any())

    ####################################################################################################################
    #################################################    Main Loop    ##################################################
    ####################################################################################################################
    iteration = 0
    iteration_time = 0
    timeBeginLoop = time.time()

    # number of dimensions
    # TODO: Generalize for multiple values
    N = len(points.columns) - 1
    while True:
        # reset index just to be on the safe side while partitioning later
        points = points.reset_index(drop=True)
        iteration += 1
        print("{:*^50}".format("Iteration {}".format(iteration)))

        len_pre_drop =  len(points.index)
        points = points.drop_duplicates(ignore_index=True)
        len_post_drop = len(points.index)
        n_dropped = len_pre_drop - len_post_drop

        if len_pre_drop > len_post_drop:
            print(f"Dropped {n_dropped} duplicate samples")

        # For plotting the triangulation TODO
        if debug_plot_2d:
            fig, ax = plt.subplots(1, n_partitions, figsize=(10 * n_partitions, 10))

        points["interpolated"] = np.nan
        points["diff"] = np.nan
        new_points = pd.DataFrame(columns=coordinates)

        timeA = time.time()
        for partition in range(n_partitions):
            assert(not points.index.duplicated().any())
            # TODO This part uses a lot of deep copies, might be inefficient
            subset = points.loc[partition::n_partitions].copy(deep=True)
            bigset = pd.concat((points, subset)).drop_duplicates(keep=False)

            # TODO again, generalize
            subset.loc[:,"interpolated"] = np.nan

            # Always include corners to ensure convex hull
            bigset = pd.concat((bigset, corners))
            bigset = bigset.drop_duplicates()
            # bigset = bigset.sort_index()

            # # Need to reset the bigset index due to the concatenation
            # # Since the bigset index is not relevant again - unlike the subset index - this should be fine
            bigset = bigset.reset_index()
            # print(bigset.loc[bigset.index.duplicated(keep=False)])
            assert(not bigset.index.duplicated().any())

            # dont consider points in margins
            for c, edges in coordinates.items():
                subset = subset.loc[(subset[c] > edges[0]) & (subset[c] < edges[1])]

            tri = spatial.Delaunay(bigset[coord_list].to_numpy())
            simplex_indices = tri.find_simplex(subset[coord_list].to_numpy())
            simplices       = tri.simplices[simplex_indices]
            transforms      = tri.transform[simplex_indices]

            # The following is adapted from
            # https://stackoverflow.com/questions/30373912/interpolation-with-delaunay-triangulation-n-dim/30401693#30401693
            # 1. barycentric coordinates of points; N-1
            bary = np.einsum(
                "ijk,ik->ij",
                transforms[:,:N,:N],
                subset[coord_list].to_numpy() - transforms[:, N, :]
            )

            # 2. Add dependent barycentric coordinate to obtain weights
            weights = np.c_[bary, 1 - bary.sum(axis=1)]

            # 3. Interpolation
            # TODO vectorize
            for i, index in enumerate(subset.index):
                subset.loc[index, "interpolated"] = np.inner(
                    bigset.loc[
                        bigset.index[simplices[i]],
                        "values"
                    ],
                    weights[i]
                )

            assert(not subset["interpolated"].isnull().to_numpy().any())

            # 4. Find points over threshold
            subset["diff"] = np.abs(subset["interpolated"] - subset["values"])

            # # Draw new samples by finding the simplex point that contributed most in the wrong direction,
            # # i.e. weight * value had the largest difference
            # # new sample is at halfway point between interpolation point and that simplex point
            #
            # # for plotting...
            # samples = pd.DataFrame(columns=coord_list)
            # for i, index in enumerate(subset[subset["diff"] > dex_threshold].index):
            #     simplex = tri.find_simplex(
            #         subset.loc[
            #             index,
            #             coord_list
            #         ].to_numpy()
            #     ).flatten()     # indices of simplices
            #
            #     simplex = tri.simplices[simplex] # indices of points
            #     simplex = simplex.flatten()
            #     simplex = bigset.loc[bigset.index[simplex], :]   # actual points
            #
            #
            #     largest_difference = 0
            #     largest_difference_pos = None
            #     w = weights[subset.index.get_loc(index)]
            #     for j, jndex in enumerate(simplex.index):
            #         diff = np.abs(subset.loc[index, "values"] - w[j] * simplex.loc[jndex, "values"])
            #
            #         if diff > largest_difference:
            #             largest_difference = diff
            #             largest_difference_pos = j
            #
            #     endpoint = simplex.iloc[largest_difference_pos,:]
            #
            #     new_point = {}
            #     for coord in coord_list:
            #         new_point[coord] = (float(subset.loc[index, coord]) + float(endpoint[coord])) / 2
            #
            #     new_point = pd.DataFrame(new_point, [0])
            #     new_points = pd.concat((new_points, new_point), ignore_index=True)
            #     samples = pd.concat((samples, new_point), ignore_index=True)




            # Draw new samples by finding geometric centers of the simplices containing the points over threshold
            if not subset[subset["diff"] > dex_threshold].empty:
                # TODO: Reusing old variables is not clean
                simplex_indices = tri.find_simplex(
                    subset.loc[
                        subset["diff"] > dex_threshold,
                        coord_list
                    ].to_numpy()
                )
                simplices       = tri.simplices[simplex_indices]

                # Shape: (N_simplices, N_points_per_simplex, N_coordinates_per_point)
                simplex_points = np.zeros(
                    (simplices.shape[0], simplices.shape[1], simplices.shape[1]-1)
                )

                for i in range(simplices.shape[0]):
                    try:
                        simplex_points[i] = bigset.loc[
                            bigset.index[simplices[i]], # effectively iloc, since simplices are positions, not indices
                            coord_list
                        ].to_numpy()
                    except ValueError:
                        print(simplex_points[i])
                        print(simplex_points[i].shape)
                        print(bigset.index[simplices[i]])
                        print(bigset.loc[
                            bigset.index[simplices[i]], # effectively iloc, since simplices are positions, not indices
                            coord_list
                        ].to_numpy())
                        print(bigset.loc[
                            bigset.index[simplices[i]], # effectively iloc, since simplices are positions, not indices
                            coord_list
                        ].to_numpy().shape)
                        exit()


                # new samples = averages (centers) of the points making up the simplices
                samples = np.sum(simplex_points, axis=1) / simplices.shape[1]
                samples = pd.DataFrame(samples, columns=coord_list)
                new_points = pd.concat((new_points, samples))
                new_points = new_points.drop_duplicates()
            else:
                print(f"No points over threshold in partition {partition}")

            # TODO: Triangulation plotting? Only really works in 2D
            if debug_plot_2d and not subset[subset["diff"] > dex_threshold].empty:
                ax[partition].triplot(bigset.loc[:, "T"], bigset.loc[:, "nH"], tri.simplices)
                ax[partition].plot(bigset.loc[:, "T"], bigset.loc[:, "nH"], "ko")
                for i, index in enumerate(subset[subset["diff"] > dex_threshold].index):
                    # Plot outliers
                    ax[partition].plot(subset.loc[index, "T"], subset.loc[index, "nH"], "ro")
                    # Plot lines to new samples
                    ax[partition].plot(
                        [subset.loc[index, "T"], samples.loc[i, "T"]],
                        [subset.loc[index, "nH"], samples.loc[i, "nH"]],
                        "red"
                    )


                    rect = patches.Rectangle(
                        (param_space["T"][0], param_space["nH"][0]),
                        param_space["T"][1] - param_space["T"][0],
                        param_space["nH"][1] - param_space["nH"][0],
                        edgecolor="green",
                        fill=False
                    )
                    ax[partition].add_patch(rect)

                ax[partition].set_title("Delaunay Partition " + str(n_partitions))
                ax[partition].set_xlabel("T")
                ax[partition].set_ylabel("nH")

            # Write interpolated and diffs back into original points dataframe
            # TODO: IMPORTANT Verify this looks as expected after all partitions are done
            # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.join.html
            # points = pd.merge(points, subset, axis=1, join="outer", sort=False)
            # Note: This requires the index to be as in the original, so we CAN NOT use reset_index
            # at the beginning!
            #points = points.join(subset)

            # couldnt find a pythonic/pandaic/vectorized solution
            # points.loc[subset.index, ["interpolated", "diff"]] = subset["interpolated", "diff"]
            points.loc[subset.index, "interpolated"] = subset["interpolated"]
            points.loc[subset.index, "diff"] = subset["diff"]
            assert(not subset.loc[:,"interpolated"].isnull().to_numpy().any())

        if debug_plot_2d:
            fig.savefig(
                os.path.join(output_folder, "Partitions" + str(iteration) + ".png"),
                bbox_inches="tight"
            )


        timeB = time.time()

        # Calculate some stats/diagnostics

        # all points
        total_count = len(points.index)

        # points not in margins
        # TODO: Wasteful
        df_copy = points.copy(deep=True)
        for c, edges in coordinates.items():
            df_copy = df_copy.loc[(df_copy[c] > edges[0]) & (df_copy[c] < edges[1])]
        in_bounds_count = len(df_copy.index)
        del df_copy

        # points in margins
        outside_bounds_count = total_count - in_bounds_count

        # number and fraction of points over threshold
        over_thresh_count    = len(points[points["diff"] > dex_threshold].index)
        over_thresh_fraction = over_thresh_count / in_bounds_count

        # biggest difference
        max_diff = points["diff"].max()


        # print("Number of nans in interpolated: ", points["interpolated"].isnull().sum())

        # diagnostics
        print("Total time to interpolate and sample:".ljust(50), seconds_to_human_readable(timeB - timeA))
        print("Total points before sampling".ljust(50), total_count)
        print("Points in core".ljust(50), in_bounds_count)
        print("Points in margins".ljust(50), outside_bounds_count)
        print(f"Points in core over threshold ({dex_threshold})".ljust(50), over_thresh_count,
              f"/ {in_bounds_count} ({round(over_thresh_fraction*100, 2)}%)")
        print("Number of new samples".ljust(50), len(new_points.index))
        print("Largest interpolation error".ljust(50), max_diff)

        if plot_iterations:
            _plot_parameter_space(points, new_points, coord_list, output_folder, iteration)


        # Check the various conditions for quitting
        threshold_condition = False
        if over_thresh_max_fraction is None or over_thresh_fraction < over_thresh_max_fraction:
            threshold_condition = True

        max_diff_condition = False
        if dex_max_allowed_diff is None or max_diff < dex_max_allowed_diff:
            max_diff_condition = True

        if threshold_condition and max_diff_condition:
            print(f"Reached desired accuracy. Quitting.")
            break

        if max_iterations and iteration > max_iterations - 1:
            print(f"Reached maximum number of iterations ({max_iterations}). Quitting.")
            break

        if max_storage_gb:
            gb_in_bytes = 1e9   # bytes per gigabyte
            output_size_gb = get_folder_size(output_folder) / gb_in_bytes
            # iteration0: initial setup
            prev_iteration_gb = get_folder_size(os.path.join(output_folder, f"iteration{iteration-1}")) / gb_in_bytes

            if output_size_gb + prev_iteration_gb > max_storage_gb:
                print(f"Reached maximum allowed storage ({output_size_gb} GB/{max_storage_gb} GB), accounting " +
                      f"for size of previous iteration ({prev_iteration_gb} GB). Quitting.")
                break

        if max_time is not None and max_time < (time.time() - timeBeginLoop) + iteration_time:
            print(f"Reached maximum calculation time ({seconds_to_human_readable(time.time() - timeBeginLoop)} " +
                  f"/ {seconds_to_human_readable(max_time)}), accounting for length of previous iteration " +
                  f"({seconds_to_human_readable(iteration_time)}). Quitting")
            break


        iteration_folder = os.path.join(output_folder, f"iteration{iteration}")
        if not os.path.exists(iteration_folder):
            os.mkdir(iteration_folder)
        else:
            # choice = input(f"Attempting to write into non-empty folder {iteration_folder}. Proceed? (y/n)")
            #
            # if choice not in ["y", "Y", "yes", "Yes"]:
            #     print("Aborting...")
            #     exit()
            raise RuntimeWarning(f"Iteration folder {iteration_folder} is not empty")


        # Add completely random new points
        random_points = [] # list of dicts for DataFrame constructor
        for i in range(random_samples_per_iteration):
            random_points.append(
                {
                    coord : (edge[1] - edge[0]) * np.random.random() + edge[0]
                    for coord, edge in coordinates.items()
                }
            )

        new_points = pd.concat((new_points, pd.DataFrame(random_points)))

        points = points.drop(["interpolated", "diff"], axis=1)
        new_points["values"] = np.nan
        points = pd.concat((points, new_points)).drop_duplicates().reset_index(drop=True)

        time3 = time.time()
        points = _cloudy_evaluate(
            cloudy_input,
            cloudy_source_path,
            iteration_folder,
            filename_pattern,
            points,
            rad_bg,
            n_jobs
        )
        print(f"{round(time.time() - time3, 2)}s to evaluate new points")
        print(f"Total time: {seconds_to_human_readable(round(time.time() - timeBeginLoop, 2))}")
        iteration_time = time.time() - timeA
        print(f"Time for current iteration: {seconds_to_human_readable(round(iteration_time, 2))}")

        print("\n\n\n")

    total_time = time.time() - timeBeginLoop
    total_time_readable = seconds_to_human_readable(total_time)

    print()
    print(f"Run complete; Calculated at least {len(points.index) - existing_point_count} new points " +
          f"({existing_point_count} initially loaded, {len(points.index)} total). Time to complete: " + total_time_readable)


def _get_corners(param_space):
    """
    For a given parameter space, returns a pandas Dataframe containing its corners, with a nan values column.

    TODO: Generalize values

    :param param_space:     As given to sample()
    :return:                Pandas Dataframe
    """
    keys = list(param_space.keys())
    values = [param_space[key] for key in param_space.keys()] # ensuring order

    corners = []
    for product in itertools.product(*values):
        corners.append({
            keys[i]:product[i] for i in range(len(product))
        })

    corners = pd.DataFrame(corners)
    corners["values"] = np.nan

    return corners


def _plot_parameter_space(points, new_points, coord_list, output_folder, suffix):
    """
    Plot parameter space using a seaborn pairplot.

    :param points:          Points dataframe as used in sample; "old points"
    :param new_points:      Points dataframe as used in sample; "new points" (different color)
    :param coord_list:      coord_list as used in sample; keys are
    :param output_folder:   Folder to save plots in
    :param suffix:          File will be saved as "iteration{suffix}.png"
    :return:
    """
    begin = time.time()
    points["hue"]     = "Older Iterations"
    new_points["hue"] = "Latest Iteration"
    fullpoints = pd.concat((points, new_points))

    grid = sns.pairplot(
        fullpoints,
        diag_kind="hist",
        vars=coord_list,
        hue="hue",
        markers="o",
        plot_kws={
            "s":1,
            "marker":"o",
            "edgecolor":None
        },
        diag_kws={
            "bins":50
        },
        height=6
    )
    grid.savefig(os.path.join(output_folder, "iteration{}.png".format(suffix)))

    points.drop("hue", axis=1, inplace=True)
    new_points.drop("hue", axis=1, inplace=True)

    end = time.time()
    print(f"{round(end - begin, 2)}s to plot current iteration")


def _load_point(filename, filename_pattern, coordinates):
    """
    Load Ctot from the given cloudy cooling output ("*.cool") file. TODO: Generalize
    Filename needs to contain the full path to the file.

    The filename pattern is used to parse the coordinates at which Ctot is given.
    TODO Load from inside the file itself instead?

    :param filename:
    :param filename_pattern:
    :return:                    Dict
    """
    result = parse(filename_pattern, os.path.splitext(os.path.basename(filename))[0])
    point = result.named

    for coordinate in coordinates:
        if coordinate not in list(point.keys()):
            raise RuntimeError(f"Missing coordinate {coordinate} while trying to read in file {filename}")

    point["values"] = float(np.loadtxt(filename, usecols=3))

    return point


def _load_existing_data(folder, filename_pattern, coordinates):
    """
    Loads Ctot from all files ending in .cool in the given folder and subfolders.

    # TODO Generalize file endings, maybe

    :param folder:
    :return:            Dataframe of points
    """
    # Find all files in the specified folder (and subfolders)
    filenames = []
    for dirpath, dirnames, fnames in os.walk(folder):
        filenames += [os.path.join(dirpath, fname) for fname in fnames]

    points = [] # list of dicts for DataFrame constructor

    for filename in filenames:
        if filename.endswith(".cool"):
            points.append(_load_point(filename, filename_pattern, coordinates))

    points = pd.DataFrame(points).astype(float)
    return points


def _set_up_grid(num_per_dim, parameter_space, margins, perturbation_scale=None):
    """
    Fills parameter space with a regular grid as starting point. Takes margins into account.
    TODO Specify number of samples per dimension

    :param num_per_dim:             Amount of samples in each dimension (int)
    :param parameter_space:         Parameter space as used in sample(); I think its called coordinates there TODO
    :param margins:                 Margins of all parameters as used in sample()
    :param perturbation_scale:      If given, perturbations of this (max) size are added to each interior point.
                                    Scale is relative to extent of space along that dimension
    :return:                        Dataframe of points
    """
    param_space_with_margins = _get_param_space_with_margins(
        parameter_space, margins
    )

    param_names =  list(param_space_with_margins.keys())
    param_values = [param_space_with_margins[k] for k in param_names]
    param_grid = [np.linspace(*pv, num_per_dim) for pv in param_values]


    # https://stackoverflow.com/a/46744050
    index = pd.MultiIndex.from_product(param_grid, names=param_names)
    points = pd.DataFrame(index=index).reset_index()

    if perturbation_scale:
        for coord in param_names:
            coord_max = max(param_space_with_margins[coord])
            coord_min = min(param_space_with_margins[coord])
            perturb_coord_scale = perturbation_scale * np.abs(coord_max - coord_min)
            for i in points.loc[
                (points[coord] > coord_min) &
                (points[coord] < coord_max),    # exclude edges
                coord].index:
                points.loc[i, coord] += perturb_coord_scale * np.random.random() - 0.5 * perturb_coord_scale

    points["values"] = np.nan

    return points


def _set_up_amorphous_grid(num_per_dim, parameter_space, margins, perturbation_scale):
    """
    Uses Poisson disc sampling to set up an amorphous (instead of regular) grid
    https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf

    First sets up a regular grid using _set_up_grid(), then cuts out the middle to leave a regular hull, then fills
    it with poisson disc sampling. Not pretty but as long as it works

    :param num_per_dim:             See _set_up_grid()
    :param parameter_space:         See _set_up_grid()
    :param margins:                 See _set_up_grid()
    :param perturbation_scale:      NOT used for perturbation, but actually passed as radius to the poisson disc
                                    sampling function TODO Prettify
    :return:
    """
    # Take regular grid, no perturbation
    points_full = _set_up_grid(num_per_dim, parameter_space, margins)

    # Cut out all the "middle values"
    points = pd.DataFrame(columns=points_full.columns)
    for column in points_full.columns:
        points = pd.concat((
            points,
            points_full.loc[
                (points_full[column] == points_full[column].min()) | (points_full[column] == points_full[column].max()),
                :
            ]
        )).drop_duplicates(ignore_index=True)


    # Fill the middle using Poisson disc sampling
    # Generate nd unit cube filled with samples
    if not perturbation_scale:
        perturbation_scale = 0.1
    core = poisson_disc_sampling(
        np.array([[0, 1]] * len(list(parameter_space.keys()))),
        perturbation_scale
    )

    # Stretch cube to fit parameter space
    core_columns = list(points.columns)
    core_columns.remove("values")
    core = pd.DataFrame(core, columns=core_columns)
    for param, limits in parameter_space.items():
        core[param] = (max(limits) - min(limits)) * core[param] + min(limits)

    core["values"] = np.nan

    points = pd.concat((points, core), ignore_index=True)

    return points


def _get_param_space_with_margins(parameter_space, margins):
    """
    Returns entire parameter space with margins "applied".

    :param parameter_space:
    :param margins:
    :return:
    """
    param_space_with_margins = {}

    for k, v in margins.items():
        if hasattr(v, "__iter__"):  # sequence
            if len(v) != 2:
                raise RuntimeError("Margins must either be number or iterable of length 2: " + str(v))
            param_space_with_margins[k] = (min(v), max(v))
        elif type(v) == float or type(v) == int:
            interval_length = abs(parameter_space[k][1] - parameter_space[k][0])
            param_space_with_margins[k] = (
                # cast to np array for element wise operations
                min(np.array(parameter_space[k]) - interval_length * margins[k]),
                max(np.array(parameter_space[k]) + interval_length * margins[k])
            )

    return param_space_with_margins


def _get_rad_bg_as_function(rad_params):
    """
    Take rad_params as given to sample(). Load the radiation files specified in the keys. Interpolate each spectrum
    using InterpolatedUnivariateSpline.

    Limit energy range to what cloudy can handle. Then resample it logarithmically at ~<4000 points (maximum input
    line count for cloudy input files)

    Return function that:
    - Takes a dict containing a multiplier for each of these radiation components
    - Evaluates each radiation component's interpolator at all energies and multiplies with respective factor
    - Returns the sum to use as the total radiation background
    :param rad_params:
    :return:
    """
    if not rad_params:
        return lambda x: None

    interpolators = {}
    rad_data = {}

    # Load radiation data and build interpolators
    for k, v in rad_params.items():
        rad_data[k] = np.loadtxt(v[0])
        interpolators[k] = interpolate.InterpolatedUnivariateSpline(rad_data[k][:,0], rad_data[k][:,1], k=1, ext=1)

    valid_keys = list(rad_params.keys())

    combined_energies = np.sort(np.concatenate([rad_data[k] for k in rad_params.keys()]).flatten())
    # Cloudy only considers energies between 3.04e-9 Ryd and 7.453e6 Ryd, see Hazy p. 33
    combined_energies = np.clip(combined_energies, 3.04e-9, 7.354e6)
    x = np.geomspace(min(combined_energies), max(combined_energies), 3900)

    def rad_bg_function(rad_multipliers):
        # TODO: Verify function behaves as expected (plots!)
        # Note: Cloudy can only handle 4000 lines of input at a time...
        # f_nu = sum([interpolators[k](combined_energies) * rad_multipliers[k] for k in rad_multipliers.keys() if k in valid_keys])
        # TODO: Make sure this works with logspace, or find better solution altogether
        f_nu = sum([interpolators[k](x) * rad_multipliers[k] for k in rad_multipliers.keys() if k in valid_keys])

        hstack_shape = (f_nu.shape[0], 1)
        spectrum = np.hstack(
            (
                np.reshape(f_nu, hstack_shape),
                # np.reshape(combined_energies, hstack_shape)
                np.reshape(x, hstack_shape)
            )
        )

        spectrum = spectrum[(spectrum[:,0] != 0) & (spectrum[:,1] != 0)]

        # Cloudy wants the log of the frequency
        # TODO Check factor 4pi
        spectrum[:,0] = np.log10(spectrum[:,0])

        # Note: We save files in the format "f(x) x",
        # because we use "f(x) at x" in the original f_nu cloudy command,
        # and it its more consistent that way
        # however
        # cloudy may *also* interpret interpolate/continue inputs as "x f(x)"
        # so we have to make sure that our first values are in the correct range
        # Quote hazy p. 46
        # "CLOUDY assumes that the log of the energy in Rydbergs was entered if thefirst number is negative;
        # that the log of the frequency (Hz) was entered if the first number isgreater than 5;
        # and linear Rydbergs otherwise."
        #
        # Yes, it is stupid
        # Yes, i should probably use the table SED command instead
        # TODO


        return spectrum

    return rad_bg_function


def _f_nu_to_string(f_nu):
    """
    Helper function to turn a spectrum into a string that can be used in a cloudy input file.

    :param f_nu:    Numpy array, Shape (N, 2). TODO whats on which side
    :return:
    """
    if f_nu is None:
        return ""

    # find the entry closest to 1 Ryd for cloudy normalization
    mindex = np.argmin(np.abs(f_nu[:,0] - 1))   # min index
    out = f"f(nu) = {f_nu[mindex,0]} at {f_nu[mindex,1]}\ninterpolate ({f_nu[1,1]}  {f_nu[1,0]})\n"

    for i in range(2, f_nu.shape[0]):
        out += f"continue ({f_nu[i,1]}  {f_nu[i,0]})\n"

    out += "iterate to convergence"
    return out


def _cloudy_evaluate(input_file, path_to_source, output_folder, filename_pattern, points, rad_bg_function, n_jobs):
    """
    Evaluate the given points with cloudy using the given file template.

    :param input_file:          String representing the cloudy in put file template
    :param path_to_source:      As in sample()
    :param output_folder:       Folder to move files to after evaluation
    :param filename_pattern:    As in sample()
    :param points:              Dataframe, as in sample()
    :param rad_bg_function:     As returned from _get_rad_bg_as_function()
    :return:
    """
    filenames = []

    for i in points.loc[points["values"].isnull()].index:
        row = points.loc[i].to_dict()
        filestring = input_file.format_map(
            {
                **row,
                **{"fname":filename_pattern.format_map(row)}
            }
        )
        filestring += "\n"
        filestring += _f_nu_to_string(rad_bg_function(row))

        # I'd love to put this somewhere more convenient, but cloudy cant handle subdirectories
        with open(filename_pattern.format_map(row) + ".in", "w") as f:
            f.write(filestring)

        filenames.append(filename_pattern.format_map(row))

    path_to_filenames = os.path.join(output_folder, "_filenames.temp")
    with open(path_to_filenames, "w") as file:
        for filename in filenames:
            file.write(filename + "\n")

    path_to_exe = os.path.join(path_to_source, "cloudy.exe")
    os.system(f"parallel -j {n_jobs} '{path_to_exe} -r' :::: {path_to_filenames}")

    # TODO: Find a way to generically incorporate other data here

    # TODO: Implement reading and moving files to output folder. Cloudy file saving needs to be solved first.
    # Step 4: Read cooling data
    # for now only Ctot
    for i, index in enumerate(points.loc[points["values"].isnull()].index):
        points.loc[index,"values"] = np.log10(np.loadtxt(filenames[i] + ".cool", usecols=3))

        os.system(f"mv {filenames[i]}* {output_folder}")


    # Step 5: Move files to output folder
    # match_pattern = filename_pattern.format(*["*" for x in range(filename_pattern.count("{"))])
    # os.system("mv " + match_pattern + " " + output_folder)

    # mv has a maximum number of arguments, this is avoided by using this construction instead of mv *
    #os.system(f"ls | grep -P {match_pattern} | xargs -I % -n 1000 mv -t {output_folder} %")

    os.remove(path_to_filenames)

    #points["values"] = np.log10(points["values"])


    return points
