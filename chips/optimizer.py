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

# TODO Figure out how to consolidate the hydrogen fractions
# TODO Add option for net cooling!

def sample(
        cloudy_input,
        cloudy_source_path,
        output_folder,
        output_filename,
        param_space,
        param_space_margins,

        rad_params=None,
        rad_params_margins=None,

        existing_data=None,
        initial_grid=5,
        poisson_disc_scale=None,
        significant_digits=3, # Set None for no rounding

        filename_pattern=None,

        accuracy_threshold=0.1,
        error_fraction=0.1,
        max_error=0.5,

        random_samples_per_iteration=30,

        interp_column=3,
        suppress_interp_column_warning=False,
        save_fractions=False,

        n_jobs=4,
        n_partitions=10,

        max_iterations=20,
        max_storage_gb=10,
        max_time=20,
        max_samples=100000,

        sep=",",

        plot_iterations=True,
        cleanup=None
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
    :param output_filename:                 Base name of the output files (.points, .tris, .neighbors)
    :param param_space:                     "key":(min, max)
                                            key: The variable to fill in in the cloudy file, e.g. z, T, hden, etc
                                                 Must match the string format syntax in the cloudy input string
                                            (min, max): The edges of the parameter space along that parameter axis
                                                        Not accounting for margins
    :param param_space_margins:             How much buffer to add around the parameter space for interpolation,
                                            per parameter. If one value per parameter: Relative, if two: Absolute
    :param rad_params:                      "key":(path, (min, max), scaling).
                                            key: name of the parameter
                                            path: path to spectrum file, must contain data in format f(x) x
                                            (min, max): The edges of the parameter space
                                            scaling: "lin" or "log"
    :param rad_params_margins:              2 values: absolute. 1 value: relative. see param_space_margins
    :param existing_data:                   Iterable of strings. Either filenames or foldernames.
                                            If filename: File is loaded as .points file as output by previous runs.
                                            If foldername: All cloudy output files in folder and subfolders will
                                            be read in. May fail if the filename_patterns differ.
    :param initial_grid:                    How many samples to use in each dimension in the grid-like hull of the
                                            margins
    :param poisson_disc_scale:              Which radius to use for poisson disc sampling in the initial sampling
                                            distribution. You can expect slightly less than 1/r^d points.
    :param significant_digits:              How many decimals to round coordinates to (before evaluation). Use to
                                            avoid e.g. long filenames due to floating point errors. Beware of using
                                            too low a value; becomes essentially a grid. Recommended 3-6.
    :param filename_pattern:                Filename pattern to use for the output files. I highly recommend you leave
                                            this at default, unless you have a good reason and know what you're doing.
    :param accuracy_threshold:              How close an interpolation has to be to the real value to be considered
                                            acceptable. Absolute value, in dex
    :param error_fraction:                  Fraction of interpolations that may be over accuracy_threshold for
                                            interpolationto be considered "good enough".
    :param max_error:                       Largest interpolation error permitted.
    :param random_samples_per_iteration:    How many completely random samples to add each iteration
    :param interp_column:                   The column to read from .cool files for data. 3 = cooling, 2 = heating
    :param suppress_interp_column_warning:  Do not throw warning when attempting to read columns other than 2 or 3.
    :param save_fractions:                  Save electron and hydrogen fractions to points file. For use with gadget.
    :param n_jobs:                          How many cloudy jobs to run in parallel
    :param n_partitions:                    How many partitions to use during Triangulation+Interpolation
    :param max_iterations:                  Exit condition: Quit after this many iterations
    :param max_storage_gb:                  Exit condition: Quit after this much storage space has been used
    :param max_time:                        Exit condition: Quit after this much time (seconds).
    :param max_samples:                     Exit condition: Quit after this many samples.
    :param plot_iterations:                 Whether to save plots of the parameter space each iteration.
    :param cleanup:                         Whether to clean up the working directory after each iteration.
                                            None: Do not perform cleanup.
                                            "outfiles": Delete all cloudy output files (which take up the bulk of disk
                                            space)
                                            "full": Clean entire working directory. (Not recommended)
    :param sep:                             Separator/Delimiter to use in .csv files. I recommend you dont change it.
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
        if type(existing_data) is str:
            existing_data = os.path.expanduser(existing_data)
        elif type(existing_data) is tuple or type(existing_data) is list:
            for i in range(len(existing_data)):
                existing_data[i] = os.path.expanduser(existing_data[i])
        else:
            raise RuntimeWarning(f"Invalid type for existing data: {type(existing_data)}")


    # Set up Output folder
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    elif len(os.listdir(output_folder)) != 0:
        raise RuntimeWarning(f"Chosen Output {output_folder} folder is not empty")


    # Make sure cloudy exists
    if not os.path.exists(os.path.join(cloudy_source_path, "cloudy.exe")):
        raise RuntimeError("No cloudy.exe found at: " + str(cloudy_source_path))

    # Make sure all margins are given
    for key in param_space.keys():
        if key not in list(param_space_margins.keys()):
            raise RuntimeError("No margins given for: " + str(key))

    # Make sure rad_params are valid
    for k, v in rad_params.items():
        if len(v) != 3:
            raise RuntimeError(f"Invalid length in radiation parameter with key {k}: {v} has {len(v)} != 3 elements")
        if not os.path.isfile(v[0]):
            raise RuntimeError(f"Not a valid spectrum file: {v[0]}")
        if v[2] not in ["log", "lin"]:
            raise RuntimeError(f"Radiation parameter can be interpreted as 'log' or 'lin'; got {v[2]}")
        if k not in list(rad_params_margins.keys()):
            raise RuntimeError(f"No margins given for radiation parameter {k}")

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
    except (KeyError, SyntaxError):
        print("Error while filling cloudy input: Missing parameter or syntax error")
        exit()

    # Check if cleanup parameter is valid
    if cleanup not in [None, "outfiles", "full"]:
        raise RuntimeError(f"Invalid value for cleanup parameter: {cleanup}")

    # Check if the column we are reading from the .cool file is valid
    if not suppress_interp_column_warning:
        if interp_column != 2 and interp_column != 3:
            raise RuntimeWarning(f"interp_column is not 2 (heating) or 3 (cooling), but {interp_column}. "
                                 "If you really wish to proceed, set suppress_interp_column_warning=True.")
    else:
        print("Interpolation column warning suppressed.")

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
    print("Interpolation column ".ljust(50) + str(interp_column), sep=" ")
    if interp_column == 2:
        print("(heating)")
    elif interp_column == 3:
        print("(cooling)")
    else:
        print("(unknown)")
    print()
    print("Poisson disc sampling radius:".ljust(50) + str(poisson_disc_scale))
    print("Threshold (dex)".ljust(50) + str(accuracy_threshold))
    print("Maximum difference (dex) ".ljust(50) + str(max_error))
    print("Maximum fraction of samples over threshold ".ljust(50) + str(error_fraction))
    print("Number of random samples per iteration ".ljust(50) + str(random_samples_per_iteration))
    print()
    print("Maximum number of iterations".ljust(50) + str(max_iterations))
    print("Maximum amount of storage (GB) ".ljust(50) + str(max_storage_gb))
    print("Maximum runtime ".ljust(50) + str(seconds_to_human_readable(max_time)))
    print("Maximum number of samples".ljust(50) + str(max_samples))
    print()
    print("Maximum number of parallel jobs ".ljust(50) + str(n_jobs))
    print("Plot parameter space ".ljust(50) + str(plot_iterations))


    ####################################################################################################################
    ###################################################    Setup    ####################################################
    ####################################################################################################################
    time1 = time.time()
    rad_bg = _get_rad_bg_as_function(rad_params, output_folder)

    # Compile coordinates and values into easily accessible lists
    core = {
        **param_space,
        **{k:v[1] for k, v in rad_params.items()}
    }
    margins = {**param_space_margins, **rad_params_margins}
    coord_list  = list(core.keys())  # guaranteed order
    values = ["values"]
    points = pd.DataFrame(columns=coord_list + values)

    if not filename_pattern:
        filename_pattern = "__".join(
            ["_".join(
                [c, "{" + str(c) + "}"]
            ) for c in core.keys()]
        )
    else:
        for c in core.keys():
            if "{" + str(c) + "}" not in filename_pattern:
                raise RuntimeError(f"Invalid file pattern: Missing {c}")

    # Load existing data if applicable
    existing_point_count = 0
    if existing_data:
        for dpath in existing_data:
            if os.path.isfile(dpath):
                print("Attempting to read datafile", dpath)
                points = pd.concat((
                    points,
                    pd.read_csv(dpath, delimiter=sep)
                ))

            elif os.path.isdir(dpath):
                print("Attempting to recursively read raw data from folder", dpath)
                points = pd.concat((
                    points,
                    load_existing_raw_data(existing_data, filename_pattern, coord_list, interp_column)
                ))
            else:
                raise RuntimeError(f"Error: {dpath} is not a valid file or folder")

        existing_point_count = len(points.index)

    if initial_grid:
        print("Setting up grid")
        grid_points = _set_up_amorphous_grid(initial_grid, core, margins, poisson_disc_scale)
        if significant_digits is not None:
            grid_points = grid_points.round(significant_digits)
        points = pd.concat((points, grid_points))

    # The "corners" of the parameter space will work as convex hull, ensuring that all other points are contained
    # within and are thus valid in the Delaunay interpolation
    corners = _get_corners(
        _get_param_space_with_margins(
            core,
            margins,
        )
    )
    if significant_digits is not None:
        corners = corners.round(significant_digits)

    # iteration 0 - setup
    it0folder = os.path.join(output_folder, "iteration0/")
    if not os.path.exists(it0folder):
        os.mkdir(it0folder)

    # Corners are evaluated separately because they are used on their own later
    corners = _cloudy_evaluate(
        cloudy_input,
        cloudy_source_path,
        it0folder,
        filename_pattern,
        corners,
        rad_bg,
        n_jobs,
        interp_column
    )
    points = pd.concat((points, corners), ignore_index=True)

    points = points.drop_duplicates(subset=coord_list, keep="last", ignore_index=True)
    assert(not points.index.duplicated().any())

    points  = _cloudy_evaluate(
        cloudy_input,
        cloudy_source_path,
        it0folder,
        filename_pattern,
        points,
        rad_bg,
        n_jobs,
        interp_column
    )
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
    D = len(points.columns) - 1
    while True:
        # reset index just to be on the safe side while partitioning later
        points = points.reset_index(drop=True)
        iteration += 1
        print("{:*^50}".format(f"Iteration {iteration}"))

        drop_duplicates_and_print(points)

        points["interpolated"] = np.nan
        points["diff"] = np.nan
        new_points = pd.DataFrame(columns=core)

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

            # Need to reset the bigset index due to the concatenation
            # Since the bigset index is not relevant again - unlike the subset index - this should be fine
            bigset = bigset.reset_index()

            # dont consider points in margins
            for c, edges in core.items():
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
                transforms[:,:D,:D],
                subset[coord_list].to_numpy() - transforms[:, D, :]
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

            # Draw new samples by finding geometric centers of the simplices containing the points over threshold
            if not subset[subset["diff"] > accuracy_threshold].empty:
                # TODO: Reusing old variables is not clean
                simplex_indices = tri.find_simplex(
                    subset.loc[
                        subset["diff"] > accuracy_threshold,
                        coord_list
                    ].to_numpy()
                )
                simplices       = tri.simplices[simplex_indices]

                # Shape: (N_simplices, N_points_per_simplex, N_coordinates_per_point)
                simplex_points = np.zeros(
                    (simplices.shape[0], simplices.shape[1], simplices.shape[1]-1)
                )

                for i in range(simplices.shape[0]):
                    simplex_points[i] = bigset.loc[
                        bigset.index[simplices[i]], # effectively iloc, since simplices are positions, not indices
                        coord_list
                    ].to_numpy()

                # new samples = averages (centers) of the points making up the simplices
                samples = np.sum(simplex_points, axis=1) / simplices.shape[1]
                samples = pd.DataFrame(samples, columns=coord_list)
                new_points = pd.concat((new_points, samples))
                new_points = new_points.drop_duplicates()
            else:
                print(f"No points over threshold in partition {partition}")

            # Write interpolated and diffs back into original points dataframe
            points.loc[subset.index, "interpolated"] = subset["interpolated"]
            points.loc[subset.index, "diff"] = subset["diff"]
            assert(not subset.loc[:,"interpolated"].isnull().to_numpy().any())

        timeB = time.time()

        # Calculate some stats/diagnostics

        # all points
        total_count = len(points.index)

        # points not in margins
        # TODO: Wasteful
        df_copy = points.copy(deep=True)
        for c, edges in core.items():
            df_copy = df_copy.loc[(df_copy[c] > edges[0]) & (df_copy[c] < edges[1])]
        in_bounds_count = len(df_copy.index)
        del df_copy

        # points in margins
        outside_bounds_count = total_count - in_bounds_count

        # number and fraction of points over threshold
        over_thresh_count    = len(points[points["diff"] > accuracy_threshold].index)
        over_thresh_fraction = over_thresh_count / in_bounds_count

        # biggest difference
        max_diff = points["diff"].max()

        # diagnostics
        print("Total time to interpolate and sample:".ljust(50), seconds_to_human_readable(timeB - timeA))
        print("Total points before sampling".ljust(50), total_count)
        print("Points in core".ljust(50), in_bounds_count)
        print("Points in margins".ljust(50), outside_bounds_count)
        print(f"Points in core over threshold ({accuracy_threshold})".ljust(50), over_thresh_count,
              f"/ {in_bounds_count} ({round(over_thresh_fraction*100, 2)}%)")
        print("Number of new samples".ljust(50), len(new_points.index))
        print("Largest interpolation error".ljust(50), max_diff)

        if plot_iterations:
            _plot_parameter_space(points, new_points, coord_list, output_folder, iteration)

        if cleanup:
            print("Performing cleanup, mode:", cleanup)
            folder = os.path.join(output_folder, f"iteration{iteration - 1}")
            if cleanup == "outfiles":
                for fname in os.listdir(folder):
                    if fname.endswith(".out"):
                        os.remove(os.path.join(folder, fname))
            if cleanup == "full":
                for fname in os.listdir(folder):
                    os.remove(os.path.join(folder, fname))

            print("Cleanup complete")


        # Check the various conditions for quitting
        threshold_condition = False
        if error_fraction is None or over_thresh_fraction < error_fraction:
            threshold_condition = True

        max_diff_condition = False
        if max_error is None or max_diff < max_error:
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
                  f"({seconds_to_human_readable(iteration_time)}). Quitting.")
            break

        if max_samples and total_count + len(new_points.index) > max_samples:
            print(f"Reached maximum number of samples ({total_count} + {len(new_points.index)} / {max_samples}). Quitting.")
            break


        iteration_folder = os.path.join(output_folder, f"iteration{iteration}")
        if not os.path.exists(iteration_folder):
            os.mkdir(iteration_folder)
        else:
            raise RuntimeWarning(f"Iteration folder {iteration_folder} is not empty")


        # Add completely random new points
        random_points = [] # list of dicts for DataFrame constructor
        for i in range(random_samples_per_iteration):
            random_points.append(
                {
                    coord : (edge[1] - edge[0]) * np.random.random() + edge[0]
                    for coord, edge in core.items()
                }
            )

        new_points = pd.concat((new_points, pd.DataFrame(random_points)))

        points = points.drop(["interpolated", "diff"], axis=1)
        new_points["values"] = np.nan
        if significant_digits is not None:
            new_points = new_points.round(significant_digits)
        points = pd.concat((points, new_points)).drop_duplicates().reset_index(drop=True)

        time3 = time.time()
        points = _cloudy_evaluate(
            cloudy_input,
            cloudy_source_path,
            iteration_folder,
            filename_pattern,
            points,
            rad_bg,
            n_jobs,
            interp_column
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

    points = points.drop(["interpolated", "diff"], axis=1)

    print("Building and saving final Delaunay triangulation...")
    points.to_csv(os.path.join(output_folder, output_filename + ".fullpoints"), index=False)
    # drop margins
    for coord in core.items():
        points = points[points[coord[0]] >= coord[1][0]]
        points = points[points[coord[0]] <= coord[1][1]]

    tri = spatial.Delaunay(points[coord_list].to_numpy())
    np.savetxt(os.path.join(output_folder, output_filename + ".tris"), tri.simplices.astype(int), delimiter=sep, fmt="%i")
    np.savetxt(os.path.join(output_folder, output_filename + ".neighbors"), tri.neighbors.astype(int), delimiter=sep, fmt="%i")
    points.to_csv(os.path.join(output_folder, output_filename + ".points"), index=False)

    if save_fractions:
        print("save_fractions == true, Loading and merging hydrogen and electron fractions")
        fracs = fracs = load_existing_raw_data(
            output_folder,
            filename_pattern,
            coord_list,
            [4, 5, 6, 7],
            file_ending=".overview",
            column_names=["ne", "H2", "HI", "HII"]
        ).drop_duplicates()

        points = points.drop_duplicates()
        merged = points.merge(
            fracs,
            "inner",
            on=coord_list
        )

        merged.to_csv(os.path.join(output_folder, output_filename + ".points"), index=False)

        if len(points.index) != len(merged.index):
            print("The length of the points table changed during the merge with the electron/hydrogen fractions.\n"
                  "This is odd. You may want to double check the results and make sure the triangulation didnt break.")

    print("Done")

    return points


def _get_corners(param_space):
    """
    For a given parameter space, returns a pandas Dataframe containing its corners, with a nan values column.

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


def _load_point(filename, filename_pattern, coordinates, column_index):
    """
    Load Ctot or Htot from the given cloudy cooling output ("*.cool") file. Applies log10 to the read values.
    Filename needs to contain the full path to the file.

    The filename pattern is used to parse the coordinates at which Ctot is given.

    :param filename:            file to load
    :param filename_pattern:    how to parse filename -> coordinates
    :param coordinates:         Coordinate list
    :param column_index:        int or list
    :return:                    Dict
    """
    result = parse(filename_pattern, os.path.splitext(os.path.basename(filename))[0])
    point = result.named

    for coordinate in coordinates:
        if coordinate not in list(point.keys()):
            raise RuntimeError(f"Missing coordinate {coordinate} while trying to read in file {filename}")

    try:
        point["values"] = np.log10(float(np.loadtxt(filename, usecols=column_index)))
        return point
    except:
        print("Could not read point from file", filename)
        return None


def _load_point_multiple_values(filename, filename_pattern, coordinates, columns):
    """
    Like _load_point, but loads multiple values and does not apply log10.

    :param filename:
    :param filename_pattern:
    :param coordinates:
    :param columns:             Dict. Keys are column names, values are column indices
    :return:                    Dict
    """
    column_names = list(columns.keys())
    column_indices = [columns[k] for k in column_names]

    result = parse(filename_pattern, os.path.splitext(os.path.basename(filename))[0])
    point = result.named

    for coordinate in coordinates:
        if coordinate not in list(point.keys()):
            raise RuntimeError(f"Missing coordinate {coordinate} while trying to read in file {filename}")

    try:
        vals = np.loadtxt(filename, usecols=column_indices)
        for i in range(vals.shape[0]):
            point[column_names[i]] = float(vals[i])

        return point
    except:
        print("Could not read point from file", filename)
        return None


def load_existing_raw_data(
        folder,
        filename_pattern,
        coordinates,
        column_index,
        file_ending=".cool",
        column_names=None):
    """
    Loads Ctot from all files ending in .cool in the given folder and subfolders.

    :param folder:              Folder to load from
    :param filename_pattern:    How to parse filenames -> coordinates
    :param coordinates:         Coordinate list
    :param column_index:        int or list, columns to load
    :param column_names:        list, name of columns. Has to be given if column_index is list. Otherwise ignored.
    :param file_ending:         File ending of files to load.
    :return:                    Dataframe of points
    """
    # Find all files in the specified folder (and subfolders)
    filenames = []
    for dirpath, dirnames, fnames in os.walk(folder):
        filenames += [os.path.join(dirpath, fname) for fname in fnames]

    points = [] # list of dicts for DataFrame constructor

    if type(column_index) == int:
        for filename in filenames:
            if filename.endswith(file_ending):
                point = _load_point(filename, filename_pattern, coordinates, column_index)
                if point:
                    points.append(point)
    elif type(column_index) == list:
        assert(type(column_names == list))
        assert(len(column_names) == len(column_index))
        columns = dict(zip(column_names, column_index))

        for filename in filenames:
            if filename.endswith(file_ending):
                point = _load_point_multiple_values(filename, filename_pattern, coordinates, columns)
                if point:
                    points.append(point)
    else:
        raise TypeError(f"Invalid type for column_index: {type(column_index)}")

    points = pd.DataFrame(points).astype(float)
    return points


def _set_up_grid(num_per_dim, parameter_space, margins, perturbation_scale=None):
    """
    Fills parameter space with a regular grid as starting point. Takes margins into account.

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


def _set_up_amorphous_grid(num_per_dim, parameter_space, margins, poisson_disc_sampling_scale):
    """
    Uses Poisson disc sampling to set up an amorphous (instead of regular) grid
    https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf

    First sets up a regular grid using _set_up_grid(), then cuts out the middle to leave a regular hull, then fills
    it with poisson disc sampling. Not pretty but as long as it works

    :param num_per_dim:             See _set_up_grid()
    :param parameter_space:         See _set_up_grid()
    :param margins:                 See _set_up_grid()
    :param poisson_disc_sampling_scale:      NOT used for perturbation, but actually passed as radius to the poisson disc
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
    core = poisson_disc_sampling(
        np.array([[0, 1]] * len(list(parameter_space.keys()))),
        poisson_disc_sampling_scale
    )

    # Stretch cube to fit parameter space
    # Note: Changing this now to stretch over margins as well, required for stability. technically core != core
    core_columns = list(points.columns)
    core_columns.remove("values")
    core = pd.DataFrame(core, columns=core_columns)
    for param, limits in _get_param_space_with_margins(parameter_space, margins).items():
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


def _get_rad_bg_as_function(rad_params, output_folder):
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
    :param output_folder:
    :return:
    """
    if not rad_params:
        return lambda x: None

    interpolators = {}
    rad_data = {}
    rad_bases = {}

    # Load radiation data and build interpolators
    for k, v in rad_params.items():
        rad_data[k] = np.loadtxt(v[0])
        interpolators[k] = interpolate.InterpolatedUnivariateSpline(rad_data[k][:,0], rad_data[k][:,1], k=1, ext=1)

        if len(v) < 3:
            raise RuntimeError(f"Missing arguments in rad param: {v}")
        if v[2] == "log":
            rad_bases[k] = 10
            print(f"Interpreting {k} as LOG")
        elif v[2] == "lin":
            rad_bases[k] = 1
            print(f"Interpreting {k} as LIN")
        else:
            raise RuntimeError(f"Invalid rad param argument {v[2]} in {v}")


    valid_keys = list(rad_params.keys())

    combined_energies = np.sort(np.concatenate([rad_data[k][:,0] for k in rad_params.keys()]).flatten())
    # Cloudy only considers energies between 3.04e-9 Ryd and 7.453e6 Ryd, see Hazy p. 33
    combined_energies = np.clip(combined_energies, 3.04e-9, 7.354e6)
    x = np.geomspace(min(combined_energies), max(combined_energies), 3900)

    def rad_bg_function(rad_multipliers):
        # Note: Cloudy can only handle 4000 lines of input at a time...
        f_nu = sum([interpolators[k](x) * pow(rad_bases[k], rad_multipliers[k]) for k in rad_multipliers.keys() if k in valid_keys])

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

    # Plot radiation background to ensure it looks like its supposed to
    # Note - does not take into account cloudy radiation bgs, e.g. HM12
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
              "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
            # if you run out of these colors you probably have other, more important issues
    c = 0
    plt.figure(figsize=(10, 10))
    for k, v in rad_params.items():
        plt.plot(rad_data[k][:,0], rad_data[k][:,1] * pow(rad_bases[k], v[1][0]), ":", color=colors[c])
        plt.plot(rad_data[k][:,0], rad_data[k][:,1] * pow(rad_bases[k], (v[1][1] + v[1][0]) / 2), color=colors[c], label=k)
        plt.plot(rad_data[k][:,0], rad_data[k][:,1] * pow(rad_bases[k], v[1][1]), "--", color=colors[c])
        c += 1

    plt.title("Radiation background components (min, avg, max)")
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$E$ in Ryd")
    plt.ylabel(r"$4\pi J_\nu / h$")  # TODO Verify units
    plt.savefig(os.path.join(output_folder, "rad_components.png"))
    plt.close()

    return rad_bg_function


def _f_nu_to_string(f_nu):
    """
    Helper function to turn a spectrum into a string that can be used in a cloudy input file.

    :param f_nu:    Numpy array, Shape (N, 2).
    :return:
    """
    if f_nu is None:
        return ""

    # find the entry closest to 1 Ryd for cloudy normalization
    mindex = np.argmin(np.abs(f_nu[:,1] - 1))   # min index
    out = f"f(nu) = {f_nu[mindex,0]} at {f_nu[mindex,1]}\ninterpolate ({f_nu[0,1]}  {f_nu[0,0]})\n"

    for i in range(1, f_nu.shape[0]):
        # energy (ryd) - fnu
        out += f"continue ({f_nu[i,1]}  {f_nu[i,0]})\n"

    out += "iterate to convergence"
    return out


def _cloudy_evaluate(input_file,
                     path_to_source,
                     output_folder,
                     filename_pattern,
                     points,
                     rad_bg_function,
                     n_jobs,
                     column_index):
    """
    Evaluate the given points with cloudy using the given file template.

    :param input_file:          String representing the cloudy in put file template
    :param path_to_source:      As in sample()
    :param output_folder:       Folder to move files to after evaluation
    :param filename_pattern:    As in sample()
    :param points:              Dataframe, as in sample()
    :param rad_bg_function:     As returned from _get_rad_bg_as_function()
    :param column_index:        Index of the column to read values from
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

    # Step 4: Read cooling data
    # for now only Ctot
    missing_values = False
    for i, index in enumerate(points.loc[points["values"].isnull()].index):
        try:
            points.loc[index,"values"] = np.log10(np.loadtxt(filenames[i] + ".cool", usecols=column_index))
        except:
            print("Could not read file:", filenames[i] + ".cool")
            points.loc[index, "values"] = None
            missing_values = True

        os.system(f"mv {filenames[i]}* {output_folder}")

    if missing_values:
        before = len(points.index)
        points = points.dropna(axis=0, subset=["values"])
        after = len(points.index)
        print(f"Dropped {after - before} rows due to issues with reading cloudy files.")
        points = points.reset_index(drop=True)


    os.remove(path_to_filenames)

    return points


def build_and_save_delaunay(points, coords, filename, sep=","):
    """
    Helper function to build and save a triangulation on a given set of points. Mostly for debugging

    :param points:      pandas Dataframe
    :param coords:      list
    :param filename:    string
    :param sep:         csv separator
    :return:            Triangulation object
    """
    tri = spatial.Delaunay(points[coords].to_numpy())
    np.savetxt(filename + ".tris", tri.simplices.astype(int), delimiter=sep, fmt="%i")
    np.savetxt(filename + ".neighbors", tri.neighbors.astype(int), delimiter=sep, fmt="%i")

    return tri