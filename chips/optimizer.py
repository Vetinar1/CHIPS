import os
from chips.utils import *
from chips.psi import *
import pandas as pd
import itertools
from scipy import interpolate, spatial
import matplotlib.pyplot as plt
import seaborn as sns
import time
from parse import parse
from matplotlib import patches
from scipy.spatial import KDTree
from scipy.interpolate import LinearNDInterpolator

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

        mode="Delaunay",

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
        use_net_cooling=False,
        suppress_interp_column_warning=False,
        save_fractions=False,
        save_triangulation=True,

        n_jobs=4,

        n_partitions=10,
        psi_nearest_neighbors=50,
        psi_nn_factor=2,
        psi_max_tries=4,

        use_mdistance_weights=True,
        plot_mdistance_hists=True,

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
    :param mode:                            "Delaunay" or "PSI". Delaunay is recommended for low dimensionalities
                                            and sample counts. PSI otherwise.
                                            The following settings have no effect if mode is "PSI":
                                            n_partitions
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
    :param use_net_cooling:                 Instead of cooling or heating, use (Heating - Cooling).
                                            Overrides interp_column.
                                            If enabled, asinh(Lambda_net) will be used instead of log10(Lambda_net),
                                            because in contrast to Lambda and Gamma, Lambda_net can be negative.
    :param suppress_interp_column_warning:  Do not throw warning when attempting to read columns other than 2 or 3.
    :param save_fractions:                  Save electron and hydrogen fractions to points file. For use with gadget.
    :param save_triangulation:              Whether to save the triangulation at the end (.tris and .neighbors files).
                                            These can be very large at high dimensions.
    :param n_jobs:                          How many cloudy jobs to run in parallel
    :param n_partitions:                    How many partitions to use during Triangulation+Interpolation. Has no
                                            effect if mode is "PSI".
    :param psi_nearest_neighbors:           Number of nearest neighbors to use in PSI algorithm. Has no effect if
                                            mode is "Delaunay".
    :param psi_nn_factor:                   Multiplicator for psi_nearest_neighbors if simplex construction algorithm
                                            fails. Has no effect if mode is "Delaunay".
    :param psi_max_tries:                   Maximum number of retries if simplex construction algorithm fails. Has no
                                            effect if mode is "Delaunay".
    :param use_mdistance_weights:           If this option is enabled the interpolation error is additionally weighted
                                            using the mean distance to neighboring points. In principle this is
                                            antithetical to the core design of the algorithm, and provides an automatic
                                            brake or "rubber band" effect. It is primarily useful when dealing with
                                            (near) discontinuities in the data (i.e. the ionization of hydrogen at 1e4K)
                                            The current implementation normalizes the weights such that the halfway
                                            point between minimum and maximum mean distance gives a weighting factor of
                                            1. The assumption is that if a discontinuity occurs, the majority of points
                                            are going to be in a dense cloud with low mean distance, and thus that the
                                            distribution has the majority of points as well as its peak to the left
                                            of the halfway point.
                                            This option currently has no effect when using Delaunay simplices.
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

    if not use_net_cooling:
        # Check if the column we are reading from the .cool file is valid
        if not suppress_interp_column_warning:
            if interp_column != 2 and interp_column != 3:
                raise RuntimeWarning(f"interp_column is not 2 (heating) or 3 (cooling), but {interp_column}. "
                                     "If you really wish to proceed, set suppress_interp_column_warning=True.")
        else:
            print(f"Interpolation column warning suppressed. (interp_column = {interp_column})")


    if mode not in ["Delaunay", "PSI"]:
        print("Invalid mode: ", mode)
        print("Valid modes: Delaunay, PSI")


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
    if use_net_cooling:
        print("Using net cooling instead of single column")
    else:
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
                indata = pd.read_csv(dpath, delimiter=sep)

                if not "values" in indata.columns:
                    print("\tNo 'values' column in provided file")
                    print("\tAssuming values column is named 'Ctot'. Columns other than coords and Ctot will be dropped")
                    indata["values"] = indata["Ctot"]
                    indata = indata[coord_list + ["values"]]
                else:
                    print("\tDropping columns other than coords and 'values'")
                    indata = indata[coord_list + ["values"]]

                points = pd.concat((
                    points,
                    indata
                ))

            elif os.path.isdir(dpath):
                print("Attempting to recursively read raw data from folder", dpath)
                points = pd.concat((
                    points,
                    load_existing_raw_data(existing_data, filename_pattern, coord_list, interp_column)
                ))
            else:
                raise RuntimeError(f"Error: {dpath} is not a valid file or folder")

            points = points.drop_duplicates(subset=coord_list, keep="last", ignore_index=True)
            points = points.reset_index(drop=True)

        points = points.dropna()
        existing_point_count = len(points.index)
        print("Loaded", existing_point_count, " points")

    if initial_grid:
        print("Setting up grid")
        grid_points = _set_up_amorphous_grid(initial_grid, core, margins, poisson_disc_scale, coord_list)
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
        coord_list,
        rad_bg,
        n_jobs,
        interp_column,
        use_net_cooling
    )

    if points[values].isnull().to_numpy().any():
        points.loc[points["values"].isna()]  = _cloudy_evaluate(
            cloudy_input,
            cloudy_source_path,
            it0folder,
            filename_pattern,
            points.loc[points["values"].isna()],
            coord_list,
            rad_bg,
            n_jobs,
            interp_column,
            use_net_cooling
        )
        
    points = pd.concat((points, corners), ignore_index=True)
    points = points.drop_duplicates(subset=coord_list, keep="last", ignore_index=True)

    time2 = time.time()
    print(round(time2 - time1, 2), "s to do initial setup")
    print("\n\n")


    if not use_net_cooling:
        try:
            assert (not points[values].isnull().to_numpy().any())
        except AssertionError:
            print(f"Warning: {len(points[points[values].isnull()].index)} of the initial values are NaN. Dropping.")
            points = points.dropna()

    ####################################################################################################################
    #################################################    Main Loop    ##################################################
    ####################################################################################################################
    iteration = 0
    iteration_time = 0
    timeBeginLoop = time.time()

    # number of dimensions
    # TODO: Generalize for multiple values
    while True:
        # reset index just to be on the safe side while partitioning later
        points = points.reset_index(drop=True)
        iteration += 1
        print("{:*^50}".format(f"Iteration {iteration}"))

        drop_duplicates_and_print(points, coord_list)
        points = points.reset_index(drop=True)

        timeA = time.time()
        if mode == "Delaunay":
            points, new_points = sample_step_delaunay(points, n_partitions, coord_list, core, corners, accuracy_threshold)
        elif mode == "PSI":
            points, new_points = sample_step_psi(points, coord_list, core, accuracy_threshold, psi_nearest_neighbors,
                                                 psi_nn_factor, psi_max_tries, n_jobs, use_mdistance_weights,
                                                 plot_mdistance_hists)
        else:
            print("Invalid mode. Something went wrong.")
            exit()

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
        if use_mdistance_weights:
            unadjusted_over_thresh_count = len(points[points['diff_orig'] > accuracy_threshold].index)
            unadjusted_over_thresh_fraction = unadjusted_over_thresh_count / in_bounds_count
            print(f"\tIf use_mdistance_weights were False, {unadjusted_over_thresh_count}"
                  f" / {in_bounds_count} were over threshold ("
                  f"{round(100 * unadjusted_over_thresh_fraction, 2)}%"
                  f")")
        print("Number of new samples".ljust(50), len(new_points.index))
        print("Largest interpolation error".ljust(50), max_diff)

        if plot_iterations:
            _plot_parameter_space(points, new_points, coord_list, output_folder, iteration)

        print("Saving intermediate .points file...")
        load_rawdata_and_save_fractions(
            folder=os.path.join(output_folder, f"iteration{iteration - 1}"),
            filename=os.path.join(output_folder, f"iteration{iteration - 1}", f"it{iteration - 1}.points"),
            coord_list=coord_list
        )

        if plot_mdistance_hists:
            os.system(f"mv temphist.png {os.path.join(output_folder, f'mean_dist_hist{iteration - 1}')}.png")

        if cleanup:
            print("Performing cleanup, mode:", cleanup)
            folder = os.path.join(output_folder, f"iteration{iteration - 1}")
            if cleanup == "outfiles":
                for fname in os.listdir(folder):
                    if fname.endswith(".out"):
                        os.remove(os.path.join(folder, fname))
            if cleanup == "full":
                for fname in os.listdir(folder):
                    if not fname.endswith(".points"):
                        os.remove(os.path.join(folder, fname))

            print("Cleanup complete")


        # Check the various conditions for quitting
        threshold_condition = False
        if error_fraction is None or over_thresh_fraction < error_fraction:
            threshold_condition = True
        elif use_mdistance_weights and unadjusted_over_thresh_fraction < error_fraction:
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
        if use_mdistance_weights:
            points = points.drop("diff_orig", axis=1)
        new_points["values"] = np.nan
        if significant_digits is not None:
            new_points = new_points.round(significant_digits)

        time3 = time.time()
        new_points = _cloudy_evaluate(
            cloudy_input,
            cloudy_source_path,
            iteration_folder,
            filename_pattern,
            new_points.reset_index(drop=True),
            coord_list,
            rad_bg,
            n_jobs,
            interp_column,
            use_net_cooling
        )
        points = pd.concat((points, new_points)).drop_duplicates(subset=coord_list).reset_index(drop=True)

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
    if use_mdistance_weights:
        points = points.drop("diff_orig", axis=1)
    # *Somewhere* an index column is added and I don't know where. Get rid off it before saving.
    # update: i think i fixed it but lets keep it just in case
    if "index" in points.columns:
        points = points.drop("index", axis=1)

    if save_fractions:
        points = points.drop("values", axis=1)
        print("save_fractions == true, Loading and merging hydrogen and electron fractions")

        dfs = []
        for i in range(iteration):
            dfs.append(
                pd.read_csv(os.path.join(output_folder, f"iteration{i}", f"it{i}.points"))
            )

        dfs = pd.concat(dfs, ignore_index=True)
        points = points.merge(dfs, how="inner", on=coord_list)

        if points[coord_list].isnull().values.any():
            print(f"Warning: {len(points[points[coord_list].isnull().any(axis=1)].index)} / {len(points.index)} points contain "
                  "NaN in their coordinates")

        value_list = ["Htot", "Ctot"] + ["ne", "H2", "HI", "HII", "HeI", "HeII", "HeIII"]
        if points[value_list].isnull().values.any():
            print(f"Warning: {points[points[value_list].isnull().any(axis=1)]} / {len(points.index)} points contain "
                  "NaN in their values")

    print("Building and saving final Delaunay triangulation...")
    points.to_csv(os.path.join(output_folder, output_filename + ".fullpoints"), index=False)

    # drop margins
    for coord in core.items():
        points = points[points[coord[0]] >= coord[1][0]]
        points = points[points[coord[0]] <= coord[1][1]]

    tri = spatial.Delaunay(points[coord_list].to_numpy())
    if save_triangulation:
        np.savetxt(os.path.join(output_folder, output_filename + ".tris"), tri.simplices.astype(int), delimiter=sep, fmt="%i")
        np.savetxt(os.path.join(output_folder, output_filename + ".neighbors"), tri.neighbors.astype(int), delimiter=sep, fmt="%i")

    points.to_csv(os.path.join(output_folder, output_filename + ".points"), index=False)

    print("Done")
    return points


def sample_step_delaunay(points, n_partitions, coord_list, core, corners, accuracy_threshold):
    """
    Do one Delaunay Triangulation based sampling step.
    1. Calculate Delaunay Triangulation
    2. Use it to interpolate points
    3. For all points that are over threshold, draw new sample at the center of the matching simplex

    Moved into separate function as part of the PSI integration.

    :param points:              Pandas dataframe. Modified in-place.
    :param n_partitions:        Number of partitions to use, default 10
    :param coord_list:          List of coordinates (names)
    :param core:                Dict describing the extents of the core of the parameter space, see sample()
    :param corners:             Pandas dataframe containing corners
    :param accuracy_threshold:  float
    :return points, new_points: Original, modified points dataframe.
                                Dataframe containing all the new sample locations. (Not yet evaluated.)
    """
    new_points = pd.DataFrame(columns=coord_list)
    points["interpolated"] = np.nan
    points["diff"] = np.nan
    D = len(coord_list)

    for partition in range(n_partitions):
        assert(not points.index.duplicated().any())
        # TODO This part uses a lot of deep copies, might be inefficient
        subset = points.loc[partition::n_partitions].copy(deep=True)
        bigset = pd.concat((points, subset)).drop_duplicates(keep=False, subset=coord_list)

        subset.loc[:,"interpolated"] = np.nan

        # Always include corners to ensure convex hull
        bigset = pd.concat((bigset, corners))
        bigset = bigset.drop_duplicates(subset=coord_list)

        # Need to reset the bigset index due to the concatenation
        # Since the bigset index is not relevant again - unlike the subset index - this should be fine
        bigset = bigset.reset_index(drop=True)

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

    points = points.reset_index(drop=True)
    new_points = new_points.reset_index(drop=True)
    return points, new_points


def sample_step_psi(points, coord_list, core, accuracy_threshold, k, factor, max_steps, n_jobs, use_mdistance_weights,
                    plot_mdistance_hists=False):
    """
    Do one sampling step using the PSI algorithm.
    https://arxiv.org/abs/2109.13926

    For each point:
    1. Get k nearest neighbors
    2. Build simplex using PSI
    3. Use simplex to interpolate
    4. If difference to interpolated value is over threshold, add center of simplex as new sample

    Drops duplicates and resets index of points.

    :param points:                  Pandas dataframe. Modified in-place.
    :param coord_list:              List of coordinates (names)
    :param core:                    Dict describing the extents of the core of the parameter space, see sample()
    :param accuracy_threshold:      float
    :param k:                       Number of nearest neighbors to use
    :param factor:                  If PSI fails, multiply k with this and try again
    :param max_steps:               Retry this many times
    :param n_jobs:                  Number of jobs to run at once
    :param use_mdistance_weights:   Use mean distance to nearest neighbors to weight errors
    :return:                        points: Original, modified dataframe
                                    new_points: Dataframe containing all the new sample locations
    """

    points = points.drop_duplicates(ignore_index=True, subset=coord_list)
    points = points.reset_index(drop=True)  # looks redundant but is necessary to turn points into a true dataframe
                                            # instead of a slice...

    points["interpolated"] = np.nan
    points["diff"] = np.nan
    D = len(coord_list)

    coreset = points.copy(deep=True)

    # dont consider points in margins
    for c, edges in core.items():
        coreset = coreset.loc[(coreset[c] > edges[0]) & (coreset[c] < edges[1])]

    psa_err_counter = 0
    psa_err_counter2 = 0
    qhull_errors = 0
    coreset_numpy = coreset[coord_list].to_numpy()
    points_numpy  = points[coord_list].to_numpy()
    tree = KDTree(points_numpy)
    new_points = np.zeros_like(coreset_numpy)
    new_points[:] = np.nan

    c = 0   # counter variable; i is dataframe index, can not use it to index numpy array
    for i, row in coreset.iterrows():

        point = row[coord_list].to_numpy()
        simplex = build_simplex_adaptive(points_numpy, point, tree, k, factor, max_steps, n_jobs)

        if simplex is None:
            psa_err_counter += 1
            simplex = build_simplex_adaptive(points_numpy, point, tree, int(k/2), factor, max_steps, n_jobs, smart_nn=True)

            if simplex is None:
                psa_err_counter2 += 1
                continue

        simplex = points.loc[simplex]

        try:
            interpolator = LinearNDInterpolator(
                simplex[coord_list].to_numpy(),
                simplex["values"].to_numpy()
            )
        except:
            qhull_errors += 1
            continue

        coreset.loc[i, "interpolated"] = interpolator(point)
        new_points[c] = np.sum(simplex[coord_list]) / (D+1)
        c += 1

    coreset["diff"] = np.abs(coreset["values"] - coreset["interpolated"])
    if use_mdistance_weights:
        QUANTILE = 0.99
        NEIGHBOR_COUNT = 10

        _, neighbors = tree.query(coreset_numpy, k=NEIGHBOR_COUNT+1)
        neighbors = neighbors[:,1:]     # first nn is self

        for i in range(coreset_numpy.shape[0]):
            index = coreset.index.values[i]
            neigh_coords = points_numpy[neighbors[i]]
            coreset.loc[index, "mean_dist"] = np.mean(
                np.sqrt(np.sum(np.square(neigh_coords - coreset_numpy[i]), axis=1)) # pythagoras
            )

        # normalize values to around 1
        normfactor = (coreset["mean_dist"].quantile(QUANTILE) + coreset["mean_dist"].min()) / 2
        coreset["diff_orig"] = coreset["diff"]
        coreset["diff"]      *= coreset["mean_dist"]

        if plot_mdistance_hists:
            plt.hist(coreset["mean_dist"] / normfactor, histtype="step", label="Normed", color=(0, 0, 1, 0.5))
            plt.hist(coreset["mean_dist"], histtype="step", label="Not normed", color=(1, 0, 0, 0.5))
            plt.title(f"Mean distance to nearest {NEIGHBOR_COUNT} neighbors")
            plt.gca().axvline(normfactor)
            plt.gca().axvline(coreset["mean_dist"].min())
            plt.gca().axvline(coreset["mean_dist"].quantile(QUANTILE))
            plt.legend()
            plt.savefig("temphist.png")
            plt.close()

        coreset["mean_dist"] = coreset["mean_dist"] / normfactor

    # decide which new points to keep
    new_points = new_points[(coreset["diff"] > accuracy_threshold).to_numpy()]
    new_points = pd.DataFrame(new_points, columns=coord_list).dropna()

    # write interpolation back from the core data set into the full data set
    points.loc[coreset.index, ["interpolated", "diff"]] = coreset.loc[:, ["interpolated", "diff"]]
    if use_mdistance_weights:
        points.loc[coreset.index, "diff_orig"] = coreset.loc[:, "diff_orig"]

    points = points.reset_index(drop=True)

    if psa_err_counter:
        print(f"Projective simplex construction failed for {psa_err_counter} out of {coreset_numpy.shape[0]} points")
    if psa_err_counter2:
        print(f"Fallback option failed for {psa_err_counter2} out of remaining {psa_err_counter} points")
    if qhull_errors:
        print(f"Encountered {qhull_errors} QHull errors")

    return points, new_points


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
    fullpoints = pd.concat((points, new_points)).reset_index(drop=True)

    grid = sns.pairplot(
        fullpoints,
        diag_kind="hist",
        vars=coord_list,
        hue="hue",
        markers="o",
        plot_kws={
            "s":4,
            "marker":"o",
            "edgecolor":None
        },
        diag_kws={
            "bins":50
        },
        height=6
    )
    grid.savefig(os.path.join(output_folder, "iteration{}.png".format(suffix)))
    plt.close()

    points.drop("hue", axis=1, inplace=True)
    new_points.drop("hue", axis=1, inplace=True)

    end = time.time()
    print(f"{round(end - begin, 2)}s to plot current iteration")


def _load_point(filename, filename_pattern, coordinates, column_index, use_net_cooling=False):
    """
    Load Ctot or Htot from the given cloudy cooling output ("*.cool") file. Applies log10 to the read values.
    Filename needs to contain the full path to the file.

    The filename pattern is used to parse the coordinates at which Ctot is given.

    :param filename:            file to load
    :param filename_pattern:    how to parse filename -> coordinates
    :param coordinates:         Coordinate list
    :param column_index:        int or list
    :param use_net_cooling:     bool
    :return:                    Dict
    """
    result = parse(filename_pattern, os.path.splitext(os.path.basename(filename))[0])
    point = result.named

    for coordinate in coordinates:
        if coordinate not in list(point.keys()):
            raise RuntimeError(f"Missing coordinate {coordinate} while trying to read in file {filename}")

    try:
        if use_net_cooling:
            vals = np.loadtxt(filename, usecols=(2, 3))
            point["values"] = np.arcsin(vals[0] - vals[1])
        else:
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
        column_names=None,
        use_net_cooling=False):
    """
    Loads Ctot from all files ending in .cool in the given folder and subfolders.

    :param folder:              Folder to load from
    :param filename_pattern:    How to parse filenames -> coordinates
    :param coordinates:         Coordinate list
    :param column_index:        int or list, columns to load
    :param column_names:        list, name of columns. Has to be given if column_index is list. Otherwise ignored.
    :param file_ending:         File ending of files to load.
    :param use_net_cooling:     Both column_names and column_index will be ignored. The net cooling rate will
                                be loaded instead. To keep it consistent with the main loop, asinh() is applied
                                to the difference. If you want to get the "regular" net cooling, apply sinh() to the
                                result.
    :return:                    Dataframe of points
    """
    # Find all files in the specified folder (and subfolders)
    filenames = []
    if type(folder) is str:
        for dirpath, dirnames, fnames in os.walk(folder):
            filenames += [os.path.join(dirpath, fname) for fname in fnames]
    elif type(folder) is list:
        for f in folder:
            for dirpath, dirnames, fnames in os.walk(f):
                filenames += [os.path.join(dirpath, fname) for fname in fnames]
    else:
        raise TypeError("Invalid type for parameter folder:", type(folder))

    points = [] # list of dicts for DataFrame constructor

    if type(column_index) == int or use_net_cooling:
        for filename in filenames:
            if filename.endswith(file_ending):
                point = _load_point(filename, filename_pattern, coordinates, column_index, use_net_cooling)
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


def load_rawdata_and_save_fractions(folder, filename, coord_list, filename_pattern=None, nolog10=False):
    """
    Loads raw data from given folder, saves it to csv file. Loads and saves both cooling/heating rates and fractions.

    :param folder:              Folder to load from (includes subfolders)
    :param filename:            File to save to
    :param coord_list:          Coordinates to look for
    :param filename_pattern:    Dont touch this
    :param nolog10:             If True, do not log10 the Ctot and Htot columns
    """

    if not filename_pattern:
        filename_pattern = "__".join(
            ["_".join(
                [c, "{" + str(c) + "}"]
            ) for c in coord_list]
        )
    else:
        for c in coord_list:
            if "{" + str(c) + "}" not in filename_pattern:
                raise RuntimeError(f"Invalid file pattern: Missing {c}")

    cooldata = load_existing_raw_data(
        folder,
        filename_pattern,
        coord_list,
        [2, 3],
        file_ending=".cool",
        column_names=["Htot", "Ctot"]
    ).drop_duplicates()

    if not nolog10:
        cooldata.loc[:,["Htot", "Ctot"]] = np.log10(cooldata.loc[:,["Htot", "Ctot"]])

    fracdata = load_existing_raw_data(
        folder,
        filename_pattern,
        coord_list,
        [4, 5, 6, 7, 8, 9, 10],
        file_ending=".overview",
        column_names=["ne", "H2", "HI", "HII", "HeI", "HeII", "HeIII"]
    ).drop_duplicates()

    merged = cooldata.merge(
        fracdata,
        "inner",
        on=coord_list
    )

    merged.to_csv(os.path.expanduser(filename), index=False)

    if len(cooldata.index) != len(merged.index):
        print("The length of the points table changed during the merge with the electron/hydrogen fractions.\n"
              "This is odd. You may want to double check the results.")


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
    points = pd.DataFrame(index=index).reset_index(drop=True)

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


def _set_up_amorphous_grid(num_per_dim, parameter_space, margins, poisson_disc_sampling_scale, coord_list):
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
    :param coord_list:              List of cooordinate names as in sample()
    :return:
    """
    # Take regular grid, no perturbation
    points_full = _set_up_grid(num_per_dim, parameter_space, margins)

    # Cut out all the "middle values"
    points = pd.DataFrame(columns=points_full.columns, dtype=np.float64)
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
    core = pd.DataFrame(core, columns=coord_list, dtype=np.float64)
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
                     coords,
                     rad_bg_function,
                     n_jobs,
                     column_index,
                     use_net_cooling=False):
    """
    Evaluate the given points with cloudy using the given file template.

    :param input_file:          String representing the cloudy in put file template
    :param path_to_source:      As in sample()
    :param output_folder:       Folder to move files to after evaluation
    :param filename_pattern:    As in sample()
    :param points:              Dataframe; all points inside will be evaluated
    :param coords:              List of coordinates
    :param rad_bg_function:     As returned from _get_rad_bg_as_function()
    :param column_index:        Index of the column to read values from
    :param use_net_cooling:     Instead of using column_index, read columns 2 and 3 and take the difference.
                                Uses asinh instead of log10.
    :return:
    """
    filenames = []

    try:
        assert(not points.empty)
    except:
        raise RuntimeWarning("Points dataframe passed to _cloudy_evaluate is empty")

    try:
        assert(points["values"].isnull().all())
    except:
        raise RuntimeError("Not all values in points dataframe passed to _cloudy_evaluate are nan. "
                           "Something has gone wrong, aborting.")

    if points[coords].isnull().values.any():
        len_before = len(points.index)
        points = points.dropna(subset=coords)
        len_after = len(points.index)
        print(f"Dropped {len_before - len_after} nan values from points dataframe in _cloudy_evaluate")
        # TODO I should really find the source of that bug and fix it

    for i in points.index:
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
    if use_net_cooling:
        for i, index in enumerate(points.index):
            try:
                vals = np.loadtxt(filenames[i] + ".cool", usecols=(2, 3))
                points.loc[index, "values"] = vals[0] - vals[1]
            except:
                print("Could not read file:", filenames[i] + ".cool")
                points.loc[index, "values"] = None
                missing_values = True

            os.system(f"mv {filenames[i]}* {output_folder}")
    else:
        for i, index in enumerate(points.loc[points["values"].isnull()].index):
            try:
                points.loc[index,"values"] = np.loadtxt(filenames[i] + ".cool", usecols=column_index)
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

    if use_net_cooling:
        zero_count = len(points.loc[points["values"] == 0,"values"].index)
        if zero_count > 0:
            print(f"{zero_count} Lambda_net values were zero before arcsinh(1/x) transformation, and are kept at 0.\n"
                  "Does this sound reasonable?")

        points.loc[:,"values"] = np.arcsinh(
            np.divide(
                1,
                points.loc[:,"values"],
                out=np.zeros_like(points.loc[:,"values"].to_numpy()),
                where=points.loc[:,"values"]!=0
            )
        )
    else:
        points.loc[:,"values"] = np.log10(points.loc[:,"values"])


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