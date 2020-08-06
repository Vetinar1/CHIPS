import os
from chips.utils import *
import pandas as pd
import itertools
from scipy import interpolate, spatial
import matplotlib.pyplot as plt
import seaborn as sns
import time
from parse import parse

sns.set()

# TODO: Options for cloudy to clean up after itself
# TODO: Check existence of radiation files
# TODO: Smart continuation: Should be able to read in existing output folders and keep working on them
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

        plot_iterations=True
):
    """

    :param cloudy_input:                    String. Cloudy input or cloudy input file.
    :param cloudy_source_path:              String. Path to source/
    :param output_folder:
    :param param_space:                    "key":(min, max)
    :param param_space_margins:             2 values: absolute. 1 value: relative
    :param rad_params:                      "key":(min, max). key doubles as filepath
    :param rad_params_margins:              2 values: absolute. 1 value: relative
    :param rad_params_names:                TODO optional prettier names
    :param existing_data:
    :param initial_grid:                    How many samples in each dimension in initial grid. 0 to disable
    :param dex_threshold:
    :param over_thresh_max_fraction:
    :param dex_max_allowed_diff:
    :param random_samples_per_iteration:
    :param n_jobs:
    :param n_partitions:
    :param max_iterations:
    :param max_storage_gb:
    :param max_time:
    :param plot_iterations:
    :return:
    """
    ####################################################################################################################
    ###############################################    Safety checks    ################################################
    ####################################################################################################################
    # Set up Output folder
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    elif len(os.listdir(output_folder)) != 0:
        choice = input("Chosen output folder is not empty. Proceed? (y/n)")
        if choice not in ["y", "Y", "yes", "Yes"]:
            print("Aborting...")
            exit()

    # Make sure cloudy exists
    # TODO: Always returns false
    # if not os.path.exists(os.path.join(cloudy_source_path, "cloudy.exe")):
    #     print(os.path.join(cloudy_source_path, "cloudy.exe"))
    #     raise RuntimeError("No cloudy.exe found at: " + str(cloudy_source_path))

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
    for rad, edges in rad_params.items():
        print(f"\t{rad}:\t{edges}\t\tMargins: {rad_params_margins[dim]}")

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
    # Compile coordinates and values into easily accessible lists
    coordinates = list(param_space.keys()) + list(rad_params.keys())
    values = ["values"]  # TODO
    points = pd.DataFrame(columns=coordinates + values)

    # Load existing data if applicable
    if existing_data:
        if not os.path.isdir(output_folder):
            raise RuntimeError("Specified existing data at " + str(existing_data) + " but is not a dir or does not exist")

        points = pd.concat(
            points,
            _load_existing_data(existing_data, filename_pattern, coordinates)
        )

    if initial_grid:
        grid_points = _set_up_grid(initial_grid, param_space, param_space_margins, rad_params, rad_params_margins)


    if not filename_pattern:
        filename_pattern = "__".join(
            ["_".join(
                [c, "{" + str(c) + "}"]
            ) for c in coordinates]
        )
    else:
        for c in coordinates:
            if "{" + str(c) + "}" not in filename_pattern:
                raise RuntimeError("Invalid file pattern")

    # TODO Measure time
    rad_bg = _get_rad_bg_as_function(rad_params)
    points = _cloudy_evaluate(cloudy_input, cloudy_source_path, output_folder, filename_pattern, points, rad_bg, n_jobs)

    # number of dimensions
    # TODO: Generalize for multiple values

    ####################################################################################################################
    #################################################    Main Loop    ##################################################
    ####################################################################################################################
    iteration = 0
    N = len(points.columns) - 1
    while True:
        iteration += 1
        print("{:*^50}".format("Iteration {}".format(iteration)))

        len_pre_drop =  len(points.index)
        points = points.drop_duplicates()
        len_post_drop = len(points.index)

        if len_pre_drop > len_post_drop:
            print("Dropped {} duplicate samples".format(len_pre_drop - len_post_drop))

        # For plotting the triangulation
        # if plot_iterations:
        #     fig, ax = plt.subplots(1, n_partitions, figsize=(10 * n_partitions, 10))

        new_points = pd.DataFrame(columns=coordinates)

        for partition in range(n_partitions):
            subset = points.iloc[partition::n_partitions]
            bigset = pd.concat((points, subset)).drop_duplicates()

            # TODO again, generalize
            subset["interpolated"] = np.nan

            # dont consider points in margins
            # TODO: Verify
            # TODO: Make it so that corners are always included in each partition
            #   => convex hull, dont need to be concerned with points not inside area
            for param, edges in param_space.items():
                subset = subset.loc[(subset[param] > edges[0]) & (subset[param] < edges[1])]

            # # Not sure this is necessary, just to be safe
            # subset.reset_index()
            # bigset.reset_index()

            tri = spatial.Delaunay(bigset[coordinates].to_numpy())
            simplex_indices = tri.find_simplex(subset[coordinates].to_numpy())
            simplices  = tri.simplices[simplex_indices]
            transforms = tri.transform[simplex_indices]

            # The following is adapted from
            # https://stackoverflow.com/questions/30373912/interpolation-with-delaunay-triangulation-n-dim/30401693#30401693
            # 1. barycentric coordinates of points; N-1
            bary = np.einsum(
                "ijk,ik->ij",
                transforms[:,:N,:N],
                subset[coordinates].to_numpy() - transforms[:, N, :]
            )

            # 2. Add dependent barycentric coordinate to obtain weights
            weights = np.c_[bary, 1 - bary.sum(axis=1)]

            # 3. Interpolation
            # TODO vectorize
            # TODO Verify index is continuous
            for i in range(subset.index):
                subset.loc[i, "interpolated"] = np.inner(
                    points.loc[simplices[i], "values"], weights[i]
                )

            assert(not subset["interpolated"].isnull().to_numpy().any())

            # TODO After this point, all np.nans should be due to out of bounds or similar,
            #   NOT due to errors while interpolating

            # 4. Find points over threshold
            # convert to log because the thresholds are given in dex
            subset.loc[:, ["values", "interpolated"]] = np.log10(
                subset.loc[:, ["values", "interpolated"]]
            )
            # Only keep points where the difference is over the threshold
            subset["diff"] = subset.loc[np.abs(subset["interpolated"] - subset["values"]) > dex_threshold]

            # Draw new samples by finding geometric centers of the simplices containing the subset
            if not subset[subset["diff"] > dex_threshold].empty:
                # TODO: Reusing old variables is bad
                simplex_indices = tri.find_simplex(points)
                simplices       = tri.simplices[simplex_indices]

                # Shape: (N_simplices, N_points_per_simplex, N_coordinates_per_point)
                simplex_points = np.zeros(
                    (simplices.shape[0], simplices.shape[1], simplices.shape[1]-1)
                )

                for i in range(simplices.shape[0]):
                    simplex_points[i] = bigset.loc[simplices[i], coordinates].to_numpy()

                samples = np.sum(simplex_points, axis=1) / simplices.shape[1]
                samples = pd.DataFrame(samples, columns=coordinates)
                new_points = pd.concat((new_points, samples))
            else:
                print("No points over threshold in partition {}".format(partition))

            # TODO: Triangulation plotting? Only really works in 2D
            # if plot_iterations:
            #     ax[partition].triplot(a[:, 0], a[:, 1], tri.simplices)
            #     ax[partition].plot(a[:, 0], a[:, 1], "ko")
            #     for j in range(over_thresh.shape[0]):
            #         ax[partition].plot([over_thresh[j, 0], samples[j][0]], [over_thresh[j, 1], samples[j][1]], "red")
            #         ax[partition].plot([over_thresh[j, 0]], [over_thresh[j, 1]], "ro")
            #
            #         if dimensions:
            #             rect = patches.Rectangle((dimensions[0][0], dimensions[1][0]),
            #                                      dimensions[0][1] - dimensions[0][0],
            #                                      dimensions[1][1] - dimensions[1][0], edgecolor="green", fill=False)
            #             ax[partition].add_patch(rect)
            #
            #     ax[partition].set_title("Delaunay Partition " + str(partitions))
            #     ax[partition].set_xlabel("T")
            #     ax[partition].set_ylabel("nH")

            # Write interpolated and diffs back into original points dataframe
            # TODO: IMPORTANT Verify this looks as expected after all partitions are done
            # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.join.html
            points = pd.merge(points, subset, axis=1, join="outer", sort=False)
            # Note: This requires the index to be as in the original, so we CAN NOT use reset_index
            # at the beginning!
            points = points.join(subset)


        # Calculate some stats/diagnostics

        # all points
        total_count = len(points.index)

        # points not in margins
        # TODO: Wasteful
        df_copy = points.copy(deep=True)
        for param, edges in param_space.items():
            df_copy = df_copy.loc[(df_copy[param] > edges[0]) & (df_copy[param] < edges[1])]
        in_bounds_count = len(df_copy.index)
        del df_copy

        # points in margins
        outside_bounds_count = total_count - in_bounds_count

        # number and fraction of points over threshold
        over_thresh_count    = len(points[points["diff"] > dex_threshold].index)
        over_thresh_fraction = over_thresh_count / in_bounds_count

        # biggest difference
        max_diff = points["diff"].max()


        print("Number of nans in interpolated: ", points["interpolated"].isnull().sum())

        # TODO diagnostics
        # log("Interpolated and sampled using Delaunay Triangulation\n")
        # log("\tTime:".ljust(50) + str(round(time.time() - start, 2)) + "s\n")
        # log("\tOut of bounds points (skipped)".ljust(50) + str(oob_count) + "\n")
        # log("\tPoints requiring IDW fallback:".ljust(50) + str(outside) + "\n")
        # log("\tPoints not within threshold:".ljust(50) +
        #     str(new_point_count) + "/" +  # Equal to number of new samples
        #     str(points.shape[0]) + " (" +
        #     str(round(new_point_count / points.shape[0], 2)) + ")\n"
        # )
        # log("\tMaximum (log) difference:".ljust(50) + str(max_diff) + "\n")

        if plot_iterations:
            _plot_parameter_space(points, new_points, coordinates, output_folder, iteration)


        # Check the various conditions for quitting
        threshold_condition = False
        if over_thresh_max_fraction is None or over_thresh_fraction < over_thresh_max_fraction:
            threshold_condition = True

        max_diff_condition = False
        if dex_max_allowed_diff is None or max_diff < dex_max_allowed_diff:
            max_diff_condition = True

        if threshold_condition and max_diff_condition:
            print("Reached desired accuracy. Quitting.")
            break

        if max_iterations and iteration > max_iterations:
            print("Reached maximum number of iterations ({}). Quitting.".format(max_iterations))
            break

        if max_storage_gb:
            gb_in_bytes = 1e9   # bytes per gigabyte
            output_size_gb = get_folder_size(output_folder) / gb_in_bytes

            if output_size_gb > max_storage_gb:
                print("Reached maximum allowed storage ({} GB/{} GB). Quitting.".format(
                    output_size_gb,
                    max_storage_gb
                ))
                break

        # if False: # TODO: Check for time. enable timestamps
        #     print("Reached maximum calculation time ({}). Quitting".format(HUMAN_READABLE_TIME))
        #     break


        if not os.path.exists(os.path.join(output_folder, "iteration{}".format(iteration))):
            os.mkdir(os.path.join(output_folder, "iteration{}".format(iteration)))
        else:
            choice = input("Attempting to write into non-empty folder {}. Proceed? (y/n)".format(
                os.path.join(output_folder, "iteration{}".format(iteration))
            ))

            if choice not in ["y", "Y", "yes", "Yes"]:
                print("Aborting...")
                exit()


        # Add completely random new points
        random_points = [] # list of dicts for DataFrame constructor
        for i in range(random_samples_per_iteration):
            random_points.append(
                {
                    coord : (edge[1] - edge[0]) * np.random.random() + edge[0]
                    for coord, edge in param_space.items()
                }
            )

        new_points = pd.concat((new_points, pd.DataFrame(random_points)))

        new_points = _cloudy_evaluate(
            cloudy_input,
            cloudy_source_path,
            os.path.join(output_folder, "iteration{}".format(iteration)),
            filename_pattern,
            new_points,
            rad_bg,
            n_jobs
        )

        points = pd.concat(points, new_points)























        print("\n\n\n")


def _plot_parameter_space(points, new_points, coordinates, output_folder, suffix):
    begin = time.time()
    points["hue"]     = "Older Iterations"
    new_points["hue"] = "Latest Iteration"
    fullpoints = pd.concat((points, new_points))

    grid = sns.pairplot(
        fullpoints,
        diag_kind="hist",
        vars=coordinates,
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
    print("{}s to plot current iteration".format(round(end - begin, 2)))


def _load_point(filename, filename_pattern, coordinates):
    """
    Load Ctot from the given file. TODO: Generalize
    Filename needs to contain the full path to the file.

    :param filename:
    :param filename_pattern:
    :return:                    Dict
    """
    point = {}
    result = parse(filename_pattern, filename)
    point = result.named

    for coordinate in coordinates:
        if coordinate not in list(point.keys()):
            raise RuntimeError(f"Missing coordinate {coordinate} while trying to read in file {filename}")

    point["values"] = np.loadtxt(filename + ".cool", usecols=3)

    return point


def _load_existing_data(folder, filename_pattern, coordinates):
    """
    Loads Ctot. TODO Needs generic version.

    # TODO Needs a way of handling file extensions

    :param folder:
    :return:
    """
    # Find all files in the specified folder (and subfolders)
    filenames = []
    for dirpath, dirnames, fnames in os.walk(folder):
        filenames += [os.path.join(dirpath, fname) for fname in fnames]

    points = [] # list of dicts for DataFrame constructor

    for filename in filenames:
        if filename.endswith(".cool"):
            points.append(_load_point(filename, filename_pattern, coordinates))

    points = pd.DataFrame(points)
    return points


def _set_up_grid(num_per_dim, parameter_space, parameter_margins, rad_parameter_space, rad_parameter_margins):
    """
    Fills parameter space with a grid as starting point. Takes margins into account.

    :param num_per_dim:
    :param parameter_space:
    :param parameter_margins:
    :param rad_parameter_space:
    :param rad_parameter_margins:
    :return:
    """
    param_space_with_margins = _get_param_space_with_margins(
        parameter_space, parameter_margins, rad_parameter_space, rad_parameter_margins
    )

    param_names =  list(param_space_with_margins.keys())
    param_values = [param_space_with_margins[k] for k in param_names]
    param_grid = [np.linspace(*pv, num_per_dim) for pv in param_values]


    # https://stackoverflow.com/a/46744050
    index = pd.MultiIndex.from_product(param_grid, names=param_names)
    points = pd.DataFrame(index=index).reset_index()

    points["values"] = np.nan

    print("Default grid:")
    print(points)
    exit()
    # TODO: Verify default grid

    return points


def _get_param_space_with_margins(parameter_space, parameter_margins, rad_parameter_space, rad_parameter_margins):
    """
    Returns entire parameter space - including radiation component - with margins.

    :param parameter_space:
    :param parameter_margins:
    :param rad_parameter_space:
    :param rad_parameter_margins:
    :return:
    """
    param_space_with_margins = {}

    for k, v in parameter_margins:
        if hasattr(v, "__iter__"):  # sequence
            if len(v) != 2:
                raise RuntimeError("Margins must either be number or iterable of length 2: " + str(v))
            param_space_with_margins[k] = (min(v), max(v))
        elif type(v) == float or type(v) == int:
            interval_length = abs(parameter_space[k][1] - parameter_space[k][0])
            param_space_with_margins[k] = (
                min(parameter_space[k] - interval_length * parameter_margins[k]),
                max(parameter_space[k] + interval_length * parameter_margins[k])
            )

    for k, v in rad_parameter_margins:
        if hasattr(v, "__iter__"):  # sequence
            if len(v) != 2:
                raise RuntimeError("Margins must either be number or iterable of length 2: " + str(v))
            param_space_with_margins[k] = (min(v), max(v))
        elif type(v) == float or type(v) == int:
            interval_length = abs(rad_parameter_space[k][1] - rad_parameter_space[k][0])
            param_space_with_margins[k] = (
                min(rad_parameter_space[k] - interval_length * rad_parameter_margins[k]),
                max(rad_parameter_space[k] + interval_length * rad_parameter_margins[k])
            )

    return param_space_with_margins


def _get_rad_bg_as_function(rad_params):
    interpolators = {}
    rad_data = {}

    # Load radiation data and build interpolators
    for k, v in rad_params.items():
        rad_data[k] = np.loadtxt(k)
        print("Loaded radiation data:")
        print(rad_data)
        exit() # TODO: Verify radiation data is loaded correctly
        interpolators[k] = interpolate.InterpolatedUnivariateSpline(rad_data[:,0], rad_data[:,1], k=1, ext=1)

    valid_keys = list(rad_params.keys())

    combined_energies = np.sort(np.concatenate([rad_data[k] for k in rad_params.keys()]).flatten())
    # Cloudy only considers energies between 3.04e-9 Ryd and 7.453e6 Ryd, see Hazy p. 33
    combined_energies = np.clip(combined_energies, 3.04e-9, 7.354e6)

    def rad_bg_function(rad_multipliers):
        # TODO: Verify function behaves as expected (plots!)
        f_nu = sum([interpolators[k](combined_energies) * rad_multipliers[k] for k in rad_multipliers.keys() if k in valid_keys])
        return f_nu

    return rad_bg_function


def _f_nu_to_string(f_nu):
    # Note: The columns are switched around!!!
    out = "f(nu) = {0} at {1}\ninterpolate ({2}  {3}\n)".format(
        f_nu[0,1],
        f_nu[0,0],
        f_nu[1,1],
        f_nu[1,0]
    )

    for i in range(2, f_nu.shape[0]):
        out += "continue ({0}  {1})\n".format(f_nu[i,1], f_nu[i,0])

    out += "iterate to convergence"
    return out


def _cloudy_evaluate(input_file, path_to_source, output_folder, filename_pattern, points, rad_bg_function, n_jobs):
    """

    :param input_file:          String representing the cloudy in put file template
    :param path_to_source:
    :param output_folder:       Note: Make sure to include iteration!
    :param filename_pattern:
    :param points:
    :param rad_bg_function:
    :return:
    """
    filenames = []

    for row in points.loc[points["value"] == np.nan].itertuples(index=False):   # TODO: Arbitrary values
        filestring = input_file.format_map(row) # TODO: Important! File save location still missing
        filestring += "\n"
        filestring += _f_nu_to_string(rad_bg_function(row))

        # TODO Verify this .. construction works
        with open(os.path.join(path_to_source, "..", filename_pattern.format(row)) + ".in") as f:
            f.write(filestring)

        filenames.append(os.path.join(path_to_source, "..", filename_pattern.format(row)))

    with open(os.path.join(output_folder, "_filenames.temp"), "w") as file:
        for filename in filenames:
            file.write(filename + "\n")


    os.system("parallel -j {n_jobs} '{cloudy_path} -r' :::: filenames".format(
        n_jobs=n_jobs,
        cloudy_path=os.path.join(path_to_source, "cloudy.exe")
    ))

    # TODO: Find a way to generically incorporate other data here

    # TODO: Implement reading and moving files to output folder. Cloudy file saving needs to be solved first.
    # # Step 4: Read cooling data
    # # for now only Ctot
    # for i, filename in enumerate(input_files):
    #     points[i, -1] = np.loadtxt(filename + ".cool", usecols=3)
    #
    # # Step 5: Move files to cache
    # pattern = get_base_filename_from_parameters("-?[0-9\.]+", "-?[0-9\.]+", "-?[0-9\.]+", "-?[0-9\.]+") + "-?[0-9\.]+"
    # #os.system("mv " + pattern + " " + cache_folder)
    # os.system("ls | grep -P {} | xargs -I % -n 1000 mv -t {} %".format(pattern, cache_folder))

    os.remove(os.path.join(output_folder, "_filenames.temp"))

    return points


def interpolate_delaunay(points, interp_coords, margins=0):
    """
    Use points to interpolate at positions interp_coords using Delaunay triangulation.
    TODO: This entire function needs to be re-reviewed for the rewrite

    :param points:          Points; coords + values; Shape (N, M+1)
    :param interp_coords:   Coords only; Shape (N', M)
    :return:                1. Array of successfully interpolated points with values at interp_coords; Shape (N'', M+1)
                            2. Mask for interp_coords that is true for each element that couldnt be interpolated/
                            was ignored; Shape (N)
                            3. The triangulation object
    """

    # TODO: Generalize for multiple values
    tri = spatial.Delaunay(points.drop("values", axis=1))              # Triangulation
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