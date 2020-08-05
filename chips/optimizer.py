import os
from chips.utils import *
import pandas as pd
import itertools
from scipy import interpolate, spatial
import matplotlib.pyplot as plt


# TODO: Options for cloudy to clean up after itself
# TODO: Check existence of radiation files
def sample(
        cloudy_input,
        cloudy_source_path,
        output_folder,
        parameter_space,
        param_space_margins,

        rad_params=None,
        rad_params_margins=None,

        existing_data=None, # TODO: Rename
        initial_grid=10,

        filename_pattern=None,

        dex_threshold=0.1,
        over_thresh_max_fraction=0.1,
        dex_max_diff=0.5,

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
    :param parameter_space:                 "key":(min, max)
    :param param_space_margins:             2 values: absolute. 1 value: relative
    :param rad_params:                      "key":(min, max). key doubles as filepath
    :param rad_params_margins:              2 values: absolute. 1 value: relative
    :param rad_params_names:                TODO optional prettier names
    :param existing_data:
    :param initial_grid:                    How many samples in each dimension in initial grid. 0 to disable
    :param dex_threshold:
    :param over_thresh_max_fraction:
    :param dex_max_diff:
    :param random_samples_per_iteration:
    :param n_jobs:
    :param n_partitions:
    :param max_iterations:
    :param max_storage_gb:
    :param max_time:
    :param plot_iterations:
    :return:
    """

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    elif len(os.listdir(output_folder)) != 0:
        choice = input("Chosen output folder is not empty. Proceed? (y/n)")
        if choice not in ["y", "Y", "yes", "Yes"]:
            print("Aborting...")
            exit()

    if not os.path.exists(os.path.join(cloudy_source_path, "cloudy.exe")):
        raise RuntimeError("No cloudy.exe found at: " + str(cloudy_source_path))

    for key in parameter_space.keys():
        if key not in list(param_space_margins.keys()):
            raise RuntimeError("No margins given for: " + str(key))

    if not filename_pattern:
        filename_pattern = "__".join(
            ["_".join(
                [key, "{" + str(key) + "}"]
            ) for key in parameter_space.keys()]
        )

    # TODO: Verify default filename_pattern
    # TODO: If filename pattern is not default, verify that it contains all necessary parameters
    # TODO: Include radiation in filename pattern
    # TODO: Ensure filenames are not too long (256 characters?)
    print("Default pattern:")
    print(filename_pattern)
    exit()

    input_filled = cloudy_input.format_map({k : v[0] for k, v in parameter_space.items()}).replace("{fname}", "")
    if "{" in input_filled or "}" in input_filled:
        raise RuntimeError("Error while filling cloudy input: Missing parameter or syntax error")


    if not initial_grid and not existing_data:
        raise RuntimeError("Need to specify existing data or use initial_grid to set up a run")

    if existing_data:
        if not os.path.isdir(output_folder):
            raise RuntimeError("Specified existing data at " + str(existing_data) + " but is not a dir or does not exist")

        # TODO: Implement existing data. Format needs to match!
        points = None

    if initial_grid:
        points = _set_up_grid(initial_grid, parameter_space, param_space_margins, rad_params, rad_params_margins)


    # TODO: Rework this section to make sure it includes ALL parameters
    print("NUMBER_OF_PARTITIONS".ljust(50) + str(n_partitions) + "\n")
    print("THRESHOLD (dex)".ljust(50) + str(dex_threshold) + "\n")
    print("OVER_THRESH_MAX_FRACTION".ljust(50) + str(over_thresh_max_fraction) + "\n")
    print("MAX_DIFF (dex)".ljust(50) + str(dex_max_diff) + "\n")
    print("RANDOM_NEW_POINTS".ljust(50) + str(random_samples_per_iteration) + "\n")
    print("\n")
    print("NUMBER_OF_JOBS".ljust(50) + str(n_jobs) + "\n")
    print("MAX_ITERATIONS".ljust(50) + str(max_iterations) + "\n")
    print("MAX_STORAGE (GB)".ljust(50) + str(max_iterations) + "\n")
    print("MAX_TIME (h)".ljust(50) + str(seconds_to_human_readable(max_time)) + "\n")
    print("\n")
    # print("Points loaded from cache:".ljust(50) + str(init_point_count) + "\n")
    print("dimensions (T, nH, Z, z) (start, stop, step) ".ljust(50) + str(parameter_space) + "\n")


    # TODO Measure time
    rad_bg = _get_rad_bg_as_function(rad_params)
    points = _cloudy_evaluate(cloudy_input, cloudy_source_path, output_folder, filename_pattern, points, rad_bg, n_jobs)

    # number of dimensions
    # TODO: Generalize for multiple values
    N = len(points.columns)-1

    iteration = 0
    while True:
        iteration += 1
        print("{:*^50}".format("Iteration {}".format(iteration)))

        len_pre_drop =  len(points.index)
        points = points.drop_duplicates()
        len_post_drop = len(points.index)

        if len_pre_drop > len_post_drop:
            print("Dropped {} duplicate samples".format(len_pre_drop - len_post_drop))

        # For plotting the triangulation
        if plot_iterations:
            fig, ax = plt.subplots(1, n_partitions, figsize=(10 * n_partitions, 10))

        for partition in range(n_partitions):
            subset = points.iloc[partition::n_partitions]
            bigset = pd.concat((points, subset)).drop_duplicates()

            # # Not sure this is necessary, just to be safe
            # subset.reset_index()
            # bigset.reset_index()

            tri = spatial.Delaunay(bigset.loc[bigset.columns - ["values"]].to_numpy())
            simplex_indices = tri.find_simplex(subset.values)



















        print("\n\n\n")







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


def _load_existing_data(folder):
    pass    # TODO

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