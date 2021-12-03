import os
import numpy as np
from itertools import product
from chips.optimizer import _get_rad_bg_as_function, _f_nu_to_string, load_existing_raw_data
import matplotlib.pyplot as plt
import sys
from itertools import product

# See also README file


# 1. Adjust path to cloudy source folder
cloudy_source_path = "path/to/cloudy/source/folder"


# 2. Choose the output folder. Folder must not exist yet.
out_folder = "output/location"


# 3. Choose the minimum redshift, maximum redshift, and redshift stepsize.
# Sampling will run once for EACH step between z_min and z_max.
# For z_min = 1, z_max = 2 and z_step = 0.1 it will run 10 times: At 1.0, 1.1, 1.2, ..., 1.9
z_min = 0       # inclusive
z_max = 10      # exclusive
z_step = 0.1


# 4.1 Uncomment the parameters you wish to use/Comment out the ones you don't want to use.
# The format is "param_name":[lower_limit, upper_limit, stepsize]
# Valid parameters are T, nH, Z
# Since DIP has some trouble interpolating at the edges of the parameter space, it is recommended to choose your limits
# wider than you need them. Depending on the density of the sampling and the length of the parameter axis you should
# add about 0.2 - 1 magnitudes padding.
param_space={
    "T":[2, 9, 0.1],             # limits in log10 kelvin
    "nH":[-4, 4, 0.1],           # limits in log10 cm^-3
    "Z":[-3, 1, 0.1]             # limits in log10 Z_sol
}

# 4.2 Choose default values for the parameters above. These are only applied if the respective parameter is missing
# from param_space.
T_default  = 5
nH_default = 0
Z_default  = 1


# 5. Uncomment the radiation fields you want to use/Comment out the ones you don't want to use.
# The format is "label":("source/path", [lower_limit, upper_limit, stepsize], "scaling")
# The limits apply to the multiplier that will be used for the SED. The scaling keywords "log" and "lin" describe
# whether this multiplier is given in log or lin space.
# If you want to use your own SEDs the source files should match the examples provided. Their units should match the
# CLOUDY f(nu) command.
#
# Important: Uncomment with care. The radiation field multipliers are treated just like T, nH, Z. Make sure your
# parameter space does not become too large!
rad_param_space={
    # "hhT6":("spectra/hhT6", [17.5, 23.5, 0.5], "log"),
    # "hhT7":("spectra/hhT7", [17.5, 23.5, 0.5], "log"),
    # "hhT8":("spectra/hhT8", [17.5, 23.5, 0.5], "log"),
    # "SFR":("spectra/SFR", [-5, 3, 0.5], "log"),
    # "old":("spectra/old", [6, 10, 0.5], "log")
}


# 7. Choose number of simultaneous jobs to run
n_jobs = 40





# You don't really need to worry about anything that happens below here
###########################################################################################
###########################################################################################

cloudy_source_path = os.path.expanduser(cloudy_source_path)
out_folder         = os.path.expanduser(out_folder)


# These tests are not exhaustive but should catch the worst mistakes
assert(os.path.exists(cloudy_source_path))
assert(z_min >= 0)
assert(z_min <= z_max)
for k, v in param_space.items():
    assert(v[0] <= v[1])
for k, v in rad_param_space.items():
    assert(v[1][0] <= v[1][1])
    assert(v[2] in ["log", "lin"])
assert(n_jobs > 0)




# Cloudy input can also be a path to a file.
cloudy_input = """CMB redshift {z}
table HM12 redshift {z}
metals {Z} log
hden {nH}
constant temperature {T}
stop zone 1
iterate to convergence
print last
print short
set save prefix "{fname}"
save overview last ".overview"
save cooling last ".cool"
"""


path_to_exe = os.path.join(cloudy_source_path, "cloudy.exe")


if os.path.exists(out_folder):
    print(f"Folder {out_folder} already exists")
    exit()

os.mkdir(out_folder)

filename_pattern = ""
dims = 0
sorted_params = sorted(list(param_space.keys()) + list(rad_param_space.keys()))
for label in sorted_params:
    filename_pattern += str(label) + "_{" + str(label) + "}__"
    dims += 1

filename_pattern = filename_pattern[:-2]

if param_space is None or type(param_space) is not dict:
    param_space = {}

if rad_param_space:
    rad_bg = _get_rad_bg_as_function(rad_param_space, out_folder)
elif rad_param_space is None or type(rad_param_space) is not dict:
    rad_param_space = {}

if "T" not in param_space:
    cloudy_input = cloudy_input.replace("{T}", str(T_default))
if "nH" not in param_space:
    cloudy_input = cloudy_input.replace("{nH}", str(nH_default))
if "Z" not in param_space:
    cloudy_input = cloudy_input.replace("{Z}", str(Z_default))

sys.stderr = open(os.path.join(out_folder, "errlog"), "w")

fullspace = []
for p in sorted_params:
    if p in param_space:
        fullspace.append(np.arange(*param_space[p]))
    elif p in rad_param_space:
        fullspace.append(np.arange(*rad_param_space[p][1]))

# Surprisingly theres no simple, readable numpy function for taking the cartesian product of two arrays
# So I'm using itertools.product here for readability
# not like the performance matters much in comparison to the cloudy runs...
points = []
for p in product(*fullspace):
    points.append(p)

points = np.array(points)
points = np.round(points, 4)

for z in np.arange(z_min, z_max, z_step):
    # create a cloudy input file for each point
    filenames = []
    for p in points:
        # Create parameter:value dict to fill filename_pattern
        point_dict = {sorted_params[i]:p[i] for i in range(dims)}

        curr_filename = filename_pattern.format(**point_dict)
        filenames.append(curr_filename)

        # add z and fname to dict for cloudy input string
        point_dict["z"] = z
        point_dict["fname"] = curr_filename

        # extract rad_params to separate dict
        rad_point_dict = {}
        for k in rad_param_space.keys():
            rad_point_dict[k] = point_dict[k]
            del point_dict[k]

        filestring = cloudy_input[:] # copy
        filestring = filestring.format(**point_dict)

        # add rad_params to cloudy input file
        if rad_param_space:
            filestring += "\n" + _f_nu_to_string(rad_bg(rad_point_dict))

        with open(curr_filename + ".in", "w") as f:
            f.write(filestring)

    path_to_filenames = "_filenames.temp"
    with open(path_to_filenames, "w") as file:
        for filename in filenames:
            file.write(filename + "\n")

    os.system(f"parallel -j {n_jobs} '{path_to_exe} -r' :::: {path_to_filenames}")


    out_subfolder = os.path.join(out_folder, f"z{z}")
    rawdata_folder = os.path.join(out_subfolder, f"rawdata")
    os.mkdir(out_subfolder)
    os.mkdir(rawdata_folder)

    # we have to move these one by one, otherwise mv fails
    for filename in filenames:
        os.system(f"mv {filename}* {rawdata_folder}")


    # load cooling and heating functions
    tempdata = load_existing_raw_data(
        rawdata_folder,
        filename_pattern,
        sorted_params,
        [2, 3],
        file_ending=".cool",
        column_names=["Htot", "Ctot"]
    ).drop_duplicates()

    # load ionization rates
    iondata = load_existing_raw_data(
        rawdata_folder,
        filename_pattern,
        sorted_params,
        [4, 5, 6, 7, 8, 9, 10],
        file_ending=".overview",
        column_names=["ne", "H2", "HI", "HII", "HeI", "HeII", "HeIII"]
    ).drop_duplicates()

    # print(f"Loaded iondata with {len(iondata.index)} rows")
    # print(f"Saving tempdata to {folder}")
    # tempdata.to_csv(os.path.join(folder, "tempdata.csv"), index=False)
    # print(f"Saving iondata to {folder}")
    # tempdata.to_csv(os.path.join(folder, "iondata.csv"), index=False)

    # Merge tempdata and iondata
    mergedata = tempdata.merge(
        iondata,
        "inner",
        on=sorted_params
    )
    outfile = os.path.join(out_folder, f"z{z}.points")
    mergedata.to_csv(outfile, index=False)
    os.system(f"cp {outfile} {out_folder}")


# create mapfile
with open(os.path.join(out_folder, "mapfile"), "w") as f:
    for z in np.arange(z_min, z_max, z_step):
        f.write(str(z) + " " + os.path.join(out_folder, f"z{z}\n"))
