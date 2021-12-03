from chips import optimizer
import numpy as np
import sys
import os

# See also README file
# If you encounter any bugs, please open an issue on github: https://github.com/Vetinar1/CHIPS


# 1. Adjust path to cloudy source folder
cloudy_source_path = "path/to/cloudy/source/folder"


# 2. Choose the output folder. Folder must not exist yet.
out_folder = "output/location"


# 3. Choose the minimum redshift, maximum redshift, and redshift stepsize.
# CHIPS will run once for EACH step between z_min and z_max.
# For z_min = 1, z_max = 2 and z_step = 0.1 it will run 10 times: At 1.0, 1.1, 1.2, ..., 1.9
z_min = 0       # inclusive
z_max = 10      # exclusive
z_step = 0.1


# 4.1 Uncomment the parameters you wish to use/Comment out the ones you don't want to use.
# The format is "param_name":[lower_limit, upper_limit]
# Valid parameters are T, nH, Z
# Since DIP has some trouble interpolating at the edges of the parameter space, it is recommended to choose your limits
# wider than you need them. Depending on the density of the sampling and the length of the parameter axis you should
# add about 0.2 - 1 magnitudes padding.
param_space={
    "T":[2, 9],             # limits in log10 kelvin
    "nH":[-4, 4],           # limits in log10 cm^-3
    "Z":[-3, 1]             # limits in log10 Z_sol
}

# 4.2 Choose default values for the parameters above. These are only applied if the respective parameter is missing
# from param_space.
T_default  = 5
nH_default = 0
Z_default  = 1


# 5. Uncomment the radiation fields you want to use/Comment out the ones you don't want to use.
# The format is "label":("source/path", [lower_limit, upper_limit], "scaling")
# The limits apply to the multiplier that will be used for the SED. The scaling keywords "log" and "lin" describe
# whether this multiplier is given in log or lin space.
# If you want to use your own SEDs the source files should match the examples provided. Their units should match the
# CLOUDY f(nu) command.
# For the provide SEDs, see
# https://ui.adsabs.harvard.edu/abs/2019MNRAS.490.1518O/abstract
# and
# https://ui.adsabs.harvard.edu/abs/2016MNRAS.458.2516K/abstract
# for background information.
#
# Important: Uncomment with care. The radiation field multipliers are treated just like T, nH, Z. Make sure your
# parameter space does not become too large!
rad_param_space={
    # "hhT6":("spectra/hhT6", [17.5, 23.5], "log"),
    # "hhT7":("spectra/hhT7", [17.5, 23.5], "log"),
    # "hhT8":("spectra/hhT8", [17.5, 23.5], "log"),
    # "SFR":("spectra/SFR", [-5, 3], "log"),
    # "old":("spectra/old", [6, 10], "log")
}


# 6. Choose the number of initial samples in EACH RUN (at each redshift). They will be uniformly and evenly distributed
# throughout the parameter space (using poisson disc sampling).
# A good *starting point* is 10^#dimensions. You will probably want to use less at high dimensions if you are
# trying to save computation time, or more if you want to cover the parameter space better.
# If you choose it to high the effect of the actual sampling algorithm will diminish, but if you choose it too low
# parts of the parameter space may end up undersampled.
initial_sample_count = 1000


# 7. Choose the number of random, uniformly sampled points to add in each iteration.
# Supposed to help cover parts of the parameter space that the algorithm might miss.
random_samples_per_iteration = 100


# 8. Choose number of simultaneous jobs to run
n_jobs = 40


# 9. Choose exit conditions. For all of these, None is a valid option.
# 9.1 Maximum number of samples before stopping. Will attempt to stop before reaching it.
max_samples = 30000

# 9.2 Maximum amount of (wall clock) runtime before stopping. Will attempt to stop before reaching it.
max_time = 3*24*3600        # in seconds

# 9.3 Maximum amount of storage to use before stopping. Will attempt to stop before reaching it.
max_storage_gb = 10         # in gigabyte, surprisingly

# 9.4 Accuracy threshold: An interpolation with a relative error of this size or smaller is considered "accurate"
accuracy_threshold = 0.1    # 0.1 = 10% error is still "accurate"

# 9.5 Error fraction: Program will not terminate if this fraction of points or more is not interpolated "accurately"
error_fraction = 0.1        # 0.1 = At most 10% of points may not fulfill accuracy_threshold

# 9.6 Max error: Program will not terminate if the lowest accuracy interpolation is worse than this threshold
# Be conservative with this value! You may also turn it off by setting it to None
max_error = 3               # 3 = The worst interpolation may be off by a factor of 300%


# 10. Cleanup options to save space:
# None: Do not perform cleanup
# "outfiles": Delete all cloudy .out files (which take up the bulk of disk space). They are not required for the
# calculation of the cooling function.
# "full": Clean entire working directory. (Not recommended)
cleanup = "outfiles"


# 11. Choose whether to save the triangulation. You do not need this if you intend to use PSI, only if you want to use
# the classic DIP algorithm. If you don't know what either of that means, keep it disabled.
# If you later decided you would like the triangulation anyway, you can laod the .points file as a csv with pandas,
# and feed the dataframe into chips.build_and_save_delaunay.
save_triangulation = False


# 12. Whether to plot samples at each step. Turn off if you expect very high sample counts.
plot_iterations = True


# 12. Choose existing data to load. This may be used to continue a previous run. The parameter space of the run must
# match exactly, but the exit conditions may be changed.
# Must be a list of filenames or foldernames. (Even if its a single one, it must be in a list!)
# Filenames will attempt to be loaded as if they are .points output files from previous runs.
# For foldernames, all contents of the folders and subfolders will be crawled for .cool files that will then be loaded.
existing_data = None


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
assert(initial_sample_count >= 0)
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

if "T" not in param_space:
    cloudy_input = cloudy_input.replace("{T}", str(T_default))
if "nH" not in param_space:
    cloudy_input = cloudy_input.replace("{nH}", str(nH_default))
if "Z" not in param_space:
    cloudy_input = cloudy_input.replace("{Z}", str(Z_default))

if not param_space:
    param_space = None
    param_space_margins = None
else:
    param_space_margins = {k:0.1 for k in param_space.keys()}

if not rad_param_space:
    rad_param_space = None
    rad_param_space_margins = None
else:
    rad_param_space_margins = {k:0.1 for k in rad_param_space.keys()}


dims = len(param_space) + len(rad_param_space)
poisson_disc_scale = 1 / np.power(initial_sample_count, 1/dims)

if os.path.exists(out_folder):
    print(f"Folder {out_folder} already exists")
    exit()

os.mkdir(out_folder)

sys.stderr = open(os.path.join(out_folder, "errlog"), "w")

for i in np.arange(z_min, z_max, z_step):
    sys.stdout = open(os.path.join(out_folder, f"z{i}.log"), "w")
    out = optimizer.sample(
        cloudy_input=cloudy_input.replace("{z}", str(i)),
        cloudy_source_path=cloudy_source_path,
        output_folder=os.path.join(out_folder, f"z{i}"),
        output_filename=f"z{i}",
        param_space=param_space,
        param_space_margins=param_space_margins,
        rad_params=rad_param_space,
        rad_params_margins=rad_param_space_margins,
        initial_grid=4,
        poisson_disc_scale=poisson_disc_scale,
        accuracy_threshold=accuracy_threshold,
        error_fraction=error_fraction,
        max_error=max_error,
        random_samples_per_iteration=random_samples_per_iteration,
        n_jobs=n_jobs,
        n_partitions=10,
        max_samples=max_samples,
        max_iterations=None,
        max_storage_gb=max_storage_gb,
        max_time=max_time,
        plot_iterations=plot_iterations,
        cleanup=cleanup,
        existing_data=existing_data,
        significant_digits=4,
        save_triangulation=save_triangulation
    )


# create mapfile
with open(os.path.join(out_folder, "mapfile")) as f:
    for z in np.arange(z_min, z_max, z_step):
        f.write(str(z) + " " + os.path.join(out_folder, f"z{z}\n"))

























