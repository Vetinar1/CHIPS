from chips.optimizer import sample
import numpy as np
import sys
import os

###########################################################################################
# Practical example of how to use CHIPS - multiple runs (for use with CoolManager in DIP) #
###########################################################################################

cloudy_input = """CMB redshift {z}
table HM12 redshift {z}
metals 0 log
hden 0
constant temperature {T}
stop zone 1
iterate to convergence
print last
print short
set save prefix "{fname}"
save overview last ".overview"
save cooling last ".cool"
"""

out_folder = "example2_data"
os.mkdir(out_folder)
sys.stderr = open(out_folder + "/errlog", "w")

for i in np.linspace(0, 3, 1):
    sys.stdout = open(out_folder + f"/z{i}.log", "w")
    out = sample(
        cloudy_input=cloudy_input.replace("{z}", str(i)),
        cloudy_source_path="cloudy/source",
        output_folder=out_folder + f"/z{i}",
        output_filename=f"z{i}",
        param_space={
            "T":[2, 9],
            "nH":[-9, 4],
        },
        param_space_margins={
            "T":0.1,
            "nH":0.1,
        },
        rad_params=None,
        initial_grid=4,
        poisson_disc_scale=0.2,
        filename_pattern=None,
        accuracy_threshold=0.1,
        error_fraction=0.2,
        max_error=1,
        random_samples_per_iteration=100,
        n_jobs=60,
        n_partitions=10,
        max_iterations=20,
        max_storage_gb=2,
        max_time=1*3600,
        plot_iterations=True,
    )
