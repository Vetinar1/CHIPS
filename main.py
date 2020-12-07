from chips.optimizer import sample
import numpy as np
import sys
import os

# cloudy heuristic iterative parameter sampler

cloudy_input = """CMB redshift 0
table HM12 redshift 0
metals 0 log
hden {nH}
constant temperature {T}
stop zone 1
iterate to convergence
print last
print short
set save prefix "{fname}"
save overview last ".overview"
save cooling last ".cool"
save heating last ".heat"
save cooling each last ".cool_by_element"
save element hydrogen last ".H_ionf"
save element helium last ".He_ionf"
save element oxygen last ".O_ionf"
save element carbon last ".C_ionf"
save element neon last ".Ne_ionf"
save element magnesium last ".Mg_ionf"
save element silicon last ".Si_ionf"
save last line emissivity ".lines"
H 1 1215.67A
H 1 1025.72A
He 2 1640.43A
C 3 977.020A
C 4 1548.19A
C 4 1550.78A
N 5 1238.82A
N 5 1242.80A
O 6 1031.91A
O 6 1037.62A
Si 3 1206.50A
Si 4 1393.75A
Si 4 1402.77A
end of line"""

out_folder = "output"
out = sample(
    cloudy_input=cloudy_input,
    cloudy_source_path="cloudy/source",
    output_folder=out_folder,
    output_filename=f"data",
    param_space={
        "T":[2, 9],
        "nH":[-9, 4],
        #"Z":[-2, 0],
        #"z":[0, 2]
    },
    param_space_margins={
        "T":0.1,
        "nH":0.1,
        #"Z":0.1,
        #"z":[0, 2.2]
    },
    rad_params={
        "hhT6":("spectra/hhT6", [17.5, 23.5], "log"),
        "SFR":("spectra/SFR", [-4, 3], "lin")
    },
    rad_params_margins={
        "hhT6":0.1,
        "SFR":0.1
    },
    existing_data=None,
    initial_grid=4,
    perturbation_scale=1,
    filename_pattern=None,

    dex_threshold=1,
    over_thresh_max_fraction=0.2,
    dex_max_allowed_diff=3,
    random_samples_per_iteration=500,
    n_jobs=50,
    n_partitions=10,
    max_iterations=50,
    max_storage_gb=20,
    max_time=0.05*3600,
    plot_iterations=True,

    debug_plot_2d=False
)
