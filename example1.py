from chips.optimizer import sample

############################################################################
# Practical of how to use CHIPS - single run (use with Cool object in DIP) #
############################################################################

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
"""

out_folder = "output"
out = sample(
    cloudy_input=cloudy_input,
    cloudy_source_path="cloudy/source",
    output_folder=out_folder,
    output_filename=f"cooling",
    param_space={
        "T":[2, 9],
        "nH":[-9, 4]
    },
    param_space_margins={
        "T":0.1,
        "nH":0.1
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
    perturbation_scale=0.2,
    filename_pattern=None,

    dex_threshold=0.1,
    over_thresh_max_fraction=0.1,
    dex_max_allowed_diff=2,
    random_samples_per_iteration=100,
    n_jobs=60,
    n_partitions=10,
    max_iterations=50,
    max_storage_gb=2,
    max_time=5*3600,
    plot_iterations=True,

    debug_plot_2d=False
)
