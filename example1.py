from chips.optimizer import sample

####################################################################################
# Practical example of how to use CHIPS - single run (use with Cool object in DIP) #
####################################################################################

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


out = sample(
    cloudy_input=cloudy_input,
    cloudy_source_path="cloudy/source",
    output_folder="example1_output/",
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
    poisson_disc_scale=0.2,
    accuracy_threshold=0.1,
    error_fraction=0.1,
    max_error=2,
    random_samples_per_iteration=100,
    n_jobs=60,
    n_partitions=10,
    max_iterations=30,
    max_storage_gb=2,
    max_time=1*3600,
    plot_iterations=True,
)
