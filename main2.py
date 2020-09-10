from chips.optimizer import sample

# cloudy h iterative parameter sampler



out = sample(
    cloudy_input="cloudy.in",
    cloudy_source_path="cloudy/source",
    output_folder="output",
    param_space={
        "T":[2, 8],
        "nH":[-4, 4],
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
        # "hhT6":("spectra/hhT6", [10, 30]),
        # "SFR":("spectra/SFR", [10, 30])  # TODO no clue if these are reasonable
    },
    rad_params_margins={
        # "hhT6":0.1,
        # "SFR":0.1
    },
    existing_data=None, # TODO: Testing
    initial_grid=7,
    perturbation_scale=0.1,
    filename_pattern=None,

    dex_threshold=0.1,
    over_thresh_max_fraction=0.1,
    dex_max_allowed_diff=0.5,
    random_samples_per_iteration=30,
    n_jobs=12,
    n_partitions=10,
    max_iterations=3,
    max_storage_gb=20,
    max_time=0.1*3600,
    plot_iterations=True, # TODO: Testing, Implementation,

    debug_plot_2d=True
)

out.to_csv("output/data.csv", index=False)







# def sample(
#         cloudy_input,
#         cloudy_source_path,
#         output_folder,
#         param_space,
#         param_space_margins,
#
#         rad_params=None,
#         rad_params_margins=None,
#
#         existing_data=None, # TODO: Rename
#         initial_grid=10,
#
#         filename_pattern=None,
#
#         dex_threshold=0.1,
#         over_thresh_max_fraction=0.1,
#         dex_max_allowed_diff=0.5,
#
#         random_samples_per_iteration=30,
#
#         n_jobs=4,
#         n_partitions=10,
#
#         max_iterations=20,
#         max_storage_gb=10,
#         max_time=20,
#
#         plot_iterations=True
# ):