# CHIPS

This is the Cloudy Heuristic Iterative Parameter Sampler, CHIPS for short.
CHIPS is a python package that samples the cooling function for cosmological simulations.
For more information on the cooling function and the goals of this package, refer to my master thesis (insert link to thesis here).

CHIPS is meant to be used in conjunction with DIP: https://github.com/Vetinar1/DIP

## How to install CHIPS

To install CHIPS, either download this repository and move the `chips` folder into your project folder, or run the following command:

`pip3 install git+https://github.com/Vetinar1/CHIPS#egg=CHIPS`

TODO: Ensure that the whole egg business works.

Make sure to have cloudy installed. What else?

TODO: requirements txt

## How does CHIPS work

It is highly recommended that you read chapter 3 of the thesis in order to understand the algorithm used in CHIPS.
In short, CHIPS samples a parameter space using an irregular mesh.
In contrast to the regular grids employed typically in cooling function interpolation, this mesh is not of uniform density.
It is built in such a way that the cooling function interpolation has consistent accuracy across the parameter space.
Areas in which more samples are required to achieve this receive more samples than others.

The algorithm does this by evaluating the interpolation quality using already existing parts of the mesh.
Interpolation on the mesh is done using a [Delaunay triangulation](https://en.wikipedia.org/wiki/Delaunay_triangulation) and [barycentric coordinates](https://en.wikipedia.org/wiki/Barycentric_coordinate_system).

## How to use CHIPS

A working example can be found in the file FILENAME.

The main function to use is the `sample` function:

```python
from chips.optimizer import sample
```

For a full documentation of this function refer to the docstring in the source file.
Its most important parameters are as follows:

`output_folder`: Where to put all the outputs and intermediary files. Folder must not exist.

`output_filename`: How to call the output files.

`cloudy_input`: This needs to be either the path to a cloudy input file template, or a string containing such a template.
A valid cloudy input file template is a cloudy file where all parameters of the parameter space have been replaced with
matching python string formatting placeholders. Here is an example:

```
CMB redshift {z}
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
```

All variables must match the parameter names given in the other arguments!
Further, while technically not necessary it is *highly* recommended to use `.cool` as the file ending for cooling files.
The cooling must be saved to a file, otherwise the program will abort.
`{fname}` refers to the automatically generated filenames and should not be changed.

Also, note the `log` in the metals command. This is important to avoid discontinuities.
See the thesis for further information.

`param_space`: A dictionary defining the extent of the parameter space. Each key defines the name of one dimension, and
the keys must 1:1 match the variables from the cloudy input file template (ignoring `fname`).
Each value defines the extent of the matching parameter dimension. It must be an iterable containing the lower bound
and the upper bound. Example:

```
param_space={
    "T":[2, 9],
    "nH":[-9, 4],
    "Z":[-2, 1],
    "z":[0, 1]
},
```

The parameter space should extend slightly beyond what you plan to use in your cosmological simulations!
Sometimes DIP encounters problems when reaching the edge of the parameter space, so leave a buffer.
How large this buffer should be depends on the density of the sampling. The less dense, the larger the buffer should be.

`param_space_margins`: This defines the "margins" of the parameter space.
The margins extend beyond the extents defined using `param_space`.
They are important for stable sampling.
These margins are *discarded* after the sampling is complete, and do not contribute to the "buffer" you should leave
in `param_space`! (There are, again, technical reasons in the DIP implementation for this.)
If you really do want to look at the points in the margins, check the `.fullpoints` output file.
However, it is recommended you don't use it.

`param_space_margins` must be a dict, with each key exactly matching each key from `param_space`.
The values may be either floats or iterables. If they are floats, they are interpeted as fractions.
E.g. if dimension X goes from 0 to 5, and the margin of X is 0.1, then the margins on each end will have a size of 0.5.
If they are iterables, the extends of the margins are given directly. Example:

```
param_space_margins={
    "T":0.1,
    "nH":0.1,
    "Z":0.1,
    "z":[0, 1.1]
},
```

Note how the redshift gets custom margins, because the redshift can't be negative.
For redshift sampling, consider using slices in conjunction with DIP CoolManager objects instead (see below).
Usually, you should be fine with margins of 0.1.

`rad_params` and `rad_params_margins`: These work analogously to `param_space` and `param_space_margins`, but they
contain spectral components instead. These must be read from files.
They are automatically scaled, overlaid, interpolated and added to the input file before each run.

Apart from the slightly different input method you can treat these like parameters.
Again, the keys define the names of the parameters.
However, the values must give the name of the input file containing the spectral component, the extents of the
multipliers applied to these components, and whether the multiplications is supposed to happen in lin or log space.
Example:

```
rad_params={
    "hhT6":("spectra/hhT6", [17.5, 23.5], "log"),
    "hhT7":("spectra/hhT7", [17.5, 23.5], "log")
    "SFR":("spectra/SFR", [-4, 3], "log"),
# The relevant file should not actually be interpreted as lin, this is only for demonstrational purposes
    "old":("spectra/old", [7, 12], "lin")
},
rad_params_margins={
    "hhT6":0.1,
    "hhT7":0.1,
    "SFR":0.1,
    "old":0.1
},
```

To see an example for what the spectrum files should look like, check the `spectra` folder.
Their units should match the cloudy `f(nu)` command.

`existing_data`: Path to a folder containing data from a previous run. Filename patterns must match. Will also check
subdirectories.

`n_jobs`: Number of cloudy instances to run at once.

`initial_grid`:  Related to margins, number of grid points to use for outer hull.
Must be two or larger. Should be four or five. Greater values probably don't lead to increased benefits.

`perturbation_scale`: TODO RENAME THIS ARGUMENT
Related to initial sampling. The minimum distance to use for the poisson disc sampling.
The number of points you can expect is about `1/r^d` or less (r is the distance, d is the number of dimensions).

`random_samples_per_iteration`: How many uniformly distributed random samples to add in each iteration.
Supposed to help against local optima. Consider your dimensionality when chosing a value.

`dex_threshold`: The error threshold to judge the interpolation quality, in dex.
Errors lower than the threshold are going to be considered good enough.
Errors larger than the threshold will lead to more samples
The higher this is the less samples you will get overall.

`over_thresh_max_fraction`: This is the maximum fraction of samples for which the interpolation error may be
`dex_threshold` or worse.
If the number of samples for which the error is `dex_threshold` or worse is smaller than `over_thresh_max_fraction`,
the sampling is considered to be good enough and the algorithm stops.

`dex_max_allowed_diff`: The maximum error any sample may have in its interpolation.
Will *prevent* the algorithm from completing, even if the other two conditions are fulfilled. See thesis.

`max_iterations`: Will forcefully quit after this many iterations.

`max_storage_gb`: Will forcefully quit after this much space (in GB) has been taken up.
Will attempt to stay below the threshold.

`cleanup`: Clean up intermediary files. Valid options: None, "outfiles", "full".
"outfiles" will remove cloudy output files immediately after they are generated; they take up by far the most space.
"full" will get rid of all intermediate files.

`max_time`: Will forcefully quit after this much (real world) time has elapsed. Will attempt to stay under threshold.

`max_samples`: Will forcefully quit after this many samples. Will attempt to stay under threshold.

After the algorithm is done, you will get four output files:

* `.points`, which contains all the samples.
* `.tris`, which contains the triangulations.
* `.neighbors`, which contains the neighborhood relations between the simplices.
* `.fullpoints`, which contains all the samples *and* the samples from the margins. You shouldn't worry about this one.

You can then use these for DIP. 