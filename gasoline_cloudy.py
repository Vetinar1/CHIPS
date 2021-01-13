import os
import numpy as np
from itertools import product
from chips.optimizer import _get_rad_bg_as_function, _f_nu_to_string, _load_existing_data
import matplotlib.pyplot as plt

cloudy_input = """CMB redshift {z}
table HM12 redshift {z}
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
end of lines
"""

print(os.getcwd())
rad_bg_function = _get_rad_bg_as_function(
    {
        # "hhT6": ("spectra/hhT6", [17.5, 23.5], "log"),
        # "hhT7": ("spectra/hhT7", [17.5, 23.5], "log"),
        # "hhT8": ("spectra/hhT8", [17.5, 23.5], "log"),
        # "SFR":("spectra/SFR", [-4, 3], "log"),
        # "old":("spectra/old", [7, 12], "log")
        "SFR":("spectra/SFR", [-5, 3], "log"),
        "old":("spectra/old", [6, 12], "log")
    },
    "."
)

n_jobs = 60
# filename_pattern = "T_{T}__nH_{nH}__hhT6_{T6}__hhT7_{T7}__hhT8_{T8}__old_{old}__SFR_{SFR}"
filename_pattern = "T_{T}__nH_{nH}__old_{old}__SFR_{SFR}"
path_to_exe = "cloudy/source/cloudy.exe"

nH  = list(np.round(np.linspace(-9, 4, 14), 3))
T   = list(np.round(np.linspace(2, 9, 71), 3))
# T6  = list(np.round(np.linspace(17.5, 23.5, 5), 3))
# T7  = list(np.round(np.linspace(17.5, 23.5, 5), 3))
# T8  = list(np.round(np.linspace(17.5, 23.5, 5), 3))
# old = list(np.round(np.linspace(7, 12, 7), 3))
# SFR = list(np.round(np.linspace(-4, 3, 9), 3))
old = list(np.round(np.linspace(6, 12, 7), 3))
SFR = list(np.round(np.linspace(-5, 3, 9), 3))

for z in np.linspace(0, 3, 1):
    filenames = []
    # for p in product(nH, T, T6, T7, T8, old, SFR):
    for p in product(nH, T, old, SFR):
        print(p)
        filestring = cloudy_input[:]
        args = dict(zip(
            # ["hhT6", "hhT7", "hhT8", "old", "SFR"],
            ["old", "SFR"],
            p[2:]
        ))
        rad_bg = rad_bg_function(args)
        filestring += _f_nu_to_string(rad_bg)

        curr_filename = filename_pattern.format(
            nH=p[0],
            T=p[1],
            # T6=args["hhT6"],
            # T7=args["hhT7"],
            # T8=args["hhT8"],
            old=args["old"],
            SFR=args["SFR"]
        )
        filenames.append(curr_filename)
        filestring = filestring.format(z=z, nH=p[0], T=p[1], fname=curr_filename)

        with open(curr_filename + ".in", "w") as f:
            f.write(filestring)

        plt.title("Radiation background")
        plt.plot(rad_bg[:,1], rad_bg[:,0])
        plt.xscale("log")
        # plt.yscale("log")
        plt.xlabel(r"$E$ in Ryd")
        plt.ylabel(r"$4\pi J_\nu / h$")
        plt.savefig(curr_filename + ".png")
        plt.close()

    path_to_filenames = "_filenames.temp"
    with open(path_to_filenames, "w") as file:
        for filename in filenames:
            file.write(filename + "\n")

    os.system(f"parallel -j {n_jobs} '{path_to_exe} -r' :::: {path_to_filenames}")

    folder = "grid_gasoline_header2/"
    os.mkdir(folder)
    for filename in filenames:
        os.system(f"mv {filename}* {folder}")

    points = _load_existing_data(folder, filename_pattern, ["T", "nH", "old", "SFR"])
    points.to_csv("grid_gasoline_header2.csv", index=False)