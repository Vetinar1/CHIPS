import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate as si
import scipy.ndimage

DATA_LABEL = "02_test2"
HIRES_LABEL = "03_fine"

# Load rough data to interpolate
data = pd.read_csv(
    "~/c17.01/runs/" + DATA_LABEL + "/" + DATA_LABEL + ".grid",
    sep="\t+",
    engine="python",
    header=0,
    #index_col=0,
    names=["Iron", "hden", "temp"],
    usecols=[6, 7, 8],
    comment="#"
)

cool_rough = pd.read_csv(
    "~/c17.01/runs/" + DATA_LABEL + "/" + DATA_LABEL + ".cool_by_element",
    sep="\s+",
    header=0,
    names=["Ctot", "CFe"],
    usecols=[2, 28],
    comment="#"
)

hires = pd.read_csv(
    "~/c17.01/runs/" + HIRES_LABEL + "/" + HIRES_LABEL + ".grid",
    sep="\t+",
    engine="python",
    header=0,
    #index_col=0,
    names=["Iron", "hden", "temp"],
    usecols=[6, 7, 8],
    comment="#"
)

cool_fine = pd.read_csv(
    "~/c17.01/runs/" + HIRES_LABEL + "/" + HIRES_LABEL + ".cool_by_element",
    sep="\s+",
    header=0,
    names=["Ctot", "CFe"],
    usecols=[2, 28],
    comment="#"
)

data["Ctot"] = cool_rough["Ctot"]
data["CFe"] = cool_rough["CFe"]
hires["Ctot"] = cool_fine["Ctot"]
hires["CFe"] = cool_fine["CFe"]


# TODO: Interpolate data sets at all points covered in fine set but not rough set
# TODO: Get maximum, minimum, median, mean error
# TODO: Plot interpolated vs fine result

# interpolated = si.griddata(
#     points=data.loc[:, ("Iron", "hden", "temp")],
#     values=data["CFe"],
#     xi=hires.loc[:, ("Iron", "hden", "temp")],
#     method="linear"
# )

# # radial based function interpolation
# RBF_interpolator = si.Rbf(
#     data["Iron"],
#     data["hden"],
#     data["temp"],
#     data["CFe"],
#     function="thin_plate"
# )
#
# interpolated = RBF_interpolator(
#     hires["Iron"],
#     hires["hden"],
#     hires["temp"]
# )
#
# hires["CFe_interp"] = interpolated

# Try ignoring Iron abundance (use same grid as coarse) -> -4.6 and -4.4
hires["CFe_interp"] = np.nan
for fe_abun in [-4.4, -4.6]:
    SBS_interpolator = si.SmoothBivariateSpline(
        data[data["Iron"] == fe_abun].loc[:, "hden"],
        data[data["Iron"] == fe_abun].loc[:, "temp"],
        data[data["Iron"] == fe_abun].loc[:, "CFe"],
    )

    interpolated = SBS_interpolator.ev(
        hires[hires["Iron"] == fe_abun].loc[:, "hden"],
        hires[hires["Iron"] == fe_abun].loc[:, "hden"],
    )

    print(interpolated.shape)
    print(hires.shape)
    print(hires.loc[hires["Iron"] == fe_abun, "CFe_interp"].shape)
    hires.loc[hires["Iron"] == fe_abun, "CFe_interp"] = interpolated


print(hires)

for i, hden in enumerate(pd.unique(hires["hden"])):
    for iron in pd.unique(hires["Iron"]):
        plt.loglog(
            10**hires[(hires["hden"] == hden) & (hires["Iron"] == iron)].loc[:, "temp"],
            hires[(hires["hden"] == hden) & (hires["Iron"] == iron)].loc[:, "CFe"],
            color="black",
            label=r"$n_H$ = " + str(hden)
        )

        plt.loglog(
            10 ** hires[(hires["hden"] == hden) & (hires["Iron"] == iron)].loc[:, "temp"],
            hires[(hires["hden"] == hden) & (hires["Iron"] == iron)].loc[:, "CFe_interp"],
            color="red",
            label=r"$n_H$ = " + str(hden)
        )

    plt.title(r"$n_H = $" + str(hden))
    plt.xlabel(r"$\log(T)$")
    #plt.ylim(min(hires["Ctot"]), max(hires["Ctot"]))
    plt.ylabel("Cooling")
    plt.show()
    #plt.savefig(RUN_LABEL + "_" + str(i) + "_nH" + str(hden).replace(".", "_") + ".png")
    plt.close()






























