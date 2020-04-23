import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate as si
import scipy.ndimage
from scipy.optimize import curve_fit

DATA_LABEL = "02_test2"
HIRES_LABEL = "03_fine"

# Load rough data to interpolate
data = pd.read_csv(
    "~/c17.01/runs/" + DATA_LABEL + "/" + DATA_LABEL + ".grid",
    sep="\t+",
    engine="python",
    #header=0,
    #index_col=0,
    names=["iron", "hden", "temp"],
    usecols=[6, 7, 8],
    comment="#"
)

cool_rough = pd.read_csv(
    "~/c17.01/runs/" + DATA_LABEL + "/" + DATA_LABEL + ".cool_by_element",
    sep="\s+",
    #header=0,
    names=["Ctot", "CFe"],
    usecols=[2, 28],
    comment="#"
)

hires = pd.read_csv(
    "~/c17.01/runs/" + HIRES_LABEL + "/" + HIRES_LABEL + ".grid",
    sep="\t+",
    engine="python",
    #header=0,
    #index_col=0,
    names=["iron", "hden", "temp"],
    usecols=[6, 7, 8],
    comment="#"
)

cool_fine = pd.read_csv(
    "~/c17.01/runs/" + HIRES_LABEL + "/" + HIRES_LABEL + ".cool_by_element",
    sep="\s+",
    #header=0,
    names=["Ctot", "CFe"],
    usecols=[2, 28],
    comment="#"
)

data["Ctot"] = cool_rough["Ctot"]
data["CFe"] = cool_rough["CFe"]
hires["Ctot"] = cool_fine["Ctot"]
hires["CFe"] = cool_fine["CFe"]


# convert to multidimensional block of data (from table)
# there is probably a much smarter way of doing this with multi indices...
# 1. dimensions
dim_iron = np.sort(data["iron"].unique())
dim_hden = np.sort(data["hden"].unique())
dim_temp = np.sort(data["temp"].unique())

# 2. matrix of functional values
matrix = np.zeros((len(dim_iron), len(dim_hden), len(dim_temp)))
for i, iron in enumerate(dim_iron):
    for j, hden in enumerate(dim_hden):
        for k, temp in enumerate(dim_temp):
            matrix[i][j][k] = data.loc[(data["iron"] == iron) & (data["hden"] == hden) & (data["temp"] == temp), "CFe"]

matrix = np.log(matrix)
hires["CFe"] = np.log(hires["CFe"])
data["CFe"] = np.log(data["CFe"])


# for i, hden in enumerate(hires["hden"].unique()):
#     plt.plot(
#         hires.loc[(hires["iron"] == -4.4) & (hires["hden"] == hden), "temp"],
#         hires.loc[(hires["iron"] == -4.4) & (hires["hden"] == hden), "CFe"],
#         label=hden
#     )
#
# plt.title("Dependence on T at different hden")
# # plt.legend()
# plt.show()
#
# for i, temp in enumerate(hires["temp"].unique()):
#     plt.plot(
#         hires.loc[(hires["iron"] == -4.4) & (hires["temp"] == temp), "hden"],
#         hires.loc[(hires["iron"] == -4.4) & (hires["temp"] == temp), "CFe"],
#         label=temp
#     )
#
# plt.title("Dependence on hden at different T")
# # plt.legend()
# plt.show()
# exit()


def func_iron(x, m, t):
    """
    linear function for fitting iron abundance -> cooling function
    """
    return m*x + t


def func_hden(x, m, t, a, b, c):
    """
    linear function + gaussian for "hump"
    """
    return m*x + t + a*np.exp(- (x-b)**2 / (2*c**2))


def func_temp(x, a, b, c, d, e, f):
    """
    simple 5th order polynomial
    """
    return a + b*x + c*x**2 + d*x**3 + e*x**4 + f*x**5

def get_func_temp(a, b, c, d, e, f):
    def func_temp(x):
        return a + b*x + c*x**2 + d*x**3 + e*x**4 + f*x**5
    return func_temp

# Interpolate along iron (linear)
# for j, hden in enumerate(dim_hden):
#     for k, temp in enumerate(dim_temp):
fe_interp = si.interp1d(
    dim_iron,           # x: The iron abundances
    matrix,             # y: The cooling values
    axis=0,             # interpolate along iron abundances
    kind="linear"
)

# the interpolation now "splinters" into a plane for each point we want to interpolate at
matrix_interp_1 = fe_interp(hires["iron"])
print(matrix_interp_1.shape)    # -> (1008, 9, 21): 1008 planes of size 9x21. Now we interpolate in each of these

nh_interps = []
for i in range(matrix_interp_1.shape[0]):
    nh_interp_param, _ = curve_fit(
        func_hden,
        dim_hden,
        matrix_interp_1
    )

    nh_interps.append(nh_interp)


# we now have an interpolator for each entry in our dataframe
# note: technically, we only need one interpolator for each entry in iron_dim. optimization potential
assert(len(nh_interps) == len(hires.index))

matrix_interp_2 = np.zeros((matrix_interp_1.shape[0], matrix_interp_1.shape[2]))
for i, interp_func in enumerate(nh_interps):
    matrix_interp_2[i] = interp_func(hires.loc[i, "hden"])

# matrix_interp_2 is now condense to a line that has to be interpolated fo reach data point
temp_interps = []
for i in range(matrix_interp_2.shape[0]):
    temp_interp_param, _ = curve_fit(
        func_temp,
        dim_temp,
        matrix_interp_2[i]
    )
    temp_interp = get_func_temp(*temp_interp_param)
    temp_interps.append(temp_interp)

interpolated = np.zeros(len(hires.index))
for i, interp_func in enumerate(temp_interps):
    interpolated[i] = interp_func(hires.loc[i, "temp"])


hires["CFe_interp"] = interpolated
hires["CFe_diff"] = hires["CFe"] - hires["CFe_interp"]

#print(hires.loc[:, ("CFe", "CFe_interp", "CFe_diff")].to_string())
print(hires.to_string())

for i, hden in enumerate(pd.unique(hires["hden"])):
    for iron in pd.unique(hires["iron"]):
        plt.plot(
            hires[(hires["hden"] == hden) & (hires["iron"] == iron)].loc[:, "temp"],
            hires[(hires["hden"] == hden) & (hires["iron"] == iron)].loc[:, "CFe"],
            color="black",
            label=r"$n_H$ = " + str(hden)
        )

        plt.plot(
            hires[(hires["hden"] == hden) & (hires["iron"] == iron)].loc[:, "temp"],
            hires[(hires["hden"] == hden) & (hires["iron"] == iron)].loc[:, "CFe_interp"],
            color="red",
            label=r"$n_H$ = " + str(hden)
        )

    plt.title(r"$n_H = $" + str(hden))
    plt.xlabel(r"$\log(T)$")
    #plt.ylim(min(hires.loc[:, ("CFe", "CFe_interp")]), max(hires.loc[:, ("CFe", "CFe_interp")]))
    #plt.ylim(-80, -50)
    plt.ylabel("Cooling")
    plt.show()
    #plt.savefig(RUN_LABEL + "_" + str(i) + "_nH" + str(hden).replace(".", "_") + ".png")
    plt.close()


# plt.loglog(
#     10 ** hires.loc[:, "temp"],
#     hires.loc[:, "CFe"],
#     color="black"
# )
#
# plt.loglog(
#     10 ** hires.loc[:, "temp"],
#     hires.loc[:, "CFe_interp"],
#     color="red"
# )
#
# plt.show()




























