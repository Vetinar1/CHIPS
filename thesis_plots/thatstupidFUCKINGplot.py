import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial, optimize
import pandas as pd
from scipy.interpolate import interpn
from matplotlib.lines import Line2D
from scipy.stats import binned_statistic_2d

show = False
FORMAT = "pdf"

GRID_DATA = "../gasoline_header2_grid/grid_gasoline_header2.csv"
RAND_DATA = "../gasoline_header2_random/random_gasoline_header2.csv"
DEL_DATA = "../run37_gasoline_z0_header2/z0.0.points"
# DEL_DATA = "run39_gasoline_z0_header2_extended2/z0.0.points"

gridpoints = pd.read_csv(GRID_DATA, delimiter=",")
delpoints = pd.read_csv(DEL_DATA, delimiter=",")
randpoints = pd.read_csv(RAND_DATA, delimiter=",")

print("No. Gridpoints:", len(gridpoints.index))
print("No. Randpoints:", len(randpoints.index))
print("No. Delpoints:", len(delpoints.index))

# ensuring column order
gridpoints = gridpoints[["T", "nH", "SFR", "old", "values"]]
delpoints = delpoints[["T", "nH", "SFR", "old", "values"]]
randpoints = randpoints[["T", "nH", "SFR", "old", "values"]]

# the coordinates at which we will interpolate
interp_coords = randpoints.drop("values", axis=1).to_numpy()

# Delaunay interpolation
N = delpoints.shape[1] - 1
tri = spatial.Delaunay(delpoints.drop("values", axis=1).to_numpy())
simplex_indices = tri.find_simplex(interp_coords)
simplices = tri.simplices[simplex_indices]
transforms = tri.transform[simplex_indices]

bary = np.einsum(
    "ijk,ik->ij",
    transforms[:, :N, :N],
    interp_coords - transforms[:, N, :]
)

weights = np.c_[bary, 1 - bary.sum(axis=1)]
vals = np.zeros(interp_coords.shape[0])
for i in range(interp_coords.shape[0]):
    vals[i] = np.inner(
        delpoints.to_numpy()[simplices[i], -1],
        weights[i]
    )

randpoints["interp_del"] = vals
randpoints["diff_del"] = randpoints["values"] - randpoints["interp_del"]

delmaxdiff = np.max(np.abs(randpoints["diff_del"]))
delmedian = np.median(np.abs(randpoints["diff_del"]))
delavg = np.average(np.abs(randpoints["diff_del"]))

# Multilinear interpolation
# Transform from 2d dataframe into Nd array
arrpoints = gridpoints.copy()
for column in arrpoints.columns[:-1]:
    colvals = np.sort(arrpoints[column].unique())
    arrpoints[column] -= colvals[0]
    colvals = np.sort(arrpoints[column].unique())
    arrpoints[column] /= colvals[1]
    arrpoints[column] = arrpoints[column].astype(int)

valgrid = np.empty(tuple(np.sort(gridpoints[column].unique()).shape[0] for column in gridpoints.columns[:-1]))
valgrid[:] = np.nan

for row in arrpoints.itertuples():
    valgrid[row[1:-1]] = row[-1]

# Clamp
randpoints = randpoints.loc[randpoints["T"] >= 2]
randpoints = randpoints.loc[randpoints["T"] <= 9]
randpoints = randpoints.loc[randpoints["nH"] >= -9]
randpoints = randpoints.loc[randpoints["nH"] <= 4]
randpoints = randpoints.loc[randpoints["SFR"] >= -5]
randpoints = randpoints.loc[randpoints["SFR"] <= 3]
randpoints = randpoints.loc[randpoints["old"] >= 6]
randpoints = randpoints.loc[randpoints["old"] <= 12]

# interpolate
randpoints["interp_grid"] = interpn(
    tuple(np.sort(gridpoints[column].unique()) for column in gridpoints.columns[:-1]),
    valgrid,
    randpoints[["T", "nH", "SFR", "old"]].to_numpy(),
    method="linear"
)
randpoints["diff_grid"] = randpoints["values"] - randpoints["interp_grid"]

gridmaxdiff = np.max(np.abs(randpoints["diff_grid"]).dropna())
gridmedian = np.median(np.abs(randpoints["diff_grid"]).dropna())
gridavg = np.average(np.abs(randpoints["diff_grid"]).dropna())

fig, ax = plt.subplots(2, 3,
                       sharex="col", sharey=True,
                       figsize=(9, 6),
                       # constrained_layout=True
                       )
# fig.tight_layout(rect=(0, 0, 0.85, 0.85))
ax[0, 0].set_ylabel("Multilinear Interpolation\n" + r"$T$")
ax[1, 0].set_ylabel("Delaunay Interpolation\n" + r"$T$")
ax[1, 0].set_xlabel(r"$n_H$")
ax[1, 0].set_xticks([-8, -6, -4, -2, 0, 2, 4])
ax[1, 1].set_xlabel("SFR")
ax[1, 2].set_xlabel("old")
kwargs = {"vmin": 0, "vmax": 0.5, "s": 0.1}
randpoints = randpoints  # .sample(30000)
def percentile16(x):
    try:
        srt = np.argsort(x)
        x = x[srt]
        w = np.ones(len(x))
        cum = np.cumsum(w)/np.sum(w)
        val = x[np.argsort(abs(cum-0.5))][0]
    except:
        val = np.nan
        pass
    return val

randpoints = randpoints.dropna()
dims = (60, 60)
vmin = 0
vmax = 0.5
mode = "mean"
interp = "nearest"
stat, xe, ye, bn = binned_statistic_2d(
    randpoints["nH"], randpoints["T"], randpoints["diff_grid"].abs(), mode, dims
)
ax[0, 0].imshow(stat.transpose(), interpolation=interp, extent=(-9, 4, 2, 9), aspect='auto', origin='lower', vmin=vmin, vmax=vmax)

stat, xe, ye, bn = binned_statistic_2d(
    randpoints["SFR"], randpoints["T"], randpoints["diff_grid"].abs(), mode, dims
)
ax[0, 1].imshow(stat.transpose(), interpolation=interp, extent=(-5, 3, 2, 9), aspect='auto', origin='lower', vmin=vmin, vmax=vmax)

stat, xe, ye, bn = binned_statistic_2d(
    randpoints["old"], randpoints["T"], randpoints["diff_grid"].abs(), mode, dims
)
ax[0, 2].imshow(stat.transpose(), interpolation=interp, extent=(6, 12, 2, 9), aspect='auto', origin='lower', vmin=vmin, vmax=vmax)


stat, xe, ye, bn = binned_statistic_2d(
    randpoints["nH"], randpoints["T"], randpoints["diff_del"].abs(), mode, dims
)
ax[1, 0].imshow(stat.transpose(), interpolation=interp, extent=(-9, 4, 2, 9), aspect='auto', origin='lower', vmin=vmin, vmax=vmax)

stat, xe, ye, bn = binned_statistic_2d(
    randpoints["SFR"], randpoints["T"], randpoints["diff_del"].abs(), mode, dims
)
ax[1, 1].imshow(stat.transpose(), interpolation=interp, extent=(-5, 3, 2, 9), aspect='auto', origin='lower', vmin=vmin, vmax=vmax)

stat, xe, ye, bn = binned_statistic_2d(
    randpoints["old"], randpoints["T"], randpoints["diff_del"].abs(), mode, dims
)
something = ax[1, 2].imshow(stat.transpose(), interpolation=interp, extent=(6, 12, 2, 9), aspect='auto', origin='lower', vmin=vmin, vmax=vmax)

# ax[0, 0].scatter(randpoints["nH"], randpoints["T"], c=randpoints["diff_grid"].abs(), **kwargs)
# ax[0, 1].scatter(randpoints["SFR"], randpoints["T"], c=randpoints["diff_grid"].abs(), **kwargs)
# ax[0, 2].scatter(randpoints["old"], randpoints["T"], c=randpoints["diff_grid"].abs(), **kwargs)
# ax[1, 0].scatter(randpoints["nH"], randpoints["T"], c=randpoints["diff_del"].abs(), **kwargs)
# ax[1, 1].scatter(randpoints["SFR"], randpoints["T"], c=randpoints["diff_del"].abs(), **kwargs)
# something = ax[1, 2].scatter(randpoints["old"], randpoints["T"], c=randpoints["diff_del"].abs(), **kwargs)


# fig.subplots_adjust(top=0.85)
fig.tight_layout(
    rect=(0, 0, 0.95, 0.95)
)
# fig.subplots_adjust(right=0.8)
bbox1 = ax[1, 2].get_position()
bbox2 = ax[0, 2].get_position()
print(bbox1.bounds[1])
cbar_ax = fig.add_axes([0.96, bbox1.bounds[1], 0.015, bbox2.bounds[1] + bbox2.bounds[3] - bbox1.bounds[1]])
fig.colorbar(something, cax=cbar_ax)

# fig.add_subplot(3, 3, 2)
fig.suptitle("Distribution of interpolation errors in parameter space, MLI vs. Delaunay")
if show:
    plt.show()
else:
    plt.savefig("11_grid_err_dist_small." + FORMAT, transparent=False, bbox_inches="tight")
plt.close()