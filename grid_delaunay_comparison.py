import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from chips import optimizer
from scipy import spatial
from scipy.interpolate import interpn
from tqdm import tqdm
import itertools
from pathlib import Path
from pprint import pprint
from parse import parse

GRID_DATA = "gasoline_header2_grid/grid_gasoline_header2.csv"
DEL_DATA = "gasoline_header2_random/random_gasoline_header2.csv"
# DEL_DATA = "run37_gasoline_z0_header2/z0.0.points"
# DEL_DATA = "run39_gasoline_z0_header2_extended2/z0.0.points"

gridpoints = pd.read_csv(GRID_DATA, delimiter=",")
delpoints = pd.read_csv(DEL_DATA, delimiter=",")

# ensuring column order
gridpoints = gridpoints[["T", "nH", "SFR", "old", "values"]]
delpoints = delpoints[["T", "nH", "SFR", "old", "values"]]

CLIP_MIN = -32
CLIP_MAX = -12


interp_coords = gridpoints.drop("values", axis=1).to_numpy()
N = delpoints.shape[1]-1
tri = spatial.Delaunay(delpoints.drop("values", axis=1).to_numpy())
simplex_indices = tri.find_simplex(interp_coords)
simplices = tri.simplices[simplex_indices]
transforms = tri.transform[simplex_indices]

# The following is adapted from
# https://stackoverflow.com/questions/30373912/interpolation-with-delaunay-triangulation-n-dim/30401693#30401693
# 1. barycentric coordinates of points; N-1

bary = np.einsum(
    "ijk,ik->ij",
    transforms[:, :N, :N],
    interp_coords - transforms[:, N, :]
)

# 2. Add dependent barycentric coordinate to obtain weights
weights = np.c_[bary, 1 - bary.sum(axis=1)]

# 3. Interpolation
# TODO vectorize
vals = np.zeros(interp_coords.shape[0])
for i in range(interp_coords.shape[0]):
    vals[i] = np.inner(
        delpoints.to_numpy()[simplices[i], -1],
        weights[i]
    )

gridpoints["interpolated"] = vals
gridpoints["diff"] = gridpoints["values"] - gridpoints["interpolated"]

print("Interpolation of grid values by Triangulation:")
print("Maximum Difference:", np.max(np.abs(gridpoints["diff"])))
print("Median Difference:", np.median(np.abs(gridpoints["diff"])))
print("Average Difference: ", np.average(np.abs(gridpoints["diff"])))
delmaxdiff = np.max(np.abs(gridpoints["diff"]))
delmedian = np.median(np.abs(gridpoints["diff"]))
delavg = np.average(np.abs(gridpoints["diff"]))

gridpoints_plot = gridpoints.copy()
gridpoints = gridpoints.drop(["interpolated", "diff"], axis=1)

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

# print(len(delpoints.index))
delpoints = delpoints.loc[delpoints["T"] >= 2]
delpoints = delpoints.loc[delpoints["T"] <= 9]
delpoints = delpoints.loc[delpoints["nH"] >= -9]
delpoints = delpoints.loc[delpoints["nH"] <= 4]
delpoints = delpoints.loc[delpoints["SFR"] >= -5]
delpoints = delpoints.loc[delpoints["SFR"] <= 3]
delpoints = delpoints.loc[delpoints["old"] >= 6]
delpoints = delpoints.loc[delpoints["old"] <= 12]
# print(len(delpoints.index))
# print(delpoints)

delpoints["interpolated"] = interpn(
    tuple(np.sort(gridpoints[column].unique()) for column in gridpoints.columns[:-1]),
    valgrid,
    delpoints.drop("values", axis=1).to_numpy(),
    method="linear"
)
delpoints["diff"] = delpoints["values"] - delpoints["interpolated"]

print()
print("Interpolation of Triangulation values by grid:")
print("Maximum Difference:", np.max(np.abs(delpoints["diff"])))
print("Median Difference:", delpoints["diff"].abs().median())
print("Average Difference: ", np.abs(delpoints["diff"].abs().mean()))

# print(gridpoints["values"].min(), gridpoints["values"].max(), gridpoints["values"].mean())
# print(arrpoints["values"].min(), arrpoints["values"].max(), arrpoints["values"].mean())
# print(delpoints["values"].min(), delpoints["values"].max(), delpoints["values"].mean())
# print(delpoints["interpolated"].min(), delpoints["interpolated"].max(), delpoints["interpolated"].mean())


# H, X1 = np.histogram(delpoints["diff"].dropna(), bins=10, density=True)
# dx = X1[1] - X1[0]
# plt.plot(X1[1:], np.cumsum(H)*dx)

plt.figure(figsize=(8, 6))
plt.title("Normalized cumsum")
plt.axvline(delmedian, c="b", linestyle="--")
plt.axvline(delpoints["diff"].abs().median(), c="orange", linestyle="--")
plt.axvline(delavg, c="b", linestyle=":")
plt.axvline(delpoints["diff"].abs().mean(), c="orange", linestyle=":")
plt.axhline(0.5, c="k", linestyle="--")
X3 = np.sort(gridpoints_plot["diff"].abs().dropna())
plt.plot(X3, np.arange(X3.shape[0])/X3.shape[0], label="Delaunay -> Grid", c="b")
X2 = np.sort(delpoints["diff"].abs().dropna())
plt.plot(X2, np.arange(X2.shape[0])/X2.shape[0], label="Grid -> Delaunay", c="orange")
plt.legend()
plt.ylabel("Normalized cumulative sum")
plt.xlabel("Absolute difference")
plt.xlim(-0.5, 2.5)
# plt.show()
plt.savefig("comparison_plots/gridvsrandom.png")
# plt.close()

# plt.title("Not-Normalized cumsum")
# X3 = np.sort(gridpoints_plot["diff"].abs().dropna())
# plt.plot(X3, np.arange(X3.shape[0]), label="Delaunay -> Grid")
# X2 = np.sort(delpoints["diff"].abs().dropna())
# plt.plot(X2, np.arange(X2.shape[0]), label="Grid -> Delaunay")
# plt.legend()
# plt.ylabel("cumulative sum")
# plt.xlabel("Absolute difference")
# plt.xlim(-0.5, 4)
# plt.show()
# # plt.savefig("cumsum-small.png")