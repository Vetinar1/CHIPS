import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial

points = np.loadtxt("ctest/data.csv", delimiter=",", skiprows=1)
tris = np.loadtxt("ctest/dtri.csv", delimiter=",").astype(int)
tree = np.loadtxt("ctest/tree", delimiter=" ", usecols=(0, 1, 2)).astype(int)
tree_radius = np.loadtxt("ctest/tree", delimiter=" ", usecols=3)

print(np.sqrt(tree_radius))

centroids = np.zeros((tris.shape[0], 3))
for i in range(tris.shape[0]):
    centroids[i] = 0
    for j in range(3):
        centroids[i] += points[tris[i,j]]

    centroids[i] /= 3

plt.figure(figsize=(10, 10))
plt.plot(points[:,0], points[:,1], "k.")
plt.plot(centroids[:,0], centroids[:,1], "r.")
plt.triplot(points[:,0], points[:,1], triangles=tris)

# for i in range(tris.shape[0]):
#     if tree_radius[i] != 0:
#         circ = plt.Circle((centroids[i, 0], centroids[i, 1]), radius=np.sqrt(tree_radius[i]), facecolor=None,
#                           edgecolor="k", linewidth=0.5, fill=False)
#         plt.gca().add_patch(circ)

for i in range(tris.shape[0]):
    if tree[i, 1] != -1:
        plt.plot(
            [centroids[i, 0], centroids[tree[i, 1], 0]],
            [centroids[i, 1], centroids[tree[i, 1], 1]],
            "magenta",
            linewidth=0.5
        )
    if tree[i, 2] != -1:
        plt.plot(
            [centroids[i, 0], centroids[tree[i, 2], 0]],
            [centroids[i, 1], centroids[tree[i, 2], 1]],
            "cyan",
            linewidth=0.5
        )

plt.xlim(1, 9)
plt.ylim(-5, 5)
plt.show()



n_count = 100*100
N = 2
interp_coords = np.zeros((n_count, 2))

count = 0
for i in range(100):
    for j in range(100):
        interp_coords[count, 0] =2 + i * (8 - 2) / 100.
        interp_coords[count, 1] = -4 + j * 8 / 100.
        count += 1

tri = spatial.Delaunay(points[:,:-1])
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
vals = np.zeros(n_count)
for i in range(n_count):
    vals[i] = np.inner(
        points[simplices[i], 2],
        weights[i]
    )

interp_loaded = np.loadtxt("ctest/interp", delimiter=" ")
plt.figure(figsize=(10, 10))
plt.plot(points[:,0], points[:,1], "k.")
plt.triplot(points[:,0], points[:,1], triangles=tris, linewidth=0.5, color="r")
plt.scatter(interp_loaded[:,0], interp_loaded[:,1], c=interp_loaded[:,2], s=8)
plt.colorbar()
plt.show()

plt.figure(figsize=(10, 10))
plt.plot(points[:,0], points[:,1], "k.")
plt.triplot(points[:,0], points[:,1], triangles=tris, linewidth=0.5, color="r")
plt.scatter(interp_coords[:,0], interp_coords[:,1], c=vals, s=8)
plt.colorbar()
plt.show()

plt.figure(figsize=(10, 10))
plt.plot(points[:,0], points[:,1], "k.")
plt.triplot(points[:,0], points[:,1], triangles=tris, linewidth=0.5, color="r")
plt.scatter(interp_coords[:,0], interp_coords[:,1], c=np.abs(vals - interp_loaded[:,2]), s=8)
plt.colorbar()
plt.show()