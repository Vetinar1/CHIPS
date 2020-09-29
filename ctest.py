import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial

points = np.loadtxt("ctest/data.csv", delimiter=",", skiprows=1)
tris = np.loadtxt("ctest/dtri.csv", delimiter=",").astype(int)
tree = np.loadtxt("ctest/tree", delimiter=" ", usecols=(0, 1, 2)).astype(int)
ptree = np.loadtxt("ctest/ptree", delimiter=" ", usecols=(0, 1, 2)).astype(int)
tree_radius = np.loadtxt("ctest/tree", delimiter=" ", usecols=3)
ptree_radius = np.loadtxt("ctest/ptree", delimiter=" ", usecols=3)

print(np.sqrt(tree_radius))

PLOT_S_TREE = False
PLOT_C_INTERP = False
PLOT_P_INTERP = False
PLOT_DIFF = False
PLOT_P_TREE = True
PLOT_VORONOI = True

if PLOT_P_TREE:
    plt.figure(figsize=(10, 10))
    plt.plot(points[:, 0], points[:, 1], "k.")
    plt.triplot(points[:, 0], points[:, 1], triangles=tris)

    # for i in range(points.shape[0]):
    #     if ptree_radius[i] != 0:
    #         circ = plt.Circle((points[i, 0], points[i, 1]), radius=np.sqrt(ptree_radius[i]), facecolor=None,
    #                           edgecolor="k", linewidth=0.5, fill=False)
    #         plt.gca().add_patch(circ)

    for i in range(points.shape[0]):
        if ptree[i, 1] != -1:
            plt.plot(
                [points[i, 0], points[ptree[i, 1], 0]],
                [points[i, 1], points[ptree[i, 1], 1]],
                "magenta",
                linewidth=2
            )
        if ptree[i, 2] != -1:
            plt.plot(
                [points[i, 0], points[ptree[i, 2], 0]],
                [points[i, 1], points[ptree[i, 2], 1]],
                "cyan",
                linewidth=2
            )

    plt.plot([2], [-3.92], "bo")
    plt.xlim(1, 9)
    plt.ylim(-5, 5)
    plt.title("point tree")
    plt.show()

if PLOT_S_TREE:
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

    for i in range(tris.shape[0]):
        if tree_radius[i] != 0:
            circ = plt.Circle((centroids[i, 0], centroids[i, 1]), radius=np.sqrt(tree_radius[i]), facecolor=None,
                              edgecolor="k", linewidth=0.5, fill=False)
            plt.gca().add_patch(circ)

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

    plt.plot([2], [-3.92], "bo")
    plt.xlim(1, 9)
    plt.ylim(-5, 5)
    plt.title("simplex tree")
    plt.show()


if PLOT_VORONOI:
    voronoi = spatial.Voronoi(points[:,:-1])
    delaunay = spatial.Delaunay(points[:,:-1])

    plt.figure(figsize=(10, 10))
    # plt.plot(points[:,0], points[:,1], "k.")
    spatial.delaunay_plot_2d(delaunay, ax=plt.gca())
    spatial.voronoi_plot_2d(voronoi, ax=plt.gca())
    plt.plot([2], [-3.92], "bo")
    plt.show()


if PLOT_P_INTERP:
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


    plt.figure(figsize=(10, 10))
    plt.plot(points[:,0], points[:,1], "k.")
    plt.triplot(points[:,0], points[:,1], triangles=tris, linewidth=0.5, color="r")
    plt.scatter(interp_coords[:,0], interp_coords[:,1], c=vals, s=8)
    plt.colorbar()
    plt.title("Python interpolation")
    plt.show()

if PLOT_C_INTERP:
    interp_loaded = np.loadtxt("ctest/interp", delimiter=" ")
    plt.figure(figsize=(10, 10))
    plt.plot(points[:,0], points[:,1], "k.")
    plt.triplot(points[:,0], points[:,1], triangles=tris, linewidth=0.5, color="r")
    plt.scatter(interp_loaded[:,0], interp_loaded[:,1], c=interp_loaded[:,2], s=8)
    plt.colorbar()
    plt.title("C interpolation")
    plt.show()

if PLOT_DIFF:
    plt.figure(figsize=(10, 10))
    plt.plot(points[:,0], points[:,1], "k.")
    plt.triplot(points[:,0], points[:,1], triangles=tris, linewidth=0.5, color="r")
    plt.scatter(interp_coords[:,0], interp_coords[:,1], c=np.abs(vals - interp_loaded[:,2]), s=8)
    plt.colorbar()
    plt.show()