import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial

points = np.loadtxt("ctest3d/data.csv", delimiter=",")#, skiprows=1)
# tree = np.loadtxt("ctest/tree", delimiter=" ", usecols=(0, 1, 2)).astype(int)
ctris = np.loadtxt("ctest3d/dtri.csv", delimiter=",").astype(int)
# ptree = np.loadtxt("ctest/ptree", delimiter=" ", usecols=(0, 1, 2)).astype(int)
interp_loaded = np.loadtxt("ctest3d/interpml", delimiter=" ")
# tree_radius = np.loadtxt("ctest/tree", delimiter=" ", usecols=3)
# ptree_radius = np.loadtxt("ctest/ptree", delimiter=" ", usecols=3)

PLOT_S_TREE = False
PLOT_C_INTERP = True
PLOT_P_INTERP = True
PLOT_DIFF = True
PLOT_P_TREE = False
PLOT_VORONOI = False
PLOT_DIRECT_TRILINEAR_INTERP = False

CLIP_FOR_COLORBARS = False
CLIP_MIN = -32
CLIP_MAX = -12


if PLOT_P_INTERP:
    n_count = 100*100
    N = points.shape[1]-1
    interp_coords = np.zeros((n_count, N))

    count = 0
    for i in range(100):
        for j in range(100):
            interp_coords[count, 0] =2 + i * (8 - 2) / 100.
            interp_coords[count, 1] = -4 + j * 8 / 100.
            interp_coords[count, 2] = -0.17
            # interp_coords[count, 3] = 20
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
    print(points)
    vals = np.zeros(n_count)
    for i in range(n_count):
        vals[i] = np.inner(
            points[simplices[i], -1],
            weights[i]
        )


    if CLIP_FOR_COLORBARS:
        vals = np.clip(vals, CLIP_MIN, CLIP_MAX)

    plt.figure(figsize=(1.2*6, 6))
    # plt.triplot(points[:,0], points[:,1], triangles=tri.simplices, linewidth=0.5, color="r")
    plt.scatter(interp_coords[:,0], interp_coords[:,1], c=vals, s=2)
    # plt.scatter(points[:,0], points[:,1], color="k", s=4, zorder=1000)
    plt.colorbar()
    plt.clim(vmin=-32, vmax=-13)
    plt.xlim(1, 9)
    plt.ylim(-5, 5)
    plt.title(r"Python interpolation of $\log(\Lambda)$, $D = " + str(N) + "$")
    plt.gcf().set_dpi(200)
    plt.xlim(1, 9)
    plt.ylim(-5, 5)
    plt.xlabel(r"Temperature $T$")
    plt.ylabel(r"Hydrogen density $n_H$")
    plt.savefig("pnotri.png", transparent=True)
    plt.show()


if PLOT_C_INTERP:
    if CLIP_FOR_COLORBARS:
        interp_loaded[:,-1] = np.clip(interp_loaded[:,-1], CLIP_MIN, CLIP_MAX)

    plt.figure(figsize=(1.2*6, 6))
    # plt.triplot(points[:,0], points[:,1], triangles=ctris, linewidth=0.5, color="r")
    plt.scatter(interp_loaded[:,0], interp_loaded[:,1], c=interp_loaded[:,2], s=2)
    # plt.scatter(points[:,0], points[:,1], color="k", s=4, zorder=1000)
    plt.colorbar()
    plt.clim(vmin=-32, vmax=-13)
    plt.xlim(1, 9)
    plt.ylim(-5, 5)
    plt.gcf().set_dpi(200)
    plt.title(r"Multilinear C interpolation of $\log \Lambda$, $D = " + str(N) + "$")
    plt.xlabel(r"Temperature $T$")
    plt.ylabel(r"Hydrogen density $n_H$")
    plt.savefig("cnotri.png", transparent=True)
    plt.show()


if PLOT_DIFF:
    plt.figure(figsize=(1.2*6, 6))
    # plt.triplot(points[:,0], points[:,1], triangles=ctris, linewidth=0.5, color="r")
    plt.scatter(interp_coords[:,0], interp_coords[:,1], c=np.abs(vals - interp_loaded[:,-1]), s=2)
    # plt.scatter(points[:,0], points[:,1], color="k", s=4, zorder=1000)
    plt.colorbar()
    plt.xlim(1, 9)
    plt.ylim(-5, 5)
    plt.gcf().set_dpi(200)
    plt.title(r"Difference between Python and Multilinear C interpolation, $D = " + str(N) + "$")
    plt.xlabel(r"Temperature $T$")
    plt.ylabel(r"Hydrogen density $n_H$")
    plt.savefig("diffnotri.png", transparent=True)
    plt.show()


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


if PLOT_VORONOI:
    voronoi = spatial.Voronoi(points[:,:-1])
    delaunay = spatial.Delaunay(points[:,:-1])

    plt.figure(figsize=(10, 10))
    # plt.plot(points[:,0], points[:,1], "k.")
    spatial.delaunay_plot_2d(delaunay, ax=plt.gca())
    spatial.voronoi_plot_2d(voronoi, ax=plt.gca())
    plt.plot([2], [-3.92], "bo")
    plt.show()


if PLOT_S_TREE:
    centroids = np.zeros((ctris.shape[0], 3))
    for i in range(ctris.shape[0]):
        centroids[i] = 0
        for j in range(3):
            centroids[i] += points[ctris[i,j]]

        centroids[i] /= 3

    plt.figure(figsize=(10, 10))
    plt.plot(points[:,0], points[:,1], "k.")
    plt.gcf().set_dpi(200)
    # plt.plot(centroids[:,0], centroids[:,1], "r.")
    plt.triplot(points[:,0], points[:,1], triangles=ctris)

    for i in range(ctris.shape[0]):
        if tree_radius[i] != 0:
            circ = plt.Circle((centroids[i, 0], centroids[i, 1]), radius=np.sqrt(tree_radius[i]), facecolor=None,
                              edgecolor="k", linewidth=0.5, fill=False)
            plt.gca().add_patch(circ)

    # for i in range(ctris.shape[0]):
    #     if tree[i, 1] != -1:
    #         plt.plot(
    #             [centroids[i, 0], centroids[tree[i, 1], 0]],
    #             [centroids[i, 1], centroids[tree[i, 1], 1]],
    #             "magenta",
    #             linewidth=0.5
    #         )
    #     if tree[i, 2] != -1:
    #         plt.plot(
    #             [centroids[i, 0], centroids[tree[i, 2], 0]],
    #             [centroids[i, 1], centroids[tree[i, 2], 1]],
    #             "cyan",
    #             linewidth=0.5
    #         )

    plt.xlim(1, 9)
    plt.ylim(-5, 5)
    plt.title("simplex tree")
    plt.show()


def trilinear_interp(data, vals, points):
    dim1 = np.unique(data[:,0])
    dim2 = np.unique(data[:,1])
    dim3 = np.unique(data[:,2])
    out = np.zeros(points.shape[0])

    ldata = pd.DataFrame(data, columns=["T", "nH", "Z"])
    ldata["values"] = vals

    # Stupid naive implementation
    # https://en.wikipedia.org/wiki/Trilinear_interpolation
    for i in range(points.shape[0]):
        mincoords = [0, 0, 0]
        maxcoords = [0, 0, 0]
        for j1 in dim1:
            if j1 <= points[i,0]:
                mincoords[0] = j1
            else:
                maxcoords[0] = j1
                break

        for j2 in dim2:
            if j2 <= points[i,1]:
                mincoords[1] = j2
            else:
                maxcoords[1] = j2
                break

        for j3 in dim3:
            if j3 <= points[i,2]:
                mincoords[2] = j3
            else:
                maxcoords[2] = j3
                break

        nns = ldata.loc[
              ((ldata["T"] == mincoords[0]) | (ldata["T"] == maxcoords[0])) &
              ((ldata["nH"] == mincoords[1]) | (ldata["nH"] == maxcoords[1])) &
              ((ldata["Z"] == mincoords[2]) | (ldata["Z"] == maxcoords[2]))
              ]
        assert(len(nns.index) == 2**3)

        xd = (points[i,0] - mincoords[0]) / abs(maxcoords[0] - mincoords[0])
        yd = (points[i,1] - mincoords[1]) / abs(maxcoords[1] - mincoords[1])
        zd = (points[i,2] - mincoords[2]) / abs(maxcoords[2] - mincoords[2])

        c000 = float(nns.loc[(nns["T"] == mincoords[0]) & (nns["nH"] == mincoords[1]) & (nns["Z"] == mincoords[2]), "values"])
        c100 = float(nns.loc[(nns["T"] == maxcoords[0]) & (nns["nH"] == mincoords[1]) & (nns["Z"] == mincoords[2]), "values"])
        c001 = float(nns.loc[(nns["T"] == mincoords[0]) & (nns["nH"] == mincoords[1]) & (nns["Z"] == maxcoords[2]), "values"])
        c101 = float(nns.loc[(nns["T"] == maxcoords[0]) & (nns["nH"] == mincoords[1]) & (nns["Z"] == maxcoords[2]), "values"])
        c010 = float(nns.loc[(nns["T"] == mincoords[0]) & (nns["nH"] == maxcoords[1]) & (nns["Z"] == mincoords[2]), "values"])
        c110 = float(nns.loc[(nns["T"] == maxcoords[0]) & (nns["nH"] == maxcoords[1]) & (nns["Z"] == mincoords[2]), "values"])
        c011 = float(nns.loc[(nns["T"] == mincoords[0]) & (nns["nH"] == maxcoords[1]) & (nns["Z"] == maxcoords[2]), "values"])
        c111 = float(nns.loc[(nns["T"] == maxcoords[0]) & (nns["nH"] == maxcoords[1]) & (nns["Z"] == maxcoords[2]), "values"])

        c00 = c000 * (1 - xd) + c100 * xd
        c01 = c001 * (1 - xd) + c101 * xd
        c10 = c010 * (1 - xd) + c110 * xd
        c11 = c011 * (1 - xd) + c111 * xd

        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd

        c = c0 * (1 - zd) + c1 * zd
        out[i] = c

    return out


if PLOT_DIRECT_TRILINEAR_INTERP:
    repr_data = np.loadtxt("repr1_ploeck_schaye/pl_schy.grid", usecols=(8, 7, 6))
    repr_vals = np.loadtxt("repr1_ploeck_schaye/pl_schy.cool", usecols=3)
    repr_vals = np.log10(repr_vals)
    print("Points loaded from reproduction: ", repr_data.shape[0])
    print("Point loaded from delaunay: ", points.shape[0])

    N = 3
    n_count = 100*100
    interp_coords = np.zeros((n_count, N))

    count = 0
    for i in range(100):
        for j in range(100):
            interp_coords[count, 0] =2 + i * (8 - 2) / 100.
            interp_coords[count, 1] = -4 + j * 8 / 100.
            interp_coords[count, 2] = -0.17
            # interp_coords[count, 3] = 20
            count += 1

    tvals = trilinear_interp(repr_data, repr_vals, interp_coords)

    plt.figure(figsize=(10, 10))
    plt.scatter(interp_coords[:,0], interp_coords[:,1], c=tvals, s=8)
    plt.colorbar()
    plt.title(r"Trilinear grid interpolation of $\log(\Lambda)$")
    plt.xlabel(r"Temperature $T$")
    plt.ylabel(r"Hydrogen density $n_H$")
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.scatter(interp_coords[:,0], interp_coords[:,1], c=np.abs(vals - tvals), s=8)
    plt.colorbar()
    plt.title(r"Trilinear grid interpolation of $\log(\Lambda)$ - difference to Delaunay")
    plt.xlabel(r"Temperature $T$")
    plt.ylabel(r"Hydrogen density $n_H$")
    plt.show()

