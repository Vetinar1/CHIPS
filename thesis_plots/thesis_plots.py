import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from chips.utils import poisson_disc_sampling
from scipy import spatial, optimize
import pandas as pd
from scipy.interpolate import interpn
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import seaborn as sns
import scipy

FORMAT = "pdf"

def plot_mli_cube(bg=True, az = -64, el = 20, show=True):
    cube = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1]
    ])
    p = [0.6, 0.3, 0.4]

    plane = np.array([
        [p[0], 0, 0],
        [p[0], 0, 1],
        [p[0], 1, 1],
        [p[0], 1, 0],
        # for plotting:
        [p[0], 0, 0],
    ])
    fig = plt.figure()
    ax = plt.axes(projection="3d", proj_type="ortho")

    side1 = [0, 1, 3, 2, 0]
    side2 = [4, 5, 7, 6, 4]

    side3 = [0, 1, 5, 4, 0]
    side4 = [2, 3, 7, 6, 2]

    ax.plot3D([p[0]], [p[1]], [p[2]], "rP", markersize=8)
    ax.plot3D([p[0], p[0]], [p[1], p[1]], [p[2], 0], "r:")

    ax.plot3D(cube[side1,0], cube[side1,1], cube[side1,2], color="gray")
    ax.plot3D(cube[side1,0], cube[side1,1], cube[side1,2], "ko")
    ax.plot3D(cube[side2,0], cube[side2,1], cube[side2,2], color="gray")
    ax.plot3D(cube[side2,0], cube[side2,1], cube[side2,2], "ko")

    ax.plot3D(cube[side3,0], cube[side3,1], cube[side3,2], color="k")
    ax.plot3D(cube[side4,0], cube[side4,1], cube[side4,2], color="k")

    ax.plot3D(plane[:,0], plane[:,1], plane[:,2], "k--")
    ax.plot3D(plane[:,0], plane[:,1], plane[:,2], "ks")

    ax.view_init(el, az)
    if bg:
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_xticklabels([" "] * 6)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_yticklabels([" "] * 6)
        ax.set_zticks(np.linspace(0, 1, 6))
        ax.set_zticklabels([" "] * 6)
    else:
        ax.grid(False)
        plt.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    if show:
        plt.show()
    else:
        plt.savefig("01_MLI_cube_a." + FORMAT, transparent=True, bbox_inches="tight")
    plt.close()


    fig = plt.figure()
    ax = plt.axes(projection="3d", proj_type="ortho")

    ax.plot3D(cube[side1,0], cube[side1,1], cube[side1,2], "o", c="gray")
    ax.plot3D(cube[side2,0], cube[side2,1], cube[side2,2], "o", c="gray")
    ax.plot3D(cube[side1,0], cube[side1,1], cube[side1,2], color="gray")
    ax.plot3D(cube[side2,0], cube[side2,1], cube[side2,2], color="gray")
    ax.plot3D(cube[side3,0], cube[side3,1], cube[side3,2], color="gray")
    ax.plot3D(cube[side4,0], cube[side4,1], cube[side4,2], color="gray")

    ax.plot3D(plane[:,0], plane[:,1], plane[:,2], "k")
    ax.plot3D(plane[:,0], plane[:,1], plane[:,2], "ko")

    ax.plot3D([p[0]] * 2, [0, 1], [p[2]] * 2, "ks")
    ax.plot3D([p[0]] * 2, [0, 1], [p[2]] * 2, "k--")

    ax.plot3D([p[0]], [p[1]], [p[2]], "rP", markersize=8)
    ax.plot3D([p[0], p[0]], [p[1], p[1]], [p[2], 0], "r:")


    ax.view_init(el, az)
    if bg:
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_xticklabels([" "] * 6)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_yticklabels([" "] * 6)
        ax.set_zticks(np.linspace(0, 1, 6))
        ax.set_zticklabels([" "] * 6)
    else:
        ax.grid(False)
        plt.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    if show:
        plt.show()
    else:
        plt.savefig("01_MLI_cube_b." + FORMAT, transparent=True, bbox_inches="tight")
    plt.close()


def plot_mli_nonlinear(stilts=False, show=True):
    coords = np.array([
        [2, 1, 1.5],
        [2, 2, 1.2],
        [1, 1, 1],
        [1, 2, 5]
    ])

    base = np.zeros(coords.shape)
    base[:, 0:2] = coords[:, 0:2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d", proj_type="ortho")

    if stilts:
        for i, point in enumerate(coords):
            ax.plot(
                [point[0], base[i, 0]],
                [point[1], base[i, 1]],
                [point[2], base[i, 2]],
                linewidth=0.5,
                color="k"
            )

    def linear_interp(x, x0, y0, x1, y1):
        return (y0 * (x1 - x) + y1 * (x - x0)) / (x1 - x0)

    N = 10
    for i in range(N + 1):
        x = coords[2, 0] + i / N
        z1 = linear_interp(x, coords[2, 0], coords[2, 2], coords[0, 0], coords[0, 2])
        z2 = linear_interp(x, coords[3, 0], coords[3, 2], coords[1, 0], coords[1, 2])
        ax.plot(
            [x, x],
            [coords[2, 1], coords[1, 1]],
            [z1, z2],
            color="blue"
        )
        y = coords[2, 1] + 1 - i / N
        z3 = linear_interp(y, coords[2, 1], z1, coords[1, 1], z2)
        if i != 0 and i != N:
            ax.plot(
                [x], [y], [z3], "rs"
            )

    ax.plot(
        [coords[0, 0], coords[2, 0]],
        [coords[0, 1], coords[2, 1]],
        [coords[0, 2], coords[2, 2]],
        color="orange"
    )
    ax.plot(
        [coords[1, 0], coords[3, 0]],
        [coords[1, 1], coords[3, 1]],
        [coords[1, 2], coords[3, 2]],
        color="orange"
    )

    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], "ko")

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])

    if show:
        plt.show()
    else:
        plt.savefig("02_MLI_nonlinear." + FORMAT, transparent=True, bbox_inches="tight")
    plt.close()


def plot_flips(show=True):
    # points = poisson_disc_sampling(np.array([[0, 1], [0, 1]]), 0.3)
    # points = np.vstack((
    #     points,
    #     np.array([
    #         [0, 0],
    #         [1, 0],
    #         [1, 1],
    #         [0, 1]
    #     ])
    # ))
    points = np.array([
        [0.08024023, 0.77803724],
        [0.46908374, 0.43756773],
        # [0.31637726, 0.9670485],
        [0.66432975, 0.70232735],
        [0.93443921, 0.48000535],
        [0.69379516, 0.00137135],
        # [0.94668336, 0.97051398],
        [0.17497168, 0.24748033],
        # [0.37061093, 0.01419692],
        [0., 0.],
        [1.,    0.],
        [1.,         1.],
        [0.,    1.]]
    )
    print(points)
    tri = spatial.Delaunay(points)
    centroids = np.zeros((tri.simplices.shape[0], 2))
    for i in range(tri.simplices.shape[0]):
        centroids[i] = np.sum(points[tri.simplices[i]], axis=0) / 3
        # plt.plot([centroids[i,0]], [centroids[i,1]], marker=f"${i}$", color="red", markersize=20)

    tx = 0.5
    ty = 0.85
    plt.plot([centroids[12,0], tx], [centroids[12,1], ty], lw=0.5, color="blue")
    for t in ((12, 13), (13, 11), (11, 9), (9, 8), (8, 4)):
        plt.annotate(
            s="",
            xy=centroids[t[1]],
            xytext=centroids[t[0]],
            arrowprops={"arrowstyle":"simple", "facecolor":"red", "edgecolor":"red", "lw":0.2}
        )
        plt.plot([centroids[t[1],0], tx], [centroids[t[1],1], ty], lw=0.5, color="blue")

    plt.plot([tx], [ty], "P", markersize=10, color="blue")
    plt.triplot(points[:,0], points[:,1], tri.simplices, color="k")
    plt.gca().xaxis.set_ticklabels([])
    plt.gca().yaxis.set_ticklabels([])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    # plt.gca().set_aspect("equal")
    plt.axis("off")


    if show:
        plt.show()
    else:
        plt.savefig("03_DIP_sequence." + FORMAT, transparent=True, bbox_inches="tight")
    plt.close()


def plot_delaunay_progression(show=True):
    # points = poisson_disc_sampling(np.array([[0, 1], [0, 1]]), 0.15)
    # points = np.vstack((
    #     points,
    #     np.array([
    #         [0, 0],
    #         [1, 0],
    #         [1, 1],
    #         [0, 1]
    #     ])
    # ))
    points = np.array([
        [3.57234003e-01, 4.31637646e-01],
        [1.67984910e-01, 3.80754537e-01],
        [2.65785126e-01, 2.18394510e-01],
        [1.29878639e-01, 6.26606557e-01],
        [2.02056228e-01, 8.76292605e-01],
        [6.11708137e-04, 3.51533021e-01],
        [9.25090637e-02, 2.13731159e-01],
        [5.00371607e-01, 6.73627422e-01],
        [3.27153731e-01, 7.33711973e-01],
        [6.45639360e-01, 4.63389234e-01],
        [4.39023529e-01, 8.44898957e-01],
        [2.37528253e-02, 9.63019263e-01],
        [6.66754892e-01, 7.74934230e-01],
        [4.74481663e-02, 7.54709681e-01],
        [3.78703076e-01, 1.13879985e-01],
        [2.96342003e-01, 5.79883693e-01],
        [6.52914020e-01, 9.28483669e-01],
        [7.36940714e-01, 2.35050872e-01],
        [4.50434154e-01, 9.98981977e-01],
        [2.12255691e-01, 6.35394895e-02],
        [4.65338868e-01, 3.26398236e-01],
        [5.64260294e-01, 1.67391972e-01],
        [7.06268945e-01, 8.34522432e-02],
        [4.96731088e-01, 5.12560687e-01],
        [9.01539024e-01, 4.69737014e-01],
        [8.13737146e-01, 8.59235537e-01],
        [7.74063010e-01, 6.65240178e-01],
        [9.94603273e-01, 8.22416337e-01],
        [9.53194241e-01, 2.19516108e-01],
        [5.43523680e-01, 1.54273449e-02],
        [9.91964321e-01, 6.40810443e-01],
        [9.45526404e-01, 2.45010453e-02],
        [9.60654107e-01, 9.83529513e-01],
        [0.00000000e+00, 0.00000000e+00],
        [1.00000000e+00, 0.00000000e+00],
        [1.00000000e+00, 1.00000000e+00],
        [0.00000000e+00, 1.00000000e+00],])
    # print(points)
    tri = spatial.Delaunay(points)
    centroids = np.zeros((tri.simplices.shape[0], 2))
    for i in range(tri.simplices.shape[0]):
        centroids[i] = np.sum(points[tri.simplices[i]], axis=0) / 3
        # plt.plot([centroids[i,0]], [centroids[i,1]], marker=f"${i}$", color="red", markersize=20)

    plt.triplot(points[:,0], points[:,1], tri.simplices, color="k")
    plt.gca().xaxis.set_ticklabels([])
    plt.gca().yaxis.set_ticklabels([])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.axis("off")

    if show:
        plt.show()
    else:
        plt.savefig("04_delaunay_progression_1." + FORMAT, transparent=True, bbox_inches="tight")
    plt.close()

    points = np.vstack((points, centroids[53:59], centroids[41:47]))

    tri = spatial.Delaunay(points)
    centroids = np.zeros((tri.simplices.shape[0], 2))
    for i in range(tri.simplices.shape[0]):
        centroids[i] = np.sum(points[tri.simplices[i]], axis=0) / 3
        # plt.plot([centroids[i,0]], [centroids[i,1]], marker=f"${i}$", color="red", markersize=20)

    plt.triplot(points[:,0], points[:,1], tri.simplices, color="k")
    plt.gca().xaxis.set_ticklabels([])
    plt.gca().yaxis.set_ticklabels([])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.axis("off")

    if show:
        plt.show()
    else:
        plt.savefig("04_delaunay_progression_2." + FORMAT, transparent=True, bbox_inches="tight")
    plt.close()

    points = np.vstack((points, centroids[48:52], centroids[70:79], centroids[85:91], centroids[81]))

    tri = spatial.Delaunay(points)
    centroids = np.zeros((tri.simplices.shape[0], 2))
    for i in range(tri.simplices.shape[0]):
        centroids[i] = np.sum(points[tri.simplices[i]], axis=0) / 3
        # plt.plot([centroids[i,0]], [centroids[i,1]], marker=f"${i}$", color="red", markersize=20)

    plt.triplot(points[:,0], points[:,1], tri.simplices, color="k")
    plt.gca().xaxis.set_ticklabels([])
    plt.gca().yaxis.set_ticklabels([])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.axis("off")

    if show:
        plt.show()
    else:
        plt.savefig("04_delaunay_progression_3." + FORMAT, transparent=True, bbox_inches="tight")
    plt.close()


def plot_bary_guide(show=True):
    points = np.array([
        [1, 1],
        [5, 3],
        [2, 5],
        [1, 1]
    ])

    x1, y1, x2, y2, x3, y3 = points.flatten()[:-2]

    def get_bary(x, y, rnd=True):
        l1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))
        l2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))
        l3 = 1 - l1 - l2
        if rnd:
            l1 = round(l1, 2)
            l2 = round(l2, 2)
            l3 = round(l3, 2)
        return l1, l2, l3

    def plot_and_annotate(p, y=None):
        if y:
            p = [p, y]
        plt.plot([p[0]], [p[1]], "o", color="red")
        plt.annotate(get_bary(p[0], p[1]), (p[0], p[1]), xytext=(p[0], p[1]+0.1), ha="center", fontsize=15)

    center = np.sum(points[:-1], axis=0) / 3

    plt.plot(points[:,0], points[:,1], color="k")
    plot_and_annotate(center)
    plot_and_annotate(2.2, 2)
    plot_and_annotate(2, 5)
    plot_and_annotate(4, 4.5)

    plt.plot([3], [2], "o", color="red")
    plt.annotate(get_bary(3, 2), (3, 2), xytext=(3.5, 2 - 0.25), ha="center", fontsize=15)

    plt.plot([1], [1], "o", color="red")
    plt.annotate(get_bary(1, 1), (1, 1), xytext=(1, 1 - 0.25), ha="center", fontsize=15)

    for p in points[:-1]:
        plt.plot()

    plt.gca().xaxis.set_ticklabels([])
    plt.gca().yaxis.set_ticklabels([])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.axis("off")

    if show:
        plt.show()
    else:
        plt.savefig("05_bary_guide." + FORMAT, transparent=True, bbox_inches="tight")
    plt.close()


def plot_complexities(show=True):
    def exp(x, a):
        return a * 2**x

    def cube(x, a, b):
        return a * x**3 + b

    # ms for 10k interps
    mesh2 = np.array([38.302, 31.095, 31.233, 31.244, 31.387, 31.359, 31.159, 31.369, 31.199, 31.252])
    mesh3 = np.array([122, 114, 114, 115, 114, 113, 113, 114, 113, 113])
    mesh4 = np.array([399, 395, 396, 392, 398, 396, 395, 396, 395, 396])

    grid2 = np.array([6.421, 5.373, 5.286, 5.445, 5.290, 5.296, 5.452, 5.287, 5.289, 5.290])

    p_mesh = [mesh2.mean(), mesh3.mean(), mesh4.mean()]
    std_mesh = [mesh2.std(), mesh3.std(), mesh4.std()]

    m_opt, pcov = optimize.curve_fit(cube, [2, 3, 4], p_mesh)

    plt.errorbar([2, 3, 4], p_mesh, yerr=std_mesh)
    plt.plot(
        np.linspace(2, 4.1, 100),
        cube(np.linspace(2, 4.1, 100), *m_opt)
    )

    plt.gca().xaxis.set_ticklabels([2, 3, 4])
    # plt.gca().yaxis.set_ticklabels([])
    plt.gca().set_xticks([2, 3, 4])
    # plt.gca().set_yticks([])

    if show:
        plt.show()
    else:
        plt.savefig("06_time_complexities." + FORMAT, transparent=True, bbox_inches="tight")
    plt.close()


def plot_cumsum(show=True, plot_errs=True):
    GRID_DATA = "../gasoline_header2_grid/grid_gasoline_header2.csv"
    RAND_DATA = "../gasoline_header2_random/random_gasoline_header2.csv"
    DEL_DATA = "../run37_gasoline_z0_header2/z0.0.points"
    # DEL_DATA = "../run55_run37_10k/z0.0.points"
    # DEL_DATA = "run39_gasoline_z0_header2_extended2/z0.0.points"

    gridpoints = pd.read_csv(GRID_DATA, delimiter=",")
    delpoints = pd.read_csv(DEL_DATA, delimiter=",")
    randpoints = pd.read_csv(RAND_DATA, delimiter=",")

    # randpoints = randpoints.loc[(randpoints["T"] > 3.5) & (randpoints["T"] < 5)]
    # randpoints = randpoints.loc[(randpoints["nH"] < -4)]

    delpoints = delpoints.sample(15000)

    print(gridpoints["T"].min(), gridpoints["T"].max())
    print(delpoints["T"].min(), delpoints["T"].max())
    print(randpoints["T"].min(), randpoints["T"].max())

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
    N = delpoints.shape[1]-1
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

    # Plotting
    plt.figure()
    plt.title(r"Cumulative Sum of Errors, $n_H < -4$")
    plt.axvline(delmedian, c="b", linestyle="--")
    plt.axvline(delavg, c="b", linestyle=":")

    plt.axvline(gridmedian, c="orange", linestyle="--")
    plt.axvline(gridavg, c="orange", linestyle=":")

    plt.axhline(0.5, c="k", lw=0.5)

    X3 = np.sort(randpoints["diff_del"].abs().dropna())
    plt.plot(X3, np.arange(X3.shape[0]) / X3.shape[0], label="Delaunay Mesh Interpolation", c="b")
    X2 = np.sort(randpoints["diff_grid"].abs().dropna())
    plt.plot(X2, np.arange(X2.shape[0]) / X2.shape[0], label="Multilinear Grid Interpolation", c="orange")

    legend_elements = [
        Line2D([0], [0], color="b", label="Delaunay Mesh Interpolation"),
        Line2D([0], [0], color="orange", label="Multilinear Grid Interpolation"),
        Line2D([0], [0], color="k", linestyle="--", label="Median"),
        Line2D([0], [0], color="k", linestyle=":", label="Mean")
    ]
    plt.legend(handles=legend_elements)
    plt.ylabel("Normalized cumulative sum")
    plt.xlabel(r"$|\Lambda - \Lambda_{\mathrm{interpolated}}|$ in dex")
    plt.xlim(-0.05, 0.5)

    print("Delmedian\t", round(delmedian, 4))
    print("Delavg\t\t", round(delavg, 4))
    print("Gridmedian\t", round(gridmedian, 4))
    print("Gridavg\t\t", round(gridavg, 4))
    print("Delmedian / Gridmedian", round(delmedian / gridmedian, 4))
    print("Delavg / Gridavg\t", round(delavg / gridavg, 4))
    print("Gridmedian / Delmedian", round(gridmedian / delmedian, 4))
    print("Gridavg / Delavg\t", round(gridavg / delavg, 4))
    print("Delmaxerror\t", delmaxdiff)
    print("Gridmaxerror\t", gridmaxdiff)

    if show:
        plt.show()
    else:
        plt.savefig("07b_cumsum." + FORMAT, transparent=True, bbox_inches="tight")
    plt.close()



def plot_flip_distribution(show=True):
    dist2 = []

    with open("../flip_distribution/mesh2", "r") as f:
        for line in f:
            line = line.replace("\n", "")
            if len(line) == 1:
                flips = int(line)
                if flips >= 0 and flips >= 0:
                    dist2.append(flips)
    dist3 = []

    with open("../flip_distribution/mesh3", "r") as f:
        for line in f:
            line = line.replace("\n", "")
            if len(line) == 1:
                flips = int(line)
                if flips >= 0 and flips >= 0:
                    dist3.append(flips)
    dist4 = []

    with open("../flip_distribution/mesh4", "r") as f:
        for line in f:
            line = line.replace("\n", "")
            if len(line) == 1:
                flips = int(line)
                if flips >= 0 and flips >= 0:
                    dist4.append(flips)

    plt.hist(dist2, bins=np.linspace(-0.5, 10.5, 12))
    plt.xlabel("Number of flips")
    plt.ylabel("Count")
    plt.title("2D")
    plt.xticks(np.linspace(0, 10, 11))
    plt.ylim(0, 6000)
    if show:
        plt.show()
    else:
        plt.savefig("10a_flips_dist_2D." + FORMAT, transparent=True, bbox_inches="tight")

    plt.close()

    plt.hist(dist3, bins=np.linspace(-0.5, 10.5, 12))
    plt.xlabel("Number of flips")
    plt.ylabel("Count")
    plt.title("3D")
    plt.xticks(np.linspace(0, 10, 11))
    plt.ylim(0, 6000)
    if show:
        plt.show()
    else:
        plt.savefig("10b_flips_dist_3D." + FORMAT, transparent=True, bbox_inches="tight")

    plt.close()

    plt.hist(dist4, bins=np.linspace(-0.5, 10.5, 12))
    plt.xlabel("Number of flips")
    plt.ylabel("Count")
    plt.title("4D")
    plt.xticks(np.linspace(0, 10, 11))
    plt.ylim(0, 6000)
    if show:
        plt.show()
    else:
        plt.savefig("10c_flips_dist_4D." + FORMAT, transparent=True, bbox_inches="tight")

    plt.close()


def plot_cooling_function():
    cloudy_cool = np.zeros(71)
    cloudy_heat = np.zeros(71)
    for i, T in enumerate(np.linspace(2, 9, 71)):
        print(np.loadtxt("../coolfct/T_" + str(T) + ".cool", usecols=3))
        cloudy_cool[i] = np.log10(np.loadtxt("../coolfct/T_" + str(T) + ".cool", usecols=3))
        cloudy_heat[i] = np.log10(np.loadtxt("../coolfct/T_" + str(T) + ".cool", usecols=2))

    DEL_DATA = "../run45_gadget/z0.0.points"

    delpoints = pd.read_csv(DEL_DATA, delimiter=",")

    delpoints = delpoints[["T", "nH", "Z", "values"]]

    # the coordinates at which we will interpolate
    interp_coords = np.zeros((70, 3))
    interp_coords[:,0] = np.linspace(2, 8.9, 70)
    interp_coords[:,1] = 0
    interp_coords[:,2] = -3

    # Delaunay interpolation
    N = delpoints.shape[1]-1
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

    plt.plot(np.linspace(2, 9, 71), cloudy_cool, label="cloudy")
    plt.plot(np.linspace(2, 8.9, 70), vals, label="interp")
    plt.legend()
    plt.show()


def plot_bary_plane(show=True):

    coords = np.array([
        [1.2, 1.3],
        [1.6, 1.8],
        [1.8, 1.2]
    ])

    vals = np.array([2, 1.2, 1])

    x1, y1, x2, y2, x3, y3 = coords.flatten()

    def get_bary(x, y, rnd=False):
        l1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))
        l2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))
        l3 = 1 - l1 - l2
        if rnd:
            l1 = round(l1, 2)
            l2 = round(l2, 2)
            l3 = round(l3, 2)
        return l1, l2, l3

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")#, proj_type="ortho")

    for i, point in enumerate(coords):
        ax.plot(
            [point[0], point[0]],
            [point[1], point[1]],
            [0, vals[i]],
            linewidth=0.5,
            color="k"
        )

    ax.plot(
        list(coords[:,0]) + [coords[0,0]],
        list(coords[:,1]) + [coords[0,1]],
        4*[0],
        color="k"
    )

    ax.plot(
        list(coords[:,0]) + [coords[0,0]],
        list(coords[:,1]) + [coords[0,1]],
        list(vals) + [vals[0]],
        color="blue"
    )

    for i in np.linspace(1.1, 1.9, 10):
        for j in np.linspace(1.1, 1.9, 10):
            bary = get_bary(i, j)
            ax.plot(
                [i],
                [j],
                [sum([bary[i] * vals[i] for i in range(len(bary))])],
                color="orange",
                marker="."
            )

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])

    ax.set_xlim(1, 2)
    ax.set_ylim(1, 2)
    ax.set_zlim(0, 2)

    # ax.azim = -30
    # ax.elev = 30
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$f(x, y)$")

    if show:
        plt.show()
    else:
        plt.savefig("12_bary_plane." + FORMAT, transparent=True, bbox_inches="tight")
    plt.close()



def plot_sf_guide(show=True):

    points = np.array([
        [1, 1.],
        [5, 3],
        [2, 5],
        [1, 1]
    ])
    points[:,0] *= 4/3.
    points *= 0.8

    interpoint = np.array([1.7, 4.5])

    fig = plt.figure()
    plt.plot(points[:,0], points[:,1], "k")

    centroid = np.sum(points[:-1], axis=0) / 3
    plt.plot([centroid[0]], centroid[1], "bo", label="Centroid")

    midpoints = np.array([
        (points[0] + points[1]) / 2,
        (points[1] + points[2]) / 2,
        (points[2] + points[0]) / 2,
    ])

    rot1 = np.array([
        [0, -1],
        [1, 0]
    ])
    rot2 = np.array([
        [0, 1],
        [-1, 0]
    ])

    n1 = points[1] - points[0]
    n1 = np.matmul(rot2, n1) / np.sqrt(np.sum(n1 ** 2))
    n2 = points[2] - points[1]
    n2 = np.matmul(rot2, n2) / np.sqrt(np.sum(n2 ** 2))
    n3 = points[0] - points[2]
    n3 = np.matmul(rot2, n3) / np.sqrt(np.sum(n3 ** 2))

    for i, n in enumerate([n1, n2, n3]):
        plt.annotate(
            s="",
            xy=midpoints[i] + n,
            xytext=midpoints[i],
            xycoords="data",
            arrowprops={"arrowstyle":"simple", "facecolor":"orange", "edgecolor":"orange", "lw":0.2},
            annotation_clip=False,
            zorder=-100
        )

    plt.annotate(s=r"$n_1$", xy=midpoints[0] + n1, xytext=midpoints[0] + n1 + np.array([0.05, 0.0]), color="orange", size=12)
    plt.annotate(s=r"$n_2$", xy=midpoints[1] + n2, xytext=midpoints[1] + n2 + np.array([0.0, 0.0]), color="orange", size=12)
    plt.annotate(s=r"$n_3$", xy=midpoints[2] + n3, xytext=midpoints[2] + n3 + np.array([0, 0.05]), color="orange", size=12, annotation_clip=False)

    d1 = midpoints[1] - interpoint
    d2 = midpoints[2] - interpoint
    # plt.annotate(s=r"$d_1$", xy=midpoints[1], xytext=midpoints[1] - 0.5 * d1 + np.array([0.05, -0.1]), color="k", size=12)

    p1 = -(d1[0] * n2[0] + d1[1] * n2[1]) * n2
    p2 = -(d2[0] * n3[0] + d2[1] * n3[1]) * n3

    plt.annotate(
        s="",
        xy=midpoints[1] + p1,
        xytext=midpoints[1],
        xycoords="data",
        arrowprops={"arrowstyle":"simple", "facecolor":"red", "edgecolor":"red", "lw":0.2},
        annotation_clip=False,
        zorder=-90
    )
    plt.annotate(s=r"$x_2$", xy=midpoints[1], xytext=midpoints[1] + p1 + np.array([0.05, -0.1]), color="r", size=12)

    plt.annotate(
        s="",
        xy=midpoints[2] + p2,
        xytext=midpoints[2],
        xycoords="data",
        arrowprops={"arrowstyle":"simple", "facecolor":"red", "edgecolor":"red", "lw":0.2},
        annotation_clip=False,
        zorder=-90
    )
    plt.annotate(s=r"$x_3$", xy=midpoints[2], xytext=midpoints[2] + p2 + np.array([0.0, -0.2]), color="r", size=12)

    plt.plot(
        [midpoints[1,0], interpoint[0], midpoints[2,0]],
        [midpoints[1,1], interpoint[1], midpoints[2,1]],
        "k--",
        lw=0.8
    )

    plt.plot(
        [midpoints[1,0] + p1[0], interpoint[0], midpoints[2,0] + p2[0]],
        [midpoints[1,1] + p1[1], interpoint[1], midpoints[2,1] + p2[1]],
        "k:",
        lw=0.8
    )

    plt.plot(midpoints[:,0], midpoints[:,1], "bo", label="Midpoints")
    plt.annotate(s=r"$M_1$", xy=midpoints[0], xytext=midpoints[0] + np.array([-0.15, 0.1]), color="b", size=16)
    plt.annotate(s=r"$M_2$", xy=midpoints[1], xytext=midpoints[1] + np.array([-0.2, -0.25]), color="b", size=16)
    plt.annotate(s=r"$M_3$", xy=midpoints[2], xytext=midpoints[2] + np.array([0.1, -0.1]), color="b", size=16)
    plt.plot([interpoint[0]], [interpoint[1]], "bP")

    plt.gca().xaxis.set_ticklabels([])
    plt.gca().yaxis.set_ticklabels([])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.axis("off")
    plt.gca().set_aspect("equal")

    if show:
        plt.show()
    else:
        plt.savefig("13_dip_guide." + FORMAT, transparent=True, bbox_inches="tight")
    plt.close()


def plot_cooling_function_components(show=True):
    # 1 T, 2 Ctot, 3 H, 4 He, 8 C, 9 N, 10 O, 28 Fe, 41 FF_H, 42 FF_M, 43 eeff, 45 Comp
    data = np.loadtxt("../lingrid/lingrid.cool_by_element", usecols=(1, 2, 3, 4, 8, 9, 10, 28, 41, 42, 43, 45))

    lw = 0.9
    plt.plot(data[:,0], data[:,1], label=r"$\Lambda$", c="k")
    plt.plot(data[:,0], data[:,2] + data[:,8], label=r"H", lw=lw)
    plt.plot(data[:,0], data[:,3], label=r"He", lw=lw)
    plt.plot(data[:,0], data[:,4], label=r"C", lw=lw)
    plt.plot(data[:,0], data[:,5], label=r"N", lw=lw)
    plt.plot(data[:,0], data[:,6], label=r"O", lw=lw)
    plt.plot(data[:,0], data[:,7], label=r"Fe", lw=lw)
    plt.plot(data[:,0], data[:,8], label=r"$\mathrm{FF}_\mathrm{H}$", ls="--", lw=lw)
    # plt.plot(data[:,0], data[:,9], label=r"FF_M")
    # plt.plot(data[:,0], data[:,10], label=r"ee_ff")
    # plt.plot(data[:,0], data[:,11], label=r"Compton")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"Temperature $T$ in K")
    plt.ylabel(r"Cooling in $\frac{\mathrm{erg}}{\mathrm{cm}^3\mathrm{s}}$")
    plt.ylim(1e-25)
    plt.legend(loc="lower right", framealpha=1)

    if show:
        plt.show()
    else:
        plt.savefig("14_coolfct_components." + FORMAT, transparent=True, bbox_inches="tight")
    plt.close()


def plot_sample_distribution(show=True):
    DEL_DATA = "../run37_gasoline_z0_header2/z0.0.points"
    data = pd.read_csv(DEL_DATA).sample(5000)

    # sns.set_style("ticks")
    grid = sns.pairplot(
        data,
        diag_kind="hist",
        vars=["T", "nH", "old", "SFR"],
        markers="o",
        plot_kws={
            "s":1,
            "marker":"o",
            "edgecolor":None
        },
        diag_kws={
            "bins":50,
            "lw":0
        },
        height=2.5
    )

    if show:
        plt.show()
    else:
        plt.savefig("15_sample_distribution." + FORMAT, transparent=True, bbox_inches="tight")
    plt.close()


def plot_radiation_fields(show=True):
    HM12 = np.loadtxt("HM12.cont", usecols=(0, 1))

    hPlanck = 6.626196e-27  # erg s
    ryd = 2.1798723611e-11 # ergs

    old = np.loadtxt("../spectra/old")
    SFR = np.loadtxt("../spectra/SFR")
    HM12[:,1] = HM12[:,1] / (HM12[:,0] * ryd / (hPlanck))
    HM12[:,1] = HM12[:,1] / hPlanck
    plt.plot(HM12[:,0], HM12[:,1], label=r"HM12 at $z = 0$")

    norm_HM12 = np.argmin(np.abs(HM12[:,0] - 1))
    norm_old = np.argmin(np.abs(old[:,0] - 1))
    norm_SFR = np.argmin(np.abs(SFR[:,0] - 1))

    old[:,1] = old[:,1] / hPlanck
    SFR[:,1] = SFR[:,1] / hPlanck

    T_old = HM12[norm_HM12,1] / old[norm_old,1]
    T_SFR = HM12[norm_HM12,1] / SFR[norm_SFR,1]

    plt.plot(old[:,0], T_old * old[:,1], label=r"$T_{old} = 10^{" + str(round(np.log10(T_old), 2)) + "}$")
    plt.plot(SFR[:,0], T_SFR * SFR[:,1], label=r"$T_{SFR} = 10^{" + str(round(np.log10(T_SFR), 2)) + "}$")

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel(r"$E$ in Ryd")
    plt.ylabel(r"$4\pi J_\nu/h$ in $\frac{\mathrm{photons}}{\mathrm{s}\cdot\mathrm{cm}^2}$")
    plt.legend()
    plt.xlim(1e-4, 1e4)
    plt.ylim(1e-4, 1e9)
    plt.axvline(1, ls=":", c="k")

    if show:
        plt.show()
    else:
        plt.savefig("16_rad_fields." + FORMAT, transparent=True, bbox_inches="tight")
    plt.close()

    HM12_0 = np.loadtxt("ch4fig2/cloudy_0_HM12.cool", usecols=(1, 3))
    Told_0 = np.loadtxt("ch4fig2/cloudy_0_Told.cool", usecols=(1, 3))
    TSFR_0 = np.loadtxt("ch4fig2/cloudy_0_TSFR.cool", usecols=(1, 3))
    HM12_2 = np.loadtxt("ch4fig2/cloudy_-2_HM12.cool", usecols=(1, 3))
    Told_2 = np.loadtxt("ch4fig2/cloudy_-2_Told.cool", usecols=(1, 3))
    TSFR_2 = np.loadtxt("ch4fig2/cloudy_-2_TSFR.cool", usecols=(1, 3))

    plt.plot(HM12_0[:,0], HM12_0[:,1], c="blue", label=r"HM12, $n_H = 1 {\rm cm}^{-3}$")
    plt.plot(HM12_2[:,0], HM12_2[:,1], c="tab:blue", label=r"HM12, $n_H = 10^{-2} {\rm cm}^{-3}$")

    plt.plot(Told_0[:,0], Told_0[:,1], c="tab:orange", label=r"$T_{\rm old}$, $n_H = 1 {\rm cm}^{-3}$")
    plt.plot(Told_2[:,0], Told_2[:,1], c="orange", label=r"$T_{\rm old}$, $n_H = 10^{-2} {\rm cm}^{-3}$")

    plt.plot(TSFR_0[:,0], TSFR_0[:,1], c="green", label=r"$T_{\rm SFR}$, $n_H = 1 {\rm cm}^{-3}$")
    plt.plot(TSFR_2[:,0], TSFR_2[:,1], c="tab:green", label=r"$T_{\rm SFR}$, $n_H = 10^{-2} {\rm cm}^{-3}$")

    plt.xlabel(r"$T$ in K")
    plt.ylabel(r"$\Lambda$ in $\frac{\rm erg}{{\rm cm}^3 {\rm s}}$")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()

    if show:
        plt.show()
    else:
        plt.savefig("16b_coolfcts." + FORMAT, transparent=True, bbox_inches="tight")
    plt.close()


def plot_edgeflip(show=True):
    points = np.array([
        [1, 1],
        [3, 4],
        [5, 1],
        [6, 3],
        [3, 2]
    ])

    plt.plot(points[:-1,0], points[:-1,1], "k")
    plt.plot(
        [points[3,0], points[1,0], points[0,0], points[2,0]],
        [points[3,1], points[1,1], points[0,1], points[2,1]],
        "k"
    )
    plt.plot(
        [points[0,0], points[-1,0], points[1,0]],
        [points[0,1], points[-1,1], points[1,1]],
        "k--"
    )
    plt.plot(
        [points[-1,0], points[2,0]],
        [points[-1,1], points[2,1]],
        "k--"
    )
    plt.plot(points[:-1,0], points[:-1,1], "ko")
    plt.plot([points[-1,0]], [points[-1,1]], "rs")

    plt.gca().add_patch(
        Circle(
            (4.75, 3),
            np.sqrt(4.0625),
            edgecolor="blue",
            fill=False
        )
    )

    plt.annotate(
        s=r"A",
        xy=points[3],
        xytext=points[3] + np.array([0.15, 0.0]),
        color="k",
        size=16
    )
    plt.annotate(
        s=r"P",
        xy=points[-1],
        xytext=points[-1] + np.array([0.15, 0.0]),
        color="r",
        size=16
    )
    plt.annotate(
        s=r"x",
        xy=0.5 * (points[1] + points[2]),
        xytext=0.5 * (points[1] + points[2]) + np.array([0.1, 0.0]),
        color="k",
        size=16
    )
    plt.gca().set_aspect("equal")
    plt.gca().xaxis.set_ticklabels([])
    plt.gca().yaxis.set_ticklabels([])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.axis("off")


    if show:
        plt.show()
    else:
        plt.savefig("17a_edgeflip." + FORMAT, transparent=True, bbox_inches="tight")
    plt.close()



    plt.plot(
        [points[3,0], points[1,0], points[0,0], points[2,0]],
        [points[3,1], points[1,1], points[0,1], points[2,1]],
        "k"
    )
    plt.plot(
        [points[0,0], points[-1,0], points[1,0]],
        [points[0,1], points[-1,1], points[1,1]],
        "k"
    )
    plt.plot(
        [points[-1,0], points[2,0]],
        [points[-1,1], points[2,1]],
        "k"
    )
    plt.plot(
        [points[2,0], points[3,0]],
        [points[2,1], points[3,1]],
        "k"
    )
    plt.plot(
        [points[2,0], points[1,0]],
        [points[2,1], points[1,1]],
        "grey",
        lw=0.8
    )
    plt.plot(
        [points[-1,0], points[3,0]],
        [points[-1,1], points[3,1]],
        "k--"
    )
    plt.plot(points[:-1,0], points[:-1,1], "ko")
    plt.plot([points[-1,0]], [points[-1,1]], "rs")

    plt.annotate(
        s=r"A",
        xy=points[3],
        xytext=points[3] + np.array([0.15, 0.0]),
        color="k",
        size=16
    )
    plt.annotate(
        s=r"P",
        xy=points[-1],
        xytext=points[-1] + np.array([0.1, 0.1]),
        color="r",
        size=16
    )
    plt.annotate(
        s=r"x",
        xy=0.5 * (points[1] + points[2]),
        xytext=0.5 * (points[1] + points[2]) + np.array([-0.05, 0.2]),
        color="grey",
        size=16
    )
    plt.annotate(
        s=r"y",
        xy=0.5 * (points[-1] + points[3]),
        xytext=0.5 * (points[-1] + points[3]) + np.array([0.0, -0.2]),
        color="k",
        size=16
    )

    plt.gca().add_patch(
        Circle(
            (4.33333333, 3),
            np.sqrt(2.7777777777),
            edgecolor="blue",
            fill=False
        )
    )

    plt.gca().add_patch(
        Circle(
            (4.5, 2.5),
            np.sqrt(2.5),
            edgecolor="blue",
            fill=False
        )
    )


    plt.gca().set_aspect("equal")
    plt.gca().xaxis.set_ticklabels([])
    plt.gca().yaxis.set_ticklabels([])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.axis("off")

    if show:
        plt.show()
    else:
        plt.savefig("17b_edgeflip." + FORMAT, transparent=True, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    show = True
    # plot_mli_nonlinear(show=show)
    # plot_mli_cube(show=show)
    # plot_flips(show=show)
    # plot_delaunay_progression(show=show)
    # plot_bary_guide(show=show)
    # plot_complexities(show=show)
    plot_cumsum(show=show, plot_errs=False)
    # plot_flip_distribution(show=show)
    # plot_cooling_function()
    # plot_bary_plane(show)
    # plot_sf_guide(show)
    # plot_cooling_function_components(show)
    # plot_sample_distribution(show)
    # plot_radiation_fields(show)
    # plot_edgeflip(show)
