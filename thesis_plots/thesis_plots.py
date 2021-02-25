import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from chips.utils import poisson_disc_sampling
from scipy import spatial

FORMAT = "svg"

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

    ax.plot(
        [coords[0, 0], coords[2, 0]],
        [coords[0, 1], coords[2, 1]],
        [coords[0, 2], coords[2, 2]],
        color="tab:green"
    )
    ax.plot(
        [coords[1, 0], coords[3, 0]],
        [coords[1, 1], coords[3, 1]],
        [coords[1, 2], coords[3, 2]],
        color="tab:green"
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
            color="tab:blue"
        )
        y = coords[2, 1] + 1 - i / N
        z3 = linear_interp(y, coords[2, 1], z1, coords[1, 1], z2)
        ax.plot(
            [x], [y], [z3], "rs"
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
    for t in ((12, 13), (13, 11), (11, 9), (9, 8), (8, 4)):
        plt.annotate(
            s="",
            xy=centroids[t[1]],
            xytext=centroids[t[0]],
            arrowprops={"arrowstyle":"simple", "facecolor":"red", "edgecolor":"red", "lw":0.2}
        )
        plt.plot([centroids[t[1],0], tx], [centroids[t[1],1], ty], lw=0.5, color="tab:blue")

    plt.plot([tx], [ty], "P", markersize=10)
    plt.triplot(points[:,0], points[:,1], tri.simplices, color="k")
    plt.gca().xaxis.set_ticklabels([])
    plt.gca().yaxis.set_ticklabels([])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
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


if __name__ == "__main__":
    show = False
    # plot_mli_nonlinear(show=show)
    # plot_mli_cube(show=show)
    # plot_flips(show=show)
    plot_delaunay_progression(show=show)
    plot_bary_guide(show=show)