import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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


if __name__ == "__main__":
    show = False
    plot_mli_nonlinear(show=show)
    plot_mli_cube(show=show)