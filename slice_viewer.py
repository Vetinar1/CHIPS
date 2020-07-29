import pandas as pd
import seaborn as sns
from cloudy_optimizer import *
import matplotlib as mpl
import matplotlib.pyplot as plt

sns.set()

def get_ndim_linear_interp(point1, point2):
    point1 = np.array(point1)
    point2 = np.array(point2)

    def ndim_linear_interp(t):
        """

        :param t:       in [0, 1]
        :return:
        """
        return np.array([point1 + t * (point2 - point1) for t in list(t)])

    return ndim_linear_interp


def view_slice(points, begin, end, steps, filename=None, title=None):
    """
    Interpolate steps points on a line between begin coordinates and end coordinates. Plot results.

    :param points:      Points to use for interpolation
    :param begin:       Start coordinates of slice to interpolate over. List/Iterable.
    :param end:         End coordinates of slice to interpolate over. List/Iterable.
    :param steps:       Number of points to evenly distribute over slice for interpolation. Int.
    :param filename:    If given, save plot to file instead of displaying it directly.
    :return:
    """

    # coord interpolator gets the coordinates along the line via linear interpolation
    coord_interpolator = get_ndim_linear_interp(begin, end)

    # interp coordinates are these coordinates along the line
    interp_coordinates = coord_interpolator(np.linspace(0, 1, steps))

    # use the delaunay interpolator to get the values at the interp coordinates
    interpolated = interpolate_delaunay(
        points,
        interp_coordinates
    )[0]

    plt.plot(np.linspace(0, 1, steps), interpolated[:,-1])
    if title:
        plt.title(title)
    plt.xlabel("x: " + str(begin) + " to " + str(end))
    plt.ylabel("Ctot")
    plt.yscale("log")
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    print("Loading...")
    #points = load_all_points_from_cache(3, cache_folder="run5/cache/")

    cool = np.loadtxt("forplot.cool", usecols=[1, 3])
    plt.plot(cool[:,0], cool[:,1])
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel(r"$\log T$")
    plt.ylabel(r"$\log \Lambda$")
    plt.title("Cooling function as calculated by cloudy - 0.1 dex steps in T")
    plt.show()

    exit()
    df_points = pd.read_csv(
        "run5/run5.csv"
    )

    for i in range(-3, 3):
        view_slice(
            df_points.values,
            [2.01, i, -1],
            [5.99, i, -1],
            100,
            title=r"$n_H = " + str(i) + "$, $Z=-1$",
            filename="n_H_" + str(i)
        )

    exit()

    print("Converting...")
    # df_points = pd.DataFrame(
    #     points,
    #     columns=["Temperature", "n_H", "Metallicity", "Value"]
    # )

    df_points["Value"] = np.log10(df_points["Value"])
    print(df_points)

    print("Plotting...")
    plotted_points = df_points.sample(10000)

    grid = sns.PairGrid(plotted_points, vars=["Temperature", "n_H", "Metallicity"], height=4)
    grid.map_diag(sns.distplot, bins=50, hist=True, kde=False, rug=False)
    grid.map_offdiag(
        sns.scatterplot,
        hue=df_points["Value"],
        markers=".",
        edgecolor=None,
        s=0.5
    )

    cmap = sns.cubehelix_palette(as_cmap=True)
    grid.fig.colorbar(
        mpl.cm.ScalarMappable(
            cmap=cmap,
            norm=mpl.colors.Normalize(plotted_points["Value"].min(), plotted_points["Value"].max())
        ),
        ax=grid.axes[1,1], shrink=0.8, panchor=(1,1)
    )

    plt.show()

    exit()
    grid = sns.pairplot(
        plotted_points,
        #hue="Value",
        diag_kind="hist",
        vars=["Temperature", "n_H", "Metallicity"],
        markers=".",
        plot_kws={
            "s":3,
            "marker":".",
            "edgecolor":None,
            "c":plotted_points["Value"]
        },
        diag_kws={
            "bins":50
        },
        height=3
    )
    print(len(df_points.index))
    # grid.savefig("test.png")
    plt.show()