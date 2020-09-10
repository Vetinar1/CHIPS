import os
import numpy as np
import pandas as pd
import re
import random
import seaborn as sns

def prod(iterable):
    """
    Return the product of all elements of the iterable. Analogous to sum().
    """
    prod = 1
    for i in iterable:
        prod *= i

    return prod


def get_folder_size(folder):
    """
    Returns size of folder in bytes. Does not consider subdirectories, links, etc.
    https://stackoverflow.com/a/1392549

    TODO: Consider subdirectories

    :param folder:
    :return:
    """
    return sum(os.path.getsize(f) for f in os.listdir(folder) if os.path.isfile(f))


def simple_print(var):
    """
    Custom print with no line separator, analogous to file.write().

    :param var:         Variable to be printed.
    """
    print(var, sep="")


def get_normalization_transform(min, max):
    """
    returns two functions, one to transform points in an interval linearly on [0, 1]
    and one to transform back
    """
    def transform(x):
        return (x - min) / abs(max - min)

    def inv_transform(x):
        return x * abs(max - min) + min

    return transform, inv_transform


def get_pruning_function(dims):
    """
    Returns a function to prune an array of points to an Nd cuboid.

    :param dims:    List: [[min, max], [min, max], ...]
                    len(list) <= points.shape[1]!
                    min_i < max_i!
    :return:        pruning function
    """
    def prune(points):
        mask = np.ones(points.shape[0], dtype=bool)
        for i, dim in enumerate(dims):
            mask[(points[:, i] < dim[0]) | (points[:, i] > dim[1])] = 0

        return points[mask]

    return prune


def drop_all_rows_containing_nan(array):
    """
    Returns a copy of the array that lacks all rows that originally contained a np.nan value.

    :param array:
    :return:
    """
    # https://stackoverflow.com/questions/22032668/numpy-drop-rows-with-all-nan-or-0-values
    return array[np.all(~np.isnan(array), axis=1)]


def seconds_to_human_readable(seconds):
    """
    Convert a number of seconds into a human readable format (string) like Xh Ym Zs

    :param seconds:
    :return:
    """
    hours = (seconds - (seconds % 3600)) / 3600
    seconds_no_hours = seconds % 3600
    minutes = (seconds_no_hours - (seconds_no_hours % 60)) / 60
    seconds_left = seconds_no_hours % 60

    return str(int(hours)) + "h " + str(int(minutes)) + "m " + str(int(seconds_left)) + "s"


def sample_simplex(simplex):
    """
    Function draw uniform samples from a simplex.
    https://www.researchgate.net/publication/275348534_Picking_a_Uniformly_Random_Point_from_an_Arbitrary_Simplex

    :param simplex:     The simplex to draw from. Numpy array containing the points of the simplex, shape
                        (n+1, n) for a n-dimensional simplex.
    :return:            n dimensional numpy array
    """
    n = simplex.shape[1]
    z = np.random.random(n+2)
    z[0] = 1
    z[-1] = 0

    l = np.zeros(n+2)
    l[0] = 1

    for j in range(1, n+1): # exlude first and last, they are 1 and 0
        l[j] = np.power(z[j], 1/(n+1-j))

    point = np.zeros(n)
    for i in range(1, n+2):
        point += simplex[i-1] * ( (1 - l[i]) * np.prod(l[:i]))  # simplex[i-1] because of 0 indexing

    return point





def sample_simplices(simplices):
    """
    Function to draw uniform samples in multiple simplices. For each given simplex one sample will be drawn.

    :param simplices:   3d array containing the simplices: (m, n+1, n)
                        m simplices
                        n+1 points per simplex
                        n dimensions
    :return:            (m, n) dimensional numpy array
    """
    samples = np.zeros((simplices.shape[0], simplices.shape[2]))

    for i in range(simplices.shape[0]):
        samples[i] = sample_simplex(simplices[i])

    return samples


def poisson_disc_sampling(space, r, k=30):
    """
    Sample a space using an evenly spaced, amorphous lattice. For the algorithm and some diagrams see:
    https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf

    This implementation does not use a grid/array, which means the runtime is a lot worse than O(N).
    (It needs to check each other point for neighbors, up to k times, should be O(N^2).)
    Should be alright if you don't want a huge amount of samples, but TODO: Optimize

    Uses euclidean distance. Normalize your parameter space.

    :param space:       ((min, max), (min, max), ...)
                        Extents of given space.
    :param r:           Minimum distance between samples. Samples will be at minimum r and at most 2r apart.
    :param k:           How many "tries" for new samples to use in each iteration; Default should be fine,
                        but you might want to increase it in large dimensions.
    :return:            A numpy array of samples, shape (M, N).
                        M: Number of samples (depends on r, size of your space)
                        N: Number of dimensions of space
    """
    n = space.shape[0]
    start = np.random.random(n)
    start = (space[:,1] - space[:,0]) * start + space[:,0]

    out = np.zeros((1, n))
    out[0] = start
    active_list = [start]

    while active_list:
        # Who would have thought this is the most straightforward way to draw a random sample without replacement?
        random.shuffle(active_list)
        curr = active_list.pop(0)

        # Uniform sampling on d-spheres and balls, method 22:
        # http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/

        for i in range(k):
            # Filter out points with distance <r, leaving only the [r, 2r] annulus
            for i in range(1000): # while True: without risk of infinite loop
                u = np.random.normal(0, 1, n+2)
                norm = np.sum(u**2)**0.5
                u = u/norm

                x = u[:n] * (2*r)

                if np.sqrt(np.sum(x**2)) >= r:
                    break

            x += curr

            outofbounds = False
            for i, limits in enumerate(space):
                if x[i] < limits[0] or x[i] > limits[1]:
                    outofbounds = True

            if outofbounds:
                continue

            x  = np.reshape(x, (1, x.shape[0]))

            # See if its too close to any nearby points
            # distances between x and all other points
            distances = np.sum(
                (out - np.repeat(x, out.shape[0], axis=0))**2,
                axis=1
            )

            if min(distances) < r**2:
                continue

            out = np.vstack((out, x))
            active_list.append(x.flatten())
            active_list.append(curr)
            break

    return out

if __name__ == "__main__":
    points = poisson_disc_sampling(
        np.array([[0, 1]] * 5),
        0.2,
        k=30
    )

    print(points.shape)

    import matplotlib.pyplot as plt
    plt.plot(points[:,0], points[:,1], "ko")
    plt.gca().set_aspect(1)
    plt.show()
    exit()

    points = pd.DataFrame(points)
    print(len(points.index))

    grid = sns.pairplot(
        points,
        diag_kind="hist",
        height=6
    )

    grid.savefig("test.png")

