import os

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