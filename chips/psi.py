import numba
import numpy as np


@numba.jit(nopython=True)
def dot(vec1, vec2):
    return np.sum(vec1 * vec2)


@numba.jit(nopython=True)
def build_simplex(neighbors, target):
    """
    Numba friendly implementation of the projective simplex construction algorithm described in
    https://arxiv.org/abs/2109.13926

    :param neighbors:   2D numpy array of points
    :param target:      1D numpy array containing target point
    :return:            Numpy array with indices of simplex in neighbors or None if no simplex could be built
    """
    D = neighbors.shape[1]

    n_indices = np.arange(neighbors.shape[0], dtype=np.intc)
    simplex   = np.zeros(D+1, dtype=np.intc)
    simplex_i = 0 # current vertex to find

    pneighbors = np.copy(neighbors)
    ptarget    = np.copy(target)

    it = D
    failflag = False
    while it > 1:
        # Find nearest neighbor in projective space
        min_dist2 = np.inf
        pnni     = -1       # projected nearest neighbor index
        if it == D:
            pnni = 0
        else:
            for i in range(pneighbors.shape[0]):
                if n_indices[i] == -1:
                    continue
                # dist2 = np.power(ptarget - pneighbors[i], 2)
                dist2 = dot(ptarget - pneighbors[i], ptarget - pneighbors[i])
                if dist2 < min_dist2:
                    min_dist2 = dist2
                    pnni = i

        # remove point so we won't find it at nearest neighbor with distance 0 later
        n_indices[pnni] = -1

        # The closest point must be part of the solution
        simplex[simplex_i] = pnni
        simplex_i += 1

        pnn = pneighbors[pnni]      # nearest neighbor
        # normal vector that we will use to build next projective space
        pnnvec = pnn - ptarget
        pnnvec2 = dot(pnnvec, pnnvec)

        # project
        for i in range(pneighbors.shape[0]):
            if n_indices[i] == -1:
                continue

            # projection on pnnvec
            pn = dot(pneighbors[i], pnnvec) * pnnvec / pnnvec2
            # projection on pnnvec shifted to origin
            pnvert = pn - ptarget

            # throw out half the points (on the wrong side of the projection plane)
            if dot(pnvert, pnnvec) > 0:
                n_indices[i] = -1
                continue

            # project onto plane
            pneighbors[i] = pneighbors[i] - pn

        # print(n_indices)
        # Verify we still have enough neighbors to continue next iteration Last step (line) requires at least 2
        if pneighbors.shape[0] - (n_indices == -1).sum() < 1 + it:
            failflag = True
            break

        # update pneighbors, ptarget
        ptarget = ptarget - dot(ptarget, pnnvec) * pnnvec / pnnvec2
        it -= 1

    if failflag:
        return None

    # all pneighbors are now on a line
    # find nearest neighbors in both directions of the line
    min_dist2 = np.inf
    pnni     = -1       # projected nearest neighbor index
    for i in range(pneighbors.shape[0]):
        if n_indices[i] == -1:
            continue
        # dist2 = np.power(ptarget - pneighbors[i], 2)
        dist2 = dot(ptarget - pneighbors[i], ptarget - pneighbors[i])
        if dist2 < min_dist2:
            min_dist2 = dist2
            pnni = i

    # first nearest neighbor
    pnn = pneighbors[pnni]
    simplex[simplex_i] = pnni
    simplex_i += 1
    # points from target to nn
    pnnvec = pnn - ptarget

    # find nearest neighbor in other direction
    best_nn2_ind = None
    best_dist2 = np.inf
    for i in range(pneighbors.shape[0]):
        if n_indices[i] == -1:
            continue

        # points from target to neigh
        diffvec = pneighbors[i] - ptarget

        # point in same direction?
        if dot(pnnvec, diffvec) > 0:
            continue

        dist2 = dot(diffvec, diffvec)
        if dist2 < best_dist2:
            best_dist2 = dist2
            best_nn2_ind = i

    if best_dist2 == np.inf:
        return None
    else:
        simplex[simplex_i] = best_nn2_ind

    return simplex


def build_simplex_adaptive(points, target, tree, k, factor, max_steps):
    """
    Adaptive interface for build_simplex. Automatically increases k if the algorithm fails.

    As a convenience feature for sample_step_psi() it automatically ignores nearest neighbors that
    are within an epsilon (1e-8). Doing this here avoids some unnecessary array copies.

    :param points:      2D numpy array of points
    :param target:      1D numpy array containing target point
    :param tree:        scipy KDTree object
    :param k:           Number of nearest neighbors to find initially
    :param factor:      Factor to increase k if algorithm fails
    :param max_steps:   Number of times to retry before giving up
    :return:            Numpy array with indices of simplex in points or None if no simplex could be built
    """
    break_early = False
    dists, neighbor_indices = tree.query(target, k)
    if dists[0] < 1e-8: # epsilon
        # one dropped value shouldnt matter except at *really* small k...
        neighbor_indices = neighbor_indices[1:]
    if neighbor_indices.shape[0] > points.shape[0]:
        neighbor_indices = neighbor_indices[:points.shape[0]-1]
        break_early = True # no point iterating if we are already at max k
    neighbors = points[neighbor_indices]

    simplex = build_simplex(neighbors, target)

    steps = 0
    while simplex is None and steps < max_steps:
        if break_early:
            break

        k *= factor
        steps += 1
        dists, neighbor_indices = tree.query(target, k)
        if dists[0] < 1e-8: # epsilon
            # one dropped value shouldnt matter except at *really* small k...
            neighbor_indices = neighbor_indices[1:]
        if neighbor_indices.shape[0] > points.shape[0]:
            neighbor_indices = neighbor_indices[:points.shape[0]-1]
            break_early = True

        neighbors = points[neighbor_indices]

        simplex = build_simplex(neighbors, target)

    if simplex is not None:
        return neighbor_indices[simplex]
    else:
        return None


if __name__ == "__main__":
    from scipy.spatial import KDTree, Delaunay
    dimensions = 3
    k = 15
    N_points = 10000
    N_targets = 1000

    points = np.random.random((N_points, dimensions)) * 2
    tree = KDTree(points)

    print("Compiling")
    target = np.random.random(dimensions)
    _, neighbors = tree.query(target, k)
    neighbors = points[neighbors]

    simplex = build_simplex(neighbors, target)

    print("Running")
    invalid = 0
    wrong = 0
    for i in range(N_targets):
        target = np.random.random(dimensions) + 0.5
        _, neighbors = tree.query(target, k)
        neighbors = points[neighbors]

        simplex = build_simplex(neighbors, target)

        if simplex is None:
            invalid += 1
            continue

        # print("Simplex", simplex)
        tri = Delaunay(neighbors[simplex])

        if tri.find_simplex(target) == -1:
            wrong += 1

    print(f"Built {N_targets} simplices on {N_points} points")
    print(f"{invalid} invalid simplices")
    print(f"{wrong} simplices did not contain their targets")
    print()


    print("Running in adaptive mode")
    invalid = 0
    wrong = 0
    for i in range(N_targets):
        target = np.random.random(dimensions) + 0.5
        simplex = build_simplex_adaptive(points, target, tree, k, 2, 4)

        if simplex is None:
            invalid += 1
            continue

        tri = Delaunay(points[simplex])

        if tri.find_simplex(target) == -1:
            wrong += 1

    print(f"Built {N_targets} simplices on {N_points} points")
    print(f"{invalid} invalid simplices")
    print(f"{wrong} simplices did not contain their targets")
    print()




















