import numba
import numpy as np
import matplotlib.pyplot as plt


@numba.jit(nopython=True, parallel=True)
def dot(vec1, vec2):
    return np.sum(vec1 * vec2)


# TODO Rewrite to be parallel
@numba.jit(nopython=True)
def build_simplex(neighbors, target, smart_nn=False):
    """
    Numba friendly implementation of the projective simplex construction algorithm described in
    https://arxiv.org/abs/2109.13926

    :param neighbors:   2D numpy array of points
    :param target:      1D numpy array containing target point
    :param smart_nn:    Instead of picking the nearest neighbor, pick the neighbor that is furthest from the
                        center of mass of the point cloud. Useful for lopsided point distributions. # TODO document
    :return:            Numpy array with indices of simplex in neighbors or None if no simplex could be built
    """
    D = neighbors.shape[1]

    neigh_mask = np.arange(neighbors.shape[0], dtype=np.intc)
    diffvecs   = np.zeros(neighbors.shape)
    dot_prods  = np.zeros(neighbors.shape[0])
    simplex    = np.zeros(D+1, dtype=np.intc)

    pneighbors = np.copy(neighbors)
    ptarget    = np.copy(target)

    it = D
    failflag = False
    while it > 1:
        # Find nearest neighbor in projective space
        min_dist2 = np.inf
        pnni      = -1       # projected nearest neighbor index
        if smart_nn:
            nn_candidates = 0
            for i in range(pneighbors.shape[0]):
                if neigh_mask[i] == -1:
                    diffvecs[i] = 0
                    continue
                diffvecs[i] = pneighbors[i] - ptarget
                nn_candidates += 1

            mean_vec = np.sum(diffvecs, axis=0) / nn_candidates
            dot_prods = np.dot(diffvecs, mean_vec)
            pnni = np.argmin(dot_prods)
        else:
            if it == D and not smart_nn:
                pnni = 0
            else:
                for i in range(pneighbors.shape[0]):
                    if neigh_mask[i] == -1:
                        continue
                    dist2 = dot(ptarget - pneighbors[i], ptarget - pneighbors[i])
                    if dist2 < min_dist2:
                        min_dist2 = dist2
                        pnni = i

        # remove point so we won't find it at nearest neighbor with distance 0 later
        neigh_mask[pnni] = -1

        # The closest point must be part of the solution
        # the simplex array is filled backwards
        simplex[it] = pnni

        pnn = pneighbors[pnni]      # nearest neighbor
        # normal vector that we will use to build next projective space
        pnnvec = pnn - ptarget
        pnnvec2 = dot(pnnvec, pnnvec)

        # project
        for i in range(pneighbors.shape[0]):
            if neigh_mask[i] == -1:
                continue

            # projection on pnnvec
            pn = dot(pneighbors[i], pnnvec) * pnnvec / pnnvec2
            # projection on pnnvec shifted to origin
            pnvert = pn - ptarget

            # throw out half the points (on the wrong side of the projection plane)
            if dot(pnvert, pnnvec) > 0:
                neigh_mask[i] = -1
                continue

            # project onto plane
            pneighbors[i] = pneighbors[i] - pn

        # print(neigh_mask)
        # Verify we still have enough neighbors to continue next iteration Last step (line) requires at least 2
        if (neigh_mask != -1).sum() < 1 + it:
            failflag = True
            break

        # update ptarget
        ptarget = ptarget - dot(ptarget, pnnvec) * pnnvec / pnnvec2
        it -= 1

    if failflag:
        return None

    linevecflag = False
    posmin = np.inf
    negmax = -np.inf
    posind = -1
    negind = -1
    for i in range(pneighbors.shape[0]):
        if neigh_mask[i] == -1:
            continue

        if not linevecflag:
            linevec = pneighbors[i] - ptarget
            linevecflag = True

        tempvec = pneighbors[i] - ptarget
        proj1d = dot(linevec, tempvec)

        if proj1d > 0 and proj1d < posmin:
            posmin = proj1d
            posind = i
        elif proj1d < 0 and proj1d > negmax:
            negmax = proj1d
            negind = i

    if posind == -1 or negind == -1:
        return None
    else:
        simplex[1] = negind
        simplex[0] = posind

    return simplex


def build_simplex_adaptive(points, target, tree, k, factor, max_steps, jobs=1, smart_nn=False):
    """
    Adaptive interface for build_simplex. Automatically increases k if the algorithm fails.

    As a convenience feature for sample_step_psi() it automatically ignores nearest neighbors that
    are within an epsilon (1e-8). Doing this here avoids some unnecessary array copies.

    :param points:          2D numpy array of points
    :param target:          1D numpy array containing target point
    :param tree:            scipy KDTree object
    :param k:               Number of nearest neighbors to find initially
    :param factor:          Factor to increase k if algorithm fails
    :param max_steps:       Number of times to retry before giving up
    :param jobs:            Number of parallel jobs to use
    :return:                Numpy array with indices of simplex in points or None if no simplex could be built
    """
    break_early = False
    dists, neighbor_indices = tree.query(target, k, workers=jobs)
    if neighbor_indices.shape[0] > points.shape[0]:
        neighbor_indices = neighbor_indices[:points.shape[0]-1]
        break_early = True # no point iterating if we are already at max k
    if dists[0] < 1e-8: # epsilon
        # Don't find the target point itself as a neighbor
        # one dropped value shouldnt matter except at *really* small k...
        neighbor_indices = neighbor_indices[1:]

    neighbors = points[neighbor_indices]
    simplex = build_simplex(neighbors, target, smart_nn)

    steps = 0
    while simplex is None and steps < max_steps:
        if break_early:
            break

        k *= factor
        steps += 1
        dists, neighbor_indices = tree.query(target, k, workers=jobs)
        if neighbor_indices.shape[0] > points.shape[0]:
            neighbor_indices = neighbor_indices[:points.shape[0]-1]
            break_early = True
        if dists[0] < 1e-8: # epsilon
            neighbor_indices = neighbor_indices[1:]

        neighbors = points[neighbor_indices]
        simplex = build_simplex(neighbors, target, smart_nn)

    if simplex is not None:
        return neighbor_indices[simplex]
    else:
        return None


if __name__ == "__main__":
    from scipy.spatial import KDTree, Delaunay
    dimensions = 3
    k = 30
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

        tri = Delaunay(neighbors[simplex])

        if tri.find_simplex(target) == -1:
            wrong += 1
            continue

    print(f"Built {N_targets} simplices on {N_points} points")
    print(f"{invalid} invalid simplices")
    print(f"{wrong} simplices did not contain their targets")
    print()


    print("Running in adaptive mode")
    invalid = 0
    wrong = 0

    D = 3
    for i in range(N_targets):
        target = np.random.random(dimensions) + 0.5
        _, neighbors = tree.query(target, k)
        simplex = build_simplex_adaptive(points, target, tree, k, 2, 4)

        if simplex is None:
            invalid += 1
            continue

        tri = Delaunay(points[simplex])

        if tri.find_simplex(target) == -1:
            wrong += 1
            continue


    print(f"Built {N_targets} simplices on {N_points} points")
    print(f"{invalid} invalid simplices")
    print(f"{wrong} simplices did not contain their targets")
    print()




















