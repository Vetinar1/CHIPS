import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import KDTree, Delaunay


def dot(vec1, vec2):
    return np.sum(vec1 * vec2)


def simplex_alg(neighbors, target):
    D = neighbors.shape[1]

    neighbor_indices = np.reshape(np.array([i for i in range(neighbors.shape[0])]), (neighbors.shape[0], 1))
    rneighbors = np.hstack((neighbors, neighbor_indices))
    rtarget = target

    pneighbors, ptarget = rneighbors, rtarget

    rsimplex = []

    it = D
    failflag = False
    while it > 1:
        # Find nearest neighbor in projective space
        diffs2 = []
        for i, neigh in enumerate(pneighbors[:, -1]):
            dist = np.sum(np.power(ptarget - neigh, 2))
            diffs2.append(dist)

        pnn  =     pneighbors[diffs2.index(min(diffs2)), :-1]
        pnni = int(pneighbors[diffs2.index(min(diffs2)), -1])
        # remove point so we won't find it at nearest neighbor with distance 0 later
        pneighbors = np.delete(pneighbors, diffs2.index(min(diffs2)), axis=0)

        # The closest point must be part of the solution
        rsimplex.append(rneighbors[pnni])

        # normal vector that we will use to build next projective space
        pnnvec = pnn - ptarget
        pnnvec2 = dot(pnnvec, pnnvec)

        # project
        new_pneighbors = []

        for pneigh in pneighbors:
            # projection on pnnvec
            pn = dot(pneigh[:-1], pnnvec) * pnnvec / pnnvec2
            # projection on pnnvec shifted to origin
            pnvert = pn - ptarget

            # throw out half the points on the wrong side of the projection plane
            pndot = dot(pnvert, pnnvec)
            if pndot > 0:
                continue

            # subtract projection pnnvec (= project onto plane)
            new_pneighbors.append(pneigh - np.append(pn, [0]))

        # update pneighbors, ptarget
        ptarget = ptarget - dot(ptarget, pnnvec) * pnnvec / pnnvec2
        if len(new_pneighbors) <= 1:
            failflag = True
            break
        pneighbors = np.array(new_pneighbors)
        it -= 1

    if failflag:
        return None

    # all pneighbors are now on a line
    # find nearest neighbors in both directions of the line
    diffs2 = []
    for i, neigh in enumerate(pneighbors[:, -1]):
        dist = np.sum(np.power(ptarget - neigh, 2))
        diffs2.append(dist)

    # first nearest neighbor
    pnn = pneighbors[diffs2.index(min(diffs2)), :-1]
    pnni = int(pneighbors[diffs2.index(min(diffs2)), -1])
    rsimplex.append(rneighbors[pnni])
    # points from target to nn
    pnnvec = pnn - ptarget

    # find nearest neighbor in other direction
    best_nn2_ind = None
    best_dist = np.inf
    for i, neigh in enumerate(pneighbors):
        # points from target to neigh
        diffvec = neigh[:-1] - ptarget

        # point in same direction?
        if dot(pnnvec, diffvec) > 0:
            continue

        dist = np.sqrt(dot(diffvec, diffvec))
        if dist < best_dist:
            best_dist = dist
            best_nn2_ind = neigh[-1]

    if best_dist == np.inf:
        return None

    rsimplex.append(rneighbors[int(best_nn2_ind)])

    return np.array(rsimplex)

data = pd.read_csv("extra_data_test1/fulldata2.csv")
tree = KDTree(data.loc[:,["T", "nH", "Z"]].to_numpy())

k = 30
target = np.array([5, 0, -1])

_, neighbours = tree.query(target, k)
print(neighbours)
neighbours = data.loc[neighbours, ["T", "nH", "Z"]].to_numpy()

simplex = simplex_alg(neighbours, target)
tri = Delaunay(simplex[:,:-1])
print(tri.find_simplex(target))
simplex = pd.DataFrame(simplex[:, :-1], columns=["T", "nH", "Z"])

data["index"] = data.index
print(simplex.merge(data, on=["T", "nH", "Z"]).loc[:, ["T", "nH", "Z", "index"]])

