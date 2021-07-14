import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

dims = 3

fulldata = pd.read_csv("extra_data_test1/z0.0.points")
data = fulldata.loc[:, ["T", "nH", "Z"]].to_numpy()
tree = KDTree(data)

point = np.random.uniform(size=dims)
point[0] = 3 + 4 * point[0]
point[1] = -2 + 4 * point[1]
point[2] = -1 + 1 * point[2]

distances, neighbor_indices = tree.query(point, 20)
neighbors = data[neighbor_indices]

plt.figure(dpi=300)
plt.xlabel("T")
plt.ylabel("nH")
plt.plot(data[:,0], data[:,1], "b.", markersize=1)
plt.plot([point[0]], [point[1]], "rx")
plt.plot(neighbors[:,0], neighbors[:,1], "r.", markersize=1.5)
plt.show()


# Pick d+1 initial points
# random
# better: 2 points furthers apart, point furthes from line, point furthest from triangle
hull = pd.DataFrame(neighbors).sample(dims+1).to_numpy()
hullcenter = np.sum(hull, axis=0) / (dims+1)

