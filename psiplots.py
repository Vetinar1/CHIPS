import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import pandas as pd
from scipy.spatial import KDTree

save = False
np.random.seed(0)

# points = [
#     [3.39431, 0.321248, 1.84975],
#     [3.89382, 0.368522, 2.12197],
#     [4.29014, 0.406031, 2.33794],
#     [4.30093, 0.407053, 2.34382],
#     [4.31987, 0.408845, 2.35414],
#     [4.41762, 0.418097, 2.40742],
#     [4.20784, 0.398242, 2.29309],
# ]
#
# points = np.array(points)
#
# points = pd.read_csv("psilog", sep=" ", names=["T", "nH", "Z"])
# points = points.to_numpy()
# target = points[0]
# points = points[1:]
#
# fullpoints = pd.read_csv("fulldata2.csv")
# fullpoints = fullpoints.to_numpy()

# fig = plt.figure()
# ax = plt.axes(projection="3d")
# ax.set_box_aspect(aspect = (1,1,1))
#
# # plt.plot(fullpoints[:,0], fullpoints[:,1], fullpoints[:,2], ".", color="orange", markersize=0.5)
# plt.plot(points[:,0], points[:,1], points[:,2], "b.")
# plt.plot([target[0]], [target[1]], [target[2]], "rx")
# # plt.plot([nn[0]], [nn[1]], [nn[2]], marker="+", color="orange")
#
# plt.xlabel("T")
# plt.ylabel("nH")
# plt.gca().set_zlabel("Z")
# plt.show()



def dot(vec1, vec2):
    return np.sum(vec1 * vec2)

np.random.seed(0)

target = np.array([0.5, 0.5, 0.5])
points = np.random.random((50, 3))

btree = KDTree(points)
_, nn = btree.query(target)
nn = points[nn]
nnvec = nn - target

fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection="3d")
ax.set_box_aspect(aspect = (1,1,1))

a1 = plt.plot(points[:,0], points[:,1], points[:,2], "b.")
a2 = plt.plot([target[0]], [target[1]], [target[2]], "g.")

a4 = plt.gca().quiver(
    target[0], target[1], target[2],
    nnvec[0], nnvec[1], nnvec[2],
    color="red", lw=1
)



plt.show(block=False)



projpoints = []
pospoints = []
negpoints = []

for p in points:
    pn = dot(p, nnvec) * nnvec / dot(nnvec, nnvec)
    pnvert = pn - target

    if dot(pnvert, nnvec) > 0:
        pospoints.append(p)
        continue
    else:
        negpoints.append(p)

    projpoints.append(p - pn)

projpoints = np.array(projpoints)
pospoints = np.array(pospoints)
negpoints = np.array(negpoints)

tpn = dot(target, nnvec) * nnvec / dot(nnvec, nnvec)
tp = target - tpn

for i in range(projpoints.shape[0]):
    projpoints[i] += target - tp

fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection="3d")
ax.set_box_aspect(aspect = (1,1,1))

plt.plot(pospoints[:,0], pospoints[:,1], pospoints[:,2], "x", color="orange")
plt.plot(negpoints[:,0], negpoints[:,1], negpoints[:,2], "b.")
plt.plot([target[0]], [target[1]], [target[2]], "rx")
# plt.plot([nn[0]], [nn[1]], [nn[2]], marker="+", color="orange")

plt.gca().quiver(
    target[0], target[1], target[2],
    nn[0] - target[0], nn[1] - target[1], nn[2] - target[2],
    color="red", lw=1
)

plt.show(block=False)


fig = plt.figure()
ax = plt.axes(projection="3d", proj_type="ortho")
ax.set_box_aspect(aspect = (1,1,1))

plt.plot(negpoints[:,0], negpoints[:,1], negpoints[:,2], "b.")
plt.plot([target[0]], [target[1]], [target[2]], "rx")
# plt.plot([nn[0]], [nn[1]], [nn[2]], marker="+", color="orange")


# plt.plot(projpoints[:,0], projpoints[:,1], projpoints[:,2], ".", color="magenta")
for i in range(projpoints.shape[0]):
    plt.plot(
        [projpoints[i,0], negpoints[i,0]],
        [projpoints[i,1], negpoints[i,1]],
        [projpoints[i,2], negpoints[i,2]],
        color="orange",
        lw=0.5
    )
    plt.plot([projpoints[i,0]], [projpoints[i,1]], [projpoints[i,2]], ".", color="magenta")

plt.gca().quiver(
    target[0], target[1], target[2],
    nn[0] - target[0], nn[1] - target[1], nn[2] - target[2],
    color="red", lw=1
)

plt.show(block=False)





btree = KDTree(projpoints)
_, nn = btree.query(target)
nn = projpoints[nn]
nnvec = nn - target

projpoints2 = []
pospoints = []
negpoints = []

for p in projpoints:
    pn = dot(p, nnvec) * nnvec / dot(nnvec, nnvec)
    pnvert = pn - target

    if dot(pnvert, nnvec) > 0:
        pospoints.append(p)
        continue
    else:
        negpoints.append(p)

    projpoints2.append(p - pn)


projpoints2 = np.array(projpoints2)
pospoints = np.array(pospoints)
negpoints = np.array(negpoints)

tpn = dot(target, nnvec) * nnvec / dot(nnvec, nnvec)
tp2 = target - tpn

for i in range(projpoints.shape[0]):
    projpoints[i] += target - tp2


fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection="3d")
ax.set_box_aspect(aspect = (1,1,1))

plt.plot(pospoints[:,0], pospoints[:,1], pospoints[:,2], "x", color="orange")
plt.plot(negpoints[:,0], negpoints[:,1], negpoints[:,2], "b.")
plt.plot([target[0]], [target[1]], [target[2]], "rx")
# plt.plot([nn[0]], [nn[1]], [nn[2]], marker="+", color="orange")

plt.gca().quiver(
    target[0], target[1], target[2],
    nnvec[0], nnvec[1], nnvec[2],
    color="red", lw=1
)

plt.show(block=False)


fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection="3d", proj_type="ortho")
ax.set_box_aspect(aspect = (1,1,1))

plt.plot(negpoints[:,0], negpoints[:,1], negpoints[:,2], "b.")
plt.plot([target[0]], [target[1]], [target[2]], "rx")
# plt.plot([nn[0]], [nn[1]], [nn[2]], marker="+", color="orange")


plt.plot(projpoints2[:,0], projpoints2[:,1], projpoints2[:,2], ".", color="magenta")
for i in range(min(projpoints2.shape[0], negpoints.shape[0])):
    plt.plot(
        [projpoints2[i,0], negpoints[i,0]],
        [projpoints2[i,1], negpoints[i,1]],
        [projpoints2[i,2], negpoints[i,2]],
        color="orange",
        lw=0.5
    )

plt.gca().quiver(
    target[0], target[1], target[2],
    nnvec[0], nnvec[1], nnvec[2],
    color="red", lw=1
)

plt.show()