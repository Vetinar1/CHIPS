import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import pandas as pd
from scipy.spatial import KDTree
# import seaborn as sns


save = True

colors = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#fb85c3', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

blue = colors[0]
orange = colors[1]
green = colors[2]
pink = colors[3]
red = colors[4]


plt.figure(figsize=(5.2, 3.6))
data = np.loadtxt("2022_03_02_lingrid/lingrid.cool")
plt.plot(data[:,1], data[:,3], c=blue, label="H")
plt.plot(data[:,1], data[:,4], c=orange, label="He")
plt.plot(data[:,1], data[:,8], c=green, label="C")
plt.plot(data[:,1], data[:,9], c=pink, label="N")
plt.plot(data[:,1], data[:,10], c=red, label="O")
plt.plot(data[:,1], data[:,29], c=colors[5], label="Fe")
plt.plot(data[:,1], data[:,42], c=colors[6], label="FF$_H$")
plt.plot(data[:,1], data[:,46], c=colors[7], label="IC")
plt.plot(data[:,1], data[:,2], "k-", lw=2, label="$\Lambda$")
plt.xscale("log")
plt.yscale("log")
plt.ylim(1e-26)
plt.legend()
plt.xlabel(r"$T$ in K")
plt.ylabel(r"$\Lambda$ in erg cm$^{-3}$ s$^{-1}$")
plt.savefig("08_coolfunc.pdf", bbox_inches="tight")
exit()


coords = ["T", "nH", "Z"]
data = pd.read_csv("psi_13/z0.0.points")
data = data.drop("index", axis=1)
# sns.pairplot(data, hue="values", diag_kind="hist")
# sns.pairplot(data, )
# sns.pairplot(
#     data,
#     plot_kws={
#         "marker":".",
#         "edgecolor":"none",
#         "facecolor":blue,
#         "size":0.25
#     },
#     height=2,
#     x_vars=["T", "nH"], y_vars=["T", "nH", "Z"]
# )
# plt.show()


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6, 4), dpi=300)
img1 = ax[0].scatter(
    data["T"], data["nH"], c=data["values"], marker=".", s=4, cmap=sns.color_palette("flare", as_cmap=True),
    rasterized=True
)
img2 = ax[1].scatter(
    data["T"], data["Z"], c=data["values"], marker=".", s=4, cmap=sns.color_palette("flare", as_cmap=True),
    rasterized=True
)


ax[0].set_xlabel(r"$\log T$ in K")
ax[0].set_ylabel(r"$\log n_H$ in cm$^{-3}$")
ax[1].set_xlabel(r"$\log T$ in K")
ax[1].set_ylabel(r"$\log Z$ in Z$_\odot$")
plt.tight_layout()
fig.colorbar(img1, ax=ax[:], location="bottom", fraction=0.05, shrink=1, aspect=40, pad=0.15, label=r"$\log \Lambda$ in erg cm$^{-3}$ s$^{-1}$")

if save:
    plt.savefig("07_exampledist.pdf")
else:
    plt.show()


exit()







np.random.seed(0)

def dot(vec1, vec2):
    return np.sum(vec1 * vec2)

# np.random.seed(0)
# points = np.random.random((50, 3))
np.random.seed(20)

lims = (0.35, 0.65)
figsizes = (2.5, 2.5)
# angles = (8, -35)
angles = (16, -45)
# points = lims[0] + (lims[1] - lims[0]) * np.random.random((20, 3))
points = 0.2 + 0.6 * np.random.random((20, 3))
points[-1] = np.array([0.55, 0.55, 0.55])

points[13] -= np.array([0.05, 0.05, 0.05])
points[15] -= np.array([0.05, 0.05, 0.05])

keep = [1, 2, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19]
points = points[keep]

target = np.array([0.5, 0.5, 0.5])

btree = KDTree(points)
_, nn = btree.query(target)
nn = points[nn]
nnvec = nn - target

# fig = plt.figure(figsize=(6, 6), dpi=150)
# ax = plt.axes(projection="3d")
# ax.set_box_aspect(aspect = (1,1,1))
#
# a1 = plt.plot(points[:,0], points[:,1], points[:,2], "b.")
# a2 = plt.plot([target[0]], [target[1]], [target[2]], "g.")
#
# a4 = plt.gca().quiver(
#     target[0], target[1], target[2],
#     nnvec[0], nnvec[1], nnvec[2],
#     color="red", lw=1
# )
#
# plt.show()


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

fig = plt.figure(figsize=figsizes, dpi=150)
ax = plt.axes(projection="3d", proj_type="ortho")
ax.set_box_aspect(aspect = (1,1,1))

plt.plot(pospoints[:,0], pospoints[:,1], pospoints[:,2], "x", color=orange)
plt.plot(negpoints[:,0], negpoints[:,1], negpoints[:,2], ".", color=blue)
plt.plot([target[0]], [target[1]], [target[2]], "rx")
# plt.plot([nn[0]], [nn[1]], [nn[2]], marker="+", color="orange")

plt.gca().quiver(
    target[0], target[1], target[2],
    nn[0] - target[0], nn[1] - target[1], nn[2] - target[2],
    color="red", lw=1
)

plt.gca().view_init(*angles)
plt.gca().set_xlim(*lims)
plt.gca().set_ylim(*lims)
plt.gca().set_zlim(*lims)
plt.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)

plt.tight_layout()
if not save:
    plt.show() # 37, 6
else:
    plt.savefig("03_alg1.pdf", transparent=True)






fig = plt.figure(figsize=figsizes, dpi=150)
ax = plt.axes(projection="3d", proj_type="ortho")
ax.set_box_aspect(aspect = (1,1,1))

plt.plot(negpoints[:,0], negpoints[:,1], negpoints[:,2], ".", color=blue)
plt.plot([target[0]], [target[1]], [target[2]], "rx")
# plt.plot([nn[0]], [nn[1]], [nn[2]], marker="+", color="orange")


# plt.plot(projpoints[:,0], projpoints[:,1], projpoints[:,2], ".", color="magenta")
for i in range(projpoints.shape[0]):
    plt.plot(
        [projpoints[i,0], negpoints[i,0]],
        [projpoints[i,1], negpoints[i,1]],
        [projpoints[i,2], negpoints[i,2]],
        color=orange,
        lw=0.5,
        zorder=-100
    )
    plt.plot([projpoints[i,0]], [projpoints[i,1]], [projpoints[i,2]], ".", color=pink)
    # plt.gca().text(projpoints[i,0], projpoints[i,1], projpoints[i,2], str(i), color="k")

# for i in range(points.shape[0]):
#     plt.gca().text(points[i,0], points[i,1], points[i,2], str(i), color="r")

plt.gca().quiver(
    target[0], target[1], target[2],
    nn[0] - target[0], nn[1] - target[1], nn[2] - target[2],
    color="red", lw=1
)

plt.gca().view_init(*angles)
plt.gca().set_xlim(*lims)
plt.gca().set_ylim(*lims)
plt.gca().set_zlim(*lims)
plt.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)


plt.tight_layout()
if not save:
    plt.show() # 37, 6
else:
    plt.savefig("04_alg2.pdf", transparent=True)




fig = plt.figure(figsize=figsizes, dpi=150)
ax = plt.axes(projection="3d", proj_type="ortho")
ax.set_box_aspect(aspect = (1,1,1))

# plt.plot([nn[0]], [nn[1]], [nn[2]], marker="+", color="orange")


# plt.plot(projpoints[:,0], projpoints[:,1], projpoints[:,2], ".", color="magenta")
for i in range(projpoints.shape[0]):
    plt.plot(
        [projpoints[i,0], negpoints[i,0]],
        [projpoints[i,1], negpoints[i,1]],
        [projpoints[i,2], negpoints[i,2]],
        color=orange,
        lw=0.5
    )
    plt.plot([projpoints[i,0]], [projpoints[i,1]], [projpoints[i,2]], ".", color=pink)

plt.gca().quiver(
    target[0], target[1], target[2],
    nn[0] - target[0], nn[1] - target[1], nn[2] - target[2],
    color="red", lw=1
)


plt.plot(negpoints[:,0], negpoints[:,1], negpoints[:,2], ".", color=blue)
# pyramid = np.array([
#     projpoints[3],
#     projpoints[20],
#     projpoints[12],
#     projpoints[3],
#     nn,
#     projpoints[20],
#     nn,
#     projpoints[12],
# ])
# # plt.plot([projpoints[12,0]], [projpoints[12,1]], [projpoints[12,2]], "ro")
# # plt.plot(pyramid[:,0], pyramid[:,1], pyramid[:,2], color=green, zorder=100)
#
# py1 = np.array([
#     projpoints[3],
#     projpoints[12],
#     nn,
#     projpoints[3]
# ])
# py2 = np.array([
#     projpoints[12],
#     projpoints[20],
#     projpoints[3]
# ])
# py3 = np.array([
#     projpoints[20],
#     nn
# ])
py1 = np.array([
    projpoints[3],
    projpoints[4],
    nn,
    projpoints[3]
])
py2 = np.array([
    projpoints[4],
    projpoints[2],
    projpoints[3]
])
py3 = np.array([
    projpoints[2],
    nn
])
plt.plot(py1[:,0], py1[:,1], py1[:,2], color=green, zorder=100)
plt.plot(py2[:,0], py2[:,1], py2[:,2], color=green)
plt.plot(py3[:,0], py3[:,1], py3[:,2], color=green)
plt.plot([target[0]], [target[1]], [target[2]], "rx")

plt.gca().view_init(*angles)

plt.gca().set_xlim(*lims)
plt.gca().set_ylim(*lims)
plt.gca().set_zlim(*lims)
plt.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)

plt.tight_layout()
if not save:
    plt.show() # 37, 6
else:
    plt.savefig("05_alg3.pdf", transparent=True)






# pyramid = np.array([
#     negpoints[3],
#     negpoints[20],
#     negpoints[12],
#     negpoints[3],
#     nn,
#     negpoints[20],
#     nn,
#     negpoints[12],
# ])

# py1 = np.array([
#     negpoints[3],
#     negpoints[12],
#     nn,
#     negpoints[3]
# ])
# py2 = np.array([
#     negpoints[12],
#     negpoints[20],
#     negpoints[3]
# ])
# py3 = np.array([
#     negpoints[20],
#     nn
# ])

py1 = np.array([
    negpoints[3],
    negpoints[4],
    nn,
    negpoints[3]
])
py2 = np.array([
    negpoints[4],
    negpoints[2],
    negpoints[3]
])
py3 = np.array([
    negpoints[2],
    nn
])





fig = plt.figure(figsize=figsizes, dpi=150)
ax = plt.axes(projection="3d", proj_type="ortho")
ax.set_box_aspect(aspect = (1,1,1))

plt.plot(negpoints[:,0], negpoints[:,1], negpoints[:,2], ".", color=blue)
# plt.plot([nn[0]], [nn[1]], [nn[2]], marker="+", color="orange")


# plt.plot(projpoints[:,0], projpoints[:,1], projpoints[:,2], ".", color="magenta")
for i in range(projpoints.shape[0]):
    plt.plot(
        [projpoints[i,0], negpoints[i,0]],
        [projpoints[i,1], negpoints[i,1]],
        [projpoints[i,2], negpoints[i,2]],
        color=orange,
        lw=0.5,
        zorder=-100
    )
    plt.plot([projpoints[i,0]], [projpoints[i,1]], [projpoints[i,2]], ".", color=pink)

plt.gca().quiver(
    target[0], target[1], target[2],
    nn[0] - target[0], nn[1] - target[1], nn[2] - target[2],
    color="red", lw=1
)
plt.gca().view_init(*angles)

# plt.plot(pyramid[:,0], pyramid[:,1], pyramid[:,2], color=green)
plt.plot(py1[:,0], py1[:,1], py1[:,2], color=green, zorder=100)
plt.plot(py2[:,0], py2[:,1], py2[:,2], color=green)
plt.plot(py3[:,0], py3[:,1], py3[:,2], color=green)
plt.plot([target[0]], [target[1]], [target[2]], "rx")
plt.gca().set_xlim(*lims)
plt.gca().set_ylim(*lims)
plt.gca().set_zlim(*lims)
plt.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)

plt.tight_layout()
if not save:
    plt.show() # 37, 6
else:
    plt.savefig("06_alg4.pdf", transparent=True)
exit()



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


fig = plt.figure(figsize=(6, 6), dpi=150)
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

