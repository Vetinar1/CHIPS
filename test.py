import pandas as pd
# from cloudy_optimizer import compile_to_dataframe
# import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# from util import sample_simplex, sample_simplices
from chips import optimizer
from scipy import spatial
from tqdm import tqdm
import itertools
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from parse import parse

maxpoints = 0
maxtris = 0
pathlist = Path("run29_3d_compiled").glob("*.points")
for path in pathlist:
    data = pd.read_csv(str(path))
    maxpoints = max(maxpoints, len(data.index))
    data = data.drop(["diff", "interpolated"], axis=1)
    data = data.to_numpy()
    tri = spatial.Delaunay(data[:,:-1])

    np.savetxt("run29_compiled2/z" + str(round(float(path.stem[1:]), 2)) + path.suffix, data, delimiter=",")
    np.savetxt("run29_compiled2/z" + str(round(float(path.stem[1:]), 2)) +
               ".tris", tri.simplices.astype(int), delimiter=",", fmt="%i")
    np.savetxt("run29_compiled2/z" + str(round(float(path.stem[1:]), 2)) +
               ".neighbors", tri.neighbors.astype(int), delimiter=",", fmt="%i")

    maxtris = max(maxtris, tri.simplices.shape[0])


    # plt.figure(figsize=(1.2*6, 6))
    # plt.triplot(data[:,0], data[:,1], triangles=tri.simplices, linewidth=0.5, color="r")
    # plt.scatter(data[:,0], data[:,1], color="k", s=4, zorder=1000)
    # plt.gcf().set_dpi(200)
    # plt.xlim(1, 9)
    # plt.ylim(-5, 5)
    # plt.show()

print(maxpoints)
print(maxtris)
exit()


points = np.loadtxt("data.csv", delimiter=",", skiprows=1)
ctris = np.loadtxt("dtri.csv", delimiter=",").astype(int)

N = points.shape[1] - 1
print(points.shape)
tri = spatial.Delaunay(points[:, :-1])

z_min = 0
z_max = 2.2
z_splits = 10

pathlist = Path(".").glob("slice_*.slice")
for c, path in enumerate(pathlist):
    str_path = str(path)
    result = parse("slice_{i}_{z_low}_{z_high}.slice", str_path)
    context = result.named
    print(context)
    simplex_indices = np.loadtxt("slice_{i}_{z_low}_{z_high}.slice".format(**context)).astype(int)
    disc = np.loadtxt("slice_{i}_{z_low}_{z_high}.disc".format(**context)).astype(int)

    tris = ctris[simplex_indices]
    disctris = ctris[disc]

    centroids = np.zeros((tris.shape[0], N))
    discc = np.zeros((disctris.shape[0], N))

    for i in range(tris.shape[0]):
        centroids[i] = np.sum(points[tris[i],:-1], axis=0) / (N+1)
    for i in range(disctris.shape[0]):
        discc[i] = np.sum(points[disctris[i],:-1], axis=0) / (N+1)


    plt.figure(figsize=(1.2 * 6, 6))
    plt.triplot(points[:,1], points[:,0], triangles=ctris, linewidth=0.5, color="k")
    plt.scatter(centroids[:,1], centroids[:,0], color="bg", s=2)
    plt.scatter(discc[:,1], discc[:,0], color="orange", s=2)
    plt.title(r"Distribution of samples, $D = " + str(N) + "$, $z = [$" +
              str(context["z_low"]) + ", " + str(context["z_high"]) + "]$")
    plt.axvline(float(context["z_low"]), color="r")
    plt.axvline(float(context["z_high"]), color="r")
    plt.gcf().set_dpi(200)
    plt.xlim(-0.2, 2.2)
    # plt.xlim(-5, 5)
    plt.ylim(1, 9)
    plt.xlabel(r"redshift $z$")
    plt.ylabel(r"Temperature $T$")
    plt.show()

exit()




data = optimizer._load_existing_data(
    "run28_2d/",
    "T_{T}__z_{z}",
    ["T", "z"]
)

data = optimizer.single_evaluation_step(
    data,
    param_space={
        "T":[2, 8],
        #"nH":[-4, 4],
        "z":[0, 2]
    },
    param_space_margins={
        "T":0.1,
        #"nH":0.1,
        "z":[0, 2.2]
    },
    rad_params={
        # "hhT6":("spectra/hhT6", [15, 25]),
        # "SFR":("spectra/SFR", [15, 25])
    },
    rad_params_margins={
        # "hhT6":0.1,
        # "SFR":0.1
    },
    z_split_partitions=10
)

data.to_csv("data.csv", index=False)
exit()



fig = plt.figure(figsize=(6.4, 6.4))
points = np.array([
    [4, 2],
    [3, 3],
    [4.5, 2.5],
    [1.5, 2.5]
])

circ = plt.Circle((4, 2), np.sqrt(2), color="b", fill=False)
plt.vlines(3, 0, 6, "k", linestyles="--", linewidth=0.8)
plt.plot([1.5, 1.5, 4-np.sqrt(2)], [2.5, 2, 2], "r")
plt.plot([4-np.sqrt(2), 4], [2, 2], "r--", linewidth=0.5)
plt.gca().add_patch(circ)
plt.plot(points[:,0], points[:,1], "k.")
plt.plot([1.5], [2], "kx")
plt.xlim(0, 6)
plt.ylim(0, 6)
plt.gcf().set_dpi(200)
plt.savefig("balltree.png", transparent=True)

exit()


coords = np.array([
    [2, 1, 1.2],
    [2, 2, 1.5],
    [1, 1, 1],
    [1, 2, 3]
])

base = np.zeros(coords.shape)
base[:,0:2] = coords[:,0:2]

fig = plt.figure()
fig.set_dpi(200)
ax = fig.add_subplot(111, projection="3d")

for i, point in enumerate(coords):
    ax.plot(
        [point[0], base[i,0]],
        [point[1], base[i,1]],
        [point[2], base[i,2]],
        linewidth=0.5,
        color="k"
    )

ax.plot(
    [coords[0,0], coords[2,0]],
    [coords[0,1], coords[2,1]],
    [coords[0,2], coords[2,2]],
    color="green"
)
ax.plot(
    [coords[1,0], coords[3,0]],
    [coords[1,1], coords[3,1]],
    [coords[1,2], coords[3,2]],
    color="green"
)

def linear_interp(x, x0, y0, x1, y1):
    return (y0 * (x1 - x) + y1 * (x - x0)) / (x1 - x0)

N = 10
for i in range(N+1):
    x = coords[2,0] + i / N
    z1 = linear_interp(x, coords[2,0], coords[2,2], coords[0,0], coords[0,2])
    z2 = linear_interp(x, coords[3,0], coords[3,2], coords[1,0], coords[1,2])
    ax.plot(
        [x, x],
        [coords[2,1], coords[1,1]],
        [z1, z2],
        color="blue"
    )
    y = coords[2,1] + 1 - i / N
    z3 = linear_interp(y, coords[2,1], z1, coords[1,1], z2)
    ax.plot(
        [x], [y], [z3], "r."
    )

ax.plot(coords[:,0], coords[:,1], coords[:,2], "k.")

ax.set_xlim(0.5, 2.5)
ax.set_ylim(0.5, 2.5)
ax.set_zlim(0, 3.5)

for line in ax.xaxis.get_ticklines():
    line.set_visible(False)
for line in ax.yaxis.get_ticklines():
    line.set_visible(False)
for line in ax.zaxis.get_ticklines():
    line.set_visible(False)

ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])
plt.savefig("multilinear.png", transparent=True)

exit()

x1 = 2
y1 = 4
x2 = 8
y2 = 2
x3 = 6
y3 = 8

bc_cx = (x1 + x2 + x3) / 3
bc_cy = (y1 + y2 + y3) / 3


def get_bary(x, y, rnd=True):
    l1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))
    l2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))
    l3 = 1 - l1 - l2
    if rnd:
        l1 = round(l1, 2)
        l2 = round(l2, 2)
        l3 = round(l3, 2)
    return l1, l2, l3

plt.figure(figsize=(6.4, 6.4))
triangle_x = [x1, x2, x3, x1]
triangle_y = [y1, y2, y3, y1]
plt.plot(triangle_x, triangle_y)
plt.arrow((x1+x3)/2, (y1+y3)/2, -1/np.sqrt(2), 1/np.sqrt(2), shape="full", head_width=0.1, head_length=0.1,
          length_includes_head=True, facecolor="k")
plt.arrow((x1+x2)/2, (y1+y2)/2, (-1/3)/(np.sqrt(1+1/9)), -1/(np.sqrt(1+1/9)), shape="full", head_width=0.1, head_length=0.1,
          length_includes_head=True, facecolor="k")
plt.arrow((x2+x3)/2, (y2+y3)/2, 1/(np.sqrt(1+1/9)), 1/3/(np.sqrt(1+1/9)), shape="full", head_width=0.1, head_length=0.1,
          length_includes_head=True, facecolor="k")
plt.plot([(x1+x3)/2, 5], [(y1+y3)/2, 8], "r--", lw=0.5)
plt.plot([(x1+x2)/2, 5], [(y1+y2)/2, 8], "r--", lw=0.5)
plt.plot([(x2+x3)/2, 5], [(y2+y3)/2, 8], "r--", lw=0.5)

plt.arrow((x1+x3)/2,
          (y1+y3)/2,
          (5-(x1+x3)/2) / np.sqrt((5-(x1+x3)/2) ** 2 + (8-(y1+y3)/2) ** 2),
          (8-(y1+y3)/2) / np.sqrt((5-(x1+x3)/2) ** 2 + (8-(y1+y3)/2) ** 2),
          head_width=0.1,
          head_length=0.1,
          color="red",
          length_includes_head=True
)
plt.arrow((x1+x2)/2,
          (y1+y2)/2,
          (5-(x1+x2)/2) / np.sqrt((5-(x1+x2)/2) ** 2 + (8-(y1+y2)/2) ** 2),
          (8-(y1+y2)/2) / np.sqrt((5-(x1+x2)/2) ** 2 + (8-(y1+y2)/2) ** 2),
          head_width=0.1,
          head_length=0.1,
          color="red",
          length_includes_head=True
)
plt.arrow((x2+x3)/2,
          (y2+y3)/2,
          (5-(x2+x3)/2) / np.sqrt((5-(x2+x3)/2) ** 2 + (8-(y2+y3)/2) ** 2),
          (8-(y2+y3)/2) / np.sqrt((5-(x2+x3)/2) ** 2 + (8-(y2+y3)/2) ** 2),
          head_width=0.1,
          head_length=0.1,
          color="red",
          length_includes_head=True
)

sp1 = -1/np.sqrt(2) * (5-(x1+x3)/2) / np.sqrt((5-(x1+x3)/2) ** 2 + (8-(y1+y3)/2) ** 2) + \
    1/np.sqrt(2) * (8-(y1+y3)/2) / np.sqrt((5-(x1+x3)/2) ** 2 + (8-(y1+y3)/2) ** 2)
plt.arrow((x1+x3)/2,
          (y1+y3)/2,
          sp1 * -1/np.sqrt(2),
          sp1 * 1/np.sqrt(2),
          head_width=0.1,
          head_length=0.1,
          color="red",
          length_includes_head=True
)
sp2 = (-1/3)/(np.sqrt(1+1/9)) * (5-(x1+x2)/2) / np.sqrt((5-(x1+x2)/2) ** 2 + (8-(y1+y2)/2) ** 2) + \
    -1/np.sqrt(1+1/9) * (8-(y1+y2)/2) / np.sqrt((5-(x1+x2)/2) ** 2 + (8-(y1+y2)/2) ** 2)
plt.arrow((x1+x2)/2,
          (y1+y2)/2,
          sp2 * (-1/3)/np.sqrt(1+1/9),
          sp2 * -1/np.sqrt(1+1/9),
          head_width=0.1,
          head_length=0.1,
          color="red",
          length_includes_head=True
)
sp3 = 1/(np.sqrt(1+1/9)) * (5-(x2+x3)/2) / np.sqrt((5-(x2+x3)/2) ** 2 + (8-(y2+y3)/2) ** 2) + \
      (1/3)/(np.sqrt(1+1/9)) * (8-(y2+y3)/2) / np.sqrt((5-(x2+x3)/2) ** 2 + (8-(y2+y3)/2) ** 2)
plt.arrow((x2+x3)/2,
          (y2+y3)/2,
          sp3 * 1/(np.sqrt(1+1/9)),
          sp3 * (1/3)/(np.sqrt(1+1/9)),
          head_width=0.1,
          head_length=0.1,
          color="red",
          length_includes_head=True
)

plt.plot([(x1+x3)/2], [(y1+y3)/2], "k.")
plt.plot([(x1+x2)/2], [(y1+y2)/2], "k.")
plt.plot([(x2+x3)/2], [(y2+y3)/2], "k.")
plt.plot([5], [8], "r.")
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.gcf().set_dpi(200)
plt.savefig("flip1.png", transparent=True)
plt.close()


plt.figure(figsize=(6.4, 6.4))
triangle_x = [x1, x2, x3, x1]
triangle_y = [y1, y2, y3, y1]
plt.plot(triangle_x, triangle_y)
plt.arrow((x1+x3)/2, (y1+y3)/2, -1/np.sqrt(2), 1/np.sqrt(2), shape="full", head_width=0.1, head_length=0.1,
          length_includes_head=True, facecolor="k")
plt.arrow((x1+x2)/2, (y1+y2)/2, (-1/3)/(np.sqrt(1+1/9)), -1/(np.sqrt(1+1/9)), shape="full", head_width=0.1, head_length=0.1,
          length_includes_head=True, facecolor="k")
plt.arrow((x2+x3)/2, (y2+y3)/2, 1/(np.sqrt(1+1/9)), 1/3/(np.sqrt(1+1/9)), shape="full", head_width=0.1, head_length=0.1,
          length_includes_head=True, facecolor="k")
plt.plot([(x1+x3)/2, 6], [(y1+y3)/2, 9], "r--", lw=0.5)
plt.plot([(x1+x2)/2, 6], [(y1+y2)/2, 9], "r--", lw=0.5)
plt.plot([(x2+x3)/2, 6], [(y2+y3)/2, 9], "r--", lw=0.5)

plt.arrow((x1+x3)/2,
          (y1+y3)/2,
          (6-(x1+x3)/2) / np.sqrt((6-(x1+x3)/2) ** 2 + (9-(y1+y3)/2) ** 2),
          (9-(y1+y3)/2) / np.sqrt((6-(x1+x3)/2) ** 2 + (9-(y1+y3)/2) ** 2),
          head_width=0.1,
          head_length=0.1,
          color="red",
          length_includes_head=True
)
plt.arrow((x1+x2)/2,
          (y1+y2)/2,
          (6-(x1+x2)/2) / np.sqrt((6-(x1+x2)/2) ** 2 + (9-(y1+y2)/2) ** 2),
          (9-(y1+y2)/2) / np.sqrt((6-(x1+x2)/2) ** 2 + (9-(y1+y2)/2) ** 2),
          head_width=0.1,
          head_length=0.1,
          color="red",
          length_includes_head=True
)
plt.arrow((x2+x3)/2,
          (y2+y3)/2,
          (6-(x2+x3)/2) / np.sqrt((6-(x2+x3)/2) ** 2 + (9-(y2+y3)/2) ** 2),
          (9-(y2+y3)/2) / np.sqrt((6-(x2+x3)/2) ** 2 + (9-(y2+y3)/2) ** 2),
          head_width=0.1,
          head_length=0.1,
          color="red",
          length_includes_head=True
)

sp1 = -1/np.sqrt(2) * (6-(x1+x3)/2) / np.sqrt((6-(x1+x3)/2) ** 2 + (9-(y1+y3)/2) ** 2) + \
    1/np.sqrt(2) * (9-(y1+y3)/2) / np.sqrt((6-(x1+x3)/2) ** 2 + (9-(y1+y3)/2) ** 2)
plt.arrow((x1+x3)/2,
          (y1+y3)/2,
          sp1 * -1/np.sqrt(2),
          sp1 * 1/np.sqrt(2),
          head_width=0.1,
          head_length=0.1,
          color="red",
          length_includes_head=True
)
sp2 = (-1/3)/(np.sqrt(1+1/9)) * (6-(x1+x2)/2) / np.sqrt((6-(x1+x2)/2) ** 2 + (9-(y1+y2)/2) ** 2) + \
    -1/np.sqrt(1+1/9) * (9-(y1+y2)/2) / np.sqrt((6-(x1+x2)/2) ** 2 + (9-(y1+y2)/2) ** 2)
plt.arrow((x1+x2)/2,
          (y1+y2)/2,
          sp2 * (-1/3)/np.sqrt(1+1/9),
          sp2 * -1/np.sqrt(1+1/9),
          head_width=0.1,
          head_length=0.1,
          color="red",
          length_includes_head=True
)
sp3 = 1/(np.sqrt(1+1/9)) * (6-(x2+x3)/2) / np.sqrt((6-(x2+x3)/2) ** 2 + (9-(y2+y3)/2) ** 2) + \
      (1/3)/(np.sqrt(1+1/9)) * (9-(y2+y3)/2) / np.sqrt((6-(x2+x3)/2) ** 2 + (9-(y2+y3)/2) ** 2)
plt.arrow((x2+x3)/2,
          (y2+y3)/2,
          sp3 * 1/(np.sqrt(1+1/9)),
          sp3 * (1/3)/(np.sqrt(1+1/9)),
          head_width=0.1,
          head_length=0.1,
          color="red",
          length_includes_head=True
)

plt.plot([(x1+x3)/2], [(y1+y3)/2], "k.")
plt.plot([(x1+x2)/2], [(y1+y2)/2], "k.")
plt.plot([(x2+x3)/2], [(y2+y3)/2], "k.")
plt.plot([6], [9], "r.")
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.gcf().set_dpi(200)
plt.savefig("flip2.png", transparent=True)
plt.close()



exit()

plt.figure(figsize=(6.4, 6.4))
triangle_x = [x1, x2, x3, x1]
triangle_y = [y1, y2, y3, y1]
plt.plot(triangle_x, triangle_y)
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.gcf().set_dpi(200)
plt.savefig("bary1.png", transparent=True)
plt.close()

plt.figure(figsize=(6.4, 6.4))
triangle_x = [x1, x2, x3, x1]
triangle_y = [y1, y2, y3, y1]
plt.plot([x1, bc_cx, x2], [y1, bc_cy, y2], "r--")
plt.plot([bc_cx, x3], [bc_cy, y3], "r--")
plt.plot([bc_cx], [bc_cy], "ro")
plt.plot(triangle_x, triangle_y)
plt.plot(triangle_x, triangle_y, "k.")
plt.annotate(get_bary(x1, y1), (x1, y1), xytext=(x1, y1+0.25), ha="center")
plt.annotate(get_bary(x2, y2), (x2, y2), xytext=(x2, y2+0.25), ha="center")
plt.annotate(get_bary(x3, y3), (x3, y3), xytext=(x3, y3+0.25), ha="center")
plt.annotate(get_bary(bc_cx, bc_cy), (bc_cx, bc_cy), xytext=(bc_cx, bc_cy+0.25), ha="center")
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.gcf().set_dpi(200)
plt.savefig("bary2.png", transparent=True)
plt.close()

plt.figure(figsize=(6.4, 6.4))
bc_cx = 4
bc_cy = 4
plt.plot([x1, bc_cx, x2], [y1, bc_cy, y2], "r--")
plt.plot([bc_cx, x3], [bc_cy, y3], "r--")
plt.plot([bc_cx], [bc_cy], "ro")
plt.plot(triangle_x, triangle_y)
plt.plot(triangle_x, triangle_y, "k.")
# plt.annotate(get_bary(x1, y1), (x1, y1), xytext=(x1, y1+0.25), ha="center")
plt.annotate(get_bary(x2, y2), (x2, y2), xytext=(x2, y2+0.25), ha="center")
plt.annotate(get_bary(x3, y3), (x3, y3), xytext=(x3, y3+0.25), ha="center")
plt.annotate(get_bary(bc_cx, bc_cy), (bc_cx, bc_cy), xytext=(bc_cx, bc_cy+0.25), ha="center")
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.gcf().set_dpi(200)
plt.savefig("bary3.png", transparent=True)
plt.close()


plt.figure(figsize=(6.4, 6.4))
bc_cx = 5
bc_cy = 3
plt.plot([x1, bc_cx, x2], [y1, bc_cy, y2], "r--")
plt.plot([bc_cx, x3], [bc_cy, y3], "r--")
plt.plot([bc_cx], [bc_cy], "ro")
plt.plot(triangle_x, triangle_y)
plt.plot(triangle_x, triangle_y, "k.")
plt.annotate(get_bary(x1, y1), (x1, y1), xytext=(x1, y1+0.25), ha="center")
plt.annotate(get_bary(x2, y2), (x2, y2), xytext=(x2, y2+0.25), ha="center")
plt.annotate(get_bary(x3, y3), (x3, y3), xytext=(x3, y3+0.25), ha="center")
plt.annotate(get_bary(bc_cx, bc_cy), (bc_cx, bc_cy), xytext=(bc_cx, bc_cy+0.25), ha="center")
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.gcf().set_dpi(200)
plt.savefig("bary4.png", transparent=True)
plt.close()

plt.figure(figsize=(6.4, 6.4))
bc_cx = 4
bc_cy = 2
plt.plot([x1, bc_cx, x2], [y1, bc_cy, y2], "r--")
plt.plot([bc_cx, x3], [bc_cy, y3], "r--")
plt.plot([bc_cx], [bc_cy], "ro")
plt.plot(triangle_x, triangle_y)
plt.plot(triangle_x, triangle_y, "k.")
plt.annotate(get_bary(x1, y1), (x1, y1), xytext=(x1, y1+0.25), ha="center")
plt.annotate(get_bary(x2, y2), (x2, y2), xytext=(x2, y2+0.25), ha="center")
plt.annotate(get_bary(x3, y3), (x3, y3), xytext=(x3, y3+0.25), ha="center")
plt.annotate(get_bary(bc_cx, bc_cy), (bc_cx, bc_cy), xytext=(bc_cx, bc_cy+0.25), ha="center")
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.gcf().set_dpi(200)
plt.savefig("bary5.png", transparent=True)
plt.close()

plt.figure(figsize=(6.4, 6.4))
bc_cx = 1.5
bc_cy = 4
plt.plot([x1, bc_cx, x2], [y1, bc_cy, y2], "r--")
plt.plot([bc_cx, x3], [bc_cy, y3], "r--")
plt.plot([bc_cx], [bc_cy], "ro")
plt.plot(triangle_x, triangle_y)
plt.plot(triangle_x, triangle_y, "k.")
# plt.annotate(get_bary(x1, y1), (x1, y1), xytext=(x1, y1+0.25), ha="center")
plt.annotate(get_bary(x2, y2), (x2, y2), xytext=(x2, y2+0.25), ha="center")
plt.annotate(get_bary(x3, y3), (x3, y3), xytext=(x3, y3+0.25), ha="center")
plt.annotate(get_bary(bc_cx, bc_cy), (bc_cx, bc_cy), xytext=(bc_cx, bc_cy+0.25), ha="center")
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.gcf().set_dpi(200)
plt.savefig("bary6.png", transparent=True)
plt.close()
exit()

data = np.loadtxt("cloudy/coolplot.cool", usecols=(1, 3))
data = np.log10(data)
plt.plot(data[:,0], data[:,1])
plt.title("The Cooling Function at $z = 0$, $Z = 0$, $n_H = 2$")
plt.xlabel(r"$\log T$")
plt.ylabel(r"$\log \Lambda$")
plt.gcf().set_dpi(200)
plt.savefig("coolplot.png", transparent=True)
exit()

# xs = np.random.random(100)
# ys = np.random.random(100)
#
# points = np.zeros((100, 2))
# points[:,0] = xs
# points[:,1] = ys
#
# tri = spatial.Delaunay(points)
#
# plt.triplot(points[:,0], points[:,1], triangles=tri.simplices, color="r")
# plt.plot(points[:,0], points[:,1], "k.")
# plt.gcf().set_dpi(200)
# plt.title("Delaunay Triangulation on a random uniform distribution of points")
# plt.show()
#
# exit()

points = np.array(
    list(itertools.product(
        list(np.arange(2, 8, 0.2)),
        list(np.arange(-4, 4, 0.4))
    ))
)

plt.plot(points[:,0], points[:,1], "k.")
plt.xlabel(r"$\log T$")
plt.ylabel(r"$\log n_H$")
plt.title("Sampling on a regular grid")
plt.gcf().set_dpi(200)
plt.xlim(1, 9)
plt.ylim(-5, 5)
plt.savefig("regular_grid.png", transparent=True)

exit()

# 2 2.96 0
# Simplex points:
# 0:	2.03617 2.93014 -0.0210123
# 1:	2.08191 2.84863 -0.0242156
# 2:	1.84995 2.85399 0.0575521
# 3:	2.01938 3.06694 0.0140355
from mpl_toolkits.mplot3d import Axes3D
points = np.array([
    [2.03617, 2.93014, -0.0210123 ],
    [2.08191, 2.84863, -0.0242156  ],
    [1.84995, 2.85399, 0.0575521  ],
    [2.01938, 3.06694, 0.0140355  ],
    [2.03617, 2.93014, -0.0210123 ],
    [1.84995, 2.85399, 0.0575521  ],
    [2.08191, 2.84863, -0.0242156  ],
    [2.01938, 3.06694, 0.0140355  ],
])

for angle in range(0, 360, 5):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(30, angle)
    plt.plot(points[:,0], points[:,1], points[:,2])
    plt.plot([2], [2.96], [0], "ro")
    plt.show()

exit()



# y1 = 2/3
# x2 = 1/3
# y2 = 2/3
# x3 = 0.5
# y3 = 1/3
#
# def get_bary(x, y):
#     l1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))
#     l2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))
#     l3 = 1 - l1 - l2
#     return l1, l2, l3
#
# plt.figure(figsize=(10, 10))
# plt.plot([x1, x2, x3, x1], [y1, y2, y3, y1])
# # plt.xlim(0, 1)
# # plt.ylim(0, 1)
#
# for i in range(10):
#     for j in range(10):
#         x, y, z = get_bary(i/10, j/10)
#         plt.plot([i/10], [j/10], "k.")
#         plt.gca().annotate("({},{},{})".format(round(x, 2), round(y, 2), round(z, 2)), xy=(i/10, j/10), fontsize=8)
#
# plt.show()
# exit()


# mins = [2, -4, 19, 15]
# block_size = [7/100, 8/90, 2/90, 10/70]
# n_blocks = [100, 90, 80, 70]
#
# print(block_size)
#
# def get_id(coords):
#     id = 0
#     for i in (3, 2, 1, 0):
#         print("id:", id)
#         ith_ind = int((coords[i] - mins[i]) / block_size[i])
#         print("ith_ind:", ith_ind, "coord:", mins[i] + ith_ind * block_size[i])
#         id = ith_ind + n_blocks[i] * id
#
#     print("id:",id)
#     return id
#
# print(get_id([5, -2, 20, 20]))
# print()
# print(get_id([4.5, 2.25, 20.2, 19.87]))
# print()
#
# def get_coords(id):
#     coords = []
#     for i in range(4):
#         print("id",id)
#         coords.append(id % n_blocks[i])
#         print("ith_ind:", coords[i])
#         id = (id - coords[i]) / n_blocks[i]
#         coords[i] = round(mins[i] + coords[i] * block_size[i], 2)
#         print("coord:", coords[i])
#
#     print("id",id)
#     return coords
#
#
# print(get_coords(25607242))
# print()
# print(get_coords(24964035))
# print()
# # print(get_coords(get_id([5, -2, 20, 20])))
# exit()
#
# data = pd.read_csv("run15/data.csv")
# data = data.drop(["interpolated", "diff"], axis=1)
# # data.to_csv("data.csv", index=False)
# tri = spatial.Delaunay(data.loc[:, ["T", "nH", "hhT6", "SFR"]].to_numpy())
# print(tri.find_simplex(np.array([2, -4, 20, 20])))
# exit()