import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = np.loadtxt("bla", usecols=[0,1])
figure = plt.figure(figsize=(10,10))
plt.plot(data[:,0], data[:,1])
plt.xlabel(r"$x$ in AU")
plt.ylabel(r"$y$ in AU")
plt.xlim(-2, 2)
plt.ylim(-2, 2)
# ylabel.set_rotation(0)
plt.show()

data = np.loadtxt("bla", usecols=3)
figure = plt.figure(figsize=(5,5))
plt.plot(data)
plt.xlabel(r"Time")
plt.ylabel(r"Energy of system")
plt.xticks([])
plt.yticks([])
# ylabel.set_rotation(0)
plt.show()






exit()

RUN_LABEL = "03_fine"

data = pd.read_csv(
    "~/c17.01/runs/" + RUN_LABEL + "/" + RUN_LABEL + ".grid",
    sep="\t+",
    engine="python",
    header=0,
    #index_col=0,
    names=["Iron", "hden", "temp"],
    usecols=[6, 7, 8],
    comment="#"
)

cool = pd.read_csv(
    "~/c17.01/runs/" + RUN_LABEL + "/" + RUN_LABEL + ".cool_by_element",
    sep="\s+",
    header=0,
    names=["Ctot", "CFe"],
    usecols=[2, 28],
    comment="#"
)

data["Ctot"] = cool["Ctot"]
data["CFe"] = cool["CFe"]

print(data)

for i, hden in enumerate(pd.unique(data["hden"])):
    for iron in pd.unique(data["Iron"]):
        plt.loglog(
            10**data[(data["hden"] == hden) & (data["Iron"] == iron)].loc[:, "temp"],
            data[(data["hden"] == hden) & (data["Iron"] == iron)].loc[:, "Ctot"],
            color="black",
            label=r"$n_H$ = " + str(hden)
        )
        plt.loglog(
            10**data[(data["hden"] == hden) & (data["Iron"] == iron)].loc[:, "temp"],
            data[(data["hden"] == hden) & (data["Iron"] == iron)].loc[:, "CFe"],
            color="red",
            label=r"$n_H$ = " + str(hden)
        )

    plt.title(r"$n_H = $" + str(hden))
    plt.xlabel(r"$\log(T)$")
    plt.ylabel("Cooling")
    plt.show()
    #plt.savefig(RUN_LABEL + "_" + str(i) + "_nH" + str(hden).replace(".", "_") + ".png")
    plt.close()

#plt.loglog(data["temperature"], data["cooling"])
#plt.show()