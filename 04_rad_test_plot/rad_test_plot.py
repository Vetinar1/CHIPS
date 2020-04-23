import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from labellines import labelLine, labelLines

RUN_LABEL = "10_rad_test"

data = pd.read_csv(
    "~/c17.01/runs/" + RUN_LABEL + "/" + RUN_LABEL + "-3.grid",
    sep="\t+",
    engine="python",
    header=0,
    #index_col=0,
    names=["hden", "temp"],
    usecols=[6, 7],
    comment="#"
)

cool = pd.read_csv(
    "~/c17.01/runs/" + RUN_LABEL + "/" + RUN_LABEL + "-3.cool_by_element",
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
    plt.loglog(
        10**data[(data["hden"] == hden)].loc[:, "temp"],
        data[(data["hden"] == hden)].loc[:, "Ctot"],
        color="black",
        label=hden
    )
    plt.loglog(
        10**data[(data["hden"] == hden)].loc[:, "temp"],
        data[(data["hden"] == hden)].loc[:, "CFe"],
        color="red",
        label=hden
    )

plt.title(r"$\Phi_{SFR} = $" + str(-3) + " at different $n_H$")
labelLines(plt.gca().get_lines(), align=False, xvals=(1e4, 1e8))
plt.xlabel(r"$\log(T)$")
plt.ylabel("Cooling")
plt.ylim(1e-32, 1e-13)
plt.show()
#plt.savefig(RUN_LABEL + "_" + str(i) + "_nH" + str(hden).replace(".", "_") + ".png")
plt.close()

#plt.loglog(data["temperature"], data["cooling"])
#plt.show()