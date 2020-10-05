import pandas as pd
from cloudy_optimizer import compile_to_dataframe
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from util import sample_simplex, sample_simplices
from chips import optimizer

data = optimizer._load_existing_data(
    "run15/",
    "T_{T}__nH_{nH}__hhT6_{hhT6}__SFR_{SFR}",
    ["T", "nH", "hhT6", "SFR"]
)

data = optimizer.single_evaluation_step(
    data,
    param_space={
        "T":[2, 8],
        "nH":[-4, 4]
    },
    param_space_margins={
        "T":0.1,
        "nH":0.1
    },
    rad_params={
        "hhT6":("spectra/hhT6", [15, 25]),
        "SFR":("spectra/SFR", [15, 25])
    },
    rad_params_margins={
        "hhT6":0.1,
        "SFR":0.1
    },

)

data.to_csv("run15/data.csv", index=False)