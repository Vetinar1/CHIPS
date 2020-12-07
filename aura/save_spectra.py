import pickle, os, glob, pdb
import numpy as np
import scipy
import scipy.interpolate
import pylab as plt
from sys import stdin
import matplotlib.colors as colors
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import seaborn as sns
import matplotlib.gridspec as gridspec
import scipy.special
from scipy import linalg
from sklearn.cluster import MiniBatchKMeans
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from scipy.spatial import Voronoi
from scipy.spatial import ConvexHull
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
from itertools import cycle

hPlanck = 6.626196e-27  # erg s
Rydberg = 1.09737312e5  # cm^-1
kpc2cm = 3.086e21

# the combinations of lpf spectra from Kannan et al. 2016 (see Section 4)
bins_Phins = [1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 1.0e1, 1.0e2, 1.0e3]
bins_Phios = [1.0e6, 1.0e7, 1.0e8, 1.0e9, 1.0e10, 1.0e11, 1.0e12]
bins_PhiT6 = [17.5, 19.5, 21.5, 23.5]
bins_PhiT7 = [17.5, 19.5, 21.5, 23.5]
bins_PhiT8 = [17.5, 19.5, 21.5, 23.5]

index_Phins = range(1, len(bins_Phins) + 1)
index_Phios = range(1, len(bins_Phios) + 1)
index_PhiT6 = range(1, len(bins_PhiT6) + 1)
index_PhiT7 = range(1, len(bins_PhiT7) + 1)
index_PhiT8 = range(1, len(bins_PhiT8) + 1)

path_to_fg11_uvb = 'data_from_rahul/fg_uvb_dec11/'
path_to_lpf_spectra = 'data_from_rahul/'

# plt.switch_backend('agg')
plt.ion()


def sfr_spectrum(Phins=1, HMXB=False):
    file_in = path_to_lpf_spectra + 'sfrx.dat'
    data = np.genfromtxt(file_in, comments='#', dtype=float)
    energy = (np.array(data[:, 0], dtype=float)).flatten()  # Ryd
    Lnu_init = (np.array(data[:, 1], dtype=float)).flatten()  # erg s^-1 Msun^-1 Hz^-1
    fnu = Lnu_init * Phins * 1.0e7 / kpc2cm ** 2 / (4. * np.pi)  # erg s^-1 Hz^-1 cm^-2

    j = np.argmin(abs(energy - 1.))
    log10fnu_at1Ryd = np.log10(fnu[j])
    x_at1Ryd = energy[j]

    with open("../spectra/SFR", "w") as f:
        f.write("# E(Ryd) fnu(erg/s/Hz/cm^2)\n")
        for i in range(energy.shape[0]):
            f.write(f"{energy[i]} {fnu[i]}\n")

    return x_at1Ryd, log10fnu_at1Ryd, energy, fnu


def postAGB_spectrum(Phios=1, LMXB=False):
    file_in = path_to_lpf_spectra + 'oldpopsed.dat'
    data = np.genfromtxt(file_in, comments='#', dtype=float)
    energy_pagb = (np.array(data[:, 0], dtype=float)).flatten()  # Ryd
    sed_pagb = (np.array(data[:, 1], dtype=float)).flatten()  # erg s^-1 Msun^-1 Hz^-1
    fnu_pagb = sed_pagb * Phios / kpc2cm ** 2 / (4. * np.pi)  # erg s^-1 Hz^-1 cm^-2

    j = np.argmin(abs(energy_pagb - 1.))
    log10fnu_at1Ryd = np.log10(fnu_pagb[j])
    x_at1Ryd = energy_pagb[j]

    with open("../spectra/old", "w") as f:
        f.write("# E(Ryd) fnu(erg/s/Hz/cm^2)\n")
        for i in range(energy_pagb.shape[0]):
            f.write(f"{energy_pagb[i]} {fnu_pagb[i]}\n")

    return x_at1Ryd, log10fnu_at1Ryd, energy_pagb, fnu_pagb

sfr_spectrum(1)
exit()

def hothaloT6_spectrum(PhiT6=0):
    file_in = path_to_lpf_spectra + 'hh6.dat'
    data = np.genfromtxt(file_in, comments='#', dtype=float)
    energy_hothalo = (np.array(data[:, 0], dtype=float)).flatten()  # Ryd
    sed_hothalo = (np.array(data[:, 1], dtype=float)).flatten()  # 4*pi*J_nu/h = photons/s/cm^2
    fnu_hothalo = hPlanck * (10. ** PhiT6) * sed_hothalo

    with open("../spectra/hhT6", "w") as f:
        f.write("# E(Ryd) fnu(erg/s/Hz/cm^2)\n")
        for i in range(energy_hothalo.shape[0]):
            f.write(f"{energy_hothalo[i]} {fnu_hothalo[i]}\n")

    return energy_hothalo, fnu_hothalo


def hothaloT7_spectrum(PhiT7=0):
    file_in = path_to_lpf_spectra + 'hh7.dat'
    data = np.genfromtxt(file_in, comments='#', dtype=float)
    energy_hothalo = (np.array(data[:, 0], dtype=float)).flatten()  # Ryd
    sed_hothalo = (np.array(data[:, 1], dtype=float)).flatten()  # 4*pi*J_nu/h = photons/s/cm^2
    fnu_hothalo = hPlanck * (10. ** PhiT7) * sed_hothalo

    with open("../spectra/hhT7", "w") as f:
        f.write("# E(Ryd) fnu(erg/s/Hz/cm^2)\n")
        for i in range(energy_hothalo.shape[0]):
            f.write(f"{energy_hothalo[i]} {fnu_hothalo[i]}\n")

    return energy_hothalo, fnu_hothalo


def hothaloT8_spectrum(PhiT8=0):
    file_in = path_to_lpf_spectra + 'hh8.dat'
    data = np.genfromtxt(file_in, comments='#', dtype=float)
    energy_hothalo = (np.array(data[:, 0], dtype=float)).flatten()  # Ryd
    sed_hothalo = (np.array(data[:, 1], dtype=float)).flatten()  # 4*pi*J_nu/h = photons/s/cm^2
    fnu_hothalo = hPlanck * (10. ** PhiT8) * sed_hothalo

    with open("../spectra/hhT8", "w") as f:
        f.write("# E(Ryd) fnu(erg/s/Hz/cm^2)\n")
        for i in range(energy_hothalo.shape[0]):
            f.write(f"{energy_hothalo[i]} {fnu_hothalo[i]}\n")

    return energy_hothalo, fnu_hothalo

sfr_spectrum()
postAGB_spectrum()
hothaloT6_spectrum()
hothaloT7_spectrum()
hothaloT8_spectrum()