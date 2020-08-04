import numpy as np
import matplotlib.pyplot as plt

# Purpose of this file is to build spectra "unit vectors":
# Spectrum files in the required format for cloudy that can be linearly combined
# Adapted from aura's full_lpf_spectra_for_cloudy.py

# TODO: Have aura verify this file is reasonably sensible

# TODO: Build UVB (Complicated due to z dependence)

# TODO: Where do these Phi come from?
Phins = 2.0e-2
Phios = 5.0e8
PhiT6 = 18.5
PhiT7 = 19.5
PhiT8 = 22.5


path_to_lpf_spectra = "aura/data_from_rahul/"

hPlanck = 6.626196e-27  # erg s
Rydberg = 1.09737312e5  # cm^-1
kpc2cm = 3.086e21


def save_sfr_spectrum(Phins):
    file_in = path_to_lpf_spectra + 'sfrx.dat'
    data = np.genfromtxt(file_in, comments='#', dtype=float)
    energy = (np.array(data[:, 0], dtype=float)).flatten()  # Ryd
    Lnu_init = (np.array(data[:, 1], dtype=float)).flatten()  # erg s^-1 Msun^-1 Hz^-1
    fnu = Lnu_init * Phins * 1.0e7 / kpc2cm ** 2 / (4. * np.pi)  # erg s^-1 Hz^-1 cm^-2

    savedata = np.hstack(
        (
            np.reshape(energy, (energy.shape[0], 1)),
            np.reshape(fnu, (fnu.shape[0], 1))
        )
    )
    np.savetxt(
        "spectra/sfr_spectrum",
        savedata,
        header="# E(Ryd) fnu(erg/s/Hz/cm^2)"
    )

    # plt.plot(savedata[:,0], savedata[:,1]/hPlanck)
    # plt.yscale("log")
    # plt.xscale("log")
    # plt.xlim(1e-1, 1e4)
    # plt.show()

    return savedata


def save_postAGB_spectrum(Phios):
    file_in = path_to_lpf_spectra + 'oldpopsed.dat'
    data = np.genfromtxt(file_in, comments='#', dtype=float)
    energy = (np.array(data[:, 0], dtype=float)).flatten()  # Ryd
    sed_pagb = (np.array(data[:, 1], dtype=float)).flatten()  # erg s^-1 Msun^-1 Hz^-1
    fnu = sed_pagb * Phios / kpc2cm ** 2 / (4. * np.pi)  # erg s^-1 Hz^-1 cm^-2

    savedata = np.hstack(
        (
            np.reshape(energy, (energy.shape[0], 1)),
            np.reshape(fnu, (fnu.shape[0], 1))
        )
    )
    np.savetxt(
        "spectra/postAGB_spectrum",
        savedata,
        header="# E(Ryd) fnu(erg/s/Hz/cm^2)"
    )

    # plt.plot(savedata[:,0], savedata[:,1]/hPlanck)
    # plt.yscale("log")
    # plt.xscale("log")
    # plt.show()

    return savedata


def save_hothaloT6_spectrum(PhiT6):
    file_in = path_to_lpf_spectra + 'hh6.dat'
    data = np.genfromtxt(file_in, comments='#', dtype=float)
    energy = (np.array(data[:, 0], dtype=float)).flatten()  # Ryd
    sed_hothalo = (np.array(data[:, 1], dtype=float)).flatten()  # 4*pi*J_nu/h = photons/s/cm^2
    fnu = hPlanck * (10. ** PhiT6) * sed_hothalo    # erg s^-1 Hz^-1 cm^-2

    savedata = np.hstack(
        (
            np.reshape(energy, (energy.shape[0], 1)),
            np.reshape(fnu, (fnu.shape[0], 1))
        )
    )
    np.savetxt(
        "spectra/hothaloT6_spectrum",
        savedata,
        header="# E(Ryd) fnu(erg/s/Hz/cm^2)"
    )

    # plt.plot(savedata[:,0], savedata[:,1]/hPlanck)
    # plt.yscale("log")
    # plt.xscale("log")
    # plt.show()

    return savedata


def save_hothaloT7_spectrum(PhiT7):
    file_in = path_to_lpf_spectra + 'hh7.dat'
    data = np.genfromtxt(file_in, comments='#', dtype=float)
    energy = (np.array(data[:, 0], dtype=float)).flatten()  # Ryd
    sed_hothalo = (np.array(data[:, 1], dtype=float)).flatten()  # 4*pi*J_nu/h = photons/s/cm^2
    fnu = hPlanck * (10. ** PhiT7) * sed_hothalo    # erg s^-1 Hz^-1 cm^-2

    savedata = np.hstack(
        (
            np.reshape(energy, (energy.shape[0], 1)),
            np.reshape(fnu, (fnu.shape[0], 1))
        )
    )
    np.savetxt(
        "spectra/hothaloT7_spectrum",
        savedata,
        header="# E(Ryd) fnu(erg/s/Hz/cm^2)"
    )

    # plt.plot(savedata[:,0], savedata[:,1]/hPlanck)
    # plt.yscale("log")
    # plt.xscale("log")
    # plt.show()

    return savedata


def save_hothaloT8_spectrum(PhiT8):
    file_in = path_to_lpf_spectra + 'hh8.dat'
    data = np.genfromtxt(file_in, comments='#', dtype=float)
    energy = (np.array(data[:, 0], dtype=float)).flatten()  # Ryd
    sed_hothalo = (np.array(data[:, 1], dtype=float)).flatten()  # 4*pi*J_nu/h = photons/s/cm^2
    fnu = hPlanck * (10. ** PhiT8) * sed_hothalo    # erg s^-1 Hz^-1 cm^-2

    savedata = np.hstack(
        (
            np.reshape(energy, (energy.shape[0], 1)),
            np.reshape(fnu, (fnu.shape[0], 1))
        )
    )
    np.savetxt(
        "spectra/hothaloT8_spectrum",
        savedata,
        header="# E(Ryd) fnu(erg/s/Hz/cm^2)"
    )

    # plt.plot(savedata[:,0], savedata[:,1]/hPlanck)
    # plt.yscale("log")
    # plt.xscale("log")
    # plt.show()

    return savedata


if __name__ == "__main__":
    sfr = save_sfr_spectrum(Phins)
    postAGB = save_postAGB_spectrum(Phios)
    hhT6 = save_hothaloT6_spectrum(PhiT6)
    hhT7 = save_hothaloT7_spectrum(PhiT7)
    hhT8 = save_hothaloT8_spectrum(PhiT8)

    plt.plot(sfr[:,0], sfr[:, 1]/hPlanck, label="SFR")
    plt.plot(postAGB[:,0], postAGB[:,1]/hPlanck, label="postAGB")
    plt.plot(hhT6[:,0], hhT6[:,1]/hPlanck, label=r"$\Phi_6")
    plt.plot(hhT7[:,0], hhT7[:,1]/hPlanck, label=r"$\Phi_7")
    plt.plot(hhT8[:,0], hhT8[:,1]/hPlanck, label=r"$\Phi_8")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(1e-1, 1e4)
    plt.ylim(1e-8, 1e12)
    plt.show()
