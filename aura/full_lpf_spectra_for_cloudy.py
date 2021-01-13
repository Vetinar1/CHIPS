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


def uvb_fg11(file_in):
    file_in = path_to_fg11_uvb + file_in
    data = np.genfromtxt(file_in, comments='#')
    E_in_RYd = (np.array(data[:, 0], dtype=float)).flatten()  # Ryd
    spec = np.array(data[:, 1], dtype=float)  # Jnu (10^-21 erg s^-1 cm^-2 Hz^-1 sr^-1)

    Jnu_in_ergperspercm2perHz = 4 * np.pi * spec * 1.0e-21

    aux, indices = np.unique(E_in_RYd, return_index=True)
    E_in_RYd = E_in_RYd[indices]
    Jnu_in_ergperspercm2perHz = Jnu_in_ergperspercm2perHz[indices]

    j = np.argmin(abs(E_in_RYd - 1.))
    log10fnu_at1Ryd = np.log10(Jnu_in_ergperspercm2perHz[j])
    x_at1Ryd = E_in_RYd[j]

    f = open(file_in)
    line1 = f.readline()
    f.close()
    zin = float(line1[3:])
    print('The file corresponds to z=%.2f' % zin)

    return zin, x_at1Ryd, log10fnu_at1Ryd, E_in_RYd, Jnu_in_ergperspercm2perHz


def sfr_spectrum(Phins, HMXB=False):
    file_in = path_to_lpf_spectra + 'sfrx.dat'
    data = np.genfromtxt(file_in, comments='#', dtype=float)
    energy = (np.array(data[:, 0], dtype=float)).flatten()  # Ryd
    Lnu_init = (np.array(data[:, 1], dtype=float)).flatten()  # erg s^-1 Msun^-1 Hz^-1
    fnu = Lnu_init * Phins * 1.0e7 / kpc2cm ** 2 / (4. * np.pi)  # erg s^-1 Hz^-1 cm^-2

    j = np.argmin(abs(energy - 1.))
    log10fnu_at1Ryd = np.log10(fnu[j])
    x_at1Ryd = energy[j]

    return x_at1Ryd, log10fnu_at1Ryd, energy, fnu


def postAGB_spectrum(Phios, LMXB=False):
    file_in = path_to_lpf_spectra + 'oldpopsed.dat'
    data = np.genfromtxt(file_in, comments='#', dtype=float)
    energy_pagb = (np.array(data[:, 0], dtype=float)).flatten()  # Ryd
    sed_pagb = (np.array(data[:, 1], dtype=float)).flatten()  # erg s^-1 Msun^-1 Hz^-1
    fnu_pagb = sed_pagb * Phios / kpc2cm ** 2 / (4. * np.pi)  # erg s^-1 Hz^-1 cm^-2

    j = np.argmin(abs(energy_pagb - 1.))
    log10fnu_at1Ryd = np.log10(fnu_pagb[j])
    x_at1Ryd = energy_pagb[j]

    return x_at1Ryd, log10fnu_at1Ryd, energy_pagb, fnu_pagb


def hothaloT6_spectrum(PhiT6):
    file_in = path_to_lpf_spectra + 'hh6.dat'
    data = np.genfromtxt(file_in, comments='#', dtype=float)
    energy_hothalo = (np.array(data[:, 0], dtype=float)).flatten()  # Ryd
    sed_hothalo = (np.array(data[:, 1], dtype=float)).flatten()  # 4*pi*J_nu/h = photons/s/cm^2
    fnu_hothalo = hPlanck * (10. ** PhiT6) * sed_hothalo

    return energy_hothalo, fnu_hothalo


def hothaloT7_spectrum(PhiT7):
    file_in = path_to_lpf_spectra + 'hh7.dat'
    data = np.genfromtxt(file_in, comments='#', dtype=float)
    energy_hothalo = (np.array(data[:, 0], dtype=float)).flatten()  # Ryd
    sed_hothalo = (np.array(data[:, 1], dtype=float)).flatten()  # 4*pi*J_nu/h = photons/s/cm^2
    fnu_hothalo = hPlanck * (10. ** PhiT7) * sed_hothalo

    return energy_hothalo, fnu_hothalo


def hothaloT8_spectrum(PhiT8):
    file_in = path_to_lpf_spectra + 'hh8.dat'
    data = np.genfromtxt(file_in, comments='#', dtype=float)
    energy_hothalo = (np.array(data[:, 0], dtype=float)).flatten()  # Ryd
    sed_hothalo = (np.array(data[:, 1], dtype=float)).flatten()  # 4*pi*J_nu/h = photons/s/cm^2
    fnu_hothalo = hPlanck * (10. ** PhiT8) * sed_hothalo

    return energy_hothalo, fnu_hothalo


def make_fig1_in_obreja2019():
    zin = 0.
    Phins = 2.0e-2
    Phios = 5.0e8
    PhiT6 = 18.5

    files_uvb = glob.glob(path_to_fg11_uvb + 'fg_uvb_dec11_z_*.dat')
    z_uvb = []
    for fl in files_uvb:
        f = open(fl)
        line1 = f.readline()
        f.close()
        z_uvb.append(float(line1[3:]))
    z_uvb = np.array(z_uvb)
    i = np.argmin(abs(zin - z_uvb))
    zout = z_uvb[i]
    data = np.genfromtxt(files_uvb[i], comments='#')
    energy_uvb = (np.array(data[:, 0], dtype=float)).flatten()  # Ryd
    spec_uvb = np.array(data[:, 1], dtype=float)  # Jnu (10^-21 erg s^-1 cm^-2 Hz^-1 sr^-1)
    fnu_uvb = 4 * np.pi * spec_uvb * 1.0e-21
    aux, indices = np.unique(energy_uvb, return_index=True)
    energy_uvb = energy_uvb[indices]  # Ryd
    fnu_uvb = fnu_uvb[indices]  # 4*pi*Jnu (erg s^-1 cm^-2 Hz^-1)

    x_at1Ryd, log10fnu_at1Ryd, energy_sfr, fnu_sfr = sfr_spectrum(Phins)
    x_at1Ryd, log10fnu_at1Ryd, energy_old, fnu_old = postAGB_spectrum(Phios)
    energy_hothaloT6, fnu_hothaloT6 = hothaloT6_spectrum(PhiT6)

    files_uvb = glob.glob(path_to_fg11_uvb + 'fg_uvb_dec11_z_*.dat')
    z_uvb2 = []
    for fl in files_uvb:
        f = open(fl)
        line1 = f.readline()
        f.close()
        z_uvb2.append(float(line1[3:]))
    z_uvb2 = np.array(z_uvb2)
    i = np.argmin(abs(2.0 - z_uvb2))
    zout2 = z_uvb[i]
    data = np.genfromtxt(files_uvb[i], comments='#')
    energy_uvb2 = (np.array(data[:, 0], dtype=float)).flatten()  # Ryd
    spec_uvb2 = np.array(data[:, 1], dtype=float)  # Jnu (10^-21 erg s^-1 cm^-2 Hz^-1 sr^-1)
    fnu_uvb2 = 4 * np.pi * spec_uvb2 * 1.0e-21
    aux, indices = np.unique(energy_uvb2, return_index=True)
    energy_uvb2 = energy_uvb2[indices]  # Ryd
    fnu_uvb2 = fnu_uvb2[indices]  # 4*pi*Jnu (erg s^-1 cm^-2 Hz^-1)

    plt.close()
    sns.set_style("ticks")
    sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})
    fig = plt.figure(figsize=(8.0, 6.0))
    gs = gridspec.GridSpec(1, 1)
    ax = plt.subplot(gs[0])
    plt.setp(ax.get_xticklabels(), fontsize=18)
    plt.setp(ax.get_yticklabels(), fontsize=18)
    ax.xaxis.labelpad = 2
    ax.yaxis.labelpad = 2
    ax.set_ylabel(r"4$\rm\pi$J$_{\rm\nu}$/h [photons s$^{\rm -1}$ cm$^{\rm -2}$]", fontsize=18)
    ax.set_xlabel(r"E [Ryd]", fontsize=18)
    gs.update(left=0.15, bottom=0.10, right=0.97, top=0.90)
    ax.loglog(energy_uvb2, fnu_uvb2 / hPlanck, color='limegreen', ls='--', lw=2, alpha=0.8,
              label=r"UVB @ z=%.2f (Faucher Gigu$\rm\`{e}$re et al. 2009)" % zout2)
    ax.loglog(energy_uvb, fnu_uvb / hPlanck, color='limegreen', ls='-', lw=2, alpha=0.8,
              label=r"UVB @ z=%.2f (Faucher Gigu$\rm\`{e}$re et al. 2009)" % zout)
    ax.loglog(energy_sfr, 0.05 * fnu_sfr / hPlanck, color='cornflowerblue', ls='-', lw=2, alpha=0.8,
              label=r"SFR=2M$_{\rm\odot}$/yr, d=10kpc (Cervi$\rm\~{n}$o et al. 2002), f$_{\rm esc}$=0.05")
    ax.loglog(energy_old, fnu_old / hPlanck, color='crimson', ls='-', lw=2, alpha=0.8,
              label=r"M$_{\rm old*}$=5x10$^{\rm 10}$M$_{\rm\odot}$, d=10kpc (Bruzual & Charlot 2003)")
    ax.loglog(energy_hothaloT6, fnu_hothaloT6 / hPlanck, color='blueviolet', ls='-', lw=2, alpha=1.0,
              label=r"M$_{\rm gas}$=10$^{\rm 9}$M$_{\rm\odot}$ with T=10$^{\rm 6}$K & n=10$^{\rm -2.5}$cm$^{\rm -2}$, d=10kpc")
    ax.plot(np.array([1., 1.]), np.array([1., 1.0e10]), ls=':', lw=2, color='black', zorder=-1)
    ax.set_ylim(1.0e-8, 1.0e10)
    ax.set_xlim(2.0e-3, 1.0e4)
    ax.legend(loc=3, fontsize=12, frameon=False, labelspacing=0.6, borderaxespad=0.5)
    plt.savefig('obreja2019_fig1.png')
    plt.close()

    return


def combine_6spectra(zin, Phins, Phios, PhiT6, PhiT7, PhiT8, base_name='cloudy_input/fg11z0.1_1_1_1_1'):
    # combines one UVB spectra with the 5 lpf ones
    # the UVB is selected by redshift zin
    # the lpf spectra are selected based on the normalizations:
    # - Phins in Msol/yr/kpc^2
    # - Phios in Msol/kpc^2
    # - PhiT6 in cm^-5
    # - PhiT7 in cm^-5
    # - PhiT8 in cm^-5

    plt.close()
    sns.set_style("ticks")
    sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})
    fig = plt.figure(figsize=(8.0, 6.0))
    gs = gridspec.GridSpec(1, 1)
    ax = plt.subplot(gs[0])
    plt.setp(ax.get_xticklabels(), fontsize=18)
    plt.setp(ax.get_yticklabels(), fontsize=18)
    ax.xaxis.labelpad = 2
    ax.yaxis.labelpad = 2
    ax.set_ylabel(r"4$\rm\pi$J$_{\rm\nu}$/h [photons s$^{\rm -1}$ cm$^{\rm -2}$]", fontsize=18)
    ax.set_xlabel(r"E [Ryd]", fontsize=18)
    gs.update(left=0.15, bottom=0.10, right=0.97, top=0.90)

    files_uvb = glob.glob(path_to_fg11_uvb + 'fg_uvb_dec11_z_*.dat')
    # for each UVB file, get first element from righ tcolumn
    z_uvb = []
    for fl in files_uvb:
        print("Loading file " + fl + "...")
        f = open(fl)
        line1 = f.readline()
        f.close()
        z_uvb.append(float(line1[3:]))
    z_uvb = np.array(z_uvb)

    # find file that matches the given z best; this is now our z
    i = np.argmin(abs(zin - z_uvb))
    zout = z_uvb[i]

    print("Loading chosen data file")
    # Load all data from that file
    data = np.genfromtxt(files_uvb[i], comments='#')
    # left column = energy
    energy_uvb = (np.array(data[:, 0], dtype=float)).flatten()  # Ryd
    # right column: spectrum (J_ny)
    spec_uvb = np.array(data[:, 1], dtype=float)  # Jnu (10^-21 erg s^-1 cm^-2 Hz^-1 sr^-1)
    # Convert to f_ny (what quantity is this?)
    fnu_uvb = 4 * np.pi * spec_uvb * 1.0e-21
    # keep only unique energies (in original order)
    aux, indices = np.unique(energy_uvb, return_index=True)
    energy_uvb = energy_uvb[indices]  # Ryd
    # get matching f_nu
    fnu_uvb = fnu_uvb[indices]  # 4*pi*Jnu (erg s^-1 cm^-2 Hz^-1)

    print("Getting spectra...")
    x_at1Ryd, log10fnu_at1Ryd, energy_sfr, fnu_sfr = sfr_spectrum(Phins)
    x_at1Ryd, log10fnu_at1Ryd, energy_old, fnu_old = postAGB_spectrum(Phios)
    energy_hothaloT6, fnu_hothaloT6 = hothaloT6_spectrum(PhiT6)
    energy_hothaloT7, fnu_hothaloT7 = hothaloT7_spectrum(PhiT7)
    energy_hothaloT8, fnu_hothaloT8 = hothaloT8_spectrum(PhiT8)

    # plot spectra
    print("Plotting spectra...")
    ax.loglog(energy_uvb, fnu_uvb / hPlanck, color='limegreen', ls='-', lw=2, alpha=0.8, label=r"UVB @ z=%.2f" % zout)
    ax.loglog(energy_sfr, 0.05 * fnu_sfr / hPlanck, color='cornflowerblue', ls='-', lw=2, alpha=0.8,
              label=r"$\phi_{\rm SFR}$ = 10$^{\rm %.1f}$M$_{\rm\odot}$yr$^{\rm -1}$kpc$^{\rm -2}$, f$_{\rm esc}$=0.05" % np.log10(
                  Phins))
    ax.loglog(energy_old, fnu_old / hPlanck, color='crimson', ls='-', lw=2, alpha=0.8,
              label=r"$\phi_{\rm old}$ = 10$^{\rm %.1f}$M$_{\rm\odot}$kpc$^{\rm -2}$" % np.log10(Phios))
    ax.loglog(energy_hothaloT6, fnu_hothaloT6 / hPlanck, color='violet', ls='-', lw=2, alpha=1.0,
              label=r"$\phi_{\rm 6}$ = 10$^{\rm %.1f}$cm$^{\rm -5}$" % PhiT6)
    ax.loglog(energy_hothaloT7, fnu_hothaloT7 / hPlanck, color='blueviolet', ls='-', lw=2, alpha=1.0,
              label=r"$\phi_{\rm 7}$ = 10$^{\rm %.1f}$cm$^{\rm -5}$" % PhiT7)
    ax.loglog(energy_hothaloT8, fnu_hothaloT8 / hPlanck, color='purple', ls='-', lw=2, alpha=1.0,
              label=r"$\phi_{\rm 8}$ = 10$^{\rm %.1f}$cm$^{\rm -5}$" % PhiT8)

    print("Interpolating spectra...")
    # To construct the final spectrum, choose from the input SEDs the one with the largest range in energies, and
    # set the min and max of the returned wavelength array accordingly 
    energy_min = np.min(np.array([min(energy_uvb), min(energy_sfr), min(energy_old),
                                  min(energy_hothaloT6), min(energy_hothaloT7), min(energy_hothaloT8)]))
    energy_max = np.max(np.array([max(energy_uvb), max(energy_sfr), max(energy_old),
                                  max(energy_hothaloT6), max(energy_hothaloT7), max(energy_hothaloT8)]))
    n_energy_bins = int(np.max(np.array([len(energy_uvb), len(energy_sfr), len(energy_old),
                                         len(energy_hothaloT6), len(energy_hothaloT7), len(energy_hothaloT8)]))) - 1
    log_energy = np.linspace(np.log10(energy_min), np.log10(energy_max),
                             num=n_energy_bins)  # array of wavelengths to be returned
    fnu_total = np.zeros(len(log_energy))  # array of fluxes to be returned

    # Interpolate the UVB and sum it up to fnu_total
    print("Interpolating UVB...")
    interp_uvb = scipy.interpolate.InterpolatedUnivariateSpline(np.log10(energy_uvb), np.log10(fnu_uvb), k=1, ext=1)
    log_fnu_uvb_interp = interp_uvb(
        log_energy[((log_energy >= min(np.log10(energy_uvb))) & (log_energy <= max(np.log10(energy_uvb))))])
    fnu_total[((log_energy >= min(np.log10(energy_uvb))) & (log_energy <= max(np.log10(energy_uvb))))] = fnu_total[(
                (log_energy >= min(np.log10(energy_uvb))) & (
                    log_energy <= max(np.log10(energy_uvb))))] + 10 ** log_fnu_uvb_interp

    # Interpolate the spectrum from star forming regions and sum it up to fnu_total
    print("Interpolating star forming regions...")
    interp_sfr = scipy.interpolate.InterpolatedUnivariateSpline(np.log10(energy_sfr), np.log10(fnu_sfr), k=1, ext=1)
    log_fnu_sfr_interp = interp_sfr(
        log_energy[((log_energy >= min(np.log10(energy_sfr))) & (log_energy <= max(np.log10(energy_sfr))))])
    fnu_total[((log_energy >= min(np.log10(energy_sfr))) & (log_energy <= max(np.log10(energy_sfr))))] = fnu_total[(
                (log_energy >= min(np.log10(energy_sfr))) & (
                    log_energy <= max(np.log10(energy_sfr))))] + 10 ** log_fnu_sfr_interp

    # Interpolate the spectrum from old stars and sum it up to fnu_total
    print("Interpolating old stars...")
    u, indices = np.unique(energy_old, return_index=True)
    energy_old = energy_old[indices]
    fnu_old = fnu_old[indices]
    interp_old = scipy.interpolate.InterpolatedUnivariateSpline(np.log10(energy_old), np.log10(fnu_old), k=1, ext=1)
    log_fnu_old_interp = interp_old(
        log_energy[((log_energy >= min(np.log10(energy_old))) & (log_energy <= max(np.log10(energy_old))))])
    fnu_total[((log_energy >= min(np.log10(energy_old))) & (log_energy <= max(np.log10(energy_old))))] = fnu_total[(
                (log_energy >= min(np.log10(energy_old))) & (
                    log_energy <= max(np.log10(energy_old))))] + 10 ** log_fnu_old_interp

    # Interpolate the spectrum from gas at 10^6 K and sum it up to fnu_total
    print("Interpolating hot gas...")
    interp_T6 = scipy.interpolate.InterpolatedUnivariateSpline(np.log10(energy_hothaloT6), np.log10(fnu_hothaloT6), k=1,
                                                               ext=1)
    log_fnu_T6_interp = interp_T6(
        log_energy[((log_energy >= min(np.log10(energy_hothaloT6))) & (log_energy <= max(np.log10(energy_hothaloT6))))])
    fnu_total[((log_energy >= min(np.log10(energy_hothaloT6))) & (log_energy <= max(np.log10(energy_hothaloT6))))] = \
    fnu_total[((log_energy >= min(np.log10(energy_hothaloT6))) & (
                log_energy <= max(np.log10(energy_hothaloT6))))] + 10 ** log_fnu_T6_interp

    # Interpolate the spectrum from gas at 10^7 K and sum it up to fnu_total
    print("Interpolating hotter gas...")
    interp_T7 = scipy.interpolate.InterpolatedUnivariateSpline(np.log10(energy_hothaloT7), np.log10(fnu_hothaloT7), k=1,
                                                               ext=1)
    log_fnu_T7_interp = interp_T7(
        log_energy[((log_energy >= min(np.log10(energy_hothaloT7))) & (log_energy <= max(np.log10(energy_hothaloT7))))])
    fnu_total[((log_energy >= min(np.log10(energy_hothaloT7))) & (log_energy <= max(np.log10(energy_hothaloT7))))] = \
    fnu_total[((log_energy >= min(np.log10(energy_hothaloT7))) & (
                log_energy <= max(np.log10(energy_hothaloT7))))] + 10 ** log_fnu_T7_interp

    # Interpolate the spectrum from gas at 10^8 K and sum it up to fnu_total
    print("Interpolating hottest gas...")
    interp_T8 = scipy.interpolate.InterpolatedUnivariateSpline(np.log10(energy_hothaloT8), np.log10(fnu_hothaloT8), k=1,
                                                               ext=1)
    log_fnu_T8_interp = interp_T8(
        log_energy[((log_energy >= min(np.log10(energy_hothaloT8))) & (log_energy <= max(np.log10(energy_hothaloT8))))])
    fnu_total[((log_energy >= min(np.log10(energy_hothaloT8))) & (log_energy <= max(np.log10(energy_hothaloT8))))] = \
    fnu_total[((log_energy >= min(np.log10(energy_hothaloT8))) & (
                log_energy <= max(np.log10(energy_hothaloT8))))] + 10 ** log_fnu_T8_interp

    ax.plot(10 ** log_energy, fnu_total / hPlanck, color='lightgrey', ls='-', lw=8, alpha=1, label=r"total", zorder=-1)
    ax.plot(np.array([1., 1.]), np.array([1., 1.0e10]), ls=':', lw=2, color='black', zorder=-1)
    ax.set_ylim(1.0e-8, 1.0e12)
    ax.set_xlim(2.0e-3, 1.0e4)
    ax.legend(loc=3, fontsize=12, frameon=False, labelspacing=0.6, borderaxespad=0.5)
    #pdb.set_trace()
    #plt.savefig(base_name + '.png')
    plt.show()
    plt.close()

    print("Writing cloudy file...")
    energy = 10 ** log_energy
    j = np.argmin(abs(energy - 1.0))
    split_name = base_name.split('/')
    prefix = split_name[-1]
    outfile = open(base_name + '.in', "w")
    # outfile.write('CMB redshift %.2f\n' % zin)
    # outfile.write('# set metallicity\n')
    # outfile.write('metals 0 # This does nothing... scales solar abundances * 1\n')
    # outfile.write('# Vary Iron\n')
    # outfile.write('element abundance iron -4.55 #vary\n')
    # outfile.write('#grid -5 -4 0.2 ncpus 8\n')
    # outfile.write('# next 4 commands vary n and Te\n')
    # outfile.write('hden -2 vary\n')
    # outfile.write('grid -4.0 4.0 1\n')
    # outfile.write('constant temperature 5 vary\n')
    # outfile.write('grid 4.0 8.0 0.1\n')
    # outfile.write('# must stop this constant Te model\n')
    # outfile.write('stop zone 1\n')
    # outfile.write('# print the results of the last iteration only\n')
    # outfile.write('print last\n')
    # outfile.write('print short\n')
    # outfile.write('set save prefix "%s"\n' % prefix)
    # # outfile.write('save element hydrogen last ".Hionf"\n')
    # # outfile.write('save element oxygen last ".Oionf"\n')
    # # outfile.write('save element carbon last ".Cionf"\n')
    # # outfile.write('save element neon last ".Neionf"\n')
    # # outfile.write('save element magnesium last ".Mgionf"\n')
    # # outfile.write('save element silicon last ".Siionf"\n')
    # outfile.write('save grid last ".grid"\n')
    # outfile.write('save overview last ".overview"\n')
    # outfile.write('save cooling last ".cool"\n')
    # outfile.write('save heating last ".heat"\n')
    # outfile.write('save cooling each last ".cool_by_element"\n')
    # #outfile.write('save continuum last ".continuum"\n')
    # outfile.write('# f(nu) is the normalization for the input continuum at ~1 Ryd\n')
    # outfile.write('# f(nu) units are log10(4piJ_nu/[erg/cm^2/s/Hz])\n')
    outfile.write('f(nu) = %.4f at %.5f\n' % (np.log10(fnu_total[j]), energy[j]))
    # outfile.write('# x = energy in Ryd, y = log10(4piJ_nu/[erg/cm^2/s/Hz])\n')
    outfile.write('interpolate (%11.5e  %12.5e)\n' % (energy[0], np.log10(fnu_total[0])))
    for j in range(len(energy) - 1): outfile.write(
        'continue (%11.5e  %12.5e)\n' % (energy[j + 1], np.log10(fnu_total[j + 1])))
    outfile.write('iterate to convergence\n')
    outfile.close()

    return


#############################################################################################################

# example of combined UVB+LPF spectrum together with the corresponding Cloudy input
# combine_6spectra(0., 2.0e-2, 5.0e8, 18.5, 19.5, 22.5, base_name='rad')

# for i in [-3, -2, -1]:
#     combine_6spectra(0., 10**i, 5.0e8, 18.5, 19.5, 21.5, base_name='cloudy_inputs/10_rad_test'+str(i))

# for x in np.linspace(-4, 1.0, 5):
#     combine_6spectra(0., 10**x, 5.0e8, 18.5, 19.5, 21.5, base_name='cloudy_inputs/03_fine')

# for x in np.linspace(7, 9, 5):
#     combine_6spectra(0., 2.0e-2, 10**x, 18.5, 19.5, 21.5, base_name='cloudy_inputs/03_fine')

# for x in np.linspace(15, 22, 5):
#     combine_6spectra(0., 2.0e-2, 5.0e8, x, 19.5, 21.5, base_name='cloudy_inputs/03_fine')

# for x in np.linspace(16, 23, 5):
#     combine_6spectra(0., 2.0e-2, 5.0e8, 18.5, x, 21.5, base_name='cloudy_inputs/03_fine')

#for x in [15, 22]: #np.linspace(18, 23, 5):
#    combine_6spectra(0., 2.0e-2, 5.0e8, 18.5, 19.5, x, base_name='cloudy_inputs/03_fine')

make_fig1_in_obreja2019()


# loop to get out the combined spectra used by Kannan et al. 2016 (for z=0 only)
# for Phins,id_Phins in zip(bins_Phins,index_Phins):
# for Phios,id_Phios in zip(bins_Phios,index_Phios):
# for PhiT6,id_PhiT6 in zip(bins_PhiT6,index_PhiT6):
# for PhiT7,id_PhiT7 in zip(bins_PhiT7,index_PhiT7):
# for PhiT8,id_PhiT8 in zip(bins_PhiT8,index_PhiT8):
# base_name = '/home/aura/cooling/fg11_lpf/cloudy_input/z0/fg11z0.%i_%i_%i_%i_%i'%(id_Phins,id_Phios,id_PhiT6,id_PhiT7,id_PhiT8)
# combine_6spectra(0.,Phins,Phios,PhiT6,PhiT7,PhiT8,base_name=base_name)


# print("Writing cloudy file...")
# energy = 10 ** log_energy
# j = np.argmin(abs(energy - 1.0))
# split_name = base_name.split('/')
# prefix = split_name[-1]
# outfile = open(base_name + '.in', "w")
# outfile.write('CMB redshift %.2f\n' % zin)
# outfile.write('# set metallicity\n')
# outfile.write('metals 0\n')
# outfile.write('# next 4 commands vary n and Te\n')
# outfile.write('hden -2 vary\n')
# outfile.write('grid -9.0 4.0 1.0 ncpus 4\n')
# outfile.write('constant temperature 5 vary\n')
# outfile.write('grid 2.0 8.0 0.1\n')
# outfile.write('# must stop this constant Te model\n')
# outfile.write('stop zone 1\n')
# outfile.write('# print the results of the last iteration only\n')
# outfile.write('print last\n')
# outfile.write('print short\n')
# outfile.write('set save prefix "%s"\n' % prefix)
# outfile.write('save element hydrogen last ".Hionf"\n')
# outfile.write('save element oxygen last ".Oionf"\n')
# outfile.write('save element carbon last ".Cionf"\n')
# outfile.write('save element neon last ".Neionf"\n')
# outfile.write('save element magnesium last ".Mgionf"\n')
# outfile.write('save element silicon last ".Siionf"\n')
# outfile.write('save grid last ".grid"\n')
# outfile.write('save overview last ".overview"\n')
# outfile.write('save cooling last ".cool"\n')
# outfile.write('save heating last ".heat"\n')
# outfile.write('save cooling each last ".cool_by_element"\n')
# outfile.write('save continuum last ".continuum"\n')
# outfile.write('# f(nu) is the normalization for the input continuum at ~1 Ryd\n')
# outfile.write('# f(nu) units are log10(4piJ_nu/[erg/cm^2/s/Hz])\n')
# outfile.write('f(nu) = %.4f at %.5f\n' % (np.log10(fnu_total[j]), energy[j]))
# outfile.write('# x = energy in Ryd, y = log10(4piJ_nu/[erg/cm^2/s/Hz])\n')
# outfile.write('interpolate (%11.5e  %12.5e)\n' % (energy[0], np.log10(fnu_total[0])))
# for j in range(len(energy) - 1): outfile.write(
#     'continue (%11.5e  %12.5e)\n' % (energy[j + 1], np.log10(fnu_total[j + 1])))
# outfile.write('iterate to convergence\n')