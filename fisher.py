"""Python script for computing the Fisher matrix"""

from camb import model, initialpower
import camb
import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interpolate
from astropy import constants as const
import sys
import platform
import os

l_speed = const.c.value / 1000


# Fiducial parameters

dict_fiducial = {'Omegam': 0.32, 'Omegab': 0.05, 'Omegade': 0.68, 'Omegach2': 0.12055785610846023,
                 'w0': -1.0, 'wa': 0, 'hubble': 0.67, 'ns': 0.96, 'sigma8': 0.815584, 'gamma': 0.55}

# Reading different matter power spectrums

dict_mps = {}
dict_mps['kh'] = np.loadtxt('mps_data/kh')
dict_mps['pk'] = np.loadtxt('mps_data/pk')
dict_mps['z'] = np.loadtxt('mps_data/z')

mps_linear = interpolate.RectBivariateSpline(
    dict_mps['z'], dict_mps['kh'], dict_mps['pk'])


# Functions required to obtain the fisher matrix


def E(z, fiducial=dict_fiducial):
    locals().update(fiducial)
    mat = Omegam * (1 + z) ** 3
    de = Omegade * (1 + z) ** (3 * (1 + w0 + wa)) * \
        np.exp(-3 * wa * z / (1 + z))
    cur = (1 - Omegam - Omegade) * (1 + z) ** 2
    return np.sqrt(mat + de + cur)


def r(z, fiducial=dict_fiducial):
    """Comoving Distance as a function of the redshift
    trapz from numpy is used.
    ===================================================
    Input: redshift z
    Output: comoving distance r
    """
    H0 = fiducial['hubble'] * 100

    def integrand(u):
        return 1/E(u, fiducial)
    if type(z) == np.ndarray:
        integral = np.zeros(200)
        for idx, redshift in enumerate(z):
            z_int = np.linspace(0, redshift, 200)
            integral[idx] = np.trapz(integrand(z_int), z_int)
    else:
        z_int = np.linspace(0, z, 200)
        integral = np.trapz(integrand(z_int), z_int)
    return l_speed / H0 * integral


dict_ndsty = {}

# Interpolating the bin_number_density
for i in range(10):
    dict_ndsty['ndensity_file%s' % (str(i))] = np.loadtxt(
        'bin_ndensity/bin_%s' % (str(i)))
    dict_ndsty['bin_ndensity_%s' % (str(i))] = interpolate.interp1d(
        dict_ndsty['ndensity_file%s' % (str(i))][:, 0], dict_ndsty['ndensity_file%s' % (str(i))][:, 1])


def window(i, z, zmax=2.5, fiducial=dict_fiducial):
    """Reduced window function as a function of redshift.
    trapz from numpy is used for integration.
    ===================================================
    Inputs: bin number i, redshift z, min/max redshift 
    zmin/zmax (optional, default=0.001/2.5)
    Output: window function evaluated at z, of the ith bin.
    """
    def integrand(w):
        return dict_ndsty['bin_ndensity_%s' % (str(i))](w) * (1 - r(z, fiducial) / r(w, fiducial))

    z_int = np.linspace(z, zmax, 200)
    integral = np.trapz(integrand(z_int), z_int)

    return integral


def weight(i, z, fiducial=dict_fiducial):
    """Weight function of the ith bin, as a function of
    redshift.
    ===================================================
    Inputs: bin number i, redshift z, matter density 
    Omega_m (optional, default provided by CAMB)
    Output: weight function evaluated at z, of the ith bin
    """
    H0 = fiducial['hubble'] * 100
    constant = 3 / 2 * (H0 / l_speed) ** 2 * fiducial['Omegam']
    z_dep = (1 + z) * r(z, fiducial) * window(i, z, fiducial=fiducial)
    return constant * z_dep


def convergence(i, j, l, fiducial=dict_fiducial, zmin=0.001, zmax=2.5):
    H0 = fiducial['hubble'] * 100

    def integrand(z):
        term1 = weight(i, z, fiducial) * weight(j, z, fiducial) / \
            (E(z, fiducial) * r(z, fiducial) ** 2)
        k = (l + 1/2) / r(z)
        term2 = mps_linear(z, k)[0]
        return term1 * term2
    integral = integrate.quad(integrand, zmin, zmax)[0]
    return l_speed / H0 * integral


def error_convergence(i, j, l, fsky=(1/15000), dl=10):
    term1 = np.sqrt(2 / ((2 * l + 1) * dl * fsky))
    term2 = convergence(i, j, l) 
    return term1 * term2


def covariance(i, j, m, n, l1, l2, fsky=(1/15000), dl=10):
    if l1 != l2:
        value = 0
    else:
        num1 = cosmic_shear_array[l1, i, m] * cosmic_shear_array[l2, j, n]
        num2 = cosmic_shear_array[l1, i, n] * cosmic_shear_array[l2, j, m]
        value = (num1 + num2) / ((2*l1 + 1) * fsky * dl)


def d_convergence(i, j, l, param, dx=0.01, fiducial=dict_fiducial):
    param_l = (1 - dx) * fiducial[param]
    param_u = (1 + dx) * fiducial[param]
    fiducial_l = fiducial.copy()
    fiducial_u = fiducial.copy()
    fiducial_l = fiducial_l[param_l]
    fiducial_u = fiducial_u[param_u]
    return (convergence(i, j, l, fiducial=fiducial_u) - convergence(i, j, l, fiducial=fiducial_l)) / (2 * dx)

