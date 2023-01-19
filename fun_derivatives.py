from camb import model, initialpower
import camb
import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interpolate
from astropy import constants as const
import sys
import platform
import os

# Assume installed from github using "git clone --recursive https://github.com/cmbant/CAMB.git"
# This file is then in the docs folders. Delete these two lines for pip/conda install.
camb_path = os.path.realpath(os.path.join(os.getcwd(), '..'))
sys.path.insert(0, camb_path)
print('Using CAMB %s installed at %s' %
      (camb.__version__, os.path.dirname(camb.__file__)))
# make sure the version and path is what you expect

l_speed = const.c.value / 1000

pars = model.CAMBparams()  # Set of parameters created
pars.set_dark_energy(w=-1.0, wa=0, dark_energy_model='fluid')
pars.set_cosmology(H0=67.4, ombh2=0.02233, omch2=0.1198, omk=0, tau=0.054)

results = camb.get_results(pars)
omde = results.get_Omega('de')
omk = pars.omk
omm = 1 - omde - omk

"""General functions defined"""


def E(z, Omm=omm, Omde=omde, OmK=omk, w_0=pars.DarkEnergy.w, w_a=pars.DarkEnergy.wa):
    mat = Omm * (1 + z) ** 3
    de = Omde * (1 + z) ** (3 * (1 + w_0 + w_a)) * \
        np.exp(-3 * w_a * z / (1 + z))
    cur = OmK * (1 + z) ** 2
    return np.sqrt(mat + de + cur)


def r(z):
    """Comoving Distance as a function of the redshift
    trapz from numpy is used.
    ===================================================
    Input: redshift z
    Output: comoving distance r
    """
    def integrand(u):
        return 1/E(u)
    if type(z) == np.ndarray:
        integral = np.zeros(200)
        for idx, redshift in enumerate(z):
            z_int = np.linspace(0, redshift, 200)
            integral[idx] = np.trapz(integrand(z_int), z_int)
    else:
        z_int = np.linspace(0, z, 200)
        integral = np.trapz(integrand(z_int), z_int)
    return l_speed / pars.H0 * integral


def D_A(z, Omk=omk):
    """Angular diameter distance as a function of the redshift.
    Note that there are different definitions for different
    space-time curvatures.
    ===================================================
    Inputs: redshift z; curvature density Omk, default given by camb (optional)
    Outputs: angular diameter distance D_A
    """
    if Omk == 0:
        return r(z) / (1 + z)

    arg = np.sqrt(np.abs(Omk)) * pars.H0 / l_speed * r(z)
    prop_const = l_speed / pars.H0 / np.sqrt(np.abs(Omk))

    if Omk < 0:
        return prop_const / (1 + z) * np.sin(arg)

    else:
        return prop_const / (1 + z) * np.sinh(arg)


z_equipop_bins = [(0.001, 0.42), (0.42, 0.56), (0.56, 0.68), (0.68, 0.79), (0.79, 0.90),
                  (0.90, 1.02), (1.02, 1.15), (1.15, 1.32), (1.32, 1.58), (1.58, 2.50)]


dict_ndsty = {}

# Interpolating the bin_number_density
for i in range(10):
    dict_ndsty['ndensity_file%s' % (str(i))] = np.loadtxt(
        'bin_ndensity/bin_%s' % (str(i)))
    dict_ndsty['bin_ndensity_%s' % (str(i))] = interpolate.interp1d(
        dict_ndsty['ndensity_file%s' % (str(i))][:, 0], dict_ndsty['ndensity_file%s' % (str(i))][:, 1])


def window(i, z, zmax=2.5):
    """Reduced window function as a function of redshift.
    trapz from numpy is used for integration.
    ===================================================
    Inputs: bin number i, redshift z, min/max redshift 
    zmin/zmax (optional, default=0.001/2.5)
    Output: window function evaluated at z, of the ith bin.
    """
    def integrand(w):
        return dict_ndsty['bin_ndensity_%s' % (str(i))](w) * (1 - r(z) / r(w))

    z_int = np.linspace(z, zmax, 200)
    integral = np.trapz(integrand(z_int), z_int)

    return integral


def weight_gamma(i, z, Omm=omm):
    """Weight function of the ith bin, as a function of
    redshift.
    ===================================================
    Inputs: bin number i, redshift z, matter density 
    Omega_m (optional, default provided by CAMB)
    Output: weight function evaluated at z, of the ith bin
    """
    constant = 3 / 2 * (pars.H0 / l_speed) ** 2 * Omm
    z_dep = (1 + z) * r(z) * window(i, z)
    return constant * z_dep


dict_mps = {}
dict_mps['kh'] = np.loadtxt('mps_data/kh')
dict_mps['kh_nonlin'] = np.loadtxt('mps_data/kh_nonlin')
dict_mps['pk'] = np.loadtxt('mps_data/pk')
dict_mps['pk_nonlin'] = np.loadtxt('mps_data/pk_nonlin')
dict_mps['z'] = np.loadtxt('mps_data/z')
dict_mps['z_nonlin'] = np.loadtxt('mps_data/z_nonlin')

mps_linear = interpolate.RectBivariateSpline(
    dict_mps['z'], dict_mps['kh'], dict_mps['pk'])
mps_nonlinear = interpolate.RectBivariateSpline(
    dict_mps['z_nonlin'], dict_mps['kh_nonlin'], dict_mps['pk_nonlin'])


def convergence_gammagamma(i, j, l, zmin=0.001, zmax=2.5):
    def integrand(z):
        term1 = weight_gamma(i, z) * weight_gamma(j, z) / (E(z) * r(z) ** 2)
        k = (l + 1/2) / r(z)
        term2 = mps_linear(z, k)[0]
        return term1 * term2
    integral = integrate.quad(integrand, zmin, zmax)[0]
    return l_speed / pars.H0 * integral


def error_convergence(i, j, l, fsky=(1/15000), dl=10):
    term1 = np.sqrt(2 / ((2 * l + 1) * dl * fsky))
    term2 = convergence_gammagamma(i, j, l)
    return term1 * term2


"""Defining derivatives for Fisher matrix"""


def epsilon(z):
    return np.log(1 + z) - z / (1 + z)


def d_omm_lnE(z):
    num = (1 + z) ** 3 - (1 + z) ** 2
    den = 2 * E(z) ** 2
    return num / den


def d_omde_lnE(z, w_0=pars.DarkEnergy.w, w_a=pars.DarkEnergy.wa):
    num = (1+z)**(3*(1 + w_0 + w_a)) * \
        np.exp(-3 * w_0 * z / (1+z)) - (1 + z) ** 2
    den = 2 * E(z) ** 2
    return num / den


def d_w0_lnE(z, w_0=pars.DarkEnergy.w, w_a=pars.DarkEnergy.wa):
    Oml = omde + omk
    term1 = 3 * Oml * (1 + z) ** (3*(1 + w_0 + w_a))
    term2 = np.exp(-3 * w_a * z / (1 + z)) * np.log(1 + z)
    term3 = 2 * E(z) ** 2
    return term1 * term2 / term3


def d_wa_lnE(z, w_0=pars.DarkEnergy.w, w_a=pars.DarkEnergy.wa):
    Oml = omde + omk
    term1 = 3 * Oml * (1 + z) ** (3*(1 + w_0 + w_a))
    term2 = np.exp(-3 * w_a * z / (1 + z)) * epsilon(z)
    term3 = 2 * E(z) ** 2
    return term1 * term2 / term3


def d_omm_lnr(z, zmin=0.001):
    r_tilde = pars.H0 / l_speed * r(z)

    def integrand(u):
        num = (1 + u) ** 3 - (1 + u) ** 2
        den = E(u) ** 3
        return num / den
    z_int = np.linspace(zmin, z, 200)
    integral = np.trapz(integrand(z_int), z_int)
    return -integral / (2 * r_tilde)


def d_omde_lnr(z, zmin=0.001, w_0=pars.DarkEnergy.w, w_a=pars.DarkEnergy.wa):
    r_tilde = pars.H0 / l_speed * r(z)

    def integrand(u):
        num = (1+u)**(3*(1 + w_0 + w_a)) * \
            np.exp(-3 * w_a * u / (1+u)) - (1 + u) ** 2
        den = E(u) ** 2
        return num / den
    z_int = np.linspace(zmin, z, 200)
    integral = np.trapz(integrand(z_int), z_int)
    return -3 * integral / (2 * r_tilde)


def d_w0_lnr(z, zmin=0.001, Omde=omde, w_0=pars.DarkEnergy.w, w_a=pars.DarkEnergy.wa):
    r_tilde = pars.H0 / l_speed * r(z)

    def integrand(u):
        num = Omde * (1+u)**(3*(1 + w_0 + w_a)) * \
            np.exp(-3 * w_a * u / (1+u)) * np.log(1 + u)
        den = E(z) ** 3
        return num / den
    z_int = np.linspace(zmin, z, 200)
    integral = np.trapz(integrand(z_int), z_int) 
    return -3 * integral / (2 * r_tilde)


def d_wa_lnr(z, zmin=0.001, Omde=omde, w_0=pars.DarkEnergy.w, w_a=pars.DarkEnergy.wa):
    r_tilde = pars.H0 / l_speed * r(z)

    def integrand(u):
        num = Omde * (1+u)**(3*(1 + w_0 + w_a)) * \
            np.exp(-3 * w_a * u / (1+u)) * epsilon(z)
        den = E(z) ** 3
        return num / den
    z_int = np.linspace(zmin, z, 200)
    integral = np.trapz(integrand(z_int), z_int)
    return -3 * integral / (2 * r_tilde)


d_h_lnr = -1 / pars.h


def d_omm_window(i, z, zmax=2.5):
    def integrand(u):
        term1 = dict_ndsty['bin_ndensity_%s' % (str(i))](u) * r(z) / r(u)
        term2 = d_omm_lnr(u) - d_omm_lnr(z)
        return term1 * term2
    z_int = np.linspace(z, zmax, 200)
    num = np.trapz(integrand(z_int), z_int)
    den = window(i, z)
    return num / den


def d_omde_window(i, z, zmax=2.5):
    def integrand(u):
        term1 = dict_ndsty['bin_ndensity_%s' % (str(i))](u) * r(z) / r(u)
        term2 = d_omde_lnr(u) - d_omde_lnr(z)
        return term1 * term2
    z_int = np.linspace(z, zmax, 200)
    num = np.trapz(integrand(z_int), z_int)
    den = window(i, z)
    return num / den


def d_w0_window(i, z, zmax=2.5):
    def integrand(u):
        term1 = dict_ndsty['bin_ndensity_%s' % (str(i))](u) * r(z) / r(u)
        term2 = d_w0_lnr(u) - d_w0_lnr(z)
        return term1 * term2
    z_int = np.linspace(z, zmax, 200)
    num = np.trapz(integrand(z_int), z_int)
    den = window(i, z)
    return num / den


def d_wa_window(i, z, zmax=2.5):
    def integrand(u):
        term1 = dict_ndsty['bin_ndensity_%s' % (str(i))](u) * r(z) / r(u)
        term2 = d_wa_lnr(u) - d_wa_lnr(z)
        return term1 * term2
    z_int = np.linspace(z, zmax, 200)
    num = np.trapz(integrand(z_int), z_int)
    den = window(i, z)
    return num / den


def d_omm_k(i, j, z, Omm=omm):
    return 2 / Omm - d_omm_lnE(z) + d_omm_window(i, z) + d_omm_window(j, z)


def d_omde_k(i, j, z):
    return -d_omde_lnE(z) + d_omde_window(i, z) + d_omde_window(j, z)


def d_w0_k(i, j, z):
    return -d_w0_lnE(z) + d_w0_window(i, z) + d_w0_window(j, z)


def d_wa_k(i, j, z):
    return -d_wa_lnE(z) + d_wa_window(i, z) + d_wa_window(j, z)


d_h_k = 3 / pars.h
