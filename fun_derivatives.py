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


"""Fiducial parameters"""


dict_fiducial = {'Omegam': 0.32, 'Omegab': 0.05, 'Omegade': 0.68,
                 'w0': -1.0, 'wa': 0, 'hubble': 0.67, 'ns': 0.96, 'sigma8': 0.815584, 'gamma': 0.55}

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


def d_lnE(z, param, w_0=pars.DarkEnergy.w, w_a=pars.DarkEnergy.wa):
    den = 2 * E(z) ** 2
    if param == 'Omegam':
        return ((1 + z) ** 3 - (1 + z) ** 2) / den
    term1 = (1 + z) ** (3*(1 + w_0 + w_a)) * np.exp(-3 * w_0 * z / (1+z))
    if param == 'Omegade':
        return (term1 - (1 + z) ** 2) / den
    elif param == 'w0':
        Oml = omde + omk
        return (3 * Oml * term1 * np.log(1 + z)) / den
    elif param == 'wa':
        Oml = omde + omk
        return (3 * Oml * term1 * epsilon(z)) / den
    else:
        print("Please give a correct parameter, such as 'Omegam', 'Omegade', 'w0', 'wa'...")


def d_lnr(z, param, zmin=0.001, w_0=pars.DarkEnergy.w, w_a=pars.DarkEnergy.wa):
    den1 = - (pars.H0 * r(z)) / (2 * l_speed)
    if param == 'Omegam':
        def integrand(u):
            num = (1 + u) ** 3 - (1 + u) ** 2
            den = E(u) ** 3
            return num / den
        z_int = np.linspace(zmin, z, 200)
        integral = np.trapz(integrand(z_int), z_int)
        return integral / den1
    elif param == 'Omegade':
        def integrand(u):
            num = (1+u)**(3*(1 + w_0 + w_a)) * \
                np.exp(-3 * w_a * u / (1+u)) - (1 + u) ** 2
            den = E(u) ** 2
            return num / den
        z_int = np.linspace(zmin, z, 200)
        integral = np.trapz(integrand(z_int), z_int)
        return -3 * integral / den1
    elif param == 'w0':
        def integrand(u):
            num = omde * (1+u)**(3*(1 + w_0 + w_a)) * \
                np.exp(-3 * w_a * u / (1+u)) * np.log(1 + u)
            den = E(z) ** 3
            return num / den
        z_int = np.linspace(zmin, z, 200)
        integral = np.trapz(integrand(z_int), z_int)
        return -3 * integral / den1
    elif param == 'wa':
        def integrand(u):
            num = omde * (1+u)**(3*(1 + w_0 + w_a)) * \
                np.exp(-3 * w_a * u / (1+u)) * epsilon(z)
            den = E(z) ** 3
            return num / den
        z_int = np.linspace(zmin, z, 200)
        integral = np.trapz(integrand(z_int), z_int)
        return -3 * integral / den1
    elif param == 'h':
        return -1 / pars.h


def d_window(i, z, param, zmax=2.5):
    def integrand(u):
        term1 = dict_ndsty['bin_ndensity_%s' % (str(i))](u) * r(z) / r(u)
        term2 = d_lnr(u, param) - d_lnr(z, param)
        return term1 * term2
    z_int = np.linspace(z, zmax, 200)
    num = np.trapz(integrand(z_int), z_int)
    den = window(i, z)
    return num / den


def d_K(i, j, z, param, Omm=omm):
    term = - d_lnE(z, param) + d_window(i, z, param) + d_window(j, z, param)
    if param == 'Omegam':
        return 2 / Omm + term
    elif param == 'h':
        return 3 / pars.h
    else:
        return term


"""Power Spectrum"""


def d_param_kl(z, l, param):
    return -(l + 1/2) / (r(z) ** 3) * d_lnr(z, param)


def d_k_mps(z, l, dk=0.01):
    k = (l + 1/2) / r(z)
    return (mps_linear(z, k+dk) - mps_linear(z, k)) / dk


def d_param_mps(param, l=200, z=1,  fiducial=dict_fiducial):
    locals().update(fiducial)
    k = (l + 1/2) / r(z)
    param_u = 1 + fiducial[param] / 10
    param_l = 1 - fiducial[param] / 10
    mps_evaluated = np.zeros(200)
    for idx, val in enumerate(np.linspace(param_l, param_u, 200)):
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=fiducial['hubble']*100, ombh2=Omegab*hubble**2, tau=0.058)
        pars.InitPower.set_params(ns=ns)
        pars.set_matter_power(redshifts=np.linspace(0.001, 2.5, 101), kmax=50)

        pars.NonLinear = model.NonLinear_none
        results = camb.get_results(pars)
        kh, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=7, npoints = 200)
        mps_linear = interpolate.RectBivariateSpline(z, kh, pk)
        mps_evaluated[idx] = mps_linear(z, k)


