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
    Quad integration from scipy is used.
    ===================================================
    Input: redshift z
    Output: comoving distance r
    """
    def integrand(u):
        return 1/E(u)
    integral = integrate.quad(integrand, 0, z)[0]
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
    Quad integration from scipy is used.
    ===================================================
    Inputs: bin number i, redshift z, min/max redshift 
    zmin/zmax (optional, default=0.001/2.5)
    Output: window function evaluated at z, of the ith bin.
    """
    def integrand(w):
        return dict_ndsty['bin_ndensity_%s' % (str(i))](w) * (1 - r(z) / r(w))

    return integrate.quad(integrand, z, zmax, limit=100)[0]


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


def error_convergence(i, j, l, fsky=1.0, dl=10):
    term1 = np.sqrt(2 / ((2 * l + 1) * dl * fsky))
    term2 = convergence_gammagamma(i, j, l)
    return term1 * term2


"""Defining derivatives for Fisher matrix"""


def d_omm_lnE(z):
    num = (1 + z) ** 3 - (1 + z) ** 2
    den = 2 * E(z) ** 2
    return num / den


def d_omde_lnE(z, w_0=pars.DarkEnergy.w, w_a=pars.DarkEnergy.wa):
    num = (1+z)**(3*(1 + w_0 + w_a)) * np.exp(-3 * w_0 * z / (1+z)) - (1 + z) ** 2
    den = 2 * E(z) ** 2
    return num / den


def d_w0_lnE(z, Oml=results.get_Omega('de'), w_0=pars.DarkEnergy.w, w_a=pars.DarkEnergy.wa):
    term1 = 3 * Oml * (1 + z) ** (3*(1 + w_0 + w_a))
    term2 = np.exp(-3 * w_a * z / (1 + z)) * np.log(1 + z)
    term3 = 2 * E(z) ** 2
    return term1 * term2 / term3


def epsilon(z):
    return np.log(1 + z) - z / (1 + z)


def d_wa_lnE(z, Oml=results.get_Omega('de'), w_0=pars.DarkEnergy.w, w_a=pars.DarkEnergy.wa):
    term1 = 3 * Oml * (1 + z) ** (3*(1 + w_0 + w_a))
    term2 = np.exp(-3 * w_a * z / (1 + z)) * epsilon(z)
    term3 = 2 * E(z) ** 2
    return term1 * term2 / term3
        