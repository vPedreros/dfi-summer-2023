# importancion de paquetes necesarios para el código

import sys
import os
import numpy as np
import scipy.integrate as integrate
import camb
from matplotlib import pyplot as plt
from scipy.stats import binned_statistic
import time
from camb import model
import pandas as pd
from astropy import constants as const
import scipy.interpolate as interpolate
from scipy.stats import linregress


# instalacion de camb

camb_path = os.path.realpath(os.path.join(os.getcwd(), '..'))
sys.path.insert(0, camb_path)


print('Using CAMB %s installed at %s' % (
      camb.__version__, os.path.dirname(camb.__file__)))


# Creación de funciones E(z) y D(z), a manera de prueba se ocuparan los
# valores de Planck 2016 para testear las funciones

params_P18 = dict()
# crearemos diccionarios en donde estaran los parametros cosmologicos que
# queremos utilizar, es facil poder crear y modificar diccionarios

params_P18['Ob'] = 0.05   # Omega_b_0
params_P18['Om'] = 0.32  # Omega_m_0
params_P18['ns'] = 0.96605       # indice espectral
params_P18['ODE'] = 0.68   # Omega_DE_0
params_P18['sigma8'] = 0.816    # amplitud de densidades de fluctuación
params_P18['H0'] = 67.32      # 100h
params_P18['sum_mv'] = 0.06  # valor de masas de neutrino
params_P18['w_0'] = -1
params_P18['w_a'] = 0
params_P18['gamma'] = 0.55
params_P18['Ov'] = 0  # en este caso tomamos la densidad de radiación como nula

# Creacion de parametros con CAMB

pars = camb.CAMBparams()

# This function sets up CosmoMC-like settings, with one massive neutrino and
# helium set using BBN consistency

pars.set_dark_energy(w=-1.0, wa=0, dark_energy_model='fluid')
pars.set_cosmology(H0=67.4, ombh2=0.02233, omch2=0.1198, omk=0, tau=0.054)
results = camb.get_results(pars)

params_CAMB = dict()
# crearemos diccionarios en donde estaran los parametros cosmologicos que
# queremos utilizar, es facil poder crear y modificar diccionarios

params_CAMB['Ob'] = results.get_Omega('baryon')
params_CAMB['Om'] = 1 - pars.omk - results.get_Omega('de')
params_CAMB['ODE'] = results.get_Omega('de')
params_CAMB['H0'] = pars.H0
params_CAMB['w_0'] = pars.DarkEnergy.w
params_CAMB['w_a'] = pars.DarkEnergy.wa
params_CAMB['Ov'] = results.get_Omega('photon')

# cration of basic background quantities


def Omega_Lambda(Omega_m):
    """La funcion Omega_Lambda nos entregara este parametro en base a los que
    tenemos, para esto tambien debemos calcular Omega c, un parametro que
    no se utilizara, por lo que no es necesario almacenar"""
    return 1 - Omega_m


def Omega_K_0(Omega_DE, Omega_m):
    """Omega_K_0 nos entrega este parametro que es depende de Omega DE y
    Omega m, en el caso de del modelo ΛCDM este valor es cero"""
    return 1 - (Omega_DE + Omega_m)


def cosmological_parameters(cosmo_pars=dict()):
    """cosmological_parameters extrae los parametros necesarios para las
    el calculo de funciones E(z) y D(z), concadena estos parametros de manera
    que sean facil de utilizar, el default son los parametros de Planck 2018"""
    H0 = cosmo_pars.get('H0', params_CAMB['H0'])
    Om = cosmo_pars.get('Om', params_CAMB['Om'])
    ODE = cosmo_pars.get('ODE', params_CAMB['ODE'])
    OL = Omega_Lambda(Om)
    OK = Omega_K_0(ODE, Om)
    wa = cosmo_pars.get('wa', params_CAMB['w_a'])
    w0 = cosmo_pars.get('w0', params_CAMB['w_0'])
    return H0, Om, ODE, OL, OK, wa, w0


def E_arb(z, cosmo_pars=dict()):
    """E_arb es la función E(z) arbitraria para cualquier modelo cosmologico"""
    H0, Om, ODE, OL, Ok, wa, w0 = cosmological_parameters(cosmo_pars)
    exp = np.exp(-3*wa*(z/1+z))
    ind = 1 + wa + w0
    return np.sqrt(Om*(1+z)**3 + ODE*((1+z)**(3*ind))*exp + Ok*(1+z)**2)


def E(z, cosmo_pars=dict()):
    """E es la función E(z) para el caso w0 = -1 y wa = 0"""
    H0, Om, ODE, OL, Ok, wa, w0 = cosmological_parameters(cosmo_pars)
    return np.sqrt(Om*(1+z)**3 + OL + Ok*(1+z)**2)


# comoving distance to an object redshift z

def f_integral(z, cosmo_pars=dict()):
    """f_integral define la funcion dentro de la integral
    ocupada para el calculo de r(z)"""
    return 1/E(z, cosmo_pars)


def r(z, cosmo_pars=dict()):
    """r calcula comoving distnace to an objecto redshift"""
    H0, Om, ODE, OL, Ok, wa, w0 = cosmological_parameters(cosmo_pars)
    if type(z) == np.ndarray:
        integral = np.zeros(200)
        for idx, redshift in enumerate(z):
            z_int = np.linspace(0, redshift, 200)
            integral[idx] = np.trapz(f_integral(z_int, cosmo_pars), z_int)
    else:
        z_int = np.linspace(0, z, 200)
        integral = np.trapz(f_integral(z_int, cosmo_pars), z_int)
    return const.c.value / 1000 / pars.H0 * integral


# transverse comoving distance


def D(z, cosmo_pars=dict()):
    """La funcion D calcula transverse comoving distance para los distintos
    casos de el parametro Omgea_K_0"""
    c = const.c.value / 1000
    H0, Om, ODE, OL, Ok, wa, w0 = cosmological_parameters(cosmo_pars)
    cte_1 = c/H0
    cte_2 = H0/c
    a = 1/(1+z)
    if Ok < 0:
        return a*(cte_1*(1/(np.abs(Ok)**(1/2))))*np.sin(
            np.abs(Ok)**(1/2)*cte_2*r(z, cosmo_pars))
    if Ok == 0:
        return a*r(z, cosmo_pars)
    if Ok > 0:
        return a*(cte_1*(1/(Ok**(1/2))))*np.sinh(
            (Ok**(1/2))*cte_2*r(z, cosmo_pars))
    else:
        return "Error"


# all plots in the same row, share the y-axis.

z_arr = np.linspace(0, 2.5, 100)
# fig, ax = plt.subplots(1, 1, sharey='row', sharex='col', figsize=(10, 8))

# Proper distance dependent on redshift plot
# ax.plot(z_arr, E_arb(z_arr), label='$E(z)$', color='mediumpurple')
# ax.set_xlabel('Redshift $z$')
# ax.set_ylabel('$E(z)$')
# ax.set_title('Proper distance $E(z) as a function of redshift $z$')
# plt.show()

# Comoving distance to an object redshift z plot

# fig, ax = plt.subplots(1, 1, sharey='row', sharex='col', figsize=(10, 8))
# for z in z_arr:
#     ax.scatter(z, r(z), s=1.0, label='$r(z)$', color='mediumpurple')
# ax.set_xlabel('Redshift $z$')
# ax.set_ylabel('$Comoving distance r(z)$')
# ax.set_title('Comoving distance $r(z)$ as a function of redshift $z$')
# plt.show()

# Angular diameter distance to an object redshift z plot

# fig, ax = plt.subplots(1, 1, sharey='row', sharex='col', figsize=(10, 8))
# for z in z_arr:
#     ax.scatter(z, D(z), s=1.0, label='$D_A(z)$', color='mediumpurple')
# ax.set_xlabel('Redshift $z$')
# ax.set_ylabel('$D_A(z)$')
# ax.set_title('Angular diameter distance $D_a(z)$ as a function of redshift $z$')
# plt.show()

# now using CAMB parameters

# Proper distance dependent on redshift plot
# fig, ax = plt.subplots(1, 1, sharey='row', sharex='col', figsize=(10, 8))
# ax.plot(z_arr, E_arb(z_arr, params_CAMB), label='$E(z)$', color='mediumpurple')
# ax.set_xlabel('Redshift $z$')
# ax.set_ylabel('$E(z)$')
# ax.set_title('Proper distance $E(z) as a function of redshift $z$')
# plt.show()

# Comoving distance to an object redshift z plot

# fig, ax = plt.subplots(1, 1, sharey='row', sharex='col', figsize=(10, 8))
# for z in z_arr:
#     ax.scatter(z, r(z, params_CAMB), s=1.0, label='$r(z)$',
#                color='mediumpurple')
# ax.set_xlabel('Redshift $z$')
# ax.set_ylabel('$Comoving distance r(z)$')
# ax.set_title('Comoving distance $r(z)$ as a function of redshift $z$')
# plt.show()

# Angular diameter distance to an object redshift z plot

# fig, ax = plt.subplots(1, 1, sharey='row', sharex='col', figsize=(10, 8))
# for z in z_arr:
#     ax.scatter(z, D(z, params_CAMB), s=1.0, label='$D_A(z)$',
#                color='mediumpurple')
# ax.set_xlabel('Redshift $z$')
# ax.set_ylabel('$D_A(z)$')
# ax.set_title('Angular diameter distance $D_a(z)$ as a function of redshift $z$')
# plt.show()


# Window Function

# Bin creation

z_bin = binned_statistic(z_arr, z_arr, bins=100)
z_bin_equi = binned_statistic(z_arr, z_arr, bins=10)
limits = [z_bin.bin_edges[0], z_bin.bin_edges[-1]]
# This are the values from the paper
z_equi = [(0.001, 0.42), (0.42, 0.56), (0.56, 0.68), (0.68, 0.79),
          (0.79, 0.90), (0.90, 1.02), (1.02, 1.15), (1.15, 1.32),
          (1.32, 1.58), (1.58, 2.50)]

# Parameters adopted to describe the photometric redshift distribution source

PRD = dict()

PRD['cb'] = 1.0
PRD['zb'] = 0.0
PRD['sigmab'] = 0.05
PRD['co'] = 1.0
PRD['zo'] = 0.1
PRD['sigmao'] = 0.05
PRD['fout'] = 0.1

# Tilde function of comoving distance


def tilde_r(z, cosmo_pars=dict()):
    c = const.c.value / 1000
    H0, Om, ODE, OL, Ok, wa, w0 = cosmological_parameters(cosmo_pars)
    cte = c/H0
    return r(z, cosmo_pars)/cte

# Photometric redshift estimates:


def n(z):
    zm = 0.9  # median redshift, value given by Euclid Red Book
    z0 = zm/(np.sqrt(2))
    frac = z/z0
    return (frac**2)*np.exp(-(frac)**(3/2))


# Photometric redshift ditribution of sources

def P_ph(zp, z):
    cb = PRD['cb']
    zb = PRD['zb']
    sigmab = PRD['sigmab']
    co = PRD['co']
    zo = PRD['zo']
    sigmao = PRD['sigmao']
    fout = PRD['fout']

    frac_1 = (1-fout)/(np.sqrt(2*np.pi)*sigmab*(1+z))

    frac_2 = fout/(np.sqrt(2*np.pi)*sigmao*(1+z))

    exp_1 = np.exp((-1/2)*((z-cb*zp-zb)/sigmab*(1+z))**2)

    exp_2 = np.exp((-1/2)*((z-co*zp-zo)/sigmao*(1+z))**2)

    return frac_1*exp_1 + frac_2*exp_2


# Defining integrals for photometric redshift estimates


def int_1(zp, z):
    return n(z)*P_ph(zp, z)


def n_i(z, i):
    ith_bin = z_equi[i]
    zi_l, zi_u = ith_bin
    z_int1 = np.linspace(zi_l, zi_u, 200)
    z_int2 = np.linspace(limits[0], limits[1], 200)
    X, Y = np.meshgrid(z_int1, z_int2)
    list1 = int_1(X, Y)
    I1 = np.trapz(int_1(z_int1, z), z_int1)
    I2 = np.trapz(np.trapz(list1, z_int2, axis=0), z_int1, axis=0)

    return I1/I2


# Matter Power spectrum following CAMB demo

pars = camb.CAMBparams()
pars.set_cosmology(H0=67.5, ombh2=0.02233, omch2=0.1198, omk=0, tau=0.054)
# Now get matter power spectra and sigma8 at redshift 0 and 0.8
pars.InitPower.set_params(ns=0.9652)
# Note non-linear corrections couples to smaller scales than you want
pars.set_matter_power(redshifts=z_bin[0], kmax=2.0)

# Linear spectra
pars.NonLinear = model.NonLinear_none
results = camb.get_results(pars)
kh, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1,
                                              npoints=200)
s8 = np.array(results.get_sigma8())

# Non-Linear spectra (Halofit)
pars.NonLinear = model.NonLinear_both
results.calc_power_spectra(pars)
kh_nonlin, z_nonlin, pk_nonlin = results.get_matter_power_spectrum(minkh=1e-4,
                                                                   maxkh=1,
                                                                   npoints=200)


# Storage power matter parameters

list_of_PMS = list(zip(kh, z, pk, kh_nonlin, z_nonlin, pk_nonlin))

# Converting lists of tuples into
# pandas Dataframe.
# df = pd.DataFrame(list_of_PMS,
#                   columns=['kh', 'z', 'pk', 'nonlinear_kh',
#                            'nonlinear_z', 'nonlinear_pk'])

# # Print data.
# df.to_csv('PMS_params.txt', sep='\t')

# Storage number density

z_list = z_bin[1]

# lst_0 = []
# for z in z_list:
#     lst_0.append([z, n_i(z, 0)])
# np.savetxt('Bin_number_d_0.txt', np.array(lst_0))


# lst_1 = []
# for z in z_list:
#     lst_1.append([z, n_i(z, 1)])
# np.savetxt('Bin_number_d_1.txt', np.array(lst_1))

# lst_2 = []
# for z in z_list:
#     lst_2.append([z, n_i(z, 2)])
# np.savetxt('Bin_number_d_2.txt', np.array(lst_2))

# lst_3 = []
# for z in z_list:
#     lst_3.append([z, n_i(z, 3)])
# np.savetxt('Bin_number_d_3.txt', np.array(lst_3))

# lst_4 = []
# for z in z_list:
#     lst_4.append([z, n_i(z, 4)])
# np.savetxt('Bin_number_d_4.txt', np.array(lst_4))

# lst_5 = []
# for z in z_list:
#     lst_5.append([z, n_i(z, 5)])
# np.savetxt('Bin_number_d_5.txt', np.array(lst_5))

# lst_6 = []
# for z in z_list:
#     lst_6.append([z, n_i(z, 6)])
# np.savetxt('Bin_number_d_6.txt', np.array(lst_6))

# lst_7 = []
# for z in z_list:
#     lst_7.append([z, n_i(z, 7)])
# np.savetxt('Bin_number_d_7.txt', np.array(lst_7))

# lst_8 = []
# for z in z_list:
#     lst_8.append([z, n_i(z, 8)])
# np.savetxt('Bin_number_d_8.txt', np.array(lst_8))

# lst_9 = []
# for z in z_list:
#     lst_9.append([z, n_i(z, 9)])
# np.savetxt('Bin_number_d_9.txt', np.array(lst_9))

# Dictionary of Bin number density

# lst_n_i = dict()

# lst_n_i["bin_0"] = np.loadtxt("Bin_number_d_0.txt")
# lst_n_i["bin_1"] = np.loadtxt("Bin_number_d_1.txt")
# lst_n_i["bin_2"] = np.loadtxt("Bin_number_d_2.txt")
# lst_n_i["bin_3"] = np.loadtxt("Bin_number_d_3.txt")
# lst_n_i["bin_4"] = np.loadtxt("Bin_number_d_4.txt")
# lst_n_i["bin_5"] = np.loadtxt("Bin_number_d_5.txt")
# lst_n_i["bin_6"] = np.loadtxt("Bin_number_d_6.txt")
# lst_n_i["bin_7"] = np.loadtxt("Bin_number_d_7.txt")
# lst_n_i["bin_8"] = np.loadtxt("Bin_number_d_8.txt")
# lst_n_i["bin_9"] = np.loadtxt("Bin_number_d_9.txt")


# # Dictionary of Inerpolation for bin number density

# interpolate_n_i = dict()

# interpolate_n_i["I_0"] = interpolate.interp1d(lst_n_i["bin_0"][:, 0],
#                                               lst_n_i["bin_0"][:, 1])
# interpolate_n_i["I_1"] = interpolate.interp1d(lst_n_i["bin_1"][:, 0],
#                                               lst_n_i["bin_1"][:, 1])
# interpolate_n_i["I_2"] = interpolate.interp1d(lst_n_i["bin_2"][:, 0],
#                                               lst_n_i["bin_2"][:, 1])
# interpolate_n_i["I_3"] = interpolate.interp1d(lst_n_i["bin_3"][:, 0],
#                                               lst_n_i["bin_3"][:, 1])
# interpolate_n_i["I_4"] = interpolate.interp1d(lst_n_i["bin_4"][:, 0],
#                                               lst_n_i["bin_4"][:, 1])
# interpolate_n_i["I_5"] = interpolate.interp1d(lst_n_i["bin_5"][:, 0],
#                                               lst_n_i["bin_5"][:, 1])
# interpolate_n_i["I_6"] = interpolate.interp1d(lst_n_i["bin_6"][:, 0],
#                                               lst_n_i["bin_6"][:, 1])
# interpolate_n_i["I_7"] = interpolate.interp1d(lst_n_i["bin_7"][:, 0],
#                                               lst_n_i["bin_7"][:, 1])
# interpolate_n_i["I_8"] = interpolate.interp1d(lst_n_i["bin_8"][:, 0],
#                                               lst_n_i["bin_8"][:, 1])
# interpolate_n_i["I_9"] = interpolate.interp1d(lst_n_i["bin_9"][:, 0],
#                                               lst_n_i["bin_9"][:, 1])


# Window Function


def W_int(z_1, z, i, cosmo_pars=dict()):
    return interpolate_n_i['I_%s' % (str(i))](z_1)*(
        1-tilde_r(z, cosmo_pars)/tilde_r(z_1, cosmo_pars))


def Window_F(z, i, cosmo_pars=dict()):
    z_int = np.linspace(z, limits[1], 200)
    return np.trapz(W_int(z_int, z, i, cosmo_pars), z_int)


# Window function for an specific bin for redshift
# start = time.time()
# fig, ax = plt.subplots(1, 1, sharey='row', sharex='col', figsize=(10, 8))
# for z in z_arr:
#     ax.scatter(z, Window_F(z, 1), s=2.0, label='$Window Function(z)$',
#                color='mediumpurple')
# ax.set_xlabel('Redshift $z$')
# ax.set_ylabel('Window Function $\tilde{W}_{1}(z)$')
# ax.set_title('Window function for an specific bin $\tilde{W}(z)$ as a function of redshift $z$')
# end = time.time()

# print("El tiempo que se demoró es "+str(end-start)+" segundos")
# fig.show()


# fig, ax = plt.subplots()

# for i in range(10):
#     ax.plot(z_arr, interpolate_n_i['I_%s'%(str(i))](z_arr), c='b')
# ax.plot(z_arr, 25*n(z_arr), c='red')
# ax.set_xlabel('Redshift $z$')
# ax.set_ylabel('Number density')
# fig.show()


# Interpolator CAMB


nz = 100  # number of steps to use for the radial/redshift integration
kmax = 7   # kmax to use with k_hunit = Mpc/h

# For Limber result, want integration over \chi, from 0 to chi_*.
# so get background results to find chistar, set up a range in chi,
# and calculate corresponding redshifts
results = camb.get_background(pars)
chistar = results.conformal_time(0) - results.tau_maxvis
chis = np.linspace(0, chistar, nz)
zs = results.redshift_at_comoving_radial_distance(chis)
# Calculate array of delta_chi, and drop first and
# last points where things go singular
dchis = (chis[2:]-chis[:-2])/2
chis = chis[1:-1]
zs = zs[1:-1]

# Get the matter power spectrum interpolation object.
# Here for lensing we want the power spectrum of the Weyl potential.
PK = camb.get_matter_power_interpolator(pars,
                                        nonlinear=True,
                                        hubble_units=False,
                                        k_hunit=True,
                                        kmax=kmax,
                                        var1=model.Transfer_tot,
                                        var2=model.Transfer_tot,
                                        zmax=zs[-1])

# Have a look at interpolated power spectrum results for a range of redshifts
# Expect linear potentials to decay a bit when Lambda becomes important,
# and change from non-linear growth

# plt.figure(figsize=(8,5))
# k = np.exp(np.log(10)*np.linspace(-4, 7, 200))
# for z in z_bin_equi[0]:
#     plt.loglog(k, PK.P(z, k), color='mediumpurple')
# plt.xlim([1e-4,kmax])
# plt.xlabel('Wave-number k (h/Mpc)')
# plt.ylabel('$P_k, Mpc^3$')
# plt.legend(['z=%s'%z for z in z_bin_equi[0]])
# plt.show()


# Calculation of Cosmic shear power spectrum:


# Weight function


def Weight_F(z, i, cosmo_pars=dict()):
    H0, Om, ODE, OL, Ok, wa, w0 = cosmological_parameters(cosmo_pars)
    c = const.c.value / 1000
    cte = (3/2)*((H0/c)**2)*Om
    return cte*(1 + z)*r(z, cosmo_pars)*Window_F(z, i, cosmo_pars)


def int_2(z, i, j, l, cosmo_pars=dict()):
    I1 = (Weight_F(z, i, cosmo_pars)*Weight_F(z, j, cosmo_pars))/(
        E(z, cosmo_pars)*(r(z, cosmo_pars)**2))
    k = (l + (1/2))/r(z, cosmo_pars)
    PMS = PK.P(z, k)
    return I1*PMS


def C(l, i, j, cosmo_pars=dict()):
    H0, Om, ODE, OL, Ok, wa, w0 = cosmological_parameters(cosmo_pars)
    c = const.c.value / 1000
    cte = (c/H0)
    I1 = integrate.quad(int_2, limits[0],
                        limits[1], args=(i, j, l, cosmo_pars))[0]
    return cte*I1

# FOR INDIVIDUAL I J

# start_1 = time.time()
# fig, ax = plt.subplots(1, 1, sharey='row', sharex='col', figsize=(10, 8))
# l_toplot = [138, 194, 271, 378, 529, 739, 1031, 1440, 2012]
# #l_toplot = np.arange(100, 300)
# i, j = 1, 1
# for l in l_toplot:
#     ax.scatter(l, l*(l+1)*C(l, i, j)/(2*np.pi), label="$l(l+1)/2\pi C_{1,1}$")
#     ax.scatter(l, l*(l+1)*C(l, 9, 9)/(2*np.pi), label="$l(l+1)/2\pi C_{9,9}$")
# ax.set_xlabel('Multipole l')
# ax.set_ylabel(r'$l(l+1)/2\pi C_{%s%s}^{\gamma\gamma(l)}$'%(str(i), str(j)))
# # ax.legend(['z=%s'%z for z in zplot])
# end_1 = time.time()

# print("El tiempo que se demoró es "+str(end_1-start_1)+" segundos")
# fig.legend()
# fig.show()


# cosmic_shear_array = np.zeros((200, 10, 10))
# for l in range(100, 301):
#     for i in range(10):
#         for j in range(10):
#             cosmic_shear_array[l-100, i, j] = C(l, i, j)
#             print("i: %f, j: %f" %(i, j), end = '\r')

# reshape_cosmic = np.reshape(cosmic_shear_array, (cosmic_shear_array.shape[0], -1))

# np.savetxt('quad_convergence/quad_file', reshape_cosmic)

C_l_n = np.loadtxt('convergence/cosmic_shear_correctls')

C_l_i_j = np.reshape(C_l_n, (C_l_n.shape[0],
                             C_l_n.shape[1]
                             // 10, 10))

np.save('quad_convergence/quad_file', C)

lst_C_l = dict()
for l in range(100):
    lst_C_l['C_bin_%s' % (str(l))] = C_l_n[l]


# FIHSER MATRIX
l_lst = np.linspace(10, 1500, 100)
ls = np.logspace(1, np.log10(1500), 101)
l_bins = [(ls[i], ls[i + 1]) for i in range(100)]
ls_eval = np.array([(l_bins[i][1] + l_bins[i][0]) / 2 for i in range(100)])


def Delta_l(i):
    lamba_min = np.log(l_lst[0])
    lamba_max = np.log(l_lst[-1])  # pessimist case
    N_l = 100
    delta_lambda = (lamba_max - lamba_min)/N_l
    lambda_k = lamba_min + (i - 1)*delta_lambda
    lambda_k_1 = lamba_min + i*delta_lambda
    return 10**(lambda_k_1) - 10**(lambda_k)


def Cov(i, j, m, n):
    f_sky = 1/15000
    M = np.zeros((100, 100))
    for x, l in enumerate(np.arange(100, 200)):
        dl = l_bins[x][1] - l_bins[x][0]
        term_1 = C_l_i_j[l-100, i, m] * C_l_i_j[l-100, j, n]
        term_2 = C_l_i_j[l-100, i, n] * C_l_i_j[l-100, j, m]
        term_3 = (2*l + 1)*f_sky*dl
        M[x, x] = (term_1 + term_2) / term_3
    return M


def Obs_E(i, j):
    f_sky = 1/15000
    # delta_l = l_lst[-1] - l_lst[0]
    M = np.zeros((100, 100))
    for x, l in enumerate(np.arange(100, 200)):
        M[x, x] = np.sqrt(2/((2*l + 1)*Delta_l(i)*f_sky)) * \
            C_l_i_j[l - 100, i, j]
    return M


def K_yy(z, i, j, cosmo_pars=dict()):
    H0, Om, ODE, OL, Ok, wa, w0 = cosmological_parameters(cosmo_pars)
    c = const.c.value / 1000
    cte = (H0/c)**3
    term_1 = ((3/2)*Om*(1 + z))**2
    term_2 = ((Window_F(z, i, cosmo_pars)*Window_F(z, j, cosmo_pars))
              / E(z, cosmo_pars))
    return term_1*cte*term_2


def shot_noice(l, i, j):
    ng = 354543085.80106884
    sigma_e = 0.30
    n = ng/10
    return (sigma_e/n)*np.kron(i, j)


i, j = 0, 0
m, n = 0, 0
fig, ax = plt.subplots()

im = ax.imshow(Cov(i, j, m, n))
ax.set_title(
    'Cov$[C_{%i%i}^{\gamma\gamma}(\ell), C_{%i%i}^{\gamma\gamma}(\ell)]$' % (i, j, m, n))
ax.set_ylim(ax.get_ylim()[::-1])
fig.colorbar(im)
fig.savefig('Cov')

fig, ax = plt.subplots()

ell = 100
im = ax.imshow(Obs_E(i, j))
ax.set_title('$\Delta C_{ij}^{\gamma\gamma}(\ell=%i)$' % ell)
ax.set_ylim(ax.get_ylim()[::-1])
fig.colorbar(im)
# fig.show()

fig, ax = plt.subplots()
ltoplt = np.arange(100, 300)
# ls = np.arange(100, 1501)


fig, ax = plt.subplots()

i, j = 9, 9

for idx, l in enumerate(ls_eval):
    ax.scatter(l, C_l_i_j[idx, i, j], c='mediumpurple', s=0.5)
ax.set_xlabel('Multipole $\ell$')
ax.set_ylabel(r'$C_{%s%s}^{\gamma\gamma}(\ell)$' % (str(i), str(j)))
ax.set_xscale('log')
ax.set_yscale('log')
fig.savefig('Con')
