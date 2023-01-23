"""Python script for computing matter power spectrum for """

from camb import model, initialpower
import camb
import numpy as np

dict_fiducial = {'Omegam': 0.32, 'Omegab': 0.05, 'Omegade': 0.68, 'Omegach2': 0.12055785610846023,
                 'w0': -1.0, 'wa': 0, 'hubble': 0.67, 'ns': 0.96, 'sigma8': 0.815584, 'gamma': 0.55}


def mps_param(param, dx=0.01, fiducial=dict_fiducial):
    locals().update(fiducial)
    pars_l = camb.CAMBparams()
    pars_u = camb.CAMBparams()
    if param == 'Omegam':
        Omegab_l, Omegab_u = (1 - dx) * Omegab, (1 + dx) * Omegab
        Omegach2_l, Omegach2_u = (1 - dx) * Omegach2, (1 + dx) * Omegach2

        pars_l.set_dark_energy(w=-1.0, wa=0, dark_energy_model='fluid')
        pars_u.set_dark_energy(w=-1.0, wa=0, dark_energy_model='fluid')
        pars_l.set_cosmology(H0=hubble*100, ombh2=Omegab_l *
                             hubble**2, omch2=Omegach2_l, tau=0.058)
        pars_u.set_cosmology(H0=hubble*100, ombh2=Omegab_u *
                             hubble**2, omch2=Omegach2_u, tau=0.058)
        pars_l.InitPower.set_params(As=2.1260500000000005e-9, ns=ns)
        pars_u.InitPower.set_params(As=2.1260500000000005e-9, ns=ns)

        # Linear spectra
        pars_l.NonLinear = model.NonLinear_none
        pars_u.NonLinear = model.NonLinear_none
        results_l = camb.get_results(pars_l)
        results_u = camb.get_results(pars_u)
        kh_l, z_l, pk_l = results_l.get_matter_power_spectrum(
            minkh=1e-4, maxkh=7, npoints=200)
        kh_u, z_u, pk_u = results_u.get_matter_power_spectrum(
            minkh=1e-4, maxkh=7, npoints=200)

        # Store data for this model
        dict_mps_data = {'kh_l': kh_l, 'z_l': z_l,
                         'pk_l': pk_l, 'kh_u': kh_u, 'z_u': z_u, 'pk_u': pk_u}

        for rslt in dict_mps_data:
            np.savetxt('mps_data/Omegam_'+rslt, dict_mps_data[rslt])

    elif param == 'ns':
        ns_l, ns_u = (1 - dx) * ns, (1 + dx) * ns

        pars_l.set_dark_energy(w=-1.0, wa=0, dark_energy_model='fluid')
        pars_u.set_dark_energy(w=-1.0, wa=0, dark_energy_model='fluid')
        pars_l.set_cosmology(H0=hubble*100, ombh2=Omegab *
                             hubble**2, omch2=Omegach2, tau=0.058)
        pars_u.set_cosmology(H0=hubble*100, ombh2=Omegab *
                             hubble**2, omch2=Omegach2, tau=0.058)
        pars_l.InitPower.set_params(As=2.1260500000000005e-9, ns=ns_l)
        pars_u.InitPower.set_params(As=2.1260500000000005e-9, ns=ns_u)

        # Linear spectra
        pars_l.NonLinear = model.NonLinear_none
        pars_u.NonLinear = model.NonLinear_none
        results_l = camb.get_results(pars_l)
        results_u = camb.get_results(pars_u)
        kh_l, z_l, pk_l = results_l.get_matter_power_spectrum(
            minkh=1e-4, maxkh=7, npoints=200)
        kh_u, z_u, pk_u = results_u.get_matter_power_spectrum(
            minkh=1e-4, maxkh=7, npoints=200)

        # Store data for this model
        dict_mps_data = {'kh_l': kh_l, 'z_l': z_l,
                         'pk_l': pk_l, 'kh_u': kh_u, 'z_u': z_u, 'pk_u': pk_u}

        for rslt in dict_mps_data:
            np.savetxt('mps_data/ns_'+rslt, dict_mps_data[rslt])
