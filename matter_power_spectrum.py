"""Python script for computing matter power spectrum for """

from camb import model, initialpower
import camb


dict_fiducial = {'Omegam': 0.32, 'Omegab': 0.05, 'Omegade': 0.68, 'Omegach2': 0.12055785610846023,
                 'w0': -1.0, 'wa': 0, 'hubble': 0.67, 'ns': 0.96, 'sigma8': 0.815584, 'gamma': 0.55}


def mps(fiducial=dict_fiducial):
    locals().update(fiducial)
    H0 = hubble * 100
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=Omegab*hubble**2, omch2=Omegach2, tau=0.058)
    

