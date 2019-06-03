from scipy.constants import physical_constants
c = physical_constants['speed of light in vacuum'][0] # speed of light in m/s
q_e = physical_constants['elementary charge'][0] # electron charge in C
m_e = physical_constants['electron mass'][0] # electon mass in Kg

import numpy as np
from scipy.signal import hilbert

from multiprocessing import cpu_count

default_nprocs = cpu_count()

def distribute(nitems,  nprocs=None):
    if nprocs is None:
        nprocs = default_nprocs
    nitems_per_proc = (nitems + nprocs -1) // nprocs
    return [(i, min(nitems, i+nitems_per_proc))
            for i in range(0, nitems, nitems_per_proc)]

def E0(lambda0=0.8e-6):
    k0 = 2 * np.pi / 0.8e-6
    E0 = m_e * c**2 * k0 / q_e #V/m
    return E0


def get_a0(ts, t=None, iteration=None, pol='x', m='all', lambda0=0.8e-6):
    
    # if pol not in ['x', 'y']:
    #     raise ValueError('The `pol` argument is missing or erroneous.')

    # if pol == 'x':
    slicing_dir = 'y'
    theta = 0
    # else:
    #     slicing_dir = 'x'
    #     theta = np.pi / 2.

    # get E_x field in V/m
    Ex, info_Ex = ts.get_field(field='E', coord=pol,
                               t=t, iteration=iteration,
                               m=m, theta=theta, slicing_dir=slicing_dir)
    #normalization
    a0 = Ex/E0(lambda0)

    # get pulse envelope
    envelope = np.abs(hilbert(a0, axis=1))
    envelope_z = envelope[envelope.shape[0]//2, :]
    
    a0_max = np.amax(envelope_z)    
        
    # index of peak
    z_idx = np.argmax(envelope_z)
    # peak position
    z0 = info_Ex.z[z_idx]

    # FWHM perpendicular size of beam, proportional to w0
    fwhm_a0_w0 = np.sum(np.greater_equal(envelope[:, z_idx], a0_max/2)) * info_Ex.dr
    # FWHM longitudinal size of the beam, proportional to ctau
    fwhm_a0_ctau = np.sum(np.greater_equal(envelope_z, a0_max/2)) * info_Ex.dz

    return z0, a0_max, fwhm_a0_w0, fwhm_a0_ctau



def get_laser_diags(ts, t=None, iteration=None, pol='x', m='all',
                    method='fit', prnt=False, units='SI', lambda0=0.8e-6):
    """
    Calculate the centroid, main frequency, a0 value, 
    beam waist and pulse length of a Gaussian laser.
    
    Parameters
    ----------
    ts : LpaDiagnostics time series instance
        An instance of the LpaDiagnostics class
    t : float (in seconds), optional
        Time at which to obtain the data (if this does not correspond to
        an available file, the last file before `t` will be used)
        Either `t` or `iteration` should be given by the user.
    iteration : int
        The iteration at which to obtain the data
        Either `t` or `iteration` should be given by the user.
    pol : string
        Polarization of the field. Options are 'x', 'y'
    m : int or str, optional
        Only used for thetaMode geometry
        Either 'all' (for the sum of all modes) 
        or an integer (for the selectrion of a particular mode)
    method : str, optional
       The method which is used to compute the waist,
       both longitudinal and transverse
       'fit': Gaussian fit of the profile
       'rms': RMS radius, weighted by the profile
       ('rms' tends to give more weight to the "wings" of the pulse)
    prnt : bool
        Weather to print the values out
    units : str
        The unit system used, either
        'SI' : rads, meters, seconds
        'mufs' : rads, microns, femtoseconds
    
    Returns
    -------
    A tuple with:
    z0 : float
        Centroid of the laser in meters
    omega0 : float
        Mean angular frequency in rad/s
    a0 : float
        Normalized vector potential
    w0 : float
        Beam waist in meters
    ctau : float
        Pulse length (longitudinal waist) in meters
    """
    if pol not in ['x', 'y']:
        raise ValueError('The `pol` argument is missing or erroneous.')

    # if pol == 'x':
    #     slicing_dir = 'y'
    #     theta = 0
    # else:
    #     slicing_dir = 'x'
    #     theta = np.pi / 2.
        
    # get pulse envelope
    #E_vs_z, info = ts.get_laser_envelope(t=t, iteration=iteration,
    #                                     pol=pol, m=m, theta=theta,
    #                                     slicing_dir=slicing_dir,
    #                                     index='center')
    #i_max = np.argmax( E_vs_z )
    #z0 = info.z[i_max]
    
    z0, a0, w0, ctau = get_a0(ts, t=t, iteration=iteration,
                                              pol=pol, m=m, lambda0=lambda0)

    #omega0 = ts.get_main_frequency(t=t, iteration=iteration, pol=pol)
    omega0 = 0.
    #a0 = ts.get_a0(t=t, iteration=iteration, pol=pol)
    #w0 = ts.get_laser_waist(t=t, iteration=iteration, pol=pol, theta=theta,
    #                     slicing_dir=slicing_dir, method=method)
    #ctau = ts.get_ctau(t=t, iteration=iteration, pol=pol, method=method)

    # assign units
    z0_u, omega0_u, w0_u, ctau_u = 'm', 'rad/s', 'm', 'm'

    if units=='mufs':
        # rescale
        z0, omega0, w0, ctau = z0*1e6, omega0*1e-15, w0*1e6, ctau*1e6
        # change units
        z0_u, omega0_u, w0_u, ctau_u = 'mu', 'rad/fs', 'mu', 'mu'
        
    if prnt and units=='mufs':
        print('z0 = {:.2f} {}, omega0 = {:.2f} {}, a0 = {:.2f}, w0 = {:.2f} {}, ctau = {:.2f} {}.'\
        .format(z0, z0_u, omega0, omega0_u, a0, w0, w0_u, ctau, ctau_u))
    return z0, omega0, a0, w0, ctau