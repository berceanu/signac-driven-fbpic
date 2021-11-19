import numpy as np
from scipy.constants import c, e, m_e, m_p

from fbpic.openpmd_diag import FieldDiagnostic, ParticleDiagnostic
import matplotlib.pyplot as pl

from fbpic.lpa_utils.laser import add_laser_pulse, GaussianLaser

# The density profile
w_matched = 400000000000000000.0e-6
ramp_up = 1000.0e-6
plateau = 20e3 * 1.0e-6
ramp_down = ramp_up

# The particles of the plasma
p_zmax = ramp_up + plateau + ramp_down

L_vacuum = ramp_up

# The interaction length of the simulation, in the lab frame (meters)
L_interact = p_zmax + L_vacuum  # the plasma length

# The density profile

# Relative change divided by w_matched^2 that allows guiding
rel_delta_n_over_w2 = 1.0 / (np.pi * 2.81e-15 * w_matched ** 4 * n_e)
# Define the density function
def dens_func(z, r):

    # Allocate relative density
    n = np.ones_like(z)
    # Make ramp up
    inv_ramp_up = 1.0 / ramp_up
    n = np.where(z < ramp_up, (z * inv_ramp_up) ** 2, n)
    # n = np.where( z<ramp_up, (np.sin(2./np.pi*z*inv_ramp_up))**2, n )
    # n = np.where( z<=ramp_up, z*inv_ramp_up, n )
    # Make ramp down
    inv_ramp_down = 1.0 / ramp_down
    n = np.where(
        (z >= ramp_up + plateau) & (z < ramp_up + plateau + ramp_down),
        -(z - (ramp_up + plateau + ramp_down)) * inv_ramp_down,
        n,
    )
    #
    n = np.where(z >= ramp_up + plateau + ramp_down, 0, n)
    n = np.where(z <= 0.0, 0.0, n)
    # Add transverse guiding parabolic profile
    n = n * (1.0 + rel_delta_n_over_w2 * r ** 2)
    # n = n * np.exp( -(r/r_cutoff)**16 )
    return n


# #Plot the density
pl.show()
axes = pl.gca()
z_dens = np.linspace(p_zmin, p_zmax, 10000)
n_dens = dens_func(z_dens, 0)
l = axes.plot(z_dens * 1e6, dens_func(z_dens, 0), linewidth=3, label="Density")
l = axes.plot(zfoc * 1e6, dens_func(zfoc, 0), "ro", markersize=8, label="Focus point")
pl.xlabel("z, $\mu m$", fontsize=14)
pl.ylabel("$n_e/n_0$", fontsize=14)
pl.savefig("DensityProfile.png")
pl.legend(fontsize=13)
