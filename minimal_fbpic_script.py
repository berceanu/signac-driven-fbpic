"""
This is a minimal input script that runs a CPU-based simulation of
laser-wakefield acceleration using FBPIC.

Runtime
-------
~3 min on 1 core for 800 time steps
"""

import numpy as np
from fbpic.lpa_utils.laser import add_laser
from fbpic.main import Simulation
from fbpic.openpmd_diag import FieldDiagnostic, ParticleDiagnostic
from scipy.constants import c  # m/s

# The simulation box
Nz = 800       # Number of gridpoints along z
zmin = -10.e-6   # Left end of the simulation box (meters)
zmax = 30.e-6  # Right end of the simulation box (meters)
Nr = 50        # Number of gridpoints along r
rmax = 20.e-6    # Length of the box along r (meters)
Nm = 2         # Number of modes used

# The particles
p_zmin = 25.e-6  # Position of the beginning of the plasma (meters)
p_rmax = 18.e-6  # Maximal radial position of the plasma (meters)
n_e = 4.e18*1.e6  # Density (electrons.meters^-3)
p_nz = 2          # Number of particles per cell along z
p_nr = 2          # Number of particles per cell along r
p_nt = 4          # Number of particles per cell along theta

# The laser
a0 = 4.          # Laser amplitude
w0 = 5.e-6       # Laser waist
ctau = 5.e-6     # Laser duration
z0 = 15.e-6      # Laser centroid


## up to here ##

p_zmax = 500.e-6  # Position of the end of the plasma (meters)
p_rmin = 0.       # Minimal radial position of the plasma (meters)

# Timestep (seconds)
dt = (zmax-zmin)/Nz/c

# The interaction length of the simulation (meters)
L_interact = 0.e-6  # increase (up to `p_zmax`) to simulate longer distance!

# Interaction time (seconds) (to calculate number of PIC iterations)
T_interact = (L_interact + (zmax-zmin)) / c
# (i.e. the time it takes for the moving window to slide across the plasma)

# Number of iterations to perform
N_step = int(T_interact/dt)

# Period in number of timesteps
diag_period = int(N_step/4)  # change to 100 for long simulations!

# The density profile
ramp_start = 30.e-6
ramp_length = 40.e-6  # increase (up to `p_zmax`) !

def dens_func(z, r):
    """Returns relative density at position z and r"""
    # Allocate relative density
    n = np.ones_like(z)
    # Make linear ramp
    n = np.where(z < ramp_start+ramp_length, (z-ramp_start)/ramp_length, n)
    # Supress density before the ramp
    n = np.where(z < ramp_start, 0., n)
    return(n)


if __name__ == '__main__':

    # Initialize the simulation object
    sim = Simulation(Nz, zmax, Nr, rmax, Nm, dt,
                     p_zmin, p_zmax, p_rmin, p_rmax, p_nz, p_nr, p_nt, n_e,
                     dens_func=dens_func, zmin=zmin, boundaries='open',
                     n_order=-1, use_cuda=False)

    # Add a laser to the fields of the simulation
    add_laser(sim, a0, w0, ctau, z0)

    # Configure the moving window
    sim.set_moving_window(v=c)

    # Add diagnostics
    sim.diags = [FieldDiagnostic(diag_period, sim.fld, comm=sim.comm),
                 ParticleDiagnostic(diag_period, {"electrons": sim.ptcl[0]},
                                    select={"uz": [1., None]}, comm=sim.comm)]

    # Run the simulation
    np.random.seed(0)  # set deterministic random seed
    sim.step(N_step)
