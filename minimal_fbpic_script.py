"""
This is a minimal input script that runs a CPU-based simulation of
laser-wakefield acceleration using FBPIC.

Usage
-----
- Type "export FBPIC_DISABLE_THREADING=1; python minimal_fbpic_script.py" in a
  terminal

ETA
---
~1 minute on 10 cores
"""

# -------
# Imports
# -------
import numpy as np
from scipy.constants import c
# Import the relevant structures in FBPIC
from fbpic.main import Simulation
from fbpic.lpa_utils.laser import add_laser
from fbpic.openpmd_diag import FieldDiagnostic, ParticleDiagnostic

# ----------
# Parameters
# ----------

# Whether to use the GPU
use_cuda = False

# The simulation box
Nz = 800        # Number of gridpoints along z
zmax = 30.e-6   # Right end of the simulation box (meters)
zmin = -70.e-6   # Left end of the simulation box (meters)
Nr = 50         # Number of gridpoints along r
rmax = 30.e-6  # Length of the box along r (meters)
Nm = 2           # Number of modes used

# The simulation timestep
dz = (zmax-zmin)/Nz  # resolution along z
dt = dz/c   # Timestep (seconds)
N_step = 200  # Number of iterations to perform (tot length/resolution)

# Order of the stencil for z derivatives in the Maxwell solver.
n_order = -1

# The particles
p_zmin = 0.e-6  # Position of the beginning of the plasma (meters)
p_zmax = 2250.e-6 # Position of the end of the plasma (meters)
p_rmin = 0.      # Minimal radial position of the plasma (meters)
p_rmax = 27.e-6  # Maximal radial position of the plasma (meters)
n_e = 7.5e18*1.e6 # Density (electrons.meters^-3)
p_nz = 2         # Number of particles per cell along z
p_nr = 2         # Number of particles per cell along r
p_nt = 4         # Number of particles per cell along theta

# The laser
a0 = 5          # Laser amplitude
w0 = 9.e-6       # Laser waist
ctau = 9.e-6     # Laser duration
z0 = 0.e-6      # Laser centroid

# The moving window
v_window = c       # Speed of the window

# The diagnostics and the checkpoints/restarts
diag_period = 50        # Period of the diagnostics in number of timesteps

# The density profile

      #################
    #                   #
  #                       #

# z1 z2               z3 z4

z1 = 0.e-6
z2 = 375.e-6
z3 = 1875.e-6
z4 = 2250.e-6

def dens_func( z, r ) :
    """Returns relative density at position z and r"""
    # Allocate relative density
    n = np.ones_like(z)
    # Make linear ramps
    n = np.where( z<z2, (z-z1)/(z2-z1), n )
    n = np.where( z>z3, (z4 - z)/(z4-z3), n )
    # Supress density before/after the ramps
    n = np.where( z<z1, 0., n )
    n = np.where( z>z4, 0., n )
    return(n)


if __name__ == '__main__':

    # Initialize the simulation object
    sim = Simulation( Nz, zmax, Nr, rmax, Nm, dt,
        p_zmin, p_zmax, p_rmin, p_rmax, p_nz, p_nr, p_nt, n_e,
        dens_func=dens_func, zmin=zmin, boundaries='open',
        n_order=n_order, use_cuda=use_cuda )

    # Load initial fields
    # Add a laser to the fields of the simulation
    add_laser( sim, a0, w0, ctau, z0 )

    # Configure the moving window
    sim.set_moving_window( v=v_window )

    # Add diagnostics
    sim.diags = [ FieldDiagnostic( diag_period, sim.fld, comm=sim.comm ),
                ParticleDiagnostic( diag_period, {"electrons" : sim.ptcl[0]},
                                select={"uz" : [1., None ]}, comm=sim.comm ) ]

    ### Run the simulation
    np.random.seed(0) # set deterministic random seed
    sim.step( N_step )
    print('')
