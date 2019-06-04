"""
This is an input script that runs a simulation of
laser-wakefield acceleration using FBPIC, based on the
parameters in doi:10.1016/j.jcp.2008.11.017. Self-injection is
observed for these parameters.

Usage
-----
- Type "python calder_injection.py" in a terminal

Help
----
All the structures implemented in FBPIC are internally documented.
Enter "print(fbpic_object.__doc__)" to have access to this documentation,
where fbpic_object is any of the objects or function of FBPIC.

ETA
---
On 1xP100: 3h
"""

# -------
# Imports
# -------
import numpy as np
from scipy.constants import c
# Import the relevant structures in FBPIC
from fbpic.main import Simulation
from fbpic.lpa_utils.laser import add_laser
from fbpic.openpmd_diag import FieldDiagnostic, ParticleDiagnostic, \
     set_periodic_checkpoint, restart_from_checkpoint

# ----------
# Parameters
# ----------

# Whether to use the GPU
use_cuda = True

# The simulation box
Nz = 4096        # Number of gridpoints along z
zmax = 30.e-6   # Right end of the simulation box (meters)
zmin = -70.e-6   # Left end of the simulation box (meters)
Nr = 256         # Number of gridpoints along r
rmax = 30.e-6  # Length of the box along r (meters)
Nm = 2           # Number of modes used

# The simulation timestep
dz = (zmax-zmin)/Nz  # resolution along z
dt = dz/c   # Timestep (seconds)
N_step = int(900e-6 / dz)          # Number of iterations to perform (tot length/resolution)

# Order of the stencil for z derivatives in the Maxwell solver.
# Use -1 for infinite order, i.e. for exact dispersion relation in
# all direction (adviced for single-GPU/single-CPU simulation).
# Use a positive number (and multiple of 2) for a finite-order stencil
# (required for multi-GPU/multi-CPU with MPI). A large `n_order` leads
# to more overhead in MPI communications, but also to a more accurate
# dispersion relation for electromagnetic waves. (Typically,
# `n_order = 32` is a good trade-off.)
# See https://arxiv.org/abs/1611.05712 for more information.
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
diag_period = 100        # Period of the diagnostics in number of timesteps
save_checkpoints = False # Whether to write checkpoint files
checkpoint_period = 50   # Period for writing the checkpoints
use_restart = False      # Whether to restart from a previous checkpoint
track_electrons = False  # Whether to track and write particle ids

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

# ---------------------------
# Carrying out the simulation
# ---------------------------

# NB: The code below is only executed when running the script,
# (`python lwfa_script.py`), but not when importing it (`import lwfa_script`).
if __name__ == '__main__':

    # Initialize the simulation object
    sim = Simulation( Nz, zmax, Nr, rmax, Nm, dt,
        p_zmin, p_zmax, p_rmin, p_rmax, p_nz, p_nr, p_nt, n_e,
        dens_func=dens_func, zmin=zmin, boundaries='open',
        n_order=n_order, use_cuda=use_cuda )

    # Load initial fields
    # Add a laser to the fields of the simulation
    add_laser( sim, a0, w0, ctau, z0 )

    if use_restart is False:
        # Track electrons if required (species 0 correspond to the electrons)
        if track_electrons:
            sim.ptcl[0].track( sim.comm )
    else:
        # Load the fields and particles from the latest checkpoint file
        restart_from_checkpoint( sim )

    # Configure the moving window
    sim.set_moving_window( v=v_window )

    # Add diagnostics
    sim.diags = [ FieldDiagnostic( diag_period, sim.fld, comm=sim.comm ),
                ParticleDiagnostic( diag_period, {"electrons" : sim.ptcl[0]},
                                select={"uz" : [1., None ]}, comm=sim.comm ) ]
    # Add checkpoints
    if save_checkpoints:
        set_periodic_checkpoint( sim, checkpoint_period )

    ### Run the simulation
    sim.step( N_step )
    print('')
