import numpy as np
from scipy.constants import c, e, m_e

# Import the relevant structures in FBPIC
from fbpic.main import Simulation
from fbpic.lpa_utils.bunch import add_particle_bunch_file
from fbpic.openpmd_diag import FieldDiagnostic, ParticleDiagnostic

# The simulation box
Nz = 8400  # Number of gridpoints along z
zmax = 3000.0e-6  # Right end of the simulation box (meters)
zmin = -39000.0e-6  # Left end of the simulation box (meters)
Nr = 250  # Number of gridpoints along r
rmax = 500.0e-6  # Length of the box along r (meters)
Nm = 3  # Number of modes used

# The simulation timestep
dt = (zmax - zmin) / Nz / c  # Timestep (seconds)

# The particles
p_zmin = 30.0e-6  # Position of the beginning of the plasma (meters)
p_zmax = 68000.0e-6  # Position of the end of the plasma (meters)
p_rmax = 400.0e-6  # Maximal radial position of the plasma (meters)
n_e = 5.0e13 * 1.0e6  # Density (electrons.meters^-3)
p_nz = 2  # Number of particles per cell along z
p_nr = 2  # Number of particles per cell along r
p_nt = 4  # Number of particles per cell along theta

# The electron beam
L0 = 100.0e-6  # Position at which the beam should be "unfreezed"
Qtot = 200.0e-12  # Charge in Coulomb

# The moving window
v_window = c  # Speed of the window

# The diagnostics
diag_period = 1000  # Period of the diagnostics in number of timesteps

# The density profile
def dens_func_ext(z, r):
    # Allocate relative density
    n = np.ones_like(z)
    # Read density data and nomalize
    filedata = "../density_1_inlet_spacers.txt"
    l = np.loadtxt(filedata, usecols=(0))
    ne = np.loadtxt(filedata, usecols=(1))
    l = (l - l[0]) * 1e-3
    ne = ne / np.amax(ne)
    # Take only some point to keep the profile not too irregular
    numElems = 56
    idx = np.round(np.linspace(0, len(l) - 1, numElems)).astype(int)
    l = l[idx]
    ne = ne[idx]
    # Interpolate data
    n = np.interp(z, l, ne)
    return n


# The interaction length of the simulation (meters)
L_interact = 68000.0e-6  # increase to simulate longer distance!
# Interaction time (seconds) (to calculate number of PIC iterations)
T_interact = (L_interact + (zmax - zmin)) / v_window
# (i.e. the time it takes for the moving window to slide across the plasma)


if __name__ == "__main__":
    # Initialize the simulation object
    sim = Simulation(
        Nz,
        zmax,
        Nr,
        rmax,
        Nm,
        dt,
        zmin=zmin,
        boundaries={"z": "open", "r": "reflective"},
        n_order=-1,
        use_cuda=True,
    )

    # Create the plasma electrons
    _ = sim.add_new_species(
        q=-e,
        m=m_e,
        n=n_e,
        dens_func=dens_func_ext,
        p_zmin=p_zmin,
        p_zmax=p_zmax,
        p_rmax=p_rmax,
        p_nz=p_nz,
        p_nr=p_nr,
        p_nt=p_nt,
    )

    # particles beam from txt file
    beam = add_particle_bunch_file(
        sim,
        q=-e,
        m=m_e,
        filename="../exp_4deg.txt",
        n_physical_particles=Qtot / e,
        z_off=0.0,
        z_injection_plane=L0,
    )

    # Configure the moving window
    sim.set_moving_window(v=v_window)

    # Add diagnostics
    sim.diags = [
        FieldDiagnostic(diag_period, sim.fld, comm=sim.comm),
        ParticleDiagnostic(diag_period, {"beam": beam}, comm=sim.comm),
    ]

    # Number of iterations to perform
    N_step = int(T_interact / sim.dt)

    ### Run the simulation
    sim.step(N_step)
    print("")
