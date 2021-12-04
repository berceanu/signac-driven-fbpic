import numpy as np
from scipy.constants import c, e, m_e, m_p
from fbpic.main import Simulation


from fbpic.openpmd_diag import FieldDiagnostic, ParticleDiagnostic
import matplotlib.pyplot as pl


from fbpic.lpa_utils.laser import add_laser_pulse, FlattenedGaussianLaser
from fbpic.fields.smoothing import BinomialSmoother

# ----------
# Parameters
# ----------

n_order = -1


# Whether to use the GPU
use_cuda = True

lambda0 = 0.815e-6  # Reference wavelength

# The Driver laser train (conversion to boosted frame is done inside 'add_laser')
a0 = 5  # Laser amplitude
w0 = 22 * 1e-6  # 127.5*1.e-6      # Laser waist
tau = 29 / np.sqrt(2.0 * np.log(2)) * 1e-15  # Laser duration   40fs FWHM
ctau = c * tau  # Pulse length
z0 = 0.0e-6  # Laser centroid
zfoc = 2000.0e-6  # Focal position
N_supergaussian = 4  # N order for supergaussian
# Reference laser wavelength
lambdad = lambda0  # Laser wavelength
Max_m = 1  # Maximum value of the azimuthal mode


# Plasma electron density
n_e = 1.0e18 * 1e6  # The density in the labframe (electrons.meters^-3)
# Critical plasma density
n_c = 1.1e15 / lambdad ** 2
# Plasma wavelength
lambda_p = lambdad * np.sqrt(n_c / n_e)
print("Plasma wavelength  %e m " % lambda_p)
gamma_wake = np.sqrt(n_c / n_e)


# The simulation box

zmax = 2.2 * ctau  # Length of the box along z (meters)
zmin = -2 * ctau - 1.9 * lambda_p
# -1.5*lambda_p
rmax = 4.5 * w0  # Length of the box along r (meters)
Nm = 2  # Number of modes used (Mm>=Max_m+2)

# n_guard = 40     # Number of guard cells
# exchange_period = 2500

# Resolution
dz = lambda0 / 24
dr = lambda0 / 4
dt = dz / c

Nz = int(np.round((zmax - zmin) / dz))  # Number of gridpoints along z
Nr = int(np.round(rmax / dr))  # Number of gridpoints along r

print("Performing a simulation with a box %d X %d " % (Nz, Nr))


# The density profile
w_matched = 400000000000000000.0e-6
ramp_up = 1000.0e-6
plateau = 20e3 * 1.0e-6
ramp_down = ramp_up


# The particles of the plasma
p_zmin = 0.0e-6  # Position of the beginning of the plasma (meters)
p_zmax = ramp_up + plateau + ramp_down
p_rmin = 0.0  # Minimal radial position of the plasma (meters)
p_rmax = 0.95 * rmax  # Maximal radial position of the plasma (meters)
p_nz = 2  # Number of particles per cell along z
p_nr = 2  # Number of particles per cell along r
p_nt = 4  # Number of particles per cell along theta
uz_m = 0.0  # Initial momentum of the electrons in the lab frame

L_vacuum = ramp_up

# The moving window
v_window = c * np.sqrt(1.0 - n_e / n_c)
# Speed of the window

# The interaction length of the simulation, in the lab frame (meters)
L_interact = p_zmax - p_zmin + L_vacuum  # the plasma length
T_interact = L_interact / c
## The diagnostics
diag_period = 20000

# Whether to tag and track the particles of the bunch

save_checkpoints = True  # Whether to write checkpoint files
checkpoint_period = 100000  # Period for writing the checkpoints
use_restart = False  # Whether to restart from a previous checkpoint
track_electrons = False  # Whether to track and write particle ids

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

# df1 = pd.DataFrame(z_dens)
# df1.to_csv('z_density.csv')
# df2 = pd.DataFrame(n_dens)
# df2.to_csv('n_density.csv')

# ---------------------------
# Carrying out the simulation
# ---------------------------
# NB: The code below is only executed when running the script,
# (`python boosted_frame_sim.py`), but not when importing it.
if __name__ == "__main__":

    # Initialize the simulation object
    smoother = BinomialSmoother({"z": 4, "r": 4}, compensator={"z": True, "r": True})

    sim = Simulation(
        Nz,
        zmax,
        Nr,
        rmax,
        Nm,
        dt,
        zmin=zmin,
        n_order=n_order,
        use_cuda=use_cuda,
        boundaries={"z": "open", "r": "reflective"},
        smoother=smoother,
    )
    # 'r': 'open' can also be used, but is more computationally expensive

    # Add the plasma electron and plasma ions
    plasma_elec = sim.add_new_species(
        q=-e,
        m=m_e,
        n=n_e,
        dens_func=dens_func,
        p_zmin=p_zmin,
        p_zmax=p_zmax,
        p_rmax=p_rmax,
        p_nz=p_nz,
        p_nr=p_nr,
        p_nt=p_nt,
    )
    # plasma_ions = sim.add_new_species( q=e, m=m_p,
    #                n=n_e, dens_func=dens_func,
    #                p_zmin=p_zmin, p_zmax=p_zmax, p_rmax=p_rmax,
    #                p_nz=p_nz, p_nr=p_nr, p_nt=p_nt )

    # profile = LaguerreGaussLaser(a0=a0, waist=w0, tau=tau, z0=z0, lambda0=lambdad, zf=zfoc, p=0, m=0)
    profile = FlattenedGaussianLaser(
        a0=a0, w0=w0, tau=tau, N=N_supergaussian, z0=z0, lambda0=lambdad, zf=zfoc
    )

    add_laser_pulse(sim, profile)

    # Configure the moving window
    sim.set_moving_window(v=v_window)

    sim.diags = [
        FieldDiagnostic(diag_period, sim.fld, comm=sim.comm, fieldtypes=["E", "rho"]),
        ParticleDiagnostic(
            diag_period,
            {"electrons": plasma_elec},
            select={"uz": [200.0, None]},
            comm=sim.comm,
        ),
    ]

    # Number of iterations to perform
    N_step = int(T_interact / sim.dt)

    ### Run the simulation
    sim.step(N_step)
    print("")
