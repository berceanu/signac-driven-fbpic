import numpy as np
from scipy.constants import c, e, m_e, m_p
from fbpic.main import Simulation
from fbpic.openpmd_diag import FieldDiagnostic, ParticleDiagnostic
import matplotlib.pyplot as pl
from fbpic.lpa_utils.laser import add_laser_pulse, GaussianLaser
from fbpic.fields.smoothing import BinomialSmoother

n_order = 64
use_cuda = True
lambda0 = 0.815e-6
a0 = 5
w0 = 22 * 1e-6
tau = 29 / np.sqrt(2.0 * np.log(2)) * 1e-15
ChirpStretching = 1.001
ChirpSign = -1
ctau = c * tau
ctau_stretched = ctau * ChirpStretching
z0 = 0.0e-6
zfoc = 2000.0e-6
GDD = ChirpSign * 0.5 * tau ** 2 * np.sqrt(ChirpStretching ** 2 - 1)
lambdad = lambda0
Max_m = 1
n_e = 0.9e18 * 1e6
n_c = 1.1e15 / lambdad ** 2
lambda_p = lambdad * np.sqrt(n_c / n_e)
print("Plasma wavelength  %e m " % lambda_p)
gamma_wake = np.sqrt(n_c / n_e)
zmax = 2.2 * ctau_stretched
zmin = -2 * ctau_stretched - 1.5 * lambda_p
rmax = 5 * w0
Nm = 2
dz = lambda0 / 24
dr = lambda0 / 8
dt = dz / c
Nz = int(np.round((zmax - zmin) / dz))
Nr = int(np.round(rmax / dr))
print("Performing a simulation with a box %d X %d " % (Nz, Nr))
w_matched = 400000000000000000.0e-6
ramp_up = 1000.0e-6
plateau = 20e3 * 1.0e-6
ramp_down = ramp_up
p_zmin = 0.0e-6
p_zmax = ramp_up + plateau + ramp_down
p_rmin = 0.0
p_rmax = 0.95 * rmax
p_nz = 2
p_nr = 3
p_nt = 6
uz_m = 0.0
L_vacuum = ramp_up
v_window = c
L_interact = p_zmax - p_zmin + L_vacuum
T_interact = L_interact / c
diag_period = 20000
save_checkpoints = True
checkpoint_period = 100000
use_restart = False
track_electrons = False
rel_delta_n_over_w2 = 1.0 / (np.pi * 2.81e-15 * w_matched ** 4 * n_e)


def dens_func(z, r):
    n = np.ones_like(z)
    inv_ramp_up = 1.0 / ramp_up
    n = np.where(z < ramp_up, (z * inv_ramp_up) ** 2, n)
    inv_ramp_down = 1.0 / ramp_down
    n = np.where(
        (z >= ramp_up + plateau) & (z < ramp_up + plateau + ramp_down),
        -(z - (ramp_up + plateau + ramp_down)) * inv_ramp_down,
        n,
    )
    n = np.where(z >= ramp_up + plateau + ramp_down, 0, n)
    n = np.where(z <= 0.0, 0.0, n)
    n = n * (1.0 + rel_delta_n_over_w2 * r ** 2)
    return n


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
if __name__ == "__main__":
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
    profile = GaussianLaser(
        a0=a0, waist=w0, tau=tau, z0=z0, lambda0=lambdad, zf=zfoc, phi2_chirp=GDD
    )
    add_laser_pulse(sim, profile)
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
    N_step = int(T_interact / sim.dt)
    sim.step(N_step)
