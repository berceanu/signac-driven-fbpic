import numpy as np
from matplotlib import pyplot
from dataclasses import dataclass

from scipy import interpolate
import pandas as pd


@dataclass
class Job:
    sp: dict


job = Job(dict(zmin=-70.0e-6,
               zmax=30.0e-6,
               p_zmin=0.0e-6,
               z0=0.0e-6,
               zf=0.0e-6,
               ramp_start=0.0e-6,
               ramp_length=7.0e-6,
               L_interact=800e-6,
               p_zmax=2250.0e-6))

data = pd.read_csv("density_16.txt", delim_whitespace=True, names=["position_mu", "density_cm_3"])
data["position_m"] = data["position_mu"] * 1e-6
interp_z_min = data["position_m"].min()
interp_z_max = data["position_m"].max()

data["norm_density"] = data["density_cm_3"] / data["density_cm_3"].max()
# check density values between 0 and 1
if not data["norm_density"].between(0, 1).any():
    raise ValueError("The density contains values outside the range [0,1].")

f = interpolate.interp1d(data.position_m.values, data.norm_density.values, bounds_error=False, fill_value=(0., 0.))


def dens_func(z, r):
    # Allocate relative density
    n = np.ones_like(z)

    # only compute n if z is inside the interpolation bounds
    n = np.where(np.logical_and(z > interp_z_min, z < interp_z_max), f(z), n)

    # Make linear ramp
    n = np.where(
        z < job.sp["ramp_start"] + job.sp["ramp_length"],
        (z - job.sp["ramp_start"]) / job.sp["ramp_length"] * f(interp_z_min),
        n,
    )

    # Supress density before the ramp
    n = np.where(z < job.sp["ramp_start"], 0.0, n)

    return n


all_z = np.linspace(-70.0e-6, 2250.0e-6, 1000)
dens = dens_func(all_z, 0.0)

width_inch = 2250.0e-6 / 1e-5
major_locator = pyplot.MultipleLocator(10)
minor_locator = pyplot.MultipleLocator(5)
major_locator.MAXTICKS = 10000
minor_locator.MAXTICKS = 10000


def mark_on_plot(*, ax, parameter: str, y=1.1):
    ax.annotate(s=parameter, xy=(job.sp[parameter] * 1e6, y), xycoords="data")
    ax.axvline(x=job.sp[parameter] * 1e6, linestyle="--", color="red")
    return ax


fig, ax = pyplot.subplots(figsize=(width_inch, 4.8))
ax.plot(all_z * 1e6, dens)
ax.set_xlabel(r"$%s \;(\mu m)$" % "z")
ax.set_ylim(-0.1, 1.2)
ax.set_xlim(-70.0e-6 * 1e6 - 20, 2250.0e-6 * 1e6 + 20)
ax.set_ylabel("Density profile $n$")
ax.xaxis.set_major_locator(major_locator)
ax.xaxis.set_minor_locator(minor_locator)

mark_on_plot(ax=ax, parameter="zmin")
mark_on_plot(ax=ax, parameter="zmax")
mark_on_plot(ax=ax, parameter="p_zmin", y=0.9)
mark_on_plot(ax=ax, parameter="z0", y=0.8)
mark_on_plot(ax=ax, parameter="zf", y=0.6)
mark_on_plot(ax=ax, parameter="ramp_start", y=0.7)
mark_on_plot(ax=ax, parameter="L_interact")
mark_on_plot(ax=ax, parameter="p_zmax")

ax.annotate(s="ramp_start + ramp_length", xy=(0.0e-6 * 1e6 + 7.0e-6 * 1e6, 1.1),
            xycoords="data")
ax.axvline(x=0.0e-6 * 1e6 + 7.0e-6 * 1e6, linestyle="--", color="red")

ax.fill_between(all_z * 1e6, dens, alpha=0.5)
fig.savefig("check_density.png")
