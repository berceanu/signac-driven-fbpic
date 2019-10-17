import os

from matplotlib import pyplot
from opmd_viewer.openpmd_timeseries.data_reader.field_reader import read_field_circ
from opmd_viewer.openpmd_timeseries.utilities import combine_cylindrical_components
from sliceplots import Plot2D

import numpy as np
from opmd_viewer import OpenPMDTimeSeries, FieldMetaInformation
from scipy.constants import physical_constants

import h5py
from src.lwfa_script import Nr, Nz, zmin, zmax, rmax

c_light = physical_constants["speed of light in vacuum"][0]
m_e = physical_constants["electron mass"][0]
q_e = physical_constants["elementary charge"][0]

# wavevector
k0 = 2 * np.pi / 0.8e-6
# field amplitude
e0 = m_e * c_light ** 2 * k0 / q_e

# This is the most high-level mode of extracting the Ex component of the electric field from the .h5 output
# h5_path = os.path.join("diags", "hdf5")
# time_series = OpenPMDTimeSeries(h5_path, check_all_files=False)
# iterations = time_series.iterations
# field, info = time_series.get_field(field="E", coord="x", iteration=iterations[0], m="all", theta=0.0)

# This is a lower-level approach, using some of fbpic's internal functions
# FFr, info = read_field_circ('/home/berceanu/Development/signac-driven-fbpic/signac/diags/hdf5/data00000000.h5', 'E/r', 'all', 0.0)
# FFt, info = read_field_circ('/home/berceanu/Development/signac-driven-fbpic/signac/diags/hdf5/data00000000.h5', 'E/t', 'all', 0.0)
# FF = combine_cylindrical_components(FFr, FFt, 0.0, 'x', info)


# And finally, this is the lowest-level approach, working directly with h5py
# # Extract the modes and recombine them properly
Er = np.zeros((2 * Nr, Nz))

# # - Sum the modes
dfile = h5py.File('/home/berceanu/Development/signac-driven-fbpic/signac/diags/hdf5/data00000000.h5', 'r')
Fr = dfile['/data/0/fields/E/r'][...]  # (Extracts all modes)

Er[Nr:, :] = Fr[0, :, :] + Fr[1, :, :]
Er[:Nr, :] = (-Fr[0, :, :] + Fr[1, :, :])[::-1, :]

# Fr[0, :, :] = fld.interp[0].Er.T.real

# There is a factor 2 here so as to comply with the convention in
# Lifschitz et al., which is also the convention adopted in Warp Circ
# Fr[1, :, :] = 2 * fld.interp[1].Er.T.real

# This is not used when theta = 0.
# Fr[2, :, :] = 2 * fld.interp[1].Er.T.imag

#######
Ex = Er
#######

# r
dr = rmax / Nr
rstart = 0.5 * dr
rend = rstart + (Nr - 1) * dr
r = np.linspace(rstart, rend, Nr, endpoint=True)
r = np.concatenate((-r[::-1], r))

# z
dz = (zmax - zmin) / Nz
zstart = zmin + 0.5 * dz
zend = zstart + (Nz - 1) * dz
z = np.linspace(zstart, zend, Nz, endpoint=True)

fig = pyplot.figure(figsize=(8, 8))
Plot2D(
    fig=fig,
    arr2d=Ex / e0,
    h_axis=z * 1e6,
    v_axis=r * 1e6,
    zlabel=r"$E_x/E_0$",
    xlabel=r"$z \;(\mu m)$",
    ylabel=r"$r \;(\mu m)$",
    extent=(
        z[0] * 1e6 + 40,
        z[-1] * 1e6 - 20,
        r[0] * 1e6 + 15,
        r[-1] * 1e6 - 15,
    ),
    cbar=True,
    vmin=-5,
    vmax=5,
    hslice_val=0.0,  # do a 1D slice through the middle of the simulation box
)
fig.show()



