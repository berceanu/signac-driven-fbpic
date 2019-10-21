# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import holoviews as hv
from holoviews import dim, opts, streams, param

hv.extension('bokeh')

# %%
import os
import numpy as np
import signac

from opmd_viewer import OpenPMDTimeSeries

# %%
np.set_printoptions(precision=2, linewidth=80)

# %%
# ugly hack to import project.py from 'signac/src'
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath('')), 'signac', 'src'))
from project import particle_energy_histogram

del sys.path[0], sys

# %%
pr = signac.get_project(root="../signac", search=False)

# %%
# We see there are a few values of a0 in the project. We now want the job id of the job with a certain a0 value.
job_id_set = pr.find_job_ids({'a0': 3})
job_id = next(iter(job_id_set))
print(job_id)  # this is also the name of the job's workspace folder

# %%
# What are the full job parameters?
print(pr.get_statepoint(job_id))

# %%
# get the job handler
job = pr.open_job(id=job_id)

# %%
# We can now access individual parameters, like so
print(job.sp.N_step)

# %%
# get path to job's hdf5 files
h5_path = os.path.join(job.ws, "diags", "hdf5")
print(h5_path)

# %%
# open the full time series and see iteration numbers
time_series = OpenPMDTimeSeries(h5_path, check_all_files=True)
print(time_series.iterations)
iteration = time_series.iterations[-1]
print(iteration)

# %%
help(particle_energy_histogram)

# %%

# %%
# compute 1D histogram
energy_hist, bin_edges, nbins = particle_energy_histogram(
    tseries=time_series,
    it=iteration,
    cutoff=np.inf,  # no cutoff
)

# %%
delta_E = bin_edges[1] - bin_edges[0]  # MeV

# %%
energy = hv.Dimension('energy', label='E', unit='MeV')
count = hv.Dimension('frequency', label='dQ/dE', unit='pC/MeV')

# %%
histogram = hv.Histogram((bin_edges, energy_hist), kdims=energy, vdims=count)
curve = hv.Curve(histogram)

# %%
e_min = dim('energy').min().apply(curve)
e_max = dim('energy').max().apply(curve)


def integral(limit_a, limit_b, y):
    limit_a = e_min if limit_a is None else np.clip(limit_a, e_min, e_max)
    limit_b = e_max if limit_b is None else np.clip(limit_b, e_min, e_max)
    area = hv.Area((curve.dimension_values('energy'), curve.dimension_values('frequency')))[limit_a:limit_b]
    charge = delta_E * histogram[limit_a:limit_b].dimension_values('frequency').sum()
    return curve * area * hv.VLine(limit_a) * hv.VLine(limit_b) * hv.Text(limit_b - 2., 5, 'Q = %.0f pC' % charge)


# %%
integral_streams = [
    streams.Stream.define('Iteration', iteration=param.Integer(default=4600, doc='Time step in the simulation'))(),
    streams.PointerX().rename(x='limit_b'),
    streams.Tap().rename(x='limit_a')
]

# %%
integral_dmap = hv.DynamicMap(integral, streams=integral_streams)

# %%
integral_dmap.opts(
    opts.Area(color='#fff8dc', line_width=2),
    opts.Curve(color='black', height=300, responsive=True, show_grid=True, xlim=(None, 25), ylim=(None, 10)),
    opts.VLine(color='red'))

# %%
