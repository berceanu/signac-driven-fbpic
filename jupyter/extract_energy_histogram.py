import os
import numpy as np
import signac
from opmd_viewer import OpenPMDTimeSeries

# ugly hack to import project.py from 'signac/src'
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath('')), 'signac', 'src'))
from project import particle_energy_histogram

del sys.path[0], sys

pr = signac.get_project(root="../signac", search=False)

# We see there are a few values of a0 in the project. We now want the job id of the job with a certain a0 value.
job_set = pr.find_job_ids({'a0': 3.2})  # 3 on laptop, 3.2 on ServerS
job_id = next(iter(job_set))

# get the job handler
job = pr.open_job(id=job_id)

# get path to job's hdf5 files
h5_path = os.path.join(job.ws, "diags", "hdf5")

# open the full time series and see iteration numbers
time_series = OpenPMDTimeSeries(h5_path, check_all_files=True)

# compute 1D histogram
energy_hist, bin_edges, nbins = particle_energy_histogram(
    tseries=time_series,
    it=time_series.iterations[-1],
    cutoff=np.inf,  # no cutoff
)

np.savez('histogram', edges=bin_edges, counts=energy_hist)

npzfile = np.load('histogram.npz')
print(npzfile.files)

edges = npzfile['edges']
counts = npzfile['counts']