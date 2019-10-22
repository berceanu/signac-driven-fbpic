import numpy as np
import sliceplots
from matplotlib import pyplot

npzfile = np.load('histogram.npz')
print(npzfile.files)

edges = npzfile['edges']
counts = npzfile['counts']


x_axis = np.array([edges[:-1], edges[1:]]).T.flatten()
y_axis = np.array([counts, counts]).T.flatten()

# plot it
fig, ax = pyplot.subplots(figsize=(10, 6))
sliceplots.plot1d(
    ax=ax,
    v_axis=y_axis,
    h_axis=x_axis,
    xlabel=r"E (MeV)",
    ylabel=r"dQ/dE (pC/MeV)",
    xlim=[1.0, 800.0],  # TODO: hard-coded magic number
    ylim=[0.0, 1.0],  # TODO: hard-coded magic number
)
fig.show()
