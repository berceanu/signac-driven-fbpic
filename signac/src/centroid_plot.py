import pathlib
import numpy as np
from matplotlib import pyplot
from openpmd_viewer import OpenPMDTimeSeries


if __name__ == "__main__":
    p = pathlib.Path(
        "/scratch/berceanu/runs/signac-driven-fbpic/workspace/5cfc1556d3e859230b25e831d55175c4"
    )
    h5_path = p / "diags" / "hdf5"
    ts: OpenPMDTimeSeries = OpenPMDTimeSeries(h5_path, check_all_files=True)

    fig, ax = pyplot.subplots(figsize=(7, 5))
    z, x = ts.get_particle(
        ["z", "x"], species="bunch", iteration=ts.iterations[-1], plot=True
    )

    img = ax.get_images()[0]
    z_min, z_max, x_min, x_max = img.get_extent()
    hist_data = img.get_array()

    r, c = np.shape(hist_data)
    z_coords = np.linspace(z_min, z_max, c)
    x_coords = np.linspace(x_min, x_max, r)
    z_m, x_m = np.meshgrid(z_coords, x_coords)

    centroid = np.ma.average(x_m, weights=hist_data, axis=0)
    
    ax.plot(z_coords, centroid)
    fig.savefig("centroid.png")
