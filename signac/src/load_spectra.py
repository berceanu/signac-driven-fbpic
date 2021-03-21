"""Load spectra from file into a 3D xarray.
Usage: python load_spectra.py
Requires xarray: http://xarray.pydata.org/en/stable/installing.html#instructions
"""
import xarray as xr
from matplotlib import pyplot as plt


def main():
    """Main entry point."""
    # load xarray dataset from file
    ds_spectra = xr.open_dataset("spectra.nc")
    # extract the (only) array from the dataset
    spectra = ds_spectra["spectra"]

    # 3D array 8x8x499, with named dimensions
    print(spectra)

    fig = plt.figure()
    axs = fig.add_subplot(111)

    # select spectrum with alpha=1.4 and electron density 7.2 x 10^18 cm^(-3)
    # see http://xarray.pydata.org/en/stable/indexing.html#indexing-and-selecting-data
    # power is alpha
    charge = spectra.sel(power=1.4, n_e=7.2e24, method="nearest")

    print(charge)
    print(charge.E)

    # .values returns a simply numpy array, similar to pandas
    axs.step(charge.E.values, charge.values)

    axs.set_xlim(71, 499)
    axs.set_ylim(0, 120)

    axs.set_xlabel(r"$E$ ($\mathrm{MeV}$)")
    axs.set_ylabel(r"$\frac{\mathrm{d} N}{\mathrm{d} E}$ ($\mathrm{a.u.}$)")

    fig.savefig("spectrum.png")


if __name__ == "__main__":
    main()
