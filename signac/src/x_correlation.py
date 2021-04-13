# -*- coding: utf-8 -*-
"""
Load spectra from file into a 3D xarray.
Usage: python load_spectra.py
Requires xarray: http://xarray.pydata.org/en/stable/installing.html#instructions
"""
import xarray as xr
from matplotlib import pyplot as plt
import numpy as np

import pandas as pd
import util

from icecream import ic

ic.configureOutput(includeContext=True)


def read_spectrum(path_to_csv):
    """Read spectrum data from CSV file, return dataframe."""
    csv_df = pd.read_csv(
        path_to_csv,
        comment="#",
        names=["E_MeV_float", "dN_over_dE"],
    )
    csv_df.loc[:, "E_MeV"] = csv_df.loc[:, "E_MeV_float"].astype(int)
    grouped = csv_df.groupby(["E_MeV"])
    df_ = grouped[["dN_over_dE"]].agg(["mean"])
    df_.columns = df_.columns.get_level_values(0)
    df_["dN_over_dE"] = df_["dN_over_dE"].astype(np.float64)
    df_["dN_over_dE_normalized"] = util.normalize_to_interval(0, 1, df_["dN_over_dE"])

    print(df_.index)
    return df_


def ExtractSpectrum(power, n_e):
    """Main entry point."""
    # load xarray dataset from file
    ds_spectra = xr.open_dataset("spectra.nc")
    # extract the (only) array from the dataset
    spectra = ds_spectra["spectra"]
    Num_Spectrum = spectra.sel(power=power, n_e=n_e, method="nearest")

    return Num_Spectrum


# Read the experimental spectrum
# Exp_Spectrum_File = open('Experimental.txt','r')
Energy_min = 70  # minimum enregy, MeV
Energy_max = 450  # maximum energy, MeV

Exp_Spectrum = np.loadtxt("Experimental.txt")
Exp_Energy_full = Exp_Spectrum[:, 0]  # Exp. energy scan
Exp_Counts_full = Exp_Spectrum[:, 1]  # Exp. spectrum

# Selected energy range

Exp_Energy = Exp_Energy_full[
    (Exp_Energy_full >= Energy_min) * (Exp_Energy_full < Energy_max)
]
Exp_Counts = Exp_Counts_full[
    (Exp_Energy_full >= Energy_min) * (Exp_Energy_full < Energy_max)
]


# Normalyze Counts:
Exp_Counts = Exp_Counts / np.max(Exp_Counts)


# SIMULATIONS
Npower = 8
Nn = 8

Power = np.linspace(1.5, 3, Npower)  # power set
N_e = np.linspace(7.4, 8.1, Nn) * 1e24  # electron density set, 1/m^3
Cutoff_from_maximum = 0.8  # Data weights are increasingly reduced below that threshold


XCorr = np.zeros((Npower, Nn))  # The output weighted chi^2  matrix
fig = plt.figure()
axs = fig.add_subplot(111)
plt.title("E4 Exp vs FB-PIC Spectrum")
axs.step(Exp_Energy, Exp_Counts, label="Experimental")

for i in range(0, Nn):
    power = Power[i]

    for j in range(0, Nn):
        n_e = N_e[j]

        # Read data
        Num_Spectrum = ExtractSpectrum(power, n_e)

        # Numerical spectrum
        Num_Energy_full = Num_Spectrum.E.values  # Exp. energy scan
        Num_Counts_full = Num_Spectrum.values  # Exp. spectrum
        Num_Energy = Num_Energy_full[
            (Num_Energy_full >= Energy_min) * (Num_Energy_full < Energy_max)
        ]
        Num_Counts = Num_Counts_full[
            (Num_Energy_full >= Energy_min) * (Num_Energy_full < Energy_max)
        ]

        # Find the (weighted) Exp/Num matching
        Num_Counts_interp = np.interp(Exp_Energy, Num_Energy, Num_Counts)
        Num_Counts_interp = Num_Counts_interp / np.max(Num_Counts_interp)

        Weight = np.tanh((Exp_Counts / Cutoff_from_maximum) ** 2)

        Rel_deviation = np.sum(Num_Counts_interp * Exp_Counts) ** 2 / (
            np.sum(Num_Counts_interp) ** 2 * np.sum(Exp_Counts) ** 2
        )
        xcorr = np.sum(Weight * Rel_deviation)
        XCorr[i, j] = xcorr


# Best case (maximum xcorr)
result = np.where(XCorr == XCorr.max())
power = Power[np.int(result[0])]
n_e = N_e[np.int(result[1])]

Num_Spectrum = ExtractSpectrum(power, n_e)
Num_Energy_full = Num_Spectrum.E.values  # Exp. energy scan
Num_Counts_full = Num_Spectrum.values  # Exp. spectrum

Num_Energy = Num_Energy_full[
    (Num_Energy_full >= Energy_min) * (Num_Energy_full < Energy_max)
]
Num_Counts = Num_Counts_full[
    (Num_Energy_full >= Energy_min) * (Num_Energy_full < Energy_max)
]
Num_Counts_interp = np.interp(Exp_Energy, Num_Energy, Num_Counts)
Num_Counts_interp = Num_Counts_interp / np.max(Num_Counts_interp)
axs.step(Exp_Energy, Num_Counts_interp, label="FB-PIC (automatic)")
axs.set_xlabel(r"$E$ ($\mathrm{MeV}$)")
axs.set_ylabel(r"$\frac{\mathrm{d} N}{\mathrm{d} E}$ ($\mathrm{a.u.}$)")

print("Optimised density  \t\t= ", n_e, "$1/m^3$")
print("Optimised power  \t\t= ", power)

# By visual inspection (another possibility)

power = 3
n_e = 8.1e24
Num_Spectrum = ExtractSpectrum(power, n_e)
Num_Energy_full = Num_Spectrum.E.values  # Exp. energy scan
Num_Counts_full = Num_Spectrum.values  # Exp. spectrum

Num_Energy = Num_Energy_full[
    (Num_Energy_full >= Energy_min) * (Num_Energy_full < Energy_max)
]
Num_Counts = Num_Counts_full[
    (Num_Energy_full >= Energy_min) * (Num_Energy_full < Energy_max)
]
Num_Counts_interp = np.interp(Exp_Energy, Num_Energy, Num_Counts)
Num_Counts_interp = Num_Counts_interp / np.max(Num_Counts_interp)


axs.legend()
fig.savefig("spectrum.png")