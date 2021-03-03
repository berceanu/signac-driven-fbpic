"""
Reimplementation of the particle energy histogram for the electron species, directly on top of h5py.
The resulting speedup compared to the openPMD-viewer-based `particle_energy_histogram` is a factor ~4.

To view all groups datasets and corresponding attributes in an .h5 file, use `h5ls -rv filename.h5`.
"""
import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict

import h5py
import numexpr as ne
import numpy as np
import pint
import signac
from fast_histogram import histogram1d

import job_util

ureg = pint.UnitRegistry()
m_e = ureg.electron_mass
e = ureg.elementary_charge
c = ureg.speed_of_light

e_pC = (1 * e).to("pC").magnitude
mc2 = (1 * m_e * c ** 2).to("MeV").magnitude
mc = (1 * m_e * c).to("kilogram * meter / second").magnitude


@dataclass
class LastH5File:
    job: Any = field(repr=False)
    iteration: int = field(init=False, repr=False)
    fpath: pathlib.Path = field(init=False)
    h5_path: pathlib.Path = field(init=False, repr=False)
    fname: str = field(init=False, repr=False)
    electrons: str = field(init=False, repr=False)
    mom: Dict[str, str] = field(init=False, repr=False)
    w: str = field(init=False, repr=False)

    def __post_init__(self):
        self.h5_path = job_util.is_h5_path(self.job)
        self.fname = self.last_fname()
        self.iteration = job_util.extract_iteration_number(self.fname)
        self.fpath = self.h5_path / self.fname
        self.file_obj = h5py.File(self.fpath, "r")
        self.electrons = f"/data/{self.iteration}/particles/electrons"
        self.w = f"{self.electrons}/weighting"
        self.mom = dict()
        for xyz in "x", "y", "z":
            self.mom[xyz] = f"{self.electrons}/momentum/{xyz}"

    def __enter__(self):
        return self.file_obj

    def __exit__(self, type, value, traceback):
        self.file_obj.close()

    def last_fname(self):
        fnames = job_util.get_diags_fnames(self.job)
        return tuple(fnames)[-1]


def energy_histogram(normalized_particle_momenta, weights, bins=499, range=(1, 500)):
    ux = normalized_particle_momenta["x"]
    uy = normalized_particle_momenta["y"]
    uz = normalized_particle_momenta["z"]
    expr = ne.evaluate("sqrt(1+ux**2+uy**2+uz**2)")
    return histogram1d(mc2 * expr, bins=bins, range=range, weights=e_pC * weights)


def job_energy_histogram(job):
    uxyz = dict()
    h5f = LastH5File(job)
    with h5f as f:
        w = np.array(f[h5f.w])
        for xyz in "x", "y", "z":
            # normalize momenta by mc
            uxyz[xyz] = np.array(f[h5f.mom[xyz]]) / mc
    return energy_histogram(uxyz, w)


def main():
    """Main entry point."""
    proj = signac.get_project(search=False)

    for job in proj:  # ~14s
        hist = job_energy_histogram(job)


if __name__ == "__main__":
    main()
