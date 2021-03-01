import xarray as xr
import numpy as np
import pint_xarray
import pint
import sys
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib import figure, rc_context, cm, colors, ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import mpl_util
import util
from pint.registry import UnitRegistry

from dataclasses import dataclass, field


@dataclass
class XSpectra:
    differential_charge: xr.DataArray
    ureg: UnitRegistry = pint.UnitRegistry()

    def __post_init__(self):
        self.differential_charge.pint.quantify()

    def matshow(self):
        mat = self.find_main_peak()

        Y = util.corners(self.differential_charge.a_0.values)

        Q = self.differential_charge.n_e.values * self.ureg(
            self.differential_charge.n_e.units
        )
        n_e = Q.to("1 / centimeter ** 3")
        X = util.corners(n_e.magnitude / 1e18)

        with rc_context():
            mpl_util.mpl_publication_style()

            fig = figure.Figure()
            canvas = FigureCanvasAgg(fig)
            ax = fig.add_subplot(1, 1, 1, aspect=1)

            img = ax.pcolorfast(
                X,
                Y,
                mat,
                norm=colors.Normalize(vmin=mat.min() - 0.5, vmax=mat.max() + 0.5),
                cmap=cm.get_cmap("Reds", np.max(mat) - np.min(mat) + 1),
            )

            ax.xaxis.set_major_locator(
                ticker.FixedLocator(self.differential_charge.n_e.values / 1.0e24)
            )
            ax.yaxis.set_major_locator(
                ticker.FixedLocator(self.differential_charge.a_0.values)
            )
            ax.xaxis.set_minor_locator(ticker.NullLocator())
            ax.yaxis.set_minor_locator(ticker.NullLocator())

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.02)
            cbar = ax.figure.colorbar(
                img,
                cax=cax,
            )
            cbar.set_label(r"E ($\mathrm{MeV}$)")

            ax.set_xlabel(r"$n_e$ ($10^{18}\,\mathrm{electrons\,cm^{-3}}$)")
            ax.set_ylabel(r"$a_0$")

            # plot mat

            fig.savefig("matshow")

    def find_main_peak(self):
        peak_indices = self.differential_charge.argmax(dim="E")
        return self.differential_charge.E[peak_indices]


def gaussian(x, mu, sig, h):
    return h * np.exp(-np.power((x - mu) / sig, 2) / 2)


def main():
    """Main entry point."""
    rng = np.random.default_rng(seed=42)

    a_0 = np.linspace(2.4, 3.1, 8)
    n_e = np.linspace(7.4, 8.1, 8) * 1.0e18 * 1.0e6
    energy = np.linspace(1, 500, 500)

    peak_sigma = rng.integers(45, 55, (a_0.size, n_e.size))
    peak_height = rng.integers(30, 50, (a_0.size, n_e.size))
    peak_energy = rng.normal(200, 50, (a_0.size, n_e.size)).round().astype(int)

    charge = gaussian(
        x=energy,
        mu=peak_energy[:, :, np.newaxis],
        sig=peak_sigma[:, :, np.newaxis],
        h=peak_height[:, :, np.newaxis],
    )

    spectra = xr.DataArray(
        charge,
        dims=("a_0", "n_e", "E"),
        coords={"a_0": a_0, "n_e": n_e, "E": energy},
    )
    spectra.attrs["long_name"] = r"$\frac{\mathrm{d} Q}{\mathrm{d} E}$"
    spectra.attrs["units"] = "pC / MeV"
    spectra.a_0.attrs["units"] = "dimensionless"
    spectra.n_e.attrs["units"] = "1 / meter ** 3"
    spectra.E.attrs["units"] = "MeV"

    xs = XSpectra(spectra)

    print(peak_energy)
    print(xs.find_main_peak())

    # print(xs.differential_charge.sel(a_0=2.4, n_e=7.6e24, method="nearest"))

    # 2.54 * ureg("dimensionless")

    # xs.matshow()


if __name__ == "__main__":
    main()
