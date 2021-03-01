from dataclasses import dataclass

import numpy as np
import pint
import pint_xarray
import xarray as xr
from matplotlib import cm, colors, figure, rc_context, ticker
from matplotlib.backends.backend_agg import FigureCanvasAgg
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pint.registry import UnitRegistry

import mpl_util
import util


def gaussian(x, mu, sig, h):
    return h * np.exp(-np.power((x - mu) / sig, 2) / 2)


@dataclass
class XSpectra:
    charge: xr.DataArray
    ureg: UnitRegistry = pint.UnitRegistry()

    def __post_init__(self):
        self.charge.attrs["units"] = "pC / MeV"
        self.charge.attrs[
            "plot_label"
        ] = r"$\frac{\mathrm{d} Q}{\mathrm{d} E}\, \left(\frac{\mathrm{pC}}{\mathrm{MeV}}\right)$"
        #
        self.charge.E.attrs["units"] = "MeV"
        self.charge.E.attrs["plot_label"] = r"E ($\mathrm{MeV}$)"
        #
        self.charge.pint.quantify()

    def get_coordinate(self, dim):
        c = self.charge.coords[dim]
        units = c.attrs.get("units", "dimensionless")
        to_units = c.attrs.get("to_units", units)
        scale = c.attrs.get("scaling_factor", 1)

        c_with_units = c.values * self.ureg(units)
        c_converted = c_with_units.to(to_units)
        c_scaled = c_converted * scale
        return c_scaled.magnitude

    def matshow(self):
        mat = self.find_main_peak()

        axes = dict()
        for dim in self.charge.dims[:-1]:
            c = self.get_coordinate(dim)
            axes[dim] = {"values": c, "corners": util.corners(c)}

        with rc_context():
            mpl_util.mpl_publication_style()

            fig = figure.Figure()
            canvas = FigureCanvasAgg(fig)
            ax = fig.add_subplot(111, aspect=1)

            img = ax.pcolorfast(
                axes["n_e"]["corners"],
                axes["a_0"]["corners"],
                mat,
                norm=colors.Normalize(vmin=mat.min() - 0.5, vmax=mat.max() + 0.5),
                cmap=cm.get_cmap("Greys", np.max(mat) - np.min(mat) + 1),
            )
            ax.xaxis.set_minor_locator(ticker.NullLocator())
            ax.yaxis.set_minor_locator(ticker.NullLocator())
            ax.xaxis.set_major_locator(ticker.FixedLocator(axes["n_e"]["values"]))
            ax.yaxis.set_major_locator(ticker.FixedLocator(axes["a_0"]["values"]))
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.02)
            cbar = ax.figure.colorbar(
                img,
                cax=cax,
            )
            for ticks, length, width in zip(("major", "minor"), (3.5, 2), (0.8, 0.6)):
                cbar.ax.tick_params(
                    axis="both",
                    which=ticks,
                    length=length,
                    width=width,
                )
            ax.set_ylabel(self.charge.a_0.plot_label)
            ax.set_xlabel(self.charge.n_e.plot_label)
            cbar.set_label(self.charge.E.plot_label)

            fig.savefig("matshow")

    def find_main_peak(self):
        peak_idx = self.charge.argmax(dim="E")
        return self.charge.E[peak_idx]


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
    #
    spectra.a_0.attrs["plot_label"] = r"$a_0$"
    #
    spectra.n_e.attrs["plot_label"] = r"$n_e$ ($10^{18}\,\mathrm{cm^{-3}}$)"
    spectra.n_e.attrs["units"] = "1 / meter ** 3"
    spectra.n_e.attrs["to_units"] = "1 / centimeter ** 3"
    spectra.n_e.attrs["scaling_factor"] = 1.0e-18
    ##

    xs = XSpectra(spectra)
    xs.matshow()

    # print(xs.charge.sel(a_0=2.4, n_e=7.6e24, method="nearest"))


if __name__ == "__main__":
    main()
