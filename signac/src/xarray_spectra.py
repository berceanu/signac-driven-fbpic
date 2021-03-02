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
        self.charge.E.attrs["plot_label"] = r"$E$ ($\mathrm{MeV}$)"
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

    def sample(self, dim_val, other_dim):
        dim = next(iter(dim_val))
        assert dim != other_dim, "other_dim can't be equal to dim."
        mat = self.charge.sel(dim_val, method="nearest")

        # get the other dimension's corners
        other_coord = self.get_coordinate(other_dim)
        other_corners = util.corners(other_coord)

        with rc_context():
            mpl_util.mpl_publication_style()

            fig = figure.Figure()
            canvas = FigureCanvasAgg(fig)
            ax = fig.add_subplot(111)

            img = ax.pcolorfast(
                self.charge.E.values,
                other_corners,
                mat.values,
                cmap=cm.get_cmap("turbo"),
                rasterized=True,
            )
            #
            ax.set_title(f"${dim} = {dim_val[dim]}$")
            ax.hlines(
                y=other_corners,
                xmin=self.charge.E[0],
                xmax=self.charge.E[-1],
                linewidth=0.5,
                linestyle="solid",
                color="white",
            )
            for v in other_coord:
                ax.annotate(
                    text=f"{v:.1f}", xy=(5, v), xycoords="data", color="white", fontsize=7
                )
            #
            ax.yaxis.set_minor_locator(ticker.NullLocator())
            ax.yaxis.set_minor_formatter(ticker.NullFormatter())
            ax.yaxis.set_major_locator(ticker.NullLocator())
            ax.yaxis.set_major_formatter(ticker.NullFormatter())
            #
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            #
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="6%", pad=0.02)
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
            cbar.set_label(self.charge.plot_label)
            #
            ax.set_ylabel(self.charge.coords[other_dim].attrs.get("plot_label", ""))
            ax.set_xlabel(self.charge.E.plot_label)
            #
            fig.savefig("sample", bbox_inches="tight")

    def matshow(self, dims=None):
        if dims is None:
            dims = {"y": "n_e", "x": "a_0"}

        mat = self.find_main_peak()

        mat_dims = {"y": mat.dims[0], "x": mat.dims[1]}

        if (mat_dims["y"] == dims["x"]) and (mat_dims["x"] == dims["y"]):
            mat = mat.transpose()

        mat = mat.values

        axes = dict()
        for ax, dim in dims.items():
            c = self.get_coordinate(dim)
            axes[ax] = {
                "values": c,
                "corners": util.corners(c),
                "label": self.charge.coords[dim].attrs.get("plot_label", ""),
            }

        with rc_context():
            mpl_util.mpl_publication_style()

            fig = figure.Figure()
            canvas = FigureCanvasAgg(fig)
            ax = fig.add_subplot(111, aspect=1)

            img = ax.pcolorfast(
                axes["x"]["corners"],
                axes["y"]["corners"],
                mat,
                norm=colors.Normalize(vmin=mat.min() - 0.5, vmax=mat.max() + 0.5),
                cmap=cm.get_cmap("Greys", mat.max() - mat.min() + 1),
                rasterized=True,
            )
            ax.xaxis.set_minor_locator(ticker.NullLocator())
            ax.yaxis.set_minor_locator(ticker.NullLocator())
            ax.xaxis.set_major_locator(ticker.FixedLocator(axes["x"]["values"]))
            ax.yaxis.set_major_locator(ticker.FixedLocator(axes["y"]["values"]))
            #
            ax.hlines(
                y=axes["y"]["corners"],
                xmin=axes["x"]["corners"][0],
                xmax=axes["x"]["corners"][-1],
                linewidth=0.5,
                linestyle="solid",
                color="white",
            )
            ax.vlines(
                x=axes["x"]["corners"],
                ymin=axes["y"]["corners"][0],
                ymax=axes["y"]["corners"][-1],
                linewidth=0.5,
                linestyle="solid",
                color="white",
            )
            #
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
            cbar.set_label(r"$E_{\mathrm{p}}$ ($\mathrm{MeV}$)")
            #
            ax.set_ylabel(axes["y"]["label"])
            ax.set_xlabel(axes["x"]["label"])
            #
            fig.savefig("matshow", bbox_inches="tight")

    def find_main_peak(self, energy_window=slice(100, 300)):
        peak_idx = self.charge.sel(E=energy_window).argmax(dim="E")
        return self.charge.E.sel(E=energy_window)[peak_idx]


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

    xs.sample({"n_e": 7.9e24}, "a_0")
    # xs.sample({"a_0": 3.1}, "n_e")

    # TODO fix n_e = .. ax title (10^18...)
    # TODO add sample line on 2D plot
    # TODO update 2D plot when sample is changed

if __name__ == "__main__":
    main()
