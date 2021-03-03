"""
Provides N-dimensional data structure for storing spectra depending on multiple
statepoint parameters.
"""
from dataclasses import dataclass, field

import numpy as np
import pint
import pint_xarray
import xarray as xr
from matplotlib import cm, colors, figure, rc_context, ticker
from matplotlib.backends.backend_agg import FigureCanvasAgg
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pint.registry import UnitRegistry
from typing import Dict

import mpl_util
import util


def gaussian(x, mu, sig, h):
    return h * np.exp(-np.power((x - mu) / sig, 2) / 2)


@dataclass
class XSpectra:
    charge: xr.DataArray
    ureg: UnitRegistry = pint.UnitRegistry()
    dim_mapping: Dict[str, str] = field(default_factory=lambda: {"y": "n_e", "x": "a0"})

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

    def get_coordinate(self, dim, values=None):
        c = self.charge.coords[dim]

        if values is None:
            values = c.values

        units = c.attrs.get("units", "dimensionless")
        to_units = c.attrs.get("to_units", units)
        scale = c.attrs.get("scaling_factor", 1)

        c_with_units = values * self.ureg(units)
        c_converted = c_with_units.to(to_units)
        c_scaled = c_converted * scale
        return c, c_scaled.magnitude

    def sample(self, dim_val, other_dim, dim_mapping=None):
        dim = next(iter(dim_val))
        assert dim != other_dim, "other_dim can't be equal to dim."

        c, c_value = self.get_coordinate(dim, values=dim_val[dim])

        def create_title():
            l = c.plot_label.split("$")
            l[2] = f" = {c_value:.1f}" + l[2]
            return "$".join(l)

        if dim_mapping is None:
            dim_mapping = self.dim_mapping
        inv_dim_mapping = {v: k for k, v in dim_mapping.items()}
        self.matshow(dim_mapping=dim_mapping, axline={inv_dim_mapping[dim]: c_value})

        mat = self.charge.sel(dim_val, method="nearest")

        # get the other dimension's corners
        c_other_coord, other_coord_val = self.get_coordinate(other_dim)
        other_corners = util.corners(other_coord_val)

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
            ax.set_title(create_title())
            ax.hlines(
                y=other_corners,
                xmin=self.charge.E[0],
                xmax=self.charge.E[-1],
                linewidth=0.5,
                linestyle="solid",
                color="white",
            )
            for v in other_coord_val:
                ax.text(5, v, f"{v:.1f}", color="white", fontsize=7)
            #
            ax.yaxis.set(
                minor_locator=ticker.NullLocator(),
                minor_formatter=ticker.NullFormatter(),
                major_locator=ticker.NullLocator(),
                major_formatter=ticker.NullFormatter(),
            )
            #
            for pos in "right", "left", "top", "bottom":
                ax.spines[pos].set_visible(False)
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
            ax.set_ylabel(c_other_coord.attrs.get("plot_label", ""))
            ax.set_xlabel(self.charge.E.plot_label)
            #
            fig.savefig("sample", bbox_inches="tight")

    def matshow(self, dim_mapping=None, axline=None):
        if dim_mapping is None:
            dim_mapping = self.dim_mapping

        mat = self.find_main_peak()

        mat_dims = {"y": mat.dims[0], "x": mat.dims[1]}

        if (mat_dims["y"] == dim_mapping["x"]) and (mat_dims["x"] == dim_mapping["y"]):
            mat = mat.transpose()

        mat = mat.values

        axes = dict()
        for ax, dim in dim_mapping.items():
            c, c_val = self.get_coordinate(dim)
            axes[ax] = {
                "values": c_val,
                "corners": util.corners(c_val),
                "label": c.attrs.get("plot_label", ""),
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
            for xy, ax_xy in zip(("x", "y"), (ax.xaxis, ax.yaxis)):
                ax_xy.set(
                    minor_locator=ticker.NullLocator(),
                    major_locator=ticker.FixedLocator(axes[xy]["values"]),
                )
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
            xy = next(iter(axline))
            {"x": ax.axvline, "y": ax.axhline}[xy](
                axline[xy],
                color="white",
                linestyle="dashed",
                linewidth=2,
            )
            #
            fig.savefig("matshow", bbox_inches="tight")

    def find_main_peak(self, energy_window=slice(100, 300)):
        peak_idx = self.charge.sel(E=energy_window).argmax(dim="E")
        return self.charge.E.sel(E=energy_window)[peak_idx]


def main():
    """Main entry point."""
    rng = np.random.default_rng(seed=42)

    a0 = np.linspace(2.4, 3.1, 8)
    n_e = np.linspace(7.4, 8.1, 8) * 1.0e18 * 1.0e6
    energy = np.linspace(1, 500, 500)

    peak_sigma = rng.integers(45, 55, (a0.size, n_e.size))
    peak_height = rng.integers(30, 50, (a0.size, n_e.size))
    peak_energy = rng.normal(200, 50, (a0.size, n_e.size)).round().astype(int)

    charge = gaussian(
        x=energy,
        mu=peak_energy[:, :, np.newaxis],
        sig=peak_sigma[:, :, np.newaxis],
        h=peak_height[:, :, np.newaxis],
    )

    spectra = xr.DataArray(
        charge,
        dims=("a0", "n_e", "E"),
        coords={"a0": a0, "n_e": n_e, "E": energy},
    )
    #
    spectra.a0.attrs["plot_label"] = r"$a0$"
    #
    spectra.n_e.attrs["plot_label"] = r"$n_e$ ($10^{18}\,\mathrm{cm^{-3}}$)"
    spectra.n_e.attrs["units"] = "1 / meter ** 3"
    spectra.n_e.attrs["to_units"] = "1 / centimeter ** 3"
    spectra.n_e.attrs["scaling_factor"] = 1.0e-18
    ##

    xs = XSpectra(spectra)
    # xs.sample({"a0": 3.1}, "n_e")
    xs.sample({"n_e": 7.9e24}, "a0")


if __name__ == "__main__":
    main()
