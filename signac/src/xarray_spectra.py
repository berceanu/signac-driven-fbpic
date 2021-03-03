"""
Provides N-dimensional data structure for storing spectra depending on multiple
statepoint parameters.
"""
from dataclasses import dataclass, field

import numpy as np
import pint
import pint_xarray
import xarray as xr
from matplotlib import cm, colors, figure, rc_context, ticker, rcParams
from matplotlib.backends.backend_agg import FigureCanvasAgg
from pint.registry import UnitRegistry
from typing import Any, Dict, Tuple
from scipy.ndimage import gaussian_filter1d
from util import first

import mpl_util
import util


def gaussian(x, mu, sig, h):
    return h * np.exp(-np.power((x - mu) / sig, 2) / 2)


@dataclass
class Slice:
    dimension_name: str
    dimension_value: float
    other_dimension_name: str

    def __post_init__(self):
        assert (
            self.dimension_name != self.other_dimension_name
        ), "Dimension names must be different."

    def to_dict(self):
        return {self.dimension_name: self.dimension_value}


def generate_slices(dimension_name, values):
    other_dimension_name = {"a0": "n_e", "n_e": "a0"}[dimension_name]
    return tuple(Slice(dimension_name, v, other_dimension_name) for v in values)


@dataclass
class XSpectra:
    charge: xr.DataArray
    ureg: UnitRegistry = pint.UnitRegistry()
    dim_mapping: Dict[str, str] = field(default_factory=lambda: {"y": "n_e", "x": "a0"})
    gaussian_std: int = 10
    vmax: float = 40.0
    left_xlim: float = 50.0

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

    def get_coordinate(self, dim, *, values=None):
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

    def sample(self, ax, s: Slice):
        c, c_value = self.get_coordinate(s.dimension_name, values=s.dimension_value)

        def create_title():
            l = c.plot_label.split("$")
            l[2] = f" = {c_value:.1f}" + l[2]
            return "$".join(l)

        # get the other dimension's corners
        c_other_coord, other_coord_val = self.get_coordinate(s.other_dimension_name)
        other_corners = util.corners(other_coord_val)

        mat = self.charge.sel(s.to_dict(), method="nearest")

        img = ax.pcolorfast(
            self.charge.E.values,
            other_corners,
            mat.values,
            norm=colors.Normalize(vmin=0.0, vmax=self.vmax),
            cmap=cm.get_cmap("turbo"),
            rasterized=True,
        )
        ax.set_title(create_title(), fontsize=6)
        ax.hlines(
            y=other_corners,
            xmin=self.charge.E[0],
            xmax=self.charge.E[-1],
            linewidth=0.5,
            linestyle="solid",
            color="white",
        )
        for v in other_coord_val:
            ax.text(self.left_xlim + 5, v, f"{v:.1f}", color="white", fontsize=6)
        #
        ax.yaxis.set(
            minor_locator=ticker.NullLocator(),
            minor_formatter=ticker.NullFormatter(),
            major_locator=ticker.NullLocator(),
            major_formatter=ticker.NullFormatter(),
        )
        for pos in "right", "left", "top", "bottom":
            ax.spines[pos].set_visible(False)
        #
        ax.set_ylabel(c_other_coord.attrs.get("plot_label", ""))
        ax.set_xlim(left=self.left_xlim)
        ax.set_xlabel(self.charge.E.plot_label)
        #
        cbar = mpl_util.add_colorbar(ax, img)
        cbar.ax.set_title(self.charge.plot_label)
        #
        return img, cbar

    def matshow(self, ax):
        mat = self.find_main_peak()
        mat_dims = {"y": mat.dims[0], "x": mat.dims[1]}
        if (mat_dims["y"] == self.dim_mapping["x"]) and (
            mat_dims["x"] == self.dim_mapping["y"]
        ):
            mat = mat.transpose()

        axes = dict()
        for xy, dim in self.dim_mapping.items():
            c, c_val = self.get_coordinate(dim)
            axes[xy] = {
                "values": c_val,
                "corners": util.corners(c_val),
                "label": c.attrs.get("plot_label", ""),
            }

        mat = mat.values
        img = ax.pcolorfast(
            axes["x"]["corners"],
            axes["y"]["corners"],
            mat,
            norm=colors.Normalize(vmin=mat.min() - 0.5, vmax=mat.max() + 0.5),
            cmap=cm.get_cmap("turbo", mat.max() - mat.min() + 1),
            rasterized=True,
        )
        for xy, ax_xy in zip(("x", "y"), (ax.xaxis, ax.yaxis)):
            ax_xy.set(  # TODO remove
                minor_locator=ticker.NullLocator(),
                major_locator=ticker.FixedLocator(axes[xy]["values"]),
            )
        for xy in "x", "y":
            {"y": ax.hlines, "x": ax.vlines}[xy](
                axes[xy]["corners"],
                axes[{"x": "y", "y": "x"}[xy]]["corners"][0],
                axes[{"x": "y", "y": "x"}[xy]]["corners"][-1],
                linewidth=0.5,
                linestyle="solid",
                color="white",
            )
        ax.set_ylabel(axes["y"]["label"])
        ax.set_xlabel(axes["x"]["label"])
        #
        cbar = mpl_util.add_colorbar(
            ax,
            img,
        )
        cbar.ax.set_title(r"$E_{\mathrm{p}}$ ($\mathrm{MeV}$)")
        #
        return img, cbar

    def find_main_peak(self, energy_window=slice(100, 300)):
        numpy_charge = self.charge.values
        smooth_charge = gaussian_filter1d(numpy_charge, self.gaussian_std, axis=-1)
        new_charge = xr.DataArray(
            smooth_charge, dims=self.charge.dims, attrs=self.charge.attrs.copy()
        )
        peak_idx = new_charge.sel(E=energy_window).argmax(dim="E")
        return new_charge.E.sel(E=energy_window)[peak_idx]


@dataclass
class XFigure:
    spectra: XSpectra = field(repr=False)
    slices: Tuple[Slice]
    fig: Any = field(init=False)
    axd: Dict[str, Any] = field(init=False)
    layout: str = """
                   ABC
                   DEF
                   GHI
                   """

    def __post_init__(self):
        self.fig = figure.Figure()
        _ = FigureCanvasAgg(self.fig)
        with rc_context():
            mpl_util.mpl_publication_style()
            self.axd = self.create_axes()

    def create_axes(self):
        axd = self.fig.subplot_mosaic(
            self.layout,
            gridspec_kw={
                "width_ratios": [1, 1, 1],
                "height_ratios": [1, 1, 1],
                "wspace": 0.26,
                "hspace": 0.5,
                "left": 0.03,
                "right": 0.99,
                "top": 0.96,
                "bottom": 0.03,
            },
        )
        return axd

    def render(self):
        with rc_context():
            mpl_util.mpl_publication_style()

            im, cb = {}, {}
            for count, character in enumerate("ABCDFGHI"):
                im[character], cb[character] = self.spectra.sample(
                    self.axd[character], self.slices[count]
                )

            im["E"], cb["E"] = self.spectra.matshow(self.axd["E"])

            for character in "BCDFGHI":
                cb[character].remove()
                self.axd[character].set_xlabel("")

            for character in "AE":
                for ticks in "minor", "major":
                    cb[character].ax.tick_params(
                        labelsize=5, which=ticks, direction="out"
                    )

    def savefig(self):
        with rc_context():
            mpl_util.mpl_publication_style(extension="png")
            self.fig.savefig("xfig")


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
    spectra.a0.attrs["plot_label"] = r"$a_0$"
    #
    spectra.n_e.attrs["plot_label"] = r"$n_e$ ($10^{18}\,\mathrm{cm^{-3}}$)"
    spectra.n_e.attrs["units"] = "1 / meter ** 3"
    spectra.n_e.attrs["to_units"] = "1 / centimeter ** 3"
    spectra.n_e.attrs["scaling_factor"] = 1.0e-18
    ##

    xs = XSpectra(
        spectra,
        dim_mapping={"y": "a0", "x": "n_e"},
        gaussian_std=10,
        vmax=40.0,
        left_xlim=50.0,
    )
    # xs.sample({"a0": 3.1}, "n_e")
    # xs.sample({"n_e": 7.9e24}, "a0", vmax=40.0, left_xlim=50.0)

    s1 = generate_slices("a0", (2.4, 2.6, 2.7, 3.1))
    s2 = generate_slices("n_e", np.array((7.4, 7.6, 7.9, 8.0)) * 1.0e24)
    s = s1 + s2

    xf = XFigure(xs, s)
    xf.render()
    xf.savefig()


if __name__ == "__main__":
    main()
