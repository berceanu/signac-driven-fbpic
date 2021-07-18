from matplotlib import pyplot, colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from fbpic.lpa_utils.laser import FlattenedGaussianLaser, GaussianLaser
import unyt as u
import colorcet as cc


def flattened_waist_far_from_focus(z, /, *, lambda0, w0, zfoc):
    return lambda0 / (np.pi * w0) * np.abs(z - zfoc)


def make_flat_laser_profile(job):
    profile = FlattenedGaussianLaser(
        a0=job.sp.a0,
        w0=job.sp.w0,
        tau=job.sp.tau,
        z0=job.sp.z0,
        N=job.sp.profile_flatness,
        zf=job.sp.zfoc,
        lambda0=job.sp.lambda0,
    )
    return profile


def make_gaussian_laser_profile(job):
    profile = GaussianLaser(
        a0=job.sp.a0,
        waist=job.sp.w0,
        tau=job.sp.tau,
        z0=job.sp.z0,
        zf=job.sp.zfoc,
        lambda0=job.sp.lambda0,
    )
    return profile


def plot_laser_intensity(
    profile,
    /,
    *,
    rmax=25.0e-6,
    Nr=64,
    zfoc=0.0e-6,
    z0=0.0e-6,
    zR=0.715e-3,
    lambda0=0.8e-6,
    w0=22.0e-6,
    vert_bars=False,
    fn="laser_intensity.png",
):
    # vacuum impedance
    wave_impedance = 377 * u.ohm

    z = {"near": zfoc * u.meter, "far": (zfoc + 2 * zR) * u.meter}

    # Initially (at t = 0), the laser is at z = z0.
    #  After time t + T, it will be at zfoc, having run the distance zfoc - z0 in time T.
    t = {field: (z[field] - z0 * u.meter) / u.clight for field in ("near", "far")}
    col_label = {
        "near": f"z={z['near'].to(u.micrometer):.1f}, t={t['near'].to(u.fs):.1f}",
        "far": f"z={z['far'].to(u.mm):.2f}, t={t['far'].to(u.ps):.2f}",
    }

    far_waist = flattened_waist_far_from_focus(
        z["far"],
        lambda0=lambda0 * u.meter,
        w0=w0 * u.meter,
        zfoc=zfoc * u.meter,
    )

    grid_near = np.linspace(-rmax, rmax, Nr) * u.meter
    grid_far = (
        np.linspace(-far_waist.to_value("m"), far_waist.to_value("m"), Nr) * u.meter
    )
    x = {
        "near": grid_near,
        "far": grid_far,
    }
    y = {
        "near": grid_near,
        "far": grid_far,
    }

    extent = {}
    X = {}
    Y = {}
    for field in "near", "far":
        extent[field] = (
            np.array([x[field][0], x[field][-1], y[field][0], y[field][-1]])
            * x[field].units
        )
        X[field], Y[field] = np.meshgrid(x[field], y[field])

    # electric_field_y will be 0 due to polarization along x
    intensity = {}
    for field in "near", "far":
        electric_field_x = (
            profile.E_field(
                x=X[field].to_value(u.meter),
                y=Y[field].to_value(u.meter),
                z=z[field].to_value(u.meter),
                t=t[field].to_value(u.second),
            )[0]
            * u.volt
            / u.meter
        )
        intensity[field] = (
            np.abs(electric_field_x) ** 2 / (2 * wave_impedance)
        ).to_value("exawatt/cm**2")

    intensity = {
        "near": {"linear": intensity["near"], "log": np.log(intensity["near"])},
        "far": {"linear": intensity["far"], "log": np.log(intensity["far"])},
    }

    norm = {
        "near": {"linear": colors.Normalize(), "log": colors.LogNorm()},
        "far": {"linear": colors.Normalize(), "log": colors.LogNorm()},
    }

    fig, axs = pyplot.subplots(2, 2)
    axes = {
        "near": {"linear": axs[0, 0], "log": axs[1, 0]},
        "far": {"linear": axs[0, 1], "log": axs[1, 1]},
    }

    for field in "near", "far":
        ax = axes[field]["linear"]
        ax.annotate(
            col_label[field],
            xy=(0.5, 1),
            xytext=(0, 5),
            xycoords="axes fraction",
            textcoords="offset points",
            size="large",
            ha="center",
            va="baseline",
        )

    for field in "near", "far":
        for scale in "linear", "log":
            ax = axes[field][scale]

            img = ax.imshow(
                intensity[field]["linear"],
                extent=extent[field].to(u.micrometer),
                origin="lower",
                cmap=cc.m_kr,
                norm=norm[field][scale],
            )
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.02)
            cbar = fig.colorbar(
                mappable=img,
                orientation="vertical",
                cax=cax,
            )
            if field == "far" and scale == "linear":
                cbar.set_label(r"$I$ ($10^{18}$ W/cm${}^{2}$)")

            # Add the name of the axes
            if field == "near":
                ax.set_ylabel(r"$y \;(\mu \mathrm{m} )$")
            if scale == "log":
                ax.set_xlabel(r"$x \;(\mu \mathrm{m} )$")

            ax.minorticks_on()

            ax.axhline(0.0, linewidth=1, linestyle="dashed", color="0.75")
            ax.plot(
                x[field].to(u.micrometer),
                intensity[field][scale][Nr // 2, :],
                color="0.75",
            )

            if vert_bars and field == "near" and scale == "linear":
                ax.axvline(-11, linewidth=1, linestyle="dashed", color="0.75")
                ax.axvline(11, linewidth=1, linestyle="dashed", color="0.75")

            if scale == "linear" and field == "near":
                contours = ax.contour(
                    x[field].to(u.micrometer),
                    y[field].to(u.micrometer),
                    intensity[field][scale],
                    levels=[
                        1 / np.e ** 2 * np.max(intensity[field]["linear"]),
                        1 / 2 * np.max(intensity[field]["linear"]),
                    ],
                    colors="0.75",
                    linewidths=1,
                )
                ax.clabel(contours, fmt="%1.1f")

    fig.savefig(
        fn,
        dpi=600,
        transparent=False,
        bbox_inches="tight",
    )
    pyplot.close()


def main():
    import random
    import signac

    random.seed(42)

    proj = signac.get_project(search=False)
    ids = [job.id for job in proj]
    job = proj.open_job(id=random.choice(ids))

    profile = make_flat_laser_profile(job)

    plot_laser_intensity(
        profile,
        rmax=job.sp.rmax,
        Nr=job.sp.Nr,
        zfoc=job.sp.zfoc,
        z0=job.sp.z0,
        zR=job.sp.zR,
        lambda0=job.sp.lambda0,
        w0=job.sp.w0,
    )

    print(
        f"zfoc={(job.sp.zfoc * u.meter).to(u.micrometer):.1f}, N={job.sp.profile_flatness:.1f}, z0={(job.sp.z0 * u.meter).to(u.micrometer):.1f}, w0={(job.sp.w0 * u.meter).to(u.micrometer):.1f}"
    )


if __name__ == "__main__":
    main()
