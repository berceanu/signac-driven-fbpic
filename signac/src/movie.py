from openpmd_viewer import OpenPMDTimeSeries
from typing import List, Optional
import sliceplots
import os
import subprocess
import pathlib
from scipy.constants import physical_constants

q_e = physical_constants["elementary charge"][0]

def sh(*cmd, **kwargs) -> str:
    """
    Run the command ``cmd`` in the shell.

    :param cmd: the command to be run, with separate arguments
    :param kwargs: optional keyword arguments for ``Popen``, eg. shell=True
    :return: the shell STDOUT and STDERR
    """
    stdout = (
        subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kwargs
        )
        .communicate()[0]
        .decode("utf-8")
    )
    return stdout


def ffmpeg_command(
    frame_rate: float = 10.0,  # CHANGEME
    input_files: str = "pic%04d.png",  # pic0001.png, pic0002.png, ...
    output_file: str = "test.mp4",
) -> str:
    """
    Build up the command string for running ``ffmpeg``.
    For details, see `this blog post <http://hamelot.io/visualization/using-ffmpeg-to-convert-a-set-of-images-into-a-video/>`_

    :param frame_rate: desired video framerate, in fps
    :param input_files: shell-like wildcard pattern (globbing)
    :param output_file: name of video output
    :return: the command that needs to be executed in the shell for producing the video from the input files
    """
    return (
        rf"ffmpeg -framerate {frame_rate} -pattern_type glob -i '{input_files}' "
        rf"-c:v libx264 -vf fps=25 -pix_fmt yuv420p {output_file}"
    )

def field_snapshot(
    tseries: OpenPMDTimeSeries,
    it: int,
    field_name: str,
    normalization_factor=1,
    coord: Optional[str] = None,
    m="all",
    theta=0.0,
    chop: Optional[List[float]] = None,
    path="./",
    **kwargs,
) -> None:
    """
    Plot the ``field_name`` field from ``tseries`` at step ``iter``.

    :param path: path to output file
    :param tseries: whole simulation time series
    :param it: time step in the simulation
    :param field_name: which field to extract, eg. 'rho', 'E', 'B' or 'J'
    :param normalization_factor: normalization factor for the extracted field
    :param coord: which component of the field to extract, eg. 'r', 't' or 'z'
    :param m: 'all' for extracting the sum of all the azimuthal modes
    :param theta: the angle of the plane of observation, with respect to the 'x' axis
    :param chop: adjusting extent of simulation box plot
    :param kwargs: extra plotting arguments, eg. labels, data limits etc.
    :return: saves field plot image to disk
    """
    if chop is None:  # how much to cut out from simulation domain
        chop = [0, 0, 0, 0]  # CHANGEME

    field, info = tseries.get_field(
        field=field_name, coord=coord, iteration=it, m=m, theta=theta
    )

    field *= normalization_factor

    plot = sliceplots.Plot2D(
        arr2d=field,
        h_axis=info.z * 1e6,
        v_axis=info.r * 1e6,
        xlabel=r"${} \;(\mu m)$".format(info.axes[1]),
        ylabel=r"${} \;(\mu m)$".format(info.axes[0]),
        extent=(
            info.zmin * 1e6 + chop[0],
            info.zmax * 1e6 + chop[1],
            info.rmin * 1e6 + chop[2],
            info.rmax * 1e6 + chop[3],
        ),
        cbar=True,
        text=f"iteration {it}",
        **kwargs,
    )

    filename = os.path.join(path, f"{field_name}{it:06d}.png")
    plot.canvas.print_figure(filename)


if __name__ == "__main__":
    p = pathlib.Path('/scratch/berceanu/runs/signac-driven-fbpic/workspace/17d602de872d8c1d80e178bbf3c386d6')
    h5_path = p / "diags" / "hdf5"
    rho_path = pathlib.Path.cwd() / "diags" / "rhos"

    time_series: OpenPMDTimeSeries = OpenPMDTimeSeries(h5_path, check_all_files=False)
    iterations = time_series.iterations

    for iteration in iterations:
        field_snapshot(
            tseries=time_series,
            it=iteration,
            field_name="rho",
            normalization_factor=1.0 / (-q_e * 5e+20),
            path=rho_path,
            zlabel=r"$n/n_e$",
            vmin=0.0,
            vmax=2.0,  # CHANGEME
            hslice_val=0,
        )

    command = ffmpeg_command(
    input_files=rho_path / "rho*.png",
    output_file="rho.mp4",
    )
    sh(command, shell=True)
