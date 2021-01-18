"""Module containing various utility functions."""
import subprocess
import logging
import numpy as np

logger = logging.getLogger(__name__)


def nozzle_center_offset(their_distance):
    """
    Given the (experimental) distance from the center of the gas nozzle, compute the fbpic distance.
    """
    our_distance = np.subtract(1500.0e-6, their_distance)
    return our_distance


def w_ave(a, weights):
    """
    Calculate the weighted average of array `a`
    Parameters
    ----------
    a : 1d array
        Calculate the weighted average for these a.
    weights : 1d array
        An array of weights for the values in a.
    Returns
    -------
    Float with the weighted average
    Returns nan if input array is empty
    """
    # Check if input contains data
    if not np.any(weights) and not np.any(a):
        # If input is empty return NaN
        return np.nan
    else:
        # Calculate the weighted average
        average = np.average(a, weights=weights)
        return average


def w_std(a, weights):
    """
    Calculate the weighted standard deviation.
    Parameters
    ----------
    a : array_like
        Calculate the weighted standard deviation for these a.
    weights : array_like
        An array of weights for the values in a.
    Returns
    -------
    Float with the weighted standard deviation.
    Returns nan if input array is empty
    """
    # Check if input contains data
    if not np.any(weights) and not np.any(a):
        # If input is empty return NaN
        return np.nan
    else:
        # Calculate the weighted standard deviation
        average = np.average(a, weights=weights)
        variance = np.average((a - average) ** 2, weights=weights)
        return np.sqrt(variance)


def ffmpeg_command(
    frame_rate: float = 10.0,  # CHANGEME
    input_files: str = "pic%04d.png",  # pic0001.png, pic0002.png, ...
    output_file: str = "test.mp4",
):
    """
    Build up the command string for running ``ffmpeg``.
    http://hamelot.io/visualization/using-ffmpeg-to-convert-a-set-of-images-into-a-video/

    :param frame_rate: desired video framerate, in fps
    :param input_files: shell-like wildcard pattern (globbing)
    :param output_file: name of video output
    :return: command to be executed in the shell for producing video from the input files
    """
    return (
        rf"ffmpeg -framerate {frame_rate} -pattern_type glob -i '{input_files}' "
        rf"-c:v libx264 -vf fps=25 -pix_fmt yuv420p {output_file}"
    )


def shell_run(*cmd, **kwargs):
    """
    Run the command ``cmd`` in the shell.

    :param cmd: the command to be run, with separate arguments
    :param kwargs: optional keyword arguments for ``Popen``, eg. shell=True
    :return: the shell STDOUT and STDERR
    """
    logger.info(cmd[0])
    stdout = (
        subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kwargs
        )
        .communicate()[0]
        .decode("utf-8")
    )
    logger.info(stdout)
    return stdout


def main():
    print(ffmpeg_command())

    d = np.array([500, 750, 1000, 1250, 1500]) * 1e-6
    print(nozzle_center_offset(d))


if __name__ == "__main__":
    main()
