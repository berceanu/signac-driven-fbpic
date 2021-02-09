"""Module containing various utility functions."""
import subprocess
import logging
import numpy as np
import time
import datetime

logger = logging.getLogger(__name__)


def convert_from(seconds):
    """
    Convert seconds to H:M:S format.
    Works for periods over 24H also.
    """
    return str(datetime.timedelta(seconds=seconds))


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""


class Timer:
    def __init__(self):
        self._start_time = None

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        runtime = convert_from(time.perf_counter() - self._start_time)
        self._start_time = None

        return runtime


def nozzle_center_offset(distance):
    """
    Convert between distance from center of gas nozzle to distance from z = 0 point.
    """
    other_distance = np.subtract(1500.0e-6, distance)
    return other_distance


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
    """Main entry point."""
    print(ffmpeg_command())

    d = np.array([500, 750, 1000, 1250, 1500]) * 1e-6
    print(d)
    print(nozzle_center_offset(d))
    print(nozzle_center_offset(nozzle_center_offset(d)))

    t = Timer()
    t.start()

    time.sleep(5.0)

    runtime = t.stop()  # A few seconds later
    print(f"Elapsed time: {runtime}")


if __name__ == "__main__":
    main()
