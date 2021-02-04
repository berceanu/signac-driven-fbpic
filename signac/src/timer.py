"""Utilities for timing code execution."""
import time
import datetime


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

        elapsed_time = convert_from(time.perf_counter() - self._start_time)
        self._start_time = None

        return elapsed_time


def main():
    """Main entry point."""
    t = Timer()
    t.start()

    time.sleep(5.0)

    elapsed_time = t.stop()  # A few seconds later
    print(f"Elapsed time: {elapsed_time}")


if __name__ == "__main__":
    main()
