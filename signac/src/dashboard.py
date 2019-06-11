#!/usr/bin/env python3
"""This module contains the web-based data visualization functions for this project.

The workflow defined in this file can be executed from the command
line with

    $ python src/dashboard.py run

See also: $ python src/dashboard.py --help
"""
from signac_dashboard import Dashboard
from signac_dashboard.modules import VideoViewer
from signac_dashboard.modules.document_list import DocumentList
from signac_dashboard.modules.file_list import FileList
from signac_dashboard.modules.image_viewer import ImageViewer
from signac_dashboard.modules.notes import Notes
from signac_dashboard.modules.statepoint_list import StatepointList


class MyDashboard(Dashboard):
    """Child class for customization."""
    def job_sorter(self, job):
        """This method returns a key that can be compared to sort jobs.

        :param job: The job being sorted.
        :type job: :py:class:`signac.contrib.job.Job`
        :returns: Key for :py:function:`sorted(jobs, key=lambda job: job_sorter(job))`
        :rtype: any comparable type
        """
        return job.sp.a0, job.sp.Nm

    def job_title(self, job):
        """This method generates custom job subtitles.

        :param job: The job being subtitled.
        :type job: :py:class:`signac.contrib.job.Job`
        :returns: Subtitle to be displayed.
        :rtype: str
        """
        return f"a₀, Nₘ = {job.sp.a0}, {job.sp.Nm}"


if __name__ == "__main__":
    # which modules are visible by default in the web page
    dashboard = MyDashboard(
        modules=[
            ImageViewer(name="Image Viewer", img_globs=["*.png"]),
            VideoViewer(name='Animation', video_globs=['*.mp4'], preload='auto'),
            StatepointList(enabled=True),
            DocumentList(max_chars=140, enabled=False),
            FileList(enabled=True),
            Notes(enabled=False),
        ],
        config={"DASHBOARD_PATHS": ["src/"]},
    )

    # launch web server
    dashboard.main()
