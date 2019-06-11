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
    def job_sorter(self, job):
        # should return key for
        # sorted(jobs, key=lambda job: job_sorter(job))
        return job.sp["Nz"], job.sp["Nm"]

    def job_title(self, job):
        return f"(Nz, Nm) = ({job.sp['Nz']}, {job.sp['Nm']})"


if __name__ == "__main__":
    config = {"DASHBOARD_PATHS": ["src/"]}
    dashboard = MyDashboard(
        modules=[
            ImageViewer(name="Image Viewer", img_globs=["*.png"]),
            VideoViewer(name='Animation', video_globs=['*.mp4'], preload='auto'),
            StatepointList(enabled=True),
            DocumentList(max_chars=140),
            FileList(enabled=True),
            Notes(enabled=False),
        ],
        config=config,
    )
    dashboard.main()

# TODO test web interface from CETAL
