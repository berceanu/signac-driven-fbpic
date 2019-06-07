#!/usr/bin/env python3
from signac_dashboard import Dashboard
from signac_dashboard.modules.document_list import DocumentList
from signac_dashboard.modules.file_list import FileList
from signac_dashboard.modules.image_viewer import ImageViewer
from signac_dashboard.modules.notes import Notes
from signac_dashboard.modules.statepoint_list import StatepointList


class MyDashboard(Dashboard):

    def job_sorter(self, job):
        # shuld return key for
        # sorted(jobs, key=lambda job: job_sorter(job))
        return job.sp['Nz'], job.sp['Nm']

    def job_title(self, job):
        return f"(Nz, Nm) = ({job.sp['Nz']}, {job.sp['Nm']})"


if __name__ == '__main__':
    config = {'DASHBOARD_PATHS': ['src/']}
    dashboard = MyDashboard(modules=[
        ImageViewer(name='Image Viewer', img_globs=['*.png']),
        StatepointList(enabled=True),
        DocumentList(max_chars=140),
        FileList(enabled=True),
        Notes(enabled=False),
    ],
        config=config
    )
    dashboard.main()

# TODO show rho.mp4 movie