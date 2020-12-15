from signac_dashboard import Dashboard
from signac_dashboard.modules import (
    StatepointList,
    DocumentList,
    ImageViewer,
    VideoViewer,
)
import unyt as u


class MyDashboard(Dashboard):
    def job_sorter(self, job):
        return job.sp.n_e

    def job_title(self, job):
        n_e = (job.sp.n_e / u.meter ** 3).to(1 / u.centimeter ** 3)
        return f"n_e = {n_e:.1e}"

    # def job_subtitle(self, job):
    #     pass


if __name__ == "__main__":
    modules = [
        StatepointList(name="Parameters", enabled=False),
        DocumentList(enabled=False),
        ImageViewer(enabled=False),
        VideoViewer(enabled=True),
    ]
    MyDashboard(modules=modules).main()
