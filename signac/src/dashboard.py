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
        ratio = job.sp.n_e / job.doc.n_bunch
        # n_e = (job.sp.n_e / u.meter ** 3).to(1 / u.centimeter ** 3)
        return f"n_e / n_bunch = {ratio:.2f}"

    # def job_subtitle(self, job):
    #     pass


if __name__ == "__main__":
    modules = [
        StatepointList(name="Parameters", enabled=True),
        DocumentList(enabled=True),
        ImageViewer(enabled=False),
        ImageViewer(name="Bunch Plots", enabled=False, img_globs=["bunch/*.png"]),
        VideoViewer(name="rho", enabled=True, video_globs=["rho.mp4"]),
        VideoViewer(name="centroid", enabled=True, video_globs=["centroid.mp4"]),
    ]
    MyDashboard(modules=modules).main()
