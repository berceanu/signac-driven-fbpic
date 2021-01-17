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
        return job.sp.w0

    def job_title(self, job):
        w0 = (job.sp.w0 * u.meter).to(u.micrometer)
        return f"w0 = {w0:.1f}"

    # def job_subtitle(self, job):
    #     pass


if __name__ == "__main__":
    modules = [
        StatepointList(name="Parameters", enabled=False),
        DocumentList(enabled=False),
        ImageViewer(enabled=True),
        ImageViewer(name="2D Histogram", enabled=False, img_globs=["hist2d.png"]),
        ImageViewer(name="Electron Spectrum", enabled=False, img_globs=["final_histogram.png"]),
        VideoViewer(enabled=False),
    ]
    MyDashboard(modules=modules).main()
