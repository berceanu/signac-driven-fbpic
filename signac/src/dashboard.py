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
        return job.sp.zfoc

    def job_title(self, job):
        zfoc = (job.sp.zfoc * u.meter).to(u.micrometer)
        return f"zfoc = {zfoc:.1f}"

    # def job_subtitle(self, job):
    #     pass


if __name__ == "__main__":
    modules = [
        StatepointList(name="Parameters", enabled=False),
        DocumentList(enabled=False),
        ImageViewer(enabled=False),
        ImageViewer(name="2D Histogram", img_globs=["hist2d.png"]),
        ImageViewer(name="Electron Spectrum", img_globs=["final_histogram.png"]),
        VideoViewer(enabled=False),
    ]
    MyDashboard(modules=modules).main()
