from signac_dashboard import Dashboard
from signac_dashboard.modules import (
    StatepointList,
    ImageViewer,
    VideoViewer,
)
import unyt as u


class MyDashboard(Dashboard):
    def job_sorter(self, job):
        return job.sp.zfoc

    def job_title(self, job):
        zfoc = (job.sp.zfoc * u.meter).to_value("micrometer")
        return f"zfoc = {zfoc:.1f} um"

    # def job_subtitle(self, job):
    #     pass


if __name__ == "__main__":
    modules = [
        StatepointList(name="Parameters", enabled=False),
        ImageViewer(enabled=False),
        ImageViewer(name="2D Histogram", img_globs=["hist2d.png"]),
        VideoViewer(enabled=False),
    ]
    MyDashboard(modules=modules).main()
