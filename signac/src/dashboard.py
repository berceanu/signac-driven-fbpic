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
        return (job.sp.z_rezolution_factor, job.sp.Nr, job.sp.rmax)

    def job_title(self, job):
        n = f"{job.sp.z_rezolution_factor:.1f}"
        rmax = (job.sp.rmax * u.meter).to(u.micrometer)
        return f"Δz = λ / {n}, Nr = {job.sp.Nr}, rmax = {rmax:.1f}; {job.sp.r_boundary_conditions} BC"


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
