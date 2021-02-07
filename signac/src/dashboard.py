from signac_dashboard import Dashboard
from signac_dashboard.modules import (
    StatepointList,
    DocumentList,
    ImageViewer,
    VideoViewer,
    FileList,
    Notes,
)
import unyt as u


class MyDashboard(Dashboard):
    def job_sorter(self, job):
        return (job.sp.z_rezolution_factor, job.sp.Nr, job.sp.rmax)

    def job_title(self, job):
        n = f"{job.sp.z_rezolution_factor:.1f}"
        rmax = (job.sp.rmax * u.meter).to(u.micrometer)
        return f"Δz = λ / {n}, Nr = {job.sp.Nr}, rmax = {rmax:.1f}; {job.sp.r_boundary_conditions} BC"


config = {
    "DASHBOARD_PATHS": ["."],
    "SECRET_KEY": b"\x99o\x90'/\rK\xf5\x10\xed\x8bC\xaa\x03\x9d\x99",
}

modules = [
    StatepointList(name="Params", enabled=False),
    DocumentList(name="Info", enabled=False),
    VideoViewer(name="Evolution", enabled=False),
    ImageViewer(name="Histogram", img_globs=["hist2d.png"], enabled=False),
    ImageViewer(name="Spectrum", img_globs=["final_histogram.png"], enabled=False),
    ImageViewer(name="All Plots", enabled=True),
    FileList(name="All Files", enabled=False),
    # Notes(name="Notes", enabled=False),
]

dashboard = MyDashboard(config=config, modules=modules)

if __name__ == "__main__":
    dashboard.main()
